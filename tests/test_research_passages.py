import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from optimizer_agent.codex_backend import observed_web_activity
from optimizer_agent.research_service import (
    ResearchBlocked, ResearchConfig, ResearchService, code_passages,
    observed_search_call_count, observed_web_queries, verify_source_passages,
)
from tests import test_codex_research as fixtures


class PassageTests(unittest.TestCase):
    def service(self, directory, body):
        return ResearchService(None, directory, fetcher=lambda url, timeout, limit: (body.encode(), url))

    def request(self, **overrides):
        return dict(kind="code", repository="NVIDIA/cutlass", file="kernel.cu",
                    url="https://raw.githubusercontent.com/NVIDIA/cutlass/main/kernel.cu",
                    symbol="kernel", section_or_locator="full source", **overrides)

    def test_after_prefix_target_and_launcher_are_actually_supplied(self):
        lines = [f"// padding line {n} " + "x" * 60 + "\n" for n in range(1100)]
        lines[750] = "int requested_tail_context = 123;\n"
        lines[900] = "__global__ void kernel() {}\n"
        lines[1050] = "dim3 launch_grid(5), launch_block(32);\n"
        lines[1051] = "kernel<<<launch_grid, launch_block>>>();\n"
        body = "".join(lines)
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, body)
            request = self.request()
            request.update(section_or_locator="Lines 750–752", line_ranges=[dict(start=750, end=752)])
            source, supplied = service._open_source(request)
            self.assertNotIn("requested_tail_context", body[:30000])
            self.assertIn("requested_tail_context", supplied)
            self.assertIn("kernel<<<launch_grid, launch_block>>>()", supplied)
            self.assertEqual(source.launcher_status, "supplied")
            self.assertEqual(source.requested_context[0]["section_or_locator"], "Lines 750–752")
            self.assertNotEqual(source.section_or_locator, "Lines 750–752")
            self.assertTrue(all(part["truncated"] for part in source.passages))
            verify_source_passages(source)
            part = source.passages[0]
            self.assertEqual(part["text_sha256"], hashlib.sha256(Path(part["cache_path"]).read_bytes()).hexdigest())
            replay = copy.deepcopy(source)
            replay.passages[0]["line_start"] += 1
            with self.assertRaisesRegex(ResearchBlocked, "selection changed"):
                verify_source_passages(replay)
            Path(part["cache_path"]).write_text("tampered")
            with self.assertRaisesRegex(ResearchBlocked, "selection changed"):
                verify_source_passages(source)
            Path(part["cache_path"]).unlink()
            with self.assertRaisesRegex(ResearchBlocked, "missing or malformed"):
                verify_source_passages(source)

    def test_missing_invalid_and_ambiguous_locators_do_not_fall_back(self):
        body = "// real source\n__global__ void kernel() {}\n" + "int x;\n" * 30
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, body)
            for changes in [dict(section_or_locator="Lines 50–60"), dict(section_or_locator="Lines 9–4"),
                            dict(symbol="absent"), dict(symbol="kernel or absent"),
                            dict(symbol="", section_or_locator="missing semantic section")]:
                request = self.request()
                request.update(changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    service._open_source(request)

    def test_multiple_selections_same_body_survive_and_change_passage_identity(self):
        body = "int first;\n" + "// unrelated\n" * 80 + "int last;\n"
        with tempfile.TemporaryDirectory() as directory:
            service = self.service(directory, body)
            first = self.request(); first.update(symbol="", section_or_locator="Lines 1–1")
            last = self.request(); last.update(symbol="", section_or_locator="Lines 82–82")
            narrow, _ = service._open_source(first)
            both, supplied = service._open_source(dict(first, requested_selections=[first, last]))
            self.assertIn("int first", supplied)
            self.assertIn("int last", supplied)
            self.assertEqual(len(both.requested_context), 2)
            self.assertEqual(len(both.passages), 2)
            self.assertNotEqual(narrow.source_id, both.source_id)
            self.assertEqual(narrow.content_sha256, both.content_sha256)

    def test_exact_prefix_and_html_fragment_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            body = "int x;\r\n" * 100
            passages, _ = code_passages(Path(directory), body, [dict(section_or_locator="full source")], 101)
            text = Path(passages[0]["cache_path"]).read_bytes().decode()
            self.assertEqual(text, "int x;\r\n" * 12)
            self.assertEqual(passages[0]["line_end"], 12)
            self.assertTrue(passages[0]["truncated"])
            for markup in ['<h2 id="topic">Only heading</h2><p>Unselected prose</p>',
                           '<a id="topic"></a><p>Unselected prose</p>']:
                service = self.service(directory, markup)
                with self.assertRaisesRegex(ValueError, "section"):
                    service._open_source(dict(kind="cuda_documentation", url="https://docs.nvidia.com/guide#topic"))

    def test_body_hash_alone_does_not_prove_prompt_inclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            dossier = fixtures.ResearchTests().make_dossier(directory)
            self.assertTrue(dossier.research_ready("hash", "tiling", "gpu-fingerprint"))
            manifest = json.loads(Path(dossier.supplied_content_path).read_text())
            manifest[0]["read_text"] = "different source actually supplied"
            # Even a self-consistent manifest hash cannot make omitted text read.
            from optimizer_agent.research_service import store_hashed_json
            dossier.supplied_content_path, dossier.supplied_content_sha256 = store_hashed_json(Path(directory), "changed.json", manifest)
            with self.assertRaisesRegex(ResearchBlocked, "not included exactly"):
                dossier.research_ready("hash", "tiling", "gpu-fingerprint")

    def test_observed_web_receipt_cannot_be_replaced_by_declared_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            dossier = fixtures.ResearchTests().make_dossier(directory)
            dossier.declared_queries = ["not an observed query"]
            self.assertTrue(dossier.research_ready("hash", "tiling", "gpu-fingerprint"))
            dossier.queries = dossier.declared_queries
            dossier.observed_web["queries"] = dossier.declared_queries
            with self.assertRaisesRegex(ResearchBlocked, "completed tool receipts"):
                dossier.research_ready("hash", "tiling", "gpu-fingerprint")

    def test_completed_search_queries_distinct_from_opens_plans_and_duplicates(self):
        def event(kind, identity, action):
            return dict(type=kind, item=dict(id=identity, type="web_search", action=action))
        actual = event("item.completed", "a", dict(type="search", queries=["q", "q"]))
        events = [event("item.started", "a", dict(type="search", queries=["planned"])), actual, actual,
                  event("item.completed", "b", dict(type="open_page", query="not a search")),
                  event("item.completed", "c", dict(type="search", query="second"))]
        activity = observed_web_activity(events)
        self.assertEqual(observed_search_call_count(events), 2)
        self.assertEqual(observed_web_queries(events), ["q", "q", "second"])
        self.assertEqual(activity["query_count"], 3)
        self.assertEqual(activity["action_counts"], {"open_page": 1, "search": 2})


if __name__ == "__main__":
    unittest.main()
