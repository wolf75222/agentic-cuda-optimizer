import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from optimizer_agent.codex_backend import (
    CodexBackend, CodexConfig, CodexError, CodexResult, PROPOSAL_SCHEMA, CandidateGenerator,
    Proposal, schema_object,
)
from optimizer_agent.pmpp_guidance import PMPP_GUIDANCE, consult_pmpp
from optimizer_agent.research_service import (
    NeedsResearch, ResearchBlocked, ResearchConfig, ResearchDossier,
    ResearchService, SourceRecord, observed_web_queries, observed_search_call_count, cuda_documentation_requests,
    SEARCH_SCHEMA,
)


def proposal_payload():
    return dict(parent_id="parent", source_parent_id="parent", contract_hash="hash",
                strategy_family="tiling", techniques=["shared_tiling"], hypothesis="Reuse input tiles",
                pmpp_principle="Cooperatively reuse shared tiles", references=["code", "cuda", "pmpp"],
                cuda_constraint="Every thread reaches each block barrier", modification="Tile GEMM",
                tradeoffs=["shared memory"], source='extern "C" __global__ void matmul() {}',
                kernel="matmul", launch_rule=dict(kind="square_2d", block=[16, 16, 1], grid=[1, 1, 1],
                    elements_key="n", shared_memory_bytes=0, version=1, tile_x=16, tile_y=16),
                parameters=[dict(name="tile", value=16)], expected_measurement="Lower target latency",
                parameter_space=[], parameter_bindings=[],
                verification_criteria=["Independent oracle and held-out tails pass"])


class BackendTests(unittest.TestCase):
    def test_structured_transport_model_usage_and_no_shell(self):
        events = [{"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}},
                  {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 5}}]

        class Process:
            returncode = 0
            def communicate(self, prompt, timeout):
                return "\n".join(json.dumps(event) for event in events), ""

        with tempfile.TemporaryDirectory() as directory:
            with patch("optimizer_agent.codex_backend.subprocess.Popen", return_value=Process()) as popen:
                backend = CodexBackend(CodexConfig(trace_dir=Path(directory)))
                result = backend.generate_json("test", schema_object({"ok": {"type": "boolean"}}))
            command = popen.call_args.args[0]
            self.assertIn("gpt-6-luna", command)
            self.assertIn('model_reasoning_effort="max"', command)
            self.assertIn("features.shell_tool=false", command)
            self.assertIn('web_search="disabled"', command)
            self.assertEqual(result.payload, {"ok": True})
            self.assertEqual(result.usage["input_tokens"], 12)
            self.assertIsNone(result.usage["cost_usd"])
            receipt = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(receipt["status"], "completed")

    def test_timeout_kills_process_group_and_records(self):
        class Process:
            pid = 123456
            calls = 0
            def communicate(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise subprocess.TimeoutExpired("codex", 1)
                return "", ""

        with tempfile.TemporaryDirectory() as directory:
            with patch("optimizer_agent.codex_backend.subprocess.Popen", return_value=Process()), \
                    patch("optimizer_agent.codex_backend.os.killpg") as kill:
                with self.assertRaisesRegex(CodexError, "timed out"):
                    CodexBackend(CodexConfig(trace_dir=Path(directory))).generate_json("test", {})
            kill.assert_called_once()
            self.assertEqual(json.loads(next(Path(directory).glob("*.json")).read_text())["status"], "timeout")

    def test_streaming_search_quota_terminates_process_before_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake-codex"
            script.write_text('''#!/usr/bin/env python3
import json,sys,time
sys.stdin.read()
def emit(identity, action, event="item.completed"):
 print(json.dumps({"type":event,"item":{"id":identity,"type":"web_search","query":"query","action":{"type":action}}}),flush=True)
emit("one","search","item.started")
emit("one","search")
emit("opened-page","open_page")
emit("two","search")
time.sleep(30)
''')
            script.chmod(0o755)
            backend = CodexBackend(CodexConfig(executable=str(script), timeout_seconds=5,
                                              trace_dir=Path(directory) / "trace"))
            started = time.monotonic()
            with self.assertRaisesRegex(CodexError, "live-search budget exceeded"):
                backend.generate_json("bounded research", {}, purpose="research", max_web_search_calls=1)
            self.assertLess(time.monotonic()-started, 3)
            receipt = json.loads(next((Path(directory) / "trace").glob("*.json")).read_text())
            self.assertEqual(receipt["status"], "tool_budget_exhausted")
            self.assertEqual(receipt["observed_search_calls"], 2)
            self.assertIn("opened-page", receipt["stdout"])

    def test_exact_parent_contract_and_launch_types(self):
        payload = proposal_payload()
        proposal = Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")
        self.assertEqual(proposal.parameters, {"tile": 16})
        self.assertEqual(Proposal.from_dict(asdict(proposal), parent_id="parent", contract_hash="hash", family="tiling"), proposal)
        payload["source_parent_id"] = "incumbent"
        with self.assertRaisesRegex(ValueError, "exact selected parent"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")
        payload = proposal_payload()
        payload["contract_hash"] = "changed"
        with self.assertRaisesRegex(ValueError, "immutable"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")
        payload = proposal_payload()
        payload["launch_rule"]["block"][0] = True
        with self.assertRaisesRegex(ValueError, "expected integer"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")

    def test_parameter_domain_and_source_macro_are_declarative(self):
        payload = proposal_payload()
        payload["parameters"] = [dict(name="TILE", value=16)]
        payload["parameter_space"] = [dict(name="TILE", values=[8, 16, 32])]
        payload["parameter_bindings"] = [dict(parameter="TILE", target="source_macro")]
        with self.assertRaisesRegex(ValueError, "matching integer #define"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")
        payload["source"] = "#define TILE 16\n" + payload["source"]
        self.assertEqual(Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling").parameters,
                         {"TILE": 16})
        payload["parameters"][0]["value"] = 4
        with self.assertRaisesRegex(ValueError, "inside its declared domain"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")

    def test_repair_keeps_tree_parent_and_exact_failed_source_chain(self):
        payload = proposal_payload()
        payload["source_parent_id"] = "failed-candidate"
        with self.assertRaisesRegex(ValueError, "explicitly supplied repair source"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")
        proposal = Proposal.from_payload(payload, parent_id="parent", source_parent_id="failed-candidate",
                                         contract_hash="hash", family="tiling")
        self.assertEqual(proposal.parent_id, "parent")
        self.assertEqual(proposal.source_parent_id, "failed-candidate")
        replay = Proposal.from_dict(asdict(proposal), parent_id="parent", source_parent_id="failed-candidate",
                                    contract_hash="hash", family="tiling")
        self.assertEqual(replay, proposal)
        with self.assertRaisesRegex(ValueError, "exact selected parent"):
            Proposal.from_dict(asdict(proposal), parent_id="different-tree-parent", source_parent_id="failed-candidate",
                               contract_hash="hash", family="tiling")

    def test_unresearched_generation_is_explicit_and_never_fabricates_references(self):
        class Backend:
            prompt = ""
            def generate_json(self, prompt, schema, **kwargs):
                self.prompt = prompt
                payload = proposal_payload()
                payload["references"] = []
                return CodexResult(payload, {"input_tokens": 23}, [], "gpt-6-luna", "max", 1, "ablation-call")
        backend = Backend()
        generator = CandidateGenerator(backend)
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            generator.propose_unresearched({"candidate_id": "parent"}, {"strategy_family": "tiling"},
                contract_hash="hash", environment_fingerprint="gpu", experimental_mode="normal")
        proposal = generator.propose_unresearched({"candidate_id": "parent"}, {"strategy_family": "tiling"},
                contract_hash="hash", environment_fingerprint="gpu", experimental_mode="ablation_no_research")
        self.assertEqual(proposal.references, [])
        self.assertEqual(proposal.experimental_mode, "ablation_no_research")
        self.assertIn("no research dossier is", backend.prompt)
        self.assertIn(PMPP_GUIDANCE, backend.prompt)
        with self.assertRaisesRegex(ValueError, "experimental mode differs"):
            Proposal.from_dict(asdict(proposal), parent_id="parent", contract_hash="hash", family="tiling")
        self.assertEqual(Proposal.from_dict(asdict(proposal), parent_id="parent", contract_hash="hash", family="tiling",
            experimental_mode="ablation_no_research"), proposal)
        payload = proposal_payload()
        with self.assertRaisesRegex(ValueError, "must not invent"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling",
                                  experimental_mode="ablation_no_research")


class ResearchTests(unittest.TestCase):
    def make_dossier(self, directory):
        sources = []
        for source_id, kind, repository in (("code", "code", "NVIDIA/cutlass"),
                ("cccl", "code", "NVIDIA/cccl"), ("cuda", "cuda_documentation", ""), ("pmpp", "pmpp", "")):
            body = f"read {source_id}".encode()
            path = Path(directory) / f"{source_id}.txt"
            path.write_bytes(body)
            digest = hashlib.sha256(body).hexdigest()
            sources.append(SourceRecord(source_id, kind, "https://docs.nvidia.com/test", repository,
                f"sha256:{digest}", "file", "symbol", "full file", "independent read", "original", digest,
                str(path), "2026-10-21T00:00:00Z", len(body)))
        return ResearchDossier("research", "hash", "tiling", "gpu-fingerprint", ["real search query"],
            "retrieved", "2026-10-21T00:00:00Z", sources, ["code"], ["16x16 block"], [],
            [dict(condition="same ABI", evidence="contract signature matched source launcher", satisfied=True)],
            ["multi-launch CUB host APIs cannot replace a single kernel"], [], ["tiling"],
            [dict(documented_fact="block barriers synchronize a block", observed_code_property="tiles reused",
                  expected_effect_to_measure="fewer input reads", source_ids=["code", "cuda"])],
            ["lower target latency"], ["fresh paired target timings"], ["shared_tiling"])

    def test_read_receipts_environment_and_new_technique(self):
        with tempfile.TemporaryDirectory() as directory:
            dossier = self.make_dossier(directory)
            self.assertTrue(dossier.research_ready("hash", "tiling", "gpu-fingerprint"))
            with self.assertRaisesRegex(ResearchBlocked, "mismatch"):
                dossier.research_ready("hash", "tiling", "different-GPU")
            proposal = Proposal.from_payload(proposal_payload(), parent_id="parent", contract_hash="hash", family="tiling")
            dossier.validate_proposal(proposal)
            dossier.validate_proposal(asdict(proposal))
            proposal.techniques.append("tensor_cores")
            with self.assertRaises(NeedsResearch):
                dossier.validate_proposal(proposal)
            Path(dossier.sources[0].cache_path).write_text("tampered")
            with self.assertRaisesRegex(ResearchBlocked, "hash changed"):
                dossier.research_ready("hash", "tiling", "gpu-fingerprint")

    def test_boolean_or_search_snippet_cannot_satisfy_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            dossier = self.make_dossier(directory)
            dossier.queries = []
            with self.assertRaises(ResearchBlocked):
                dossier.research_ready("hash", "tiling", "gpu-fingerprint")
            dossier.queries = ["query"]
            dossier.sources = [source for source in dossier.sources if source.kind != "code"]
            with self.assertRaisesRegex(ResearchBlocked, "opened code"):
                dossier.research_ready("hash", "tiling", "gpu-fingerprint")

    def test_pmpp_ablation_excludes_guide_but_keeps_real_research_guards(self):
        class Backend:
            prompt = ""
            def generate_json(self, prompt, schema, **kwargs):
                self.prompt = prompt
                payload = proposal_payload()
                payload["references"] = ["code", "cuda"]
                payload["pmpp_principle"] = "disabled: ablation_no_pmpp"
                return CodexResult(payload, {}, [], "gpt-6-luna", "max", 1)
        with tempfile.TemporaryDirectory() as directory:
            dossier = self.make_dossier(directory)
            dossier.sources = [source for source in dossier.sources if source.kind != "pmpp"]
            dossier.pmpp_topics = []
            dossier.experimental_mode = "ablation_no_pmpp"
            self.assertTrue(dossier.research_ready("hash", "tiling", "gpu-fingerprint"))
            backend = Backend()
            generator = CandidateGenerator(backend)
            with self.assertRaisesRegex(ValueError, "explicit ablation_no_pmpp"):
                generator.propose({"candidate_id": "parent"}, {"strategy_family": "tiling"}, dossier,
                    contract_hash="hash", environment_fingerprint="gpu-fingerprint", include_pmpp=False)
            proposal = generator.propose({"candidate_id": "parent"}, {"strategy_family": "tiling"}, dossier,
                contract_hash="hash", environment_fingerprint="gpu-fingerprint", include_pmpp=False,
                experimental_mode="ablation_no_pmpp")
            self.assertEqual(proposal.experimental_mode, "ablation_no_pmpp")
            self.assertNotIn(PMPP_GUIDANCE, backend.prompt)
            dossier.validate_proposal(proposal)
            dossier.experimental_mode = None
            with self.assertRaisesRegex(ResearchBlocked, "installed PMPP"):
                dossier.research_ready("hash", "tiling", "gpu-fingerprint")

    def test_failed_live_search_stays_blocked_and_persists_reason(self):
        class Backend:
            def generate_json(self, *args, **kwargs):
                return CodexResult({"sources": [], "rejected_examples": []}, {}, [], "gpt-6-luna", "max", 1)
        with tempfile.TemporaryDirectory() as directory:
            service = ResearchService(Backend(), directory)
            with self.assertRaisesRegex(ResearchBlocked, "no observed web_search"):
                service.prepare({"contract_hash": "hash"}, {}, {"strategy_family": "tiling"},
                                {"environment_fingerprint": "gpu"})
            self.assertTrue(list(Path(directory).glob("*-blocked.json")))

    def test_prepare_fetches_sources_synthesizes_actual_contents_and_guards(self):
        class Backend:
            prompts = []
            def generate_json(self, prompt, schema, **kwargs):
                self.prompts.append(prompt)
                if len(self.prompts) == 1:
                    return CodexResult({"sources": [], "rejected_examples": []}, {"input_tokens": 10},
                        [{"type": "item.completed", "item": {"type": "web_search", "query": "CUTLASS CCCL tiling"}}],
                        "gpt-6-luna", "max", 1, "search-call")
                contents = json.loads(prompt.split("Actually read source content:\n")[1])
                references = [entry["source"]["source_id"] for entry in contents if entry["source"]["kind"] == "code"]
                payload = dict(covered_techniques=["shared_tiling"], reference_kernels=references,
                    launch_requirements=["all threads reach barriers"], dependencies=[],
                    applicability_checks=[dict(condition="same ABI", evidence="launcher and contract checked", satisfied=True)],
                    incompatible_conditions=[], missing_evidence=[], pmpp_topics=(
                        ["tiling"] if any(item["source"]["kind"] == "pmpp" for item in contents) else []),
                    hypotheses=[dict(documented_fact="synchronization applies per block",
                        observed_code_property="cooperative tile reuse", expected_effect_to_measure="fewer reads",
                        source_ids=references)], expected_effects=["lower GPU latency"],
                    validation_metrics=["paired fresh target timings"], negative_search=[], original_hypothesis="")
                return CodexResult(payload, {"input_tokens": 20}, [], "gpt-6-luna", "max", 1, "synthesis-call")
        def fetcher(url, timeout, limit):
            if "raw.githubusercontent.com" in url:
                return ("// read complete source\n" + "__global__ void kernel() {}\n" * 10).encode(), url
            return b"<html><p>Official NVIDIA synchronization documentation</p></html>", url
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend()
            dossier = ResearchService(backend, directory, fetcher=fetcher).prepare(
                {"contract_hash": "hash"}, {}, {"strategy_family": "tiling"},
                {"environment_fingerprint": "gpu"})
            self.assertTrue(dossier.research_ready("hash", "tiling", "gpu"))
            self.assertEqual(dossier.usage["input_tokens"], 30)
            self.assertIn("__global__ void kernel", backend.prompts[1])
            self.assertIn("Official NVIDIA synchronization", backend.prompts[1])
            ablation_backend = Backend()
            ablation_backend.prompts = []
            ablation = ResearchService(ablation_backend, Path(directory) / "no-pmpp", fetcher=fetcher,
                config=ResearchConfig(experimental=True, include_pmpp=False)).prepare(
                {"contract_hash": "hash"}, {}, {"strategy_family": "tiling"},
                {"environment_fingerprint": "gpu"})
            self.assertEqual(ablation.experimental_mode, "ablation_no_pmpp")
            self.assertFalse(ablation.pmpp_topics)
            self.assertNotIn("pmpp", {source.kind for source in ablation.sources})
            self.assertNotIn(PMPP_GUIDANCE, ablation_backend.prompts[1])
            self.assertTrue(ablation.research_ready("hash", "tiling", "gpu"))

    def test_code_is_downloaded_as_source_and_rejects_untrusted_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def fetcher(url, timeout, limit):
                calls.append(url)
                return b"// Complete example\n" + b"__global__ void matmul(){}\n" * 12, url
            service = ResearchService(None, directory, fetcher=fetcher)
            source, text = service._open_source(dict(kind="code", repository="NVIDIA/cutlass",
                url="https://github.com/NVIDIA/cutlass/blob/main/example.cu", file="example.cu", symbol="matmul"))
            self.assertEqual(calls, ["https://raw.githubusercontent.com/NVIDIA/cutlass/main/example.cu"])
            self.assertEqual(source.content_sha256, hashlib.sha256(Path(source.cache_path).read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "allowed public HTTPS"):
                service._normalize_url("https://localhost/private")
            with self.assertRaisesRegex(ValueError, "allowed public HTTPS"):
                service._normalize_url("file:///etc/passwd")

    def test_cuda_fragments_read_actual_sections_and_share_one_download(self):
        body = b'''<html><nav>Table of contents and unrelated claims</nav>
<section id="shared-memory"><h3>Shared Memory</h3><p>Blocks share tile storage.</p>
 <section id="nested"><p>Initialize padded lanes safely.</p></section>
</section><section id="unrelated"><p>Must not be included.</p></section>
<section id="synchronization-functions"><h2>Synchronization</h2><p>__syncthreads() waits for all block threads.</p></section></html>'''
        downloads = []
        def fetcher(url, timeout, limit):
            downloads.append(url)
            return body, url
        requests = cuda_documentation_requests({"environment": {"nvrtc_version": "12.6"}})
        self.assertEqual(len(requests), 2)
        self.assertIn("archive/12.6.0", requests[0]["url"])
        with tempfile.TemporaryDirectory() as directory:
            service = ResearchService(None, directory, fetcher=fetcher)
            shared, shared_text = service._open_source(requests[0])
            sync, sync_text = service._open_source(requests[1])
            self.assertEqual(len(downloads), 1)
            self.assertIn("Initialize padded lanes", shared_text)
            self.assertNotIn("Must not be included", shared_text)
            self.assertNotIn("Table of contents", shared_text)
            self.assertNotIn("__syncthreads", shared_text)
            self.assertIn("__syncthreads", sync_text)
            self.assertEqual(shared.content_sha256, sync.content_sha256)
            self.assertNotEqual(shared.source_id, sync.source_id)
            missing = dict(requests[0], url=requests[0]["url"].replace("#shared-memory", "#missing"))
            with self.assertRaisesRegex(ValueError, "section #missing was not found"):
                service._open_source(missing)

    def test_real_query_receipts_and_explicit_experimental_mode(self):
        events = [{"type": "item.completed", "item": {"type": "web_search", "query": "NVIDIA CUTLASS tiling"}}]
        self.assertEqual(observed_web_queries(events), ["NVIDIA CUTLASS tiling"])
        with self.assertRaises(ValueError):
            ResearchConfig(required=False)
        with self.assertRaises(ValueError):
            ResearchConfig(include_pmpp=False)
        self.assertTrue(ResearchConfig(required=False, experimental=True).experimental)
        self.assertIsNone(ResearchConfig().max_search_calls_per_dossier)
        self.assertNotIn("maxItems", SEARCH_SCHEMA["properties"]["sources"])

    def test_search_quota_counts_actions_once_and_service_rejects_overrun(self):
        events = [
            {"type": "item.started", "item": {"id": "a", "type": "web_search", "action": {"type": "other"}}},
            {"type": "item.started", "item": {"id": "a", "type": "web_search", "action": {"type": "search"}}},
            {"type": "item.completed", "item": {"id": "a", "type": "web_search", "query": "query", "action": {"type": "search"}}},
            {"type": "item.completed", "item": {"id": "b", "type": "web_search", "action": {"type": "open_page"}}},
            {"type": "item.completed", "item": {"id": "c", "type": "web_search", "query": "more", "action": {"type": "search"}}},
        ]
        self.assertEqual(observed_search_call_count(events), 2)
        class Backend:
            def generate_json(self, *args, **kwargs):
                self.options = kwargs
                return CodexResult({"sources": [], "rejected_examples": []}, {}, events, "gpt-6-luna", "max", 1)
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend()
            with self.assertRaisesRegex(ResearchBlocked, "exceeded the dossier quota"):
                ResearchService(backend, directory, ResearchConfig(max_search_calls_per_dossier=1)).prepare(
                    {"contract_hash": "hash"}, {}, {"strategy_family": "tiling"},
                    {"environment_fingerprint": "gpu"})
            self.assertEqual(backend.options["max_web_search_calls"], 1)

    def test_pmpp_original_guide_retained_and_cards_provenance(self):
        self.assertIn("Use the performance methodology", PMPP_GUIDANCE)
        card = consult_pmpp(["tiling"])[0]
        self.assertIn("no full book text", card["provenance"])
        self.assertIn("fourth-edition", card["edition"])
        self.assertEqual(len(card["content_sha256"]), 64)
        with self.assertRaises(ValueError):
            consult_pmpp(["invented-book-page"])


if __name__ == "__main__":
    unittest.main()
