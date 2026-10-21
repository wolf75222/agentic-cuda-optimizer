import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from optimizer_agent.codex_backend import (
    CodexBackend, CodexConfig, CodexError, CodexResult, PROPOSAL_SCHEMA, CandidateGenerator,
    Proposal, schema_object, proposal_schema_for_dossier,
)
from optimizer_agent.pmpp_guidance import PMPP_GUIDANCE, consult_pmpp
from optimizer_agent.research_service import (
    NeedsResearch, ResearchBlocked, ResearchConfig, ResearchDossier,
    ResearchService, SourceRecord, observed_web_queries, observed_search_call_count, cuda_documentation_requests,
    SEARCH_SCHEMA, synthesis_schema, normalize_research_references,
    _passage, render_passages, store_hashed_json,
    reject_external_pmpp_actions,
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
                constraints=[],
                verification_criteria=["Independent oracle and held-out tails pass"])


class BackendTests(unittest.TestCase):
    def test_optional_constraints_validate_names_and_preserve_legacy_empty_default(self):
        self.assertNotIn("default", PROPOSAL_SCHEMA["properties"]["constraints"])
        self.assertIn("constraints", PROPOSAL_SCHEMA["required"])
        legacy = proposal_payload()
        legacy.pop("constraints")
        self.assertEqual(Proposal.from_payload(legacy, parent_id="parent", contract_hash="hash", family="tiling").constraints, [])
        payload = proposal_payload()
        payload["constraints"] = [dict(op="bounds", parameters=["tile"], values=[8,32])]
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")
        payload["parameter_space"] = [dict(name="tile", values=[16,32])]
        payload["parameter_bindings"] = [dict(parameter="tile", target="tile_x")]
        proposal = Proposal.from_payload(payload, parent_id="parent", contract_hash="hash", family="tiling")
        self.assertEqual(proposal.constraints, payload["constraints"])
        replay = Proposal.from_dict(asdict(proposal), parent_id="parent", contract_hash="hash", family="tiling")
        self.assertEqual(replay.constraints, proposal.constraints)

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

    def test_nonobject_jsonl_and_accessed_field_shapes_record_failed_receipts(self):
        malformed_events = [None, [], 42, "text", {"type": "item.completed", "item": []},
                            {"type": "item.completed", "item": {"action": []}},
                            {"type": "item.completed", "item": {"type": "agent_message", "text": []}},
                            {"type": "turn.completed", "usage": True}]
        prefix = {"type": "thread.started", "thread_id": "observed-prefix"}
        for malformed in malformed_events:
            with self.subTest(event=malformed), tempfile.TemporaryDirectory() as directory:
                class Process:
                    returncode = 0
                    def communicate(self, prompt, timeout):
                        return "\n".join(map(json.dumps, [prefix, malformed])), "diagnostic"

                backend = CodexBackend(CodexConfig(trace_dir=Path(directory)))
                with patch("optimizer_agent.codex_backend.subprocess.Popen", return_value=Process()):
                    with self.assertRaisesRegex(CodexError, "Codex JSONL") as caught:
                        backend.generate_json("test", {})
                receipt = json.loads(next(Path(directory).glob("*.json")).read_text())
                self.assertEqual(receipt["status"], "failed")
                self.assertEqual(receipt["error"], str(caught.exception))
                self.assertEqual(receipt["events"], [prefix])
                self.assertEqual(receipt["returncode"], 0)
                self.assertEqual(receipt["stderr"], "diagnostic")
                self.assertNotIn("usage", receipt)
                self.assertEqual(backend.calls, [])

    def test_streaming_nonobject_jsonl_terminates_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fake-codex"
            script.write_text(f"#!{sys.executable}\nimport sys,time\nsys.stdin.read()\n"
                              "print('[]',flush=True)\ntime.sleep(30)\n")
            script.chmod(0o755)
            backend = CodexBackend(CodexConfig(executable=str(script), timeout_seconds=5,
                                              trace_dir=Path(directory) / "trace"))
            started = time.monotonic()
            with self.assertRaisesRegex(CodexError, "event must be an object") as caught:
                backend.generate_json("research", {}, purpose="research", max_web_search_calls=1)
            self.assertLess(time.monotonic() - started, 3)
            receipt = json.loads(next((Path(directory) / "trace").glob("*.json")).read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["error"], str(caught.exception))
            self.assertIn("[]", receipt["stdout"])
            self.assertLess(receipt["returncode"], 0)
            self.assertNotIn("usage", receipt)
            self.assertEqual(backend.calls, [])

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
            sources[-1].passages = [_passage(Path(directory), body.decode(), role="full_file", start=1, end=1)]
            sources[-1].read_receipt_version = 2
        dossier = ResearchDossier("research", "hash", "tiling", "gpu-fingerprint", ["real search query"],
            "retrieved", "2026-10-21T00:00:00Z", sources, ["code"], ["16x16 block"], [],
            [dict(condition="same ABI", evidence="contract signature matched source launcher", satisfied=True)],
            ["multi-launch CUB host APIs cannot replace a single kernel"], [], ["tiling"],
            [dict(documented_fact="block barriers synchronize a block", observed_code_property="tiles reused",
                  expected_effect_to_measure="fewer input reads", source_ids=["code", "cuda"])],
            ["lower target latency"], ["fresh paired target timings"], ["shared_tiling"])
        events = [{"type": "item.completed", "item": {"id": "search", "type": "web_search",
                   "action": {"type": "search", "queries": ["real search query"]}}}]
        from optimizer_agent.codex_backend import observed_web_activity
        dossier.receipt_version = 2
        dossier.observed_web = observed_web_activity(events)
        dossier.web_events_path, dossier.web_events_sha256 = store_hashed_json(Path(directory), "web.json", events)
        dossier.supplied_content_path, dossier.supplied_content_sha256 = store_hashed_json(Path(directory), "supplied.json",
            [{"source": asdict(source), "read_text": render_passages(source.passages)} for source in sources])
        return dossier

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

    def test_real_luna_descriptive_reference_shape_preserves_originals(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/luna_research_reference_shape.json").read_text())
        sources = [SourceRecord(item["source_id"], item["kind"], "", "", "", "", "", "", "", "", "", "", "", 1)
                   for item in fixture["sources"]]
        normalized = normalize_research_references(fixture, sources)
        self.assertEqual(normalized["reference_kernels"], ["source-8be5b7397513d692"])
        self.assertEqual(normalized["pmpp_topics"], ["tiling", "coalescing"])
        originals = {entry["original"] for entry in normalized["reference_context"]}
        self.assertTrue(set(fixture["reference_kernels"]) <= originals)
        self.assertTrue(set(fixture["pmpp_topics"]) <= originals)
        self.assertEqual(normalize_research_references(normalized, sources), normalized)
        fixture["reference_kernels"].append("source-0000000000000000 — never retrieved")
        with self.assertRaisesRegex(ResearchBlocked, "Unretrieved reference"):
            normalize_research_references(fixture, sources)

    def test_dynamic_reference_schemas_only_allow_retrieved_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            dossier = self.make_dossier(directory)
            schema = synthesis_schema(dossier.sources)
            self.assertEqual(set(schema["properties"]["reference_kernels"]["items"]["enum"]), {"code", "cccl"})
            self.assertEqual(set(schema["properties"]["hypotheses"]["items"]["properties"]["source_ids"]["items"]["enum"]),
                             {"code", "cccl", "cuda", "pmpp"})
            self.assertIn("tiling", schema["properties"]["pmpp_topics"]["items"]["enum"])
            proposal_schema = proposal_schema_for_dossier(dossier)
            self.assertEqual(set(proposal_schema["properties"]["references"]["items"]["enum"]), {"code", "cccl", "cuda", "pmpp"})
            # Declaring a new technique must reach the dossier's semantic
            # NeedsResearch guard; this never authorizes an unread source.
            self.assertEqual(proposal_schema["properties"]["techniques"], PROPOSAL_SCHEMA["properties"]["techniques"])
            self.assertNotIn("enum", proposal_schema["properties"]["techniques"]["items"])

    def test_adaptation_prerequisite_schema_preserves_genuinely_false_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            dossier = self.make_dossier(directory)
            schema = synthesis_schema(dossier.sources)
            description = schema["properties"]["applicability_checks"]["description"]
            self.assertIn("EXCLUSIVELY", description)
            self.assertIn("standalone adaptation", description)
            self.assertIn("unchanged-reference copyability", description)
            self.assertIn("unavailable from the supplied contract", schema["properties"]["missing_evidence"]["description"])
            raw = dossier.to_dict()
            raw["applicability_checks"] = [dict(condition="Memory alias semantics are defined",
                evidence="A, B and C overlap is unspecified in this deliberately incomplete test contract", satisfied=False)]
            replay = ResearchDossier.from_dict(raw)
            self.assertFalse(replay.applicability_checks[0]["satisfied"])
            with self.assertRaisesRegex(ResearchBlocked, "unsatisfied"):
                replay.research_ready("hash", "tiling", "gpu-fingerprint")

    def test_actual_saved_luna_dossier_is_readable_but_requires_passage_refresh(self):
        path = Path(__file__).resolve().parent.parent / "results/gemm-comparison-20261021/agent/research/e2c1998322464ea88717a70a5545176a.json"
        if not path.is_file():
            self.skipTest("Real campaign receipt is local integration evidence, not a checked-in fixture")
        dossier = ResearchDossier.from_dict(json.loads(path.read_text()))
        self.assertEqual(dossier.reference_kernels, ["source-8be5b7397513d692"])
        self.assertEqual(len(dossier.reference_context), 8)
        with self.assertRaisesRegex(ResearchBlocked, "Legacy dossier"):
            dossier.research_ready(dossier.workload_hash, dossier.strategy_family, dossier.environment_fingerprint)

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
            dossier.supplied_content_path, dossier.supplied_content_sha256 = store_hashed_json(Path(directory), "supplied-no-pmpp.json",
                [{"source": asdict(source), "read_text": render_passages(source.passages)} for source in dossier.sources])
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
                    return CodexResult({"sources": [], "rejected_examples": [], "queries": ["declared but not observed"]}, {"input_tokens": 10},
                        [{"type": "item.completed", "item": {"id": "search", "type": "web_search",
                          "action": {"type": "search", "queries": ["CUTLASS CCCL tiling"]}}}],
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
                return ("// read complete source\nstruct BlockReduce {};\n__global__ void gemm_device() {}\n" + "__global__ void kernel() {}\n" * 10).encode(), url
            return b"<html><p>Official NVIDIA synchronization documentation</p></html>", url
        with tempfile.TemporaryDirectory() as directory:
            backend = Backend()
            dossier = ResearchService(backend, directory, fetcher=fetcher).prepare(
                {"contract_hash": "hash"}, {}, {"strategy_family": "tiling"},
                {"environment_fingerprint": "gpu"})
            self.assertTrue(dossier.research_ready("hash", "tiling", "gpu"))
            self.assertEqual(dossier.usage["input_tokens"], 30)
            self.assertEqual(dossier.declared_queries, ["declared but not observed"])
            self.assertEqual(dossier.queries, ["CUTLASS CCCL tiling"])
            self.assertEqual(dossier.observed_web["search_calls"], 1)
            self.assertIn("__global__ void kernel", backend.prompts[1])
            self.assertIn("Official NVIDIA synchronization", backend.prompts[1])
            self.assertIn("applicability_checks must EXCLUSIVELY list prerequisites", backend.prompts[1])
            self.assertIn("copying any\nreference kernel is unnecessary", backend.prompts[1])
            self.assertIn("unspecified aliasing cannot be assumed safe", backend.prompts[1])
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
            class PermanentBackend(Backend):
                def generate_json(self, *args, **kwargs):
                    result = super().generate_json(*args, **kwargs)
                    if len(self.prompts) == 1:
                        result.payload["sources"] = [dict(kind=kind, repository="", file="", symbol="",
                            url="https://shop.elsevier.com/books/book-companion/9780323912310",
                            section_or_locator="full source", reason="excluded") for kind in ("pmpp", "cuda_documentation")]
                    return result
            permanent_backend = PermanentBackend()
            permanent_backend.prompts = []
            with patch("optimizer_agent.research_service.consult_pmpp", side_effect=AssertionError("targeted cards excluded")) as consult:
                permanent = ResearchService(permanent_backend, Path(directory)/"permanent", fetcher=fetcher,
                    config=ResearchConfig(experimental=True, pmpp_permanent_only=True)).prepare(
                        {"contract_hash":"hash"}, {}, {"strategy_family":"tiling"}, {"environment_fingerprint":"gpu"})
                consult.assert_not_called()
            self.assertEqual(permanent.experimental_mode, "ablation_pmpp_permanent_only")
            self.assertEqual(permanent.pmpp_topics, [])
            self.assertNotIn("pmpp", {source.kind for source in permanent.sources})
            guide = next(source for source in permanent.sources if source.kind == "pmpp_permanent")
            self.assertEqual(Path(guide.cache_path).read_text(), PMPP_GUIDANCE)
            self.assertIn("unchanged original permanent guide", permanent_backend.prompts[1])
            self.assertTrue(permanent.research_ready("hash", "tiling", "gpu"))
            altered = ResearchDossier.from_dict(permanent.to_dict())
            altered.pmpp_topics = ["tiling"]
            with self.assertRaisesRegex(ResearchBlocked, "excluded targeted"):
                altered.research_ready("hash", "tiling", "gpu")
            altered.pmpp_topics, altered.experimental_mode = [], None
            with self.assertRaisesRegex(ResearchBlocked, "explicit experimental marker"):
                altered.research_ready("hash", "tiling", "gpu")
            class ManySourcesBackend(Backend):
                def generate_json(self, *args, **kwargs):
                    result = super().generate_json(*args, **kwargs)
                    if len(self.prompts) == 1:
                        result.payload["sources"] = [dict(kind="code", repository="NVIDIA/cutlass",
                            url=f"https://raw.githubusercontent.com/NVIDIA/cutlass/main/example{index}.cu",
                            file=f"example{index}.cu", symbol="kernel", section_or_locator="full source", reason="test")
                            for index in range(12)]
                    return result
            many_backend = ManySourcesBackend()
            many_backend.prompts = []
            # A hypothetical long elapsed duration must not activate a shared
            # phase deadline under the default uncapped configuration.
            with patch("optimizer_agent.research_service.time.monotonic", side_effect=[0] + [1e8] * 20) as clock:
                many = ResearchService(many_backend, Path(directory) / "many", fetcher=fetcher).prepare(
                    {"contract_hash": "hash"}, {}, {"strategy_family": "tiling"},
                    {"environment_fingerprint": "gpu"})
            self.assertGreater(len(many.sources), 8)
            self.assertEqual(clock.call_count, 1)
            self.assertTrue(many.research_ready("hash", "tiling", "gpu"))

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
            missing = dict(requests[0], url=requests[0]["url"].replace("#shared-memory", "#missing"),
                           section_or_locator="#missing")
            with self.assertRaisesRegex(ValueError, "section #missing was not found"):
                service._open_source(missing)

    def test_real_query_receipts_and_explicit_experimental_mode(self):
        events = [{"type": "item.completed", "item": {"id": "search", "type": "web_search",
                  "action": {"type": "search", "queries": ["NVIDIA CUTLASS tiling"]}}}]
        self.assertEqual(observed_web_queries(events), ["NVIDIA CUTLASS tiling"])
        with self.assertRaises(ValueError):
            ResearchConfig(required=False)
        with self.assertRaises(ValueError):
            ResearchConfig(include_pmpp=False)
        self.assertTrue(ResearchConfig(required=False, experimental=True).experimental)
        self.assertIsNone(ResearchConfig().max_search_calls_per_dossier)
        self.assertIsNone(ResearchConfig().max_sources)
        self.assertIsNone(ResearchConfig().total_timeout_seconds)
        explicit = ResearchConfig(max_sources=8, total_timeout_seconds=1200)
        self.assertEqual((explicit.max_sources, explicit.total_timeout_seconds), (8, 1200))
        with self.assertRaises(ValueError):
            ResearchConfig(total_timeout_seconds=0)
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

    def test_permanent_only_requires_explicit_marker_and_rejects_external_consultation(self):
        with self.assertRaisesRegex(ValueError, "explicit experimental"):
            ResearchConfig(pmpp_permanent_only=True)
        with self.assertRaisesRegex(ValueError, "permanent guide"):
            ResearchConfig(experimental=True, pmpp_permanent_only=True, include_pmpp=False)
        for action in [dict(type="search", queries=["PMPP tiling chapter"]),
                       dict(type="open_page", url="https://shop.elsevier.com/books/book-companion/9780323912310")]:
            with self.assertRaises(ResearchBlocked):
                reject_external_pmpp_actions([dict(type="item.completed", item=dict(id="actual", type="web_search", action=action))])
        reject_external_pmpp_actions([dict(type="item.completed",item=dict(id="actual",type="web_search",
            action=dict(type="search",queries=["NVIDIA CCCL shared memory implementation"])))])


if __name__ == "__main__":
    unittest.main()
