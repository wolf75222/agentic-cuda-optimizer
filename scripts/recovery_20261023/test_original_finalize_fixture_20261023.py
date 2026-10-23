"""Public synthetic CPU fixture for cold finalization of the pinned original graph.

Only this module is an implementation artifact. ``build_fixture`` writes solely
inside its caller's new temporary directory. No existing campaign DB, Slurm,
SSH, model process, GPU or private scientific input is accessed. All timings,
token envelopes and arrays created by the reused test doubles are synthetic.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "optimizer_agent") not in sys.path:
    sys.path.append(str(REPO / "optimizer_agent"))

from campaign_config import CampaignConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from optimizer_agent import upstream_original as upstream


def _test_doubles():
    spec = importlib.util.spec_from_file_location(
        "_public_original_finalize_test_doubles", REPO / "tests/test_upstream_original.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _compile_graph(original, client, runner, config, saver, counters=None):
    class FixtureGraph(StateGraph):
        def add_node(self, node, action=None, **kwargs):
            if counters is not None:
                if node == "finalize_result":
                    actual = action

                    def action(state):
                        counters["finalize_node_calls"] += 1
                        return actual(state)
                else:
                    name = node

                    def action(state):
                        counters["forbidden_node_calls"] += 1
                        raise AssertionError("Finalization-only fixture executed forbidden node: " + name)
            return super().add_node(node, action, **kwargs)

        def compile(self, *args, **kwargs):
            kwargs["checkpointer"] = saver
            return super().compile(*args, **kwargs)

    original.workflow.ChatOpenAI = lambda **kwargs: client
    original.workflow.StateGraph = FixtureGraph
    return original.workflow.build_graph(client, runner, config)


def build_fixture(tmp_path):
    """Return a new authentic-original, synthetic checkpoint pending finalize.

    Returns generation/snapshot Paths; CampaignConfig JSON; four public history
    rows; the selected source SHA; and mutable resume-only action counters.
    The snapshot is materialized from Git1e9464d and each of its16Python blobs
    is authenticated by the existing snapshot_upstream implementation.
    """
    root = Path(tmp_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    generation = root / "synthetic-original-generation"
    generation.mkdir(exist_ok=False)
    snapshot, pin = upstream.snapshot_upstream(root / "pinned-original-source")
    if pin["commit"] != upstream.UPSTREAM_COMMIT or len(pin["source_hashes"]) != 16:
        raise AssertionError("Fixture requires the exact16original Python source blobs")
    doubles = _test_doubles()

    class ThreeProposals(doubles.OriginalResponses):
        def generate_json(self, prompt, schema, **kwargs):
            if self.proposals < 2:
                return super().generate_json(prompt, schema, **kwargs)
            if self.proposals != 2:
                raise AssertionError("Unexpected model action after third synthetic proposal")
            # Reuse the valid original-schema test response, then make a third
            # valid but slower synthetic candidate. Real original nodes route,
            # validate, rank and checkpoint all three proposals themselves.
            self.proposals = 1
            result = super().generate_json(prompt, schema, **kwargs)
            self.proposals = 3
            payload = json.loads(result.payload["text"])
            payload["kernel_source"] = payload["kernel_source"].replace("UNIT_REPAIRED", "UNIT_SLOWER")
            payload["hypothesis"]["starting_iteration"] = 2
            payload["hypothesis"]["change_summary"] = "Public CPU fixture third valid slower attempt"
            result.payload["text"] = json.dumps(payload)
            return result

    config = CampaignConfig(
        run_id="public-synthetic-original-finalize", workload="scan",
        search_sizes=(8, 32, 64, 128), final_sizes=(8,), max_trials=3, benchmark_series=1,
        confirmation_batches=2, launches_per_series=2, warmup=0, profile=False,
    )
    config_data = json.loads(json.dumps(config.to_dict()))
    _write_json(generation / "config.json", config_data)
    suite = upstream.common_workloads.WorkloadSuite("scan", generation / "inputs", config)
    baseline = suite.baseline(REPO / config.baseline_source)
    cases = suite.cases()
    if len(cases) != 7:
        raise AssertionError("Fixture requires the seven public CPU cases used by the cold recovery guard")
    runner = doubles.CPUHarness(generation / "cpu-harness")
    backend = ThreeProposals(tuple(case.metadata["n"] for case in cases), "scan", repair=True)
    client = upstream.OriginalCodexTransport(backend, directory=generation / "cpu-model-transport")
    graph_config = {"configurable": {"thread_id": config.run_id}, "recursion_limit": 512}
    with upstream.original_modules(snapshot, generation) as original:
        with SqliteSaver.from_conn_string(str(generation / "original-langgraph.sqlite")) as saver:
            original_config = original.optimizer_config.OptimizerConfig(
                generation, 0, 3, 0, 2, "gpt-6-luna", config.max_repairs_per_trial,
                nvidia_research=False, use_nsight=False, reasoning_effort="high",
            )
            original_config.validate()
            original_cases = [original.agent_workload.InputCase.from_dict({
                "arguments": asdict(case.request)["arguments"],
                "comparisons": [asdict(c) for c in case.comparisons],
                "size_summary": "n=" + str(case.metadata["n"]),
                "purpose": "performance" if case.metadata["n"] in config.search_sizes else "correctness",
            }, generation) for case in cases]
            workload = original.agent_workload.Workload(
                "PUBLIC SYNTHETIC CPU FIXTURE ONLY", suite.signature,
                original_cases, baseline.source, baseline.source,
            )
            graph = _compile_graph(original, client, runner, original_config, saver)
            initial = dict(
                config=original_config, session_dir=generation, workload=workload,
                device_properties=runner.get_device_properties(0), research_brief="",
                kernel_source="", kernel_role="reference", kernel_proposal=None,
                iteration=0, validation_ok=False, reference_results=[], reference_attempt=None,
                messages=[], optimization_history=[], best_attempt=None, proposal_retries=0,
                model_calls=0, error=None, profiling_unavailable=None, stop_reason=None,
            )
            stream = graph.stream(initial, graph_config, stream_mode="values")
            stopped = False
            try:
                for state in stream:
                    if state["iteration"] == 3 and len(state["optimization_history"]) == 4:
                        stopped = True
                        break
            finally:
                stream.close()
            if not stopped:
                raise AssertionError("True original graph did not yield iteration3/history4")
            checkpoint = graph.get_state(graph_config)
            if checkpoint.next != ("finalize_result",):
                raise AssertionError("Unexpected pending original nodes: " + repr(checkpoint.next))
            if checkpoint.values["best_attempt"].iteration != 2:
                raise AssertionError("Original selection did not retain iteration2")
            # Simulate the real wrapper's external observer-failure catch: write
            # original failed summary/best files without mutating its previously
            # committed, clean graph checkpoint pending finalize_result.
            original.nodes.finalize_result(dict(checkpoint.values, error="PUBLIC SYNTHETIC observer infrastructure failure"))
    leftovers = [p.name for p in generation.glob("original-langgraph.sqlite*")
                 if p.name != "original-langgraph.sqlite"]
    if leftovers:
        raise AssertionError("Fixture checkpoint is not cold: " + repr(leftovers))
    history = json.loads((generation / "history.json").read_text())
    if [r["iteration"] for r in history] != [0, 1, 2, 3]:
        raise AssertionError("Fixture history must contain all four actual original attempts")
    selected = generation / "best.cu"
    source_sha = hashlib.sha256(selected.read_bytes()).hexdigest()
    if source_sha != hashlib.sha256((generation / "candidate-2.cu").read_bytes()).hexdigest():
        raise AssertionError("Fixture exported best differs from original selected candidate2")
    fixture = {
        "generation": generation, "snapshot": snapshot, "config": config_data,
        "history": history, "source_sha256": source_sha, "source_identity": pin,
        "graph_config": graph_config,
        "counters": {"model_calls": 0, "harness_calls": 0, "forbidden_node_calls": 0,
                     "finalize_node_calls": 0, "actual_model_cli_calls": 0, "actual_gpu_calls": 0},
        "setup_counts": {"cpu_backend_calls": len(backend.prompts), "cpu_harness_calls": len(runner.calls)},
        "synthetic_only": True,
    }
    return fixture


@contextlib.contextmanager
def open_fixture_graph(fixture, destination_generation=None):
    """Open a caller-owned cold copy in its ORIGINAL source pickle namespace.

    The caller must rebase session_dir through update_state before stream(None)
    if destination_generation differs from generation. All other graph actions,
    model invocations and harness calls fail and increment dedicated counters.
    """
    source = Path(fixture["generation"])
    target = Path(destination_generation or source)
    counters = fixture["counters"]

    class DeniedClient:
        def bind_tools(self, tools):
            return self

        def invoke(self, *args, **kwargs):
            counters["model_calls"] += 1
            raise AssertionError("Forbidden model action during cold finalize")

        def create(self, *args, **kwargs):
            return self.invoke(*args, **kwargs)

    class DeniedRunner:
        results_dir = target

        def get_device_properties(self, *args, **kwargs):
            counters["harness_calls"] += 1
            raise AssertionError("Forbidden harness properties call during cold finalize")

        def run(self, *args, **kwargs):
            counters["harness_calls"] += 1
            raise AssertionError("Forbidden harness run during cold finalize")

    with upstream.original_modules(fixture["snapshot"], source) as original:
        with SqliteSaver.from_conn_string(str(target / "original-langgraph.sqlite")) as saver:
            config = original.optimizer_config.OptimizerConfig(
                target, 0, 3, 0, 2, "gpt-6-luna", 2,
                nvidia_research=False, use_nsight=False, reasoning_effort="high",
            )
            graph = _compile_graph(original, DeniedClient(), DeniedRunner(), config, saver, counters)
            yield {"graph": graph, "graph_config": fixture["graph_config"], "original": original}


class OriginalFinalizeFixtureTests(unittest.TestCase):
    def test_authentic_graph_checkpoint_get_state_and_finalize_only_cold_copy(self):
        with tempfile.TemporaryDirectory(prefix="public-original-finalize-fixture-") as tmp:
            fixture = build_fixture(Path(tmp))
            source = fixture["generation"]
            source_files = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in source.rglob("*") if p.is_file()}
            target = Path(tmp).resolve() / "cold-copy-generation"
            shutil.copytree(source, target)
            with open_fixture_graph(fixture, target) as opened:
                graph, config = opened["graph"], opened["graph_config"]
                state = graph.get_state(config)
                self.assertEqual(state.next, ("finalize_result",))
                self.assertEqual(state.values["iteration"], 3)
                self.assertEqual(state.values["config"].max_iterations, 3)
                self.assertEqual(state.values["best_attempt"].iteration, 2)
                self.assertEqual(len(state.values["optimization_history"]), 4)
                graph.update_state(config, {"session_dir": target}, as_node="evaluate_candidate")
                self.assertEqual(graph.get_state(config).next, ("finalize_result",))
                list(graph.stream(None, config, stream_mode="values"))
                self.assertEqual(graph.get_state(config).next, ())
            summary = json.loads((target / "summary.json").read_text())
            self.assertEqual((summary["status"], summary["termination"], summary["best_iteration"]),
                             ("complete", "iteration_limit", 2))
            self.assertEqual(hashlib.sha256((target / "best.cu").read_bytes()).hexdigest(), fixture["source_sha256"])
            self.assertEqual(fixture["counters"], {"model_calls": 0, "harness_calls": 0,
                             "forbidden_node_calls": 0, "finalize_node_calls": 1,
                             "actual_model_cli_calls": 0, "actual_gpu_calls": 0})
            self.assertEqual(source_files, {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in source.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
