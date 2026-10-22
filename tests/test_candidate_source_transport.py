"""Exercise scoped history wiring and durable transport receipts without a model."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from dataclasses import asdict

import pytest

import optimizer_agent.codex_backend as transport
from optimizer_agent.codex_backend import CandidateGenerator, CodexBackend, CodexConfig, schema_object
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.evaluation import Candidate, LaunchRule


CONTRACT = "c" * 64
ENVIRONMENT = "e" * 64
SCHEMA = schema_object({"ok": {"type": "boolean"}})


def history_context():
    source = 'extern "C" __global__ void reduction(const float* a, float* out, int n) {}'
    launch = LaunchRule(block=(32, 1, 1), tile_x=32)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "fixture.cu"
        path.write_text(source)
        candidate = Candidate("archived-parent", path, "reduction", launch)
        artifact_hash = candidate.artifact_hash
    return {"run_id": "search-run", "contract_hash": CONTRACT,
            "environment_fingerprint": ENVIRONMENT, "records": [{
                "candidate_id": "archived-parent", "source": source,
                "source_hash": hashlib.sha256(source.encode()).hexdigest(),
                "artifact_hash": artifact_hash, "kernel": "reduction", "parameters": {},
                "launch_rule": asdict(launch),
                "references": [], "evaluations": [],
                "input_file": "PRIVATE_FINAL_INPUT_SENTINEL",
                "oracle": "PRIVATE_ORACLE_SENTINEL",
            }]}


def receipt(trace):
    paths = list(trace.glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_scoped_stdio_tool_is_staged_before_one_cli_launch_with_shell_disabled(tmp_path, monkeypatch):
    trace = tmp_path / "trace"
    backend = CodexBackend(CodexConfig(executable="fake-codex", trace_dir=trace))
    launches = []

    class Process:
        pid = 73123
        returncode = 0

        def communicate(self, prompt, timeout):
            assert prompt == "untrusted CUDA reference data"
            assert receipt(trace)["status"] == "running"
            events = [
                {"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}},
                {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 3}},
            ]
            return "\n".join(json.dumps(event) for event in events), ""

    def popen(command, **kwargs):
        launches.append(command)
        prepared = receipt(trace)
        assert prepared["status"] == "prepared"
        history = prepared["candidate_source_tool"]
        snapshot = Path(history["snapshot"]["path"])
        raw = snapshot.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == history["snapshot"]["sha256"]
        assert b"PRIVATE_FINAL_INPUT_SENTINEL" not in raw
        assert b"PRIVATE_ORACLE_SENTINEL" not in raw
        assert history["model_can_choose_run_or_path"] is False
        config = {command[index + 1].split("=", 1)[0]: command[index + 1].split("=", 1)[1]
                  for index, value in enumerate(command[:-1]) if value == "-c"}
        assert config["features.shell_tool"] == "false"
        assert config["web_search"] == '"disabled"'
        prefix = "mcp_servers.candidate_source."
        assert json.loads(config[prefix + "command"]) == sys.executable
        assert json.loads(config[prefix + "enabled_tools"]) == ["get_candidate_source"]
        assert json.loads(config[prefix + "enabled"]) is True
        assert json.loads(config[prefix + "required"]) is True
        args = json.loads(config[prefix + "args"])
        assert Path(args[0]).is_absolute() and Path(args[0]).name == "candidate_source_tools.py"
        assert args[args.index("--snapshot") + 1] == str(snapshot)
        assert args[args.index("--run-id") + 1] == "search-run"
        assert args[args.index("--contract-hash") + 1] == CONTRACT
        assert args[args.index("--environment-fingerprint") + 1] == ENVIRONMENT
        assert command[-1] == "-" and "--ignore-user-config" in command
        return Process()

    monkeypatch.setattr(transport.subprocess, "Popen", popen)
    result = backend.generate_json("untrusted CUDA reference data", SCHEMA,
                                   candidate_source_context=history_context())
    assert result.payload == {"ok": True} and len(launches) == 1
    saved = receipt(trace)
    assert saved["status"] == "completed"
    assert saved["candidate_source_tool"]["observed_call_intents"] == 0
    assert saved["candidate_source_tool"]["observed_call_results"] == 0
    assert saved["candidate_source_tool"]["audit_files"] == []


@pytest.mark.parametrize("purpose", ["research", "vanilla", "healthcheck"])
def test_history_cannot_be_added_to_research_vanilla_or_healthcheck(tmp_path, monkeypatch, purpose):
    called = []
    monkeypatch.setattr(transport.subprocess, "Popen", lambda *args, **kwargs: called.append(args))
    backend = CodexBackend(CodexConfig(trace_dir=tmp_path))
    with pytest.raises(ValueError, match="proposal transport"):
        backend.generate_json("x", SCHEMA, purpose=purpose, candidate_source_context=history_context())
    assert called == []


def test_snapshot_context_mismatch_is_rejected_before_calling_model():
    class Backend:
        def generate_json(self, *args, **kwargs):
            pytest.fail("Mismatched candidate history must not reach the model")
    generator = CandidateGenerator(Backend(), candidate_source_provider=history_context)
    with pytest.raises(ValueError, match="immutable proposal context"):
        generator._proposal_call("x", SCHEMA, "d" * 64, ENVIRONMENT)


def test_generator_supplies_discoverable_ids_and_same_fixed_context_in_one_call():
    context = history_context()
    calls = []
    class Backend:
        def generate_json(self, prompt, schema, **kwargs):
            calls.append((prompt, schema, kwargs))
            return "synthetic-result"
    generator = CandidateGenerator(Backend(), candidate_source_provider=lambda: context)
    assert generator._proposal_call("immutable prompt", SCHEMA, CONTRACT, ENVIRONMENT) == "synthetic-result"
    assert len(calls) == 1
    prompt, schema, options = calls[0]
    assert prompt.startswith("immutable prompt") and '"candidate_ids": ["archived-parent"]' in prompt
    assert options == {"purpose": "proposal", "candidate_source_context": context}
    assert context["records"][0]["source"] not in prompt


@pytest.mark.parametrize("prefix", ["", "Immutable header\nContext:\n"])
def test_structured_proposal_remains_one_json_frame_with_controller_index(prefix):
    context = history_context()
    calls = []
    class Backend:
        def generate_json(self, prompt, schema, **kwargs):
            assert prompt.startswith(prefix)
            calls.append(json.loads(prompt[len(prefix):]))
    generator = CandidateGenerator(Backend(), candidate_source_provider=lambda: context)
    generator._proposal_call(prefix + '{"immutable_contract":"fixture"}', SCHEMA, CONTRACT, ENVIRONMENT)
    assert len(calls) == 1 and calls[0]["immutable_contract"] == "fixture"
    assert calls[0]["controller_candidate_history"]["candidate_ids"] == ["archived-parent"]


def test_registry_history_is_run_scoped_and_keeps_invalid_slower_candidates(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.sqlite")
    # Campaign identity and all data in this test are synthetic public fixtures.
    store.initialize_campaign("search-run", {"contract": CONTRACT})
    store.initialize_campaign("other-run", {"contract": "d" * 64})
    first = history_context()["records"][0]
    store.archive_candidate("search-run", first)
    store.archive_candidate("search-run", {**first, "candidate_id": "slower-parent"})
    store.set_candidate_validity("search-run", "slower-parent", False, "synthetic numerical failure")
    store.archive_candidate("other-run", {**first, "candidate_id": "private-other-run"})
    records = store.candidate_source_records("search-run")
    assert [record["candidate_id"] for record in records] == ["archived-parent", "slower-parent"]
    assert records[1]["current_validity"]["valid"] is False
    assert all(record["source"] == first["source"] for record in records)
    with pytest.raises(ValueError, match="explicit campaign"):
        store.candidate_source_records("")


def test_runtime_supplies_only_its_fixed_campaign_history():
    from optimizer_agent.campaign import CampaignRuntime
    from types import SimpleNamespace
    observed = []
    instance = object.__new__(CampaignRuntime)
    instance.config = SimpleNamespace(run_id="search-run")
    instance.contract_hash = CONTRACT
    instance.environment_hash = ENVIRONMENT
    instance.store = SimpleNamespace(candidate_source_records=lambda run: observed.append(run) or [])
    assert instance.candidate_source_context() == {
        "run_id": "search-run", "contract_hash": CONTRACT,
        "environment_fingerprint": ENVIRONMENT, "records": []}
    assert observed == ["search-run"]
