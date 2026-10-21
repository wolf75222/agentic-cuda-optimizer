"""Actual backend/generator/runtime protocol, with fake CLI and CUDA reports.

No model, network, SSH or CUDA is run. Supplied research passages and returned
usage are explicit fixtures, not evidence of a live research campaign.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from optimizer_agent import campaign
from optimizer_agent.codex_backend import CodexBackend, CodexConfig, validate_json_shape
from optimizer_agent.campaign_workflow import build_campaign_graph
from tests import test_codex_research as research_fixtures
from tests.test_campaign_runtime import runtime_setup, selected


NEW_TECHNIQUE = "register_blocking"


def bound_dossier(runtime, action, directory, *, renewed=False, diagnostic=None):
    directory.mkdir(parents=True, exist_ok=True)
    dossier = research_fixtures.ResearchTests().make_dossier(directory)
    dossier.research_id = "renewed-research" if renewed else "initial-research"
    dossier.workload_hash = runtime.contract_hash
    dossier.strategy_family = action
    dossier.environment_fingerprint = runtime.environment_hash
    if renewed:
        dossier.covered_techniques.append(NEW_TECHNIQUE)
        dossier.diagnostic_ids = [diagnostic["diagnostic_id"]] if diagnostic else []
    dossier.research_ready(runtime.contract_hash, action, runtime.environment_hash)
    return dossier


def install_cli(monkeypatch, *, techniques, change_payload=None):
    calls = []
    def popen(command, **options):
        assert options["text"] is True and "shell" not in options
        schema_path = Path(command[command.index("--output-schema") + 1])
        schema = json.loads(schema_path.read_text())
        assert "enum" not in schema["properties"]["techniques"]["items"]
        assert set(schema["properties"]["references"]["items"]["enum"]) == {"code", "cccl", "cuda", "pmpp"}
        assert "features.shell_tool=false" in command and 'web_search="disabled"' in command
        class Process:
            returncode = 0
            def communicate(self, prompt, timeout):
                context = json.loads(prompt.split("Context:\n", 1)[1])
                payload = research_fixtures.proposal_payload()
                payload.update(parent_id=context["parent"]["candidate_id"],
                    source_parent_id=context["effective_source_parent"]["candidate_id"],
                    contract_hash=context["contract"]["contract_hash"],
                    strategy_family=context["action"]["strategy_family"],
                    techniques=list(techniques), source=context["effective_source_parent"]["source"],
                    kernel=context["contract"]["operation"])
                if change_payload:
                    change_payload(payload)
                calls.append({"context": context, "prompt": prompt, "schema": schema, "payload": payload})
                # This emulates the provider schema for genuine structured
                # output. The actual CodexBackend independently checks it too.
                if not change_payload:
                    validate_json_shape(payload, schema)
                events = [{"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(payload)}},
                          {"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 3}}]
                return "\n".join(json.dumps(event) for event in events), ""
        return Process()
    monkeypatch.setattr("optimizer_agent.codex_backend.subprocess.Popen", popen)
    return calls


def protocol_runtime(runtime_setup, tmp_path):
    _, config, runner, _ = runtime_setup
    backend = CodexBackend(CodexConfig(executable="fixture-codex-never-executed", trace_dir=tmp_path / "protocol-run/model"))
    runtime = campaign.CampaignRuntime(replace(config, max_trials=1), runner, tmp_path / "protocol-run", model_backend=backend)
    return runtime, backend


def test_actual_structured_backend_routes_new_technique_to_targeted_research_before_any_candidate(runtime_setup, tmp_path, monkeypatch):
    runtime, backend = protocol_runtime(runtime_setup, tmp_path)
    cli_calls = install_cli(monkeypatch, techniques=["shared_tiling", NEW_TECHNIQUE])
    research_rounds = []
    def prepare(workload, parent, action, evidence):
        renewed = bool(research_rounds)
        diagnostic = evidence.get("diagnostic")
        if renewed:
            assert len(backend.calls) == len(cli_calls) == 1
            assert runtime.evaluator.benchmarks == ["baseline"]
            assert runtime.evaluator.validations == []
            assert not list((runtime.directory / "candidates").glob("*.cu"))
            assert not list((runtime.directory / "trials").glob("*/proposal-*.json"))
            assert not [event for event in runtime.store.events(runtime.config.run_id) if event["kind"] == "generation"]
            assert NEW_TECHNIQUE in diagnostic["proposal_hypothesis"]["techniques"]
            assert "insufficient_hypothesis" in diagnostic["triggers"]
            assert "needs_research" in evidence["previous_error"]
            trace = campaign.read_json(backend.config.trace_dir / (backend.calls[0].call_id + ".json"))
            assert trace["status"] == "completed"  # Structured response, not transport failure.
            assert trace["payload"]["techniques"] == ["shared_tiling", NEW_TECHNIQUE]
            assert runtime.store.budget_state(runtime.config.run_id)["spent"]["model_calls"] == 1
        else:
            assert diagnostic is None
        dossier = bound_dossier(runtime, action["strategy_family"], tmp_path / f"research-{len(research_rounds)}",
                                renewed=renewed, diagnostic=diagnostic)
        research_rounds.append(dossier)
        return dossier
    runtime.research_service.prepare = prepare
    with SqliteSaver.from_conn_string(str(tmp_path / "ac02.sqlite")) as checkpoints:
        graph = build_campaign_graph(runtime, checkpoints)
        final = graph.invoke({"run_id": runtime.config.run_id},
            {"configurable": {"thread_id": runtime.config.run_id}, "recursion_limit": 60})
    assert final["finished"] and final["trials_done"] == 1
    assert final["research_round"] == 2 and final["needs_research"] is False
    assert len(research_rounds) == len(cli_calls) == len(backend.calls) == 2
    assert NEW_TECHNIQUE not in research_rounds[0].covered_techniques
    assert NEW_TECHNIQUE in research_rounds[1].covered_techniques
    assert "request for additional research" in cli_calls[0]["prompt"]
    assert "never authorization" in cli_calls[0]["prompt"]
    assert len([path for path in (runtime.directory / "candidates").glob("*.cu") if path.name != "baseline.cu"]) == 1
    assert len(runtime.evaluator.benchmarks) == 2  # Baseline, then the newly covered candidate only.
    generations = [event for event in runtime.store.events(runtime.config.run_id) if event["kind"] == "generation"]
    assert len(generations) == 1
    assert not [event for event in runtime.store.events(runtime.config.run_id) if event["kind"] == "model_uncertain"]
    assert runtime.store.budget_state(runtime.config.run_id)["spent"]["model_calls"] == 2
    dossier = campaign.read_json(next((runtime.directory / "trials").glob("*/dossier.json")))
    assert dossier["research_id"] == "renewed-research" and dossier["diagnostic_ids"]


@pytest.mark.parametrize("defect,error,model_called", [
    ("unread_reference", "unsupported value", True),
    ("changed_contract", "immutable workload contract", True),
    ("changed_kernel", "kernel ABI", True),
    ("inapplicable", "unsatisfied", False),
    ("tampered_source", "hash changed", False),
])
def test_new_declaration_channel_keeps_reference_applicability_integrity_contract_and_abi_guards(
        runtime_setup, tmp_path, monkeypatch, defect, error, model_called):
    runtime, backend = protocol_runtime(runtime_setup, tmp_path)
    state = selected(runtime)
    trial = runtime.trial(state)
    dossier = bound_dossier(runtime, trial["action"], tmp_path / "guard-research")
    if defect == "inapplicable":
        dossier.applicability_checks[0]["satisfied"] = False
    elif defect == "tampered_source":
        Path(dossier.sources[0].cache_path).write_text("altered source fixture")
    campaign.write_json(runtime.trial_dir(state) / "dossier.json", dossier.to_dict())
    state.update(research_round=1, research_ready=True)
    def change(payload):
        if defect == "unread_reference":
            payload["references"] = ["not-a-retrieved-source"]
        elif defect == "changed_contract":
            payload["contract_hash"] = "different-contract"
        elif defect == "changed_kernel":
            payload["kernel"] = "different_kernel"
    calls = install_cli(monkeypatch, techniques=["shared_tiling"], change_payload=change)
    update = runtime.generate(state)
    assert error in update["error"] and update["valid"] is False
    assert bool(calls) is model_called
    assert update.get("candidate_id") is None
    assert not list((runtime.directory / "candidates").glob("*.cu"))
    assert runtime.evaluator.benchmarks == ["baseline"] and runtime.evaluator.validations == []
    assert not [event for event in runtime.store.events(runtime.config.run_id) if event["kind"] == "generation"]
    if defect == "unread_reference":
        assert not update.get("needs_research")
        assert backend.calls == []  # The strict dynamic references schema still fails in the actual backend.
