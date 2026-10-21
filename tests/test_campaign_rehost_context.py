"""Runtime hooks for unscored rehosting; evaluator/accounting are CPU fixtures.

The real policy, registry, candidate identities and copied graph/RNG are used.
No model, GPU, scheduler or SSH command is executed.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer_agent import campaign, suite_orchestrator as suite
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.evaluation import Candidate, LaunchRule, WorkloadContract
from optimizer_agent.research_service import ResearchBlocked
from test_unscored_rehost import setup, prepare


def no_activity(*args, **kwargs):
    pytest.fail("This hook must not execute a model, GPU, SSH or scheduler")


class ModelFixture:
    def __init__(self):
        self.calls = []

    def generate_json(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        no_activity()


@pytest.fixture
def transferred_runtime(tmp_path, monkeypatch):
    value, task, original, old_store, _, _, _ = setup(tmp_path)
    audit = prepare(value, task)
    generation = Path(audit["new_generation"])
    config = suite.config_from_dict(task["payload"]["config"])
    environment = {"uuid": "GPU-new", "name": "SIMULATED_ONLY", "architecture": "sm_90", "harness_build_id": "unit"}
    suite.write_json(generation.parent / "worker/slurm-session.json", {"session_id": audit["new_session"], "job_id": "5678"})
    old_contract = json.loads((generation / "prior-context/contract.json").read_text())["contract"]
    baseline_state = old_store.get_candidate_source("baseline", config.run_id)
    protected_before = {str(path.relative_to(generation)): path.read_bytes()
                        for path in [generation / "langgraph.sqlite", *sorted((generation / "model").glob("*.json"))]}
    with old_store.connection() as db:
        old_snapshot = db.execute("SELECT snapshot FROM campaigns WHERE run_id=?", (config.run_id,)).fetchone()[0]
    calls = []

    class Workloads:
        signature = "fixed ABI"
        def __init__(self, *args):
            pass
        def baseline(self, path):
            # Match the fixture's admitted immutable original baseline exactly.
            return Candidate("baseline", generation / "baseline.cu", "matmul", LaunchRule(**baseline_state["launch_rule"]))
        def cases(self, final=False):
            return [SimpleNamespace(case_id="search-" + str(n), metadata={"n": n}) for n in config.search_sizes]
        def oracle(self, case):
            no_activity()

    def contract_fixture(*args, **kwargs):
        # Only the environment changes; the rehost validator checks the real
        # old/new contract bodies, store, worker identity and raw-series shape.
        body = deepcopy(old_contract)
        metadata = json.loads(body["metadata_json"])
        metadata["environment"] = environment
        body["metadata_json"] = json.dumps(metadata)
        return WorkloadContract(**body)

    class EvaluatorFixture:
        def __init__(self, runner, contract, oracle, suites, **kwargs):
            self.contract_hash = campaign.digest(asdict(contract))
            self.performance = suites["performance"]
            self.contract = contract
        def benchmark(self, candidate):
            calls.append(candidate.candidate_id)
            return {"ok": True, "status": "measured", "evaluation_id": "fresh-independent-baseline",
                    "candidate_id": candidate.candidate_id, "artifact_hash": candidate.artifact_hash,
                    "contract_hash": self.contract_hash, "environment": environment,
                    "cases": {case.case_id: {"latency_ms": 1.0, "series": [
                        {"raw_ms": [1.0] * self.contract.launches_per_series,
                         "report": {"ok": True, "slurm_job_id": "5678"}}
                        for _ in range(self.contract.benchmark_series)]} for case in self.performance}}

    monkeypatch.setattr(campaign, "WorkloadSuite", Workloads)
    monkeypatch.setattr(campaign, "WorkloadContract", contract_fixture)
    monkeypatch.setattr(campaign, "Evaluator", EvaluatorFixture)
    model = ModelFixture()
    runner = SimpleNamespace(get_device_properties=lambda device: deepcopy(environment))
    runtime = campaign.CampaignRuntime(config, runner, generation, model_backend=model, generation_only=True)
    return SimpleNamespace(value=value, task=task, audit=audit, original=original, generation=generation,
                           runtime=runtime, model=model, runner=runner, calls=calls,
                           protected_before=protected_before, old_snapshot=old_snapshot)


def test_real_runtime_rebind_preserves_three_calls_checkpoint_rng_and_resource_history(transferred_runtime):
    case = transferred_runtime
    runtime, audit = case.runtime, case.audit
    assert case.calls == ["baseline"] and case.model.calls == []
    assert {name: (case.generation / name).read_bytes() for name in case.protected_before} == case.protected_before
    assert len(list((case.generation / "model").glob("*.json"))) == 3
    assert runtime.contract_hash != audit["old_proof"]["identity"]["contract"]
    assert runtime.policy.identity["environment"] != audit["old_proof"]["identity"]["environment"]
    with runtime.store.connection() as db:
        snapshot = db.execute("SELECT snapshot FROM campaigns WHERE run_id=?", (runtime.config.run_id,)).fetchone()[0]
        assert snapshot == case.old_snapshot
        assert db.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM reward_claims").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM integration_events").fetchone()[0] == 0
        reservation_rows = list(db.execute("SELECT reservation_id,actual FROM reservations"))
        reservations = {row["reservation_id"]: json.loads(row["actual"]) for row in reservation_rows if row["actual"] is not None}
        assert sum(row["actual"] is None for row in reservation_rows) == 1  # Pending trial remains unconsumed.
        assert reservations["baseline-bootstrap"] == {"gpu_seconds": 20}
        assert audit["baseline_reservation_id"] != "baseline-bootstrap"
        assert reservations[audit["baseline_reservation_id"]]["gpu_seconds"] >= 0
        assert sum(row.get("model_calls", 0) for row in reservations.values()) == 3
        assert sum(row.get("model_tokens", 0) for row in reservations.values()) == 333
    budget = runtime.store.budget_state(runtime.config.run_id)
    assert budget["spent"]["model_calls"] == 3 and budget["spent"]["model_tokens"] == 333


def test_copied_repair_cannot_use_archived_dossier_or_make_a_prompt_before_fresh_research(transferred_runtime):
    case = transferred_runtime
    runtime = case.runtime
    state = {**case.audit["old_proof"]["checkpoint"]["state"], "research_ready": True, "needs_research": False}
    directory = runtime.trial_dir(state)
    assert not (directory / "dossier.json").exists()
    assert (case.generation / "prior-context" / directory.relative_to(case.generation) / "dossier.json").is_file()
    result = runtime.generate(state)
    assert result["needs_research"] is True and result["valid"] is False
    assert case.model.calls == []
    requests = []
    def fresh_web(workload, parent, action, evidence):
        requests.append((workload, parent, action, evidence))
        raise ResearchBlocked("Simulated live web research unavailable")
    runtime.research_service.prepare = fresh_web
    outcome = runtime.research({**state, **result})
    assert outcome["research_ready"] is False and len(requests) == 1
    assert requests[0][0]["contract_hash"] == runtime.contract_hash
    assert requests[0][3]["environment_fingerprint"] == runtime.environment_hash
    assert case.model.calls == []


def test_historical_compilation_context_is_old_contract_only_and_never_a_score(transferred_runtime):
    case = transferred_runtime
    runtime = case.runtime
    state = {**case.audit["old_proof"]["checkpoint"]["state"], "repair": 1}
    receipt_path = runtime.trial_dir(state) / "evaluation-invalid.json"
    original = receipt_path.read_bytes()
    evidence = runtime.generation_evidence(state)
    historical = evidence["previous_evaluation"]
    assert historical["status"] == "compilation_error" and isinstance(historical["cases"], list)
    assert historical["contract_hash"] == case.audit["old_proof"]["identity"]["contract"]
    assert historical["historical_context"]["ranking_eligible"] is False
    assert historical["historical_context"]["previous_environment_fingerprint"] == case.audit["old_proof"]["identity"]["environment"]
    assert receipt_path.read_bytes() == original
    with runtime.store.connection() as db:
        assert db.execute("SELECT count(*) FROM evaluations").fetchone()[0] == 0
    old = json.loads(original)
    for wrong in [{**old, "contract_hash": "unrelated-third-contract"},
                  {**old, "candidate_id": "unrelated-candidate"}, {**old, "status": "measured", "ok": True}]:
        with pytest.raises(ValueError):
            runtime._historical_error_context(wrong)


@pytest.mark.parametrize("marker", [{}, {"phase": "prepared"}, {"phase": "committed", "old_proof": {"run_id": "wrong"}}, []])
def test_uncommitted_or_malformed_marker_blocks_before_device_query_or_model(tmp_path, marker):
    directory = tmp_path / "uncommitted"
    suite.write_json(directory / "unscored-rehost.json", marker)
    runner = SimpleNamespace(get_device_properties=no_activity)
    model = ModelFixture()
    with pytest.raises(ValueError, match="committed unscored generation transfer"):
        campaign.CampaignRuntime(CampaignConfig(run_id="blocked"), runner, directory,
                                 model_backend=model, generation_only=True)
    assert model.calls == []


def test_pending_source_recovery_dispatches_resume_true_and_keeps_original_three_receipts(tmp_path):
    value, task, _, _, _, _, _ = setup(tmp_path)
    audit = prepare(value, task)
    task = value.journal.tasks("generation")[0]
    assert task["state"] == "pending" and task["payload"]["source_recovery"]
    calls = []
    class BackendFixture:
        job_id = "5678"
        def start(self): calls.append("start-owned-fixture")
        def wait_until_ready(self): calls.append("ready-fixture")
        def runner(self): return SimpleNamespace()
        def close(self): calls.append("close-owned-fixture")
        def wait_until_finished(self, timeout):
            return {"job_id": "5678", "terminal": True, "state": "COMPLETED", "exit_code": "0:0"}
    def generator(entry, runner, directory, session, *, resume):
        assert resume is True and session == audit["new_session"]
        assert len(list((Path(directory) / "model").glob("*.json"))) == 3
        calls.append("resume-generation-fixture")
        return {"generation_complete": True, "usage": {"model_calls": 3, "model_tokens": 333}}
    value.backend_factory = lambda task, config, directory: BackendFixture()
    value.generate = generator
    value._task(task)
    row = value.journal.tasks("generation")[0]
    assert row["state"] == "completed" and row["result"]["execution_complete"] is True
    assert calls == ["start-owned-fixture", "ready-fixture", "resume-generation-fixture", "close-owned-fixture"]


@pytest.mark.parametrize("session", ["previous_session", "new_session"])
def test_prepared_global_transfer_intent_blocks_both_sessions_before_backend(tmp_path, session):
    value, old_task, _, _, _, _, _ = setup(tmp_path)
    audit = prepare(value, old_task)
    task = value.journal.tasks("generation")[0]
    suite.write_json(value.directory / "unscored-rehost-intents" / (task["id"] + ".json"),
                     {**audit, "phase": "prepared"})
    value.backend_factory = no_activity
    value.generate = no_activity
    value._task({**task, "session": audit[session]})
    row = value.journal.tasks("generation")[0]
    assert row["state"] == "pending" and row["session"] == audit["new_session"]
    assert row["job_id"] is None and row["result"] is None
