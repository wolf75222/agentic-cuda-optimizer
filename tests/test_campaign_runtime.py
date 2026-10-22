"""Exercise the real runtime/graph with a declared non-CUDA evaluator fixture.

These tests establish orchestration and recovery behavior, not GPU performance.
TreeQuest, registry, workloads, candidate identities and checkpoints are real.
"""
from dataclasses import asdict, replace
import uuid

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from optimizer_agent import campaign
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.campaign_workflow import build_campaign_graph
from optimizer_agent.greedy_search_policy import GreedySearchPolicy
from optimizer_agent.research_service import ResearchBlocked


class RunnerFixture:
    def __init__(self,uuid="fixture-gpu"):
        self.uuid = uuid
    def get_device_properties(self,device):
        assert device == 0
        return {"name":"simulated orchestration GPU","uuid":self.uuid,"architecture":"sm_90",
                "driver_version":12000,"nvrtc_version":"12.0","max_threads_per_block":1024,
                "max_block_dimensions":[1024,1024,64],"max_shared_memory_per_block_bytes":49152}


class ModelFixture:
    def __init__(self):
        self.calls = []
    def generate_json(self,*args,**kwargs):
        self.calls.append((args,kwargs))
        raise AssertionError("A blocked-research test must never call the generator")


class EvaluatorFixture:
    instances = []
    confirmations = []
    def __init__(self,runner,contract,oracle,suites,**kwargs):
        self.runner,self.contract,self._suites = runner,contract,suites
        self.contract_hash = campaign.digest(asdict(contract))
        self.benchmarks,self.validations = [],[]
        self.instances.append(self)
    def benchmark(self,candidate,protocol=None):
        self.benchmarks.append(candidate.candidate_id)
        tile = candidate.parameters.get("TILE",16)
        latency = 10. if candidate.candidate_id == "baseline" else 80./tile
        return {"evaluation_id":uuid.uuid4().hex,"ok":True,"status":"measured",
                "contract_hash":self.contract_hash,"candidate_id":candidate.candidate_id,
                "cases":{case.case_id:{"latency_ms":latency,"weight":1/len(self._suites["performance"])}
                         for case in self._suites["performance"]}}
    def validate(self,candidate,suite="search"):
        self.validations.append(candidate.candidate_id)
        return {"evaluation_id":uuid.uuid4().hex,"ok":True,"status":"valid"}
    def confirm(self,candidate,incumbent,protocol=None):
        self.confirmations.append((candidate.candidate_id,incumbent.candidate_id))
        return {"evaluation_id":uuid.uuid4().hex,"status":"promoted","ok":True,
                "candidate_id":candidate.candidate_id,"incumbent_id":incumbent.candidate_id,
                "interval":[1.5,2.5]}
    def finalize(self,candidate,baseline):
        return {"ok":True,"status":"validated_no_gain" if candidate.candidate_id == "baseline" else "certified",
                "candidate_id":candidate.candidate_id,"artifact_hash":candidate.artifact_hash,
                "contract_hash":self.contract_hash}


@pytest.fixture
def runtime_setup(tmp_path,monkeypatch):
    EvaluatorFixture.instances,EvaluatorFixture.confirmations = [],[]
    monkeypatch.setattr(campaign,"Evaluator",EvaluatorFixture)
    config = CampaignConfig(run_id="runtime-fixture",backend="local",search_sizes=(8,),final_sizes=(11,),
        max_trials=2,profile=False,benchmark_series=3,confirmation_batches=3,launches_per_series=2)
    runner,model = RunnerFixture(),ModelFixture()
    runtime = campaign.CampaignRuntime(config,runner,tmp_path/"run",model_backend=model)
    return runtime,config,runner,model


def selected(runtime):
    return {"run_id":runtime.config.run_id,"trials_done":0,**runtime.initialize({}),**runtime.select({})}


def test_generator_memory_facts_are_bound_into_immutable_contract(runtime_setup):
    import json
    runtime, config, runner, model = runtime_setup
    policy = runtime.workload_description["memory_policy"]
    assert policy == json.loads(runtime.evaluator.contract.metadata_json)["memory_policy"]
    assert "disjoint" in policy["allocations"]
    assert "fully overwrite" in policy["output"]
    assert "read-only" in policy["inputs"]
    assert "float32 FMA is allowed" in policy["arithmetic"]


def test_real_runtime_initialization_and_pending_trial_resume(runtime_setup):
    runtime,config,runner,model = runtime_setup
    state = selected(runtime)
    trial = runtime.trial(state)
    assert trial["parent"]["candidate_id"] == "baseline"
    assert trial["parent"]["source"] == runtime.baseline.source.read_text()
    assert campaign.digest(runtime.policy.snapshot()["tree"].tree.root.state) == campaign.digest(trial["parent"])
    assert runtime.store.budget_state(config.run_id)["reserved"]["trials"] == 1
    restored = campaign.CampaignRuntime(config,runner,runtime.directory,model_backend=model)
    resumed = restored.select({"trials_done":0})
    assert resumed["trial_id"] == state["trial_id"]
    assert restored.trial(resumed) == trial
    assert restored.evaluator.benchmarks == []  # Reuse the complete baseline receipt.


def test_actual_runtime_graph_blocks_model_on_negative_research(runtime_setup,tmp_path):
    runtime,config,runner,model = runtime_setup
    def unavailable(*args,**kwargs):
        raise ResearchBlocked("live search unavailable; no fabricated provenance")
    runtime.research_service.prepare = unavailable
    with SqliteSaver.from_conn_string(str(tmp_path/"graph.sqlite")) as checkpoints:
        graph = build_campaign_graph(runtime,checkpoints)
        final = graph.invoke({"run_id":config.run_id},{"configurable":{"thread_id":config.run_id},"recursion_limit":100})
        assert final["finished"]
        assert final["trials_done"] == 2
        assert final["termination_reason"] == "trial_budget"
        assert not graph.get_state({"configurable":{"thread_id":config.run_id}}).next
    assert model.calls == []
    assert len(runtime.policy.snapshot()["tree"].tree) == 1
    assert runtime.store.incumbent(config.run_id)["candidate_id"] is None
    blocked = [event for event in runtime.store.events(config.run_id) if event["kind"] == "research_blocked"]
    assert len(blocked) == 2
    assert final["certification_status"] == "validated_no_gain"


def test_runtime_missing_dossier_reenters_research(runtime_setup):
    runtime,_,_,model = runtime_setup
    state = selected(runtime)
    update = runtime.generate(state)
    assert update["needs_research"] and not update["valid"]
    assert model.calls == []


def test_runtime_closed_pending_does_not_block_other_selection(runtime_setup):
    runtime,_,_,model = runtime_setup
    state = selected(runtime)
    runtime.policy.close("candidate","baseline","No applicable action remains")
    outcome = runtime.select({"trials_done":0})
    assert outcome["termination_reason"] == "no_eligible_branch"
    assert runtime.policy.pending_trials() == []
    assert len(runtime.policy.snapshot()["tree"].tree) == 1
    assert model.calls == []


def test_runtime_changed_gpu_rejects_resume(runtime_setup):
    runtime,config,_,model = runtime_setup
    with pytest.raises(ValueError,match="GPU/toolchain changed"):
        campaign.CampaignRuntime(config,RunnerFixture("different-device"),runtime.directory,model_backend=model)


def archive_candidate(runtime,state,candidate_id="fast",source=None,parameters=None):
    path = runtime.directory/(candidate_id+".cu")
    path.write_text(source or runtime.baseline.source.read_text())
    candidate = campaign.Candidate(candidate_id,path,runtime.config.workload,runtime.baseline.launch_rule,parameters or {})
    tree_parent_id = runtime.trial(state)["parent"]["candidate_id"]
    runtime.store.archive_candidate(runtime.config.run_id,runtime.candidate_state(candidate,
        tree_parent_id=tree_parent_id,source_parent_id=tree_parent_id,trial_id=state["trial_id"]))
    return candidate


def record_measurement(runtime, state, candidate, measurement=None):
    measurement = measurement or runtime.evaluator.benchmark(candidate)
    campaign.write_json(runtime.trial_dir(state)/f"evaluation-{candidate.candidate_id}.json",measurement)
    state.update(candidate_id=candidate.candidate_id,research_ready=True,
                 valid=measurement["ok"],evaluation_id=measurement["evaluation_id"],
                 evaluation_status=measurement["status"])
    return measurement


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("stop", ["exhausted", "repair_remaining", "needs_research", "research_blocked", "wrong_binding"])
def test_only_current_exhausted_cuda_failure_is_a_zero_observation(runtime_setup,legacy,stop):
    import json
    runtime,config,_,_ = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime,state)
    measurement = {"evaluation_id":"stable-cuda-error","candidate_id":candidate.candidate_id,
                   "ok":False,"status":"compilation_error","error":"Explicit CPU fixture"}
    record_measurement(runtime,state,candidate,measurement)
    state.update(retryable=True,repair=runtime.max_repairs,error=json.dumps(measurement)[:5000])
    if legacy:
        state.pop("evaluation_id"); state.pop("evaluation_status")
    if stop == "repair_remaining":state["repair"] -= 1
    elif stop == "needs_research":state["needs_research"] = True
    elif stop == "research_blocked":state.update(research_ready=False,error="Research failed after the CUDA error")
    elif stop == "wrong_binding":
        if legacy:state["error"] = "Malformed later proposal"
        else:state["evaluation_id"] = "another-evaluation"
    result = runtime.prepare_result(state)
    if stop == "exhausted":
        assert result["tell"]["valid"] is False
        assert result["tell"]["evaluation_id"] == measurement["evaluation_id"]
        runtime.integrate(state)
        assert len(runtime.policy.snapshot()["tree"].tree) == 2
    else:
        assert "tell" not in result
        runtime.integrate(state)
        assert len(runtime.policy.snapshot()["tree"].tree) == 1
        assert not runtime.policy.pending_trials()
    assert runtime.store.incumbent(config.run_id)["candidate_id"] is None


@pytest.mark.parametrize("already_consumed", [False,True])
def test_cached_legacy_documentary_zero_is_preserved_but_never_newly_told(runtime_setup,already_consumed):
    runtime,config,_,_ = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime,state)
    state.update(candidate_id=candidate.candidate_id,valid=False,retryable=True,repair=1,
                 research_ready=False,needs_research=True,error="research_blocked: explicit CPU fixture")
    consumption = {"trials":1,"gpu_seconds":0}
    result = {"candidate_id":candidate.candidate_id,"error":state["error"],"consumption":consumption,
        "tell":{"evaluation_id":"legacy-documentary-zero","valid":False,"purpose":"tree_reward",
                "consumption":consumption,"state":runtime.store.get_candidate_source(candidate.candidate_id,config.run_id)}}
    path = runtime.trial_dir(state)/"result.json"
    campaign.write_json(path,result)
    original = path.read_bytes()
    if already_consumed:
        runtime.policy.tell_once(state["trial_id"],result["tell"])
        runtime.policy.close("candidate",candidate.candidate_id,"repair budget exhausted",evidence=[state["error"]])
    tree_size = len(runtime.policy.snapshot()["tree"].tree)
    closures = sum(event["kind"] == "branch_decision" for event in runtime.store.events(config.run_id))
    assert runtime.integrate(state)["trials_done"] == 1
    assert runtime.integrate(state)["trials_done"] == 1
    assert path.read_bytes() == original
    assert len(runtime.policy.snapshot()["tree"].tree) == tree_size == (2 if already_consumed else 1)
    assert sum(event["kind"] == "branch_decision" for event in runtime.store.events(config.run_id)) == closures
    assert len(list(runtime.trial_dir(state).glob("legacy-result-integration-refused-*.json"))) == 1
    assert not runtime.policy.pending_trials()


def test_cached_real_exhausted_cuda_failure_keeps_legacy_stable_reward_id(runtime_setup):
    import json
    runtime,config,_,_ = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime,state)
    measurement = {"evaluation_id":"actual-old-compile-attempt","candidate_id":candidate.candidate_id,
                   "ok":False,"status":"compilation_error","error":"CPU fixture genuine compile failure"}
    record_measurement(runtime,state,candidate,measurement)
    state.update(retryable=True,repair=runtime.max_repairs,error=json.dumps(measurement)[:5000])
    state.pop("evaluation_id");state.pop("evaluation_status")
    result = runtime.prepare_result(state)
    result["tell"]["evaluation_id"] = "old-durable-reward-id"
    path = runtime.trial_dir(state)/"result.json"
    campaign.write_json(path,result)
    original = path.read_bytes()
    runtime.integrate(state)
    assert path.read_bytes() == original and len(runtime.policy.snapshot()["tree"].tree) == 2
    assert not list(runtime.trial_dir(state).glob("legacy-result-integration-refused-*.json"))
    rewards = [event for event in runtime.store.events(config.run_id) if event["kind"] == "tree_reward"]
    assert len(rewards) == 1 and rewards[0]["payload"]["evaluation_id"] == "old-durable-reward-id"


def test_confirmation_and_promotion_replay_reuses_durable_evidence(runtime_setup):
    runtime,config,runner,model = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime,state)
    record_measurement(runtime,state,candidate)
    real_promote = runtime.store.promote
    def crash_after_promotion(*args,**kwargs):
        promoted = real_promote(*args,**kwargs)
        assert promoted
        raise RuntimeError("interrupted after durable promotion")
    runtime.store.promote = crash_after_promotion
    with pytest.raises(RuntimeError,match="after durable promotion"):
        runtime.integrate(state)
    assert runtime.store.incumbent(config.run_id) == {"candidate_id":"fast","version":1}
    restored = campaign.CampaignRuntime(config,runner,runtime.directory,model_backend=model)
    restored.integrate(state)
    assert EvaluatorFixture.confirmations == [("fast","baseline")]
    assert restored.store.incumbent(config.run_id) == {"candidate_id":"fast","version":1}
    assert len(restored.policy.snapshot()["tree"].tree) == 2
    restored.integrate(state)
    assert len(restored.policy.snapshot()["tree"].tree) == 2


@pytest.mark.parametrize("constraint,winning_tile", [([],32), ([{"op":"le","parameters":["TILE"],"values":[16]}],16)])
def test_runtime_tuner_uses_exact_variant_and_fresh_reward_measurement(runtime_setup,constraint,winning_tile):
    runtime,_,_,_ = runtime_setup
    state = selected(runtime)
    source = "#ifndef TILE\n#define TILE 16\n#endif\n"+runtime.baseline.source.read_text()
    candidate = archive_candidate(runtime,state,source=source)
    state.update(candidate_id=candidate.candidate_id,repair=0,research_round=1)
    proposal = {"source":source,"kernel":runtime.config.workload,"launch_rule":asdict(candidate.launch_rule),
                "parameter_space":[{"name":"TILE","values":[16,32]}],
                "parameter_bindings":[{"parameter":"TILE","target":"source_macro"}],"constraints":constraint}
    campaign.write_json(runtime.trial_dir(state)/"proposal-0-1.json",proposal)
    outcome = runtime.evaluate(state)
    assert outcome["valid"] and outcome["candidate_id"] != "fast"
    winner = runtime.candidate(outcome["candidate_id"])
    assert winner.parameters == {"TILE":winning_tile}
    assert f"#define TILE {winning_tile}" in winner.source.read_text()
    assert runtime.evaluator.benchmarks.count(winner.candidate_id) == 2
    stored = runtime.store.get_candidate_source(winner.candidate_id,runtime.config.run_id)
    assert stored["tree_parent_id"] == "baseline" and stored["source_parent_id"] == "fast"
    tuning = campaign.read_json(runtime.trial_dir(state)/"autotuning.json")
    assert tuning["constraints"] == constraint
    if constraint:
        rejected = next(row for row in tuning["configurations"] if row["parameters"]["TILE"] == 32)
        assert rejected["status"] == "filtered"
        assert rejected["candidate_id"] not in runtime.evaluator.validations
        assert rejected["candidate_id"] not in runtime.evaluator.benchmarks


@pytest.mark.parametrize("autotuning", [True, False])
def test_runtime_unknown_constraint_reference_rejected_before_candidate_gpu_evaluation(runtime_setup,autotuning):
    runtime,config,runner,model = runtime_setup
    if not autotuning:
        runtime = campaign.CampaignRuntime(replace(config,autotuning_enabled=False),runner,
            runtime.directory.parent/"no-autotuning",model_backend=model)
    state = selected(runtime)
    candidate = archive_candidate(runtime,state)
    state.update(candidate_id=candidate.candidate_id,repair=0,research_round=1)
    campaign.write_json(runtime.trial_dir(state)/"proposal-0-1.json",{
        "parameter_space":[],"constraints":[{"op":"le","parameters":["unknown"],"values":[16]}]})
    before = list(runtime.evaluator.benchmarks)
    result = runtime.evaluate(state)
    assert result["valid"] is False and "unknown parameter" in result["error"]
    assert runtime.evaluator.benchmarks == before and not runtime.evaluator.validations
    assert campaign.read_json(runtime.trial_dir(state)/"constraints.json")["status"] == "rejected"


def test_runtime_no_autotuning_still_applies_constraints_to_concrete_parameters(runtime_setup):
    original,config,runner,model = runtime_setup
    runtime = campaign.CampaignRuntime(replace(config,autotuning_enabled=False),runner,
        original.directory.parent/"fixed-configuration",model_backend=model)
    state = selected(runtime)
    candidate = archive_candidate(runtime,state,parameters={"TILE":32})
    state.update(candidate_id=candidate.candidate_id,repair=0,research_round=1)
    campaign.write_json(runtime.trial_dir(state)/"proposal-0-1.json",{
        "parameter_space":[{"name":"TILE","values":[16,32]}],
        "constraints":[{"op":"le","parameters":["TILE"],"values":[16]}]})
    before = list(runtime.evaluator.benchmarks)
    outcome = runtime.evaluate(state)
    assert not outcome["valid"] and "violates declared tuning constraints" in outcome["error"]
    assert runtime.evaluator.benchmarks == before and not runtime.evaluator.validations


def test_same_tuned_executable_in_distinct_trials_preserves_both_lineages(runtime_setup):
    runtime,_,_,_ = runtime_setup
    source = "#ifndef TILE\n#define TILE 16\n#endif\n"+runtime.baseline.source.read_text()
    winners = []
    for index in range(2):
        state = {"run_id":runtime.config.run_id,"trials_done":index,**runtime.select({"trials_done":index})}
        candidate = archive_candidate(runtime,state,candidate_id=f"initial-{index}",source=source)
        state.update(candidate_id=candidate.candidate_id,repair=0,research_round=1,research_ready=True)
        proposal = {"source":source,"kernel":runtime.config.workload,"launch_rule":asdict(candidate.launch_rule),
                    "parameter_space":[{"name":"TILE","values":[16,32]}],
                    "parameter_bindings":[{"parameter":"TILE","target":"source_macro"}]}
        campaign.write_json(runtime.trial_dir(state)/"proposal-0-1.json",proposal)
        update = runtime.evaluate(state)
        assert update["valid"],update.get("error")
        state.update(update)
        runtime.integrate(state)
        winner = runtime.store.get_candidate_source(state["candidate_id"],runtime.config.run_id)
        assert winner["source_parent_id"] == candidate.candidate_id
        assert winner["trial_id"] == state["trial_id"]
        winners.append(winner)
    assert winners[0]["source"] == winners[1]["source"]
    assert winners[0]["candidate_id"] != winners[1]["candidate_id"]


def test_runtime_no_tree_ablation_uses_genuine_confirmed_greedy_parent(runtime_setup):
    original,config,runner,model = runtime_setup
    control = campaign.CampaignRuntime(replace(config,experimental_mode="ablation_no_tree"),runner,
        original.directory.parent/"greedy-control",model_backend=model)
    assert isinstance(control.policy,GreedySearchPolicy)
    first = selected(control)
    assert control.trial(first)["action"] == config.actions[0]
    candidate = archive_candidate(control,first)
    record_measurement(control,first,candidate)
    control.integrate(first)
    second = control.select({"trials_done":1})
    assert control.trial(second)["parent"]["candidate_id"] == "fast"
    assert control.trial(second)["action"] == config.actions[1]
    assert "tree" not in control.policy.snapshot()


def test_closed_pending_trial_records_elapsed_gpu_usage(runtime_setup,monkeypatch):
    runtime,config,_,_ = runtime_setup
    state = selected(runtime)
    selected_at = runtime.trial(state)["selected_at"]
    before = runtime.store.budget_state(config.run_id)["spent"].get("gpu_seconds",0)
    monkeypatch.setattr(campaign.time,"time",lambda:selected_at+7)
    runtime.policy.close("candidate","baseline","pending action became inapplicable")
    assert runtime.select({"trials_done":0})["termination_reason"] == "no_eligible_branch"
    after = runtime.store.budget_state(config.run_id)["spent"]["gpu_seconds"]
    assert after-before == pytest.approx(7)


def test_profile_records_actual_gpu_usage_without_global_cap(runtime_setup,monkeypatch):
    from types import SimpleNamespace
    runtime,config,_,_ = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime,state)
    record_measurement(runtime,state,candidate)
    runtime.integrate(state)
    runtime.config = replace(config,profile=True)
    runtime.evaluator._request = lambda candidate,case:object()
    before = runtime.store.budget_state(config.run_id)
    receipt = {}
    def profile(*args,**kwargs):
        receipt["budget_during_profile"] = runtime.store.budget_state(config.run_id)
        return {"ok":True,"diagnostic_only":True}
    runtime.profiler = SimpleNamespace(profile=profile)
    clock = iter((100.,102.))
    monkeypatch.setattr(campaign.time,"monotonic",lambda:next(clock))
    runtime.diagnose(state)
    after = runtime.store.budget_state(config.run_id)
    assert "gpu_seconds" not in receipt["budget_during_profile"]["limits"]
    assert receipt["budget_during_profile"]["reserved"].get("gpu_seconds",0) == before["reserved"].get("gpu_seconds",0)
    assert after["reserved"].get("gpu_seconds",0) == before["reserved"].get("gpu_seconds",0)
    assert after["spent"]["gpu_seconds"]-before["spent"]["gpu_seconds"] == pytest.approx(2.)


def test_model_can_exceed_2_9_million_tokens_and_continue_without_global_cap(runtime_setup):
    from optimizer_agent.codex_backend import CodexResult
    runtime,config,_,_ = runtime_setup
    class LargeUsageBackend:
        calls = 0
        def generate_json(self,*args,**kwargs):
            self.calls += 1
            return CodexResult(payload={"ok":True},
                usage={"input_tokens":2_900_000,"output_tokens":100_001},events=[],
                model="gpt-6-luna",reasoning_effort="max",elapsed_seconds=1.,
                call_id=f"large-usage-{self.calls}")
    provider = LargeUsageBackend()
    runtime.backend.backend = provider
    assert runtime.backend.generate_json("first",{},purpose="test").payload["ok"]
    assert runtime.backend.generate_json("second",{},purpose="test").payload["ok"]
    recorded = runtime.store.budget_state(config.run_id)
    assert provider.calls == 2
    assert recorded["spent"]["model_tokens"] == 6_000_002
    assert recorded["spent"]["model_calls"] == 2
    assert set(recorded["limits"]) == {"trials"}
    removed_caps = {"model_tokens","max_model_calls","wall_time_seconds","gpu_lease_seconds","finalization_reserve_fraction"}
    assert not removed_caps.intersection(config.to_dict())


def test_final_receipt_cannot_certify_different_late_incumbent_on_resume(runtime_setup):
    runtime,config,_,_ = runtime_setup
    state = selected(runtime)
    receipt = runtime.evaluator.finalize(runtime.baseline,runtime.baseline)
    campaign.write_json(runtime.directory/"final.json",receipt)
    late = archive_candidate(runtime,state,source=runtime.baseline.source.read_text()+"\n// distinct late artifact\n")
    runtime.store.promote(config.run_id,late.candidate_id,0,{"status":"improved","evidence":["late-confirmation"]})
    outcome = runtime.finalize({"trials_done":1,"termination_reason":"trial_budget"})
    exported = campaign.read_json(runtime.directory/"best-launch.json")
    assert (outcome["certification_status"] == "finalization_incomplete"
            or exported["artifact_hash"] == receipt["artifact_hash"])


# These regressions use the non-CUDA fixture above. They establish receipt
# identity/repair history and recovery, never real kernel performance.
def tuning_request(runtime, state, *, values=(16, 32), constraints=(), bindings=None):
    source = "#ifndef TILE\n#define TILE 16\n#endif\n" + runtime.baseline.source.read_text()
    candidate = archive_candidate(runtime, state, candidate_id="request-template", source=source, parameters={"TILE":16})
    state.update(candidate_id=candidate.candidate_id, repair=state.get("repair", 0),
                 research_round=state.get("research_round") or 1, research_ready=True)
    proposal = {"source": source, "kernel": candidate.kernel, "launch_rule": asdict(candidate.launch_rule),
        "parameter_space": [{"name": "TILE", "values": list(values)}],
        "parameter_bindings": bindings or [{"parameter": "TILE", "target": "source_macro"}],
        "constraints": list(constraints)}
    campaign.write_json(runtime.trial_dir(state) / f'proposal-{state["repair"]}-{state["research_round"]}.json', proposal)
    return candidate, proposal


def completed_requests(runtime, state):
    return sorted((runtime.trial_dir(state) / "evaluation-requests").glob("*/completion.json"))


def test_repair_history_is_immutable_and_collector_counts_no_winner_alias_twice(runtime_setup, tmp_path, monkeypatch):
    import shutil
    from optimizer_agent.suite_orchestrator import observed_metrics
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    tuning_request(runtime, state)
    validate = runtime.evaluator.validate
    monkeypatch.setattr(runtime.evaluator, "validate", lambda *args, **kwargs:
        {"evaluation_id": uuid.uuid4().hex, "ok": False, "status": "compilation_error"})
    first = runtime.evaluate(state)
    assert not first["valid"] and first["retryable"]
    original = completed_requests(runtime, state)[0].parent
    frozen = {p.name: p.read_bytes() for p in original.glob("*.json")}
    state.update(first, repair=1)
    tuning_request(runtime, state, values=(8, 16, 32), constraints=({"op":"le", "parameters":["TILE"], "values":[16]},))
    monkeypatch.setattr(runtime.evaluator, "validate", validate)
    second = runtime.evaluate(state)
    assert second["valid"]
    assert {p.name:p.read_bytes() for p in original.glob("*.json")} == frozen
    records = [campaign.evaluation_record(p.parent) for p in completed_requests(runtime, state)]
    assert {row[0]["repair"] for row in records} == {0, 1}
    assert len({row[1]["attempt_id"] for row in records}) == 2
    latest = campaign.read_json(runtime.trial_dir(state) / "autotuning.json")
    assert latest["receipt_alias"]["authority"] == "non_authoritative_latest_alias"
    assert latest["winner_remeasurement"]["evaluation_id"] == second["evaluation_id"]
    assert latest["winner_selection_evaluation"]["evaluation_id"] != second["evaluation_id"]
    assert latest["winner_validation"]["ok"]
    scope = tmp_path / "study"
    shutil.copytree(runtime.trial_dir(state).parent, scope / "generation/trials")
    metrics = observed_metrics(scope, {})
    assert metrics["configuration_observations"] == 5
    assert metrics["configurations_filtered_before_compilation"] == 1
    assert len(metrics["configuration_observation_sources"]) == 2
    assert metrics["legacy_configuration_snapshots"] == []


@pytest.mark.parametrize("change", ["domain", "binding", "constraint", "compiler_options", "hardware_limit"])
def test_changed_request_same_template_cannot_use_previous_measurement(runtime_setup, change):
    runtime, config, _, _ = runtime_setup
    state = selected(runtime)
    _, proposal = tuning_request(runtime, state)
    first = runtime.evaluate(state)
    before = len(runtime.evaluator.benchmarks)
    if change == "domain":
        proposal["parameter_space"][0]["values"] = [16]
    elif change == "binding":
        proposal["parameter_bindings"][0]["target"] = "tile_x"
    elif change == "constraint":
        proposal["constraints"] = [{"op":"le", "parameters":["TILE"], "values":[16]}]
    elif change == "compiler_options":
        runtime.evaluator._suites["performance"][0].request.compiler_options.append("-DREQUEST_CHANGE=1")
    else:
        runtime.properties["max_threads_per_block"] = 512
    campaign.write_json(runtime.trial_dir(state) / "proposal-0-1.json", proposal)
    second = runtime.evaluate(state)
    assert second["valid"] and first["evaluation_id"] != second["evaluation_id"]
    assert len(runtime.evaluator.benchmarks) > before
    assert len(completed_requests(runtime, state)) == 2
    count = len(runtime.evaluator.benchmarks)
    spent = runtime.store.budget_state(config.run_id)
    measurement_events = [e for e in runtime.store.events(config.run_id) if e["kind"] == "measurement"]
    # Replay both a pre-evaluate template checkpoint and a post-evaluate winner.
    assert runtime.evaluate(state) == second
    assert runtime.evaluate({**state, **second}) == second
    assert len(runtime.evaluator.benchmarks) == count
    assert runtime.store.budget_state(config.run_id) == spent
    assert len([e for e in runtime.store.events(config.run_id) if e["kind"] == "measurement"]) == len(measurement_events)


def test_exact_research_round_is_used_before_any_evaluation_cache(runtime_setup):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    state["research_round"] = 9
    _, proposal = tuning_request(runtime, state, values=(16,))
    proposal["parameter_space"][0]["values"] = [32]
    campaign.write_json(runtime.trial_dir(state) / "proposal-0-10.json", proposal)
    update = runtime.evaluate(state)
    assert runtime.candidate(update["candidate_id"]).parameters == {"TILE":16}


def test_durable_canonical_measurement_survives_failure_before_latest_alias(runtime_setup, monkeypatch):
    runtime, config, runner, model = runtime_setup
    state = selected(runtime)
    tuning_request(runtime, state)
    monkeypatch.setattr(runtime, "_evaluation_aliases", lambda *args: (_ for _ in ()).throw(RuntimeError("alias publication interrupted")))
    with pytest.raises(RuntimeError, match="alias publication"):
        runtime.evaluate(state)
    record = campaign.evaluation_record(completed_requests(runtime, state)[0].parent)
    assert not (runtime.trial_dir(state) / "autotuning.json").exists()
    resumed = campaign.CampaignRuntime(config, runner, runtime.directory, model_backend=model)
    result = resumed.evaluate(state)
    assert result["evaluation_id"] == record[2]["evaluation_id"]
    assert resumed.evaluator.benchmarks == [] and resumed.evaluator.validations == []
    bound = {**state, **result}
    resumed.integrate(bound)
    confirmations = list(resumed.evaluator.confirmations)
    resumed.integrate(bound)
    assert resumed.evaluator.confirmations == confirmations
    rewards = [e for e in resumed.store.events(config.run_id) if e["kind"] == "tree_reward"]
    assert len(rewards) == 1
    assert campaign.read_json(resumed.trial_dir(state)/"result.json")["evaluation_evidence"]["request_id"] == record[1]["request_id"]


def test_new_request_with_only_unbound_legacy_cache_suspends_without_work(runtime_setup):
    runtime, config, _, _ = runtime_setup
    state = selected(runtime)
    candidate, _ = tuning_request(runtime, state)
    record_measurement(runtime, state, candidate)
    path = runtime.trial_dir(state) / f'evaluation-{candidate.candidate_id}.json'
    original = path.read_bytes()
    count = len(runtime.evaluator.benchmarks)
    update = runtime.evaluate(state)
    assert update["evaluation_status"] == "legacy_evaluation_unbound"
    assert not update["retryable"] and update["evaluation_id"] is None
    runtime.integrate({**state, **update})
    assert path.read_bytes() == original and len(runtime.evaluator.benchmarks) == count
    assert runtime.evaluator.confirmations == []
    assert not [e for e in runtime.store.events(config.run_id) if e["kind"] == "tree_reward"]


def test_legacy_bound_integrate_replay_preserves_old_guard_and_one_reward(runtime_setup):
    runtime, config, _, _ = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime, state)
    record_measurement(runtime, state, candidate)
    count = len(runtime.evaluator.benchmarks)
    runtime.integrate(state)
    original = (runtime.trial_dir(state)/"result.json").read_bytes()
    runtime.integrate(state)
    assert (runtime.trial_dir(state)/"result.json").read_bytes() == original
    assert len(runtime.evaluator.benchmarks) == count
    assert len(runtime.evaluator.confirmations) == 1
    assert len([e for e in runtime.store.events(config.run_id) if e["kind"] == "tree_reward"]) == 1


def test_canonical_corruption_and_unfinished_intent_never_rerun_evaluator(runtime_setup):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    tuning_request(runtime, state)
    runtime.evaluate(state)
    directory = completed_requests(runtime, state)[0].parent
    count = len(runtime.evaluator.benchmarks)
    original = (directory / "measurement.json").read_bytes()
    (directory / "measurement.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime.evaluate(state)
    (directory / "measurement.json").write_bytes(original)
    (directory / "completion.json").unlink()
    with pytest.raises(RuntimeError, match="unfinished"):
        runtime.evaluate(state)
    assert len(runtime.evaluator.benchmarks) == count


def test_immutable_publication_refuses_different_content_and_symlink(tmp_path):
    path = tmp_path / "receipt.json"
    campaign.write_immutable_json(path, {"value":1})
    campaign.write_immutable_json(path, {"value":1})
    with pytest.raises(ValueError, match="Immutable"):
        campaign.write_immutable_json(path, {"value":2})
    link = tmp_path / "alias.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="Immutable"):
        campaign.write_immutable_json(link, {"value":1})



def test_preserved_legacy_flat_files_cannot_replace_current_diagnostic_evidence(runtime_setup, tmp_path):
    import shutil
    from optimizer_agent.suite_orchestrator import observed_metrics
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    _, proposal = tuning_request(runtime, state)
    directory = runtime.trial_dir(state)
    legacy = {
        "constraints.json": {"status":"legacy_compiled", "constraints":["stale constraints"]},
        "autotuning.json": {"winner":"old-winner", "configurations":[{"status":"filtered"}]},
        "evaluation-old-winner.json": {"evaluation_id":"old-evaluation", "candidate_id":"old-winner",
                                       "status":"numerical_error", "ok":False}}
    for name, value in legacy.items():
        campaign.write_json(directory/name, value)
    frozen = {name:(directory/name).read_bytes() for name in legacy}
    update = runtime.evaluate(state)
    state.update(update)
    assert update["valid"]
    assert {name:(directory/name).read_bytes() for name in legacy} == frozen
    for name in ("constraints.latest.json", "autotuning.latest.json"):
        assert campaign.read_json(directory/name)["receipt_alias"]["authority"] == "non_authoritative_latest_alias"
    # Even a stale flat winner alias cannot override the checkpoint-bound bytes.
    campaign.write_json(directory/f'evaluation-{state["candidate_id"]}.json', legacy["evaluation-old-winner.json"])
    update = runtime.diagnose(state)
    diagnostic = campaign.read_json(runtime.directory/"diagnostics"/(update["diagnostic_id"]+".json"))
    observed = next(row["value"] for row in diagnostic["observations"] if row["kind"] == "evaluation")
    assert observed["evaluation_id"] == state["evaluation_id"] and observed["ok"]
    refs = {row["kind"]:row["path"] for row in diagnostic["evidence_references"]}
    assert all("/evaluation-requests/" in refs[kind] for kind in ("evaluation", "evaluation_request", "tuning_constraints", "autotuning"))
    evidence = runtime.generation_evidence({**state, "repair":1})
    assert evidence["previous_evaluation"]["evaluation_id"] == state["evaluation_id"]
    assert evidence["previous_evaluation_request"]["request_id"] in refs["evaluation"]
    scope=tmp_path/"legacy-study"
    shutil.copytree(directory.parent, scope/"generation/trials")
    metrics=observed_metrics(scope,{})
    assert metrics["configuration_observations"] == 3
    assert len(metrics["configuration_observation_sources"]) == 1
    assert len(metrics["legacy_configuration_snapshots"]) == 1
    assert "earlier overwritten repairs" in metrics["legacy_configuration_snapshots"][0]["scope"]


def test_canonical_request_directory_cannot_be_replaced_by_external_symlink(runtime_setup, tmp_path):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    tuning_request(runtime, state)
    result = runtime.evaluate(state)
    directory = completed_requests(runtime, state)[0].parent
    external = tmp_path / "external-receipt"
    directory.rename(external)
    directory.symlink_to(external, target_is_directory=True)
    count = len(runtime.evaluator.benchmarks)
    with pytest.raises(ValueError, match="Unsafe canonical"):
        runtime.evaluate({**state, **result})
    assert len(runtime.evaluator.benchmarks) == count


def test_volatile_gpu_properties_do_not_change_a_completed_request(runtime_setup):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    tuning_request(runtime, state)
    result = runtime.evaluate(state)
    count = len(runtime.evaluator.benchmarks)
    runtime.properties.update(free_memory_bytes=1234, temperature=75, observed_at=1000, slurm_job_id="dynamic")
    assert runtime.evaluate(state) == result
    assert len(runtime.evaluator.benchmarks) == count



def test_request_parent_publication_is_durable_before_reservation_and_evaluator(runtime_setup, monkeypatch):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    tuning_request(runtime, state)
    flushed = []
    flush = campaign._fsync_evaluation_directory
    def record_flush(path):
        flush(path)
        flushed.append(path)
    monkeypatch.setattr(campaign, "_fsync_evaluation_directory", record_flush)
    reserve = runtime.store.reserve
    observed = []
    def checked_reserve(run_id, reservation_id, amount, **kwargs):
        if reservation_id.startswith("evaluation-"):
            request_directory = runtime.trial_dir(state) / "evaluation-requests" / reservation_id.removeprefix("evaluation-")
            required = [runtime.trial_dir(state).parent, runtime.trial_dir(state), request_directory.parent, request_directory]
            assert all(path in flushed for path in required)
            assert [flushed.index(path) for path in required] == sorted(flushed.index(path) for path in required)
            assert runtime.evaluator.validations == []
            observed.append(reservation_id)
        return reserve(run_id, reservation_id, amount, **kwargs)
    monkeypatch.setattr(runtime.store, "reserve", checked_reserve)
    assert runtime.evaluate(state)["valid"]
    assert len(observed) == 1


def test_failed_parent_fsync_cannot_reserve_or_evaluate(runtime_setup, monkeypatch):
    runtime, config, _, _ = runtime_setup
    state = selected(runtime)
    tuning_request(runtime, state)
    spent = runtime.store.budget_state(config.run_id)
    count = len(runtime.evaluator.benchmarks)
    def interrupted(path):
        raise OSError("parent fsync interrupted")
    monkeypatch.setattr(campaign, "_fsync_evaluation_directory", interrupted)
    with pytest.raises(OSError, match="parent fsync interrupted"):
        runtime.evaluate(state)
    assert runtime.store.budget_state(config.run_id) == spent
    assert len(runtime.evaluator.benchmarks) == count and runtime.evaluator.validations == []



def test_equivalent_launch_defaults_and_parameter_list_match_candidate(runtime_setup):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    candidate, proposal = tuning_request(runtime, state)
    # Persisted helpers may omit defaults that LaunchRule fills identically.
    for key in ("grid", "shared_memory_bytes", "version"):
        proposal["launch_rule"].pop(key)
    proposal["parameters"] = [{"name":key,"value":value} for key,value in candidate.parameters.items()]
    campaign.write_json(runtime.trial_dir(state)/"proposal-0-1.json",proposal)
    result = runtime.evaluate(state)
    assert result["valid"]
    assert runtime.evaluate(state) == result


def test_local_angle_header_content_change_invalidates_same_template_cache(runtime_setup, tmp_path):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    include = tmp_path/"include"
    include.mkdir()
    header = include/"parameter.cuh"
    header.write_text("#define CONSTANT 1\n")
    source = '#include <parameter.cuh>\n' + runtime.baseline.source.read_text()
    candidate = archive_candidate(runtime, state, "header-template", source=source)
    state.update(candidate_id=candidate.candidate_id,repair=0,research_round=1,research_ready=True)
    for name in ("search","performance"):
        for case in runtime.evaluator._suites[name]:
            case.request.compiler_options = ["-I",str(include)]
    proposal = {"source":source,"kernel":candidate.kernel,"launch_rule":asdict(candidate.launch_rule),
                "parameters":candidate.parameters,"parameter_space":[],"constraints":[]}
    campaign.write_json(runtime.trial_dir(state)/"proposal-0-1.json",proposal)
    first = runtime.evaluate(state)
    before = len(runtime.evaluator.benchmarks)
    header.write_text("#define CONSTANT 2\n")
    second = runtime.evaluate(state)
    assert first["valid"] and second["valid"] and first["evaluation_id"] != second["evaluation_id"]
    assert len(runtime.evaluator.benchmarks) == before+1
    records = [campaign.evaluation_record(p.parent)[0] for p in completed_requests(runtime,state)]
    header_hashes = {manifest["files"][str(header.resolve())] for record in records
                     for manifest in record["compilation_inputs"].values()}
    assert len(header_hashes) == 2
