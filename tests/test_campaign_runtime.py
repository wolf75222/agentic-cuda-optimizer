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


def archive_candidate(runtime,state,candidate_id="fast",source=None):
    path = runtime.directory/(candidate_id+".cu")
    path.write_text(source or runtime.baseline.source.read_text())
    candidate = campaign.Candidate(candidate_id,path,runtime.config.workload,runtime.baseline.launch_rule)
    tree_parent_id = runtime.trial(state)["parent"]["candidate_id"]
    runtime.store.archive_candidate(runtime.config.run_id,runtime.candidate_state(candidate,
        tree_parent_id=tree_parent_id,source_parent_id=tree_parent_id,trial_id=state["trial_id"]))
    return candidate


def test_confirmation_and_promotion_replay_reuses_durable_evidence(runtime_setup):
    runtime,config,runner,model = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime,state)
    state.update(candidate_id=candidate.candidate_id,valid=True)
    campaign.write_json(runtime.trial_dir(state)/"evaluation-fast.json",runtime.evaluator.benchmark(candidate))
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


def test_runtime_tuner_uses_exact_variant_and_fresh_reward_measurement(runtime_setup):
    runtime,_,_,_ = runtime_setup
    state = selected(runtime)
    source = "#ifndef TILE\n#define TILE 16\n#endif\n"+runtime.baseline.source.read_text()
    candidate = archive_candidate(runtime,state,source=source)
    state.update(candidate_id=candidate.candidate_id,repair=0,research_round=1)
    proposal = {"source":source,"kernel":runtime.config.workload,"launch_rule":asdict(candidate.launch_rule),
                "parameter_space":[{"name":"TILE","values":[16,32]}],
                "parameter_bindings":[{"parameter":"TILE","target":"source_macro"}]}
    campaign.write_json(runtime.trial_dir(state)/"proposal-0-1.json",proposal)
    outcome = runtime.evaluate(state)
    assert outcome["valid"] and outcome["candidate_id"] != "fast"
    winner = runtime.candidate(outcome["candidate_id"])
    assert winner.parameters == {"TILE":32}
    assert "#define TILE 32" in winner.source.read_text()
    assert runtime.evaluator.benchmarks.count(winner.candidate_id) == 2
    stored = runtime.store.get_candidate_source(winner.candidate_id,runtime.config.run_id)
    assert stored["tree_parent_id"] == "baseline" and stored["source_parent_id"] == "fast"


def test_same_tuned_executable_in_distinct_trials_preserves_both_lineages(runtime_setup):
    runtime,_,_,_ = runtime_setup
    source = "#ifndef TILE\n#define TILE 16\n#endif\n"+runtime.baseline.source.read_text()
    winners = []
    for index in range(2):
        state = {"run_id":runtime.config.run_id,"trials_done":index,**runtime.select({"trials_done":index})}
        candidate = archive_candidate(runtime,state,candidate_id=f"initial-{index}",source=source)
        state.update(candidate_id=candidate.candidate_id,repair=0,research_round=1)
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
    first.update(candidate_id=candidate.candidate_id,valid=True)
    campaign.write_json(control.trial_dir(first)/"evaluation-fast.json",control.evaluator.benchmark(candidate))
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
    state.update(candidate_id=candidate.candidate_id,valid=True)
    campaign.write_json(runtime.trial_dir(state)/"evaluation-fast.json",runtime.evaluator.benchmark(candidate))
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
