"""Acceptance tests use real pinned TreeQuest; no replacement search engine."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import sqlite3

import pytest

from optimizer_agent.autotuning import Autotuner
from optimizer_agent.experiment_store import BudgetExceeded, ExperimentStore, IncompatibleCampaign
from optimizer_agent.search_policy import SearchPolicy, stationary_reward, verify_treequest_pin
from optimizer_agent.treequest_eligibility import NoEligibleActions


BASELINE = {"candidate_id":"baseline","source":"baseline exact source","parameters":{"block":64},"valid":True}


def make_policy(tmp_path,**overrides):
    store = ExperimentStore(tmp_path/"campaign.sqlite")
    options = dict(store=store,run_id="campaign",baseline_state=BASELINE,
                   baseline_latencies={"small":10.,"large":20.},actions=["tile","vector"],
                   budget_limits={"trials":50},contract_fingerprint="contract-v1",
                   environment_fingerprint="gpu-a",seed=7)
    options.update(overrides)
    return SearchPolicy(**options)


def result(candidate_id="candidate",evaluation_id="measurement-1",latencies=None):
    return {"state":{"candidate_id":candidate_id,"source":candidate_id+" exact source","parameters":{"block":128}},
            "evaluation_id":evaluation_id,"valid":True,"purpose":"tree_reward","fresh":True,
            "latencies":latencies or {"small":8.,"large":16.}}


def test_upstream_pin_and_exact_slower_parent(tmp_path):
    verify_treequest_pin()
    policy = make_policy(tmp_path)
    first = policy.ask()
    assert first.parent_state == BASELINE
    policy.tell_once(first.trial_id,result("slow","slow-eval",{"small":12.,"large":24.}))
    policy.close("candidate","slow","temporarily force root for a second branch",state="suspended")
    fast_trial = policy.ask()
    assert fast_trial.parent_state == BASELINE
    policy.tell_once(fast_trial.trial_id,result("fast","fast-eval"))
    policy.close("candidate","slow","resume slower valid parent",state="active")
    policy.close("candidate","baseline","root generation complete")
    policy.close("candidate","fast","force independent slower-parent coverage",state="suspended")
    chosen = policy.ask()
    assert chosen.parent_state == dict(result("slow")["state"],valid=True)
    assert chosen.parent_state["source"] == "slow exact source"
    assert chosen.parent_state["parameters"] == {"block":128}
    assert policy.snapshot()["tree"].__class__.__module__.startswith("treequest.")


def test_scope_masks_before_trial_creation_and_batch(tmp_path):
    policy = make_policy(tmp_path)
    policy.close("parent_action",("baseline","tile"),"impossible alignment")
    trials = policy.ask_batch(4)
    assert {trial.action for trial in trials} == {"vector"}
    for index,trial in enumerate(trials):
        policy.tell_once(trial.trial_id,result(f"child-{index}",f"e-{index}"))
    policy.close("subtree","child-0","local stagnation")
    policy.close("strategy_family","vector","configured family budget exhausted")
    # Other descendants can still use tile, despite the root/action closure.
    remaining = policy.ask_batch(4)
    assert all(trial.action == "tile" and trial.parent_state["candidate_id"] != "child-0" for trial in remaining)
    policy.close("subtree","baseline","all search funds exhausted")
    before = len(policy.pending_trials())
    with pytest.raises(NoEligibleActions):
        policy.ask_batch(3)
    assert len(policy.pending_trials()) == before
    with pytest.raises(NoEligibleActions):
        policy.assert_eligible(remaining[0].trial_id)


def test_stationary_score_and_small_gains():
    assert stationary_reward({"a":10},{"a":10}) == .5
    first = stationary_reward({"a":10,"b":20},{"a":8,"b":16},{"a":.25,"b":.75})
    second = stationary_reward({"a":10,"b":20},{"a":8,"b":16},{"a":.25,"b":.75})
    assert first == second
    assert stationary_reward({"a":10},{"a":9.999}) > .5
    assert stationary_reward({"a":10},{"a":9.998}) > stationary_reward({"a":10},{"a":9.999})
    for bad in (0,-1,float("nan"),float("inf")):
        with pytest.raises(ValueError):
            stationary_reward({"a":10},{"a":bad})


def test_invalid_kernel_does_not_close_valid_parent(tmp_path):
    policy = make_policy(tmp_path)
    trial = policy.ask()
    evidence = result()
    evidence["valid"] = False
    policy.tell_once(trial.trial_id,evidence)
    child = policy.snapshot()["tree"].tree.get_nodes()[1]
    assert child.score == 0
    assert child.state["valid"] is False
    assert all(trial.parent_state["candidate_id"] == "baseline" for trial in policy.ask_batch(4))
    infrastructure = policy.ask()
    with pytest.raises(ValueError,match="Infrastructure"):
        policy.tell_once(infrastructure.trial_id,dict(evidence,infrastructure_error="SSH unavailable"))
    policy.suspend_or_cancel(infrastructure.trial_id,"SSH unavailable",consumption={"trials":1})
    assert len(policy.snapshot()["tree"].tree) == 2


@pytest.mark.parametrize("failure_point",["after_result","after_tell"])
def test_crash_replay_is_atomic_once_and_restores_pending(tmp_path,failure_point):
    policy = make_policy(tmp_path)
    first,second = policy.ask_batch(2)
    if failure_point == "after_result":
        policy.integrate_pending = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("simulated crash"))
    else:
        policy._before_tree_commit = lambda *args: (_ for _ in ()).throw(RuntimeError("simulated crash"))
    with pytest.raises(RuntimeError,match="simulated crash"):
        policy.tell_once(first.trial_id,result())
    restored = make_policy(tmp_path)
    assert len(restored.snapshot()["tree"].tree) == 2
    assert [trial.trial_id for trial in restored.pending_trials()] == [second.trial_id]
    assert restored.tell_once(first.trial_id,result()) is False
    assert len(restored.snapshot()["tree"].tree) == 2
    assert restored.store.budget_state("campaign")["spent"]["trials"] == 1
    assert sum(event["kind"] == "tree_reward" for event in restored.store.events("campaign")) == 1


def test_measurement_dedup_and_intentional_remeasurement(tmp_path):
    policy = make_policy(tmp_path)
    first,duplicate,new = policy.ask_batch(3)
    assert policy.tell_once(first.trial_id,result()) is True
    assert policy.tell_once(duplicate.trial_id,result()) is False
    assert len(policy.snapshot()["tree"].tree) == 2
    assert policy.snapshot()["tree"].trial_store.finished_trials[duplicate.trial_id].trial_status == "INVALID"
    assert policy.tell_once(new.trial_id,result(evaluation_id="independent-new-measurement")) is True
    assert len(policy.snapshot()["tree"].tree) == 3
    with pytest.raises(ValueError,match="rewritten"):
        policy.tell_once(first.trial_id,result(evaluation_id="incorrect-score-rewrite"))


def test_cancel_result_ready_and_late_result_never_reopens_trial(tmp_path):
    policy = make_policy(tmp_path)
    trial = policy.ask()
    integrate = policy.integrate_pending
    policy.integrate_pending = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("before integration"))
    with pytest.raises(RuntimeError):
        policy.tell_once(trial.trial_id,result())
    policy.integrate_pending = integrate
    policy.suspend_or_cancel(trial.trial_id,"result became inapplicable",consumption={"trials":1})
    restored = make_policy(tmp_path)
    assert len(restored.snapshot()["tree"].tree) == 1
    assert restored.tell_once(trial.trial_id,result()) is False
    assert len(restored.pending_trials()) == 0
    assert not any(event["kind"] == "tree_reward" for event in restored.store.events("campaign"))


def test_rng_restored_and_global_rng_untouched(tmp_path):
    import numpy as np
    policy = make_policy(tmp_path)
    for i in range(4):
        trial = policy.ask()
        policy.tell_once(trial.trial_id,result(f"c-{i}",f"e-{i}"))
    copy_path = tmp_path/"copy"
    copy_path.mkdir()
    with sqlite3.connect(policy.store.path) as original,sqlite3.connect(copy_path/"campaign.sqlite") as copied:
        original.backup(copied)
    restored = make_policy(copy_path)
    np.random.seed(918)
    original_rng = np.random.get_state()
    a,b = policy.ask(),restored.ask()
    assert (a.node_to_expand,a.action,a.parent_state) == (b.node_to_expand,b.action,b.parent_state)
    current_rng = np.random.get_state()
    assert original_rng[0] == current_rng[0]
    assert np.array_equal(original_rng[1],current_rng[1])
    assert original_rng[2:] == current_rng[2:]


def test_budget_reservation_rolls_back_ask_and_parallel_writers(tmp_path):
    policy = make_policy(tmp_path,budget_limits={"trials":3})
    with pytest.raises(BudgetExceeded):
        policy.ask_batch(4)
    assert len(policy.pending_trials()) == 0
    other = make_policy(tmp_path,budget_limits={"trials":3})
    def ask_once(index):
        try:
            return (policy if index%2 else other).ask().trial_id
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes = list(workers.map(ask_once,range(8)))
    assert sum(value is not None for value in outcomes) == 3
    assert len(policy.pending_trials()) == 3


def test_finalization_protected_compatibility_and_atomic_promotion(tmp_path):
    policy = make_policy(tmp_path,budget_limits={"trials":10,"gpu_seconds":10},finalization_reserve={"gpu_seconds":3})
    policy.store.reserve("campaign","search",{"gpu_seconds":7})
    with pytest.raises(BudgetExceeded):
        policy.store.reserve("campaign","too-much",{"gpu_seconds":.01})
    policy.store.reserve_finalization("campaign","final",{"gpu_seconds":3})
    assert policy.store.budget_state("campaign")["reserved"]["gpu_seconds"] == 10
    restored = make_policy(tmp_path,budget_limits={"trials":10,"gpu_seconds":10},finalization_reserve={"gpu_seconds":3})
    assert restored.store.budget_state("campaign")["reserved"]["gpu_seconds"] == 10
    with pytest.raises(IncompatibleCampaign):
        make_policy(tmp_path,environment_fingerprint="different-gpu")
    policy.store.archive_candidate("campaign",{"candidate_id":"winner","source":"kernel"})
    decision = {"status":"improved","evidence":["confirmation-1"]}
    assert policy.store.promote("campaign","winner",0,decision)
    assert not policy.store.promote("campaign","baseline",0,decision)
    assert policy.store.incumbent("campaign") == {"candidate_id":"winner","version":1}


def test_immutable_sources_and_gpu_expiry_requires_stopped_worker(tmp_path):
    import time
    policy = make_policy(tmp_path)
    source = policy.store.get_candidate_source("baseline","campaign")
    assert source["source"] == BASELINE["source"]
    with pytest.raises(ValueError,match="immutable"):
        policy.store.archive_candidate("campaign",dict(BASELINE,source="changed"))
    with pytest.raises(ValueError,match="escapes"):
        policy.store.atomic_artifact("../../outside",b"no")
    policy.store.reserve("campaign","gpu-worker",{"gpu_seconds":8},expires_at=time.time()-1)
    with pytest.raises(ValueError,match="worker is stopped"):
        policy.store.expire_reservation("campaign","gpu-worker",{"gpu_seconds":6})
    assert policy.store.budget_state("campaign")["reserved"]["gpu_seconds"] == 8
    policy.store.expire_reservation("campaign","gpu-worker",{"gpu_seconds":6},worker_confirmed_stopped=True)
    assert policy.store.budget_state("campaign")["spent"]["gpu_seconds"] == 6


def test_finalization_transfer_includes_free_funds_once_without_release_window(tmp_path):
    search = make_policy(tmp_path,budget_limits={"trials":10,"gpu_seconds":10},finalization_reserve={"gpu_seconds":3})
    search.store.reserve("campaign","search-stage",{"gpu_seconds":4})
    search.store.reconcile("campaign","search-stage",{"gpu_seconds":2})
    reserved = search.store.reserve_finalization_remaining("campaign","final-stage")
    assert reserved == {"gpu_seconds":8}
    assert search.store.reserve_finalization_remaining("campaign","final-stage") == reserved
    with pytest.raises(BudgetExceeded):
        search.store.reserve("campaign","concurrent-search",{"gpu_seconds":.01})
    restored = make_policy(tmp_path,budget_limits={"trials":10,"gpu_seconds":10},finalization_reserve={"gpu_seconds":3})
    assert restored.store.budget_state("campaign")["reserved"]["gpu_seconds"] == 8
    search.store.reconcile("campaign","final-stage",{"gpu_seconds":1})
    assert search.store.budget_state("campaign")["spent"]["gpu_seconds"] == 3


def test_late_invalidity_preserves_score_but_masks_parent(tmp_path):
    policy = make_policy(tmp_path)
    trial = policy.ask()
    policy.tell_once(trial.trial_id,result())
    score = policy.snapshot()["tree"].tree.get_nodes()[1].score
    policy.set_candidate_validity("candidate",False,"new correctness failure")
    assert policy.snapshot()["tree"].tree.get_nodes()[1].score == score
    assert all(trial.parent_state["candidate_id"] != "candidate" for trial in policy.ask_batch(5))


def test_tuning_filters_validates_and_remeasures_winner():
    class Evaluator:
        def __init__(self):
            self.validations,self.measurements = [],[]
        def validate(self,candidate,suite):
            assert suite == "search"
            self.validations.append(candidate["parameters"]["block"])
            return {"ok":candidate["parameters"]["block"] != 64,"status":"correctness"}
        def benchmark(self,candidate):
            self.measurements.append(candidate["parameters"]["block"])
            return {"evaluation_id":f"measurement-{len(self.measurements)}","cases":{"a":{"latency_ms":128/candidate["parameters"]["block"]}}}
    evaluator = Evaluator()
    def materialize(template,parameters):
        return {"candidate_id":str(parameters["block"]),"parameters":parameters,
                "launch_rule":{"block":[parameters["block"],1,1]}}
    tuner = Autotuner(evaluator,materialize=materialize)
    outcome = tuner.search({}, {"block":[2048,64,128,256,512]},3)
    assert [observation.status for observation in outcome.configurations] == ["filtered","invalid","measured"]
    assert evaluator.validations == [64,128]
    assert evaluator.measurements == [128,128]
    assert outcome.remeasurement["evaluation_id"] != outcome.evaluation["evaluation_id"]
    assert outcome.termination_reason == "configuration_budget"


def test_tuning_explicit_source_and_launch_bindings(tmp_path):
    tuner = Autotuner(None)
    template = {"source":"#ifndef TILE\n#define TILE 16\n#endif\n__global__ void k() {}\n", "kernel":"k",
                "launch_rule":{"kind":"square_2d","block":(16,16,1),"tile_x":16,"tile_y":16},
                "parameter_bindings":[{"parameter":"TILE","target":"source_macro"},
                                      {"parameter":"TILE","target":"block_x"},
                                      {"parameter":"TILE","target":"tile_x"}]}
    concrete = tuner.materialize(template,{"TILE":32})
    assert "#define TILE 32" in concrete.source.read_text()
    assert concrete.launch_rule.block == (32,16,1)
    assert concrete.launch_rule.tile_x == 32
    assert concrete.parameters == {"TILE":32}
    with pytest.raises(ValueError,match="one existing"):
        tuner.materialize(dict(template,source="__global__ void k() {}"),{"TILE":32})
