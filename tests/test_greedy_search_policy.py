"""The no-tree ablation shares evidence rules, not MCTS parent selection."""
import pytest

from optimizer_agent.experiment_store import BudgetExceeded, ExperimentStore, IncompatibleCampaign
from optimizer_agent.greedy_search_policy import GreedySearchPolicy
from optimizer_agent.search_policy import SearchPolicy, stationary_reward
from optimizer_agent.treequest_eligibility import NoEligibleActions


BASELINE = {"candidate_id":"baseline","source":"baseline source","parameters":{"block":32},"valid":True}
ACTIONS = ["tiling","coalescing","work_per_thread"]


def policy(tmp_path,**overrides):
    kwargs = dict(store=ExperimentStore(tmp_path/"greedy.sqlite"),run_id="greedy",baseline_state=BASELINE,
        baseline_latencies={"small":10.,"large":20.},actions=ACTIONS,
        budget_limits={"trials":20,"gpu_seconds":100.},finalization_reserve={"gpu_seconds":10.},
        contract_fingerprint="contract",environment_fingerprint="gpu-a")
    kwargs.update(overrides)
    return GreedySearchPolicy(**kwargs)


def result(candidate_id="fast",evaluation_id="measurement-1",valid=True):
    return {"state":{"candidate_id":candidate_id,"source":candidate_id+" exact source","parameters":{"block":128}},
        "evaluation_id":evaluation_id,"valid":valid,"purpose":"tree_reward","fresh":True,
        "latencies":{"small":5.,"large":10.}}


def tell(search,trial,outcome):
    search.store.archive_candidate(search.run_id,outcome["state"])
    return search.tell_once(trial.trial_id,outcome)


def test_real_greedy_never_creates_treequest_algorithm(tmp_path,monkeypatch):
    from treequest.algos.ab_mcts_a.algo import ABMCTSA
    def forbidden(*args,**kwargs):
        raise AssertionError("The no-tree control must not construct AB-MCTS")
    monkeypatch.setattr(ABMCTSA,"__init__",forbidden)
    search = policy(tmp_path)
    selected = search.ask_batch(3)
    assert [trial.action for trial in selected] == ACTIONS
    assert all(trial.parent_state == BASELINE for trial in selected)
    state = search.snapshot()
    assert state["algorithm"] == "greedy_parent"
    assert "tree" not in state and "rng" not in state


def test_only_confirmed_incumbent_becomes_parent(tmp_path):
    search = policy(tmp_path)
    first = search.ask()
    assert tell(search,first,result())
    # A high search reward is insufficient for promotion under the common
    # evaluator's confirmation protocol.
    assert search.ask().parent_state == BASELINE
    assert search.store.promote("greedy","fast",0,{"status":"improved","evidence":["confirmation-1"]})
    chosen = search.ask()
    assert chosen.parent_state["candidate_id"] == "fast"
    assert chosen.parent_state["source"] == "fast exact source"
    assert chosen.parent_state["parameters"] == {"block":128}
    assert search.snapshot()["observations"][first.trial_id]["reward"] == stationary_reward({"small":10.,"large":20.},{"small":5.,"large":10.})


def test_promoted_candidate_before_tell_is_available_durably(tmp_path):
    search = policy(tmp_path)
    candidate = {"candidate_id":"confirmed-before-tell","source":"source","tree_parent_id":"baseline"}
    search.store.archive_candidate("greedy",candidate)
    search.store.promote("greedy",candidate["candidate_id"],0,{"status":"improved","evidence":["confirmation-1"]})
    assert search.ask().parent_state["candidate_id"] == candidate["candidate_id"]


def test_eligibility_before_creation_and_no_fallback_to_slower_parent(tmp_path):
    search = policy(tmp_path)
    trial = search.ask()
    tell(search,trial,result())
    search.store.promote("greedy","fast",0,{"status":"improved","evidence":["confirmation-1"]})
    search.close("parent_action",("fast","tiling"),"not applicable")
    search.close("strategy_family","coalescing","family budget exhausted")
    batch = search.ask_batch(4)
    assert {item.action for item in batch} == {"work_per_thread"}
    search.close("candidate","fast","greedy incumbent suspended",state="suspended")
    pending_before = len(search.pending_trials())
    with pytest.raises(NoEligibleActions):
        search.ask_batch(3)
    assert len(search.pending_trials()) == pending_before
    with pytest.raises(NoEligibleActions):
        search.assert_eligible(batch[0].trial_id)
    search.close("candidate","fast","resume incumbent",state="active")
    assert search.ask().parent_state["candidate_id"] == "fast"
    search.close("subtree","baseline","all descendants closed")
    with pytest.raises(NoEligibleActions):
        search.ask()


@pytest.mark.parametrize("point",["after_result","after_in_memory_integration"])
def test_crash_replay_once_and_pending_cursor_restored(tmp_path,point):
    search = policy(tmp_path)
    first,second = search.ask_batch(2)
    if point == "after_result":
        search.integrate_pending = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("crash"))
    else:
        search._before_tree_commit = lambda *args: (_ for _ in ()).throw(RuntimeError("crash"))
    with pytest.raises(RuntimeError,match="crash"):
        search.tell_once(first.trial_id,result())
    restored = policy(tmp_path)
    assert list(restored.snapshot()["observations"]) == [first.trial_id]
    assert [trial.trial_id for trial in restored.pending_trials()] == [second.trial_id]
    assert restored.ask().action == "work_per_thread"
    assert restored.tell_once(first.trial_id,result()) is False
    assert restored.store.budget_state("greedy")["spent"]["trials"] == 1
    assert len([event for event in restored.store.events("greedy") if event["kind"] == "tree_reward"]) == 1


def test_cached_measurement_cancelled_and_new_observation_allowed(tmp_path):
    search = policy(tmp_path)
    first,duplicate,independent = search.ask_batch(3)
    assert tell(search,first,result())
    assert not tell(search,duplicate,result())
    assert search.snapshot()["finished"][duplicate.trial_id].trial_status == "INVALID"
    assert tell(search,independent,result(evaluation_id="independent-new-series"))
    assert len(search.snapshot()["observations"]) == 2
    with pytest.raises(ValueError,match="rewritten"):
        search.tell_once(first.trial_id,result(evaluation_id="rewrite"))


@pytest.mark.parametrize("policy_class",[GreedySearchPolicy,SearchPolicy])
def test_reused_evidence_has_same_bytes_despite_distinct_trial_costs(tmp_path,policy_class):
    search = policy_class(ExperimentStore(tmp_path/"evidence.sqlite"),"cost-control",BASELINE,
        {"small":10.,"large":20.},actions=ACTIONS,budget_limits={"trials":3,"gpu_seconds":10})
    first,second = search.ask_batch(2,budget_reservation={"gpu_seconds":2})
    initial = result()
    initial["consumption"] = {"trials":1,"gpu_seconds":.2}
    assert search.tell_once(first.trial_id,initial)
    original_evidence = search.store.get_evaluation(search.run_id,initial["evaluation_id"])
    cached = result()
    cached["consumption"] = {"trials":1,"gpu_seconds":.1}
    cached["state"]["evaluations"] = [{"evaluation_id":"newly-linked-diagnostic"}]
    cached["state"]["current_validity"] = {"valid":True,"reason":"latest applicability check"}
    assert not search.tell_once(second.trial_id,cached)
    assert search.store.get_evaluation(search.run_id,initial["evaluation_id"]) == original_evidence
    assert "consumption" not in original_evidence
    assert search.store.budget_state(search.run_id)["spent"]["gpu_seconds"] == pytest.approx(.3)


def test_budget_rolls_back_selection_and_action_cursor(tmp_path):
    search = policy(tmp_path,budget_limits={"trials":2,"gpu_seconds":100})
    with pytest.raises(BudgetExceeded):
        search.ask_batch(3)
    assert search.pending_trials() == []
    assert search.ask().action == "tiling"
    assert search.ask().action == "coalescing"
    with pytest.raises(BudgetExceeded):
        search.ask()


def test_invalid_and_infrastructure_failures_never_change_incumbent(tmp_path):
    search = policy(tmp_path)
    first,infra = search.ask_batch(2)
    tell(search,first,result(valid=False))
    assert search.snapshot()["observations"][first.trial_id]["reward"] == 0
    with pytest.raises(ValueError,match="Infrastructure"):
        search.tell_once(infra.trial_id,dict(result(),infrastructure_error=True))
    search.suspend_or_cancel(infra.trial_id,"SSH unavailable",consumption={"trials":1})
    assert search.ask().parent_state == BASELINE
    assert search.store.incumbent("greedy")["candidate_id"] is None


def test_resume_rejects_mixed_algorithms_and_hardware(tmp_path):
    search = policy(tmp_path)
    with pytest.raises(IncompatibleCampaign):
        policy(tmp_path,environment_fingerprint="different-gpu")
    with pytest.raises(IncompatibleCampaign):
        SearchPolicy(search.store,"greedy",BASELINE,{"small":10.,"large":20.},actions=ACTIONS,
            contract_fingerprint="contract",environment_fingerprint="gpu-a",
            budget_limits={"trials":20,"gpu_seconds":100.},finalization_reserve={"gpu_seconds":10.})
