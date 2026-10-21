import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from campaign_config import CampaignConfig
from campaign_workflow import build_campaign_graph


class FakeRuntime:
    max_repairs = 2

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def initialize(self, state):
        return {"initialized": True, "trials_done": state.get("trials_done", 0)}

    def select(self, state):
        self.calls.append("select")
        if state["trials_done"] >= 2:
            return {"termination_reason": "trial_budget"}
        return {"trial_id": str(state["trials_done"]), "repair": 0,
                "research_round": 0, "error": None}

    def research(self, state):
        self.calls.append("research")
        return {"research_ready": not self.fail,
                "research_round": state["research_round"] + 1}

    def generate(self, state):
        assert state["research_ready"]
        self.calls.append("generate")
        return {"candidate_id": "c" + state["trial_id"], "error": None,
                "needs_research": False}

    def evaluate(self, state):
        self.calls.append("evaluate")
        return {"valid": state["repair"] == 2, "retryable": True}

    def integrate(self, state):
        self.calls.append("integrate")
        return {"trials_done": state["trials_done"] + 1}

    def diagnose(self, state):
        return {}

    def finalize(self, state):
        return {"finished": True, "certification_status": "uncertified"}


def test_research_blocks_generation_and_exploration_continues(tmp_path):
    runtime = FakeRuntime(fail=True)
    graph = build_campaign_graph(runtime, SqliteSaver(sqlite3.connect(tmp_path / "g.db", check_same_thread=False)))
    state = graph.invoke({"run_id": "test"}, {"configurable": {"thread_id": "test"}})
    assert state["finished"] and state["trials_done"] == 2
    assert "generate" not in runtime.calls


def test_repairs_bounded_per_trial_and_checkpoint_replay(tmp_path):
    runtime = FakeRuntime()
    graph = build_campaign_graph(runtime, SqliteSaver(sqlite3.connect(tmp_path / "g.db", check_same_thread=False)))
    config = {"configurable": {"thread_id": "test"}, "recursion_limit": 100}
    state = graph.invoke({"run_id": "test"}, config)
    assert runtime.calls.count("generate") == 6
    assert state["trials_done"] == 2 and state["finished"]
    saved = graph.get_state(config)
    assert not saved.next
    assert saved.values["termination_reason"] == "trial_budget"


def test_normal_config_cannot_disable_research():
    with pytest.raises(ValueError, match="require code"):
        CampaignConfig(research_required=False)
    assert CampaignConfig(research_required=False, experimental_mode="ablation_no_research")


def test_config_stable_safe_identifiers():
    with pytest.raises(ValueError, match="run_id"):
        CampaignConfig(run_id="../escape")
    with pytest.raises(ValueError, match="positive integer"):
        CampaignConfig(max_trials=0)
