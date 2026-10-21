"""Campaign control flow; search tree and artifacts live in the durable ledger.

The runtime owns deterministic guards and services. Checkpoints contain stable
identifiers rather than CUDA sources, raw outputs, or model conversation copies.
"""
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph


class CampaignState(TypedDict, total=False):
    run_id: str
    initialized: bool
    trials_done: int
    trial_id: str | None
    candidate_id: str | None
    repair: int
    research_round: int
    research_ready: bool
    needs_research: bool
    valid: bool
    retryable: bool
    error: str | None
    termination_reason: str | None
    certification_status: str | None
    finished: bool
    diagnostic_id: str | None
    research_refresh_required: bool


class Runtime(Protocol):
    max_repairs: int
    def initialize(self, state: CampaignState) -> dict: ...
    def select(self, state: CampaignState) -> dict: ...
    def research(self, state: CampaignState) -> dict: ...
    def generate(self, state: CampaignState) -> dict: ...
    def evaluate(self, state: CampaignState) -> dict: ...
    def integrate(self, state: CampaignState) -> dict: ...
    def diagnose(self, state: CampaignState) -> dict: ...
    def finalize(self, state: CampaignState) -> dict: ...


def build_experiment_graph(runtime: Runtime):
    graph = StateGraph(CampaignState)
    graph.add_node("generate", runtime.generate)
    graph.add_node("evaluate", runtime.evaluate)

    def repair(state):
        return {"repair": state.get("repair", 0) + 1}

    def after_generation(state):
        if state.get("needs_research") or state.get("error"):
            return "end"
        return "evaluate"

    def after_evaluation(state):
        if (not state.get("valid") and state.get("retryable")
                and state.get("repair", 0) < runtime.max_repairs):
            return "repair"
        return "end"

    graph.add_node("repair", repair)
    graph.add_edge(START, "generate")
    graph.add_conditional_edges("generate", after_generation, {"evaluate": "evaluate", "end": END})
    graph.add_conditional_edges("evaluate", after_evaluation, {"repair": "repair", "end": END})
    graph.add_edge("repair", "generate")
    return graph.compile()


def build_campaign_graph(runtime: Runtime, checkpointer: Any):
    graph = StateGraph(CampaignState)
    graph.add_node("initialize", runtime.initialize)
    graph.add_node("select", runtime.select)
    graph.add_node("research", runtime.research)
    graph.add_node("experiment", build_experiment_graph(runtime))
    graph.add_node("integrate", runtime.integrate)
    graph.add_node("diagnose", runtime.diagnose)
    graph.add_node("diagnose_research", runtime.diagnose)
    graph.add_node("finalize", runtime.finalize)
    graph.add_edge(START, "initialize")

    def initialized(state):
        if state.get("finished"):
            return "end"
        return "finalize" if state.get("termination_reason") else "select"

    def selected(state):
        return "finalize" if state.get("termination_reason") else "research"

    def researched(state):
        return "experiment" if state.get("research_ready") else "integrate"

    def experimented(state):
        # One additional dossier can cover a newly declared technique. Failure
        # to support a second proposal suspends this trial instead of spinning.
        if state.get("needs_research") and state.get("research_round", 0) < 2:
            return "diagnose_research"
        return "integrate"

    def diagnosed(state):
        return "finalize" if state.get("termination_reason") else "select"

    graph.add_conditional_edges("initialize", initialized, {"end": END, "finalize": "finalize", "select": "select"})
    graph.add_conditional_edges("select", selected, {"finalize": "finalize", "research": "research"})
    graph.add_conditional_edges("research", researched, {"experiment": "experiment", "integrate": "integrate"})
    graph.add_conditional_edges("experiment", experimented, {"diagnose_research": "diagnose_research", "integrate": "integrate"})
    graph.add_edge("diagnose_research", "research")
    graph.add_edge("integrate", "diagnose")
    graph.add_conditional_edges("diagnose", diagnosed, {"finalize": "finalize", "select": "select"})
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
