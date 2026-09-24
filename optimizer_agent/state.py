from pathlib import Path
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from agent_workload import Workload
from optimizer_config import OptimizerConfig
from optimization_history import OptimizationAttempt
from kernel_proposal import KernelProposal


class OptimizerState(TypedDict):
    config: OptimizerConfig
    session_dir: Path
    workload: Workload
    device_properties: dict
    research_brief: str
    kernel_source: str
    kernel_role: Literal["reference", "initial", "candidate"]
    kernel_proposal: KernelProposal | None
    iteration: int
    validation_ok: bool
    reference_results: list[dict]
    reference_attempt: OptimizationAttempt | None
    messages: Annotated[list[AnyMessage], add_messages]
    optimization_history: list[OptimizationAttempt]
    best_attempt: OptimizationAttempt | None
    proposal_retries: int
    model_calls: int
    error: str | None
    profiling_unavailable: str | None
    stop_reason: str | None
    
