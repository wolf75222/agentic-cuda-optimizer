from pydantic import BaseModel, Field, TypeAdapter
from math import prod
from typing import Literal


class LaunchConfig(BaseModel):
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory_bytes: int = Field(description="Dynamic shared-memory bytes only. Static __shared__ arrays are allocated by the compiler; do not include them. Use 0 if no dynamic storage is consumed.")


class CaseLaunch(BaseModel):
    case_index: int
    config: LaunchConfig
    dynamic_shared_memory_reason: str = Field(description="Identify the dynamic shared-memory declaration and byte formula for this case, or explain why no dynamic allocation is needed. Exclude static arrays.")


class LaunchPlan(BaseModel):
    launches: list[CaseLaunch]


class CaseExecutionPath(BaseModel):
    case_index: int
    active_path: str
    condition_evaluation: str
    executes_changed_code: bool


class CounterEvidence(BaseModel):
    iteration: int
    case_index: int
    metric_name: str
    value: str
    unit: str


class HardwareClaim(BaseModel):
    claim: str
    status: Literal["measured", "unverified"]
    counters: list[CounterEvidence]
    interpretation: str


class OptimizationHypothesis(BaseModel):
    change_summary: str = Field(description="Two to eight words describing the actual change, not a claimed speedup.")
    starting_iteration: int
    supporting_measurements: list[str]
    proposed_change: str
    expected_benefit: str
    case_execution_paths: list[CaseExecutionPath]
    hardware_claims: list[HardwareClaim]


class KernelProposal(BaseModel):
    kernel_source: str
    launches: list[CaseLaunch]
    hypothesis: OptimizationHypothesis | None = None


class StopOptimization(BaseModel):
    model_config = {"extra": "forbid"}
    action: Literal["stop"]
    reason: str


optimization_decision = TypeAdapter(KernelProposal | StopOptimization)


def validate_launches(launches: list[CaseLaunch], case_count: int, device: dict) -> None:
    if sorted(launch.case_index for launch in launches) != list(range(case_count)):
        raise ValueError("Each input case needs exactly one launch config")
    for launch in launches:
        if not launch.dynamic_shared_memory_reason.strip():
            raise ValueError(f"Case {launch.case_index}: explain the dynamic shared-memory allocation")
        config = launch.config
        for name, dimensions, limits in (
            ("grid", config.grid, device["max_grid_dimensions"]),
            ("block", config.block, device["max_block_dimensions"]),
        ):
            if any(value <= 0 or value > limit for value, limit in zip(dimensions, limits)):
                raise ValueError(f"Case {launch.case_index}: {name} dimensions exceed device limits")
        if prod(config.block) > device["max_threads_per_block"]:
            raise ValueError(f"Case {launch.case_index}: too many threads per block")
        if not 0 <= config.shared_memory_bytes <= device["max_shared_memory_per_block_bytes"]:
            raise ValueError(f"Case {launch.case_index}: dynamic shared memory exceeds the non-opt-in limit")
