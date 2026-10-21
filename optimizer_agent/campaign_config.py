"""Validated, serializable settings for the local Codex/LangGraph campaign."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class CampaignConfig:
    schema_version: int = 1
    run_id: str = "gemm-luna"
    results_dir: str = "results"
    backend: str = "romeo"
    execution_config: str = "configs/romeo.json"
    workload: str = "matmul"
    baseline_source: str = "examples/naive_matmul.cu"
    model: str = "gpt-6-luna"
    reasoning_effort: str = "max"
    max_trials: int = 3
    max_repairs_per_trial: int = 2
    max_configurations_per_trial: int = 4
    model_timeout_seconds: int = 600
    research_required: bool = True
    web_search_required: bool = True
    experimental_mode: str | None = None
    autotuning_enabled: bool = True
    actions: tuple[str, ...] = ("tiling", "coalescing", "work_per_thread")
    search_sizes: tuple[int, ...] = (64, 128, 256, 512)
    final_sizes: tuple[int, ...] = (96, 192, 384)
    benchmark_series: int = 7
    confirmation_batches: int = 20
    launches_per_series: int = 20
    warmup: int = 3
    confidence_level: float = .95
    min_relative_gain: float = .01
    max_critical_regression: float = .05
    required_sanitizers: tuple[str, ...] = ("memcheck",)
    profile: bool = True
    local_profile_harness: str | None = None
    seed: int = 1729

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported configuration schema_version")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.run_id):
            raise ValueError("run_id must be a safe, stable identifier")
        if self.backend not in ("romeo", "local"):
            raise ValueError("backend must be romeo or local")
        if self.experimental_mode not in (None, "vanilla", "ablation_no_research", "ablation_no_tree", "ablation_no_pmpp", "test"):
            raise ValueError("Unknown experimental mode")
        if self.experimental_mode is None and not (self.research_required and self.web_search_required):
            raise ValueError("Normal campaigns require code, documentation, web search and PMPP")
        if not self.actions or len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be nonempty and unique")
        if self.reasoning_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("Invalid reasoning effort")
        for name in ("max_trials", "max_configurations_per_trial", "model_timeout_seconds", "benchmark_series", "confirmation_batches", "launches_per_series"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_repairs_per_trial < 0 or self.warmup < 0:
            raise ValueError("Repair and warmup counts must be nonnegative")
        if not .5 < self.confidence_level < 1 or self.min_relative_gain < 0 or self.max_critical_regression < 0:
            raise ValueError("Invalid fixed measurement policy")
        if not self.search_sizes or not self.final_sizes or any(type(n) is not int or n < 1 for n in self.search_sizes + self.final_sizes):
            raise ValueError("Workload sizes must be positive integers")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        # Compatibility with early fork configurations. Resource usage is
        # recorded; these removed global ceilings never gate execution.
        for name in ("max_model_calls", "model_tokens", "wall_time_seconds", "gpu_lease_seconds", "finalization_reserve_fraction", "max_profiles", "max_profile_seconds"):
            data.pop(name, None)
        for name in ("actions", "search_sizes", "final_sizes", "required_sanitizers"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)
