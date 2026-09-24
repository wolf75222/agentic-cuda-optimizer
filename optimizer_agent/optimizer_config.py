from dataclasses import dataclass
from pathlib import Path


@dataclass
class OptimizerConfig:
    results_dir: Path
    device: int
    max_iterations: int
    warmup: int
    iterations: int
    model: str
    max_retries: int
    nvidia_research: bool = False
    use_nsight: bool = False
    reasoning_effort: str = "medium"

    def validate(self) -> None:
        if self.reasoning_effort not in ("low", "medium", "high"):
            raise ValueError("reasoning_effort must be low, medium, or high")
        if type(self.nvidia_research) is not bool:
            raise ValueError("nvidia_research must be a boolean")
        if type(self.use_nsight) is not bool:
            raise ValueError("use_nsight must be a boolean")
        for name in ("device", "max_iterations", "warmup", "max_retries"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.iterations) is not int or self.iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not self.model.strip():
            raise ValueError("model must not be empty")
