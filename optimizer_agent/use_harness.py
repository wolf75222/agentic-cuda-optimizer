import json
import subprocess
from pathlib import Path
import sys
from dataclasses import dataclass, field, asdict
from enum import StrEnum

class BufferDirection(StrEnum):
    Input = "input"
    Output = "output"
    InOut = "inout"

@dataclass
class BufferArgument:
    bytes: int
    input_file: str | None
    direction: BufferDirection
    kind: str = field(default="buffer", init=False)

@dataclass
class ScalarArgument:
    type: str
    value: int | float
    kind: str = field(default="scalar", init=False)


def parse_arguments(data: list[dict]) -> list[BufferArgument | ScalarArgument]:
    arguments = []
    for item in data:
        values = item.copy()
        kind = values.pop("kind")
        if kind == "buffer":
            values["direction"] = BufferDirection(values["direction"])
            argument = BufferArgument(**values)
        elif kind == "scalar":
            argument = ScalarArgument(**values)
        else:
            raise ValueError(f"Unknown argument kind: {kind}")
        arguments.append(argument)
    return arguments


@dataclass
class KernelRequest:
    source: str
    kernel: str
    mode: str
    device: int
    grid: list[int]
    block: list[int]
    shared_memory_bytes: int
    warmup: int
    iterations: int
    arguments: list[BufferArgument | ScalarArgument]
    series: int = 1
    compiler_options: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    cache_dir: str | None = None
    compile_cache_key: str | None = None
    guard_outputs: bool = True
    sentinel_byte: int = 255

    @classmethod
    def from_dict(cls, data:dict) -> "KernelRequest":
        fields = data.copy()
        fields["arguments"] = parse_arguments(fields["arguments"])
        return cls(**fields)

    @staticmethod
    def load_request(path: Path) -> tuple[str, Path, "KernelRequest"]:
        path = path.resolve()

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("The request JSON must contain an object")
        
        run_id = data.pop("id")
        results_dir = (path.parent / data.pop("results_dir")).resolve()
        request = KernelRequest.from_dict(data)
        request.source = str((path.parent / request.source).resolve())

        for argument in request.arguments:
            if isinstance(argument, BufferArgument) and argument.input_file is not None:
                argument.input_file = str(
                    (path.parent / argument.input_file).resolve()
                )
        return run_id, results_dir, request

project_dir = Path(__file__).resolve().parent.parent
if sys.platform == "win32":
    executable = (
        project_dir 
        / "cuda_test_harness"
        / "build"
        / "Release"
        / "cuda_test_harness.exe"
    )
else:
    executable = (
        project_dir 
        / "cuda_test_harness"
        / "build"
        / "cuda_test_harness"
    )

class HarnessRuner:
    def __init__(self, executable: Path, results_dir: Path):
        self.executable = executable.resolve()
        self.results_dir = results_dir.resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write_request(self, run_id: str, request: KernelRequest) -> Path:
        request_dir = self.results_dir / "requests"
        request_dir.mkdir(parents=True, exist_ok=True)

        request_path = request_dir / f"{run_id}.json"

        payload = {
            **asdict(request),
            "id":run_id,
            "results_dir":str(self.results_dir),
        }

        with request_path.open("x",encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        return request_path

    def run(self, run_id : str, request: KernelRequest) -> dict:
        request_path = self.write_request(run_id, request)

        try:
            process = subprocess.run(
                [str(self.executable), "--request", str(request_path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                timeout=1200,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "execute", "error": "Harness timed out after 1200 seconds", "outputs": [], "latency_ms": None}
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "stage": "harness", "error": f"Harness did not return JSON: {process.stderr or process.stdout}", "outputs": [], "latency_ms": None}
        if process.returncode != 0:
            result["ok"] = False
            result.setdefault("error", f"Harness exited with code {process.returncode}")
        return result

    def run_sanitizer(self, run_id: str, request: KernelRequest, tool: str) -> dict:
        """Instrument a fresh request; instrumented timings never become a score."""
        import shutil
        if tool not in {"memcheck", "racecheck", "initcheck", "synccheck"}:
            raise ValueError("Unsupported Compute Sanitizer tool")
        sanitizer = shutil.which("compute-sanitizer")
        if not sanitizer:
            return {"ok": False, "status": "tool_unavailable", "tool": tool}
        request_path = self.write_request(run_id, request)
        try:
            process = subprocess.run(
                [sanitizer, "--tool", tool, "--error-exitcode", "91", str(self.executable), "--request", str(request_path)],
                capture_output=True, text=True, timeout=1200, check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "status": "infrastructure_failure", "tool": tool, "error": "sanitizer timeout"}
        return {"ok": process.returncode == 0, "status": "passed" if process.returncode == 0 else "memory_error",
                "tool": tool, "returncode": process.returncode, "stdout": process.stdout[-16000:], "stderr": process.stderr[-16000:]}

    def get_device_properties(self, device: int)->dict:
        process = subprocess.run(
            [
                str(self.executable),
                "--device-info",
                "--device",
                str(device)
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=30
        )
        result = json.loads(process.stdout)

        if process.returncode != 0 or not result["ok"]:
            raise RuntimeError(
                result.get("error","Device info query failed!")
            )
        
        return result["properties"]
