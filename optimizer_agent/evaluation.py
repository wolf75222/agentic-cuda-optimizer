"""Independent, contract-bound CUDA evaluation and fresh paired confirmation.

The generator owns candidate source and launch metadata only. The evaluator owns
the oracle, immutable test snapshot, numerical rules and undisclosed final suite.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
from typing import Callable
import uuid

import numpy as np

try:
    from .agent_workload import OutputComparison
    from .compile_cache import compilation_manifest, content_hash
    from .gpu_scheduler import GPUScheduler
    from .use_harness import BufferArgument, BufferDirection, KernelRequest
except ImportError:
    from agent_workload import OutputComparison
    from compile_cache import compilation_manifest, content_hash
    from gpu_scheduler import GPUScheduler
    from use_harness import BufferArgument, BufferDirection, KernelRequest


@dataclass(frozen=True)
class LaunchRule:
    kind: str = "elementwise"
    block: tuple[int, int, int] = (256, 1, 1)
    grid: tuple[int, int, int] = (1, 1, 1)
    elements_key: str = "n"
    shared_memory_bytes: int = 0
    version: int = 1
    tile_x: int | None = None
    tile_y: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "block", tuple(self.block))
        object.__setattr__(self, "grid", tuple(self.grid))
        if self.kind not in {"elementwise", "square_2d", "fixed"} or self.version != 1:
            raise ValueError("Unsupported launch rule")
        for dimensions in (self.grid, self.block):
            if len(dimensions) != 3 or any(type(d) is not int or d <= 0 for d in dimensions):
                raise ValueError("Launch dimensions must contain three positive integers")
        if type(self.shared_memory_bytes) is not int or self.shared_memory_bytes < 0:
            raise ValueError("Shared memory must be nonnegative")
        if any(t is not None and (type(t) is not int or t <= 0) for t in (self.tile_x, self.tile_y)):
            raise ValueError("Tile sizes must be positive integers")

    def apply(self, metadata: dict, properties: dict | None = None) -> dict:
        grid = self.grid
        if self.kind != "fixed":
            n = metadata[self.elements_key]
            if type(n) is not int or n <= 0:
                raise ValueError("Launch rule requires a positive integer element count")
            grid = ((n + (self.tile_x or self.block[0]) - 1) // (self.tile_x or self.block[0]), 1, 1)
            if self.kind == "square_2d":
                grid = (grid[0], (n + (self.tile_y or self.block[1]) - 1) // (self.tile_y or self.block[1]), 1)
        if properties:
            if math.prod(self.block) > properties.get("max_threads_per_block", 1024):
                raise ValueError("Launch exceeds worker thread limit")
            for values, limit_name in ((self.block, "max_block_dimensions"), (grid, "max_grid_dimensions")):
                if limit_name in properties and any(a > b for a, b in zip(values, properties[limit_name])):
                    raise ValueError("Launch exceeds worker dimension limit")
            # The worker does not opt in to larger dynamic shared memory limits.
            if self.shared_memory_bytes > properties.get("max_shared_memory_per_block_bytes", 48 * 1024):
                raise ValueError("Launch exceeds configured shared-memory limit")
        return {"grid": list(grid), "block": list(self.block), "shared_memory_bytes": self.shared_memory_bytes}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source: Path
    kernel: str
    launch_rule: LaunchRule
    parameters: dict = field(default_factory=dict)
    _source_hash: str = field(init=False, repr=False)
    _parameters_json: str = field(init=False, repr=False)
    _launch_rule_json: str = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "source", Path(self.source).resolve())
        object.__setattr__(self, "_source_hash", hashlib.sha256(self.source.read_bytes()).hexdigest())
        object.__setattr__(self, "_parameters_json", json.dumps(self.parameters, sort_keys=True, allow_nan=False))
        object.__setattr__(self, "_launch_rule_json", json.dumps(asdict(self.launch_rule), sort_keys=True, allow_nan=False))

    @property
    def artifact_hash(self) -> str:
        self.check_unchanged()
        return content_hash({"source": self._source_hash, "kernel": self.kernel,
                             "launch_rule": asdict(self.launch_rule), "parameters": self._parameters_json})

    def check_unchanged(self):
        if hashlib.sha256(self.source.read_bytes()).hexdigest() != self._source_hash:
            raise ValueError("Candidate source changed after freezing")
        if json.dumps(self.parameters, sort_keys=True, allow_nan=False) != self._parameters_json:
            raise ValueError("Candidate parameters changed after freezing")
        if json.dumps(asdict(self.launch_rule), sort_keys=True, allow_nan=False) != self._launch_rule_json:
            raise ValueError("Candidate launch rule changed after freezing")


@dataclass(frozen=True)
class WorkloadContract:
    contract_id: str
    version: int
    operation: str
    signature: str
    oracle_provenance: str
    benchmark_series: int = 7
    launches_per_series: int = 20
    warmup: int = 3
    confidence: float = 0.95
    min_relative_gain: float = 0.01
    max_critical_regression: float = 0.0
    bootstrap_samples: int = 2000
    required_sanitizers: tuple[str, ...] = ()
    metadata_json: str = "{}"
    correctness_repetitions: int = 2
    confirmation_batches: int = 20
    nonfinite_policy: str = "reject"
    allowed_precision: str | None = None

    def __post_init__(self):
        if not self.oracle_provenance.strip():
            raise ValueError("blocked_oracle: independent oracle provenance is required")
        if any(type(v) is not int or v <= 0 for v in (self.version, self.benchmark_series, self.launches_per_series,
                                                     self.bootstrap_samples, self.correctness_repetitions, self.confirmation_batches)):
            raise ValueError("Contract counts must be positive integers")
        if self.confirmation_batches < 2 or self.bootstrap_samples < 100:
            raise ValueError("Confirmation requires at least two paired series and 100 bootstrap samples")
        if (not all(math.isfinite(v) for v in (self.confidence, self.min_relative_gain, self.max_critical_regression)) or
                not 0 < self.confidence < 1 or self.min_relative_gain < 0 or self.max_critical_regression < 0 or
                type(self.warmup) is not int or self.warmup < 0):
            raise ValueError("Invalid numerical measurement protocol")
        object.__setattr__(self, "required_sanitizers", tuple(self.required_sanitizers))
        if set(self.required_sanitizers) - {"memcheck", "racecheck", "initcheck", "synccheck"}:
            raise ValueError("Unknown sanitizer requirement")
        if self.nonfinite_policy not in {"reject", "match"}:
            raise ValueError("Unknown nonfinite numerical policy")
        if self.allowed_precision not in {None, "float32", "float64", "integer"}:
            raise ValueError("Unknown arithmetic precision policy")
        json.loads(self.metadata_json)

    @property
    def contract_hash(self) -> str:
        return content_hash(asdict(self))


@dataclass
class EvaluationCase:
    case_id: str
    request: KernelRequest
    comparisons: list[OutputComparison]
    metadata: dict
    seed: int = 0
    weight: float = 1.0
    critical: bool = False


def _source_contract_violations(source: str, kernel: str, signature: str, precision: str | None) -> list[str]:
    """Conservative source admission guard, not a proof of CUDA program semantics.

    This catches explicit precision intrinsics/types and tensor instructions. It
    cannot establish the meaning of arbitrary aliases, includes or bit tricks;
    the independent oracle and instrumented controls remain necessary.
    """
    # Preserve string contents only when examining assembly instructions. Ignore
    # comments and normal string literals so documentation does not trigger it.
    lexical = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.DOTALL)
    comments_removed = lexical.sub(lambda m: " " if m.group().startswith(("//", "/*")) else m.group(), source)
    code = lexical.sub(lambda m: "\n" * m.group().count("\n") + " ", source)
    code = re.sub(r"__launch_bounds__\s*\([^)]*\)", " ", code)
    violations = []
    global_count = len(re.findall(r"\b__global__\b", code))
    if global_count != 1:
        violations.append(f"single-kernel contract requires exactly one __global__ definition; found {global_count}")

    def arguments(text, name):
        match = re.search(r"\bvoid\s+" + re.escape(name) + r"\s*\(", text)
        if not match:
            return None
        position, depth = match.end(), 1
        end = position
        while end < len(text) and depth:
            if text[end] == "(": depth += 1
            elif text[end] == ")": depth -= 1
            end += 1
        if depth:
            return None
        value = text[position:end-1].strip()
        return [] if value in {"", "void"} else [x.strip() for x in value.split(",")]

    reference_name = re.search(r"\bvoid\s+([A-Za-z_]\w*)\s*\(", signature)
    expected = arguments(signature, reference_name.group(1)) if reference_name else None
    actual = arguments(code, kernel)
    if actual is None:
        violations.append("candidate kernel entry point is missing or has an unsupported declaration")
    elif expected is not None:
        if len(actual) != len(expected):
            violations.append("candidate kernel argument count differs from immutable ABI")
        else:
            def abi_type(argument):
                argument = re.sub(r"\b(?:__restrict__|__restrict|restrict)\b", "", argument)
                argument = re.sub(r"\s*=.*$", "", argument).strip()
                # The contract uses named, flat scalar/pointer parameters.
                argument = re.sub(r"\b[A-Za-z_]\w*\s*$", "", argument).strip()
                argument = re.sub(r"\bfloat\s+const\b", "const float", argument)
                return re.sub(r"\s+", "", argument)
            if any(abi_type(a) != abi_type(b) for a, b in zip(actual, expected)):
                violations.append("candidate kernel argument types differ from immutable ABI")
    if precision in {"float32", "float64"}:
        lower_types = re.compile(
            r"\b(?:half|half2|__half\w*|__nv_bfloat\w*|nv_bfloat\w*|__nv_fp8\w*|"
            r"__\w*(?:half|bfloat|tf32|fp8)\w*|tf32|tfloat32|wmma|mma_sync|"
            r"tensorcore|tensor_core|OpClassTensorOp|cuda_fp16|cuda_bf16|cuda_fp8)\b", re.IGNORECASE)
        if lower_types.search(code):
            violations.append("explicit lower-precision or tensor-core arithmetic violates declared precision")
        for assembly in re.findall(r"\b(?:asm|__asm__)\b([^;]*)", comments_removed, re.DOTALL):
            if re.search(r"(?:\b(?:mma|wgmma|wmma|tcgen\w*)\b|\.(?:f16|bf16|tf32|e4m3|e5m2)\b)", assembly):
                violations.append("inline tensor-core/lower-precision instruction violates declared precision")
                break
        if precision == "float64" and re.search(r"\bfloat\b", code):
            violations.append("float32 arithmetic violates declared float64 precision")
        if precision == "float32" and re.search(r"\bdouble\b", code):
            violations.append("explicit double/long double arithmetic violates declared float32 precision")
    return violations


def paired_bootstrap(candidate: dict[str, list[float]], incumbent: dict[str, list[float]],
                     weights: dict[str, float], *, confidence=0.95, samples=2000, seed=0) -> dict:
    """Resample series pairs jointly; individual launches are never observations."""
    ids = list(weights)
    if not ids or candidate.keys() != incumbent.keys() or set(ids) != set(candidate):
        raise ValueError("Paired series must cover identical weighted cases")
    lengths = {len(candidate[k]) for k in ids} | {len(incumbent[k]) for k in ids}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("Confirmation requires at least two complete paired series")
    if samples < 100 or not 0 < confidence < 1 or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Invalid bootstrap configuration")
    count = next(iter(lengths))
    a = np.array([candidate[k] for k in ids], dtype=np.float64)
    b = np.array([incumbent[k] for k in ids], dtype=np.float64)
    if not np.all(np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)):
        raise ValueError("Latency series must be finite and positive")
    normalized = np.array([weights[k] for k in ids], dtype=float)
    normalized /= normalized.sum()

    def gains(indices):
        case_gains = np.median(b[:, indices], axis=2) / np.median(a[:, indices], axis=2)
        return np.exp(np.sum(normalized[:, None] * np.log(case_gains), axis=0)), case_gains

    rng = np.random.default_rng(seed)
    distribution, case_distribution = gains(rng.integers(0, count, size=(samples, count)))
    alpha = (1 - confidence) / 2
    estimates = np.median(b, axis=1) / np.median(a, axis=1)
    return {"gain": float(np.exp(np.sum(normalized * np.log(estimates)))),
            "lower": float(np.quantile(distribution, alpha)), "upper": float(np.quantile(distribution, 1-alpha)),
            "confidence": confidence, "bootstrap_samples": samples, "seed": seed,
            "case_gains": {k: {"gain": float(estimates[i]), "lower": float(np.quantile(case_distribution[i], alpha)),
                                "upper": float(np.quantile(case_distribution[i], 1-alpha))} for i, k in enumerate(ids)}}


class Evaluator:
    def __init__(self, runner, contract: WorkloadContract, oracle: Callable[[EvaluationCase], list[np.ndarray]],
                 suites: dict[str, list[EvaluationCase]], scheduler=None, *, final_suite_factory=None,
                 final_suite_identity: str | None = None, input_identity=None):
        self.runner, self.contract, self.oracle, self.scheduler = runner, contract, oracle, scheduler
        self._suites = deepcopy(suites)
        self._final_suite_factory = final_suite_factory
        if final_suite_factory and not final_suite_identity:
            raise ValueError("Lazy final suite requires a frozen generator/distribution/seed identity")
        if not self._suites.get("search") or not self._suites.get("performance"):
            raise ValueError("Search validation and performance suites are required")
        if not self._suites.get("short"):
            self._suites["short"] = self._suites["search"][:1]
        final_ids = {c.case_id for c in self._suites.get("final", [])}
        if final_ids & {c.case_id for name, cases in self._suites.items() if name != "final" for c in cases}:
            raise ValueError("Final holdout must use independent case identities")
        for name, cases in self._suites.items():
            if len({c.case_id for c in cases}) != len(cases):
                raise ValueError("Duplicate case identity in suite")
            for case in cases:
                if not math.isfinite(case.weight) or case.weight <= 0:
                    raise ValueError("Case weights must be positive and finite")
                outputs = [a for a in case.request.arguments if isinstance(a, BufferArgument) and a.direction != BufferDirection.Input]
                if sorted(c.output_index for c in case.comparisons) != list(range(len(outputs))):
                    raise ValueError("Every output requires exactly one comparison rule")
                for comparison in case.comparisons:
                    if comparison.dtype not in {"float32", "float64", "int32", "uint32", "int64", "uint64"}:
                        raise ValueError("Unsupported comparison dtype")
                    if any(value is not None and (not math.isfinite(value) or value < 0)
                           for value in (comparison.absolute_tolerance, comparison.relative_tolerance)):
                        raise ValueError("Numerical tolerances must be finite and nonnegative")
        self._input_hashes = {a.input_file: hashlib.sha256(Path(a.input_file).read_bytes()).hexdigest()
                              for cases in self._suites.values() for case in cases for a in case.request.arguments
                              if isinstance(a, BufferArgument) and a.input_file}
        self._input_identity = input_identity
        if input_identity is not None:
            from optimizer_agent.evaluation_recovery import RecoveryInputIdentity
            if not isinstance(input_identity, RecoveryInputIdentity):
                raise ValueError("Input identity requires an authenticated recovery capsule")
            input_identity.verify_inputs(self._input_hashes)
        self._suite_hash = content_hash({name: [self._identity_case(c) for c in cases] for name, cases in self._suites.items()})
        self._contract_hash = content_hash({"contract": self.contract.contract_hash, "suites": self._suite_hash,
                                           "input_hashes": self._identity_inputs(), "final_suite_identity": final_suite_identity})
        self._validated: set[tuple[str, str]] = set()
        self._properties = None
        self._final_exposed = False
        self._final_timing_authorized = False

    @property
    def contract_hash(self):
        return self._contract_hash

    def _identity_path(self, path):
        return (self._input_identity.project(path, self._input_hashes[path])
                if self._input_identity is not None else path)

    def _identity_inputs(self, case=None):
        names = (self._input_hashes if case is None else
                 [a.input_file for a in case.request.arguments if isinstance(a, BufferArgument) and a.input_file])
        return {self._identity_path(path): self._input_hashes[path] for path in names}

    def _identity_case(self, case):
        value = asdict(case)
        if self._input_identity is not None:
            # Typed argument projection only: source, options, dependencies,
            # scalar values, metadata, numerical rules and seed stay exact.
            for original, argument in zip(case.request.arguments, value["request"]["arguments"]):
                if isinstance(original, BufferArgument) and original.input_file:
                    argument["input_file"] = self._identity_path(original.input_file)
        return value

    def _reservation(self, purpose):
        if self.scheduler is None:
            if self._properties is None:
                first = self._suites["search"][0]
                self._properties = deepcopy(self.runner.get_device_properties(first.request.device))
            device_key = str(self._properties.get("uuid", self._properties.get("architecture", "device-0")))
            self.scheduler = GPUScheduler(Path(tempfile.gettempdir()) / "agentic-cuda-gpu-reservations", device_key)
        return self.scheduler.reserve(purpose)

    def _guard(self, candidate):
        candidate.check_unchanged()
        for path, digest in self._input_hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError("Frozen evaluator input changed")

    def _request(self, candidate, case, *, benchmark=False, series=1):
        self._guard(candidate)
        if self._properties is None:
            self._properties = deepcopy(self.runner.get_device_properties(case.request.device))
        expected_environment = json.loads(self.contract.metadata_json).get("target_environment", {})
        if any(self._properties.get(k) != v for k, v in expected_environment.items()):
            raise ValueError("Execution environment differs from the immutable workload contract")
        launch = candidate.launch_rule.apply(case.metadata, self._properties)
        request = replace(deepcopy(case.request), source=str(candidate.source), kernel=candidate.kernel,
                          mode="benchmark" if benchmark else "run", **launch,
                          warmup=self.contract.warmup if benchmark else 0,
                          iterations=self.contract.launches_per_series if benchmark else 1, series=series)
        if request.cache_dir is None and hasattr(self.runner, "results_dir"):
            request.cache_dir = str(Path(self.runner.results_dir) / "compile-cache")
        # Include options and recursive dependencies even when the worker disables
        # binary reuse; it remains an exact build identity in every receipt.
        toolchain = {k: self._properties[k] for k in ("driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")
                     if k in self._properties}
        if toolchain:
            manifest = compilation_manifest(candidate.source, dependencies=request.dependencies,
                                            options=request.compiler_options,
                                            architecture=self._properties.get("architecture", "unknown"), toolchain=toolchain)
            request.compile_cache_key = manifest["key"]
        return request

    def _base(self, candidate, status):
        return {"evaluation_id": uuid.uuid4().hex, "candidate_id": candidate.candidate_id,
                "artifact_hash": candidate.artifact_hash, "contract_hash": self.contract_hash, "status": status}

    @staticmethod
    def _failure_status(report):
        if report.get("status") in {"tool_unavailable", "infrastructure_failure"}:
            return report["status"]
        if report.get("stage") == "compile":
            return "compilation_error"
        if report.get("stage") in {"execute", "warmup", "load", "guards"}:
            return "memory_error"
        return "infrastructure_failure"

    def _check_environment(self, report):
        actual = report.get("environment")
        if not isinstance(actual, dict) or not isinstance(self._properties, dict):
            return False
        keys = ("uuid", "architecture", "driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")
        return all(actual.get(k) == self._properties.get(k) for k in keys if k in self._properties)

    def _compare(self, case, report, expected):
        if len(expected) != len(case.comparisons) or len(report.get("outputs", [])) != len(case.comparisons):
            return {"ok": False, "error": "Oracle/candidate output count differs from contract"}
        checks = []
        for comparison in case.comparisons:
            reference = np.asarray(expected[comparison.output_index]).ravel()
            actual = comparison.read_output(report)
            if reference.dtype != actual.dtype:
                checks.append({"ok": False, "error": "Oracle dtype differs from declared output dtype"})
                continue
            if reference.shape != actual.shape:
                checks.append({"ok": False, "error": "Output shape differs"})
                continue
            if reference.dtype.kind in "iu":
                matches = actual == reference
            else:
                matches = np.isfinite(reference) & np.isfinite(actual) & np.isclose(
                    actual.astype(np.float64), reference.astype(np.float64),
                    rtol=comparison.relative_tolerance or 0, atol=comparison.absolute_tolerance or 0)
                if self.contract.nonfinite_policy == "match":
                    matches |= (np.isnan(actual) & np.isnan(reference)) | (np.isinf(actual) & (actual == reference))
            checks.append({"ok": bool(np.all(matches)), "mismatch_count": int(np.count_nonzero(~matches))})
        return {"ok": all(c["ok"] for c in checks), "outputs": checks}

    def _validate(self, candidate, suite):
        if suite == "final" and not self._final_exposed:
            raise ValueError("Final suite is reserved for finalize()")
        cases = self._suites.get(suite, [])
        if not cases:
            return {**self._base(candidate, "finalization_incomplete"), "ok": False, "error": "Missing suite"}
        result = {**self._base(candidate, "validated"), "ok": True, "suite": suite, "cases": []}
        precision = self.contract.allowed_precision or json.loads(self.contract.metadata_json).get("precision")
        violations = _source_contract_violations(candidate.source.read_text(), candidate.kernel, self.contract.signature, precision)
        if violations:
            return {**result, "ok": False, "status": "contract_violation", "violations": violations,
                    "source_guard_scope": "Explicit ABI/precision/single-kernel checks; not a semantic proof for arbitrary CUDA"}
        for case in cases:
            # Invoke the independent oracle with a copy, never candidate code.
            expected = self.oracle(deepcopy(case))
            for repetition in range(self.contract.correctness_repetitions):
                request = self._request(candidate, case)
                request.sentinel_byte = 255 if repetition % 2 == 0 else 127
                report = self.runner.run(uuid.uuid4().hex, request)
                entry = {"case_id": case.case_id, "seed": case.seed, "repetition": repetition,
                         "launch": {k: getattr(request, k) for k in ("grid", "block", "shared_memory_bytes")}, "report": report}
                if not report.get("ok"):
                    result.update(ok=False, status=self._failure_status(report))
                elif not self._check_environment(report):
                    result.update(ok=False, status="infrastructure_failure", error="Worker hardware/toolchain identity changed or was omitted")
                elif report.get("guard_readonly_ok") is not True:
                    result.update(ok=False, status="memory_error", error="Worker did not certify read-only input guards")
                elif report.get("guard_outputs_poisoned") is not True:
                    result.update(ok=False, status="memory_error", error="Worker did not apply missing-output sentinels")
                else:
                    try:
                        entry["comparison"] = self._compare(case, report, expected)
                    except (OSError, ValueError, KeyError) as error:
                        entry["comparison"] = {"ok": False, "error": str(error)}
                    if not entry["comparison"]["ok"]:
                        result.update(ok=False, status="numerical_error")
                result["cases"].append(entry)
                if not result["ok"]:
                    return result
        self._guard(candidate)
        self._validated.add((candidate.artifact_hash, suite))
        return result

    def validate(self, candidate, suite="search"):
        if suite == "final":
            raise ValueError("Final suite is reserved for finalize()")
        with self._reservation("validation"):
            return self._validate(candidate, suite)

    def _ensure_valid(self, candidate):
        self._guard(candidate)
        artifact_hash = candidate.artifact_hash
        for suite in ("short", "search", "performance"):
            if (artifact_hash, suite) not in self._validated:
                if suite == "performance" and (artifact_hash, "search") in self._validated:
                    # Reuse only the same frozen case, not a shape-only match.
                    # Identity covers arguments/input paths, numerical rules,
                    # metadata, seed and bytes pinned in the evaluator snapshot.
                    def identity(case):
                        return content_hash({"case": self._identity_case(case), "input_hashes": self._identity_inputs(case)})
                    search_cases = {identity(case) for case in self._suites["search"]}
                    performance_cases = [identity(case) for case in self._suites["performance"]]
                    if performance_cases and all(case in search_cases for case in performance_cases):
                        self._validated.add((artifact_hash, "performance"))
                        continue
                result = self._validate(candidate, suite)
                if not result["ok"]:
                    return result
        return None

    def _measure_case(self, candidate, case):
        request = self._request(candidate, case, benchmark=True)
        report = self.runner.run(uuid.uuid4().hex, request)
        if not report.get("ok"):
            raise RuntimeError(f"{self._failure_status(report)}: {report.get('error', report)}")
        if not self._check_environment(report):
            raise RuntimeError("infrastructure_failure: worker hardware/toolchain identity changed or was omitted")
        raw = report.get("latency_samples_ms")
        if not raw or len(raw) != self.contract.launches_per_series or any(not math.isfinite(v) or v <= 0 for v in raw):
            raise RuntimeError("infrastructure_failure: worker omitted valid raw latency series")
        if report.get("guard_readonly_ok") is not True:
            raise RuntimeError("memory_error: benchmark modified read-only input")
        return {"mean_ms": statistics.fmean(raw), "raw_ms": list(raw), "report": report,
                "launch": {k: getattr(request, k) for k in ("grid", "block", "shared_memory_bytes")}}

    def qualification_context(self, candidate, suite="performance"):
        """Bind one group's qualification reuse, rechecking inputs and hardware.

        This never validates or instruments a kernel. Source locations are
        excluded only for path-independent builds; compiler/dependency bytes,
        launches, cases and the immutable evaluator contract remain bound.
        """
        self._guard(candidate)
        cases = self._suites.get(suite, [])
        if suite == "final" or not cases:
            raise ValueError("Qualification reuse requires an existing non-final suite")
        with self._reservation("qualification identity"):
            actual = deepcopy(self.runner.get_device_properties(cases[0].request.device))
            if not isinstance(actual, dict):
                raise RuntimeError("infrastructure_failure: qualification hardware/toolchain identity is missing")
            expected = json.loads(self.contract.metadata_json).get("target_environment", {})
            if any(actual.get(key) != value for key, value in expected.items()):
                raise RuntimeError("infrastructure_failure: qualification environment differs from contract")
            # An injected scheduler does not initialize a hardware snapshot.
            # Freeze the actual worker observation only after the contract guard.
            if self._properties is None:
                self._properties = deepcopy(actual)
            if not self._check_environment({"environment": actual}):
                raise RuntimeError("infrastructure_failure: qualification hardware/toolchain identity changed")
            names = ("uuid", "gpu_uuid", "architecture", "compute_capability", "driver_version", "nvrtc_version",
                     "cuda_toolkit_version", "harness_build_id")
            environment = {name: actual[name] for name in names if name in actual}
            required_identity = ("uuid", "architecture", "driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")
            builds = []
            reusable = all(actual.get(key) not in (None, "") for key in required_identity)
            reusable = reusable and actual.get("harness_build_id") != "unidentified-build-cache-disabled"
            for case in cases:
                request = self._request(candidate, case)
                toolchain = {key: actual[key] for key in ("driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id") if key in actual}
                manifest = compilation_manifest(candidate.source, dependencies=request.dependencies,
                    options=request.compiler_options, architecture=actual.get("architecture", "unknown"),
                    toolchain=toolchain or {"unidentified_toolchain": True})
                files = {}
                for path, digest in manifest["files"].items():
                    dependency = Path(path)
                    files["<source>" if dependency == candidate.source else path] = digest
                    # Conservatively avoid preprocessing/path-dependent builds.
                    # This includes token-pasted builtins, not only literal ones.
                    if re.search(r"\b__(?:FILE|BASE_FILE|DATE|TIME|TIMESTAMP)__\b|##|^\s*#",
                                 dependency.read_text(), re.MULTILINE):
                        reusable = False
                builds.append({"files": files, "options": manifest["options"], "abi": manifest["abi"],
                    "architecture": manifest["architecture"], "toolchain": manifest["toolchain"],
                    "launch": {key: getattr(request, key) for key in ("grid", "block", "shared_memory_bytes")}})
                # Options can inject macros or implicit headers. Their builds
                # remain admissible; only qualification reuse is disabled.
                reusable = reusable and not request.compiler_options
            identity = {"artifact_hash": candidate.artifact_hash, "contract_hash": self.contract_hash,
                "environment": environment, "suite": suite, "suite_hash": content_hash([self._identity_case(case) for case in cases]),
                "input_hashes": self._identity_inputs(), "required_sanitizers": list(self.contract.required_sanitizers),
                "builds": builds}
            self._guard(candidate)
            return {"key": content_hash(identity) if reusable else None, "identity": identity,
                    "scope": "Only reusable within the same final-group invocation; no validation or GPU kernel executed"}

    def benchmark(self, candidate, protocol=None):
        if protocol is not None:
            raise ValueError("Measurement protocol is frozen in WorkloadContract")
        with self._reservation("benchmark"):
            failure = self._ensure_valid(candidate)
            if failure:
                return failure
            result = {**self._base(candidate, "measured"), "ok": True, "cases": {}, "protocol": asdict(self.contract),
                      "environment": deepcopy(self._properties)}
            for case in self._suites["performance"]:
                rows = [self._measure_case(candidate, case) for _ in range(self.contract.benchmark_series)]
                result["cases"][case.case_id] = {"latency_ms": statistics.median(r["mean_ms"] for r in rows),
                                               "series": rows, "weight": case.weight, "critical": case.critical}
            return result

    def confirm(self, candidate, incumbent, protocol=None):
        if protocol is not None:
            raise ValueError("Confirmation protocol is frozen in WorkloadContract")
        self._guard(candidate)
        self._guard(incumbent)
        if candidate.artifact_hash == incumbent.artifact_hash:
            candidate_context = self.qualification_context(candidate)
            incumbent_context = self.qualification_context(incumbent)
            proven = bool(candidate_context["key"] and candidate_context["key"] == incumbent_context["key"])
            return {**self._base(candidate, "exact_artifact_identity" if proven else "artifact_identity_context_unproven"), "ok": proven,
                "incumbent_id": incumbent.candidate_id, "incumbent_artifact_hash": incumbent.artifact_hash,
                "environment": deepcopy(self._properties), "pairs": [], "protocol": asdict(self.contract),
                "identity_context": candidate_context, "paired_measurement_performed": False,
                "ranking_eligible": False, "correctness_qualification_inferred": False,
                "reason": ("Exact frozen artifact and bound execution context" if proven else "Source artifact matches but execution-context identity is unproven")
                    + "; no performance gain or interval measured"}
        if candidate.candidate_id == incumbent.candidate_id:
            raise ValueError("Distinct artifacts require distinct candidate_id labels for paired confirmation")
        with self._reservation("paired confirmation"):
            for item in (candidate, incumbent):
                failure = self._ensure_valid(item)
                if failure:
                    return failure
            return self._confirm_suite(candidate, incumbent, "performance")

    def _confirm_suite(self, candidate, incumbent, suite):
        """Time an already qualified suite while the caller owns the GPU lease."""
        if suite not in {"performance", "final"}:
            raise ValueError("Unknown confirmation suite")
        if suite == "final" and not (self._final_exposed and self._final_timing_authorized):
            raise ValueError("Final confirmation is reserved for qualified finalize()")
        for item in (candidate, incumbent):
            self._guard(item)
            if (item.artifact_hash, suite) not in self._validated:
                raise ValueError("Confirmation requires validation on its exact suite")
        if candidate.candidate_id == incumbent.candidate_id:
            raise ValueError("Distinct artifacts require distinct candidate_id labels for paired confirmation")
        cases = self._suites[suite]
        result = {**self._base(candidate, "inconclusive"), "ok": True, "incumbent_id": incumbent.candidate_id,
                  "incumbent_artifact_hash": incumbent.artifact_hash, "environment": deepcopy(self._properties),
                  "pairs": [], "protocol": asdict(self.contract), "suite": suite,
                  "case_ids": [case.case_id for case in cases]}
        a = {c.case_id: [] for c in cases}
        b = {c.case_id: [] for c in cases}
        for index in range(self.contract.confirmation_batches):
            # Predetermined alternating AB/BA; no repeated decision until win.
            for case in cases:
                order = [incumbent, candidate] if index % 2 == 0 else [candidate, incumbent]
                measured = [(item, self._measure_case(item, case)) for item in order]
                pair = {"case_id": case.case_id, "series": index, "order": [x.candidate_id for x in order], "measurements": {}}
                for item, row in measured:
                    (a if item is candidate else b)[case.case_id].append(row["mean_ms"])
                    pair["measurements"][item.candidate_id] = row
                result["pairs"].append(pair)
        interval = paired_bootstrap(a, b, {c.case_id: c.weight for c in cases}, confidence=self.contract.confidence,
                                    samples=self.contract.bootstrap_samples)
        result["interval"] = interval
        critical_ok = all(interval["case_gains"][c.case_id]["lower"] >= 1 / (1 + self.contract.max_critical_regression)
                          for c in cases if c.critical)
        result["critical_cases_ok"] = critical_ok
        threshold = 1 + self.contract.min_relative_gain
        if interval["lower"] > threshold and critical_ok:
            result["status"] = "promoted"
        elif interval["upper"] <= threshold or not critical_ok:
            result["status"] = "not_improved"
        return result

    def qualify_sanitizers(self, candidate, suite="performance"):
        """Fresh correctness and instrumented checks; never collect ranking data.

        This is a separate receipt for an exact artifact, including generated
        comparison candidates that were not promoted to campaign incumbent.
        Required tools belong to the frozen contract; an empty list is not a
        successful sanitizer qualification.
        """
        if suite == "final":
            raise ValueError("Final suite is reserved for finalize()")
        with self._reservation("sanitizer qualification"):
            self._guard(candidate)
            result = {**self._base(candidate, "sanitizer_qualification_incomplete"), "ok": False,
                      "suite": suite, "source_hash": candidate._source_hash,
                      "launch_rule": asdict(candidate.launch_rule), "parameters": json.loads(candidate._parameters_json),
                      "environment": deepcopy(self._properties), "required_sanitizers": list(self.contract.required_sanitizers),
                      "sanitizers": [], "ranking_eligible": False, "instrumented_timings_excluded": True}
            cases = self._suites.get(suite, [])
            if not cases or not self.contract.required_sanitizers:
                return {**result, "error": "A nonempty bound suite and required sanitizer tools are necessary"}
            # Deliberately bypass _validated: post-hoc qualification gets fresh,
            # complete correctness evidence for every case it instruments.
            validation = self._validate(candidate, suite)
            result["validation"] = validation
            if not validation["ok"]:
                if validation["status"] in {"numerical_error", "memory_error", "compilation_error", "contract_violation"}:
                    result["status"] = "sanitizer_qualification_failed"
                return result
            if not hasattr(self.runner, "run_sanitizer"):
                return {**result, "error": "Sanitizer runner is unavailable"}
            for tool in self.contract.required_sanitizers:
                for case in cases:
                    request = self._request(candidate, case)
                    row = {"tool": tool, "case_id": case.case_id, "seed": case.seed,
                           "launch": {k: getattr(request, k) for k in ("grid", "block", "shared_memory_bytes")}}
                    try:
                        before = deepcopy(self.runner.get_device_properties(request.device))
                        if not self._check_environment({"environment": before}):
                            row.update(ok=False, status="infrastructure_failure", error="Hardware/toolchain changed before sanitizer")
                            result["sanitizers"].append(row)
                            return result
                        report = deepcopy(self.runner.run_sanitizer(uuid.uuid4().hex, request, tool))
                        after = deepcopy(self.runner.get_device_properties(request.device))
                    except Exception as error:
                        row.update(ok=False, status="infrastructure_failure", error=str(error))
                        result["sanitizers"].append(row)
                        return result
                    row = {**report, **row,
                           "hardware_guard": {"before": before, "after": after,
                                              "reported_environment": report.get("environment")}}
                    # Local wrappers may return tool logs instead of native JSON;
                    # authenticate their explicitly bound device before and after.
                    hardware_ok = self._check_environment({"environment": after})
                    if "environment" in report:
                        hardware_ok = hardware_ok and self._check_environment(report)
                    if report.get("tool", tool) != tool:
                        hardware_ok = False
                    if not hardware_ok:
                        row.update(ok=False, status="infrastructure_failure", error="Hardware/toolchain changed during sanitizer")
                    self._guard(candidate)
                    result["sanitizers"].append(row)
                    if not row.get("ok"):
                        diagnostics = str(row.get("error", "")).lower()
                        unavailable = (row.get("status") in {"tool_unavailable", "infrastructure_failure"}
                                       or any(word in diagnostics for word in ("unavailable", "not found", "permission denied")))
                        result["status"] = "sanitizer_qualification_incomplete" if unavailable else "sanitizer_qualification_failed"
                        return result
            self._guard(candidate)
            return {**result, "ok": True, "status": "sanitizer_qualified"}

    def finalize(self, candidate, baseline):
        if self._final_exposed:
            return {**self._base(candidate, "finalization_incomplete"), "ok": False,
                    "error": "Holdout already exposed; a new independent final suite is required"}
        self._final_exposed = True
        with self._reservation("final validation"):
            if self._final_suite_factory:
                final = deepcopy(self._final_suite_factory())
                seen = {c.case_id for name, cases in self._suites.items() if name != "final" for c in cases}
                if len({c.case_id for c in final}) != len(final) or seen & {c.case_id for c in final}:
                    raise ValueError("Lazy final suite contains repeated or research case identities")
                self._suites["final"] = final
                for case in final:
                    for argument in case.request.arguments:
                        if isinstance(argument, BufferArgument) and argument.input_file:
                            self._input_hashes[argument.input_file] = hashlib.sha256(Path(argument.input_file).read_bytes()).hexdigest()
                self._final_suite_factory = None
            if not self._suites.get("final"):
                return {**self._base(candidate, "finalization_incomplete"), "ok": False, "error": "Missing final suite"}
            validation = self._validate(candidate, "final")
            if not validation["ok"]:
                return {**self._base(candidate, "final_validation_failed"), "ok": False, "validation": validation}
            baseline_validation = self._validate(baseline, "final")
            if not baseline_validation["ok"]:
                return {**self._base(candidate, "final_validation_failed"), "ok": False,
                        "validation": validation, "baseline_validation": baseline_validation}
            cases = self._suites["final"]
            expected_validation = {(case.case_id, repetition) for case in cases
                                   for repetition in range(self.contract.correctness_repetitions)}
            for item, receipt in ((candidate, validation), (baseline, baseline_validation)):
                rows = receipt.get("cases", [])
                complete = (isinstance(rows, list) and len(rows) == len(expected_validation)
                            and all(isinstance(row, dict) and isinstance(row.get("case_id"), str)
                                    and type(row.get("repetition")) is int for row in rows)
                            and {(row["case_id"], row["repetition"]) for row in rows} == expected_validation
                            and all(row.get("comparison", {}).get("ok") is True for row in rows))
                if (receipt.get("suite") != "final" or receipt.get("contract_hash") != self.contract_hash
                        or receipt.get("artifact_hash") != item.artifact_hash
                        or not complete):
                    return {**self._base(candidate, "finalization_incomplete"), "ok": False,
                            "error": "Final correctness receipts do not cover the exact final cases",
                            "validation": validation, "baseline_validation": baseline_validation}
            sanitizers = []
            for tool in self.contract.required_sanitizers:
                for case in cases:
                    if not hasattr(self.runner, "run_sanitizer"):
                        report = {"ok": False, "status": "tool_unavailable", "tool": tool}
                    else:
                        request = self._request(candidate, case)
                        before = deepcopy(self.runner.get_device_properties(request.device))
                        if not self._check_environment({"environment": before}):
                            report = {"ok": False, "status": "infrastructure_failure", "tool": tool,
                                      "error": "Hardware/toolchain changed before final sanitizer"}
                        else:
                            report = deepcopy(self.runner.run_sanitizer(uuid.uuid4().hex, request, tool))
                            after = deepcopy(self.runner.get_device_properties(request.device))
                            report["hardware_guard"] = {"before": before, "after": after,
                                                        "reported_environment": report.get("environment")}
                            hardware_ok = self._check_environment({"environment": after})
                            if "environment" in report:
                                hardware_ok = hardware_ok and self._check_environment(report)
                            if not hardware_ok:
                                report.update(ok=False, status="infrastructure_failure",
                                              error="Hardware/toolchain changed during final sanitizer")
                    if report.get("tool", tool) != tool or report.get("case_id", case.case_id) != case.case_id:
                        report.update(ok=False, error="Final sanitizer receipt has a different tool or case identity")
                    report.setdefault("tool", tool)
                    self._guard(candidate)
                    sanitizers.append({**report, "case_id": case.case_id})
                    if not report.get("ok"):
                        return {**self._base(candidate, "finalization_incomplete"), "ok": False,
                                "validation": validation, "baseline_validation": baseline_validation,
                                "sanitizers": sanitizers}
            expected_sanitizers = {(tool, case.case_id) for tool in self.contract.required_sanitizers for case in cases}
            if (len(sanitizers) != len(expected_sanitizers)
                    or {(row["tool"], row["case_id"]) for row in sanitizers} != expected_sanitizers):
                return {**self._base(candidate, "finalization_incomplete"), "ok": False,
                        "error": "Final sanitizer receipts do not cover the exact final cases",
                        "validation": validation, "baseline_validation": baseline_validation, "sanitizers": sanitizers}
            # Fresh timing uses exactly the independently validated final cases,
            # under the same lease. Public confirm() remains a search operation.
            self._final_timing_authorized = True
            try:
                confirmation = (self._confirm_suite(candidate, baseline, "final")
                                if candidate.artifact_hash != baseline.artifact_hash
                                else {"status": "not_improved", "reason": "baseline artifact returned",
                                      "suite": "final", "case_ids": [case.case_id for case in cases]})
            finally:
                self._final_timing_authorized = False
        status = "certified" if confirmation["status"] == "promoted" else "validated_gain_uncertified"
        if candidate.artifact_hash == baseline.artifact_hash:
            status = "validated_no_gain"
        return {**self._base(candidate, status), "ok": True, "validation": validation,
                "baseline_validation": baseline_validation, "confirmation": confirmation, "sanitizers": sanitizers, "source_hash": candidate._source_hash,
                "launch_rule": asdict(candidate.launch_rule), "parameters": json.loads(candidate._parameters_json)}
