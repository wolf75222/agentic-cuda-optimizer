"""Independent live GPU acceptance probes. Import and call explicitly.

run_safe_probes uses only valid launches, a partial-output candidate and a compile
error. destructive_probe deliberately mutates a read-only input; call it LAST in
an allocation and start a new allocation before any later accepted measurement.
This file does not submit jobs and does not run a probe when executed directly.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer_agent"))
from agent_workload import OutputComparison
from compile_cache import compilation_manifest
from evaluation import Candidate, EvaluationCase, Evaluator, LaunchRule, WorkloadContract
from use_harness import BufferArgument, BufferDirection, KernelRequest, ScalarArgument


GOOD_SOURCE = '''extern "C" __global__ void probe(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = 2.0f * a[i];
}
'''
PARTIAL_SOURCE = GOOD_SOURCE.replace("if (i < n)", "if (i < n - 1)")
MUTATING_SOURCE = '''extern "C" __global__ void probe(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) const_cast<float*>(a)[0] = 1234.0f;
    if (i < n) c[i] = 2.0f * a[i];
}
'''


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class RecordingRunner:
    def __init__(self, runner):
        self.runner, self.calls = runner, []

    def __getattr__(self, name):
        return getattr(self.runner, name)

    def run(self, run_id, request):
        report = self.runner.run(run_id, request)
        self.calls.append({"run_id": run_id, "mode": request.mode,
            "source_hash": hashlib.sha256(Path(request.source).read_bytes()).hexdigest(),
            "compile_cache_key": request.compile_cache_key, "request": asdict(request), "report": report})
        return report


def _fixture(runner, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    n = 257  # Includes a tail and two blocks; exactly represented CPU values.
    values = np.arange(n, dtype="<f4") * np.float32(0.25)
    input_file = directory / "input.bin"
    values.tofile(input_file)
    for name, source in (("correct", GOOD_SOURCE), ("partial", PARTIAL_SOURCE),
                         ("mutating", MUTATING_SOURCE), ("compile_error", GOOD_SOURCE.replace("2.0f * a[i]", "not_defined(a[i])")),
                         ("changed_source", GOOD_SOURCE + "\n// distinct compilation content\n")):
        (directory / (name + ".cu")).write_text(source)
    request = KernelRequest(str(directory / "correct.cu"), "probe", "run", 0, [2, 1, 1], [256, 1, 1],
        0, 0, 1, [BufferArgument(values.nbytes, str(input_file), BufferDirection.Input),
                  BufferArgument(values.nbytes, None, BufferDirection.Output), ScalarArgument("int32", n)])
    case = EvaluationCase("probe-tail-257", request, [OutputComparison(0, "float32", 0, 0)],
                           {"n": n}, seed=0, critical=True)
    contract = WorkloadContract("live-independent-probes", 1, "double_float32", "probe(a,c,n)",
        "Independent NumPy float32 multiplication by exactly representable 2.0",
        benchmark_series=2, launches_per_series=2, warmup=1, confirmation_batches=2, bootstrap_samples=200)
    recording = RecordingRunner(runner)
    evaluator = Evaluator(recording, contract, lambda _: [values * np.float32(2)],
                          {"short": [case], "search": [case], "performance": [case]})
    candidates = {name: Candidate("probe-" + name, directory / (name + ".cu"), "probe", LaunchRule())
                  for name in ("correct", "partial", "mutating", "compile_error", "changed_source")}
    return directory, recording, evaluator, candidates, request, values


def _output_matches(report, expected):
    try:
        comparison = OutputComparison(0, "float32", 0, 0)
        actual = comparison.read_output(report)
        return bool(report.get("ok") and len(report["outputs"]) == 1 and np.array_equal(actual, expected))
    except (OSError, KeyError, ValueError):
        return False


def run_safe_probes(runner, directory):
    """Run real correctness, no-benchmark-on-error and native binary-cache probes."""
    directory, recording, evaluator, candidates, template, values = _fixture(runner, directory)
    evidence = {"scope": "live CUDA correctness/cache policy; no optimization gain claimed", "checks": {}, "calls": []}
    try:
        good = evaluator.benchmark(candidates["correct"])
        evidence["valid_candidate"] = good
        evidence["checks"]["valid_before_benchmark"] = bool(good.get("ok"))
        if not good.get("ok"):
            return evidence
        begin = len(recording.calls)
        partial = evaluator.benchmark(candidates["partial"])
        partial_calls = recording.calls[begin:]
        evidence["partial_output"] = partial
        evidence["checks"]["partial_output_rejected"] = partial.get("status") == "numerical_error" and not partial.get("ok")
        evidence["checks"]["incorrect_candidate_never_benchmarked"] = bool(partial_calls) and all(c["mode"] == "run" for c in partial_calls)
        invalid = evaluator.validate(candidates["compile_error"])
        evidence["compile_error"] = invalid
        evidence["checks"]["compile_error_classified"] = invalid.get("status") == "compilation_error"
        recovery = evaluator.validate(candidates["correct"])
        evidence["after_compile_error"] = recovery
        evidence["checks"]["compile_error_did_not_poison_worker"] = bool(recovery.get("ok"))

        properties = runner.get_device_properties(0)
        toolchain = {k: properties[k] for k in ("driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id") if k in properties}
        cache_runs = []
        # A unique define avoids accidentally reusing a binary from earlier probes.
        define = "-DPROBE_CACHE_ID=" + str(uuid.uuid4().int % (2**31 - 1))
        for name, options in (("first", [define]), ("same", [define]),
                              ("changed_options", [define, "--use_fast_math"]),
                              ("same_changed_options", [define, "--use_fast_math"]),
                              ("changed_source", [define])):
            source = candidates["changed_source" if name == "changed_source" else "correct"].source
            manifest = compilation_manifest(source, options=options, architecture=properties["architecture"], toolchain=toolchain)
            request = replace(template, source=str(source), compiler_options=options,
                              cache_dir=str(directory / "compile-cache"), compile_cache_key=manifest["key"])
            report = recording.run(uuid.uuid4().hex, request)
            cache_runs.append({"name": name, "manifest": manifest, "report": report})
        evidence["cache_runs"] = cache_runs
        evidence["checks"]["cache_runs_correct"] = all(_output_matches(c["report"], values * np.float32(2)) for c in cache_runs)
        observed = [c["report"].get("compile_cache_hit") for c in cache_runs]
        evidence["checks"]["native_cache_source_and_option_invalidation"] = observed == [False, True, False, True, False]
        evidence["dependency_cache_scope"] = "Recursive dependency-content invalidation is unit-tested; this remote transport explicitly rejects unstaged local dependencies. No remote dependency claim."
        evidence["completed"] = True
        return evidence
    finally:
        evidence["calls"] = recording.calls
        evidence["passed"] = evidence.get("completed", False) and bool(evidence["checks"]) and all(evidence["checks"].values())
        _save(directory / "safe-probes.json", evidence)


def destructive_probe(runner, directory):
    """LAST CUDA work in this allocation: mutate read-only input and require quarantine.

    No recovery launch is attempted. The caller must inspect the worker/allocation
    terminal state separately, then allocate a new worker for a recovery test.
    """
    directory, recording, evaluator, candidates, _, _ = _fixture(runner, directory)
    evidence = {"scope": "deliberate read-only mutation; retire this allocation after probe",
                "must_start_new_allocation": True, "safe_to_reuse_allocation": False}
    try:
        result = evaluator.validate(candidates["mutating"])
        evidence["validation"] = result
        reports = [entry["report"] for entry in result.get("cases", [])]
        evidence["checks"] = {
            "input_mutation_rejected": result.get("status") == "memory_error" and not result.get("ok"),
            "native_worker_marked_nonreusable": any(r.get("worker_reusable") is False for r in reports),
            "no_benchmark_submitted": all(c["mode"] == "run" for c in recording.calls),
        }
        evidence["passed"] = all(evidence["checks"].values())
        return evidence
    finally:
        evidence["calls"] = recording.calls
        _save(directory / "destructive-probe.json", evidence)


if __name__ == "__main__":
    print("Import run_safe_probes(runner, directory); call destructive_probe(runner, directory) last. No GPU work was executed.")
