"""Evaluator policy tests; actual CUDA evidence is produced by the ROMEO run."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from agent_workload import OutputComparison
from compile_cache import CompilationCache, compilation_manifest
from evaluation import Candidate, EvaluationCase, Evaluator, LaunchRule, WorkloadContract, paired_bootstrap, _source_contract_violations
from gpu_scheduler import GPUScheduler, GPUInterferenceError
from use_harness import BufferArgument, BufferDirection, KernelRequest


class FakeRunner:
    def __init__(self, directory, *, wrong=False, readonly=True, fail_stage=None):
        self.directory = directory
        self.wrong, self.readonly, self.fail_stage = wrong, readonly, fail_stage
        self.calls = []

    def get_device_properties(self, device):
        return {"architecture": "sm_90", "uuid": "fake", "max_threads_per_block": 1024,
                "max_shared_memory_per_block_bytes": 49152, "driver_version": 12000, "nvrtc_version": "12.0",
                "cuda_toolkit_version":12000,"harness_build_id":"cpu-test-harness"}

    def run(self, run_id, request):
        self.calls.append((run_id, request))
        if self.fail_stage:
            return {"ok": False, "stage": self.fail_stage, "error": "controlled failure"}
        value = np.fromfile(request.arguments[0].input_file, dtype="<f4") * 2
        if self.wrong:
            value[-1] = np.nan
        path = self.directory / (run_id + ".bin")
        value.tofile(path)
        latency = 1.0 if "fast" in Path(request.source).name else 2.0
        return {"ok": True, "guard_readonly_ok": self.readonly, "guard_outputs_poisoned": True,
                "environment": self.get_device_properties(0),
                "outputs": [{"path": str(path), "bytes": value.nbytes}], "latency_ms": latency,
                "latency_samples_ms": [latency] * request.iterations}


def setup(tmp_path, runner=None, final=True, **overrides):
    source = tmp_path / "baseline.cu"
    source.write_text('extern "C" __global__ void f(float* x) {}')
    fast = tmp_path / "fast.cu"
    fast.write_text(source.read_text() + "\n// different compiled artifact")
    candidate = Candidate("fast", fast, "f", LaunchRule())
    baseline = Candidate("baseline", source, "f", LaunchRule())
    inp = tmp_path / "a.bin"
    np.arange(17, dtype="<f4").tofile(inp)
    request = KernelRequest(str(source), "f", "run", 0, [1, 1, 1], [256, 1, 1], 0, 0, 1,
                            [BufferArgument(68, str(inp), BufferDirection.Input), BufferArgument(68, None, BufferDirection.Output)])
    case = EvaluationCase("search", request, [OutputComparison(0, "float32", 0, 0)], {"n": 17}, critical=True)
    contract = WorkloadContract("test", 1, "double", "f(x)", "CPU NumPy multiplication",
                                benchmark_series=3, confirmation_batches=4, launches_per_series=2,
                                bootstrap_samples=200, **overrides)
    suites = {"short": [case], "search": [case], "performance": [case]}
    if final:
        suites["final"] = [replace(case, case_id="holdout", seed=900)]
    runner = runner or FakeRunner(tmp_path)
    oracle = lambda c: [np.fromfile(c.request.arguments[0].input_file, dtype="<f4") * 2]
    evaluator = Evaluator(runner, contract, oracle, suites)
    return evaluator, candidate, baseline, case


def test_validation_precedes_benchmark_and_preserves_raw_series(tmp_path):
    evaluator, candidate, _, _ = setup(tmp_path)
    result = evaluator.benchmark(candidate)
    assert result["ok"] and result["status"] == "measured"
    modes = [request.mode for _, request in evaluator.runner.calls]
    # Short + full search each run twice; an exact performance subset adds no
    # duplicate correctness launches before the three fresh timing series.
    assert modes == ["run"] * 4 + ["benchmark"] * 3
    assert result["cases"]["search"]["latency_ms"] == 1
    assert result["cases"]["search"]["series"][0]["raw_ms"] == [1, 1]
    assert evaluator.runner.calls[0][1].sentinel_byte != evaluator.runner.calls[1][1].sentinel_byte


@pytest.mark.parametrize("changed", ["seed", "metadata", "comparison", "input_bytes"])
def test_performance_validation_is_reused_only_for_exact_frozen_search_cases(tmp_path, changed):
    evaluator, candidate, _, case = setup(tmp_path)
    performance = replace(case)
    if changed == "seed":
        performance.seed += 1
    elif changed == "metadata":
        performance.metadata = {"n": 17, "different_layout": "new"}
    elif changed == "comparison":
        performance.comparisons = [OutputComparison(0, "float32", 0.1, 0)]
    else:
        from copy import deepcopy
        performance.request = deepcopy(case.request)
        new_input = tmp_path / "different-input.bin"
        (np.arange(17, dtype="<f4") + 1).tofile(new_input)
        performance.request.arguments[0].input_file = str(new_input)
    suites = {"short": [case], "search": [case], "performance": [performance]}
    exact = Evaluator(evaluator.runner, evaluator.contract, evaluator.oracle, suites)
    assert exact.benchmark(candidate)["ok"]
    assert [request.mode for _, request in evaluator.runner.calls] == ["run"] * 6 + ["benchmark"] * 3


@pytest.mark.parametrize("issue,status", [("wrong", "numerical_error"), ("readonly", "memory_error"), ("compile", "compilation_error")])
def test_invalid_candidates_never_benchmark(tmp_path, issue, status):
    runner = FakeRunner(tmp_path, wrong=issue == "wrong", readonly=issue != "readonly",
                        fail_stage="compile" if issue == "compile" else None)
    evaluator, candidate, _, _ = setup(tmp_path, runner)
    result = evaluator.benchmark(candidate)
    assert not result["ok"] and result["status"] == status
    assert all(request.mode == "run" for _, request in runner.calls)


def test_candidate_source_parameters_and_contract_are_pinned(tmp_path):
    evaluator, candidate, _, case = setup(tmp_path)
    case.comparisons[0].absolute_tolerance = 1e9
    # Constructor deep-copies the caller's test/numerical metadata.
    assert evaluator._suites["search"][0].comparisons[0].absolute_tolerance == 0
    candidate.parameters["precision"] = "fp16"
    with pytest.raises(ValueError, match="parameters changed"):
        evaluator.validate(candidate)
    candidate.parameters.clear()
    candidate.source.write_text("modified")
    with pytest.raises(ValueError, match="source changed"):
        evaluator.validate(candidate)


def test_input_modification_is_rejected_before_gpu(tmp_path):
    evaluator, candidate, _, case = setup(tmp_path)
    Path(case.request.arguments[0].input_file).write_bytes(b"x" * 68)
    with pytest.raises(ValueError, match="input changed"):
        evaluator.validate(candidate)
    assert not evaluator.runner.calls


def test_confirmation_alternates_pairs_and_uses_new_samples(tmp_path):
    evaluator, candidate, baseline, _ = setup(tmp_path)
    measured = evaluator.benchmark(candidate)
    result = evaluator.confirm(candidate, baseline)
    assert result["evaluation_id"] != measured["evaluation_id"]
    assert result["status"] == "promoted"
    assert result["interval"]["lower"] == pytest.approx(2)
    assert [p["order"] for p in result["pairs"]] == [["baseline", "fast"], ["fast", "baseline"]] * 2


def test_exact_artifact_confirmation_has_no_measured_interval_or_gpu_runs(tmp_path):
    evaluator, _, baseline, _ = setup(tmp_path)
    alias_source = tmp_path / "baseline-copy.cu"
    alias_source.write_bytes(baseline.source.read_bytes())
    alias = Candidate("baseline", alias_source, baseline.kernel, baseline.launch_rule)
    for candidate in (alias, baseline):
        result = evaluator.confirm(candidate, baseline)
        assert result["ok"] and result["status"] == "exact_artifact_identity"
        assert result["pairs"] == [] and "interval" not in result
        assert result["paired_measurement_performed"] is False
        assert result["correctness_qualification_inferred"] is False
        assert result["ranking_eligible"] is False
    assert not evaluator.runner.calls


def test_qualification_context_initializes_hardware_with_a_provided_scheduler(tmp_path,monkeypatch):
    evaluator,_,baseline,_ = setup(tmp_path,final=False)
    evaluator.scheduler = GPUScheduler(tmp_path/"provided-scheduler","cpu-test-device")
    properties = evaluator.runner.get_device_properties(0)
    monkeypatch.setattr(evaluator.runner,"get_device_properties",lambda device:properties)
    assert evaluator._properties is None
    context = evaluator.qualification_context(baseline)
    assert context["key"] and evaluator._properties == properties
    properties["driver_version"] += 1
    assert evaluator._properties["driver_version"] != properties["driver_version"]
    with pytest.raises(RuntimeError,match="hardware/toolchain identity changed"):
        evaluator.qualification_context(baseline)
    assert not evaluator.runner.calls


def test_provided_scheduler_does_not_freeze_hardware_before_contract_guard(tmp_path):
    original,_,baseline,case = setup(tmp_path,final=False)
    evaluator = Evaluator(original.runner,replace(original.contract,metadata_json=json.dumps({
        "target_environment":{"uuid":"a-different-expected-device"}})),original.oracle,
        {"short":[case],"search":[case],"performance":[case]},
        scheduler=GPUScheduler(tmp_path/"provided-scheduler","cpu-test-device"))
    with pytest.raises(RuntimeError,match="environment differs from contract"):
        evaluator.qualification_context(baseline)
    assert evaluator._properties is None and not evaluator.runner.calls


def test_distinct_artifacts_with_colliding_labels_fail_before_gpu_runs(tmp_path):
    evaluator, candidate, baseline, _ = setup(tmp_path)
    candidate = Candidate("baseline", candidate.source, candidate.kernel, candidate.launch_rule)
    with pytest.raises(ValueError, match="Distinct artifacts require distinct candidate_id"):
        evaluator.confirm(candidate, baseline)
    assert not evaluator.runner.calls


def test_qualification_context_binds_launch_parameters_contract_cases_and_suite(tmp_path):
    evaluator, candidate, _, case = setup(tmp_path, final=False)
    original = evaluator.qualification_context(candidate, "search")["key"]
    assert original
    assert evaluator.qualification_context(candidate, "performance")["key"] != original
    changed_launch = Candidate("launch", candidate.source, candidate.kernel, LaunchRule(block=(128,1,1)))
    changed_parameters = Candidate("parameters", candidate.source, candidate.kernel, candidate.launch_rule, {"tile":16})
    assert evaluator.qualification_context(changed_launch, "search")["key"] != original
    assert evaluator.qualification_context(changed_parameters, "search")["key"] != original
    for contract, cases in ((replace(evaluator.contract, min_relative_gain=.02), [case]),
                            (replace(evaluator.contract, required_sanitizers=("memcheck",)), [case]),
                            (evaluator.contract, [replace(case, seed=72)]),
                            (evaluator.contract, [replace(case, request=replace(case.request,compiler_options=["--fmad=false"]))])):
        other = Evaluator(evaluator.runner, contract, evaluator.oracle,
                          {"short":cases,"search":cases,"performance":cases})
        assert other.qualification_context(candidate, "search")["key"] != original
    assert not evaluator.runner.calls


def test_qualification_context_tracks_quoted_dependency_bytes(tmp_path):
    evaluator, candidate, _, _ = setup(tmp_path)
    header = tmp_path/"helper.h"
    header.write_text("#define HELPER 1\n")
    candidate.source.write_text('#include "helper.h"\n'+candidate.source.read_text())
    candidate = Candidate(candidate.candidate_id,candidate.source,candidate.kernel,candidate.launch_rule)
    first = evaluator.qualification_context(candidate)
    header.write_text("#define HELPER 2\n")
    second = evaluator.qualification_context(candidate)
    assert first["identity"] != second["identity"]
    assert first["key"] is None and second["key"] is None
    assert not evaluator.runner.calls


def test_qualification_alias_identity_rechecks_source_inputs_and_environment(tmp_path, monkeypatch):
    evaluator, candidate, baseline, case = setup(tmp_path)
    evaluator.qualification_context(candidate)
    properties = evaluator.runner.get_device_properties(0)
    monkeypatch.setattr(evaluator.runner,"get_device_properties",lambda device:{**properties,"uuid":"changed"})
    with pytest.raises(RuntimeError, match="hardware/toolchain identity changed"):
        evaluator.qualification_context(candidate)
    with pytest.raises(RuntimeError, match="hardware/toolchain identity changed"):
        evaluator.confirm(baseline, baseline)
    monkeypatch.setattr(evaluator.runner,"get_device_properties",lambda device:dict(properties))
    candidate.source.write_text(candidate.source.read_text()+"\n// mutated")
    with pytest.raises(ValueError, match="source changed"):
        evaluator.qualification_context(candidate)
    Path(case.request.arguments[0].input_file).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        evaluator.qualification_context(baseline)
    assert not evaluator.runner.calls


def test_path_or_time_sensitive_source_cannot_reuse_qualification(tmp_path):
    evaluator, candidate, _, _ = setup(tmp_path)
    candidate.source.write_text(candidate.source.read_text()+"\n#define FILE_NAME __FILE__\n")
    candidate = Candidate(candidate.candidate_id,candidate.source,candidate.kernel,candidate.launch_rule)
    assert evaluator.qualification_context(candidate)["key"] is None
    comparison = evaluator.confirm(candidate,candidate)
    assert comparison["status"] == "artifact_identity_context_unproven" and not comparison["ok"]
    assert comparison["pairs"] == [] and "interval" not in comparison
    assert not evaluator.runner.calls


@pytest.mark.parametrize("mechanism", ["token_paste", "compiler_macro"])
def test_indirect_path_builtins_disable_reuse_without_rejecting_source(tmp_path,mechanism):
    evaluator, candidate, _, case = setup(tmp_path,final=False)
    if mechanism == "token_paste":
        candidate.source.write_text("#define CAT(a,b) a##b\n#define LOCATION CAT(__FI,LE__)\n"+candidate.source.read_text())
        candidate = Candidate(candidate.candidate_id,candidate.source,candidate.kernel,candidate.launch_rule)
    else:
        case = replace(case,request=replace(case.request,compiler_options=["-DLOCATION=__FILE__"]))
        evaluator = Evaluator(evaluator.runner,evaluator.contract,evaluator.oracle,
                              {"short":[case],"search":[case],"performance":[case]})
    assert evaluator.qualification_context(candidate)["key"] is None
    assert evaluator.validate(candidate)["ok"]  # The ordinary validation remains available.


@pytest.mark.parametrize("missing", ["uuid","architecture","driver_version","nvrtc_version","cuda_toolkit_version","harness_build_id"])
def test_partial_worker_identity_disables_qualification_reuse(tmp_path,monkeypatch,missing):
    evaluator,candidate,_,_ = setup(tmp_path)
    properties = evaluator.runner.get_device_properties(0)
    properties.pop(missing)
    monkeypatch.setattr(evaluator.runner,"get_device_properties",lambda device:dict(properties))
    assert evaluator.qualification_context(candidate)["key"] is None
    assert not evaluator.runner.calls


def test_bootstrap_resamples_complete_series_and_critical_regression(tmp_path):
    result = paired_bootstrap({"a": [1, 1, 1], "b": [10, 10, 10]},
                              {"a": [2, 2, 2], "b": [9, 9, 9]}, {"a": 1, "b": 1}, samples=200)
    assert result["gain"] > 1
    assert result["case_gains"]["b"]["upper"] < 1
    evaluator, fast, baseline, _ = setup(tmp_path)
    # The candidate under test is slower on the critical case.
    result = evaluator.confirm(baseline, fast)
    assert result["status"] == "not_improved" and not result["critical_cases_ok"]


def test_final_holdout_is_one_use_and_artifact_is_exact(tmp_path):
    evaluator, candidate, baseline, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="reserved"):
        evaluator.validate(candidate, "final")
    result = evaluator.finalize(candidate, baseline)
    assert result["status"] == "certified"
    assert result["artifact_hash"] == candidate.artifact_hash
    assert result["source_hash"] == hashlib.sha256(candidate.source.read_bytes()).hexdigest()
    assert evaluator.finalize(candidate, baseline)["status"] == "finalization_incomplete"


def test_lazy_holdout_created_after_search_preserves_contract_hash(tmp_path):
    evaluator, candidate, baseline, case = setup(tmp_path, final=False)
    calls = []
    factory = lambda: (calls.append("final") or [replace(case, case_id="holdout", seed=333)])
    lazy = Evaluator(evaluator.runner, evaluator.contract, evaluator.oracle, evaluator._suites,
                     final_suite_factory=factory, final_suite_identity="frozen-final-generator-and-seed-commitment")
    original_hash = lazy.contract_hash
    lazy.benchmark(candidate)
    assert not calls
    assert lazy.finalize(candidate, baseline)["status"] == "certified"
    assert calls == ["final"] and lazy.contract_hash == original_hash


def test_missing_required_sanitizer_blocks_certification(tmp_path):
    evaluator, candidate, baseline, _ = setup(tmp_path, required_sanitizers=("racecheck",))
    result = evaluator.finalize(candidate, baseline)
    assert result["status"] == "finalization_incomplete"
    assert result["sanitizers"][0]["status"] == "tool_unavailable"


def test_final_numerical_failure_is_explicit_and_cannot_certify(tmp_path):
    evaluator, candidate, baseline, _ = setup(tmp_path)
    assert evaluator.benchmark(candidate)["ok"]
    evaluator.runner.wrong = True
    result = evaluator.finalize(candidate, baseline)
    assert result["status"] == "final_validation_failed" and not result["ok"]
    assert "confirmation" not in result


def test_returning_baseline_requires_final_validation_but_no_gain_claim(tmp_path):
    evaluator, _, baseline, _ = setup(tmp_path)
    result = evaluator.finalize(baseline, baseline)
    assert result["status"] == "validated_no_gain" and result["validation"]["ok"]
    assert all(request.mode == "run" for _, request in evaluator.runner.calls)


def test_hardware_drift_invalidates_a_measurement(tmp_path):
    class DriftingRunner(FakeRunner):
        def run(self, run_id, request):
            result = super().run(run_id, request)
            result["environment"]["uuid"] = "a-different-gpu"
            return result
    evaluator, candidate, _, _ = setup(tmp_path, DriftingRunner(tmp_path))
    result = evaluator.benchmark(candidate)
    assert result["status"] == "infrastructure_failure" and not result["ok"]


def test_integer_comparison_is_exact_regardless_of_float_tolerances(tmp_path):
    evaluator, _, _, case = setup(tmp_path)
    integer_case = replace(case, comparisons=[OutputComparison(0, "int32", 1000, 1000)])
    output = tmp_path / "integer.bin"
    np.array([43], dtype="<i4").tofile(output)
    report = {"outputs": [{"path": str(output), "bytes": 4}]}
    assert not evaluator._compare(integer_case, report, [np.array([42], dtype="<i4")])["ok"]


def test_launch_rule_applies_tails_without_generator_or_optin_memory():
    rule = LaunchRule(kind="square_2d", block=[16, 16, 1], tile_x=32, tile_y=16)
    assert rule.apply({"n": 63})["grid"] == [2, 4, 1]
    with pytest.raises(ValueError, match="shared-memory"):
        replace(rule, shared_memory_bytes=64 * 1024).apply({"n": 63}, {"max_shared_memory_per_block_bytes": 49152})


def test_compilation_cache_invalidates_content_dependencies_options_target_and_toolchain(tmp_path):
    source = tmp_path / "kernel.cu"
    dependency = tmp_path / "dep.h"
    source.write_text('#include "dep.h"\n')
    dependency.write_text("#define X 1")
    kwargs = {"architecture": "sm_90", "toolchain": {"nvrtc": "12.0"}, "options": ["--std=c++17"]}
    manifest = compilation_manifest(source, **kwargs)
    assert compilation_manifest(source, **kwargs) == manifest
    cache = CompilationCache(tmp_path / "cache")
    cache.store(manifest, b"cubin")
    assert cache.lookup(manifest).read_bytes() == b"cubin"
    dependency.write_text("#define X 2")
    assert compilation_manifest(source, **kwargs)["key"] != manifest["key"]
    dependency.write_text("#define X 1")
    for key, value in (("architecture", "sm_80"), ("toolchain", {"nvrtc": "12.1"}), ("options", ["--use_fast_math"])):
        assert compilation_manifest(source, **{**kwargs, key: value})["key"] != manifest["key"]
    cache.lookup(manifest).write_bytes(b"corrupted")
    assert cache.lookup(manifest) is None


def test_gpu_reservations_exclude_other_threads_and_detect_interference(tmp_path):
    scheduler = GPUScheduler(tmp_path, "gpu-uuid")
    other = GPUScheduler(tmp_path, "gpu-uuid")
    entered = threading.Event()
    def contend():
        with other.reserve():
            entered.set()
    with scheduler.reserve("benchmark"):
        thread = threading.Thread(target=contend)
        thread.start()
        assert not entered.wait(0.1)
    thread.join(timeout=2)
    assert entered.is_set()
    with pytest.raises(GPUInterferenceError):
        with GPUScheduler(tmp_path, "gpu-uuid", interference_probe=lambda: True).reserve():
            pass


@pytest.mark.parametrize("operation", [
    "__half h;", "half2 h;", "__nv_bfloat16 h;", "auto x = __float2half(1.0f);",
    "nvcuda::wmma::fragment<int,1,1,1> f;", "auto x = __float_to_tf32(1.0f);",
    'asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32;");',
    'asm volatile("cvt.rn.f16.f32 %0, %1;" : : );',
])
def test_explicit_lower_precision_is_rejected_under_float32_contract(operation):
    signature = 'extern "C" __global__ void f(const float* a,float* c,int n)'
    source = signature + " { " + operation + " }"
    assert _source_contract_violations(source, "f", signature, "float32")


def test_precision_guard_ignores_comments_and_strings_and_preserves_float32_vectorization():
    signature = 'extern "C" __global__ void f(const float* a,float* c,int n)'
    source = '''// __half tf32 wmma and asm("mma.sync") are forbidden in actual code
    /* half2 __nv_bfloat16 */
    extern "C" __global__ __launch_bounds__(256) void f(const float* __restrict__ a,float* c,int n) {
        const char* documentation = "half tf32 __global__ void incorrect(double a)";
        float4 values; c[0] = a[0];
    }'''
    assert not _source_contract_violations(source, "f", signature, "float32")


@pytest.mark.parametrize("source", [
    '__global__ void f(const float* a,float* c) {}',
    '__global__ void f(const double* a,float* c,int n) {}',
    '__global__ void f(const float* a,float* c,int n) {} __global__ void second(float* x) {}',
])
def test_abi_changes_and_extra_global_kernels_are_rejected(source):
    assert _source_contract_violations(source, "f", '__global__ void f(const float* a,float* c,int n)', "float32")


def test_contract_precision_violation_never_reaches_compiler_or_benchmark(tmp_path):
    evaluator, _, _, _ = setup(tmp_path, allowed_precision="float32")
    source = tmp_path / "hidden_half.cu"
    source.write_text('__global__ void f(float* x) { __half h; }')
    result = evaluator.benchmark(Candidate("half", source, "f", LaunchRule()))
    assert result["status"] == "contract_violation" and not result["ok"]
    assert not evaluator.runner.calls


@pytest.mark.parametrize("accumulator_type", ["double", "long double"])
def test_float32_contract_rejects_explicit_widening_before_gpu(tmp_path, accumulator_type):
    evaluator, _, _, _ = setup(tmp_path, allowed_precision="float32")
    source = tmp_path / "wider_precision.cu"
    source.write_text('__global__ void f(float* x) { '
                      + accumulator_type + ' sum=0; '
                      + 'for(int i=0;i<17;++i) sum += (' + accumulator_type + ')x[i]; '
                      + 'x[0]=(float)sum; }')
    result = evaluator.benchmark(Candidate("wider", source, "f", LaunchRule()))
    assert result["status"] == "contract_violation" and not result["ok"]
    assert any("double/long double" in violation for violation in result["violations"])
    assert not evaluator.runner.calls


class SanitizerRunner(FakeRunner):
    def __init__(self, directory, *, sanitizer_status="passed", drift=False, wrong=False):
        super().__init__(directory, wrong=wrong)
        self.sanitizer_calls = []
        self.sanitizer_status = sanitizer_status
        self.drift = drift

    def run_sanitizer(self, run_id, request, tool):
        self.sanitizer_calls.append((run_id, request, tool))
        environment = self.get_device_properties(request.device)
        if self.drift:
            environment["uuid"] = "different-sanitizer-gpu"
        return {"ok": self.sanitizer_status == "passed", "tool": tool, "status": self.sanitizer_status,
                "environment": environment, "latency_ms": 123456.0}


def test_sanitizer_qualification_validates_every_case_then_runs_all_tools_without_benchmark(tmp_path):
    tools = ("memcheck", "racecheck", "initcheck", "synccheck")
    runner = SanitizerRunner(tmp_path)
    original, candidate, _, case = setup(tmp_path, runner, required_sanitizers=tools)
    other = replace(case, case_id="second-shape", seed=551)
    evaluator = Evaluator(runner, original.contract, original.oracle,
                          {"short": [case], "search": [case], "performance": [case, other]})
    result = evaluator.qualify_sanitizers(candidate)
    assert result["ok"] and result["status"] == "sanitizer_qualified"
    assert result["artifact_hash"] == candidate.artifact_hash
    assert result["contract_hash"] == evaluator.contract_hash
    assert result["source_hash"] == hashlib.sha256(candidate.source.read_bytes()).hexdigest()
    assert result["launch_rule"]["block"] == candidate.launch_rule.block
    assert not result["ranking_eligible"] and result["instrumented_timings_excluded"]
    assert len(runner.calls) == 4  # Fresh full correctness: two cases, two repetitions.
    assert len(runner.sanitizer_calls) == 8
    assert {(row["tool"], row["case_id"]) for row in result["sanitizers"]} == {
        (tool, case_id) for tool in tools for case_id in ("search", "second-shape")}
    assert all(request.mode == "run" for _, request in runner.calls)
    assert all(request.mode == "run" and request.warmup == 0 and request.iterations == 1
               for _, request, _ in runner.sanitizer_calls)
    # Instrumented latency is retained as a tool diagnostic, never a score.
    assert "cases" not in result and "interval" not in result
    second = evaluator.qualify_sanitizers(candidate)
    assert second["evaluation_id"] != result["evaluation_id"] and len(runner.calls) == 8


@pytest.mark.parametrize("status,expected", [("tool_unavailable", "sanitizer_qualification_incomplete"),
                                             ("memory_error", "sanitizer_qualification_failed")])
def test_sanitizer_qualification_does_not_treat_missing_or_failed_tools_as_success(tmp_path, status, expected):
    runner = SanitizerRunner(tmp_path, sanitizer_status=status)
    evaluator, candidate, _, _ = setup(tmp_path, runner, required_sanitizers=("memcheck", "racecheck"))
    result = evaluator.qualify_sanitizers(candidate)
    assert not result["ok"] and result["status"] == expected
    assert len(runner.sanitizer_calls) == 1


def test_sanitizer_qualification_rejects_hardware_drift_and_incorrect_candidate(tmp_path):
    runner = SanitizerRunner(tmp_path, drift=True)
    evaluator, candidate, _, _ = setup(tmp_path, runner, required_sanitizers=("memcheck",))
    result = evaluator.qualify_sanitizers(candidate)
    assert not result["ok"] and result["status"] == "sanitizer_qualification_incomplete"
    assert result["sanitizers"][0]["status"] == "infrastructure_failure"
    runner.drift, runner.wrong = False, True
    runner.sanitizer_calls.clear()
    result = evaluator.qualify_sanitizers(candidate)
    assert not result["ok"] and result["validation"]["status"] == "numerical_error"
    assert not runner.sanitizer_calls


def test_sanitizer_qualification_requires_declared_tools_and_cannot_expose_holdout(tmp_path):
    evaluator, candidate, _, _ = setup(tmp_path)
    result = evaluator.qualify_sanitizers(candidate)
    assert not result["ok"] and result["status"] == "sanitizer_qualification_incomplete"
    assert not evaluator.runner.calls
    with pytest.raises(ValueError, match="reserved"):
        evaluator.qualify_sanitizers(candidate, "final")


def test_sanitizer_hardware_identity_snapshot_cannot_change_through_runner_alias(tmp_path):
    class MutablePropertiesRunner(SanitizerRunner):
        def __init__(self, directory):
            super().__init__(directory)
            self.properties = super().get_device_properties(0)
        def get_device_properties(self, device):
            return self.properties
        def run_sanitizer(self, run_id, request, tool):
            self.properties["uuid"] = "changed-in-place"
            return {"ok": True, "tool": tool, "environment": self.properties}
    runner = MutablePropertiesRunner(tmp_path)
    evaluator, candidate, _, _ = setup(tmp_path, runner, required_sanitizers=("memcheck",))
    result = evaluator.qualify_sanitizers(candidate)
    assert not result["ok"] and result["status"] == "sanitizer_qualification_incomplete"
    assert result["environment"]["uuid"] == "fake"
    assert result["sanitizers"][0]["hardware_guard"]["before"]["uuid"] == "fake"
    assert result["sanitizers"][0]["hardware_guard"]["after"]["uuid"] == "changed-in-place"
