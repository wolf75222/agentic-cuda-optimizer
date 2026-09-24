from dataclasses import dataclass
from typing import Literal
from use_harness import KernelRequest
import math

@dataclass
class CaseRun:
    case_index: int
    request: KernelRequest
    harness_result: dict
    comparison_result: list[dict]
    validation_ok: bool
    size_summary: str = ""
    purpose: Literal["correctness", "performance"] = "performance"

@dataclass
class OptimizationAttempt:
    iteration: int
    kernel_source: str
    case_runs: list[CaseRun]
    hypothesis: dict | None = None
    change_summary: str = "Initial implementation"

    def performance_report(self) -> dict:
        """Model-facing measurements; full harness diagnostics stay in history.json."""
        cases = []
        for case in self.case_runs:
            result = case.harness_result
            latency = result.get("latency_ms")
            measured = type(latency) in (int, float) and math.isfinite(latency) and latency > 0
            report = {
                "case_index": case.case_index,
                "purpose": case.purpose,
                "execution_ok": bool(result.get("ok")),
                "validation_ok": case.validation_ok,
                "kernel_latency_ms": latency if measured else None,
                "grid": case.request.grid,
                "block": case.request.block,
                "shared_memory_bytes": case.request.shared_memory_bytes,
                "warmup": case.request.warmup,
                "timed_launches": case.request.iterations,
                "comparison_results": case.comparison_result,
            }
            if not result.get("ok"):
                report["execution_error"] = result.get("error")
                report["compiler_log"] = result.get("compiler_log", "")
            cases.append(report)
        score = geometric_mean_time(self)
        return {
            "iteration": self.iteration,
            "change_summary": self.change_summary,
            "eligible_for_ranking": score is not None,
            "performance_geometric_mean_kernel_latency_ms": score,
            "cases": cases,
        }


def attempt_time(attempt: OptimizationAttempt) -> float | None:
    if not attempt.case_runs:
        return None
    total = 0.0
    for case in attempt.case_runs:
        latency = case.harness_result.get("latency_ms")
        if not case.validation_ok or not case.harness_result.get("ok"):
            return None
        if case.purpose == "correctness":
            continue
        if not isinstance(latency, (float, int)) or not math.isfinite(latency) or latency <= 0:
            return None
        total += latency
    return total if total > 0 and math.isfinite(total) else None


def experiment_history(history: list[OptimizationAttempt]) -> list[dict]:
    reports = []
    previous = {}
    for attempt in history:
        report = attempt.performance_report()
        hypothesis = attempt.hypothesis or {}
        parent = previous.get(hypothesis.get("starting_iteration"))
        report["experiment"] = {
            "starting_iteration": hypothesis.get("starting_iteration"),
            "claimed_change_unverified": hypothesis.get("proposed_change"),
            "claimed_execution_paths_unverified": hypothesis.get("case_execution_paths", []),
            "hardware_claims": hypothesis.get("hardware_claims", []),
            "source_text_changed_from_parent": (
                attempt.kernel_source != parent.kernel_source if parent is not None else None
            ),
            "exact_source_matches": [
                entry.iteration for entry in previous.values()
                if entry.kernel_source == attempt.kernel_source
            ],
            "performance_speedup_vs_parent": (
                geometric_mean_speedup(parent, attempt) if parent is not None else None
            ),
        }
        reports.append(report)
        previous[attempt.iteration] = attempt
    return reports


def geometric_mean_time(attempt: OptimizationAttempt) -> float | None:
    if attempt_time(attempt) is None:
        return None
    performance_cases = [case for case in attempt.case_runs if case.purpose == "performance"]
    return math.exp(math.fsum(math.log(case.harness_result["latency_ms"])
                              for case in performance_cases) / len(performance_cases))


def geometric_mean_speedup(baseline: OptimizationAttempt, attempt: OptimizationAttempt) -> float | None:
    baseline_time = geometric_mean_time(baseline)
    candidate_time = geometric_mean_time(attempt)
    baseline_indices = sorted((case.case_index, case.purpose) for case in baseline.case_runs)
    candidate_indices = sorted((case.case_index, case.purpose) for case in attempt.case_runs)
    if baseline_time is None or candidate_time is None or baseline_indices != candidate_indices:
        return None
    return baseline_time / candidate_time
