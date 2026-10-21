"""A retained valid artifact never erases an observed method failure."""
from pathlib import Path

import pytest

from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.evaluation import Candidate, LaunchRule
from optimizer_agent.quality_suite import plan_suite, summarize_suite
from optimizer_agent.suite_orchestrator import candidate_receipt, generation_adapter


@pytest.mark.parametrize("method_status", ["completed", "failed"])
def test_original_adapter_preserves_method_outcome_and_retained_artifact(tmp_path, monkeypatch, method_status):
    import optimizer_agent.upstream_original as original

    plan = plan_suite(CampaignConfig(run_id="method-status"), seeds=(1729, 2718))
    entry = next(e for e in plan["entries"] if e["mode"] == "true_upstream")
    source = tmp_path / "retained.cu"
    source.write_text('extern "C" __global__ void matmul(const float*a,const float*b,float*c,int n){}')
    receipt = candidate_receipt(Candidate("retained-initial", source, "matmul", LaunchRule()))
    failed = method_status == "failed"
    result = {"candidate": receipt, "status": "method_failed" if failed else "generated",
        "method_status": method_status, "method_error": "CodexError: interrupted" if failed else None,
        "candidate_fallback": {"retained": True, "reason": "method_failed"} if failed else None,
        "generation_complete": True, "usage": {"input_tokens": 123, "output_tokens": 7}}
    monkeypatch.setattr(original, "run_upstream_campaign", lambda *a, **kw: result)
    observed = generation_adapter(entry, object(), tmp_path, "owned-session")
    assert observed["generation_complete"] is True
    assert observed["generation_stage_status"] == "generation_complete"
    assert observed["method_status"] == method_status
    assert observed["status"] == ("method_failed" if failed else "generation_complete")
    assert observed["candidate"]["artifact_hash"] == receipt["artifact_hash"]
    assert observed["method_error"] == result["method_error"]
    assert observed["candidate_fallback"] == result["candidate_fallback"]
    assert observed["usage"]["model_tokens"] == 130


def test_measured_fallback_failure_remains_in_denominator_and_is_not_method_success():
    plan = plan_suite(CampaignConfig(run_id="outcomes"), seeds=(1729, 2718))
    entry = next(e for e in plan["entries"] if e["mode"] == "true_upstream")
    fallback = {"retained": True, "selected_origin": "supplied_initial", "reason": "method_failed"}
    generation = {"status": "method_failed", "method_status": "failed", "method_error": "CodexError: failure",
        "candidate_fallback": fallback, "usage": {"model_tokens": 234}}
    group = {"candidates": {"true_upstream": {"qualified": True, "status": "validated_no_gain"}}}
    summary = summarize_suite(plan, {entry["config"]["run_id"]: generation}, {entry["group_id"]: group})
    row = next(r for r in summary["rows"] if r["run_id"] == entry["config"]["run_id"])
    assert row["gain"] == 1.0 and row["final_qualified"] is True
    assert row["generation_status"] == "method_failed"
    assert row["generation_method_status"] == "failed"
    assert row["generation_method_error"] == generation["method_error"]
    assert row["candidate_fallback"] == fallback
    counts = summary["aggregates"]["true_upstream"]
    assert counts["method_failed"] == 1 and counts["method_completed"] == 0
    assert counts["method_status_unknown"] == counts["planned"] - 1
    assert counts["final_measured"] == 1 and counts["geometric_mean_all_measured_gain"] == 1.0
