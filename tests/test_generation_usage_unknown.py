"""Generation adaptation preserves missing/malformed counters without inventing totals."""
import pytest

from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.evaluation import Candidate, LaunchRule
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.suite_orchestrator import candidate_receipt, generation_adapter


@pytest.mark.parametrize("usage,total", [
    ({}, None), (None, None), ({"input_tokens": 12}, None),
    ({"input_tokens": 12, "output_tokens": None}, None),
    ({"input_tokens": True, "output_tokens": 3}, None),
    ({"input_tokens": -1, "output_tokens": 5}, None),
    ({"input_tokens": "1", "output_tokens": "2"}, None),
    ({"input_tokens": 12, "output_tokens": 3}, 15),
    ({"input_tokens": 0, "output_tokens": 0}, 0),
    ({"input_tokens": 0, "output_tokens": 0, "model_tokens":0, "model_calls":0, "usage_complete":False}, None),
    ({"input_tokens": 12, "output_tokens": 3, "model_tokens":15,
      "model_calls":1, "uncertain_model_intents":["synthetic/pending/intent.json"]}, None),
    ({"input_tokens": 12, "output_tokens": 3, "model_calls":1, "usage_complete":False}, None),
    ({"input_tokens": 12, "output_tokens": 3, "model_calls":1, "known_model_calls":True,
      "uncertain_model_records":[{"record_index":0}]}, None),
    ({"input_tokens": 12, "output_tokens": 3, "model_calls":1, "known_model_calls":"1",
      "uncertain_model_records":[{"record_index":0}]}, None),
    ({"input_tokens": 12, "output_tokens": 3, "model_calls":1, "known_model_calls":0,
      "uncertain_model_records":[{"record_index":0}]}, None),
    ({"input_tokens": 12, "output_tokens": 3, "model_tokens":0}, 15),
    ({"model_tokens":0}, None),
])
def test_adapter_does_not_fill_missing_output_as_zero_or_coerce_provider_counts(tmp_path, monkeypatch, usage, total):
    import optimizer_agent.upstream_original as original
    plan = plan_suite(CampaignConfig(run_id="usage-adapter"), seeds=(1729, 2718))
    entry = next(entry for entry in plan["entries"] if entry["mode"] == "true_upstream")
    source = tmp_path / "fixture.cu"
    source.write_text('extern "C" __global__ void matmul(const float*a,const float*b,float*c,int n){}')
    candidate = candidate_receipt(Candidate("retained-fixture", source, "matmul", LaunchRule()))
    result = {"candidate": candidate, "method_status": "completed", "status": "generated", "usage": usage}
    monkeypatch.setattr(original, "run_upstream_campaign", lambda *args, **kwargs: result)
    observed = generation_adapter(entry, object(), tmp_path, "fixture-session")
    assert observed["usage"]["model_tokens"] == total
    assert observed["usage_observation"]["model_tokens"] == total
    assert observed["generation_usage_reported"] == usage
    if isinstance(usage, dict) and (usage.get("usage_complete") is False or usage.get("uncertain_model_intents")
                                   or usage.get("uncertain_model_records")):
        assert observed["usage"]["input_tokens"] is None
        assert observed["usage"]["output_tokens"] is None
        assert observed["usage"]["known_input_tokens"] == usage.get("input_tokens")
        assert observed["usage"]["known_model_tokens"] == usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        if usage.get("uncertain_model_intents") or usage.get("uncertain_model_records") or usage.get("model_calls") == 0:
            assert observed["usage"]["model_calls"] is None
        elif usage.get("model_calls") == 1:
            assert observed["usage"]["model_calls"] == 1
        if "known_model_calls" in usage:
            expected = usage["known_model_calls"] if type(usage["known_model_calls"]) is int else None
            assert observed["usage"]["known_model_calls"] == expected
    assert observed["generation_complete"] is True and observed["method_status"] == "completed"
    assert observed["candidate"]["artifact_hash"] == candidate["artifact_hash"]
