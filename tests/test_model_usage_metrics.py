"""CPU metadata propagation: no real study DB, model, SSH or GPU."""
import json

import pytest

from optimizer_agent.suite_orchestrator import observed_metrics, write_json
from optimizer_agent.codex_backend import CodexResult
from optimizer_agent.research_service import ResearchService


def observe(tmp_path, events):
    write_json(tmp_path / "generation/clock.json", {"started_at": 0})
    write_json(tmp_path / "generation/events.json", events)
    return observed_metrics(tmp_path, {})


def model(usage, *, call_id="call", purpose="research", at=1, reservation_id="reserved"):
    return {"kind": "model", "created_at": at, "payload": {"call_id": call_id,
        "reservation_id": reservation_id, "purpose": purpose, "usage": usage, "elapsed_seconds": 2}}


@pytest.mark.parametrize("usage", [{}, {"cost_usd": None}, {"input_tokens": 3},
                                   {"input_tokens": True, "output_tokens": 2},
                                   {"input_tokens": -1, "output_tokens": 5},
                                   {"input_tokens": None, "output_tokens": 0}])
def test_completed_unknown_usage_is_not_a_zero_research_or_first_gain_total(tmp_path, usage):
    metrics = observe(tmp_path, [model(usage), {"kind": "promotion", "created_at": 4}])
    assert metrics["research_model_tokens"] is None
    assert metrics["model_tokens_to_first_useful_gain"] is None
    assert metrics["model_calls_to_first_useful_gain"] == 1
    assert metrics["research_model_usage"]["observed_calls"] == 1
    assert metrics["research_model_usage"]["usage_complete"] is False


def test_reported_zero_remains_known_and_matched_uncertainty_is_not_a_second_call(tmp_path):
    events = [model({"input_tokens": 0, "output_tokens": 0}),
              {"kind": "model_uncertain", "created_at": 2, "payload": {
                  "call_id": "call", "purpose": "research", "reservation_id": "reserved"}},
              {"kind": "promotion", "created_at": 4}]
    metrics = observe(tmp_path, events)
    assert metrics["research_model_tokens"] == metrics["model_tokens_to_first_useful_gain"] == 0
    assert metrics["model_calls_to_first_useful_gain"] == 1
    assert metrics["research_model_usage"]["observed_calls"] == 1
    assert metrics["unmatched_model_uncertain_observations"] == 0


def test_unknown_usage_and_matching_uncertainty_keep_single_call_unknown(tmp_path):
    metrics = observe(tmp_path, [model({}),
        {"kind": "model_uncertain", "created_at": 2, "payload": {"call_id": "call", "purpose": "research"}},
        {"kind": "promotion", "created_at": 4}])
    assert metrics["research_model_usage"]["observed_calls"] == 1
    assert metrics["research_model_usage"]["unknown_usage_calls"] == 1
    assert metrics["research_model_tokens"] is None
    assert metrics["model_calls_to_first_useful_gain"] == 1


def test_unmatched_uncertain_call_is_conservative_without_double_counting_duplicate_events(tmp_path):
    missing = {"kind": "model_uncertain", "created_at": 2, "payload": {
        "call_id": "missing", "purpose": "research", "reservation_id": "missing-reservation"}}
    metrics = observe(tmp_path, [model({"input_tokens": 3, "output_tokens": 2}), missing, missing,
                                {"kind": "promotion", "created_at": 4}])
    assert metrics["research_model_tokens"] is None
    assert metrics["research_model_usage"]["known_model_tokens"] == 5
    assert metrics["research_model_usage"]["observed_calls"] == 2
    assert metrics["unmatched_model_uncertain_observations"] == 1
    assert metrics["model_calls_to_first_useful_gain"] is None
    assert metrics["known_model_calls_to_first_useful_gain"] == 1
    assert metrics["research_model_wall_seconds"] is None


def test_uncertain_purpose_not_recorded_cannot_prove_zero_research(tmp_path):
    metrics = observe(tmp_path, [model({"input_tokens": 3, "output_tokens": 2}, purpose="proposal"),
        {"kind": "model_uncertain", "created_at": 2, "payload": {"reservation_id": "legacy-unknown"}}])
    assert metrics["research_model_tokens"] is None
    assert metrics["research_model_usage"]["unreported_calls"] == 1


def test_no_usage_evidence_remains_unknown_while_proven_no_research_calls_stays_zero(tmp_path):
    missing = observed_metrics(tmp_path, {})
    assert missing["research_model_tokens"] is None and missing["research_model_usage"] is None
    known = observe(tmp_path, [model({"input_tokens": 3, "output_tokens": 2}, purpose="proposal")])
    assert known["research_model_tokens"] == 0
    assert known["research_model_usage"]["observed_calls"] == 0


def test_promotion_alone_cannot_establish_zero_model_calls_or_tokens(tmp_path):
    metrics = observe(tmp_path, [{"kind": "promotion", "created_at": 4}])
    assert metrics["model_tokens_to_first_useful_gain"] is None
    assert metrics["model_calls_to_first_useful_gain"] is None
    assert metrics["model_usage_to_first_useful_gain"] is None
    assert "model_tokens_to_first_useful_gain" in metrics["unknown_token_metrics"]


@pytest.mark.parametrize("usages", [({}, {}), ({"input_tokens": 0, "output_tokens": 0},) * 2,
    ({"input_tokens": 10}, {"input_tokens": 20, "output_tokens": 2})])
def test_real_research_service_aggregation_preserves_unknowns_and_zero_without_relaxing_guard(tmp_path, usages):
    class Backend:
        prompts = []
        def generate_json(self, prompt, schema, **kwargs):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return CodexResult({"sources": [], "rejected_examples": [], "queries": []}, usages[0],
                    [{"type": "item.completed", "item": {"id": "web-search", "type": "web_search",
                      "action": {"type": "search", "queries": ["CUTLASS CCCL tiling"]}}}],
                    "gpt-6-luna", "max", 1, "search-call")
            contents = json.loads(prompt.split("Actually read source content:\n")[1])
            references = [entry["source"]["source_id"] for entry in contents if entry["source"]["kind"] == "code"]
            payload = dict(covered_techniques=["shared_tiling"], reference_kernels=references,
                launch_requirements=["all threads reach barriers"], dependencies=[],
                applicability_checks=[dict(condition="same ABI", evidence="launcher and contract checked", satisfied=True)],
                incompatible_conditions=[], missing_evidence=[], pmpp_topics=["tiling"],
                hypotheses=[dict(documented_fact="synchronization applies per block",
                    observed_code_property="cooperative tile reuse", expected_effect_to_measure="fewer reads", source_ids=references)],
                expected_effects=["lower GPU latency"], validation_metrics=["paired fresh target timings"],
                negative_search=[], original_hypothesis="")
            return CodexResult(payload, usages[1], [], "gpt-6-luna", "max", 1, "synthesis-call")
    def fetcher(url, timeout, limit):
        if "raw.githubusercontent.com" in url:
            return ("// source fixture\nstruct BlockReduce {};\n__global__ void gemm_device() {}\n" + "__global__ void kernel() {}\n" * 10).encode(), url
        return b"<html><p>Official NVIDIA synchronization documentation</p></html>", url
    dossier = ResearchService(Backend(), tmp_path, fetcher=fetcher).prepare(
        {"contract_hash": "hash"}, {}, {"strategy_family": "tiling"}, {"environment_fingerprint": "gpu"})
    assert dossier.research_ready("hash", "tiling", "gpu")
    if usages[0] == {}:
        assert dossier.usage["model_tokens"] is dossier.usage["known_model_tokens"] is None
        assert dossier.usage["unknown_usage_calls"] == 2
    elif "output_tokens" not in usages[0]:
        assert dossier.usage["input_tokens"] == 30 and dossier.usage["output_tokens"] is None
        assert dossier.usage["known_model_tokens"] == 32 and dossier.usage["usage_complete"] is False
    else:
        assert dossier.usage["model_tokens"] == 0 and dossier.usage["usage_complete"] is True
    assert dossier.usage["observed_calls"] == 2 and dossier.usage["cost_usd"] is None
