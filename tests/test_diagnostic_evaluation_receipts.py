"""CPU regressions from archived v2 CUDA receipts; no GPU/model is invoked."""
from copy import deepcopy
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from optimizer_agent import campaign
from optimizer_agent.diagnostics import build_diagnostic, evaluation_summary, compact_candidate_evidence
from optimizer_agent.research_service import ResearchBlocked, ResearchService
from tests.test_campaign_runtime import runtime_setup, selected


FIXTURES = Path(__file__).parent / "fixtures" / "diagnostics"


def recorded(name):
    if name.endswith(".gz"):
        return json.loads(gzip.decompress((FIXTURES / name).read_bytes()))
    return json.loads((FIXTURES / name).read_text())


def test_real_validation_list_preserves_compiler_error_launch_and_actual_environment():
    receipt = recorded("validation-compilation-error.json")
    original = deepcopy(receipt)
    summary = evaluation_summary(receipt)
    assert summary["status"] == "compilation_error" and summary["ok"] is False
    assert isinstance(summary["cases"], list) and len(summary["cases"]) == 1
    row, expected = summary["cases"][0], receipt["cases"][0]
    assert {key: row[key] for key in ("case_id", "seed", "repetition", "launch")} == {
        key: expected[key] for key in ("case_id", "seed", "repetition", "launch")}
    assert row["report"]["error"] == expected["report"]["error"]
    assert row["report"]["compiler_log"] == expected["report"]["compiler_log"]
    assert 'cannot open source file "math.h"' in row["report"]["compiler_log"]
    assert row["report"]["stage"] == "compile" and row["report"]["worker_reusable"] is True
    assert row["report"]["environment_identity"]["uuid"] == expected["report"]["environment"]["uuid"]
    assert summary["attempt_id"] == receipt["attempt_id"]
    assert receipt == original
    row["launch"]["block"][0] = 999
    assert receipt == original  # Summarization cannot mutate the immutable receipt.


def test_real_benchmark_mapping_preserves_raw_series_means_resources_and_protocol():
    receipt = recorded("benchmark-measured.json")
    original = deepcopy(receipt)
    summary = evaluation_summary(receipt)
    assert isinstance(summary["cases"], dict) and set(summary["cases"]) == {"search-1"}
    actual, expected = summary["cases"]["search-1"], receipt["cases"]["search-1"]
    assert actual["latency_ms"] == expected["latency_ms"]
    assert actual["weight"] == expected["weight"] and actual["critical"] == expected["critical"]
    raw_summary = actual["series"][0]["raw_timing_summary"]
    assert raw_summary["count"] == raw_summary["finite_count"] == 20 and raw_summary["unit"] == "ms"
    assert raw_summary["min_ms"] == min(expected["series"][0]["raw_ms"])
    assert raw_summary["max_ms"] == max(expected["series"][0]["raw_ms"])
    assert actual["series"][0]["mean_ms"] == expected["series"][0]["mean_ms"]
    assert actual["series"][0]["launch"] == expected["series"][0]["launch"]
    assert actual["series"][0]["report"]["kernel_resources"] == expected["series"][0]["report"]["kernel_resources"]
    assert actual["series"][0]["report"]["measurement_protocol"] == expected["series"][0]["report"]["measurement_protocol"]
    assert summary["protocol"] == receipt["protocol"] and summary["environment"] == receipt["environment"]
    assert "compilation_identity" not in actual["series"][0]["report"]  # Source stays in the referenced full receipt.
    assert receipt == original


def test_validation_repetitions_are_not_collapsed_by_case_id():
    receipt = recorded("validation-compilation-error.json")
    repeated = deepcopy(receipt["cases"][0])
    repeated["repetition"] = 1  # Synthetic second repetition tests representation, not CUDA correctness.
    receipt["cases"].append(repeated)
    summary = evaluation_summary(receipt)
    assert [row["case_id"] for row in summary["cases"]] == ["search-1", "search-1"]
    assert [row["repetition"] for row in summary["cases"]] == [0, 1]


def test_numeric_comparison_and_source_guard_failures_remain_typed():
    receipt = recorded("validation-compilation-error.json")
    receipt.update(status="numerical_error")
    receipt["cases"][0]["comparison"] = {"ok": False, "outputs": [{"ok": False, "mismatch_count": 3}]}
    summary = evaluation_summary(receipt)
    assert summary["status"] == "numerical_error"
    assert summary["cases"][0]["comparison"] == receipt["cases"][0]["comparison"]
    guard = {"ok": False, "status": "contract_violation", "cases": [],
             "violations": ["explicit half arithmetic"], "source_guard_scope": "source admission"}
    assert {key: evaluation_summary(guard)[key] for key in guard} == guard


def test_diagnostic_keeps_real_failure_and_profile_as_distinct_observations():
    receipt = recorded("validation-compilation-error.json")
    profile = {"ok": True, "metrics": {"fixture_counter": {"value": 12.5, "unit": "percent"}},
               "execution_environment": "local", "hardware_fingerprint": {"uuid": "simulated-local-gpu"},
               "diagnostic_only": True, "ranking_eligible": False}
    diagnostic = build_diagnostic(run_id="cpu-regression", trial_id="trial", candidate_id=receipt["candidate_id"],
        strategy_family="tiling", contract_hash=receipt["contract_hash"], environment_fingerprint="cpu-fixture",
        stage="result", error=None, needs_research=False, proposal=None, measurement=receipt,
        confirmation=None, profile=profile, previous_results=[], evidence_references=[], candidate=None, pmpp_guidance=None)
    observed = {row["kind"]: row for row in diagnostic["observations"]}
    assert observed["typed_failure"]["status"] == "compilation_error"
    assert observed["evaluation"]["value"]["cases"][0]["report"]["error"] == receipt["cases"][0]["report"]["error"]
    assert observed["profile"]["value"] == profile
    assert observed["profile"]["value"]["hardware_fingerprint"] != receipt["cases"][0]["report"]["environment"]


def test_repair_generation_evidence_consumes_real_receipt_without_model_or_gpu(runtime_setup):
    runtime, _, _, model = runtime_setup
    receipt = recorded("validation-compilation-error.json")
    # This historical GPU fixture has another contract than the CPU runtime.
    # Make that provenance explicit rather than relabeling its observed error.
    runtime.rehost_context = {"context_id":"cpu-test-historical-error",
        "old_proof":{"identity":{"contract":receipt["contract_hash"],"environment":"historical-gpu"},
                     "checkpoint":{"state":{"candidate_id":receipt["candidate_id"]}}}}
    state = {**selected(runtime), "candidate_id": receipt["candidate_id"], "repair": 1,
             "error": "compilation_error"}
    path = runtime.trial_dir(state) / f'evaluation-{state["candidate_id"]}.json'
    campaign.write_json(path, receipt)
    before = path.read_bytes()
    evidence = runtime.generation_evidence(state)
    assert evidence["previous_error"] == "compilation_error"
    assert evidence["previous_evaluation"]["status"] == "compilation_error"
    assert evidence["previous_evaluation"]["historical_context"]["ranking_eligible"] is False
    assert 'cannot open source file "math.h"' in evidence["previous_evaluation"]["cases"][0]["report"]["compiler_log"]
    reference = evidence["previous_evaluation_reference"]
    assert reference["path"] == str(path.relative_to(runtime.directory))
    assert reference["sha256"] == hashlib.sha256(before).hexdigest()
    assert reference["identity"] == receipt["evaluation_id"]
    assert path.read_bytes() == before and model.calls == []


def test_fixture_excerpt_hashes_are_portable_without_reading_the_original_campaign():
    provenance = recorded("provenance.json")
    assert "CPU diagnostic regressions" in provenance["evidence_kind"]
    for source in provenance["sources"]:
        assert hashlib.sha256((FIXTURES / source["fixture"]).read_bytes()).hexdigest() == source["fixture_sha256"]
        assert not Path(source["source_relative_to_repo"]).is_absolute()
        if source["fixture"].endswith(".gz"):
            decoded = gzip.decompress((FIXTURES / source["fixture"]).read_bytes())
            assert hashlib.sha256(decoded).hexdigest() == source["decoded_fixture_sha256"]


@pytest.mark.parametrize("value", [None, "search-1", 3])
def test_unknown_case_container_is_explicit_schema_error_not_empty_success(value):
    with pytest.raises(ValueError, match="case mapping or a validation list"):
        evaluation_summary({"status": "validated", "ok": True, "cases": value})


def test_actual_two_megabyte_parent_has_compact_exact_source_and_authenticated_raw_pairs(tmp_path):
    parent = recorded("confirmed-parent.json.gz")
    original = deepcopy(parent)
    assert len(json.dumps(parent)) > 2_000_000
    compact = compact_candidate_evidence(parent, tmp_path)
    assert {k: v for k, v in compact.items() if k != "evaluations"} == {
        k: v for k, v in parent.items() if k != "evaluations"}
    evaluation = compact["evaluations"][0]
    assert len(evaluation["pairs"]) == len(parent["evaluations"][0]["pairs"]) == 140
    pair, raw_pair = evaluation["pairs"][0], parent["evaluations"][0]["pairs"][0]
    assert pair["order"] == raw_pair["order"] and pair["case_id"] == raw_pair["case_id"]
    for candidate_id, row in evaluation["paired_case_statistics"][pair["case_id"]].items():
        assert row["series_means_ms"][0] == raw_pair["measurements"][candidate_id]["mean_ms"]
        assert len(row["series_indices"]) == row["report_count"] == 20
        assert row["raw_timing_summary"]["count"] == 400
        assert row["raw_timing_summary"]["unit"] == "ms"
        assert row["errors"] == [] and len(row["environment_identities"]) == 1
    reference = evaluation["raw_evidence_reference"]
    path = tmp_path / reference["path"]
    assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    assert json.loads(path.read_bytes()) == parent["evaluations"][0]
    assert parent == original
    assert len(json.dumps({"parent": compact, "effective_source_parent": compact})) < 1_048_576
    assert compact_candidate_evidence(parent, tmp_path) == compact
    path.write_text("tampered archive")
    with pytest.raises(ValueError, match="Immutable raw evaluation"):
        compact_candidate_evidence(parent, tmp_path)


def test_real_research_prompt_for_archived_large_parent_stays_below_cli_limit(tmp_path):
    parent = compact_candidate_evidence(recorded("confirmed-parent.json.gz"), tmp_path)
    prompts = []
    class Capture:
        def generate_json(self, prompt, *args, **kwargs):
            prompts.append(prompt)
            raise ResearchBlocked("CPU capture before any model or network invocation")
    service = ResearchService(Capture(), tmp_path / "research")
    with pytest.raises(ResearchBlocked, match="CPU capture"):
        service.prepare({"contract_hash": "fixture-contract"}, parent,
                        {"strategy_family": "tiling"}, {"environment_fingerprint": "fixture-environment"})
    assert len(prompts) == 1 and len(prompts[0]) < 1_048_576
    context = json.loads(prompts[0].rsplit("Context: ", 1)[1])
    assert context["parent"]["source"] == parent["source"]
    assert context["parent"]["evaluations"][0]["raw_evidence_reference"] == parent["evaluations"][0]["raw_evidence_reference"]


def test_campaign_research_and_generation_compact_parent_without_changing_lineage(runtime_setup, monkeypatch):
    runtime, config, runner, model = runtime_setup
    parent = recorded("confirmed-parent.json.gz")
    state = selected(runtime)
    trial_path = runtime.trial_dir(state) / "trial.json"
    trial = campaign.read_json(trial_path)
    trial["parent"] = parent
    campaign.write_json(trial_path, trial)
    original = trial_path.read_bytes()
    seen = []
    def capture_research(workload, prompt_parent, action, evidence):
        seen.append(prompt_parent)
        raise ResearchBlocked("CPU boundary capture; no dossier bypass")
    monkeypatch.setattr(runtime.research_service, "prepare", capture_research)
    assert runtime.research(state)["research_ready"] is False
    assert len(seen) == 1 and seen[0]["source"] == parent["source"]
    assert len(json.dumps(seen[0])) < 1_048_576 and trial_path.read_bytes() == original
    runtime.config = replace(config, experimental_mode="ablation_no_research")
    def capture_generation(prompt_parent, action, **kwargs):
        seen.append(prompt_parent)
        assert kwargs["source_parent"] == prompt_parent
        raise ResearchBlocked("CPU proposal boundary capture")
    monkeypatch.setattr(runtime.generator, "propose_unresearched", capture_generation)
    assert runtime.generate(state)["valid"] is False
    assert len(seen) == 2 and seen[1] == seen[0]
    assert trial_path.read_bytes() == original and model.calls == []


def test_bootstrap_prefix_must_be_consumed_before_store_or_model_construction(runtime_setup, monkeypatch, tmp_path):
    _, config, runner, model = runtime_setup
    def incomplete():
        raise ValueError("CPU bootstrap prefix incomplete")
    runner.require_complete = incomplete
    monkeypatch.setattr(campaign, "ExperimentStore", lambda *args, **kwargs: pytest.fail("Store precedes bootstrap completion"))
    monkeypatch.setattr(campaign, "RecordedCodex", lambda *args, **kwargs: pytest.fail("Model precedes bootstrap completion"))
    with pytest.raises(ValueError, match="bootstrap prefix incomplete"):
        campaign.CampaignRuntime(config, runner, tmp_path / "incomplete", model_backend=model)
    assert model.calls == []
