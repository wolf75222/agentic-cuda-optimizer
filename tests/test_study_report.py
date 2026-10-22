"""Observed consumption cannot double count cached/duplicate/reasoning tokens."""
import json
import hashlib
from copy import deepcopy
import subprocess
from pathlib import Path
import shutil

import pytest

from scripts.report_study_fr import (project_consumption, frozen_pilot_summary,
    frozen_pilot_panel, plot_frozen_pilot, FROZEN_ROLES, FROZEN_PAIRS)
from scripts.report_study_fr import checkout_source_identity, ci_qualifies_checkout, control_receipts
from scripts.report_study_fr import gpu_lease_consumption
from scripts.report_study_fr import (study_aggregate_table, study_campaign_table, case_comparison_table,
    campaign_metric_views, campaign_metrics_panel, campaign_call_observations, campaign_call_panel,
    workload_dispersion_table)
from scripts.report_study_fr import campaign_research_model_consumption, campaign_research_model_panel
from scripts.report_study_fr import DIAGNOSTIC_V2_AUDIT, POLICY_AUTHORITY, PARENT_OBSERVATION
from scripts.report_study_fr import (ORIGINAL_EXECUTION, COMMON_FINAL_FREEZE, RESEARCH_REFRESH,
    UPSTREAM_COMMIT, REFRESH_CHECKS, FREEZE_METHOD_CHECKS, LABELS, execution_evidence_panel)
from scripts.report_study_fr import ORIGINAL_CAMPAIGN, FINAL_DIRECTORY, FIRST_GROUP
from scripts.report_study_fr import comparison_view, direct_comparison_views, direct_comparison_table, plot
from scripts.report_study_fr import (RESEARCH_REFRESH_RESUMED, RESEARCH_REFRESH_RESUMED_2,
    RESEARCH_REFRESH_RESUMED_3, REFRESH_RESUMPTION_CHECKS, STANDALONE_FINAL_SUITE, FINAL_SUITE_CHECKS)
from optimizer_agent.suite_report import write_report, method_outcome_cells


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def method_outcome_summary():
    from optimizer_agent.quality_suite import summarize_suite
    modes = ["true_upstream", "full", "native_codex_vanilla"]
    plan = {"modes": modes, "workloads": ["convolution"], "entries": [
        {"workload": "convolution", "seed": 1729, "mode": mode, "group_id": "shared",
         "config": {"run_id": mode}} for mode in modes]}
    manifests = {
        "true_upstream": {"status": "method_failed", "method_status": "failed",
            "method_error": "compiler | <worker>\nexit 1", "generation_complete": True,
            "candidate_fallback": {"retained": True, "selected_iteration": 3,
                "selected_origin": "original_generated_proposal", "reason": "method_failed"},
            "candidate": {"artifact_hash": "retained-before-failure"}},
        "full": {"status": "completed", "method_status": "completed", "candidate_fallback": None},
        # A historical artifact can have a final measurement without a method outcome receipt.
        "native_codex_vanilla": {"candidate": {"artifact_hash": "historical-artifact"}}}
    candidates = {mode: {"status": "certified", "qualified": True,
        "artifact_hash": manifests[mode].get("candidate", {}).get("artifact_hash", "full-artifact"),
        "confirmation": {"interval": {"gain": gain, "lower": gain-.01, "upper": gain+.01}}}
        for mode, gain in zip(modes, [1.25, 1.1, .9])}
    return summarize_suite(plan, manifests, {"shared": {"execution_complete": True,
        "all_modes_available": True, "candidates": candidates,
        "method_comparisons": {"full_vs_"+other:{"ok":True,"status":"inconclusive",
            "interval":{"gain":1.1/gain,"lower":1.1/gain-.01,"upper":1.1/gain+.01}}
            for other,gain in (("true_upstream",1.25),("native_codex_vanilla",.9))}}})


def test_html_separates_failed_method_from_qualified_measured_fallback():
    summary = method_outcome_summary()
    html = study_campaign_table(summary)
    aggregate = study_aggregate_table(summary)
    assert summary["scientific_qualified"] is True  # Artifact qualification is a separate claim.
    assert "Statut méthode" in html and "Statut final de l’artefact" in html
    assert "<td>échouée</td>" in html and "<td>certified</td>" in html
    assert "compiler | &lt;worker&gt;\nexit 1" in html
    assert "<worker>" not in html
    assert "proposition de l’agent original, itération 3 (échec de la méthode)" in html
    assert "<td>1.25</td>" in html and "<td>retained-before-failure</td>" in html
    assert "Méthodes réussies" in aggregate and "Méthodes échouées" in aggregate
    assert "Statuts méthode inconnus" in aggregate
    assert summary["aggregates"]["true_upstream"]["method_completed"] == 0
    assert summary["aggregates"]["true_upstream"]["method_failed"] == 1
    assert summary["aggregates"]["native_codex_vanilla"]["method_status_unknown"] == 1


def test_markdown_preserves_failure_reason_fallback_gain_and_raw_json(tmp_path):
    summary = method_outcome_summary()
    report = write_report(tmp_path, {"summary": summary})
    markdown = (tmp_path/"rapport-fr.md").read_text()
    original = summary["rows"][0]
    assert "Méthodes réussies | Méthodes échouées | Statuts méthode inconnus" in markdown
    assert "échouée | compiler &#124; &lt;worker&gt;<br>exit 1" in markdown
    assert "itération 3 (échec de la méthode) | certified | 1.25 | [1.24 ; 1.26]" in markdown
    assert "transformer cet échec en succès" in markdown
    assert "| native_codex_vanilla | inconnu |" in markdown
    assert report["summary"]["rows"][0]["candidate_fallback"] == original["candidate_fallback"]
    assert report["summary"]["rows"][0]["generation_method_error"] == "compiler | <worker>\nexit 1"
    assert json.loads((tmp_path/"report.json").read_text())["summary"]["rows"][0]["gain"] == 1.25


def test_initial_fallback_iteration_zero_and_unknown_method_are_not_successes():
    status, error, fallback = method_outcome_cells({"generation_method_status": "failed",
        "candidate_fallback": {"retained": True, "selected_iteration": 0,
            "selected_origin": "supplied_initial", "reason": "method_failed"}})
    assert status == "échouée" and error == "non renseignée"
    assert "noyau initial fourni, itération 0" in fallback
    assert method_outcome_cells({}) == ["inconnu", "non renseignée", "inconnu"]
    assert method_outcome_cells({"generation_method_status": "completed", "candidate_fallback": None}) == [
        "réussie", "aucune signalée", "aucun"]


def test_consumption_deduplicates_calls_and_preserves_unknown_failures(tmp_path):
    call = {"call_id": "a"*32, "status": "completed", "usage": {
        "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20,
        "reasoning_output_tokens": 10}}
    save(tmp_path/"one/model"/("a"*32+".json"), call)
    save(tmp_path/"copied/model"/("a"*32+".json"), call)
    save(tmp_path/"failed/model"/("b"*32+".json"), {"status":"failed", "usage":{}})
    native = {"status":"completed", "model_started":True, "thread_ids":["native-thread"],
              "usage":{"input_tokens":200,"cached_input_tokens":160,"output_tokens":40}}
    save(tmp_path/"native/evidence/receipt.json", native)
    save(tmp_path/"copy/evidence/receipt.json", native)
    totals = project_consumption(tmp_path)
    assert totals["cli_call_receipts"] == 3
    assert totals["input_tokens"] == 300
    assert totals["cached_input_tokens"] == 240
    assert totals["output_tokens"] == 60
    assert totals["model_tokens"] == 360  # No second addition of cache/reasoning.
    assert totals["unreported_call_usage"] == 1
    assert totals["cost_usd"] is None


def test_same_call_with_conflicting_usage_cannot_silently_change_totals(tmp_path):
    save(tmp_path/"one/model"/("a"*32+".json"), {"status":"completed","usage":{"input_tokens":100,"output_tokens":20}})
    save(tmp_path/"two/model"/("a"*32+".json"), {"status":"completed","usage":{"input_tokens":200,"output_tokens":20}})
    with pytest.raises(ValueError, match="Conflicting observed usage"):
        project_consumption(tmp_path)


def test_original_intent_cli_and_legacy_copies_share_one_call_without_counting_acks(tmp_path):
    call_id = "d"*32
    receipt = {"call_id": call_id, "status": "completed", "usage": {
        "input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20}}
    paths = [tmp_path/"original/model-transport/intents/prompt-key/0/cli"/(call_id+".json"),
             tmp_path/"rehost/archive/model-transport/intents/prompt-key/0/cli"/(call_id+".json"),
             tmp_path/"legacy/model"/(call_id+".json")]
    for path in paths:
        save(path, receipt)
    # Even an ACK exposing the same usage is never a second provider call.
    save(tmp_path/"original/model-transport/intents/prompt-key/0/result.json", receipt)
    save(tmp_path/"original/model-transport/calls.json", [receipt])
    save(tmp_path/"unrelated/cli"/("e"*32+".json"), dict(receipt, call_id="e"*32))
    value = project_consumption(tmp_path)
    assert value["cli_call_receipts"] == 1 and value["model_tokens"] == 120
    assert value["cached_input_tokens"] == 80 and value["unreported_call_usage"] == 0
    call = value["call_provenance"][0]
    assert call["identity"] == "structured:"+call_id
    assert {row["path"] for row in call["receipts"]} == {str(path) for path in paths}
    assert {row["receipt_layout"] for row in call["receipts"]} == {"model", "original_intent_cli"}
    assert all(row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
               for row in call["receipts"] for path in paths if str(path) == row["path"])


@pytest.mark.parametrize("status,usage", [("failed", None), ("timeout", {})])
def test_original_intent_unknown_failures_survive_rehost_without_zero_invoice(tmp_path, status, usage):
    receipt = {"call_id": "e"*32, "status": status, "usage": usage, "error": "unreported CLI failure"}
    for name in ["original", "rehost"]:
        save(tmp_path/name/"model-transport/intents/key/0/cli"/("e"*32+".json"), receipt)
    value = project_consumption(tmp_path)
    assert value["cli_call_receipts"] == value["unreported_call_usage"] == 1
    assert value["calls_without_any_reported_usage"] == 1 and value["cost_usd"] is None
    assert value["call_provenance"][0]["observed_usage"] == {}
    assert len(value["call_provenance"][0]["receipts"]) == 2


def test_original_intent_conflicting_legacy_usage_fails_closed(tmp_path):
    receipt = {"call_id": "f"*32, "status": "completed", "usage": {"input_tokens": 100, "output_tokens": 20}}
    save(tmp_path/"original/model-transport/intents/key/0/cli"/("f"*32+".json"), receipt)
    save(tmp_path/"legacy/model"/("f"*32+".json"), dict(receipt, usage={"input_tokens": 200, "output_tokens": 20}))
    with pytest.raises(ValueError, match="Conflicting observed usage"):
        project_consumption(tmp_path)


def test_original_distinct_attempts_are_not_deduplicated_by_prompt_key(tmp_path):
    for attempt, char in enumerate(["a", "b"]):
        save(tmp_path/"original/model-transport/intents/shared-prompt"/str(attempt)/"cli"/(char*32+".json"),
             {"call_id": char*32, "status": "failed"})
    value = project_consumption(tmp_path)
    assert value["cli_call_receipts"] == value["unreported_call_usage"] == 2


def test_failed_timeout_and_unreported_structured_receipts_remain_visible(tmp_path):
    for char, status, extra in [("a", "failed", {}), ("b", "timeout", {"usage": None}),
                                ("c", "failed", {"usage": {"input_tokens": 0, "output_tokens": 0}})]:
        receipt = {"call_id": char*32, "status": status, "error": "CLI failed before usage", **extra}
        save(tmp_path/"original/model"/(char*32+".json"), receipt)
        save(tmp_path/"rehost/archive/model"/(char*32+".json"), receipt)
    value = project_consumption(tmp_path)
    assert value["cli_call_receipts"] == value["unreported_call_usage"] == 3
    assert value["calls_without_any_reported_usage"] == 3
    assert value["model_tokens"] == 0 and value["observed_lower_bounds"]
    assert value["cost_usd"] is None
    for call in value["call_provenance"]:
        assert len(call["receipts"]) == 2 and call["usage_reporting"] == "unreported"
        assert all(row["has_error"] and row["sha256"] for row in call["receipts"])


def native_before_thread():
    return {"model_started": True, "worker_session": "actual-native-session", "child_pid": 123,
            "started_at": 1234.5, "thread_ids": [], "status": "failed", "returncode": 1,
            "usage_events": [], "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}}


def test_native_started_before_first_thread_is_unknown_and_deduplicated_by_invocation(tmp_path):
    receipt = native_before_thread()
    for name in ["original", "copied-rehost"]:
        save(tmp_path/name/"evidence/receipt.json", receipt)
    value = project_consumption(tmp_path)
    assert value["cli_call_receipts"] == value["unreported_call_usage"] == 1
    assert value["call_provenance"][0]["kind"] == "native"
    assert value["call_provenance"][0]["identity_bases"] == ["session_pid_started_at"]
    assert value["call_provenance"][0]["observed_usage"] == {}
    assert value["cost_usd"] is None


def test_native_later_thread_snapshot_links_the_initial_invocation_and_copies(tmp_path):
    initial = native_before_thread()
    final = dict(initial, status="completed", thread_ids=["provider-thread"], returncode=0,
                 usage={"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20},
                 usage_events=[{"usage": {"input_tokens": 100, "output_tokens": 20}}])
    save(tmp_path/"before/evidence/receipt.json", initial)
    save(tmp_path/"completed/evidence/receipt.json", final)
    save(tmp_path/"archive/evidence/receipt.json", final)
    value = project_consumption(tmp_path)
    assert value["cli_call_receipts"] == 1 and value["unreported_call_usage"] == 0
    assert value["model_tokens"] == 120 and value["cached_input_tokens"] == 80
    assert len(value["call_provenance"][0]["receipts"]) == 3


def test_distinct_native_invocations_without_threads_are_not_merged(tmp_path):
    first = native_before_thread()
    save(tmp_path/"one/evidence/receipt.json", first)
    save(tmp_path/"two/evidence/receipt.json", dict(first, child_pid=456, started_at=1235.5))
    assert project_consumption(tmp_path)["unreported_call_usage"] == 2


def test_requested_native_thread_identity_and_unidentifiable_receipt_fail_closed(tmp_path):
    receipt = {"model_started": True, "requested_thread_id": "reserved-thread", "status": "failed"}
    save(tmp_path/"one/evidence/receipt.json", receipt)
    save(tmp_path/"copy/evidence/receipt.json", receipt)
    assert project_consumption(tmp_path)["unreported_call_usage"] == 1
    save(tmp_path/"unknown/evidence/receipt.json", {"model_started": True, "status": "failed"})
    with pytest.raises(ValueError, match="without thread or session/pid/start"):
        project_consumption(tmp_path)


def test_complementary_cached_usage_and_partial_failure_are_not_zero_invoices(tmp_path):
    receipt = {"call_id": "a"*32, "status": "completed", "usage": {"input_tokens": 100, "output_tokens": 20}}
    save(tmp_path/"one/model"/("a"*32+".json"), receipt)
    save(tmp_path/"copy/model"/("a"*32+".json"), dict(receipt, usage={**receipt["usage"], "cached_input_tokens": 80}))
    save(tmp_path/"failed/model"/("b"*32+".json"), {"status": "timeout", "usage": {
        "input_tokens": 200, "cached_input_tokens": 160, "output_tokens": 40, "reasoning_output_tokens": 30}})
    value = project_consumption(tmp_path)
    assert value["model_tokens"] == 360 and value["cached_input_tokens"] == 240
    assert value["cli_call_receipts"] == 2 and value["unreported_call_usage"] == 1
    assert value["partly_reported_call_usage"] == 1 and value["calls_without_any_reported_usage"] == 0


@pytest.mark.parametrize("usage", [{"input_tokens": 100, "cached_input_tokens": 101, "output_tokens": 20},
                                   {"input_tokens": 100, "output_tokens": 20, "reasoning_output_tokens": 21},
                                   {"input_tokens": -1, "output_tokens": 20}])
def test_impossible_usage_cannot_be_reported_as_a_lower_bound(tmp_path, usage):
    save(tmp_path/"one/model"/("a"*32+".json"), {"status": "completed", "usage": usage})
    with pytest.raises(ValueError, match="token"):
        project_consumption(tmp_path)


def lease(directory, job="123", elapsed="00:00:39", state="COMPLETED", gres="gpu:h100:1"):
    save(directory/"slurm-session.json", {"job_id": job, "session_id": "owned-"+job,
        "config": {"host": "romeo", "account": "project", "gres": gres}})
    if elapsed is not None:
        save(directory/"slurm-diagnostics/accounting.json", {"job_id": job, "state": state, "elapsed": elapsed})


@pytest.mark.parametrize("elapsed,seconds", [("00:00:39", 39), ("49:02:03", 176523),
                                             ("2-03:04:05", 183845), ("03:04", 184)])
def test_gpu_leases_use_recorded_slurm_elapsed_formats(tmp_path, elapsed, seconds):
    lease(tmp_path/"worker", elapsed=elapsed)
    value = project_consumption(tmp_path)
    assert value["observed_terminal_allocated_gpu_seconds"] == seconds
    assert value["observed_terminal_job_count"] == 1
    assert value["nonterminal_or_unreported_jobIDs"] == []


def test_failed_timeout_leases_and_rehost_copies_retain_cost_once(tmp_path):
    lease(tmp_path/"one/worker", "1", "00:00:05", "FAILED")
    lease(tmp_path/"archive/worker", "1", "00:00:05", "FAILED")
    lease(tmp_path/"two/worker", "2", "00:00:07", "TIMEOUT")
    lease(tmp_path/"three/worker", "3", "1-00:00:00", "COMPLETED")
    lease(tmp_path/"equivalent-copy/worker", "3", "24:00:00", "COMPLETED")
    value = gpu_lease_consumption(tmp_path)
    assert value["observed_terminal_allocated_gpu_seconds"] == 86412
    assert value["observed_terminal_job_count"] == 3
    assert len(next(row for row in value["gpu_lease_provenance"] if row["job_id"] == "1")["provenance"]) == 4


def test_active_unparsed_or_unverified_leases_remain_unknown_not_zero(tmp_path):
    lease(tmp_path/"active/worker", "1", elapsed=None)
    lease(tmp_path/"unparsed/worker", "2", elapsed="unknown")
    lease(tmp_path/"multi-gpu/worker", "3", gres="gpu:h100:2")
    save(tmp_path/"orphan/worker/slurm-diagnostics/accounting.json", {"job_id": "4", "state": "COMPLETED", "elapsed": "00:00:39"})
    value = gpu_lease_consumption(tmp_path)
    assert value["observed_terminal_job_count"] == 0
    assert value["nonterminal_or_unreported_jobIDs"] == ["1", "2", "3", "4"]
    assert value["unpaired_gpu_lease_receipts"]


@pytest.mark.parametrize("change", ["job_id", "duration", "session"])
def test_inconsistent_slurm_receipts_cannot_falsify_allocation_cost(tmp_path, change):
    lease(tmp_path/"one/worker")
    if change == "job_id":
        save(tmp_path/"one/worker/slurm-diagnostics/accounting.json", {"job_id": "other", "state": "COMPLETED", "elapsed": "00:00:39"})
    elif change == "duration":
        lease(tmp_path/"copy/worker", elapsed="00:01:00")
    else:
        lease(tmp_path/"copy/worker")
        save(tmp_path/"copy/worker/slurm-session.json", {"job_id": "123", "session_id": "different-owner",
             "config": {"host": "romeo", "account": "project", "gres": "gpu:h100:1"}})
    with pytest.raises(ValueError, match="Slurm"):
        gpu_lease_consumption(tmp_path)


def frozen_receipt():
    candidates = {role: dict(candidate_id=role, source="source " + role,
        source_hash=hashlib.sha256(("source " + role).encode()).hexdigest(), artifact_hash=role + "-artifact",
        provenance=dict(kind={"native_codex":"native_codex_one_prompt",
                              "single_prompt":"generation_only_structured_single_prompt"}.get(role, role)))
        for role in FROZEN_ROLES}
    pairs = {}
    values = {"agent_vs_baseline":(.830731,.823756,.835579),
              "single_prompt_vs_baseline":(1.000320,.988295,1.019778),
              "native_codex_vs_baseline":(1.120191,1.101939,1.142764),
              "agent_vs_native_codex":(.742984,.734017,.751730)}
    for name,(role,reference) in FROZEN_PAIRS.items():
        gain,low,high = values[name]
        pairs[name] = dict(ok=True,candidate_role=role,reference_role=reference,
            artifact_hash=candidates[role]["artifact_hash"],incumbent_artifact_hash=candidates[reference]["artifact_hash"],
            contract_hash="contract",environment={"uuid":"same-GPU"},status="measured",
            interval=dict(gain=gain,lower=low,upper=high,confidence=.95,case_gains={}))
    return dict(kind="independent_frozen_kernel_comparison",ok=True,status="frozen_comparison_complete",job_id="770762",
        terminal=dict(job_id="770762",state="COMPLETED",exit_code="0:0"),candidates=candidates,
        hardware_identity={"uuid":"same-GPU"},contract_hash="contract",shapes=[122,169,316],model_calls=0,
        validations={role:{"complete":{"ok":True}} for role in FROZEN_ROLES},
        sanitizer_qualification={role:{"ok":True} for role in FROZEN_ROLES},paired_comparisons=pairs)


def test_frozen_pilot_preserves_raw_regression_native_identity_and_separate_scope(tmp_path):
    receipt = frozen_receipt()
    original = deepcopy(receipt)
    path = tmp_path/"comparison.json"
    save(path, receipt)
    pilot = frozen_pilot_summary(path)
    assert receipt == original
    assert pilot["receipt_verified"]
    assert pilot["scope"]["included_in_section16_cohorts"] is False
    assert pilot["scope"]["retained_best_substitution"] is False
    assert pilot["scope"]["original_upstream_present"] is False
    assert len(pilot["sources"]) == 4
    assert pilot["comparisons"]["agent_vs_baseline"]["interval"]["gain"] == .830731
    assert pilot["comparisons"]["agent_vs_native_codex"]["interval"]["gain"] == .742984
    assert pilot["comparisons"]["native_codex_vs_baseline"]["interval"]["gain"] == 1.120191
    # Independent AB/BA comparison must not be replaced by a ratio of aggregates.
    assert pilot["comparisons"]["agent_vs_native_codex"]["interval"]["gain"] != .830731/1.120191
    html = frozen_pilot_panel(pilot, None)
    assert "aucun agent original upstream" in html
    assert "cohortes de l’étude §16" in html
    assert "sans substitution" in html and "régression reste visible" in html
    assert "sans outils" in html and "véritable Codex CLI vanilla natif" in html


@pytest.mark.parametrize("change", ["terminal", "source_hash", "native_identity", "pair_gpu", "pair_reference"])
def test_partial_or_mismatched_pilot_cannot_produce_a_qualified_plot(tmp_path, change):
    receipt = frozen_receipt()
    if change == "terminal": receipt["terminal"]["state"] = "RUNNING"
    if change == "source_hash": receipt["candidates"]["agent"]["source_hash"] = "wrong"
    if change == "native_identity": receipt["candidates"]["native_codex"]["provenance"]["kind"] = "structured_single_prompt"
    if change == "pair_gpu": receipt["paired_comparisons"]["agent_vs_baseline"]["environment"]["uuid"] = "other-GPU"
    if change == "pair_reference": receipt["paired_comparisons"]["agent_vs_native_codex"]["incumbent_artifact_hash"] = "baseline-artifact"
    path = tmp_path/"comparison.json"
    save(path, receipt)
    pilot = frozen_pilot_summary(path)
    assert not pilot["receipt_verified"]
    assert pilot["validation_errors"]
    assert plot_frozen_pilot(pilot, tmp_path) is None


def test_frozen_plot_and_missing_receipt(tmp_path):
    missing = frozen_pilot_summary(tmp_path/"absent.json")
    assert not missing["receipt_verified"]
    assert missing["comparisons"] == {}
    path = tmp_path/"comparison.json"
    save(path, frozen_receipt())
    target = plot_frozen_pilot(frozen_pilot_summary(path), tmp_path)
    assert target.name == "pilote-gemm-fige-romeo.png"
    assert target.read_bytes().startswith(b"\x89PNG")


def test_green_ci_on_same_head_does_not_qualify_tracked_or_untracked_changes(tmp_path):
    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "--quiet")
    (tmp_path/"source.py").write_text("original source\n")
    (tmp_path/".gitignore").write_text("outputs/\nresults/\nwork/\n")
    git("add", "source.py", ".gitignore")
    git("-c", "user.name=Report fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Source fixture")
    clean = checkout_source_identity(tmp_path)
    ci = {"headSha":clean["head"],"status":"completed","conclusion":"success"}
    assert ci_qualifies_checkout(ci, clean)
    save(tmp_path/"results/generated.json", {"generated":True})
    assert not checkout_source_identity(tmp_path)["dirty"]
    (tmp_path/"source.py").write_text("different generator source\n")
    dirty = checkout_source_identity(tmp_path)
    assert dirty["head"] == ci["headSha"] and dirty["dirty"]
    assert dirty["tracked_changes"] == [{"status":" M","path":"source.py"}]
    assert ci_qualifies_checkout(ci, dirty) is False
    (tmp_path/"source.py").write_text("original source\n")
    (tmp_path/"new config.toml").write_text("new_generator = true\n")
    untracked = checkout_source_identity(tmp_path)
    assert untracked["tracked_changes"] == []
    assert untracked["untracked_files"] == ["new config.toml"]
    assert ci_qualifies_checkout(ci, untracked) is False


def test_new_gpu_receipts_keep_executed_source_head_and_hashes_separate_from_report_head(tmp_path):
    executed_head = "6292b7d88467ebd528c816d5bdb9daf1760d3b86"
    for family,job in [("gpu-rpc-acceptance","770863"),("gpu-final-isolation-acceptance","770862")]:
        save(tmp_path/"results"/family/"20261021-current/allocation-receipt.json", {
            "ok":True,"job_id":job,"worker_terminal":{"job_id":job,"state":"COMPLETED","exit_code":"0:0"},
            "source_provenance":{"HEAD":executed_head,"dirty_state":[],"sources":{"worker.py":"frozen-source-sha"}},
            "sources_unchanged":True,"scope":"Only the explicitly executed acceptance control"})
    rows = [row for row in control_receipts(tmp_path) if row["job_id"] in ("770862","770863")]
    assert len(rows) == 2
    for row in rows:
        assert row["source_head"] == executed_head and row["source_dirty_state"] == []
        assert row["execution_source_sha256"] == {"worker.py":"frozen-source-sha"}
        assert row["terminal"]["state"] == "COMPLETED"
        assert row["receipt_sha256"] and row["sources_unchanged"]


def test_reconstructed_real_control_and_missing_diagnostic_v2_have_honest_scope(tmp_path):
    save(tmp_path/"results/gpu-reconstructed-acceptance/20261021-current/acceptance-summary.json", {
        "ok": True, "job_id": "771414", "source_HEAD": "412d857",
        "source_sha256_map": {"portable_export.py": "executed-source-hash"}, "snapshot_dirty_after": [],
        "terminal": {"job_id": "771414", "state": "COMPLETED", "exit_code": "0:0"},
        "scope": "One reconstructed correctness-only run;zero benchmark or model calls"})
    rows = control_receipts(tmp_path)
    actual = next(row for row in rows if row["job_id"] == "771414")
    assert actual["available"] and actual["source_head"] == "412d857"
    assert actual["execution_source_sha256"] == {"portable_export.py": "executed-source-hash"}
    pending = next(row for row in rows if row["label"] == "Diagnostic ciblé avec lecteur v2")
    assert not pending["available"] and pending["ok"] is None
    assert pending["path"] == DIAGNOSTIC_V2_AUDIT
    assert pending["receipt_sha256"] is pending["job_id"] is None
    assert "en attente" in pending["status"]


def diagnostic_v2_audit_fixture(tmp_path):
    def linked(name, value):
        path = tmp_path/"audit-linked"/name
        save(path, value)
        return {"path":str(path.relative_to(tmp_path)), "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    chains = []
    for index, mode in enumerate(["full", "greedy_parent", "no_autotuning"]):
        diagnostic_id = "diag-"+mode
        diagnostic = {**linked(mode+"-diagnostic.json", {"diagnostic_id":diagnostic_id}),
                      "diagnostic_id":diagnostic_id, "immutable_identity_and_evidence_hashes_verified":True}
        dossier = {**linked(mode+"-dossier.json", {"diagnostic_ids":[diagnostic_id]}),
                   "diagnostic_ids":[diagnostic_id], "research_ready_guard_replayed":True,
                   "all_source_bodies_passages_and_supplied_content_verified":True,
                   "pmpp_installed_original_and_cards_verified":True,
                   "supplied_content":linked(mode+"-supplied.json", {}),
                   "web_events":linked(mode+"-web.json", [])}
        proposal = {**linked(mode+"-proposal.json", {}),
                    "strict_schema_parent_contract_family_and_dossier_guard_replayed":True}
        measurement = {**linked(mode+"-measurement.json", {"ok":True}), "ok":True, "status":"measured",
                       "candidate_artifact_recomputed":True,
                       "all_series_owned_job_same_hardware_and_exact_proposal_source":True,
                       "candidate_source":linked(mode+"-source.json", {"test_source":"copy"}),
                       "cases":[{"case_id":str(n), "samples":140} for n in range(7)]}
        chains.append({"mode":mode, "status":"diagnostic_targeted_research_proposal_measured",
                       "diagnostic":diagnostic,"targeted_dossier":dossier,"next_proposal":proposal,
                       "measurement":measurement,"next_trial":{"diagnostic_id":diagnostic_id},
                       "worker":{"job_id":str(771747+index)}})
    value = {"kind":"actual_convolution_diagnostic_targeted_research_read_only_audit",
             "controller_launch_source":"b9bda60145ebf438fecd2837bdc2b1b00474674c",
             "all_three_chains_measured":True,"chains":chains,
             "relevant_source_versions":{"research_service.py":{"frozen_sha256":"a"*64,"launch_sha256":"a"*64}}}
    save(tmp_path/DIAGNOSTIC_V2_AUDIT, value)
    return value


def test_actual_diagnostic_audit_replaces_pending_only_with_verified_measured_chain_links(tmp_path):
    audit = diagnostic_v2_audit_fixture(tmp_path)
    row = next(r for r in control_receipts(tmp_path) if r["label"]=="Diagnostic ciblé avec lecteur v2")
    assert row["available"] and row["ok"] is True and all(row["authentication_checks"].values())
    assert row["source_head"] == audit["controller_launch_source"]
    assert row["job_ids"] == ["771747","771748","771749"]
    assert row["new_gpu_execution"] is False and "7 cas" in row["scope"]
    assert "Profilage indisponible" in row["scope"] and "Aucune nouvelle exécution GPU" in row["scope"]
    path = tmp_path/DIAGNOSTIC_V2_AUDIT
    assert row["receipt_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # A past assertion is not sufficient if a linked measured artifact changed.
    measured = tmp_path/audit["chains"][0]["measurement"]["path"]
    measured.write_text('{"ok":false}')
    failed = next(r for r in control_receipts(tmp_path) if r["label"]==row["label"])
    assert failed["available"] and failed["ok"] is False
    assert failed["authentication_checks"]["full"] is False


def test_present_but_unproved_diagnostic_receipt_is_not_qualification(tmp_path):
    save(tmp_path/DIAGNOSTIC_V2_AUDIT, {"all_three_chains_measured":True})
    row = next(r for r in control_receipts(tmp_path) if r["label"]=="Diagnostic ciblé avec lecteur v2")
    assert row["available"] and row["ok"] is False
    assert "incomplète" in row["status"]


def test_partial_authorities_and_secondary_parent_observation_remain_separate(tmp_path):
    terminal = {"state":"COMPLETED","exit_code":"0:0"}
    save(tmp_path/"results/diagnostic-loop-v2-20261021/diagnostic-loop-acceptance.json", {
        "status":"incomplete","ok":False,"job_id":"773095","terminal":dict(terminal,job_id="773095")})
    save(tmp_path/POLICY_AUTHORITY, {"status":"partial_or_incomplete","ok":False,"job_id":"773380",
        "terminal":dict(terminal,job_id="773380"),"criteria_checks":{"AC-05":False,"AC-06":True,"AC-10":True}})
    primary = tmp_path/POLICY_AUTHORITY
    save(tmp_path/PARENT_OBSERVATION, {"status":"observed_general_AC05_with_strict_control_still_false",
        "job_id":"773380","terminal":dict(terminal,job_id="773380"),
        "authority":{"primary_ok":False,"strict_AC05":False,
            "primary_receipt":{"path":primary.name,"sha256":hashlib.sha256(primary.read_bytes()).hexdigest()}},
        "execution_source":{"HEAD":"498d3be","source_sha256":{"search_policy.py":"observed-sha"}}})
    rows = control_receipts(tmp_path)
    helper = next(r for r in rows if r["job_id"]=="773095")
    policy = next(r for r in rows if r["path"]==POLICY_AUTHORITY)
    observation = next(r for r in rows if r["path"]==PARENT_OBSERVATION)
    assert helper["ok"] is False and helper["status"]=="incomplete"
    assert "sans exécution/mesure/intégration" in helper["scope"]
    assert policy["ok"] is False and policy["criteria_checks"]=={"AC-05":False,"AC-06":True,"AC-10":True}
    assert "AC-05 faux" in policy["scope"]
    assert observation["ok"] is None and observation["authority_receipt_verified"] is True
    assert observation["authority"]["strict_AC05"] is False and "ne les requalifie pas" in observation["scope"]
    assert observation["source_head"]=="498d3be"


def test_tuning_constraint_control_remains_pending_until_success_is_explicit(tmp_path):
    relative = "results/tuning-constraints-20261021/tuning-constraints-acceptance.json"
    row = next(r for r in control_receipts(tmp_path) if r["path"]==relative)
    assert row["available"] is False and row["ok"] is None and row["status"]=="en attente"
    save(tmp_path/relative, {"status":"started"})
    present = next(r for r in control_receipts(tmp_path) if r["path"]==relative)
    assert present["available"] is True and present["ok"] is None and present["status"]=="started"


def linked_report_fixture(root, name, value):
    """Invented CPU fixture receipts, never production evidence."""
    path = root/name
    save(path, value)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def original_execution_fixture(root):
    upstream = [{"path": "optimizer_agent/fixture-"+str(n)+".py", "sha256": format(n, "064x"), "matches_git_blob": True}
                for n in range(16)]
    adapters = [{"path": name, "sha256": format(n+20, "064x"), "matches_executed_frozen_source": True}
                for n,name in enumerate(("upstream_original.py", "frozen_launch.py", "codex_backend.py", "gain_observer.py"))]
    calls = []
    for index in range(10):
        call_id = format(index+1, "032x")
        usage = {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20}
        ref = linked_report_fixture(root, ORIGINAL_CAMPAIGN+"/generation/model-transport/intents/"+"a"*64+"/0/cli/"+call_id+".json", {
            "call_id": call_id, "status": "completed", "model": "gpt-6-luna", "reasoning_effort": "max", "usage": usage})
        calls.append({**ref, "call_id": call_id, "requested_model": "gpt-6-luna", "reasoning_effort": "max", "usage": usage})
    source = linked_report_fixture(root, ORIGINAL_CAMPAIGN+"/generation/frozen-original.cu", {"invented_fixture_source": True})
    terminal = {"job_id": "123", "state": "COMPLETED", "exit_code": "0:0"}
    usage = {"model_calls": 10, "input_tokens": 1000, "cached_input_tokens": 800,
             "output_tokens": 200, "model_tokens": 1200, "gain_observer_wall_seconds": 12.5}
    candidate = {"candidate_id": "initial-0", "artifact_hash": "a"*64, "source_hash": source["sha256"]}
    manifest = {"method": "true_upstream_original", "generation_complete": True, "method_status": "completed",
        "upstream": {"commit": UPSTREAM_COMMIT, "source_hashes": {item["path"]: item["sha256"] for item in upstream}},
        "adapter_source_hashes": {item["path"]: item["sha256"] for item in adapters}, "usage": usage, "candidate": candidate}
    value = {"schema_version": 1, "kind": "actual_pinned_original_generation", "ok": True,
        "source_controller": "e"*40, "job_id": "123", "session_id": "suite-fixture", "worker_terminal": terminal,
        "upstream_commit": UPSTREAM_COMMIT, "upstream_source_checks": upstream, "adapter_checks": adapters,
        "completed_cli_calls": calls, "usage": usage, "candidate": {"candidate_id": "initial-0", "artifact_hash": "a"*64, "source": source},
        "original_manifest": linked_report_fixture(root, ORIGINAL_CAMPAIGN+"/generation/manifest.json", manifest),
        "result": linked_report_fixture(root, ORIGINAL_CAMPAIGN+"/result.json", {**manifest, "job_id": "123",
            "session_id": "suite-fixture", "worker_terminal": terminal}),
        "history": linked_report_fixture(root, ORIGINAL_CAMPAIGN+"/generation/history.json", [{"iteration": n,
            "case_runs": [{"validation_ok": True} for _ in range(7)]} for n in range(4)]),
        "checks": {name: True for name in ("actual_original_workflow_completed", "all_28_search_case_runs_valid",
            "all_upstream_sources_match_pinned_git", "all_adapter_bytes_match_frozen_controller",
            "all_ten_cli_receipts_completed", "candidate_artifact_recomputed")}}
    save(root/ORIGINAL_EXECUTION, value)
    return value


def test_original_actual_generation_links_are_presented_without_final_success_or_attested_model(tmp_path):
    value = original_execution_fixture(tmp_path)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == ORIGINAL_EXECUTION)
    assert row["ok"] is True and all(row["authentication_checks"].values())
    assert row["final_qualified"] is False and row["new_gpu_execution"] is False
    assert row["source_head"] == value["source_controller"] and row["upstream_commit"] == UPSTREAM_COMMIT
    html = execution_evidence_panel([row])
    assert "1200 tokens observés" in html and "10 invocations CLI" in html
    assert "observateur séparé 12.5 s" in html and "ne remplacent pas une comparaison finale" in html
    assert "poids fournisseur restent non attestés" in row["scope"]


@pytest.mark.parametrize("change", ["linked_file", "pinned_revision", "missing_adapter", "bad_terminal", "failed_method", "missing_check"])
def test_original_source_or_cli_or_method_failures_cannot_be_relabelled_completed(tmp_path, change):
    value = original_execution_fixture(tmp_path)
    if change == "linked_file":
        save(tmp_path/value["completed_cli_calls"][0]["path"], {"status": "failed"})
    elif change == "pinned_revision":
        value["upstream_commit"] = "f"*40
    elif change == "missing_adapter":
        value["adapter_checks"].pop()
    elif change == "bad_terminal":
        value["worker_terminal"]["state"] = "RUNNING"
    elif change == "missing_check":
        value["checks"].pop("candidate_artifact_recomputed")
    else:
        # A valid retained artifact can survive method failure; the execution
        # control must not reclassify that failure from a generic stage flag.
        result = json.loads((tmp_path/value["result"]["path"]).read_text())
        result.update(method_status="failed", generation_complete=True, candidate_fallback={"retained": True})
        value["result"] = linked_report_fixture(tmp_path, value["result"]["path"], result)
    save(tmp_path/ORIGINAL_EXECUTION, value)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == ORIGINAL_EXECUTION)
    assert row["available"] is True and row["ok"] is False and row["final_qualified"] is False
    assert row["usage"] == {}


def common_freeze_fixture(root):
    naive_ref = linked_report_fixture(root, FINAL_DIRECTORY+"/sealed-inputs/baseline.cu", {"invented_fixture_source": "naive"})
    naive = {"source_path": naive_ref["path"], "source_sha256": naive_ref["sha256"], "artifact_sha256": "a"*64}
    methods = []
    for index,mode in enumerate(LABELS):
        different = mode == "native_codex_vanilla"
        body = {"invented_fixture_source": "native" if different else "naive"}
        source = linked_report_fixture(root, FINAL_DIRECTORY+"/frozen/"+mode+"/kernel.cu", body)
        campaign = "results/quality-study-v2/campaigns/"+FIRST_GROUP+"."+mode
        source_name = {"true_upstream": "frozen-original.cu", "native_codex_vanilla": "evidence/kernel.cu"}.get(mode, "best.cu")
        gen_source = linked_report_fixture(root, campaign+"/generation/"+source_name, body)
        artifact = "b"*64 if mode == "true_upstream" else "c"*64 if different else "a"*64
        candidate = {"candidate_id": mode, "artifact_hash": artifact, "source_hash": source["sha256"]}
        job,session = str(200+index), "fixture-"+mode
        result = linked_report_fixture(root, campaign+"/result.json", {
            "candidate": candidate, "job_id": job, "session_id": session, "execution_complete": True})
        methods.append({"method": mode, "candidate_id": mode, "artifact_sha256": artifact,
            "frozen_source": source, "generation_source": gen_source, "generation_result": result,
            "generation_job_id": job, "generation_session_id": session,
            "same_source_as_naive": not different, "same_artifact_as_naive": artifact == naive["artifact_sha256"],
            "checks": {name: True for name in FREEZE_METHOD_CHECKS}})
    value = {"schema_version": 1, "kind": "read_only_common_final_candidate_freeze_audit",
        "freeze_identity_checks_pass": True, "errors": [], "naive": naive, "methods": methods,
        "frozen_candidates": linked_report_fixture(root, FINAL_DIRECTORY+"/frozen-candidates.json", {"fixture_only": True}),
        "worker": {"job_id": "456", "session_id": "suite-final-fixture",
            "hardware_receipt": linked_report_fixture(root, FINAL_DIRECTORY+"/worker/hardware.json", {"slurm_job_id": "456", "environment": "romeo"}),
            "session_receipt": linked_report_fixture(root, FINAL_DIRECTORY+"/worker/slurm-session.json", {"job_id": "456", "session_id": "suite-final-fixture"})},
        "source_observation": {"HEAD": "d"*40},
        "checks": {name: True for name in ("all_eight_generation_to_frozen_identity_checks_pass",
            "all_observed_files_unchanged_before_receipt", "frozen_has_exact_eight_methods_plus_naive",
            "naive_load_candidate_accepts_artifact", "naive_source_matches_static_convolution", "worker_identity_checks_pass")}}
    save(root/COMMON_FINAL_FREEZE, value)
    return value


def test_final_freeze_aliases_are_identities_not_eight_distinct_kernels_or_gains(tmp_path):
    value = common_freeze_fixture(tmp_path)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == COMMON_FINAL_FREEZE)
    assert row["ok"] is True and all(row["authentication_checks"].values())
    assert row["final_qualified"] is False and row["new_gpu_execution"] is False
    assert row["source_head"] is None and row["audit_source_head"] == value["source_observation"]["HEAD"]
    assert sum(r["same_artifact_as_naive"] for r in row["artifact_aliases"]) == 6
    html = execution_evidence_panel([row])
    assert html.count("identique au naïf") == 6
    assert "même source, lancement/paramètres distincts" in html and "source distincte" in html
    assert "aucune différence de performance" in html


@pytest.mark.parametrize("change", ["same_source_flag", "same_artifact_flag", "method_check", "worker_job", "changed_frozen_copy"])
def test_frozen_identity_links_and_alias_declarations_fail_closed(tmp_path, change):
    value = common_freeze_fixture(tmp_path)
    if change == "same_source_flag":
        value["methods"][0]["same_source_as_naive"] = False
    elif change == "same_artifact_flag":
        value["methods"][0]["same_artifact_as_naive"] = False
    elif change == "method_check":
        value["methods"][0]["checks"].pop("artifact_hash_matches_generation")
    elif change == "worker_job":
        value["worker"]["job_id"] = "other"
    else:
        save(tmp_path/value["methods"][0]["frozen_source"]["path"], {"source": "changed"})
    save(tmp_path/COMMON_FINAL_FREEZE, value)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == COMMON_FINAL_FREEZE)
    assert row["ok"] is False and row["artifact_aliases"] == []


def refresh_documentary_fixture(root):
    directory = root/RESEARCH_REFRESH
    directory = directory.parent
    files = {}
    def artifact(name, value):
        reference = linked_report_fixture(directory, name, value)
        files[name] = reference["sha256"]
    calls = []
    for index in range(4):
        call_id = format(index+1, "032x")
        calls.append({"call_id": call_id})
        artifact("campaign/model/"+call_id+".json", {"call_id": call_id, "status": "completed", "model": "gpt-6-luna", "reasoning_effort": "max"})
    artifact("plan.json", {"kind": "documentary_research_refresh_acceptance"})
    artifact("run-intent.json", {"run_id": "fixture-documentary", "source_provenance": {"HEAD": "e"*40}, "new_gpu_requests": 0})
    artifact("events.json", [])
    artifact("campaign/acceptance-controls/invocations.json", [])
    artifact("campaign/acceptance-evidence/initial-dossier.json", {"research_id": "initial", "receipt_version": 2, "covered_techniques": ["tiling"]})
    artifact("campaign/trials/fixture-trial/dossier.json", {"research_id": "renewed", "receipt_version": 2,
        "covered_techniques": ["tiling", "warp_shuffle"], "diagnostic_ids": ["diag-fixture"]})
    artifact("campaign/acceptance-evidence/blocked-refusal.json", {"technique": "warp_shuffle", "call_id": calls[2]["call_id"],
        "checks": {name: True for name in ("needs_research_returned", "exact_semantic_guard_reason", "model_actually_declared_control_technique",
            "no_new_generation_event", "no_new_candidate_or_source_change", "no_accepted_proposal_file")}})
    proof = linked_report_fixture(directory, "proof-sha256.json", {"schema_version": 1, "hash_algorithm": "sha256", "files": files})
    value = {"schema_version": 1, "kind": "documentary_research_refresh_acceptance", "run_id": "fixture-documentary",
        "ok": True, "status": "documentary_research_refresh_acceptance_passed", "checks": {name: True for name in REFRESH_CHECKS},
        "new_gpu_requests": 0, "new_gpu_requests_attempted": 0, "hardware_currently_observed": False,
        "bootstrap_files_unchanged": True, "source_provenance": {"HEAD": "e"*40}, "proof_manifest": proof,
        "initial_call_id": calls[0]["call_id"], "compatible_call_id": calls[1]["call_id"],
        "blocked_call_id": calls[2]["call_id"], "final_call_id": calls[3]["call_id"], "model_calls": calls,
        "initial_research_id": "initial", "renewed_research_id": "renewed", "diagnostic_id": "diag-fixture", "uncovered_technique": "warp_shuffle",
        "initial_dossier_sha256": files["campaign/acceptance-evidence/initial-dossier.json"],
        "renewed_dossier_sha256": files["campaign/trials/fixture-trial/dossier.json"], "historical_bootstrap": {"job_id": "old-job"}}
    save(root/RESEARCH_REFRESH, value)
    return value


def test_absent_documentary_result_is_pending_or_unknown_never_gpu_success(tmp_path):
    row = next(r for r in control_receipts(tmp_path) if r["path"] == RESEARCH_REFRESH)
    assert row["ok"] is None and row["job_id"] is None and not row["available"]
    assert "en attente" in row["status"] and row["new_gpu_execution"] is False
    save((tmp_path/RESEARCH_REFRESH).parent/"run-intent.json", {"started": True})
    row = next(r for r in control_receipts(tmp_path) if r["path"] == RESEARCH_REFRESH)
    assert "en cours ou interrompu" in row["status"] and row["ok"] is None


def test_documentary_control_success_is_limited_to_authenticated_json_subset(tmp_path):
    refresh_documentary_fixture(tmp_path)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == RESEARCH_REFRESH)
    assert row["ok"] is True and all(row["authentication_checks"].values())
    assert row["job_id"] is None and row["historical_bootstrap_job_id"] == "old-job"
    assert row["new_gpu_execution"] is False and row["final_qualified"] is False
    assert row["database_or_final_inputs_read"] is False
    assert all(name.endswith(".json") for name in row["verified_json_subset"])
    assert "sans compilation, benchmark, sanitizer" in row["scope"]


def test_documentary_authentication_never_opens_manifest_databases_rng_or_numeric_inputs(tmp_path, monkeypatch):
    value = refresh_documentary_fixture(tmp_path)
    directory = (tmp_path/RESEARCH_REFRESH).parent
    proof = json.loads((directory/"proof-sha256.json").read_text())
    forbidden = ("campaign/checkpoints.sqlite3", "campaign/search.pkl", "private/final-input.bin")
    proof["files"].update({name: "a"*64 for name in forbidden})
    value["proof_manifest"] = linked_report_fixture(directory, "proof-sha256.json", proof)
    save(tmp_path/RESEARCH_REFRESH, value)
    read = Path.read_bytes
    def checked_read(path):
        assert not any(str(path).endswith(name) for name in forbidden)
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", checked_read)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == RESEARCH_REFRESH)
    assert row["ok"] is True and row["database_or_final_inputs_read"] is False
    assert not set(forbidden).intersection(row["verified_json_subset"])


@pytest.mark.parametrize("change", ["missing_check", "new_gpu", "current_hardware", "same_dossier", "changed_proof", "changed_model_call", "incomplete"])
def test_documentary_partial_or_unverified_proof_never_becomes_acquired(tmp_path, change):
    value = refresh_documentary_fixture(tmp_path)
    if change == "missing_check":
        value["checks"].pop("uncovered_technique_blocked_before_candidate")
    elif change == "new_gpu":
        value["new_gpu_requests_attempted"] = 1
    elif change == "current_hardware":
        value["hardware_currently_observed"] = True
    elif change == "same_dossier":
        value["renewed_research_id"] = value["initial_research_id"]
    elif change == "changed_proof":
        save((tmp_path/RESEARCH_REFRESH).parent/"proof-sha256.json", {"files": {}})
    elif change == "changed_model_call":
        save((tmp_path/RESEARCH_REFRESH).parent/("campaign/model/"+value["initial_call_id"]+".json"), {"status": "failed"})
    else:
        value.update(ok=False, status="incomplete")
    save(tmp_path/RESEARCH_REFRESH, value)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == RESEARCH_REFRESH)
    assert row["available"] is True and row["ok"] is False and row["final_qualified"] is False
    assert row["verified_json_subset"] == []


@pytest.mark.parametrize("relative", [ORIGINAL_EXECUTION, COMMON_FINAL_FREEZE, RESEARCH_REFRESH])
@pytest.mark.parametrize("malformed", [[1], 17, "scalar", None])
def test_malformed_control_root_cannot_crash_or_qualify_other_receipts(tmp_path, relative, malformed):
    save(tmp_path/relative, malformed)
    rows = control_receipts(tmp_path)
    row = next(r for r in rows if r["path"] == relative)
    assert row["ok"] is False and "invalide" in row["status"]
    assert len(rows) > 10 and row["authentication_error"]


@pytest.mark.parametrize("fixture,relative,field", [
    (original_execution_fixture, ORIGINAL_EXECUTION, "checks"),
    (original_execution_fixture, ORIGINAL_EXECUTION, "candidate"),
    (original_execution_fixture, ORIGINAL_EXECUTION, "usage"),
    (original_execution_fixture, ORIGINAL_EXECUTION, "upstream_source_checks"),
    (common_freeze_fixture, COMMON_FINAL_FREEZE, "checks"),
    (common_freeze_fixture, COMMON_FINAL_FREEZE, "methods"),
    (common_freeze_fixture, COMMON_FINAL_FREEZE, "naive"),
    (refresh_documentary_fixture, RESEARCH_REFRESH, "checks"),
    (refresh_documentary_fixture, RESEARCH_REFRESH, "proof_manifest"),
    (refresh_documentary_fixture, RESEARCH_REFRESH, "model_calls")])
@pytest.mark.parametrize("malformed", [[1], 19])
def test_nested_list_or_scalar_fields_fail_closed_without_global_report_loss(tmp_path, fixture, relative, field, malformed):
    value = fixture(tmp_path)
    value[field] = malformed
    save(tmp_path/relative, value)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == relative)
    assert row["ok"] is False and row.get("final_qualified") is not True


@pytest.mark.parametrize("malformed", [[1], 23])
def test_hash_matched_linked_manifest_must_still_be_expected_json_object(tmp_path, malformed):
    value = original_execution_fixture(tmp_path)
    value["original_manifest"] = linked_report_fixture(tmp_path, ORIGINAL_CAMPAIGN+"/generation/manifest.json", malformed)
    save(tmp_path/ORIGINAL_EXECUTION, value)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == ORIGINAL_EXECUTION)
    assert row["ok"] is False


@pytest.mark.parametrize("role", ["original_source", "original_result", "frozen_source", "generation_source", "worker_hardware", "proof_manifest"])
@pytest.mark.parametrize("forbidden_name", ["protected.sqlite3", "input.bin", "holdout-reveal.json"])
def test_new_control_links_are_role_whitelisted_before_any_protected_read(tmp_path, monkeypatch, role, forbidden_name):
    original = original_execution_fixture(tmp_path)
    frozen = common_freeze_fixture(tmp_path)
    refresh = refresh_documentary_fixture(tmp_path)
    protected = tmp_path/FINAL_DIRECTORY/"private"/forbidden_name
    protected.parent.mkdir(parents=True, exist_ok=True)
    protected.write_text("Invented forbidden fixture, not final study data")
    reference = {"path": str(protected.relative_to(tmp_path)), "sha256": "a"*64}
    relative = ORIGINAL_EXECUTION if role.startswith("original") else RESEARCH_REFRESH if role == "proof_manifest" else COMMON_FINAL_FREEZE
    if role == "original_source": original["candidate"]["source"] = reference
    elif role == "original_result": original["result"] = reference
    elif role == "frozen_source": frozen["methods"][0]["frozen_source"] = reference
    elif role == "generation_source": frozen["methods"][0]["generation_source"] = reference
    elif role == "worker_hardware": frozen["worker"]["hardware_receipt"] = reference
    else: refresh["proof_manifest"] = reference
    for path, value in ((ORIGINAL_EXECUTION, original), (COMMON_FINAL_FREEZE, frozen), (RESEARCH_REFRESH, refresh)):
        save(tmp_path/path, value)
    read_bytes, read_text = Path.read_bytes, Path.read_text
    def checked_bytes(path):
        assert path.resolve() != protected.resolve(), "forbidden bytes opened before authorization"
        return read_bytes(path)
    def checked_text(path, *args, **kwargs):
        assert path.resolve() != protected.resolve(), "forbidden JSON opened before authorization"
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_bytes", checked_bytes)
    monkeypatch.setattr(Path, "read_text", checked_text)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == relative)
    assert row["ok"] is False


def test_absent_or_incomplete_documentary_scope_never_claims_completed_real_sequence(tmp_path):
    rows = control_receipts(tmp_path)
    for row in rows:
        if row["path"] in (ORIGINAL_EXECUTION, COMMON_FINAL_FREEZE, RESEARCH_REFRESH):
            assert "prévu" in row["scope"] and "aucun" in row["scope"].lower()
    html = execution_evidence_panel(rows)
    assert "Le protocole prévoit" in html
    assert "Le reçu documentaire authentifie" not in html and "Ce contrôle utilise de vrais appels" not in html
    value = refresh_documentary_fixture(tmp_path)
    value.update(ok=False, status="incomplete")
    save(tmp_path/RESEARCH_REFRESH, value)
    rows = control_receipts(tmp_path)
    row = next(r for r in rows if r["path"] == RESEARCH_REFRESH)
    assert row["ok"] is False and "aucune séquence complète" in row["scope"]


@pytest.mark.parametrize("status", ["confirmed", "exact_artifact_identity"])
def test_direct_legacy_exact_artifacts_are_identity_without_changing_raw_intervals(status):
    pair = {"status": status, "ok": True, "artifact_hash": "a"*64, "incumbent_artifact_hash": "a"*64,
            "interval": {"gain": .87, "lower": .84, "upper": .90, "case_gains": {"case": {"gain": .85}}}}
    original = deepcopy(pair)
    views = direct_comparison_views({"group-fixture": {"method_comparisons": {"full_vs_greedy": pair}}})
    assert pair == original and views[0]["gain_basis"] == "exact_artifact_identity"
    assert views[0]["gain"] == 1 and views[0]["lower"] is views[0]["upper"] is None
    assert views[0]["source_interval_preserved"] == pair["interval"] and views[0]["case_gains"] == {}
    html = direct_comparison_table(views)
    assert "identité exacte des artefacts" in html and "sans intervalle : identité" in html
    assert "0.87" not in html and "0.84" not in html


@pytest.mark.parametrize("refusal", [
    {"status": "artifact_identity_context_unproven", "ok": False},
    {"status": "artifact_identity_context_unproven", "ok": True},
    {"status": "confirmed", "gain_basis": "unproven_artifact_context", "ok": True},
    {"status": "exact_artifact_identity", "ok": False},
    {"gain_basis": "exact_artifact_identity", "ok": False},
])
@pytest.mark.parametrize("noisy_interval", [False, True])
def test_explicit_unproven_context_cannot_be_normalized_to_an_identity_ratio(refusal, noisy_interval):
    pair = {"artifact_hash": "a"*64, "incumbent_artifact_hash": "a"*64,
            "paired_measurement_performed": False, "ranking_eligible": False, **refusal}
    if noisy_interval:
        pair["interval"] = {"gain": .87, "lower": .84, "upper": .90, "case_gains": {"case": {"gain": .85}}}
    original = deepcopy(pair)
    view = comparison_view(pair)
    assert view["gain_basis"] == "unproven_artifact_context"
    assert view["gain"] is view["lower"] is view["upper"] is None and view["case_gains"] == {}
    assert view["source_interval_preserved"] == pair.get("interval", {}) and pair == original
    assert view["source_status"] == pair.get("status") and view["source_ok"] == pair["ok"]
    views = direct_comparison_views({"group-fixture": {"method_comparisons": {"full_vs_greedy": pair}}})
    html = direct_comparison_table(views)
    assert "contexte d’exécution non prouvé" in html and "IC non établi" in html
    assert "inconnu" in html and "0.87" not in html and "sans intervalle : identité" not in html


@pytest.mark.parametrize("status", ["numerical_error", "infrastructure_failure", None])
def test_unmeasured_failure_is_unavailable_instead_of_a_paired_measurement(status):
    pair = {"status": status, "ok": False, "artifact_hash": "a"*64, "incumbent_artifact_hash": "b"*64}
    view = comparison_view(pair)
    assert view["gain_basis"] == "comparison_unavailable" and view["gain"] is None
    assert view["lower"] is view["upper"] is None
    views = direct_comparison_views({"group-fixture": {"method_comparisons": {"full_vs_greedy": pair}}})
    html = direct_comparison_table(views)
    assert "comparaison indisponible" in html and "IC non établi" in html and "mesure appariée" not in html


def test_legacy_valid_identity_stays_available_without_an_interval():
    view = comparison_view({"status": "confirmed", "ok": True,
                            "artifact_hash": "a"*64, "incumbent_artifact_hash": "a"*64})
    assert view["gain_basis"] == "exact_artifact_identity" and view["gain"] == 1


def test_case_table_exposes_regression_intervals_and_uses_independent_direct_pairs():
    summary = {"rows": [{"group_id": "convolution.1729", "mode": "true_upstream", "gain_basis": "paired_measurement",
        "case_gains": {"final-1": {"gain": .91, "lower": .85, "upper": .95},
                       "final-17": {"gain": .99, "lower": .96, "upper": 1.03}}}]}
    views = direct_comparison_views({"convolution.1729": {"method_comparisons": {"full_vs_original": {
        "artifact_hash": "a"*64, "incumbent_artifact_hash": "b"*64, "ok": True,
        "interval": {"gain": 1.11, "lower": 1.02, "upper": 1.20,
                     "case_gains": {"final-1": {"gain": 1.12, "lower": 1.03, "upper": 1.21}}}}}}})
    original = deepcopy((summary, views))
    html = case_comparison_table(summary, views)
    assert "0.91" in html and "[0.85 ; 0.95]" in html and "IC entièrement sous 1" in html
    assert "[0.96 ; 1.03]" in html and "IC recouvre 1" in html
    assert "full_vs_original" in html and "[1.03 ; 1.21]" in html and "IC entièrement au-dessus de 1" in html
    assert "sans correction pour comparaisons multiples" in html and "propres paires" in html
    assert (summary, views) == original


@pytest.mark.parametrize("basis", ["exact_baseline_artifact", "exact_artifact_identity", "unproven_artifact_context", "comparison_unavailable", None, "unknown_legacy_basis"])
def test_case_table_never_turns_identity_or_unproven_context_into_measured_cases(basis):
    cases = {"old-noisy-case": {"gain": .734567, "lower": .70, "upper": .79}}
    row = {"group_id": "group", "mode": "full", "gain_basis": basis,
           "case_gains": cases}
    original = deepcopy(row)
    html = case_comparison_table({"rows": [row]}, [])
    assert "old-noisy-case" not in html and "0.734567" not in html and row == original
    if basis in ("exact_baseline_artifact", "exact_artifact_identity"):
        assert "sans intervalle : identité" in html and "ensemble des cas" in html
    else:
        assert "IC non établi" in html and "inconnu" in html and "sans intervalle : identité" not in html


@pytest.mark.parametrize("interval", [None, {}, {"gain": 1, "lower": .9},
    {"gain": float("nan"), "lower": .9, "upper": 1.1},
    {"gain": 1, "lower": 1.2, "upper": 1.1}, {"gain": True, "lower": .9, "upper": 1.1}])
def test_case_table_keeps_incomplete_or_invalid_intervals_unknown(interval):
    html = case_comparison_table({"rows": [{"group_id": "<group>", "mode": "full", "gain_basis": "paired_measurement",
        "case_gains": {"<case>": interval}}]}, [])
    assert "IC non établi" in html and "intervalle incomplet ou incohérent" in html
    assert "&lt;case&gt;" in html and "&lt;group&gt;" in html and "<case>" not in html


def test_case_table_does_not_require_percentile_bootstrap_interval_to_contain_point_estimate():
    html = case_comparison_table({"rows": [{"group_id": "group", "mode": "full", "gain_basis": "paired_measurement",
        "case_gains": {"final-1": {"gain": 1.01, "lower": .90, "upper": .99}}}]}, [])
    assert "1.01" in html and "[0.9 ; 0.99]" in html and "IC entièrement sous 1" in html


def test_case_table_displays_partial_measurement_failure_and_artifact_qualification_separately():
    cases = {"final-1": {"gain": 1.12, "lower": 1.03, "upper": 1.21}}
    summary = {"rows": [{"group_id": "group", "mode": "full", "status": "final_measurement_failed",
        "final_qualified": False, "generation_method_status": "completed", "gain_basis": "paired_measurement", "case_gains": cases}]}
    views = [{"group_id": "group", "comparison": "full_vs_original", "source_status": "infrastructure_failure",
              "source_ok": False, "gain_basis": "paired_measurement", "case_gains": cases}]
    html = case_comparison_table(summary, views)
    assert "final_measurement_failed; artefact qualifié : non" in html
    assert "infrastructure_failure; indicateur ok du reçu : faux" in html
    assert html.count("[1.03 ; 1.21]") == 2 and "un intervalle ne qualifie ni l’artefact ni la méthode" in html


@pytest.mark.parametrize("first,second", [(None, None), ("", ""), ("source-only", "source-only"), ("a"*64, "b"*64)])
def test_identity_requires_equal_complete_artifact_hashes_not_missing_or_source_only(first, second):
    view = comparison_view({"artifact_hash": first, "incumbent_artifact_hash": second,
                            "source_hash": "a"*64, "interval": {"gain": .87, "lower": .84, "upper": .90}})
    assert view["gain_basis"] == "paired_measurement" and view["gain"] == .87


def identity_pilot_receipt():
    receipt = frozen_receipt()
    for role in receipt["candidates"]:
        receipt["candidates"][role]["artifact_hash"] = hashlib.sha256(role.encode()).hexdigest()
    receipt["candidates"]["native_codex"]["artifact_hash"] = receipt["candidates"]["agent"]["artifact_hash"]
    receipt["candidates"]["native_codex"]["source"] = receipt["candidates"]["agent"]["source"]
    receipt["candidates"]["native_codex"]["source_hash"] = receipt["candidates"]["agent"]["source_hash"]
    for name, (role, reference) in FROZEN_PAIRS.items():
        pair = receipt["paired_comparisons"][name]
        pair["artifact_hash"] = receipt["candidates"][role]["artifact_hash"]
        pair["incumbent_artifact_hash"] = receipt["candidates"][reference]["artifact_hash"]
    return receipt


def test_pilot_identity_direct_pair_has_no_legacy_interval_in_panel_or_plot(tmp_path):
    receipt = identity_pilot_receipt()
    save(tmp_path/"comparison.json", receipt)
    pilot = frozen_pilot_summary(tmp_path/"comparison.json")
    assert pilot["receipt_verified"]
    direct = pilot["comparisons"]["agent_vs_native_codex"]
    assert direct["gain_basis"] == "exact_artifact_identity" and direct["interval"]["lower"] is None
    assert direct["source_interval_preserved"]["gain"] == .742984
    html = frozen_pilot_panel(pilot, None)
    assert "ratio 1 sans intervalle statistique" in html and "0.742984" not in html
    assert plot_frozen_pilot(pilot, tmp_path).is_file()


@pytest.mark.parametrize("refusal", [
    {"status": "artifact_identity_context_unproven", "ok": False},
    {"status": "artifact_identity_context_unproven", "ok": True},
    {"status": "measured", "gain_basis": "unproven_artifact_context", "ok": True},
    {"status": "exact_artifact_identity", "ok": False},
])
@pytest.mark.parametrize("noisy_interval", [False, True])
def test_pilot_refuses_explicit_unproven_context_before_qualification_and_plot(tmp_path, refusal, noisy_interval):
    receipt = identity_pilot_receipt()
    pair = receipt["paired_comparisons"]["agent_vs_native_codex"]
    pair.update(refusal, paired_measurement_performed=False, ranking_eligible=False)
    if not noisy_interval:
        pair.pop("interval")
    path = tmp_path/"comparison.json"
    save(path, receipt)
    original = path.read_bytes()
    pilot = frozen_pilot_summary(path)
    assert not pilot["receipt_verified"] and any("Contexte d’exécution non prouvé" in error for error in pilot["validation_errors"])
    derived = pilot["comparisons"]["agent_vs_native_codex"]
    assert derived["gain_basis"] == "unproven_artifact_context"
    assert derived["interval"]["gain"] is derived["interval"]["lower"] is derived["interval"]["upper"] is None
    assert derived["source_interval_preserved"] == pair.get("interval", {})
    output = tmp_path/"not-created-plot"
    assert plot_frozen_pilot(pilot, output) is None and not output.exists()
    html = frozen_pilot_panel(pilot, None)
    assert "Comparaison figée non qualifiée" in html and "ratio 1 sans intervalle statistique" not in html
    assert "0.742984" not in html and path.read_bytes() == original


def resumed_documentary_fixture(root):
    """Synthetic archive continuation; no live model/GPU activity in tests."""
    old = refresh_documentary_fixture(root)
    old_root, new_root = (root/name for name in (str(Path(RESEARCH_REFRESH).parent), str(Path(RESEARCH_REFRESH_RESUMED).parent)))
    for index in (5, 6, 7):
        call_id = format(index, "032x")
        old["model_calls"].append({"call_id": call_id})
        linked_report_fixture(old_root, "campaign/model/"+call_id+".json", {"call_id": call_id,
            "status": "completed", "model": "gpt-6-luna", "reasoning_effort": "max"})
    old.update(ok=False, status="incomplete")
    old_proof = {"schema_version": 1, "hash_algorithm": "sha256", "files": {
        str(path.relative_to(old_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in old_root.rglob("*.json") if path.name not in ("proof-sha256.json", "research-refresh-acceptance.json")}}
    old["proof_manifest"] = linked_report_fixture(old_root, "proof-sha256.json", old_proof)
    save(root/RESEARCH_REFRESH, old)
    shutil.copytree(old_root, new_root)
    for source, target in (("research-refresh-acceptance.json", "previous-receipt.json"),
            ("proof-sha256.json", "previous-proof.json"), ("events.json", "previous-events.json"),
            ("campaign/trials/fixture-trial/dossier.json", "previous-dossier.json")):
        destination = new_root/"source-evidence"/target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_root/source, destination)
    new = deepcopy(old)
    new.update(run_id="fixture-resumed", kind="documentary_research_refresh_resumption_acceptance", ok=True,
        status="documentary_research_refresh_resumption_acceptance_passed", inherited_model_calls=old["model_calls"],
        model_calls=[], source_provenance={"HEAD": "f"*40}, new_usage_delta={"model_calls": 3, "model_tokens": 900})
    for index in (8, 9, 10):
        call_id = format(index, "032x")
        new["model_calls"].append({"call_id": call_id})
        linked_report_fixture(new_root, "campaign/model/"+call_id+".json", {"call_id": call_id,
            "status": "completed", "model": "gpt-6-luna", "reasoning_effort": "max"})
    new["final_call_id"] = new["model_calls"][0]["call_id"]
    new["renewed_research_id"] = "resumed-renewed"
    dossier = {"research_id": "resumed-renewed", "receipt_version": 2, "covered_techniques": ["tiling", "warp_shuffle"],
               "diagnostic_ids": ["diag-fixture"]}
    new["renewed_dossier_sha256"] = linked_report_fixture(new_root, "campaign/trials/fixture-trial/dossier.json", dossier)["sha256"]
    linked_report_fixture(new_root, "resume-state.json", {"kind": "derived_from_archived_public_prefix"})
    linked_report_fixture(new_root, "campaign/hardware-evidence-context.json", {"driver_api_raw": 13010, "driver_package": "590.48.01"})
    linked_report_fixture(new_root, "campaign/acceptance-controls-resume/invocations.json", [])
    linked_report_fixture(new_root, "run-intent.json", {"run_id": new["run_id"], "source_provenance": new["source_provenance"], "new_gpu_requests": 0})
    new["checks"].update({name: True for name in REFRESH_RESUMPTION_CHECKS})
    new["resumption"] = {"source_directory": str(Path(RESEARCH_REFRESH).parent),
        "source_receipt": {"path": RESEARCH_REFRESH, "path_base": "repository",
                           "sha256": hashlib.sha256((root/RESEARCH_REFRESH).read_bytes()).hexdigest()},
        "source_proof_manifest": {"path": str(Path(RESEARCH_REFRESH).parent/"proof-sha256.json"), "path_base": "repository",
                                 "sha256": hashlib.sha256((old_root/"proof-sha256.json").read_bytes()).hexdigest()},
        "inherited_completed_model_calls": 7, "no_initial_proposal_or_diagnostic_replay": True,
        "state_derivation": "Explicit synthetic fixture derivation, never an historical checkpoint"}
    new_proof = {"schema_version": 1, "hash_algorithm": "sha256", "files": {
        str(path.relative_to(new_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in new_root.rglob("*.json") if path.name not in ("proof-sha256.json", "research-refresh-acceptance.json")}}
    new["proof_manifest"] = linked_report_fixture(new_root, "proof-sha256.json", new_proof)
    save(root/RESEARCH_REFRESH_RESUMED, new)
    return old, new


def test_resumed_documentary_authority_preserves_incomplete_original_and_separates_new_calls(tmp_path):
    old, new = resumed_documentary_fixture(tmp_path)
    original_bytes = (tmp_path/RESEARCH_REFRESH).read_bytes()
    rows = control_receipts(tmp_path)
    previous = next(r for r in rows if r["path"] == RESEARCH_REFRESH)
    resumed = next(r for r in rows if r["path"] == RESEARCH_REFRESH_RESUMED)
    assert previous["ok"] is False and resumed["ok"] is True
    assert all(resumed["authentication_checks"].values())
    assert resumed["inherited_model_call_count"] == 7 and resumed["observed_new_model_call_count"] == 3
    assert resumed["declared_new_ledger_usage_delta"] == new["new_usage_delta"]
    assert "non réconcilié" in resumed["new_usage_delta_scope"]
    assert resumed["new_gpu_execution"] is False and resumed["database_or_final_inputs_read"] is False
    assert (tmp_path/RESEARCH_REFRESH).read_bytes() == original_bytes
    html = execution_evidence_panel(rows)
    assert "Autorité séparée de reprise AC-02" in html and "7 anciens appels hérités" in html
    assert "3 nouveaux appels observés" in html and "Le reçu initial incomplet garde son statut" in html


def test_missing_resumption_is_an_independent_unknown_result(tmp_path):
    old = refresh_documentary_fixture(tmp_path)
    old.update(ok=False, status="incomplete")
    save(tmp_path/RESEARCH_REFRESH, old)
    rows = control_receipts(tmp_path)
    resumed = next(r for r in rows if r["path"] == RESEARCH_REFRESH_RESUMED)
    assert resumed["ok"] is None and resumed["available"] is False
    assert "en attente" in resumed["status"] and "aucune séquence complète" in resumed["scope"]
    assert "ne permettent pas de conclure" in execution_evidence_panel(rows)


@pytest.mark.parametrize("change", ["missing_check", "overlapping_calls", "altered_inherited_calls", "wrong_old_pin", "current_hardware", "incomplete"])
def test_resume_cannot_erase_prior_failure_or_claim_uncertain_repeated_calls(tmp_path, change):
    old, new = resumed_documentary_fixture(tmp_path)
    if change == "missing_check": new["checks"].pop("resume_no_old_model_call_repeated")
    elif change == "overlapping_calls": new["model_calls"][0]["call_id"] = new["inherited_model_calls"][0]["call_id"]
    elif change == "altered_inherited_calls": new["inherited_model_calls"][0]["usage"] = {"model_tokens": 0}
    elif change == "wrong_old_pin": new["resumption"]["source_receipt"]["sha256"] = "a"*64
    elif change == "current_hardware": new["hardware_currently_observed"] = True
    else: new.update(ok=False, status="incomplete")
    save(tmp_path/RESEARCH_REFRESH_RESUMED, new)
    rows = control_receipts(tmp_path)
    assert next(r for r in rows if r["path"] == RESEARCH_REFRESH)["ok"] is False
    resumed = next(r for r in rows if r["path"] == RESEARCH_REFRESH_RESUMED)
    assert resumed["ok"] is False and resumed["observed_new_model_call_count"] is None


def _documentary_proof(root, receipt):
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in root.rglob("*.json") if path.name not in ("proof-sha256.json", "research-refresh-acceptance.json")}
    receipt["proof_manifest"] = linked_report_fixture(root, "proof-sha256.json",
        {"schema_version":1,"hash_algorithm":"sha256","files":files})
    save(root/"research-refresh-acceptance.json",receipt)


def twice_resumed_documentary_fixture(root):
    """Explicit synthetic 7/16 → 9/19 refusal → successful new control."""
    anchor, previous = resumed_documentary_fixture(root)
    a, p, n = (root/Path(name).parent for name in (RESEARCH_REFRESH,RESEARCH_REFRESH_RESUMED,RESEARCH_REFRESH_RESUMED_2))
    def call(index, purpose, directory):
        row = {"call_id":format(index,"032x"),"usage":{"input_tokens":index*10,"output_tokens":1},"elapsed_seconds":float(index)}
        payload = {"fixture":index}
        linked_report_fixture(directory,"campaign/model/"+row["call_id"]+".json",dict(row,purpose=purpose,
            status="completed",returncode=0,model="gpt-6-luna",reasoning_effort="max",payload=payload))
        return row, {"call_id":row["call_id"],"purpose":purpose,"stage":"research_service","kernel_proposal":False,
            "status":"completed","returned_payload_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
    anchor["model_calls"] = [call(i,purpose,a)[0] for i,purpose in enumerate(
        ("research","proposal","proposal","proposal","proposal","research","proposal"),1)]
    kinds = ["campaign_initialized","checkpoint","selection","selection","model","model","research","model",
        "generation","model","generation","model","diagnostic","model","model","research_blocked"]
    model_index, events = 0, []
    for index,kind in enumerate(kinds,1):
        payload = {}
        if kind == "model": payload = deepcopy(anchor["model_calls"][model_index]); model_index += 1
        if kind == "diagnostic": payload = {"diagnostic_id":"diag-fixture"}
        if kind == "research_blocked": payload = {"error":"research_blocked: fixture initial guard refusal"}
        events.append({"event_id":index,"kind":kind,"payload":payload})
    save(a/"events.json",events)
    initial = json.loads((a/"campaign/acceptance-evidence/initial-dossier.json").read_text())
    save(a/"campaign/trials/fixture-trial/dossier.json",initial)
    refusal_path=a/"campaign/acceptance-evidence/blocked-refusal.json"
    refusal=json.loads(refusal_path.read_text());refusal["call_id"]=anchor["model_calls"][4]["call_id"];save(refusal_path,refusal)
    anchor.update(checks={},error="Actual targeted research blocked: "+events[-1]["payload"]["error"],
        ledger_usage={"spent":{"model_calls":7,"model_tokens":sum(c["usage"]["input_tokens"]+1 for c in anchor["model_calls"])}})
    _documentary_proof(a,anchor)
    shutil.rmtree(p/"campaign");shutil.copytree(a/"campaign",p/"campaign")
    pairs = [call(8,"research",p),call(9,"proposal",p)]
    calls,invocations = [pair[0] for pair in pairs],[pair[1] for pair in pairs]
    save(p/"campaign/acceptance-controls-resume/invocations.json",invocations)
    save(p/"campaign/hardware-evidence-context.json",{"historical":True})
    previous["resumption"].update(evidence_repository=str(root.resolve()),
        source_receipt={"path":RESEARCH_REFRESH,"path_base":"repository","sha256":hashlib.sha256((a/"research-refresh-acceptance.json").read_bytes()).hexdigest()},
        source_proof_manifest={"path":str(Path(RESEARCH_REFRESH).parent/"proof-sha256.json"),"path_base":"repository",
            "sha256":hashlib.sha256((a/"proof-sha256.json").read_bytes()).hexdigest()})
    error="research_blocked: fixture resumed applicability refusal"
    suffix=[{"event_id":17+i,"kind":"model","payload":dict(row,purpose=inv["purpose"])} for i,(row,inv) in enumerate(pairs)]
    suffix.append({"event_id":19,"kind":"research_blocked","payload":{"error":error}})
    save(p/"events.json",events+suffix)
    research_id="a"*32
    save(p/("campaign/research/"+research_id+".json"),{"research_id":research_id,"receipt_version":2,
        "call_ids":[c["call_id"] for c in calls],"diagnostic_ids":["diag-fixture"],"missing_evidence":["fixture missing evidence"],
        "applicability_checks":[{"satisfied":False}]})
    save(p/("campaign/research/"+research_id+"-blocked.json"),{"research_id":research_id,
        "retrieval_status":"research_blocked","error":"fixture resumed applicability refusal"})
    tokens=sum(c["usage"]["input_tokens"]+c["usage"]["output_tokens"] for c in calls)
    previous.update(ok=False,status="incomplete",checks={},inherited_model_calls=deepcopy(anchor["model_calls"]),model_calls=calls,
        error="Actual resumed targeted research blocked: "+error,resume_source_unchanged=True,
        ledger_usage={"spent":{"model_calls":9,"model_tokens":anchor["ledger_usage"]["spent"]["model_tokens"]+tokens}},
        new_usage_delta={"model_calls":2,"model_tokens":tokens})
    save(p/"run-intent.json",{"run_id":previous["run_id"],"source_provenance":previous["source_provenance"],
        "new_gpu_requests":0,"resumption":previous["resumption"]})
    for source,target in (("research-refresh-acceptance.json","previous-receipt.json"),("proof-sha256.json","previous-proof.json"),
        ("events.json","previous-events.json"),("campaign/acceptance-evidence/initial-dossier.json","previous-dossier.json")):
        shutil.copy2(a/source,p/"source-evidence"/target)
    _documentary_proof(p,previous)
    shutil.copytree(p,n)
    new=deepcopy(previous)
    new.update(run_id="fixture-resumed-2",ok=True,status="documentary_research_refresh_resumption_acceptance_passed",
        checks={name:True for name in REFRESH_CHECKS+REFRESH_RESUMPTION_CHECKS},
        inherited_model_calls=anchor["model_calls"]+calls,model_calls=[call(i,purpose,n)[0] for i,purpose in
            ((10,"research"),(11,"proposal"),(12,"proposal"))],source_provenance={"HEAD":"d"*40},
        initial_call_id=anchor["model_calls"][2]["call_id"],compatible_call_id=anchor["model_calls"][3]["call_id"],
        blocked_call_id=anchor["model_calls"][4]["call_id"],final_call_id=format(12,"032x"),
        initial_research_id="initial",renewed_research_id="renewed-2",diagnostic_id="diag-fixture",uncovered_technique="warp_shuffle")
    new["resumption"].update(source_directory=str(Path(RESEARCH_REFRESH_RESUMED).parent),
        source_receipt={"path":RESEARCH_REFRESH_RESUMED,"path_base":"repository","sha256":hashlib.sha256((p/"research-refresh-acceptance.json").read_bytes()).hexdigest()},
        source_proof_manifest={"path":str(Path(RESEARCH_REFRESH_RESUMED).parent/"proof-sha256.json"),"path_base":"repository",
            "sha256":hashlib.sha256((p/"proof-sha256.json").read_bytes()).hexdigest()},
        inherited_completed_model_calls=9,controls_directory="acceptance-controls-resume-2",
        research_round_derivation={"initial_anchor_round":2,"completed_failed_round":3,"next_fresh_round":4})
    new["initial_dossier_sha256"]=hashlib.sha256((n/"campaign/acceptance-evidence/initial-dossier.json").read_bytes()).hexdigest()
    new["renewed_dossier_sha256"]=linked_report_fixture(n,"campaign/trials/fixture-trial/dossier.json",
        {"research_id":"renewed-2","receipt_version":2,"covered_techniques":["tiling","warp_shuffle"],"diagnostic_ids":["diag-fixture"]})["sha256"]
    new["runtime_evidence_context"]=linked_report_fixture(n,"campaign/runtime-evidence-context.json",{"historical_baseline":True})
    save(n/"campaign/acceptance-controls-resume-2/invocations.json",[])
    save(n/"run-intent.json",{"run_id":new["run_id"],"source_provenance":new["source_provenance"],"new_gpu_requests":0,"resumption":new["resumption"]})
    for source,target in (("research-refresh-acceptance.json","previous-receipt.json"),("proof-sha256.json","previous-proof.json"),
        ("events.json","previous-events.json"),("campaign/acceptance-evidence/initial-dossier.json","previous-dossier.json")):
        shutil.copy2(p/source,n/"source-evidence"/target)
    _documentary_proof(n,new)
    return anchor,previous,new


def _repin_second_resume(root, previous, new):
    p,n=(root/Path(name).parent for name in (RESEARCH_REFRESH_RESUMED,RESEARCH_REFRESH_RESUMED_2))
    _documentary_proof(p,previous)
    new["resumption"]["source_receipt"]["sha256"]=hashlib.sha256((p/"research-refresh-acceptance.json").read_bytes()).hexdigest()
    new["resumption"]["source_proof_manifest"]["sha256"]=hashlib.sha256((p/"proof-sha256.json").read_bytes()).hexdigest()
    for source,target in (("research-refresh-acceptance.json","previous-receipt.json"),("proof-sha256.json","previous-proof.json")):
        shutil.copy2(p/source,n/"source-evidence"/target)
    save(n/"run-intent.json",{"run_id":new["run_id"],"source_provenance":new["source_provenance"],"new_gpu_requests":0,"resumption":new["resumption"]})
    _documentary_proof(n,new)


def third_resumed_documentary_fixture(root):
    """Invented public11-call accepted alternative, then fresh exact coverage."""
    anchor,previous,accepted_authority=twice_resumed_documentary_fixture(root)
    old=root/Path(RESEARCH_REFRESH_RESUMED_2).parent
    target=root/Path(RESEARCH_REFRESH_RESUMED_3).parent
    accepted_authority["model_calls"]=accepted_authority["model_calls"][:2]
    (old/"campaign/model"/(format(12,"032x")+".json")).unlink()
    invocations=[]
    for call,purpose in zip(accepted_authority["model_calls"],("research","proposal")):
        trace=json.loads((old/"campaign/model"/(call["call_id"]+".json")).read_text())
        invocations.append(dict(call_id=call["call_id"],purpose=purpose,stage="research_service",
            kernel_proposal=False,status="completed",returned_payload_sha256=hashlib.sha256(
                json.dumps(trace["payload"],sort_keys=True).encode()).hexdigest()))
    save(old/"campaign/acceptance-controls-resume-2/invocations.json",invocations)
    research_id="b"*32
    accepted={"research_id":research_id,"receipt_version":2,"retrieval_status":"retrieved",
        "covered_techniques":["supported alternative"],"diagnostic_ids":["diag-fixture"],"missing_evidence":[],
        "applicability_checks":[{"satisfied":True}],"call_ids":[c["call_id"] for c in accepted_authority["model_calls"]],
        "reference_kernels":["fixture-code"],"launch_requirements":["future oracle"],"dependencies":[],
        "incompatible_conditions":["requested transformation not established"],"pmpp_topics":["tiling"],
        "hypotheses":[{"documented_fact":"explicit CPU fixture","observed_code_property":"alternative",
            "expected_effect_to_measure":"future timing","source_ids":["fixture-code"]}],
        "expected_effects":["unmeasured"],"validation_metrics":["independent oracle"],
        "negative_search":["requested technique unsupported in fixture"],"original_hypothesis":"alternative",
        "usage":{"input_tokens":sum(c["usage"]["input_tokens"] for c in accepted_authority["model_calls"]),
            "output_tokens":sum(c["usage"]["output_tokens"] for c in accepted_authority["model_calls"]),"cost_usd":None}}
    fields=("covered_techniques","reference_kernels","launch_requirements","dependencies","applicability_checks",
        "incompatible_conditions","missing_evidence","pmpp_topics","hypotheses","expected_effects",
        "validation_metrics","negative_search","original_hypothesis")
    synthesis_path=old/"campaign/model"/(accepted_authority["model_calls"][1]["call_id"]+".json")
    trace=json.loads(synthesis_path.read_text());trace["payload"]={key:accepted[key] for key in fields};save(synthesis_path,trace)
    invocations[1]["returned_payload_sha256"]=hashlib.sha256(json.dumps(trace["payload"],sort_keys=True).encode()).hexdigest()
    save(old/"campaign/acceptance-controls-resume-2/invocations.json",invocations)
    dossier_name="campaign/trials/fixture-trial/dossier.json"
    save(old/dossier_name,accepted);save(old/("campaign/research/"+research_id+".json"),accepted)
    events=json.loads((old/"events.json").read_text())
    events += [dict(event_id=index,kind="model",payload=dict(call,purpose=invocation["purpose"]))
        for index,call,invocation in zip((20,21),accepted_authority["model_calls"],invocations)]
    events.append({"event_id":22,"kind":"research","payload":{"research_id":research_id,
        "diagnostic_id":"diag-fixture","targeted_refresh":True,"trial_id":"fixture-trial"}})
    save(old/"events.json",events)
    state={"repair":2,"research_round":3,"research_ready":False,"needs_research":True,
        "research_refresh_required":True,"error":"prior blocked research","trial_id":"fixture-trial"}
    save(old/"resume-state.json",{"state":state})
    tokens=sum(c["usage"]["input_tokens"]+c["usage"]["output_tokens"] for c in accepted_authority["model_calls"])
    prior_tokens=previous["ledger_usage"]["spent"]["model_tokens"]
    accepted_authority.update(ok=False,status="incomplete",checks={},
        error="Actual resumed dossier does not cover the exact blocked technique/diagnostic",
        ledger_usage={"spent":{"model_calls":11,"model_tokens":prior_tokens+tokens}},
        new_usage_delta={"model_calls":2,"model_tokens":tokens})
    for key in ("initial_call_id","compatible_call_id","blocked_call_id","final_call_id","initial_research_id",
                "renewed_research_id","diagnostic_id","uncovered_technique"):
        accepted_authority.pop(key,None)
    _documentary_proof(old,accepted_authority)
    shutil.copytree(old,target)
    new=deepcopy(accepted_authority)
    new.update(run_id="fixture-third-resume",ok=True,status="documentary_research_refresh_resumption_acceptance_passed",
        checks={name:True for name in REFRESH_CHECKS+REFRESH_RESUMPTION_CHECKS},
        inherited_model_calls=accepted_authority["inherited_model_calls"]+accepted_authority["model_calls"],model_calls=[],
        source_provenance={"HEAD":"e"*40},initial_research_id="initial",renewed_research_id="c"*32,
        diagnostic_id="diag-fixture",uncovered_technique="warp_shuffle",
        initial_call_id=anchor["model_calls"][2]["call_id"],compatible_call_id=anchor["model_calls"][3]["call_id"],
        blocked_call_id=anchor["model_calls"][4]["call_id"],final_call_id=format(14,"032x"))
    for index in (12,13,14):
        call={"call_id":format(index,"032x"),"usage":{"input_tokens":100,"output_tokens":1}}
        new["model_calls"].append(call)
        linked_report_fixture(target,"campaign/model/"+call["call_id"]+".json",dict(call,status="completed",model="gpt-6-luna",reasoning_effort="max"))
    post=dict(state,research_round=4,research_ready=True,needs_research=False,research_refresh_required=False,error=None)
    coverage={"kind":"explicit_complementary_coverage_refresh","accepted_research_id":research_id,
        "accepted_dossier":{"path":dossier_name,"sha256":hashlib.sha256((old/dossier_name).read_bytes()).hexdigest()},
        "missing_techniques":["warp_shuffle"],"diagnostic_id":"diag-fixture",
        "source_helper_error":accepted_authority["error"],"prior_research_succeeded":True,
        "semantic_equivalence_is_not_exact_coverage":True}
    new["resumption"].update(source_directory=str(Path(RESEARCH_REFRESH_RESUMED_2).parent),
        source_receipt={"path":RESEARCH_REFRESH_RESUMED_2,"sha256":hashlib.sha256((old/"research-refresh-acceptance.json").read_bytes()).hexdigest(),"path_base":"repository"},
        source_proof_manifest={"path":str(Path(RESEARCH_REFRESH_RESUMED_2).parent/"proof-sha256.json"),
            "sha256":hashlib.sha256((old/"proof-sha256.json").read_bytes()).hexdigest(),"path_base":"repository"},
        inherited_completed_model_calls=11,phase="complementary_coverage_refresh",coverage_refresh=coverage,
        controls_directory="acceptance-controls-resume-3",research_round_derivation={"initial_anchor_round":2,
            "completed_failed_round":3,"completed_accepted_round":4,"next_fresh_round":5})
    save(target/"run-intent.json",{"run_id":new["run_id"],"source_provenance":new["source_provenance"],
        "new_gpu_requests":0,"resumption":new["resumption"]})
    save(target/"resume-state.json",{"phase":"accepted_research_coverage_gap_reconstructed","state":post,
        "source_events_sha256":hashlib.sha256((old/"events.json").read_bytes()).hexdigest()})
    request=dict(coverage,phase="complementary_coverage_refresh",source_receipt=new["resumption"]["source_receipt"],
        post_research_state=post,requested_state=dict(post,needs_research=True,research_refresh_required=True),next_fresh_round=5)
    new["coverage_refresh_request"]=linked_report_fixture(target,"coverage-refresh-request.json",request)
    for name,source in (("previous-receipt.json","research-refresh-acceptance.json"),("previous-proof.json","proof-sha256.json"),
        ("previous-events.json","events.json"),("previous-dossier.json","campaign/acceptance-evidence/initial-dossier.json"),
        ("previous-accepted-dossier.json",dossier_name)):
        shutil.copy2(old/source,target/"source-evidence"/name)
    new["initial_dossier_sha256"]=hashlib.sha256((target/"campaign/acceptance-evidence/initial-dossier.json").read_bytes()).hexdigest()
    new["renewed_dossier_sha256"]=linked_report_fixture(target,dossier_name,{"research_id":"c"*32,
        "receipt_version":2,"covered_techniques":["warp_shuffle"],"diagnostic_ids":["diag-fixture"]})["sha256"]
    save(target/"campaign/acceptance-controls-resume-3/invocations.json",[])
    _documentary_proof(target,new)
    return accepted_authority,new


def test_third_resume_accepted_research_lineage_keeps_all_prior_incomplete_authorities_separate(tmp_path):
    old,new=third_resumed_documentary_fixture(tmp_path)
    rows=control_receipts(tmp_path);row=next(r for r in rows if r["path"]==RESEARCH_REFRESH_RESUMED_3)
    assert row["ok"] is True,{k:v for k,v in row.get("authentication_checks",{}).items() if not v}
    assert row["inherited_model_call_count"]==11 and row["observed_new_model_call_count"]==3
    assert all(next(r for r in rows if r["path"]==name)["ok"] is False for name in (
        RESEARCH_REFRESH,RESEARCH_REFRESH_RESUMED,RESEARCH_REFRESH_RESUMED_2))
    assert json.loads((tmp_path/RESEARCH_REFRESH_RESUMED_2).read_text())["ok"] is False
    assert "recherche acceptée au round 4 conservée" in execution_evidence_panel(rows)


@pytest.mark.parametrize("fault",["accepted_relabel","accepted_event","old_call_usage","source_pin","round5",
    "force_requested_state","accepted_copy","ledger","wrong_parent_count","cyclic_source","synthesis_mismatch"])
def test_third_resume_rejects_tampered_accepted_lineage_or_explicit_refresh(tmp_path,fault):
    old,new=third_resumed_documentary_fixture(tmp_path)
    target=tmp_path/Path(RESEARCH_REFRESH_RESUMED_3).parent
    if fault=="round5":new["resumption"]["research_round_derivation"]["next_fresh_round"]=4
    elif fault=="source_pin":new["resumption"]["source_receipt"]["sha256"]="0"*64
    elif fault=="wrong_parent_count":new["resumption"]["inherited_completed_model_calls"]=9
    elif fault=="old_call_usage":new["inherited_model_calls"][-1]["usage"]["input_tokens"]+=1
    elif fault=="force_requested_state":
        path=target/"coverage-refresh-request.json";request=json.loads(path.read_text());request["requested_state"]["research_round"]=5
        new["coverage_refresh_request"]=linked_report_fixture(target,"coverage-refresh-request.json",request)
    elif fault=="accepted_copy":save(target/"source-evidence/previous-accepted-dossier.json",{})
    elif fault=="ledger":old["new_usage_delta"]["model_calls"]=3
    elif fault=="cyclic_source":
        old["resumption"]["inherited_completed_model_calls"]=11
        old["resumption"]["source_receipt"]["path"]=RESEARCH_REFRESH_RESUMED_2
    elif fault=="synthesis_mismatch":
        original=tmp_path/Path(RESEARCH_REFRESH_RESUMED_2).parent
        path=original/"campaign/trials/fixture-trial/dossier.json";dossier=json.loads(path.read_text())
        dossier["covered_techniques"].append("altered alternative");save(path,dossier)
        save(original/("campaign/research/"+dossier["research_id"]+".json"),dossier)
        shutil.copy2(path,target/"source-evidence/previous-accepted-dossier.json")
        new["resumption"]["coverage_refresh"]["accepted_dossier"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        request=json.loads((target/"coverage-refresh-request.json").read_text())
        request["accepted_dossier"]=new["resumption"]["coverage_refresh"]["accepted_dossier"]
        new["coverage_refresh_request"]=linked_report_fixture(target,"coverage-refresh-request.json",request)
    else:
        original=tmp_path/Path(RESEARCH_REFRESH_RESUMED_2).parent
        if fault=="accepted_relabel":
            path=original/"campaign/trials/fixture-trial/dossier.json";dossier=json.loads(path.read_text())
            dossier["covered_techniques"].append("warp_shuffle");save(path,dossier)
        else:
            path=original/"events.json";events=json.loads(path.read_text());events[-1]["kind"]="research_blocked";save(path,events)
    # Re-pin fixture authority metadata so shape/phase checks, not a stale
    # outer receipt alone, must reject these invented negative cases.
    original=tmp_path/Path(RESEARCH_REFRESH_RESUMED_2).parent
    _documentary_proof(original,old)
    new["resumption"]["source_proof_manifest"]["sha256"]=hashlib.sha256((original/"proof-sha256.json").read_bytes()).hexdigest()
    if fault!="source_pin":new["resumption"]["source_receipt"]["sha256"]=hashlib.sha256((original/"research-refresh-acceptance.json").read_bytes()).hexdigest()
    for name in ("research-refresh-acceptance.json","proof-sha256.json"):
        shutil.copy2(original/name,target/"source-evidence"/({"research-refresh-acceptance.json":"previous-receipt.json","proof-sha256.json":"previous-proof.json"}[name]))
    save(target/"run-intent.json",{"run_id":new["run_id"],"source_provenance":new["source_provenance"],
        "new_gpu_requests":0,"resumption":new["resumption"]})
    _documentary_proof(target,new)
    row=next(r for r in control_receipts(tmp_path) if r["path"]==RESEARCH_REFRESH_RESUMED_3)
    assert row["ok"] is False
    if fault=="synthesis_mismatch":assert row["authentication_checks"]["accepted_dossier_matches_exact_synthesis"] is False


def test_third_resume_absent_is_unknown_and_does_not_complete_old_accepted_research(tmp_path):
    third_resumed_documentary_fixture(tmp_path)
    shutil.rmtree(tmp_path/Path(RESEARCH_REFRESH_RESUMED_3).parent)
    rows=control_receipts(tmp_path)
    assert next(r for r in rows if r["path"]==RESEARCH_REFRESH_RESUMED_3)["ok"] is None
    assert next(r for r in rows if r["path"]==RESEARCH_REFRESH_RESUMED_2)["ok"] is False
    assert "Aucun succès de ce complément n’est acquis" in execution_evidence_panel(rows)


def test_standalone_final_control_absent_or_plan_cannot_prove_execution(tmp_path):
    row=next(r for r in control_receipts(tmp_path) if r["path"]==STANDALONE_FINAL_SUITE)
    assert row["ok"] is None and row["new_gpu_execution"] is False
    save(tmp_path/STANDALONE_FINAL_SUITE,{"schema_version":1,"kind":"standalone_distinct_final_suite_acceptance",
        "submitted":False,"model_calls":0,"model_tokens":0})
    row=next(r for r in control_receipts(tmp_path) if r["path"]==STANDALONE_FINAL_SUITE)
    assert row["ok"] is False and row["new_gpu_execution"] is False


@pytest.mark.parametrize("fault",[None,"missing_check","failed_terminal","different_job","model_call","sources_changed"])
def test_standalone_final_control_requires_result_checks_owned_terminal_and_sources(tmp_path,fault):
    value={"schema_version":1,"kind":"standalone_distinct_final_suite_acceptance","ok":True,
        "status":"final_suite_acceptance_passed","job_id":"123","model_calls":0,"model_tokens":0,
        "sources_unchanged":True,"source_provenance":{"HEAD":"a"*40,"sources":{"fixture.py":"b"*64}},
        "worker_terminal":{"job_id":"123","terminal":True,"state":"COMPLETED","exit_code":"0:0"},
        "result":{"ok":True,"checks":{name:True for name in FINAL_SUITE_CHECKS}}}
    if fault=="missing_check":value["result"]["checks"].pop("final_input_buffers_used")
    elif fault=="failed_terminal":value["worker_terminal"]["state"]="FAILED"
    elif fault=="different_job":value["worker_terminal"]["job_id"]="456"
    elif fault=="model_call":value["model_calls"]=1
    elif fault=="sources_changed":value["sources_unchanged"]=False
    save(tmp_path/STANDALONE_FINAL_SUITE,value)
    row=next(r for r in control_receipts(tmp_path) if r["path"]==STANDALONE_FINAL_SUITE)
    assert row["ok"] is (fault is None) and row["new_gpu_execution"] is (fault is None)
    assert row["final_qualified"] is False


def test_second_documentary_resumption_authenticates_exact_seven_to_nine_refusal_chain(tmp_path):
    _,_,new=twice_resumed_documentary_fixture(tmp_path)
    rows=control_receipts(tmp_path)
    assert all(next(row for row in rows if row["path"]==name)["ok"] is False for name in (RESEARCH_REFRESH,RESEARCH_REFRESH_RESUMED))
    result=next(row for row in rows if row["path"]==RESEARCH_REFRESH_RESUMED_2)
    assert result["ok"] is True, result["authentication_checks"]
    assert all(result["authentication_checks"].values())
    assert result["inherited_model_call_count"]==9 and result["observed_new_model_call_count"]==3
    assert not result["new_gpu_execution"] and not result["database_or_final_inputs_read"]
    assert "Chaîne authentifiée 7 → 9" in execution_evidence_panel(rows)


@pytest.mark.parametrize("fault",["missing_anchor_pin","unsupported_count","wrong_round","source_was_successful",
    "altered_event_prefix","wrong_refusal_suffix","missing_failed_dossier","dossier_not_failed","missing_ancestor_copy","ledger_delta"])
def test_second_resumption_rejects_hash_resealed_but_unsupported_ancestry(tmp_path,fault):
    _,previous,new=twice_resumed_documentary_fixture(tmp_path)
    p,n=(tmp_path/Path(name).parent for name in (RESEARCH_REFRESH_RESUMED,RESEARCH_REFRESH_RESUMED_2))
    if fault=="missing_anchor_pin":previous["resumption"]["source_receipt"].pop("sha256")
    elif fault=="unsupported_count":new["resumption"]["inherited_completed_model_calls"]=10
    elif fault=="wrong_round":new["resumption"]["research_round_derivation"]["next_fresh_round"]=5
    elif fault=="source_was_successful":previous["ok"]=True
    elif fault in ("altered_event_prefix","wrong_refusal_suffix"):
        events=json.loads((p/"events.json").read_text());events[0 if fault=="altered_event_prefix" else -1]["kind"]="generation";save(p/"events.json",events)
    elif fault=="missing_failed_dossier":(p/("campaign/research/"+"a"*32+"-blocked.json")).unlink()
    elif fault=="dossier_not_failed":
        file=p/("campaign/research/"+"a"*32+".json");dossier=json.loads(file.read_text());dossier.update(missing_evidence=[],applicability_checks=[{"satisfied":True}]);save(file,dossier)
    elif fault=="missing_ancestor_copy":(p/"source-evidence/previous-receipt.json").unlink()
    else:previous["new_usage_delta"]["model_tokens"]+=1
    _repin_second_resume(tmp_path,previous,new)
    result=next(row for row in control_receipts(tmp_path) if row["path"]==RESEARCH_REFRESH_RESUMED_2)
    assert result["ok"] is False and result["observed_new_model_call_count"] is None


def test_absent_second_resumption_remains_separate_unknown_authority(tmp_path):
    resumed_documentary_fixture(tmp_path)
    result=next(row for row in control_receipts(tmp_path) if row["path"]==RESEARCH_REFRESH_RESUMED_2)
    assert result["ok"] is None and not result["available"] and "en attente" in result["status"]


def test_second_resume_authenticates_source_trial_bytes_not_only_manifest_claim(tmp_path):
    twice_resumed_documentary_fixture(tmp_path)
    source=tmp_path/Path(RESEARCH_REFRESH_RESUMED).parent/"campaign/trials/fixture-trial/dossier.json"
    save(source,{"research_id":"unrecorded-admission"})
    result=next(row for row in control_receipts(tmp_path) if row["path"]==RESEARCH_REFRESH_RESUMED_2)
    assert result["ok"] is False
    assert result["authentication_checks"]["source_nine_ancestor_copies_and_trace_hashes"] is False


def test_campaign_ratio_table_identifies_baseline_alias_without_measured_interval():
    summary = method_outcome_summary()
    row = summary["rows"][0]
    row.update(gain_basis="exact_baseline_artifact", gain=1, lower=.9, upper=1.1)
    html = study_campaign_table(summary)
    assert "Origine du ratio" in html and "identité exacte avec le naïf" in html
    assert "identité exacte, sans IC" in html and "[0.9 ; 1.1]" not in html
    assert "Ratios disponibles (mesures/identités)" in study_aggregate_table(summary)
    assert "Ratio géométrique (mesures/identités)" in study_aggregate_table(summary)


@pytest.mark.parametrize("directory_link", [False, True])
def test_authorized_role_cannot_follow_symlink_to_private_in_repo_payload(tmp_path, monkeypatch, directory_link):
    value = original_execution_fixture(tmp_path)
    expected = tmp_path/ORIGINAL_CAMPAIGN/"generation/frozen-original.cu"
    protected = tmp_path/"private/input.bin"
    protected.parent.mkdir()
    protected.write_text("Invented forbidden fixture")
    expected.unlink()
    if directory_link:
        original_generation = expected.parent
        shutil.rmtree(original_generation)
        original_generation.symlink_to(protected.parent, target_is_directory=True)
        protected.rename(protected.parent/expected.name)
        protected = protected.parent/expected.name
    else:
        expected.symlink_to(protected)
    read = Path.read_bytes
    def checked(path):
        assert path.resolve() != protected.resolve(), "private payload opened through an authorized-looking role"
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", checked)
    row = next(r for r in control_receipts(tmp_path) if r["path"] == ORIGINAL_EXECUTION)
    assert row["ok"] is False


def metric_summary(mode="full", metrics=None, **fields):
    return {"rows": [{"run_id": "fixture.<campaign>", "workload": "matmul", "seed": 17,
        "mode": mode, "generation_method_status": "failed", "status": "pending_common_final",
        "model_tokens": 999999, "model_calls": 99, "metrics": metrics, **fields}]}


def test_workload_dispersion_uses_recorded_family_aggregates_without_repooling():
    summary = {"aggregates": {"full": {"gain_stddev_between_campaigns": 999, "per_workload": {
        "matmul": {"planned": 3, "measured": 2, "unmeasured_or_failed": 1,
            "geometric_mean_gain": 1.125, "gain_stddev_between_independent_seeds": .375,
            "seeds": [{"seed": 1, "gain": 1, "status": "validated_no_gain"},
                      {"seed": 2, "gain": None, "status": "<missing>"}]},
        "scan": {"planned": 3, "measured": 1, "unmeasured_or_failed": 2,
            "geometric_mean_gain": 1, "gain_stddev_between_independent_seeds": None, "seeds": []}}}}}
    original = deepcopy(summary)
    html = workload_dispersion_table(summary)
    assert "GEMM" in html and "Scan inclusif" in html and "Écart-type entre graines" in html
    assert "0.375" in html and "1.12" in html and "999" not in html
    assert "<td>1</td><td>inconnu</td><td>non enregistrées</td>" in html
    assert "&lt;missing&gt;" in html and "<missing>" not in html
    assert "sans regrouper des opérations différentes" in html
    assert summary == original


def closure_fixture(**fields):
    cost = {"observed_rpc_calls": 0, "unique_request_ids": 0, "rpc_wall_seconds": 0,
            "reported_native_reservation_seconds": None, "reported_compile_seconds": None,
            "native_duration_missing_request_ids": [], "native_duration_conflict_request_ids": []}
    return {"status": "closed", "outcome": "validated_no_gain", "group_id": "shared-group",
            "seconds_since_generation_task_start": 300, "final_group_wall_seconds": 20,
            "reported_generation_usage": {"model_tokens": 17, "model_calls": 7, "gpu_seconds": None},
            "candidate_request_cost": cost,
            "shared_group_request_cost": {**cost, "unique_request_ids": 2, "observed_rpc_calls": 3,
                "reported_native_reservation_seconds": 2.25, "rpc_wall_seconds": 4.5},
            "cost_evidence": {"path": "final-cost-observation.json", "sha256": "a" * 64},
            "final_model_tokens": None, "final_model_calls": None, "cost_usd": None, **fields}


@pytest.mark.parametrize("mode", ["full", "true_upstream", "native_codex_vanilla"])
def test_final_closure_view_preserves_recorded_usage_and_shared_costs_once(mode):
    closure = closure_fixture()
    summary = metric_summary(mode, {"method_consumption": {"model_tokens": 123}}, final_closure=closure)
    summary["rows"].append({**summary["rows"][0], "run_id": "second", "mode": "no_pmpp"})
    original = deepcopy(summary)
    views = campaign_metric_views(summary)
    assert views[0]["final_closure"] == closure
    html = campaign_metrics_panel(views)
    assert html.count("<td>shared-group</td>") == 1  # The group cost is never allocated or summed across methods.
    assert "<td>Tokens génération enregistrés à la clôture</td><td>17</td>" in html
    assert "<td>Compteur méthode génération à la clôture</td><td>7</td>" in html
    assert "<td>Tokens / appels modèles pendant FINAL</td><td>inconnu / inconnu</td>" in html
    assert "<td>RPC observés du candidat</td><td>0</td>" in html
    assert "<td>Durées natives rapportées : sous-total connu</td><td>inconnu</td>" in html
    assert "close / validated_no_gain" in html and "ni réparties" in html
    assert summary == original


@pytest.mark.parametrize("closure", [None, {}, {"status": "pending", "seconds_since_generation_task_start": 500}])
def test_absent_or_pending_final_closure_cannot_be_inferred_from_old_task_or_gain_times(closure):
    summary = metric_summary(metrics={}, final_closure=closure, total_generation_task_wall_seconds=100,
        task_finished_at_unix=999, time_to_independently_confirmed_final_gain_seconds=200)
    view = campaign_metric_views(summary)[0]
    html = campaign_metrics_panel([view])
    assert "<td>Statut / résultat</td><td>inconnu / non enregistré</td>" in html
    assert "<td>Délai depuis le début de génération (s)</td><td>inconnu</td>" in html
    assert "Coûts FINAL partagés, une fois par groupe et reçu" not in html
    assert view["independent_final_gain"]["seconds"] == 200


def test_closed_numerically_failed_evaluation_is_not_relabelled_as_final_gain():
    summary = metric_summary(metrics={}, status="final_validation_failed", final_qualified=False,
        final_closure=closure_fixture(outcome="final_validation_failed"))
    view = campaign_metric_views(summary)[0]
    html = campaign_metrics_panel([view])
    assert "close / final_validation_failed" in html
    assert view["independent_final_gain"]["seconds"] is None
    assert view["generation_method_status"] == "failed"


def test_conflicting_shared_cost_copies_do_not_select_or_sum_a_duration():
    closure = closure_fixture()
    summary = metric_summary(metrics={}, final_closure=closure)
    summary["rows"].append({**summary["rows"][0], "run_id": "second",
        "final_closure": {**closure, "shared_group_request_cost": {"rpc_wall_seconds": 9999}}})
    html = campaign_metrics_panel(campaign_metric_views(summary))
    shared = html.split("<h3>Coûts FINAL partagés", 1)[1]
    assert "9999" not in shared and "<td>inconnu</td>" in shared


@pytest.mark.parametrize("mode", ["full", "true_upstream", "native_codex_vanilla"])
def test_metric_view_keeps_merged_method_values_and_raw_unknown_list(mode):
    metrics = {"first_useful_gain_status": "observed", "time_to_first_useful_gain_seconds": 869.6877868175507,
        "model_tokens_to_first_useful_gain": 145, "model_calls_to_first_useful_gain": 2,
        "first_useful_gain_scope": "Recorded observer scope", "first_useful_gain_note": "No feedback",
        "research_model_wall_seconds": 301.5568711250089,
        "unknown_metrics": ["time_to_first_useful_gain_seconds", "research_seconds"],
        "method_consumption": {"method_gpu_evaluation_wall_seconds": 53.174054876872106},
        "observer_wall_seconds": 12.125, "combined_method_and_observer_gpu_wall_seconds": 65.3}
    summary = metric_summary(mode, metrics, time_to_first_useful_gain_seconds=1,
        time_to_independently_confirmed_final_gain_seconds=1234.567890123)
    original = deepcopy(summary)
    view = campaign_metric_views(summary)[0]
    assert view["generation_method_status"] == "failed"
    assert view["first_useful_search_gain"] == {"status": "observed", "seconds": 869.6877868175507,
        "model_tokens": 145, "model_calls": 2, "scope": "Recorded observer scope", "note": "No feedback"}
    assert view["independent_final_gain"]["seconds"] == 1234.567890123
    costs = {cost["key"]: cost["value"] for cost in view["recorded_costs"]}
    assert costs["method_consumption.method_gpu_evaluation_wall_seconds"] == 53.174054876872106
    assert costs["combined_method_and_observer_gpu_wall_seconds"] == 65.3  # Preserve the recorded total, even if rounded.
    assert view["raw_unknown_metrics"] == original["rows"][0]["metrics"]["unknown_metrics"]
    assert view["view_annotations"] == [
        {"declared_unknown": "time_to_first_useful_gain_seconds", "recorded_key": "time_to_first_useful_gain_seconds", "recorded_value": 869.6877868175507},
        {"declared_unknown": "research_seconds", "recorded_key": "research_model_wall_seconds", "recorded_value": 301.5568711250089}]
    html = campaign_metrics_panel([view])
    assert "SEARCH" in html and "FINAL" in html and "budget égal" in html
    assert "Annotation de vue seulement" in html and "research_seconds" in html
    assert "fixture.&lt;campaign&gt;" in html and "<campaign>" not in html
    assert summary == original


@pytest.mark.parametrize("value", [None, 0, 0.0, 2.125])
def test_metric_view_distinguishes_recorded_zero_from_unknown(value):
    metrics = {"native_compile_stage_wall_seconds": value, "research_model_wall_seconds": value,
        "compile_cache_hits": value, "observer_wall_seconds": value,
        "unknown_metrics": ["research_seconds"]}
    view = campaign_metric_views(metric_summary(metrics=metrics))[0]
    costs = {cost["key"]: cost["value"] for cost in view["recorded_costs"]}
    assert costs["native_compile_stage_wall_seconds"] == value
    assert costs["compile_cache_misses"] is None
    assert view["first_useful_search_gain"]["seconds"] is None
    assert view["first_useful_search_gain"]["model_tokens"] is None
    assert view["first_useful_search_gain"]["model_calls"] is None
    assert bool(view["view_annotations"]) is (value is not None)
    html = campaign_metrics_panel([view])
    display = "inconnu" if value is None else str(value).removesuffix(".0")
    assert "<td>Compilation native</td><td>" + display + "</td><td>s</td>" in html
    assert "<td>Cache : absences</td><td>inconnu</td>" in html


@pytest.mark.parametrize("status", ["no_promoted_gain", "no_confirmed_gain", "unknown_incident", "unknown_identity_context", "not_recorded_by_method"])
def test_metric_view_never_infers_gain_or_milestone_usage_from_totals(status):
    view = campaign_metric_views(metric_summary("native_codex_vanilla", {
        "first_useful_gain_status": status, "first_valid_search_measurement_unix": 1234.5,
        "method_consumption": {"model_calls": None, "model_tokens": None}},
        time_to_independently_confirmed_final_gain_seconds=200))[0]
    assert view["first_useful_search_gain"]["status"] == status
    assert all(view["first_useful_search_gain"][key] is None for key in ("seconds", "model_tokens", "model_calls"))
    assert view["independent_final_gain"]["seconds"] == 200
    assert status in campaign_metrics_panel([view])


def test_metric_view_does_not_fill_a_missing_research_metric_from_legacy_spelling():
    summary = metric_summary(metrics={"research_seconds": 8, "unknown_metrics": ["research_seconds"]})
    view = campaign_metric_views(summary)[0]
    assert view["view_annotations"] == []
    assert next(cost["value"] for cost in view["recorded_costs"] if cost["key"] == "research_model_wall_seconds") is None
    assert summary["rows"][0]["metrics"] == {"research_seconds": 8, "unknown_metrics": ["research_seconds"]}


@pytest.mark.parametrize("metrics", [None, [], {"unknown_metrics": "not a list", "first_useful_gain_status": {"bad": "type"}, "observer_wall_seconds": False}])
def test_metric_panel_handles_absent_or_malformed_optional_metadata(metrics):
    view = campaign_metric_views(metric_summary(metrics=metrics))[0]
    html = campaign_metrics_panel([view])
    assert "<td>Observateur indépendant</td><td>inconnu</td>" in html
    assert view["view_annotations"] == []


def test_report_artifact_exports_additive_metric_view_and_prioritizes_section16(tmp_path, monkeypatch):
    import scripts.report_study_fr as renderer
    import optimizer_agent.suite_orchestrator as orchestrator
    import optimizer_agent.suite_report as suite_report
    study = tmp_path/"fixture-study"
    save(study/"plan.json", {"entries": [{"group_id": "fixture"}], "workloads": ["matmul"], "seeds": [17], "modes": ["full"]})
    summary = method_outcome_summary()
    summary["rows"][0]["metrics"] = {"first_useful_gain_status": "observed", "time_to_first_useful_gain_seconds": 3.125,
        "research_model_wall_seconds": 8.25, "unknown_metrics": ["research_seconds"]}
    raw = deepcopy(summary)
    class FixtureSuite:
        def __init__(self, plan, directory):
            assert directory == study and plan["entries"] == [{"group_id": "fixture"}]
        def status(self):
            return {"fixture_only": True}
    def fixture_report(directory, status, evidence):
        assert directory == study and status == {"fixture_only": True}
        return {"summary": deepcopy(summary), "method_outcome_policy": "Fixture method policy", "final_groups": {}}
    monkeypatch.setattr(orchestrator, "SuiteOrchestrator", FixtureSuite)
    monkeypatch.setattr(suite_report, "write_report", fixture_report)
    inventory_calls = []
    def fixture_consumption(_):
        inventory_calls.append(True)
        return {"cli_call_receipts": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0,
            "unreported_call_usage": 0, "call_provenance": [], "observed_terminal_allocated_gpu_seconds": None,
            "observed_terminal_job_count": 0, "nonterminal_or_unreported_jobIDs": []}
    monkeypatch.setattr(renderer, "project_consumption", fixture_consumption)
    monkeypatch.setattr(renderer, "plot", lambda *_: None)
    pilot = {"receipt_verified": False, "validation_errors": ["Fixture only, no qualification"]}
    monkeypatch.setattr(renderer, "frozen_pilot_summary", lambda _: pilot)
    monkeypatch.setattr(renderer, "plot_frozen_pilot", lambda *_: None)
    monkeypatch.setattr(renderer, "checkout_source_identity", lambda _: {"head": "fixture", "dirty": False, "tracked_changes": [], "untracked_files": []})
    monkeypatch.setattr(renderer, "ci_qualifies_checkout", lambda *_: False)
    monkeypatch.setattr(renderer, "control_receipts", lambda _: [])
    monkeypatch.setattr(renderer, "audit_table", lambda _: "")
    output = tmp_path/"report"
    monkeypatch.setattr(renderer.sys, "argv", ["report_study_fr", "--study", str(study), "--output", str(output)])
    renderer.main()
    html = (output/"rapport-validation.html").read_text()
    report = json.loads((output/"rapport-donnees.json").read_text())
    assert html.index("Étude §16 : trois méthodes") < html.index(renderer.frozen_pilot_panel(pilot, None))
    assert html.index(renderer.frozen_pilot_panel(pilot, None)) < html.index("Consommation, régressions et provenance")
    assert "Jalons et coûts enregistrés par campagne" in html and "ratios ponctuels &lt; 1" in html
    assert "il ne démontre pas une régression statistique" in html
    assert report["summary"] == raw and summary == raw
    assert report["campaign_metric_views"][0]["first_useful_search_gain"]["seconds"] == 3.125
    assert report["campaign_metric_views"][0]["raw_unknown_metrics"] == ["research_seconds"]
    assert report["independent_frozen_gemm_pilot"] == pilot
    assert inventory_calls == [True]
    assert report["campaign_call_observations"]["inventory_available"] is True
    assert "Coûts modèles du service de recherche" in html
    assert report["campaign_research_model_consumption"]["phase_inventory_available"] is False


@pytest.mark.parametrize("mode", ["full", "true_upstream", "native_codex_vanilla"])
def test_recorded_validity_view_preserves_method_overrides_without_deriving_fraction(mode):
    metrics = {"candidate_configurations_with_recorded_validity": 3, "correct_candidates": 2,
        "correct_candidate_fraction": .625, "configuration_observations": 9,
        "configurations_filtered_before_compilation": 4, "configurations_independently_validated": 1,
        "validation_evidence_basis": "Recorded independent <oracle> basis",
        "method_consumption": {"usage_complete": False}}
    summary = metric_summary(mode, metrics, configurations=999, correct_candidate_fraction=1)
    raw = deepcopy(summary)
    view = campaign_metric_views(summary)[0]
    projected = {entry["key"]: entry["value"] for entry in view["recorded_quality"]}
    assert projected == {key: metrics[key] for key in (
        "candidate_configurations_with_recorded_validity", "correct_candidates", "correct_candidate_fraction",
        "configuration_observations", "configurations_filtered_before_compilation", "configurations_independently_validated")}
    assert projected["correct_candidate_fraction"] == .625  # No 2/3 recomputation or denominator chosen by the renderer.
    assert view["validation_evidence_basis"] == "Recorded independent <oracle> basis"
    assert view["method_callback_usage_complete"] is False
    html = campaign_metrics_panel([view])
    assert "<td>0.625</td><td>fraction de 0 à 1</td>" in html
    assert "&lt;oracle&gt;" in html and "<oracle>" not in html
    assert "ne qualifient pas automatiquement le dernier artefact exporté ni FINAL" in html
    assert "Usage complet indiqué au dernier callback méthode de l’observateur : non" in html
    assert "sous-totaux incomplets" in html and "aucun coût facturé en USD n’est déduit" in html
    assert summary == raw


@pytest.mark.parametrize("present,value", [(False, None), (True, None), (True, 0), (True, 0.0)])
def test_recorded_validity_view_distinguishes_missing_null_and_explicit_zero(present, value):
    keys = ("candidate_configurations_with_recorded_validity", "correct_candidates", "correct_candidate_fraction",
            "configuration_observations", "configurations_filtered_before_compilation", "configurations_independently_validated")
    metrics = dict.fromkeys(keys, value) if present else {}
    view = campaign_metric_views(metric_summary(metrics=metrics))[0]
    assert {entry["key"]: entry["value"] for entry in view["recorded_quality"]} == dict.fromkeys(keys, value)
    assert view["validation_evidence_basis"] is None and view["method_callback_usage_complete"] is None
    html = campaign_metrics_panel([view])
    expected = "0" if value is not None else "inconnu"
    assert "<td>Part correcte enregistrée</td><td>" + expected + "</td><td>fraction de 0 à 1</td>" in html
    assert "Fondement de validation enregistré : non enregistré" in html
    assert "Usage complet indiqué au dernier callback méthode de l’observateur : inconnu" in html


def call_inventory_fixture(study, run="campaign-a", *, identity="a"*32, status="completed", usage=None,
                           kind="structured", suffix=None):
    suffix = suffix or ("generation/evidence/receipt.json" if kind == "native" else "generation/model/"+identity+".json")
    return {"identity": ("native-thread:" if kind == "native" else "structured:")+identity,
        "aliases": [("native-thread:" if kind == "native" else "structured:")+identity], "kind": kind,
        "receipts": [{"path": str(study/"campaigns"/run/suffix), "sha256": "c"*64,
                      "status": status, "returncode": 0 if status == "completed" else None}],
        "usage_reporting": "complete" if usage and all(key in usage for key in ("input_tokens","output_tokens")) and status == "completed" else "partial" if usage else "unreported",
        "observed_usage": usage or {},
        "missing_usage_fields": [key for key in ("input_tokens", "cached_input_tokens", "output_tokens") if key not in (usage or {})]}


def test_campaign_receipts_distinguish_seven_completed_ledger_calls_from_eight_attempts(tmp_path, monkeypatch):
    study = tmp_path/"study"
    calls = [call_inventory_fixture(study, identity=format(i+1,"032x"), usage={"input_tokens":100,"output_tokens":20}) for i in range(7)]
    calls.append(call_inventory_fixture(study, identity="f"*32, status="timeout"))
    summary = metric_summary(model_calls=7, model_tokens=840, run_id="campaign-a")
    inventory = {"call_provenance": calls, "model_tokens": 840}
    original = deepcopy((summary, inventory))
    monkeypatch.setattr(Path, "rglob", lambda *_: pytest.fail("campaign view rescanned receipts"))
    view = campaign_call_observations(summary, inventory, study)
    row = view["campaigns"][0]
    assert row["ledger_model_calls"] == 7 and row["observed_unique_cli_receipts"] == 8
    assert row["status_counts"] == {"completed":7,"failed":1,"incomplete":0,"mixed":0,"unknown":0}
    assert row["usage_reporting_counts"] == {"complete":7,"partial":0,"unreported":1}
    assert row["usage_incomplete_calls"] == 1 and row["provider_usage_complete_for_observed_calls"] is False
    assert row["known_usage_lower_bounds"]["model_tokens"] == 840
    html = campaign_call_panel(view)
    assert "<td>7</td><td>8</td>" in html and "7 / 1 / 0 / 0 / 0" in html
    assert "borne inférieure" in html and "zéro coût" in html
    assert "Compteur brut d’appels méthode" in study_campaign_table(method_outcome_summary())
    assert (summary, inventory) == original


@pytest.mark.parametrize("kind", ["structured", "native", "original"])
def test_campaign_receipts_reuse_global_alias_dedup_for_copies_and_attempts(tmp_path, kind):
    results = tmp_path/"results";study=results/"study";run=study/"campaigns/campaign-a"
    suffix = ("evidence/receipt.json" if kind == "native" else
              "model-transport/intents/"+"b"*64+"/0/cli/"+"a"*32+".json" if kind == "original" else "model/"+"a"*32+".json")
    value = {"status":"completed","usage":{"input_tokens":100,"cached_input_tokens":80,"output_tokens":20}}
    value.update({"model_started":True,"thread_ids":["native-thread"]} if kind == "native" else {"call_id":"a"*32})
    for directory in [run/"generation",run/"attempts/001/generation",results/"copy-outside/generation"]:save(directory/suffix,value)
    # Transport receipts and acknowledgements cannot manufacture an extra attempt.
    save(run/"generation/model-transport/calls.json",[value])
    save(run/"generation/model-transport/intents/key/0/result.json",value)
    consumption = project_consumption(results)
    assert consumption["cli_call_receipts"] == 1
    row = campaign_call_observations(metric_summary(run_id="campaign-a"),consumption,study)["campaigns"][0]
    assert row["observed_unique_cli_receipts"] == 1 and row["known_usage_lower_bounds"]["model_tokens"] == 120
    assert len(row["calls"][0]["canonical_receipts"]) == 2 and len(row["calls"][0]["other_receipt_origins"]) == 1


@pytest.mark.parametrize("other_known", [True,False])
def test_shared_identity_is_ambiguous_even_when_other_campaign_is_not_in_summary(tmp_path, other_known):
    study=tmp_path/"study";call=call_inventory_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    call["receipts"]+=call_inventory_fixture(study,"campaign-b",usage={"input_tokens":100,"output_tokens":20})["receipts"]
    summary=metric_summary(run_id="campaign-a")
    if other_known:summary["rows"]+=metric_summary(run_id="campaign-b")["rows"]
    view=campaign_call_observations(summary,{"call_provenance":[call]},study)
    assert len(view["ambiguous_shared_identities"])==1 and not view["unattributed_identities"]
    for row in view["campaigns"]:
        assert row["observed_unique_cli_receipts"]==0 and row["known_usage_lower_bounds"]["model_tokens"] is None
        assert row["ambiguous_identities"]==[call["identity"]] and row["attribution_status"]=="attributed_with_ambiguous_exclusions"


@pytest.mark.parametrize("layout", ["outside","substring","ack","relative","traversal"])
def test_campaign_attribution_requires_exact_canonical_root_run_and_cli_layout(tmp_path, layout):
    study=tmp_path/"study";call=call_inventory_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    receipt=call["receipts"][0]
    if layout=="outside":receipt["path"]=str(tmp_path/"archive/study/campaigns/campaign-a/generation/model"/("a"*32+".json"))
    elif layout=="substring":receipt["path"]=str(study/"campaigns/campaign-a-extra/generation/model"/("a"*32+".json"))
    elif layout=="ack":receipt["path"]=str(study/"campaigns/campaign-a/generation/model-transport/intents/key/0/result.json")
    elif layout=="relative":receipt["path"]="campaigns/campaign-a/generation/model/"+"a"*32+".json"
    else:receipt["path"]=str(study/"campaigns/campaign-a/../campaign-a/generation/model"/("a"*32+".json"))
    view=campaign_call_observations(metric_summary(run_id="campaign-a"),{"call_provenance":[call]},study)
    assert view["campaigns"][0]["observed_unique_cli_receipts"]==0 and len(view["unattributed_identities"])==1
    assert view["campaigns"][0]["known_usage_lower_bounds"]["model_tokens"] is None


@pytest.mark.parametrize("usage", [None,{}, {"input_tokens":0,"output_tokens":0}])
def test_campaign_unknown_usage_is_not_a_zero_cost_or_total(tmp_path, usage):
    study=tmp_path/"study";call=call_inventory_fixture(study,usage=usage)
    view=campaign_call_observations(metric_summary(run_id="campaign-a"),{"call_provenance":[call]},study)
    row=view["campaigns"][0]
    assert row["observed_unique_cli_receipts"]==1
    assert row["known_usage_lower_bounds"]["model_tokens"]==(0 if usage else None)
    assert row["provider_usage_complete_for_observed_calls"] is bool(usage)
    assert row["calls_without_reported_cached_usage"]==1


@pytest.mark.parametrize("consumption", [{},{"call_provenance":[]}])
def test_absent_campaign_receipts_do_not_establish_zero_usage(tmp_path, consumption):
    row=campaign_call_observations(metric_summary(run_id="campaign-a"),consumption,tmp_path/"study")["campaigns"][0]
    assert row["observed_unique_cli_receipts"]==(0 if "call_provenance" in consumption else None)
    assert row["known_usage_lower_bounds"]==dict.fromkeys(["input_tokens","cached_input_tokens","output_tokens","model_tokens"])
    assert row["provider_usage_complete_for_observed_calls"] is None


def test_mixed_snapshots_and_partial_usage_have_disjoint_status_categories(tmp_path):
    study=tmp_path/"study";call=call_inventory_fixture(study,usage={"input_tokens":100})
    call["usage_reporting"]="partial"
    call["receipts"].append({**call["receipts"][0],"status":"running","returncode":None})
    view=campaign_call_observations(metric_summary(run_id="campaign-a"),{"call_provenance":[call]},study)
    row=view["campaigns"][0]
    assert row["status_counts"]=={"completed":0,"failed":0,"incomplete":0,"mixed":1,"unknown":0}
    assert row["calls"][0]["statuses_observed"]==["completed","running"]
    assert row["usage_reporting_counts"]=={"complete":0,"partial":1,"unreported":0}
    assert row["known_usage_lower_bounds"]["model_tokens"]==100 and row["known_usage_lower_bounds"]["output_tokens"] is None


def test_completed_receipt_stderr_does_not_by_itself_mean_failed_call(tmp_path):
    study=tmp_path/"study";call=call_inventory_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    call["receipts"][0]["has_error"]=True
    row=campaign_call_observations(metric_summary(run_id="campaign-a"),{"call_provenance":[call]},study)["campaigns"][0]
    assert row["status_counts"]["completed"]==1 and row["status_counts"]["failed"]==0


def test_origin_metadata_cannot_override_campaign_derived_from_exact_path(tmp_path):
    study=tmp_path/"study";call=call_inventory_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    call["receipts"][0]["run_id"]="forged-other-campaign"
    view=campaign_call_observations(metric_summary(run_id="campaign-a"),{"call_provenance":[call]},study)
    assert view["campaigns"][0]["observed_unique_cli_receipts"]==1
    assert view["campaigns"][0]["calls"][0]["canonical_receipts"][0]["run_id"]=="campaign-a"
    assert call["receipts"][0]["run_id"]=="forged-other-campaign"


def research_phase_fixture(study, *, run="campaign-a", phase="search", research_id="b"*32,
                           call_id="a"*32, status="completed", usage=None, elapsed=1.25,
                           result=True, cli=True, payload=None):
    directory = study/"campaigns"/run/"generation"
    intent_path = directory/"research"/(research_id+"-"+phase+"-context-intent.json")
    result_path = directory/"research"/(research_id+"-"+phase+"-context-result.json")
    cli_path = directory/"model"/(call_id+".json")
    payload = payload if payload is not None else {"fixture": "returned model payload"}
    intent = {"schema_version":1,"kind":"research_prompt_context_intent","status":"intent",
        "research_id":research_id,"phase":phase,"prompt_sha256":"c"*64,
        "public_context":{"fixture_private_input":"must not be copied"}}
    save(intent_path,intent)
    returned = {"schema_version":1,"kind":"research_prompt_context_result","status":"completed",
        "research_id":research_id,"phase":phase,"call_id":call_id,
        "intent":{"path":str(intent_path),"sha256":hashlib.sha256(intent_path.read_bytes()).hexdigest()},
        "returned_payload_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}
    if result:save(result_path,returned)
    if cli:
        value={"call_id":call_id,"purpose":"research" if phase=="search" else "proposal",
            "status":status,"elapsed_seconds":elapsed}
        if usage is not None:value["usage"]=usage
        if status=="completed":value["payload"]=payload
        save(cli_path,value)
    return {"intent":intent_path,"result":result_path,"cli":cli_path,"returned":returned}


def research_projection(study, consumption, *, runs=("campaign-a",)):
    summary=metric_summary(run_id=runs[0])
    for run in runs[1:]:summary["rows"]+=metric_summary(run_id=run)["rows"]
    calls=campaign_call_observations(summary,consumption,study)
    return campaign_research_model_consumption(calls,consumption,study)


def test_research_costs_include_both_phases_and_refused_dossiers_without_second_CLI_read(tmp_path,monkeypatch):
    results=tmp_path/"results";study=results/"study"
    search=research_phase_fixture(study,usage={"input_tokens":100,"cached_input_tokens":80,"output_tokens":20})
    synthesis=research_phase_fixture(study,phase="synthesis",call_id="d"*32,
        usage={"input_tokens":40,"output_tokens":10},elapsed=2.5)
    # Neither admission nor a second aggregate invoice establishes phase cost.
    save(search["intent"].parent/("b"*32+"-blocked.json"),{"error":"guard refused after synthesis"})
    save(search["intent"].parent/("b"*32+".json"),{"usage":{"input_tokens":9999999},"research_ready":False})
    reads=[];original_read=Path.read_bytes
    def observed_read(path):
        reads.append(path)
        return original_read(path)
    monkeypatch.setattr(Path,"read_bytes",observed_read)
    consumption=project_consumption(results)
    assert reads.count(search["cli"])==reads.count(synthesis["cli"])==1
    assert consumption["model_tokens"]==170 and consumption["cli_call_receipts"]==2
    assert "fixture_private_input" not in json.dumps(consumption["research_phase_receipts"])
    raw=deepcopy(consumption)
    monkeypatch.setattr(Path,"rglob",lambda *_: pytest.fail("pure phase projection rescanned"))
    monkeypatch.setattr(Path,"read_bytes",lambda *_: pytest.fail("pure phase projection reread"))
    view=research_projection(study,consumption);row=view["campaigns"][0]
    assert row["phases"]["search"]["known_usage_lower_bounds"]["model_tokens"]==120
    assert row["phases"]["synthesis"]["known_usage_lower_bounds"]["model_tokens"]==50
    assert row["attributed_total"]["observed_unique_cli_calls"]==2
    assert row["attributed_total"]["known_usage_lower_bounds"]["model_tokens"]==170
    assert row["attributed_total"]["sum_known_CLI_elapsed_seconds"]==3.75
    assert row["outside_identified_phases"]["observed_unique_cli_calls"]==0
    assert not row["intents_without_result"] and not row["unverified_results_or_receipts"]
    assert row["documentary_end_to_end_seconds"] is row["cost_usd"] is None
    html=campaign_research_model_panel(view)
    assert "Recherche" in html and "Synthèse" in html and "<td>170</td><td>3.75</td>" in html
    assert "dossier est ensuite refusé" in html and "bout en bout, qui reste inconnu" in html
    assert "USD inconnus" in html and "hors phases identifiées" in html
    assert consumption==raw


@pytest.mark.parametrize("status",["timeout","failed","prepared"])
def test_research_intent_without_return_never_assigns_failed_or_prepared_CLI_to_a_phase(tmp_path,status):
    study=tmp_path/"study"
    research_phase_fixture(study,phase="synthesis",status=status,result=False,elapsed=9)
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption)["campaigns"][0]
    assert row["attributed_total"]["observed_unique_cli_calls"]==0
    assert row["attributed_total"]["known_usage_lower_bounds"]["model_tokens"] is None
    assert row["outside_identified_phases"]["observed_unique_cli_calls"]==1
    assert row["outside_identified_phases"]["known_usage_lower_bounds"]["model_tokens"] is None
    assert row["outside_identified_phases"]["sum_known_CLI_elapsed_seconds"]==9
    assert row["outside_identified_phases"]["status_counts"]["incomplete" if status=="prepared" else "failed"]==1
    assert len(row["intents_without_result"])==1 and row["intent_counts"]=={"search":0,"synthesis":1}


def test_research_intent_alone_is_not_an_invocation_or_zero_invoice(tmp_path):
    study=tmp_path/"study";research_phase_fixture(study,result=False,cli=False)
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption)["campaigns"][0]
    assert consumption["cli_call_receipts"]==0 and row["intent_counts"]["search"]==1
    assert row["attributed_total"]["known_usage_lower_bounds"]["model_tokens"] is None
    assert row["attributed_total"]["sum_known_CLI_elapsed_seconds"] is None
    assert len(row["intents_without_result"])==1


@pytest.mark.parametrize("change",["phase","research_id","kind","status","intent_sha","intent_path","malformed_intent","payload","call_id","missing_CLI","malformed_root","boolean_schema"])
def test_research_phase_join_fails_closed_without_losing_global_CLI_consumption(tmp_path,change):
    study=tmp_path/"study"
    paths=research_phase_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    value=paths["returned"]
    if change=="phase":value["phase"]="synthesis"
    elif change=="research_id":value["research_id"]="f"*32
    elif change=="kind":value["kind"]="research_prompt_context_intent"
    elif change=="status":value["status"]="intent"
    elif change=="intent_sha":value["intent"]["sha256"]="f"*64
    elif change=="intent_path":value["intent"]["path"]=str(tmp_path/"private.sqlite")
    elif change=="malformed_intent":value["intent"]={"path":[1],"sha256":[2]}
    elif change=="payload":value["returned_payload_sha256"]="f"*64
    elif change=="call_id":value["call_id"]="e"*32
    elif change=="malformed_root":value=[1]
    elif change=="boolean_schema":value["schema_version"]=True
    else:paths["cli"].unlink()
    save(paths["result"],value)
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption)["campaigns"][0]
    assert row["attributed_total"]["observed_unique_cli_calls"]==0 and row["unverified_results_or_receipts"]
    assert consumption["model_tokens"]==(0 if change=="missing_CLI" else 120)
    assert row["outside_identified_phases"]["known_usage_lower_bounds"]["model_tokens"]==(None if change=="missing_CLI" else 120)


def test_research_phase_copies_and_attempts_are_one_cost(tmp_path):
    results=tmp_path/"results";study=results/"study"
    paths=research_phase_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    source=study/"campaigns/campaign-a/generation"
    shutil.copytree(source,study/"campaigns/campaign-a/attempts/001/generation")
    shutil.copytree(source,results/"archive/study/campaigns/campaign-a/generation")
    consumption=project_consumption(results)
    row=research_projection(study,consumption)["campaigns"][0]
    assert consumption["cli_call_receipts"]==row["attributed_total"]["observed_unique_cli_calls"]==1
    assert row["attributed_total"]["known_usage_lower_bounds"]["model_tokens"]==120
    assert row["attributed_total"]["sum_known_CLI_elapsed_seconds"]==1.25
    assert row["intent_counts"]["search"]==1
    assert len(row["phases"]["search"]["calls"][0]["phase_evidence"])==2


def test_research_phase_cannot_reassign_another_campaigns_CLI(tmp_path):
    study=tmp_path/"study"
    research_phase_fixture(study,run="campaign-a",cli=False)
    research_phase_fixture(study,run="campaign-b",usage={"input_tokens":100,"output_tokens":20},result=False)
    consumption=project_consumption(tmp_path)
    rows=research_projection(study,consumption,runs=("campaign-a","campaign-b"))["campaigns"]
    assert all(row["attributed_total"]["observed_unique_cli_calls"]==0 for row in rows)
    assert rows[0]["unverified_results_or_receipts"] and rows[1]["outside_identified_phases"]["known_usage_lower_bounds"]["model_tokens"]==120


def test_research_phase_rejects_an_intent_owned_by_another_campaign(tmp_path):
    study=tmp_path/"study"
    first=research_phase_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    second=research_phase_fixture(study,run="campaign-b",cli=False,result=False)
    value=first["returned"];value["intent"]={"path":str(second["intent"]),"sha256":hashlib.sha256(second["intent"].read_bytes()).hexdigest()}
    save(first["result"],value)
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption,runs=("campaign-a","campaign-b"))["campaigns"][0]
    assert row["attributed_total"]["observed_unique_cli_calls"]==0
    assert row["unverified_results_or_receipts"][0]["reason"]=="intent_link_unverified"
    assert row["outside_identified_phases"]["known_usage_lower_bounds"]["model_tokens"]==120


def test_shared_CLI_identity_and_conflicting_phase_evidence_are_not_counted_twice(tmp_path):
    study=tmp_path/"study"
    research_phase_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    research_phase_fixture(study,phase="synthesis",research_id="d"*32,usage={"input_tokens":100,"output_tokens":20})
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption)["campaigns"][0]
    assert row["ambiguous_phase_identities"]==["structured:"+"a"*32]
    assert row["attributed_total"]["observed_unique_cli_calls"]==0
    assert row["outside_identified_phases"]["known_usage_lower_bounds"]["model_tokens"]==120
    research_phase_fixture(study,run="campaign-b",usage={"input_tokens":100,"output_tokens":20})
    consumption=project_consumption(tmp_path)
    for row in research_projection(study,consumption,runs=("campaign-a","campaign-b"))["campaigns"]:
        assert row["attributed_total"]["observed_unique_cli_calls"]==0
        assert row["ambiguous_campaign_identities"]==["structured:"+"a"*32]
    assert consumption["cli_call_receipts"]==1 and consumption["model_tokens"]==120


@pytest.mark.parametrize("elapsed,other,expected,reason",[(0,0,0,None),(None,None,None,"unreported"),(1,2,None,"conflicting"),(1,-2,None,"invalid"),(1,True,None,"invalid")])
def test_research_CLI_durations_preserve_zero_unknown_and_conflicting_snapshots(tmp_path,elapsed,other,expected,reason):
    study=tmp_path/"study";paths=research_phase_fixture(study,elapsed=elapsed,usage={"input_tokens":0,"output_tokens":0})
    value=json.loads(paths["cli"].read_text());value["elapsed_seconds"]=other
    save(tmp_path/"copy/model"/paths["cli"].name,value)
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption)["campaigns"][0]["attributed_total"]
    assert row["sum_known_CLI_elapsed_seconds"]==expected
    assert row["known_usage_lower_bounds"]["model_tokens"]==0
    assert row["CLI_duration_complete_for_observed_calls"] is (reason is None)
    if reason:assert row["CLI_duration_unknown_calls"][0]["reason"]==reason


def test_research_phase_partial_usage_and_uninstrumented_methods_remain_explicit(tmp_path):
    study=tmp_path/"study";research_phase_fixture(study,usage={"input_tokens":0})
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption)["campaigns"][0]["attributed_total"]
    assert row["usage_reporting_counts"]=={"complete":0,"partial":1,"unreported":0}
    assert row["known_usage_lower_bounds"]["model_tokens"]==0 and row["known_usage_lower_bounds"]["output_tokens"] is None
    for kind in ["native","structured"]:
        call=call_inventory_fixture(study,kind=kind,usage={"input_tokens":100,"output_tokens":20})
        call["cli_metadata"]=[{"purpose":"research","elapsed_seconds":2}]
        raw={"call_provenance":[call],"research_phase_receipts":[]}
        view=research_projection(study,raw)["campaigns"][0]
        assert view["attributed_total"]["known_usage_lower_bounds"]["model_tokens"] is None
        assert view["outside_identified_phases"]["known_usage_lower_bounds"]["model_tokens"]==120


@pytest.mark.parametrize("symlink",["file","directory"])
def test_research_phase_index_never_reads_symlinks_or_follows_a_supplied_private_reference(tmp_path,monkeypatch,symlink):
    study=tmp_path/"study";paths=research_phase_fixture(study,usage={"input_tokens":100,"output_tokens":20})
    private=tmp_path/"private.sqlite";private.write_text("NEVER OPEN")
    value=paths["returned"];value["intent"]={"path":str(private),"sha256":"c"*64};save(paths["result"],value)
    if symlink=="file":
        extra=paths["intent"].parent/("f"*32+"-search-context-intent.json");extra.symlink_to(private)
    else:
        target=tmp_path/"private-directory";target.mkdir();(target/("f"*32+"-search-context-intent.json")).write_text("NEVER OPEN")
        extra=study/"campaigns/campaign-a/attempts/001/generation/research";extra.parent.mkdir(parents=True);extra.symlink_to(target)
    original=Path.read_bytes
    def guarded(path):
        assert path!=private and not path.is_symlink() and "private-directory" not in path.parts
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    consumption=project_consumption(tmp_path)
    row=research_projection(study,consumption)["campaigns"][0]
    assert row["attributed_total"]["observed_unique_cli_calls"]==0 and row["unverified_results_or_receipts"]
