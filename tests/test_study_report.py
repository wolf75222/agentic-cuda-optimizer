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
from scripts.report_study_fr import study_aggregate_table, study_campaign_table
from scripts.report_study_fr import DIAGNOSTIC_V2_AUDIT, POLICY_AUTHORITY, PARENT_OBSERVATION
from scripts.report_study_fr import (ORIGINAL_EXECUTION, COMMON_FINAL_FREEZE, RESEARCH_REFRESH,
    UPSTREAM_COMMIT, REFRESH_CHECKS, FREEZE_METHOD_CHECKS, LABELS, execution_evidence_panel)
from scripts.report_study_fr import ORIGINAL_CAMPAIGN, FINAL_DIRECTORY, FIRST_GROUP
from scripts.report_study_fr import comparison_view, direct_comparison_views, direct_comparison_table, plot
from scripts.report_study_fr import RESEARCH_REFRESH_RESUMED, REFRESH_RESUMPTION_CHECKS
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


def test_direct_legacy_exact_artifacts_are_identity_without_changing_raw_intervals():
    pair = {"status": "confirmed", "ok": True, "artifact_hash": "a"*64, "incumbent_artifact_hash": "a"*64,
            "interval": {"gain": .87, "lower": .84, "upper": .90, "case_gains": {"case": {"gain": .85}}}}
    original = deepcopy(pair)
    views = direct_comparison_views({"group-fixture": {"method_comparisons": {"full_vs_greedy": pair}}})
    assert pair == original and views[0]["gain_basis"] == "exact_artifact_identity"
    assert views[0]["gain"] == 1 and views[0]["lower"] is views[0]["upper"] is None
    assert views[0]["source_interval_preserved"] == pair["interval"] and views[0]["case_gains"] == {}
    html = direct_comparison_table(views)
    assert "identité exacte des artefacts" in html and "sans intervalle : identité" in html
    assert "0.87" not in html and "0.84" not in html


@pytest.mark.parametrize("first,second", [(None, None), ("", ""), ("source-only", "source-only"), ("a"*64, "b"*64)])
def test_identity_requires_equal_complete_artifact_hashes_not_missing_or_source_only(first, second):
    view = comparison_view({"artifact_hash": first, "incumbent_artifact_hash": second,
                            "source_hash": "a"*64, "interval": {"gain": .87, "lower": .84, "upper": .90}})
    assert view["gain_basis"] == "paired_measurement" and view["gain"] == .87


def test_pilot_identity_direct_pair_has_no_legacy_interval_in_panel_or_plot(tmp_path):
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
    save(tmp_path/"comparison.json", receipt)
    pilot = frozen_pilot_summary(tmp_path/"comparison.json")
    assert pilot["receipt_verified"]
    direct = pilot["comparisons"]["agent_vs_native_codex"]
    assert direct["gain_basis"] == "exact_artifact_identity" and direct["interval"]["lower"] is None
    assert direct["source_interval_preserved"]["gain"] == .742984
    html = frozen_pilot_panel(pilot, None)
    assert "ratio 1 sans intervalle statistique" in html and "0.742984" not in html
    assert plot_frozen_pilot(pilot, tmp_path).is_file()


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
