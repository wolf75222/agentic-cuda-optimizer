"""Observed consumption cannot double count cached/duplicate/reasoning tokens."""
import json
import hashlib
from copy import deepcopy
import subprocess

import pytest

from scripts.report_study_fr import (project_consumption, frozen_pilot_summary,
    frozen_pilot_panel, plot_frozen_pilot, FROZEN_ROLES, FROZEN_PAIRS)
from scripts.report_study_fr import checkout_source_identity, ci_qualifies_checkout, control_receipts
from scripts.report_study_fr import gpu_lease_consumption
from scripts.report_study_fr import study_aggregate_table, study_campaign_table
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
        "all_modes_available": True, "candidates": candidates}})


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
    assert pending["path"] is pending["receipt_sha256"] is pending["job_id"] is None
    assert "en attente" in pending["status"]
