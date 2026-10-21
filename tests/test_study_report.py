"""Observed consumption cannot double count cached/duplicate/reasoning tokens."""
import json
import hashlib
from copy import deepcopy
import subprocess

import pytest

from scripts.report_study_fr import (project_consumption, frozen_pilot_summary,
    frozen_pilot_panel, plot_frozen_pilot, FROZEN_ROLES, FROZEN_PAIRS)
from scripts.report_study_fr import checkout_source_identity, ci_qualifies_checkout, control_receipts


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


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
