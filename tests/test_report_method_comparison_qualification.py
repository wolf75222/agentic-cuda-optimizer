"""CPU fixtures: scientific display decisions never fall back to stale groups."""
from copy import deepcopy
import json

import pytest

from optimizer_agent.quality_suite import summarize_suite
from optimizer_agent import suite_report
from scripts.report_study_fr import (case_comparison_table, comparison_view,
    direct_comparison_table, direct_comparison_views)


KIND = "artifact_aware_method_comparison_summary"
GROUP, PAIR = "fixture-group", "full_vs_native_codex_vanilla"


def pair(**updates):
    return {"status": "promoted", "ok": True, "artifact_hash": "a" * 64,
            "incumbent_artifact_hash": "b" * 64,
            "interval": {"gain": 9.875, "lower": 9.765, "upper": 9.985,
                         "case_gains": {"public-final-case": {"gain": 9.875, "lower": 9.765, "upper": 9.985}}},
            **updates}


def raw_groups(value=None):
    return {GROUP: {"method_comparisons": {PAIR: pair() if value is None else value}}}


def projected(value=None):
    value = pair() if value is None else value
    return {"method_comparisons_view_kind": KIND,
            "method_comparisons": {GROUP: {PAIR: {**value, "kind": KIND, "group_id": GROUP, "pair_key": PAIR}}}}


def actual_summary_with_incident():
    modes = ["full", "native_codex_vanilla"]
    entries = [{"workload": "matmul", "seed": 17, "mode": mode, "group_id": GROUP,
                "config": {"run_id": "run-" + mode}} for mode in modes]
    plan = {"entries": entries, "modes": modes, "workloads": ["matmul"]}
    manifests = {"run-" + mode: {"status": "generation_complete", "method_status": "completed"} for mode in modes}
    groups = raw_groups()
    groups[GROUP].update(execution_complete=True, all_modes_available=True, candidates={
        mode: {"status": "certified", "qualified": True, "artifact_hash": artifact,
               "confirmation": {"interval": {"gain": 1.25, "lower": 1.1, "upper": 1.4}}}
        for mode, artifact in zip(modes, ["a" * 64, "b" * 64])})
    original = deepcopy(groups)
    summary = summarize_suite(plan, manifests, groups, artifact_qualification_incidents={
        "a" * 64: [{"incident_id": "public-static-incident"}]})
    assert groups == original
    return summary, groups


def test_actual_incident_projection_controls_html_markdown_json_and_case_views(tmp_path, monkeypatch):
    summary, groups = actual_summary_with_incident()
    before = deepcopy((summary, groups))
    views = direct_comparison_views(groups, summary)
    assert views == suite_report.direct_method_comparison_views(summary, groups)
    view = views[0]
    assert view["comparison_source"] == "summary.method_comparisons"
    assert view["gain_basis"] == view["source_status"] == "artifact_qualification_not_admitted"
    assert view["source_ok"] is view["source_qualified"] is False
    assert view["artifact_qualification_incident_ids"] == ["public-static-incident"]
    assert view["gain"] is view["lower"] is view["upper"] is None and view["case_gains"] == {}
    html = direct_comparison_table(views) + case_comparison_table(summary, views)
    assert "artefact non admis scientifiquement" in html and "IC non établi" in html
    assert "9.88" not in html and "public-final-case" not in html
    monkeypatch.setattr(suite_report, "build_requirement_status", lambda _: [])
    report = suite_report.write_report(tmp_path, {"summary": summary, "final_groups": groups})
    markdown = (tmp_path / "rapport-fr.md").read_text()
    assert "## Contrastes directs des groupes finaux" in markdown
    assert "artefact non admis scientifiquement | inconnu | IC non établi" in markdown
    assert "9.875" not in markdown and "summary.method_comparisons" in markdown
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["direct_comparison_views"] == report["direct_comparison_views"] == views
    assert saved["final_groups"] == groups and saved["summary"] == summary
    assert saved["final_groups"][GROUP]["method_comparisons"][PAIR]["interval"]["gain"] == 9.875
    assert summary["method_comparisons"][GROUP][PAIR]["observed_unqualified_interval"]["gain"] == 9.875
    assert (summary, groups) == before


@pytest.mark.parametrize("block", [
    {"status": "artifact_qualification_not_admitted"},
    {"artifact_incident_blocked": True},
    {"status": "artifact_qualification_not_admitted", "gain_basis": "exact_artifact_identity", "ok": False},
])
@pytest.mark.parametrize("same_artifact", [False, True])
def test_disqualified_interval_and_identity_are_not_reenabled(block, same_artifact):
    value = pair(**block)
    if same_artifact:
        value["incumbent_artifact_hash"] = value["artifact_hash"]
    before = deepcopy(value)
    view = comparison_view(value)
    assert view["gain_basis"] == "artifact_qualification_not_admitted"
    assert view["gain"] is view["lower"] is view["upper"] is None and view["case_gains"] == {}
    assert value == before


@pytest.mark.parametrize("summary", [
    {"method_comparisons": {}},
    {"method_comparisons_view_kind": KIND},
    {"method_comparisons_view_kind": KIND, "method_comparisons": {}},
    {"method_comparisons_view_kind": KIND, "method_comparisons": None},
    {"method_comparisons_view_kind": KIND, "method_comparisons": []},
    {"method_comparisons_view_kind": "future-unknown-kind", "method_comparisons": {GROUP: {PAIR: pair()}}},
])
def test_present_empty_or_unknown_projection_never_falls_back_to_raw(summary):
    raw = raw_groups()
    before = deepcopy((summary, raw))
    assert direct_comparison_views(raw, summary) == []
    assert (summary, raw) == before


@pytest.mark.parametrize("change", [
    {"kind": "legacy"}, {"kind": None}, {"group_id": "another-group"}, {"pair_key": "other-pair"},
])
def test_malformed_projected_pair_is_explicitly_unavailable(change):
    summary = projected()
    summary["method_comparisons"][GROUP][PAIR].update(change)
    view = direct_comparison_views(raw_groups(), summary)[0]
    assert view["comparison_source"] == "summary.method_comparisons"
    assert view["source_status"] == "summary_projection_unavailable"
    assert view["gain_basis"] == "comparison_unavailable" and view["gain"] is None
    assert view["case_gains"] == {}


def test_positive_projection_is_used_exactly_and_raw_only_pairs_are_not_reinserted():
    summary = projected(pair(interval={"gain": 1.234, "lower": 1.1, "upper": 1.3}))
    groups = raw_groups()
    groups[GROUP]["method_comparisons"]["unprojected-raw-contrast"] = pair()
    view, = direct_comparison_views(groups, summary)
    assert view["gain"] == 1.234 and view["lower"] == 1.1 and view["upper"] == 1.3
    assert view["comparison"] == PAIR and view["comparison_source"] == "summary.method_comparisons"


def test_legacy_without_a_summary_projection_keeps_identity_rules():
    value = pair(incumbent_artifact_hash="a" * 64)
    view, = direct_comparison_views(raw_groups(value), {"rows": []})
    assert view["comparison_source"] == "legacy_final_group"
    assert view["gain_basis"] == "exact_artifact_identity" and view["gain"] == 1
    assert view["lower"] is view["upper"] is None and view["case_gains"] == {}


@pytest.mark.parametrize("bad", [True, "9.875", [], float("inf")])
def test_malformed_legacy_ratio_does_not_crash_new_markdown_view(bad):
    view = comparison_view(pair(interval={"gain": bad, "lower": bad, "upper": bad}))
    assert view["gain"] is view["lower"] is view["upper"] is None
    assert "inconnu" in "\n".join(suite_report._direct_comparison_lines([{
        **view, "group_id": GROUP, "comparison": PAIR, "comparison_source": "legacy_final_group"}]))
