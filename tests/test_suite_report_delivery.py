"""CPU fixtures for presentation only; no study, GPU or provider is contacted."""
from copy import deepcopy
import json

import pytest

from optimizer_agent import suite_report


@pytest.fixture
def render(tmp_path, monkeypatch):
    monkeypatch.setattr(suite_report, "build_requirement_status", lambda evidence: [])

    def write(summary):
        original = deepcopy(summary)
        suite_report.write_report(tmp_path, {"summary": summary})
        assert summary == original
        assert json.loads((tmp_path / "report.json").read_text())["summary"] == original
        return (tmp_path / "rapport-fr.md").read_text()

    return write


def row(mode="full", **updates):
    return {"workload": "matmul", "seed": 1729, "mode": mode,
            "generation_method_status": "completed", "candidate_fallback": None,
            "status": "validated_no_gain", "gain": None, "lower": None, "upper": None,
            "artifact_hash": "artifact", **updates}


def summary(rows=None, per_workload=None):
    return {"complete": False, "scientific_qualified": False, "rows": rows or [row()],
            "aggregates": {"full": {"planned": 3, "method_completed": 1, "method_failed": 1,
                "method_status_unknown": 1, "final_measured": 1, "qualified": 1,
                "regressions": 1, "unmeasured_or_failed": 2,
                "geometric_mean_all_measured_gain": .8, "gain_stddev_between_campaigns": None,
                "per_workload": per_workload or {}}}}


def cost(native=2.25):
    return {"observed_rpc_calls": 7, "unique_request_ids": 6, "rpc_wall_seconds": 11.5,
            "reported_native_reservation_seconds": native, "reported_compile_seconds": None,
            "native_duration_missing_request_ids": ["missing|request"],
            "native_duration_conflict_request_ids": [], "request_ids": ["request"]}


def closure(**updates):
    return {"status": "closed", "outcome": "final_validation_failed", "group_id": "shared-group",
            "closed_at_unix": 200, "seconds_since_generation_task_start": 120.5,
            "final_group_wall_seconds": 42.25,
            "reported_generation_usage": {"model_tokens": None, "model_calls": 0, "gpu_seconds": 3.5},
            "reported_generation_usage_source": "manifest.usage",
            "candidate_request_cost": cost(), "shared_group_request_cost": cost(9.25),
            "cost_evidence": {"path": "group/request-costs.json", "sha256": "a" * 64},
            "final_model_tokens": None, "final_model_calls": None, "cost_usd": None, **updates}


def test_markdown_labels_distinguish_identity_point_ratio_and_method_failure(render):
    markdown = render(summary([row(generation_method_status="failed", generation_method_error="exit | <1>",
        candidate_fallback={"retained": True, "selected_origin": "supplied_initial", "selected_iteration": 0,
                            "reason": "method_failed"},
        status="certified", gain=1, gain_basis="exact_baseline_artifact")]))
    assert "Ratios disponibles (mesures/identités)" in markdown
    assert "Régressions : ratios ponctuels < 1" in markdown
    assert "ne suffit pas à établir une régression statistique" in markdown
    assert "Écart-type descriptif toutes familles" in markdown
    assert "échouée | exit &#124; &lt;1&gt;" in markdown
    assert "certified | 1 | identité exacte, sans IC | identité exacte avec le naïf" in markdown
    assert "ne créent aucune mesure indépendante" in markdown


def test_workload_dispersion_projects_recorded_aggregates_without_pooling(render):
    families = {"matmul": {"planned": 3, "measured": 2, "unmeasured_or_failed": 1,
        "geometric_mean_gain": 7.25, "gain_stddev_between_independent_seeds": 8.125,
        "seeds": [{"seed": 1729, "gain": .8, "status": "not_improved"},
                  {"seed": 2718, "gain": None, "status": "failed"}]},
        "reduction": {"planned": 3, "measured": 1, "unmeasured_or_failed": 2,
            "geometric_mean_gain": 1, "gain_stddev_between_independent_seeds": None, "seeds": []}}
    markdown = render(summary(per_workload=families))
    # These intentionally independent stored values must not be recalculated from rows/seeds.
    assert "| full | matmul | 3 | 2 | 1 | 7.25 | 8.125 |" in markdown
    assert "1729: 0.8 (not_improved); 2718: inconnu (failed)" in markdown
    assert "| full | reduction | 3 | 1 | 2 | 1 | inconnu (n&lt;2) | non enregistrées |" in markdown
    assert "sans recalcul ni regroupement des temps bruts" in markdown


def test_final_closure_is_distinct_from_gain_and_unknown_is_not_zero(render):
    markdown = render(summary([row(final_closure=closure()), row("native", final_closure=None),
                              row("pending", final_closure={"status": "pending", "final_model_calls": 999})]))
    assert "close / final_validation_failed | 120.5 | 42.25 | inconnu | 0 | 3.5 | inconnu | inconnu | inconnu" in markdown
    assert "native | inconnu / non enregistré" in markdown
    assert "pending | inconnu / non enregistré" in markdown
    assert "999" not in markdown
    assert "snapshot brut du ledger" in markdown and "potentiellement incomplet" in markdown
    assert "après les décisions et comparaisons du groupe" in markdown
    assert "n’atteste aucun terminal Slurm" in markdown
    assert "IDs durée native manquante" in markdown
    assert "1 : missing&#124;request | 0 (aucun)" in markdown
    assert "sous-totaux connus dédupliqués" in markdown


def test_shared_costs_display_once_without_adding_to_candidate_cost(render):
    markdown = render(summary([row(final_closure=closure()), row("native", final_closure=closure())]))
    shared = markdown.split("### Coûts FINAL partagés, une fois par groupe et reçu", 1)[1]
    assert shared.count("| shared-group | " + "a" * 64) == 1
    assert "| 7 | 6 | 11.5 | 9.25 | inconnu |" in shared
    candidate = markdown.split("### Requêtes FINAL propres au candidat", 1)[1].split("### Coûts FINAL partagés", 1)[0]
    assert candidate.count("| 7 | 6 | 11.5 | 2.25 | inconnu |") == 2
    assert "11.5 | 11.5" not in candidate
    assert "ni réparties entre méthodes ni ajoutées" in shared


def test_conflicting_shared_copies_remain_unknown_even_after_third_copy(render):
    changed = cost(99)
    markdown = render(summary([row(final_closure=closure()),
        row("native", final_closure=closure(shared_group_request_cost=changed)),
        row("greedy", final_closure=closure())]))
    shared = markdown.split("### Coûts FINAL partagés, une fois par groupe et reçu", 1)[1]
    assert shared.count("| shared-group | " + "a" * 64) == 1
    assert "| inconnu | inconnu | inconnu | inconnu | inconnu | inconnu | inconnu |" in shared
    assert "9.25" not in shared and "99" not in shared


def test_historical_absence_does_not_reconstruct_final_closure(render):
    markdown = render(summary([row(task_started_at_unix=10, task_finished_at_unix=400, wall_seconds=390)]))
    assert "inconnu / non enregistré" in markdown
    assert "Coûts partagés non enregistrés : inconnu" in markdown
    assert "anciens timestamps ne sont pas reconstruits" in markdown
    assert "Agrégats par famille non enregistrés : inconnu" in markdown
