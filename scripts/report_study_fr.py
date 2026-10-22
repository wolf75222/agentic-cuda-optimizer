#!/usr/bin/env python3
"""Render a French, evidence-scoped HTML report and standalone scientific plot."""
import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LABELS = {"full": "Optimiseur amélioré", "true_upstream": "Agent original 1e9464d",
          "native_codex_vanilla": "Codex vanilla Luna Max", "greedy_parent": "Parent glouton",
          "no_autotuning": "Sans autotuning", "no_research": "Sans recherche",
          "no_pmpp": "Sans PMPP", "pmpp_permanent_only": "Guide PMPP permanent seul"}
FAMILIES = {"reduction": "Réduction", "scan": "Scan inclusif", "stencil": "Stencil",
            "convolution": "Convolution", "matmul": "GEMM", "normalization": "Normalisation RMS"}
FROZEN_ROLES = {"baseline": "Kernel naïf", "agent": "Agent amélioré, kernel brut",
                "single_prompt": "Proposition structurée sans outils", "native_codex": "Codex CLI vanilla natif"}
FROZEN_PAIRS = {"agent_vs_baseline": ("agent", "baseline"),
                "single_prompt_vs_baseline": ("single_prompt", "baseline"),
                "native_codex_vs_baseline": ("native_codex", "baseline"),
                "agent_vs_native_codex": ("agent", "native_codex")}
FROZEN_RECEIPT = "results/gemm-frozen-final-20261021/comparison.json"
DIAGNOSTIC_V2_AUDIT = "work/live/research-convolution-diagnostic-b9bda60.json"
POLICY_AUTHORITY = "results/policy-controls-20261021/search-policy-acceptance.json"
PARENT_OBSERVATION = "results/policy-controls-20261021/selected-parent-observation.json"
ORIGINAL_EXECUTION = "work/live/true-upstream-first-generation-ef057b3.json"
COMMON_FINAL_FREEZE = "work/live/common-final-convolution-1729-freeze-audit.json"
RESEARCH_REFRESH = "results/research-refresh-ac02-20261021/research-refresh-acceptance.json"
RESEARCH_REFRESH_RESUMED = "results/research-refresh-ac02-resumed-20261021/research-refresh-acceptance.json"
RESEARCH_REFRESH_RESUMED_2 = "results/research-refresh-ac02-resumed-2-20261021/research-refresh-acceptance.json"
RESEARCH_REFRESH_RESUMED_3 = "results/research-refresh-ac02-resumed-3-20261021/research-refresh-acceptance.json"
STANDALONE_FINAL_SUITE = "results/final-suite-acceptance-20261021/final-suite-acceptance.json"
FINAL_SUITE_CHECKS = ("search_oracle_passed", "distinct_artifacts", "final_oracles_passed", "two_disjoint_final_cases",
    "confirmation_explicitly_final", "fresh_final_shapes_only", "final_input_buffers_used", "exact_fixed_pairs",
    "alternating_AB_BA", "all_four_sanitizers_on_final_cases", "sanitizer_requests_use_exact_final_inputs",
    "finalization_completed", "exposed_holdout_not_reused", "final_authorization_revoked")
FIRST_GROUP = "quality-20261021-v2.convolution.1729"
ORIGINAL_CAMPAIGN = "results/quality-study-v2/campaigns/" + FIRST_GROUP + ".true_upstream"
FINAL_DIRECTORY = "results/quality-study-v2/final-groups/" + FIRST_GROUP
UPSTREAM_COMMIT = "1e9464da54dfc651337a97c3643bfeefec712bc1"
REFRESH_CHECKS = (
    "new_family_real_applicable_dossier", "compatible_adjustment_exact_dossier_reuse_without_research",
    "compatible_adjustment_uses_previous_source_parent", "compatible_proposal_changes_source_launch_or_parameters",
    "uncovered_technique_blocked_before_candidate", "diagnostic_contains_exact_blocked_technique",
    "targeted_new_dossier_covers_technique", "targeted_dossier_links_immutable_diagnostic",
    "new_covered_proposal_admitted", "only_three_admitted_generation_events",
    "no_trial_integration_or_measurement", "no_final_holdout_exposed")
FREEZE_METHOD_CHECKS = (
    "artifact_hash_matches_generation", "candidate_identity_matches", "copied_source_bytes_identical",
    "distinct_copied_source_path", "frozen_candidate_source_sha_matches", "generation_candidate_source_sha_matches",
    "generation_reports_execution_complete", "kernel_matches_generation", "launch_rule_matches_generation",
    "load_candidate_accepts_frozen", "load_candidate_accepts_generation", "normalized_launch_rules_match",
    "parameters_match_generation", "source_hash_matches_generation")
REFRESH_RESUMPTION_CHECKS = (
    "resume_original_prefix_unchanged", "resume_no_old_model_call_repeated", "resume_no_new_selection_or_diagnostic",
    "resume_hardware_context_bound", "resume_source_unchanged")
PROOF_PATHS = (
    ("Véritable agent original — première génération ROMEO", ORIGINAL_EXECUTION),
    ("Gel du premier groupe final — identités et aliases", COMMON_FINAL_FREEZE),
    ("AC-02 — renouvellement documentaire réel", RESEARCH_REFRESH),
    ("AC-02 — reprise documentaire isolée", RESEARCH_REFRESH_RESUMED),
    ("AC-02 — deuxième reprise documentaire isolée", RESEARCH_REFRESH_RESUMED_2),
    ("AC-02 — complément documentaire après recherche acceptée", RESEARCH_REFRESH_RESUMED_3),
    ("Finalisation autonome : mesures sur les cas FINAL distincts", STANDALONE_FINAL_SUITE),
    ("Autotuning réel", "results/gpu-autotuning-acceptance/20261021-current/autotuning-acceptance.json"),
    ("Précision, cache et reconstruction", "results/gpu-contract-acceptance/20261021-current/acceptance-summary.json"),
    ("Diagnostic et nouvelle recherche", "results/diagnostic-loop-20261021/diagnostic-loop-acceptance.json"),
    ("Diagnostic ciblé avec lecteur v2", DIAGNOSTIC_V2_AUDIT),
    ("Helper diagnostic v2 — exécution incomplète (773095)", "results/diagnostic-loop-v2-20261021/diagnostic-loop-acceptance.json"),
    ("Contrôles TreeQuest — autorité partielle (773380)", POLICY_AUTHORITY),
    ("Parent sélectionné — observation distincte du contrôle strict", PARENT_OBSERVATION),
    ("Contraintes d’autotuning — contrôle réel", "results/tuning-constraints-20261021/tuning-constraints-acceptance.json"),
    ("Comparaison GEMM figée, quatre artefacts bruts", FROZEN_RECEIPT),
    ("Transport GPU RPC réel", "results/gpu-rpc-acceptance/20261021-current/allocation-receipt.json"),
    ("Isolation profil/benchmark et contrôle final négatif", "results/gpu-final-isolation-acceptance/20261021-current/allocation-receipt.json"),
    ("Finaliste natif reconstruit et revalidé", "results/gpu-reconstructed-acceptance/20261021-current/acceptance-summary.json"),
    ("Reprise après arrêt brutal", "results/process-recovery-acceptance-20261021-release/process-recovery-acceptance.json"),
    ("§11.2 — contrôle 001, échec avant soumission", "work/live/evaluation-recovery-functional-control-001-terminal-audit.json"),
    ("§11.2 — contrôle 002, reprise incomplète", "work/live/evaluation-recovery-functional-control-002-terminal-audit.json"),
    ("§11.2 — contrôle 003, reprise statique sans modèle", "work/live/evaluation-recovery-functional-control-003-terminal-audit.json"),
    ("Codex vanilla natif", "results/native-vanilla-20261021/evidence/receipt.json"))

RECOVERY_AUDITS = {"work/live/evaluation-recovery-functional-control-"+number+"-terminal-audit.json": number
                   for number in ("001", "002", "003")}
RECOVERY_ACCEPTANCE_CHECKS = (
    "actual_predeclared_placement_before_measurement", "both_worker_terminal", "independent_oracle_measurement",
    "new_epoch_request_attempt", "new_session_job", "old_canonical_unfinished_preserved", "old_cost_remains_uncertain",
    "old_measurement_before_tell", "old_process_exit_exact", "preparation_submits_no_job", "real_native_both_jobs",
    "replay_no_gpu_or_tell", "same_contract", "same_real_uuid_toolchain", "unique_native_request_ids")
RECOVERY_AUDIT_CHECKS = (
    "fifteen_authority_checks_true", "source_helper_api_pins", "phase1_crash73_after_real_measured_before_canonical_tell",
    "owned_one_gpu_placement_same_uuid_two_distinct_jobs", "both_terminal_completed_zero", "new_epoch_request_attempt",
    "old_uncertain_cost_not_zeroed", "same_contract_and_environment", "copied_input_identity_preparation_commit_sha_links",
    "new_evaluation_id_same_exact_artifact", "all63_public_native_receipts_hash_job_uuid_source_guards_match",
    "replay_no_additional_rpc_or_tell", "no_final_exposure_no_model", "bounded_public_manifest_subset_sha_verified")


def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def fmt(value, digits=3):
    return f"{value:.{digits}g}" if isinstance(value, (int, float)) and math.isfinite(value) else "inconnu"


def comparison_view(pair):
    """Keep a legacy pair intact and derive an identity-aware presentation."""
    pair = _mapping(pair)
    interval = _mapping(pair.get("interval"))
    identity_declared = pair.get("status") == "exact_artifact_identity" or pair.get("gain_basis") == "exact_artifact_identity"
    unproven = (pair.get("status") == "artifact_identity_context_unproven"
                or pair.get("gain_basis") == "unproven_artifact_context"
                or (identity_declared and pair.get("ok") is not True))
    exact = not unproven and _hash(pair.get("artifact_hash")) and pair["artifact_hash"] == pair.get("incumbent_artifact_hash")
    unavailable = not exact and not unproven and not interval
    return {"gain_basis": "unproven_artifact_context" if unproven else "exact_artifact_identity" if exact else "comparison_unavailable" if unavailable else "paired_measurement",
            "gain": None if unproven else 1.0 if exact else interval.get("gain"),
            "lower": None if exact or unproven else interval.get("lower"), "upper": None if exact or unproven else interval.get("upper"),
            "case_gains": {} if exact or unproven else interval.get("case_gains", {}),
            "source_status": pair.get("status"), "source_ok": pair.get("ok"),
            "source_interval_preserved": interval,
            "artifact_hash": pair.get("artifact_hash"), "incumbent_artifact_hash": pair.get("incumbent_artifact_hash")}


def direct_comparison_views(final_groups):
    return [{"group_id": group_id, "comparison": name, **comparison_view(pair)}
            for group_id, group in _mapping(final_groups).items() if isinstance(group, dict)
            for name, pair in _mapping(group.get("method_comparisons")).items()]


def direct_comparison_table(views):
    if not views:
        return ""
    return "<h2>Contrastes directs des groupes finaux</h2>" + table(
        ["Groupe", "Contraste", "Fondement", "Ratio", "IC95%", "Indicateur ok du reçu source"],
        [[entry["group_id"], entry["comparison"],
          {"exact_artifact_identity": "identité exacte des artefacts",
           "unproven_artifact_context": "contexte d’exécution non prouvé",
           "comparison_unavailable": "comparaison indisponible"}.get(entry["gain_basis"], "mesure appariée"),
          fmt(entry["gain"]), "sans intervalle : identité" if entry["gain_basis"] == "exact_artifact_identity" else
          "IC non établi" if entry["gain_basis"] in ("unproven_artifact_context", "comparison_unavailable") else
          "["+fmt(entry["lower"])+" ; "+fmt(entry["upper"])+"]", entry["source_ok"]] for entry in views]) + (
        "<p>Les comparaisons héritées avec empreintes complètes identiques gardent le ratio 1, "
        "sauf en présence d’un refus explicite du contexte d’exécution. Ce refus conserve un ratio inconnu et un IC non établi. "
        "Les intervalles hérités bruités sont conservés dans les données sources et exclus de cette interprétation. "
        "Une identité ne qualifie ni le kernel ni la méthode. Une même source avec un lancement différent reste un autre artefact.</p>")


def table(headers, rows):
    return "<div class='scroll'><table><thead><tr>" + "".join("<th>" + escape(str(h)) + "</th>" for h in headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join("<td>" + escape(str(x)) + "</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def study_aggregate_table(summary):
    from optimizer_agent.suite_report import method_count_cells
    rows = [[LABELS.get(mode, mode), value["planned"], *method_count_cells(value),
             value["final_measured"], value["qualified"], value["regressions"],
             value["unmeasured_or_failed"], fmt(value["geometric_mean_all_measured_gain"]),
             fmt(value["gain_stddev_between_campaigns"])]
            for mode, value in summary["aggregates"].items()]
    return table(["Méthode", "Prévus", "Méthodes réussies", "Méthodes échouées",
                  "Statuts méthode inconnus", "Ratios disponibles (mesures/identités)", "Artefacts qualifiés (mesures/identités)",
                  "Régressions : ratios ponctuels < 1", "Absents/échecs sans mesure", "Ratio géométrique (mesures/identités)",
                  "Écart-type descriptif toutes familles"], rows)


def workload_dispersion_table(summary):
    rows = []
    for mode, aggregate in summary.get("aggregates", {}).items():
        for workload, value in _mapping(aggregate.get("per_workload")).items():
            seeds = value.get("seeds") if isinstance(value.get("seeds"), list) else []
            observations = "; ".join(str(item.get("seed")) + ": " + fmt(item.get("gain")) +
                " (" + str(item.get("status", "inconnu")) + ")" for item in seeds if isinstance(item, dict))
            rows.append([LABELS.get(mode, mode), FAMILIES.get(workload, workload), value.get("planned", "inconnu"),
                value.get("measured", "inconnu"), value.get("unmeasured_or_failed", "inconnu"),
                fmt(value.get("geometric_mean_gain")), fmt(value.get("gain_stddev_between_independent_seeds")),
                observations or "non enregistrées"])
    return ("<h3>Dispersion par famille entre graines indépendantes</h3><p>Les valeurs sont celles des agrégats enregistrés, "
            "séparées par famille et méthode, sans regrouper des opérations différentes. Moins de deux ratios disponibles "
            "ne permettent pas un écart-type entre graines ; une absence reste inconnue. Les ratios peuvent inclure des identités déterministes.</p>" +
            table(["Méthode", "Famille", "Prévus", "Ratios disponibles", "Absents/échecs sans ratio", "Ratio géométrique",
                   "Écart-type entre graines", "Graines : ratio et statut"], rows))


def study_campaign_table(summary):
    from optimizer_agent.suite_report import method_outcome_cells
    rows = [[FAMILIES.get(row["workload"], row["workload"]), row["seed"],
             LABELS.get(row["mode"], row["mode"]), *method_outcome_cells(row), row["status"],
             {"exact_baseline_artifact": "identité exacte avec le naïf", "exact_artifact_identity": "identité exacte d’artefact",
              "paired_measurement": "mesure appariée", "unproven_artifact_context": "contexte d’artefact non prouvé"}.get(row.get("gain_basis"), "non renseignée"),
             fmt(1.0 if row.get("gain_basis") in ("exact_artifact_identity", "exact_baseline_artifact") else row.get("gain")), "identité exacte, sans IC" if row.get("gain_basis") in ("exact_artifact_identity", "exact_baseline_artifact") else
             "["+fmt(row.get("lower"))+" ; "+fmt(row.get("upper"))+"]",
             fmt(row.get("model_tokens"), 9), fmt(row.get("model_calls")), fmt(row.get("wall_seconds")),
             row.get("artifact_hash") or "absent"] for row in summary["rows"]]
    return table(["Famille", "Graine", "Méthode", "Statut méthode", "Défaillance méthode",
                  "Candidat de repli", "Statut final de l’artefact", "Origine du ratio", "Ratio final", "IC95%",
                  "Tokens bruts méthode", "Compteur brut d’appels méthode", "Durée génération (s)", "Artefact"], rows)


def campaign_call_observations(summary, consumption, study):
    """Attribute the existing deduplicated inventory once; never reopen receipts."""
    campaign_root = Path(study).resolve() / "campaigns"
    rows, by_run = [], {}
    for source in summary["rows"]:
        run = source.get("run_id")
        row = {"run_id": run, "mode": source.get("mode"), "ledger_model_calls": source.get("model_calls"),
               "ledger_model_tokens": source.get("model_tokens"), "calls": [], "ambiguous_identities": []}
        rows.append(row)
        if isinstance(run, str) and run and run not in (".", "..") and "/" not in run:
            if run in by_run:
                raise ValueError("Duplicate campaign run_id in summary: " + run)
            by_run[run] = row
    inventory = consumption.get("call_provenance")
    available = isinstance(inventory, list)
    ambiguous, unattributed = [], []
    def owner(origin, kind):
        raw = origin.get("path")
        if not isinstance(raw, str) or not Path(raw).is_absolute() or str(Path(raw)) != raw or ".." in raw.split("/"):
            return None
        try:
            parts = Path(raw).relative_to(campaign_root).parts
        except ValueError:
            return None
        if len(parts) < 2 or not _hash(origin.get("sha256")):
            return None
        suffix = "/".join(parts[1:])
        prefix = r"(?:attempts/[0-9]+/)?generation/"
        layout = (r"(?:model/[0-9a-f]{32}\.json|model-transport/intents/[^/]+/[0-9]+/cli/[0-9a-f]{32}\.json)"
                  if kind == "structured" else r"evidence/receipt\.json" if kind == "native" else r"(?!)")
        return parts[0] if re.fullmatch(prefix + layout, suffix) else None
    for call in inventory if available else []:
        call = _mapping(call)
        raw_origins = call.get("receipts")
        origins = [origin for origin in raw_origins if isinstance(origin, dict)] if isinstance(raw_origins, list) else []
        canonical, other_origins = [], []
        for origin in origins:
            run = owner(origin, call.get("kind"))
            (canonical if run is not None else other_origins).append({**origin, "run_id": run} if run is not None else origin)
        owners = sorted({origin["run_id"] for origin in canonical})
        categories = set()
        for origin in origins:
            status, code = origin.get("status"), origin.get("returncode")
            failed = status in ("failed", "timeout", "cancelled", "canceled") or (type(code) is int and code != 0)
            categories.add("mixed" if status == "completed" and failed else "failed" if failed else
                           "completed" if status == "completed" else "incomplete" if status in ("prepared", "pending", "running", "started", "interrupted") else "unknown")
        category = next(iter(categories)) if len(categories) == 1 else "mixed" if categories else "unknown"
        entry = {"identity": call.get("identity"), "aliases": call.get("aliases", []), "kind": call.get("kind"),
                 "canonical_campaigns": owners, "canonical_receipts": canonical,
                 "other_receipt_origins": other_origins,
                 "statuses_observed": sorted({str(origin.get("status")) for origin in origins}),
                 "status_category": category, "usage_reporting": call.get("usage_reporting"),
                 "observed_usage": _mapping(call.get("observed_usage")),
                 "missing_usage_fields": call.get("missing_usage_fields", [])}
        if len(owners) > 1:
            ambiguous.append(entry)
            for run in owners:
                if run in by_run:
                    by_run[run]["ambiguous_identities"].append(entry["identity"])
        elif len(owners) == 1 and owners[0] in by_run:
            by_run[owners[0]]["calls"].append(entry)
        else:
            unattributed.append(entry)
    for row in rows:
        calls = row["calls"]
        row["observed_unique_cli_receipts"] = len(calls) if available else None
        row["status_counts"] = {status: sum(call["status_category"] == status for call in calls)
                                for status in ("completed", "failed", "incomplete", "mixed", "unknown")}
        row["usage_reporting_counts"] = {status: sum(call["usage_reporting"] == status for call in calls)
                                         for status in ("complete", "partial", "unreported")}
        row["usage_incomplete_calls"] = sum(call["usage_reporting"] != "complete" for call in calls) if available else None
        row["provider_usage_complete_for_observed_calls"] = all(call["usage_reporting"] == "complete" for call in calls) if calls else None
        known = {}
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            values = [call["observed_usage"][key] for call in calls if type(call["observed_usage"].get(key)) is int and call["observed_usage"][key] >= 0]
            known[key] = sum(values) if values else None
        known["model_tokens"] = sum(known[key] or 0 for key in ("input_tokens", "output_tokens")) if any(
            known[key] is not None for key in ("input_tokens", "output_tokens")) else None
        row["known_usage_lower_bounds"] = known
        row["calls_without_reported_cached_usage"] = sum("cached_input_tokens" not in call["observed_usage"] for call in calls) if available else None
        row["attribution_status"] = ("inventory_unavailable" if not available else "attributed_with_ambiguous_exclusions" if row["ambiguous_identities"] else
                                     "attributed" if calls else "no_retained_canonical_receipt")
    return {"campaigns": rows, "ambiguous_shared_identities": ambiguous, "unattributed_identities": unattributed,
            "inventory_available": available, "canonical_campaign_root": str(campaign_root),
            "scope": "Unique groups from the already-deduplicated retained inventory, attributed only by absolute canonical campaign receipt layouts, including attempts. Shared campaign identities are excluded from every campaign count. Outside archive copies do not establish ownership. No HTTP-cycle or exact launched-process count; missing receipts/usage do not establish zero cost."}


def campaign_call_panel(observations):
    rows = []
    for row in observations["campaigns"]:
        statuses, usage = row["status_counts"], row["usage_reporting_counts"]
        rows.append([str(row["run_id"]) + " · " + str(LABELS.get(row["mode"], row["mode"])),
            fmt(row["ledger_model_calls"], 9), fmt(row["observed_unique_cli_receipts"], 9),
            " / ".join(str(statuses[key]) for key in ("completed", "failed", "incomplete", "mixed", "unknown")) if observations["inventory_available"] else "inconnu",
            " / ".join(str(usage[key]) for key in ("complete", "partial", "unreported")) if observations["inventory_available"] else "inconnu",
            fmt(row["known_usage_lower_bounds"]["model_tokens"], 9), len(row["ambiguous_identities"]), row["attribution_status"]])
    return ("<details><summary>Reçus CLI attribués aux campagnes</summary>" + table(
        ["Campagne", "Compteur brut méthode", "Reçus uniques / tentatives documentées", "Statuts terminé / échec / incomplet / mixte / inconnu",
         "Usage entrée-sortie complet / partiel / absent", "Tokens connus : borne inférieure", "Identités ambiguës exclues", "Attribution"], rows) +
        "<p>Le compteur brut de la méthode peut ne compter que les appels achevés ; il n’est pas remplacé par le nombre de reçus. "
        "Les reçus sont regroupés une seule fois par identité, y compris leurs copies et aliases. Une identité présente sous plusieurs campagnes reste ambiguë, sans double comptage. "
        "Les statuts mixtes conservent les snapshots divergents ; les catégories ne se chevauchent pas. L’usage complet concerne les entrées/sorties déclarées des seuls groupes observés, pas un inventaire exhaustif ni le cache toujours connu. "
        "Le JSON conserve chemins, empreintes, statuts et champs d’usage manquants. Les tokens connus sont une borne inférieure ; cache et raisonnement ne sont pas ajoutés une seconde fois. "
        "Une identité prepared atteste une tentative préparée, sans prouver le lancement du CLI. "
        "Zéro reçu observé ne prouve ni zéro appel exécuté ni zéro coût. Aucun nombre de cycles HTTP ou de processus lancés sans PID attesté n’est déduit. "
        + str(len(observations["ambiguous_shared_identities"])) + " identités ambiguës et " + str(len(observations["unattributed_identities"])) +
        " identités extérieures/non attribuables restent visibles dans le JSON ; leurs coûts ne sont pas répartis entre campagnes.</p></details>")


def _research_context_identity(path, campaign_root=None):
    """Recognize a public phase receipt layout without following its references."""
    raw = str(path)
    if not Path(raw).is_absolute() or str(Path(raw)) != raw or ".." in raw.split("/"):
        return None
    if campaign_root is not None:
        try:
            raw = "/" + str(Path(raw).relative_to(campaign_root))
        except ValueError:
            return None
        prefix = r"/"
    else:
        prefix = r".*/campaigns/"
    match = re.fullmatch(prefix + r"([^/]+)/(?:attempts/[0-9]+/)?generation/research/"
                         r"([A-Za-z0-9_-]+)-(search|synthesis)-context-(intent|result)\.json", raw)
    return dict(zip(("run_id", "research_id", "phase", "role"), match.groups())) if match else None


def _research_phase_receipts(results):
    """One bounded public index; never open a path supplied inside a receipt."""
    root, records = Path(results).resolve(), []
    paths = set(root.rglob("generation/research/*-context-intent.json")) | set(root.rglob("generation/research/*-context-result.json"))
    for path in sorted(paths):
        identity = _research_context_identity(path)
        if not identity or any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
            continue
        try:
            body = path.read_bytes()
        except OSError:
            records.append({"path": str(path), **identity, "valid": False, "error": "unreadable_public_phase_receipt"})
            continue
        record = {"path": str(path), "sha256": hashlib.sha256(body).hexdigest(), "layout": identity}
        try:
            value = json.loads(body)
        except (ValueError, UnicodeError):
            value = None
        value = _mapping(value)
        # No public_context, supplied corpus, prompt, or linked file is copied/read.
        for key in ("schema_version", "kind", "status", "research_id", "phase", "call_id", "returned_payload_sha256"):
            record[key] = value.get(key)
        record["intent"] = {key: _mapping(value.get("intent")).get(key) for key in ("path", "sha256")}
        record["valid"] = (type(value.get("schema_version")) is int and value["schema_version"] == 1 and value.get("research_id") == identity["research_id"]
            and value.get("phase") == identity["phase"] and value.get("kind") == "research_prompt_context_" + identity["role"]
            and value.get("status") == ("intent" if identity["role"] == "intent" else "completed"))
        records.append(record)
    return records


def campaign_research_model_consumption(call_observations, consumption, study):
    """Pure phase projection of the existing CLI inventory, including refusals."""
    root = Path(study).resolve() / "campaigns"
    contexts = consumption.get("research_phase_receipts")
    available = isinstance(contexts, list)
    contexts = [_mapping(value) for value in contexts] if available else []
    metadata = {call.get("identity"): call for call in _records(consumption.get("call_provenance"))}
    indexed = {(record.get("path"), record.get("sha256")): record for record in contexts if _hash(record.get("sha256"))}
    rows, outside, by_run = [], [], {}
    known_runs = {row["run_id"] for row in call_observations["campaigns"]}
    for record in contexts:
        owner = _research_context_identity(record.get("path", ""), root)
        if owner and owner["run_id"] in known_runs:
            by_run.setdefault(owner["run_id"], []).append(record)
        else:
            outside.append(record)
    for source in call_observations["campaigns"]:
        run, calls = source["run_id"], {call["identity"]: call for call in source["calls"]}
        records = by_run.get(run, [])
        intents, results, failures, bindings = {}, {}, [], {}
        for record in records:
            layout = _research_context_identity(record["path"], root)
            key = (layout["research_id"], layout["phase"], record.get("sha256"))
            if (not record.get("valid") or record.get("research_id") != layout["research_id"]
                    or record.get("phase") != layout["phase"]):
                failures.append({"receipt": record, "reason": "invalid_phase_receipt"})
            elif layout["role"] == "intent":
                intents.setdefault(key, []).append(record)
            else:
                results.setdefault(key, []).append(record)
        resolved_intents = set()
        for copies in results.values():
            for record in copies:
                intent_ref = _mapping(record.get("intent"))
                intent = indexed.get((intent_ref.get("path"), intent_ref.get("sha256"))) if isinstance(intent_ref.get("path"), str) and _hash(intent_ref.get("sha256")) else None
                owner = _research_context_identity(intent_ref.get("path", ""), root)
                valid_intent = (intent and intent.get("valid") and owner and owner["role"] == "intent" and owner["run_id"] == run
                    and intent.get("research_id") == record["research_id"] and intent.get("phase") == record["phase"])
                if not valid_intent:
                    failures.append({"receipt": record, "reason": "intent_link_unverified"})
                    continue
                resolved_intents.add((intent["research_id"], intent["phase"], intent["sha256"]))
                call_id = record.get("call_id")
                identity = "structured:" + call_id if isinstance(call_id, str) and re.fullmatch(r"[0-9a-f]{32}", call_id) else None
                call = calls.get(identity)
                cli = _records(_mapping(metadata.get(identity)).get("cli_metadata"))
                canonical = {origin["path"] for origin in call["canonical_receipts"]} if call else set()
                payloads = {item.get("payload_sha256") for item in cli if item.get("path") in canonical
                    and item.get("call_id") == call_id and Path(item["path"]).stem == call_id and _hash(item.get("payload_sha256"))}
                if (not call or call.get("kind") != "structured" or not _hash(record.get("returned_payload_sha256"))
                        or payloads != {record["returned_payload_sha256"]}):
                    failures.append({"receipt": record, "reason": "CLI_identity_owner_or_payload_unverified"})
                    continue
                bindings.setdefault(identity, []).append({"phase": record["phase"], "research_id": record["research_id"],
                    "result": {key: record[key] for key in ("path", "sha256")}, "intent": intent_ref})
        ambiguous = sorted(identity for identity, links in bindings.items() if len({link["phase"] for link in links}) != 1)
        phase_calls = {phase: [] for phase in ("search", "synthesis")}
        for identity, links in bindings.items():
            if identity not in ambiguous:
                phase_calls[links[0]["phase"]].append({**calls[identity], "phase_evidence": links})
        attributed = {call["identity"] for group in phase_calls.values() for call in group}
        def aggregate(group):
            known, durations, duration_issues = {}, [], []
            def valid_duration(value):
                try:
                    return type(value) in (int, float) and value >= 0 and math.isfinite(value)
                except OverflowError:
                    return False
            for field in ("input_tokens", "cached_input_tokens", "output_tokens"):
                values = [call["observed_usage"][field] for call in group if type(call["observed_usage"].get(field)) is int and call["observed_usage"][field] >= 0]
                known[field] = sum(values) if values else None
            known["model_tokens"] = sum(known[field] or 0 for field in ("input_tokens", "output_tokens")) if any(known[field] is not None for field in ("input_tokens", "output_tokens")) else None
            for call in group:
                observations = _records(_mapping(metadata.get(call["identity"])).get("cli_metadata"))
                values = {item["elapsed_seconds"] for item in observations if valid_duration(item.get("elapsed_seconds"))}
                invalid = any(item.get("elapsed_seconds") is not None and not valid_duration(item["elapsed_seconds"]) for item in observations)
                if len(values) == 1 and not invalid:
                    durations.append(next(iter(values)))
                else:
                    duration_issues.append({"identity": call["identity"], "reason": "invalid" if invalid else "conflicting" if values else "unreported", "observations": observations})
            return {"observed_unique_cli_calls": len(group), "known_usage_lower_bounds": known,
                "status_counts": {status: sum(call["status_category"] == status for call in group) for status in ("completed", "failed", "incomplete", "mixed", "unknown")},
                "usage_reporting_counts": {status: sum(call["usage_reporting"] == status for call in group) for status in ("complete", "partial", "unreported")},
                "sum_known_CLI_elapsed_seconds": sum(durations) if durations else None,
                "duration_estimator": "CLI receipt elapsed_seconds only; outer model-event duration not combined",
                "CLI_duration_complete_for_observed_calls": not duration_issues if group else None,
                "CLI_duration_unknown_calls": duration_issues, "calls": group}
        phases = {phase: aggregate(group) for phase, group in phase_calls.items()}
        rows.append({"run_id": run, "mode": source["mode"], "phases": phases,
            "attributed_total": aggregate([call for group in phase_calls.values() for call in group]),
            "outside_identified_phases": aggregate([call for identity, call in calls.items() if identity not in attributed]),
            "intent_counts": {phase: sum(key[1] == phase for key in intents) for phase in phase_calls},
            "intents_without_result": [copies[0] for key, copies in intents.items() if key not in resolved_intents],
            "unverified_results_or_receipts": failures, "ambiguous_phase_identities": ambiguous,
            "ambiguous_campaign_identities": source["ambiguous_identities"],
            "phase_inventory_available": available, "documentary_end_to_end_seconds": None, "cost_usd": None})
    return {"campaigns": rows, "unattributed_context_receipts": outside, "phase_inventory_available": available,
        "scope": "Explicit ResearchService search/synthesis links only, including calls whose dossier is later refused. No admission filter, timestamp/purpose inference, extra dossier usage, provider-internal cycles or exhaustive inventory claim. Intent-only records do not attest an executed call. Other/unresolved CLI consumption stays visible separately; documentary end-to-end latency and USD are unknown."}


def campaign_research_model_panel(view):
    html = "<details><summary>Coûts modèles du service de recherche : recherche et synthèse</summary>"
    for row in view["campaigns"]:
        values = []
        for label, group in [("Recherche", row["phases"]["search"]), ("Synthèse", row["phases"]["synthesis"]),
                             ("Total explicitement attribué", row["attributed_total"]),
                             ("CLI hors phases identifiées / phase inconnue", row["outside_identified_phases"])]:
            values.append([label, group["observed_unique_cli_calls"],
                " / ".join(str(group["status_counts"][status]) for status in ("completed", "failed", "incomplete", "mixed", "unknown")),
                " / ".join(str(group["usage_reporting_counts"][status]) for status in ("complete", "partial", "unreported")),
                fmt(group["known_usage_lower_bounds"]["model_tokens"], 9), fmt(group["sum_known_CLI_elapsed_seconds"], 9),
                len(group["CLI_duration_unknown_calls"])])
        html += "<details><summary>" + escape(str(row["run_id"])) + "</summary>" + table(
            ["Phase", "Identités CLI", "Terminé / échec / incomplet / mixte / inconnu", "Usage complet / partiel / absent",
             "Tokens connus : borne inférieure", "Somme des durées CLI connues (s)", "Durées inconnues ou divergentes"], values)
        html += ("<p>Intents recherche / synthèse observés : " + str(row["intent_counts"]["search"]) + " / " + str(row["intent_counts"]["synthesis"]) +
                 " ; sans résultat lié : " + str(len(row["intents_without_result"])) + " ; liens non vérifiés : " + str(len(row["unverified_results_or_receipts"])) +
                 " ; identités de phase ambiguës : " + str(len(row["ambiguous_phase_identities"])) + ". " +
                 "Identités de campagne ambiguës exclues : " + str(len(row["ambiguous_campaign_identities"])) + ". " +
                 ("Index de phases disponible." if row["phase_inventory_available"] else "Index de phases indisponible : attribution inconnue.") + "</p></details>")
    return html + ("<p>Ces coûts comprennent les appels explicitement liés même si le dossier est ensuite refusé. Une intention seule ne prouve pas un appel exécuté ; les échecs sans résultat lié restent hors phases identifiées. "
        "Les copies ne multiplient pas les coûts et aucune phase n’est inférée pour vanilla/original. Zéro lien ne prouve pas zéro consommation. "
        "La somme des durées CLI connues est distincte du temps modèle extérieur des événements et du temps documentaire de bout en bout, qui reste inconnu. "
        "Les durées divergentes sont exclues de cette somme et conservées dans le JSON. Les tokens sont des bornes inférieures ; le cache est déjà inclus, sans addition de ResearchDossier.usage. "
        "USD inconnus ; aucun compteur historique, override, résultat ou classement n’est réécrit.</p></details>")


def campaign_metric_views(summary):
    """Project recorded, already-merged metrics without recalculating any cost."""
    quality = (
        ("candidate_configurations_with_recorded_validity", "Configurations à validité enregistrée", "nombre"),
        ("correct_candidates", "Configurations correctes enregistrées", "nombre"),
        ("correct_candidate_fraction", "Part correcte enregistrée", "fraction de 0 à 1"),
        ("configuration_observations", "Observations de configurations", "nombre"),
        ("configurations_filtered_before_compilation", "Configurations filtrées avant compilation", "nombre"),
        ("configurations_independently_validated", "Validations indépendantes enregistrées (périmètre méthode)", "nombre"))
    costs = (
        ("native_compile_stage_wall_seconds", "Compilation native", "s", "Temps des étapes de compilation enregistré par le harness ; exclu des latences kernel."),
        ("compile_cache_hits", "Cache : réutilisations", "nombre", "Compteur de cache enregistré ; pas un nombre de candidats distincts."),
        ("compile_cache_misses", "Cache : absences", "nombre", "Compteur de cache enregistré ; distinct des configurations explorées."),
        ("research_model_wall_seconds", "Recherche : temps modèle", "s", "Appels de recherche identifiés ; une synthèse classée proposal peut être exclue. Pas le temps GPU du kernel."),
        ("research_model_tokens", "Recherche : tokens modèle", "tokens", "Appels de recherche identifiés ; pas nécessairement tout le coût du dossier ni les cycles internes du fournisseur."),
        ("research_events", "Recherche : événements", "nombre", "Événements enregistrés ; ne compte ni les requêtes web ni les dossiers admis."),
        ("native_reservation_wall_seconds", "Réservations du harness", "s", "Temps des requêtes natives réservé enregistré ; distinct de la durée totale Slurm."),
        ("cuda_event_kernel_seconds", "Kernels : CUDA Events cumulés", "s", "Temps kernel cumulé enregistré ; aucune nouvelle latence de classement calculée."),
        ("observer_wall_seconds", "Observateur indépendant", "s", "Coût enregistré de l’observation inline ; aucun résultat transmis à la méthode."),
        ("observer_wall_seconds_known_lower_bound", "Observateur : borne inférieure connue", "s", "Coût partiel connu enregistré ; ne remplace pas un total observateur inconnu."),
        ("observer_gpu_requests", "Observateur : requêtes GPU", "nombre", "Requêtes GPU de l’observateur enregistrées séparément de la méthode."),
        ("combined_method_and_observer_gpu_wall_seconds", "Méthode et observateur : total enregistré", "s", "Somme déjà présente dans le reçu ; aucune somme recalculée et aucune durée Slurm déduite."))
    views = []
    for row in summary["rows"]:
        metrics = _mapping(row.get("metrics"))
        method = _mapping(metrics.get("method_consumption"))
        recorded = [{"key": key, "label": label, "value": metrics.get(key), "unit": unit, "scope": scope}
                    for key, label, unit, scope in costs]
        recorded += [{"key": "method_consumption." + key, "label": label, "value": method.get(key),
                      "unit": unit, "scope": "Consommation méthode au dernier callback durable de l’observateur ; peut être partielle, séparée de son propre coût."}
                     for key, label, unit in (
                         ("method_gpu_evaluation_wall_seconds", "Méthode : évaluations GPU", "s"),
                         ("method_harness_calls", "Méthode : appels harness", "nombre"),
                         ("method_gpu_evaluation_calls", "Méthode : appels évaluation GPU", "nombre"))]
        recorded.append({"key": "gpu_reservation_wall_seconds", "label": "Ledger des essais : temps écoulé",
                         "value": row.get("gpu_reservation_wall_seconds"), "unit": "s",
                         "scope": metrics.get("gpu_seconds_ledger_scope") or
                         "Ledger incluant modèle, recherche, SSH et parfois interruptions ; ce n’est pas une durée GPU allouée Slurm."})
        raw_unknown = metrics.get("unknown_metrics")
        annotations = []
        for declared in raw_unknown if isinstance(raw_unknown, list) else []:
            key = "research_model_wall_seconds" if declared == "research_seconds" else declared
            if isinstance(key, str) and metrics.get(key) is not None:
                annotations.append({"declared_unknown": declared, "recorded_key": key,
                                    "recorded_value": metrics[key]})
        views.append({"run_id": row.get("run_id"), "workload": row.get("workload"),
            "seed": row.get("seed"), "mode": row.get("mode"),
            "generation_method_status": row.get("generation_method_status"),
            "first_useful_search_gain": {"status": metrics.get("first_useful_gain_status"),
                "seconds": metrics.get("time_to_first_useful_gain_seconds"),
                "model_tokens": metrics.get("model_tokens_to_first_useful_gain"),
                "model_calls": metrics.get("model_calls_to_first_useful_gain"),
                "scope": metrics.get("first_useful_gain_scope"), "note": metrics.get("first_useful_gain_note")},
            "independent_final_gain": {"status": row.get("status"),
                "seconds": row.get("time_to_independently_confirmed_final_gain_seconds"),
                "scope": "Délai depuis le début de la tâche de génération jusqu’au gain FINAL indépendamment confirmé, attente et interruptions comprises ; pas la seule durée de mesure finale."},
            "final_closure": row.get("final_closure"),
            "recorded_quality": [{"key": key, "label": label, "value": metrics.get(key), "unit": unit}
                                 for key, label, unit in quality],
            "validation_evidence_basis": metrics.get("validation_evidence_basis"),
            "method_callback_usage_complete": method.get("usage_complete"),
            "recorded_costs": recorded, "raw_unknown_metrics": raw_unknown,
            "view_annotations": annotations})
    return views


def campaign_metrics_panel(views):
    statuses = {"observed": "gain utile observé", "no_promoted_gain": "aucune promotion enregistrée",
                "no_confirmed_gain": "aucun gain confirmé enregistré", "not_recorded_by_method": "non enregistré par la méthode",
                "unknown_incident": "inconnu après incident", "unknown_identity_context": "contexte d’identité inconnu"}
    def number(value):
        return fmt(value, 9) if isinstance(value, (int, float)) and not isinstance(value, bool) else "inconnu"
    blocks, shared_groups = [], {}
    for view in views:
        search, final = view["first_useful_search_gain"], view["independent_final_gain"]
        status = search["status"]
        label = (statuses.get(status, "statut non traduit") + " (" + status + ")"
                 if isinstance(status, str) and status else "inconnu")
        milestones = table(["Jalon", "Statut enregistré", "Délai (s)", "Tokens bruts au jalon", "Compteur méthode au jalon"], [
            ["Premier gain utile SEARCH", label, number(search["seconds"]), number(search["model_tokens"]), number(search["model_calls"])],
            ["Gain FINAL indépendant", final["status"] or "inconnu", number(final["seconds"]), "non enregistrés", "non enregistrées"]])
        closure = _mapping(view.get("final_closure"))
        closed = closure.get("status") == "closed"
        closure_usage = _mapping(closure.get("reported_generation_usage")) if closed else {}
        closure_table = table(["Clôture de l’évaluation FINAL", "Valeur", "Périmètre"], [
            ["Statut / résultat", ("close / " + str(closure.get("outcome", "inconnu"))) if closed else "inconnu / non enregistré",
             "Clôture distincte du gain, de la qualification numérique et de la fin du worker."],
            ["Délai depuis le début de génération (s)", number(closure.get("seconds_since_generation_task_start") if closed else None),
             "Comprend attente et interruptions avant cette clôture ; aucun ancien timestamp n’est reconstruit."],
            ["Durée du groupe FINAL (s)", number(closure.get("final_group_wall_seconds") if closed else None),
             "Durée du groupe partagé, répétée comme contexte ; ne pas additionner les campagnes."],
            ["Tokens génération enregistrés à la clôture", number(closure_usage.get("model_tokens")),
             "Snapshot brut du ledger de génération, potentiellement incomplet ; pas les tokens consommés pendant FINAL."],
            ["Compteur méthode génération à la clôture", number(closure_usage.get("model_calls")),
             "Compteur brut achevé, distinct des identités CLI observées."],
            ["Tokens / appels modèles pendant FINAL", number(closure.get("final_model_tokens") if closed else None) + " / " + number(closure.get("final_model_calls") if closed else None),
             "Aucune valeur supposée nulle, aucune interpolation depuis les totaux."],
            ["Coût facturé USD", number(closure.get("cost_usd") if closed else None), "Inconnu en l’absence de reçu de facturation."]])
        candidate_cost = _mapping(closure.get("candidate_request_cost")) if closed else {}
        request_rows = [[label, number(candidate_cost.get(key)), unit] for key, label, unit in (
            ("observed_rpc_calls", "RPC observés du candidat", "nombre"),
            ("unique_request_ids", "Identifiants de requêtes uniques du candidat", "nombre"),
            ("rpc_wall_seconds", "Durées RPC cumulées du candidat", "s"),
            ("reported_native_reservation_seconds", "Durées natives rapportées : sous-total connu", "s"),
            ("reported_compile_seconds", "Compilation rapportée : sous-total connu", "s"))]
        request_rows += [[label, len(candidate_cost[key]) if isinstance(candidate_cost.get(key), list) else "inconnu", "nombre"]
                        for key, label in (("native_duration_missing_request_ids", "IDs avec durée native manquante"),
                                           ("native_duration_conflict_request_ids", "IDs avec durée native contradictoire"))]
        request_cost_table = table(["Requêtes FINAL propres au candidat", "Valeur", "Unité"], request_rows)
        evidence = _mapping(closure.get("cost_evidence"))
        if closed and closure.get("group_id") is not None and evidence.get("sha256"):
            key = (str(closure["group_id"]), str(evidence["sha256"]))
            shared_cost = _mapping(closure.get("shared_group_request_cost"))
            if key in shared_groups and shared_groups[key] != shared_cost:
                shared_groups[key] = {}  # Contradictory copies do not select a preferred cost.
            else:
                shared_groups.setdefault(key, shared_cost)
        costs = table(["Coût enregistré", "Valeur", "Unité", "Périmètre"],
            [[cost["label"], number(cost["value"]), cost["unit"], cost["scope"]] for cost in view["recorded_costs"]])
        quality = table(["Validité / configurations enregistrées", "Valeur", "Unité"],
            [[entry["label"], number(entry["value"]), entry["unit"]] for entry in view["recorded_quality"]])
        basis = view["validation_evidence_basis"]
        callback_complete = view["method_callback_usage_complete"]
        completeness = "oui" if callback_complete is True else "non" if callback_complete is False else "inconnu"
        scope = " ".join(str(value) for value in (search.get("scope"), search.get("note")) if value)
        unknown = "absente" if view["raw_unknown_metrics"] is None else json.dumps(view["raw_unknown_metrics"], ensure_ascii=False)
        annotations = " ".join(str(note["declared_unknown"]) + " déclaré inconnu, mais " + note["recorded_key"] +
            " contient une valeur enregistrée : " + number(note["recorded_value"]) + "." for note in view["view_annotations"])
        name = (str(FAMILIES.get(view["workload"], view["workload"])) + " · graine " + str(view["seed"]) + " · " +
                str(LABELS.get(view["mode"], view["mode"])) + " · méthode " + str(view["generation_method_status"] or "inconnue"))
        blocks.append("<details><summary>" + escape(name) + "</summary><p>Campagne : <code>" +
            escape(str(view["run_id"] or "non enregistré")) + "</code>.</p>" + milestones +
            "<p>SEARCH : " + escape(scope or "périmètre détaillé non enregistré") + "</p><p>FINAL : " + escape(final["scope"]) +
            "</p>" + closure_table + request_cost_table +
            "<p>Reçus de coût FINAL : " + escape(str(evidence) if evidence else "non enregistrés") +
            ". Temps RPC = transport et attente compris ; durées natives = sous-totaux rapportés dédupliqués par identifiant de requête, "
            "jamais une durée Slurm. Valeurs natives manquantes ou contradictoires restent inconnues. "
            "La confirmation propre au candidat comprend les deux côtés de sa comparaison au naïf. "
            "Les autres coûts partagés du naïf et des comparaisons directes sont présentés séparément, sans répartition.</p>" + quality + "<p>Fondement de validation enregistré : " +
            escape(str(basis) if basis is not None else "non enregistré") +
            ". Ces validités SEARCH ne qualifient pas automatiquement le dernier artefact exporté ni FINAL. "
            "La fraction et ses compteurs sont repris tels quels, sans recalcul ni dénominateur supposé. Les compteurs d’observations de configurations et de validations indépendantes dépendent de l’instrumentation de la méthode : un zéro ne signifie pas zéro candidat évalué. Les validités et leurs preuves restent distinctes.</p>" + costs +
            "<p>Usage complet indiqué au dernier callback méthode de l’observateur : " + completeness +
            " ; ce callback peut précéder la fin de génération.</p><p>Liste historique <code>unknown_metrics</code> inchangée : " + escape(unknown) +
            ".</p>" + ("<p>Annotation de vue seulement : " + escape(annotations) + "</p>" if annotations else "") + "</details>")
    shared_rows = [[group_id, fingerprint, *[number(cost.get(key)) for key in (
        "observed_rpc_calls", "unique_request_ids", "rpc_wall_seconds", "reported_native_reservation_seconds", "reported_compile_seconds")],
        *[len(cost[key]) if isinstance(cost.get(key), list) else "inconnu" for key in (
            "native_duration_missing_request_ids", "native_duration_conflict_request_ids")]]
        for (group_id, fingerprint), cost in shared_groups.items()]
    shared_panel = ("<h3>Coûts FINAL partagés, une fois par groupe et reçu</h3><p>Qualification/mesure du naïf et comparaisons directes "
        "restent partagées. Ces valeurs ne sont ni réparties entre méthodes ni ajoutées à leurs sous-totaux propres.</p>" +
        table(["Groupe", "SHA du reçu", "RPC observés", "IDs uniques", "Durées RPC (s)", "Durées natives connues (s)", "Compilation connue (s)",
               "IDs durée manquante", "IDs durée contradictoire"], shared_rows)) if shared_rows else ""
    return ("<h3>Jalons et coûts enregistrés par campagne</h3><p>Les valeurs viennent des métriques déjà enregistrées et fusionnées par méthode. "
            "Le premier gain utile SEARCH reste distinct de la certification FINAL ; une première mesure valide ne suffit pas à le démontrer. "
            "L’observation inline peut augmenter le délai du premier gain. Les tokens et appels à ce jalon restent inconnus s’ils ne sont pas enregistrés, "
            "sans interpolation depuis les totaux. Ces coûts ne constituent pas une comparaison à budget égal. "
            "Les tokens enregistrés peuvent être des sous-totaux incomplets lorsqu’un appel échoué n’a pas retourné d’usage. "
            "Aucun coût absent ne devient zéro ; aucun coût facturé en USD n’est déduit.</p>" + "".join(blocks) + shared_panel)


def case_comparison_table(summary, direct_views):
    """Expose case intervals without inventing samples for identities or absences."""
    rows = []
    comparisons = [(row.get("group_id"), LABELS.get(row["mode"], row["mode"]) + " / naïf", row)
                   for row in summary["rows"]]
    comparisons += [(entry["group_id"], entry["comparison"], entry) for entry in direct_views]
    for group, contrast, value in comparisons:
        basis = value.get("gain_basis")
        if "final_qualified" in value:
            qualified = value["final_qualified"]
            qualification = "oui" if qualified is True else "non" if qualified is False else "inconnue"
            status = str(value.get("status") or "inconnu") + "; artefact qualifié : " + qualification
        else:
            source_ok = value.get("source_ok")
            ok_label = "vrai" if source_ok is True else "faux" if source_ok is False else "inconnu"
            status = str(value.get("source_status") or "inconnu") + "; indicateur ok du reçu : " + ok_label
        prefix = [group, contrast, status]
        if basis in ("exact_baseline_artifact", "exact_artifact_identity"):
            rows.append(prefix + ["ensemble des cas", "1", "sans intervalle : identité", "identité exacte"])
            continue
        cases = _mapping(value.get("case_gains")) if basis == "paired_measurement" else {}
        if not cases:
            reason = "contexte d’exécution non prouvé" if basis == "unproven_artifact_context" else "mesure par cas indisponible"
            rows.append(prefix + ["non établi", "inconnu", "IC non établi", reason])
            continue
        for case, interval in cases.items():
            interval = _mapping(interval)
            gain, lower, upper = (interval.get(key) for key in ("gain", "lower", "upper"))
            finite = all(isinstance(number, (int, float)) and not isinstance(number, bool) and
                         math.isfinite(number) and number > 0 for number in (gain, lower, upper))
            if not finite or lower > upper:
                rows.append(prefix + [case, "inconnu", "IC non établi", "intervalle incomplet ou incohérent"])
                continue
            interpretation = ("IC entièrement sous 1" if upper < 1 else
                              "IC entièrement au-dessus de 1" if lower > 1 else "IC recouvre 1")
            rows.append(prefix + [case, fmt(gain, 6), "["+fmt(lower, 6)+" ; "+fmt(upper, 6)+"]", interpretation])
    return ("<details><summary>Ratios et intervalles par cas final</summary>" +
            "<p>Chaque ratio divise la latence de la référence par celle du candidat. Les contrastes directs "
            "proviennent de leurs propres paires ; ils ne sont pas déduits des gains face au naïf. "
            "Ces IC à 95 % sont individuels, sans correction pour comparaisons multiples. "
            "Un IC recouvrant 1 ne démontre pas une différence. Le contrôle de non-régression des cas critiques "
            "reste distinct et suit le contrat figé. Les mesures partielles restent visibles avec le statut du reçu ; "
            "un intervalle ne qualifie ni l’artefact ni la méthode. Les absences et identités ne créent aucune série mesurée.</p>" +
            table(["Groupe", "Candidat / référence", "Statut du reçu / qualification", "Cas", "Ratio", "IC95% individuel", "Lecture de l’intervalle"], rows) + "</details>")


def audit_table(path):
    lines = Path(path).read_text().splitlines()
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in lines if line.startswith("|")]
    return table(rows[0], rows[2:]) if rows else ""


def checkout_source_identity(repo):
    """Capture all Git-visible source changes; ignored generated outputs stay out."""
    repo = Path(repo).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo, text=True).strip()
    fields = subprocess.check_output(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], cwd=repo).decode("utf-8", errors="surrogateescape").split("\0")
    entries, index = [], 0
    while index < len(fields):
        record = fields[index]
        index += 1
        if not record:
            continue
        entry = {"status": record[:2], "path": record[3:]}
        if "R" in entry["status"] or "C" in entry["status"]:
            entry["original_path"] = fields[index]
            index += 1
        entries.append(entry)
    return {"checkout": str(repo), "head": head, "branch": branch, "dirty": bool(entries),
            "dirty_entries": entries,
            "tracked_changes": [entry for entry in entries if entry["status"] != "??"],
            "untracked_files": [entry["path"] for entry in entries if entry["status"] == "??"],
            "scope": "All Git tracked changes and nonignored untracked files; Git-ignored generated outputs excluded"}


def ci_qualifies_checkout(ci, source):
    return bool(source.get("head") and not source["dirty"] and ci.get("headSha") == source["head"] and
                ci.get("conclusion") == "success" and ci.get("status") == "completed")


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _records(value):
    return value if isinstance(value, list) and all(isinstance(item, dict) for item in value) else []


def _hash(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{"+str(length)+"}", value) is not None


def _linked_receipt_verified(repo, reference, *, suffix=None, within=None, exact=None):
    """Authorize type and location before opening any bytes for authentication."""
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str) or not _hash(reference.get("sha256")):
        return False
    relative = Path(reference["path"])
    if relative.is_absolute() or (suffix and relative.suffix != suffix):
        return False
    repo = Path(repo).resolve()
    unresolved = repo/relative
    # An authorized role must be the regular file at that location, not a
    # symlink (including a parent directory) to a private but in-repo payload.
    for component in (unresolved, *unresolved.parents):
        if component == repo:
            break
        if component.is_symlink():
            return False
    path = unresolved.resolve()
    allowed = (repo/within).resolve() if within else repo
    return bool(allowed.is_relative_to(repo) and path.is_relative_to(allowed) and
                (exact is None or path == (repo/exact).resolve()) and path.is_file() and
                hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"])


def _verified_json(repo, reference, default=None, *, within=None, exact=None, expected_type=dict):
    """Only open a declared, hash-matched JSON link inside its evidence root."""
    if not _linked_receipt_verified(repo, reference, suffix=".json", within=within, exact=exact):
        return default
    try:
        value = load(Path(repo)/reference["path"], default)
        return value if isinstance(value, expected_type) else default
    except (ValueError, OSError):
        return default


def _required_checks(value, names):
    checks = _mapping(_mapping(value).get("checks"))
    return all(checks.get(name) is True for name in names)


def _original_execution_control(repo, value):
    """Recheck archived execution links; never turn search results into finals."""
    result = _verified_json(repo, value.get("result"), exact=ORIGINAL_CAMPAIGN+"/result.json") or {}
    manifest = _verified_json(repo, value.get("original_manifest"), exact=ORIGINAL_CAMPAIGN+"/generation/manifest.json") or {}
    history = _records(_verified_json(repo, value.get("history"), [], exact=ORIGINAL_CAMPAIGN+"/generation/history.json", expected_type=list))
    upstream, adapters, calls = (_records(value.get(key)) for key in ("upstream_source_checks", "adapter_checks", "completed_cli_calls"))
    candidate, terminal = (_mapping(value.get(key)) for key in ("candidate", "worker_terminal"))
    checks = {
        "receipt_kind": value.get("kind") == "actual_pinned_original_generation" and value.get("schema_version") == 1,
        "archived_execution_checks": value.get("ok") is True and _required_checks(value, (
            "actual_original_workflow_completed", "all_28_search_case_runs_valid", "all_upstream_sources_match_pinned_git",
            "all_adapter_bytes_match_frozen_controller", "all_ten_cli_receipts_completed", "candidate_artifact_recomputed")),
        "owned_terminal": bool(value.get("job_id") and value.get("session_id") and
            str(terminal.get("job_id")) == str(value["job_id"]) and terminal.get("state") == "COMPLETED" and
            terminal.get("exit_code") == "0:0" and result.get("job_id") == value["job_id"] and
            result.get("session_id") == value["session_id"] and result.get("worker_terminal") == terminal),
        "executed_method_completed": all(record.get("method") == "true_upstream_original" and
            record.get("generation_complete") is True and record.get("method_status") == "completed"
            for record in (result, manifest)),
        "pinned_original_sources": bool(len(upstream) == 16 and value.get("upstream_commit") == UPSTREAM_COMMIT and
            all(record.get("upstream", {}).get("commit") == UPSTREAM_COMMIT for record in (result, manifest)) and
            len({entry.get("path") for entry in upstream}) == 16 and all(
                entry.get("matches_git_blob") is True and _hash(entry.get("sha256")) and
                all(record.get("upstream", {}).get("source_hashes", {}).get(entry.get("path")) == entry["sha256"]
                    for record in (result, manifest)) for entry in upstream)),
        "adapter_hashes_linked": bool({entry.get("path") for entry in adapters} ==
            {"upstream_original.py", "frozen_launch.py", "codex_backend.py", "gain_observer.py"} and all(
                entry.get("matches_executed_frozen_source") is True and
                _hash(entry.get("sha256")) and all(
                    record.get("adapter_source_hashes", {}).get(entry.get("path")) == entry["sha256"]
                    for record in (result, manifest)) for entry in adapters)),
        "four_search_iterations_seven_valid_cases": isinstance(history, list) and len(history) == 4 and
            [entry.get("iteration") for entry in history] == [0, 1, 2, 3] and all(
                len(entry.get("case_runs", [])) == 7 and all(run.get("validation_ok") is True
                    for run in entry["case_runs"]) for entry in history),
        "retained_candidate_linked": bool(_linked_receipt_verified(repo, candidate.get("source"), suffix=".cu", exact=ORIGINAL_CAMPAIGN+"/generation/frozen-original.cu") and all(
            record.get("candidate", {}).get("candidate_id") == candidate.get("candidate_id") and
            record.get("candidate", {}).get("artifact_hash") == candidate.get("artifact_hash") and
            record.get("candidate", {}).get("source_hash") == candidate.get("source", {}).get("sha256")
            for record in (result, manifest))),
    }
    cli_prefix = ORIGINAL_CAMPAIGN+"/generation/model-transport/intents"
    call_records = [_verified_json(repo, call, within=cli_prefix) or {} if
        _hash(call.get("call_id"), 32) and isinstance(call.get("path"), str) and
        re.fullmatch(re.escape(cli_prefix)+r"/[0-9a-f]{64}/[0-9]+/cli/"+call["call_id"]+r"\.json", call["path"])
        else {} for call in calls]
    usage = _mapping(value.get("usage"))
    checks["ten_unique_completed_cli_receipts"] = bool(len(calls) == 10 and
        len({call.get("call_id") for call in calls}) == 10 and all(
            record.get("call_id") == call.get("call_id") and record.get("status") == "completed" and
            record.get("model") == call.get("requested_model") == "gpt-6-luna" and
            record.get("reasoning_effort") == call.get("reasoning_effort") == "max" and
            record.get("usage") == call.get("usage") for call, record in zip(calls, call_records)))
    checks["observed_usage_linked"] = (all(type(usage.get(key)) is int for key in ("input_tokens", "output_tokens")) and usage.get("model_calls") == 10 and
        usage.get("model_tokens") == usage.get("input_tokens", 0) + usage.get("output_tokens", 0) and
        all(record.get("usage") == usage for record in (result, manifest)) and all(
        all(type(record.get("usage", {}).get(key)) is int for record in call_records) and
        usage.get(key) == sum(record["usage"][key] for record in call_records)
        for key in ("input_tokens", "cached_input_tokens", "output_tokens")))
    verified = all(checks.values())
    return {"ok": verified if value else None, "authentication_checks": checks,
        "status": "génération originale ROMEO authentifiée — finale distincte" if verified else
                  "preuve originale incomplète ou liens non authentifiés" if value else "en attente — original non authentifié",
        "source_head": value.get("source_controller"), "upstream_commit": value.get("upstream_commit"),
        "usage": usage if verified else {}, "new_gpu_execution": False, "final_qualified": False,
        "scope": "Workflow original épinglé, 16 empreintes de sources et 4 adapters liés aux manifests ; "
                 "28 cas SEARCH et 10 invocations CLI archivées. Liens SHA revérifiés sans nouvelle exécution. "
                 "Itération 0 fournie, puis 3 propositions ; paramètres/outils adaptés à Codex et ROMEO. "
                 "Le modèle gpt-6-luna et l’effort max sont demandés, les poids fournisseur restent non attestés. "
                 "L’observateur est extérieur à la méthode et son surcoût reste séparé. Aucun gain final ni succès global déduit."
                 if verified else "Protocole prévu : authentifier l’exécution du workflow original épinglé et ses liens SEARCH. "
                 "Preuves absentes, partielles ou non authentifiées ; aucune génération ni finale réussie déduite."}


def _common_freeze_control(repo, value):
    naive, worker = (_mapping(value.get(key)) for key in ("naive", "worker"))
    methods = _records(value.get("methods"))
    naive_source = {"path": naive.get("source_path"), "sha256": naive.get("source_sha256")}
    hardware = _verified_json(repo, worker.get("hardware_receipt"), exact=FINAL_DIRECTORY+"/worker/hardware.json") or {}
    session = _verified_json(repo, worker.get("session_receipt"), exact=FINAL_DIRECTORY+"/worker/slurm-session.json") or {}
    checks = {
        "receipt_kind": value.get("kind") == "read_only_common_final_candidate_freeze_audit" and value.get("schema_version") == 1,
        "archived_freeze_checks": value.get("freeze_identity_checks_pass") is True and not value.get("errors") and
            _required_checks(value, ("all_eight_generation_to_frozen_identity_checks_pass",
                "all_observed_files_unchanged_before_receipt", "frozen_has_exact_eight_methods_plus_naive",
                "naive_load_candidate_accepts_artifact", "naive_source_matches_static_convolution", "worker_identity_checks_pass")),
        "naive_source_link": _linked_receipt_verified(repo, naive_source, suffix=".cu", exact=FINAL_DIRECTORY+"/sealed-inputs/baseline.cu"),
        "naive_artifact_identity": _hash(naive.get("artifact_sha256")),
        "frozen_manifest_link": _linked_receipt_verified(repo, value.get("frozen_candidates"), suffix=".json", exact=FINAL_DIRECTORY+"/frozen-candidates.json"),
        "eight_declared_methods": len(methods) == 8 and {entry.get("method") for entry in methods} == set(LABELS),
        "worker_links": bool(worker.get("job_id") and worker.get("session_id") and
            str(hardware.get("slurm_job_id")) == str(worker["job_id"]) and hardware.get("environment") == "romeo" and
            str(session.get("job_id")) == str(worker["job_id"]) and session.get("session_id") == worker["session_id"]),
    }
    aliases = []
    for entry in methods:
        method = entry.get("method", "unknown")
        if method not in LABELS:
            checks["unknown_method"] = False
            continue
        campaign = "results/quality-study-v2/campaigns/"+FIRST_GROUP+"."+method
        source_name = {"true_upstream": "frozen-original.cu", "native_codex_vanilla": "evidence/kernel.cu"}.get(method, "best.cu")
        result = _verified_json(repo, entry.get("generation_result"), exact=campaign+"/result.json") or {}
        candidate = _mapping(result.get("candidate"))
        frozen_source = _mapping(entry.get("frozen_source"))
        same_source = _hash(naive.get("source_sha256")) and frozen_source.get("sha256") == naive["source_sha256"]
        same_artifact = _hash(naive.get("artifact_sha256")) and entry.get("artifact_sha256") == naive["artifact_sha256"]
        checks[method] = bool(_required_checks(entry, FREEZE_METHOD_CHECKS) and
            _linked_receipt_verified(repo, frozen_source, suffix=".cu", exact=FINAL_DIRECTORY+"/frozen/"+method+"/kernel.cu") and
            _linked_receipt_verified(repo, entry.get("generation_source"), suffix=".cu", exact=campaign+"/generation/"+source_name) and
            bool(result) and
            candidate.get("candidate_id") == entry.get("candidate_id") and
            candidate.get("artifact_hash") == entry.get("artifact_sha256") and
            candidate.get("source_hash") == frozen_source.get("sha256") and
            result.get("job_id") == entry.get("generation_job_id") and
            result.get("session_id") == entry.get("generation_session_id") and result.get("execution_complete") is True and
            entry.get("same_source_as_naive") is same_source and entry.get("same_artifact_as_naive") is same_artifact)
        aliases.append({"mode": method, "same_source_as_naive": same_source, "same_artifact_as_naive": same_artifact,
                        "artifact_hash": entry.get("artifact_sha256"), "source_sha256": frozen_source.get("sha256")})
    verified = all(checks.values())
    return {"ok": verified if value else None, "authentication_checks": checks,
        "status": "identités gelées authentifiées — qualification finale distincte" if verified else
                  "gel incomplet ou liens non authentifiés" if value else "en attente — aucun gel authentifié",
        "job_id": worker.get("job_id"), "terminal": {}, "source_head": None,
        "audit_source_head": _mapping(value.get("source_observation")).get("HEAD"),
        "artifact_aliases": aliases if verified else [], "new_gpu_execution": False, "final_qualified": False,
        "scope": "Audit local du gel du naïf et des huit entrées méthode ; aucun input final numérique ni graine privée lu. "
                 "Les copies de fichiers ne créent pas huit kernels distincts. Un artefact exact partagé reste une identité, "
                 "sans gain ni régression statistique entre ses labels, même si un ancien reçu donne des temps bruités. "
                 "Même source avec lancement différent signifie un artefact différent. Les reçus worker sont historiques ; "
                 "cet audit n’atteste ni terminal Slurm actuel, ni validation, sanitizers ou mesure finale."
                 if verified else "Protocole prévu : authentifier les neuf identités gelées et les aliases du naïf. "
                 "Preuves absentes, partielles ou non authentifiées ; aucun gel, worker ou résultat final qualifié déduit."}


def _standalone_final_control(value):
    result, terminal, provenance = (_mapping(value.get(name)) for name in ("result","worker_terminal","source_provenance"))
    checks = {"receipt_kind":value.get("schema_version")==1 and value.get("kind")=="standalone_distinct_final_suite_acceptance",
        "executed_final_controls":value.get("ok") is True and value.get("status")=="final_suite_acceptance_passed" and
            result.get("ok") is True and _required_checks(result,FINAL_SUITE_CHECKS),
        "owned_clean_terminal":str(value.get("job_id","")).isdigit() and terminal.get("terminal") is True and
            terminal.get("state")=="COMPLETED" and terminal.get("exit_code")=="0:0" and str(terminal.get("job_id"))==str(value.get("job_id")),
        "executed_sources_preserved":_hash(provenance.get("HEAD"),40) and value.get("sources_unchanged") is True and
            bool(_mapping(provenance.get("sources"))) and all(_hash(item) for item in _mapping(provenance.get("sources")).values()),
        "no_model_calls":value.get("model_calls")==value.get("model_tokens")==0}
    verified=all(checks.values())
    return {"ok":verified if value else None,"authentication_checks":checks,"final_qualified":False,
        "new_gpu_execution":verified,"status":"Chemin FINAL autonome exécuté et contrôlé — hors cohortes qualité" if verified else
            "Contrôle FINAL autonome incomplet ou non authentifié" if value else "en attente — aucune exécution du contrôle FINAL autonome",
        "scope":"Contrôle fonctionnel GPU déclaré : cas FINAL distincts, oracles, paires fraîches, quatre sanitizers et refus de réutilisation. "
            "Les checks du résultat, les sources et le terminal Slurm sont distincts du plan. Aucun gain de l’optimiseur ni qualification §16 déduit."
            if verified else "Plan ou contrôle autonome sans résultat qualifié ; aucune mesure GPU, finalisation réussie ou qualification §16 déduite."}


def _accepted_eleven_source_checks(repo, value, original, old_files, linked, files, resumption, diagnostic_id, technique):
    """Authenticate the fixed7→9→11 documentary chain; never qualify old11."""
    chain = _mapping(original.get("resumption"))
    if (type(chain.get("inherited_completed_model_calls")) is not int or
            chain.get("inherited_completed_model_calls") != 9 or
            _mapping(chain.get("source_receipt")).get("path") != RESEARCH_REFRESH_RESUMED):
        # Explicitly bound the only supported lower edge before calling its
        # validator; malformed/cyclic authorities must never recurse.
        return {"strict_seven_nine_eleven_accepted_lineage":False}
    root = repo/Path(RESEARCH_REFRESH_RESUMED_2).parent
    def source_json(name, expected_type=dict):
        return _verified_json(root, {"path":name,"sha256":old_files.get(name)}, exact=name,
                              expected_type=expected_type)
    # This call inspects the lower9-call lineage only. The old11 authority
    # remains incomplete: its acceptance-success and exact-coverage checks are
    # false and are deliberately not converted into success here.
    lower = _research_refresh_control(repo, original, root/"research-refresh-acceptance.json",
        resumed=True, lineage_diagnostic_id=diagnostic_id)["authentication_checks"]
    lower_names = ("original_incomplete_authority_preserved", "completed_call_prefix_inherited_without_repetition",
        "copied_prefix_json_links_preserved", "strict_seven_to_nine_source_chain",
        "source_nine_exact_failed_round_events_and_calls", "source_nine_method_dossier_refusal_linked",
        "source_nine_ancestor_copies_and_trace_hashes", "source_nine_declared_ledger_delta",
        "run_intent_identity", "proof_manifest_json_subset", "historical_bootstrap_preserved",
        "no_new_gpu_or_current_hardware_claim")
    events = source_json("events.json",list) or []
    prior = _verified_json(repo,chain.get("source_receipt"),exact=RESEARCH_REFRESH_RESUMED) or {}
    prior_proof = _verified_json(repo,chain.get("source_proof_manifest"),
        exact=str(Path(RESEARCH_REFRESH_RESUMED).parent/"proof-sha256.json")) or {}
    prior_files = _mapping(prior_proof.get("files"))
    prefix = _verified_json(repo/Path(RESEARCH_REFRESH_RESUMED).parent,
        {"path":"events.json","sha256":prior_files.get("events.json")},exact="events.json",expected_type=list) or []
    calls = _records(original.get("model_calls"))
    invocations = source_json("campaign/acceptance-controls-resume-2/invocations.json",list) or []
    traces = [source_json("campaign/model/"+call["call_id"]+".json") or {} for call in calls if _hash(call.get("call_id"),32)]
    suffix = bool(len(calls)==len(invocations)==len(traces)==2 and
        [row.get("purpose") for row in invocations]==["research","proposal"] and all(
        invocation.get("stage")=="research_service" and invocation.get("kernel_proposal") is False and
        invocation.get("status")=="completed" and invocation.get("call_id")==call.get("call_id")==trace.get("call_id") and
        trace.get("status")=="completed" and trace.get("returncode")==0 and trace.get("model")=="gpt-6-luna" and
        trace.get("reasoning_effort")=="max" and trace.get("usage")==call.get("usage") and
        trace.get("purpose")==invocation.get("purpose") and
        hashlib.sha256(json.dumps(trace.get("payload"),sort_keys=True,default=str,allow_nan=False).encode()).hexdigest()
            ==invocation.get("returned_payload_sha256") for call,invocation,trace in zip(calls,invocations,traces)))
    event = _mapping(events[-1].get("payload")) if events else {}
    research_id = event.get("research_id")
    dossiers = [name for name in old_files if isinstance(name,str) and
                re.fullmatch(r"campaign/trials/[A-Za-z0-9_-]+/dossier\.json",name)]
    accepted = source_json(dossiers[0]) or {} if len(dossiers)==1 else {}
    body = source_json("campaign/research/"+research_id+".json") if _hash(research_id,32) else None
    entry = source_json("resume-state.json") or {}
    old_entry = _mapping(entry.get("state"))
    initial = source_json("campaign/acceptance-evidence/initial-dossier.json") or {}
    synthesis_payload = _mapping(traces[-1].get("payload")) if traces else {}
    synthesis_fields = ("covered_techniques", "reference_kernels", "launch_requirements", "dependencies",
        "applicability_checks", "incompatible_conditions", "missing_evidence", "pmpp_topics", "hypotheses",
        "expected_effects", "validation_metrics", "negative_search", "original_hypothesis")
    post = dict(old_entry,research_round=4,research_ready=True,needs_research=False,research_refresh_required=False,error=None)
    requested = dict(post,needs_research=True,research_refresh_required=True)
    request = linked.get("coverage-refresh-request.json") or {}
    coverage = _mapping(resumption.get("coverage_refresh"))
    wanted_coverage = {"kind":"explicit_complementary_coverage_refresh","accepted_research_id":research_id,
        "accepted_dossier":{"path":dossiers[0],"sha256":old_files.get(dossiers[0])} if len(dossiers)==1 else {},
        "missing_techniques":[technique],"diagnostic_id":diagnostic_id,"source_helper_error":original.get("error"),
        "prior_research_succeeded":True,"semantic_equivalence_is_not_exact_coverage":True}
    spent = _mapping(_mapping(original.get("ledger_usage")).get("spent"))
    prior_spent = _mapping(_mapping(prior.get("ledger_usage")).get("spent"))
    delta = _mapping(original.get("new_usage_delta"))
    return {
        "strict_seven_nine_eleven_accepted_lineage":all(lower.get(name) is True for name in lower_names),
        "source_eleven_accepted_research_events_calls":bool(len(prefix)==19 and len(events)==22 and events[:19]==prefix and
            [row.get("kind") for row in events[19:]]==["model","model","research"] and suffix and
            original.get("error")=="Actual resumed dossier does not cover the exact blocked technique/diagnostic" and
            original.get("resume_source_unchanged") is True and original.get("bootstrap_files_unchanged") is True and
            original.get("new_gpu_requests")==original.get("new_gpu_requests_attempted")==0 and
            original.get("hardware_currently_observed") is False and not original.get("checks") and
            event.get("diagnostic_id")==diagnostic_id and event.get("targeted_refresh") is True and
            bool(old_entry.get("trial_id")) and event.get("trial_id")==old_entry.get("trial_id") and
            all(_mapping(row.get("payload")).get("call_id")==call.get("call_id") and
                _mapping(row.get("payload")).get("usage")==call.get("usage") and
                _mapping(row.get("payload")).get("purpose")==invocation.get("purpose")
                for row,call,invocation in zip(events[19:21],calls,invocations))),
        "accepted_dossier_preserved_without_relabel":bool(len(dossiers)==1 and accepted==body and
            accepted.get("research_id")==research_id and accepted.get("receipt_version")==2 and
            research_id != initial.get("research_id") and
            accepted.get("retrieval_status")=="retrieved" and not accepted.get("missing_evidence") and
            bool(accepted.get("applicability_checks")) and all(row.get("satisfied") is True for row in _records(accepted.get("applicability_checks"))) and
            diagnostic_id in accepted.get("diagnostic_ids",[]) and technique not in accepted.get("covered_techniques",[]) and
            accepted.get("call_ids")==[call.get("call_id") for call in calls] and
            files.get("source-evidence/previous-accepted-dossier.json")==old_files.get(dossiers[0]) and
            linked.get("source-evidence/previous-accepted-dossier.json")==accepted and
            value.get("renewed_research_id")!=research_id),
        "accepted_dossier_matches_exact_synthesis":bool(all(key in accepted and key in synthesis_payload and
            accepted[key]==synthesis_payload[key] for key in synthesis_fields) and accepted.get("usage")=={
                "input_tokens":sum(_mapping(call.get("usage")).get("input_tokens",0) for call in calls),
                "output_tokens":sum(_mapping(call.get("usage")).get("output_tokens",0) for call in calls),"cost_usd":None}),
        "complementary_refresh_explicit_round4_to5":bool(resumption.get("evidence_repository")==str(repo.resolve()) and
            _mapping(linked.get("run-intent.json")).get("resumption")==resumption and
            resumption.get("phase")=="complementary_coverage_refresh" and
            resumption.get("controls_directory")=="acceptance-controls-resume-3" and coverage==wanted_coverage and
            resumption.get("research_round_derivation")=={"initial_anchor_round":2,"completed_failed_round":3,"completed_accepted_round":4,"next_fresh_round":5} and
            value.get("coverage_refresh_request")=={"path":"coverage-refresh-request.json","sha256":files.get("coverage-refresh-request.json")} and
            all(request.get(key)==item for key,item in wanted_coverage.items()) and
            request.get("phase")=="complementary_coverage_refresh" and request.get("next_fresh_round")==5 and
            request.get("source_receipt")==resumption.get("source_receipt") and
            request.get("post_research_state")==post and request.get("requested_state")==requested and
            _mapping(linked.get("resume-state.json")).get("phase")=="accepted_research_coverage_gap_reconstructed" and
            _mapping(linked.get("resume-state.json")).get("state")==post and
            _mapping(linked.get("resume-state.json")).get("source_events_sha256")==old_files.get("events.json")),
        "source_eleven_declared_ledger_delta":bool(spent.get("model_calls")==11 and prior_spent.get("model_calls")==9 and
            delta.get("model_calls")==2 and delta.get("model_tokens")==sum(
                _mapping(call.get("usage")).get("input_tokens",0)+_mapping(call.get("usage")).get("output_tokens",0) for call in calls) and
            delta=={key:spent.get(key,0)-prior_spent.get(key,0) for key in set(spent)|set(prior_spent)}),
    }


def _research_refresh_control(repo, value, path, *, resumed=False, lineage_diagnostic_id=None):
    """Authenticate explicit documentary proof files, without reading a database."""
    root = path.parent
    proof = _verified_json(root, value.get("proof_manifest"), exact="proof-sha256.json") or {}
    files = _mapping(proof.get("files"))
    # This is deliberately a declared JSON subset: no SQLite, pickle, RNG or
    # numerical input is opened by the reporting authentication pass.
    required = ["plan.json", "run-intent.json", "events.json", "campaign/acceptance-controls/invocations.json",
                "campaign/acceptance-evidence/initial-dossier.json", "campaign/acceptance-evidence/blocked-refusal.json"]
    call_ids = [value.get(key) for key in ("initial_call_id", "compatible_call_id", "blocked_call_id", "final_call_id")]
    new_calls = _records(value.get("model_calls"))
    inherited_calls = _records(value.get("inherited_model_calls")) if resumed else []
    resumption = _mapping(value.get("resumption")) if resumed else {}
    inherited_count = resumption.get("inherited_completed_model_calls")
    controls_directory = {9:"acceptance-controls-resume-2",11:"acceptance-controls-resume-3"}.get(
        inherited_count,"acceptance-controls-resume")
    calls = inherited_calls + new_calls
    if resumed:
        required += ["resume-state.json", "campaign/"+controls_directory+"/invocations.json",
                     "campaign/hardware-evidence-context.json",
                     "source-evidence/previous-receipt.json", "source-evidence/previous-proof.json",
                     "source-evidence/previous-events.json", "source-evidence/previous-dossier.json"]
        if inherited_count in (9,11):
            required += ["campaign/runtime-evidence-context.json"]
        if inherited_count == 11:
            required += ["coverage-refresh-request.json", "source-evidence/previous-accepted-dossier.json"]
    valid_call_ids = bool(calls) and all(_hash(call.get("call_id"), 32) for call in calls)
    required += ["campaign/model/"+call["call_id"]+".json" for call in calls if
                 _hash(call.get("call_id"), 32)]
    renewed = [name for name in files if isinstance(name, str) and re.fullmatch(r"campaign/trials/[A-Za-z0-9_-]+/dossier\.json", name)]
    required += renewed
    list_files = {"events.json", "campaign/acceptance-controls/invocations.json",
                  "source-evidence/previous-events.json", "campaign/"+controls_directory+"/invocations.json"}
    linked = {name: _verified_json(root, {"path": name, "sha256": files.get(name)}, expected_type=list if name in list_files else dict) for name in required}
    initial_name = "campaign/acceptance-evidence/initial-dossier.json"
    initial = linked.get(initial_name) or {}
    renewed_dossier = linked.get(renewed[0]) or {} if len(renewed) == 1 else {}
    refusal = linked.get("campaign/acceptance-evidence/blocked-refusal.json") or {}
    intent = linked.get("run-intent.json") or {}
    technique, diagnostic_id = value.get("uncovered_technique"), value.get("diagnostic_id") or lineage_diagnostic_id
    checks = {
        "receipt_kind": value.get("schema_version") == 1 and value.get("kind") == (
            "documentary_research_refresh_resumption_acceptance" if resumed else "documentary_research_refresh_acceptance"),
        "explicit_documentary_success": value.get("ok") is True and value.get("status") == (
            "documentary_research_refresh_resumption_acceptance_passed" if resumed else "documentary_research_refresh_acceptance_passed"),
        "required_controls": _required_checks(value, REFRESH_CHECKS),
        "no_new_gpu_or_current_hardware_claim": value.get("new_gpu_requests") == 0 and
            value.get("new_gpu_requests_attempted") == 0 and value.get("hardware_currently_observed") is False,
        "historical_bootstrap_preserved": value.get("bootstrap_files_unchanged") is True,
        "run_intent_identity": bool(intent.get("run_id") == value.get("run_id") and
            intent.get("source_provenance") == value.get("source_provenance") and intent.get("new_gpu_requests") == 0),
        "proof_manifest_json_subset": proof.get("schema_version") == 1 and proof.get("hash_algorithm") == "sha256" and
            len(renewed) == 1 and all(item is not None for item in linked.values()),
        "actual_control_calls_linked": bool(valid_call_ids and all(_hash(call_id, 32) for call_id in call_ids) and len(set(call_ids)) == 4 and
            set(call_ids).issubset({call.get("call_id") for call in calls}) and all(
                all((linked.get("campaign/model/"+call.get("call_id", "")+".json") or {}).get(key) == expected
                    for key, expected in (("call_id", call.get("call_id")), ("status", "completed"),
                                          ("model", "gpt-6-luna"), ("reasoning_effort", "max")))
                for call in calls)),
        "distinct_dossiers": bool(value.get("initial_research_id") and value.get("renewed_research_id") and
            value["initial_research_id"] != value["renewed_research_id"] and diagnostic_id and technique and
            initial.get("research_id") == value["initial_research_id"] and
            renewed_dossier.get("research_id") == value["renewed_research_id"] and
            initial.get("receipt_version") == renewed_dossier.get("receipt_version") == 2 and
            technique not in initial.get("covered_techniques", []) and technique in renewed_dossier.get("covered_techniques", []) and
            diagnostic_id in renewed_dossier.get("diagnostic_ids", []) and
            files.get(initial_name) == value.get("initial_dossier_sha256") and
            len(renewed) == 1 and files.get(renewed[0]) == value.get("renewed_dossier_sha256")),
        "semantic_refusal_linked": bool(refusal.get("technique") == technique and refusal.get("call_id") == value.get("blocked_call_id") and
            _required_checks(refusal, ("needs_research_returned", "exact_semantic_guard_reason", "model_actually_declared_control_technique",
                "no_new_generation_event", "no_new_candidate_or_source_change", "no_accepted_proposal_file"))),
    }
    if path == repo/RESEARCH_REFRESH_RESUMED_3:
        checks["third_authority_requires_eleven_call_source"] = inherited_count == 11
    if resumed:
        source_path = {9:RESEARCH_REFRESH_RESUMED,11:RESEARCH_REFRESH_RESUMED_2}.get(inherited_count,RESEARCH_REFRESH)
        original = _verified_json(repo, resumption.get("source_receipt"), exact=source_path) or {}
        old_proof = _verified_json(repo, resumption.get("source_proof_manifest"), exact=str(Path(source_path).parent/"proof-sha256.json")) or {}
        old_files = _mapping(old_proof.get("files"))
        old_dossiers = [name for name in old_files if isinstance(name, str) and re.fullmatch(r"campaign/trials/[A-Za-z0-9_-]+/dossier\.json", name)]
        old_ids, new_ids = [call.get("call_id") for call in inherited_calls], [call.get("call_id") for call in new_calls]
        checks.update({
            "resumption_required_checks": _required_checks(value, REFRESH_RESUMPTION_CHECKS),
            "original_incomplete_authority_preserved": bool(
                type(inherited_count) is int and inherited_count in (7, 9, 11) and original.get("kind") == ("documentary_research_refresh_resumption_acceptance"
                    if inherited_count in (9,11) else "documentary_research_refresh_acceptance") and original.get("ok") is False and
                original.get("status") == "incomplete" and resumption.get("source_directory") == str(Path(source_path).parent) and
                _mapping(resumption.get("source_receipt")).get("path_base") == "repository" and
                _mapping(resumption.get("source_proof_manifest")).get("path_base") == "repository" and
                old_proof.get("schema_version") == 1 and old_proof.get("hash_algorithm") == "sha256" and
                _mapping(original.get("proof_manifest")).get("sha256") == _mapping(resumption.get("source_proof_manifest")).get("sha256") and
                files.get("source-evidence/previous-receipt.json") == _mapping(resumption.get("source_receipt")).get("sha256") and
                files.get("source-evidence/previous-proof.json") == _mapping(resumption.get("source_proof_manifest")).get("sha256")),
            "completed_call_prefix_inherited_without_repetition": bool(
                inherited_count in (7, 9, 11) and inherited_calls == (
                    _records(original.get("inherited_model_calls"))+_records(original.get("model_calls"))
                    if inherited_count in (9,11) else original.get("model_calls")) and
                len(old_ids) == len(set(old_ids)) == inherited_count and bool(new_ids) and len(new_ids) == len(set(new_ids)) and
                set(old_ids).isdisjoint(new_ids) and resumption.get("no_initial_proposal_or_diagnostic_replay") is True and
                all(_hash(call_id, 32) and files.get("campaign/model/"+call_id+".json") ==
                    old_files.get("campaign/model/"+call_id+".json") for call_id in old_ids)),
            "copied_prefix_json_links_preserved": bool(len(old_dossiers) == 1 and all(
                files.get(name) == old_files.get(name) and _hash(files.get(name)) for name in (
                    "campaign/acceptance-evidence/initial-dossier.json", "campaign/acceptance-evidence/blocked-refusal.json",
                    "campaign/acceptance-controls/invocations.json")) and
                files.get("source-evidence/previous-events.json") == old_files.get("events.json") and
                files.get("source-evidence/previous-dossier.json") == old_files.get(
                    initial_name if inherited_count == 11 else old_dossiers[0])),
            "final_call_new_and_initial_controls_inherited": bool(all(_hash(call_id, 32) for call_id in call_ids) and
                set(call_ids[:3]).issubset(old_ids) and call_ids[3] in new_ids),
        })
        if inherited_count == 9:
            # Only the observed seven-call anchor followed by its two-call
            # research refusal is supported. This is not an arbitrary ancestry
            # walk, and no database, RNG or numerical input is opened.
            chain = _mapping(original.get("resumption"))
            anchor = _verified_json(repo, chain.get("source_receipt"), exact=RESEARCH_REFRESH) or {}
            anchor_proof = _verified_json(repo, chain.get("source_proof_manifest"),
                exact=str(Path(RESEARCH_REFRESH).parent/"proof-sha256.json")) or {}
            anchor_files = _mapping(anchor_proof.get("files"))
            source_root, anchor_root = repo/Path(RESEARCH_REFRESH_RESUMED).parent, repo/Path(RESEARCH_REFRESH).parent
            def source_json(name, *, anchor_file=False, expected_type=dict):
                file_map, base = (anchor_files, anchor_root) if anchor_file else (old_files, source_root)
                return _verified_json(base, {"path":name, "sha256":file_map.get(name)},
                                      exact=name, expected_type=expected_type)
            prefix = source_json("events.json", anchor_file=True, expected_type=list) or []
            prior_events = source_json("events.json", expected_type=list) or []
            prior_intent = source_json("run-intent.json") or {}
            prior_calls = _records(original.get("model_calls"))
            anchor_calls = _records(anchor.get("model_calls"))
            prior_invocations = source_json("campaign/acceptance-controls-resume/invocations.json", expected_type=list) or []
            initial_kinds = ["campaign_initialized", "checkpoint", "selection", "selection", "model", "model",
                "research", "model", "generation", "model", "generation", "model", "diagnostic", "model", "model", "research_blocked"]
            last_error = _mapping(prior_events[-1].get("payload")).get("error") if prior_events else None
            prefix_error = _mapping(prefix[-1].get("payload")).get("error") if prefix else None
            source_trace_rows = [source_json("campaign/model/"+call["call_id"]+".json") or {}
                                 for call in prior_calls if _hash(call.get("call_id"), 32)]
            exact_suffix = bool(len(prior_calls) == len(prior_invocations) == len(source_trace_rows) == 2 and
                [row.get("purpose") for row in prior_invocations] == ["research", "proposal"] and all(
                invocation.get("stage") == "research_service" and invocation.get("kernel_proposal") is False and
                invocation.get("status") == "completed" and invocation.get("call_id") == call.get("call_id") == trace.get("call_id") and
                trace.get("status") == "completed" and trace.get("returncode") == 0 and
                trace.get("model") == "gpt-6-luna" and trace.get("reasoning_effort") == "max" and
                trace.get("purpose") == invocation.get("purpose") and trace.get("usage") == call.get("usage") and
                hashlib.sha256(json.dumps(trace.get("payload"), sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()
                    == invocation.get("returned_payload_sha256")
                for call, invocation, trace in zip(prior_calls, prior_invocations, source_trace_rows)))
            diagnostic_ids = [_mapping(row.get("payload")).get("diagnostic_id") for row in prefix if row.get("kind") == "diagnostic"]
            failed_dossiers = []
            for name in old_files:
                if not isinstance(name, str) or not re.fullmatch(r"campaign/research/[0-9a-f]{32}\.json", name):
                    continue
                dossier = source_json(name) or {}
                if dossier.get("call_ids") == [row.get("call_id") for row in prior_calls]:
                    blocked = source_json(name[:-5]+"-blocked.json") or {}
                    failed_dossiers.append((dossier, blocked))
            method_refusal = bool(len(failed_dossiers) == 1 and len(diagnostic_ids) == 1 and diagnostic_ids[0] == diagnostic_id)
            if method_refusal:
                dossier, blocked = failed_dossiers[0]
                method_refusal = bool(dossier.get("receipt_version") == 2 and diagnostic_id in dossier.get("diagnostic_ids", []) and
                    (dossier.get("missing_evidence") or any(row.get("satisfied") is False for row in _records(dossier.get("applicability_checks")))) and
                    blocked.get("research_id") == dossier.get("research_id") and blocked.get("retrieval_status") == "research_blocked" and
                    last_error == "research_blocked: "+str(blocked.get("error")))
            prior_spent = _mapping(_mapping(original.get("ledger_usage")).get("spent"))
            anchor_spent = _mapping(_mapping(anchor.get("ledger_usage")).get("spent"))
            prior_delta = _mapping(original.get("new_usage_delta"))
            checks.update({
                "strict_seven_to_nine_source_chain": bool(
                    resumption.get("evidence_repository") == str(repo.resolve()) and intent.get("resumption") == resumption and
                    value.get("runtime_evidence_context") == {"path":"campaign/runtime-evidence-context.json",
                        "sha256":files.get("campaign/runtime-evidence-context.json")} and
                    resumption.get("controls_directory") == "acceptance-controls-resume-2" and
                    resumption.get("research_round_derivation") == {"initial_anchor_round":2,"completed_failed_round":3,"next_fresh_round":4} and
                    chain.get("source_directory") == str(Path(RESEARCH_REFRESH).parent) and
                    chain.get("evidence_repository") == str(repo.resolve()) and
                    chain.get("inherited_completed_model_calls") == 7 and chain.get("no_initial_proposal_or_diagnostic_replay") is True and
                    all(_mapping(chain.get(role)).get("path_base") == "repository" for role in ("source_receipt","source_proof_manifest")) and
                    anchor.get("kind") == "documentary_research_refresh_acceptance" and anchor.get("ok") is False and anchor.get("status") == "incomplete" and
                    anchor_proof.get("schema_version") == 1 and anchor_proof.get("hash_algorithm") == "sha256" and
                    _mapping(anchor.get("proof_manifest")).get("sha256") == _mapping(chain.get("source_proof_manifest")).get("sha256") and
                    len(anchor_calls) == 7 and original.get("inherited_model_calls") == anchor_calls and
                    original.get("resume_source_unchanged") is True and original.get("bootstrap_files_unchanged") is True and not original.get("checks") and
                    original.get("new_gpu_requests") == original.get("new_gpu_requests_attempted") == 0 and original.get("hardware_currently_observed") is False and
                    value.get("historical_bootstrap") == original.get("historical_bootstrap") == anchor.get("historical_bootstrap") and
                    prior_intent.get("resumption") == chain and prior_intent.get("source_provenance") == original.get("source_provenance")),
                "source_nine_exact_failed_round_events_and_calls": bool(
                    [row.get("kind") for row in prefix] == initial_kinds and len(prior_events) == 19 and prior_events[:16] == prefix and
                    [row.get("kind") for row in prior_events[16:]] == ["model","model","research_blocked"] and
                    anchor.get("error") == "Actual targeted research blocked: "+str(prefix_error) and
                    original.get("error") == "Actual resumed targeted research blocked: "+str(last_error) and exact_suffix and
                    [_mapping(row.get("payload")).get("call_id") for row in prior_events if row.get("kind") == "model"] == old_ids and
                    all(_mapping(row.get("payload")).get("usage") == call.get("usage") and
                        _mapping(row.get("payload")).get("purpose") == invocation.get("purpose")
                        for row, call, invocation in zip(prior_events[16:18], prior_calls, prior_invocations))),
                "source_nine_method_dossier_refusal_linked": method_refusal,
                "source_nine_ancestor_copies_and_trace_hashes": bool(all(old_files.get("source-evidence/"+target) == expected and _hash(expected)
                    for target, expected in (("previous-receipt.json",_mapping(chain.get("source_receipt")).get("sha256")),
                        ("previous-proof.json",_mapping(chain.get("source_proof_manifest")).get("sha256")),
                        ("previous-events.json",anchor_files.get("events.json")),
                        ("previous-dossier.json",anchor_files.get(initial_name)))) and
                    source_json("source-evidence/previous-receipt.json") == anchor and
                    source_json("source-evidence/previous-proof.json") == anchor_proof and
                    source_json("source-evidence/previous-events.json",expected_type=list) == prefix and
                    source_json("source-evidence/previous-dossier.json") == source_json(initial_name,anchor_file=True) and
                    all(old_files.get(name) == expected and _hash(expected) for name, expected in anchor_files.items()
                        if name.startswith("campaign/model/") or name in (initial_name,"campaign/acceptance-evidence/blocked-refusal.json",
                            "campaign/acceptance-controls/invocations.json")) and
                    all(old_files.get(name) == anchor_files.get(name) and source_json(name) is not None and
                        source_json(name) == source_json(name,anchor_file=True) for name in old_dossiers)),
                "source_nine_declared_ledger_delta": bool(prior_spent.get("model_calls") == 9 and anchor_spent.get("model_calls") == 7 and
                    prior_delta.get("model_calls") == 2 and prior_delta.get("model_tokens") ==
                        sum(_mapping(call.get("usage")).get("input_tokens",0)+_mapping(call.get("usage")).get("output_tokens",0) for call in prior_calls) and
                    prior_delta == {key:prior_spent.get(key,0)-anchor_spent.get(key,0) for key in set(prior_spent)|set(anchor_spent)}),
            })
        if inherited_count == 11:
            checks.update(_accepted_eleven_source_checks(repo,value,original,old_files,linked,files,
                resumption,diagnostic_id,technique))
    verified = all(checks.values())
    pending = (root/"run-intent.json").is_file()
    return {"ok": verified if value else None, "authentication_checks": checks,
        "status": ("AC-02 reprise documentaire vérifiée — sans nouveau GPU" if resumed else "AC-02 documentaire vérifié — sans nouveau GPU") if verified else
                  "AC-02 documentaire incomplet ou liens non authentifiés" if value else
                  "en cours ou interrompu — résultat encore inconnu" if pending else "en attente — aucun résultat documentaire",
        "job_id": None, "terminal": {}, "historical_bootstrap_job_id": value.get("historical_bootstrap", {}).get("job_id"),
        "new_gpu_execution": False, "final_qualified": False,
        "verified_json_subset": required if verified else [], "database_or_final_inputs_read": False,
        "resumption": value.get("resumption") if resumed else None,
        "observed_new_model_call_count": len(new_calls) if verified else None,
        "inherited_model_call_count": len(inherited_calls) if verified and resumed else None,
        "declared_new_ledger_usage_delta": value.get("new_usage_delta") if verified and resumed else None,
        "new_usage_delta_scope": "Delta déclaré par le ledger du contrôle ; usage fournisseur non réconcilié ici, aucun prix déduit" if resumed else None,
        "scope": (f"Reprise isolée : {len(inherited_calls)} appels précédents hérités sans répétition, nouveaux appels distincts ; "
                 "ancien reçu incomplet conservé comme autorité séparée, état de reprise dérivé explicitement des preuves publiques. "
                 "Le contexte matériel documentaire conserve API CUDA et package pilote séparés, sans compatibilité forcée. "
                 if resumed and verified else "") + ("Contrôle documentaire conduit par de vrais appels Codex/web : dossier neuf, variation compatible sans recherche, "
                 "technique non couverte refusée, diagnostic, recherche ciblée, dossier renouvelé et proposition admise. "
                 "Les gardes normales restent actives. Le manifeste SHA et son sous-ensemble JSON déclaré sont vérifiés ; "
                 "les bases/checkpoints et inputs numériques ne sont pas lus. Le bootstrap ROMEO est historique, "
                 "sans compilation, benchmark, sanitizer, nouveau GPU ni qualification de l’optimiseur §16."
                 if verified else "Protocole documentaire prévu : dossier initial, variation compatible, refus d’une technique non couverte, "
                 "diagnostic, recherche ciblée et proposition après renouvellement. Le reçu est absent, incomplet ou non authentifié ; "
                 "aucune séquence complète, proposition admise, nouvelle exécution GPU ou qualification §16 déduite.")}


def _diagnostic_v2_control(repo, value):
    """Check the archived audit's links, without repeating research or GPU work."""
    chains = _records(value.get("chains"))
    checks = {"audit_kind": value.get("kind") == "actual_convolution_diagnostic_targeted_research_read_only_audit",
              "three_measured_chains": value.get("all_three_chains_measured") is True and len(chains) == 3,
              "three_declared_modes": {c.get("mode") for c in chains} == {"full", "greedy_parent", "no_autotuning"},
              "launch_source_identified": bool(re.fullmatch(r"[0-9a-f]{40}", value.get("controller_launch_source", "")))}
    versions = _mapping(value.get("relevant_source_versions"))
    checks["frozen_source_hashes"] = bool(versions) and all(
        isinstance(v, dict) and _hash(v.get("frozen_sha256")) and
        v.get("frozen_sha256") == v.get("launch_sha256") for v in versions.values())
    for chain in chains:
        diagnostic, dossier, proposal, measurement, trial = (_mapping(chain.get(key)) for key in
            ("diagnostic", "targeted_dossier", "next_proposal", "measurement", "next_trial"))
        linked = [diagnostic, dossier, proposal, measurement, measurement.get("candidate_source"),
                  dossier.get("supplied_content"), dossier.get("web_events")]
        checks[chain.get("mode", "unknown")] = bool(
            chain.get("status") == "diagnostic_targeted_research_proposal_measured" and
            diagnostic.get("immutable_identity_and_evidence_hashes_verified") is True and
            diagnostic.get("diagnostic_id") and trial.get("diagnostic_id") == diagnostic["diagnostic_id"] and
            dossier.get("diagnostic_ids") == [diagnostic["diagnostic_id"]] and
            dossier.get("research_ready_guard_replayed") is True and
            dossier.get("all_source_bodies_passages_and_supplied_content_verified") is True and
            dossier.get("pmpp_installed_original_and_cards_verified") is True and
            proposal.get("strict_schema_parent_contract_family_and_dossier_guard_replayed") is True and
            measurement.get("ok") is True and measurement.get("status") == "measured" and
            measurement.get("candidate_artifact_recomputed") is True and
            measurement.get("all_series_owned_job_same_hardware_and_exact_proposal_source") is True and
            len(measurement.get("cases", [])) == 7 and
            all(c.get("samples", 0) > 0 for c in measurement.get("cases", [])) and
            all(_linked_receipt_verified(repo, reference) for reference in linked))
    verified = all(checks.values())
    jobs = sorted({str(c["worker"]["job_id"]) for c in chains if c.get("worker", {}).get("job_id")})
    return {"ok": verified if value else None, "authentication_checks": checks,
        "status": "trois chaînes vérifiées — mesures de recherche existantes" if verified else
                  "preuve incomplète ou liens non authentifiés" if value else "en attente — aucun reçu acquis",
        "job_id": ", ".join(jobs) or None, "job_ids": jobs,
        "source_head": value.get("controller_launch_source"),
        "execution_source_sha256": {k:v.get("frozen_sha256") for k,v in versions.items()},
        "new_gpu_execution": False,
        "scope": "Audit en lecture seule de trois chaînes diagnostic→web→dossier v2→proposition acceptée→mesure SEARCH, "
                 "7 cas par chaîne, sources figées b9bda60. Authentification et gardes déjà rejouées dans le reçu ; "
                 "liens SHA revérifiés pour le rapport. Profilage indisponible conservé. Aucune nouvelle exécution GPU, "
                 "aucune finale commune ni qualification globale déduite."}


def _evaluation_recovery_control(repo, value, number):
    """Project only public control metadata; never follow the proof inventory."""
    base = "results/evaluation-recovery-functional-control-"+number
    scope = ("Contrôle fonctionnel statique du §11.2, sans modèle et hors cohortes §16 ; "
             "aucune qualification FINAL ni aucun gain scientifique. Les preuves d’implémentation, "
             "tests locaux et exécution ROMEO restent distinctes. ")
    result = {"control_number": number, "functional_acceptance": None, "ok": None,
              "final_qualified": False, "scientific_gain_qualified": False,
              "excluded_from_quality_cohorts": True, "model_calls": None,
              "jobs": [], "allocated_gpu_wall_seconds": None, "old_work_actual_cost": None,
              "old_work_cost_status": "inconnu", "validation_level": "inconnu",
              "scope": scope+"Reçu absent : résultat encore inconnu.", "verified_public_subset": []}
    if not value:
        return result
    if number == "001":
        verified = (value.get("kind") == "isolated_recovery_control_pre_submission_failure"
                    and value.get("status") == "incomplete_retained" and value.get("returncode") == 2
                    and type(value.get("recorded_owned_jobs")) is int and value["recorded_owned_jobs"] == 0
                    and type(value.get("phase_launch_intents")) is int and value["phase_launch_intents"] == 0
                    and type(value.get("model_calls")) is int and value["model_calls"] == 0
                    and value.get("artifacts_retained") is True and value.get("retry_same_output") is False)
        result.update(ok=False, functional_acceptance=False, evidence_verified=verified,
                      status="incomplet — erreur du helper avant soumission" if verified else "reçu invalide — échec non authentifié",
                      source_head=value.get("runtime_source_head"), source_status=value.get("status"),
                      validation_level="échec avant soumission" if verified else "inconnu",
                      model_calls=0 if verified else None, allocated_gpu_wall_seconds=0 if verified else None,
                      scope=scope+("Échec du helper, aucune allocation atteinte ; aucun succès de reprise. "
                                   "Ce résultat historique reste inchangé." if verified else
                                   "Forme du reçu non reconnue ; aucune exécution déduite."))
        return result

    authority = _verified_json(repo, value.get("authority"), exact=base+"/evaluation-recovery-acceptance.json") or {}
    proof = _verified_json(repo, value.get("proof_manifest"), exact=base+"/proof-sha256.json") or {}
    files = _mapping(proof.get("files"))
    public_names = ["uncertain-old-cost-observation.json", "control-proof/phase-1-placement.json",
                    "control-proof/phase-2-placement.json"]
    if number == "003":
        public_names.append("control-proof/replay-observation.json")
    linked = {name: _verified_json(Path(repo)/base, {"path": name, "sha256": files.get(name)}, exact=name)
              for name in public_names}
    old_cost = _mapping(linked.get(public_names[0]))
    jobs = _records(authority.get("jobs"))
    jobs_valid = (len(jobs) == 2 and [job.get("phase") for job in jobs] == [1, 2]
                  and len({job.get("job_id") for job in jobs}) == 2
                  and len({job.get("session") for job in jobs}) == 2)
    for job, name in zip(jobs, public_names[1:3]):
        placement = _mapping(linked.get(name))
        terminal = _mapping(job.get("terminal"))
        jobs_valid = jobs_valid and bool(
            isinstance(job.get("job_id"), str) and re.fullmatch(r"[1-9][0-9]*", job["job_id"])
            and isinstance(job.get("session"), str) and job["session"]
            and re.fullmatch(r"gpu(?::[A-Za-z0-9_]+)?:1", str(job.get("allocated_gres")))
            and terminal.get("job_id") == job["job_id"] and terminal.get("terminal") is True
            and terminal.get("state") == "COMPLETED" and terminal.get("exit_code") == "0:0"
            and placement.get("job_id") == job["job_id"] and placement.get("session_id") == job["session"]
            and placement.get("gres") == job["allocated_gres"]
            and placement.get("actual_hostname") == terminal.get("nodes")
            and _hash(placement.get("device_uuid"), 32))
    placements = [_mapping(linked.get(name)) for name in public_names[1:3]]
    same_uuid = bool(jobs_valid and placements[0].get("device_uuid") == placements[1].get("device_uuid"))
    checks = {
        "authority_and_public_proof_links": bool(authority and proof and all(linked.values())),
        "static_no_model_outside_quality": authority.get("kind") == "evaluation_recovery_acceptance"
            and authority.get("control_method") == "functional_static_no_model"
            and authority.get("excluded_from_quality_cohorts") is True
            and type(authority.get("model_calls")) is int and authority["model_calls"] == 0,
        "two_distinct_owned_terminal_one_gpu_allocations": jobs_valid and same_uuid,
        "uncertain_old_work_preserved": authority.get("uncertain_old_work_preserved") is True
            and old_cost.get("uncertain_preserved") is True and old_cost.get("old_attempt_status") == "running"
            and "old_reservation_actual" in old_cost and old_cost["old_reservation_actual"] is None,
    }
    if number == "003":
        replay = _mapping(linked.get("control-proof/replay-observation.json"))
        audit_replay = _mapping(value.get("replay"))
        pairs = (("gpu_rpc_calls_before_evaluation_replay", "gpu_rpc_calls_after_evaluation_replay"),
                 ("gpu_rpc_calls_before_replay", "gpu_rpc_calls_after_replay"),
                 ("tree_rewards_before_replay", "tree_rewards_after_replay"))
        checks.update({
            "required_authority_and_audit_checks": set(_mapping(authority.get("checks"))) == set(RECOVERY_ACCEPTANCE_CHECKS)
                and _required_checks(authority, RECOVERY_ACCEPTANCE_CHECKS)
                and _required_checks(value, RECOVERY_AUDIT_CHECKS),
            "successful_separate_authority": authority.get("ok") is True
                and authority.get("status") == "evaluation_recovery_acceptance_passed"
                and not authority.get("error") and not authority.get("cleanup_error"),
            "audit_source_and_job_bindings": value.get("kind") == "independent_public_terminal_audit"
                and value.get("jobs") == jobs and value.get("uuid") == placements[0].get("device_uuid")
                and value.get("old_uncertain_cost_preserved") is True
                and value.get("excluded_from_quality_cohorts") is True
                and type(value.get("model_calls")) is int and value["model_calls"] == 0
                and _hash(authority.get("source_head"), 40)
                and authority.get("source_head") == _mapping(value.get("source")).get("head")
                and _hash(authority.get("helper_sha256"))
                and authority.get("helper_sha256") == _mapping(value.get("source")).get("helper_sha256"),
            "replay_has_no_new_rpc_or_tell": all(
                type(replay.get(before)) is int and type(replay.get(after)) is int
                and type(audit_replay.get(before)) is int and type(audit_replay.get(after)) is int
                and replay[before] > 0 and replay[after] == replay[before]
                and audit_replay[before] == replay[before] and audit_replay[after] == replay[after]
                for before, after in pairs)
                and replay.get("tree_rewards_before_replay") == 1
                and type(replay.get("model_calls")) is int and replay["model_calls"] == 0
                and replay.get("request_id") == value.get("new_request_id")
                and replay.get("previous_request_id") == value.get("old_request_id")
                and _hash(replay.get("request_id")) and _hash(replay.get("previous_request_id"))
                and replay["request_id"] != replay["previous_request_id"]
                and _hash(value.get("new_measured_evaluation_id"), 32)
                and _hash(value.get("old_measured_evaluation_id"), 32)
                and replay.get("evaluation_id") == value.get("new_measured_evaluation_id")
                and value.get("new_measured_evaluation_id") != value.get("old_measured_evaluation_id"),
        })
    else:
        checks["historical_failure_remains_incomplete"] = (
            value.get("kind") == "evaluation_recovery_functional_control_terminal_audit"
            and value.get("status") == "incomplete_contract_resume_refused"
            and value.get("recovery_acceptance_ok") is False and authority.get("ok") is False
            and authority.get("status") == "incomplete" and bool(authority.get("error"))
            and _required_checks(authority, ("old_cost_remains_uncertain", "preparation_submits_no_job")))
    verified = all(checks.values())
    durations = [_elapsed_seconds(_mapping(job.get("terminal")).get("elapsed")) for job in jobs]
    allocated = sum(durations) if verified and len(durations) == 2 and all(v is not None for v in durations) else None
    accepted = verified and number == "003"
    result.update(ok=accepted, functional_acceptance=accepted, evidence_verified=verified,
                  status=("acquis — reprise fonctionnelle statique §11.2 sans modèle" if accepted else
                          "incomplet — contrat refusé après copie froide" if verified else
                          "reçu invalide — contrôle non authentifié"),
                  source_status=authority.get("status"), authority=value.get("authority"),
                  authentication_checks=checks, source_head=authority.get("source_head"),
                  execution_source_sha256={"helper": authority.get("helper_sha256")},
                  jobs=jobs if verified else [], job_id=", ".join(job["job_id"] for job in jobs) if verified else None,
                  terminal={"state": "COMPLETED — deux allocations distinctes"} if verified else {},
                  allocated_gpu_wall_seconds=allocated, model_calls=0 if verified else None,
                  old_work_cost_status="incertain, jamais ramené à zéro" if verified else "inconnu",
                  validation_level="ROMEO — contrôle statique" if verified else "inconnu",
                  verified_public_subset=[name for name in public_names if linked.get(name)],
                  scope=scope+("Deux allocations distinctes au même UUID, ancien travail non commis à coût incertain. "
                               "Les durées Slurm sont séparées du temps CUDA et du ledger ; aucun ajout au bilan global. "
                               "Le même UUID futur reste une condition, sans garantie de réallocation. "
                               + ("Rejeu sans nouvelle requête ni nouveau tell acquis ; les échecs 001/002 restent inchangés. "
                                  "La reprise avec agent modèle, les autres placements et les campagnes §16 restent des preuves séparées."
                                  if accepted else "Le hash lié aux chemins bloque la suite avant toute nouvelle requête native ; aucune reprise complète acquise.")
                               if verified else "Liens publics ou contrôles incomplets ; aucune réussite déduite."))
    return result


def control_receipts(repo):
    receipts = []
    for name, relative in PROOF_PATHS:
        path = Path(repo) / relative
        malformed = None
        unsafe_audit = False
        try:
            if relative in RECOVERY_AUDITS:
                for component in (path, *path.parents):
                    if component == Path(repo):
                        break
                    if component.is_symlink():
                        unsafe_audit = True
                        raise ValueError("Lien symbolique dans le chemin du contrôle")
            raw_value = load(path, {})
            if not isinstance(raw_value, dict):
                malformed = "Le reçu doit être un objet JSON"
            value = _mapping(raw_value)
        except (ValueError, OSError) as error:
            value, malformed = {}, type(error).__name__
        terminal = _mapping(value.get("terminal") or value.get("worker_terminal"))
        provenance = _mapping(value.get("source_provenance") or value.get("execution_source"))
        row = {"label": name, "path": relative,
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not unsafe_audit else None,
            "status": value.get("status", "reçu disponible" if value else "en attente"),
            "ok": value.get("ok"), "job_id": terminal.get("job_id", value.get("job_id")),
            "terminal": terminal, "source_head": provenance.get("HEAD", value.get("source_HEAD")),
            "source_dirty_state": provenance.get("dirty_state", value.get("snapshot_dirty_after")),
            "execution_source_sha256": provenance.get("sources", provenance.get("source_sha256", value.get("execution_source_sha256", value.get("source_sha256_map", {})))),
            "sources_unchanged": value.get("sources_unchanged"), "available": path.is_file(),
            "scope": value.get("scope", value.get("evidence_kind")),
            "criteria_checks": value.get("criteria_checks")}
        authenticators = {ORIGINAL_EXECUTION: lambda: _original_execution_control(repo, value),
            COMMON_FINAL_FREEZE: lambda: _common_freeze_control(repo, value),
            RESEARCH_REFRESH: lambda: _research_refresh_control(repo, value, path),
            RESEARCH_REFRESH_RESUMED: lambda: _research_refresh_control(repo, value, path, resumed=True),
            RESEARCH_REFRESH_RESUMED_2: lambda: _research_refresh_control(repo, value, path, resumed=True),
            RESEARCH_REFRESH_RESUMED_3: lambda: _research_refresh_control(repo, value, path, resumed=True),
            STANDALONE_FINAL_SUITE: lambda: _standalone_final_control(value),
            DIAGNOSTIC_V2_AUDIT: lambda: _diagnostic_v2_control(repo, value)}
        if relative in RECOVERY_AUDITS:
            authenticators[relative] = lambda: _evaluation_recovery_control(repo, value, RECOVERY_AUDITS[relative])
        if relative in authenticators:
            try:
                row.update(authenticators[relative]())
            except (TypeError, AttributeError, KeyError, IndexError, ValueError, OSError) as error:
                # Unknown nested JSON shapes are invalid evidence, not a reason
                # to lose the report or silently qualify the remaining fields.
                malformed = "Structure de preuve invalide : "+type(error).__name__
        elif relative == PARENT_OBSERVATION:
            authority = _mapping(value.get("authority"))
            primary = _mapping(authority.get("primary_receipt"))
            row.update(ok=None, authority=authority,
                authority_receipt_verified=_linked_receipt_verified(path.parent, primary, suffix=".json", exact=Path(POLICY_AUTHORITY).name),
                scope="Observation secondaire d’un parent réellement plus lent, entrée générateur exacte. "
                      "Aucune nouvelle mesure. Le contrôle strict prédéclaré AC-05 et l’autorité du helper restent faux ; "
                      "cette observation ne les requalifie pas.")
        elif relative == POLICY_AUTHORITY and value:
            mark = lambda criterion: "acquis" if _mapping(value.get("criteria_checks")).get(criterion) is True else (
                "faux" if _mapping(value.get("criteria_checks")).get(criterion) is False else "inconnu")
            row["scope"] = (f"Autorité du contrôle statique : AC-05 {mark('AC-05')}, AC-06 {mark('AC-06')}, "
                            f"AC-10 {mark('AC-10')}. Le statut du helper reste distinct du terminal Slurm. Hors cohortes §16.")
        elif "diagnostic-loop-v2-20261021/" in relative and value:
            row["scope"] = ("Helper 773095 incomplet : diagnostic, recherche ciblée et proposition acceptée ; "
                            "compilation math.h échouée, sans exécution/mesure/intégration du quatrième candidat. "
                            "Le terminal Slurm COMPLETED ne valide pas la boucle.")
        if malformed:
            row.update(ok=False, status="reçu invalide — aucune qualification déduite", authentication_error=malformed,
                       scope="Preuve JSON absente de la forme attendue ; aucune exécution réussie ni séquence complète déduite.",
                       artifact_aliases=[], verified_json_subset=[], final_qualified=False)
        receipts.append(row)
    return receipts


def execution_evidence_panel(receipts):
    """Identity and documentary controls remain separate from measured scores."""
    by_path = {row["path"]: row for row in receipts}
    original = by_path.get(ORIGINAL_EXECUTION, {})
    frozen = by_path.get(COMMON_FINAL_FREEZE, {})
    refresh = by_path.get(RESEARCH_REFRESH, {})
    resumed = by_path.get(RESEARCH_REFRESH_RESUMED, {})
    html = "<h2>Original exécuté, gel des candidats et contrôle documentaire</h2>"
    html += "<p>"+escape(original.get("status", "Original : inconnu"))+". "
    if original.get("ok") is True:
        usage = original["usage"]
        html += (f"Job {escape(str(original['job_id']))}, commit original <code>{UPSTREAM_COMMIT}</code> : "
                 f"{fmt(usage.get('model_tokens'), 9)} tokens observés, {fmt(usage.get('model_calls'))} invocations CLI ; "
                 f"observateur séparé {fmt(usage.get('gain_observer_wall_seconds'))} s. ")
    html += "La génération et les cas SEARCH ne remplacent pas une comparaison finale commune.</p>"
    html += "<p>"+escape(frozen.get("status", "Gel : inconnu"))+". "
    html += ("Un gel authentifié prouve l’identité des artefacts ; il ne prouve ni succès des méthodes, "
             "ni qualification ou gain final. Les identités exactes ne constituent aucune différence de performance.</p>")
    aliases = frozen.get("artifact_aliases", [])
    if aliases:
        rows = [[LABELS.get(entry["mode"], entry["mode"]),
                 "identique au naïf" if entry["same_artifact_as_naive"] else
                 "même source, lancement/paramètres distincts" if entry["same_source_as_naive"] else "source distincte",
                 entry["artifact_hash"]] for entry in aliases]
        html += table(["Méthode gelée", "Relation au noyau naïf", "Identité complète de l’artefact"], rows)
    refresh_scope = ("Le reçu documentaire authentifie des appels Codex/web et le renouvellement complet du dossier. "
                     if refresh.get("ok") is True else
                     "Le protocole prévoit de vrais appels Codex/web ; les preuves disponibles ne permettent pas d’en déduire la séquence complète. ")
    html += ("<p>"+escape(refresh.get("status", "AC-02 : résultat inconnu"))+". "+refresh_scope+
             "Aucune nouvelle exécution GPU, compilation, mesure ou qualification §16 n’en est déduite. "
             "Un reçu absent ou incomplet reste inconnu ou incomplet.</p>")
    html += "<p>Autorité séparée de reprise AC-02 : "+escape(resumed.get("status", "résultat encore inconnu"))+". "
    if resumed.get("ok") is True:
        html += (f"{fmt(resumed.get('inherited_model_call_count'))} anciens appels hérités et préservés, "
                 f"{fmt(resumed.get('observed_new_model_call_count'))} nouveaux appels observés ; "
                 "aucun appel initial ou diagnostic rejoué. Le reçu initial incomplet garde son statut. ")
    else:
        html += "Les preuves ne permettent pas de conclure au renouvellement complet par la reprise ; le reçu initial garde son statut. "
    html += "Ce contrôle reste documentaire : aucun nouveau GPU ni résultat de performance §16.</p>"
    resumed2 = by_path.get(RESEARCH_REFRESH_RESUMED_2, {})
    html += "<p>Deuxième autorité séparée de reprise AC-02 : "+escape(resumed2.get("status", "résultat encore inconnu"))+". "
    if resumed2.get("ok") is True:
        html += (f"Chaîne authentifiée 7 → 9 : {fmt(resumed2.get('inherited_model_call_count'))} appels hérités, "
                 f"{fmt(resumed2.get('observed_new_model_call_count'))} nouveaux appels observés ; "
                 "les deux autorités précédentes incomplètes conservent leur statut. ")
    else:
        html += "Aucune réussite de cette deuxième reprise n’est déduite ; les deux autorités précédentes gardent leur statut. "
    html += "Contrôle documentaire sans nouveau GPU ni qualification de performance.</p>"
    resumed3 = by_path.get(RESEARCH_REFRESH_RESUMED_3, {})
    html += "<p>Autorité séparée du complément après recherche acceptée : "+escape(resumed3.get("status", "résultat encore inconnu"))+". "
    if resumed3.get("ok") is True:
        html += (f"{fmt(resumed3.get('inherited_model_call_count'))} appels hérités ; recherche acceptée au round 4 conservée, "
                 "complément explicite au round 5 puis proposition effectivement admise. "
                 "Le dossier précédent sans couverture exacte et les trois autorités incomplètes restent inchangés. ")
    else:
        html += ("Aucun succès de ce complément n’est acquis. Une recherche peut soutenir une alternative sans couvrir le nom exact demandé ; "
                 "son acceptation ne termine pas ce contrôle AC-02. ")
    html += "Aucun nouveau GPU ni résultat de performance §16 n’en est déduit.</p>"
    return html+evaluation_recovery_panel(receipts)


def evaluation_recovery_panel(receipts):
    rows = [row for row in receipts if row.get("path") in RECOVERY_AUDITS]
    if not rows:
        return ""
    cells = [[row.get("control_number", RECOVERY_AUDITS[row["path"]]), row["status"], row.get("validation_level", "inconnu"),
              row.get("job_id") or ("aucune allocation atteinte" if row.get("evidence_verified") else "inconnu"),
              fmt(row.get("allocated_gpu_wall_seconds")), row.get("old_work_cost_status", "inconnu")]
             for row in rows]
    return ("<h2>Reprise d’évaluation — contrôles fonctionnels du §11.2</h2>"
            +table(["Contrôle", "Résultat fonctionnel", "Preuve exécutée", "Allocations distinctes",
                    "Secondes allouées Slurm (un GPU par allocation)", "Ancien coût du travail non commis"], cells)
            +"<p>Contrôle statique sans modèle, hors cohortes §16 : aucune génération Codex vanilla, "
             "aucun gain scientifique et aucune qualification FINAL. Implémentation et tests locaux ont leurs reçus séparés. "
             "Les secondes additionnent seulement les durées terminales des allocations indiquées ; "
             "elles ne sont ni du temps CUDA ni un coût ancien régularisé et ne s’ajoutent pas au bilan global déjà dédupliqué. "
             "Les contrôles 001/002 conservent leurs statuts ; un succès séparé ne les requalifie pas. "
             "Le même UUID reste obligatoire pour une future reprise, sans garantie Slurm.</p>")


def _receipt_snapshot(path):
    body = Path(path).read_bytes()
    return json.loads(body), {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}


def _reported_tokens(value, kind):
    usage = value.get("usage") or {}
    if not isinstance(usage, dict):
        raise ValueError("Malformed model usage receipt")
    tokens = {key: usage[key] for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
              if usage.get(key) is not None}
    for key, amount in tokens.items():
        if type(amount) is not int or amount < 0:
            raise ValueError("Invalid observed token count: " + key)
    for subset, total in (("cached_input_tokens", "input_tokens"), ("reasoning_output_tokens", "output_tokens")):
        if subset in tokens and total in tokens and tokens[subset] > tokens[total]:
            raise ValueError("Observed token subset exceeds its total: " + subset)
    # The native driver initializes zero counters before its first provider
    # event. Those counters are not a provider report of zero billed tokens.
    if not any(tokens.values()) and (value.get("status") != "completed" or
            (kind == "native" and not value.get("usage_events"))):
        tokens = {}
    complete = value.get("status") == "completed" and all(k in tokens for k in ("input_tokens", "output_tokens"))
    return tokens, complete


def _elapsed_seconds(value):
    """Parse recorded sacct Elapsed, never controller/ledger wall estimates."""
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?", value)
    if not match:
        return None
    days, first, second, third = match.groups()
    if days is not None and third is None:
        return None
    if third is None:  # MM:SS
        return int(first)*60 + int(second) if int(second) < 60 else None
    hours, minutes, seconds = int(first), int(second), int(third)
    if minutes >= 60 or seconds >= 60 or (days is not None and hours >= 24):
        return None
    return int(days or 0)*86400 + hours*3600 + minutes*60 + seconds


def gpu_lease_consumption(results):
    """Read authenticated local session/accounting pairs; never contact Slurm."""
    jobs, orphans = {}, []
    terminal_states = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
                       "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
    for path in sorted(Path(results).rglob("slurm-session.json")):
        session, origin = _receipt_snapshot(path)
        job = str(session.get("job_id") or "")
        if not re.fullmatch(r"\d+", job):
            orphans.append({**origin, "reason": "session has no valid job_id"})
            continue
        config = session.get("config") or {}
        session_id, gres = session.get("session_id"), config.get("gres")
        valid = isinstance(session_id, str) and bool(session_id) and bool(re.fullmatch(r"gpu(?::[A-Za-z0-9_-]+)?:1", str(gres)))
        identity = (session_id, gres, config.get("host"), config.get("account"))
        row = jobs.setdefault(job, {"job_id": job, "identity": identity, "provenance": [], "terminal": None,
                                    "unknown_reasons": [], "one_gpu_session_verified": valid})
        if row["identity"] != identity:
            raise ValueError("Conflicting Slurm session identity for job " + job)
        row["provenance"].append(origin)
        if not valid:
            row["unknown_reasons"].append("session_id or single-GPU gres not verified")
        accounting = path.parent/"slurm-diagnostics/accounting.json"
        if not accounting.is_file():
            row["unknown_reasons"].append("terminal accounting unavailable")
            continue
        receipt, source = _receipt_snapshot(accounting)
        row["provenance"].append(source)
        if str(receipt.get("job_id")) != job:
            raise ValueError("Slurm accounting/session job_id mismatch for " + job)
        state = str(receipt.get("state", "")).split(" ")[0].rstrip("+")
        seconds = _elapsed_seconds(receipt.get("elapsed"))
        if state not in terminal_states or seconds is None or not valid:
            row["unknown_reasons"].append("nonterminal state, unparsed elapsed or unverified GPU count")
            continue
        observation = {"state": state, "elapsed": receipt["elapsed"], "allocated_gpu_seconds": seconds}
        if row["terminal"] is not None and any(row["terminal"][key] != observation[key]
                for key in ("state", "allocated_gpu_seconds")):
            raise ValueError("Conflicting terminal Slurm accounting for job " + job)
        row["terminal"] = observation
    for path in sorted(Path(results).rglob("slurm-diagnostics/accounting.json")):
        if not (path.parent.parent/"slurm-session.json").is_file():
            value, origin = _receipt_snapshot(path)
            orphans.append({**origin, "job_id": value.get("job_id"), "reason": "no sibling session authenticating this allocation"})
    known = [row for row in jobs.values() if row["terminal"] is not None]
    return {"observed_terminal_allocated_gpu_seconds": sum(row["terminal"]["allocated_gpu_seconds"] for row in known),
        "observed_terminal_job_count": len(known),
        "nonterminal_or_unreported_jobIDs": sorted({job for job, row in jobs.items() if row["terminal"] is None} |
            {str(row["job_id"]) for row in orphans if re.fullmatch(r"\d+", str(row.get("job_id", "")))}),
        "gpu_lease_provenance": [{key: value for key, value in row.items() if key != "identity"} for row in jobs.values()],
        "unpaired_gpu_lease_receipts": orphans,
        "gpu_lease_scope": "Observed lower bound from unique terminal Slurm jobs with matching session and one-GPU gres. Includes model/research/idle during the allocation; excludes queue time. Active/missing/unparsed leases remain unknown. Ledger GPU estimates are not used as allocated wall time."}


def project_consumption(results):
    """Known totals are lower bounds; missing provider usage is never a zero bill."""
    snapshots, aliases, parents = [], {}, []
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def remember(kind, value, origin, keys, basis):
        index = len(snapshots)
        parents.append(index)
        snapshots.append({"kind": kind, "value": value, "origin": origin, "aliases": keys, "identity_basis": basis})
        for key in keys:
            if key in aliases:
                parents[find(index)] = find(aliases[key])
            aliases[key] = index
    # The original adapter now isolates each Codex invocation below its durable
    # prompt intent. ACKs/calls.json are transport records, not extra CLI calls.
    structured_paths = {}
    for pattern, layout in (("model/*.json", "model"),
                            ("model-transport/intents/*/*/cli/*.json", "original_intent_cli")):
        for path in Path(results).rglob(pattern):
            structured_paths[path] = layout
    for path, layout in sorted(structured_paths.items()):
        value, origin = _receipt_snapshot(path)
        origin["receipt_layout"] = layout
        identity = value.get("call_id") or path.stem
        if not re.fullmatch(r"[0-9a-f]{32}", str(identity)):
            if re.fullmatch(r"[0-9a-f]{32}", path.stem):
                raise ValueError("Invalid structured model call identity: "+str(path))
            continue
        remember("structured", value, origin, ["structured:"+identity], "call_id")
    for path in sorted(Path(results).rglob("evidence/receipt.json")):
        value, origin = _receipt_snapshot(path)
        if not value.get("model_started"):
            continue
        keys, basis = [], []
        for key in ("requested_thread_id",):
            if value.get(key):
                keys.append("native-thread:"+str(value[key])); basis.append(key)
        threads = value.get("thread_ids") or []
        if not isinstance(threads, list) or any(not isinstance(thread, str) or not thread for thread in threads):
            raise ValueError("Malformed native thread identities: "+str(path))
        for thread in sorted(set(threads)):
            keys.append("native-thread:"+str(thread)); basis.append("observed_thread_id")
        session = value.get("worker_session", value.get("session_id"))
        if isinstance(session, str) and session and type(value.get("child_pid")) is int and value["child_pid"] > 0 and value.get("started_at") is not None:
            keys.append("native-invocation:"+json.dumps([session, value["child_pid"], value["started_at"]], separators=(",", ":")))
            basis.append("session_pid_started_at")
        if not keys:
            raise ValueError("Cannot deduplicate model-started native receipt without thread or session/pid/start identity: "+str(path))
        remember("native", value, origin, keys, ",".join(sorted(set(basis))))
    groups = {}
    for index, snapshot in enumerate(snapshots):
        groups.setdefault(find(index), []).append(snapshot)
    totals = {key: 0 for key in ("input_tokens", "cached_input_tokens", "output_tokens")}
    provenance = []
    for group in groups.values():
        merged, complete, all_aliases = {}, False, sorted({key for item in group for key in item["aliases"]})
        identity = all_aliases[0]
        for item in group:
            tokens, done = _reported_tokens(item["value"], item["kind"])
            for key, amount in tokens.items():
                if key in merged and merged[key] != amount:
                    raise ValueError("Conflicting observed usage for the same model call: "+identity)
                merged[key] = amount
            complete |= done
        # Validate subsets again after merging complementary receipt fields.
        for subset, total in (("cached_input_tokens", "input_tokens"), ("reasoning_output_tokens", "output_tokens")):
            if subset in merged and total in merged and merged[subset] > merged[total]:
                raise ValueError("Observed token subset exceeds its total: "+identity)
        for key in totals:
            totals[key] += merged.get(key, 0)
        origins = [{**item["origin"], "status": item["value"].get("status"),
                    "returncode": item["value"].get("returncode"),
                    "has_error": bool(item["value"].get("error") or item["value"].get("stderr"))} for item in group]
        provenance.append({"identity": identity, "aliases": all_aliases, "kind": group[0]["kind"],
            "identity_bases": sorted({item["identity_basis"] for item in group}), "receipts": origins,
            "usage_reporting": "complete" if complete else "partial" if merged else "unreported",
            "observed_usage": merged, "missing_usage_fields": [key for key in totals if key not in merged],
            "cli_metadata": [{**item["origin"], "call_id": item["value"].get("call_id"),
                "purpose": item["value"].get("purpose"), "elapsed_seconds": item["value"].get("elapsed_seconds"),
                "payload_sha256": hashlib.sha256(json.dumps(item["value"]["payload"], sort_keys=True).encode()).hexdigest()
                    if isinstance(item["value"].get("payload"), dict) else None} for item in group]})
    totals.update(cli_call_receipts=len(groups), unreported_call_usage=sum(row["usage_reporting"] != "complete" for row in provenance),
        calls_without_any_reported_usage=sum(row["usage_reporting"] == "unreported" for row in provenance),
        partly_reported_call_usage=sum(row["usage_reporting"] == "partial" for row in provenance),
        calls_without_reported_cached_usage=sum("cached_input_tokens" in row["missing_usage_fields"] for row in provenance),
        call_provenance=provenance, model_tokens=totals["input_tokens"]+totals["output_tokens"], cost_usd=None,
        observed_lower_bounds=True,
        scope="All retained CLI-call receipts including original intent-scoped CLI traces, failures, pilots, diagnostics and copied archives; deduplicated by structured call ID or linked native invocation/thread identities. Original transport ACKs/calls.json are not additional calls. Missing/partial usage remains unknown, not a zero invoice.",
        cached_tokens_included_in_input=True, reasoning_tokens_included_in_output=True)
    totals["research_phase_receipts"] = _research_phase_receipts(results)
    totals.update(gpu_lease_consumption(results))
    return totals


def frozen_pilot_summary(path):
    """Keep this four-source pilot outside every section-16 cohort/selection."""
    path = Path(path)
    receipt = load(path, {})
    result = {"scope": {"kind": "independent_frozen_gemm_pilot", "operations": ["matmul"],
              "included_in_section16_cohorts": False, "artifact_selection": "raw_frozen_sources",
              "retained_best_substitution": False, "original_upstream_present": False,
              "artifact_roles": list(FROZEN_ROLES), "inference_scope": "These four fixed artifacts and these final cases only"},
              "source_receipt": str(path), "source_receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
              "receipt_verified": False, "status": receipt.get("status", "receipt_missing"),
              "terminal": receipt.get("terminal", {}), "job_id": receipt.get("job_id"),
              "contract_hash": receipt.get("contract_hash"), "hardware_identity": receipt.get("hardware_identity", {}),
              "shapes": receipt.get("shapes", []), "protocol": receipt.get("protocol", {}),
              "generation_boundary": receipt.get("generation_boundary"),
              "model_calls_during_final": receipt.get("model_calls"), "sources": [], "comparisons": {}, "validation_errors": []}
    errors = result["validation_errors"]
    if not receipt:
        errors.append("Reçu du pilote absent")
        return result
    if receipt.get("kind") != "independent_frozen_kernel_comparison" or receipt.get("ok") is not True:
        errors.append("Reçu de comparaison figée non achevé ou invalide")
    terminal = result["terminal"]
    if terminal.get("state") != "COMPLETED" or terminal.get("exit_code") != "0:0" or terminal.get("job_id") != result["job_id"]:
        errors.append("Terminal Slurm réussi non confirmé pour ce job")
    candidates = receipt.get("candidates", {})
    if set(candidates) != set(FROZEN_ROLES):
        errors.append("Les quatre rôles figés attendus ne correspondent pas au reçu")
    for role, candidate in candidates.items():
        if role not in FROZEN_ROLES:
            continue
        if (not candidate.get("source") or hashlib.sha256(candidate["source"].encode()).hexdigest() != candidate.get("source_hash")
                or not candidate.get("artifact_hash")):
            errors.append("Source figée non authentifiée : " + role)
        complete = receipt.get("validations", {}).get(role, {}).get("complete", {})
        sanitized = receipt.get("sanitizer_qualification", {}).get(role, {})
        if complete.get("ok") is not True or sanitized.get("ok") is not True:
            errors.append("Validation complète ou sanitizers non qualifiés : " + role)
        result["sources"].append({"role": role, "label": FROZEN_ROLES[role], "candidate_id": candidate.get("candidate_id"),
            "source_hash": candidate.get("source_hash"), "artifact_hash": candidate.get("artifact_hash"),
            "generation_kind": candidate.get("provenance", {}).get("kind"),
            "correctness_ok": complete.get("ok"), "sanitizers_ok": sanitized.get("ok")})
    if candidates.get("native_codex", {}).get("provenance", {}).get("kind") != "native_codex_one_prompt":
        errors.append("Le rôle vanilla natif ne prouve pas un véritable appel Codex CLI")
    if candidates.get("single_prompt", {}).get("provenance", {}).get("kind") != "generation_only_structured_single_prompt":
        errors.append("La proposition structurée sans outils est mal identifiée")
    for name, (role, reference) in FROZEN_PAIRS.items():
        pair = receipt.get("paired_comparisons", {}).get(name, {})
        interval = _mapping(pair.get("interval"))
        view = comparison_view(pair)
        values = [interval.get(key) for key in ("gain", "lower", "upper")]
        valid_interval = all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in values)
        if view["gain_basis"] == "unproven_artifact_context":
            errors.append("Contexte d’exécution non prouvé : " + name)
        elif view["gain_basis"] != "exact_artifact_identity" and (not valid_interval or not values[1] <= values[0] <= values[2]):
            errors.append("Intervalle apparié manquant ou invalide : " + name)
        candidate, incumbent = candidates.get(role, {}), candidates.get(reference, {})
        if (pair.get("ok") is not True or pair.get("candidate_role") != role or pair.get("reference_role") != reference
                or pair.get("artifact_hash") != candidate.get("artifact_hash")
                or pair.get("incumbent_artifact_hash") != incumbent.get("artifact_hash")
                or pair.get("contract_hash") != result["contract_hash"]):
            errors.append("Identité candidate/référence/contrat différente : " + name)
        if any(pair.get("environment", {}).get(key) != value for key, value in result["hardware_identity"].items()):
            errors.append("Environnement différent du GPU final commun : " + name)
        result["comparisons"][name] = {key: pair.get(key) for key in (
            "evaluation_id", "status", "candidate_role", "reference_role", "gain_direction", "critical_cases_ok")}
        result["comparisons"][name]["gain_basis"] = view["gain_basis"]
        result["comparisons"][name]["source_interval_preserved"] = interval
        result["comparisons"][name]["interval"] = ({key: view[key] for key in ("gain", "lower", "upper", "case_gains")}
            if view["gain_basis"] in ("exact_artifact_identity", "unproven_artifact_context") else interval)
    result["receipt_verified"] = not errors
    return result


def plot_frozen_pilot(pilot, output):
    if not pilot["receipt_verified"]:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax, direct_ax = axes
    colors = {"agent": "#b04c42", "single_prompt": "#586f7c", "native_codex": "#167c80"}
    ax.scatter(1, 0, marker="D", color="#71828d", s=35)
    ax.annotate("1 : référence", (1, 0), xytext=(8, 8), textcoords="offset points", fontsize=9)
    for y, role in enumerate(("agent", "single_prompt", "native_codex"), 1):
        pair = pilot["comparisons"][role + "_vs_baseline"]
        interval = pair["interval"]
        gain, low, high = (interval[key] for key in ("gain", "lower", "upper"))
        exact = pair.get("gain_basis") == "exact_artifact_identity"
        if exact:
            ax.scatter(gain, y, marker="D", color=colors[role])
        else:
            ax.errorbar(gain, y, xerr=[[gain-low], [high-gain]], fmt="o", color=colors[role], capsize=4)
        ax.annotate("1 : identité exacte" if exact else f"{gain:.3f} [{low:.3f} ; {high:.3f}]", (gain, y), xytext=(0, 12),
                    textcoords="offset points", fontsize=9, ha="center")
    ax.set_yticks(range(4), list(FROZEN_ROLES.values()))
    ax.invert_yaxis()
    ax.set_ylim(3.6, -.65)
    ax.set_xlim(.67, 1.24)
    ax.set_xlabel("Latence naïve / latence du kernel figé")
    ax.set_title("Quatre artefacts bruts\nGains géométriques, IC appariés à 95 %")
    direct = pilot["comparisons"]["agent_vs_native_codex"]["interval"]
    intervals = [("Ensemble des cas", direct)] + [("n = " + case.removeprefix("final-"), interval)
                  for case, interval in direct.get("case_gains", {}).items()]
    for y, (_, interval) in enumerate(intervals):
        gain, low, high = (interval[key] for key in ("gain", "lower", "upper"))
        exact = pilot["comparisons"]["agent_vs_native_codex"].get("gain_basis") == "exact_artifact_identity"
        if exact:
            direct_ax.scatter(gain, y, marker="D", color="#6c62a5")
        else:
            direct_ax.errorbar(gain, y, xerr=[[gain-low], [high-gain]], fmt="o", color="#6c62a5", capsize=4)
        direct_ax.annotate("1 : identité exacte" if exact else f"{gain:.3f}", (gain, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    direct_ax.set_yticks(range(len(intervals)), [name for name, _ in intervals])
    direct_ax.invert_yaxis()
    direct_ax.set_ylim(len(intervals)-.4, -.65)
    direct_ax.set_xlim(.59, 1.06)
    direct_ax.set_xlabel("Latence CLI natif / latence agent brut")
    direct_ax.set_title("Comparaison directe appariée\nUn ratio < 1 favorise le CLI natif")
    for axis in axes:
        axis.axvline(1, color="#71828d", linestyle="--", linewidth=1)
        axis.grid(axis="x", alpha=.15)
    fig.suptitle("Pilote GEMM indépendant — ROMEO GH200 — n = " + ", ".join(map(str, pilot["shapes"])))
    target = output / "pilote-gemm-fige-romeo.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def frozen_pilot_panel(pilot, figure):
    heading = "<h2>Pilote GEMM indépendant : quatre kernels figés</h2>"
    if not pilot["receipt_verified"]:
        return heading + "<p>Comparaison figée non qualifiée : " + escape(" ; ".join(pilot["validation_errors"])) + ".</p>"
    rows = []
    for role in ("agent", "single_prompt", "native_codex"):
        pair = pilot["comparisons"][role + "_vs_baseline"]
        interval = pair["interval"]
        rows.append([FROZEN_ROLES[role], fmt(interval["gain"], 6),
                     "identité exacte, sans IC" if pair.get("gain_basis") == "exact_artifact_identity" else
                     "[" + fmt(interval["lower"], 6) + " ; " + fmt(interval["upper"], 6) + "]", pair["status"]])
    direct = pilot["comparisons"]["agent_vs_native_codex"]["interval"]
    direct_sentence = ("Le contraste direct est une identité exacte d’artefacts : ratio 1 sans intervalle statistique. "
        "Les mesures héritées bruitées sont conservées comme données sources, sans différence de performance déduite. "
        if pilot["comparisons"]["agent_vs_native_codex"].get("gain_basis") == "exact_artifact_identity" else
        f"Le ratio direct CLI natif/agent brut vaut {fmt(direct['gain'], 6)} [{fmt(direct['lower'], 6)} ; {fmt(direct['upper'], 6)}] ; un ratio inférieur à1 favorise le CLI natif. ")
    agent_pair = pilot["comparisons"]["agent_vs_baseline"]
    agent_sentence = ("L’agent brut et le naïf partagent exactement le même artefact : identité 1 sans intervalle. "
        if agent_pair.get("gain_basis") == "exact_artifact_identity" else
        f"Le gain géométrique de l’agent brut face au naïf vaut {fmt(agent_pair['interval']['gain'], 6)}" +
        (" : la régression reste visible. " if agent_pair["interval"]["gain"] < 1 else ". "))
    return (heading + "<p>Ce pilote compare le naïf, le kernel brut de l’agent amélioré, une proposition structurée sans outils et le véritable Codex CLI vanilla natif. "
            "Il ne contient aucun agent original upstream et n’entre pas dans les cohortes de l’étude §16. Les kernels bruts sont conservés, sans substitution par le meilleur parent retenu.</p>"
            + (f"<img src='{escape(figure.name)}' alt='Pilote GEMM : quatre artefacts bruts et comparaison directe agent contre Codex CLI natif'>" if figure else "")
            + table(["Artefact face au naïf", "Gain", "IC95% apparié", "Statut du reçu"], rows)
            + "<p>" + agent_sentence
            + direct_sentence
            + "Ce ratio provient de ses propres paires AB/BA et n’est pas déduit des deux gains agrégés.</p>"
            + f"<p>Sources figées avant révélation des nouvelles tailles {escape(', '.join(map(str, pilot['shapes'])))} ; quatre validations complètes et quatre qualifications aux sanitizers. "
            + f"Mesures neuves sur une allocation GH200 commune, job <code>{escape(str(pilot['job_id']))}</code>, terminal COMPLETED 0:0. "
            + "Les anciennes séries ne sont pas regroupées. Ce résultat porte sur ces artefacts et ces cas, sans conclusion de supériorité universelle.</p>")


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = ("full", "true_upstream", "native_codex_vanilla")
    colors = ("#167c80", "#be743a", "#6c62a5")
    rows = summary["rows"]
    if not any(row.get("gain") for row in rows):
        return None
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, (operation, title) in zip(axes.flat, FAMILIES.items()):
        for x, (mode, color) in enumerate(zip(methods, colors)):
            selected = [row for row in rows if row["workload"] == operation and row["mode"] == mode and row.get("gain")]
            for index, row in enumerate(selected):
                offset = (index-(len(selected)-1)/2)*.10
                exact = row.get("gain_basis") in ("exact_artifact_identity", "exact_baseline_artifact")
                y = 1.0 if exact else row["gain"]
                ax.scatter(x+offset, y, color=color, s=40, zorder=3)
                if exact:
                    ax.annotate("identité", (x+offset, y), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
                elif row.get("lower") and row.get("upper"):
                    ax.vlines(x+offset, row["lower"], row["upper"], color=color, alpha=.65)
        ax.axhline(1, color="#71828d", linestyle="--", linewidth=1)
        ax.set_xticks(range(3), ("Amélioré", "Original", "Vanilla"))
        ax.set_title(title)
        ax.set_ylabel("Latence naïve / latence retenue")
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("ROMEO : identités exactes et mesures finales, intervalles appariés à 95 %")
    target = output / "comparaison-methodes-romeo.png"
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ci", type=Path)
    parser.add_argument("--requirements", type=Path)
    parser.add_argument("--audit", type=Path, default=REPO / "docs/SPEC_AUDIT.fr.md")
    parser.add_argument("--frozen-comparison", type=Path, default=REPO / FROZEN_RECEIPT)
    args = parser.parse_args()
    from optimizer_agent.suite_orchestrator import SuiteOrchestrator
    from optimizer_agent.suite_report import write_report
    plan = load(args.study / "plan.json")
    planned_generations = len(plan["entries"])
    planned_groups = len({entry["group_id"] for entry in plan["entries"]})
    status = SuiteOrchestrator(plan, args.study).status()  # No job or model action.
    evidence = load(args.requirements, {}) if args.requirements else {}
    report = write_report(args.study, status, evidence=evidence)
    report["project_observed_consumption"] = project_consumption(REPO / "results")
    summary = report["summary"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure = plot(summary, output)
    pilot = frozen_pilot_summary(args.frozen_comparison)
    report["independent_frozen_gemm_pilot"] = pilot
    pilot_figure = plot_frozen_pilot(pilot, output)
    ci = load(args.ci, {}) if args.ci else {}
    source = checkout_source_identity(REPO)
    head = source["head"]
    ci_ok = ci_qualifies_checkout(ci, source)
    report["source_checkout"] = source
    report["ci_evidence"] = {"receipt": ci, "exact_checkout_qualified": ci_ok,
                             "policy": "Successful completed CI on HEAD qualifies only a clean source checkout"}
    report["control_receipts"] = control_receipts(REPO)
    report["direct_comparison_views"] = direct_comparison_views(report.get("final_groups"))
    report["campaign_metric_views"] = campaign_metric_views(summary)
    report["campaign_call_observations"] = campaign_call_observations(summary, report["project_observed_consumption"], args.study)
    report["campaign_research_model_consumption"] = campaign_research_model_consumption(
        report["campaign_call_observations"], report["project_observed_consumption"], args.study)
    receipts = [[value["label"], value["status"], value["ok"] if value["ok"] is not None else "voir périmètre",
                 value["job_id"] or "sans nouveau job", value["terminal"].get("state", "sans objet / en attente"),
                 value["source_head"] or "non enregistré", value["path"] or "aucun reçu acquis",
                 value["scope"] or "non renseigné"] for value in report["control_receipts"]]
    links = " · ".join(f"<a href='{escape(p.name)}'>{escape(p.name)}</a>" for p in sorted(output.glob("*.zip")))
    ci_html = (f"<a href='{escape(ci.get('url',''))}'>CI du SHA publié</a> : " + ("réussie sur les sources propres de ce SHA" if ci_ok else "ne qualifie pas ce checkout source"))
    consumed = report["project_observed_consumption"]
    consumption_html = table(["Périmètre", "Reçus CLI uniques (identités)", "Entrée observée", "Dont cache", "Sortie observée", "Appels à usage inconnu"],
        [["Tout le projet retenu, pilotes et diagnostics inclus", consumed["cli_call_receipts"], consumed["input_tokens"],
          consumed["cached_input_tokens"], consumed["output_tokens"], consumed["unreported_call_usage"]]])
    consumption_html += table(["Périmètre GPU", "Secondes GPU allouées observées", "Jobs terminaux authentifiés", "Jobs sans durée terminale connue"],
        [["Borne inférieure des leases Slurm à un GPU, sans temps de queue", consumed["observed_terminal_allocated_gpu_seconds"],
          consumed["observed_terminal_job_count"], ", ".join(consumed["nonterminal_or_unreported_jobIDs"]) or "aucun parmi les reçus retenus"]])
    consumption_html += ("<p>Les totaux sont des bornes inférieures observées : un appel sans usage déclaré reste inconnu, sans facture nulle inventée. "
        "Les secondes allouées proviennent des durées terminales Slurm archivées et incluent recherche, modèle et attente pendant la réservation. "
        "Les compteurs GPU du ledger peuvent inclure une interruption entre sélection et résultat après rehost ; ils ne représentent pas ces leases Slurm réelles. "
        "Les coûts en dollars restent inconnus.</p>")
    chart_html = f"<img src='{figure.name}' alt='Trois méthodes comparées par famille et graine avec intervalles appariés'>" if figure else "<p>Aucun gain final commun mesuré à cette date.</p>"
    html = f"""<!doctype html><html lang='fr'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Comparaison vérifiée des agents CUDA sur ROMEO</title><style>
body{{font:15px/1.6 system-ui,sans-serif;color:#173542;max-width:1280px;margin:40px auto;padding:0 24px}}
h1{{font-size:30px}}h2{{margin-top:36px}}a{{color:#136b80}}table{{border-collapse:collapse;width:100%;font-size:12px}}
td,th{{padding:9px;border-bottom:1px solid #d6e2e5;text-align:left;vertical-align:top}}th{{background:#eef4f5}}
.scroll{{overflow:auto}}.status{{padding:18px;background:#eef4f5;border-left:5px solid #167c80}}img{{max-width:100%}}
code{{overflow-wrap:anywhere}}details{{margin-top:20px}}summary{{cursor:pointer;font-weight:600}}
</style><h1>Comparaison vérifiée des agents CUDA sur ROMEO</h1>
<p class='status'>Campagnes et évaluations communes terminées : <strong>{'oui' if summary['complete'] else 'non'}</strong>.
Tous les artefacts finaux qualifiés : <strong>{'oui' if summary.get('artifact_evaluations_qualified', summary['scientific_qualified']) else 'non'}</strong>.
Incidents de protocole des méthodes : <strong>{'état inconnu' if summary.get('method_protocols_qualified', True) is None else 'aucun incident signalé' if summary.get('method_protocols_qualified', True) else 'incident vérifié'}</strong>.
Qualification globale de l'étude : <strong>{'oui' if summary['scientific_qualified'] else 'non'}</strong>.
Les échecs, pertes et résultats encore inconnus restent visibles.</p>
<p><a href='https://github.com/wolf75222/agentic-cuda-optimizer'>Fork publié</a> · source du rapport : <code>{head}</code> · {ci_html}.</p>
<p>Checkout source : <strong>{'modifié' if source['dirty'] else 'propre'}</strong> ; {len(source['tracked_changes'])} chemins suivis modifiés et {len(source['untracked_files'])} fichiers non suivis non ignorés. Les chemins exacts sont enregistrés dans le JSON du rapport.</p>
<p>Agent local Codex, modèle demandé <code>gpt-6-luna</code>, effort <code>max</code> ; orchestration LangGraph et recherche TreeQuest.
L'inférence utilise le service OpenAI authentifié. Aucun plafond global de tokens, coût, appels ou durée cumulée n'est imposé.
Les limites techniques et les durées obligatoires de Slurm restent enregistrées.</p>
<h2>Étude §16 : trois méthodes, un évaluateur final</h2>
<p>L'optimiseur amélioré, le véritable agent original au commit <code>1e9464d</code> et Codex vanilla natif sont des méthodes distinctes.
Le vanilla reçoit un seul prompt et dispose de ses outils de compilation et benchmark. Le parent glouton constitue une ablation supplémentaire.
Le protocole déclaré couvre {len(plan['workloads'])} opérations, {len(plan['seeds'])} graines et {len(plan['modes'])} modes : {planned_generations} générations et {planned_groups} groupes finaux. Chaque groupe fige toutes les sources avant de créer de nouvelles données sur le domaine de tailles déclaré.</p>
<p>Les contrastes PMPP prévus séparent la consultation ciblée (méthode complète face au guide permanent seul) et le guide permanent (guide seul face au mode sans PMPP).</p>
{execution_evidence_panel(report['control_receipts'])}
<p>Chaque groupe compare ses artefacts sur un même GPU réservé, avec oracle CPU indépendant, toutes les sorties, quatre sanitizers,
puis confirmations AB/BA et bootstrap des séries. Les chronométrages d'allocations différentes ne sont jamais regroupés.
Les mesures locales ne remplacent aucune performance ROMEO. Les points et intervalles ci-dessous montrent les campagnes individuelles ; un gain supérieur à1 favorise le candidat face au naïf.</p>
<p>Les identités exactes d’artefact donnent un ratio déterministe 1 sans intervalle. Plusieurs labels pour ce même artefact ne sont pas plusieurs mesures indépendantes.</p>
{chart_html}{study_aggregate_table(summary)}{workload_dispersion_table(summary)}{direct_comparison_table(report['direct_comparison_views'])}
{case_comparison_table(summary, report['direct_comparison_views'])}
<p>{escape(report['method_outcome_policy'])} Les points représentent l’artefact retenu, y compris un candidat de repli.</p>
<p>Les agrégats incluent les retours au noyau naïf. Le compteur « Régressions » dénombre les ratios ponctuels inférieurs à 1 ;
il ne démontre pas une régression statistique si l’IC recouvre 1 et ne remplace pas un veto de cas critique. Les absences ne reçoivent aucun gain inventé.
Cette étude sur des domaines déclarés à un seul kernel ne promet pas de gain universel ni de supériorité sur cuBLAS.</p>
{frozen_pilot_panel(pilot, pilot_figure)}
<h2>Consommation, régressions et provenance</h2>
{consumption_html}
<p>Les tokens comptent les entrées et sorties observées, y compris les tokens d'entrée servis du cache lorsqu'ils sont inclus par le fournisseur.
Les traces conservent séparément les tokens mis en cache. Aucun coût en dollars n'est déduit sans reçu de facturation.
Le temps d'allocation inclut l'attente des modèles, les échanges SSH et les outils ; les CUDA Events ne mesurent que le kernel.
Les coûts absents sont inconnus. Les vues suivantes présentent les jalons, la compilation, le cache, la recherche et les coûts méthode/observateur enregistrés, sans recalculer les métriques historiques.</p>
<details><summary>Résultats des {planned_generations} campagnes, y compris les pertes et échecs</summary>{study_campaign_table(summary)}</details>
{campaign_call_panel(report['campaign_call_observations'])}
{campaign_research_model_panel(report['campaign_research_model_consumption'])}
{campaign_metrics_panel(report['campaign_metric_views'])}
<h2>Contrôles réellement exécutés et limites</h2>{table(['Contrôle','Statut','Résultat','Job','Terminal','HEAD des sources exécutées','Reçu','Périmètre et limites'], receipts)}
<p>Sur le pilote GH200, Nsight Compute a signalé <code>ERR_NVGPUCTRPERM</code>. Ce Mac ne fournit pas de GPU CUDA pour le repli local.
Ces indisponibilités restent des diagnostics explicites ; aucun compteur n'est inventé et aucun temps instrumenté ne participe au classement.
Les scénarios destructifs antérieurs ont vérifié la mise en quarantaine et le remplacement du worker. Les reprises de processus rejouent des preuves CUDA archivées sans produire de nouvelle mesure GPU.</p>
<p>Les essais historiques interrompus, notamment le job770184 terminé par TIMEOUT, sont conservés. Ses deux finales individuelles ne constituent pas une comparaison directe complète.
Sa proposition structurée sans outils est identifiée séparément du véritable Codex vanilla natif.</p>
<h2>Exigences : implémenté, testé localement, exécuté sur ROMEO, restant</h2>
<p>Cette confrontation couvre la spécification intégralement relue. Chaque affirmation est limitée à son périmètre ; la présence d'un fichier ou d'un test ne prouve pas un fonctionnement de bout en bout.</p>
{audit_table(args.audit)}
<h2>Reproduction et artefacts</h2><p>{links or 'Archive portable en attente de préparation.'}</p>
<p><a href='rapport-donnees.json'>Résultats structurés et consommation observée</a>. Les archives contiennent sources figées, configurations,
entrées relatives et empreintes, contrats, environnements, journaux, sauvegardes SQLite cohérentes et provenance du dépôt original.
Les références documentaires conservent URL, version, section et empreinte ; leurs corps externes ne sont pas redistribués.</p>
<p>Commandes et configuration : <code>README.md</code>, <code>docs/quality_suite.fr.md</code> et <code>docs/upstream_original.md</code> dans les sources exportées.
La reconstruction vérifie les hashes avant toute compilation. Les preuves locales, la CI du SHA publié et les qualifications ROMEO restent trois validations distinctes.</p></html>"""
    (output / "rapport-validation.html").write_text(html)
    (output / "rapport-donnees.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    paths = [output / "rapport-validation.html", output / "rapport-donnees.json"] + ([figure] if figure else []) + ([pilot_figure] if pilot_figure else [])
    (output / "RAPPORT-SHA256SUMS.txt").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.name+"\n" for p in paths))
    print(json.dumps({"report": str(output / "rapport-validation.html"), "complete": summary["complete"],
                      "ci_exact_sha": ci_ok, "source_head": head, "source_dirty": source["dirty"]}, indent=2))


if __name__ == "__main__":
    main()
