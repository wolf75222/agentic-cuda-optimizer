"""French reports that keep implementation, local tests and GPU proof separate."""
from pathlib import Path
from html import escape
import json
import re

from .suite_orchestrator import REPO, write_json


METHOD_OUTCOME_POLICY = (
    "Le statut de la méthode décrit sa tentative de génération. La disponibilité et la qualification "
    "d’un artefact retenu sont indépendantes : un candidat de repli peut être mesuré et qualifié après "
    "l’échec de la méthode. Son gain final reste affiché sans transformer cet échec en succès. "
    "Un statut de méthode absent reste inconnu."
)


def method_outcome_cells(row):
    """Display method failure without changing the measured fallback outcome."""
    status = row.get("generation_method_status")
    label = {"completed": "réussie", "failed": "échouée"}.get(status, "inconnu")
    error = row.get("generation_method_error")
    if error is None:
        error = "aucune signalée" if status == "completed" else "non renseignée"
    elif not isinstance(error, str):
        error = json.dumps(error, ensure_ascii=False, sort_keys=True)
    if "candidate_fallback" not in row:
        fallback = "inconnu"
    else:
        value = row["candidate_fallback"]
        if value is None or value is False:
            fallback = "aucun"
        elif isinstance(value, dict) and value.get("retained") is True:
            origin = {"supplied_initial": "noyau initial fourni",
                      "original_generated_proposal": "proposition de l’agent original"}.get(
                          value.get("selected_origin"), value.get("selected_origin") or "origine inconnue")
            iteration = value.get("selected_iteration")
            reason = "échec de la méthode" if value.get("reason") == "method_failed" else value.get("reason")
            fallback = f"oui : {origin}, itération {iteration if iteration is not None else 'inconnue'}"
            if reason:
                fallback += f" ({reason})"
        else:
            fallback = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return [label, error, fallback]


def method_count_cells(aggregate):
    return [aggregate.get(key, "inconnu") for key in
            ("method_completed", "method_failed", "method_status_unknown")]


def _markdown_cell(value):
    return escape(str(value), quote=False).replace("|", "&#124;").replace("\n", "<br>")


def _number(value):
    return f"{value:.4g}" if value is not None else "inconnu"


def _markdown_table(headers, rows):
    return ["| " + " | ".join(_markdown_cell(value) for value in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(_markdown_cell(value) for value in row) + " |" for row in rows)]


def _workload_dispersion_lines(summary):
    rows = []
    for mode, aggregate in summary.get("aggregates", {}).items():
        for workload, value in (aggregate.get("per_workload") or {}).items():
            seeds = value.get("seeds") or []
            observations = "; ".join(str(item.get("seed")) + ": " + _number(item.get("gain")) +
                " (" + str(item.get("status", "inconnu")) + ")" for item in seeds)
            deviation = _number(value.get("gain_stddev_between_independent_seeds"))
            if deviation == "inconnu" and isinstance(value.get("measured"), int) and value["measured"] < 2:
                deviation += " (n<2)"
            rows.append([mode, workload, value.get("planned", "inconnu"), value.get("measured", "inconnu"),
                value.get("unmeasured_or_failed", "inconnu"), _number(value.get("geometric_mean_gain")),
                deviation, observations or "non enregistrées"])
    return ["", "## Dispersion par famille entre graines indépendantes", "",
        "Ces valeurs reprennent les agrégats enregistrés par famille et méthode, sans recalcul ni regroupement des temps bruts. Moins de deux ratios disponibles ne permettent pas un écart-type entre graines ; une absence reste inconnue. Les ratios peuvent inclure des identités déterministes.", "",
        *_markdown_table(["Méthode", "Famille", "Prévus", "Ratios disponibles", "Absents/échecs sans ratio",
                         "Ratio géométrique", "Écart-type entre graines", "Graines : ratio et statut"], rows),
        *( ["", "Agrégats par famille non enregistrés : inconnu."] if not rows else [])]


def _duration_ids(value):
    if not isinstance(value, list):
        return "inconnu"
    return str(len(value)) + (" : " + ", ".join(str(item) for item in value) if value else " (aucun)")


def _request_cost_cells(cost):
    return [*[_number(cost.get(key)) for key in ("observed_rpc_calls", "unique_request_ids", "rpc_wall_seconds",
        "reported_native_reservation_seconds", "reported_compile_seconds")],
        _duration_ids(cost.get("native_duration_missing_request_ids")),
        _duration_ids(cost.get("native_duration_conflict_request_ids")), _number(cost.get("unidentified_rpc_calls"))]


def _final_closure_lines(summary):
    closures, candidates, shared, conflicts = [], [], {}, set()
    for row in summary.get("rows", []):
        closure = row.get("final_closure") or {}
        closed = closure.get("status") == "closed"
        value = closure if closed else {}
        usage = value.get("reported_generation_usage") or {}
        evidence = value.get("cost_evidence") or {}
        context = [row["workload"], row["seed"], row["mode"]]
        closures.append([*context, "close / " + str(value.get("outcome", "inconnu")) if closed else "inconnu / non enregistré",
            _number(value.get("seconds_since_generation_task_start")), _number(value.get("final_group_wall_seconds")),
            _number(usage.get("model_tokens")), _number(usage.get("model_calls")), _number(usage.get("gpu_seconds")),
            _number(value.get("final_model_tokens")), _number(value.get("final_model_calls")), _number(value.get("cost_usd")),
            evidence.get("path") or "inconnu", evidence.get("sha256") or "inconnu"])
        candidates.append([*context, *_request_cost_cells(value.get("candidate_request_cost") or {})])
        if closed and value.get("group_id") is not None and evidence.get("sha256"):
            key = (str(value["group_id"]), str(evidence["sha256"]))
            cost = value.get("shared_group_request_cost") or {}
            if key in shared and shared[key] != cost:
                conflicts.add(key)
            else:
                shared.setdefault(key, cost)
    cost_headers = ["RPC observés", "IDs uniques", "Durées RPC (s)", "Durées natives connues (s)",
                    "Compilation connue (s)", "IDs durée native manquante", "IDs durée native contradictoire", "RPC sans ID"]
    lines = ["", "## Clôture de l’évaluation FINAL et coûts enregistrés", "",
        "La clôture est enregistrée après les décisions et comparaisons du groupe, distincte du verdict individuel, du gain, de la qualification et de la fin du worker ou de la tâche. Elle n’atteste aucun terminal Slurm. Le délai depuis le début de génération comprend attente et interruptions. La durée du groupe FINAL est un contexte partagé, jamais une somme de durées par méthode. Les anciens timestamps ne sont pas reconstruits.", "",
        *_markdown_table(["Famille", "Graine", "Méthode", "Clôture / résultat", "Délai génération→clôture (s)",
            "Durée groupe FINAL (s)", "Tokens génération bruts", "Compteur méthode génération brut", "GPU génération brut (s)",
            "Tokens modèles FINAL", "Appels modèles FINAL", "Coût facturé USD", "Reçu de coût", "SHA du reçu"], closures), "",
        "L’usage de génération est le snapshot brut du ledger à la clôture, potentiellement incomplet ; il ne décrit ni toutes les identités CLI ni l’usage pendant FINAL. Les tokens, appels modèles et coûts facturés absents restent inconnus.", "",
        "### Requêtes FINAL propres au candidat", "",
        *_markdown_table(["Famille", "Graine", "Méthode", *cost_headers], candidates), "",
        "Les durées RPC incluent transport et attente. Les durées natives et de compilation sont des sous-totaux connus dédupliqués par identifiant de requête, jamais des durées Slurm. Les IDs de durée native manquante ou contradictoire restent explicites ; les observations sans ID ne fournissent aucune durée native dédupliquée. La confirmation du candidat comprend les deux côtés de sa comparaison au naïf.", "",
        "### Coûts FINAL partagés, une fois par groupe et reçu", "",
        "Qualification/mesure du naïf et comparaisons directes restent partagées : elles ne sont ni réparties entre méthodes ni ajoutées aux sous-totaux propres. Des copies contradictoires restent inconnues.", "",
        *_markdown_table(["Groupe", "SHA du reçu", *cost_headers],
            [[*key, *_request_cost_cells({} if key in conflicts else cost)] for key, cost in shared.items()])]
    if not shared:
        lines += ["", "Coûts partagés non enregistrés : inconnu."]
    return lines


def requirement_registry(spec=None):
    text = Path(spec or REPO / "docs/spec.md").read_text()
    rows = [{"id": "USER-LOCAL", "title": "Codex Luna Max local et orchestration LangGraph"},
            {"id": "USER-HPC", "title": "SSH ROMEO, compilation et jobs GPU Slurm"},
            {"id": "USER-PROFILE", "title": "Profilage hybride et séparation des matériels"}]
    for line in text.splitlines():
        match = re.match(r"^#{2,3} (\d+(?:\.\d+)?)\.?\s+(.+)", line)
        if match and 1 <= int(match[1].split(".")[0]) <= 19:
            rows.append({"id": "SEC-" + match[1], "title": match[2]})
        match = re.match(r"^\| ([A-Z]+-\d+) \| (.+?) \|", line)
        if match:
            rows.append({"id": match[1], "title": match[2]})
    return rows


def build_requirement_status(evidence=None):
    """External evidence entries must explicitly identify scope and outcome.

    A source path or a planned test never establishes executed qualification.
    """
    evidence = evidence or {}
    if "requirements" in evidence:
        if not isinstance(evidence["requirements"],dict):
            raise ValueError("Requirement evidence catalog must contain an ID-to-evidence mapping")
        evidence = evidence["requirements"]
    rows = []
    for requirement in requirement_registry():
        supplied = evidence.get(requirement["id"], {})
        row = {**requirement}
        for phase in ("implemented", "local_tested", "romeo_executed"):
            item = supplied.get(phase, {})
            applicable = item.get("applicable") is not False
            row[phase] = {"ok": item.get("ok") is True and applicable, "evidence": item.get("evidence", []),
                          "applicable": applicable, "scope": item.get("scope"),
                          "status": "not_applicable" if not applicable else "verified" if item.get("ok") is True and item.get("evidence") else "pending"}
            if not item.get("evidence"):
                row[phase]["ok"] = False
        row["pending"] = [phase for phase in ("implemented", "local_tested", "romeo_executed") if row[phase]["applicable"] and not row[phase]["ok"]]
        row["remaining"] = supplied.get("remaining", row["pending"])
        rows.append(row)
    return rows


def write_report(directory, status, *, evidence=None):
    directory = Path(directory)
    report = {**status, "requirements": build_requirement_status(evidence),
        "evidence_policy": "Une implémentation, un test local et une preuve ROMEO sont trois affirmations indépendantes. Les absences restent en attente.",
        "method_outcome_policy": METHOD_OUTCOME_POLICY}
    if isinstance(evidence,dict) and "requirements" in evidence:
        report["requirement_evidence_catalog"] = evidence.get("evidence_catalog",{})
        report["requirement_audit"] = {key:value for key,value in evidence.items()
                                      if key not in {"requirements","evidence_catalog"}}
    report["remaining"] = [{"id":row["id"],"items":row["remaining"]} for row in report["requirements"] if row["remaining"]]
    write_json(directory / "report.json", report)
    summary = report["summary"]
    lines = ["# Campagnes comparatives et ablations — état vérifié", "",
        f"Exécution finale complète : {'oui' if summary['complete'] else 'non'}. Qualification de tous les résultats : {'oui' if summary['scientific_qualified'] else 'non'}.", "",
        "Les scores proviennent uniquement des groupes finaux communs sur un GPU exclusif. Chaque méthode conserve ses coûts observés. Aucun plafond global de tokens, coût, appels ou durée cumulée n’est imposé. Les limites techniques par requête et les nombres d’itérations restent explicites.", "",
        "| Méthode | Prévus | Méthodes réussies | Méthodes échouées | Statuts méthode inconnus | Ratios disponibles (mesures/identités) | Artefacts qualifiés (mesures/identités) | Régressions : ratios ponctuels < 1 | Absents/échecs sans mesure | Ratio géométrique (mesures/identités) | Écart-type descriptif toutes familles |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt = _number
    for mode, row in summary["aggregates"].items():
        cells = [mode, row['planned'], *method_count_cells(row), row['final_measured'], row['qualified'],
                 row['regressions'], row['unmeasured_or_failed'], fmt(row['geometric_mean_all_measured_gain']),
                 fmt(row['gain_stddev_between_campaigns'])]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
    lines += ["", METHOD_OUTCOME_POLICY, "",
        "Ces agrégats incluent les ratios ponctuels inférieurs à 1 et les retours au noyau initial. Un ratio ponctuel inférieur à 1 ne suffit pas à établir une régression statistique. Les identités exactes donnent un ratio déterministe 1 sans intervalle et ne créent aucune mesure indépendante. Les échecs et valeurs inconnues restent dans le dénominateur ; aucun gain fictif ne leur est attribué. Les temps bruts de matériels ou allocations différents ne sont jamais regroupés."]
    lines += _workload_dispersion_lines(summary)
    lines += ["",
        "## Résultats par campagne", "",
        "| Famille | Graine | Méthode | Statut méthode | Défaillance méthode | Candidat de repli | Statut final de l’artefact | Ratio final | IC95% | Origine du ratio | Artefact |",
        "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | --- | --- |"]
    for row in summary["rows"]:
        identity = row.get("gain_basis") in ("exact_artifact_identity", "exact_baseline_artifact")
        basis = {"exact_baseline_artifact": "identité exacte avec le naïf", "exact_artifact_identity": "identité exacte d’artefact",
                 "paired_measurement": "mesure appariée", "unproven_artifact_context": "contexte d’artefact non prouvé"}.get(row.get("gain_basis"), "non renseignée")
        cells = [row['workload'], row['seed'], row['mode'], *method_outcome_cells(row), row['status'],
                 fmt(1.0 if identity else row.get('gain')), "identité exacte, sans IC" if identity else f"[{fmt(row.get('lower'))} ; {fmt(row.get('upper'))}]", basis,
                 row.get('artifact_hash') or 'absent']
        lines.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
    lines += _final_closure_lines(summary)
    lines += ["", "Les résultats par cas, intervalles, sources figées, paramètres, contrats, empreintes GPU et coûts détaillés sont disponibles dans report.json et les reçus des groupes finaux. Une absence de mesure n’est pas une absence de coût.", "", "## Exigences", "",
        "| Identifiant | Exigence | Implémenté | Test local | Exécuté ROMEO | En attente |", "| --- | --- | --- | --- | --- | --- |"]
    for row in report["requirements"]:
        mark = lambda key: "sans objet" if not row[key]["applicable"] else "vérifié" if row[key]["ok"] else "en attente"
        lines.append(f"| {row['id']} | {row['title'].replace('|', '/')} | {mark('implemented')} | {mark('local_tested')} | {mark('romeo_executed')} | {', '.join(row['pending']) or 'aucun'} |")
    lines += ["", "Un reçu doit préciser son périmètre pour valider une exigence. Un fichier source présent ou un scénario préparé ne prouve pas une exécution réussie. Le protocole section16 constitue une étude reproductible, sans promesse de gain universel.", ""]
    (directory / "rapport-fr.md").write_text("\n".join(lines))
    return report
