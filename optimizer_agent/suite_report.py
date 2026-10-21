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
        "| Méthode | Prévus | Méthodes réussies | Méthodes échouées | Statuts méthode inconnus | Artefacts mesurés | Artefacts qualifiés | Régressions | Absents/échecs sans mesure | Gain géométrique mesuré | Écart-type entre campagnes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fmt = lambda value: f"{value:.4g}" if value is not None else "inconnu"
    for mode, row in summary["aggregates"].items():
        cells = [mode, row['planned'], *method_count_cells(row), row['final_measured'], row['qualified'],
                 row['regressions'], row['unmeasured_or_failed'], fmt(row['geometric_mean_all_measured_gain']),
                 fmt(row['gain_stddev_between_campaigns'])]
        lines.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
    lines += ["", METHOD_OUTCOME_POLICY, "",
        "Ces agrégats incluent les régressions mesurées et les retours au noyau initial. Les échecs et valeurs inconnues restent dans le dénominateur ; aucun gain fictif ne leur est attribué. Les temps bruts de matériels ou allocations différents ne sont jamais regroupés.", "",
        "## Résultats par campagne", "",
        "| Famille | Graine | Méthode | Statut méthode | Défaillance méthode | Candidat de repli | Statut final de l’artefact | Gain final mesuré | IC95% | Artefact |",
        "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | --- |"]
    for row in summary["rows"]:
        cells = [row['workload'], row['seed'], row['mode'], *method_outcome_cells(row), row['status'],
                 fmt(row.get('gain')), f"[{fmt(row.get('lower'))} ; {fmt(row.get('upper'))}]",
                 row.get('artifact_hash') or 'absent']
        lines.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
    lines += ["", "Les résultats par cas, intervalles, sources figées, paramètres, contrats, empreintes GPU et coûts détaillés sont disponibles dans report.json et les reçus des groupes finaux. Une absence de mesure n’est pas une absence de coût.", "", "## Exigences", "",
        "| Identifiant | Exigence | Implémenté | Test local | Exécuté ROMEO | En attente |", "| --- | --- | --- | --- | --- | --- |"]
    for row in report["requirements"]:
        mark = lambda key: "sans objet" if not row[key]["applicable"] else "vérifié" if row[key]["ok"] else "en attente"
        lines.append(f"| {row['id']} | {row['title'].replace('|', '/')} | {mark('implemented')} | {mark('local_tested')} | {mark('romeo_executed')} | {', '.join(row['pending']) or 'aucun'} |")
    lines += ["", "Un reçu doit préciser son périmètre pour valider une exigence. Un fichier source présent ou un scénario préparé ne prouve pas une exécution réussie. Le protocole section16 constitue une étude reproductible, sans promesse de gain universel.", ""]
    (directory / "rapport-fr.md").write_text("\n".join(lines))
    return report
