"""French reports that keep implementation, local tests and GPU proof separate."""
from pathlib import Path
import re

from .suite_orchestrator import REPO, write_json


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
        "evidence_policy": "Une implémentation, un test local et une preuve ROMEO sont trois affirmations indépendantes. Les absences restent en attente."}
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
        "| Méthode | Prévus | Mesurés | Qualifiés | Régressions | Absents/échecs sans mesure | Gain géométrique mesuré | Écart-type entre campagnes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for mode, row in summary["aggregates"].items():
        fmt = lambda value: f"{value:.4g}" if value is not None else "inconnu"
        lines.append(f"| {mode} | {row['planned']} | {row['final_measured']} | {row['qualified']} | {row['regressions']} | {row['unmeasured_or_failed']} | {fmt(row['geometric_mean_all_measured_gain'])} | {fmt(row['gain_stddev_between_campaigns'])} |")
    lines += ["", "Ces agrégats incluent les régressions mesurées et les retours au noyau initial. Les échecs et valeurs inconnues restent dans le dénominateur ; aucun gain fictif ne leur est attribué. Les temps bruts de matériels ou allocations différents ne sont jamais regroupés.", "",
        "Les résultats par cas, intervalles, sources figées, paramètres, contrats, empreintes GPU et coûts détaillés sont disponibles dans report.json et les reçus des groupes finaux. Une absence de mesure n’est pas une absence de coût.", "", "## Exigences", "",
        "| Identifiant | Exigence | Implémenté | Test local | Exécuté ROMEO | En attente |", "| --- | --- | --- | --- | --- | --- |"]
    for row in report["requirements"]:
        mark = lambda key: "sans objet" if not row[key]["applicable"] else "vérifié" if row[key]["ok"] else "en attente"
        lines.append(f"| {row['id']} | {row['title'].replace('|', '/')} | {mark('implemented')} | {mark('local_tested')} | {mark('romeo_executed')} | {', '.join(row['pending']) or 'aucun'} |")
    lines += ["", "Un reçu doit préciser son périmètre pour valider une exigence. Un fichier source présent ou un scénario préparé ne prouve pas une exécution réussie. Le protocole section16 constitue une étude reproductible, sans promesse de gain universel.", ""]
    (directory / "rapport-fr.md").write_text("\n".join(lines))
    return report
