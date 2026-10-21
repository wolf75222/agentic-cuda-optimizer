#!/usr/bin/env python3
"""Export completed GPU evidence, exact sources and a readable local report."""
import argparse
from datetime import datetime
import hashlib
from html import escape
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from optimizer_agent.campaign import comparison_completed
    campaign = args.campaign.resolve()
    summary_path = campaign / "comparison.json"
    if not summary_path.is_file():
        raise SystemExit("The comparison has no completed receipt; refusing a completed-results export")
    summary = json.loads(summary_path.read_text())
    if summary.get("ok") is not True or summary.get("status") != "completed" or not comparison_completed(summary):
        raise SystemExit("A complete generated-agent/vanilla paired comparison is required before export")
    terminal_path = campaign / "worker-terminal.json"
    terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
    if not terminal.get("terminal") or terminal.get("state") != "COMPLETED" or terminal.get("exit_code") != "0:0":
        raise SystemExit("Successful terminal Slurm accounting is required before export")
    if str(terminal.get("job_id")) != str(summary.get("job_id")) or not summary.get("job_id"):
        raise SystemExit("The comparison and terminal receipt must identify the same Slurm job")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise SystemExit("Commit the complete source changes before exporting an exact-commit bundle")
    source_tar = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=root)
    archive_path = output / "agentic-cuda-optimizer-romeo.zip"
    included = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def add_bytes(name, payload):
            archive.writestr(name, payload)
            included.append({"path": name, "sha256": hashlib.sha256(payload).hexdigest()})

        with tarfile.open(fileobj=io.BytesIO(source_tar)) as sources:
            for member in sources.getmembers():
                if member.isfile():
                    add_bytes("agentic-cuda-optimizer/" + member.name, sources.extractfile(member).read())
        evidence_roots = [campaign, root / "results/gpu-acceptance", root / "work/live/gpu-smoke"]
        for directory in evidence_roots:
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if "research" in path.parts and "sources" in path.parts:
                    # Dossiers retain canonical URLs, section locators and
                    # hashes; do not redistribute downloaded documentation.
                    continue
                if path.is_file() and path.suffix not in (".sqlite", ".db", ".pyc") and not path.name.endswith(("-wal", "-shm")):
                    name = "validation/" + str(path.relative_to(root))
                    add_bytes(name, path.read_bytes())
        manifest = {"source_commit": commit, "repository": "https://github.com/wolf75222/agentic-cuda-optimizer",
            "created_at": datetime.now(ZoneInfo("Europe/Paris")).isoformat(), "comparison": summary,
            "reference_provenance": "Canonical URLs, section locators and content hashes retained in dossiers; downloaded reference bodies stay in the local research cache",
            "terminal": terminal, "files": included}
        archive.writestr("validation/EXPORT-MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive_path) as archive:
        failure = archive.testzip()
        if failure:
            raise SystemExit("Invalid ZIP entry: " + failure)

    direct = summary.get("generated_agent_vs_vanilla", {})
    interval = direct.get("interval", {})
    figure_path = output / "comparaison-romeo.png"
    figure_path.unlink(missing_ok=True)
    if interval.get("case_gains"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pairs = sorted(interval["case_gains"].items(), key=lambda item: int(item[0].rsplit("-", 1)[-1]))
        fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
        for index, (name, values) in enumerate(pairs):
            ax.hlines(index, values["lower"], values["upper"], color="#377d8b", linewidth=3)
            ax.scatter(values["gain"], index, color="#134b5b", s=55, zorder=3)
        ax.axvline(1, color="#ad6243", linestyle="--", linewidth=1)
        ax.set_yticks(range(len(pairs)), ["N = " + name.rsplit("-", 1)[-1] for name, _ in pairs])
        ax.set_xlabel("Latence vanilla / latence agent — valeur > 1 en faveur de l'agent")
        ax.set_title("Kernels générés : comparaison directe sur ROMEO GH200")
        ax.grid(axis="x", alpha=.15)
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)
    rows = []
    for mode, value in summary.get("results", {}).items():
        usage = value.get("usage", {})
        rows.append("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in (
            mode, value.get("status"), value.get("candidate_id"), usage.get("model_tokens", "indisponible"),
            round(value.get("elapsed_seconds", 0), 1))) + "</tr>")
    gain_text = (f"Rapport latence vanilla / latence agent : {interval['gain']:.3f}× ; intervalle {interval.get('confidence', .95):.0%} "
                 f"[{interval['lower']:.3f}, {interval['upper']:.3f}]." if interval else
                 "Comparaison directe non qualifiée : " + str(direct.get("status", "incomplète")))
    hardware = summary.get("hardware", {}).get("device_properties", {})
    image = f'<img src="{figure_path.name}" alt="Comparaison par forme et intervalles appariés">' if figure_path.exists() else ""
    html = f'''<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Validation CUDA sur ROMEO</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1000px;margin:48px auto;padding:0 24px;color:#183c48}}
h1{{font-size:32px}}h2{{margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #ccd9dc;text-align:left}}
code{{background:#eef4f5;padding:2px 5px}}img{{max-width:100%}}.note{{padding:16px;background:#eef4f5;border-radius:8px}}a{{color:#14667f}}</style>
<h1>Validation du fork CUDA sur ROMEO</h1>
<p><a href="https://github.com/wolf75222/agentic-cuda-optimizer">Fork GitHub</a> · sources exportées au commit <code>{commit}</code></p>
<p>{escape(str(hardware.get('name', 'GPU ROMEO')))}, {escape(str(hardware.get('architecture', '')))},
allocation <code>{escape(str(summary.get('job_id')))}</code> : <strong>{escape(str(terminal.get('state')))}</strong>, sortie {escape(str(terminal.get('exit_code')))}.</p>
<p>Codex local, <code>gpt-6-luna</code>, effort <code>max</code>, orchestration LangGraph.
Aucun plafond global de tokens, de coût ou de temps de campagne ; consommation observée.</p>
<h2>Résultats retenus après évaluation indépendante</h2>
<table><tr><th>Mode</th><th>Qualification</th><th>Artefact retenu</th><th>Tokens observés</th><th>Durée (s)</th></tr>{''.join(rows)}</table>
<h2>Comparaison directe des kernels générés</h2><p>{escape(gain_text)}</p>
<p>Décision : <code>{escape(str(direct.get('status')))}</code> ; contrôle des régressions critiques :
<code>{escape(str(direct.get('critical_cases_ok', 'voir reçu brut')))}</code>.</p>{image}
<p>Données et formes indépendantes après la fin des deux générations, mêmes règles numériques, séries A/B alternées et bootstrap apparié.
Les kernels générés sont mesurés même si une régression a conduit à conserver la baseline.</p>
<h2>Contrôles réels</h2><p>Oracle CPU indépendant, validation complète avant mesure, sentinelles de sortie,
entrées en lecture seule, cache natif, contexte CUDA persistant et remplacement après quarantaine.
Les reçus distinguent chaque sanitizer requis, chaque profil et les mesures ordinaires.</p>
<p class="note">Une campagne GEMM par mode ne permet pas de conclure à une supériorité générale de l'optimiseur.
Les études sur six familles et plusieurs graines sont planifiées séparément. Les performances locales ne remplacent aucune mesure ROMEO.</p>
<h2>Sources et preuves</h2><p><a href="{archive_path.name}">Archive : sources, configurations, résultats bruts, dossiers documentaires et comptabilité Slurm</a>.</p>
<p>La suite locale et la CI GitHub sont documentées dans <code>docs/VALIDATION.md</code>. Les empreintes de chaque fichier exporté figurent dans <code>validation/EXPORT-MANIFEST.json</code>.</p></html>'''
    report_path = output / "rapport-validation.html"
    report_path.write_text(html)
    checksum_paths = [archive_path, report_path] + ([figure_path] if figure_path.exists() else [])
    (output / "SHA256SUMS.txt").write_text("".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+p.name+"\n" for p in checksum_paths))
    print(json.dumps({"report": str(report_path), "archive": str(archive_path), "files": len(included),
                      "archive_bytes": archive_path.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
