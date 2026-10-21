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
PROOF_PATHS = (
    ("Autotuning réel", "results/gpu-autotuning-acceptance/20261021-current/autotuning-acceptance.json"),
    ("Précision, cache et reconstruction", "results/gpu-contract-acceptance/20261021-current/acceptance-summary.json"),
    ("Diagnostic et nouvelle recherche", "results/diagnostic-loop-20261021/diagnostic-loop-acceptance.json"),
    ("Comparaison GEMM figée, quatre artefacts bruts", FROZEN_RECEIPT),
    ("Transport GPU RPC réel", "results/gpu-rpc-acceptance/20261021-current/allocation-receipt.json"),
    ("Isolation profil/benchmark et contrôle final négatif", "results/gpu-final-isolation-acceptance/20261021-current/allocation-receipt.json"),
    ("Finaliste natif reconstruit et revalidé", "results/gpu-reconstructed-acceptance/20261021-current/acceptance-summary.json"),
    ("Reprise après arrêt brutal", "results/process-recovery-acceptance-20261021-release/process-recovery-acceptance.json"),
    ("Codex vanilla natif", "results/native-vanilla-20261021/evidence/receipt.json"))


def load(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else default


def fmt(value, digits=3):
    return f"{value:.{digits}g}" if isinstance(value, (int, float)) and math.isfinite(value) else "inconnu"


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
                  "Statuts méthode inconnus", "Artefacts mesurés", "Artefacts qualifiés",
                  "Régressions", "Absents/échecs sans mesure", "Gain géométrique mesuré",
                  "Dispersion entre campagnes"], rows)


def study_campaign_table(summary):
    from optimizer_agent.suite_report import method_outcome_cells
    rows = [[FAMILIES.get(row["workload"], row["workload"]), row["seed"],
             LABELS.get(row["mode"], row["mode"]), *method_outcome_cells(row), row["status"],
             fmt(row.get("gain")), "["+fmt(row.get("lower"))+" ; "+fmt(row.get("upper"))+"]",
             fmt(row.get("model_tokens"), 9), fmt(row.get("model_calls")), fmt(row.get("wall_seconds")),
             row.get("artifact_hash") or "absent"] for row in summary["rows"]]
    return table(["Famille", "Graine", "Méthode", "Statut méthode", "Défaillance méthode",
                  "Candidat de repli", "Statut final de l’artefact", "Gain final mesuré", "IC95%",
                  "Tokens", "Appels CLI", "Durée génération (s)", "Artefact"], rows)


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


def control_receipts(repo):
    receipts = []
    for name, relative in PROOF_PATHS:
        path = Path(repo) / relative
        value = load(path, {})
        terminal = value.get("terminal") or value.get("worker_terminal") or {}
        provenance = value.get("source_provenance", {})
        receipts.append({"label": name, "path": relative,
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "status": value.get("status", "reçu disponible" if value else "en attente"),
            "ok": value.get("ok"), "job_id": terminal.get("job_id", value.get("job_id")),
            "terminal": terminal, "source_head": provenance.get("HEAD", value.get("source_HEAD")),
            "source_dirty_state": provenance.get("dirty_state", value.get("snapshot_dirty_after")),
            "execution_source_sha256": provenance.get("sources", value.get("execution_source_sha256", value.get("source_sha256_map", {}))),
            "sources_unchanged": value.get("sources_unchanged"), "available": path.is_file(),
            "scope": value.get("scope", value.get("evidence_kind"))})
    receipts.append({"label": "Diagnostic ciblé avec lecteur v2", "path": None,
        "receipt_sha256": None, "status": "en attente — aucun reçu acquis", "ok": None,
        "job_id": None, "terminal": {}, "source_head": None, "source_dirty_state": None,
        "execution_source_sha256": {}, "sources_unchanged": None, "available": False,
        "scope": "Les dossiers de démarrage v2 ne prouvent pas une nouvelle boucle diagnostic→recherche v2; le reçu historique reste distinct."})
    return receipts


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
            "observed_usage": merged, "missing_usage_fields": [key for key in totals if key not in merged]})
    totals.update(cli_call_receipts=len(groups), unreported_call_usage=sum(row["usage_reporting"] != "complete" for row in provenance),
        calls_without_any_reported_usage=sum(row["usage_reporting"] == "unreported" for row in provenance),
        partly_reported_call_usage=sum(row["usage_reporting"] == "partial" for row in provenance),
        calls_without_reported_cached_usage=sum("cached_input_tokens" in row["missing_usage_fields"] for row in provenance),
        call_provenance=provenance, model_tokens=totals["input_tokens"]+totals["output_tokens"], cost_usd=None,
        observed_lower_bounds=True,
        scope="All retained CLI-call receipts including original intent-scoped CLI traces, failures, pilots, diagnostics and copied archives; deduplicated by structured call ID or linked native invocation/thread identities. Original transport ACKs/calls.json are not additional calls. Missing/partial usage remains unknown, not a zero invoice.",
        cached_tokens_included_in_input=True, reasoning_tokens_included_in_output=True)
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
        interval = pair.get("interval", {})
        values = [interval.get(key) for key in ("gain", "lower", "upper")]
        valid_interval = all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in values)
        if not valid_interval or not values[1] <= values[0] <= values[2]:
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
        result["comparisons"][name]["interval"] = interval
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
        interval = pilot["comparisons"][role + "_vs_baseline"]["interval"]
        gain, low, high = (interval[key] for key in ("gain", "lower", "upper"))
        ax.errorbar(gain, y, xerr=[[gain-low], [high-gain]], fmt="o", color=colors[role], capsize=4)
        ax.annotate(f"{gain:.3f} [{low:.3f} ; {high:.3f}]", (gain, y), xytext=(0, 12),
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
        direct_ax.errorbar(gain, y, xerr=[[gain-low], [high-gain]], fmt="o", color="#6c62a5", capsize=4)
        direct_ax.annotate(f"{gain:.3f}", (gain, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
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
                     "[" + fmt(interval["lower"], 6) + " ; " + fmt(interval["upper"], 6) + "]", pair["status"]])
    direct = pilot["comparisons"]["agent_vs_native_codex"]["interval"]
    return (heading + "<p>Ce pilote compare le naïf, le kernel brut de l’agent amélioré, une proposition structurée sans outils et le véritable Codex CLI vanilla natif. "
            "Il ne contient aucun agent original upstream et n’entre pas dans les cohortes de l’étude §16. Les kernels bruts sont conservés, sans substitution par le meilleur parent retenu.</p>"
            + (f"<img src='{escape(figure.name)}' alt='Pilote GEMM : quatre artefacts bruts et comparaison directe agent contre Codex CLI natif'>" if figure else "")
            + table(["Artefact face au naïf", "Gain", "IC95% apparié", "Statut du reçu"], rows)
            + f"<p>Le gain géométrique de l’agent brut face au naïf vaut {fmt(pilot['comparisons']['agent_vs_baseline']['interval']['gain'], 6)} : la régression reste visible. "
            + f"Le ratio direct CLI natif/agent brut vaut {fmt(direct['gain'], 6)} [{fmt(direct['lower'], 6)} ; {fmt(direct['upper'], 6)}] ; un ratio inférieur à1 favorise le CLI natif. "
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
                y = row["gain"]
                ax.scatter(x+offset, y, color=color, s=40, zorder=3)
                if row.get("lower") and row.get("upper"):
                    ax.vlines(x+offset, row["lower"], row["upper"], color=color, alpha=.65)
        ax.axhline(1, color="#71828d", linestyle="--", linewidth=1)
        ax.set_xticks(range(3), ("Amélioré", "Original", "Vanilla"))
        ax.set_title(title)
        ax.set_ylabel("Latence naïve / latence retenue")
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("ROMEO : résultats finaux par campagne, intervalles appariés à 95 %")
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
    receipts = [[value["label"], value["status"], value["ok"] if value["ok"] is not None else "voir périmètre",
                 value["job_id"] or "sans nouveau job", value["terminal"].get("state", "sans objet / en attente"),
                 value["source_head"] or "non enregistré", value["path"] or "aucun reçu acquis"] for value in report["control_receipts"]]
    links = " · ".join(f"<a href='{escape(p.name)}'>{escape(p.name)}</a>" for p in sorted(output.glob("*.zip")))
    ci_html = (f"<a href='{escape(ci.get('url',''))}'>CI du SHA publié</a> : " + ("réussie sur les sources propres de ce SHA" if ci_ok else "ne qualifie pas ce checkout source"))
    consumed = report["project_observed_consumption"]
    consumption_html = table(["Périmètre", "Reçus appelsCLI", "Entrée observée", "Dont cache", "Sortie observée", "Appels à usage inconnu"],
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
Tous les artefacts finaux qualifiés : <strong>{'oui' if summary['scientific_qualified'] else 'non'}</strong>.
Les échecs, pertes et résultats encore inconnus restent visibles.</p>
<p><a href='https://github.com/wolf75222/agentic-cuda-optimizer'>Fork publié</a> · source du rapport : <code>{head}</code> · {ci_html}.</p>
<p>Checkout source : <strong>{'modifié' if source['dirty'] else 'propre'}</strong> ; {len(source['tracked_changes'])} chemins suivis modifiés et {len(source['untracked_files'])} fichiers non suivis non ignorés. Les chemins exacts sont enregistrés dans le JSON du rapport.</p>
<p>Agent local Codex, modèle demandé <code>gpt-6-luna</code>, effort <code>max</code> ; orchestration LangGraph et recherche TreeQuest.
L'inférence utilise le service OpenAI authentifié. Aucun plafond global de tokens, coût, appels ou durée cumulée n'est imposé.
Les limites techniques et les durées obligatoires de Slurm restent enregistrées.</p>
{frozen_pilot_panel(pilot, pilot_figure)}
<h2>Étude §16 : trois méthodes, un évaluateur final</h2>
<p>L'optimiseur amélioré, le véritable agent original au commit <code>1e9464d</code> et Codex vanilla natif sont des méthodes distinctes.
Le vanilla reçoit un seul prompt et dispose de ses outils de compilation et benchmark. Le parent glouton constitue une ablation supplémentaire.
Le protocole déclaré couvre {len(plan['workloads'])} opérations, {len(plan['seeds'])} graines et {len(plan['modes'])} modes : {planned_generations} générations et {planned_groups} groupes finaux. Chaque groupe fige toutes les sources avant de créer de nouvelles données sur le domaine de tailles déclaré.</p>
<p>Les contrastes PMPP prévus séparent la consultation ciblée (méthode complète face au guide permanent seul) et le guide permanent (guide seul face au mode sans PMPP).</p>
<p>Chaque groupe compare ses artefacts sur un même GPU réservé, avec oracle CPU indépendant, toutes les sorties, quatre sanitizers,
puis confirmations AB/BA et bootstrap des séries. Les chronométrages d'allocations différentes ne sont jamais regroupés.
Les mesures locales ne remplacent aucune performance ROMEO. Les points et intervalles ci-dessous montrent les campagnes individuelles ; un gain supérieur à1 favorise le candidat face au naïf.</p>
{chart_html}{study_aggregate_table(summary)}
<p>{escape(report['method_outcome_policy'])} Les points représentent l’artefact retenu, y compris un candidat de repli.</p>
<p>Les agrégats incluent les retours au noyau naïf et les régressions mesurées. Les absences ne reçoivent aucun gain inventé.
Cette étude sur des domaines déclarés à un seul kernel ne promet pas de gain universel ni de supériorité sur cuBLAS.</p>
<h2>Consommation, régressions et provenance</h2>
{consumption_html}
<p>Les tokens comptent les entrées et sorties observées, y compris les tokens d'entrée servis du cache lorsqu'ils sont inclus par le fournisseur.
Les traces conservent séparément les tokens mis en cache. Aucun coût en dollars n'est déduit sans reçu de facturation.
Le temps d'allocation inclut l'attente des modèles, les échanges SSH et les outils ; les CUDA Events ne mesurent que le kernel.
Les coûts absents sont inconnus. Le rapport JSON contient les compteurs de compilation, cache, recherche, configurations et premier gain utile lorsque la méthode les observe.</p>
<details><summary>Résultats des {planned_generations} campagnes, y compris les pertes et échecs</summary>{study_campaign_table(summary)}</details>
<h2>Contrôles réellement exécutés et limites</h2>{table(['Contrôle','Statut','Résultat','Job','Terminal','HEAD des sources exécutées','Reçu'], receipts)}
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
