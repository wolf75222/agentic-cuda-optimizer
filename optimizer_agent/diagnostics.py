"""Deterministic research triggers grounded in immutable campaign receipts.

Diagnostics explain an observation and ask the next research question. They do
not change a reward, promote a candidate, infer hardware counters, or turn a
local profile into target performance evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


POLICY_VERSION = "measured-diagnostic-v1"
STAGNATION_TRIALS = 3


def diagnostic_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()).hexdigest()


def evidence_reference(path: Path, root: Path, *, kind: str, identity: str | None = None) -> dict:
    path, root = path.resolve(), root.resolve()
    relative = path.relative_to(root)
    return {"kind": kind, "identity": identity, "path": str(relative),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_diagnostic(value: dict, root: Path, *, contract_hash: str,
                      environment_fingerprint: str) -> dict:
    value = deepcopy(value)
    identity = value.pop("diagnostic_id", None)
    if identity != "diag-" + diagnostic_digest(value):
        raise ValueError("Diagnostic content differs from its immutable identity")
    if value.get("contract_hash") != contract_hash or value.get("environment_fingerprint") != environment_fingerprint:
        raise ValueError("Diagnostic contract/environment differs from this campaign")
    root = root.resolve()
    for reference in value.get("evidence_references", []):
        path = (root / reference["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Diagnostic evidence is missing or outside this campaign")
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]:
            raise ValueError("Diagnostic evidence hash changed")
    value["diagnostic_id"] = identity
    return value


def evaluation_summary(value: dict | None) -> dict | None:
    if value is None:
        return None
    summary = {key: deepcopy(value[key]) for key in (
        "evaluation_id", "attempt_id", "candidate_id", "artifact_hash", "contract_hash",
        "ok", "status", "error", "interval", "critical_cases_ok", "incumbent_id",
        "incumbent_version", "kernel_resources", "diagnostics") if key in value}
    summary["cases"] = {}
    for case_id, case in value.get("cases", {}).items():
        if isinstance(case, dict):
            summary["cases"][case_id] = {key: deepcopy(case[key]) for key in (
                "latency_ms", "weight", "critical", "latency_series_ms", "series_means_ms",
                "kernel_resources", "status", "ok", "error") if key in case}
    return summary


def profile_summary(value: dict | None) -> dict | None:
    if value is None:
        return None
    # Preserve measured metric names/values/units and their environment exactly.
    return {key: deepcopy(value[key]) for key in (
        "ok", "error", "error_code", "metrics", "hardware_fingerprint",
        "execution_environment", "hardware_transfer_note", "profiling_conditions",
        "diagnostic_only", "ranking_eligible", "remote_unavailable", "profile_attempts") if key in value}


def build_diagnostic(*, run_id: str, trial_id: str, candidate_id: str | None,
                     strategy_family: str, contract_hash: str, environment_fingerprint: str,
                     stage: str, error: str | None, needs_research: bool,
                     proposal: dict | None, measurement: dict | None,
                     confirmation: dict | None, profile: dict | None,
                     previous_results: list[dict], evidence_references: list[dict],
                     candidate: dict | None, pmpp_guidance: str | None) -> dict:
    triggers, questions, observations = [], [], []
    if needs_research:
        triggers.append("insufficient_hypothesis")
        questions.append("Which consulted implementations and CUDA constraints cover the proposed technique or resolve the generation guard failure?")
    if measurement is not None:
        observations.append({"kind": "evaluation", "value": evaluation_summary(measurement)})
        if not measurement.get("ok"):
            status = measurement.get("status", "unknown")
            observations.append({"kind": "typed_failure", "status": status, "error": measurement.get("error")})
            if status in ("compilation_error", "numerical_error", "memory_error", "contract_violation"):
                triggers.append("unexplained_result")
                questions.append(f"Which source-level CUDA constraint explains the recorded {status}, and what compatible transformation addresses it?")
    if confirmation is not None:
        observations.append({"kind": "confirmation", "value": evaluation_summary(confirmation)})
        if confirmation.get("status") in ("not_improved", "inconclusive") and proposal is not None:
            triggers.append("unexplained_result")
            questions.append("Why was the proposed performance hypothesis unconfirmed by the recorded paired measurements, and which alternative fits these case sizes and resources?")
    if proposal is not None:
        required = ("hypothesis", "cuda_constraint", "expected_measurement", "references", "techniques")
        missing = [name for name in required if not proposal.get(name)]
        if missing:
            triggers.append("insufficient_hypothesis")
            observations.append({"kind": "missing_hypothesis_fields", "fields": missing})
            questions.append("Supply source-backed constraints and a falsifiable measured expectation for the incomplete hypothesis.")
    elif candidate_id is not None and stage == "result":
        triggers.append("insufficient_hypothesis")
        observations.append({"kind": "missing_proposal_receipt"})
        questions.append("Recover a technically sourced hypothesis before expanding this candidate's strategy.")
    compatible = [result for result in previous_results if result.get("strategy_family") == strategy_family
                  and result.get("valid") is True]
    recent = compatible[-STAGNATION_TRIALS:]
    if len(recent) == STAGNATION_TRIALS and all(not result.get("promoted") for result in recent):
        triggers.append("stagnation")
        observations.append({"kind": "stagnation", "policy": POLICY_VERSION,
                             "window": STAGNATION_TRIALS,
                             "trial_ids": [result["trial_id"] for result in recent],
                             "evaluation_ids": [result.get("evaluation_id") for result in recent]})
        questions.append("Which different documented transformation addresses three valid trials in this family without a confirmed promotion?")
    if profile is not None:
        observations.append({"kind": "profile", "value": profile_summary(profile)})
        if profile.get("ok"):
            questions.append("Relate the actual profile counters and compiler resource data to the unconfirmed hypothesis; respect the profile's recorded hardware environment.")
        else:
            observations.append({"kind": "profile_unavailable", "reason": profile.get("error")})
    if error:
        observations.append({"kind": "recorded_error", "text": error})
    value = {"schema_version": 1, "policy_version": POLICY_VERSION,
             "run_id": run_id, "trial_id": trial_id, "candidate_id": candidate_id,
             "strategy_family": strategy_family, "contract_hash": contract_hash,
             "environment_fingerprint": environment_fingerprint, "stage": stage,
             "triggers": list(dict.fromkeys(triggers)), "research_questions": list(dict.fromkeys(questions)),
             "requires_targeted_research": bool(triggers), "observations": observations,
             "proposal_hypothesis": ({key: deepcopy(proposal[key]) for key in (
                 "hypothesis", "expected_measurement", "cuda_constraint", "techniques", "references") if key in proposal}
                 if proposal is not None else None),
             "candidate": candidate, "evidence_references": evidence_references,
             "pmpp_guidance": pmpp_guidance,
             "provenance_note": "Observed evaluation/profile evidence is distinct from hypotheses; no absent counter or target performance is inferred."}
    value["diagnostic_id"] = "diag-" + diagnostic_digest(value)
    return value
