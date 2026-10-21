#!/usr/bin/env python3
"""Documentary AC-02 control: plan by default; real Codex/web only with --run.

A completed campaign supplies an unchanged, authenticated historical baseline
and hardware identity. Nothing is compiled, evaluated, integrated or finalized.
The control asks for an uncovered technique BEFORE the model call; it never
edits a returned payload or a dossier to obtain the expected refusal/refresh.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import sys
from tempfile import TemporaryDirectory
import traceback
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from optimizer_agent.campaign import (CampaignRuntime, RecordedCodex, WorkloadSuite, Candidate,
    Evaluator, LaunchRule, WorkloadContract, device_identity, digest, read_json, write_json)
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.codex_backend import CandidateGenerator, CodexBackend, CodexConfig, Proposal, default_codex_executable
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.research_service import ResearchConfig, ResearchDossier, ResearchService, NeedsResearch
from optimizer_agent.diagnostics import verify_diagnostic
from optimizer_agent.search_policy import SearchPolicy
from use_harness import BufferArgument
from optimizer_agent.workload_semantics import operation_spec
from scripts.acceptance_intent import controller_provenance

RECEIPT = "research-refresh-acceptance.json"
TECHNIQUES = ("register_blocking", "vectorized_load", "warp_shuffle", "async_copy", "thread_block_clusters")
BOOTSTRAP_FILES = ("result.json", "generation/manifest.json", "generation/config.json",
                   "generation/environment.json", "generation/contract.json",
                   "generation/baseline-measurement.json", "generation/evaluator_data/baseline.cu",
                   "worker/hardware.json", "worker/slurm-session.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path, base):
    return {"path": str(Path(path).relative_to(base)), "sha256": sha(path)}


def inspect_bootstrap(directory, result_sha256=None):
    """Read only the named public artifacts; never inspect private final inputs."""
    directory = Path(directory).resolve()
    for name in BOOTSTRAP_FILES:
        if not (directory / name).is_file():
            raise ValueError("Missing completed-campaign bootstrap artifact: " + name)
    hashes = {name: sha(directory / name) for name in BOOTSTRAP_FILES}
    if result_sha256 and hashes["result.json"] != result_sha256:
        raise ValueError("Bootstrap result SHA differs from the explicitly pinned receipt")
    result, manifest = (read_json(directory / name) for name in ("result.json", "generation/manifest.json"))
    terminal = result.get("worker_terminal", {})
    if (result.get("execution_complete") is not True or result.get("generation_complete") is not True
            or manifest.get("certification_status") != "generation_complete"
            or terminal.get("terminal") is not True or terminal.get("state") != "COMPLETED"
            or terminal.get("exit_code") != "0:0"):
        raise ValueError("Bootstrap must be a completed generation with its exact clean terminal receipt")
    config = CampaignConfig.load(directory / "generation/config.json")
    if config.experimental_mode is not None or not (config.research_required and config.web_search_required):
        raise ValueError("The control requires a normal research-enabled bootstrap, without ablation bypass")
    hardware, ownership = (read_json(directory / name) for name in ("worker/hardware.json", "worker/slurm-session.json"))
    job = str(result.get("job_id", ""))
    if (not job.isdigit() or int(job) <= 0 or any(str(value) != job for value in
            (terminal.get("job_id"), hardware.get("slurm_job_id"), ownership.get("job_id")))
            or result.get("session_id") != ownership.get("session_id")):
        raise ValueError("Bootstrap job/session ownership is inconsistent")
    properties = hardware["device_properties"]
    environment = read_json(directory / "generation/environment.json")
    contract = read_json(directory / "generation/contract.json")
    baseline = read_json(directory / "generation/baseline-measurement.json")
    original_config = read_json(directory / "generation/config.json")
    if (result.get("run_id") != config.run_id or manifest.get("run_id") != config.run_id
            or result.get("config") != original_config or manifest.get("config") != original_config
            or contract["contract"].get("contract_id") != config.run_id):
        raise ValueError("Bootstrap original configuration/run identity is inconsistent")
    if (environment != device_identity(properties) or environment != device_identity(baseline.get("environment", {}))
            or result.get("environment") != environment or manifest.get("environment") != environment):
        raise ValueError("Bootstrap archived hardware identities disagree")
    if (any(value != contract["contract_hash"] for value in
                   (baseline.get("contract_hash"), result.get("contract_hash"), manifest.get("contract_hash")))
            or baseline.get("ok") is not True or baseline.get("status") != "measured"):
        raise ValueError("Bootstrap contract or historical baseline is inconsistent")
    rule = LaunchRule(kind="square_2d", block=(16, 16, 1)) if config.workload == "matmul" else LaunchRule(block=(128, 1, 1))
    candidate = Candidate("baseline", directory / "generation/evaluator_data/baseline.cu", config.workload, rule)
    if baseline.get("candidate_id") != "baseline" or baseline.get("artifact_hash") != candidate.artifact_hash:
        raise ValueError("Bootstrap baseline source/launch artifact differs from its measurement")
    reports = [series["report"] for case in baseline.get("cases", {}).values() for series in case.get("series", [])]
    if not reports:
        raise ValueError("Bootstrap needs original native baseline series, not only aggregate latencies")
    for report in reports:
        raw = report.get("latency_samples_ms", [])
        if (report.get("ok") is not True or str(report.get("slurm_job_id")) != job
                or device_identity(report.get("environment", {})) != environment
                or report.get("compilation_identity", {}).get("source") != candidate.source.read_text()
                or not raw or any(not isinstance(x, (float, int)) or isinstance(x, bool)
                                  or not math.isfinite(x) or x <= 0 for x in raw)):
            raise ValueError("Bootstrap native series do not authenticate the baseline/job/hardware/source")
    origin = {"directory": directory, "config": config, "properties": properties, "contract": contract,
            "baseline": baseline, "hashes": hashes, "terminal": terminal, "job_id": job,
            "session_id": ownership["session_id"], "historical_native_series": len(reports)}
    # Reconstruct the REAL Evaluator hash, which includes absolute archived
    # paths and search-input bytes. Generate only a temporary search suite;
    # never invoke the final factory or write into the original campaign.
    with TemporaryDirectory(prefix="research-bootstrap-auth-") as temporary:
        _, _, evaluator, inputs = archived_evaluator(origin, Path(temporary))
        if evaluator.contract_hash != contract["contract_hash"]:
            raise ValueError("Recomputed archived Evaluator/suite/input contract differs")
        if set(baseline.get("cases", {})) != {case.case_id for case in evaluator._suites["performance"]}:
            raise ValueError("Archived baseline does not contain every exact public performance case")
        recomputed_latencies = {}
        for case in evaluator._suites["performance"]:
            measured = baseline["cases"][case.case_id]
            if (measured.get("weight") != case.weight or measured.get("critical") != case.critical
                    or len(measured.get("series", [])) != evaluator.contract.benchmark_series
                    or not isinstance(measured.get("latency_ms"), (int, float))
                    or not math.isfinite(measured["latency_ms"]) or measured["latency_ms"] <= 0):
                raise ValueError("Archived baseline case metadata/series are incomplete")
            for series in measured["series"]:
                report = series["report"]
                if (len(report["latency_samples_ms"]) != evaluator.contract.launches_per_series
                        or series.get("raw_ms") != report["latency_samples_ms"]
                        or report.get("guard_readonly_ok") is not True
                        or report.get("guard_outputs_poisoned") is not True):
                    raise ValueError("Archived baseline lacks exact raw timings or native memory guards")
                # These are the exact estimators in Evaluator._measure_case
                # and Evaluator.benchmark; no timing or score is remeasured.
                if series.get("mean_ms") != statistics.fmean(report["latency_samples_ms"]):
                    raise ValueError("Archived baseline series mean differs from its raw native timings")
            recomputed_latencies[case.case_id] = statistics.median(series["mean_ms"] for series in measured["series"])
            if measured["latency_ms"] != recomputed_latencies[case.case_id]:
                raise ValueError("Archived baseline aggregate differs from its raw native series estimator")
        for record in (result, manifest):
            if "baseline_latencies_ms" in record and record["baseline_latencies_ms"] != recomputed_latencies:
                raise ValueError("Archived baseline latencies differ from the completed result/manifest")
        authentication = {"original_result_sha256":hashes["result.json"],
            "explicit_result_pin_checked":result_sha256 is not None,
            "recomputed_evaluator_contract_hash":evaluator.contract_hash,
            "workload_contract_hash":evaluator.contract.contract_hash,
            "suite_hash":evaluator._suite_hash,"search_input_hashes":inputs,
            "baseline_source_sha256":sha(candidate.source),"baseline_artifact_hash":candidate.artifact_hash,
            "search_case_ids":[case.case_id for case in evaluator._suites["search"]],
            "baseline_case_ids":list(recomputed_latencies),"recomputed_historical_latencies_ms":recomputed_latencies,
            "native_series_count":len(reports),"archived_environment":environment,
            "job_id":job,"session_id":ownership["session_id"],"terminal":terminal,
            "checked_components":["completed result and manifest", "original configuration/run identity",
                "exact owned clean terminal job/session", "unchanged native hardware/toolchain/build identity",
                "baseline source and launch artifact", "all public baseline cases and raw guarded native series",
                "deterministic public input SHA256", "full original Evaluator contract/suite/path/input hash"],
            "scope":"Local consistency and explicit receipt pin of an historical archive; no current hardware observation, new GPU validation or optimizer-quality result"}
    hashes.update(inputs)
    origin["authenticated_search_input_hashes"] = inputs
    origin["authentication"] = authentication
    return origin


def archived_evaluator(origin, directory, runner=None):
    """Replay archive identity from immutable public inputs, without evaluation."""
    config, original = origin["config"], origin["directory"] / "generation/evaluator_data"
    data = Path(directory) / "evaluator_data"
    data.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(original / "baseline.cu", data / "baseline.cu")
    suite = WorkloadSuite(config.workload, data, config)
    baseline = suite.baseline(REPO / config.baseline_source)
    if sha(baseline.source) != sha(original / "baseline.cu"):
        raise ValueError("Current baseline reconstruction differs from the authenticated historical source")
    cases, inputs = suite.cases(), {}
    archived_cases = []
    for case in cases:
        arguments = []
        for argument in case.request.arguments:
            if isinstance(argument, BufferArgument) and argument.input_file:
                generated = Path(argument.input_file)
                archived = original / generated.relative_to(data)
                if (not archived.resolve().is_relative_to(original.resolve()) or not archived.is_file()
                        or sha(archived) != sha(generated)):
                    raise ValueError("Archived search input differs from deterministic public suite: " + str(archived))
                inputs[str(archived.relative_to(origin["directory"]))] = sha(archived)
                argument = replace(argument, input_file=str(archived))
            arguments.append(argument)
        # The ordinary suite hashes an empty request source and supplies the
        # candidate only at evaluation time. Preserve that exact ABI identity.
        request = replace(case.request, arguments=arguments)
        archived_cases.append(replace(case, request=request))
    contract = WorkloadContract(**origin["contract"]["contract"])
    metadata, semantics = json.loads(contract.metadata_json), operation_spec(config.workload)
    if (contract.operation != config.workload or contract.signature != suite.signature
            or contract.allowed_precision not in (None, "float32")
            or metadata.get("precision") != "float32"
            or metadata.get("memory_policy") != semantics["memory_policy"]
            or metadata.get("layout") != semantics["layout"]
            or metadata.get("environment") != device_identity(origin["properties"])):
        raise ValueError("Archived workload ABI/precision/environment is inconsistent")
    evaluator = Evaluator(runner or ArchivedPropertiesOnlyRunner(origin["properties"]), contract, suite.oracle,
        {"short": archived_cases[:2], "search": archived_cases,
         "performance": [case for case in archived_cases if case.metadata["n"] in config.search_sizes]},
        final_suite_factory=lambda: (_ for _ in ()).throw(RuntimeError("No final suite in documentary acceptance")),
        final_suite_identity=digest({"generator":"benchmark_workloads-v1", "sizes":config.final_sizes,
                                     "seed_commitment":digest(config.seed + 1000003)}))
    return suite, baseline, evaluator, inputs


class DocumentaryRuntime(CampaignRuntime):
    """Load a verified historical context; inherit all three research guards.

    The ordinary constructor measures a new GPU baseline and hashes relocated
    input paths. This restricted adapter authenticates the original hash first
    and supports only the existing research/generate/diagnose phase methods.
    """
    def __init__(self, config, runner, directory, *, model_backend, origin):
        control_fields = {"actions", "max_trials", "max_repairs_per_trial", "profile",
                          "autotuning_enabled", "model", "reasoning_effort"}
        if any(value != origin["config"].to_dict().get(key) for key, value in config.to_dict().items()
               if key not in control_fields):
            raise ValueError("Documentary adapter changed the authenticated workload configuration")
        if not all((origin["directory"] / name).is_file() and sha(origin["directory"] / name) == value
                   for name, value in origin["hashes"].items()):
            raise ValueError("Authenticated bootstrap artifact changed before documentary initialization")
        self.config, self.runner, self.directory = config, runner, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.generation_only, self.rehost_context = True, None
        self.max_repairs = config.max_repairs_per_trial
        self.properties, self.environment = dict(origin["properties"]), device_identity(origin["properties"])
        self.environment_hash = digest(self.environment)
        self.suite, self.baseline, self.evaluator, inputs = archived_evaluator(origin, self.directory, runner)
        self.performance_case_ids = [case.case_id for case in self.evaluator._suites["performance"]]
        self.contract_hash = self.evaluator.contract_hash
        if self.contract_hash != origin["contract"]["contract_hash"] or inputs != origin["authenticated_search_input_hashes"]:
            raise ValueError("Documentary context changed its authenticated archive identity")
        self.baseline_measurement = read_json(origin["directory"] / "generation/baseline-measurement.json")
        self.baseline_latencies = self.latencies(self.baseline_measurement)
        self.store = ExperimentStore(self.directory / "experiments.sqlite")
        self.policy = SearchPolicy(self.store, config.run_id, self.candidate_state(self.baseline),
            self.baseline_latencies, actions=list(config.actions), contract_fingerprint=self.contract_hash,
            environment_fingerprint=self.environment_hash, budget_limits={"trials":1}, seed=config.seed)
        self.raw_backend = model_backend
        self.backend = RecordedCodex(model_backend, self.store, config.run_id)
        self.generator = CandidateGenerator(self.backend)
        self.research_service = ResearchService(self.backend, self.directory / "research",
            ResearchConfig(required=True, web_required=True))
        semantics = operation_spec(config.workload)
        metadata = json.loads(self.evaluator.contract.metadata_json)
        if metadata.get("memory_policy") != semantics["memory_policy"]:
            raise ValueError("Archived memory/precision contract differs from the normal generator contract")
        self.memory_policy = metadata["memory_policy"]
        self.workload_description = {**semantics, "contract_hash":self.contract_hash,
            "operation":config.workload,"signature":self.evaluator.contract.signature,
            "precision":"float32 input/output, full float32 arithmetic", "layout":metadata["layout"],
            "memory_policy":self.memory_policy,"search_sizes":list(config.search_sizes),
            "shape_domain":"all positive n; tails required","atol":2e-4,"rtol":2e-4,
            "single_kernel":True,"environment":self.environment}
        write_json(self.directory / "config.json", config.to_dict())
        write_json(self.directory / "environment.json", self.environment)
        write_json(self.directory / "contract.json", origin["contract"])
        write_json(self.directory / "archival-context.json", {"kind":"documentary_only_archive_adapter",
            "authentication":origin["authentication"],"archived_files_sha256":origin["hashes"],
            "recomputed_evaluator_contract_hash":self.contract_hash,"authenticated_search_inputs":inputs,
            "hardware_currently_observed":False,"historical_baseline_used_only_as_unranked_root":True,
            "baseline_gpu_cost_transferred":False,"research_generate_diagnose_methods":"Inherited unchanged from CampaignRuntime",
            "prohibited_phases":["evaluate","integrate","finalize"]})

    def evaluate(self, state):
        raise RuntimeError("Documentary adapter forbids evaluation")

    def integrate(self, state):
        raise RuntimeError("Documentary adapter forbids integration")

    def finalize(self, state):
        raise RuntimeError("Documentary adapter forbids finalization")


class ArchivedPropertiesOnlyRunner:
    """Archived metadata is a research target, never a current GPU observation."""
    def __init__(self, properties):
        self.properties, self.gpu_attempts = dict(properties), []

    def get_device_properties(self, device):
        if device != 0:
            raise ValueError("The archived context has only device0")
        return dict(self.properties)

    def run(self, *args, **kwargs):
        self.gpu_attempts.append("run")
        raise RuntimeError("Documentary acceptance forbids every native/GPU request")

    def run_sanitizer(self, *args, **kwargs):
        self.gpu_attempts.append("sanitizer")
        raise RuntimeError("Documentary acceptance forbids sanitizer/GPU requests")

    def require_complete(self):
        if self.gpu_attempts:
            raise RuntimeError("Unexpected GPU request attempted during archival bootstrap")


class PromptControls:
    """Insert declared controls before CLI; return the exact backend result."""
    def __init__(self, backend, directory):
        self.backend, self.directory = backend, Path(directory)
        self.pending, self.invocations = None, []
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def calls(self):
        return self.backend.calls

    @property
    def config(self):
        return self.backend.config

    def queue(self, stage, instruction):
        if self.pending:
            raise RuntimeError("An earlier controlled proposal has not reached the CLI")
        self.pending = (stage, instruction)

    def generate_json(self, prompt, schema, **options):
        kernel = {"source", "kernel", "techniques", "parent_id"} <= set(schema.get("properties", {}))
        record = {"kernel_proposal": kernel, "purpose": options.get("purpose"),
                  "original_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        if kernel and self.pending:
            stage, instruction = self.pending
            self.pending = None
            marker = "Context:\n"
            if marker not in prompt:
                raise ValueError("The actual generator prompt lost its context boundary")
            prefix, context = prompt.split(marker, 1)
            prompt = prefix + "\nExplicit documentary acceptance control (not optimizer-quality data):\n" + instruction + "\n" + marker + context
            path = self.directory / (stage + "-prompt.txt")
            with path.open("x") as stream:
                stream.write(prompt)
            record.update(stage=stage, instruction=instruction, outgoing_prompt=str(path),
                          outgoing_prompt_sha256=sha(path))
        else:
            record["stage"] = "research_service"
        self.invocations.append(record)
        write_json(self.directory / "invocations.json", self.invocations)
        try:
            result = self.backend.generate_json(prompt, schema, **options)
        except BaseException as error:
            record.update(status="uncertain_or_failed", error=str(error))
            write_json(self.directory / "invocations.json", self.invocations)
            raise
        record.update(status="completed", call_id=result.call_id,
                      returned_payload_sha256=digest(result.payload))
        write_json(self.directory / "invocations.json", self.invocations)
        return result


def generate_control(runtime, controls, state, stage, instruction):
    controls.queue(stage, instruction)
    state.update(runtime.generate(state))
    if state.get("error") or not state.get("candidate_id"):
        raise RuntimeError(stage + " did not produce an admitted proposal: " + str(state.get("error")))
    proposal = read_json(runtime.trial_dir(state) / f'proposal-{state["repair"]}-{state["research_round"]}.json')
    return proposal


def exercise(runtime, controls, *, uncovered_technique=None):
    """Use unmodified runtime research/generation/diagnostic guards, no evaluator."""
    state = {"run_id": runtime.config.run_id, "trials_done": 0, **runtime.initialize({})}
    state.update(runtime.select(state))
    initial_dossier_missing = not (runtime.trial_dir(state) / "dossier.json").exists()
    state.update(runtime.research(state))
    if not state.get("research_ready"):
        raise RuntimeError("Initial real mandatory research blocked: " + str(state.get("error")))
    dossier_path = runtime.trial_dir(state) / "dossier.json"
    initial = ResearchDossier.from_dict(read_json(dossier_path))
    initial_guard_ready = initial.research_ready(runtime.contract_hash, runtime.trial(state)["action"], runtime.environment_hash)
    archive = runtime.directory / "acceptance-evidence"
    archive.mkdir(exist_ok=True)
    shutil.copyfile(dossier_path, archive / "initial-dossier.json")
    dossier_hash = sha(dossier_path)
    first = generate_control(runtime, controls, state, "initial-compatible",
        "Produce a genuine proposal supported by this dossier. Use only covered techniques; preserve every normal guard.")
    first_id = state["candidate_id"]
    before_compatible = len(controls.invocations)
    # The repair field names a local variation slot here. No failed compilation,
    # validation or measurement is invented for this documentary adjustment.
    state.update(repair=1)
    second = generate_control(runtime, controls, state, "compatible-adjustment",
        "Make a compatible local adjustment of the supplied source parent, entirely within the same dossier coverage. "
        "No compilation or measurement has occurred; this is an explicitly controlled variation slot, not a claimed failed repair.")
    second_id = state["candidate_id"]
    between = controls.invocations[before_compatible:]
    exact_reuse = sha(dossier_path) == dossier_hash and len(between) == 1 and between[0]["kernel_proposal"]
    if not exact_reuse:
        raise RuntimeError("Compatible adjustment did not reuse the exact dossier without a research call")
    technique = uncovered_technique or next((name for name in TECHNIQUES if name not in initial.covered_techniques), None)
    if not technique or technique in initial.covered_techniques:
        raise RuntimeError("The declared uncovered-technique control is already covered; do not alter the dossier")
    generations_before = [e for e in runtime.store.events(runtime.config.run_id) if e["kind"] == "generation"]
    candidates_before = {p.name: sha(p) for p in (runtime.directory / "candidates").glob("*.cu")}
    state.update(repair=2)
    controls.queue("uncovered-technique", "For this explicit refusal control, propose a transformation requiring the real CUDA technique "
        + repr(technique) + ". Declare that exact name in techniques even though the current dossier does not cover it. "
        "This is a request for additional research, not authorization to execute. Cite only actually read source IDs; invent no support. "
        "Preserve the parent, ABI, contract and all other guards. Return the actual complete structured proposal.")
    blocked = runtime.generate(state)
    state.update(blocked)
    raw = controls.calls[-1]
    refusal = {"needs_research_returned":blocked.get("needs_research") is True,
        "exact_semantic_guard_reason":"uncovered technique" in str(blocked.get("error")),
        "model_actually_declared_control_technique":technique in raw.payload.get("techniques", []),
        "no_new_generation_event":generations_before == [e for e in runtime.store.events(runtime.config.run_id) if e["kind"] == "generation"],
        "no_new_candidate_or_source_change":candidates_before == {p.name:sha(p) for p in (runtime.directory / "candidates").glob("*.cu")},
        "no_accepted_proposal_file":not (runtime.trial_dir(state) / 'proposal-2-1.json').exists()}
    write_json(archive / "blocked-refusal.json", {"checks":refusal,"runtime_update":blocked,
        "technique":technique,"call_id":raw.call_id,"raw_payload_sha256":digest(raw.payload),
        "candidates_before":candidates_before})
    if not all(refusal.values()):
        raise RuntimeError("Model/control did not yield the required semantic refusal before a new candidate; leave incomplete")
    blocked_call_id = raw.call_id
    state.update(runtime.diagnose(state))
    diagnostic = runtime.diagnostic_context(state)
    if (not diagnostic["requires_targeted_research"]
            or technique not in diagnostic["proposal_hypothesis"].get("techniques", [])):
        raise RuntimeError("Actual diagnostic lost the blocked technique or refresh requirement")
    state.update(runtime.research(state))
    if not state.get("research_ready"):
        raise RuntimeError("Actual targeted research blocked: " + str(state.get("error")))
    renewed = ResearchDossier.from_dict(read_json(dossier_path))
    if (renewed.research_id == initial.research_id or diagnostic["diagnostic_id"] not in renewed.diagnostic_ids
            or technique not in renewed.covered_techniques):
        raise RuntimeError("Real renewed dossier does not cover the requested technique/linkage; leave incomplete")
    final = generate_control(runtime, controls, state, "newly-covered-proposal",
        "Now propose the transformation requiring " + repr(technique) + " only if this renewed dossier genuinely covers it. "
        "Declare the technique and cite actual renewed source IDs. Preserve the normal parent, ABI and contract guards.")
    if technique not in final["techniques"]:
        raise RuntimeError("Final model proposal omitted the newly covered technique; leave incomplete")
    final_id = state["candidate_id"]
    checks = {"new_family_real_applicable_dossier": initial_dossier_missing and initial_guard_ready is True,
        "compatible_adjustment_exact_dossier_reuse_without_research": exact_reuse and sha(archive / "initial-dossier.json") == dossier_hash,
        "compatible_adjustment_uses_previous_source_parent": second["source_parent_id"] == first_id,
        "compatible_proposal_changes_source_launch_or_parameters": any(
            first[field] != second[field] for field in ("source", "launch_rule", "parameters")),
        "uncovered_technique_blocked_before_candidate": all(refusal.values()),
        "diagnostic_contains_exact_blocked_technique": technique in diagnostic["proposal_hypothesis"]["techniques"],
        "targeted_new_dossier_covers_technique": technique in renewed.covered_techniques,
        "targeted_dossier_links_immutable_diagnostic": diagnostic["diagnostic_id"] in renewed.diagnostic_ids,
        "new_covered_proposal_admitted": bool(final_id and technique in final["techniques"]),
        "only_three_admitted_generation_events": len([e for e in runtime.store.events(runtime.config.run_id) if e["kind"] == "generation"]) == 3,
        "no_trial_integration_or_measurement": not list((runtime.directory / "trials").glob("*/evaluation-*.json"))
            and not list((runtime.directory / "trials").glob("*/result.json")),
        "no_final_holdout_exposed": not (runtime.directory / "holdout-exposed.json").exists()}
    return {"checks": checks, "initial_research_id": initial.research_id, "initial_dossier_sha256": dossier_hash,
        "renewed_research_id": renewed.research_id, "renewed_dossier_sha256": sha(dossier_path),
        "diagnostic_id": diagnostic["diagnostic_id"], "uncovered_technique": technique,
        "blocked_call_id": blocked_call_id, "initial_call_id": first["call_id"],
        "compatible_call_id": second["call_id"], "final_call_id": final["call_id"],
        "admitted_candidates": [first_id, second_id, final_id],
        "runtime_variation_slots": {"0": "initial", "1": "compatible documentary adjustment, no failed compilation claimed",
                                    "2": "uncovered request then genuinely covered proposal after research"}}


def _inspect_initial_resume_source(directory, *, receipt_sha256, manifest_sha256):
    """Authenticate only the observed seven-call targeted-research refusal.

    The v1 control did not serialize its last in-memory state. Its exact phase
    is reconstructed from the pinned event sequence, SQL rows and CLI receipts;
    no successful research/applicability condition is assigned by this adapter.
    """
    directory = Path(directory).resolve()
    if not receipt_sha256 or not manifest_sha256:
        raise ValueError("Resume requires explicit receipt and proof-manifest SHA256 pins")
    receipt_path, manifest_path = directory / RECEIPT, directory / "proof-sha256.json"
    if sha(receipt_path) != receipt_sha256 or sha(manifest_path) != manifest_sha256:
        raise ValueError("Resume receipt/proof-manifest differs from its explicit pin")
    receipt, manifest = read_json(receipt_path), read_json(manifest_path)
    if (receipt.get("kind") != "documentary_research_refresh_acceptance"
            or receipt.get("status") != "incomplete" or receipt.get("ok") is not False
            or not str(receipt.get("error", "")).startswith("Actual targeted research blocked: ")
            or receipt.get("new_gpu_requests_attempted") != 0 or receipt.get("bootstrap_files_unchanged") is not True
            or receipt.get("proof_manifest") != {"path":"proof-sha256.json", "sha256":manifest_sha256}):
        raise ValueError("Resume supports only the completed targeted-research refusal, with zero GPU requests")
    files = manifest.get("files", {})
    if (manifest.get("hash_algorithm") != "sha256" or not isinstance(files, dict) or not files
            or set(manifest.get("excluded_paths", [])) != {RECEIPT, "proof-sha256.json"}):
        raise ValueError("Unsupported resume proof manifest")
    for name, expected in files.items():
        path = directory / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or not path.is_file()
                or path.is_symlink() or not path.resolve().is_relative_to(directory) or sha(path) != expected):
            raise ValueError("Resume source artifact is missing, unsafe or changed: " + name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if (actual != set(files) | {RECEIPT, "proof-sha256.json"}
            or any(path.is_symlink() for path in directory.rglob("*"))):
        raise ValueError("Resume source contains unmanifested artifacts or symlinks")
    intent = read_json(directory / "run-intent.json")
    if not intent.get("source_provenance") or intent["source_provenance"] != receipt.get("source_provenance"):
        raise ValueError("Resume source-controller provenance differs between intent and receipt")
    origin = inspect_bootstrap(intent["bootstrap"], receipt["historical_bootstrap"]["file_hashes"]["result.json"])
    if (origin["hashes"] != receipt["historical_bootstrap"]["file_hashes"]
            or intent.get("bootstrap_hashes") != origin["hashes"]
            or intent.get("internal_campaign_run_id") != origin["config"].run_id):
        raise ValueError("Resume bootstrap identity differs from the authenticated original")
    events = read_json(directory / "events.json")
    kinds = [event["kind"] for event in events]
    expected_kinds = ["campaign_initialized", "checkpoint", "selection", "selection", "model", "model",
        "research", "model", "generation", "model", "generation", "model", "diagnostic", "model", "model", "research_blocked"]
    if kinds != expected_kinds or receipt["error"] != "Actual targeted research blocked: " + events[-1]["payload"]["error"]:
        raise ValueError("Resume phase is not the observed initial/reuse/refusal/diagnostic/research-blocked sequence")
    db_path = directory / "campaign/experiments.sqlite"
    if any(db_path.with_name(db_path.name + suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("Resume requires a settled archived SQLite database")
    with closing(sqlite3.connect(db_path.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db_events = [dict(row, payload=json.loads(row["payload"])) for row in db.execute("SELECT * FROM events ORDER BY event_id")]
        trials = [dict(row) for row in db.execute("SELECT * FROM trials")]
        candidates = [dict(row, payload=json.loads(row["payload"])) for row in db.execute("SELECT * FROM candidates")]
        forbidden = sum(db.execute("SELECT count(*) FROM " + table).fetchone()[0]
                        for table in ("evaluations", "reward_claims", "integration_events"))
        uncertain = [row for row in db.execute("SELECT amount FROM reservations WHERE actual IS NULL")
                     if json.loads(row["amount"]).get("model_calls", 0)]
    if (db_events != events or len(trials) != 1 or trials[0]["status"] != "pending"
            or trials[0]["result"] is not None or forbidden or uncertain or len(candidates) != 3):
        raise ValueError("Resume SQLite/events/pending-trial state is incompatible or uncertain")
    trial_id = trials[0]["trial_id"]
    if any(event["payload"].get("trial_id", trial_id) != trial_id for event in events):
        raise ValueError("Resume events name more than one trial")
    invocations = read_json(directory / "campaign/acceptance-controls/invocations.json")
    expected_stages = ["research_service", "research_service", "initial-compatible", "compatible-adjustment",
                       "uncovered-technique", "research_service", "research_service"]
    calls = receipt.get("model_calls", [])
    if (len(calls) != 7 or [row.get("stage") for row in invocations] != expected_stages
            or [row.get("kernel_proposal") for row in invocations] != [False,False,True,True,True,False,False]
            or any(row.get("status") != "completed" for row in invocations)
            or [row.get("call_id") for row in invocations] != [row.get("call_id") for row in calls]
            or [event["payload"].get("call_id") for event in events if event["kind"] == "model"] != [row["call_id"] for row in calls]
            or len({row["call_id"] for row in calls}) != 7):
        raise ValueError("Resume needs the seven completed, distinct original CLI receipts in exact order")
    traces = []
    for call, invocation in zip(calls, invocations):
        call_id = call["call_id"]
        if len(call_id) != 32 or any(char not in "0123456789abcdef" for char in call_id):
            raise ValueError("Unsafe original model call identity")
        trace = read_json(directory / "campaign/model" / (call_id + ".json"))
        if (trace.get("call_id") != call_id or trace.get("status") != "completed" or trace.get("returncode") != 0
                or trace.get("purpose") != invocation.get("purpose")
                or trace.get("usage") != call["usage"] or trace.get("model") != "gpt-6-luna"
                or trace.get("reasoning_effort") != "max"
                or digest(trace.get("payload")) != invocation.get("returned_payload_sha256")):
            raise ValueError("Original completed CLI payload/usage identity differs")
        traces.append(trace)
    initial_path = directory / "campaign/acceptance-evidence/initial-dossier.json"
    dossier_path = directory / "campaign/trials" / trial_id / "dossier.json"
    if sha(initial_path) != sha(dossier_path):
        raise ValueError("The blocked refresh must still retain the exact initial dossier")
    initial_dict = read_json(initial_path)
    prompt_dossier_matches = []
    for invocation in invocations[2:5]:
        prompt_path = Path(invocation["outgoing_prompt"])
        if (not prompt_path.resolve().is_relative_to(directory) or sha(prompt_path) != invocation["outgoing_prompt_sha256"]
                or read_json(directory / "run-intent.json")["run_id"] != receipt["run_id"]):
            raise ValueError("Original controlled prompt is not bound to this source")
        context = json.loads(prompt_path.read_text().split("Context:\n", 1)[1])
        if context["research"] != initial_dict:
            raise ValueError("The three original controls did not use the exact same dossier")
        prompt_dossier_matches.append(context["research"] == initial_dict)
    blocked = read_json(directory / "campaign/acceptance-evidence/blocked-refusal.json")
    if (not blocked.get("checks") or not all(value is True for value in blocked["checks"].values())
            or blocked.get("call_id") != calls[4]["call_id"]
            or blocked.get("raw_payload_sha256") != digest(traces[4]["payload"])
            or blocked.get("runtime_update", {}).get("needs_research") is not True):
        raise ValueError("Original semantic refusal lacks its exact completed payload")
    generations = [event["payload"]["candidate_id"] for event in events if event["kind"] == "generation"]
    diagnostic = next(event["payload"] for event in events if event["kind"] == "diagnostic")
    state = {"run_id":origin["config"].run_id, "initialized":True, "trials_done":0,
        "trial_id":trial_id, "candidate_id":generations[1], "repair":2, "research_round":2,
        "research_ready":False, "needs_research":True, "valid":False, "retryable":False,
        "error":events[-1]["payload"]["error"], "termination_reason":None,
        "diagnostic_id":diagnostic["diagnostic_id"], "research_refresh_required":True}
    return {"directory":directory, "receipt":receipt, "manifest":manifest, "intent":intent, "origin":origin,
        "source_hashes":{**files, RECEIPT:receipt_sha256, "proof-sha256.json":manifest_sha256},
        "events":events, "calls":calls, "invocations":invocations, "traces":traces,
        "state":state, "candidate_ids":generations, "technique":blocked["technique"],
        "initial_dossier":initial_dict, "initial_dossier_sha256":sha(initial_path),
        "dossier_reuse_verified":all(prompt_dossier_matches) and sha(initial_path) == sha(dossier_path)
            and [row["stage"] for row in invocations[2:4]] == ["initial-compatible","compatible-adjustment"]}


def _resume_archive(directory, receipt_sha256, manifest_sha256):
    """Authenticate a settled, exclusively owned archive before following links."""
    directory = Path(directory).absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError("Resume source path must not contain symlinks")
    directory = directory.resolve()
    for value in (receipt_sha256, manifest_sha256):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Resume requires explicit receipt and proof-manifest SHA256 pins")
    if (sha(directory / RECEIPT) != receipt_sha256
            or sha(directory / "proof-sha256.json") != manifest_sha256):
        raise ValueError("Resume receipt/proof-manifest differs from its explicit pin")
    receipt, manifest = read_json(directory / RECEIPT), read_json(directory / "proof-sha256.json")
    if not isinstance(receipt, dict) or not isinstance(manifest, dict):
        raise ValueError("Resume receipt and proof manifest must be objects")
    files = manifest.get("files")
    if (manifest.get("hash_algorithm") != "sha256" or not isinstance(files, dict) or not files
            or set(manifest.get("excluded_paths", [])) != {RECEIPT, "proof-sha256.json"}
            or receipt.get("proof_manifest") != {"path":"proof-sha256.json", "sha256":manifest_sha256}):
        raise ValueError("Unsupported resume proof manifest")
    inventory = list(directory.rglob("*"))
    if (any(path.is_symlink() for path in inventory)
            or {str(path.relative_to(directory)) for path in inventory if path.is_file()}
                != set(files) | {RECEIPT, "proof-sha256.json"}):
        raise ValueError("Resume source contains unmanifested artifacts or symlinks")
    for name, expected in files.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or not (directory / name).is_file() or sha(directory / name) != expected):
            raise ValueError("Resume source artifact is missing, unsafe or changed")
    return directory, receipt, manifest, {**files, RECEIPT:receipt_sha256, "proof-sha256.json":manifest_sha256}


def _inspect_once_resumed_source(directory, *, receipt_sha256, manifest_sha256):
    """Accept only the observed 7+2-call refusal, never an arbitrary resume chain.

    The saved resume-state is the entry to the failed research (round 2), not
    a final checkpoint. The two completed calls and blocked event establish
    failed round 3; the next unchanged runtime research therefore uses round 4.
    """
    directory, receipt, manifest, hashes = _resume_archive(directory, receipt_sha256, manifest_sha256)
    if (receipt.get("kind") != "documentary_research_refresh_resumption_acceptance"
            or receipt.get("status") != "incomplete" or receipt.get("ok") is not False
            or not str(receipt.get("error", "")).startswith("Actual resumed targeted research blocked: ")
            or receipt.get("new_gpu_requests_attempted") != 0 or receipt.get("new_gpu_requests") != 0
            or receipt.get("resume_source_unchanged") is not True
            or receipt.get("bootstrap_files_unchanged") is not True or receipt.get("checks")):
        raise ValueError("Resume supports only the completed first-resumption research refusal")
    intent = read_json(directory / "run-intent.json")
    chain = receipt.get("resumption", {})
    evidence_repository = next((p for p in (directory, *directory.parents) if (p / ".git").exists()), REPO).resolve()
    if (not isinstance(intent, dict) or not isinstance(chain, dict)
            or intent.get("kind") != "before_documentary_resumption"
            or intent.get("run_id") != receipt.get("run_id") or intent.get("resumption") != chain
            or not intent.get("source_provenance") or intent["source_provenance"] != receipt.get("source_provenance")
            or intent.get("new_gpu_requests") != 0
            or chain.get("evidence_repository") != str(evidence_repository)
            or chain.get("inherited_completed_model_calls") != 7
            or chain.get("controls_directory", "acceptance-controls-resume") != "acceptance-controls-resume"
            or chain.get("research_round_derivation", {
                "initial_anchor_round":2, "completed_failed_round":2, "next_fresh_round":3}) != {
                "initial_anchor_round":2, "completed_failed_round":2, "next_fresh_round":3}
            or chain.get("no_initial_proposal_or_diagnostic_replay") is not True):
        raise ValueError("First resumption intent, provenance or initial anchor differs")
    relative = chain.get("source_directory")
    if (not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts):
        raise ValueError("Initial refusal anchor must be relative to the evidence checkout")
    anchor = evidence_repository / relative
    if (anchor == directory or anchor.is_relative_to(directory) or directory.is_relative_to(anchor)
            or any(p.is_symlink() for p in (anchor, *anchor.parents))):
        raise ValueError("Initial refusal and resumed source must be separate immutable archives")
    def pinned(name, role):
        ref = chain.get(role)
        expected_path = str(Path(relative) / name)
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256", "path_base"}
                or ref.get("path") != expected_path or ref.get("path_base") != "repository"):
            raise ValueError("Initial refusal reference has an unsafe or ambiguous role")
        return ref["sha256"]
    original = _inspect_initial_resume_source(anchor,
        receipt_sha256=pinned(RECEIPT, "source_receipt"),
        manifest_sha256=pinned("proof-sha256.json", "source_proof_manifest"))
    if (receipt.get("historical_bootstrap") != original["receipt"]["historical_bootstrap"]
            or intent.get("internal_campaign_run_id") != original["origin"]["config"].run_id
            or receipt.get("inherited_model_calls") != original["calls"]):
        raise ValueError("First resumption changed its original bootstrap or seven-call prefix")
    for old, copied in ((RECEIPT,"previous-receipt.json"), ("proof-sha256.json","previous-proof.json"),
            ("events.json","previous-events.json"), ("campaign/acceptance-evidence/initial-dossier.json","previous-dossier.json")):
        if hashes.get("source-evidence/"+copied) != original["source_hashes"][old]:
            raise ValueError("First resumption copied anchor evidence differs")
    # A refusal may add documentary research evidence and exactly two new CLI
    # traces; every inherited campaign byte except the live ledger stays exact.
    for name, expected in original["source_hashes"].items():
        if name.startswith("campaign/") and name != "campaign/experiments.sqlite" and hashes.get(name) != expected:
            raise ValueError("First resumption changed inherited campaign evidence: " + name)
    events = read_json(directory / "events.json")
    if (not isinstance(events, list) or len(events) != 19 or events[:16] != original["events"]
            or [row.get("kind") for row in events[16:]] != ["model", "model", "research_blocked"]
            or receipt["error"] != "Actual resumed targeted research blocked: " + events[-1]["payload"].get("error", "")):
        raise ValueError("First resumption does not preserve the exact16 prefix and two-call blocked suffix")
    db_path = directory / "campaign/experiments.sqlite"
    if any(db_path.with_name(db_path.name+suffix).exists() for suffix in ("-wal", "-shm")):
        raise ValueError("Resume requires a settled archived SQLite database")
    with closing(sqlite3.connect(db_path.as_uri()+"?mode=ro&immutable=1", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("ATTACH DATABASE ? AS initial", (
            (original["directory"] / "campaign/experiments.sqlite").as_uri()+"?mode=ro&immutable=1",))
        # Compare policy snapshots opaquely inside SQLite: no RNG state is
        # decoded or exported, and no ask/selection/snapshot update is allowed.
        immutable_tables_equal = all(not db.execute(
            "SELECT EXISTS(SELECT * FROM main."+table+" EXCEPT SELECT * FROM initial."+table+
            ") OR EXISTS(SELECT * FROM initial."+table+" EXCEPT SELECT * FROM main."+table+")").fetchone()[0]
            for table in ("campaigns", "trials", "candidates", "candidate_status", "attempts"))
        db_events = [dict(row, payload=json.loads(row["payload"])) for row in db.execute("SELECT * FROM events ORDER BY event_id")]
        trials = [dict(row) for row in db.execute("SELECT * FROM trials")]
        forbidden = sum(db.execute("SELECT count(*) FROM "+table).fetchone()[0]
            for table in ("evaluations", "reward_claims", "integration_events"))
        uncertain = any(json.loads(row[0]).get("model_calls", 0)
            for row in db.execute("SELECT amount FROM reservations WHERE actual IS NULL"))
        candidate_count = db.execute("SELECT count(*) FROM candidates").fetchone()[0]
        ledger = ExperimentStore._budget(None, db, original["origin"]["config"].run_id)
    trial_id = original["state"]["trial_id"]
    if (db_events != events or len(trials) != 1 or trials[0]["trial_id"] != trial_id
            or trials[0]["status"] != "pending" or trials[0]["result"] is not None
            or forbidden or uncertain or candidate_count != 3 or not immutable_tables_equal
            or ledger != receipt.get("ledger_usage")
            or any(row["payload"].get("trial_id", trial_id) != trial_id for row in events)):
        raise ValueError("First resumption SQL/events/pending-trial state is incompatible or uncertain")
    calls = receipt.get("model_calls")
    invocations = read_json(directory / "campaign/acceptance-controls-resume/invocations.json")
    if (not isinstance(calls, list) or len(calls) != 2 or not isinstance(invocations, list) or len(invocations) != 2
            or any(not isinstance(row, dict) for row in calls+invocations)
            or [row.get("purpose") for row in invocations] != ["research", "proposal"]
            or any(row.get("stage") != "research_service" or row.get("kernel_proposal") is not False
                or row.get("status") != "completed" for row in invocations)):
        raise ValueError("First resumption requires two completed research/synthesis calls only")
    all_calls = original["calls"]+calls
    ids = [row.get("call_id") for row in all_calls]
    if (any(not isinstance(value,str) or len(value)!=32 or any(c not in "0123456789abcdef" for c in value) for value in ids)
            or len(set(ids)) != 9 or [row.get("call_id") for row in invocations] != ids[7:]
            or [row["payload"].get("call_id") for row in events if row["kind"] == "model"] != ids
            or {path.stem for path in (directory / "campaign/model").glob("*.json")} != set(ids)):
        raise ValueError("First resumption CLI identities are duplicated, missing or ambiguous")
    traces = list(original["traces"])
    for call, invocation in zip(calls, invocations):
        trace = read_json(directory / "campaign/model" / (call["call_id"]+".json"))
        if (trace.get("call_id") != call["call_id"] or trace.get("status") != "completed" or trace.get("returncode") != 0
                or trace.get("purpose") != invocation["purpose"] or trace.get("usage") != call.get("usage")
                or trace.get("model") != "gpt-6-luna" or trace.get("reasoning_effort") != "max"
                or digest(trace.get("payload")) != invocation.get("returned_payload_sha256")):
            raise ValueError("First resumption completed CLI payload/usage differs")
        traces.append(trace)
    for event, call, invocation in zip(events[16:18], calls, invocations):
        if (event["payload"].get("usage") != call["usage"]
                or event["payload"].get("purpose") != invocation["purpose"]
                or event["payload"].get("elapsed_seconds") != call.get("elapsed_seconds")):
            raise ValueError("First resumption event/call usage identity differs")
    added = set(hashes)-set(original["source_hashes"])
    allowed = {"campaign/acceptance-controls-resume/invocations.json", "campaign/hardware-evidence-context.json",
               "campaign/runtime-evidence-context.json",
               *("campaign/model/"+call["call_id"]+".json" for call in calls)}
    if any(name.startswith("campaign/") and name not in allowed
            and not (name.startswith("campaign/research/") and Path(name).suffix in (".json", ".txt")) for name in added):
        raise ValueError("First resumption added a non-research campaign artifact")
    entry = read_json(directory / "resume-state.json")
    if (entry.get("phase") != "targeted_research_blocked_reconstructed" or entry.get("state") != original["state"]
            or entry.get("source_receipt") != chain["source_receipt"]
            or entry.get("source_events_sha256") != original["source_hashes"]["events.json"]
            or entry.get("derivation") != chain.get("state_derivation")):
        raise ValueError("First resumption entry state is not the exact reconstructed round2 anchor")
    if read_json(directory / "campaign/hardware-evidence-context.json") != resume_hardware_context(original["origin"]):
        raise ValueError("First resumption historical hardware annex differs from its pinned allocation")
    if "campaign/runtime-evidence-context.json" in hashes:
        if (read_json(directory / "campaign/runtime-evidence-context.json") != resume_runtime_context(original["origin"])
                or receipt.get("runtime_evidence_context") != {
                    "path":"campaign/runtime-evidence-context.json", "sha256":hashes["campaign/runtime-evidence-context.json"]}):
            raise ValueError("First resumption baseline runtime annex differs from its independently pinned archive")
    inherited = original["receipt"]["ledger_usage"].get("spent", {})
    current = receipt.get("ledger_usage", {}).get("spent", {})
    delta = {key:current.get(key,0)-inherited.get(key,0) for key in set(current)|set(inherited)}
    tokens = sum(call["usage"].get("input_tokens",0)+call["usage"].get("output_tokens",0) for call in calls)
    if (current.get("model_calls") != 9 or delta.get("model_calls") != 2 or delta.get("model_tokens") != tokens
            or delta != receipt.get("new_usage_delta")):
        raise ValueError("First resumption combined ledger does not preserve exactly seven plus two calls")
    state = dict(original["state"], research_round=3, error=events[-1]["payload"]["error"])
    return {**original, "directory":directory, "receipt":receipt, "manifest":manifest, "source_hashes":hashes,
        "events":events, "calls":all_calls, "traces":traces, "state":state,
        "ancestor_archives":[{"directory":original["directory"], "source_hashes":original["source_hashes"]}],
        "research_round_derivation":{"initial_anchor_round":2, "completed_failed_round":3, "next_fresh_round":4},
        "controls_directory":"acceptance-controls-resume-2"}


def inspect_resume_source(directory, *, receipt_sha256, manifest_sha256):
    """Only the observed original7 or its first9-call refusal is resumable."""
    try:
        directory, receipt, _, _ = _resume_archive(directory, receipt_sha256, manifest_sha256)
        if receipt.get("kind") == "documentary_research_refresh_acceptance":
            return _inspect_initial_resume_source(directory, receipt_sha256=receipt_sha256, manifest_sha256=manifest_sha256)
        return _inspect_once_resumed_source(directory, receipt_sha256=receipt_sha256, manifest_sha256=manifest_sha256)
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as error:
        raise ValueError("Malformed or unavailable resume evidence") from error


def validate_resume_runtime(runtime, prepared):
    """Replay actual admission/diagnostic guards before any resumed model call."""
    state, ids = prepared["state"], prepared["candidate_ids"]
    if runtime.store.events(runtime.config.run_id) != prepared["events"]:
        raise ValueError("Reconstructed runtime changed the original event prefix")
    trial = runtime.trial(state)
    if (trial["action"] != prepared["intent"]["family"] or trial["parent"]["candidate_id"] != "baseline"
            or trial["trial_id"] != state["trial_id"]):
        raise ValueError("Resume changed the selected original trial/parent/family")
    initial = ResearchDossier.from_dict(prepared["initial_dossier"])
    ready = initial.research_ready(runtime.contract_hash, trial["action"], runtime.environment_hash)
    proposals = []
    for repair, candidate_id in enumerate(ids):
        data = read_json(runtime.trial_dir(state) / f"proposal-{repair}-1.json")
        parsed = Proposal.from_dict(data, parent_id="baseline", source_parent_id="baseline" if repair == 0 else ids[0],
            contract_hash=runtime.contract_hash, family=trial["action"])
        initial.validate_proposal(parsed)
        raw = prepared["traces"][2+repair]["payload"]
        provider = Proposal.from_payload(raw, parent_id="baseline", source_parent_id=parsed.source_parent_id,
            contract_hash=runtime.contract_hash, family=trial["action"])
        if any(asdict(parsed)[key] != asdict(provider)[key] for key in asdict(provider) if key not in ("usage", "call_id")):
            raise ValueError("Persisted proposal differs from its exact provider payload")
        candidate = runtime.candidate(candidate_id)
        stored = runtime.store.get_candidate_source(candidate_id, runtime.config.run_id)
        expected_id = digest({"source":data["source"], "launch":data["launch_rule"], "parameters":data["parameters"],
                              "trial_id":state["trial_id"], "repair":repair})[:24]
        if (candidate_id != expected_id or candidate.source.read_text() != data["source"]
                or candidate.kernel != data["kernel"] or digest(asdict(candidate.launch_rule)) != digest(data["launch_rule"])
                or candidate.artifact_hash != stored.get("artifact_hash")
                or candidate.parameters != data["parameters"] or data["call_id"] != prepared["calls"][2+repair]["call_id"]):
            raise ValueError("Original candidate source/parameters/identity differ")
        proposals.append(data)
    blocked = read_json(runtime.directory / "acceptance-evidence/blocked-refusal.json")
    current_files = {path.name:sha(path) for path in (runtime.directory / "candidates").glob("*.cu")}
    if (current_files != blocked["candidates_before"] or (runtime.trial_dir(state) / "proposal-2-1.json").exists()
            or list(runtime.trial_dir(state).glob("evaluation-*.json")) or (runtime.trial_dir(state) / "result.json").exists()):
        raise ValueError("The blocked request was followed by a candidate, measurement or integration")
    rejected = Proposal.from_payload(prepared["traces"][4]["payload"], parent_id="baseline", source_parent_id=ids[1],
        contract_hash=runtime.contract_hash, family=trial["action"])
    try:
        initial.validate_proposal(rejected)
    except NeedsResearch as error:
        semantic_refusal = "uncovered technique" in str(error)
        if not semantic_refusal:
            raise ValueError("The original payload fails a different research guard") from error
    else:
        raise ValueError("The original uncovered payload is admitted; refuse resumption")
    diagnostic = verify_diagnostic(read_json(runtime.directory / "diagnostics" / (state["diagnostic_id"] + ".json")),
        runtime.directory, contract_hash=runtime.contract_hash, environment_fingerprint=runtime.environment_hash)
    markers = list(runtime.trial_dir(state).glob("diagnostic-*.json"))
    if (len(markers) != 1 or read_json(markers[0]) != diagnostic
            or diagnostic.get("trial_id") != state["trial_id"] or diagnostic.get("candidate_id") != ids[1]
            or not diagnostic.get("requires_targeted_research")
            or prepared["technique"] not in diagnostic["proposal_hypothesis"].get("techniques", [])):
        raise ValueError("Original immutable targeted diagnostic linkage differs")
    return {"initial_ready":ready, "initial":initial, "proposals":proposals, "diagnostic":diagnostic,
        "semantic_refusal_verified":semantic_refusal and prepared["technique"] in rejected.techniques
            and prepared["technique"] not in initial.covered_techniques
            and current_files == blocked["candidates_before"]}


def resume_hardware_context(origin):
    from optimizer_agent.hardware_research_context import archived_hardware_research_context
    return archived_hardware_research_context(origin["directory"] / "worker", origin["properties"],
        expected_job_id=origin["job_id"], expected_session_id=origin["session_id"],
        expected_receipt_hashes={name:origin["hashes"]["worker/"+name] for name in ("hardware.json", "slurm-session.json")})


def resume_runtime_context(origin):
    """Bind ordinary, already executed baseline work; never infer VBIOS/features."""
    from optimizer_agent.hardware_research_context import baseline_runtime_research_context
    rule = LaunchRule(kind="square_2d", block=(16,16,1)) if origin["config"].workload == "matmul" else LaunchRule(block=(128,1,1))
    baseline = Candidate("baseline", origin["directory"] / "generation/evaluator_data/baseline.cu",
                         origin["config"].workload, rule)
    requests = origin["directory"] / "request-receipts"
    if not requests.is_dir():
        raise ValueError("Resumed research requires the exact public baseline native request receipts")
    return baseline_runtime_research_context(origin["directory"] / "worker", device_identity(origin["properties"]),
        origin["directory"] / "generation/baseline-measurement.json",
        expected_contract_hash=origin["contract"]["contract_hash"], baseline_candidate=baseline,
        expected_case_ids=origin["authentication"]["baseline_case_ids"],
        benchmark_series=origin["config"].benchmark_series, launches_per_series=origin["config"].launches_per_series,
        expected_job_id=origin["job_id"], expected_session_id=origin["session_id"],
        expected_receipt_hashes={name:origin["hashes"]["worker/"+name] for name in ("hardware.json", "slurm-session.json")},
        expected_baseline_sha256=origin["hashes"]["generation/baseline-measurement.json"],
        request_receipt_directory=requests)


def prepare(directory, run_id, bootstrap=None, *, family="tiling", result_sha256=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    origin = inspect_bootstrap(bootstrap, result_sha256) if bootstrap else None
    plan = {"schema_version": 1, "kind": "documentary_research_refresh_acceptance", "run_id": run_id,
        "submitted": False, "model_invoked": False, "new_gpu_requests": 0, "family": family,
        "bootstrap_required_for_run": True, "explicit_result_sha256_pin_required_for_run": True,
        "bootstrap_directory": str(origin["directory"]) if origin else None,
        "bootstrap_file_hashes": origin["hashes"] if origin else None,
        "bootstrap_authentication": origin["authentication"] if origin else None,
        "bootstrap_completed_job": origin["job_id"] if origin else None,
        "hardware_identity_scope": "Historical archived research target; not a currently observed GPU",
        "protocol": ["new-family mandatory research", "first covered proposal", "compatible adjustment reuses exact dossier",
                     "prompt-declared uncovered technique rejected before new candidate", "actual immutable diagnostic",
                     "targeted real research and renewed dossier", "new covered proposal admitted for future evaluation"],
        "scope": "Documentary acceptance only, explicitly driven controls, no evaluation/integration/finalization or section16 quality claim",
        "historical_baseline_scope": "Exact archived baseline is reused only to construct an unchanged contract/runtime. Its scores/cost are not new measurements or GPU usage."}
    write_json(directory / "plan.json", plan)
    return origin, plan


def run_acceptance(directory, run_id, bootstrap, *, result_sha256, family="tiling", uncovered_technique=None, executable=None):
    directory = Path(directory).resolve()
    if not result_sha256:
        raise ValueError("--run requires --bootstrap-result-sha256 to pin the completed source receipt")
    if (directory / "run-intent.json").exists() or (directory / "campaign").exists() or (directory / RECEIPT).exists():
        raise ValueError("Existing intent/evidence requires manual reconciliation; never repeat an uncertain model call")
    origin, plan = prepare(directory, run_id, bootstrap, family=family, result_sha256=result_sha256)
    config = replace(origin["config"], actions=(family,), max_trials=1, max_repairs_per_trial=2,
                     profile=False, autotuning_enabled=False,
                     model="gpt-6-luna", reasoning_effort="max")
    campaign_dir = directory / "campaign"
    intent = {"run_id": run_id, "bootstrap": str(origin["directory"]), "bootstrap_hashes": origin["hashes"],
              "internal_campaign_run_id": config.run_id, "family": family, "new_gpu_requests": 0,
              "source_provenance": controller_provenance(REPO, __file__, origin["directory"] / "generation/config.json")}
    with (directory / "run-intent.json").open("x") as stream:
        json.dump(intent, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    receipt = {"schema_version": 1, "kind": plan["kind"], "run_id": run_id, "ok": False, "status": "incomplete",
               "scope": plan["scope"], "source_provenance": intent["source_provenance"],
               "historical_bootstrap": {"job_id": origin["job_id"], "session_id": origin["session_id"],
                   "terminal": origin["terminal"], "file_hashes": origin["hashes"],
                   "authentication":origin["authentication"],
                   "baseline_scope": plan["historical_baseline_scope"]},
               "hardware_currently_observed": False, "new_gpu_requests": 0}
    runner = ArchivedPropertiesOnlyRunner(origin["properties"])
    runtime = controls = None
    try:
        for source_name, target_name in (("generation/baseline-measurement.json", "baseline-measurement.json"),
                ("generation/environment.json", "environment.json"),
                ("generation/evaluator_data/baseline.cu", "evaluator_data/baseline.cu")):
            target = campaign_dir / target_name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin["directory"] / source_name, target)
        codex = executable or default_codex_executable()
        backend = CodexBackend(CodexConfig(executable=codex, model="gpt-6-luna", reasoning_effort="max",
            timeout_seconds=config.model_timeout_seconds, trace_dir=campaign_dir / "model"))
        controls = PromptControls(backend, campaign_dir / "acceptance-controls")
        runtime = DocumentaryRuntime(config, runner, campaign_dir, model_backend=controls, origin=origin)
        if (runtime.contract_hash != origin["contract"]["contract_hash"]
                or sha(campaign_dir / "baseline-measurement.json") != origin["hashes"]["generation/baseline-measurement.json"]):
            raise ValueError("Runtime contract or exact archived baseline changed before any acceptance model call")
        receipt.update(exercise(runtime, controls, uncovered_technique=uncovered_technique))
        receipt["ok"] = all(receipt["checks"].values()) and not runner.gpu_attempts
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        receipt["new_gpu_requests_attempted"] = len(runner.gpu_attempts)
        receipt["bootstrap_files_unchanged"] = all((origin["directory"] / name).is_file()
            and sha(origin["directory"] / name) == value for name, value in origin["hashes"].items())
        receipt["ok"] &= receipt["bootstrap_files_unchanged"] and not runner.gpu_attempts
        if controls:
            receipt["model_calls"] = [{"call_id": c.call_id, "usage": c.usage, "elapsed_seconds": c.elapsed_seconds} for c in controls.calls]
        if runtime:
            write_json(directory / "events.json", runtime.store.events(config.run_id))
            receipt["ledger_usage"] = runtime.store.budget_state(config.run_id)
            receipt["ledger_scope"] = "Historical baseline scores only initialize the unranked root; no old GPU cost is charged. Model usage is actual new transport usage."
        receipt["status"] = "documentary_research_refresh_acceptance_passed" if receipt["ok"] else "incomplete"
        excluded = [RECEIPT,"proof-sha256.json"]
        write_json(directory / "proof-sha256.json", {"schema_version":1,"hash_algorithm":"sha256",
            "excluded_paths":excluded,"scope":"Owned documentary artifacts at controller completion; no self-reference.",
            "files":{str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob("*"))
                     if p.is_file() and str(p.relative_to(directory)) not in excluded}})
        receipt["proof_manifest"] = reference(directory / "proof-sha256.json", directory)
        write_json(directory / RECEIPT, receipt)
    return receipt


def prepare_resumption(directory, run_id, source, *, receipt_sha256, manifest_sha256):
    supplied_source = Path(source).absolute()
    if any(path.is_symlink() for path in (supplied_source, *supplied_source.parents)):
        raise ValueError("Resume source path must not contain symlinks")
    directory, source = Path(directory).resolve(), Path(source).resolve()
    if directory.is_relative_to(source) or source.is_relative_to(directory):
        raise ValueError("Resume output and its immutable source must be separate directories")
    prepared = inspect_resume_source(source, receipt_sha256=receipt_sha256, manifest_sha256=manifest_sha256)
    immutable_roots = [prepared["origin"]["directory"],
                       *(ancestor["directory"] for ancestor in prepared.get("ancestor_archives", []))]
    if any(directory.is_relative_to(root) or root.is_relative_to(directory) for root in immutable_roots):
        raise ValueError("Resume output must be disjoint from every immutable ancestor and bootstrap")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "plan.json", {"schema_version":1, "kind":"documentary_research_refresh_resumption_acceptance",
        "run_id":run_id, "submitted":False, "model_invoked":False, "new_gpu_requests":0,
        "source_directory":str(source), "source_receipt_sha256":receipt_sha256,
        "source_proof_manifest_sha256":manifest_sha256,
        "state_derivation":"Only the observed targeted-research refusal; reconstructed from pinned SQL/events/CLI/diagnostic artifacts, not an old serialized checkpoint",
        "research_round_derivation":prepared.get("research_round_derivation", {
            "initial_anchor_round":2, "completed_failed_round":2, "next_fresh_round":3}),
        "inherited_completed_model_calls":len(prepared["calls"]),
        "resume_phases":["fresh normal targeted research with authenticated historical hardware annex", "new covered proposal"],
        "prohibited_replays":["initialize", "select", "initial research", "first three proposals", "diagnose"],
        "scope":"Documentary acceptance; no new GPU request, measurement, integration, finalization or section16 claim"})
    return prepared


def run_resumption_acceptance(directory, run_id, source, *, receipt_sha256, manifest_sha256, executable=None):
    directory = Path(directory).resolve()
    if any((directory / name).exists() for name in ("run-intent.json", "campaign", RECEIPT)):
        raise ValueError("Existing resume intent/evidence requires reconciliation; never repeat uncertain calls")
    prepared = prepare_resumption(directory, run_id, source, receipt_sha256=receipt_sha256, manifest_sha256=manifest_sha256)
    source, origin = prepared["directory"], prepared["origin"]
    config = CampaignConfig.load(source / "campaign/config.json")
    # Frozen controllers can run from another checkout while the immutable
    # evidence stays in the primary repository. Keep these identities distinct.
    evidence_repository = next((parent for parent in (source, *source.parents) if (parent / ".git").exists()), REPO)
    def source_ref(name, value):
        path = source / name
        return {"path":str(path.relative_to(evidence_repository)) if path.is_relative_to(evidence_repository) else str(path),
                "sha256":value, "path_base":"repository" if path.is_relative_to(evidence_repository) else "absolute"}
    resumption = {"source_directory":str(source.relative_to(evidence_repository)) if source.is_relative_to(evidence_repository) else str(source),
        "evidence_repository":str(evidence_repository),
        "source_receipt":source_ref(RECEIPT, receipt_sha256),
        "source_proof_manifest":source_ref("proof-sha256.json", manifest_sha256),
        "inherited_completed_model_calls":len(prepared["calls"]),
        "state_derivation":"Reconstructed targeted-research refusal from pinned16event anchor, settled SQLite, completed CLI prefix, two accepted proposals, exact semantic refusal and immutable diagnostic; any later failed round is derived from its authenticated two-call blocked suffix, not an old serialized checkpoint",
        "research_round_derivation":prepared.get("research_round_derivation", {
            "initial_anchor_round":2, "completed_failed_round":2, "next_fresh_round":3}),
        "controls_directory":prepared.get("controls_directory", "acceptance-controls-resume"),
        "no_initial_proposal_or_diagnostic_replay":True}
    intent = {"schema_version":1, "kind":"before_documentary_resumption", "run_id":run_id,
        "internal_campaign_run_id":config.run_id, "resumption":resumption, "new_gpu_requests":0,
        "source_provenance":controller_provenance(REPO, __file__, source / "campaign/config.json")}
    with (directory / "run-intent.json").open("x") as stream:
        json.dump(intent, stream, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    receipt = {"schema_version":1, "kind":"documentary_research_refresh_resumption_acceptance", "run_id":run_id,
        "ok":False, "status":"incomplete", "scope":"Isolated documentary resumption, historical hardware target, no GPU measurement or optimizer-quality claim",
        "source_provenance":intent["source_provenance"], "resumption":resumption,
        "historical_bootstrap":prepared["receipt"]["historical_bootstrap"],
        "inherited_model_calls":prepared["calls"], "hardware_currently_observed":False, "new_gpu_requests":0}
    runner, runtime, controls = ArchivedPropertiesOnlyRunner(origin["properties"]), None, None
    try:
        shutil.copytree(source / "campaign", directory / "campaign")
        for old_name, new_name in ((RECEIPT,"previous-receipt.json"), ("proof-sha256.json","previous-proof.json"),
                ("events.json","previous-events.json"), ("campaign/acceptance-evidence/initial-dossier.json","previous-dossier.json")):
            target = directory / "source-evidence" / new_name
            target.parent.mkdir(exist_ok=True); shutil.copyfile(source / old_name, target)
        state = dict(prepared["state"])
        write_json(directory / "resume-state.json", {"phase":"targeted_research_blocked_reconstructed", "state":state,
            "derivation":resumption["state_derivation"], "source_receipt":resumption["source_receipt"],
            "source_events_sha256":prepared["source_hashes"]["events.json"]})
        campaign_dir = directory / "campaign"
        backend = CodexBackend(CodexConfig(executable=executable or default_codex_executable(), model="gpt-6-luna",
            reasoning_effort="max", timeout_seconds=config.model_timeout_seconds, trace_dir=campaign_dir / "model"))
        control_directory = campaign_dir / resumption["controls_directory"]
        if control_directory.exists():
            raise ValueError("New resumption control directory already exists; preserve inherited invocations")
        controls = PromptControls(backend, control_directory)
        runtime = DocumentaryRuntime(config, runner, campaign_dir, model_backend=controls, origin=origin)
        validated = validate_resume_runtime(runtime, prepared)
        before_environment = (digest(runtime.environment), runtime.contract_hash)
        context = resume_hardware_context(origin)
        runtime.workload_description["hardware_evidence_context"] = context
        write_json(campaign_dir / "hardware-evidence-context.json", context)
        runtime_context = resume_runtime_context(origin)
        runtime.workload_description["runtime_evidence_context"] = runtime_context
        write_json(campaign_dir / "runtime-evidence-context.json", runtime_context)
        receipt["runtime_evidence_context"] = reference(campaign_dir / "runtime-evidence-context.json", directory)
        if before_environment != (digest(runtime.environment), runtime.contract_hash):
            raise ValueError("Historical hardware annex changed the immutable contract/environment")
        state.update(runtime.research(state))
        if not state.get("research_ready"):
            raise RuntimeError("Actual resumed targeted research blocked: " + str(state.get("error")))
        dossier_path = runtime.trial_dir(state) / "dossier.json"
        renewed = ResearchDossier.from_dict(read_json(dossier_path))
        diagnostic, technique = validated["diagnostic"], prepared["technique"]
        if (renewed.research_id == validated["initial"].research_id
                or diagnostic["diagnostic_id"] not in renewed.diagnostic_ids or technique not in renewed.covered_techniques):
            raise RuntimeError("Actual resumed dossier does not cover the exact blocked technique/diagnostic")
        final = generate_control(runtime, controls, state, "newly-covered-proposal",
            "Now propose the transformation requiring " + repr(technique) + " only if this renewed dossier genuinely covers it. "
            "Declare the technique and cite actual renewed source IDs. Preserve the normal parent, ABI and contract guards.")
        if technique not in final["techniques"]:
            raise RuntimeError("Resumed final proposal omitted the newly covered technique")
        events = runtime.store.events(config.run_id); suffix = events[len(prepared["events"]):]
        first, second = validated["proposals"]
        checks = {"new_family_real_applicable_dossier":validated["initial_ready"] is True,
            "compatible_adjustment_exact_dossier_reuse_without_research":prepared["dossier_reuse_verified"],
            "compatible_adjustment_uses_previous_source_parent":second["source_parent_id"] == prepared["candidate_ids"][0],
            "compatible_proposal_changes_source_launch_or_parameters":any(first[key] != second[key] for key in ("source","launch_rule","parameters")),
            "uncovered_technique_blocked_before_candidate":validated["semantic_refusal_verified"],
            "diagnostic_contains_exact_blocked_technique":technique in diagnostic["proposal_hypothesis"]["techniques"],
            "targeted_new_dossier_covers_technique":technique in renewed.covered_techniques,
            "targeted_dossier_links_immutable_diagnostic":diagnostic["diagnostic_id"] in renewed.diagnostic_ids,
            "new_covered_proposal_admitted":bool(state.get("candidate_id") and technique in final["techniques"]),
            "only_three_admitted_generation_events":sum(event["kind"] == "generation" for event in events) == 3,
            "no_trial_integration_or_measurement":not list(runtime.trial_dir(state).glob("evaluation-*.json")) and not (runtime.trial_dir(state) / "result.json").exists(),
            "no_final_holdout_exposed":not (runtime.directory / "holdout-exposed.json").exists(),
            "resume_original_prefix_unchanged":events[:len(prepared["events"])] == prepared["events"],
            "resume_no_old_model_call_repeated":not ({call.call_id for call in controls.calls} & {call["call_id"] for call in prepared["calls"]}),
            "resume_no_new_selection_or_diagnostic":not any(event["kind"] in ("checkpoint","selection","diagnostic") for event in suffix),
            "resume_hardware_context_bound":context.get("job_id") == origin["job_id"] and context.get("session_id") == origin["session_id"] and context.get("hardware_currently_observed") is False,
            "resume_source_unchanged":False}
        receipt.update(checks=checks, initial_research_id=validated["initial"].research_id,
            initial_dossier_sha256=prepared["initial_dossier_sha256"], renewed_research_id=renewed.research_id,
            renewed_dossier_sha256=sha(dossier_path), diagnostic_id=diagnostic["diagnostic_id"], uncovered_technique=technique,
            initial_call_id=first["call_id"], compatible_call_id=second["call_id"], blocked_call_id=prepared["calls"][4]["call_id"],
            final_call_id=final["call_id"], admitted_candidates=prepared["candidate_ids"]+[state["candidate_id"]],
            checks_scope="First compatible/reuse/refusal/diagnostic checks revalidated from immutable inherited evidence; fresh research/final checks executed by the unchanged runtime now")
        receipt["ok"] = all(value for key,value in checks.items() if key != "resume_source_unchanged")
    except Exception as error:
        receipt.update(error=str(error), traceback=traceback.format_exc())
    finally:
        def archive_unchanged(archive, expected_hashes):
            inventory = list(archive.rglob("*"))
            return (not any(path.is_symlink() for path in inventory)
                and {str(path.relative_to(archive)) for path in inventory if path.is_file()} == set(expected_hashes)
                and all((archive / name).is_file() and sha(archive / name) == expected for name,expected in expected_hashes.items()))
        unchanged = archive_unchanged(source, prepared["source_hashes"]) and all(
            archive_unchanged(ancestor["directory"], ancestor["source_hashes"])
            for ancestor in prepared.get("ancestor_archives", []))
        bootstrap_unchanged = all((origin["directory"] / name).is_file() and sha(origin["directory"] / name) == expected
                                  for name,expected in origin["hashes"].items())
        receipt.update(resume_source_unchanged=unchanged, bootstrap_files_unchanged=bootstrap_unchanged,
                       new_gpu_requests_attempted=len(runner.gpu_attempts))
        if "checks" in receipt:receipt["checks"]["resume_source_unchanged"] = unchanged
        receipt["ok"] &= unchanged and bootstrap_unchanged and not runner.gpu_attempts
        receipt["model_calls"] = [{"call_id":call.call_id,"usage":call.usage,"elapsed_seconds":call.elapsed_seconds} for call in controls.calls] if controls else []
        if runtime:
            write_json(directory / "events.json", runtime.store.events(config.run_id))
            receipt["ledger_usage"] = runtime.store.budget_state(config.run_id)
            inherited = prepared["receipt"]["ledger_usage"].get("spent", {})
            current = receipt["ledger_usage"].get("spent", {})
            receipt["new_usage_delta"] = {key:current.get(key,0)-inherited.get(key,0) for key in set(current)|set(inherited)}
            receipt["ledger_scope"] = "Combined cloned ledger retains all inherited completed CLI calls; new_usage_delta and model_calls describe only actual resumed consumption. No old GPU cost is newly charged."
        receipt["status"] = "documentary_research_refresh_resumption_acceptance_passed" if receipt["ok"] else "incomplete"
        excluded = [RECEIPT,"proof-sha256.json"]
        write_json(directory / "proof-sha256.json", {"schema_version":1,"hash_algorithm":"sha256","excluded_paths":excluded,
            "scope":"Owned isolated resumption artifacts, including unchanged inherited copies; no self-reference",
            "files":{str(path.relative_to(directory)):sha(path) for path in sorted(directory.rglob("*")) if path.is_file() and str(path.relative_to(directory)) not in excluded}})
        receipt["proof_manifest"] = reference(directory / "proof-sha256.json", directory)
        write_json(directory / RECEIPT, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Default: local plan only, no model/network/GPU")
    mode.add_argument("--run", action="store_true", help="Explicitly run actual local Codex/web documentary controls, no GPU")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default="research-refresh-" + uuid.uuid4().hex[:12])
    parser.add_argument("--bootstrap-campaign", type=Path)
    parser.add_argument("--bootstrap-result-sha256")
    parser.add_argument("--resume-from", type=Path, help="Explicitly resume only an authenticated targeted-research refusal into a new directory")
    parser.add_argument("--resume-receipt-sha256")
    parser.add_argument("--resume-proof-manifest-sha256")
    parser.add_argument("--family", choices=("tiling", "coalescing", "work_per_thread"))
    parser.add_argument("--uncovered-technique", choices=TECHNIQUES)
    parser.add_argument("--codex-executable")
    args = parser.parse_args(argv)
    directory = args.output or REPO / "results/research-refresh-acceptance" / args.run_id
    if args.resume_from:
        if args.bootstrap_campaign or args.bootstrap_result_sha256 or args.family or args.uncovered_technique:
            parser.error("Resume inherits the original bootstrap/family/technique; do not override them")
        if args.run:
            receipt = run_resumption_acceptance(directory, args.run_id, args.resume_from,
                receipt_sha256=args.resume_receipt_sha256, manifest_sha256=args.resume_proof_manifest_sha256,
                executable=args.codex_executable)
            print(json.dumps({key:receipt.get(key) for key in ("ok","status","checks","new_gpu_requests_attempted")},indent=2))
            return 0 if receipt["ok"] else 1
        if (Path(directory) / "run-intent.json").exists():
            raise ValueError("Existing resume evidence cannot be replanned")
        prepare_resumption(directory, args.run_id, args.resume_from,
            receipt_sha256=args.resume_receipt_sha256, manifest_sha256=args.resume_proof_manifest_sha256)
        print(json.dumps({"status":"local_resumption_plan_only","model_invoked":False,"new_gpu_requests":0,"plan":str(Path(directory)/"plan.json")},indent=2))
        return 0
    if args.resume_receipt_sha256 or args.resume_proof_manifest_sha256:
        parser.error("Resume SHA256 pins require --resume-from")
    if args.run:
        if not args.bootstrap_campaign:
            parser.error("--run requires --bootstrap-campaign and its --bootstrap-result-sha256")
        receipt = run_acceptance(directory, args.run_id, args.bootstrap_campaign,
            result_sha256=args.bootstrap_result_sha256, family=args.family or "tiling",
            uncovered_technique=args.uncovered_technique, executable=args.codex_executable)
        print(json.dumps({k: receipt.get(k) for k in ("ok", "status", "checks", "new_gpu_requests_attempted")}, indent=2))
        return 0 if receipt["ok"] else 1
    if (Path(directory) / "run-intent.json").exists():
        raise ValueError("Existing execution evidence cannot be replanned")
    _, plan = prepare(directory, args.run_id, args.bootstrap_campaign,
                      family=args.family or "tiling", result_sha256=args.bootstrap_result_sha256)
    print(json.dumps({"status": "local_plan_only", "model_invoked": False, "new_gpu_requests": 0,
                      "plan": str(Path(directory) / "plan.json"), "bootstrap": plan["bootstrap_directory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
