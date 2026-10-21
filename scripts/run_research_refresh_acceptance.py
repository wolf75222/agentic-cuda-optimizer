#!/usr/bin/env python3
"""Documentary AC-02 control: plan by default; real Codex/web only with --run.

A completed campaign supplies an unchanged, authenticated historical baseline
and hardware identity. Nothing is compiled, evaluated, integrated or finalized.
The control asks for an uncovered technique BEFORE the model call; it never
edits a returned payload or a dossier to obtain the expected refusal/refresh.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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
from optimizer_agent.codex_backend import CandidateGenerator, CodexBackend, CodexConfig, default_codex_executable
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.research_service import ResearchConfig, ResearchDossier, ResearchService
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="Default: local plan only, no model/network/GPU")
    mode.add_argument("--run", action="store_true", help="Explicitly run actual local Codex/web documentary controls, no GPU")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id", default="research-refresh-" + uuid.uuid4().hex[:12])
    parser.add_argument("--bootstrap-campaign", type=Path)
    parser.add_argument("--bootstrap-result-sha256")
    parser.add_argument("--family", choices=("tiling", "coalescing", "work_per_thread"), default="tiling")
    parser.add_argument("--uncovered-technique", choices=TECHNIQUES)
    parser.add_argument("--codex-executable")
    args = parser.parse_args(argv)
    directory = args.output or REPO / "results/research-refresh-acceptance" / args.run_id
    if args.run:
        if not args.bootstrap_campaign:
            parser.error("--run requires --bootstrap-campaign and its --bootstrap-result-sha256")
        receipt = run_acceptance(directory, args.run_id, args.bootstrap_campaign,
            result_sha256=args.bootstrap_result_sha256, family=args.family,
            uncovered_technique=args.uncovered_technique, executable=args.codex_executable)
        print(json.dumps({k: receipt.get(k) for k in ("ok", "status", "checks", "new_gpu_requests_attempted")}, indent=2))
        return 0 if receipt["ok"] else 1
    if (Path(directory) / "run-intent.json").exists():
        raise ValueError("Existing execution evidence cannot be replanned")
    _, plan = prepare(directory, args.run_id, args.bootstrap_campaign,
                      family=args.family, result_sha256=args.bootstrap_result_sha256)
    print(json.dumps({"status": "local_plan_only", "model_invoked": False, "new_gpu_requests": 0,
                      "plan": str(Path(directory) / "plan.json"), "bootstrap": plan["bootstrap_directory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
