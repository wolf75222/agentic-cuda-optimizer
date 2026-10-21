"""Local Codex + durable LangGraph + TreeQuest, with CUDA workers on Slurm.

Run from the repository: python -m optimizer_agent.campaign run --config FILE
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback
import uuid

# Original repository modules use direct imports; retain their public CLI while
# allowing the new package entry point to share those exact buffer contracts.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "optimizer_agent"))

from langgraph.checkpoint.sqlite import SqliteSaver

from .campaign_config import CampaignConfig
from .campaign_workflow import build_campaign_graph
from .codex_backend import CodexBackend, CodexConfig, CandidateGenerator, Proposal, PROPOSAL_SCHEMA
from .experiment_store import ExperimentStore, BudgetExceeded
from .search_policy import SearchPolicy
from .greedy_search_policy import GreedySearchPolicy
from .treequest_eligibility import NoEligibleActions
from .research_service import ResearchService, ResearchConfig, ResearchDossier, ResearchBlocked
from .autotuning import Autotuner
from .tuning_constraints import compile_constraints, ConstraintSpecificationError
from .pmpp_guidance import PMPP_GUIDANCE
from .diagnostics import build_diagnostic, evidence_reference, verify_diagnostic, evaluation_summary
from .workload_semantics import operation_spec
from benchmark_workloads import WorkloadSuite
from evaluation import Candidate, Evaluator, LaunchRule, WorkloadContract
from execution_backend import SlurmBackend, SlurmConfig
from use_harness import HarnessRuner, executable
from hybrid_profiler import HybridProfiler


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def device_identity(properties):
    # Memory availability and job IDs change even on the same reserved GPU.
    names = ("name", "architecture", "compute_capability", "uuid", "gpu_uuid",
             "driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")
    return {name: properties[name] for name in names if name in properties}


class RecordedCodex:
    """Record actual usage without a global token, cost or elapsed-time cap."""
    def __init__(self, backend, store, run_id):
        self.backend, self.store, self.run_id = backend, store, run_id

    def generate_json(self, *args, **kwargs):
        reservation = "model-" + uuid.uuid4().hex
        self.store.reserve(self.run_id, reservation, {"model_calls": 1})
        try:
            result = self.backend.generate_json(*args, **kwargs)
        except BaseException:
            # Provider consumption may be unknown after a transport failure.
            self.store.event(self.run_id, "model_uncertain", {"reservation_id": reservation})
            raise
        usage = result.usage
        tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        self.store.reconcile(self.run_id, reservation, {"model_calls": 1, "model_tokens": tokens})
        self.store.event(self.run_id, "model", {"call_id": result.call_id, "purpose": kwargs.get("purpose"),
                         "usage": usage, "elapsed_seconds": result.elapsed_seconds})
        return result


class CampaignRuntime:
    def __init__(self, config, runner, directory, *, model_backend=None, generation_only=False):
        bootstrap_started = time.monotonic()
        self.config, self.runner, self.directory = config, runner, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.generation_only = bool(generation_only)
        mode_path = self.directory / "execution-mode.json"
        execution_mode = {"generation_only": self.generation_only}
        if mode_path.exists() and read_json(mode_path) != execution_mode:
            raise ValueError("Finalization policy changed at resume; use a new run_id")
        if self.generation_only and (self.directory / "holdout-exposed.json").exists():
            raise ValueError("Generation-only campaign cannot reuse an exposed final suite")
        write_json(mode_path, execution_mode)
        self.max_repairs = config.max_repairs_per_trial
        config_path = self.directory / "config.json"
        if config_path.exists() and read_json(config_path) != json.loads(json.dumps(config.to_dict())):
            raise ValueError("Configuration changed at resume; use a new run_id")
        write_json(config_path, config.to_dict())
        self.clock_path = self.directory / "clock.json"
        if not self.clock_path.exists():
            write_json(self.clock_path, {"started_at": time.time()})
        self.started_at = read_json(self.clock_path)["started_at"]
        self.properties = runner.get_device_properties(0)
        self.environment = device_identity(self.properties)
        self.environment_hash = digest(self.environment)
        env_path = self.directory / "environment.json"
        if env_path.exists() and read_json(env_path) != self.environment:
            raise ValueError("GPU/toolchain changed; scores cannot be merged at resume")
        write_json(env_path, self.environment)
        self.suite = WorkloadSuite(config.workload, self.directory / "evaluator_data", config)
        semantics = operation_spec(config.workload)
        self.memory_policy = semantics["memory_policy"]
        self.baseline = self.suite.baseline(REPO / config.baseline_source)
        cases = self.suite.cases()
        perf = [c for c in cases if c.metadata["n"] in config.search_sizes]
        contract = WorkloadContract(
            config.run_id, 1, config.workload, self.suite.signature,
            "Independent NumPy CPU implementation in benchmark_workloads.py, float64 accumulation with float32 outputs",
            benchmark_series=config.benchmark_series, confirmation_batches=config.confirmation_batches,
            launches_per_series=config.launches_per_series, warmup=config.warmup,
            confidence=config.confidence_level, min_relative_gain=config.min_relative_gain,
            max_critical_regression=config.max_critical_regression,
            required_sanitizers=config.required_sanitizers,
            metadata_json=json.dumps({"operation": config.workload, "precision": "float32",
                "mathematics": semantics["mathematics"], "layout": semantics["layout"],
                "timed_scope": "single CUDA kernel, CUDA Events, transfers/compilation excluded",
                "cache_regime": "warm, inputs reset before every launch",
                "memory_policy": self.memory_policy,
                "environment": self.environment,
                "numerical_tolerance": {"atol": 2e-4, "rtol": 2e-4}} ,sort_keys=True))
        self.evaluator = Evaluator(runner, contract, self.suite.oracle,
            {"short": cases[:2], "search": cases, "performance": perf},
            final_suite_factory=lambda: self.suite.cases(final=True),
            final_suite_identity=digest({"generator": "benchmark_workloads-v1", "sizes": config.final_sizes,
                                         "seed_commitment": digest(config.seed + 1000003)}))
        self.contract_hash = self.evaluator.contract_hash
        write_json(self.directory / "contract.json", {"contract": asdict(contract), "contract_hash": self.contract_hash})
        baseline_path = self.directory / "baseline-measurement.json"
        if baseline_path.exists():
            self.baseline_measurement = read_json(baseline_path)
            if self.baseline_measurement["contract_hash"] != self.contract_hash:
                raise ValueError("Baseline measurement contract differs at resume")
        else:
            self.baseline_measurement = self.evaluator.benchmark(self.baseline)
            self.baseline_measurement["resource_wall_seconds"] = time.monotonic()-bootstrap_started
            write_json(baseline_path, self.baseline_measurement)
        if not self.baseline_measurement.get("ok"):
            raise RuntimeError("Baseline independent validation failed: " + json.dumps(self.baseline_measurement)[:1000])
        self.baseline_latencies = self.latencies(self.baseline_measurement)
        self.store = ExperimentStore(self.directory / "experiments.sqlite")
        policy_class = GreedySearchPolicy if config.experimental_mode == "ablation_no_tree" else SearchPolicy
        self.policy = policy_class(self.store, config.run_id, self.candidate_state(self.baseline),
            self.baseline_latencies, actions=list(config.actions), contract_fingerprint=self.contract_hash,
            environment_fingerprint=self.environment_hash,
            budget_limits={"trials": config.max_trials},
            seed=config.seed)
        baseline_cost = self.baseline_measurement.get("resource_wall_seconds", 0)
        self.store.reserve(config.run_id, "baseline-bootstrap", {"gpu_seconds": baseline_cost})
        self.store.reconcile(config.run_id, "baseline-bootstrap", {"gpu_seconds": baseline_cost})
        bundled = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")
        codex_executable = os.environ.get("CODEX_CLI_PATH") or (str(bundled) if bundled.is_file() else "codex")
        self.raw_backend = model_backend or CodexBackend(CodexConfig(executable=codex_executable,
            model=config.model, reasoning_effort=config.reasoning_effort,
            timeout_seconds=config.model_timeout_seconds, trace_dir=self.directory / "model"))
        self.backend = RecordedCodex(self.raw_backend, self.store, config.run_id)
        self.generator = CandidateGenerator(self.backend)
        self.research_service = ResearchService(self.backend, self.directory / "research", ResearchConfig(
            required=config.research_required, web_required=config.web_search_required,
            experimental=config.experimental_mode is not None,
            include_pmpp=config.experimental_mode != "ablation_no_pmpp",
            pmpp_permanent_only=config.experimental_mode == "ablation_pmpp_permanent_only"))
        local_harness = Path(config.local_profile_harness).expanduser().resolve() if config.local_profile_harness else executable
        local_runner = runner if config.backend == "local" else (HarnessRuner(local_harness, self.directory / "local-profiles") if local_harness.is_file() else None)
        self.profiler = HybridProfiler(remote_runner=runner if config.backend == "romeo" else None,
                                       local_runner=local_runner,
                                       max_profiles=None, max_seconds=None)
        self.workload_description = {**semantics, "contract_hash": self.contract_hash, "operation": config.workload,
            "signature": self.suite.signature, "precision": "float32 input/output, full float32 arithmetic",
            "layout": "row-major contiguous matrices a[n,n], b[n,n], c[n,n], c[row*n+col]=sum_k a[row*n+k]*b[k*n+col]" if config.workload == "matmul" else "contiguous one-dimensional arrays",
            "memory_policy": self.memory_policy,
            "gemm_semantics": "C=A*B, alpha=1, beta=0, no transpose/strides/bias" if config.workload == "matmul" else None,
            "search_sizes": list(config.search_sizes), "shape_domain": "all positive n; tails required",
            "atol": 2e-4, "rtol": 2e-4, "single_kernel": True, "environment": self.environment}

    @staticmethod
    def latencies(measurement):
        return {name: item["latency_ms"] for name, item in measurement["cases"].items()}

    def candidate_state(self, candidate, **extra):
        return {"candidate_id": candidate.candidate_id, "source": candidate.source.read_text(),
                "kernel": candidate.kernel, "launch_rule": asdict(candidate.launch_rule),
                "parameters": candidate.parameters, "artifact_hash": candidate.artifact_hash, **extra}

    def candidate(self, candidate_id):
        state = self.store.get_candidate_source(candidate_id, self.config.run_id)
        path = self.directory / "candidates" / (candidate_id + ".cu")
        path.parent.mkdir(exist_ok=True)
        if path.exists() and path.read_text() != state["source"]:
            raise ValueError("Immutable archived candidate differs on disk")
        if not path.exists():
            path.write_text(state["source"])
        return Candidate(candidate_id, path, state["kernel"], LaunchRule(**state["launch_rule"]), state["parameters"])

    def trial_dir(self, state):
        return self.directory / "trials" / state["trial_id"]

    def trial(self, state):
        return read_json(self.trial_dir(state) / "trial.json")

    def diagnostic_context(self, state, trial=None):
        trial = trial or self.trial(state)
        diagnostic_id = state.get("diagnostic_id") or trial.get("diagnostic_id")
        if not diagnostic_id:
            return None
        return verify_diagnostic(read_json(self.directory / "diagnostics" / (diagnostic_id + ".json")),
            self.directory, contract_hash=self.contract_hash, environment_fingerprint=self.environment_hash)

    def generation_evidence(self, state, trial=None):
        evidence = {"previous_error": state.get("error"),
                    "diagnostic": self.diagnostic_context(state, trial)}
        if state.get("candidate_id") and state.get("repair", 0) > 0:
            path = self.trial_dir(state) / f'evaluation-{state["candidate_id"]}.json'
            if path.exists():
                receipt = read_json(path)
                evidence["previous_evaluation"] = evaluation_summary(receipt)
                evidence["previous_evaluation_reference"] = evidence_reference(path, self.directory,
                    kind="evaluation", identity=receipt.get("evaluation_id"))
        return evidence

    def initialize(self, state):
        self.store.event(self.config.run_id, "checkpoint", {"phase": "initialize", "resume": bool(state.get("initialized"))})
        return {"initialized": True, "trials_done": state.get("trials_done", 0), "error": None}

    def select(self, state):
        if state.get("trials_done", 0) >= self.config.max_trials:
            return {"termination_reason": "trial_budget"}
        try:
            pending = self.policy.pending_trials()
            eligible_pending = []
            for saved_trial in pending:
                try:
                    self.policy.assert_eligible(saved_trial.trial_id)
                    eligible_pending.append(saved_trial)
                except NoEligibleActions:
                    previous_path = self.directory / "trials" / str(saved_trial.trial_id) / "trial.json"
                    elapsed_pending = max(0, time.time()-read_json(previous_path)["selected_at"]) if previous_path.exists() else 0
                    self.policy.suspend_or_cancel(saved_trial.trial_id, "Pending trial became ineligible before resume", consumption={"trials": 1, "gpu_seconds": elapsed_pending})
            pending = eligible_pending
            trial = pending[0] if pending else self.policy.ask(budget_reservation={"trials": 1})
        except BudgetExceeded:
            return {"termination_reason": "budget"}
        except NoEligibleActions:
            return {"termination_reason": "no_eligible_branch"}
        trial_id = str(trial.trial_id)
        directory = self.directory / "trials" / trial_id
        directory.mkdir(parents=True, exist_ok=True)
        trial_path = directory / "trial.json"
        if not trial_path.exists():
            write_json(trial_path, {"trial_id": trial_id, "parent": trial.parent_state,
                                  "action": str(trial.action), "selected_at": time.time(),
                                  "diagnostic_id": state.get("diagnostic_id"),
                                  "research_refresh_required": state.get("research_refresh_required", False)})
        selected_trial = read_json(trial_path)
        self.store.event(self.config.run_id, "selection", read_json(trial_path))
        return {"trial_id": trial_id, "candidate_id": None, "repair": 0, "research_round": 0,
                "research_ready": False, "needs_research": False, "valid": False,
                "retryable": False, "error": None, "termination_reason": None,
                "diagnostic_id": selected_trial.get("diagnostic_id"),
                "research_refresh_required": selected_trial.get("research_refresh_required", False)}

    def research(self, state):
        trial = self.trial(state)
        round_number = state.get("research_round", 0) + 1
        if self.config.experimental_mode in ("vanilla", "ablation_no_research"):
            self.store.event(self.config.run_id, "experimental_research_disabled", {"mode": self.config.experimental_mode})
            return {"research_ready": True, "research_round": round_number, "needs_research": False, "error": None}
        path = self.trial_dir(state) / "dossier.json"
        try:
            diagnostic = self.diagnostic_context(state, trial)
            refresh = bool(state.get("needs_research") or state.get("research_refresh_required"))
            if path.exists() and not refresh:
                dossier = ResearchDossier.from_dict(read_json(path))
                dossier.research_ready(self.contract_hash, trial["action"], self.environment_hash)
                if (diagnostic and diagnostic["requires_targeted_research"] and
                        diagnostic["diagnostic_id"] not in dossier.diagnostic_ids):
                    refresh = True
            if not path.exists() or refresh:
                dossier = self.research_service.prepare(self.workload_description, trial["parent"],
                    {"strategy_family": trial["action"]},
                    {"environment_fingerprint": self.environment_hash, "diagnostic": diagnostic,
                     "previous_error": state.get("error")})
                dossier.research_ready(self.contract_hash, trial["action"], self.environment_hash)
                if (diagnostic and diagnostic["requires_targeted_research"] and
                        diagnostic["diagnostic_id"] not in dossier.diagnostic_ids):
                    raise ResearchBlocked("Targeted dossier does not identify its triggering diagnostic")
                write_json(path, dossier.to_dict())
            self.store.event(self.config.run_id, "research", {"trial_id": state["trial_id"], "research_id": dossier.research_id,
                "diagnostic_id": diagnostic["diagnostic_id"] if diagnostic else None, "targeted_refresh": refresh})
            return {"research_ready": True, "research_round": round_number, "needs_research": False,
                    "research_refresh_required": False, "error": None}
        except (ResearchBlocked, BudgetExceeded, RuntimeError, ValueError) as error:
            self.store.event(self.config.run_id, "research_blocked", {"trial_id": state["trial_id"], "error": str(error)})
            return {"research_ready": False, "research_round": round_number, "error": str(error)}

    def generate(self, state):
        try:
            self.policy.assert_eligible(state["trial_id"])
        except NoEligibleActions as error:
            return {"error": str(error), "valid": False, "retryable": False}
        trial = self.trial(state)
        path = self.trial_dir(state) / f'proposal-{state["repair"]}-{state["research_round"]}.json'
        try:
            experimental = self.config.experimental_mode in ("vanilla", "ablation_no_research")
            if not experimental and not (self.trial_dir(state) / "dossier.json").exists():
                return {"needs_research": True, "error": "needs_research: missing durable dossier", "valid": False}
            dossier = None if experimental else ResearchDossier.from_dict(read_json(self.trial_dir(state) / "dossier.json"))
            evidence = self.generation_evidence(state, trial)
            if not experimental and evidence["diagnostic"] and evidence["diagnostic"]["requires_targeted_research"]:
                if evidence["diagnostic"]["diagnostic_id"] not in dossier.diagnostic_ids:
                    return {"needs_research": True, "error": "needs_research: diagnostic lacks an applicable targeted dossier", "valid": False}
            source_parent = (self.store.get_candidate_source(state["candidate_id"], self.config.run_id)
                             if state.get("repair", 0) > 0 and state.get("candidate_id") else trial["parent"])
            if path.exists():
                data = read_json(path)
            else:
                if self.config.experimental_mode == "vanilla":
                    minimal_schema = {"type": "object", "additionalProperties": False,
                        "properties": {key: PROPOSAL_SCHEMA["properties"][key] for key in ("source", "kernel", "launch_rule")},
                        "required": ["source", "kernel", "launch_rule"]}
                    prompt = ("Optimize this naive CUDA kernel for the stated GPU. Preserve its signature, row-major float32 semantics, "
                              "full precision, all tails and all positive n. Use exactly one kernel without host preprocessing. "
                              "Return the complete source and a deterministic launch rule. square_2d grid is "
                              "ceil(n/tile_x),ceil(n/tile_y),1, with explicit block dimensions.\n"+
                              json.dumps({"workload": self.workload_description, "source": trial["parent"]["source"]}))
                    response = self.backend.generate_json(prompt, minimal_schema, purpose="vanilla")
                    data = {**response.payload, "parameters": {}, "parent_id": trial["parent"]["candidate_id"],
                            "source_parent_id": trial["parent"]["candidate_id"], "contract_hash": self.contract_hash,
                            "strategy_family": trial["action"], "hypothesis": "Vanilla single-prompt proposal",
                            "references": [], "parameter_space": [], "parameter_bindings": [],
                            "usage": response.usage, "call_id": response.call_id}
                elif self.config.experimental_mode == "ablation_no_research":
                    proposal = self.generator.propose_unresearched(trial["parent"], {"strategy_family": trial["action"]},
                        contract_hash=self.contract_hash, environment_fingerprint=self.environment_hash,
                        experimental_mode="ablation_no_research", source_parent=source_parent,
                        evidence=evidence, workload=self.workload_description)
                    data = asdict(proposal)
                else:
                    proposal = self.generator.propose(trial["parent"], {"strategy_family": trial["action"]}, dossier,
                        contract_hash=self.contract_hash, environment_fingerprint=self.environment_hash,
                        source_parent=source_parent,
                        include_pmpp=self.config.experimental_mode != "ablation_no_pmpp",
                        experimental_mode=self.config.experimental_mode if self.config.experimental_mode in ("ablation_no_pmpp", "ablation_pmpp_permanent_only") else None,
                        evidence=evidence, workload=self.workload_description)
                    data = asdict(proposal)
                write_json(path, data)
            # Validate the complete contract and covered techniques again on replay.
            if data["parent_id"] != trial["parent"]["candidate_id"] or data["contract_hash"] != self.contract_hash:
                raise ValueError("Proposal changed exact parent or contract")
            if self.config.experimental_mode != "vanilla":
                Proposal.from_dict(data, parent_id=trial["parent"]["candidate_id"], source_parent_id=source_parent["candidate_id"],
                                   contract_hash=self.contract_hash, family=trial["action"],
                                   experimental_mode=self.config.experimental_mode if self.config.experimental_mode in ("ablation_no_research", "ablation_no_pmpp", "ablation_pmpp_permanent_only") else None)
            if dossier:
                dossier.validate_proposal(data)
            if data["kernel"] != self.config.workload:
                raise ValueError("Candidate changed the kernel ABI")
            candidate_id = digest({"source": data["source"], "launch": data["launch_rule"], "parameters": data["parameters"],
                                   "trial_id": state["trial_id"], "repair": state["repair"]})[:24]
            source_path = self.directory / "candidates" / (candidate_id + ".cu")
            source_path.parent.mkdir(exist_ok=True)
            if source_path.exists() and source_path.read_text() != data["source"]:
                raise ValueError("Candidate identity collision")
            source_path.write_text(data["source"])
            candidate = Candidate(candidate_id, source_path, data["kernel"], LaunchRule(**data["launch_rule"]), data["parameters"])
            candidate_state = self.candidate_state(candidate, tree_parent_id=trial["parent"]["candidate_id"],
                source_parent_id=data["source_parent_id"], trial_id=state["trial_id"], strategy_family=trial["action"],
                hypothesis=data["hypothesis"], references=data["references"])
            self.store.archive_candidate(self.config.run_id, candidate_state)
            self.store.event(self.config.run_id, "generation", {"trial_id": state["trial_id"], "candidate_id": candidate_id})
            return {"candidate_id": candidate_id, "needs_research": False, "error": None}
        except (ValueError, ResearchBlocked) as error:
            return {"needs_research": "needs_research" in str(error), "error": str(error), "valid": False}
        except (BudgetExceeded, RuntimeError) as error:
            return {"error": str(error), "valid": False, "retryable": False}

    def evaluate(self, state):
        self.policy.assert_eligible(state["trial_id"])
        candidate = self.candidate(state["candidate_id"])
        path = self.trial_dir(state) / f'evaluation-{candidate.candidate_id}.json'
        if path.exists():
            measurement = read_json(path)
            if measurement.get("ok") and measurement.get("candidate_id") != candidate.candidate_id:
                candidate = self.candidate(measurement["candidate_id"])
        else:
            reservation = "evaluation-" + uuid.uuid4().hex
            self.store.reserve(self.config.run_id, reservation, {"configurations": 1})
            attempt_id = self.store.new_attempt(self.config.run_id, state["trial_id"], {"candidate_id": candidate.candidate_id})
            started = time.monotonic()
            try:
                proposal_paths = sorted(self.trial_dir(state).glob(f'proposal-{state["repair"]}-*.json'))
                proposal = read_json(proposal_paths[-1])
                space = {p["name"]: p["values"] for p in proposal.get("parameter_space", [])}
                constraints = proposal.get("constraints", [])
                restrictions = compile_constraints(constraints, space.keys())
                constraint_receipt = {"status": "compiled", "parameter_names": list(space),
                                      "constraints": [restriction.to_dict() for restriction in restrictions]}
                write_json(self.trial_dir(state) / "constraints.json", constraint_receipt)
                self.store.event(self.config.run_id, "tuning_constraints", {"trial_id": state["trial_id"],
                    "candidate_id": candidate.candidate_id, **constraint_receipt})
                if self.config.autotuning_enabled and space and self.config.experimental_mode != "vanilla":
                    tuner = Autotuner(self.evaluator, self.store, self.config.run_id)
                    materialize = tuner.materialize

                    def archive_variant(template, parameters):
                        variant = materialize(template, parameters)
                        variant = replace(variant, candidate_id=digest({"executable": variant.candidate_id,
                            "trial_id": state["trial_id"], "source_parent_id": candidate.candidate_id})[:24])
                        archived = self.candidate_state(variant, tree_parent_id=self.trial(state)["parent"]["candidate_id"],
                            source_parent_id=candidate.candidate_id, trial_id=state["trial_id"], strategy_family=self.trial(state)["action"])
                        self.store.archive_candidate(self.config.run_id, archived)
                        return self.candidate(variant.candidate_id)

                    tuner.materialize = archive_variant
                    tuned = tuner.search(proposal, space, {"configurations": self.config.max_configurations_per_trial,
                        "compilations": self.config.max_configurations_per_trial}, restrictions=restrictions, hardware_limits=self.properties)
                    write_json(self.trial_dir(state) / "autotuning.json", {"configurations": [asdict(x) for x in tuned.configurations],
                        "termination_reason": tuned.termination_reason,
                        "winner": tuned.best_candidate.candidate_id if tuned.best_candidate else None,
                        "constraints": constraint_receipt["constraints"]})
                    if tuned.best_candidate is not None:
                        candidate, measurement = tuned.best_candidate, tuned.remeasurement
                    else:
                        statuses = {getattr(o, field).get("status") for o in tuned.configurations
                                    for field in ("validation", "measurement") if getattr(o, field, None)}
                        candidate_errors = {"compilation_error", "numerical_error", "memory_error", "contract_violation"}
                        status = (next(iter(sorted(statuses & candidate_errors))) if statuses & candidate_errors else
                                  "contract_violation" if tuned.configurations and all(o.status == "filtered" for o in tuned.configurations) else
                                  "infrastructure_failure")
                        measurement = {"evaluation_id": uuid.uuid4().hex, "ok": False, "status": status,
                                       "error": "No valid tuning configuration", "local_statuses": sorted(str(x) for x in statuses)}
                else:
                    if not all(restriction(candidate.parameters) for restriction in restrictions):
                        raise ConstraintSpecificationError("Concrete candidate violates declared tuning constraints")
                    measurement = self.evaluator.benchmark(candidate)
            except ConstraintSpecificationError as error:
                write_json(self.trial_dir(state) / "constraints.json", {"status": "rejected", "error": str(error),
                    "constraints": proposal.get("constraints", [])})
                self.store.event(self.config.run_id, "tuning_constraints_rejected", {"trial_id": state["trial_id"],
                    "candidate_id": candidate.candidate_id, "constraints": proposal.get("constraints", []), "error": str(error)})
                measurement = {"evaluation_id": uuid.uuid4().hex, "ok": False, "status": "contract_violation", "error": str(error)}
            except Exception as error:
                measurement = {"evaluation_id": uuid.uuid4().hex, "ok": False, "status": "infrastructure_failure", "error": str(error)}
            measurement["attempt_id"] = attempt_id
            write_json(path, measurement)
            self.store.finish_attempt(attempt_id, measurement["status"], {"path": str(path), "elapsed_seconds": time.monotonic()-started})
            self.store.reconcile(self.config.run_id, reservation, {"configurations": 1})
            if measurement.get("ok"):
                # The selected tuner winner has a distinct immutable source ID.
                write_json(self.trial_dir(state) / f'evaluation-{candidate.candidate_id}.json', measurement)
        self.store.event(self.config.run_id, "measurement", {"trial_id": state["trial_id"], "candidate_id": candidate.candidate_id,
                        "evaluation_id": measurement["evaluation_id"], "status": measurement["status"]})
        retryable = measurement["status"] in ("compilation_error", "numerical_error", "memory_error", "contract_violation")
        return {"candidate_id": candidate.candidate_id, "valid": bool(measurement.get("ok")), "retryable": retryable,
                "error": None if measurement.get("ok") else json.dumps(measurement)[:5000]}

    def integrate(self, state):
        result_path = self.trial_dir(state) / "result.json"
        if result_path.exists():
            result = read_json(result_path)
        else:
            result = self.prepare_result(state)
            # Complete result, promotion evidence and exact consumption become
            # durable together before any tree/selection effect is applied.
            write_json(result_path, result)
        if result.get("promotion"):
            promotion = result["promotion"]
            self.store.promote(self.config.run_id, result["candidate_id"], promotion["expected_version"], promotion["decision"])
        if result.get("tell"):
            self.policy.tell_once(state["trial_id"], result["tell"])
            if not result["tell"]["valid"]:
                self.policy.close("candidate", result["candidate_id"], "repair budget exhausted", evidence=[result.get("error")])
        else:
            self.policy.suspend_or_cancel(state["trial_id"], result.get("error") or "Research/infrastructure unavailable",
                                          consumption=result["consumption"])
        return {"trials_done": state.get("trials_done", 0)+1}

    def prepare_result(self, state):
        trial = self.trial(state)
        consumption = {"trials": 1, "gpu_seconds": max(0, time.time()-trial["selected_at"])}
        result = {"candidate_id": state.get("candidate_id"), "error": state.get("error"), "consumption": consumption}
        if state.get("valid"):
            candidate = self.candidate(state["candidate_id"])
            measurement = read_json(self.trial_dir(state) / f'evaluation-{candidate.candidate_id}.json')
            incumbent = self.store.incumbent(self.config.run_id)
            incumbent_candidate = self.candidate(incumbent["candidate_id"] or "baseline")
            confirmation_path = self.trial_dir(state) / "confirmation.json"
            if confirmation_path.exists():
                confirmation = read_json(confirmation_path)
            else:
                confirmation = self.evaluator.confirm(candidate, incumbent_candidate)
                confirmation["incumbent_version"] = incumbent["version"]
                write_json(confirmation_path, confirmation)
            self.store.add_evaluation(self.config.run_id, confirmation)
            if confirmation["status"] == "promoted":
                result["promotion"] = {"expected_version": confirmation["incumbent_version"],
                    "decision": {"status": "improved", "evidence": [confirmation["evaluation_id"]], "interval": confirmation["interval"]}}
            consumption["gpu_seconds"] = max(0, time.time()-trial["selected_at"])
            tell = {"evaluation_id": measurement["evaluation_id"], "valid": True,
                    "latencies": self.latencies(measurement), "purpose": "tree_reward", "fresh": True,
                    "consumption": consumption,
                    "state": self.store.get_candidate_source(candidate.candidate_id, self.config.run_id)}
            result["tell"] = tell
        elif state.get("candidate_id") and state.get("retryable"):
            tell = {"evaluation_id": uuid.uuid4().hex, "valid": False, "purpose": "tree_reward",
                    "consumption": consumption,
                    "state": self.store.get_candidate_source(state["candidate_id"], self.config.run_id)}
            result["tell"] = tell
        return result

    def diagnose(self, state):
        directory = self.trial_dir(state)
        stage = "result" if (directory / "result.json").exists() else "proposal"
        marker = directory / ("diagnostic-" + digest({"stage": stage,
            "candidate_id": state.get("candidate_id"), "repair": state.get("repair", 0),
            "research_round": state.get("research_round", 0), "error": state.get("error"),
            "needs_research": state.get("needs_research", False)})[:24] + ".json")
        if marker.exists():
            diagnostic = verify_diagnostic(read_json(marker), self.directory,
                contract_hash=self.contract_hash, environment_fingerprint=self.environment_hash)
            return {"diagnostic_id": diagnostic["diagnostic_id"],
                    "research_refresh_required": diagnostic["requires_targeted_research"]}
        if self.config.profile and state.get("valid"):
            path = directory / "profile.json"
            if not path.exists():
                reservation = "profile-"+uuid.uuid4().hex
                self.store.reserve(self.config.run_id, reservation, {}, purpose="profile")
                started = time.monotonic()
                candidate = self.candidate(state["candidate_id"])
                case = self.evaluator._suites["performance"][-1]
                request = self.evaluator._request(candidate, case)
                try:
                    profile = self.profiler.profile("profile-"+uuid.uuid4().hex, request,
                        question="Why did this strategy's measured case latencies and paired confirmation support or contradict its hypothesis: occupancy, memory throughput or instruction execution?",
                        metrics=["gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "dram__throughput.avg.pct_of_peak_sustained_elapsed"])
                except Exception as error:
                    profile = {"ok": False, "error": str(error), "diagnostic_only": True,
                               "ranking_eligible": False}
                write_json(path, profile)
                self.store.reconcile(self.config.run_id, reservation, {"gpu_seconds": time.monotonic()-started})
                self.store.event(self.config.run_id, "profile", {"trial_id": state["trial_id"], "ok": profile.get("ok"), "diagnostic_only": True})
        trial, references = self.trial(state), []
        candidate = self.candidate(state["candidate_id"]) if state.get("candidate_id") else None
        candidate_state = self.candidate_state(candidate) if candidate else None
        if candidate:
            references.append(evidence_reference(candidate.source, self.directory,
                kind="candidate_source", identity=candidate.candidate_id))
        references.append(evidence_reference(directory / "trial.json", self.directory,
            kind="trial", identity=state["trial_id"]))

        def receipt(path, kind, identity_key=None):
            if not path.exists():
                return None
            value = read_json(path)
            references.append(evidence_reference(path, self.directory, kind=kind,
                identity=value.get(identity_key) if identity_key else None))
            return value

        proposal_path = directory / f'proposal-{state.get("repair", 0)}-{state.get("research_round", 0)}.json'
        if not proposal_path.exists():
            paths = sorted(directory.glob(f'proposal-{state.get("repair", 0)}-*.json'))
            proposal_path = paths[-1] if paths else proposal_path
        proposal = receipt(proposal_path, "proposal", "call_id")
        if proposal is None and state.get("needs_research"):
            calls = getattr(self.raw_backend, "calls", [])
            if calls and isinstance(calls[-1].payload, dict) and "source" in calls[-1].payload:
                proposal = {**calls[-1].payload, "call_id": calls[-1].call_id,
                            "execution_status": "blocked_before_execution"}
                trace_dir = getattr(getattr(self.raw_backend, "config", None), "trace_dir", None)
                trace = Path(trace_dir) / (calls[-1].call_id + ".json") if trace_dir else None
                if trace and trace.is_file():
                    references.append(evidence_reference(trace, self.directory,
                        kind="blocked_proposal_model_receipt", identity=calls[-1].call_id))
        measurement = receipt(directory / f'evaluation-{state.get("candidate_id")}.json', "evaluation", "evaluation_id")
        confirmation = receipt(directory / "confirmation.json", "confirmation", "evaluation_id")
        profile = receipt(directory / "profile.json", "profile")
        history = []
        for result_path in (self.directory / "trials").glob("*/result.json"):
            trial_path = result_path.parent / "trial.json"
            if not trial_path.exists():
                continue
            result, recorded_trial = read_json(result_path), read_json(trial_path)
            tell = result.get("tell") or {}
            history.append({"trial_id": recorded_trial["trial_id"], "strategy_family": recorded_trial["action"],
                "selected_at": recorded_trial["selected_at"], "valid": tell.get("valid"),
                "evaluation_id": tell.get("evaluation_id"), "promoted": bool(result.get("promotion"))})
            references.append(evidence_reference(result_path, self.directory,
                kind="trial_result", identity=recorded_trial["trial_id"]))
        history.sort(key=lambda result: result["selected_at"])
        diagnostic = build_diagnostic(run_id=self.config.run_id, trial_id=state["trial_id"],
            candidate_id=state.get("candidate_id"), strategy_family=trial["action"],
            contract_hash=self.contract_hash, environment_fingerprint=self.environment_hash,
            stage=stage, error=state.get("error"), needs_research=state.get("needs_research", False),
            proposal=proposal, measurement=measurement, confirmation=confirmation, profile=profile,
            previous_results=history, evidence_references=references, candidate=candidate_state,
            pmpp_guidance=PMPP_GUIDANCE if self.config.experimental_mode != "ablation_no_pmpp" else None)
        path = self.directory / "diagnostics" / (diagnostic["diagnostic_id"] + ".json")
        if path.exists() and read_json(path) != diagnostic:
            raise ValueError("Immutable diagnostic identity collision")
        if not path.exists():
            write_json(path, diagnostic)
        write_json(marker, diagnostic)
        self.store.event(self.config.run_id, "diagnostic", {"trial_id": state["trial_id"],
            "candidate_id": state.get("candidate_id"), "diagnostic_id": diagnostic["diagnostic_id"],
            "triggers": diagnostic["triggers"], "requires_targeted_research": diagnostic["requires_targeted_research"]})
        return {"diagnostic_id": diagnostic["diagnostic_id"],
                "research_refresh_required": diagnostic["requires_targeted_research"]}

    def finalize(self, state):
        final_path = self.directory / "final.json"
        incumbent = self.store.incumbent(self.config.run_id)
        best = self.candidate(incumbent["candidate_id"] or "baseline")
        shutil.copyfile(best.source, self.directory / "best.cu")
        if self.generation_only:
            # The section-16 coordinator seals all methods before constructing
            # their common independent final suite on one reserved GPU.
            frozen = {"candidate_id": best.candidate_id, "kernel": best.kernel,
                      "source": str((self.directory / "best.cu").resolve()),
                      "source_hash": best._source_hash, "artifact_hash": best.artifact_hash,
                      "launch_rule": asdict(best.launch_rule), "parameters": best.parameters,
                      "certification_status": "generation_complete"}
            write_json(self.directory / "best-launch.json", frozen)
            write_json(self.directory / "candidate.json", frozen)
            usage = self.store.budget_state(self.config.run_id)
            manifest = {"run_id": self.config.run_id,
                        "termination_reason": state.get("termination_reason") or "completed",
                        "certification_status": "generation_complete", "generation_only": True,
                        "config": self.config.to_dict(), "contract_hash": self.contract_hash,
                        "environment": self.environment, "model": self.config.model,
                        "reasoning_effort": self.config.reasoning_effort,
                        "trials_done": state.get("trials_done", 0),
                        "best_candidate_id": best.candidate_id, "artifact_hash": best.artifact_hash,
                        "candidate": frozen, "baseline_latencies_ms": self.baseline_latencies,
                        "usage": usage["spent"], "pending_usage": usage["reserved"],
                        "global_resource_caps": None, "elapsed_seconds": time.time()-self.started_at,
                        "experimental_mode": self.config.experimental_mode,
                        "final": {"status": "deferred_common_final_evaluation", "ok": False}}
            write_json(self.directory / "manifest.json", manifest)
            write_json(self.directory / "events.json", self.store.events(self.config.run_id))
            return {"finished": True, "certification_status": "generation_complete"}
        write_json(self.directory / "best-launch.json", {"candidate_id": best.candidate_id,
                   "artifact_hash": best.artifact_hash, "launch_rule": asdict(best.launch_rule), "parameters": best.parameters,
                   "certification_status": "finalization_incomplete"})
        if final_path.exists():
            final = read_json(final_path)
            if any(final.get(key) != value for key, value in (
                ("candidate_id", best.candidate_id), ("artifact_hash", best.artifact_hash),
                ("contract_hash", self.contract_hash))):
                write_json(self.directory / "previous-final-receipt.json", final)
                final = {"ok": False, "status": "finalization_incomplete", "candidate_id": best.candidate_id,
                         "artifact_hash": best.artifact_hash, "contract_hash": self.contract_hash,
                         "error": "Saved final evidence belongs to a different candidate or contract; original receipt preserved"}
                write_json(final_path, final)
        else:
            # Persist exposure BEFORE invoking final evaluation. An interrupted
            # finalization cannot silently reuse a disclosed holdout at resume.
            exposure = self.directory / "holdout-exposed.json"
            if exposure.exists():
                final = {"ok": False, "status": "finalization_incomplete", "error": "Finalization interrupted after holdout exposure; independent new final suite required"}
            else:
                write_json(exposure, {"candidate_id": best.candidate_id, "artifact_hash": best.artifact_hash})
                self.store.reserve(self.config.run_id, "finalization-execution", {}, purpose="finalization")
                started = time.monotonic()
                try:
                    final = self.evaluator.finalize(best, self.baseline)
                except Exception as error:
                    final = {"ok": False, "status": "finalization_incomplete", "error": str(error)}
                self.store.reconcile(self.config.run_id, "finalization-execution", {"gpu_seconds": time.monotonic()-started})
            write_json(final_path, final)
        shutil.copyfile(best.source, self.directory / "best.cu")
        write_json(self.directory / "best-launch.json", {"candidate_id": best.candidate_id,
                   "artifact_hash": best.artifact_hash, "launch_rule": asdict(best.launch_rule), "parameters": best.parameters,
                   "certification_status": final["status"]})
        usage = self.store.budget_state(self.config.run_id)
        manifest = {"run_id": self.config.run_id, "termination_reason": state.get("termination_reason") or "completed",
                    "certification_status": final["status"], "config": self.config.to_dict(),
                    "contract_hash": self.contract_hash, "environment": self.environment,
                    "model": self.config.model, "reasoning_effort": self.config.reasoning_effort,
                    "trials_done": state.get("trials_done", 0), "best_candidate_id": best.candidate_id,
                    "artifact_hash": best.artifact_hash, "baseline_latencies_ms": self.baseline_latencies,
                    "usage": usage["spent"], "pending_usage": usage["reserved"],
                    "global_resource_caps": None, "elapsed_seconds": time.time()-self.started_at,
                    "experimental_mode": self.config.experimental_mode,
                    "final": final}
        write_json(self.directory / "manifest.json", manifest)
        write_json(self.directory / "events.json", self.store.events(self.config.run_id))
        return {"finished": True, "certification_status": final["status"]}


def run_on_runner(config, runner, directory, *, resume=False, generation_only=False):
    directory = Path(directory)
    runtime = CampaignRuntime(config, runner, directory, generation_only=generation_only)
    with SqliteSaver.from_conn_string(str(directory / "langgraph.sqlite")) as checkpointer:
        graph = build_campaign_graph(runtime, checkpointer)
        graph_config = {"configurable": {"thread_id": config.run_id}, "recursion_limit": max(100, config.max_trials*40)}
        existing = graph.get_state(graph_config)
        if existing.values and not resume:
            raise ValueError("Campaign already exists; use resume or a new run_id")
        if existing.values and not existing.next:
            return read_json(directory / "manifest.json")
        initial = None if existing.values else {"run_id": config.run_id, "trials_done": 0}
        for value in graph.stream(initial, graph_config, stream_mode="updates"):
            print(json.dumps({"event": "graph", "run_id": config.run_id, "updates": value}, default=str), flush=True)
    return read_json(directory / "manifest.json")


def execute(config, *, resume=False, worker_session_id=None):
    directory = (REPO / config.results_dir / config.run_id).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        if resume:
            return read_json(directory / "manifest.json")
        raise ValueError("Campaign already completed; use resume to inspect it or a new run_id")
    backend = None
    try:
        if config.backend == "romeo":
            backend = SlurmBackend(SlurmConfig.load(REPO / config.execution_config), REPO, directory / "worker", session_id=worker_session_id or config.run_id)
            print(json.dumps({"event": "slurm_submitting", "run_id": config.run_id}), flush=True)
            backend.start()
            print(json.dumps({"event": "slurm_started", "job_id": backend.job_id}), flush=True)
            backend.wait_until_ready()
            runner = backend.runner()
        else:
            if not executable.is_file():
                raise FileNotFoundError("Local CUDA harness is unavailable; configure a CUDA-capable local worker")
            runner = HarnessRuner(executable, directory / "worker")
        return run_on_runner(config, runner, directory, resume=resume)
    except BaseException as error:
        write_json(directory / "failure.json", {"status": "finalization_incomplete", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if backend:
            backend.close()
            write_json(directory / "worker-terminal.json", backend.wait_until_finished(timeout=60))


def compare(config, *, worker_session_id=None):
    """Naive baseline, full agent, vanilla one-prompt Luna on one allocation.

    Protocols are identical; actual consumption is published independently.
    This paired pilot is not the multi-campaign ablation study from spec section16.
    """
    if config.backend != "romeo":
        raise ValueError("The reproducible ROMEO comparison requires backend=romeo")
    directory = (REPO / config.results_dir / config.run_id).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    backend = SlurmBackend(SlurmConfig.load(REPO / config.execution_config), REPO, directory / "worker", session_id=worker_session_id or config.run_id)
    manifests = {}
    try:
        print(json.dumps({"event": "slurm_started", "job_id": backend.start()}), flush=True)
        backend.wait_until_ready()
        runner = backend.runner()
        for mode in ("agent", "vanilla"):
            selected = replace(config, run_id=config.run_id + "." + mode)
            if mode == "vanilla":
                selected = replace(selected, experimental_mode="vanilla", max_trials=1, max_repairs_per_trial=0,
                                   autotuning_enabled=False, research_required=False, web_search_required=False,
                                   actions=("vanilla",), profile=False)
            manifests[mode] = run_on_runner(selected, runner, directory / mode,
                                          resume=(directory / mode / "langgraph.sqlite").exists())
        direct = compare_generated_candidates(config, directory, runner)
        result = {"kind": "single_campaign_pilot", "same_allocation": True,
                  "job_id": backend.job_id, "hardware": backend.hardware,
                  "resource_comparison": "No global token/GPU resource ceilings; actual usage recorded, vanilla uses one generation prompt",
                  "generated_agent_vs_vanilla": direct,
                  "results": {mode: {"status": value["certification_status"],
                        "candidate_id": value["best_candidate_id"],
                        "artifact_hash": value["artifact_hash"],
                        "final_gain": value["final"].get("confirmation", {}).get("interval"),
                        "baseline_latencies_ms": value["baseline_latencies_ms"],
                        "usage": value["usage"], "elapsed_seconds": value["elapsed_seconds"]}
                              for mode, value in manifests.items()},
                  "limits": ["One workload and one campaign per mode; no universal optimizer superiority claim",
                             "Vanilla and agent have different orchestration and consumed resources",
                             "No local profiling time is included in ROMEO performance"]}
        result["ok"] = comparison_completed(result)
        result["status"] = "completed" if result["ok"] else "incomplete"
        write_json(directory / "comparison.json", result)
        return result
    except BaseException as error:
        write_json(directory / "failure.json", {"status": "incomplete", "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        backend.close()
        write_json(directory / "worker-terminal.json", backend.wait_until_finished(timeout=60))


def load_direct_candidates(config, directory):
    """Reconstruct the exact generated artifacts selected for the paired pilot."""
    candidates = {}
    for mode in ("agent", "vanilla"):
        results = [read_json(p) for p in (directory / mode / "trials").glob("*/result.json")]
        valid = [r for r in results if r.get("tell", {}).get("valid")]
        if not valid:
            raise ValueError(f"Missing valid generated kernel: {mode}")
        chosen = min(valid, key=lambda r: sum(math.log(x) for x in r["tell"]["latencies"].values()))
        state = ExperimentStore(directory / mode / "experiments.sqlite").get_candidate_source(
            chosen["candidate_id"], config.run_id + "." + mode)
        path = directory / "direct-comparison" / (mode + ".cu")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != state["source"]:
            raise ValueError(f"Direct comparison source changed: {mode}")
        path.write_text(state["source"])
        candidates[mode] = Candidate(mode+"-"+chosen["candidate_id"], path, state["kernel"],
                                     LaunchRule(**state["launch_rule"]), state["parameters"])
    return candidates


def direct_evaluator(config, directory, runner):
    """Independent post-generation cases shared by measurement and qualification."""
    independent = replace(config, seed=config.seed+3000017)
    suite = WorkloadSuite(config.workload, directory / "direct-comparison" / "inputs", independent)
    cases = suite.cases(final=True)
    contract = WorkloadContract(config.run_id+".direct", 1, config.workload, suite.signature,
        "Independent NumPy oracle on new data, after both generators have ended",
        benchmark_series=config.benchmark_series, confirmation_batches=config.confirmation_batches,
        launches_per_series=config.launches_per_series, warmup=config.warmup,
        confidence=config.confidence_level, min_relative_gain=config.min_relative_gain,
        max_critical_regression=config.max_critical_regression,
        required_sanitizers=config.required_sanitizers,
        metadata_json=json.dumps({"precision": "float32", "scope": "generated kernels on independent final shapes"}))
    return Evaluator(runner, contract, suite.oracle, {"short": cases[:1], "search": cases, "performance": cases})


def compare_generated_candidates(config, directory, runner):
    """Compare and sanitize generated kernels independently after generation."""
    try:
        candidates = load_direct_candidates(config, directory)
    except ValueError as error:
        return {"ok": False, "status": "missing_valid_generated_kernel", "error": str(error)}
    evaluator = direct_evaluator(config, directory, runner)
    result = evaluator.confirm(candidates["agent"], candidates["vanilla"])
    result.update(scope="fresh data/shapes after generation; independent direct kernel comparison",
                  gain_direction="vanilla latency / agent latency; >1 favors agent",
                  candidate_artifacts={mode: c.artifact_hash for mode, c in candidates.items()},
                  required_sanitizers=list(config.required_sanitizers))
    qualification = {mode: evaluator.qualify_sanitizers(candidate)
                     for mode, candidate in candidates.items()}
    result["sanitizer_qualification"] = {"ok": all(item.get("ok") for item in qualification.values()),
                                        "candidates": qualification}
    write_json(directory / "direct-comparison" / "comparison.json", result)
    return result


def comparison_completed(result):
    """A baseline fallback alone cannot complete the generated-kernel pilot."""
    modes = ("agent", "vanilla")
    outcomes = result.get("results", {})
    if any(outcomes.get(mode, {}).get("status") not in
           {"certified", "validated_no_gain", "validated_gain_uncertified"} for mode in modes):
        return False
    direct = result.get("generated_agent_vs_vanilla", {})
    if direct.get("ok") is not True or direct.get("status") not in {"promoted", "not_improved", "inconclusive"}:
        return False
    artifacts = direct.get("candidate_artifacts", {})
    if any(not artifacts.get(mode) for mode in modes) or not direct.get("pairs"):
        return False
    known_tools = {"memcheck", "racecheck", "initcheck", "synccheck"}
    required = direct.get("required_sanitizers")
    if not isinstance(required, (list, tuple)) or not required or set(required)-known_tools:
        return False
    required = set(required)
    pairs = direct["pairs"]
    if not isinstance(pairs, list) or any(not isinstance(pair, dict) or not isinstance(pair.get("case_id"), str)
                                          or not pair["case_id"] for pair in pairs):
        return False
    paired_cases = {pair["case_id"] for pair in pairs}
    qualification = direct.get("sanitizer_qualification", {})
    if not isinstance(qualification, dict) or qualification.get("ok") is not True:
        return False
    qualified_candidates = qualification.get("candidates", {})
    if not isinstance(qualified_candidates, dict):
        return False
    for mode in modes:
        qualified = qualified_candidates.get(mode, {})
        if (not isinstance(qualified, dict) or qualified.get("ok") is not True
                or qualified.get("status") != "sanitizer_qualified"
                or qualified.get("artifact_hash") != artifacts[mode]):
            return False
        tools = qualified.get("required_sanitizers")
        if not isinstance(tools, (list, tuple)) or set(tools) != required:
            return False
        validation = qualified.get("validation", {})
        if not isinstance(validation, dict) or validation.get("ok") is not True:
            return False
        cases = validation.get("cases", [])
        if (not isinstance(cases, list) or not cases
                or any(not isinstance(case, dict) or not isinstance(case.get("case_id"), str)
                       or not case["case_id"] for case in cases)):
            return False
        validated_cases = {case["case_id"] for case in cases}
        if validated_cases != paired_cases:
            return False
        reports = qualified.get("sanitizers", [])
        if not isinstance(reports, list) or not reports:
            return False
        coverage = set()
        for report in reports:
            if (not isinstance(report, dict) or report.get("ok") is not True
                    or report.get("tool") not in known_tools
                    or report.get("case_id") not in validated_cases):
                return False
            coverage.add((report["tool"], report["case_id"]))
        if any((tool, case_id) not in coverage for tool in required for case_id in validated_cases):
            return False
    interval = direct.get("interval", {})
    return all(isinstance(interval.get(key), (int, float)) and math.isfinite(interval[key])
               and interval[key] > 0 for key in ("gain", "lower", "upper"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "resume", "compare", "plan-suite", "suite", "report-suite"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--worker-session", help="New isolated Slurm session for an unfinished resume after the old worker has stopped")
    parser.add_argument("--concurrency", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--include-greedy", action="store_true", help="Add the isolated parent-selection ablation to the actual upstream comparison")
    args = parser.parse_args(argv)
    config = CampaignConfig.load(args.config)
    if args.command in ("plan-suite", "suite", "report-suite"):
        from .quality_suite import plan_suite
        from .suite_orchestrator import SuiteOrchestrator
        from .suite_report import write_report
        plan = plan_suite(config, include_greedy=args.include_greedy)
        plan_directory = (REPO / config.results_dir / config.run_id).resolve()
        # Every entry must dispatch through its method-specific adapter. The
        # original and native Codex baselines cannot use execute(selected),
        # which would silently run the improved agent under another label.
        orchestrator = SuiteOrchestrator(plan, plan_directory, concurrency=args.concurrency)
        status = orchestrator.run() if args.command == "suite" else orchestrator.status()
        write_report(plan_directory, status)
        print(json.dumps({"plan": str(plan_directory / "plan.json"), "campaigns": len(plan["entries"]),
                          "complete": status["summary"]["complete"],
                          "submitted": args.command == "suite", "global_resource_caps": None}, indent=2))
        return 0 if args.command != "suite" or status["summary"]["complete"] else 1
    if args.command == "compare":
        result = compare(config, worker_session_id=args.worker_session)
        print(json.dumps(result, indent=2))
        return 0 if comparison_completed(result) else 1
    result = execute(config, resume=args.command == "resume", worker_session_id=args.worker_session)
    print(json.dumps({"run_id": config.run_id, "certification_status": result["certification_status"],
                      "manifest": str(REPO / config.results_dir / config.run_id / "manifest.json")}, indent=2))
    return 0 if result["certification_status"] in ("certified", "validated_no_gain", "validated_gain_uncertified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
