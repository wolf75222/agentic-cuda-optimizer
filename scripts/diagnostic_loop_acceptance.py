#!/usr/bin/env python3
"""Separate ROMEO proof of diagnostic -> targeted research -> guarded proposal.

Three explicitly labeled baseline-equivalent controls establish real measured
same-family stagnation. They are static acceptance inputs, not model-generated
optimizations and never enter the section-16 quality cohorts. The fourth trial
uses actual Luna Max research and generation, then the independent CUDA path.
Without --run, prepare a CPU-only inspectable plan and contact no service.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent

from optimizer_agent.campaign import (
    CampaignConfig, CampaignRuntime, Candidate, LaunchRule, SlurmBackend, SlurmConfig,
    read_json, write_json,
)
from optimizer_agent.research_service import ResearchDossier
from optimizer_agent.codex_backend import Proposal
from optimizer_agent.campaign_workflow import build_experiment_graph
from optimizer_agent.diagnostics import evidence_reference
from scripts.gpu_autotuning_acceptance import ReceiptRunner


CONTROL_MARKER = "// diagnostic-loop-acceptance control"


def prepare(directory, run_id, execution_config, *, series=3, launches=10, confirmation_batches=3):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    slurm = SlurmConfig.load(execution_config)
    hours,minutes,seconds = (int(value) for value in slurm.walltime.split(":"))
    if hours*3600+minutes*60+seconds > 24*3600:
        raise ValueError("Diagnostic acceptance requires a configured Slurm walltime of at most 24 hours")
    config = CampaignConfig(run_id=run_id, backend="romeo", workload="matmul",
        max_trials=4, max_repairs_per_trial=2, autotuning_enabled=False, actions=("tiling",),
        search_sizes=(64,128), final_sizes=(96,), benchmark_series=series,
        launches_per_series=launches, confirmation_batches=confirmation_batches,
        min_relative_gain=.25, profile=True, required_sanitizers=())
    plan = {"schema_version":1, "kind":"diagnostic_loop_acceptance", "run_id":run_id,
        "scope":"Functional diagnostic loop; static controls excluded from optimizer quality cohorts",
        "controls":3, "control_source":"Exact selected-parent CUDA body plus a trailing comment",
        "control_expectation":"No real optimization and no material gain; an unexpected promotion fails this acceptance",
        "fourth_trial":"Actual Luna Max targeted web/code/CUDA/PMPP research, verified dossier, proposal and independent CUDA evaluation",
        "search_sizes":list(config.search_sizes), "tail_validation_sizes":[1,17,63],
        "protocol":config.to_dict(), "slurm":asdict(slurm),
        "global_resource_caps":None, "holdout_exposure":False,
        "certification_scope":"No general optimizer gain or section-16 study certification"}
    write_json(directory/"plan.json",plan)
    return config, slurm, plan


def install_control(runtime, state, index, dossier):
    """Archive a declared control; no fabricated generation or source citation."""
    trial = runtime.trial(state)
    source = trial["parent"]["source"].rstrip()+f"\n{CONTROL_MARKER} {index}\n"
    candidate_id = f"diagnostic-control-{index}-"+hashlib.sha256(source.encode()).hexdigest()[:16]
    path = runtime.directory/"candidates"/(candidate_id+".cu")
    path.parent.mkdir(exist_ok=True)
    path.write_text(source)
    candidate = Candidate(candidate_id,path,trial["parent"]["kernel"],
        LaunchRule(**trial["parent"]["launch_rule"]),dict(trial["parent"].get("parameters",{})))
    runtime.store.archive_candidate(runtime.config.run_id,runtime.candidate_state(candidate,
        tree_parent_id=trial["parent"]["candidate_id"],source_parent_id=trial["parent"]["candidate_id"],
        trial_id=state["trial_id"],strategy_family=trial["action"],acceptance_control=True))
    # This declared static control is evaluated through the common evaluator.
    # Its empty optimization hypothesis/citations are deliberate, never passed
    # off as a valid LLM proposal. The fourth trial uses all normal proposal guards.
    proposal = {"source":source,"kernel":candidate.kernel,"launch_rule":asdict(candidate.launch_rule),
        "parameters":candidate.parameters,"parameter_space":[],"parameter_bindings":[],
        "parent_id":trial["parent"]["candidate_id"],"source_parent_id":trial["parent"]["candidate_id"],
        "contract_hash":runtime.contract_hash,"strategy_family":trial["action"],
        "acceptance_control":True,"generation_mode":"static baseline-equivalent acceptance input",
        "hypothesis":"Baseline-equivalence control; no optimization transformation is claimed",
        "expected_measurement":"No material gain; establish measured same-family control history",
        "references":[],"techniques":[],"source_research_id":dossier.research_id}
    write_json(runtime.trial_dir(state)/f'proposal-0-{state["research_round"]}.json',proposal)
    return candidate


def execute_on_runner(runtime):
    """Run actual controls and the fourth genuine model trial on an owned worker."""
    controls = []
    state = {"run_id":runtime.config.run_id,"trials_done":0,**runtime.initialize({})}
    initial_dossier = None
    for index in range(1,4):
        state.update(runtime.select(state))
        if state.get("termination_reason"):
            raise RuntimeError("Acceptance selection ended before its three controls")
        if initial_dossier is None:
            state.update(runtime.research(state))
            if not state.get("research_ready"):
                raise RuntimeError("Actual initial mandatory research blocked: "+str(state.get("error")))
            initial_dossier = ResearchDossier.from_dict(read_json(runtime.trial_dir(state)/"dossier.json"))
        else:
            initial_dossier.research_ready(runtime.contract_hash,"tiling",runtime.environment_hash)
            write_json(runtime.trial_dir(state)/"dossier.json",initial_dossier.to_dict())
            state.update(research_ready=True,research_round=1)
        candidate = install_control(runtime,state,index,initial_dossier)
        state.update(candidate_id=candidate.candidate_id)
        state.update(runtime.evaluate(state))
        if not state.get("valid"):
            raise RuntimeError("Actual baseline-equivalent control failed validation: "+str(state.get("error")))
        state.update(runtime.integrate(state))
        confirmation = read_json(runtime.trial_dir(state)/"confirmation.json")
        measurement = read_json(runtime.trial_dir(state)/f'evaluation-{candidate.candidate_id}.json')
        control = {"trial_id":state["trial_id"],"candidate_id":candidate.candidate_id,
            "source_hash":candidate._source_hash,"artifact_hash":candidate.artifact_hash,
            "evaluation_id":measurement["evaluation_id"],"confirmation_id":confirmation["evaluation_id"],
            "measurement":measurement,"confirmation":confirmation,
            "static_acceptance_control":True,"excluded_from_quality_cohorts":True}
        controls.append(control)
        write_json(runtime.directory/"acceptance-controls.json",controls)
        if confirmation["status"] == "promoted":
            raise RuntimeError("Unexpected control promotion; preserve measured evidence, do not manufacture stagnation")
    state.update(runtime.diagnose(state))
    diagnostic = runtime.diagnostic_context(state)
    if "stagnation" not in diagnostic["triggers"]:
        raise RuntimeError("Three actual controls did not produce the required deterministic stagnation diagnostic")
    tree_after_controls = len(runtime.policy.snapshot()["tree"].tree)
    fourth = {"run_id":runtime.config.run_id,"trials_done":3,
              "diagnostic_id":diagnostic["diagnostic_id"],"research_refresh_required":True}
    fourth.update(runtime.select(fourth))
    fourth.update(runtime.research(fourth))
    if not fourth.get("research_ready"):
        raise RuntimeError("Actual diagnostic-informed research blocked: "+str(fourth.get("error")))
    dossier = ResearchDossier.from_dict(read_json(runtime.trial_dir(fourth)/"dossier.json"))
    if diagnostic["diagnostic_id"] not in dossier.diagnostic_ids:
        raise RuntimeError("Targeted dossier lost its immutable diagnostic linkage")
    fourth, attempts = run_guarded_experiment(runtime, fourth)
    if (not attempts or attempts[-1]["repair"] != fourth["repair"]
            or not attempts[-1].get("evaluation_reference")):
        raise RuntimeError("Actual guarded Luna proposal blocked: "+str(fourth.get("error")))
    proposal = verify_real_proposal(runtime, fourth, dossier,
        source_parent_id=attempts[-1]["effective_source_parent_id"],
        proposal_reference=attempts[-1]["proposal_reference"])
    if not fourth.get("valid"):
        raise RuntimeError("Actual Luna proposal failed independent CUDA evaluation after configured repairs: "+str(fourth.get("error")))
    fourth.update(runtime.integrate(fourth))
    candidate = runtime.candidate(fourth["candidate_id"])
    measured = read_json(runtime.trial_dir(fourth)/f'evaluation-{candidate.candidate_id}.json')
    profile = read_json(runtime.trial_dir(state)/"profile.json") if (runtime.trial_dir(state)/"profile.json").exists() else None
    checks = {"three_independent_actual_control_evaluations":len({x["evaluation_id"] for x in controls}) == 3,
        "controls_valid_and_not_promoted":all(x["measurement"].get("ok") and x["confirmation"]["status"] != "promoted" for x in controls),
        "stagnation_links_actual_measurements":set(x["evaluation_id"] for x in controls) <= {
            reference.get("identity") for reference in diagnostic["evidence_references"]} | {
                identity for observation in diagnostic["observations"] if observation["kind"] == "stagnation"
                for identity in observation["evaluation_ids"]},
        "actual_targeted_new_dossier":dossier.research_id != initial_dossier.research_id and bool(dossier.queries),
        "diagnostic_dossier_proposal_chain":diagnostic["diagnostic_id"] in dossier.diagnostic_ids and bool(proposal.get("call_id")),
        "guarded_proposal_actually_validated_and_measured":bool(measured.get("ok")),
        "three_controls_integrated_once":tree_after_controls == 4,
        "fourth_trial_integrated_once":len(runtime.policy.snapshot()["tree"].tree) == 5,
        "no_final_holdout_exposed":not (runtime.directory/"holdout-exposed.json").exists()}
    return {"controls":controls,"diagnostic_id":diagnostic["diagnostic_id"],"diagnostic":diagnostic,
        "initial_research_id":initial_dossier.research_id,"targeted_research_id":dossier.research_id,
        "fourth_trial_id":fourth["trial_id"],"proposal_call_id":proposal["call_id"],
        "fourth_attempts":attempts,
        "candidate_id":candidate.candidate_id,"candidate_artifact_hash":candidate.artifact_hash,
        "candidate_measurement":measured,"actual_profile_or_unavailability":profile,"checks":checks}


def run_guarded_experiment(runtime, state):
    """Use the campaign's actual repair routing and retain every attempt receipt."""
    state = dict(state)
    attempts, events = [], []
    path = runtime.trial_dir(state) / "acceptance-experiment.json"
    status = "running"
    try:
        for update in build_experiment_graph(runtime).stream(state, stream_mode="updates"):
            for node, values in update.items():
                previous_candidate = state.get("candidate_id")
                if node == "generate" and values.get("candidate_id") and not values.get("error"):
                    proposal_path = runtime.trial_dir(state) / f'proposal-{state["repair"]}-{state["research_round"]}.json'
                    proposal = read_json(proposal_path)
                    attempts.append({"repair":state["repair"], "candidate_id":values["candidate_id"],
                        "effective_source_parent_id":(previous_candidate if state["repair"] > 0
                            else runtime.trial(state)["parent"]["candidate_id"]),
                        "proposal_reference":evidence_reference(proposal_path, runtime.directory,
                            kind="proposal", identity=proposal.get("call_id"))})
                state.update(values)
                if node == "evaluate":
                    evaluation_path = runtime.trial_dir(state) / f'evaluation-{state["candidate_id"]}.json'
                    evaluation = read_json(evaluation_path)
                    attempts[-1].update(status=evaluation["status"], valid=bool(state.get("valid")),
                        evaluation_reference=evidence_reference(evaluation_path, runtime.directory,
                            kind="evaluation", identity=evaluation.get("evaluation_id")))
                events.append({"node":node, "repair":state.get("repair"), "update":values})
                write_json(path, {"schema_version":1, "status":status, "max_repairs":runtime.max_repairs,
                                  "attempts":attempts, "events":events})
        status = "completed" if state.get("valid") else "failed"
    except Exception as error:
        status = "graph_error"
        events.append({"node":"exception", "error":str(error)})
        raise
    finally:
        write_json(path, {"schema_version":1, "status":status, "max_repairs":runtime.max_repairs,
                          "attempts":attempts, "events":events})
    return state, attempts


def verify_real_proposal(runtime,state,dossier,*,source_parent_id=None,proposal_reference=None):
    """The fourth model proposal has every normal guard, independent of controls."""
    if runtime.config.experimental_mode is not None:
        raise RuntimeError("The actual fourth proposal requires the normal research-enabled campaign mode")
    trial = runtime.trial(state)
    dossier.research_ready(runtime.contract_hash,trial["action"],runtime.environment_hash)
    path = runtime.trial_dir(state)/f'proposal-{state["repair"]}-{state["research_round"]}.json'
    if proposal_reference and (proposal_reference["path"] != str(path.relative_to(runtime.directory))
            or proposal_reference["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise RuntimeError("The final proposal differs from its exact experiment attempt receipt")
    proposal = read_json(path)
    if proposal.get("acceptance_control") or proposal.get("generation_mode"):
        raise RuntimeError("A static acceptance control cannot serve as the fourth actual model proposal")
    validated = Proposal.from_dict(proposal,parent_id=trial["parent"]["candidate_id"],
        source_parent_id=(source_parent_id or trial["parent"]["candidate_id"]),
        contract_hash=runtime.contract_hash,family=trial["action"])
    dossier.validate_proposal(proposal)
    if not proposal.get("call_id"):
        raise RuntimeError("The fourth actual model proposal is missing its Codex call receipt ID")
    candidate = runtime.candidate(state["candidate_id"])
    if (validated.source != candidate.source.read_text() or validated.kernel != candidate.kernel
            or json.dumps(validated.launch_rule,sort_keys=True) != json.dumps(asdict(candidate.launch_rule),sort_keys=True)
            or validated.parameters != candidate.parameters):
        raise RuntimeError("The final proposal differs from the exact candidate independently evaluated")
    return proposal


def run_acceptance(directory,run_id,execution_config,**protocol):
    directory = Path(directory).resolve()
    if any((directory/name).exists() for name in ("allocation-intent.json","worker/slurm-session.json","diagnostic-loop-acceptance.json")):
        raise ValueError("Use a fresh acceptance directory; never reuse an existing allocation or overwrite evidence")
    config,slurm,plan = prepare(directory,run_id,execution_config,**protocol)
    session = "diagnostic-acceptance-"+uuid.uuid4().hex
    backend = SlurmBackend(slurm,REPO,directory/"worker",session_id=session)
    receipt = {"schema_version":1,"kind":plan["kind"],"run_id":run_id,"session_id":session,
        "job_id":None,"ok":False,"status":"incomplete","scope":plan["scope"],"global_resource_caps":None}
    receipt["remote_dir"] = backend.remote_dir
    receipt["source_provenance"] = controller_provenance(REPO, __file__, execution_config)
    receipt["allocation_intent"] = persist_allocation_intent(directory, backend, receipt["source_provenance"])
    runtime,recording = None,None
    started = time.monotonic()
    try:
        receipt["job_id"] = backend.start()
        print(json.dumps({"event":"diagnostic_acceptance_started","job_id":backend.job_id,"session_id":session}),flush=True)
        backend.wait_until_ready()
        recording = ReceiptRunner(backend.runner(),directory)
        runtime = CampaignRuntime(config,recording,directory/"campaign",generation_only=True)
        receipt.update(execute_on_runner(runtime))
        receipt["contract_hash"],receipt["environment"] = runtime.contract_hash,runtime.environment
    except Exception as error:
        receipt.update(error=str(error),traceback=traceback.format_exc())
    finally:
        if runtime:
            receipt["observed_usage"] = runtime.store.budget_state(run_id)
            write_json(directory/"events.json",runtime.store.events(run_id))
        if recording:
            receipt["native_usage"] = recording.usage()
        if backend.job_id:
            try:
                backend.close()
                ended = backend.wait_until_finished(timeout=60)
                if not ended.get("terminal"):
                    backend.close(cancel=True)
                    ended = backend.wait_until_finished(timeout=60)
                receipt["terminal"] = ended
            except Exception as error:
                receipt["cleanup_error"] = str(error)
        terminal = receipt.get("terminal",{})
        receipt["helper_wall_seconds"] = time.monotonic()-started
        receipt["ok"] = (not receipt.get("error") and not receipt.get("cleanup_error")
            and bool(receipt.get("checks")) and all(receipt["checks"].values())
            and terminal.get("terminal") is True and terminal.get("state") == "COMPLETED"
            and terminal.get("exit_code") == "0:0" and str(terminal.get("job_id")) == str(receipt["job_id"]))
        receipt["status"] = "diagnostic_loop_acceptance_passed" if receipt["ok"] else "incomplete"
        write_json(directory/"diagnostic-loop-acceptance.json",receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--run-id",default="diagnostic-loop-"+uuid.uuid4().hex[:12])
    parser.add_argument("--execution-config",type=Path,default=REPO/"configs/romeo.json")
    args = parser.parse_args(argv)
    directory = (args.output or REPO/"results/diagnostic-loop-acceptance"/args.run_id).resolve()
    if (directory/"allocation-intent.json").exists():
        raise ValueError("Use a fresh acceptance directory; reconcile the recorded allocation intent")
    if args.run:
        receipt = run_acceptance(directory,args.run_id,args.execution_config)
        print(json.dumps({"ok":receipt["ok"],"status":receipt["status"],"job_id":receipt["job_id"],
                          "receipt":str(directory/"diagnostic-loop-acceptance.json")},indent=2))
        return 0 if receipt["ok"] else 1
    prepare(directory,args.run_id,args.execution_config)
    print(json.dumps({"submitted":0,"model_calls":0,"plan":str(directory/"plan.json")},indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
