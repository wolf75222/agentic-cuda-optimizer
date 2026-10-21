#!/usr/bin/env python3
"""Scoped AC-05/06/10 control: real TreeQuest, static CUDA, independent oracle.

Default only writes a local plan. --run owns one fresh Slurm allocation and
never calls a model. Delays are declared test controls, not optimizations or
quality-study samples. A slow parent must be selected naturally by the pinned
AB-MCTS-A policy; a missed selection leaves AC-05 partial. AC-10 specifically
requires aggregate gain above threshold with rejection due to a critical case.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import uuid

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from optimizer_agent.campaign import device_identity, write_json
from optimizer_agent.evaluation import Candidate, EvaluationCase, Evaluator, LaunchRule, WorkloadContract
from optimizer_agent.execution_backend import SlurmBackend, SlurmConfig
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.search_policy import SearchPolicy, verify_treequest_pin, TREEQUEST_REVISION, ELIGIBILITY_EXTENSION_VERSION
from optimizer_agent.agent_workload import OutputComparison
from optimizer_agent.use_harness import BufferArgument, BufferDirection, KernelRequest, ScalarArgument
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent
from scripts.gpu_autotuning_acceptance import ReceiptRunner

SEED = 1729
SELECTION_TRIALS = 16
MAX_REPAIRS = 2
DELAY_CYCLES = {"baseline":1_000_000,"fast":100_000,"slow":250_000,"critical_regression":0}
CRITICAL_SIZE = 16384
CRITICAL_DELAY_CYCLES = 2_000_000
SIGNATURE = 'extern "C" __global__ void copy_control(const float* input, float* output, int n)'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def source(variant):
    delay=(f"n == {CRITICAL_SIZE} ? {CRITICAL_DELAY_CYCLES}ULL : CONTROL_DELAY"
           if variant=="critical_regression" else "CONTROL_DELAY")
    return f'''// Declared static acceptance control: {variant}; not an optimizer sample.
#define CONTROL_DELAY {DELAY_CYCLES[variant]}ULL
{SIGNATURE} {{
    // clock64 plus volatile assembly keeps the intentionally timed control loop.
    if (threadIdx.x == 0) {{
        unsigned long long delay_cycles = {delay};
        unsigned long long start = clock64();
        while (clock64() - start < delay_cycles) {{ asm volatile("" ::: "memory"); }}
    }}
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i];
}}
'''


def state(candidate,**extra):
    return {"candidate_id":candidate.candidate_id,"source":candidate.source.read_text(),
        "kernel":candidate.kernel,"launch_rule":asdict(candidate.launch_rule),
        "parameters":deepcopy(candidate.parameters),"artifact_hash":candidate.artifact_hash,**extra}


def materialize(directory,name,text,parameters):
    path=Path(directory)/"candidates"/(name+".cu")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)
    return Candidate(name,path,"copy_control",LaunchRule(block=(256,1,1)),deepcopy(parameters))


def static_generator(directory,trial,name,*,variant=None,invalid=False):
    """An explicit test generator, receiving the unmodified selected parent.

    Seed controls use declared static sources. Later expansions use the selected
    parent's source/parameters verbatim, with a provenance comment; no incumbent
    is ever substituted. Invalid repairs deliberately contain a compiler symbol.
    """
    parent=deepcopy(trial.parent_state)
    parameters=(dict(control_variant=variant,delay_cycles=DELAY_CYCLES[variant])
                if variant else deepcopy(parent["parameters"]))
    text=source(variant) if variant else parent["source"]
    if invalid:
        if "output[i] = input[i]" not in text:
            raise ValueError("The selected static source is not the declared copy control")
        text=text.replace("output[i] = input[i]","output[i] = CONTROL_INVALID_SYMBOL")
    text+="\n// Static acceptance expansion: "+name+"\n"
    candidate=materialize(directory,name,text,parameters)
    receipt={"trial_id":trial.trial_id,"selected_parent":parent,"generator_input":deepcopy(parent),
        "generator_input_sha256":digest(parent),"candidate":state(candidate),
        "static_acceptance_control":True,"model_calls":0,"variant_override":variant,
        "invalid_compilation_control":invalid,"excluded_from_quality_cohorts":True}
    write_json(Path(directory)/"generation-receipts"/(name+".json"),receipt)
    return candidate,receipt


def prepare(directory,run_id,*,runner=None,properties=None,selection_trials=SELECTION_TRIALS):
    directory=Path(directory).resolve(); directory.mkdir(parents=True,exist_ok=True)
    if type(selection_trials) is not int or selection_trials < 1:
        raise ValueError("The declared number of selection trials must be positive")
    baseline=materialize(directory,"baseline",source("baseline"),
        dict(control_variant="baseline",delay_cycles=DELAY_CYCLES["baseline"]))
    cases=[]
    for n in (1,17,63,CRITICAL_SIZE,65536):
        inputs=directory/"inputs"/str(n); inputs.mkdir(parents=True,exist_ok=True)
        values=np.random.default_rng(SEED+n).uniform(-1,1,n).astype("<f4")
        values.tofile(inputs/"input.bin")
        request=KernelRequest(str(baseline.source),"copy_control","run",0,[1,1,1],[256,1,1],0,1,5,
            [BufferArgument(values.nbytes,str(inputs/"input.bin"),BufferDirection.Input),
             BufferArgument(values.nbytes,None,BufferDirection.Output),ScalarArgument("int32",n)])
        cases.append(EvaluationCase("copy-"+str(n),request,[OutputComparison(0,"float32",0,0)],{"n":n},SEED+n,
            .05 if n==CRITICAL_SIZE else .95,n==CRITICAL_SIZE))
    contract=WorkloadContract(run_id,1,"copy_control",SIGNATURE,
        "Independent NumPy byte-for-byte float32 input copy; outputs must be fully written and inputs read-only",
        benchmark_series=3,launches_per_series=5,warmup=1,confirmation_batches=4,
        min_relative_gain=.01,max_critical_regression=.05,allowed_precision="float32",
        metadata_json=json.dumps({"target_environment":device_identity(properties or {}),
            "static_acceptance_controls":True,"excluded_from_quality_cohorts":True,
            "timed_scope":"one CUDA kernel; warm inputs reset before each launch; transfers/compilation excluded",
            "control_delay_cycles":DELAY_CYCLES,"critical_delay_cycles":CRITICAL_DELAY_CYCLES,
            "selection_seed":SEED,"selection_trials":selection_trials}))
    evaluator=Evaluator(runner,contract,lambda case:[np.fromfile(case.request.arguments[0].input_file,dtype="<f4")],
        {"short":cases[:2],"search":cases,"performance":cases[-2:]})
    plan={"schema_version":1,"kind":"targeted_search_policy_acceptance","run_id":run_id,
        "submitted":False,"model_calls":0,"model_tokens":0,"selection_seed":SEED,
        "selection_trials":selection_trials,"max_repairs":MAX_REPAIRS,"post_closure_batch_size":4,
        "actions":["static_expand","static_repair"],"contract":asdict(contract),"contract_hash":evaluator.contract_hash,
        "baseline":state(baseline),"static_sources":{key:source(key) for key in DELAY_CYCLES},
        "inputs_sha256":{str(Path(path).relative_to(directory)):value for path,value in evaluator._input_hashes.items()},
        "performance_cases":[{"case_id":case.case_id,"weight":case.weight,"critical":case.critical,"metadata":case.metadata}
            for case in cases[-2:]],"compiler_options":cases[-1].request.compiler_options,
        "planned_criteria":["AC-05","AC-06","AC-10"],
        "critical_control":{"case_id":"copy-"+str(CRITICAL_SIZE),"weight":.05,"delay_cycles":CRITICAL_DELAY_CYCLES,
            "principal_case_id":"copy-65536","principal_weight":.95,
            "required":"Aggregate lower confidence bound above promotion threshold, but critical regression rejects promotion"},
        "selection_policy":"Natural pinned AB-MCTS-A; no forced parent, mask or patched score to acquire AC-05",
        "scope":"Functional controls only; no model, optimizer gain, final certification or section-16 quality claim"}
    write_json(directory/"plan.json",plan)
    return evaluator,baseline,plan


def exercise(directory,evaluator,baseline,plan,*,execution_kind="cpu_simulated"):
    """Exercise the real durable policy. CPU injections never acquire GPU ACs."""
    directory=Path(directory)
    verify_treequest_pin()
    store=ExperimentStore(directory/"experiments.sqlite")
    measurements={}
    def measure(candidate):
        value=evaluator.benchmark(candidate)
        write_json(directory/"measurements"/(value["evaluation_id"]+".json"),value)
        if not value.get("ok"):
            raise RuntimeError("Valid static control failed independent CUDA validation/measurement")
        measurements[candidate.candidate_id]=value
        return value
    baseline_measurement=evaluator.benchmark(baseline)
    if not baseline_measurement.get("ok"):
        raise RuntimeError("Fresh static baseline failed")
    write_json(directory/"baseline-measurement.json",baseline_measurement)
    latencies={name:row["latency_ms"] for name,row in baseline_measurement["cases"].items()}
    options=dict(store=store,run_id=plan["run_id"],baseline_state=state(baseline),baseline_latencies=latencies,
        weights={name:row["weight"] for name,row in baseline_measurement["cases"].items()},
        actions=plan["actions"],contract_fingerprint=evaluator.contract_hash,
        environment_fingerprint=digest(device_identity(evaluator._properties)),seed=plan["selection_seed"])
    policy=SearchPolicy(**options)
    store.add_evaluation(plan["run_id"],baseline_measurement)
    measurements["baseline"]=baseline_measurement
    def archive(candidate,trial,valid=True):
        value=state(candidate,tree_parent_id=trial.parent_state["candidate_id"],
            source_parent_id=trial.parent_state["candidate_id"],trial_id=trial.trial_id,valid=valid)
        store.archive_candidate(plan["run_id"],value)
        return value
    def tell(candidate,trial,measurement,confirmation=None):
        policy.tell_once(trial.trial_id,{"state":archive(candidate,trial),"evaluation_id":measurement["evaluation_id"],
            "valid":True,"purpose":"tree_reward","fresh":True,
            "measurement":measurement,"confirmation":confirmation,
            "latencies":{key:row["latency_ms"] for key,row in measurement["cases"].items()}})
    first=policy.ask_batch(2)
    controls={}
    for trial,variant in zip(first,("fast","slow")):
        controls[variant],_=static_generator(directory,trial,variant,variant=variant)
        measure(controls[variant])
    confirmation=evaluator.confirm(controls["fast"],baseline)
    store.add_evaluation(plan["run_id"],confirmation)
    slow_comparison=evaluator.confirm(controls["slow"],controls["fast"])
    store.add_evaluation(plan["run_id"],slow_comparison)
    # Measurements and the declared paired decisions are durable before either
    # one-shot tree reward. Confirmation samples never replace reward samples.
    for trial,variant in zip(first,("fast","slow")):
        tell(controls[variant],trial,measurements[variant],confirmation if variant=="fast" else slow_comparison)
    promoted=confirmation.get("status")=="promoted" and store.promote(plan["run_id"],"fast",0,
        {"status":"improved","evidence":[confirmation["evaluation_id"]]})
    write_json(directory/"initial-confirmations.json",{"fast_vs_baseline":confirmation,"slow_vs_fast":slow_comparison})
    slower_proved=bool(slow_comparison.get("ok") and slow_comparison.get("interval",{}).get("upper",float("inf")) < 1)
    selections=[]; selected_slow=False
    def weighted_latency(measurement):
        rows=measurement["cases"]
        return math.exp(sum(row["weight"]*math.log(row["latency_ms"]) for row in rows.values())/sum(row["weight"] for row in rows.values()))
    for index in range(plan["selection_trials"]):
        trial=policy.ask()
        policy.assert_eligible(trial.trial_id)
        parent=trial.parent_state
        best=min(measurements,key=lambda key:weighted_latency(measurements[key]))
        parent_measurement=measurements[parent["candidate_id"]]
        parent_ms=weighted_latency(parent_measurement)
        best_ms=weighted_latency(measurements[best])
        slow=bool(slower_proved and promoted and parent["parameters"].get("control_variant")=="slow" and parent_ms>best_ms)
        selections.append({"trial_id":trial.trial_id,"parent":parent,"best_measured_candidate_id":best,
            "parent_latency_ms":parent_ms,"best_latency_ms":best_ms,"selected_slow_measured_parent":slow})
        if slow or index==plan["selection_trials"]-1:
            selected_slow=slow
            break
        candidate,_=static_generator(directory,trial,"expansion-"+str(index))
        tell(candidate,trial,measure(candidate))
    attempts=[]; generator_receipts=[]
    for repair in range(MAX_REPAIRS+1):
        candidate,generation=static_generator(directory,trial,"invalid-repair-"+str(repair),invalid=True)
        archive(candidate,trial,False)
        attempt=store.new_attempt(plan["run_id"],trial.trial_id,{"repair":repair,"candidate_id":candidate.candidate_id})
        validation=evaluator.validate(candidate)
        store.finish_attempt(attempt,validation["status"],validation)
        if repair < MAX_REPAIRS:
            store.add_evaluation(plan["run_id"],validation)
        attempts.append(validation); generator_receipts.append(generation)
        write_json(directory/"repairs"/(str(repair)+".json"),validation)
        if validation.get("status")!="compilation_error":
            raise RuntimeError("A declared invalid repair did not produce an authenticated compilation error")
    policy.tell_once(trial.trial_id,{"state":state(candidate),"evaluation_id":attempts[-1]["evaluation_id"],
        "measurement":attempts[-1],"valid":False})
    closed_parent=trial.parent_state["candidate_id"]
    def snapshot_hash():
        with store.connection() as db:
            return hashlib.sha256(db.execute("SELECT snapshot FROM campaigns WHERE run_id=?",(plan["run_id"],)).fetchone()[0]).hexdigest()
    snapshot_before=snapshot_hash()
    decision=policy.close("parent_action",(closed_parent,trial.action),
        "Declared static broken transformation: initial compilation and two local repairs exhausted",
        evidence=[value["evaluation_id"] for value in attempts],author="static_acceptance_control",
        counters={"repairs":MAX_REPAIRS,"compilation_failures":len(attempts)})
    snapshot_closed=snapshot_hash()
    # Restore the very same SQLite/tree/RNG to prove the closure survives resume.
    policy=SearchPolicy(**options)
    snapshot_restored=snapshot_hash()
    following=policy.ask_batch(plan["post_closure_batch_size"])
    forbidden=[value for value in following if (value.parent_state["candidate_id"],value.action)==(closed_parent,trial.action)]
    independent=None
    for pending in following:
        policy.assert_eligible(pending.trial_id)
        if independent is None and pending.parent_state["candidate_id"]!=closed_parent:
            next_candidate,_=static_generator(directory,pending,"continued-branch")
            measurement=measure(next_candidate)
            tell(next_candidate,pending,measurement)
            independent={"trial_id":pending.trial_id,"selected_parent":pending.parent_state,
                "measurement":measurement,"candidate":state(next_candidate)}
        else:
            policy.suspend_or_cancel(pending.trial_id,"The fixed acceptance batch records selection only; no GPU execution requested",consumption={"trials":1})
    archived=store.get_candidate_source(closed_parent,plan["run_id"])
    incumbent_before=store.incumbent(plan["run_id"])
    critical=materialize(directory,"critical-regression",source("critical_regression"),
        dict(control_variant="critical_regression",delay_cycles=0,critical_delay_cycles=CRITICAL_DELAY_CYCLES,critical_size=CRITICAL_SIZE))
    store.archive_candidate(plan["run_id"],state(critical,source_parent_id="fast",static_acceptance_control=True))
    critical_confirmation=evaluator.confirm(critical,controls["fast"])
    store.add_evaluation(plan["run_id"],critical_confirmation)
    critical_decision={"candidate_id":critical.candidate_id,"confirmation_id":critical_confirmation["evaluation_id"],
        "status":critical_confirmation["status"],"critical_cases_ok":critical_confirmation.get("critical_cases_ok"),
        "aggregate_interval":critical_confirmation.get("interval"),"incumbent_before":incumbent_before}
    if critical_confirmation.get("status")=="promoted":
        store.promote(plan["run_id"],critical.candidate_id,incumbent_before["version"],
            {"status":"improved","evidence":[critical_confirmation["evaluation_id"]]})
    store.event(plan["run_id"],"acceptance_promotion_decision",critical_decision)
    incumbent=store.incumbent(plan["run_id"])
    checks={"initial_controls_really_validated_and_measured":all(measurements[key]["ok"] for key in ("fast","slow")),
        "fast_incumbent_freshly_confirmed":bool(promoted),"slow_control_paired_interval_below_one":slower_proved,
        "slow_parent_naturally_selected":selected_slow,
        "selected_source_and_parameters_received_exactly":all(value["selected_parent"]==value["generator_input"]==trial.parent_state for value in generator_receipts),
        "initial_plus_two_repairs_really_failed_compilation":len(attempts)==3 and all(value["status"]=="compilation_error" for value in attempts),
        "forbidden_pair_absent_from_new_batch":not forbidden,
        "closure_restored_from_same_durable_snapshot":snapshot_closed==snapshot_restored,
        "invalid_candidate_never_selected_as_parent":all(value.parent_state.get("valid") is not False for value in following),
        "other_parent_branch_really_benchmarked":independent is not None,
        "closed_parent_source_and_results_consultable":archived["source"]==trial.parent_state["source"] and bool(archived["evaluations"]),
        "critical_control_aggregate_lower_bound_above_threshold":critical_confirmation.get("interval",{}).get("lower",0)>1+evaluator.contract.min_relative_gain,
        "critical_regression_alone_rejects_promotion":critical_confirmation.get("status")=="not_improved" and critical_confirmation.get("critical_cases_ok") is False,
        "confirmed_incumbent_preserved":incumbent==incumbent_before and incumbent["candidate_id"]=="fast" and incumbent["version"]==1}
    criteria={"AC-05":all(checks[key] for key in ("fast_incumbent_freshly_confirmed","slow_control_paired_interval_below_one",
        "slow_parent_naturally_selected","selected_source_and_parameters_received_exactly")),
        "AC-06":all(checks[key] for key in ("initial_plus_two_repairs_really_failed_compilation","forbidden_pair_absent_from_new_batch","closure_restored_from_same_durable_snapshot",
        "invalid_candidate_never_selected_as_parent","other_parent_branch_really_benchmarked","closed_parent_source_and_results_consultable")),
        "AC-10":all(checks[key] for key in ("critical_control_aggregate_lower_bound_above_threshold",
            "critical_regression_alone_rejects_promotion","confirmed_incumbent_preserved"))}
    result={"kind":plan["kind"],"run_id":plan["run_id"],"execution_kind":execution_kind,
        "ok":False,"functional_checks_passed":all(criteria.values()),"criteria_checks":criteria,"checks":checks,
        "status":"policy_controls_observed" if execution_kind=="romeo_gpu" else "cpu_simulated_checks_only",
        "model_calls":0,"model_tokens":0,"static_acceptance_controls":True,"excluded_from_quality_cohorts":True,
        "selection_seed":plan["selection_seed"],"selections":selections,"generator_receipts":generator_receipts,
        "initial_confirmations":{"fast_vs_baseline":confirmation,"slow_vs_fast":slow_comparison},
        "repair_results":attempts,"branch_decision":decision.to_dict(),"snapshot_before_closure_sha256":snapshot_before,
        "snapshot_after_closure_sha256":snapshot_closed,"restored_snapshot_sha256":snapshot_restored,
        "critical_control":{"candidate":state(critical),"confirmation":critical_confirmation,"decision":critical_decision,
            "incumbent_after":incumbent},
        "post_closure_batch":[asdict(value) for value in following],"continued_branch":independent,"incumbent":incumbent,
        "contract_hash":evaluator.contract_hash,"contract":asdict(evaluator.contract),"hardware_identity":device_identity(evaluator._properties),
        "treequest":{"revision":TREEQUEST_REVISION,"eligibility_extension":ELIGIBILITY_EXTENSION_VERSION,"source_pin_verified":True},
        "scope":plan["scope"]}
    write_json(directory/"policy-controls.json",result)
    write_json(directory/"events.json",store.events(plan["run_id"]))
    return result


def run_acceptance(directory,run_id,execution_config,*,selection_trials=SELECTION_TRIALS):
    directory=Path(directory).resolve()
    if any((directory/name).exists() for name in ("allocation-intent.json","worker/slurm-session.json","experiments.sqlite")):
        raise ValueError("Use a fresh directory; existing allocations/evidence must be reconciled explicitly")
    directory.mkdir(parents=True,exist_ok=True)
    backend=SlurmBackend(SlurmConfig.load(execution_config),REPO,directory/"worker",session_id="ac0506-"+uuid.uuid4().hex)
    receipt={"kind":"targeted_search_policy_acceptance","run_id":run_id,"ok":False,"status":"incomplete",
        "model_calls":0,"model_tokens":0,"job_id":None,"session_id":backend.session_id,"remote_dir":backend.remote_dir}
    receipt["source_provenance"]=controller_provenance(REPO,__file__,execution_config)
    receipt["source_provenance"]["sources"]["scripts/gpu_autotuning_acceptance.py"]=hashlib.sha256(
        (REPO/"scripts/gpu_autotuning_acceptance.py").read_bytes()).hexdigest()
    receipt["allocation_intent"]=persist_allocation_intent(directory,backend,receipt["source_provenance"])
    started,recording=time.monotonic(),None
    try:
        receipt["job_id"]=backend.start(); backend.wait_until_ready()
        recording=ReceiptRunner(backend.runner(),directory)
        properties=recording.get_device_properties(0)
        receipt["hardware"]=backend.hardware
        evaluator,baseline,plan=prepare(directory,run_id,runner=recording,properties=properties,selection_trials=selection_trials)
        receipt.update(exercise(directory,evaluator,baseline,plan,execution_kind="romeo_gpu"))
        if callable(getattr(recording,"require_complete",None)):
            recording.require_complete()
    except Exception as error:
        receipt.update(error=str(error),traceback=traceback.format_exc())
    finally:
        if recording:
            receipt["usage"]=recording.usage()
            reports=[value.get("report",{}) for value in recording.calls]
            receipt["all_requests_owned_job"]=bool(reports) and all(str(value.get("slurm_job_id"))==str(backend.job_id) for value in reports)
        if backend.job_id:
            try:
                receipt["job_id"]=str(backend.job_id)
                backend.close(); terminal=backend.wait_until_finished(timeout=60)
                if not terminal.get("terminal"):
                    backend.close(cancel=True); terminal=backend.wait_until_finished(timeout=60)
                receipt["terminal"]=terminal
            except Exception as error:
                receipt["cleanup_error"]=str(error)
        terminal=receipt.get("terminal",{})
        receipt["helper_wall_seconds"]=time.monotonic()-started
        receipt["ok"]=bool(receipt.get("functional_checks_passed") and receipt.get("all_requests_owned_job")
            and not receipt.get("error") and not receipt.get("cleanup_error")
            and terminal.get("state")=="COMPLETED" and terminal.get("exit_code")=="0:0"
            and str(terminal.get("job_id"))==str(receipt.get("job_id")) and terminal.get("terminal") is True)
        receipt["status"]="search_policy_acceptance_passed" if receipt["ok"] else "partial_or_incomplete"
        write_json(directory/"search-policy-acceptance.json",receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",action="store_true",help="Explicitly submit this independent ROMEO control; default is local plan only")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--run-id",default="ac0506-"+uuid.uuid4().hex[:12])
    parser.add_argument("--execution-config",type=Path,default=REPO/"configs/romeo.json")
    parser.add_argument("--selection-trials",type=int,default=SELECTION_TRIALS)
    args=parser.parse_args(argv)
    directory=(args.output or REPO/"results/gpu-search-policy-acceptance"/args.run_id).resolve()
    if args.run:
        result=run_acceptance(directory,args.run_id,args.execution_config,selection_trials=args.selection_trials)
        print(json.dumps({key:result.get(key) for key in ("ok","status","job_id","criteria_checks")},indent=2))
        return 0 if result["ok"] else 1
    if (directory/"allocation-intent.json").exists() or (directory/"experiments.sqlite").exists():
        raise ValueError("An existing execution directory cannot be replanned")
    _,_,plan=prepare(directory,args.run_id,selection_trials=args.selection_trials)
    print(json.dumps({"status":"local_plan_only","submitted":False,"model_calls":0,"plan":str(directory/"plan.json"),
        "selection_seed":plan["selection_seed"],"selection_trials":plan["selection_trials"]},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
