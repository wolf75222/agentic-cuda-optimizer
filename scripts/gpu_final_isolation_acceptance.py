#!/usr/bin/env python3
"""Scoped AC-19 negative holdout and AC-14 concurrent-worker acceptance.

Default prepares a CPU plan only. --run owns one fresh allocation and calls
no model. Deliberately incorrect/racy controls never enter the §16 cohorts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from optimizer_agent.campaign import (
    CampaignConfig,Candidate,Evaluator,LaunchRule,SlurmBackend,SlurmConfig,
    WorkloadContract,WorkloadSuite,device_identity,write_json,
)
from scripts.gpu_autotuning_acceptance import ReceiptRunner, candidate_receipt


CONTROL_SOURCE='''// AC19 declared negative control: intentionally wrong ONLY for final n=96.
extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n) {
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<n && col<n) {
        if(n==96) { c[row*n+col]=0.0f; return; }
        float sum=0.0f;
        for(int k=0;k<n;k++) sum+=a[row*n+k]*b[k*n+col];
        c[row*n+col]=sum;
    }
}'''

SHARED_RACE_SOURCE='''// Optional declared negative control: in-bounds shared-memory WAW race.
extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n) {
    __shared__ volatile int scratch[1];
    scratch[0]=(int)threadIdx.x;
    __syncthreads();
    if(threadIdx.x==0) c[0]=(float)scratch[0];
}'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def provenance(execution_config):
    paths=[Path(__file__),REPO/"optimizer_agent/evaluation.py",REPO/"optimizer_agent/benchmark_workloads.py",
        REPO/"optimizer_agent/campaign.py",REPO/"optimizer_agent/campaign_config.py",
        REPO/"optimizer_agent/execution_backend.py",REPO/"optimizer_agent/remote_worker.py",
        REPO/"optimizer_agent/use_harness.py",REPO/"scripts/gpu_autotuning_acceptance.py",
        REPO/"examples/naive_matmul.cu",Path(execution_config).resolve()]
    paths += [p for p in (REPO/"cuda_test_harness").rglob("*") if p.is_file()
        and p.suffix in (".cpp",".cu",".h",".hpp",".cuh",".txt")
        and not any(part.startswith(("build","cmake-build")) for part in p.relative_to(REPO).parts)]
    return {"HEAD":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "dirty_state":subprocess.check_output(["git","status","--short"],cwd=REPO,text=True).splitlines(),
        "sources":{str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p):sha256(p.read_bytes()) for p in paths}}


def prepare(directory,run_id,*,runner=None,properties=None,shared_race=False):
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    config=CampaignConfig(run_id=run_id,experimental_mode="test",research_required=False,
        web_search_required=False,workload="matmul",search_sizes=(17,63,64),final_sizes=(96,),
        benchmark_series=2,confirmation_batches=2,launches_per_series=5,warmup=1,
        required_sanitizers=(),profile=False)
    suite=WorkloadSuite("matmul",directory/"inputs",config)
    cases=[case for case in suite.cases() if case.metadata["n"] in config.search_sizes]
    baseline=suite.baseline(REPO/"examples/naive_matmul.cu")
    source=directory/"declared-ac19-negative.cu"
    source.write_text(CONTROL_SOURCE)
    control=Candidate("declared-ac19-negative",source,"matmul",LaunchRule(kind="square_2d",block=(16,16,1)))
    identity=sha256(json.dumps({"sizes":[96],"seed":config.seed+1000003,"generator":"WorkloadSuite float32 GEMM v1"},sort_keys=True).encode())
    contract=WorkloadContract(run_id,1,"matmul",suite.signature,
        "Independent NumPy float64 matrix multiplication with float32 outputs",
        benchmark_series=2,launches_per_series=5,warmup=1,confirmation_batches=2,allowed_precision="float32",
        metadata_json=json.dumps({"target_environment":device_identity(properties or {}),"precision":"float32",
            "scope":"Explicit functional negative controls; excluded from quality cohorts; no performance gain claim",
            "final_generator_identity":identity},sort_keys=True))
    evaluator=Evaluator(runner,contract,suite.oracle,{"short":cases[:1],"search":cases,"performance":cases},
        final_suite_factory=lambda:suite.cases(final=True),final_suite_identity=identity)
    plan={"schema_version":1,"kind":"final_and_isolation_acceptance","run_id":run_id,"submitted":False,
        "model_calls":0,"generator_calls":0,"tree_tell_calls":0,"promotion_calls":0,
        "ac19":{"declared_negative_control":True,"correct_search_shapes":[17,63,64],"intentionally_wrong_final_shape":96,
            "expected_status":"final_validation_failed","candidate":candidate_receipt(control)},
        "ac14":{"concurrent_requests":["benchmark","profile"],"same_worker_and_uuid":True,
            "benchmark_request_protocol":{"warmup":1,"iterations":5,"series":1},
            "ordering":"Unspecified queue order; disjoint server reservation intervals required",
            "serialization_clock":"time.monotonic_ns from the same serve PID/job/session; started and response markers must agree",
            "wall_clock_scope":"UTC and filesystem mtime are diagnostic only; never compared for serialization",
            "profile_question":"Can the same GPU serialize a benchmark and a real Nsight Compute request?",
            "profile_metrics":["gpu__time_duration.sum"],"denied_counters_are_successful_profile":False},
        "shared_race_requested":shared_race,"shared_race":{"source_sha256":sha256(SHARED_RACE_SOURCE.encode()),
            "scope":"Optional in-bounds shared WAW control; no illegal addresses; race finding requires actual tool evidence",
            "proof_status":"not_executed","limitations":"Compiler/tool behavior can differ; no racecheck proof is claimed without a recorded hazard"},
        "scope":"Functional AC19/AC14 acceptance only; not an optimizer/vanilla quality result",
        "contract_hash":evaluator.contract_hash,"helper_sha256":sha256(Path(__file__).read_bytes())}
    write_json(directory/"plan.json",plan)
    return evaluator,baseline,control,cases,plan


def check_negative_final(evaluator,baseline,control,recording):
    before_hash=control.artifact_hash
    search=evaluator.validate(control,"search")
    before_calls=len(recording.calls)
    final=evaluator.finalize(control,baseline)
    final_calls=recording.calls[before_calls:]
    result={"search":search,"final":final,"source_hash_before":control._source_hash,
        "source_hash_after":sha256(control.source.read_bytes()),"artifact_hash_before":before_hash,
        "artifact_hash_after":control.artifact_hash,"final_harness_requests":len(final_calls),
        "final_benchmark_requests":sum(row["request"]["mode"] == "benchmark" for row in final_calls),
        "model_calls":0,"generator_calls":0,"tree_tell_calls":0,"promotion_calls":0,
        "declared_negative_control":True,"quality_cohort_eligible":False}
    result["ok"]=bool(search.get("ok") and not final.get("ok") and final.get("status") == "final_validation_failed"
        and final.get("validation",{}).get("status") == "numerical_error"
        and result["artifact_hash_before"] == result["artifact_hash_after"]
        and result["source_hash_before"] == result["source_hash_after"] and result["final_benchmark_requests"] == 0)
    return result


def observe_remote_requests(backend,identities):
    """Read matching worker monotonic markers; filesystem time is diagnostic."""
    script='''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out={}
for identity in sys.argv[2:]:
 p=root/'requests'/identity
 started=json.loads((p/'started.json').read_text())
 raw=(p/'response.json').read_bytes(); response=json.loads(raw)
 out[identity]={'started':started,'worker_reservation':response.get('worker_reservation',{}),
  'diagnostic_started_at_unix_ns':int(started['time']*1000000000),
  'diagnostic_response_mtime_unix_ns':(p/'response.json').stat().st_mtime_ns,
  'response_job_id':str(response['slurm_job_id']),
  'request_fingerprint':response['request_fingerprint'],'response_sha256':hashlib.sha256(raw).hexdigest()}
print(json.dumps(out))
'''
    process=backend.transport.run([backend.config.python,"-c",script,backend.remote_dir,*identities])
    return json.loads(process.stdout)


def report_uuid(report):
    return report.get("environment",{}).get("uuid") or report.get("hardware_fingerprint",{}).get("device_properties",{}).get("uuid")


def check_simultaneous(runner,evaluator,baseline,case,*,observe,run_id):
    validation=evaluator.validate(baseline,"search")
    if not validation.get("ok"):
        return {"ok":False,"error":"Baseline failed independent oracle before concurrency probe","validation":validation}
    target=runner.get_device_properties(0)
    benchmark_request=evaluator._request(baseline,case,benchmark=True)
    profile_request=evaluator._request(baseline,case)
    identities={kind:run_id+"-"+kind+"-"+uuid.uuid4().hex[:12] for kind in ("benchmark","profile")}
    barrier=threading.Barrier(3)
    def submit(kind):
        barrier.wait(timeout=30)
        sent_monotonic=time.monotonic_ns()
        sent=time.time_ns()
        if kind == "benchmark":
            report=runner.run(identities[kind],benchmark_request)
        else:
            report=runner.profile(identities[kind],profile_request,
                question="Can an actual benchmark and Nsight Compute request share one GPU without overlap?",
                metrics=["gpu__time_duration.sum"])
        return {"request_id":identities[kind],"client_submitted_at_unix_ns":sent,
                "client_received_at_unix_ns":time.time_ns(),"client_submitted_monotonic_ns":sent_monotonic,
                "client_received_monotonic_ns":time.monotonic_ns(),
                "request":asdict(benchmark_request if kind == "benchmark" else profile_request),"report":report}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={kind:pool.submit(submit,kind) for kind in identities}
        barrier.wait(timeout=30)
        rows={kind:future.result() for kind,future in futures.items()}
    observations=observe(list(identities.values()))
    intervals=[observations[identities[kind]].get("worker_reservation",{}) for kind in identities]
    def marker_valid(kind):
        row=rows[kind]
        observed=observations[row["request_id"]]
        started=observed.get("started",{})
        marker=observed.get("worker_reservation",{})
        return (marker.get("clock") == "time.monotonic_ns"
            and type(marker.get("start_monotonic_ns")) is int and type(marker.get("end_monotonic_ns")) is int
            and 0 <= marker["start_monotonic_ns"] <= marker["end_monotonic_ns"]
            and type(marker.get("worker_pid")) is int and marker["worker_pid"] > 0
            and isinstance(marker.get("session_id"),str) and bool(marker["session_id"])
            and isinstance(marker.get("job_id"),str) and marker["job_id"].isdigit()
            and type(started.get("monotonic_ns")) is int and type(started.get("worker_pid")) is int
            and started.get("monotonic_ns") == marker["start_monotonic_ns"]
            and started.get("worker_pid") == marker["worker_pid"]
            and started.get("session_id") == marker["session_id"]
            and str(started.get("job_id")) == str(marker["job_id"])
            and str(observed.get("response_job_id")) == str(marker["job_id"])
            and row["report"].get("worker_reservation") == marker)
    valid_intervals=all(marker_valid(kind) for kind in identities)
    same_worker=valid_intervals and len({(row["worker_pid"],row["session_id"],str(row["job_id"])) for row in intervals}) == 1
    ordered=sorted(intervals,key=lambda row:row["start_monotonic_ns"]) if valid_intervals else []
    disjoint=same_worker and ordered[0]["end_monotonic_ns"] <= ordered[1]["start_monotonic_ns"]
    benchmark=rows["benchmark"]["report"]
    profile=rows["profile"]["report"]
    profile_denied=profile.get("error_code") == "ERR_NVGPUCTRPERM"
    profile_supported=bool(profile.get("ok") is True and profile.get("metrics")) or (profile_denied and profile.get("ok") is False)
    oracle=evaluator._compare(case,benchmark,evaluator.oracle(case)) if benchmark.get("ok") else {"ok":False}
    identities_match=bool(target.get("uuid")) and all(report_uuid(row["report"]) == target.get("uuid")
        and str(row["report"].get("slurm_job_id")) == str(observations[row["request_id"]].get("started",{}).get("job_id"))
        and observations[row["request_id"]]["response_job_id"] == str(observations[row["request_id"]].get("started",{}).get("job_id"))
        and bool(row["report"].get("request_fingerprint"))
        and row["report"].get("request_fingerprint") == observations[row["request_id"]]["request_fingerprint"] for row in rows.values())
    overlap=max(row["client_submitted_monotonic_ns"] for row in rows.values()) < min(row["client_received_monotonic_ns"] for row in rows.values())
    checks={"client_request_intervals_overlap":overlap,"server_intervals_present":valid_intervals,
        "same_monotonic_worker":same_worker,"server_execution_serialized":disjoint,"same_worker_and_gpu_uuid":identities_match,
        "benchmark_oracle_and_environment":bool(oracle.get("ok") and evaluator._check_environment(benchmark)),
        "benchmark_readonly_and_output_guards":benchmark.get("guard_readonly_ok") is True and benchmark.get("guard_outputs_poisoned") is True,
        "benchmark_raw_count":len(benchmark.get("latency_samples_ms",[])) == benchmark_request.iterations,
        "profile_success_or_explicit_counter_denial":profile_supported}
    return {"ok":all(checks.values()),"checks":checks,"requests":rows,"server_observations":observations,
        "benchmark_oracle":oracle,"profile_succeeded":bool(profile.get("ok")),"profile_counter_permission_denied":profile_denied,
        "profile_times_used_for_ranking":False,"hardware_identity":device_identity(target),
        "serialization_clock":"Same serve process time.monotonic_ns; started/response PID, job and session agree",
        "filesystem_mtime_used_for_serialization":False,
        "scope":"Concurrent submission and serialized complete worker reservations; diagnostic profiling is distinct from benchmark timing"}


def check_shared_race(runner,evaluator,case,directory):
    path=Path(directory)/"declared-shared-race.cu"
    path.write_text(SHARED_RACE_SOURCE)
    candidate=Candidate("declared-shared-race",path,"matmul",LaunchRule(kind="fixed",grid=(1,1,1),block=(32,1,1)))
    request=evaluator._request(candidate,case)
    run_id="shared-race-"+uuid.uuid4().hex
    report=runner.run_sanitizer(run_id,request,"racecheck")
    logs=[]
    for name in ("stdout.log","stderr.log"):
        log=Path(runner.results_dir)/("run-"+run_id)/name
        if log.exists():
            logs.append(log.read_text(errors="replace"))
    text="\n".join([str(report.get("error","")),*logs])
    hazard=bool(re.search(r"(?:WAW|WAR|RAW)\s+hazard|Race reported|race hazard",text,re.I))
    count=bool(re.search(r"(?:ERROR SUMMARY|RACECHECK SUMMARY):\s*[1-9]\d*",text,re.I))
    detected=hazard and (count or report.get("returncode") == 86)
    return {"ok":bool(detected and not report.get("worker_poisoned")),"candidate":candidate_receipt(candidate),
        "request_id":run_id,"report":report,"hazard_observed":hazard,"race_detected":detected,
        "status":"race_detected" if detected else "not_demonstrated","quality_cohort_eligible":False,
        "scope":"Only an actual recorded racecheck hazard establishes this optional negative control; no correctness or gain certificate"}


def run_acceptance(directory,run_id,execution_config,*,shared_race=False,backend_factory=None):
    directory=Path(directory).resolve()
    if any((directory/name).exists() for name in ("allocation-intent.json","allocation-receipt.json","worker/slurm-session.json")):
        raise ValueError("Use a fresh isolated acceptance directory; previous jobs/evidence are preserved")
    directory.mkdir(parents=True,exist_ok=True)
    backend=(backend_factory or SlurmBackend)(SlurmConfig.load(execution_config),REPO,directory/"worker",
        session_id="final-isolation-"+uuid.uuid4().hex)
    receipt={"kind":"final_and_isolation_acceptance","ok":False,"model_calls":0,"job_id":None,"submitted":None,
             "submission_status":"intent_recorded","session_id":backend.session_id,"remote_dir":backend.remote_dir,
             "config":asdict(backend.config),
             "evidence_kind":"simulated_cpu_test" if getattr(backend,"simulation",False) else "romeo_execution",
             "source_provenance":provenance(execution_config),
             "scope":"Functional acceptance controls excluded from all quality cohorts"}
    intent={"schema_version":1,"kind":"final_isolation_allocation_intent","created_at_unix":time.time(),
        "session_id":backend.session_id,"remote_dir":backend.remote_dir,"config":asdict(backend.config),
        "source_provenance":receipt["source_provenance"],"model_calls":0,
        "instruction":"Reconcile this exact session after uncertain submission; never submit a replacement implicitly"}
    # Exclusive creation closes the concurrent-launch race as well as the
    # lost-ACK window. A partial intent also blocks another submission.
    with (directory/"allocation-intent.json").open("x") as file:
        json.dump(intent,file,indent=2,allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    try:
        receipt["job_id"]=backend.start()
        receipt["submitted"]=True
        receipt["submission_status"]="acknowledged"
        print(json.dumps({"event":"final_isolation_started","job_id":backend.job_id,"session_id":backend.session_id}),flush=True)
        backend.wait_until_ready()
        recording=ReceiptRunner(backend.runner(),directory)
        evaluator,baseline,control,cases,plan=prepare(directory/"controls",run_id,runner=recording,
            properties=recording.get_device_properties(0),shared_race=shared_race)
        receipt["plan"]=plan
        receipt["ac19"]=check_negative_final(evaluator,baseline,control,recording)
        write_json(directory/"ac19.json",receipt["ac19"])
        receipt["ac14"]=check_simultaneous(recording,evaluator,baseline,cases[-1],
            observe=lambda ids:observe_remote_requests(backend,ids),run_id=run_id)
        write_json(directory/"ac14.json",receipt["ac14"])
        if shared_race:
            receipt["shared_race"]=check_shared_race(recording,evaluator,cases[0],directory)
        receipt["usage"]=recording.usage()
        receipt["usage"]["profile_native_reservation_wall_seconds"]=receipt["ac14"].get("requests",{}).get("profile",{}).get("report",{}).get("reservation_wall_seconds")
    except Exception as error:
        receipt.update(error=str(error),traceback=traceback.format_exc())
        if receipt["submission_status"] == "intent_recorded":
            receipt["submission_status"]="uncertain"
    finally:
        if backend.job_id:
            try:
                backend.close()
                receipt["worker_terminal"]=backend.wait_until_finished(timeout=60)
            except Exception as error:
                receipt["cleanup_error"]=str(error)
        ended=receipt.get("worker_terminal",{})
        receipt["source_provenance_after"]=provenance(execution_config)
        unchanged=receipt["source_provenance"]["sources"] == receipt["source_provenance_after"]["sources"]
        receipt["sources_unchanged"]=unchanged
        receipt["ok"]=bool(receipt.get("ac19",{}).get("ok") and receipt.get("ac14",{}).get("ok")
            and (not shared_race or receipt.get("shared_race",{}).get("ok")) and not receipt.get("error")
            and unchanged
            and not receipt.get("cleanup_error") and ended.get("terminal") and ended.get("state") == "COMPLETED"
            and ended.get("exit_code") == "0:0" and str(ended.get("job_id")) == str(backend.job_id))
        write_json(directory/"allocation-receipt.json",receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run-id",default="final-isolation-"+uuid.uuid4().hex[:12])
    parser.add_argument("--execution-config",type=Path,default=REPO/"configs/romeo.json")
    parser.add_argument("--shared-race",action="store_true",help="Optional final in-bounds WAW racecheck negative control")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args(argv)
    if any((args.output/name).exists() for name in ("allocation-intent.json","allocation-receipt.json","worker/slurm-session.json")):
        raise ValueError("Existing GPU evidence/allocation is never overwritten; use a fresh output directory")
    if not args.run:
        prepare(args.output,args.run_id,shared_race=args.shared_race)
        print(json.dumps({"submitted":False,"model_calls":0,"plan":str(args.output.resolve()/"plan.json")}))
        return 0
    result=run_acceptance(args.output,args.run_id,args.execution_config,shared_race=args.shared_race)
    print(json.dumps({"ok":result["ok"],"job_id":result["job_id"],"receipt":str(args.output.resolve()/"allocation-receipt.json")}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
