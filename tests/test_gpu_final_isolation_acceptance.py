"""CPU simulations verify control flow only; these are not CUDA/GPU proofs."""
import hashlib
import copy
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np
import pytest

from scripts import gpu_final_isolation_acceptance as acceptance
from scripts.gpu_autotuning_acceptance import ReceiptRunner


class SimulatedRunner:
    environment="simulated_cpu_test"
    def __init__(self,directory):
        self.results_dir=Path(directory)
        self.results_dir.mkdir(parents=True,exist_ok=True)
        self.lock=threading.Lock()
        self.observations={}
        self.session_id="simulated-owned-session"
        self.profile_worker_pid=7654
    def get_device_properties(self,device):
        return {"uuid":"SIMULATED-CPU-NO-GPU","architecture":"sm_90","driver_version":12000,
            "nvrtc_version":"simulated","max_threads_per_block":1024,"max_shared_memory_per_block_bytes":49152}
    def _begin(self,run_id):
        start=time.monotonic_ns()
        pid=self.profile_worker_pid if "-profile-" in run_id else 7654
        self.observations[run_id]={"started":{"time":time.time(),"monotonic_ns":start,"worker_pid":pid,
            "session_id":self.session_id,"job_id":"123456"},
            "worker_reservation":{"clock":"time.monotonic_ns","worker_pid":pid,"session_id":self.session_id,
                "job_id":"123456","start_monotonic_ns":start,"end_monotonic_ns":None},
            "diagnostic_started_at_unix_ns":time.time_ns(),"response_job_id":"123456",
            "request_fingerprint":hashlib.sha256(run_id.encode()).hexdigest()}
    def _end(self,run_id):
        self.observations[run_id]["worker_reservation"]["end_monotonic_ns"]=time.monotonic_ns()
        self.observations[run_id]["diagnostic_response_mtime_unix_ns"]=time.time_ns()
    def run(self,run_id,request):
        with self.lock:
            self._begin(run_id)
            n=int(request.arguments[-1].value)
            a,b=(np.fromfile(request.arguments[i].input_file,dtype="<f4").astype(np.float64).reshape(n,n) for i in (0,1))
            value=(a@b).astype("<f4").ravel()
            if "declared-ac19-negative" in Path(request.source).name and n == 96:
                value[:]=0
            path=self.results_dir/(run_id+".bin")
            value.tofile(path)
            time.sleep(.005)
            self._end(run_id)
            return {"ok":True,"stage":"complete","environment":self.get_device_properties(0),
                "evidence_kind":"simulated_cpu_test","guard_readonly_ok":True,"guard_outputs_poisoned":True,
                "outputs":[{"path":str(path),"bytes":value.nbytes}],"latency_samples_ms":[.1]*request.iterations if request.mode == "benchmark" else [],
                "worker_reservation":self.observations[run_id]["worker_reservation"],
                "slurm_job_id":"123456","request_fingerprint":self.observations[run_id]["request_fingerprint"],"worker_reusable":True}
    def profile(self,run_id,request,*,question,metrics):
        assert question and metrics == ["gpu__time_duration.sum"]
        with self.lock:
            self._begin(run_id)
            time.sleep(.005)
            self._end(run_id)
            return {"ok":False,"error_code":"ERR_NVGPUCTRPERM","error":"simulated counter permission denial",
                "evidence_kind":"simulated_cpu_test","environment":self.get_device_properties(0),
                "worker_reservation":self.observations[run_id]["worker_reservation"],
                "slurm_job_id":"123456","request_fingerprint":self.observations[run_id]["request_fingerprint"]}


def prepared(tmp_path):
    runner=SimulatedRunner(tmp_path/"simulated-worker")
    recording=ReceiptRunner(runner,tmp_path/"receipts")
    evaluator,baseline,control,cases,plan=acceptance.prepare(tmp_path/"prepared","unit-control",
        runner=recording,properties=runner.get_device_properties(0))
    return runner,recording,evaluator,baseline,control,cases,plan


def test_cpu_plan_is_explicit_negative_control_and_never_submits(tmp_path,monkeypatch):
    monkeypatch.setattr(acceptance,"SlurmBackend",lambda *args,**kwargs:pytest.fail("Planning cannot submit a job"))
    assert acceptance.main(["--output",str(tmp_path/"plan")]) == 0
    plan=json.loads((tmp_path/"plan/plan.json").read_text())
    assert plan["submitted"] is False and plan["model_calls"] == 0
    assert plan["ac19"]["correct_search_shapes"] == [17,63,64]
    assert plan["ac19"]["intentionally_wrong_final_shape"] == 96
    assert plan["shared_race"]["proof_status"] == "not_executed"
    assert not (tmp_path/"plan/inputs/final-96").exists()


def test_actual_evaluator_rejects_static_final_control_using_simulated_cpu_outputs(tmp_path):
    _,recording,evaluator,baseline,control,cases,_=prepared(tmp_path)
    result=acceptance.check_negative_final(evaluator,baseline,control,recording)
    assert result["ok"] is True
    assert result["final"]["status"] == "final_validation_failed"
    assert result["final"]["validation"]["status"] == "numerical_error"
    assert result["artifact_hash_before"] == result["artifact_hash_after"]
    assert result["source_hash_before"] == result["source_hash_after"]
    assert result["final_benchmark_requests"] == 0
    assert result["generator_calls"] == result["tree_tell_calls"] == result["promotion_calls"] == 0
    assert all(row["report"]["evidence_kind"] == "simulated_cpu_test" for row in recording.calls)


def test_simulated_serial_worker_proves_controller_checks_but_not_actual_gpu_isolation(tmp_path):
    runner,recording,evaluator,baseline,_,cases,_=prepared(tmp_path)
    proof=acceptance.check_simultaneous(recording,evaluator,baseline,cases[-1],
        observe=lambda ids:{key:runner.observations[key] for key in ids},run_id="simulated-overlap")
    assert proof["ok"] and proof["checks"]["server_execution_serialized"]
    assert proof["checks"]["client_request_intervals_overlap"]
    assert proof["checks"]["same_monotonic_worker"]
    assert proof["filesystem_mtime_used_for_serialization"] is False
    assert proof["profile_succeeded"] is False and proof["profile_counter_permission_denied"] is True
    assert proof["profile_times_used_for_ranking"] is False
    assert all(row["report"]["evidence_kind"] == "simulated_cpu_test" for row in proof["requests"].values())


def test_overlap_or_identity_drift_cannot_be_reported_as_success(tmp_path):
    runner,recording,evaluator,baseline,_,cases,_=prepared(tmp_path)
    def wrong_intervals(ids):
        rows={key:runner.observations[key] for key in ids}
        start=min(row["worker_reservation"]["start_monotonic_ns"] for row in rows.values())
        end=max(row["worker_reservation"]["end_monotonic_ns"] for row in rows.values())
        for row in rows.values():
            row["started"]["monotonic_ns"]=start
            row["worker_reservation"]["start_monotonic_ns"]=start
            row["worker_reservation"]["end_monotonic_ns"]=end
        return rows
    proof=acceptance.check_simultaneous(recording,evaluator,baseline,cases[-1],observe=wrong_intervals,run_id="bad-overlap")
    assert not proof["ok"] and not proof["checks"]["server_execution_serialized"]
    def wrong_identity(ids):
        rows={key:copy.deepcopy(runner.observations[key]) for key in ids}
        rows[ids[0]]["started"]["job_id"]="unrelated-allocation"
        return rows
    mismatched=acceptance.check_simultaneous(recording,evaluator,baseline,cases[-1],observe=wrong_identity,run_id="wrong-owner")
    assert not mismatched["ok"] and not mismatched["checks"]["same_worker_and_gpu_uuid"]


def test_filesystem_clock_skew_is_diagnostic_and_cannot_change_monotonic_proof(tmp_path):
    runner,recording,evaluator,baseline,_,cases,_=prepared(tmp_path)
    def skewed_mtime(ids):
        rows={key:copy.deepcopy(runner.observations[key]) for key in ids}
        for row in rows.values():
            row["diagnostic_response_mtime_unix_ns"]=1
            row["diagnostic_started_at_unix_ns"]=10**30
        return rows
    proof=acceptance.check_simultaneous(recording,evaluator,baseline,cases[-1],observe=skewed_mtime,run_id="skew")
    assert proof["ok"] and proof["checks"]["server_execution_serialized"]
    assert not proof["filesystem_mtime_used_for_serialization"]


def test_different_worker_pid_rejects_comparison_even_when_monotonic_numbers_do_not_overlap(tmp_path):
    runner,recording,evaluator,baseline,_,cases,_=prepared(tmp_path)
    runner.profile_worker_pid=7655
    proof=acceptance.check_simultaneous(recording,evaluator,baseline,cases[-1],
        observe=lambda ids:{key:runner.observations[key] for key in ids},run_id="different-worker")
    assert proof["checks"]["server_intervals_present"]
    assert not proof["checks"]["same_monotonic_worker"] and not proof["ok"]


@pytest.mark.parametrize("missing",["monotonic_ns","worker_pid","session_id"])
def test_legacy_or_inconsistent_start_marker_never_establishes_isolation(tmp_path,missing):
    runner,recording,evaluator,baseline,_,cases,_=prepared(tmp_path)
    def incomplete(ids):
        rows={key:copy.deepcopy(runner.observations[key]) for key in ids}
        rows[ids[0]]["started"].pop(missing)
        return rows
    proof=acceptance.check_simultaneous(recording,evaluator,baseline,cases[-1],observe=incomplete,run_id="incomplete-marker")
    assert not proof["ok"] and not proof["checks"]["server_intervals_present"]


def test_optional_race_requires_actual_hazard_text_and_is_not_assumed_from_source(tmp_path):
    runner,_,evaluator,_,_,cases,_=prepared(tmp_path)
    runner.run_sanitizer=lambda *args,**kwargs:{"ok":True,"tool":"racecheck","error":"RACECHECK SUMMARY: 0 hazards"}
    proof=acceptance.check_shared_race(runner,evaluator,cases[0],tmp_path)
    assert proof["race_detected"] is False and proof["ok"] is False
    assert proof["status"] == "not_demonstrated" and proof["quality_cohort_eligible"] is False
    runner.run_sanitizer=lambda *args,**kwargs:{"ok":False,"tool":"racecheck","returncode":86,
        "error":"Race reported: WAW hazard at scratch; RACECHECK SUMMARY: 1 hazard"}
    assert acceptance.check_shared_race(runner,evaluator,cases[0],tmp_path)["race_detected"] is True


def test_owned_allocation_terminal_required_in_fully_simulated_controller(tmp_path):
    class FakeTransport:
        def __init__(self,runner):
            self.runner=runner
        def run(self,argv):
            ids=argv[-2:]
            return subprocess.CompletedProcess(argv,0,json.dumps({key:self.runner.observations[key] for key in ids}).encode(),b"")
    class Backend:
        simulation=True
        def __init__(self,config,repo,directory,session_id):
            assert config.poll_seconds == 17
            self.directory=Path(directory)
            self.config=config;self.session_id=session_id;self.job_id=None;self.remote_dir="/simulated-only"
            self.simulated=SimulatedRunner(directory);self.transport=FakeTransport(self.simulated)
        def start(self):
            intent=json.loads((self.directory.parent/"allocation-intent.json").read_text())
            assert intent["session_id"] == self.session_id and intent["remote_dir"] == self.remote_dir
            self.job_id="123456";return self.job_id
        def wait_until_ready(self):
            pass
        def runner(self):
            return self.simulated
        def close(self):
            pass
        def wait_until_finished(self,timeout):
            return {"terminal":True,"state":"COMPLETED","exit_code":"0:0","job_id":self.job_id}
    selected_config=tmp_path/"selected external config.json"
    config=json.loads((acceptance.REPO/"configs/romeo.json").read_text())
    config["poll_seconds"]=17
    selected_config.write_text(json.dumps(config))
    result=acceptance.run_acceptance(tmp_path/"simulated", "test-allocation",selected_config,backend_factory=Backend)
    assert result["ok"] and result["ac19"]["ok"] and result["ac14"]["ok"]
    assert result["evidence_kind"] == "simulated_cpu_test"
    assert result["model_calls"] == 0 and result["sources_unchanged"]
    sources=result["source_provenance"]["sources"]
    assert sources[str(selected_config.resolve())] == hashlib.sha256(selected_config.read_bytes()).hexdigest()
    assert all(key in sources for key in ("optimizer_agent/campaign.py","optimizer_agent/campaign_config.py"))
    assert "configs/romeo.json" not in sources
    with pytest.raises(ValueError,match="fresh"):
        acceptance.run_acceptance(tmp_path/"simulated", "same",selected_config,backend_factory=Backend)


def test_lost_submission_ack_keeps_exact_durable_intent_and_never_submits_again(tmp_path):
    attempts=[]
    class AckLostBackend:
        simulation=True
        def __init__(self,config,repo,directory,session_id):
            self.config=config;self.session_id=session_id;self.remote_dir="/protected/sessions/"+session_id
            self.directory=Path(directory);self.job_id=None
        def start(self):
            intent=json.loads((self.directory.parent/"allocation-intent.json").read_text())
            assert intent["session_id"] == self.session_id and intent["remote_dir"] == self.remote_dir
            assert intent["config"]["account"] == self.config.account
            assert "optimizer_agent/campaign.py" in intent["source_provenance"]["sources"]
            attempts.append(self.session_id)
            raise RuntimeError("Simulated lost scheduler acknowledgement after submission")
    directory=tmp_path/"lost-ack-simulated"
    result=acceptance.run_acceptance(directory,"ack-loss",acceptance.REPO/"configs/romeo.json",backend_factory=AckLostBackend)
    intent=json.loads((directory/"allocation-intent.json").read_text())
    receipt=json.loads((directory/"allocation-receipt.json").read_text())
    assert result["ok"] is False and receipt["submission_status"] == "uncertain"
    assert receipt["submitted"] is None and receipt["job_id"] is None
    assert receipt["session_id"] == intent["session_id"] == attempts[0]
    assert receipt["remote_dir"] == intent["remote_dir"]
    with pytest.raises(ValueError,match="fresh"):
        acceptance.run_acceptance(directory,"no-duplicate",acceptance.REPO/"configs/romeo.json",backend_factory=AckLostBackend)
    assert len(attempts) == 1
