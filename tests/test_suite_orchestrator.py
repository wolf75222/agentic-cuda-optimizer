from dataclasses import replace
import hashlib
import fcntl
import json
from pathlib import Path
import threading
import pytest

from optimizer_agent.campaign import Candidate, LaunchRule
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.suite_orchestrator import (
    SuiteJournal, SuiteOrchestrator, candidate_receipt, evaluate_group, load_candidate, write_json,
    observed_metrics,
)
from optimizer_agent.suite_report import build_requirement_status, write_report


def small_plan():
    plan = plan_suite(CampaignConfig(run_id="unit", search_sizes=(64,), profile=False), seeds=(1,2))
    plan["entries"] = [e for e in plan["entries"] if e["workload"] == "matmul" and e["seed"] == 1 and e["mode"] in ("full","true_upstream")]
    plan["modes"] = ("full","true_upstream")
    return plan


def frozen(directory, name="candidate"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / (name + ".cu")
    source.write_text('extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n){}')
    return candidate_receipt(Candidate(name, source, "matmul", LaunchRule(kind="square_2d",block=(8,8,1))))


class FakeBackend:
    def __init__(self, task, config, directory, jobs, lock):
        self.task, self.jobs, self.lock = task, jobs, lock
        self.job_id = None
    def start(self, allow_terminal=False):
        with self.lock:
            if self.task["session"] not in self.jobs:
                self.jobs[self.task["session"]] = str(len(self.jobs)+1)
            self.job_id = self.jobs[self.task["session"]]
        return self.job_id
    def wait_until_ready(self):
        pass
    def runner(self):
        return object()
    def close(self):
        pass
    def status(self):
        return {"state":"COMPLETED","exit_code":"0:0","job_id":self.job_id}
    def wait_until_finished(self, timeout):
        return {"terminal":True,"state":"COMPLETED","exit_code":"0:0","job_id":self.job_id}


def test_durable_intents_single_controller_no_duplicate_completed_jobs(tmp_path):
    jobs, lock, calls = {}, threading.Lock(), []
    def generate(entry, runner, directory, session, resume=False):
        calls.append(entry["mode"])
        return {"generation_complete":True,"candidate":frozen(directory, entry["mode"]),"metrics":{"correct_candidates":77}}
    def qualify(entries, generations, runner, directory, seed, commitment):
        assert len(generations) == 2 and len(calls) == 2
        assert hashlib.sha256(str(seed).encode()).hexdigest() == commitment
        return {"evaluation_finished":True,"all_modes_available":True,"baseline":frozen(directory),"candidates":{}}
    factory = lambda task, config, directory: FakeBackend(task,config,directory,jobs,lock)
    suite = SuiteOrchestrator(small_plan(),tmp_path,backend_factory=factory,generate=generate,qualify=qualify)
    before = suite.journal.commitments()
    assert len(before) == 1 and not jobs
    with suite.journal.controller():
        with pytest.raises(RuntimeError, match="Another suite controller"):
            with SuiteJournal(tmp_path,small_plan()).controller():
                pass
    result = suite.run()
    assert len(jobs) == 3 and len(calls) == 2
    assert suite.journal.tasks("generation")[0]["result"]["metrics"]["correct_candidates"] == 77
    assert result["summary"]["complete"]
    assert not result["summary"]["scientific_qualified"]
    resumed = SuiteOrchestrator(small_plan(),tmp_path,backend_factory=factory,generate=generate,qualify=qualify)
    resumed.run()
    assert len(jobs) == 3 and len(calls) == 2 and resumed.journal.commitments() == before
    candidate = resumed.journal.tasks("generation")[0]["result"]["candidate"]
    Path(candidate["source_path"]).write_text("changed")
    with pytest.raises(ValueError,match="hash mismatch"):
        resumed.run()


def test_active_intent_reuses_exact_session_and_interrupted_final_is_never_repeated(tmp_path):
    plan = small_plan()
    journal = SuiteJournal(tmp_path,plan)
    task = journal.tasks("generation")[0]
    journal.update(task["id"],state="running",job_id="previous")
    jobs, lock, resumes = {task["session"]:"previous"}, threading.Lock(), []
    def generate(entry,runner,directory,session,resume=False):
        resumes.append((session,resume))
        return {"generation_complete":True,"candidate":frozen(directory,entry["mode"])}
    final = journal.tasks("final")[0]
    target = tmp_path/"final-groups"/final["id"]
    write_json(target/"holdout-reveal.json",{"already_exposed":True})
    def no_repeat(*args):
        pytest.fail("An exposed final must not run again")
    suite = SuiteOrchestrator(plan,tmp_path,backend_factory=lambda t,c,d:FakeBackend(t,c,d,jobs,lock),generate=generate,qualify=no_repeat)
    suite.run()
    assert (task["session"],True) in resumes
    assert len(jobs) == 2
    assert suite.journal.tasks("final")[0]["state"] == "recovery_required"


def test_frozen_candidate_hash_and_plan_drift_rejected(tmp_path):
    receipt = frozen(tmp_path/"source")
    load_candidate(receipt)
    forged = {**receipt,"parameters":{"TILE":32}}
    with pytest.raises(ValueError,match="artifact hash mismatch"):
        load_candidate(forged)
    SuiteJournal(tmp_path/"suite",small_plan())
    # Identical public run IDs in independent experiments never share a lease.
    other = SuiteJournal(tmp_path/"other-suite",small_plan())
    assert SuiteJournal(tmp_path/"suite",small_plan()).tasks()[0]["session"] != other.tasks()[0]["session"]
    changed = small_plan()
    changed["entries"][0]["config"]["max_trials"] += 1
    with pytest.raises(ValueError,match="plan changed"):
        SuiteJournal(tmp_path/"suite",changed)


def test_live_orphan_worker_task_lock_prevents_duplicate_dispatch(tmp_path):
    suite = SuiteOrchestrator(small_plan(),tmp_path)
    task = suite.journal.tasks("generation")[0]
    directory = tmp_path/"task-locks"
    directory.mkdir()
    path = directory/(hashlib.sha256(task["id"].encode()).hexdigest()+".lock")
    with path.open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        suite._task(task)
        assert suite.journal.tasks("generation")[0]["state"] == "pending"


def test_explicit_infrastructure_recovery_keeps_losses_changes_seed_session_and_never_regenerates(tmp_path):
    jobs, lock = {}, threading.Lock()
    suite = SuiteOrchestrator(small_plan(),tmp_path,backend_factory=lambda t,c,d:FakeBackend(t,c,d,jobs,lock))
    task = suite.journal.tasks("final")[0]
    old_seed = suite.journal.seed(task["id"])
    loss={"status":"not_improved","interval":{"gain":.7,"lower":.6,"upper":.8},"artifact_hash":"old-loss"}
    failure={"infrastructure_category":"transport_timeout","partial_result":{"confirmation":loss}}
    suite.journal.update(task["id"],state="recovery_required",job_id="old-job",result=failure)
    directory=suite._directory(task)
    write_json(directory/"failure.json",failure)
    audit=suite.recover_final(task["id"],"SSH transport interrupted after the first recorded loss")
    new=suite.journal.tasks("final")[0]
    assert new["attempt"] == 1 and new["session"] != task["session"] and new["state"] == "pending"
    assert suite.journal.seed(task["id"]) != old_seed
    assert not jobs and audit["new_job_submitted"] is False and audit["model_calls"] == 0
    assert audit["prior_statistical_decisions_retained"][0]["interval"]["gain"] == .7
    assert (directory/"failure.json").exists()
    assert suite.status()["previous_attempts"][0]["receipt"]["prior_statistical_decisions"][0]["interval"]["gain"] == .7
    # A valid complete group cannot be requested again, irrespective of gain.
    suite.journal.update(task["id"],state="completed",result={"execution_complete":True})
    with pytest.raises(ValueError,match="Only an incomplete"):
        suite.recover_final(task["id"],"Another transport incident cannot justify retrying a complete group")


@pytest.mark.parametrize("status",["numerical_error","contract_violation","sanitizer_qualification_failed"])
def test_infrastructure_label_cannot_mask_invalid_kernel_or_sanitizer(tmp_path,status):
    suite=SuiteOrchestrator(small_plan(),tmp_path)
    task=suite.journal.tasks("final")[0]
    suite.journal.update(task["id"],state="recovery_required",job_id="old",result={
        "infrastructure_category":"worker_terminal_incomplete","qualification":{"status":status}})
    with pytest.raises(ValueError,match="cannot be retried"):
        suite.recover_final(task["id"],"Worker terminated after an invalid kernel, this must be refused")


def test_native_observation_counts_include_invalid_artifacts_without_inventing_gain(tmp_path):
    evidence=tmp_path/"generation/evidence"
    for request_id,artifact,operation,valid,status in (
            ("a","source-a","validate",True,"validated"),
            ("b","source-b","validate",False,"numerical_error"),
            ("c","source-a","benchmark",True,"measured")):
        write_json(evidence/"requests"/request_id/"response.json",{
            "operation":operation,"artifact_hash":artifact,"ok":valid,"result":{"status":status}})
    (evidence/"requests.jsonl").write_text(json.dumps({"timestamp":123,"operation":"benchmark","ok":True,"status":"complete"})+"\n")
    metrics=observed_metrics(tmp_path,{"usage":{}})
    assert metrics["candidate_configurations_with_recorded_validity"] == 2
    assert metrics["correct_candidates"] == 1 and metrics["correct_candidate_fraction"] == .5
    assert metrics["configuration_observations"] == 2
    assert metrics["first_valid_search_measurement_unix"] == 123
    assert metrics["time_to_first_useful_gain_seconds"] is None
    assert metrics["first_useful_gain_status"] == "not_recorded_by_method"


def test_explicit_recovery_requires_recorded_incident_and_terminal_owned_job(tmp_path):
    class ActiveBackend(FakeBackend):
        def status(self):
            return {"state":"RUNNING","job_id":self.job_id}
    suite=SuiteOrchestrator(small_plan(),tmp_path,backend_factory=lambda t,c,d:ActiveBackend(t,c,d,{},threading.Lock()))
    task=suite.journal.tasks("final")[0]
    suite.journal.update(task["id"],state="recovery_required",job_id="own",result={"error":"unknown"})
    with pytest.raises(ValueError,match="No qualified"):
        suite.recover_final(task["id"],"Unknown incident is insufficient evidence for a retry")
    suite.journal.update(task["id"],result={"infrastructure_category":"transport_timeout"})
    seed=suite.journal.seed(task["id"])
    with pytest.raises(ValueError,match="not confirmed terminal"):
        suite.recover_final(task["id"],"The worker is still active and must never receive a duplicate allocation")
    assert suite.journal.seed(task["id"]) == seed and suite.journal.tasks("final")[0]["attempt"] == 0


@pytest.mark.parametrize("pmpp_contrast", [False, True])
def test_group_seals_all_candidates_then_validates_everything_before_timing(tmp_path,monkeypatch,pmpp_contrast):
    import optimizer_agent.campaign as campaign
    order = []
    expected_qualifications = 2  # The generated fixture sources are exact aliases.
    class Runner:
        def get_device_properties(self,device):
            return {"uuid":"GPU-fixed","architecture":"sm_90"}
    class Evaluator:
        contract_hash = "contract"
        def __init__(self,runner,contract,oracle,suites):
            assert contract.required_sanitizers == ("memcheck","racecheck","initcheck","synccheck")
            assert [case.metadata["n"] for case in suites["performance"]] == [1,17,63,64]
            for case in suites["search"]:
                assert oracle(case)[0].dtype.name == "float32"
        def qualify_sanitizers(self,candidate,suite):
            assert suite == "search"
            order.append("qualify:"+candidate.candidate_id)
            return {"ok":True,"status":"sanitizer_qualified","evaluation_id":"cpu-"+candidate.candidate_id,
                    "artifact_hash":candidate.artifact_hash,"contract_hash":self.contract_hash,"suite":suite}
        def qualification_context(self,candidate,suite):
            return {"key":candidate.artifact_hash,"identity":{"suite":suite}}
        def benchmark(self,candidate):
            assert len([row for row in order if row.startswith("qualify")]) == expected_qualifications
            order.append("measure:baseline")
            return {"ok":True,"status":"measured","cases":{}}
        def confirm(self,candidate,baseline):
            order.append("measure:"+candidate.candidate_id)
            assert len([row for row in order if row.startswith("qualify")]) == expected_qualifications
            return {"ok":True,"status":"not_improved","candidate_id":candidate.candidate_id,
                    "incumbent_id":baseline.candidate_id,"interval":{"gain":.8,"lower":.7,"upper":.9}}
    monkeypatch.setattr(campaign,"Evaluator",Evaluator)
    plan = small_plan()
    if pmpp_contrast:
        expanded = plan_suite(CampaignConfig(run_id="unit", search_sizes=(64,), profile=False), seeds=(1,2))
        plan["entries"] += [entry for entry in expanded["entries"] if entry["workload"] == "matmul" and
                            entry["seed"] == 1 and entry["mode"] in ("pmpp_permanent_only", "no_pmpp")]
    generations = {e["config"]["run_id"]:{"candidate":frozen(tmp_path/"generation"/e["mode"],e["mode"])} for e in plan["entries"]}
    seed=123456
    proof=evaluate_group(plan["entries"],generations,Runner(),tmp_path/"final",seed,hashlib.sha256(str(seed).encode()).hexdigest())
    assert proof["all_modes_available"] and proof["generation_scores_used"] is False
    assert all(r["status"] == "validated_gain_uncertified" for r in proof["candidates"].values())
    assert len(proof["input_hashes"]) == 8
    if pmpp_contrast:
        assert "full_vs_pmpp_permanent_only" in proof["method_comparisons"]
        contrast = proof["method_comparisons"]["pmpp_permanent_only_vs_no_pmpp"]
        assert contrast["candidate_id"] == "pmpp_permanent_only"
        assert contrast["incumbent_id"] == "no_pmpp"


def test_requirement_report_never_promotes_plans_or_source_paths_to_test_proof(tmp_path):
    evidence={"AC-08":{"implemented":{"ok":True,"evidence":["evaluation.py"],"scope":"guards"},
                         "local_tested":{"ok":True},"romeo_executed":{"ok":False,"evidence":["planned.json"]}}}
    rows={r["id"]:r for r in build_requirement_status(evidence)}
    assert all("AC-%02d"%i in rows for i in range(1,21))
    assert rows["AC-08"]["implemented"]["ok"]
    assert not rows["AC-08"]["local_tested"]["ok"] and not rows["AC-08"]["romeo_executed"]["ok"]
    suite=SuiteOrchestrator(small_plan(),tmp_path/"suite")
    report=write_report(tmp_path,suite.status(),evidence=evidence)
    assert not report["summary"]["complete"]
    assert "en attente" in (tmp_path/"rapport-fr.md").read_text()


def test_explicit_not_applicable_gpu_phase_does_not_leave_local_policy_pending():
    rows={r["id"]:r for r in build_requirement_status({"AC-05":{
        "implemented":{"ok":True,"evidence":["search_policy.py"]},
        "local_tested":{"ok":True,"evidence":["local-tests.json"]},
        "romeo_executed":{"applicable":False,"scope":"Pure local parent-routing policy"},"remaining":[]}})}
    assert rows["AC-05"]["romeo_executed"]["status"] == "not_applicable"
    assert rows["AC-05"]["pending"] == [] and rows["AC-05"]["remaining"] == []
    # A simulated CUDA test is never inferred to make the GPU phase inapplicable.
    assert rows["AC-08"]["romeo_executed"]["applicable"] is True
