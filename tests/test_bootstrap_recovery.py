"""Pre-model recovery acceptance with local files and fake read-only backends.

These tests never start a provider, SSH process, Slurm allocation or GPU.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from optimizer_agent import bootstrap_recovery as recovery, suite_orchestrator as suite
from optimizer_agent.bootstrap_replay import BootstrapReplayRunner
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.execution_backend import BackendError, SlurmBackend, SlurmConfig
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.use_harness import BufferArgument, BufferDirection, KernelRequest


REASON = "The local network disappeared before any generator or model call"
JOB = "771111"
HARDWARE = {"slurm_job_id":JOB,"hostname":"SIMULATED_NODE_ONLY",
    "device_properties":{"uuid":"SIMULATED_GPU_ONLY","architecture":"sm_90","harness_build_id":"cpu-test"}}


def plan(mode):
    value = plan_suite(CampaignConfig(run_id="bootstrap-test",search_sizes=(8,)),seeds=(1,2))
    value["entries"] = [entry for entry in value["entries"] if entry["mode"] == mode
        and entry["workload"] == "matmul" and entry["seed"] == 1]
    value["modes"] = [mode]
    return value


def no_activity(*args,**kwargs):
    pytest.fail("Recovery preparation cannot submit, stage, execute a model or request GPU work")


class LocalTransport:
    def __init__(self,state,actions):
        self.state,self.actions = state,actions

    def read(self,path,optional=False):
        name = Path(path).name
        self.actions.append(("read",name))
        value = self.state["remote"].get(name)
        return None if value is None else json.dumps(value).encode()

    def run(self,argv,**kwargs):
        self.actions.append(("accounting",argv[0]))
        if argv[0] == "squeue":
            body = f"{JOB}|RUNNING|SIMULATED_NODE_ONLY\n" if self.state["status"] == "RUNNING" else ""
        elif argv[0] == "sacct":
            body = f"{JOB}|{self.state['status']}|1:0|00:01:00|SIMULATED_NODE_ONLY\n"
        else:
            no_activity()
        return subprocess.CompletedProcess(argv,0,body.encode(),b"")

    write = no_activity
    exchange = no_activity


class ReadonlyBackend(SlurmBackend):
    def __init__(self,task,directory,state,actions):
        self.state,self.actions = state,actions
        super().__init__(SlurmConfig(),suite.REPO,Path(directory)/"worker",task["session"],
            transport=LocalTransport(state,actions))

    start = no_activity
    stage = no_activity
    wait_until_ready = no_activity
    runner = no_activity
    close = no_activity


def recorded_prefix(directory,*,inside_generation=False):
    root = directory/("generation/evidence/search-data" if inside_generation else "bootstrap-inputs")
    root.mkdir(parents=True,exist_ok=True)
    source = root/"baseline.cu"
    source.write_text('extern "C" __global__ void f(const float* a,float* b) {}')
    inp = root/"input.bin"; inp.write_bytes(b"data")
    output = directory/"worker-result.bin"; output.write_bytes(b"done")
    request = KernelRequest(str(source),"f","run",0,[1,1,1],[32,1,1],0,0,1,
        [BufferArgument(4,str(inp),BufferDirection.Input),BufferArgument(4,None,BufferDirection.Output)])
    report = {"ok":True,"hardware_fingerprint":deepcopy(HARDWARE),"slurm_job_id":JOB,
        "environment":deepcopy(HARDWARE["device_properties"]),
        "outputs":[{"path":str(output),"bytes":4}],"latency_ms":2.0,"latency_samples_ms":[2.0]}
    for index,current in enumerate((request,replace(request,mode="benchmark",iterations=2))):
        row = {"run_id":"original-"+str(index),"request":asdict(current),
            "started_at_unix":10.0*index,"ended_at_unix":10.0*index+1}
        row.update({"report":report} if index == 0 else {"error":"ssh: connect to host romeo port 22: Can't assign requested address"})
        suite.write_json(directory/"request-receipts"/(str(index)+".json"),row)
    return (request,replace(request,mode="benchmark",iterations=2))


def study(tmp_path,*,mode="full",status="RUNNING",submitted=True,prefix=True):
    state = {"status":status,"remote":{}}
    actions,instances = [],[]
    def factory(task,config,directory):
        actions.append(("backend",task["session"]))
        value = ReadonlyBackend(task,directory,state,actions)
        instances.append(value)
        return value
    orchestrator = suite.SuiteOrchestrator(plan(mode),tmp_path,backend_factory=factory,
        generate=no_activity,qualify=no_activity)
    task = orchestrator.journal.tasks("generation")[0]
    directory = orchestrator._directory(task)
    backend = ReadonlyBackend(task,directory,state,actions)
    if submitted:
        state["remote"]["job_id.json"] = {"session_id":task["session"],"job_id":JOB,"remote_dir":backend.remote_dir}
        state["remote"]["ready.json"] = {"ok":True,"hardware":deepcopy(HARDWARE)}
        suite.write_json(directory/"worker/slurm-session.json",{**state["remote"]["job_id.json"],"config":asdict(backend.config)})
        suite.write_json(directory/"worker/hardware.json",HARDWARE)
    failure = {"error":"ssh: connect to host romeo port 22: Can't assign requested address",
        "infrastructure_category":"ssh_transport_failure","job_id":JOB if submitted else None}
    suite.write_json(directory/"failure.json",failure)
    requests = recorded_prefix(directory,inside_generation=mode == "native_codex_vanilla") if prefix else ()
    if mode == "native_codex_vanilla":
        suite.write_json(directory/"generation/evidence/receipt.json",{"status":"failed","model_started":False,
            "usage_events":[],"thread_ids":[],"usage":{"input_tokens":0,"output_tokens":0}})
        workspace = directory/"generation/workspace"; workspace.mkdir(parents=True,exist_ok=True)
        (workspace/"baseline.cu").write_text("parent supplied baseline")
    orchestrator.journal.update(task["id"],state="recovery_required",job_id=JOB if submitted else None,
        result=failure,error=failure["error"],started=1.0,finished=2.0)
    return orchestrator,orchestrator.journal.tasks("generation")[0],directory,state,actions,instances,requests


@pytest.mark.parametrize("artifact",[
    "model/pending-call.json","model-transport/calls.json","research/source.json","trials/proposal.cu",
    "langgraph.sqlite","experiments.sqlite","manifest.json","candidate.json","best.cu",
    "original-langgraph.sqlite",
])
def test_any_model_intent_or_checkpoint_blocks_before_backend_or_remote_read(tmp_path,artifact):
    value,task,directory,_,actions,_,_ = study(tmp_path)
    path = directory/"generation"/artifact
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("{}")
    with pytest.raises(ValueError):
        value.recover_bootstrap(task["id"],REASON)
    assert actions == [] and value.journal.tasks("generation")[0]["state"] == "recovery_required"


@pytest.mark.parametrize("marker",[
    {"model_started":True},{"child_pid":1234},{"thread_ids":["provider-thread"]},
    {"usage_events":[{"type":"turn.started"}]},{"usage":{"input_tokens":1}},
    {"status":"running"},{"actual_models":["gpt-6-luna"]},
])
def test_native_partial_process_or_provider_evidence_is_never_reprompted(tmp_path,marker):
    value,task,directory,_,actions,_,_ = study(tmp_path,mode="native_codex_vanilla")
    path = directory/"generation/evidence/receipt.json"
    row = json.loads(path.read_text()); row.update(marker); suite.write_json(path,row)
    with pytest.raises(ValueError):
        value.recover_bootstrap(task["id"],REASON)
    assert actions == []


@pytest.mark.parametrize("artifact",["codex.jsonl","last-message.json","response.json","candidate.json"])
def test_native_output_intent_blocks_even_when_empty(tmp_path,artifact):
    value,task,directory,_,actions,_,_ = study(tmp_path,mode="native_codex_vanilla")
    (directory/"generation/evidence"/artifact).write_bytes(b"")
    with pytest.raises(ValueError):
        value.recover_bootstrap(task["id"],REASON)
    assert actions == []


@pytest.mark.parametrize("mismatch",["remote_session","remote_directory","local_session","local_job","journal_job","failure_job"])
def test_remote_local_journal_or_failure_ownership_mismatch_blocks_recovery(tmp_path,mismatch):
    value,task,directory,state,_,_,_ = study(tmp_path)
    local = directory/"worker/slurm-session.json"
    if mismatch == "remote_session":
        state["remote"]["job_id.json"]["session_id"] = "someone-else"
    elif mismatch == "remote_directory":
        state["remote"]["job_id.json"]["remote_dir"] = "/other/owned/directory"
    elif mismatch.startswith("local_"):
        row = json.loads(local.read_text()); row["session_id" if mismatch == "local_session" else "job_id"] = "other"
        suite.write_json(local,row)
    elif mismatch == "journal_job":
        value.journal.update(task["id"],job_id="771999")
    else:
        row = json.loads((directory/"failure.json").read_text()); row["job_id"] = "771999"
        suite.write_json(directory/"failure.json",row)
    with pytest.raises((ValueError,BackendError)):
        value.recover_bootstrap(task["id"],REASON)
    assert value.journal.tasks("generation")[0]["state"] == "recovery_required"
    assert not (directory/"bootstrap-recovery.json").exists()


def test_terminal_worker_gets_new_attempt_without_pooling_or_touching_old_measurements(tmp_path):
    value,task,directory,_,actions,_,_ = study(tmp_path,status="FAILED")
    before = {str(p.relative_to(directory)):p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    audit = value.recover_bootstrap(task["id"],REASON)
    row = value.journal.tasks("generation")[0]
    assert row["state"] == "pending" and row["attempt"] == 1 and row["job_id"] is None
    assert row["session"] != task["session"] and value._directory(row) != directory
    assert row["result"] is None and not value._directory(row).exists()
    assert not audit["new_job_submitted"] and audit["model_calls"] == audit["gpu_requests"] == 0
    assert audit["raw_timings_pooled_between_allocations"] is False
    assert all((directory/name).read_bytes() == data for name,data in before.items())
    with value.journal.connection() as db:
        attempts = db.execute("SELECT * FROM task_attempts").fetchall()
    assert len(attempts) == 1 and json.loads(attempts[0]["snapshot"])["task"]["session"] == task["session"]
    count = len(actions)
    with pytest.raises(ValueError):
        value.recover_bootstrap(task["id"],REASON)
    assert len(actions) == count


def test_live_recovery_keeps_allocation_and_replay_proof_without_any_execution(tmp_path):
    value,task,directory,_,actions,_,requests = study(tmp_path)
    receipts = {p.name:p.read_bytes() for p in (directory/"request-receipts").glob("*.json")}
    audit = value.recover_bootstrap(task["id"],REASON)
    row = value.journal.tasks("generation")[0]
    assert row["state"] == "running" and row["session"] == task["session"] and row["job_id"] == JOB and row["attempt"] == 0
    assert audit["same_live_allocation"] and audit["new_job_submitted"] is False
    manifest = Path(audit["replay_manifest"])
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == audit["replay_manifest_sha256"]
    assert {p.name:p.read_bytes() for p in (directory/"request-receipts").glob("*.json")} == receipts
    assert audit["phase"] == "committed" and audit["model_calls"] == audit["gpu_requests"] == 0
    assert all(action[0] in {"backend","read","accounting"} for action in actions)
    class Runner:
        hardware = HARDWARE
        run = no_activity
    replay = BootstrapReplayRunner(Runner(),manifest,HARDWARE,directory/"verify-replay")
    assert replay.run("discarded-new-id",requests[0])["latency_samples_ms"] == [2.0]


def test_live_native_archives_old_workspace_but_frozen_prefix_remains_reconstructible(tmp_path):
    value,task,directory,_,_,_,requests = study(tmp_path,mode="native_codex_vanilla")
    old_receipt = (directory/"generation/evidence/receipt.json").read_bytes()
    audit = value.recover_bootstrap(task["id"],REASON)
    archive = Path(audit["native_pre_model_archive"])
    assert (archive/"evidence/receipt.json").read_bytes() == old_receipt
    assert not (directory/"generation").exists()  # Fresh one-prompt filesystem.
    class Runner:
        hardware = HARDWARE
        run = no_activity
    replay = BootstrapReplayRunner(Runner(),audit["replay_manifest"],HARDWARE,directory/"verify-native-replay",
        defer_request_files=True)
    # A fresh service regenerates its deterministic public search inputs before
    # it issues the first baseline request. Receipt/output checks stayed active
    # during construction; source/input bytes are checked before each run.
    for name in ("baseline.cu","input.bin"):
        target = directory/"generation/evidence/search-data"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((archive/"evidence/search-data"/name).read_bytes())
    assert replay.run("discarded-new-id",requests[0])["latency_samples_ms"] == [2.0]


def test_not_submitted_requires_absent_remote_submission_markers_then_preserves_session(tmp_path):
    value,task,_,_,actions,_,_ = study(tmp_path,submitted=False,prefix=False)
    audit = value.recover_bootstrap(task["id"],REASON)
    row = value.journal.tasks("generation")[0]
    assert row["state"] == "running" and row["session"] == task["session"] and row["job_id"] is None
    assert audit["ownership_status"]["state"] == "NOT_SUBMITTED" and not audit["new_job_submitted"]
    assert ("read","job_id.json") in actions and ("read","submission_started.json") in actions and ("read","submission_error.json") in actions
    assert not any(action[0] == "accounting" for action in actions)


@pytest.mark.parametrize("marker",["submission_started.json","submission_error.json","journal_job","failure_job",
    "local_receipt","hardware_receipt","gpu_receipt"])
def test_absent_remote_job_does_not_erase_uncertain_submission_intent(tmp_path,marker):
    value,task,directory,state,_,_,_ = study(tmp_path,submitted=False,prefix=False)
    if marker.endswith(".json"):
        state["remote"][marker] = {}  # Even an empty durable intent is uncertain.
    elif marker == "journal_job":
        value.journal.update(task["id"],job_id=JOB)
    elif marker == "failure_job":
        row = json.loads((directory/"failure.json").read_text()); row["job_id"] = JOB
        suite.write_json(directory/"failure.json",row)
    elif marker == "hardware_receipt":
        suite.write_json(directory/"worker/hardware.json",HARDWARE)
    elif marker == "gpu_receipt":
        suite.write_json(directory/"request-receipts/unknown-allocation.json",{"report":{"ok":True,"slurm_job_id":JOB}})
    else:
        suite.write_json(directory/"worker/slurm-session.json",{"job_id":JOB})
    with pytest.raises(ValueError,match="Uncertain scheduler ownership"):
        value.recover_bootstrap(task["id"],REASON)
    assert not (directory/"bootstrap-recovery.json").exists()


@pytest.mark.parametrize("lock_kind",["controller","task"])
def test_concurrent_controller_or_task_prevents_all_recovery_actions(tmp_path,lock_kind):
    value,task,_,_,actions,_,_ = study(tmp_path)
    if lock_kind == "controller":
        with value.journal.controller():
            with pytest.raises(RuntimeError,match="Another suite controller"):
                value.recover_bootstrap(task["id"],REASON)
    else:
        path = value.directory/"task-locks"/(hashlib.sha256(task["id"].encode()).hexdigest()+".lock")
        path.parent.mkdir(exist_ok=True)
        with path.open("a+") as handle:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            with pytest.raises(ValueError,match="still active"):
                value.recover_bootstrap(task["id"],REASON)
    assert actions == []


@pytest.mark.parametrize("mismatch",["hardware_job","remote_ready_gpu","quarantine","queued","uncertain_remote"])
def test_live_target_must_be_same_ready_gpu_and_cannot_bypass_quarantine(tmp_path,mismatch):
    value,task,directory,state,_,_,_ = study(tmp_path)
    if mismatch == "hardware_job":
        row = deepcopy(HARDWARE); row["slurm_job_id"] = "771999"
        suite.write_json(directory/"worker/hardware.json",row)
    elif mismatch == "remote_ready_gpu":
        state["remote"]["ready.json"]["hardware"]["device_properties"]["uuid"] = "different-gpu"
    elif mismatch == "quarantine":
        state["remote"]["quarantined.json"] = {}
    elif mismatch == "queued":
        state["status"] = "PENDING"
    else:
        state["remote"]["job_id.json"].update(uncertain=True,job_ids=[JOB,"771999"])
    with pytest.raises((ValueError,BackendError)):
        value.recover_bootstrap(task["id"],REASON)
    assert value.journal.tasks("generation")[0]["state"] == "recovery_required"
    assert not (directory/"bootstrap-recovery.json").exists()


@pytest.mark.parametrize("mode",["full","native_codex_vanilla"])
def test_orchestrator_replay_uses_same_lease_and_original_failed_request_id_only(tmp_path,mode):
    value,task,directory,state,actions,_,requests = study(tmp_path,mode=mode)
    value.recover_bootstrap(task["id"],REASON)
    row = value.journal.tasks("generation")[0]
    old_receipts = {path.name:path.read_bytes() for path in (directory/"request-receipts").glob("*.json")}
    calls = []
    class Runner:
        hardware = HARDWARE
        def run(self,run_id,request):
            calls.append((run_id,asdict(request)))
            assert run_id == "original-1"  # No random replacement execution ID.
            output = directory/"recovered-output.bin"; output.write_bytes(b"done")
            return {"ok":True,"slurm_job_id":JOB,"hardware_fingerprint":deepcopy(HARDWARE),
                "environment":deepcopy(HARDWARE["device_properties"]),
                "outputs":[{"path":str(output),"bytes":4}],"latency_ms":7.0,"latency_samples_ms":[7.0]}
    runner = Runner()
    class SameLeaseBackend(ReadonlyBackend):
        def start(self):
            actions.append(("resume_existing_lease",self.session_id))
            assert self.state["remote"]["job_id.json"]["job_id"] == JOB
            self.job_id = JOB
        def wait_until_ready(self):
            self.hardware = deepcopy(HARDWARE)
        def runner(self):
            return runner
        def close(self):
            actions.append(("close_same_fake_lease",self.job_id))
        def wait_until_finished(self,timeout):
            return {"job_id":JOB,"terminal":True,"state":"COMPLETED","exit_code":"0:0"}
    value.backend_factory = lambda task,config,directory:SameLeaseBackend(task,directory,state,actions)
    def generate(entry,replayed,generation,session,resume=False):
        assert resume and session == task["session"] and isinstance(replayed,BootstrapReplayRunner)
        if mode == "native_codex_vanilla":
            assert not Path(generation).exists()
            archive = directory/"bootstrap-archives/000/generation"
            for name in ("baseline.cu","input.bin"):
                target = Path(generation)/"evidence/search-data"/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes((archive/"evidence/search-data"/name).read_bytes())
        assert replayed.run("new-random-0",requests[0])["latency_ms"] == 2.0
        assert replayed.run("new-random-1",requests[1])["latency_ms"] == 7.0
        assert replayed.replay_complete
        return {"generation_complete":True,"status":"generated","candidate":None}
    value.generate = generate
    value._task(row)
    completed = value.journal.tasks("generation")[0]
    assert completed["state"] == "completed" and completed["job_id"] == JOB and completed["session"] == task["session"]
    assert len(calls) == 1 and calls[0][0] == "original-1"
    assert {path.name:path.read_bytes() for path in (directory/"request-receipts").glob("*.json")} == old_receipts
    assert (directory/"request-receipts/recovered/original-1.json").is_file()
    count = len(actions)
    value._task(completed)
    assert len(actions) == count and len(calls) == 1


@pytest.mark.parametrize("boundary",["prepared_intent","native_rename","journal_transition"])
def test_interrupted_live_preparation_remains_fail_closed_without_double_prompt_or_allocation(tmp_path,monkeypatch,boundary):
    value,task,directory,_,actions,_,_ = study(tmp_path,mode="native_codex_vanilla")
    original_write,original_rename,original_update = suite.write_json,Path.rename,value.journal.update
    intent = directory/"bootstrap-recovery.json"
    if boundary == "prepared_intent":
        def write(path,data):
            original_write(path,data)
            if Path(path) == intent and data.get("phase") == "prepared":
                raise KeyboardInterrupt("crash after durable prepared intent")
        monkeypatch.setattr(suite,"write_json",write)
    elif boundary == "native_rename":
        def rename(path,target):
            result = original_rename(path,target)
            if path == directory/"generation":
                raise KeyboardInterrupt("crash after native archive rename")
            return result
        monkeypatch.setattr(Path,"rename",rename)
    else:
        def update(key,**values):
            original_update(key,**values)
            if key == task["id"] and values.get("state") == "running":
                raise KeyboardInterrupt("crash after journal transition")
        monkeypatch.setattr(value.journal,"update",update)
    with pytest.raises(KeyboardInterrupt):
        value.recover_bootstrap(task["id"],REASON)
    persisted = json.loads(intent.read_text())
    assert persisted["phase"] == "prepared" and persisted["new_job_submitted"] is False
    row = value.journal.tasks("generation")[0]
    assert row["session"] == task["session"] and row["job_id"] == JOB and row["attempt"] == 0
    assert row["state"] == ("running" if boundary == "journal_transition" else "recovery_required")
    assert (directory/"generation").exists() == (boundary == "prepared_intent")
    assert (directory/"bootstrap-archives/000/generation").exists() == (boundary != "prepared_intent")
    count = len(actions)
    with pytest.raises(ValueError):
        value.recover_bootstrap(task["id"],REASON)
    assert len(actions) == count  # Dedicated reconciliation is required; never prepare twice.
    # The ordinary scheduler must leave a prepared transaction completely idle.
    value._task(row)
    assert len(actions) == count
    monkeypatch.setattr(suite,"write_json",original_write)
    monkeypatch.setattr(Path,"rename",original_rename)
    monkeypatch.setattr(value.journal,"update",original_update)
    resumed = value.resume_bootstrap_preparation(task["id"])
    completed = value.journal.tasks("generation")[0]
    assert resumed["phase"] == "committed" and completed["state"] == "running"
    assert completed["session"] == task["session"] and completed["job_id"] == JOB and completed["attempt"] == 0
    assert (directory/"bootstrap-archives/000/generation").exists() and not (directory/"generation").exists()
    assert resumed["model_calls"] == resumed["gpu_requests"] == 0 and not resumed["new_job_submitted"]
    for name,digest in resumed["previous_artifacts_sha256"].items():
        path = (Path(resumed["native_pre_model_archive"])/name[len("generation/"):]
            if name.startswith("generation/") else directory/name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert all(action[0] in {"backend","read","accounting"} for action in actions)
    count = len(actions)
    with pytest.raises(ValueError):
        value.resume_bootstrap_preparation(task["id"])
    assert len(actions) == count


@pytest.mark.parametrize("boundary",["before_db","after_db"])
def test_terminal_preparation_resume_commits_exactly_one_new_attempt_after_crash(tmp_path,monkeypatch,boundary):
    value,task,directory,_,actions,_,_ = study(tmp_path,status="FAILED")
    original_write = suite.write_json
    intent = directory/"bootstrap-recovery.json"
    def crash(path,data):
        if Path(path) == intent:
            if boundary == "before_db" and data.get("phase") == "prepared":
                original_write(path,data)
                raise KeyboardInterrupt("crash before attempt transaction")
            if boundary == "after_db" and data.get("phase") == "committed":
                raise KeyboardInterrupt("crash after committed attempt transaction before final ACK")
        original_write(path,data)
    monkeypatch.setattr(suite,"write_json",crash)
    with pytest.raises(KeyboardInterrupt):
        value.recover_bootstrap(task["id"],REASON)
    prepared = json.loads(intent.read_text())
    assert prepared["phase"] == "prepared" and prepared["new_session"] != task["session"]
    row = value.journal.tasks("generation")[0]
    assert row["attempt"] == (0 if boundary == "before_db" else 1)
    count = len(actions)
    value._task(row)
    assert len(actions) == count  # No allocation before reconciliation finishes.
    monkeypatch.setattr(suite,"write_json",original_write)
    resumed = value.resume_bootstrap_preparation(task["id"])
    row = value.journal.tasks("generation")[0]
    assert row["state"] == "pending" and row["attempt"] == 1 and row["job_id"] is None
    assert row["session"] == prepared["new_session"] and resumed["phase"] == "committed"
    assert row["result"] is None and not value._directory(row).exists()
    with value.journal.connection() as db:
        attempts = db.execute("SELECT * FROM task_attempts").fetchall()
    assert len(attempts) == 1
    assert json.loads(attempts[0]["snapshot"])["task"]["session"] == task["session"]
    for name,digest in resumed["previous_artifacts_sha256"].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
    count = len(actions)
    with pytest.raises(ValueError):
        value.resume_bootstrap_preparation(task["id"])
    assert len(actions) == count


def prepared_live(tmp_path,monkeypatch):
    value,task,directory,state,actions,_,_ = study(tmp_path,mode="native_codex_vanilla")
    original_write = suite.write_json
    def crash(path,data):
        original_write(path,data)
        if Path(path) == directory/"bootstrap-recovery.json" and data.get("phase") == "prepared":
            raise KeyboardInterrupt("prepared transaction awaiting local reconciliation")
    monkeypatch.setattr(suite,"write_json",crash)
    with pytest.raises(KeyboardInterrupt):
        value.recover_bootstrap(task["id"],REASON)
    monkeypatch.setattr(suite,"write_json",original_write)
    return value,task,directory,state,actions


@pytest.mark.parametrize("corruption",[
    "input","failure_receipt","manifest","new_model_intent","remote_missing","remote_session",
    "remote_hardware","journal_job","new_quarantine",
])
def test_prepared_resume_refuses_changed_proofs_new_model_intent_or_unknown_owner(tmp_path,monkeypatch,corruption):
    value,task,directory,state,actions = prepared_live(tmp_path,monkeypatch)
    if corruption == "input":
        (directory/"generation/evidence/search-data/input.bin").write_bytes(b"changed")
    elif corruption == "failure_receipt":
        (directory/"failure.json").write_text("changed")
    elif corruption == "manifest":
        path = directory/"bootstrap-replay-manifest.json"; path.write_bytes(path.read_bytes()+b" ")
    elif corruption == "new_model_intent":
        path = directory/"generation/model/call-intent.json"; path.parent.mkdir(); path.write_text("{}")
    elif corruption == "remote_missing":
        state["remote"].pop("job_id.json")
    elif corruption == "remote_session":
        state["remote"]["job_id.json"]["session_id"] = "unknown-owner"
    elif corruption == "remote_hardware":
        state["remote"]["ready.json"]["hardware"]["device_properties"]["uuid"] = "different-gpu"
    elif corruption == "journal_job":
        value.journal.update(task["id"],job_id="771999")
    else:
        state["remote"]["quarantined.json"] = {"context_reusable":False}
    with pytest.raises((ValueError,BackendError)):
        value.resume_bootstrap_preparation(task["id"])
    row = value.journal.tasks("generation")[0]
    assert row["state"] == "recovery_required" and row["attempt"] == 0
    assert json.loads((directory/"bootstrap-recovery.json").read_text())["phase"] == "prepared"
    assert not (directory/"bootstrap-archives/000/generation").exists()
    assert all(action[0] in {"backend","read","accounting"} for action in actions)


@pytest.mark.parametrize("status",["numerical_error","memory_error","compilation_error","contract_violation"])
def test_transport_label_cannot_erase_a_failed_native_or_correctness_report(tmp_path,status):
    value,task,directory,_,actions,_,_ = study(tmp_path)
    path = directory/"request-receipts/0.json"
    row = json.loads(path.read_text()); row["report"].update(ok=False,status=status)
    suite.write_json(path,row)
    with pytest.raises(ValueError,match="cannot be retried as transport bootstrap"):
        value.recover_bootstrap(task["id"],REASON)
    assert actions == [] and not (directory/"bootstrap-recovery.json").exists()
