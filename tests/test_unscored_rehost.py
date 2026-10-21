"""Unscored-only context transfer; fake accounting, never SSH/model/GPU calls."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from optimizer_agent import unscored_rehost as module
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.evaluation import Candidate,LaunchRule,WorkloadContract
from optimizer_agent.execution_backend import SlurmConfig
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.search_policy import SearchPolicy
from optimizer_agent.suite_orchestrator import SuiteOrchestrator,config_from_dict,write_json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def no_activity(*args,**kwargs):
    pytest.fail("Preparation/identity binding cannot submit, invoke a model or execute GPU work")


def checkpoint(path,run_id,trial_id,candidate_id):
    serde = JsonPlusSerializer()
    state = {"run_id":run_id,"initialized":True,"trial_id":trial_id,"candidate_id":candidate_id,
        "repair":1,"research_round":1,"trials_done":0,"valid":False,"retryable":True,
        "research_ready":True,"needs_research":False,"diagnostic_id":None,"branch:to:generate":None}
    with sqlite3.connect(path) as db:
        db.executescript("""CREATE TABLE checkpoints(thread_id TEXT,checkpoint_ns TEXT,checkpoint_id TEXT,
            parent_checkpoint_id TEXT,type TEXT,checkpoint BLOB,metadata BLOB);
            CREATE TABLE writes(thread_id TEXT,checkpoint_ns TEXT,checkpoint_id TEXT,task_id TEXT,
            task_path TEXT,idx INTEGER,channel TEXT,type TEXT,value BLOB);""")
        for ns,cid in [("","checkpoint-parent"),("experiment:unit","checkpoint-leaf")]:
            kind,blob = serde.dumps_typed({"channel_values":state})
            db.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?)",(run_id,ns,cid,None,kind,blob,b"{}"))
        kind,error = serde.dumps_typed("AttributeError: list object has no attribute items")
        db.execute("INSERT INTO writes VALUES(?,?,?,?,?,?,?,?,?)",(run_id,"experiment:unit","checkpoint-leaf",
            "failed-generate","generate",-1,"__error__",kind,error))


def setup(tmp_path,*,model_calls=3):
    plan = plan_suite(CampaignConfig(run_id="unscored-test",search_sizes=(8,),final_sizes=(8,)),seeds=(1,2))
    plan["entries"] = [entry for entry in plan["entries"] if entry["workload"] == "matmul"
        and entry["seed"] == 1 and entry["mode"] == "no_autotuning"]
    plan["modes"] = ["no_autotuning"]
    actions,backend_state = [],{"status":"COMPLETED","quarantine":None}
    class Backend:
        def __init__(self,task,directory):
            self.task,self.results_dir = task,Path(directory)/"worker"
            self.config,self.session_id,self.job_id = SlurmConfig(),task["session"],None
        def attach_existing(self):
            actions.append("authenticate_terminal_ownership")
            receipt = json.loads((self.results_dir/"slurm-session.json").read_text())
            assert receipt["session_id"] == self.task["session"]
            self.job_id = receipt["job_id"]
            return {"job_id":self.job_id,"state":backend_state["status"],"terminal":backend_state["status"] != "RUNNING","exit_code":"0:0"}
        def _json(self,name,optional=False):
            actions.append(name)
            assert name == "quarantined.json"
            return backend_state["quarantine"]
        start = no_activity
        stage = no_activity
        runner = no_activity
        close = no_activity
    orchestrator = SuiteOrchestrator(plan,tmp_path,backend_factory=lambda t,c,d:Backend(t,d),generate=no_activity,qualify=no_activity)
    task = orchestrator.journal.tasks("generation")[0]
    config = config_from_dict(task["payload"]["config"])
    directory = orchestrator._directory(task); generation=directory/"generation"; generation.mkdir(parents=True)
    source = 'extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n){}'
    baseline_path=generation/"baseline.cu"; baseline_path.write_text(source)
    baseline=Candidate("baseline",baseline_path,"matmul",LaunchRule(kind="square_2d",block=(16,16,1)))
    baseline_state={"candidate_id":"baseline","source":source,"kernel":"matmul","launch_rule":asdict(baseline.launch_rule),
        "parameters":{},"artifact_hash":baseline.artifact_hash}
    environment={"uuid":"GPU-old","name":"SIMULATED_ONLY","architecture":"sm_90","harness_build_id":"unit"}
    latencies={"search-"+str(n):2.0 for n in config.search_sizes}
    store=ExperimentStore(generation/"experiments.sqlite")
    policy=SearchPolicy(store,config.run_id,baseline_state,latencies,actions=list(config.actions),
        contract_fingerprint="old-contract",environment_fingerprint=digest(environment),
        budget_limits={"trials":config.max_trials},seed=config.seed)
    trial=policy.ask()
    trial_dir=generation/"trials"/trial.trial_id; trial_dir.mkdir(parents=True)
    child_path=generation/"candidates/invalid.cu"; child_path.parent.mkdir(); child_path.write_text('#include <math.h>\n'+source)
    child=Candidate("invalid",child_path,"matmul",baseline.launch_rule)
    store.archive_candidate(config.run_id,{"candidate_id":"invalid","source":child_path.read_text(),"kernel":"matmul",
        "launch_rule":asdict(child.launch_rule),"parameters":{},"artifact_hash":child.artifact_hash,
        "tree_parent_id":"baseline","source_parent_id":"baseline","trial_id":trial.trial_id,"strategy_family":trial.action})
    evaluation={"ok":False,"status":"compilation_error","candidate_id":"invalid","artifact_hash":child.artifact_hash,
        "contract_hash":"old-contract","evaluation_id":"old-invalid-evaluation","cases":[{"report":{"ok":False,"compiler_log":"math.h not found"}}]}
    write_json(trial_dir/"evaluation-invalid.json",evaluation)
    attempt=store.new_attempt(config.run_id,trial.trial_id,{"candidate_id":"invalid"})
    store.finish_attempt(attempt,"compilation_error",{"path":str(trial_dir/"evaluation-invalid.json")})
    for index in range(model_calls):
        call=hashlib.sha256(str(index).encode()).hexdigest()[:32]; usage={"input_tokens":100+index,"output_tokens":10}
        write_json(generation/"model"/(call+".json"),{"call_id":call,"status":"completed","returncode":0,"usage":usage})
        store.reserve(config.run_id,"model-"+call,{"model_calls":1})
        store.reconcile(config.run_id,"model-"+call,{"model_calls":1,"model_tokens":sum(usage.values())})
        store.event(config.run_id,"model",{"call_id":call,"usage":usage,"purpose":"proposal"})
    store.reserve(config.run_id,"baseline-bootstrap",{"gpu_seconds":20})
    store.reconcile(config.run_id,"baseline-bootstrap",{"gpu_seconds":20})
    write_json(generation/"config.json",config.to_dict()); write_json(generation/"environment.json",environment)
    contract=asdict(WorkloadContract(config.run_id,1,"matmul","fixed ABI","CPU oracle",benchmark_series=2,
        launches_per_series=2,metadata_json=json.dumps({"environment":environment,"precision":"float32"})))
    write_json(generation/"contract.json",{"contract":contract,"contract_hash":"old-contract"})
    write_json(generation/"baseline-measurement.json",{"ok":True,"evaluation_id":"old-baseline","contract_hash":"old-contract",
        "cases":{key:{"latency_ms":value} for key,value in latencies.items()}})
    write_json(trial_dir/"dossier.json",{"research_id":"original-dossier","workload_hash":"old-contract",
        "environment_fingerprint":digest(environment),"strategy_family":trial.action})
    write_json(trial_dir/"trial.json",{"trial_id":trial.trial_id,"action":trial.action,"parent":baseline_state,"selected_at":1})
    checkpoint(generation/"langgraph.sqlite",config.run_id,trial.trial_id,"invalid")
    failure={"job_id":"1234","error":"'list' object has no attribute 'items'",
        "traceback":"evaluation_summary: for case_id, case in value.get('cases', {}).items()"}
    write_json(directory/"failure.json",failure)
    write_json(directory/"worker/slurm-session.json",{"session_id":task["session"],"job_id":"1234"})
    orchestrator.journal.update(task["id"],state="recovery_required",job_id="1234",result=failure,error=failure["error"])
    task=orchestrator.journal.tasks("generation")[0]
    return orchestrator,task,generation,store,policy,actions,backend_state


def prepare(value,task):
    return module.prepare_unscored_rehost(value,task["id"],"Repair the diagnosed list-cases diagnostic adapter without replaying completed model calls")


def fresh_baseline(audit):
    generation=Path(audit["new_generation"])
    old=audit["old_proof"]["identity"]
    environment={"uuid":"GPU-new","name":"SIMULATED_ONLY","architecture":"sm_90","harness_build_id":"unit"}
    identity={**old,"contract":"new-contract","environment":digest(environment),
        "baseline_latencies":{key:1.0 for key in old["baseline_latencies"]}}
    contract=json.loads((generation/"prior-context/contract.json").read_text())
    metadata=json.loads(contract["contract"]["metadata_json"]); metadata["environment"]=environment
    contract["contract"]["metadata_json"]=json.dumps(metadata); contract["contract_hash"]="new-contract"
    write_json(generation/"contract.json",contract); write_json(generation/"environment.json",environment)
    write_json(generation.parent/"worker/slurm-session.json",{"session_id":audit["new_session"],"job_id":"5678"})
    measurement={"ok":True,"status":"measured","evaluation_id":"fresh-baseline","candidate_id":"baseline",
        "environment":environment,
        "artifact_hash":audit["old_proof"]["baseline_artifact_hash"],"contract_hash":"new-contract",
        "cases":{key:{"latency_ms":1.0,"series":[{"raw_ms":[1.0,1.0],"report":{"ok":True,"slurm_job_id":"5678"}}
            for _ in range(2)]} for key in identity["baseline_latencies"]}}
    write_json(generation/"baseline-measurement.json",measurement)
    return ExperimentStore(generation/"experiments.sqlite"),identity


def test_prepare_is_pure_copies_coherent_databases_and_preserves_all_original_evidence(tmp_path):
    value,task,source,_,_,actions,_=setup(tmp_path)
    before=module._inventory(source)
    audit=prepare(value,task)
    destination=Path(audit["new_generation"])
    assert module._inventory(source)==before and audit["phase"]=="committed"
    assert actions==["authenticate_terminal_ownership","quarantined.json"]
    assert audit["new_job_submitted"] is False and audit["gpu_requests_during_preparation"]==audit["model_calls_during_preparation"]==0
    assert audit["scores_inherited"] is False and audit["fresh_research_required"]
    assert not (destination/"baseline-measurement.json").exists()
    assert not (destination/"trials"/audit["old_proof"]["checkpoint"]["state"]["trial_id"]/"dossier.json").exists()
    for name in audit["archived_context_files"]:
        assert (destination/"prior-context"/name).read_bytes()==(source/name).read_bytes()
    for name in ["experiments.sqlite","langgraph.sqlite"]:
        with sqlite3.connect(destination/name) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0]=="ok"
    assert module._unscored_proof(destination,task["id"])==audit["old_proof"]
    row=value.journal.tasks("generation")[0]
    assert row["attempt"]==1 and row["state"]=="pending" and row["session"]==audit["new_session"] and row["job_id"] is None
    assert row["result"] is None and row["payload"]["source_recovery"]["marker_path"]==audit["marker_path"]
    with value.journal.connection() as db:
        assert db.execute("SELECT count(*) FROM task_attempts").fetchone()[0]==1
    count=len(actions)
    with pytest.raises(module.RehostRefused):prepare(value,task)
    assert len(actions)==count


@pytest.mark.parametrize("calls",[1,4])
def test_completed_model_calls_are_preserved_without_an_arbitrary_call_count_cap(tmp_path,calls):
    value,task,_,_,_,_,_=setup(tmp_path,model_calls=calls)
    audit=prepare(value,task)
    assert audit["old_proof"]["model_calls"]==calls


@pytest.mark.parametrize("change",["evaluation","reward","integration","incumbent","model_uncertain","model_pending","cli_pending","root_score","rng","candidate"])
def test_activation_refuses_any_learning_or_changed_frozen_state(tmp_path,change):
    value,task,_,_,_,_,_=setup(tmp_path)
    audit=prepare(value,task); store,identity=fresh_baseline(audit)
    with store.transaction() as db:
        if change=="evaluation":db.execute("INSERT INTO evaluations VALUES(?,?,?)",(task["id"],"score","{}"))
        elif change=="reward":db.execute("INSERT INTO reward_claims VALUES(?,?,?)",(task["id"],"score","trial"))
        elif change=="integration":db.execute("INSERT INTO integration_events VALUES(?,?,?)",(task["id"],"trial",1))
        elif change=="incumbent":db.execute("UPDATE campaigns SET incumbent='invalid',incumbent_version=1")
        elif change in {"root_score","rng"}:
            snapshot=pickle.loads(db.execute("SELECT snapshot FROM campaigns").fetchone()[0])
            if change=="root_score":snapshot["tree"].tree.root.score=.7
            else:snapshot["rng"][1][0]^=1
            db.execute("UPDATE campaigns SET snapshot=?",(pickle.dumps(snapshot),))
        elif change=="candidate":
            row=json.loads(db.execute("SELECT payload FROM candidates WHERE candidate_id='invalid'").fetchone()[0]); row["source"]+="changed"
            db.execute("UPDATE candidates SET payload=? WHERE candidate_id='invalid'",(json.dumps(row),))
    if change=="model_uncertain":store.event(task["id"],"model_uncertain",{})
    elif change=="model_pending":store.reserve(task["id"],"model-unacknowledged",{"model_calls":1})
    elif change=="cli_pending":
        path=next((Path(audit["new_generation"])/"model").glob("*.json")); row=json.loads(path.read_text()); row["status"]="running"; write_json(path,row)
    with pytest.raises(module.RehostRefused):
        module.verified_rehost_identity(store,task["id"],audit["marker_path"],identity,{"trials":3})
    with store.connection() as db:
        assert json.loads(db.execute("SELECT identity FROM campaigns").fetchone()[0])["contract"]=="old-contract"


def test_first_activation_changes_only_identity_then_is_idempotent_even_after_learning(tmp_path):
    value,task,_,_,_,_,_=setup(tmp_path)
    audit=prepare(value,task); store,identity=fresh_baseline(audit)
    tables=["trials","candidates","reservations","attempts"]
    with store.connection() as db:
        before={name:module._rows_hash(db,name) for name in tables}
        snapshot=db.execute("SELECT snapshot FROM campaigns").fetchone()[0]
    proof=module.verified_rehost_identity(store,task["id"],audit["marker_path"],identity,{"trials":3})
    assert proof["old_scores_inherited"] is False and proof["new_identity"]["baseline_latencies"]!=proof["old_identity"]["baseline_latencies"]
    with store.connection() as db:
        assert all(module._rows_hash(db,name)==before[name] for name in tables)
        assert db.execute("SELECT snapshot FROM campaigns").fetchone()[0]==snapshot
    store.add_evaluation(task["id"],{"evaluation_id":"fresh-future-evaluation","ok":True})
    assert module.verified_rehost_identity(store,task["id"],audit["marker_path"],identity,{"trials":3})==proof
    changed={**identity,"environment":"second-GPU"}
    with pytest.raises(module.RehostRefused):
        module.verified_rehost_identity(store,task["id"],audit["marker_path"],changed,{"trials":3})
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM events WHERE kind='unscored_context_rebound'").fetchone()[0]==1
    marker=json.loads(Path(audit["marker_path"]).read_text()); marker["reason"]="changed origin story"
    write_json(audit["marker_path"],marker)
    with pytest.raises(module.RehostRefused):
        module.verified_rehost_identity(store,task["id"],audit["marker_path"],identity,{"trials":3})


@pytest.mark.parametrize("change",["old_receipt","wrong_job","wrong_artifact","wrong_protocol","wrong_seed","wrong_actions","missing_baseline"])
def test_activation_requires_fresh_owned_baseline_and_unchanged_semantic_contract(tmp_path,change):
    value,task,_,_,_,_,_=setup(tmp_path)
    audit=prepare(value,task); store,identity=fresh_baseline(audit); root=Path(audit["new_generation"])
    if change in {"old_receipt","wrong_job","wrong_artifact"}:
        row=json.loads((root/"baseline-measurement.json").read_text())
        if change=="old_receipt":row["evaluation_id"]="old-baseline"
        elif change=="wrong_artifact":row["artifact_hash"]="modified-source"
        else:row["cases"][next(iter(row["cases"]))]["series"][0]["report"]["slurm_job_id"]="1234"
        write_json(root/"baseline-measurement.json",row)
    elif change=="wrong_protocol":
        row=json.loads((root/"contract.json").read_text()); row["contract"]["warmup"]+=1; write_json(root/"contract.json",row)
    elif change=="wrong_seed":
        row=json.loads((root/"config.json").read_text()); row["seed"]+=1; write_json(root/"config.json",row)
    elif change=="wrong_actions":identity["actions"]=["replacement"]
    else:(root/"baseline-measurement.json").unlink()
    with pytest.raises(module.RehostRefused):
        module.verified_rehost_identity(store,task["id"],audit["marker_path"],identity,{"trials":3})


@pytest.mark.parametrize("change",["running","quarantined","wrong_failure","completed_repair"])
def test_prepare_refuses_unsafe_or_ambiguous_sources_without_new_attempt(tmp_path,change):
    value,task,generation,_,_,_,state=setup(tmp_path)
    if change=="running":state["status"]="RUNNING"
    elif change=="quarantined":state["quarantine"]={"context_reusable":False}
    elif change=="wrong_failure":
        row=json.loads((generation.parent/"failure.json").read_text()); row["error"]="model timeout"; write_json(generation.parent/"failure.json",row)
    else:
        trial=next((generation/"trials").iterdir()); write_json(trial/"proposal-1-1.json",{"complete":"unknown prior model result"})
    with pytest.raises(module.RehostRefused):prepare(value,task)
    assert value.journal.tasks("generation")[0]["attempt"]==0


def test_preparation_intent_precedes_copy_and_a_crash_never_creates_a_second_attempt(tmp_path,monkeypatch):
    value,task,source,_,_,_,_=setup(tmp_path)
    before=module._inventory(source)
    def interrupted(*args):
        assert (value.directory/"unscored-rehost-intents"/(task["id"]+".json")).is_file()
        raise KeyboardInterrupt("controller stopped before copying")
    monkeypatch.setattr(module,"_copy_generation",interrupted)
    with pytest.raises(KeyboardInterrupt):prepare(value,task)
    assert module._inventory(source)==before and value.journal.tasks("generation")[0]["attempt"]==0
    with pytest.raises(module.RehostRefused):prepare(value,task)


@pytest.mark.parametrize("boundary",["before_copy","mid_copy","after_moves","after_db","before_ack"])
def test_resume_prepared_transfer_at_each_crash_boundary_is_pure_and_commits_one_attempt(tmp_path,monkeypatch,boundary):
    value,task,source,_,_,actions,_=setup(tmp_path)
    original_inventory=module._inventory(source)
    copy=module._copy_generation; commit=module._commit_preparation; write=module._write
    def interrupted_copy(old,new):
        if boundary=="mid_copy":
            new.mkdir(parents=True)
            (new/"baseline.cu").write_bytes((old/"baseline.cu").read_bytes()[:19])
        raise KeyboardInterrupt("during safe copying")
    def interrupted_commit(*args):
        raise KeyboardInterrupt("after copied context was archived")
    def interrupted_write(path,record):
        if record.get("phase")=="committed":
            if boundary=="after_db" or Path(path).parent.name=="unscored-rehost-intents":
                raise KeyboardInterrupt("after journal commit before acknowledgment")
        return write(path,record)
    if boundary in {"before_copy","mid_copy"}:monkeypatch.setattr(module,"_copy_generation",interrupted_copy)
    elif boundary=="after_moves":monkeypatch.setattr(module,"_commit_preparation",interrupted_commit)
    else:monkeypatch.setattr(module,"_write",interrupted_write)
    with pytest.raises(KeyboardInterrupt):prepare(value,task)
    intent=value.directory/"unscored-rehost-intents"/(task["id"]+".json")
    planned=json.loads(intent.read_text()); assert planned["phase"]=="prepared"
    monkeypatch.setattr(module,"_copy_generation",copy)
    monkeypatch.setattr(module,"_commit_preparation",commit)
    monkeypatch.setattr(module,"_write",write)
    audit=module.resume_unscored_preparation(value,task["id"])
    assert audit["phase"]=="committed" and audit["context_id"]==planned["context_id"]
    assert audit["new_session"]==planned["new_session"] and audit["new_attempt"]==planned["new_attempt"]==1
    assert module._inventory(source)==original_inventory
    assert module._unscored_proof(Path(audit["new_generation"]),task["id"])==audit["old_proof"]
    assert audit["model_calls_during_preparation"]==audit["gpu_requests_during_preparation"]==0
    assert actions==["authenticate_terminal_ownership","quarantined.json"]*2
    if boundary=="mid_copy":
        archive=Path(audit["interrupted_copy_archives"][0])
        assert archive.name=="interrupted-copy-"+audit["context_id"]
        assert (archive/"baseline.cu").read_bytes()==(source/"baseline.cu").read_bytes()[:19]
    with value.journal.connection() as db:
        snapshots=[json.loads(row[0]) for row in db.execute("SELECT snapshot FROM task_attempts")]
        assert len(snapshots)==1 and snapshots[0]["task"]==task
    current=value.journal.tasks("generation")[0]
    assert current["state"]=="pending" and current["attempt"]==1 and current["job_id"] is None
    assert current["session"]==planned["new_session"]
    count=len(actions)
    with pytest.raises(module.RehostRefused):module.resume_unscored_preparation(value,task["id"])
    assert len(actions)==count


@pytest.mark.parametrize("change",["source","copied_source","new_model","new_worker","journal","quarantine","live","job"])
def test_resume_preparation_refuses_changed_or_active_evidence(tmp_path,monkeypatch,change):
    value,task,source,_,_,actions,state=setup(tmp_path)
    commit=module._commit_preparation
    monkeypatch.setattr(module,"_commit_preparation",lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):prepare(value,task)
    monkeypatch.setattr(module,"_commit_preparation",commit)
    intent=value.directory/"unscored-rehost-intents"/(task["id"]+".json")
    audit=json.loads(intent.read_text()); destination=Path(audit["new_generation"])
    if change=="source":(source/"baseline.cu").write_text("modified original")
    elif change=="copied_source":(destination/"baseline.cu").write_text("modified complete copied source")
    elif change=="new_model":write_json(destination/"model/new-call.json",{"status":"running"})
    elif change=="new_worker":write_json(destination.parent/"worker/slurm-session.json",{"job_id":"9999","session_id":audit["new_session"]})
    elif change=="journal":value.journal.update(task["id"],state="running")
    elif change=="quarantine":state["quarantine"]={"context_reusable":False}
    elif change=="live":state["status"]="RUNNING"
    else:write_json(source.parent/"worker/slurm-session.json",{"job_id":"different-owned-job","session_id":task["session"]})
    with pytest.raises(module.RehostRefused):module.resume_unscored_preparation(value,task["id"])
    assert json.loads(intent.read_text())["phase"]=="prepared"
    with value.journal.connection() as db:
        assert db.execute("SELECT count(*) FROM task_attempts").fetchone()[0]==0


@pytest.mark.parametrize("which",["controller","task"])
def test_resume_respects_exclusive_controller_and_generation_locks(tmp_path,monkeypatch,which):
    import fcntl
    value,task,_,_,_,actions,_=setup(tmp_path)
    copy=module._copy_generation
    monkeypatch.setattr(module,"_copy_generation",lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):prepare(value,task)
    monkeypatch.setattr(module,"_copy_generation",copy)
    lock=(value.directory/"controller.lock" if which=="controller" else
        value.directory/"task-locks"/(hashlib.sha256(task["id"].encode()).hexdigest()+".lock"))
    count=len(actions)
    with lock.open("a+") as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises((module.RehostRefused,RuntimeError)):
            module.resume_unscored_preparation(value,task["id"])
    assert len(actions)==count
