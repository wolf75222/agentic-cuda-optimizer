"""Synthetic cold recovery/export tests: no real study DB, GPU, SSH or model."""
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import zipfile

import pytest

from optimizer_agent import campaign, evaluation_recovery as recovery
from optimizer_agent.portable_export import export_study
from test_evaluation_recovery import uncertain, prepare, resumed_runtime
from test_portable_export import repo


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_bytes(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob("*")
            if p.is_file() and not p.name.endswith(("-wal","-shm","-journal"))}


def concluded(uncertain, *, pending_tip=False, second_recovery=False, references=False):
    suite, task, old, state, config, runner, model, request_id, worker, actions = uncertain
    if references:
        body=old.directory/"research/sources/downloaded.txt"
        passage=old.directory/"research/passages/passage.txt"
        supplied=old.directory/"research/fixture-supplied-content.json"
        body.parent.mkdir(parents=True,exist_ok=True);passage.parent.mkdir(parents=True,exist_ok=True)
        body.write_text("OMITTED_EXTERNAL_BODY_CPU_FIXTURE")
        passage.write_text("OMITTED_EXTERNAL_PASSAGE_CPU_FIXTURE")
        campaign.write_json(supplied,{"read_text":"OMITTED_SYNTHESIS_INPUT_CPU_FIXTURE"})
        campaign.write_json(old.directory/"research/reference-metadata.json",{"source_id":"fixture-source","kind":"code",
            "url":"https://example.invalid/pinned-fixture","commit_or_version":"fixture-v1","content_sha256":sha(body),
            "cache_path":str(body),"passages":[{"passage_id":"fixture-passage","sha256":sha(passage),"cache_path":str(passage)}]})
        (old.directory/"closed-fixture.lock").write_text("LOCK_STATE_BYTES_NOT_REDISTRIBUTED_CPU_FIXTURE")
    receipt = prepare(uncertain)
    runtime = resumed_runtime(uncertain)
    if second_recovery:
        original = runtime.evaluator.benchmark
        runtime.evaluator.benchmark = lambda *a,**k: (_ for _ in ()).throw(KeyboardInterrupt("CPU fixture second interruption"))
        with pytest.raises(KeyboardInterrupt):runtime.evaluate(state)
        runtime.evaluator.benchmark = original
        request,_=runtime._evaluation_request(state,runtime.candidate(state["candidate_id"]))
        request_id=campaign.digest(recovery.recovery_request(runtime.directory,request))
        previous=suite.journal.task(task["id"])
        backend=suite._backend(previous,config,suite._directory(previous))
        campaign.write_json(backend.results_dir/"slurm-session.json",{"session_id":previous["session"],"job_id":"102",
            "remote_dir":backend.remote_dir,"config":asdict(backend.config)})
        suite.journal.update(task["id"],state="recovery_required",job_id="102")
        worker["job"]="102"
        receipt=suite.recover_evaluation(task["id"],request_id,"Second declared CPU fixture interruption with no fabricated score")
        runtime=resumed_runtime(uncertain)
    measured = runtime.evaluate(state)
    runtime.integrate({**state, **measured})
    if pending_tip:
        runtime.select({"trials_done":1})
    runtime.finalize({"trials_done":1})
    current = suite.journal.task(task["id"])
    directory = suite._directory(current)
    backend = suite._backend(current,config,directory)
    current_job="103" if second_recovery else "102"
    campaign.write_json(directory/"worker/slurm-session.json",{"session_id":current["session"],"job_id":current_job,
        "remote_dir":backend.remote_dir,"config":asdict(backend.config)})
    result = {"status":"generation_complete","generation_complete":True,"execution_complete":True,
        "method_status":"failed" if pending_tip else "completed","session_id":current["session"],"job_id":current_job,
        "worker_terminal":{"terminal":True,"state":"COMPLETED","exit_code":"0:0","job_id":current_job}}
    campaign.write_json(directory/"result.json",result)
    suite.journal.update(current["id"],state="completed",job_id=current_job,result=result)
    # The declared fake harness request makes the retained baseline portable.
    # It is fixture evidence only; no correctness/performance observation.
    request = asdict(runtime.suite.cases()[-1].request)
    request.update(source=str(runtime.directory/"best.cu"),kernel=config.workload,mode="run",
        **runtime.baseline.launch_rule.apply({"n":config.search_sizes[-1]}))
    campaign.write_json(directory/"request-receipts/fixture.json",{"request":request,"report":{"ok":True,"scope":"CPU fixture only"}})
    final = suite.journal.tasks("final")[0]
    suite.journal.update(final["id"],state="completed")
    return suite,task,old,runtime,current,receipt


def test_authenticated_stopped_ancestor_pending_remains_archived_but_does_not_mean_live_work(uncertain,repo,tmp_path):
    suite,task,old,runtime,current,receipt = concluded(uncertain)
    before = tree_bytes(suite.directory)
    manifest = export_study(suite.directory,tmp_path/"export",repo_dir=repo)
    assert manifest["status"]=="complete",manifest["errors"]
    assert manifest["inflight_search_trials"]==[]
    assert len(manifest["historical_search_trials"])==1
    row=manifest["historical_search_trials"][0]
    assert row["database"]==str((old.directory/"experiments.sqlite").relative_to(suite.directory))
    assert row["pending_trial_ids"] and row["classification"]=="historical_nonexecutable_ancestor"
    assert row["lineage"][0]["raw_source_database_sha256"]==sha(old.directory/"experiments.sqlite")
    assert manifest["pending_search_trial_observations"]==[row]
    with zipfile.ZipFile(tmp_path/"export/study-evidence.zip") as archive:
        readable=json.loads(archive.read("observability/"+row["database"]+".json"))
        assert readable["search_snapshots"][0]["pending_trials"]
        assert "evidence/"+row["database"] in archive.namelist()
    assert tree_bytes(suite.directory)==before


def test_two_authenticated_recoveries_preserve_each_pending_ancestor_and_exact_chain(uncertain,repo,tmp_path):
    suite,task,old,runtime,current,receipt=concluded(uncertain,second_recovery=True)
    before=tree_bytes(suite.directory)
    manifest=export_study(suite.directory,tmp_path/"export",repo_dir=repo)
    assert manifest["status"]=="complete",manifest["errors"]
    assert not manifest["inflight_search_trials"] and len(manifest["historical_search_trials"])==2
    assert sorted(len(row["lineage"]) for row in manifest["historical_search_trials"])==[1,2]
    assert tree_bytes(suite.directory)==before


@pytest.mark.parametrize("damage",[None,"body_tamper","held_lock"])
def test_omitted_reference_and_lock_hashes_authenticate_cold_ancestor_without_redistributing_text(uncertain,repo,tmp_path,damage):
    suite,task,old,runtime,current,receipt=concluded(uncertain,references=True)
    body=old.directory/"research/sources/downloaded.txt"
    if damage=="body_tamper":body.write_text(body.read_text()+" changed")
    lock=(old.directory/"closed-fixture.lock").open("rb")
    try:
        if damage=="held_lock":fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=tree_bytes(suite.directory)
        manifest=export_study(suite.directory,tmp_path/"export",repo_dir=repo)
        assert manifest["status"]==("complete" if damage is None else "partial"),manifest["errors"]
        assert bool(manifest["historical_search_trials"]) is (damage is None)
        assert bool(manifest["inflight_search_trials"]) is (damage is not None)
        omitted={row["original_path"]:row for row in manifest["reference_files_omitted"]}
        relative=str(body.relative_to(suite.directory))
        assert omitted[relative]["sha256"]==sha(body)
        locks={row["original_path"]:row for row in manifest["lock_files_omitted"]}
        assert locks[str((old.directory/"closed-fixture.lock").relative_to(suite.directory))]["sha256"]==sha(old.directory/"closed-fixture.lock")
        with zipfile.ZipFile(tmp_path/"export/study-evidence.zip") as archive:
            assert "evidence/"+relative not in archive.namelist()
            assert not any(name.endswith(".lock") for name in archive.namelist())
            for name in archive.namelist():
                assert b"OMITTED_EXTERNAL_BODY_CPU_FIXTURE" not in archive.read(name)
                assert b"OMITTED_SYNTHESIS_INPUT_CPU_FIXTURE" not in archive.read(name)
                assert b"LOCK_STATE_BYTES_NOT_REDISTRIBUTED_CPU_FIXTURE" not in archive.read(name)
            refs=json.loads(archive.read("reference-rehydration.json"))
            assert any(row["source"]["source_id"]=="fixture-source" for row in refs["sources"])
            assert refs["generation_guard_ready"] is False and refs["automatic_generation_resume_available"] is False
        assert tree_bytes(suite.directory)==before
    finally:lock.close()


@pytest.mark.parametrize("damage",["marker","preparation","copied_preparation","commit","source_inventory",
                                    "missing_ancestor","missing_database","tip_terminal","tip_session","task_not_closed"])
def test_uncertain_or_changed_recovery_authority_stays_partial_without_source_mutation(uncertain,repo,tmp_path,damage):
    suite,task,old,runtime,current,receipt=concluded(uncertain)
    marker=runtime.directory/"evaluation-recoveries"/(receipt["previous_request_id"]+".json")
    intent=Path(suite.journal.task(task["id"])["payload"]["source_recovery"]["preparation_path"])
    if damage=="source_inventory":
        path=old.directory/"config.json";path.write_bytes(path.read_bytes()+b" ")
    elif damage=="missing_ancestor":
        with suite.journal.connection() as db:db.execute("DELETE FROM task_attempts")
    elif damage=="missing_database":(old.directory/"experiments.sqlite").unlink()
    elif damage=="task_not_closed":suite.journal.update(task["id"],state="running")
    elif damage in ("tip_terminal","tip_session"):
        path=suite._directory(current)/"result.json";result=campaign.read_json(path)
        if damage=="tip_terminal":result["worker_terminal"]["job_id"]="foreign"
        else:result["session_id"]="foreign"
        campaign.write_json(path,result);suite.journal.update(task["id"],result=result)
    else:
        path={"marker":marker,"preparation":intent,
            "copied_preparation":marker.parent/("preparation-"+receipt["epoch_id"]+".json"),
            "commit":marker.parent/("committed-"+receipt["epoch_id"]+".json")}[damage]
        campaign.write_json(path,{**campaign.read_json(path),"tampered":True})
    before=tree_bytes(suite.directory)
    manifest=export_study(suite.directory,tmp_path/"export",repo_dir=repo)
    assert manifest["status"]=="partial" and (manifest["inflight_search_trials"] or manifest["errors"])
    assert manifest["historical_search_trials"]==[]
    assert tree_bytes(suite.directory)==before


@pytest.mark.parametrize("copy_kind",["interrupted-copy","unknown-database"])
def test_unmapped_cold_copies_never_inherit_the_ancestor_authorization(uncertain,repo,tmp_path,copy_kind):
    suite,task,old,runtime,current,receipt=concluded(uncertain)
    target=suite._directory(current)/copy_kind/"experiments.sqlite"
    target.parent.mkdir(parents=True)
    shutil.copy2(old.directory/"experiments.sqlite",target)
    before=tree_bytes(suite.directory)
    manifest=export_study(suite.directory,tmp_path/"export",repo_dir=repo)
    assert manifest["status"]=="partial" and len(manifest["historical_search_trials"])==1
    assert len(manifest["inflight_search_trials"])==1
    assert manifest["inflight_search_trials"][0]["database"]==str(target.relative_to(suite.directory))
    assert tree_bytes(suite.directory)==before


def test_failed_method_completed_task_with_tip_pending_is_still_partial_without_fake_tell(uncertain,repo,tmp_path):
    suite,task,old,runtime,current,receipt=concluded(uncertain,pending_tip=True)
    before=tree_bytes(suite.directory)
    manifest=export_study(suite.directory,tmp_path/"export",repo_dir=repo)
    assert manifest["status"]=="partial" and len(manifest["historical_search_trials"])==1
    assert len(manifest["inflight_search_trials"])==1
    assert manifest["inflight_search_trials"][0]["database"]==str((runtime.directory/"experiments.sqlite").relative_to(suite.directory))
    assert suite.journal.task(task["id"])["result"]["method_status"]=="failed"
    assert tree_bytes(suite.directory)==before


def test_live_ancestor_is_not_historical_even_when_preparation_and_commits_agree(uncertain,repo,tmp_path):
    suite,task,old,runtime,current,receipt=concluded(uncertain)
    current=suite.journal.task(task["id"])
    prep=Path(current["payload"]["source_recovery"]["preparation_path"])
    audit=campaign.read_json(prep)
    audit["marker"]["previous_worker_observation"]={"state":"RUNNING","terminal":False,"job_id":"101"}
    marker=runtime.directory/"evaluation-recoveries"/(audit["request_id"]+".json")
    # Only fixture provenance is changed below, never an actual recovery.
    campaign.write_json(marker,audit["marker"])
    campaign.write_json(prep,audit)
    copied=marker.parent/("preparation-"+audit["epoch_id"]+".json")
    copied.write_bytes(prep.read_bytes())
    pin=sha(prep)
    commit={"epoch_id":audit["epoch_id"],"marker_sha256":sha(marker),"preparation_sha256":pin,
            "new_session":current["session"],"new_task_attempt":current["attempt"]}
    campaign.write_json(prep.with_suffix(".committed.json"),commit)
    campaign.write_json(marker.parent/("committed-"+audit["epoch_id"]+".json"),commit)
    campaign.write_json(prep.with_suffix(".ready.json"),recovery._ready_record(audit,pin,sha(marker)))
    with suite.journal.connection() as db:
        row=db.execute("SELECT snapshot FROM task_attempts").fetchone()
        snapshot=json.loads(row[0]);snapshot["evaluation_recovery_preparation_sha256"]=pin
        db.execute("UPDATE task_attempts SET snapshot=?",(json.dumps(snapshot),))
        payload=current["payload"];payload["source_recovery"]["preparation_sha256"]=pin
        db.execute("UPDATE tasks SET payload=? WHERE id=?",(json.dumps(payload),task["id"]))
    before=tree_bytes(suite.directory)
    manifest=export_study(suite.directory,tmp_path/"export",repo_dir=repo)
    assert manifest["status"]=="partial" and not manifest["historical_search_trials"]
    assert any("terminal" in row["classification_reason"] for row in manifest["inflight_search_trials"])
    assert tree_bytes(suite.directory)==before


@pytest.mark.parametrize("damage",[None,"marker"])
def test_real_unscored_transfer_authority_can_classify_only_its_exact_old_snapshot(tmp_path,repo,damage):
    from copy import copy
    from optimizer_agent.experiment_store import ExperimentStore
    from optimizer_agent.evaluation import Candidate,LaunchRule
    from optimizer_agent.execution_backend import SlurmConfig
    from test_unscored_rehost import setup,prepare as unscored_prepare
    suite,task,old,store,policy,actions,_=setup(tmp_path/"study")
    config=SlurmConfig()
    campaign.write_json(old.parent/"worker/slurm-session.json",{"session_id":task["session"],"job_id":"1234",
        "remote_dir":config.remote_root+"/sessions/"+task["session"],"config":asdict(config)})
    audit=unscored_prepare(suite,task)
    current=suite.journal.task(task["id"]);generation=Path(audit["new_generation"])
    # End the copied fixture trial explicitly without any reward or GPU score.
    resumed_policy=copy(policy);resumed_policy.store=ExperimentStore(generation/"experiments.sqlite")
    trial=resumed_policy.pending_trials()[0]
    resumed_policy.suspend_or_cancel(trial.trial_id,"Declared CPU fixture method stop; no fabricated reward")
    source=generation/"best.cu";shutil.copy2(generation/"baseline.cu",source)
    candidate=Candidate("baseline",source,"matmul",LaunchRule(kind="square_2d",block=(16,16,1)))
    campaign.write_json(generation/"candidate.json",{"candidate_id":"baseline","kernel":"matmul","source_path":str(source),
        "source_hash":candidate._source_hash,"artifact_hash":candidate.artifact_hash,"launch_rule":asdict(candidate.launch_rule),"parameters":{}})
    directory=suite._directory(current)
    campaign.write_json(directory/"worker/slurm-session.json",{"session_id":current["session"],"job_id":"5678",
        "remote_dir":config.remote_root+"/sessions/"+current["session"],"config":asdict(config)})
    result={"generation_complete":True,"execution_complete":True,"method_status":"failed","session_id":current["session"],"job_id":"5678",
        "worker_terminal":{"terminal":True,"state":"COMPLETED","exit_code":"0:0","job_id":"5678"}}
    campaign.write_json(directory/"result.json",result)
    suite.journal.update(task["id"],state="completed",job_id="5678",result=result)
    final=suite.journal.tasks("final")[0];suite.journal.update(final["id"],state="completed")
    for name in ("a","b"):(generation/(name+".bin")).write_bytes(b"\0"*36)
    request={"source":str(source),"kernel":"matmul","mode":"run","device":0,"grid":[1,1,1],"block":[16,16,1],
        "shared_memory_bytes":0,"warmup":0,"iterations":1,"arguments":[
            {"kind":"buffer","bytes":36,"input_file":str(generation/"a.bin"),"direction":"input"},
            {"kind":"buffer","bytes":36,"input_file":str(generation/"b.bin"),"direction":"input"},
            {"kind":"buffer","bytes":36,"input_file":None,"direction":"output"},
            {"kind":"scalar","type":"int32","value":3}]}
    campaign.write_json(directory/"request-receipts/fixture.json",{"request":request,"report":{"scope":"No CUDA observation: CPU fixture only"}})
    if damage:
        marker=generation/"unscored-rehost.json"
        campaign.write_json(marker,{**campaign.read_json(marker),"tampered":True})
    before=tree_bytes(suite.directory)
    manifest=export_study(suite.directory,tmp_path/"export",repo_dir=repo)
    assert manifest["status"]==("partial" if damage else "complete"),manifest["errors"]
    assert bool(manifest["historical_search_trials"]) is (damage is None)
    assert bool(manifest["inflight_search_trials"]) is (damage is not None)
    assert tree_bytes(suite.directory)==before
