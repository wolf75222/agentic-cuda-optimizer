"""No scheduler, CUDA or model calls: exercise portable bytes and SQLite WAL."""
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import zipfile

import pytest

from optimizer_agent.evaluation import Candidate, LaunchRule
from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.portable_export import export_study,reconstruct_candidate,sqlite_backup
from optimizer_agent.search_policy import SearchPolicy


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path/"repo"
    repo.mkdir()
    subprocess.run(["git","init","-q",str(repo)],check=True)
    (repo/"README.md").write_text("Exact committed source\n")
    subprocess.run(["git","-C",str(repo),"add","README.md"],check=True)
    subprocess.run(["git","-C",str(repo),"-c","user.name=Unit Test","-c","user.email=unit@example.invalid","commit","-qm","fixture"],check=True)
    return repo


@pytest.fixture
def study(tmp_path):
    root = tmp_path/"study"
    group = root/"final-groups"/"scan-seed-1"
    group.mkdir(parents=True)
    source = group/"frozen"/"agent"/"kernel.cu"
    source.parent.mkdir(parents=True)
    source.write_text('#include "gain.cuh"\nextern "C" __global__ void scan(const float* a,const float* b,float* c,int n){int i=threadIdx.x;if(i<n)c[i]=a[i]*GAIN;}')
    header = source.parent/"gain.cuh"
    header.write_text("#define GAIN 1.0f\n")
    candidate = Candidate("selected-agent",source,"scan",LaunchRule(block=(128,1,1)),{"GAIN":1})
    receipt = {"candidate_id":candidate.candidate_id,"source_path":str(source),"source_hash":candidate._source_hash,
        "artifact_hash":candidate.artifact_hash,"kernel":"scan","launch_rule":asdict(candidate.launch_rule),"parameters":candidate.parameters}
    write(group/"frozen-candidates.json",{"agent":receipt})
    write(group/"result.json",{"contract":{"operation":"scan","oracle_provenance":"immutable independent test oracle"},
        "contract_hash":"frozen-contract","environment":{"uuid":"simulated-only"},"evaluation_finished":True})
    input_dir = group/"sealed-inputs"
    input_dir.mkdir()
    (input_dir/"a.bin").write_bytes(b"\x00\x00\x80\x3f"*17)
    (input_dir/"b.bin").write_bytes(b"\x00"*68)
    request = {"source":str(source),"kernel":"scan","mode":"benchmark","device":0,"grid":[1,1,1],"block":[128,1,1],
        "shared_memory_bytes":0,"warmup":0,"iterations":2,"compiler_options":["--fmad=true"],"dependencies":[str(header)],
        "arguments":[{"kind":"buffer","bytes":68,"input_file":str(input_dir/"a.bin"),"direction":"input"},
                     {"kind":"buffer","bytes":68,"input_file":str(input_dir/"b.bin"),"direction":"input"},
                     {"kind":"buffer","bytes":68,"input_file":None,"direction":"output"},
                     {"kind":"scalar","type":"int32","value":17}]}
    write(group/"request-receipts"/"real-request.json",{"request":request,"report":{"ok":True}})
    db = sqlite3.connect(root/"suite.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE tasks(id TEXT,kind TEXT,state TEXT,job_id TEXT)")
    db.execute("INSERT INTO tasks VALUES('generation','generation','completed','70001')")
    db.execute("INSERT INTO tasks VALUES('scan-seed-1','final','completed','70002')")
    db.execute("CREATE TABLE sealed_groups(id TEXT,seed TEXT,commitment TEXT)")
    db.execute("INSERT INTO sealed_groups VALUES('scan-seed-1','123456','precommitted')")
    db.commit()
    research = root/"campaigns"/"agent"/"research"
    write(research/"dossier.json",{"sources":[{"url":"https://docs.nvidia.com/cuda/","version":"12.6","content_hash":"body-hash","section_or_locator":"section 3"}]})
    (research/"sources").mkdir()
    (research/"sources"/"downloaded.txt").write_text("EXTERNAL_CORPUS_BODY_MUST_NOT_BE_REDISTRIBUTED")
    yield root,candidate,db
    db.close()


def test_export_complete_reconstructs_exact_artifact_options_dependencies_and_inputs(study,repo,tmp_path):
    root,candidate,db = study
    assert Path(str(root/"suite.sqlite")+"-wal").is_file()
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    assert manifest["status"] == "complete",manifest["errors"]
    assert not manifest["source"]["dirty"]
    assert manifest["database_snapshots"][0]["method"] == "sqlite.Connection.backup"
    assert len(manifest["candidates"]) == 1
    row = manifest["candidates"][0]
    assert row["contract_hash"] == "frozen-contract"
    assert row["environment"]["uuid"] == "simulated-only"
    rebuilt,requests,receipt = reconstruct_candidate(tmp_path/"output/study-evidence.zip",row["id"],tmp_path/"restored")
    assert rebuilt.artifact_hash == candidate.artifact_hash
    assert rebuilt.source.read_bytes() == candidate.source.read_bytes()
    assert requests[0].compiler_options == ["--fmad=true"]
    assert Path(requests[0].dependencies[0]).read_text() == "#define GAIN 1.0f\n"
    assert all(Path(argument.input_file).is_relative_to(tmp_path/"restored") for argument in requests[0].arguments if getattr(argument,"input_file",None))
    assert requests[0].arguments[0].bytes == Path(requests[0].arguments[0].input_file).stat().st_size
    with zipfile.ZipFile(tmp_path/"output/study-evidence.zip") as archive:
        assert not any(name.endswith(("-wal","-shm")) for name in archive.namelist())
        assert not any("downloaded.txt" in name for name in archive.namelist())
        assert "https://docs.nvidia.com/cuda/" in archive.read("evidence/campaigns/agent/research/dossier.json").decode()
        backup = tmp_path/"backup.sqlite"
        backup.write_bytes(archive.read("evidence/suite.sqlite"))
        with sqlite3.connect(backup) as check:
            assert check.execute("SELECT count(*) FROM tasks").fetchone()[0] == 2
    assert not json.loads((tmp_path/"restored/reconstruction.json").read_text())["executed"]


def test_active_controller_and_observed_job_are_protected_partial_not_certified(study,repo,tmp_path):
    root,candidate,db = study
    db.execute("UPDATE tasks SET state='running' WHERE kind='final'")
    db.commit()
    with (root/"controller.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = export_study(root,tmp_path/"partial",repo_dir=repo)
    assert manifest["status"] == "partial"
    assert manifest["controller_or_worker_lock_observed_active"]
    assert manifest["active_jobs_observed"][0]["job_id"] == "70002"
    assert manifest["private_partial_snapshot"]
    assert (tmp_path/"partial").stat().st_mode & 0o777 == 0o700
    assert (tmp_path/"partial/study-evidence.zip").stat().st_mode & 0o777 == 0o600
    assert "No certification" in manifest["certification"]


def test_readable_tree_rewards_closures_and_checkpoint_databases_are_preserved(study,repo,tmp_path):
    root,candidate,db = study
    store = ExperimentStore(root/"campaigns"/"agent"/"experiments.sqlite")
    baseline = {"candidate_id":"baseline","source":"baseline source","valid":True}
    policy = SearchPolicy(store,"agent",baseline,{"case":2.0},actions=["tiling"],seed=7)
    trial = policy.ask()
    policy.tell_once(trial.trial_id,{"evaluation_id":"measured-once","valid":True,"latencies":{"case":1.0},"purpose":"tree_reward","fresh":True,
        "state":{"candidate_id":"child","source":"child source","valid":True}})
    policy.close("strategy_family","tiling","no supported variant",evidence=["measured-once"])
    checkpoint = root/"campaigns"/"agent"/"langgraph.sqlite"
    with sqlite3.connect(checkpoint) as check:
        check.execute("CREATE TABLE checkpoints(thread_id TEXT,state BLOB)")
        check.execute("INSERT INTO checkpoints VALUES('agent',?)",(b"exact-checkpoint-bytes",))
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    with zipfile.ZipFile(tmp_path/"output/study-evidence.zip") as archive:
        readable = json.loads(archive.read("observability/campaigns/agent/experiments.sqlite.json"))
        tree = readable["search_snapshots"][0]
        assert len(tree["nodes"]) == 2
        assert tree["eligibility"]["decisions"][0]["reason"] == "no supported variant"
        assert readable["tables"]["reward_claims"][0]["evaluation_id"] == "measured-once"
        assert readable["tables"]["integration_events"][0]["processed"] == 1
        assert "evidence/campaigns/agent/langgraph.sqlite" in archive.namelist()


def test_dirty_source_is_reported_and_missing_request_makes_export_partial(study,repo,tmp_path):
    root,candidate,db = study
    (repo/"README.md").write_text("Uncommitted changed source\n")
    (root/"final-groups/scan-seed-1/request-receipts/real-request.json").unlink()
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    assert manifest["status"] == "partial"
    assert manifest["source"]["dirty"]
    assert not manifest["candidates"][0]["reconstructable"]
    assert any("real request" in row["error"] for row in manifest["errors"])
    with zipfile.ZipFile(tmp_path/"output/study-evidence.zip") as archive:
        assert b"Uncommitted changed source" in archive.read("source/working-tree.patch")


@pytest.mark.parametrize("member_kind",["source","input","dependency"])
def test_changed_candidate_bytes_cannot_be_exported_or_reconstructed(study,repo,tmp_path,member_kind):
    root,candidate,db = study
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    selected = manifest["candidates"][0]
    target = selected["source"]
    if member_kind == "input":
        target = next(name for name in manifest["files"] if name.startswith(selected["prefix"]+"inputs/"))
    elif member_kind == "dependency":
        target = next(name for name in selected["source_files"].values() if name != selected["source"])
    with zipfile.ZipFile(tmp_path/"output/study-evidence.zip") as incoming,zipfile.ZipFile(tmp_path/"tampered.zip","w") as outgoing:
        for name in incoming.namelist():
            outgoing.writestr(name,b"altered" if name == target else incoming.read(name))
    with pytest.raises(ValueError,match="checksum mismatch"):
        reconstruct_candidate(tmp_path/"tampered.zip",selected["id"],tmp_path/"rejected")
    candidate.source.write_text("extern C changed source")
    rejected = export_study(root,tmp_path/"changed",repo_dir=repo)
    assert rejected["status"] == "partial" and not rejected["candidates"]
    assert any("frozen hashes" in row["error"] for row in rejected["errors"])


def test_sqlite_backup_contains_committed_wal_but_excludes_uncommitted_transaction(tmp_path):
    path = tmp_path/"live.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE evidence(value TEXT)")
        db.execute("INSERT INTO evidence VALUES('committed')")
        db.commit()
        db.execute("INSERT INTO evidence VALUES('not committed')")
        sqlite_backup(path,tmp_path/"backup.sqlite")
        with sqlite3.connect(tmp_path/"backup.sqlite") as result:
            assert result.execute("SELECT value FROM evidence").fetchall() == [("committed",)]


def test_cli_help_requires_no_model_or_gpu():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable,str(repo/"scripts/export_study.py"),"--help"],cwd=repo,capture_output=True,text=True)
    assert result.returncode == 0 and "reconstruct" in result.stdout


def test_include_path_is_rebased_with_original_option_hash_retained(study,repo,tmp_path):
    root,candidate,db = study
    path = root/"final-groups/scan-seed-1/request-receipts/real-request.json"
    document = json.loads(path.read_text())
    original = "--include-path="+str(candidate.source.parent)
    document["request"]["compiler_options"].append(original)
    write(path,document)
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    row = manifest["candidates"][0]
    rebuilt,requests,_ = reconstruct_candidate(tmp_path/"output/study-evidence.zip",row["id"],tmp_path/"restored")
    assert row["requests"][0]["compiler_options"][-1] == original
    rebased = requests[0].compiler_options[-1].split("=",1)[1]
    assert Path(rebased).is_relative_to(tmp_path/"restored")
    assert (Path(rebased)/"gain.cuh").read_text() == "#define GAIN 1.0f\n"
    assert rebuilt.artifact_hash == candidate.artifact_hash


def test_actual_original_commit_snapshot_and_current_bridge_are_archived(study,tmp_path):
    from optimizer_agent.portable_export import REPO,UPSTREAM_COMMIT
    root,candidate,db = study
    manifest = export_study(root,tmp_path/"output")
    pinned = manifest["source"]["upstream_original"]
    assert pinned["commit"] == UPSTREAM_COMMIT and "status" not in pinned
    with zipfile.ZipFile(tmp_path/"output/study-evidence.zip") as archive:
        path = "optimizer_agent/workflow.py"
        payload = archive.read("source/upstream-original/"+path)
        expected = subprocess.check_output(["git","-C",str(REPO),"show",UPSTREAM_COMMIT+":"+path])
        assert payload == expected
        assert archive.read("source/bridge-working-tree/optimizer_agent/upstream_original.py") == (REPO/"optimizer_agent/upstream_original.py").read_bytes()


def test_missing_terminal_job_and_pending_tree_trials_remain_explicit(study,repo,tmp_path):
    root,candidate,db = study
    write(root/"worker/slurm-session.json",{"job_id":"79999","session_id":"owned-session"})
    store = ExperimentStore(root/"campaigns"/"agent"/"experiments.sqlite")
    policy = SearchPolicy(store,"agent",{"candidate_id":"baseline","source":"source","valid":True},{"case":1.0},actions=["tiling"])
    pending = policy.ask()
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    assert manifest["status"] == "partial"
    assert manifest["active_jobs_observed"][0]["job_id"] == "79999"
    assert manifest["inflight_search_trials"][0]["pending_trial_ids"] == [pending.trial_id]


def test_archive_manifest_cannot_redirect_reconstruction_to_host_paths(study,repo,tmp_path):
    root,candidate,db = study
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    selected = manifest["candidates"][0]
    with zipfile.ZipFile(tmp_path/"output/study-evidence.zip") as incoming,zipfile.ZipFile(tmp_path/"escape.zip","w") as outgoing:
        for name in incoming.namelist():
            data = incoming.read(name)
            if name == "export-manifest.json":
                value = json.loads(data)
                value["candidates"][0]["source"] = "../../outside.cu"
                data = json.dumps(value).encode()
            outgoing.writestr(name,data)
    with pytest.raises(ValueError,match="Unsafe archive path"):
        reconstruct_candidate(tmp_path/"escape.zip",selected["id"],tmp_path/"rejected")


def test_fresh_cli_exports_and_reconstructs_without_pytest_path_setup(study,tmp_path):
    from optimizer_agent.portable_export import REPO
    root,candidate,db = study
    script = REPO/"scripts/export_study.py"
    command = [sys.executable,str(script)]
    export = subprocess.run(command+["export","--study",str(root),"--output",str(tmp_path/"cli-export")],
        cwd=tmp_path,capture_output=True,text=True)
    assert export.returncode == 0,export.stderr
    selection = json.loads(export.stdout)["candidates"][0]["id"]
    restored = subprocess.run(command+["reconstruct","--bundle",str(tmp_path/"cli-export/study-evidence.zip"),
        "--candidate",selection,"--output",str(tmp_path/"cli-restored")],cwd=tmp_path,capture_output=True,text=True)
    assert restored.returncode == 0,restored.stderr
    result = json.loads(restored.stdout)
    assert result["artifact_hash"] == candidate.artifact_hash
    assert result["requests"] >= 1 and result["executed"] is False


def test_v2_reference_text_is_omitted_but_exact_receipts_and_rehydration_metadata_survive(study,repo,tmp_path):
    import hashlib
    import uuid
    root,candidate,db = study
    research = root/"campaigns/agent/research"
    passage_text = "EXTERNAL_DOCUMENT_PASSAGE_"+uuid.uuid4().hex
    passage_hash = hashlib.sha256(passage_text.encode()).hexdigest()
    passage_path = research/"passages"/(passage_hash+".txt")
    passage_path.parent.mkdir()
    passage_path.write_text(passage_text)
    source = {"source_id":"source-test","kind":"cuda_documentation","url":"https://docs.nvidia.com/cuda/guide.html#section",
        "repository":"","commit_or_version":"sha256:original-body","file":"","symbol":"",
        "section_or_locator":"#section","content_sha256":"original-body","cache_path":str(research/"sources/downloaded.txt"),
        "read_receipt_version":2,"read_characters":len(passage_text),"passages":[{
            "passage_id":"passage-test","text_sha256":passage_hash,"cache_path":str(passage_path),
            "role":"html_fragment","fragment":"section","line_start":0,"line_end":0,"characters":len(passage_text),"truncated":False}]}
    supplied = research/"research-test-supplied-content.json"
    write(supplied,[{"source":source,"read_text":passage_text}])
    dossier = research/"research-test.json"
    receipt = {"research_id":"research-test","sources":[source],"receipt_version":2,
        "supplied_content_path":str(supplied),"supplied_content_sha256":hashlib.sha256(supplied.read_bytes()).hexdigest()}
    write(dossier,receipt)
    exact_receipt = dossier.read_bytes()
    manifest = export_study(root,tmp_path/"output",repo_dir=repo)
    assert manifest["status"] == "complete",manifest["errors"]
    assert manifest["generation_resume"]["guard_ready"] is False
    assert manifest["generation_resume"]["rehydration_required"] is True
    with zipfile.ZipFile(tmp_path/"output/study-evidence.zip") as archive:
        assert not any("/research/passages/" in name or name.endswith("-supplied-content.json") for name in archive.namelist())
        assert archive.read("evidence/campaigns/agent/research/research-test.json") == exact_receipt
        index = json.loads(archive.read("reference-rehydration.json"))
        assert not index["retrieval_performed_by_exporter"]
        assert not index["kernel_reconstruction_requires_reference_download"]
        assert any(row["source"]["url"] == source["url"] and row["passages"][0]["text_sha256"] == passage_hash for row in index["sources"])
        assert any(row["original_path"].endswith("-supplied-content.json") and row["sha256"] == receipt["supplied_content_sha256"] for row in index["omitted_files"])
        for name in archive.namelist():
            assert passage_text.encode() not in archive.read(name)
    selected = manifest["candidates"][0]
    rebuilt,requests,_ = reconstruct_candidate(tmp_path/"output/study-evidence.zip",selected["id"],tmp_path/"restored")
    assert rebuilt.artifact_hash == candidate.artifact_hash and requests
    reconstruction = json.loads((tmp_path/"restored/reconstruction.json").read_text())
    assert reconstruction["generation_resume"]["guard_ready"] is False
    assert reconstruction["generation_resume"]["rehydration_required"] is True
