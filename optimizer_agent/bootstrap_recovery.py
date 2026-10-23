"""Explicit infrastructure recovery before the first generator/model call.

No submission, CUDA execution or model call occurs here. A live owned lease is
retained; a terminal lease receives an independent attempt, without pooling its
measurements. Incomplete/uncertain model calls are never treated as bootstrap.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import secrets
import subprocess
import time


_TAR_STAGE_CATEGORY = "legacy_worker_stage_extraction_storage_failure"
_LEGACY_STAGE_UID = 100267
_LEGACY_STAGE_GID = 100267
# Explicit historical authorities, not whatever worker happens to be dirty in
# the checkout now. Both commits preceded the interrupted ROMEO extraction.
_LEGACY_STAGE_COMMITS = (
    "fefd63cd874fcc77f5c8db3b874b1e87971e5366",
    "577d20a8b8a7de333309a32dddc0c6ce79104df9",
)

# Read-only probe. Its complete program and request are hashed into the audit.
# Only compiler/worker sources can be opened: unknown entries, ACKs, RPCs,
# links, special files and other owners all stop before reading their contents.
_PARTIAL_STAGE_PROBE = r'''
import hashlib,json,os,socket,stat,sys
request=json.loads(sys.argv[1])
directory=request['directory']; expected=request['files']; allowed_dirs=set(request['directories'])
uid=request['uid']; gid=request['gid']; entries={};hostname=socket.gethostname()
def require(ok,message):
    if not ok: raise RuntimeError(message)
def ident(s):
    return [s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink]
require(directory.startswith('/') and all(p not in ('','..','.') for p in directory.split('/')[1:]),'Noncanonical owned session path')
root=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
try:
    for part in directory.split('/')[1:]:
        nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=root)
        os.close(root);root=nxt
    original_root=ident(os.fstat(root))
    require(original_root[3:5]==[uid,gid],'Foreign session owner')
    def visit(parent,name,key):
        before=os.stat(name,dir_fd=parent,follow_symlinks=False)
        isdir=stat.S_ISDIR(before.st_mode)
        require((isdir and key in allowed_dirs) or (stat.S_ISREG(before.st_mode) and key in expected),'Unknown/symlink/special stage entry')
        require(before.st_uid==uid and before.st_gid==gid,'Foreign stage owner')
        if not isdir: require(before.st_nlink==1,'Linked stage file')
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|(os.O_DIRECTORY if isdir else os.O_NONBLOCK),dir_fd=parent)
        try:
            require(ident(os.fstat(fd))==ident(before),'Stage open identity drift')
            row={'kind':'directory' if isdir else 'file','identity':ident(before)}
            if isdir:
                children=sorted(os.listdir(fd))
                for child in children: visit(fd,child,key+'/'+child)
                require(sorted(os.listdir(fd))==children,'Stage directory changed')
            else:
                require(0<=before.st_size<=expected[key],'Stage file exceeds authorized source')
                h=hashlib.sha256();size=0
                while True:
                    block=os.read(fd,min(1048576,expected[key]+1-size))
                    if not block: break
                    size+=len(block);require(size<=expected[key],'Stage file grew beyond authorized source');h.update(block)
                require(size==before.st_size,'Stage file size changed')
                row.update(size=size,sha256=h.hexdigest())
            require(ident(os.fstat(fd))==ident(before) and ident(os.stat(name,dir_fd=parent,follow_symlinks=False))==ident(before),'Stage read identity drift')
            entries[key]=row
        finally: os.close(fd)
    children=sorted(os.listdir(root))
    absent_markers=['job_id.json','submission_started.json','submission_error.json','ready.json','hardware.json','build.json','requests','objects','quarantined.json']
    require(not any(name in children for name in absent_markers),'ACK/submission/worker/RPC evidence exists')
    for child in children: visit(root,child,child)
    require(sorted(os.listdir(root))==children and ident(os.fstat(root))==original_root,'Owned session changed during probe')
    check=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in directory.split('/')[1:]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=check)
            os.close(check);check=nxt
        require(ident(os.fstat(check))==original_root,'Owned absolute session path changed')
    finally: os.close(check)
    require(socket.gethostname()==hostname,'Probe hostname changed')
    print(json.dumps({'schema':1,'directory':directory,'uid':uid,'gid':gid,'hostname':hostname,'absent_markers':absent_markers,'root_identity':original_root,'entries':entries},sort_keys=True))
finally: os.close(root)
'''.strip()


def _legacy_stage_sources(repo):
    """Authenticate the complete old staging member list against Git objects."""
    authorities = []
    for commit in _LEGACY_STAGE_COMMITS:
        def git(*args):
            return subprocess.run(["git", "-C", str(repo), *args], check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        if git("rev-parse", commit + "^{commit}").decode().strip() != commit:
            raise ValueError("Legacy worker source authority differs")
        paths = git("ls-tree", "-r", "--name-only", commit, "cuda_test_harness").decode().splitlines()
        selected = sorted(p for p in paths if PurePosixPath(p).suffix in {".cpp", ".cu", ".h", ".hpp", ".cuh", ".txt"}
                          and not any(part.startswith(("build", "cmake-build")) for part in PurePosixPath(p).parts))
        selected.append("optimizer_agent/remote_worker.py")
        files = {p: git("show", commit + ":" + p) for p in selected}
        if not files or any(PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts for p in files):
            raise ValueError("Invalid legacy worker member authority")
        authorities.append({"commit": commit, "files": files})
    if any(set(a["files"]) != set(authorities[0]["files"]) for a in authorities):
        raise ValueError("Legacy source authorities do not share one exact member list")
    return authorities


def _tar_stage_diagnostic(error, members):
    """Only complete GNU-tar EDQUOT/ENOSPC extraction diagnostics qualify."""
    from .suite_orchestrator import _infrastructure_diagnostic_lines
    if not isinstance(error, str):
        return None
    lines = _infrastructure_diagnostic_lines(error)
    if len(lines) < 2 or lines[-1] != "tar: Exiting with failure status due to previous errors":
        return None
    directories = {str(p) for name in members for p in PurePosixPath(name).parents if str(p) != "."}
    failed_directories, seen, records = set(), set(), []
    for line in lines[:-1]:
        match = re.fullmatch(r"tar: ([A-Za-z0-9_./-]+): (Cannot open|Cannot mkdir): (Disk quota exceeded|No space left on device|No such file or directory)", line)
        if match is None:
            return None
        member, operation, reason = match.groups()
        if member in seen or member not in (directories if operation == "Cannot mkdir" else members):
            return None
        if reason == "No such file or directory" and not any(str(parent) in failed_directories for parent in PurePosixPath(member).parents):
            return None
        if operation == "Cannot mkdir":
            failed_directories.add(member)
        seen.add(member)
        records.append({"member": member, "operation": operation, "reason": reason})
    if not any(r["reason"] in {"Disk quota exceeded", "No space left on device"} for r in records):
        return None
    return {"schema": 1, "records": records, "error_sha256": hashlib.sha256(error.encode()).hexdigest()}


def _partial_stage_local_guard(directory, task, failure):
    if (failure.get("status") != "failed" or failure.get("session_id") != task["session"]
            or failure.get("job_id") is not None or failure.get("journal_job_id") is not None
            or task.get("job_id") is not None or failure.get("execution_complete") is not False
            or failure.get("partial_result") is not None):
        raise ValueError("Legacy extraction failure does not belong to the exact never-submitted session")
    for name in ("worker", "request-receipts"):
        path = Path(directory) / name
        if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise ValueError("Local worker/ACK/hardware/RPC evidence forbids extraction bootstrap")


def _qualify_partial_stage(backend, directory, task, failure, authorities=None):
    """Prove one owned, never-started partial source extraction; read-only."""
    _partial_stage_local_guard(directory, task, failure)
    expected_directory = str(PurePosixPath(backend.config.remote_root) / "sessions" / task["session"])
    if backend.session_id != task["session"] or backend.remote_dir != expected_directory:
        raise ValueError("Legacy extraction backend session differs from the failure owner")
    authorities = authorities or _legacy_stage_sources(backend.repo_dir)
    diagnostic = _tar_stage_diagnostic(failure.get("error"), set(authorities[0]["files"]))
    if diagnostic is None or failure.get("infrastructure_category") not in {None, _TAR_STAGE_CATEGORY}:
        raise ValueError("Unqualified legacy worker extraction diagnostic")
    members = authorities[0]["files"]
    directories = sorted({str(p) for name in members for p in PurePosixPath(name).parents if str(p) != "."})
    request = {"directory": backend.remote_dir, "uid": _LEGACY_STAGE_UID, "gid": _LEGACY_STAGE_GID,
               "files": {name: max(len(a["files"][name]) for a in authorities) for name in members},
               "directories": directories}
    raw_request = json.dumps(request, sort_keys=True, separators=(",", ":"))
    process = backend.transport.run([backend.config.python, "-c", _PARTIAL_STAGE_PROBE, raw_request], timeout=120)
    if process.returncode != 0 or len(process.stdout) > 65536:
        raise ValueError("Read-only owned partial-stage probe failed")
    remote = json.loads(process.stdout)
    if (remote.get("schema") != 1 or remote.get("directory") != backend.remote_dir
            or remote.get("uid") != _LEGACY_STAGE_UID or remote.get("gid") != _LEGACY_STAGE_GID
            or not isinstance(remote.get("entries"), dict)):
        raise ValueError("Partial-stage proof differs from the owned probe request")
    def checked_identity(value, *, directory_entry, size=None):
        import stat
        return (isinstance(value, list) and len(value) == 9 and all(type(n) is int and n >= 0 for n in value)
                and (stat.S_ISDIR(value[2]) if directory_entry else stat.S_ISREG(value[2]))
                and value[3:5] == [_LEGACY_STAGE_UID, _LEGACY_STAGE_GID]
                and (directory_entry or (value[8] == 1 and value[5] == size)))
    if not checked_identity(remote.get("root_identity"), directory_entry=True):
        raise ValueError("Partial-stage root identity/owner is invalid")
    for name, row in remote["entries"].items():
        if (not isinstance(row, dict) or row.get("kind") not in {"directory", "file"}
                or (row["kind"] == "directory" and name not in directories)
                or (row["kind"] == "file" and name not in members)
                or not checked_identity(row.get("identity"), directory_entry=row["kind"] == "directory", size=row.get("size"))):
            raise ValueError("Partial-stage entry identity/owner/namespace is invalid")
    matching = []
    for authority in authorities:
        valid = True
        for name, row in remote["entries"].items():
            if row.get("kind") == "directory" and name in directories:
                continue
            if row.get("kind") != "file" or name not in authority["files"] or type(row.get("size")) is not int:
                valid = False
                break
            content, size = authority["files"][name], row["size"]
            if not 0 <= size <= len(content) or row.get("sha256") != hashlib.sha256(content[:size]).hexdigest():
                valid = False
                break
        if valid:
            matching.append(authority["commit"])
    if not matching:
        raise ValueError("Partial worker/harness content is not a prefix of an authorized old source snapshot")
    return {"schema": 1, "category": _TAR_STAGE_CATEGORY, "diagnostic": diagnostic,
            "failure_sha256": hashlib.sha256((Path(directory) / "failure.json").read_bytes()).hexdigest(),
            "authorized_source_commits": matching,
            "source_members_sha256": {a["commit"]: {name: hashlib.sha256(data).hexdigest() for name, data in a["files"].items()}
                                      for a in authorities if a["commit"] in matching},
            "probe_script_sha256": hashlib.sha256(_PARTIAL_STAGE_PROBE.encode()).hexdigest(),
            "probe_request_sha256": hashlib.sha256(raw_request.encode()).hexdigest(),
            "remote_inventory": remote, "remote_inventory_sha256": hashlib.sha256(json.dumps(remote, sort_keys=True).encode()).hexdigest(),
            "model_calls": 0, "gpu_requests": 0, "new_job_submitted": False}


def prove_pre_model(directory, mode):
    """Fail closed on even an incomplete model intent or graph checkpoint."""
    directory = Path(directory)
    generation = directory / "generation"
    for name in ("model", "model-transport", "research", "trials"):
        path = generation / name
        if path.exists() and any(p.is_file() for p in path.rglob("*")):
            raise ValueError(f"Generator artifacts already exist in {name}; bootstrap recovery is forbidden")
    for name in ("langgraph.sqlite", "original-langgraph.sqlite", "experiments.sqlite", "manifest.json", "best.cu", "candidate.json"):
        if (generation / name).exists():
            raise ValueError(f"Generator/checkpoint artifact {name} requires its dedicated resume protocol")
    if mode == "native_codex_vanilla":
        receipt = generation / "evidence/receipt.json"
        if receipt.exists():
            row = json.loads(receipt.read_text())
            if (row.get("model_started") is True or row.get("child_pid") or row.get("thread_ids") or row.get("actual_models")
                    or row.get("usage_events") or any(row.get("usage", {}).values())
                    or row.get("status") not in {"starting", "failed"}):
                raise ValueError("An existing native prompt/process is never repeated by bootstrap recovery")
        for name in ("codex.jsonl", "last-message.json", "response.json", "candidate.json"):
            if (generation / "evidence" / name).exists():
                raise ValueError("Native model/output intent exists; preserve the one-prompt attempt")
        # Model tools cannot have produced child content before their Popen.
        workspace = generation / "workspace"
        if workspace.exists() and any(p.name not in {"baseline.cu", "GPU_TOOLS.md"} for p in workspace.iterdir()):
            raise ValueError("Unexpected native workspace artifacts require manual provenance review")
    return {"model_started": False, "generator_checkpoint_exists": False,
            "scope": "Only pre-model infrastructure bootstrap; not generic model replay"}


def _inventory(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file() and not path.is_symlink()}


def recover_pre_model(orchestrator, task_id, reason):
    from .execution_backend import TERMINAL_STATES, slurm_state
    from .suite_orchestrator import (config_from_dict, infrastructure_category,
                                     remote_storage_failure_directory, read_json, write_json)
    orchestrator._require_active_study()
    if not isinstance(reason, str) or len(reason.strip()) < 20:
        raise ValueError("Describe the concrete infrastructure incident in at least twenty characters")
    with orchestrator.journal.controller():
        task = next((row for row in orchestrator.journal.tasks("generation") if row["id"] == task_id), None)
        if not task or task["state"] != "recovery_required":
            raise ValueError("Only a failed, inactive generation can receive bootstrap recovery")
        lock = orchestrator.directory / "task-locks" / (hashlib.sha256(task_id.encode()).hexdigest() + ".lock")
        lock.parent.mkdir(exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("The previous generation process is still active") from error
            directory = orchestrator._directory(task)
            existing_intent = directory / "bootstrap-recovery.json"
            if existing_intent.exists():
                raise ValueError("A bootstrap recovery intent already exists; reconcile it instead of preparing twice")
            proof = prove_pre_model(directory, task["payload"]["mode"])
            for request_path in (directory / "request-receipts").glob("*.json"):
                request_receipt = read_json(request_path)
                if "report" in request_receipt and request_receipt["report"].get("ok") is not True:
                    raise ValueError("A failed native/precision/memory result cannot be retried as transport bootstrap")
            failure_path = directory / "failure.json"
            if not failure_path.is_file():
                raise ValueError("Bootstrap recovery requires the original failure receipt")
            failure = read_json(failure_path)
            category = failure.get("infrastructure_category") or infrastructure_category(failure.get("error", ""))
            stage_authorities = None
            if category not in {"transport_timeout", "ssh_transport_failure", "worker_terminal_incomplete", "remote_storage_failure"}:
                from .suite_orchestrator import _infrastructure_diagnostic_lines
                lines = _infrastructure_diagnostic_lines(failure.get("error", ""))
                if (not lines or not all(line.startswith("tar: ") for line in lines)
                        or lines[-1] != "tar: Exiting with failure status due to previous errors"):
                    raise ValueError("The original failure is not a qualified transport/worker infrastructure incident")
                stage_authorities = _legacy_stage_sources(Path(__file__).resolve().parents[1])
                if (_tar_stage_diagnostic(failure.get("error"), set(stage_authorities[0]["files"])) is None
                        or failure.get("infrastructure_category") not in {None, _TAR_STAGE_CATEGORY}):
                    raise ValueError("The original failure is not a qualified legacy extraction incident")
                category = _TAR_STAGE_CATEGORY
            config = config_from_dict(task["payload"]["config"])
            backend = orchestrator._backend(task, config, directory)
            if (category == "remote_storage_failure"
                    and remote_storage_failure_directory(failure.get("error", "")) != backend.remote_dir):
                raise ValueError("Storage failure directory differs from the owned worker session")
            # Inspect remote ownership before deciding that a new allocation is
            # permitted. A lost sbatch ACK/intent is never interpreted as no job.
            stage_proof = (_qualify_partial_stage(backend, directory, task, failure, stage_authorities)
                           if category == _TAR_STAGE_CATEGORY else None)
            # The complete FD-walk for this one legacy case already proves the
            # absence of ALL ACK/intent/worker/RPC names, including symlinks.
            remote = None if stage_proof else backend._json("job_id.json", optional=True)
            local_path = directory / "worker/slurm-session.json"
            if remote is None:
                markers = {} if stage_proof else {name: backend._json(name, optional=True)
                           for name in ("submission_started.json", "submission_error.json")}
                if (task.get("job_id") or failure.get("job_id") or local_path.exists()
                        or (directory/"worker/hardware.json").exists()
                        or any((directory/"request-receipts").glob("*.json"))
                        or any(value is not None for value in markers.values())):
                    raise ValueError("Uncertain scheduler ownership must be reconciled before bootstrap recovery")
                status = {"state": "NOT_SUBMITTED", "terminal": False}
                live, terminal = False, False
            else:
                status = backend.attach_existing()
                if (task.get("job_id") is not None and str(task["job_id"]) != str(backend.job_id)):
                    raise ValueError("Remote owned job differs from the suite journal")
                if failure.get("job_id") is not None and str(failure["job_id"]) != str(backend.job_id):
                    raise ValueError("Remote owned job differs from the failure receipt")
                quarantined = backend._json("quarantined.json", optional=True)
                if quarantined is not None:
                    raise ValueError("CUDA quarantine is not a transport bootstrap incident")
                live, terminal = slurm_state(status) == "RUNNING", slurm_state(status) in TERMINAL_STATES
                if not live and not terminal:
                    raise ValueError("Owned worker is neither RUNNING nor confirmed terminal; preserve it")
            inventory = _inventory(directory)
            audit = {"schema_version": 1, "task_id": task_id, "reason": reason,
                "created_at_unix": time.time(), "previous_session": task["session"],
                "previous_job_id": backend.job_id, "ownership_status": status,
                "pre_model_proof": proof, "infrastructure_category": category,
                "previous_artifacts_sha256": inventory, "model_calls": 0,
                "new_job_submitted": False, "gpu_requests": 0,
                "same_live_allocation": live, "raw_timings_pooled_between_allocations": False,
                "phase": "prepared"}
            if stage_proof:
                audit["legacy_stage_extraction_proof"] = stage_proof
            if live:
                from .bootstrap_replay import create_manifest
                hardware = read_json(directory / "worker/hardware.json")
                if str(hardware.get("slurm_job_id")) != str(backend.job_id):
                    raise ValueError("Bootstrap hardware belongs to a different owned allocation")
                ready = backend._json("ready.json")
                if not isinstance(ready, dict) or ready.get("ok") is not True or ready.get("hardware") != hardware:
                    raise ValueError("Live worker readiness/hardware differs from the frozen local receipt")
                manifest = create_manifest(directory / "request-receipts", hardware)
                manifest_path = directory / "bootstrap-replay-manifest.json"
                write_json(manifest_path, manifest)
                audit.update(replay_manifest=str(manifest_path), replay_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
                if task["payload"]["mode"] == "native_codex_vanilla" and (directory / "generation").exists():
                    audit["native_pre_model_archive"] = str(directory / "bootstrap-archives/000/generation")
                baseline = directory / "generation/baseline-measurement.json"
                if baseline.exists() and read_json(baseline).get("ok") is not True:
                    audit["failed_baseline_archive"] = str(directory / "bootstrap-archives/failed-baseline.json")
                # Commit the intent before moving any pre-model workspace.
                # An interrupted preparation is retained and cannot cause an
                # automatic second prompt or a second allocation.
                write_json(existing_intent, audit)
                # Native CLI insists on a fresh filesystem workspace. Its old
                # pre-model artifacts remain under the evaluator's denied root.
                if task["payload"]["mode"] == "native_codex_vanilla" and (directory / "generation").exists():
                    archive = directory / "bootstrap-archives" / "000"
                    archive.mkdir(parents=True, exist_ok=False)
                    (directory / "generation").rename(archive / "generation")
                    audit["native_pre_model_archive"] = str(archive / "generation")
                baseline = directory / "generation/baseline-measurement.json"
                if baseline.exists() and read_json(baseline).get("ok") is not True:
                    archive = directory / "bootstrap-archives" / "failed-baseline.json"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    baseline.rename(archive)
                write_json(existing_intent, audit)
                # "running" makes the existing graph/metadata resume path
                # explicit, with no new lease or generation identity.
                orchestrator.journal.update(task_id, state="running", job_id=str(backend.job_id), error=None, finished=None)
            elif terminal:
                session = "suite-gen-" + secrets.token_hex(16)
                audit.update(new_session=session, new_attempt=task["attempt"]+1,
                    previous_measurements_scope="Retained only; target identity changes require a fresh bootstrap")
                write_json(existing_intent, audit)
                with orchestrator.journal.connection() as db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?)", (task_id, task["attempt"],
                        json.dumps({"task":task,"bootstrap_recovery":audit},sort_keys=True), None, None, reason, time.time()))
                    db.execute("UPDATE tasks SET state='pending',session=?,job_id=NULL,result=NULL,error=NULL,started=NULL,finished=NULL,attempt=attempt+1 WHERE id=?", (session,task_id))
            else:
                # Nothing was submitted and the generator never started. Stage
                # can finish under the exact stable session and worker intent.
                write_json(existing_intent, audit)
                orchestrator.journal.update(task_id, state="running", error=None, finished=None)
            audit["phase"] = "committed"
            write_json(existing_intent, audit)
            write_json(orchestrator.directory / "recovery-audit" / (task_id + ".bootstrap-%03d.json" % task["attempt"]), audit)
            return audit


def resume_preparation(orchestrator, task_id):
    """Finish only a durable preparation interrupted before its final ACK.

    This does not run the prepared generation. Remote ownership and every old
    artifact are checked again before the pending local transaction is finished.
    """
    from .execution_backend import TERMINAL_STATES, slurm_state
    from .suite_orchestrator import (config_from_dict, remote_storage_failure_directory,
                                     read_json, write_json)
    orchestrator._require_active_study()
    with orchestrator.journal.controller():
        task = next((r for r in orchestrator.journal.tasks("generation") if r["id"] == task_id), None)
        if not task or task["state"] not in {"pending", "running", "recovery_required"}:
            raise ValueError("No inactive bootstrap preparation can be resumed")
        lock = orchestrator.directory / "task-locks" / (hashlib.sha256(task_id.encode()).hexdigest()+".lock")
        lock.parent.mkdir(exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("The prepared generation process is already active") from error
            directories = [orchestrator._directory(task)]
            if task["attempt"]:
                directories.append(orchestrator._directory({**task,"attempt":task["attempt"]-1}))
            intents = [(d,read_json(d/"bootstrap-recovery.json")) for d in directories
                       if (d/"bootstrap-recovery.json").exists()]
            prepared = [(d,a) for d,a in intents if a.get("phase") == "prepared" and a.get("task_id") == task_id]
            if len(prepared) != 1:
                raise ValueError("Exactly one uncommitted bootstrap preparation is required")
            directory,audit = prepared[0]
            old_task = {**task,"session":audit["previous_session"],"job_id":audit["previous_job_id"]}
            if task["session"] not in {audit["previous_session"],audit.get("new_session")}:
                raise ValueError("Prepared bootstrap and journal sessions differ")
            if (task["session"] == audit["previous_session"] and task.get("job_id") is not None
                    and str(task["job_id"]) != str(audit["previous_job_id"])):
                raise ValueError("Prepared bootstrap and journal job ownership differ")
            archive = Path(audit["native_pre_model_archive"]) if audit.get("native_pre_model_archive") else None
            baseline_archive = Path(audit["failed_baseline_archive"]) if audit.get("failed_baseline_archive") else None
            for moved in (archive,baseline_archive):
                if moved is not None and not moved.resolve().is_relative_to(directory.resolve()):
                    raise ValueError("Prepared archive leaves its owned task directory")
            for name,sha in audit["previous_artifacts_sha256"].items():
                original = directory/name
                possible = [original]
                if archive is not None and name.startswith("generation/"):
                    possible.append(archive / name[len("generation/"):])
                if baseline_archive is not None and name == "generation/baseline-measurement.json":
                    possible.append(baseline_archive)
                actual = [p for p in possible if p.is_file() and not p.is_symlink()]
                if len(actual) != 1 or hashlib.sha256(actual[0].read_bytes()).hexdigest() != sha:
                    raise ValueError("Prepared bootstrap artifact changed, disappeared or was duplicated")
            # Re-run the pre-model guard against the retained directory, even
            # when its native workspace has already been atomically moved.
            prove_pre_model(directory,task["payload"]["mode"])
            if archive is not None and archive.exists():
                prove_pre_model(archive.parent,task["payload"]["mode"])
            config = config_from_dict(task["payload"]["config"])
            backend = orchestrator._backend(old_task,config,directory)
            if (audit.get("infrastructure_category") == "remote_storage_failure"
                    and remote_storage_failure_directory(read_json(directory / "failure.json").get("error", "")) != backend.remote_dir):
                raise ValueError("Prepared storage failure directory differs from the owned worker session")
            legacy_stage = audit.get("infrastructure_category") == _TAR_STAGE_CATEGORY
            if legacy_stage:
                fresh = _qualify_partial_stage(backend, directory, old_task, read_json(directory / "failure.json"))
                if fresh != audit.get("legacy_stage_extraction_proof"):
                    raise ValueError("Prepared legacy partial-stage proof changed; preserve the intent")
            if audit["same_live_allocation"] or audit.get("new_session"):
                status = backend.attach_existing()
                if str(backend.job_id) != str(audit["previous_job_id"]):
                    raise ValueError("Prepared bootstrap remote ownership changed")
                if backend._json("quarantined.json",optional=True) is not None:
                    raise ValueError("Prepared CUDA context is now quarantined; preserve the incident")
                wanted = "RUNNING" if audit["same_live_allocation"] else None
                if (wanted and slurm_state(status) != wanted) or (not wanted and slurm_state(status) not in TERMINAL_STATES):
                    raise ValueError("Prepared worker state changed; preserve the intent for explicit reconciliation")
            else:
                if not legacy_stage and any(backend._json(name,optional=True) is not None for name in ("job_id.json","submission_started.json","submission_error.json")):
                    raise ValueError("An allocation appeared during pre-submission preparation")
            if audit["same_live_allocation"]:
                manifest = directory/"bootstrap-replay-manifest.json"
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != audit["replay_manifest_sha256"]:
                    raise ValueError("Prepared replay manifest changed")
                hardware = read_json(directory/"worker/hardware.json")
                ready = backend._json("ready.json")
                if ready.get("ok") is not True or ready.get("hardware") != hardware or str(hardware.get("slurm_job_id")) != str(backend.job_id):
                    raise ValueError("Prepared live hardware changed")
                for origin,destination in ((directory/"generation",archive),
                                           (directory/"generation/baseline-measurement.json",baseline_archive)):
                    if destination is not None and origin.exists():
                        if destination.exists():
                            raise ValueError("Both source and destination of prepared archive exist")
                        destination.parent.mkdir(parents=True,exist_ok=True)
                        origin.rename(destination)
                orchestrator.journal.update(task_id,state="running",job_id=str(backend.job_id),error=None,finished=None)
            elif audit.get("new_session"):
                if task["session"] == audit["previous_session"]:
                    with orchestrator.journal.connection() as db:
                        db.execute("BEGIN IMMEDIATE")
                        db.execute("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?)",(task_id,task["attempt"],
                            json.dumps({"task":task,"bootstrap_recovery":audit},sort_keys=True),None,None,audit["reason"],time.time()))
                        db.execute("UPDATE tasks SET state='pending',session=?,job_id=NULL,result=NULL,error=NULL,started=NULL,finished=NULL,attempt=attempt+1 WHERE id=?",(audit["new_session"],task_id))
                elif task["attempt"] != audit["new_attempt"] or task["state"] != "pending":
                    raise ValueError("Prepared independent attempt has already advanced")
            else:
                orchestrator.journal.update(task_id,state="running",error=None,finished=None)
            audit.update(phase="committed",preparation_resumed_at_unix=time.time())
            write_json(directory/"bootstrap-recovery.json",audit)
            old_attempt = audit.get("new_attempt",task["attempt"]+1)-1
            write_json(orchestrator.directory/"recovery-audit"/(task_id+".bootstrap-%03d.json"%old_attempt),audit)
            return audit
