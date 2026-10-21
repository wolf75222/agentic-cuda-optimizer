"""Explicit transfer of an interrupted, entirely unscored LangGraph campaign.

This is not general tree migration or model replay. Preparation only authenticates
terminal ownership and copies local evidence. Activation requires a genuinely
fresh baseline on the new allocation before changing its performance identity.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import secrets
import shutil
import sqlite3
import time
import uuid

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .compile_cache import content_hash


class RehostRefused(ValueError):
    pass


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError,ValueError) as error:
        raise RehostRefused("Required evidence is missing or corrupt: "+str(path)) from error


def _write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex)
    with temporary.open("w") as stream:
        stream.write(_json(value)+"\n"); stream.flush(); os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary,path)
    descriptor = os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _readonly(path):
    db = sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro",uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        yield db
    finally:
        db.close()


def _rows_hash(db,table):
    rows = []
    for row in db.execute("SELECT * FROM "+table):
        rows.append({key:hashlib.sha256(value).hexdigest() if isinstance(value,bytes) else value
                     for key,value in dict(row).items()})
    return _hash(sorted(rows,key=_json))


def _checkpoint(generation,run_id):
    with _readonly(generation/"langgraph.sqlite") as db:
        rows = db.execute("SELECT * FROM checkpoints ORDER BY checkpoint_id").fetchall()
        if not rows or {row["thread_id"] for row in rows} != {run_id}:
            raise RehostRefused("Checkpoint thread identity is missing or ambiguous")
        namespaces = {row["checkpoint_ns"] for row in rows}
        leaves = [row for row in rows if row["checkpoint_ns"].startswith("experiment:")]
        if len(namespaces) != 2 or "" not in namespaces or not leaves:
            raise RehostRefused("Only one original pending experiment namespace can be transferred")
        leaf = leaves[-1]
        value = JsonPlusSerializer().loads_typed((leaf["type"],leaf["checkpoint"]))["channel_values"]
        if ("branch:to:generate" not in value or value.get("repair",0) < 1
                or value.get("research_round",0) < 1 or value.get("trials_done") != 0
                or value.get("valid") is not False or value.get("retryable") is not True
                or value.get("finished") or value.get("diagnostic_id")):
            raise RehostRefused("Checkpoint must be an unfinished first-trial repair before generate")
        pending_errors = db.execute("SELECT task_path FROM writes WHERE checkpoint_id=? AND channel='__error__'",
                                    (leaf["checkpoint_id"],)).fetchall()
        if not pending_errors or not any("generate" in row[0] for row in pending_errors):
            raise RehostRefused("The interrupted generate task is not authenticated")
        proof = {"checkpoint_id":leaf["checkpoint_id"],"namespace":leaf["checkpoint_ns"],
                 "rows_sha256":_rows_hash(db,"checkpoints"),"writes_sha256":_rows_hash(db,"writes"),
                 "state":{key:value.get(key) for key in ("run_id","trial_id","candidate_id","repair",
                    "research_round","trials_done","valid","retryable","diagnostic_id")}}
        return proof


def _unscored_proof(generation,run_id):
    generation = Path(generation)
    configuration = _read(generation/"config.json")
    checkpoint = _checkpoint(generation,run_id)
    state = checkpoint["state"]
    if state["run_id"] != run_id:
        raise RehostRefused("Checkpoint campaign differs from the ledger")
    with _readonly(generation/"experiments.sqlite") as db:
        campaigns = db.execute("SELECT * FROM campaigns").fetchall()
        if len(campaigns) != 1 or campaigns[0]["run_id"] != run_id:
            raise RehostRefused("Exactly one campaign identity is required")
        campaign = campaigns[0]
        if campaign["incumbent"] is not None or campaign["incumbent_version"] != 0:
            raise RehostRefused("An incumbent was already selected")
        for table in ("evaluations","reward_claims","integration_events","candidate_status"):
            if db.execute("SELECT count(*) FROM "+table).fetchone()[0]:
                raise RehostRefused("Any evaluation, reward, integration or candidate decision prohibits transfer")
        attempts = db.execute("SELECT status,trial_id FROM attempts").fetchall()
        if not attempts or any(row["status"] != "compilation_error" or row["trial_id"] != state["trial_id"] for row in attempts):
            raise RehostRefused("Every historical execution attempt must be a completed compilation failure")
        snapshot = pickle.loads(campaign["snapshot"])  # Trusted local campaign snapshot only.
        if not isinstance(snapshot,dict) or not {"tree","eligibility","rng"}.issubset(snapshot):
            raise RehostRefused("Only the authenticated TreeQuest snapshot format is supported")
        tree = snapshot["tree"]
        root = tree.tree.root
        if (tree.tree.size != 1 or root.score != .5 or root.children or root.parent is not None
                or tree.trial_store.finished_trials or any(tree.all_rewards_store.values())
                or snapshot["eligibility"].decisions):
            raise RehostRefused("Tree must contain only its neutral, unscored baseline")
        running = tree.trial_store.running_trials
        trials = db.execute("SELECT * FROM trials").fetchall()
        if len(running) != 1 or len(trials) != 1 or trials[0]["status"] != "pending":
            raise RehostRefused("Exactly one pending, unscored trial is required")
        trial_id = state["trial_id"]
        if trial_id not in running or trials[0]["trial_id"] != trial_id or trials[0]["result"] is not None or trials[0]["evaluation_id"] is not None:
            raise RehostRefused("Pending trial/checkpoint/result identities disagree")
        trial = running[trial_id]
        if trial.score is not None or trial.trial_status != "RUNNING" or trial.parent_state != root.state:
            raise RehostRefused("The pending trial is no longer an unscored baseline expansion")
        candidates = {row["candidate_id"]:json.loads(row["payload"]) for row in db.execute("SELECT * FROM candidates")}
        if set(candidates) != {"baseline",state["candidate_id"]} or state["candidate_id"] == "baseline":
            raise RehostRefused("Only baseline and its single invalid repair source can be transferred")
        for name,candidate in candidates.items():
            source_hash = hashlib.sha256(candidate["source"].encode()).hexdigest()
            artifact = content_hash({"source":source_hash,"kernel":candidate["kernel"],
                "launch_rule":candidate["launch_rule"],"parameters":json.dumps(candidate["parameters"],sort_keys=True,allow_nan=False)})
            if source_hash != candidate["source_hash"] or artifact != candidate["artifact_hash"]:
                raise RehostRefused("Immutable candidate source/launch/parameters changed")
        child = candidates[state["candidate_id"]]
        if child.get("tree_parent_id") != "baseline" or child.get("source_parent_id") != "baseline" or child.get("trial_id") != trial_id:
            raise RehostRefused("Invalid candidate is not the first baseline proposal")
        model_events = []
        for row in db.execute("SELECT kind,payload FROM events"):
            if row["kind"] == "model_uncertain":
                raise RehostRefused("An uncertain model intent cannot be replayed or transferred")
            if row["kind"] == "model":
                model_events.append(json.loads(row["payload"]))
        traces = {}
        for path in (generation/"model").glob("*.json"):
            trace = _read(path)
            if trace.get("status") != "completed" or trace.get("returncode") != 0 or trace.get("call_id") != path.stem:
                raise RehostRefused("Every provider call must have its completed, successful receipt")
            traces[path.stem] = {"sha256":_file_hash(path),"usage":trace["usage"]}
        if not traces or len(model_events) != len(traces) or {event["call_id"] for event in model_events} != set(traces):
            raise RehostRefused("Model receipts and durable ledger events differ")
        for event in model_events:
            if event["usage"] != traces[event["call_id"]]["usage"]:
                raise RehostRefused("Model token consumption changed")
        reservations = db.execute("SELECT amount,actual FROM reservations").fetchall()
        model_costs = []
        for row in reservations:
            amount = json.loads(row["amount"])
            if amount.get("model_calls"):
                if row["actual"] is None:
                    raise RehostRefused("A model reservation has uncertain consumption")
                model_costs.append(json.loads(row["actual"]))
        tokens = sum(trace["usage"].get("input_tokens",0)+trace["usage"].get("output_tokens",0) for trace in traces.values())
        if sum(row.get("model_calls",0) for row in model_costs) != len(traces) or sum(row.get("model_tokens",0) for row in model_costs) != tokens:
            raise RehostRefused("Model resource accounting is incomplete")
        identity = json.loads(campaign["identity"])
        if identity["baseline"] != _hash(root.state) or any(_json(candidates["baseline"].get(key)) != _json(value) for key,value in root.state.items()):
            raise RehostRefused("The root baseline and immutable candidate archive disagree")
        proof = {"run_id":run_id,"identity":identity,"tree_version":campaign["tree_version"],
            "configuration":configuration,
            "snapshot_sha256":hashlib.sha256(campaign["snapshot"]).hexdigest(),
            "rng_sha256":hashlib.sha256(pickle.dumps(snapshot["rng"],protocol=5)).hexdigest(),
            "candidates_sha256":_rows_hash(db,"candidates"),"trials_sha256":_rows_hash(db,"trials"),
            "reservations_sha256":_rows_hash(db,"reservations"),"attempts_sha256":_rows_hash(db,"attempts"),
            "model_receipts":traces,"model_tokens":tokens,"model_calls":len(traces),"checkpoint":checkpoint,
            "baseline_artifact_hash":candidates["baseline"]["artifact_hash"]}
    trial_dir = generation/"trials"/trial_id
    evaluation = _read(trial_dir/("evaluation-"+state["candidate_id"]+".json"))
    if (evaluation.get("ok") is not False or evaluation.get("status") != "compilation_error"
            or evaluation.get("candidate_id") != state["candidate_id"] or evaluation.get("contract_hash") != identity["contract"]
            or evaluation.get("artifact_hash") != child["artifact_hash"] or not isinstance(evaluation.get("cases"),list)):
        raise RehostRefused("Only the historical list-cases compilation diagnostic is admitted")
    if (trial_dir/f"proposal-{state['repair']}-{state['research_round']}.json").exists():
        raise RehostRefused("A completed proposal for the pending repair must use its own replay protocol")
    for path in generation.rglob("summary.json"):
        if "gain-observer" in path.parts and _read(path).get("observations"):
            raise RehostRefused("An observed paired decision already exists")
    return proof


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RehostRefused("Symlinked campaign evidence requires manual provenance review")
        if path.is_file() and not path.name.endswith(("-shm","-wal","-journal")):
            result[str(path.relative_to(directory))] = _file_hash(path)
    return result


def _copy_generation(source,destination):
    destination.mkdir(parents=True,exist_ok=False)
    for path in sorted(source.rglob("*")):
        target = destination/path.relative_to(source)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        elif not path.name.endswith(("-wal","-shm","-journal")):
            target.parent.mkdir(parents=True,exist_ok=True)
            if path.suffix == ".sqlite":
                with _readonly(path) as original:
                    copied = sqlite3.connect(target)
                    try:
                        original.backup(copied)
                        if copied.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise RehostRefused("Copied SQLite evidence failed integrity check")
                    finally:
                        copied.close()
            else:
                shutil.copy2(path,target)


def _context_files(audit):
    trial_id = audit["old_proof"]["checkpoint"]["state"]["trial_id"]
    return ["baseline-measurement.json","environment.json","contract.json",f"trials/{trial_id}/dossier.json"]


def _payload(audit):
    return {**audit["original_task"]["payload"],"source_recovery":{
        "kind":audit["kind"],"context_id":audit["context_id"],
        "marker_path":audit["marker_path"],"fresh_research_required":True}}


def _transferred_task(audit):
    return {**audit["original_task"],"payload":_payload(audit),"state":"pending","session":audit["new_session"],
        "job_id":None,"result":None,"error":None,"started":None,"finished":None,"attempt":audit["new_attempt"]}


def _task_phase(orchestrator,audit):
    current = next((row for row in orchestrator.journal.tasks("generation") if row["id"] == audit["task_id"]),None)
    original = audit["original_task"]
    expected = _transferred_task(audit)
    if current == original:
        return "original"
    if current == expected:
        return "transferred"
    raise RehostRefused("The task journal is neither the frozen original nor the exact new pending attempt")


def _sqlite_copy_state(source,target):
    """A partial backup may be absent/corrupt; a complete changed ledger is refused."""
    try:
        with _readonly(target) as copied:
            if copied.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                return False
            names = [row[0] for row in copied.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
            if not names:
                return False
            with _readonly(source) as original:
                expected = [row[0] for row in original.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                if set(names) != set(expected) or any(_rows_hash(copied,name) != _rows_hash(original,name) for name in expected):
                    raise RehostRefused("An interrupted copy contains a changed, complete SQLite ledger")
            return True
    except sqlite3.DatabaseError:
        return False


def _copy_complete(audit):
    destination,source = Path(audit["new_generation"]),Path(audit["old_generation"])
    if not destination.exists():
        return False
    expected,seen = audit["source_files_sha256"],set()
    complete = True
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise RehostRefused("An interrupted copy contains symlinked evidence")
        if not path.is_file():
            continue
        name = str(path.relative_to(destination))
        if name.endswith(("-wal","-shm","-journal")):
            database = name.rsplit("-",1)[0]
            if database not in expected or not database.endswith(".sqlite"):
                raise RehostRefused("Unexpected SQLite sidecar in copied evidence")
            # Read-only SQLite access may create its own empty WAL/SHM. The
            # database comparison below reads the coherent view, including WAL.
            continue
        if name == "unscored-rehost.json":
            marker = _read(path)
            ignored = {"phase","archived_context_files","interrupted_copy_archives"}
            if ({key:value for key,value in marker.items() if key not in ignored}
                    != {key:value for key,value in audit.items() if key not in ignored}):
                raise RehostRefused("The interrupted transfer marker differs from its durable intent")
            continue
        if name.startswith("prior-context/"):
            name = name.removeprefix("prior-context/")
            if name not in _context_files(audit):
                raise RehostRefused("Unexpected archived performance context")
        if name not in expected or name in seen:
            raise RehostRefused("Unexpected or duplicated file in the interrupted transfer")
        seen.add(name)
        if path.suffix == ".sqlite":
            complete = _sqlite_copy_state(source/name,path) and complete
        elif _file_hash(path) != expected[name]:
            # copy2 can be interrupted part way through a file; preserve the
            # partial bytes, but never accept a modified complete document.
            partial,original = path.read_bytes(),(source/name).read_bytes()
            if len(partial) >= len(original) or not original.startswith(partial):
                raise RehostRefused("A copied evidence file was modified after preparation")
            complete = False
    return complete and seen == set(expected)


def _commit_preparation(orchestrator,audit):
    original = audit["original_task"]
    with orchestrator.journal.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM tasks WHERE id=?",(audit["task_id"],)).fetchone()
        current = ({**dict(row),"payload":json.loads(row["payload"]),
                    "result":json.loads(row["result"]) if row["result"] else None} if row else None)
        if current == original:
            if db.execute("SELECT 1 FROM task_attempts WHERE task_id=? AND attempt=?",(audit["task_id"],original["attempt"])).fetchone():
                raise RehostRefused("The original attempt was already archived inconsistently")
            db.execute("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?)",(audit["task_id"],original["attempt"],
                _json({"task":original,"unscored_rehost":audit}),None,None,audit["reason"],time.time()))
            db.execute("UPDATE tasks SET payload=?,state='pending',session=?,job_id=NULL,result=NULL,error=NULL,started=NULL,finished=NULL,attempt=? WHERE id=?",
                       (_json(_payload(audit)),audit["new_session"],audit["new_attempt"],audit["task_id"]))
        else:
            if current != _transferred_task(audit):
                raise RehostRefused("The owned task advanced during preparation")
            row = db.execute("SELECT snapshot FROM task_attempts WHERE task_id=? AND attempt=?",(audit["task_id"],original["attempt"])).fetchone()
            snapshot = json.loads(row[0]) if row else {}
            if snapshot.get("task") != original or snapshot.get("unscored_rehost",{}).get("context_id") != audit["context_id"]:
                raise RehostRefused("The preserved original attempt does not authenticate this transfer")


def _finish_preparation(orchestrator,audit,intent):
    source,destination = Path(audit["old_generation"]),Path(audit["new_generation"])
    phase = _task_phase(orchestrator,audit)
    if not _copy_complete(audit):
        if phase != "original":
            raise RehostRefused("A committed journal cannot have an incomplete copied campaign")
        if destination.exists():
            archives = list(audit.get("interrupted_copy_archives",[]))
            planned = Path(archives[-1]) if archives and not Path(archives[-1]).exists() else None
            archive = planned or destination.parent/("interrupted-copy-"+audit["context_id"]+("-"+str(len(archives)+1) if archives else ""))
            if archive.parent != destination.parent or not archive.name.startswith("interrupted-copy-"+audit["context_id"]):
                raise RehostRefused("The preserved partial-copy path is outside its owned attempt")
            if archive.exists():
                raise RehostRefused("An interrupted-copy archive is already occupied")
            audit["interrupted_copy_archives"] = archives if planned else [*archives,str(archive)]
            _write(intent,audit)  # Record the preservation path before rename.
            destination.rename(archive)
        _copy_generation(source,destination)
    if _unscored_proof(destination,audit["old_proof"]["run_id"]) != audit["old_proof"] or _inventory(source) != audit["source_files_sha256"]:
        raise RehostRefused("Source/copy changed during transfer preparation")
    for name in _context_files(audit):
        path,target = destination/name,destination/"prior-context"/name
        if path.exists() and target.exists():
            raise RehostRefused("Duplicate old performance context in copied campaign")
        if path.is_file():
            target.parent.mkdir(parents=True,exist_ok=True)
            path.rename(target)
        if not target.is_file() or _file_hash(target) != audit["source_files_sha256"][name]:
            raise RehostRefused("Missing or modified original performance/research context: "+name)
    audit["archived_context_files"] = _context_files(audit)
    _write(destination/"unscored-rehost.json",audit)
    if _inventory(source) != audit["source_files_sha256"]:
        raise RehostRefused("The original evidence changed before the journal commit")
    _commit_preparation(orchestrator,audit)
    audit["phase"] = "committed"
    _write(destination/"unscored-rehost.json",audit)
    _write(intent,audit)
    return audit


def prepare_unscored_rehost(orchestrator,task_id,reason):
    from .execution_backend import TERMINAL_STATES, slurm_state
    from .suite_orchestrator import config_from_dict
    orchestrator._require_active_study()
    if not isinstance(reason,str) or len(reason.strip()) < 20:
        raise RehostRefused("Describe the diagnosed application failure in at least twenty characters")
    with orchestrator.journal.controller():
        task = next((row for row in orchestrator.journal.tasks("generation") if row["id"] == task_id),None)
        if not task or task["state"] != "recovery_required" or task["payload"]["mode"] in {"true_upstream","native_codex_vanilla"}:
            raise RehostRefused("Only an inactive agent LangGraph campaign can enter this transfer")
        lock = orchestrator.directory/"task-locks"/(hashlib.sha256(task_id.encode()).hexdigest()+".lock")
        lock.parent.mkdir(exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RehostRefused("The original generation process is still active") from error
            intent = orchestrator.directory/"unscored-rehost-intents"/(task_id+".json")
            if intent.exists() or task["payload"].get("source_recovery"):
                raise RehostRefused("A transfer intent already exists; never prepare the generation twice")
            directory = orchestrator._directory(task)
            generation = directory/"generation"
            failure = _read(directory/"failure.json")
            if (failure.get("error") != "'list' object has no attribute 'items'"
                    or "evaluation_summary" not in failure.get("traceback","")
                    or ".items()" not in failure.get("traceback","") or not task.get("job_id")
                    or str(failure.get("job_id")) != str(task["job_id"])):
                raise RehostRefused("The exact diagnosed list-cases application failure is required")
            config = config_from_dict(task["payload"]["config"])
            if _read(generation/"config.json") != json.loads(json.dumps(config.to_dict())):
                raise RehostRefused("The campaign ABI/workload/seed/settings changed")
            proof = _unscored_proof(generation,config.run_id)
            prior_baseline = _read(generation/"baseline-measurement.json")
            prior_environment = _read(generation/"environment.json")
            prior_contract = _read(generation/"contract.json")
            if (digest_environment(prior_environment) != proof["identity"]["environment"]
                    or prior_contract.get("contract_hash") != proof["identity"]["contract"]
                    or prior_baseline.get("contract_hash") != proof["identity"]["contract"]
                    or prior_baseline.get("ok") is not True
                    or {key:row["latency_ms"] for key,row in prior_baseline.get("cases",{}).items()} != proof["identity"]["baseline_latencies"]):
                raise RehostRefused("The original environment/contract/baseline identity is inconsistent")
            backend = orchestrator._backend(task,config,directory)
            terminal = backend.attach_existing()
            if str(backend.job_id) != str(task["job_id"]) or slurm_state(terminal) not in TERMINAL_STATES:
                raise RehostRefused("The original allocation is not authenticated terminal")
            if backend._json("quarantined.json",optional=True) is not None:
                raise RehostRefused("A quarantined context cannot be transferred by this application repair")
            context_id,new_session = uuid.uuid4().hex,"suite-gen-"+secrets.token_hex(16)
            next_task = {**task,"attempt":task["attempt"]+1}
            destination = orchestrator._directory(next_task)/"generation"
            if destination.exists():
                raise RehostRefused("The new attempt directory is already occupied")
            inventory = _inventory(generation)
            audit = {"schema_version":1,"kind":"unscored_performance_context_transfer","phase":"prepared",
                "context_id":context_id,"task_id":task_id,"reason":reason,"created_at_unix":time.time(),
                "previous_attempt":task["attempt"],"new_attempt":next_task["attempt"],
                "previous_session":task["session"],"previous_job_id":str(task["job_id"]),
                "new_session":new_session,"old_generation":str(generation),"new_generation":str(destination),
                "marker_path":str(destination/"unscored-rehost.json"),"old_proof":proof,
                "original_task":task,"failure_sha256":_file_hash(directory/"failure.json"),
                "source_files_sha256":inventory,"worker_terminal":terminal,
                "model_calls_during_preparation":0,"gpu_requests_during_preparation":0,"new_job_submitted":False,
                "scores_inherited":False,"fresh_research_required":True,
                "baseline_reservation_id":"baseline-bootstrap-rehost-"+context_id,
                "bridge_source_sha256":_file_hash(__file__)}
            _write(intent,audit)  # Before any copied/moved campaign evidence.
            return _finish_preparation(orchestrator,audit,intent)


def resume_unscored_preparation(orchestrator,task_id):
    """Reconcile a prepared transfer without any model/GPU/submission activity.

    Only the frozen original terminal allocation is contacted to authenticate
    ownership. Incomplete copies are preserved before a new coherent backup;
    the already chosen context/session/attempt are never generated again.
    """
    from .execution_backend import TERMINAL_STATES, slurm_state
    from .suite_orchestrator import config_from_dict
    orchestrator._require_active_study()
    with orchestrator.journal.controller():
        lock = orchestrator.directory/"task-locks"/(hashlib.sha256(task_id.encode()).hexdigest()+".lock")
        lock.parent.mkdir(exist_ok=True)
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RehostRefused("The generation process is still active") from error
            intent = orchestrator.directory/"unscored-rehost-intents"/(task_id+".json")
            audit = _read(intent)
            if audit.get("kind") != "unscored_performance_context_transfer" or audit.get("phase") != "prepared" or audit.get("task_id") != task_id:
                raise RehostRefused("Only one prepared, uncommitted transfer can be reconciled")
            if audit.get("bridge_source_sha256") != _file_hash(__file__):
                raise RehostRefused("The transfer bridge changed after the durable preparation intent")
            original = audit.get("original_task",{})
            if (original.get("id") != task_id or original.get("kind") != "generation"
                    or original.get("state") != "recovery_required" or original.get("attempt") != audit["previous_attempt"]
                    or original.get("session") != audit["previous_session"] or str(original.get("job_id")) != audit["previous_job_id"]
                    or audit["new_attempt"] != audit["previous_attempt"]+1 or audit["new_session"] == audit["previous_session"]
                    or original["payload"]["mode"] in {"true_upstream","native_codex_vanilla"}):
                raise RehostRefused("The frozen task identity is inconsistent")
            directory = orchestrator._directory(original)
            destination = orchestrator._directory({**original,"attempt":audit["new_attempt"]})/"generation"
            if (str(directory/"generation") != audit["old_generation"] or str(destination) != audit["new_generation"]
                    or str(destination/"unscored-rehost.json") != audit["marker_path"]):
                raise RehostRefused("The transfer paths do not belong to the owned task attempts")
            _task_phase(orchestrator,audit)
            if (destination.parent/"worker/slurm-session.json").exists():
                raise RehostRefused("The new attempt already has an allocation receipt")
            config = config_from_dict(original["payload"]["config"])
            if (_file_hash(directory/"failure.json") != audit["failure_sha256"]
                    or _inventory(directory/"generation") != audit["source_files_sha256"]
                    or _unscored_proof(directory/"generation",config.run_id) != audit["old_proof"]
                    or json.loads(json.dumps(config.to_dict())) != audit["old_proof"]["configuration"]):
                raise RehostRefused("The original immutable evidence changed after the prepared intent")
            backend = orchestrator._backend(original,config,directory)
            terminal = backend.attach_existing()
            if str(backend.job_id) != audit["previous_job_id"] or slurm_state(terminal) not in TERMINAL_STATES:
                raise RehostRefused("The original allocation is not authenticated terminal")
            if backend._json("quarantined.json",optional=True) is not None:
                raise RehostRefused("A quarantined original context cannot be reconciled")
            return _finish_preparation(orchestrator,audit,intent)


def verified_rehost_identity(store,run_id,marker,new_identity,budget_limits):
    """Bind the copied, still-unscored tree to exactly one fresh context.

    Repeated initialization only accepts the exact previously activated identity;
    a marker never authorizes a second hardware/context migration after learning.
    """
    marker_path = Path(marker).resolve() if isinstance(marker,(str,Path)) else None
    if marker_path is None or marker_path.parent != Path(store.path).resolve().parent:
        raise RehostRefused("The active store's committed transfer marker is required")
    record = _read(marker_path)
    if record.get("phase") != "committed" or record.get("kind") != "unscored_performance_context_transfer" or record.get("old_proof",{}).get("run_id") != run_id:
        raise RehostRefused("Transfer marker is uncommitted or belongs to another campaign")
    identity = dict(new_identity,budget_limits=budget_limits or {},finalization_reserve={})
    old = record["old_proof"]["identity"]
    if set(identity) != set(old) or any(identity[key] != old[key] for key in old if key not in {"contract","environment","baseline_latencies"}):
        raise RehostRefused("Only contract/environment/fresh baseline latencies may change")
    latencies = identity["baseline_latencies"]
    if (set(latencies) != set(old["baseline_latencies"]) or any(not math.isfinite(value) or value <= 0 for value in latencies.values())
            or not identity["contract"] or not identity["environment"] or identity["contract"] == old["contract"]):
        raise RehostRefused("A distinct contract and complete fresh positive baseline are required")
    with store.transaction() as db:
        current = db.execute("SELECT identity FROM campaigns WHERE run_id=?",(run_id,)).fetchone()
        previous = [json.loads(row[0]) for row in db.execute("SELECT payload FROM events WHERE run_id=? AND kind='unscored_context_rebound'",(run_id,))]
        if previous:
            if (len(previous) != 1 or previous[0]["context_id"] != record["context_id"]
                    or previous[0]["marker_sha256"] != _file_hash(marker_path)
                    or previous[0]["new_identity"] != identity or json.loads(current[0]) != identity):
                raise RehostRefused("This marker cannot authorize another performance-context change")
            return previous[0]
    if _unscored_proof(marker_path.parent,run_id) != record["old_proof"]:
        raise RehostRefused("The unscored tree/RNG/trial/candidates/model ledger changed before activation")
    if _read(marker_path.parent/"config.json") != record["old_proof"]["configuration"]:
        raise RehostRefused("ABI/workload/seed/configuration changed after preparation")
    contract = _read(marker_path.parent/"contract.json")
    prior_contract = _read(marker_path.parent/"prior-context/contract.json")
    current_body,prior_body = dict(contract["contract"]),dict(prior_contract["contract"])
    for body in (current_body,prior_body):
        metadata = json.loads(body.get("metadata_json","{}"))
        metadata.pop("environment",None); metadata.pop("target_environment",None)
        body["metadata_json"] = metadata
    environment = _read(marker_path.parent/"environment.json")
    environment_hash = digest_environment(environment)
    if current_body != prior_body or contract.get("contract_hash") != identity["contract"] or environment_hash != identity["environment"]:
        raise RehostRefused("The workload protocol changed or new environment binding is inconsistent")
    baseline = _read(marker_path.parent/"baseline-measurement.json")
    worker = _read(marker_path.parent.parent/"worker/slurm-session.json")
    if (worker.get("session_id") != record["new_session"] or str(worker.get("job_id")) == record["previous_job_id"]
            or not str(worker.get("job_id","")).isdigit() or baseline.get("ok") is not True
            or baseline.get("status") != "measured" or baseline.get("contract_hash") != identity["contract"]
            or baseline.get("candidate_id") != "baseline" or baseline.get("artifact_hash") != record["old_proof"]["baseline_artifact_hash"]
            or {key:row["latency_ms"] for key,row in baseline.get("cases",{}).items()} != latencies):
        raise RehostRefused("A fresh successful baseline on the new owned allocation is required")
    if any(baseline.get("environment",{}).get(key) != value for key,value in environment.items()):
        raise RehostRefused("Fresh baseline hardware differs from the new environment binding")
    prior = _read(marker_path.parent/"prior-context/baseline-measurement.json")
    if not baseline.get("evaluation_id") or baseline["evaluation_id"] == prior.get("evaluation_id"):
        raise RehostRefused("The old baseline receipt cannot be relabelled as fresh")
    for case in baseline["cases"].values():
        if len(case.get("series",[])) != contract["contract"]["benchmark_series"] or any(
            str(row.get("report",{}).get("slurm_job_id")) != str(worker["job_id"])
            or row["report"].get("ok") is not True
            or len(row.get("raw_ms",[])) != contract["contract"]["launches_per_series"]
            or any(not math.isfinite(value) or value <= 0 for value in row["raw_ms"])
            for row in case["series"]):
            raise RehostRefused("Fresh raw baseline series must belong to the new allocation")
    proof = {"context_id":record["context_id"],"old_identity":old,"new_identity":identity,
        "old_snapshot_sha256":record["old_proof"]["snapshot_sha256"],"old_rng_sha256":record["old_proof"]["rng_sha256"],
        "baseline_evaluation_id":baseline["evaluation_id"],"new_job_id":str(worker["job_id"]),
        "marker_sha256":_file_hash(marker_path),"baseline_sha256":_file_hash(marker_path.parent/"baseline-measurement.json"),
        "old_scores_inherited":False,"fresh_research_required":True,"bound_at_unix":time.time()}
    with store.transaction() as db:
        current = db.execute("SELECT identity,snapshot FROM campaigns WHERE run_id=?",(run_id,)).fetchone()
        if json.loads(current["identity"]) != old or hashlib.sha256(current["snapshot"]).hexdigest() != record["old_proof"]["snapshot_sha256"]:
            raise RehostRefused("The campaign advanced during identity binding")
        db.execute("UPDATE campaigns SET identity=? WHERE run_id=?",(_json(identity),run_id))
        store._event(db,run_id,"unscored_context_rebound",proof)
    return proof


def digest_environment(value):
    """Match CampaignRuntime.device_identity's persisted JSON fingerprint."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
