"""Prepare an opaque copy and finalize only the already stopped original graph.

No scheduler, model, observer or harness operation is permitted. Historical
generation files and the suite journal remain unchanged. Commands other than
plan require primary review before execution against the real campaign.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import contextlib
import ctypes
import dataclasses
import datetime
import enum
import hashlib
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
HELPER_PATH = Path(__file__).with_name("inspect_original_775348_readonly.py")
HELPER_SHA = "5710bcad870dbd260971b79fb43ca9c31553f3bc2ac1b255f23b5a3c485d9d4d"
INSPECTION_SHA = "3903de251c9a45644277c5c8133fee98ae3bde511301ea866948d58a1f9b6604"
SPEC = importlib.util.spec_from_file_location("trusted_original_phase", HELPER_PATH)
h = importlib.util.module_from_spec(SPEC)
assert hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() == HELPER_SHA
SPEC.loader.exec_module(h)
Refused, require = h.Refused, h.require


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()


def sha(body):
    return hashlib.sha256(body).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, value):
    """Immutable phase record. An existing unequal record is never replaced."""
    body = encoded(value)
    if path.exists():
        require(path.read_bytes() == body, "transaction_record_collision")
        return
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    libc = ctypes.CDLL(None,use_errno=True)
    if hasattr(libc,"renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(os.fsencode(temporary),os.fsencode(path),4) # Darwin RENAME_EXCL
    elif hasattr(libc,"renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
        function.restype = ctypes.c_int
        result = function(-100,os.fsencode(temporary),-100,os.fsencode(path),1) # Linux NOREPLACE
    else:
        raise Refused("exclusive_atomic_publication_unavailable")
    if result:
        code = ctypes.get_errno()
        raise OSError(code,"Exclusive phase publication refused")
    sync_directory(path.parent)


def canonical(value):
    """Private comparison only; no object repr/pickle fallback or public values."""
    from pydantic import BaseModel
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return {"float_hex": value.hex()}
    if isinstance(value,enum.Enum):
        return {"enum":type(value).__qualname__,"value":canonical(value.value)}
    if isinstance(value, Path):
        return {"path": str(value)}
    if type(value) is bytes:
        return {"bytes_sha256": sha(value), "bytes_length": len(value)}
    if dataclasses.is_dataclass(value):
        return {"dataclass": type(value).__qualname__,
                "fields": {f.name: canonical(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, BaseModel):
        return {"model": type(value).__qualname__, "fields": canonical(value.model_dump(mode="python"))}
    if isinstance(value, dict):
        require(all(type(k) is str for k in value), "unsupported_private_mapping_key")
        return {k: canonical(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return {"set": sorted([canonical(v) for v in value], key=lambda x: encoded(x))}
    raise Refused("unsupported_private_state_type")


def scientific_digest(values):
    # session_dir is the sole typed relocation. All science, old messages,
    # configuration, inputs, history and model counters remain unchanged.
    return sha(encoded(canonical({k:v for k,v in values.items() if k != "session_dir"})))


def inventory(root):
    """Opaque bytes, never decode input buffers/checkpoints/prompts or array data."""
    root = h.safe(root, root)
    require(root.is_dir(), "source_directory_missing")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "opaque_copy_symlink_refused")
        info = path.stat()
        require(info.st_uid == os.getuid(), "opaque_copy_foreign_owner")
        name = str(path.relative_to(root))
        if stat.S_ISDIR(info.st_mode):
            result[name] = {"kind": "directory", "mode": stat.S_IMODE(info.st_mode)}
            continue
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "opaque_copy_special_or_alias_refused")
        require(not path.name.endswith(("-wal", "-shm", "-journal")), "opaque_source_not_cold")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        require(h.signature(info) == h.signature(path.stat()), "opaque_source_changed_while_reading")
        result[name] = {"kind": "file", "mode": stat.S_IMODE(info.st_mode), "bytes": info.st_size,
                        "mtime_ns": info.st_mtime_ns, "sha256": digest.hexdigest()}
    return result


def copy_opaque(source, destination, expected, crash_hook=None, staging_root=None):
    require(inventory(source) == expected, "opaque_source_drifted")
    h.safe(destination,destination)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(not destination.is_symlink(), "opaque_destination_aliased")
    staging_root = Path(staging_root or destination.parent/"opaque-copy-stage")
    h.safe(staging_root,staging_root)
    staging_root.mkdir(mode=0o700,parents=True,exist_ok=True)
    for relative, item in expected.items():
        target, origin = h.safe(destination/relative,destination), source/relative
        if item["kind"] == "directory":
            target.mkdir(mode=item["mode"], parents=True, exist_ok=True)
            os.chmod(target,item["mode"])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = staging_root/hashlib.sha256(relative.encode()).hexdigest()
        staging.mkdir(mode=0o700,exist_ok=True)
        if target.exists():
            require(not target.is_symlink() and target.is_file() and sha(target.read_bytes()) == item["sha256"],
                    "partial_copy_collision")
            # os.link publishes exclusively. A process crash between publication
            # and removal of its duplicate staging name is safely reconcilable.
            info = target.stat()
            aliases = [p for p in staging.iterdir() if not p.is_symlink()
                       and (p.stat().st_dev,p.stat().st_ino)==(info.st_dev,info.st_ino)]
            require(info.st_nlink == len(aliases)+1, "opaque_copy_unowned_hardlink")
            for alias in aliases: alias.unlink()
            shutil.copystat(origin,target,follow_symlinks=False)
            sync_directory(staging)
            continue
        temporary = staging/("%06d.part" % len(list(staging.iterdir())))
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with origin.open("rb") as src, os.fdopen(fd, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024*1024)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(origin,temporary,follow_symlinks=False)
        with temporary.open("rb") as stream: os.fsync(stream.fileno())
        require(sha(temporary.read_bytes()) == item["sha256"], "opaque_staging_bytes_changed")
        os.link(temporary,target,follow_symlinks=False)
        sync_directory(target.parent)
        temporary.unlink()
        sync_directory(staging)
        if crash_hook:
            crash_hook("copied_file")
    require(inventory(source) == expected and inventory(destination) == expected, "opaque_copy_identity_mismatch")
    for path in sorted((p for p in destination.rglob("*") if p.is_dir()), reverse=True):
        sync_directory(path)
    sync_directory(destination)


def outputs(destination, expected_source):
    summary = h.read(destination/"summary.json", destination)
    require(summary["status"] == "complete" and summary["error"] is None
            and summary["termination"] == "iteration_limit" and summary["best_iteration"] == 2
            and summary["attempt_count"] == 4, "finalized_summary_invalid")
    require(sha(h.raw(destination/"best.cu", destination)) == expected_source, "finalized_source_changed")
    cases = summary["cases"]
    require(sorted(c["case_index"] for c in cases) == list(range(7)), "finalized_case_set_changed")
    paths = [destination/"summary.json", destination/"best.cu"]
    for case in cases:
        path = destination/("best-case-"+str(case["case_index"])+".json")
        require(case["request_path"] == str(path), "finalized_request_path_changed")
        request = h.read(path, destination)
        require(request["source"] == str(destination/"best.cu") and request["results_dir"] == str(destination),
                "finalized_request_destination_changed")
        paths.append(path)
    if summary.get("timing_plot_path"):
        require(summary["timing_plot_path"] == str(destination/"heatmap.png"), "finalized_plot_path_changed")
        paths.append(destination/"heatmap.png")
    for path in paths:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    sync_directory(destination)
    return [h.proof(p, destination) for p in paths]


def recover_owned_copy_journal(source,destination,transaction):
    """SQLite may leave WAL after process death. Archive it; never delete it.

    Only the operation's independent destination can be recovered, after a
    durable typed relocation intent. SQLite itself checkpoints/closes its WAL.
    The outer command owns the historical task/controller locks throughout.
    """
    database = destination/"original-langgraph.sqlite"
    suffixes = ("-wal","-shm","-journal")
    sidecars = [Path(str(database)+s) for s in suffixes if Path(str(database)+s).exists()]
    if not sidecars: return
    require(source != destination and (transaction/"relocation-intent.json").exists(),
            "unowned_copy_journal_refused")
    intent = h.read(transaction/"relocation-intent.json",transaction)
    require(intent["source"] == str(source) and intent["destination"] == str(destination),
            "copy_journal_ownership_changed")
    archives = transaction/"copy-crash-opaque-archives"
    archives.mkdir(mode=0o700,exist_ok=True)
    archive = archives/("%03d" % len(list(archives.iterdir())))
    archive.mkdir(mode=0o700,exist_ok=False)
    pins = []
    for path in [database,*sidecars]:
        body = h.raw(path,destination)
        target = archive/(path.name+".opaque")
        fd = os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"wb") as stream:
            stream.write(body);stream.flush();os.fsync(stream.fileno())
        pins.append({"name":path.name,"sha256":sha(body),"bytes":len(body)})
    for pin in pins:
        require(sha(h.raw(destination/pin["name"],destination)) == pin["sha256"], "copy_journal_changed_before_recovery")
    publish(archive/"prepared.json",{"kind":"owned_copy_sqlite_recovery_preparation","files":pins})
    with contextlib.closing(sqlite3.connect(database.as_uri()+"?mode=rw",uri=True)) as connection:
        require(connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "copy_sqlite_integrity_refused")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        require(checkpoint is not None and checkpoint[0] == 0,"copy_sqlite_recovery_busy")
    require(not any(Path(str(database)+s).exists() for s in suffixes), "copy_sqlite_journal_still_active")
    publish(archive/"committed.json",{"kind":"owned_copy_sqlite_recovery_completed",
        "database_sha256":sha(h.raw(database,destination)),"manually_deleted_sidecars":False})


def finalize_copy_only(source_generation, destination_generation, snapshot_path, config, history,
                       transaction_dir, expected_source_sha256, *, crash_hook=None):
    """No historical writes. Engine is also exercised by synthetic graph fixtures."""
    for path in (source_generation,destination_generation,transaction_dir):
        h.safe(path,path)
    source, destination, transaction = map(lambda p: Path(p).resolve(),
        (source_generation, destination_generation, transaction_dir))
    require(source != destination and not destination.is_relative_to(source), "copy_not_independent")
    require(transaction.is_dir() and not transaction.is_relative_to(source), "transaction_not_independent")
    require(not (transaction/"committed.json").exists(), "use_readonly_committed_replay")
    sys.path.insert(0, str(REPO))
    from optimizer_agent import upstream_original as bridge
    from langgraph.graph import StateGraph
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpoint = destination/"original-langgraph.sqlite"
    recover_owned_copy_journal(source,destination,transaction)
    require(checkpoint.is_file() and not checkpoint.is_symlink()
            and not any(Path(str(checkpoint)+s).exists() for s in ("-wal", "-shm", "-journal")), "copy_checkpoint_not_cold")
    source_before = inventory(source)
    node_record = transaction/"node-returned.json"
    relocation_path = transaction/"relocated.json"
    executed = 0
    replayed_return = 0
    forbidden_nodes = []
    started = time.monotonic()
    class NoExecution:
        def bind_tools(self, *args, **kwargs): return self
        def __getattr__(self, name): raise Refused("model_or_gpu_execution_forbidden")
    with bridge.original_modules(snapshot_path, source) as original:
        original_config = original.optimizer_config.OptimizerConfig(source,0,config["max_trials"],
            config["warmup"],config["launches_per_series"],"gpt-6-luna",config["max_repairs_per_trial"],
            nvidia_research=False,use_nsight=False,reasoning_effort="max")
        with SqliteSaver.from_conn_string(str(checkpoint)) as saver:
            class FinalizationGraph(StateGraph):
                def add_node(self, name, action=None, **kwargs):
                    require(name in h.NODES, "unrecognized_original_node")
                    original_action = action
                    if name != "finalize_result":
                        def protected(state):
                            forbidden_nodes.append(name)
                            raise Refused("non_finalizer_node_execution_forbidden")
                    else:
                        def protected(state):
                            nonlocal executed, replayed_return
                            require(Path(state["session_dir"]) == destination, "finalizer_would_write_historical_directory")
                            require(scientific_digest(state) == private_before, "private_science_changed_before_finalizer")
                            if node_record.exists():
                                saved = h.read(node_record, transaction)
                                require(saved["result"] == {"error": None}
                                        and saved["private_scientific_digest"] == private_before
                                        and saved["outputs"] == outputs(destination, expected_source_sha256), "durable_finalizer_return_changed")
                                replayed_return += 1
                                return saved["result"]
                            # A crash before node-returned may repeat this CPU
                            # export only; no kernel/validation/model is repeated.
                            attempt = len(list(transaction.glob("cpu-finalize-started-*.json")))
                            publish(transaction/("cpu-finalize-started-%03d.json" % attempt),
                                    {"kind":"cpu_only_original_finalizer_started", "attempt":attempt,
                                     "private_scientific_digest":private_before})
                            result = original_action(state)
                            executed += 1
                            require(result == {"error": None}, "original_finalizer_return_changed")
                            publish(node_record, {"kind":"durable_original_finalizer_return", "result":result,
                                "private_scientific_digest":private_before,
                                "outputs":outputs(destination, expected_source_sha256)})
                            if crash_hook: crash_hook("after_node_returned")
                            return result
                    return super().add_node(name, protected, **kwargs)
                def compile(self, *args, **kwargs):
                    kwargs["checkpointer"] = saver
                    return super().compile(*args, **kwargs)
            original.workflow.StateGraph = FinalizationGraph
            original.workflow.ChatOpenAI = lambda **kwargs: NoExecution()
            graph = original.workflow.build_graph(NoExecution(),NoExecution(),original_config)
            graph_config = {"configurable":{"thread_id":config["run_id"]}, "recursion_limit":8}
            state = graph.get_state(graph_config)
            phase = h.project_snapshot(state,history,expected_source_sha256,original.nodes.choose_next)
            private_before = scientific_digest(state.values)
            if relocation_path.exists():
                relocation = h.read(relocation_path, transaction)
                require(Path(state.values["session_dir"]) == destination
                        and relocation["private_scientific_digest"] == private_before, "relocated_checkpoint_changed")
            else:
                intent = {"kind":"typed_original_output_directory_relocation_intent",
                    "private_scientific_digest":private_before,"source":str(source),"destination":str(destination)}
                intent_path = transaction/"relocation-intent.json"
                require(phase["next_nodes"] == ["finalize_result"], "unprepared_original_phase")
                if Path(state.values["session_dir"]) == source:
                    publish(intent_path,intent)
                    # Only this typed output directory is changed in the COPY.
                    graph.update_state(graph_config,{"session_dir":destination},as_node="evaluate_candidate")
                    if crash_hook: crash_hook("after_checkpoint_relocation_before_receipt")
                else:
                    require(Path(state.values["session_dir"]) == destination and intent_path.exists()
                            and h.read(intent_path,transaction) == intent,"unprepared_relocated_checkpoint")
                relocated = graph.get_state(graph_config)
                require(scientific_digest(relocated.values) == private_before
                        and Path(relocated.values["session_dir"]) == destination, "typed_relocation_changed_science")
                h.project_snapshot(relocated,history,expected_source_sha256,original.nodes.choose_next)
                publish(relocation_path,{"kind":"typed_original_output_directory_relocation",
                    "private_scientific_digest":private_before,"source":str(source),"destination":str(destination)})
                if crash_hook: crash_hook("after_relocation")
            state = graph.get_state(graph_config)
            if state.next:
                for private_state in graph.stream(None,graph_config,stream_mode="values"):
                    require(scientific_digest(private_state) == private_before, "private_science_changed_during_finalization")
            ended = graph.get_state(graph_config)
            end_projection = h.project_snapshot(ended,history,expected_source_sha256,original.nodes.choose_next)
            require(not ended.next and not ended.tasks and node_record.exists()
                    and scientific_digest(ended.values) == private_before and not forbidden_nodes,
                    "original_graph_not_safely_ended")
            output_proof = outputs(destination, expected_source_sha256)
    require(inventory(source) == source_before, "historical_generation_changed")
    require(not any(Path(str(checkpoint)+s).exists() for s in ("-wal", "-shm", "-journal")), "copy_checkpoint_still_open")
    if crash_hook: crash_hook("after_graph_end_before_commit")
    return {"kind":"isolated_original_graph_finalization", "phase":end_projection["phase"],
        "private_scientific_digest":private_before,"outputs":output_proof,
        "cpu_finalizer_bodies_executed_this_invocation":executed,
        "durable_node_returns_replayed_this_invocation":replayed_return,
        "cpu_elapsed_seconds_this_invocation":time.monotonic()-started,
        "durable_cpu_start_records":len(list(transaction.glob("cpu-finalize-started-*.json"))),
        "model_calls":0,"gpu_requests":0,"observer_calls":0,"scheduler_calls":0,
        "historical_generation_unchanged":True,"method_status":"historical_failed_not_relabelled",
        "new_observer_epoch":"required_before_method_admission",
        "selected_export_independent_validation":"required_before_method_admission"}


def paths():
    study = REPO/"results/quality-study-v2"
    source = study/"campaigns"/h.TASK/"generation"
    target = source.parent/"attempts/001/generation"
    transaction = study/"original-finalize-recoveries"/hashlib.sha256(h.TASK.encode()).hexdigest()/"000"
    return study,source,target,transaction


def replay(transaction, target):
    commit = h.read(transaction/"committed.json", transaction)
    require(commit["destination"] == str(target)
            and commit["operational_tool_sha256"] == sha(Path(__file__).read_bytes())
            and commit["prepared_sha256"] == sha(h.raw(transaction/"prepared.json",transaction))
            and sha(encoded(inventory(target))) == commit["final_copy_inventory_sha256"], "committed_copy_drifted")
    return {"phase":"committed_readonly_replay", "committed_sha256":sha(h.raw(transaction/"committed.json",transaction)),
            "model_calls":0,"gpu_requests":0,"observer_calls":0,"scheduler_calls":0,"checkpoint_api_calls":0}


def operate(command):
    study,source,target,transaction = paths()
    if command == "replay": return replay(transaction,target)
    if command == "prepare":
        assessment = h.assess()
        actual = h.raw(REPO/"work/live/original775348-checkpoint-readonly-002.json",REPO)
        require(sha(actual) == INSPECTION_SHA, "reviewed_initial_inspection_changed")
        reviewed = json.loads(actual)
        require(assessment["checkpoint_projection"]["checkpoint_opaque_sha256"] ==
                reviewed["checkpoint_projection"]["checkpoint_opaque_sha256"], "source_checkpoint_changed_since_review")
    with h.existing_lock(study/"controller.lock",REPO), h.existing_lock(
            study/"task-locks"/(hashlib.sha256(h.TASK.encode()).hexdigest()+".lock"),REPO):
        h.public_task(study)
        require(not (transaction/"committed.json").exists(), "use_readonly_committed_replay")
        if command == "prepare":
            source_inventory = inventory(source)
            if transaction.exists():
                prepared = h.read(transaction/"prepared.json",transaction)
                require(prepared["source_inventory_sha256"] == sha(encoded(source_inventory))
                        and prepared["destination"] == str(target)
                        and prepared["operational_tool_sha256"] == sha(Path(__file__).read_bytes()),
                        "prepared_source_or_destination_changed")
                require(not (transaction/"relocated.json").exists(), "copy_already_finalizing")
            else:
                require(not target.exists(), "new_attempt_destination_exists")
                transaction.mkdir(mode=0o700,parents=True,exist_ok=False)
                publish(transaction/"source-opaque-inventory.json",source_inventory)
                prepared = {"schema_version":1,"kind":"cold_original_finalize_only_preparation",
                    "task":h.TASK,"previous_job_id":h.JOB,"previous_session":h.SESSION,
                    "source":str(source),"destination":str(target),"new_attempt":1,
                    "source_inventory_sha256":sha(encoded(source_inventory)),
                    "source_assessment_sha256":sha(encoded(assessment)),
                    "helper_sha256":HELPER_SHA,"reviewed_initial_inspection_sha256":INSPECTION_SHA,
                    "operational_tool_sha256":sha(Path(__file__).read_bytes()),
                    "historical_state":"failed_preserved","new_worker_session":None,
                    "model_calls":0,"gpu_requests":0,"scheduler_calls":0,
                    "scope":"Opaque local copy only. No suite journal activation, new observation, validation, performance claim or method relabeling."}
                publish(transaction/"assessment-public.json",assessment)
                publish(transaction/"prepared.json",prepared)
            copy_opaque(source,target,source_inventory,staging_root=transaction/"opaque-copy-stage")
            publish(transaction/"copy-complete.json",{"kind":"verified_opaque_generation_copy",
                "inventory_sha256":sha(encoded(source_inventory)),"files":sum(v["kind"]=="file" for v in source_inventory.values())})
            return {"phase":"prepared_copy_only","transaction":str(transaction),"destination":str(target),
                "model_calls":0,"gpu_requests":0,"scheduler_calls":0,"suite_journal_updated":False}
        prepared = h.read(transaction/"prepared.json",transaction)
        require(prepared["operational_tool_sha256"] == sha(Path(__file__).read_bytes()), "operational_source_changed")
        require(sha(h.raw(transaction/"assessment-public.json",transaction)) == prepared["source_assessment_sha256"],
                "prepared_assessment_changed")
        frozen_inventory = h.read(transaction/"source-opaque-inventory.json",transaction)
        require(sha(encoded(frozen_inventory)) == prepared["source_inventory_sha256"]
                and inventory(source) == frozen_inventory, "historical_generation_drifted")
        copy_receipt = h.read(transaction/"copy-complete.json",transaction)
        require(copy_receipt["inventory_sha256"] == prepared["source_inventory_sha256"], "copy_receipt_changed")
        if not (transaction/"relocation-intent.json").exists():
            require(inventory(target) == frozen_inventory, "unfinalized_copy_drifted")
        manifest = h.read(source/"manifest.json",source)
        snapshot,_ = h.authenticate_original(REPO,source,manifest)
        sys.path.insert(0,str(REPO))
        from optimizer_agent import upstream_original as bridge
        h.closed_cli_receipts(source,bridge,manifest)
        sink = io.StringIO()
        logging_before = logging.root.manager.disable
        prior_mpl = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = str(transaction/"matplotlib-config")
        try:
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stdout(sink),contextlib.redirect_stderr(sink):
                result = finalize_copy_only(source,target,snapshot,h.read(source/"config.json",source),
                    h.read(source/"history.json",source),transaction,h.SOURCE)
        finally:
            logging.disable(logging_before)
            if prior_mpl is None: os.environ.pop("MPLCONFIGDIR",None)
            else: os.environ["MPLCONFIGDIR"] = prior_mpl
        require(inventory(source) == frozen_inventory, "historical_generation_changed_after_finalization")
        completed_inventory = inventory(target)
        publish(transaction/"final-copy-opaque-inventory.json",completed_inventory)
        publish(transaction/"committed.json",{"schema_version":1,"kind":"isolated_original_finalize_only_commit",
            "task":h.TASK,"destination":str(target),"source_inventory_sha256":prepared["source_inventory_sha256"],
            "operational_tool_sha256":prepared["operational_tool_sha256"],
            "prepared_sha256":sha(h.raw(transaction/"prepared.json",transaction)),
            "final_copy_inventory_sha256":sha(encoded(completed_inventory)),"result":result,
            "suite_journal_updated":False,"historical_method_status":"failed",
            "method_admission":"pending_new_observer_epoch_and_independent_export_validation"})
        return {"phase":"committed_isolated_copy_only","transaction":str(transaction),"result":result,
                "suite_journal_updated":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","prepare","finalize","replay"),nargs="?",default="plan")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps({"phase":"protocol_only","requires_primary_review":True,"actual_checkpoint_opened":False,
            "scope":__doc__,"destination":str(paths()[2])},indent=2))
        return 0
    try:
        result = operate(args.command)
    except Refused as error:
        print(json.dumps({"status":"refused","code":str(error),"private_values_exposed":False}))
        return 2
    except Exception:
        print(json.dumps({"status":"refused","code":"isolated_finalization_internal_error","private_values_exposed":False}))
        return 2
    print(json.dumps(result,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
