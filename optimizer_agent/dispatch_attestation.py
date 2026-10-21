"""Observed local source identity before model/SSH/Slurm task actions.

This records on-disk Python bytes and loaded module paths, not loaded
bytecode, model weights, or evidence that a particular action happened.
New controller versions append epochs. Publication never overwrites existing
records, and recorded digest links are verified.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
SCOPE = "observed_on_disk_source_and_module_paths_before_model_ssh_slurm"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _git_identity(repo):
    def git(*args):
        return subprocess.check_output(["git", "--no-optional-locks", "-C", str(repo), *args],
                                       stderr=subprocess.PIPE, timeout=10)
    try:
        if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != repo:
            raise ValueError("Git root differs from the source directory")
        head = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain=v1", "--untracked-files=normal", "-z").decode(errors="surrogateescape")
        return {"available": True, "HEAD_at_dispatch": head, "dirty": bool(status),
                "porcelain_status": [item for item in status.split("\0") if item]}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"available": False, "HEAD_at_dispatch": None, "dirty": None,
                "limitation": "No verified Git root/HEAD available at capture"}


def _source_hashes(repo):
    files = sorted(path for folder in ("optimizer_agent", "scripts")
                   for path in (repo / folder).rglob("*.py") if path.is_file())
    if not files:
        raise ValueError("No application Python sources available for dispatch attestation")
    hashes = {}
    for path in files:
        if path.is_symlink():
            raise ValueError("Dispatch source must not be a symlink: " + str(path))
        hashes[str(path.relative_to(repo))] = _file_hash(path)
    return hashes


def _file_hash(path):
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during dispatch attestation: " + str(path))
    return hashlib.sha256(raw).hexdigest()


def _module_paths(repo):
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name != "__main__" and not name.startswith(("optimizer_agent", "scripts.")):
            continue
        path = getattr(module, "__file__", None)
        if path:
            actual = Path(path).resolve()
            paths[name] = {"path": str(actual),
                           "repository_relative": str(actual.relative_to(repo)) if actual.is_relative_to(repo) else None}
            if actual.is_file() and actual.suffix in {".py", ".pyc"}:
                paths[name]["on_disk_sha256"] = _file_hash(actual)
    return paths


def _archive_matches(study, sources):
    """Optional observed metadata/TAR digests; not a full Git TAR verification."""
    matches = []
    for metadata_path in sorted((study / "controller-sources").glob("*/source.json")):
        try:
            raw = metadata_path.read_bytes()
            metadata = json.loads(raw)
            archived = metadata.get("source_hashes", {})
            if not (metadata.get("dirty") == [] or metadata.get("dirty") is False) or not all(archived.get(name) == value for name, value in sources.items()):
                continue
            archive_path = metadata_path.with_name("source.tar.gz")
            archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            if archive_sha != metadata.get("archive_sha256"):
                raise ValueError("Matching controller source archive digest differs from its metadata")
            matches.append({"metadata_path": str(metadata_path.relative_to(study)),
                "metadata_sha256": hashlib.sha256(raw).hexdigest(), "archive_path": str(archive_path.relative_to(study)),
                "archive_sha256": archive_sha, "recorded_head": metadata.get("head"),
                "scope": "Observed archive digest and listed Python hashes match; full TAR/Git authentication is separate"})
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            # Optional unreadable/nonmatching archives confer no association.
            continue
    return matches


def _reference(study, path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Dispatch epoch record must be an object")
    body = {key:item for key,item in value.items() if key != "record_hash"}
    if value.get("schema_version") != 1 or value.get("record_hash") != _hash(body):
        raise ValueError("Dispatch epoch record digest/schema mismatch")
    return {"path": str(path.relative_to(study)), "sha256": hashlib.sha256(raw).hexdigest(),
            "epoch_id": value["epoch_id"]}, value


@contextmanager
def _append_lock(root):
    _mkdir_durable(root)
    with (root / ".append.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path):
    if path.is_dir():
        # Also finish a directory publication whose prior parent fsync failed.
        _fsync_directory(path.parent)
        return
    _mkdir_durable(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if not path.is_dir():
            raise
    _fsync_directory(path.parent)


def _chain(study, folder, kind, task_id=None):
    records, previous_ref = [], None
    for path in sorted(folder.glob("*.json")):
        reference, record = _reference(study, path)
        if (record.get("kind") != kind or record.get("study_directory") != str(study)
                or record.get("previous_epoch") != previous_ref
                or task_id is not None and record.get("task", {}).get("task_id") != task_id):
            raise ValueError("Dispatch epoch chain identity mismatch")
        records.append((reference, record))
        previous_ref = reference
    return records


def _write_immutable(path, raw):
    descriptor, temporary = tempfile.mkstemp(prefix=".epoch-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # Atomic publication with no overwrite.
        os.unlink(temporary)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_dispatch_attestation(study, *, task=None, controller_epoch=None, repo=REPO):
    """Append a source observation; configuration is retained as a hash only.

    Source changes during capture, a corrupt predecessor, or a different
    source than the current parent controller fail before task execution.
    A new controller may use a different source than earlier epochs.
    """
    study, repo = Path(study).resolve(), Path(repo).resolve()
    git = _git_identity(repo)
    sources = _source_hashes(repo)
    module_paths = _module_paths(repo)
    archives = _archive_matches(study, sources)
    git_after = _git_identity(repo)
    if (_source_hashes(repo) != sources
            or any(git_after.get(key) != git.get(key) for key in ("available", "HEAD_at_dispatch"))
            or any(_file_hash(Path(item["path"])) != item["on_disk_sha256"]
                   for item in module_paths.values() if "on_disk_sha256" in item)):
        raise ValueError("Source identity changed during dispatch attestation")
    identity = {"git": git, "python_source_hashes": sources}
    # Output receipts or unrelated documentation can change Git's dirty list.
    # They are observed context, not a change to the parent's runtime Python.
    identity["source_identity_hash"] = _hash({"git_available": git["available"],
        "HEAD_at_dispatch": git["HEAD_at_dispatch"], "python_source_hashes": sources})
    identity["source_identity_hash_scope"] = "Git HEAD/availability and application Python bytes; dirty status and loaded-module inventory are observed separately"
    root = study / "dispatch-attestations"
    kind = "task" if task is not None else "controller"
    binding = None
    if task is not None:
        if (task.get("kind") not in {"generation", "final"} or not isinstance(task.get("id"), str)
                or not task["id"] or not isinstance(task.get("session"), str) or not task["session"]
                or type(task.get("attempt")) is not int or task["attempt"] < 0):
            raise ValueError("Invalid dispatch task identity")
        binding = {"task_id": task["id"], "kind": task["kind"], "session": task["session"],
                   "attempt": task["attempt"], "payload_sha256": _hash(task["payload"])}
    with _append_lock(root):
        if controller_epoch is not None:
            parent_path = (study / controller_epoch["path"]).resolve()
            if parent_path.parent != root / "controller":
                raise ValueError("Controller epoch reference escapes the controller evidence directory")
            parents = _chain(study, root / "controller", "controller")
            parent = next((record for reference, record in parents if reference == controller_epoch), None)
            if (parent is None or parent.get("source_identity", {}).get("source_identity_hash") != identity["source_identity_hash"]):
                raise ValueError("Task source identity differs from its current controller epoch")
        folder = root / ("tasks/" + hashlib.sha256(task["id"].encode()).hexdigest() if task is not None else "controller")
        _mkdir_durable(folder)
        previous = _chain(study, folder, kind, task["id"] if task is not None else None)
        previous_ref = previous[-1][0] if previous else None
        epoch = "%06d-%s" % (len(previous) + 1, uuid.uuid4().hex)
        record = {"schema_version": 1, "epoch_id": epoch, "kind": kind, "scope": SCOPE,
            "study_directory": str(study), "observed_at_unix": time.time(), "pid": os.getpid(), "ppid": os.getppid(),
            "python": {"executable": sys.executable, "version": sys.version}, "source_identity": identity,
            "module_paths": module_paths, "matching_frozen_archive_observations": archives,
            "task": binding, "controller_epoch": controller_epoch, "previous_epoch": previous_ref,
            "limitations": ["On-disk observation does not attest loaded bytecode or model weights",
                            "An attested dispatch is not evidence that a model, SSH or GPU action occurred",
                            "No retrospective controller revision is assigned to earlier activity"]}
        record["record_hash"] = _hash(record)
        path = folder / (epoch + ".json")
        _write_immutable(path, _json(record) + b"\n")
        return _reference(study, path)[0]
