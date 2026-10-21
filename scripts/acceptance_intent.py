"""Durable ownership before any acceptance helper can submit a Slurm job."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess


def controller_provenance(repo, controller_file, execution_config):
    repo = Path(repo).resolve()
    paths = {Path(controller_file).resolve(), Path(execution_config).resolve(), Path(__file__).resolve()}
    paths.update((repo / "optimizer_agent").glob("*.py"))
    paths.update(path for path in (repo / "cuda_test_harness").rglob("*") if path.is_file()
        and path.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh", ".txt")
        and not any(part.startswith(("build", "cmake-build")) for part in path.relative_to(repo).parts))
    return {"HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).splitlines(),
        "sources": {str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path):
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}}


def persist_allocation_intent(directory, backend, source_provenance):
    """Exclusive and fsynced; a lost ACK still leaves one exact session to audit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "phase": "before_submission", "session_id": backend.session_id,
             "remote_dir": backend.remote_dir, "config": asdict(backend.config), "source_provenance": source_provenance}
    path = directory / "allocation-intent.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("Allocation intent already exists; reconcile the recorded session before any new submission") from error
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return value
