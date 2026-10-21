"""Authenticated allocation metadata for research, separate from timing identity."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import re


def _uuid(value):
    value = str(value).removeprefix("GPU-").replace("-", "").lower()
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("Hardware research context requires an unambiguous physical GPU UUID")
    return value


def archived_hardware_research_context(worker_directory, environment, *, expected_job_id,
                                      expected_session_id, expected_remote_dir=None,
                                      expected_hardware=None, expected_receipt_hashes=None):
    """Read only the fixed hardware/ownership receipts, never infer compatibility.

    An allocation-start observation remains historical even if its worker is
    still running. This annex does not alter a contract or environment hash.
    """
    directory = Path(worker_directory).resolve()
    records, references = {}, {}
    for name in ("hardware.json", "slurm-session.json"):
        path = directory / name
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError("Invalid hardware research receipt")
        body = path.read_bytes()
        record = json.loads(body)
        if not isinstance(record, dict):
            raise ValueError("Hardware research receipt must be an object")
        records[name] = record
        references[name] = {"path": str(path), "sha256": hashlib.sha256(body).hexdigest()}
    hardware, session = records["hardware.json"], records["slurm-session.json"]
    if expected_hardware is None and not expected_receipt_hashes:
        raise ValueError("Hardware research context needs an independent hardware/receipt binding")
    if expected_hardware is not None and hardware != expected_hardware:
        raise ValueError("Hardware research receipt differs from the authenticated runner")
    if expected_receipt_hashes is not None and any(expected_receipt_hashes.get(name) != ref["sha256"]
                                                  for name, ref in references.items()):
        raise ValueError("Hardware research receipt differs from its pinned archive")
    properties = hardware.get("device_properties")
    if (not isinstance(environment, dict) or not isinstance(properties, dict)
            or hardware.get("environment") != "romeo" or not session.get("session_id")
            or not str(expected_job_id).isdigit() or int(expected_job_id) <= 0
            or str(hardware.get("slurm_job_id")) != str(expected_job_id)
            or str(session.get("job_id")) != str(expected_job_id)
            or session["session_id"] != expected_session_id
            or session.get("uncertain") or "job_ids" in session
            or (expected_remote_dir is not None and session.get("remote_dir") != str(expected_remote_dir))):
        raise ValueError("Hardware research context lacks matching ROMEO ownership")
    physical_uuid = _uuid(properties.get("uuid"))
    if _uuid(environment.get("uuid")) != physical_uuid:
        raise ValueError("Hardware research context differs from the evaluator GPU")
    for key, value in environment.items():
        if key == "uuid":
            continue
        if properties.get(key) != value:
            raise ValueError("Hardware research context differs from evaluator " + key)
    rows = list(csv.reader(str(hardware.get("nvidia_smi", "")).splitlines()))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise ValueError("Hardware research context requires exactly one nvidia-smi GPU row")
    gpu, name, package, pci = [value.strip() for value in rows[0]]
    if (_uuid(gpu) != physical_uuid or not re.fullmatch(r"\d+(?:\.\d+){1,3}", package)
            or [_uuid(value) for value in hardware.get("gpu_uuids", [])] != [physical_uuid]):
        raise ValueError("Hardware research nvidia-smi identity is inconsistent")
    stable_keys = ("machine", "device_properties", "nvidia_smi", "toolkit", "compiler")
    if not all(key in hardware for key in stable_keys):
        raise ValueError("Hardware research fingerprint fields are incomplete")
    stable = {key: hardware[key] for key in stable_keys}
    fingerprint = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    if fingerprint != hardware.get("fingerprint_id"):
        raise ValueError("Hardware research fingerprint differs from its archived bytes")
    return {"schema_version": 1, "kind": "authenticated_allocation_start_hardware_context",
        "scope": "Historical allocation-start observation; not a current GPU query or a compatibility decision",
        "hardware_currently_observed": False, "job_id": str(hardware["slurm_job_id"]),
        "session_id": session["session_id"], "gpu_uuid": physical_uuid,
        "machine": hardware["machine"], "hostname": hardware.get("hostname"),
        "nvidia_driver_package_version": package, "nvidia_smi_gpu_name": name, "pci_bus_id": pci,
        "native_reported_versions": {key: deepcopy(properties[key]) for key in
            ("driver_version", "nvrtc_version", "cuda_toolkit_version") if key in properties},
        "nvcc_version_output": hardware["toolkit"], "host_compiler": hardware["compiler"],
        "hardware_fingerprint_id": fingerprint, "evidence": references,
        "producer": "remote_worker.fingerprint: nvidia-smi --query-gpu=uuid,name,driver_version,pci.bus_id --format=csv,noheader",
        "interpretation": "The package release is separate from cuDriverGetVersion's encoded CUDA API version. Consult official platform/version documentation; no applicability condition is supplied or satisfied by this annex."}


def runner_hardware_research_context(runner, environment):
    """Return an annex only for a runner carrying the owned ROMEO receipts."""
    if getattr(runner, "environment", None) != "romeo":
        return None
    backend = getattr(runner, "backend", None)
    if backend is None or not isinstance(getattr(runner, "hardware", None), dict):
        raise ValueError("ROMEO research context needs the authenticated runner/backend ownership")
    return archived_hardware_research_context(runner.results_dir, environment,
        expected_job_id=backend.job_id, expected_session_id=backend.session_id,
        expected_remote_dir=backend.remote_dir, expected_hardware=runner.hardware)
