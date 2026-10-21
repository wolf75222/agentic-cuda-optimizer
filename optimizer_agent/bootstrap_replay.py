"""Replay an authenticated, interrupted pre-model bootstrap prefix only.

This is not a measurement cache. It consumes one ordered, frozen recording,
then delegates every fresh request. Only the recording's final transport error
can contact the runner during replay, with its original ID and exact payload.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .use_harness import KernelRequest


class BootstrapReplayError(RuntimeError):
    """Frozen bootstrap evidence or its execution context changed."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _read(path, label):
    try:
        return json.loads(path.read_bytes())
    except (ValueError, OSError) as error:
        raise BootstrapReplayError(f"Invalid {label}: {path}") from error


def _snapshot(path, role):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise BootstrapReplayError(f"Missing or unsafe {role} artifact: {path}")
    content = path.read_bytes()
    return {"path": str(path.absolute()), "role": role, "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def _check_file(snapshot):
    if _snapshot(snapshot["path"], snapshot["role"]) != snapshot:
        raise BootstrapReplayError(f"Changed {snapshot['role']} artifact: {snapshot['path']}")


def _request_data(request):
    data = asdict(request) if is_dataclass(request) else deepcopy(request)
    if not isinstance(data, dict):
        raise BootstrapReplayError("A complete KernelRequest is required")
    try:
        # Reject unknown fields/kinds instead of silently omitting them from
        # the semantic fingerprint. Preserve recorded path strings verbatim.
        return json.loads(_json(asdict(KernelRequest.from_dict(data))))
    except (ValueError, TypeError, KeyError) as error:
        raise BootstrapReplayError(f"Invalid bootstrap request: {error}") from error


def _semantic(request):
    data = _request_data(request)
    files = []

    def blob(path, role):
        item = _snapshot(path, role)
        files.append(item)
        return {"sha256": item["sha256"], "size": item["size"]}

    data["source"] = blob(data["source"], "source")
    for argument in data["arguments"]:
        if argument.get("kind") == "buffer" and argument.get("input_file") is not None:
            argument["input_file"] = blob(argument["input_file"], "input")
    data["dependencies"] = [{"name": Path(path).name, **blob(path, "dependency")}
                            for path in data["dependencies"]]
    # A compile-cache directory is a host path, not a compiler option or part
    # of the numerical request. All other request fields remain in the hash.
    if data["cache_dir"] is not None:
        data["cache_dir"] = {"local_compile_cache": True}
    return _hash(data), files


def _hardware(hardware):
    if not isinstance(hardware, dict):
        raise BootstrapReplayError("An authenticated hardware receipt is required")
    properties = hardware.get("device_properties", hardware)
    gpu = (properties.get("uuid") or properties.get("gpu_uuid")) if isinstance(properties, dict) else None
    job = hardware.get("slurm_job_id", hardware.get("job_id", ""))
    if (not isinstance(gpu, str) or not gpu or isinstance(job, bool)
            or not re.fullmatch(r"[1-9][0-9]*", str(job))):
        raise BootstrapReplayError("Hardware receipt must identify the allocation and GPU UUID")
    return json.loads(_json(hardware))


def _report_files(report, hardware):
    if not isinstance(report, dict) or report.get("hardware_fingerprint") != hardware:
        raise BootstrapReplayError("Report hardware differs from the authenticated allocation")
    job = str(hardware.get("slurm_job_id", hardware.get("job_id")))
    if report.get("slurm_job_id") is not None and str(report["slurm_job_id"]) != job:
        raise BootstrapReplayError("Report belongs to a different Slurm job")
    environment = report.get("environment", {})
    properties = hardware.get("device_properties", hardware)
    gpu_key = "uuid" if "uuid" in properties else "gpu_uuid"
    if not isinstance(environment, dict) or environment.get(gpu_key) != properties[gpu_key]:
        raise BootstrapReplayError("Native report hardware must identify the same GPU UUID")
    # Native free-memory counters can legitimately vary within one allocation.
    for key in ("uuid", "gpu_uuid", "architecture", "harness_build_id", "driver_version", "nvrtc_version"):
        if key in environment and key in properties and environment[key] != properties[key]:
            raise BootstrapReplayError(f"Native report hardware changed: {key}")
    outputs = []
    for output in report.get("outputs", []):
        item = _snapshot(output["path"], "output")
        if output.get("bytes") is not None and output["bytes"] != item["size"]:
            raise BootstrapReplayError("Output artifact has the wrong byte count")
        outputs.append(item)
    return outputs


def create_manifest(receipt_directory, hardware):
    """Freeze the complete ordered receipt prefix; does no runner/GPU work.

    The caller must establish that generation is still before its first model
    call and hold its campaign lock. Persist the returned JSON before resume.
    Original receipts, sources, inputs and returned output paths must remain.
    """
    hardware = _hardware(hardware)
    directory = Path(receipt_directory).absolute()
    recordings = []
    for path in directory.glob("*.json"):
        snapshot = _snapshot(path, "request receipt")
        recording = _read(path, "request receipt")
        if not isinstance(recording, dict):
            raise BootstrapReplayError("Request receipt must be an object")
        started = recording.get("started_at_unix")
        if isinstance(started, bool) or not isinstance(started, (int, float)) or not math.isfinite(started):
            raise BootstrapReplayError("Receipt requires an unambiguous started_at_unix timestamp")
        recordings.append((started, path.name, snapshot, recording))
    recordings.sort(key=lambda item: (item[0], item[1]))
    if not recordings:
        raise BootstrapReplayError("Bootstrap receipt prefix is empty")
    if len({item[0] for item in recordings}) != len(recordings):
        raise BootstrapReplayError("Ambiguous bootstrap request order: duplicate start timestamp")
    entries, identifiers, failures = [], set(), []
    for index, (started, _, snapshot, row) in enumerate(recordings):
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id or run_id in identifiers:
            raise BootstrapReplayError("Missing or duplicate original request ID")
        identifiers.add(run_id)
        if ("report" in row) == ("error" in row):
            raise BootstrapReplayError("Receipt must contain exactly one report or transport error")
        ended = row.get("ended_at_unix")
        if (isinstance(ended, bool) or not isinstance(ended, (int, float))
                or not math.isfinite(ended) or ended < started):
            raise BootstrapReplayError("Bootstrap receipt has no valid completion timestamp")
        if index + 1 < len(recordings) and ended > recordings[index + 1][0]:
            raise BootstrapReplayError("Bootstrap receipts overlap; a serial prefix is required")
        original = _request_data(row.get("request"))
        fingerprint, files = _semantic(original)
        entry = {"position": index, "original_run_id": run_id, "original_request": original,
                 "semantic_fingerprint": fingerprint, "request_files": files,
                 "receipt_file": snapshot, "started_at_unix": started, "ended_at_unix": ended}
        if "error" in row:
            failures.append(index)
            entry.update(kind="reconcile", original_error=str(row["error"]))
        else:
            report = deepcopy(row["report"])
            entry.update(kind="completed", report=report, report_hash=_hash(report),
                         output_files=_report_files(report, hardware))
        entries.append(entry)
    if failures and failures != [len(entries) - 1]:
        raise BootstrapReplayError("Only one final interrupted request can be reconciled")
    manifest = {"schema_version": 1, "kind": "pre_model_bootstrap_prefix",
                "receipt_directory": str(directory), "hardware": hardware,
                "hardware_hash": _hash(hardware), "entries": entries}
    manifest["manifest_hash"] = _hash(manifest)
    _verify_manifest(manifest)
    return manifest


def _verify_manifest(manifest, *, verify_request_files=True):
    if not isinstance(manifest, dict):
        raise BootstrapReplayError("Bootstrap manifest must be an object")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "pre_model_bootstrap_prefix"
            or manifest.get("manifest_hash") != _hash(body)):
        raise BootstrapReplayError("Bootstrap manifest SHA256/schema mismatch")
    hardware = _hardware(manifest["hardware"])
    if _hash(hardware) != manifest.get("hardware_hash"):
        raise BootstrapReplayError("Bootstrap hardware hash mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise BootstrapReplayError("Bootstrap manifest prefix is empty")
    for index, entry in enumerate(entries):
        if entry.get("position") != index or entry.get("kind") not in ("completed", "reconcile"):
            raise BootstrapReplayError("Invalid bootstrap prefix position")
        _check_file(entry["receipt_file"])
        _request_data(entry["original_request"])
        if verify_request_files:
            fingerprint, files = _semantic(entry["original_request"])
            if fingerprint != entry["semantic_fingerprint"] or files != entry["request_files"]:
                raise BootstrapReplayError("Frozen bootstrap source/input/request changed")
            for file in files:
                _check_file(file)
        if entry["kind"] == "completed":
            if _hash(entry["report"]) != entry["report_hash"]:
                raise BootstrapReplayError("Frozen bootstrap report changed")
            if _report_files(entry["report"], hardware) != entry["output_files"]:
                raise BootstrapReplayError("Frozen bootstrap outputs changed")
        elif index != len(entries) - 1:
            raise BootstrapReplayError("An interrupted request must end the bootstrap prefix")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as file:
            file.write(_json(value) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_position(path, state):
    body = {key: value for key, value in state.items() if key != "journal_hash"}
    _write(path, {**body, "journal_hash": _hash(body)})


class BootstrapReplayRunner:
    def __init__(self, runner, manifest, expected_hardware, audit_directory, *, defer_request_files=False):
        """Bind evidence before a native workspace regenerates its inputs.

        Explicit ``defer_request_files=True`` only defers source/input/dependency
        byte checks at construction. Every run (including the first) verifies
        the entire prefix's request files before returning a replay or calling
        a runner. Receipt/output/manifest/hardware checks are never deferred.
        """
        if type(defer_request_files) is not bool:
            raise BootstrapReplayError("defer_request_files must be an explicit boolean")
        self.runner = runner
        self._manifest_path = Path(manifest).absolute() if isinstance(manifest, (str, Path)) else None
        self._manifest_file = _snapshot(self._manifest_path, "bootstrap manifest") if self._manifest_path else None
        self._manifest = _read(self._manifest_path, "bootstrap manifest") if self._manifest_path else deepcopy(manifest)
        self._expected_hardware = _hardware(expected_hardware)
        self.audit_directory = Path(audit_directory).absolute()
        self._position = 0
        self._check_context(verify_request_files=not defer_request_files)
        self.audit_directory.mkdir(parents=True, exist_ok=True)
        with (self.audit_directory / "replay.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            binding = self.audit_directory / "manifest.json"
            if binding.exists():
                if _read(binding, "replay journal manifest") != self._manifest:
                    raise BootstrapReplayError("Replay journal belongs to another frozen manifest")
            else:
                _write(binding, self._manifest)

    def __getattr__(self, name):
        return getattr(self.runner, name)

    @property
    def replay_position(self):
        return self._position

    @property
    def replay_complete(self):
        return self._position == len(self._manifest["entries"])

    def require_complete(self):
        """Call at the pre-model boundary; a shorter bootstrap must not pass."""
        self._check_context()
        if not self.replay_complete:
            raise BootstrapReplayError("Bootstrap ended before the frozen request prefix was consumed")

    def _check_context(self, *, verify_request_files=True):
        if self._manifest_file:
            _check_file(self._manifest_file)
        _verify_manifest(self._manifest, verify_request_files=verify_request_files)
        if (self._manifest["hardware"] != self._expected_hardware
                or getattr(self.runner, "hardware", None) != self._expected_hardware):
            raise BootstrapReplayError("Bootstrap requires the same authenticated allocation/GPU hardware")
        binding = self.audit_directory / "manifest.json"
        if binding.exists() and _read(binding, "replay journal manifest") != self._manifest:
            raise BootstrapReplayError("Frozen replay journal manifest changed")

    def run(self, run_id, request):
        with (self.audit_directory / "replay.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._check_context()
            if self.replay_complete:
                # Never reuse a matching measurement beyond the frozen prefix.
                return self.runner.run(run_id, request)
            entry = self._manifest["entries"][self._position]
            if _semantic(request)[0] != entry["semantic_fingerprint"]:
                raise BootstrapReplayError(f"Bootstrap request order/content mismatch at position {self._position}")
            path = self.audit_directory / f"position-{self._position:06d}.json"
            identity = {"manifest_hash": self._manifest["manifest_hash"], "position": self._position,
                        "original_run_id": entry["original_run_id"], "semantic_fingerprint": entry["semantic_fingerprint"]}
            previous = _read(path, "replay position journal") if path.exists() else None
            if path.exists() and not isinstance(previous, dict):
                raise BootstrapReplayError("Replay position journal must be an object")
            if previous is not None and previous.get("journal_hash") != _hash(
                    {key: value for key, value in previous.items() if key != "journal_hash"}):
                raise BootstrapReplayError("Replay position journal SHA256 changed")
            if previous is not None and any(previous.get(key) != value for key, value in identity.items()):
                raise BootstrapReplayError("Replay position journal identity changed")
            if previous is not None and previous.get("status") == "completed":
                report = previous["report"]
                if (_hash(report) != previous.get("report_hash")
                        or _report_files(report, self._expected_hardware) != previous.get("output_files")):
                    raise BootstrapReplayError("Completed replay journal/artifacts changed")
            else:
                if previous is not None and previous.get("status") not in ("prepared", "awaiting_reconciliation"):
                    raise BootstrapReplayError("Unknown replay position journal state")
                state = {**identity, "status": "prepared", "kind": entry["kind"],
                         "attempts": (previous or {}).get("attempts", 0) + 1}
                _write_position(path, state)
                try:
                    report = (deepcopy(entry["report"]) if entry["kind"] == "completed" else
                              self.runner.run(entry["original_run_id"],
                                              KernelRequest.from_dict(deepcopy(entry["original_request"]))))
                    self._check_context()
                    outputs = _report_files(report, self._expected_hardware)
                    state.update(status="completed", report=deepcopy(report), report_hash=_hash(report), output_files=outputs)
                    _write_position(path, state)
                except Exception as error:
                    state.update(status="awaiting_reconciliation", error=str(error))
                    _write_position(path, state)
                    raise
            self._position += 1
            return deepcopy(report)
