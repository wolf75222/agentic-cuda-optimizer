#!/usr/bin/env python3
"""Standalone Python 3.9 worker; CUDA commands only run under a Slurm job."""
import argparse
import base64
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import select
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time


EXCHANGE_CAPABILITY = "exchange_v1"


def exchange_liveness(directory, payload):
    """Login-side scheduler/file checks; this function never executes CUDA."""
    directory = Path(directory)
    try:
        receipt = json.loads((directory / "job_id.json").read_text())
        job_id = str(receipt["job_id"])
        if not job_id.isdigit() or receipt["session_id"] != directory.name:
            raise ValueError("Invalid worker ownership receipt")
        if payload.get("job_id", job_id) != job_id or payload.get("session_id", directory.name) != directory.name:
            raise ValueError("Exchange ownership does not match this worker")
        ready = json.loads((directory / "ready.json").read_text())
        recorded_job = ready.get("hardware", {}).get("slurm_job_id", ready.get("slurm_job_id"))
        if recorded_job is not None and str(recorded_job) != job_id:
            raise ValueError("Readiness belongs to another allocation")
    except (OSError, ValueError, KeyError) as error:
        return {"ok": False, "error_code": "ownership", "error": str(error)}
    if (directory / "quarantined.json").exists():
        return {"ok": False, "error_code": "quarantined", "error": "CUDA worker is quarantined; use a new worker session"}
    if (directory / "STOP").exists():
        return {"ok": False, "error_code": "not_running", "error": "Worker shutdown has already been requested"}
    process = run(["squeue", "-h", "-j", job_id, "-o", "%i|%T"], timeout=15)
    rows = [line.split("|") for line in process.stdout.splitlines()]
    row = next((row for row in rows if len(row) >= 2 and row[0] == job_id), None)
    if process.returncode == 0 and row and row[1] == "RUNNING" and ready.get("ok"):
        return None
    if process.returncode:
        return {"ok": False, "error_code": "scheduler", "error": process.stderr[-3000:]}
    process = run(["sacct", "-n", "-P", "-j", job_id, "--format=JobIDRaw,State"], timeout=15)
    rows = [line.split("|") for line in process.stdout.splitlines()]
    row = next((row for row in rows if len(row) >= 2 and row[0] == job_id), row)
    state = row[1].split()[0].rstrip("+") if row else "UNKNOWN"
    terminal = state in {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED",
                         "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"}
    return {"ok": False, "error_code": "terminal" if terminal else "not_running",
            "error": "Worker job " + job_id + " is " + state + "; use a new worker session if terminal", "state": state}


def exchange(directory, payload, wait_seconds):
    """Publish one immutable request and collect it with one idempotent SSH RPC.

    Only ``serve`` executes envelopes, inside the GPU allocation. Repeating this
    exchange after a lost acknowledgement can never create another attempt ID.
    Blob objects are scoped to this allocation and verified on every use.
    """
    directory = Path(directory).resolve()
    try:
        envelope = payload["envelope"]
        identity = envelope["identity"]
        expected = envelope["fingerprint"]
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", identity):
            raise ValueError("Unsafe request identity")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Invalid request fingerprint")
        if envelope["operation"] not in ("run", "profile", "sanitize"):
            raise ValueError("Unsupported request operation")
        manifest, uploads = payload.get("files", {}), payload.get("blobs", {})
        if not isinstance(manifest, dict) or not isinstance(uploads, dict):
            raise ValueError("Invalid blob manifest")
        for name, entry in manifest.items():
            if not re.fullmatch(r"(?:kernel\.cu|input-[0-9]+\.bin)", name):
                raise ValueError("Unsafe request file name")
            if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) or type(entry["size"]) is not int or entry["size"] < 0:
                raise ValueError("Invalid blob identity")
        wait_seconds = float(wait_seconds)
        if not 0 <= wait_seconds <= 60:
            raise ValueError("Exchange wait must be between zero and sixty seconds")
    except (KeyError, TypeError, ValueError) as error:
        return {"ok": False, "error_code": "invalid_request", "error": str(error)}
    request_dir = directory / "requests" / identity
    request_dir.mkdir(parents=True, exist_ok=True)
    counts = {"blob_cache_hits": 0, "blob_uploads": 0, "blob_upload_bytes": 0}

    def completed():
        response_path = request_dir / "response.json"
        if not response_path.exists():
            return None
        response = json.loads(response_path.read_text())
        artifact_path = request_dir / "artifacts.tar.gz"
        artifact = artifact_path.read_bytes() if artifact_path.is_file() else b""
        return {"ok": True, "response": response, "artifacts_b64": base64.b64encode(artifact).decode(),
                "artifacts_sha256": hashlib.sha256(artifact).hexdigest(), **counts}

    # A duplicate RPC can overlap the original after an SSH disconnect. Lock
    # only publication, never the wait, so both collect the same durable result.
    with (request_dir / "publish.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        envelope_path = request_dir / "envelope.json"
        existing = json.loads(envelope_path.read_text()) if envelope_path.exists() else None
        if existing and existing.get("fingerprint") != expected:
            return {"ok": False, "error_code": "fingerprint_conflict", "error": "Request ID reused for a different experiment"}
        result = completed() if existing else None
        if result is not None:
            return result
        failure = exchange_liveness(directory, payload)
        if failure:
            return failure
        if not existing:
            objects = directory / "blobs" / "sha256"
            objects.mkdir(parents=True, exist_ok=True)
            expected_blobs = {entry["sha256"] for entry in manifest.values()}
            if set(uploads) - expected_blobs:
                return {"ok": False, "error_code": "invalid_request", "error": "Unexpected blob upload"}
            decoded = {}
            try:
                for digest, encoded in uploads.items():
                    content = base64.b64decode(encoded, validate=True)
                    if hashlib.sha256(content).hexdigest() != digest:
                        raise ValueError("Uploaded blob SHA256 mismatch")
                    decoded[digest] = content
            except (ValueError, TypeError) as error:
                return {"ok": False, "error_code": "blob_corrupt", "error": str(error)}
            contents = {}
            missing = set()
            for name, entry in manifest.items():
                digest, size = entry["sha256"], entry["size"]
                path = objects / digest
                if path.is_symlink() or (path.exists() and not path.is_file()):
                    return {"ok": False, "error_code": "blob_corrupt", "error": "Blob object is not a regular immutable file"}
                if path.exists():
                    content = path.read_bytes()
                    counts["blob_cache_hits"] += 1
                elif digest in decoded:
                    content = decoded[digest]
                else:
                    missing.add(digest)
                    continue
                if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                    return {"ok": False, "error_code": "blob_corrupt", "error": "Immutable blob content or length changed"}
                contents[name] = content
            if missing:
                return {"ok": False, "error_code": "missing_blobs", "missing_blobs": sorted(missing), **counts}
            fingerprint = hashlib.sha256((envelope["operation"] + "\0" + json.dumps(envelope["request"], sort_keys=True)).encode())
            for name in sorted(contents):
                fingerprint.update(name.encode() + b"\0" + contents[name])
            if fingerprint.hexdigest() != expected:
                return {"ok": False, "error_code": "fingerprint_conflict", "error": "Manifest content does not match request fingerprint"}
            for name, content in contents.items():
                digest = manifest[name]["sha256"]
                path = objects / digest
                if not path.exists():
                    descriptor, temporary_name = tempfile.mkstemp(prefix=digest + ".upload-", dir=str(objects))
                    os.close(descriptor)
                    temporary = Path(temporary_name)
                    try:
                        temporary.write_bytes(content)
                        temporary.chmod(0o444)
                        try:
                            os.link(str(temporary), str(path))
                            counts["blob_uploads"] += 1
                            counts["blob_upload_bytes"] += len(content)
                        except FileExistsError:
                            if path.is_symlink() or path.read_bytes() != content:
                                return {"ok": False, "error_code": "blob_corrupt", "error": "Concurrent immutable blob mismatch"}
                    finally:
                        if temporary.exists():
                            temporary.unlink()
                target = request_dir / name
                if target.exists() or target.is_symlink():
                    if target.is_symlink() or target.read_bytes() != content:
                        return {"ok": False, "error_code": "blob_corrupt", "error": "Request file changed before publication"}
                else:
                    os.link(str(path), str(target))
            # The existing serve loop consumes the exact same envelope schema.
            # Publication last prevents it from seeing partial input transfers.
            atomic_json(envelope_path, envelope)
    deadline = time.monotonic() + wait_seconds
    next_status = time.monotonic() + 5
    while True:
        result = completed()
        if result is not None:
            return result
        if (directory / "quarantined.json").exists():
            return {"ok": False, "error_code": "quarantined", "error": "CUDA worker is quarantined; use a new worker session"}
        if time.monotonic() >= deadline:
            return {"ok": False, "error_code": "pending", "error": "Request remains tracked; repeat the same request ID", **counts}
        if time.monotonic() >= next_status:
            failure = exchange_liveness(directory, payload)
            if failure:
                return failure
            next_status = time.monotonic() + 5
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(str(temporary), str(path))


def run(argv, timeout=30):
    process = subprocess.Popen([str(v) for v in argv], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, errors="replace",
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def recover_job(config):
    """Scheduler queries use exact job name and owner, never positional guessing."""
    user = os.environ.get("USER", "")
    found = set()
    commands = [
        ["squeue", "-h", "-u", user, "-n", config["job_name"], "-o", "%A|%j"],
        ["sacct", "-n", "-P", "-u", user, "--starttime", "today", "--name", config["job_name"],
         "--format=JobIDRaw,JobName%128"],
    ]
    for command in commands:
        process = run(command)
        if process.returncode:
            continue
        for line in process.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[0].isdigit() and fields[1] == config["job_name"]:
                found.add(fields[0])
    if len(found) > 1:
        raise RuntimeError("Multiple jobs share this session identity; manual reconciliation required")
    return next(iter(found)) if found else None


def submit(directory):
    directory = Path(directory).resolve()
    config = json.loads((directory / "submission.json").read_text())
    with (directory / "submission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        receipt = directory / "job_id.json"
        if receipt.exists():
            return {"ok": True, **json.loads(receipt.read_text()), "reused": True}
        marker = directory / "submission_started.json"
        if marker.exists():
            job_id = recover_job(config)
            if job_id is None:
                return {"ok": False, "uncertain": True,
                        "error": "Previous sbatch acknowledgement was interrupted. No unique scheduler receipt yet; retry recovery, never blindly resubmit this session."}
        else:
            atomic_json(marker, {"time": time.time(), "session_id": config["session_id"]})
            command = ["sbatch", "--parsable", "--job-name=" + config["job_name"],
                       "--account=" + config["account"], "--partition=" + config["partition"],
                       "--constraint=" + config.get("constraint", "armgpu"),
                       "--gres=" + config["gres"], "--ntasks=1",
                       "--cpus-per-task=" + str(config["cpus"]), "--mem=" + config["memory"],
                       "--time=" + config["walltime"], "--chdir=" + str(directory),
                       "--output=" + str(directory / "slurm-%j.out"),
                       "--error=" + str(directory / "slurm-%j.err")]
            if config.get("exclude"):
                command.append("--exclude=" + config["exclude"])
            command.append(str(directory / "job.sh"))
            try:
                process = run(command, timeout=45)
            except subprocess.TimeoutExpired:
                return {"ok": False, "uncertain": True,
                        "error": "sbatch timed out; reconcile this session before resubmission"}
            if process.returncode:
                atomic_json(directory / "submission_error.json", {"stderr": process.stderr,
                            "returncode": process.returncode, "command": command})
                return {"ok": False, "error": process.stderr[-6000:]}
            job_id = process.stdout.strip().split(";")[0]
            if not job_id.isdigit():
                return {"ok": False, "uncertain": True, "error": "Unrecognized sbatch acknowledgement"}
        value = {"job_id": job_id, "session_id": config["session_id"], "job_name": config["job_name"]}
        atomic_json(receipt, value)
        return {"ok": True, **value}


def build_harness(directory):
    """Small upstream harness builds without cmake (unavailable on ROMEO login)."""
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc absent after module initialization inside allocation")
    cuda = Path(nvcc).resolve().parent.parent
    architecture = "aarch64-linux" if platform.machine() == "aarch64" else "x86_64-linux"
    target = cuda / "targets" / architecture
    include = target / "include" if (target / "include/cuda.h").exists() else cuda / "include"
    library = target / "lib" if (target / "lib").exists() else cuda / "lib64"
    executable = directory / "build/cuda_test_harness"
    executable.parent.mkdir(exist_ok=True)
    source = directory / "cuda_test_harness"
    digest = hashlib.sha256()
    digest.update(run(["g++", "--version"]).stdout.encode())
    digest.update(run([nvcc, "--version"]).stdout.encode())
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh", ".txt"):
            digest.update(str(path.relative_to(source)).encode() + b"\0" + path.read_bytes())
    build_id = digest.hexdigest()
    command = ["g++", "-std=c++17", "-O2", "-pthread", str(source / "harness.cpp"),
               "-I" + str(source), "-I" + str(source / "third_party"), "-I" + str(include),
               '-DCUDA_HARNESS_INCLUDE_DIR="' + str(include) + '"', "-L" + str(library),
               '-DCUDA_HARNESS_BUILD_ID="' + build_id + '"',
               "-Wl,-rpath," + str(library), "-lcuda", "-lnvrtc", "-o", str(executable)]
    process = run(command, timeout=180)
    atomic_json(directory / "build.json", {"command": command, "returncode": process.returncode,
                                           "stdout": process.stdout, "stderr": process.stderr})
    if process.returncode:
        raise RuntimeError("Harness compilation failed: " + process.stderr[-6000:])
    return executable


class PersistentHarness:
    """One native CUDA context, supervised independently of the JSONL worker."""
    def __init__(self, executable, directory):
        self.executable = executable
        self.directory = directory
        self.process = None
        self.stderr_file = None
        self.pending = b""

    @property
    def pid(self):
        return self.process.pid if self.process and self.process.poll() is None else None

    def execute(self, request_path, timeout):
        if self.process is None or self.process.poll() is not None:
            self.close()
            self.stderr_file = (self.directory / "native-worker-stderr.log").open("ab")
            self.process = subprocess.Popen([str(self.executable), "--worker"],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.stderr_file, start_new_session=True)
            self.pending = b""
        payload = (json.dumps({"request_path": str(request_path)}) + "\n").encode()
        self.process.stdin.write(payload)
        self.process.stdin.flush()
        deadline = time.monotonic() + timeout
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise subprocess.TimeoutExpired([str(self.executable), "--worker"], timeout)
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                self.close()
                raise subprocess.TimeoutExpired([str(self.executable), "--worker"], timeout)
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self.close()
                raise RuntimeError("Native CUDA worker ended without a complete result; allocation quarantined")
            self.pending += chunk
            if len(self.pending) > 16 * 1024 * 1024:
                self.close()
                raise RuntimeError("Native worker response exceeded 16 MiB; allocation quarantined")
        line, self.pending = self.pending.split(b"\n", 1)
        # Validation occurs before this response reaches the controller. A
        # malformed response must not leave an untracked CUDA process alive.
        try:
            result = json.loads(line)
        except ValueError:
            self.close()
            raise RuntimeError("Native worker returned malformed JSON; allocation quarantined")
        if result.get("worker_reusable") is False:
            self.close()
        return subprocess.CompletedProcess([str(self.executable), "--worker"], 0,
                                           line.decode(errors="replace"), "")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.close()
                    self.process.wait(timeout=2)
                except (subprocess.TimeoutExpired, BrokenPipeError, OSError):
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
            if self.process.stdout:
                self.process.stdout.close()
            self.process = None
        if self.stderr_file:
            self.stderr_file.close()
            self.stderr_file = None


def fingerprint(executable):
    process = run([str(executable), "--device-info", "--device", "0"])
    device = json.loads(process.stdout)
    if process.returncode or not device.get("ok"):
        raise RuntimeError("GPU device query failed: " + process.stdout + process.stderr)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    command = ["nvidia-smi", "--query-gpu=uuid,name,driver_version,pci.bus_id", "--format=csv,noheader"]
    raw_uuid = device["properties"].get("uuid", "")
    if len(raw_uuid) == 32 and all(c in "0123456789abcdefABCDEF" for c in raw_uuid):
        # CUDA's physical UUID avoids Slurm/cgroup ordinal remapping ambiguity.
        uuid = "GPU-" + "-".join((raw_uuid[:8], raw_uuid[8:12], raw_uuid[12:16], raw_uuid[16:20], raw_uuid[20:]))
        command += ["--id=" + uuid]
    elif visible and "," not in visible:
        command += ["--id=" + visible]
    inventory = run(command)
    value = {"environment": "romeo", "hostname": platform.node(), "machine": platform.machine(),
             "slurm_job_id": os.environ["SLURM_JOB_ID"], "cuda_visible_devices": visible,
             "device_properties": device["properties"], "nvidia_smi": inventory.stdout.strip(),
             "toolkit": run(["nvcc", "--version"]).stdout.strip(),
             "compiler": run(["g++", "--version"]).stdout.splitlines()[0],
             "profiling_available": shutil.which("ncu") is not None,
             "sanitizer_available": shutil.which("compute-sanitizer") is not None}
    value["gpu_uuids"] = [row.split(",")[0].strip() for row in inventory.stdout.splitlines() if row.strip()]
    # Job IDs identify receipts, but cannot define comparability across allocations.
    stable = {k: value[k] for k in ("machine", "device_properties", "nvidia_smi", "toolkit", "compiler")}
    value["fingerprint_id"] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    return value


def foreign_gpu_processes(hardware, ignored_pids=()):
    process = run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], timeout=10)
    if process.returncode:
        return []
    allowed = set(hardware.get("gpu_uuids", []))
    ignored = {str(pid) for pid in ignored_pids if pid is not None}
    return [line.strip() for line in process.stdout.splitlines()
            if len(line.split(",")) == 2 and line.split(",")[1].strip() in allowed
            and line.split(",")[0].strip() not in ignored]


def parse_metrics(text):
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "Metric Name" in line and "Metric Value" in line), None)
    if start is None:
        return []
    result = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if row.get("Metric Name") and row.get("Metric Value"):
            result.append({"name": row["Metric Name"], "value": row["Metric Value"],
                           "unit": row.get("Metric Unit", "")})
    return result


def execute_request(directory, executable, envelope, timeout, hardware, native_worker=None):
    operation = envelope["operation"]
    payload = envelope["request"]
    request = payload["kernel_request"] if operation in ("profile", "sanitize") else payload
    request_path = directory / "request.json"
    atomic_json(request_path, request)
    command = [str(executable), "--request", str(request_path)]
    conditions = None
    if operation == "profile":
        ncu = shutil.which("ncu")
        if ncu is None:
            return {"ok": False, "retryable": False, "error_code": "NCU_UNAVAILABLE", "error": "Nsight Compute is unavailable on ROMEO"}
        if any(not isinstance(metric, str) or not metric or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_." for c in metric)
               for metric in payload["metrics"]):
            raise ValueError("Expected explicit Nsight metric names")
        command = [ncu, "--config-file", "off", "--target-processes", "application-only",
                   "--kernel-name-base", "function", "--kernel-name", request["kernel"],
                   "--launch-count", "1", "--replay-mode", "kernel", "--cache-control", "all",
                   "--clock-control", "none", "--csv", "--page", "details", "--metrics", ",".join(payload["metrics"]),
                   "--log-file", str(directory / "metrics.csv"), "--export", str(directory / "profile"),
                   *command]
        conditions = {"question": payload["question"], "expected_metrics": payload["metrics"],
                      "cache_control": "all", "clock_control": "none", "replay_mode": "kernel",
                      "warmup": 0, "launches": 1, "diagnostic_only": True,
                      "note": "Profiler durations are never benchmark observations"}
    elif operation == "sanitize":
        tool = payload["tool"]
        if tool not in ("memcheck", "racecheck", "synccheck", "initcheck"):
            raise ValueError("Unsupported sanitizer tool")
        sanitizer = shutil.which("compute-sanitizer")
        if sanitizer is None:
            return {"ok": False, "stage": "sanitizer", "error": "compute-sanitizer unavailable", "tool": tool}
        command = [sanitizer, "--tool", tool, "--error-exitcode", "86", *command]
    elif operation != "run":
        raise ValueError("Unsupported worker operation")
    busy = foreign_gpu_processes(hardware, (native_worker.pid,) if native_worker else ())
    if busy:
        return {"ok": False, "stage": "interference", "error": "Foreign CUDA process on allocated GPU", "foreign_processes": busy}
    atomic_json(directory / "command.json", command)
    start = time.monotonic()
    try:
        process = native_worker.execute(request_path, timeout) if operation == "run" and native_worker else run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        # A hung CUDA context is not considered clean: terminate the allocation
        # rather than allowing another measurement to inherit an uncertain GPU.
        atomic_json(directory / "poisoned.json", {"reason": "request timeout"})
        return {"ok": False, "stage": "timeout", "error": "GPU request timed out; worker quarantined", "worker_poisoned": True}
    (directory / "stdout.log").write_text(process.stdout)
    (directory / "stderr.log").write_text(process.stderr)
    result_path = directory / "results/run-evaluation/result.json"
    try:
        result = json.loads(result_path.read_text()) if result_path.exists() else json.loads(process.stdout)
    except (ValueError, OSError):
        result = {"ok": False, "stage": "worker", "error": (process.stderr + process.stdout)[-6000:]}
    result["reservation_wall_seconds"] = time.monotonic() - start
    result["persistent_native_worker"] = operation == "run" and native_worker is not None
    if result.get("worker_reusable") is False:
        result["worker_poisoned"] = True
    if process.returncode:
        result.update(ok=False, error=(process.stderr + process.stdout)[-6000:], returncode=process.returncode)
    if operation == "profile":
        text = (directory / "metrics.csv").read_text() if (directory / "metrics.csv").exists() else ""
        diagnostics = text + process.stdout + process.stderr
        result["profiling_conditions"] = conditions
        result["metrics"] = parse_metrics(text)
        if "ERR_NVGPUCTRPERM" in diagnostics:
            result.update(ok=False, retryable=False, error_code="ERR_NVGPUCTRPERM",
                          error="ROMEO GPU performance counters denied; benchmark remains available. Profiling fallback is diagnostic only.")
        elif not result["metrics"]:
            result.update(ok=False, retryable=False, error="Nsight returned no diagnostic metrics")
    if operation == "sanitize":
        result["tool"] = payload["tool"]
        result["sanitizer_ok"] = bool(result.get("ok")) and process.returncode == 0
    busy = foreign_gpu_processes(hardware, (native_worker.pid,) if native_worker else ())
    if busy:
        result.update(ok=False, stage="interference", foreign_processes=busy,
                      error="Observed foreign CUDA process; measurement invalidated")
    # An illegal access or device-side assertion invalidates the persistent
    # context and quarantines the allocation before another request can run.
    diagnostics = str(result.get("error", "")).lower()
    if any(word in diagnostics for word in ("illegal address", "illegal memory access", "device-side assert", "launch timeout")):
        result["worker_poisoned"] = True
    return result


def save_artifacts(directory, result):
    outputs = directory / "outputs"
    outputs.mkdir(exist_ok=True)
    for output in result.get("outputs", []):
        path = Path(output["path"]).resolve()
        if not str(path).startswith(str(directory.resolve()) + os.sep):
            raise RuntimeError("Harness output escaped request directory")
        shutil.copy2(str(path), str(outputs / path.name))
    temporary = directory / "artifacts.tmp.tar.gz"
    with tarfile.open(str(temporary), "w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            is_input = path.parent == directory and path.name.startswith("input-")
            redundant_binary = path.suffix == ".bin" and path.parent != outputs
            if path.is_file() and not is_input and not redundant_binary and path.name not in ("artifacts.tmp.tar.gz", "artifacts.tar.gz", "response.json"):
                archive.add(str(path), arcname=str(path.relative_to(directory)), recursive=False)
    os.replace(str(temporary), str(directory / "artifacts.tar.gz"))


def serve(directory, timeout):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU worker may only run within a Slurm allocation")
    directory = Path(directory).resolve()
    os.umask(0o077)
    with (directory / "gpu-worker.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            executable = build_harness(directory)
            hardware = fingerprint(executable)
            atomic_json(directory / "ready.json", {"ok": True, "hardware": hardware,
                                                  "transport_capabilities": [EXCHANGE_CAPABILITY]})
        except Exception as error:
            atomic_json(directory / "ready.json", {"ok": False, "error": str(error)})
            raise
        requests = directory / "requests"
        requests.mkdir(exist_ok=True)
        native_worker = PersistentHarness(executable, directory)
        try:
            while not (directory / "STOP").exists():
                pending = sorted(requests.glob("*/envelope.json"), key=lambda p: (p.stat().st_mtime_ns, str(p)))
                for envelope_path in pending:
                    request_dir = envelope_path.parent
                    if (request_dir / "response.json").exists():
                        continue
                    envelope = json.loads(envelope_path.read_text())
                    started = request_dir / "started.json"
                    if started.exists():
                        start_record = json.loads(started.read_text())
                        result = {"ok": False, "stage": "interrupted", "error": "Previous worker stopped during this request; archive its cost and use a new attempt_id", "worker_poisoned": True}
                    else:
                        start_record = {"time": time.time(), "job_id": os.environ["SLURM_JOB_ID"],
                                        "worker_pid": os.getpid(), "session_id": directory.name,
                                        "monotonic_ns": time.monotonic_ns()}
                        atomic_json(started, start_record)
                        try:
                            result = execute_request(request_dir, executable, envelope, timeout, hardware, native_worker=native_worker)
                        except Exception as error:
                            native_worker.close()
                            result = {"ok": False, "stage": "worker", "error": str(error), "worker_poisoned": True}
                    result.update(execution_environment="romeo", hardware_fingerprint=hardware,
                                  request_fingerprint=envelope["fingerprint"], slurm_job_id=os.environ["SLURM_JOB_ID"])
                    save_artifacts(request_dir, result)
                    # Both bounds come from this serve process, after complete
                    # profiling/execution and artifact collection. GPFS mtimes
                    # and clocks from other hosts are never execution bounds.
                    result["worker_reservation"] = {"clock": "time.monotonic_ns", "worker_pid": os.getpid(),
                        "session_id": directory.name, "job_id": os.environ["SLURM_JOB_ID"],
                        "start_monotonic_ns": start_record.get("monotonic_ns"),
                        "end_monotonic_ns": time.monotonic_ns()}
                    atomic_json(request_dir / "response.json", result)
                    if result.get("worker_poisoned"):
                        atomic_json(directory / "quarantined.json", result)
                        return 3
                time.sleep(0.1)
        finally:
            native_worker.close()
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("submit", "serve", "exchange"))
    parser.add_argument("--directory", required=True)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--wait-seconds", type=float, default=30)
    args = parser.parse_args()
    if args.operation == "submit":
        print(json.dumps(submit(args.directory)))
        return 0
    if args.operation == "exchange":
        try:
            result = exchange(args.directory, json.load(sys.stdin), args.wait_seconds)
        except Exception as error:
            # An acknowledgement may have been lost after publication. Never
            # turn this into a new attempt or an automatic legacy resubmission.
            result = {"ok": False, "error_code": "exchange_error", "error": str(error)}
        print(json.dumps(result))
        return 0
    return serve(args.directory, args.request_timeout)


if __name__ == "__main__":
    sys.exit(main())
