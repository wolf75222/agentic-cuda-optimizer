"""A local controller for one isolated SSH/Slurm GPU worker.

Only Slurm's allocation runs CUDA. SSH on the login host transfers files,
submits, and inspects accounting. Candidate requests share a serialized worker.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import time
import uuid


class BackendError(RuntimeError):
    pass


class SubmissionUncertain(BackendError):
    """Never blindly resubmit a job whose scheduler acknowledgement was lost."""


TERMINAL_STATES = frozenset({"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY",
                             "PREEMPTED", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "REVOKED"})


def slurm_state(receipt: dict) -> str:
    return receipt.get("state", "UNKNOWN").split()[0].rstrip("+")


def transient_ssh_failure(process: subprocess.CompletedProcess) -> bool:
    message = process.stderr.decode(errors="replace").lower()
    return process.returncode == 255 and any(value in message for value in (
        "could not resolve hostname", "temporary failure in name resolution", "connection timed out",
        "connection reset", "connection closed", "network is unreachable", "no route to host",
        "connection refused", "kex_exchange_identification", "broken pipe"))


def safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", value):
        raise ValueError("Identifiers must contain only letters, numbers, underscore, dot or dash")
    return value


@dataclass(frozen=True)
class SlurmConfig:
    host: str = "romeo"
    remote_root: str = "/gpfs/scratch/rmdraux/agentic-cuda-optimizer-20261021"
    account: str = "r250127"
    partition: str = "instant"
    gres: str = "gpu:h100:1"
    constraint: str = "armgpu"
    cpus: int = 4
    memory: str = "8G"
    walltime: str = "00:30:00"
    modules: tuple[str, ...] = ("cuda/12.6",)
    python: str = "python3"
    poll_seconds: float = 2.0
    startup_timeout: int = 600
    request_timeout: int = 300
    exclude: str = ""

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*", self.host):
            raise ValueError("Invalid SSH alias/host")
        root = PurePosixPath(self.remote_root)
        if not root.is_absolute() or ".." in root.parts or len(root.parts) < 4:
            raise ValueError("remote_root must be an isolated absolute directory")
        if any(c in self.remote_root for c in "\x00\n\r"):
            raise ValueError("Invalid remote_root")
        for value in (self.account, self.partition, self.memory, self.python, self.constraint):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", value):
                raise ValueError("Invalid Slurm configuration value")
        if not re.fullmatch(r"gpu(?::[A-Za-z0-9_-]+)?:1", self.gres):
            raise ValueError("One GPU per worker is required")
        if not re.fullmatch(r"[0-9]+:[0-5][0-9]:[0-5][0-9]", self.walltime):
            raise ValueError("walltime must be HH:MM:SS")
        if type(self.cpus) is not int or self.cpus < 1:
            raise ValueError("cpus must be positive")
        for value in self.modules:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", value):
                raise ValueError("Invalid module name")
        if self.exclude and not re.fullmatch(r"[A-Za-z0-9_,.\[\]-]+", self.exclude):
            raise ValueError("Invalid node exclusion")
        if self.poll_seconds <= 0 or min(self.startup_timeout, self.request_timeout) <= 0:
            raise ValueError("Timeouts and polling interval must be positive")

    @classmethod
    def load(cls, path: str | Path) -> "SlurmConfig":
        values = json.loads(Path(path).read_text())
        values.pop("description", None)
        values["modules"] = tuple(values.get("modules", ("cuda/12.6",)))
        return cls(**values)


class SSHTransport:
    def __init__(self, host: str):
        self.host = host

    def run(self, argv: list[str], *, data: bytes | None = None, timeout: int = 60,
            check: bool = True) -> subprocess.CompletedProcess:
        # OpenSSH passes its remote command through a shell. Quote each argument
        # before composing that command, even though local execution has no shell.
        process = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "--", self.host,
             shlex.join([str(arg) for arg in argv])], input=data,
            capture_output=True, timeout=timeout, check=False)
        if check and process.returncode:
            raise BackendError(process.stderr.decode(errors="replace")[-6000:])
        return process

    def write(self, path: str, data: bytes):
        # Verify before commit: a disconnected SSH stdin can otherwise look
        # like EOF and atomically publish a *truncated* input. Both an original
        # delayed upload and its retry may commit only these exact same bytes.
        script = """import hashlib,os,pathlib,sys
p=pathlib.Path(sys.argv[1])
payload=sys.stdin.buffer.read()
if len(payload)!=int(sys.argv[2]) or hashlib.sha256(payload).hexdigest()!=sys.argv[3]:
    print('Atomic upload rejected: incomplete or changed payload',file=sys.stderr)
    sys.exit(65)
p.parent.mkdir(parents=True,exist_ok=True)
t=p.with_name(p.name+'.upload-'+str(os.getpid()))
try:
    t.write_bytes(payload)
    os.replace(str(t),str(p))
finally:
    if t.exists(): t.unlink()
"""
        command = ["python3", "-c", script, path, str(len(data)), hashlib.sha256(data).hexdigest()]
        for attempt in range(3):
            try:
                process = self.run(command, data=data, check=False)
            except subprocess.TimeoutExpired as error:
                if attempt == 2:
                    raise BackendError("Atomic SSH upload timed out after 3 identical attempts; "
                                       "resume this request ID to reconcile, never submit duplicate execution") from error
                time.sleep(1)
                continue
            if process.returncode == 0:
                return
            if not transient_ssh_failure(process) or attempt == 2:
                raise BackendError(process.stderr.decode(errors="replace")[-4000:])
            time.sleep(1)

    def read(self, path: str, *, optional: bool = False) -> bytes | None:
        command = ["python3", "-c",
                   "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                   "sys.stdout.buffer.write(p.read_bytes()) if p.is_file() else sys.exit(44)", path]
        # Reads are safe to repeat after an uncertain connection outcome. This
        # never retries a scheduler/execution command. Atomic same-body uploads
        # use their own verified-content retry protocol above.
        for attempt in range(3):
            try:
                process = self.run(command, check=False)
            except subprocess.TimeoutExpired as error:
                if attempt == 2:
                    raise BackendError("Remote artifact read timed out after 3 attempts") from error
                time.sleep(1)
                continue
            if optional and process.returncode == 44:
                return None
            if process.returncode == 0:
                return process.stdout
            message = process.stderr.decode(errors="replace")[-4000:]
            if not transient_ssh_failure(process) or attempt == 2:
                raise BackendError(message)
            time.sleep(1)


class SlurmBackend:
    def __init__(self, config: SlurmConfig, repo_dir: str | Path, results_dir: str | Path,
                 session_id: str | None = None, *, transport=None):
        self.config = config
        self.repo_dir = Path(repo_dir).resolve()
        self.results_dir = Path(results_dir).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = safe_id(session_id or uuid.uuid4().hex)
        self.remote_dir = str(PurePosixPath(config.remote_root) / "sessions" / self.session_id)
        self.transport = transport or SSHTransport(config.host)
        self.job_id: str | None = None
        self.hardware: dict = {}
        self._ready = False

    def _json(self, name: str, *, optional: bool = False):
        payload = self.transport.read(f"{self.remote_dir}/{name}", optional=optional)
        return None if payload is None else json.loads(payload)

    def stage(self):
        # Ship the immutable evaluator only. Candidate sources/inputs arrive as
        # independent requests. Never upload credentials, .git or the local venv.
        files = sorted((self.repo_dir / "cuda_test_harness").rglob("*"))
        selected = [p for p in files if p.is_file() and p.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh", ".txt")
                    and not any(part.startswith(("build", "cmake-build")) for part in p.relative_to(self.repo_dir).parts)]
        selected.append(Path(__file__).with_name("remote_worker.py"))
        digest = hashlib.sha256()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for path in selected:
                relative = str(path.relative_to(self.repo_dir))
                content = path.read_bytes()
                digest.update(relative.encode() + b"\0" + content)
                archive.add(path, arcname=relative, recursive=False)
        snapshot = digest.hexdigest()
        existing = self._json("snapshot.json", optional=True)
        if existing:
            if existing["sha256"] != snapshot:
                raise BackendError("Worker source differs from this session; use a new session_id")
            return
        self.transport.run(["mkdir", "-p", self.remote_dir])
        self.transport.run(["tar", "-xzf", "-", "-C", self.remote_dir], data=buffer.getvalue())
        self.transport.write(f"{self.remote_dir}/job.sh", self.job_script().encode())
        submission = {"session_id": self.session_id, "directory": self.remote_dir,
                      "job_name": "aco-" + self.session_id,
                      "account": self.config.account, "partition": self.config.partition,
                      "gres": self.config.gres, "constraint": self.config.constraint, "cpus": self.config.cpus,
                      "memory": self.config.memory, "walltime": self.config.walltime,
                      "exclude": self.config.exclude}
        self.transport.write(f"{self.remote_dir}/submission.json", json.dumps(submission).encode())
        # Commit the stage receipt last; an interrupted stage can be repeated.
        self.transport.write(f"{self.remote_dir}/snapshot.json", json.dumps({"sha256": snapshot}).encode())

    def job_script(self) -> str:
        cfg = self.config
        lines = ["#!/bin/bash", "set -euo pipefail", "umask 077",
                 "test -n \"${SLURM_JOB_ID:-}\" || { echo 'Slurm allocation required' >&2; exit 2; }"]
        lines += ["module load " + shlex.quote(module) for module in cfg.modules]
        lines += ["cd " + shlex.quote(self.remote_dir),
                  "exec " + shlex.join([cfg.python, f"{self.remote_dir}/optimizer_agent/remote_worker.py",
                                         "serve", "--directory", self.remote_dir,
                                         "--request-timeout", str(cfg.request_timeout)])]
        return "\n".join(lines) + "\n"

    def start(self, *, allow_terminal: bool = False) -> str:
        """Submit/recover this session; terminal sessions require a new identity.

        ``allow_terminal`` is for collecting historical status/artifacts only;
        it never makes an ended worker ready for new GPU work.
        """
        self.stage()
        process = self.transport.run([
            self.config.python, f"{self.remote_dir}/optimizer_agent/remote_worker.py",
            "submit", "--directory", self.remote_dir], timeout=60)
        result = json.loads(process.stdout)
        if result.get("uncertain"):
            raise SubmissionUncertain(result["error"])
        if not result.get("ok"):
            raise BackendError(result.get("error", "Submission failed"))
        self.job_id = str(result["job_id"])
        if not self.job_id.isdigit():
            raise BackendError("Invalid scheduler job ID")
        (self.results_dir / "slurm-session.json").write_text(json.dumps(
            {"session_id": self.session_id, "job_id": self.job_id, "remote_dir": self.remote_dir,
             "config": asdict(self.config)}, indent=2))
        if not allow_terminal:
            self.require_live_worker()
        return self.job_id

    def status(self) -> dict:
        if self.job_id is None:
            return {"state": "NOT_SUBMITTED"}
        # squeue reflects current allocation state while accounting can lag,
        # particularly during startup, requeue, and preemption.
        active = self.transport.run(["squeue", "-h", "-j", self.job_id,
                                     "-o", "%i|%T|%N"], check=False)
        active_rows = [line.split("|") for line in active.stdout.decode().splitlines() if line.strip()]
        active_row = next((row for row in active_rows if len(row) >= 3 and row[0] == self.job_id), None)
        if active.returncode == 0 and active_row:
            return {"job_id": self.job_id, "state": active_row[1], "nodes": active_row[2], "source": "squeue"}
        result = self.transport.run(["sacct", "-n", "-P", "-j", self.job_id,
                                     "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList"])
        rows = [line.split("|") for line in result.stdout.decode().splitlines() if line.strip()]
        row = next((r for r in rows if r[0] == self.job_id), None)
        return ({"job_id": self.job_id, "state": row[1], "exit_code": row[2],
                 "elapsed": row[3], "nodes": row[4]} if row else {"state": "ACCOUNTING_PENDING"})

    def require_live_worker(self, receipt: dict | None = None) -> dict:
        if self.job_id is None:
            raise BackendError("No Slurm job is attached to this worker session")
        receipt = self.status() if receipt is None else receipt
        if slurm_state(receipt) in TERMINAL_STATES:
            self._ready = False
            raise BackendError(
                f"Worker session {self.session_id} is terminal (job {self.job_id}, {receipt['state']}). "
                "Resume with a new worker session (--worker-session NEW); historical artifacts remain available.")
        quarantined = self._json("quarantined.json", optional=True)
        if quarantined is not None:
            self._ready = False
            raise BackendError(
                f"Worker session {self.session_id} is quarantined after a CUDA/context error. "
                "Resume with a new worker session (--worker-session NEW).")
        return receipt

    def wait_until_ready(self) -> dict:
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            receipt = self.require_live_worker()
            ready = self._json("ready.json", optional=True)
            if ready:
                hardware = ready.get("hardware", {})
                recorded_job = hardware.get("slurm_job_id", ready.get("slurm_job_id"))
                if recorded_job is not None and str(recorded_job) != self.job_id:
                    raise BackendError("Worker readiness belongs to a different Slurm job; use a new worker session")
                # A previous allocation's ready file cannot make a queued,
                # suspended, or requeued job immediately usable.
                if slurm_state(receipt) != "RUNNING":
                    time.sleep(self.config.poll_seconds)
                    continue
                if not ready.get("ok"):
                    raise BackendError(ready.get("error", "Worker startup failed"))
                self.hardware = ready["hardware"]
                self._ready = True
                (self.results_dir / "hardware.json").write_text(json.dumps(self.hardware, indent=2))
                return ready
            time.sleep(self.config.poll_seconds)
        raise BackendError("Timed out waiting for Slurm worker; allocation remains tracked for resume/cancel")

    def runner(self) -> "RemoteHarnessRunner":
        if not self._ready:
            self.wait_until_ready()
        return RemoteHarnessRunner(self)

    def close(self, *, cancel: bool = False):
        if not self.job_id:
            return
        # Never use scancel -u or touch another campaign's jobs.
        record = self._json("job_id.json")
        if str(record["job_id"]) != self.job_id or record["session_id"] != self.session_id:
            raise BackendError("Refusing cancellation: job ownership receipt mismatch")
        if cancel:
            self.transport.run(["scancel", self.job_id])
        else:
            self.transport.write(f"{self.remote_dir}/STOP", b"stop\n")

    def collect_diagnostics(self) -> dict:
        """Fetch bounded session receipts even if compilation or startup failed."""
        directory = self.results_dir / "slurm-diagnostics"
        directory.mkdir(exist_ok=True)
        names = ["ready.json", "build.json", "job_id.json", "snapshot.json", "submission_error.json", "quarantined.json"]
        if self.job_id:
            names += [f"slurm-{self.job_id}.out", f"slurm-{self.job_id}.err"]
        for name in names:
            payload = self.transport.read(f"{self.remote_dir}/{name}", optional=True)
            if payload is not None:
                (directory / name).write_bytes(payload)
        receipt = self.status()
        (directory / "accounting.json").write_text(json.dumps(receipt, indent=2))
        return receipt

    def wait_until_finished(self, timeout: float = 60) -> dict:
        deadline = time.monotonic() + timeout
        receipt = self.status()
        while slurm_state(receipt) not in TERMINAL_STATES:
            if time.monotonic() >= deadline:
                return {**receipt, "terminal": False}
            time.sleep(self.config.poll_seconds)
            receipt = self.status()
        return {**self.collect_diagnostics(), "terminal": True}


class RemoteHarnessRunner:
    def __init__(self, backend: SlurmBackend):
        self.backend = backend
        self.results_dir = backend.results_dir
        self.executable = Path(backend.remote_dir) / "build/cuda_test_harness"
        self.hardware = backend.hardware
        self.environment = "romeo"

    def get_device_properties(self, device: int = 0) -> dict:
        if device != 0:
            raise ValueError("The Slurm allocation exposes one GPU, indexed 0")
        return self.hardware["device_properties"]

    def _request(self, operation: str, request: dict, files: dict[str, bytes], identity: str) -> dict:
        identity = safe_id(identity)
        remote = f"{self.backend.remote_dir}/requests/{identity}"
        fingerprint = hashlib.sha256((operation + "\0" + json.dumps(request, sort_keys=True)).encode())
        for name in sorted(files):
            fingerprint.update(name.encode() + b"\0" + files[name])
        envelope = {"operation": operation, "request": request, "identity": identity,
                    "fingerprint": fingerprint.hexdigest()}
        existing = self.backend._json(f"requests/{identity}/envelope.json", optional=True)
        if existing and existing["fingerprint"] != envelope["fingerprint"]:
            raise BackendError("Request ID reused for a different experiment; use a fresh attempt ID")
        if not existing:
            self.backend.require_live_worker()
            for name, content in files.items():
                self.backend.transport.write(f"{remote}/{name}", content)
            self.backend.transport.write(f"{remote}/envelope.json", json.dumps(envelope).encode())
        deadline = time.monotonic() + self.backend.config.request_timeout + 30
        next_status_check = time.monotonic()
        while time.monotonic() < deadline:
            result = self.backend._json(f"requests/{identity}/response.json", optional=True)
            if result is not None:
                directory = self.results_dir / ("run-" + identity)
                directory.mkdir(parents=True, exist_ok=True)
                artifact_path = f"{remote}/artifacts.tar.gz"
                payload = self.backend.transport.read(artifact_path, optional=True)
                if payload:
                    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
                        for member in archive.getmembers():
                            target = (directory / member.name).resolve()
                            if not target.is_relative_to(directory.resolve()) or not member.isfile():
                                raise BackendError("Unsafe artifact archive member")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(archive.extractfile(member).read())
                for output in result.get("outputs", []):
                    output["remote_path"] = output["path"]
                    output["path"] = str(directory / "outputs" / Path(output["path"]).name)
                result["execution_environment"] = "romeo"
                result["hardware_fingerprint"] = self.hardware
                result["slurm_job_id"] = self.backend.job_id
                (directory / "result.json").write_text(json.dumps(result, indent=2))
                return result
            if time.monotonic() >= next_status_check:
                self.backend.require_live_worker()
                next_status_check = time.monotonic() + 10
            time.sleep(self.backend.config.poll_seconds)
        raise BackendError("Remote request timed out; resume the same request ID to recover, do not remeasure silently")

    def run(self, run_id, request) -> dict:
        payload = asdict(request)
        if payload.get("dependencies"):
            raise ValueError("Remote dependencies must first be explicitly staged; remote execution rejects local dependencies")
        files = {"kernel.cu": Path(payload["source"]).read_bytes()}
        remote = f"{self.backend.remote_dir}/requests/{safe_id(run_id)}"
        payload["source"] = f"{remote}/kernel.cu"
        payload["id"] = "evaluation"
        payload["results_dir"] = f"{remote}/results"
        payload["cache_dir"] = f"{self.backend.remote_dir}/compile-cache"
        for index, arg in enumerate(payload["arguments"]):
            if arg.get("kind") == "buffer" and arg.get("input_file"):
                name = f"input-{index}.bin"
                files[name] = Path(arg["input_file"]).read_bytes()
                arg["input_file"] = f"{remote}/{name}"
        return self._request("run", payload, files, run_id)

    def profile(self, run_id, request, *, question: str, metrics: list[str]) -> dict:
        if not question.strip() or not metrics:
            raise ValueError("Profiling requires a diagnostic question and expected metrics")
        payload = asdict(request)
        if payload.get("dependencies"):
            raise ValueError("Remote profiling rejects unstaged dependencies")
        files = {"kernel.cu": Path(payload["source"]).read_bytes()}
        remote = f"{self.backend.remote_dir}/requests/{safe_id(run_id)}"
        payload.update(source=f"{remote}/kernel.cu", id="evaluation", results_dir=f"{remote}/results",
                       mode="run", warmup=0, iterations=1, series=1,
                       cache_dir=f"{self.backend.remote_dir}/compile-cache")
        for index, arg in enumerate(payload["arguments"]):
            if arg.get("kind") == "buffer" and arg.get("input_file"):
                name = f"input-{index}.bin"
                files[name] = Path(arg["input_file"]).read_bytes()
                arg["input_file"] = f"{remote}/{name}"
        return self._request("profile", {"kernel_request": payload, "question": question,
                                         "metrics": metrics}, files, run_id)

    def run_sanitizer(self, run_id, request, tool="memcheck") -> dict:
        if tool not in ("memcheck", "racecheck", "synccheck", "initcheck"):
            raise ValueError("Unsupported sanitizer tool")
        payload = asdict(request)
        if payload.get("dependencies"):
            raise ValueError("Remote sanitizers reject unstaged dependencies")
        files = {"kernel.cu": Path(payload["source"]).read_bytes()}
        remote = f"{self.backend.remote_dir}/requests/{safe_id(run_id)}"
        payload.update(source=f"{remote}/kernel.cu", id="evaluation", results_dir=f"{remote}/results",
                       mode="run", warmup=0, iterations=1, series=1,
                       cache_dir=f"{self.backend.remote_dir}/compile-cache")
        for index, arg in enumerate(payload["arguments"]):
            if arg.get("kind") == "buffer" and arg.get("input_file"):
                name = f"input-{index}.bin"
                files[name] = Path(arg["input_file"]).read_bytes()
                arg["input_file"] = f"{remote}/{name}"
        return self._request("sanitize", {"kernel_request": payload, "tool": tool}, files, run_id)
