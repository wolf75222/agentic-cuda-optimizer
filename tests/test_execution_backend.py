import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from execution_backend import (
    BackendError, RemoteHarnessRunner, SlurmBackend, SlurmConfig, SSHTransport, safe_id,
)
from hybrid_profiler import HybridProfiler
import remote_worker


@pytest.mark.parametrize("values", [
    {"host": "-oProxyCommand=evil"}, {"host": "romeo;id"},
    {"remote_root": "/tmp/../escape"}, {"remote_root": "/"},
    {"modules": ("cuda/12.6;id",)}, {"gres": "gpu:h100:4"},
    {"account": "r250127\n--exclusive"}, {"walltime": "1;id"},
    {"request_timeout": 0}, {"exclude": "romeo;id"},
    {"constraint": "armgpu;id"},
])
def test_invalid_config_rejected(values):
    with pytest.raises(ValueError):
        SlurmConfig(**values)


def test_ssh_shell_boundary_is_quoted(monkeypatch):
    captured = {}
    def invoke(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, b"", b"")
    monkeypatch.setattr(subprocess, "run", invoke)
    SSHTransport("romeo").run(["python3", "-c", "print('safe')", "/tmp/a;$(id) space"])
    assert captured["command"][-1].endswith("'/tmp/a;$(id) space'")
    assert "--" in captured["command"]


@pytest.mark.parametrize("value", ["../../a", "x y", ";id", "", "-option"])
def test_artifact_identifiers_are_safe(value):
    with pytest.raises(ValueError):
        safe_id(value)


def test_worker_requires_slurm(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm allocation"):
        remote_worker.serve(tmp_path, 1)


def submission_config(tmp_path):
    config = {"session_id": "test", "job_name": "aco-test", "account": "r250127",
              "partition": "instant", "gres": "gpu:h100:1", "cpus": 4,
              "memory": "8G", "walltime": "00:05:00", "exclude": "romeo-c003"}
    (tmp_path / "submission.json").write_text(json.dumps(config))
    return config


def test_submission_receipt_reused_without_sbatch(monkeypatch, tmp_path):
    submission_config(tmp_path)
    commands = []
    def invoke(argv, timeout=30):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "987654;cluster\n", "")
    monkeypatch.setattr(remote_worker, "run", invoke)
    first = remote_worker.submit(tmp_path)
    second = remote_worker.submit(tmp_path)
    assert first["job_id"] == second["job_id"] == "987654"
    assert second["reused"]
    assert len(commands) == 1
    assert "--gres=gpu:h100:1" in commands[0]
    assert "--constraint=armgpu" in commands[0]
    assert "--exclude=romeo-c003" in commands[0]


def test_uncertain_submission_recovers_exact_name(monkeypatch, tmp_path):
    submission_config(tmp_path)
    (tmp_path / "submission_started.json").write_text("{}")
    commands = []
    def invoke(argv, timeout=30):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "123|aco-test\n999|aco-other\n", "")
    monkeypatch.setattr(remote_worker, "run", invoke)
    assert remote_worker.submit(tmp_path)["job_id"] == "123"
    assert all(command[0] != "sbatch" for command in commands)


def test_uncertain_submission_never_blindly_resubmits(monkeypatch, tmp_path):
    submission_config(tmp_path)
    (tmp_path / "submission_started.json").write_text("{}")
    commands = []
    def invoke(argv, timeout=30):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(remote_worker, "run", invoke)
    result = remote_worker.submit(tmp_path)
    assert result["uncertain"] and not result["ok"]
    assert all(command[0] != "sbatch" for command in commands)


def test_cancellation_checks_owned_job_receipt(tmp_path):
    class Transport:
        def read(self, path, optional=False):
            return json.dumps({"job_id": "888", "session_id": "other"}).encode()
        def run(self, *args, **kwargs):
            raise AssertionError("must never cancel unrelated jobs")
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "results", "mine", transport=Transport())
    backend.job_id = "888"
    with pytest.raises(BackendError, match="ownership"):
        backend.close(cancel=True)


def test_job_script_gpu_requires_scheduler_and_quotes_paths(tmp_path):
    config = SlurmConfig(remote_root="/gpfs/scratch/rmdraux/path with spaces")
    backend = SlurmBackend(config, tmp_path, tmp_path / "out", "safe")
    script = backend.job_script()
    assert "SLURM_JOB_ID" in script
    assert "cd '/gpfs/scratch/rmdraux/path with spaces/sessions/safe'" in script
    assert "serve --directory" in script


def test_hybrid_permission_failure_is_sticky_and_local_never_ranked():
    class Remote:
        calls = 0
        def profile(self, *args, **kwargs):
            self.calls += 1
            return {"ok": False, "error_code": "ERR_NVGPUCTRPERM", "retryable": False,
                    "error": "GPU counters denied"}
    remote = Remote()
    profiler = HybridProfiler(remote_runner=remote, max_profiles=4)
    request = SimpleNamespace()
    for identity in ("first", "second"):
        result = profiler.profile(identity, request, question="Memory bound?", metrics=["dram__bytes.sum"])
        assert result["execution_environment"] == "local"
        assert result["ranking_eligible"] is False
        assert result["diagnostic_only"] is True
        assert "ROMEO" in result["hardware_transfer_note"]
    assert remote.calls == 1


def test_profile_question_and_budget_required():
    profiler = HybridProfiler(max_profiles=0)
    with pytest.raises(ValueError):
        profiler.profile("x", None, question="", metrics=["foo"])
    result = profiler.profile("x", None, question="Why?", metrics=["foo"])
    assert "budget" in result["error"]


def test_unsafe_artifact_archive_rejected(tmp_path, monkeypatch):
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:gz") as archive:
        info = tarfile.TarInfo("../outside")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    class Transport:
        def read(self, path, optional=False):
            if path.endswith("envelope.json"):
                return None
            if path.endswith("response.json"):
                return b'{"ok":true,"outputs":[]}'
            if path.endswith("artifacts.tar.gz"):
                return memory.getvalue()
            return None
        def write(self, *args):
            pass
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "safe", transport=Transport())
    backend.job_id = "123"
    backend._ready = True
    monkeypatch.setattr(backend, "status", lambda: {"job_id": "123", "state": "RUNNING"})
    runner = RemoteHarnessRunner(backend)
    with pytest.raises(BackendError, match="Unsafe artifact"):
        runner._request("run", {}, {}, "request")


def test_profile_metrics_csv_parser():
    assert remote_worker.parse_metrics('noise\n"Metric Name","Metric Unit","Metric Value"\n"dram__bytes.sum","byte","1024"') == [
        {"name": "dram__bytes.sum", "unit": "byte", "value": "1024"}]


def test_worker_serializes_benchmark_profile_requests(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(remote_worker, "build_harness", lambda directory: directory / "harness")
    monkeypatch.setattr(remote_worker, "fingerprint", lambda executable: {"gpu_uuids": ["GPU-test"]})
    order = []
    def execute(directory, executable, envelope, timeout, hardware, native_worker=None):
        # The worker calls one operation synchronously before selecting another.
        assert len(order) < 2
        assert not (tmp_path / "requests/second/response.json").exists()
        order.append(envelope["operation"])
        return {"ok": True, "outputs": [], "worker_poisoned": len(order) == 2}
    monkeypatch.setattr(remote_worker, "execute_request", execute)
    for name, operation in (("first", "run"), ("second", "profile")):
        directory = tmp_path / "requests" / name
        directory.mkdir(parents=True)
        (directory / "envelope.json").write_text(json.dumps(
            {"operation": operation, "identity": name, "request": {}, "fingerprint": name}))
    assert remote_worker.serve(tmp_path, 10) == 3
    assert order == ["run", "profile"]
    assert (tmp_path / "quarantined.json").exists()
    assert json.loads((tmp_path / "requests/first/response.json").read_text())["ok"]


def test_worker_interrupted_request_is_not_silently_reexecuted(monkeypatch, tmp_path):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(remote_worker, "build_harness", lambda directory: directory / "harness")
    monkeypatch.setattr(remote_worker, "fingerprint", lambda executable: {})
    def unexpected(*args):
        raise AssertionError("uncertain execution must not be repeated under the same attempt id")
    monkeypatch.setattr(remote_worker, "execute_request", unexpected)
    directory = tmp_path / "requests/interrupted"
    directory.mkdir(parents=True)
    (directory / "envelope.json").write_text(json.dumps({"fingerprint": "same"}))
    (directory / "started.json").write_text("{}")
    assert remote_worker.serve(tmp_path, 10) == 3
    result = json.loads((directory / "response.json").read_text())
    assert result["stage"] == "interrupted"
    assert result["worker_poisoned"]


def test_native_jsonl_worker_reuses_process_and_stops_poisoned_context(tmp_path):
    executable = tmp_path / "fake-harness"
    executable.write_text("#!" + sys.executable + "\n"
                          "import json,os,sys\n"
                          "for line in sys.stdin:\n"
                          "    value=json.loads(line)\n"
                          "    reusable=not value['request_path'].endswith('poison')\n"
                          "    print(json.dumps({'ok':True,'pid':os.getpid(),'worker_reusable':reusable}),flush=True)\n"
                          "    if not reusable: break\n")
    executable.chmod(0o700)
    worker = remote_worker.PersistentHarness(executable, tmp_path)
    try:
        first = json.loads(worker.execute(tmp_path / "first", 5).stdout)
        second = json.loads(worker.execute(tmp_path / "second", 5).stdout)
        assert first["pid"] == second["pid"] == worker.pid
        poison = json.loads(worker.execute(tmp_path / "poison", 5).stdout)
        assert poison["worker_reusable"] is False
        assert worker.pid is None
    finally:
        worker.close()


def test_foreign_process_probe_excludes_persistent_worker(monkeypatch):
    def invoke(command, timeout=30):
        return subprocess.CompletedProcess(command, 0, "123, GPU-owned\n456, GPU-owned\n789, GPU-other\n", "")
    monkeypatch.setattr(remote_worker, "run", invoke)
    assert remote_worker.foreign_gpu_processes({"gpu_uuids": ["GPU-owned"]}, [123]) == ["456, GPU-owned"]


def test_direct_build_provides_stable_harness_identity(monkeypatch, tmp_path):
    toolkit = tmp_path / "cuda"
    (toolkit / "bin").mkdir(parents=True)
    nvcc = toolkit / "bin/nvcc"
    nvcc.touch()
    (tmp_path / "cuda_test_harness").mkdir()
    (tmp_path / "cuda_test_harness/harness.cpp").write_text("int main(){}")
    monkeypatch.setattr(remote_worker.shutil, "which", lambda name: str(nvcc))
    commands = []
    def invoke(command, timeout=30):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "compiler and toolkit version", "")
    monkeypatch.setattr(remote_worker, "run", invoke)
    remote_worker.build_harness(tmp_path)
    command = next(command for command in commands if "-O2" in command)
    define = next(arg for arg in command if str(arg).startswith("-DCUDA_HARNESS_BUILD_ID="))
    assert len(define.split('"')[1]) == 64
    assert "-DCUDA_HARNESS_INCLUDE_DIR=" in " ".join(command)


@pytest.mark.parametrize("state", ["COMPLETED", "FAILED", "CANCELLED by 123", "PREEMPTED", "NODE_FAIL", "OUT_OF_MEMORY+"])
def test_stale_readiness_never_revives_terminal_job(tmp_path, monkeypatch, state):
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "old")
    backend.job_id = "123"
    monkeypatch.setattr(backend, "status", lambda: {"job_id": "123", "state": state})
    monkeypatch.setattr(backend, "_json", lambda *args, **kwargs: {"ok": True, "hardware": {"slurm_job_id": "123"}})
    with pytest.raises(BackendError, match="new worker session"):
        backend.wait_until_ready()
    assert not backend._ready


def test_readiness_requires_current_job_identity(tmp_path, monkeypatch):
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "old")
    backend.job_id = "123"
    monkeypatch.setattr(backend, "status", lambda: {"state": "RUNNING"})
    monkeypatch.setattr(backend, "_json", lambda name, **kwargs: None if name == "quarantined.json" else {
        "ok": True, "hardware": {"slurm_job_id": "999"}})
    with pytest.raises(BackendError, match="different Slurm job"):
        backend.wait_until_ready()


def test_queued_job_does_not_accept_previous_ready_file(tmp_path, monkeypatch):
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "resumed")
    backend.job_id = "123"
    states = iter(("PENDING", "RUNNING"))
    monkeypatch.setattr(backend, "status", lambda: {"state": next(states)})
    sleeps = []
    monkeypatch.setattr("execution_backend.time.sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(backend, "_json", lambda name, **kwargs: None if name == "quarantined.json" else {
        "ok": True, "hardware": {"slurm_job_id": "123"}})
    assert backend.wait_until_ready()["ok"]
    assert len(sleeps) == 1
    assert backend._ready


def test_quarantined_running_job_fails_before_new_request(tmp_path, monkeypatch):
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "broken")
    backend.job_id = "123"
    backend._ready = True
    monkeypatch.setattr(backend, "status", lambda: {"state": "RUNNING"})
    monkeypatch.setattr(backend, "_json", lambda name, **kwargs: {} if name == "quarantined.json" else None)
    with pytest.raises(BackendError, match="quarantined"):
        RemoteHarnessRunner(backend)._request("run", {}, {}, "new-request")
    assert not backend._ready


def test_start_recovered_terminal_job_requires_new_session_but_allows_history(tmp_path, monkeypatch):
    class Transport:
        def run(self, command, **kwargs):
            return subprocess.CompletedProcess(command, 0, b'{"ok":true,"job_id":"123"}', b"")
        def read(self, path, optional=False):
            assert path.endswith("/job_id.json")
            return b'{"job_id":"123","session_id":"old"}'
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "old", transport=Transport())
    monkeypatch.setattr(backend, "stage", lambda: None)
    monkeypatch.setattr(backend, "status", lambda: {"state": "COMPLETED"})
    with pytest.raises(BackendError, match="--worker-session NEW"):
        backend.start()
    assert backend.start(allow_terminal=True) == "123"
    assert not backend._ready


def test_status_prefers_live_scheduler_to_lagging_accounting(tmp_path):
    class Transport:
        def run(self, command, **kwargs):
            assert command[0] == "squeue", "active jobs should use authoritative current scheduler state"
            return subprocess.CompletedProcess(command, 0, b"123|RUNNING|romeo-a057\n", b"")
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "active", transport=Transport())
    backend.job_id = "123"
    assert backend.status()["state"] == "RUNNING"


def test_pending_response_detects_terminal_worker_without_waiting_request_timeout(tmp_path, monkeypatch):
    import hashlib
    fingerprint = hashlib.sha256(b"run\0{}").hexdigest()
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "ended")
    backend.job_id = "123"
    monkeypatch.setattr(backend, "status", lambda: {"state": "PREEMPTED"})
    monkeypatch.setattr(backend, "_json", lambda name, **kwargs: {"fingerprint": fingerprint} if name.endswith("envelope.json") else None)
    with pytest.raises(BackendError, match="terminal"):
        RemoteHarnessRunner(backend)._request("run", {}, {}, "pending")


def test_completed_response_remains_collectible_after_worker_exits(tmp_path, monkeypatch):
    import hashlib
    fingerprint = hashlib.sha256(b"run\0{}").hexdigest()
    class Transport:
        def read(self, path, optional=False):
            assert path.endswith("artifacts.tar.gz")
            return None
    backend = SlurmBackend(SlurmConfig(), tmp_path, tmp_path / "out", "ended", transport=Transport())
    backend.job_id = "123"
    def read_json(name, **kwargs):
        return {"fingerprint": fingerprint} if name.endswith("envelope.json") else {"ok": True, "outputs": []}
    monkeypatch.setattr(backend, "_json", read_json)
    monkeypatch.setattr(backend, "require_live_worker", lambda: pytest.fail("completed result collection must not require a living worker"))
    assert RemoteHarnessRunner(backend)._request("run", {}, {}, "finished")["ok"]


def test_read_only_artifact_fetch_retries_transient_dns_errors(monkeypatch):
    transport = SSHTransport("romeo")
    calls = []
    monkeypatch.setattr("execution_backend.time.sleep", lambda seconds: None)
    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0 if len(calls) == 3 else 255,
                                           b"ready" if len(calls) == 3 else b"",
                                           b"ssh: Could not resolve hostname romeo: nodename nor servname provided")
    monkeypatch.setattr(transport, "run", invoke)
    assert transport.read("/isolated/ready.json", optional=True) == b"ready"
    assert len(calls) == 3


def test_read_only_fetch_does_not_retry_permissions_or_missing_optional_file(monkeypatch):
    transport = SSHTransport("romeo")
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, b"", b"Permission denied (publickey)")
    monkeypatch.setattr(transport, "run", invoke)
    with pytest.raises(BackendError, match="Permission denied"):
        transport.read("/isolated/ready.json")
    assert len(calls) == 1
    monkeypatch.setattr(transport, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 44, b"", b""))
    assert transport.read("/isolated/missing.json", optional=True) is None


def test_read_only_transient_retries_are_bounded(monkeypatch):
    transport = SSHTransport("romeo")
    calls = []
    monkeypatch.setattr("execution_backend.time.sleep", lambda seconds: None)
    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, b"", b"Connection reset by peer")
    monkeypatch.setattr(transport, "run", invoke)
    with pytest.raises(BackendError, match="Connection reset"):
        transport.read("/isolated/ready.json")
    assert len(calls) == 3


def test_scheduler_command_is_never_blindly_replayed_after_network_failure(monkeypatch):
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, b"", b"Connection reset by peer")
    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(BackendError, match="Connection reset"):
        SSHTransport("romeo").run(["sbatch", "--parsable", "/isolated/job.sh"])
    assert len(calls) == 1


def test_atomic_upload_retries_identical_content_after_lost_acknowledgement(tmp_path, monkeypatch):
    transport = SSHTransport("romeo")
    target = tmp_path / "input-0.bin"
    payload = b"GPU input bytes\0" * 1024
    calls = []
    monkeypatch.setattr("execution_backend.time.sleep", lambda seconds: None)
    def invoke(command, *, data, **kwargs):
        calls.append((command, data))
        # Execute the exact remote atomic-upload program locally. First commit
        # succeeds physically, then its SSH acknowledgement is lost.
        result = subprocess.run([sys.executable, "-c", command[2], *command[3:]],
                                input=data, capture_output=True, timeout=5)
        assert result.returncode == 0
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 60)
        return result
    monkeypatch.setattr(transport, "run", invoke)
    transport.write(str(target), payload)
    assert target.read_bytes() == payload
    assert len(calls) == 2 and calls[0] == calls[1]
    assert not list(tmp_path.glob("*.upload-*"))


def test_atomic_upload_rejects_truncated_stdin_without_overwriting_valid_file(tmp_path, monkeypatch):
    transport = SSHTransport("romeo")
    target = tmp_path / "input.bin"
    target.write_bytes(b"previous complete input")
    calls = []
    def invoke(command, *, data, **kwargs):
        calls.append(command)
        return subprocess.run([sys.executable, "-c", command[2], *command[3:]],
                              input=data[:-1], capture_output=True, timeout=5)
    monkeypatch.setattr(transport, "run", invoke)
    with pytest.raises(BackendError, match="incomplete or changed payload"):
        transport.write(str(target), b"new expected input")
    assert target.read_bytes() == b"previous complete input"
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.upload-*"))


def test_atomic_upload_checks_hash_even_when_changed_payload_length_matches(tmp_path, monkeypatch):
    transport = SSHTransport("romeo")
    target = tmp_path / "input.bin"
    def invoke(command, *, data, **kwargs):
        changed = bytes([data[0] ^ 1]) + data[1:]
        return subprocess.run([sys.executable, "-c", command[2], *command[3:]],
                              input=changed, capture_output=True, timeout=5)
    monkeypatch.setattr(transport, "run", invoke)
    with pytest.raises(BackendError, match="incomplete or changed payload"):
        transport.write(str(target), b"expected bytes")
    assert not target.exists()


def test_atomic_upload_transient_retries_are_identical_and_bounded(monkeypatch):
    transport = SSHTransport("romeo")
    calls = []
    monkeypatch.setattr("execution_backend.time.sleep", lambda seconds: None)
    def invoke(command, *, data, **kwargs):
        calls.append((command, data))
        return subprocess.CompletedProcess(command, 255, b"", b"Connection reset by peer")
    monkeypatch.setattr(transport, "run", invoke)
    with pytest.raises(BackendError, match="Connection reset"):
        transport.write("/isolated/input.bin", b"input")
    assert len(calls) == 3
    assert calls[0] == calls[1] == calls[2]


def test_atomic_upload_timeouts_are_bounded_and_report_reconciliation(monkeypatch):
    transport = SSHTransport("romeo")
    calls = []
    monkeypatch.setattr("execution_backend.time.sleep", lambda seconds: None)
    def invoke(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, 60)
    monkeypatch.setattr(transport, "run", invoke)
    with pytest.raises(BackendError, match="3 identical attempts.*resume this request ID"):
        transport.write("/isolated/input.bin", b"input")
    assert len(calls) == 3


def test_atomic_upload_does_not_retry_ssh_authentication_failure(monkeypatch):
    transport = SSHTransport("romeo")
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 255, b"", b"Permission denied (publickey)")
    monkeypatch.setattr(transport, "run", invoke)
    with pytest.raises(BackendError, match="Permission denied"):
        transport.write("/isolated/input.bin", b"input")
    assert len(calls) == 1
