"""Historical collection must never stage sources, submit, or execute CUDA."""
import json
import subprocess

import pytest

from execution_backend import BackendError, RemoteHarnessRunner, SlurmBackend, SlurmConfig


class ReadonlyTransport:
    def __init__(self, remote=None, *, state="COMPLETED"):
        self.remote = {"job_id": "123", "session_id": "history", "job_name": "aco-history"} if remote is None else remote
        self.state = state
        self.reads = []
        self.commands = []

    def read(self, path, optional=False):
        self.reads.append(path)
        if path.endswith("/job_id.json"):
            return self.remote if isinstance(self.remote, bytes) else json.dumps(self.remote).encode()
        if path.endswith("/snapshot.json"):
            return b'{"sha256":"different-from-current-controller"}'
        return None

    def run(self, command, **kwargs):
        self.commands.append(command)
        assert command[0] in {"squeue", "sacct"}, "historical collection must not mutate the remote session"
        assert command[command.index("-j") + 1] == "123"
        if command[0] == "squeue":
            output = b"123|RUNNING|romeo-a001\n" if self.state == "RUNNING" else b""
        else:
            output = f"123|{self.state}|0:0|00:00:11|romeo-a001\n".encode()
        return subprocess.CompletedProcess(command, 0, output, b"")

    def write(self, *args, **kwargs):
        pytest.fail("historical attachment must not write remote files")

    def exchange(self, *args, **kwargs):
        pytest.fail("historical attachment must not execute an RPC")


def backend(tmp_path, transport=None, *, local=True):
    result = SlurmBackend(SlurmConfig(), tmp_path / "changed-source-tree", tmp_path / "results",
                          "history", transport=transport or ReadonlyTransport())
    if local:
        (result.results_dir / "slurm-session.json").write_text(json.dumps({
            "job_id": "123", "session_id": "history", "remote_dir": result.remote_dir}))
    return result


def test_attach_after_lost_submission_ack_uses_remote_ownership_without_creating_local_receipt(tmp_path):
    instance = backend(tmp_path, local=False)
    before = set(instance.results_dir.iterdir())
    receipt = instance.attach_existing()
    assert receipt == {"job_id": "123", "state": "COMPLETED", "exit_code": "0:0",
                       "elapsed": "00:00:11", "nodes": "romeo-a001", "terminal": True}
    assert instance.job_id == "123" and instance.historical_only and not instance._ready
    assert set(instance.results_dir.iterdir()) == before
    assert instance.transport.reads == [f"{instance.remote_dir}/job_id.json"]


def test_changed_controller_can_collect_without_reading_or_replacing_worker_snapshot(tmp_path, monkeypatch):
    instance = backend(tmp_path)
    original = (instance.results_dir / "slurm-session.json").read_bytes()
    monkeypatch.setattr(instance, "stage", lambda: pytest.fail("must not stage changed controller sources"))
    assert instance.attach_readonly()["terminal"]
    assert instance.collect_diagnostics()["state"] == "COMPLETED"
    assert instance.wait_until_finished()["terminal"]
    assert (instance.results_dir / "slurm-session.json").read_bytes() == original
    assert (instance.results_dir / "slurm-diagnostics/snapshot.json").read_bytes() == b'{"sha256":"different-from-current-controller"}'
    assert all(command[0] in {"squeue", "sacct"} for command in instance.transport.commands)


@pytest.mark.parametrize("update", [
    {"job_id": "456"}, {"session_id": "foreign"}, {"remote_dir": "/tmp/foreign"},
    {"job_id": True}, {"job_id": ["123"]}, {"job_id": "123_1"},
])
def test_conflicting_or_invalid_local_receipt_fails_before_status_or_mutation(tmp_path, update):
    instance = backend(tmp_path)
    path = instance.results_dir / "slurm-session.json"
    value = json.loads(path.read_bytes())
    value.update(update)
    path.write_text(json.dumps(value))
    original = path.read_bytes()
    with pytest.raises(BackendError, match="ownership"):
        instance.attach_existing()
    assert instance.job_id is None and instance.historical_only
    assert not instance.transport.commands and path.read_bytes() == original


@pytest.mark.parametrize("remote", [
    b"null", b"[]", b"invalid-json", b"{}", b'{"job_ids":["123","456"],"job_id":"123","session_id":"history"}',
    {"job_id": "123", "session_id": "history", "uncertain": True},
    {"job_id": "123", "session_id": "foreign"},
    {"job_id": "123", "session_id": "history", "directory": "/tmp/foreign"},
    {"job_id": "123", "session_id": "history", "job_name": "aco-foreign"},
])
def test_remote_receipt_must_unambiguously_authenticate_session_and_directory(tmp_path, remote):
    instance = backend(tmp_path, ReadonlyTransport(remote), local=False)
    with pytest.raises(BackendError, match="ownership"):
        instance.attach_existing()
    assert instance.job_id is None and not instance.transport.commands


@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, [], {}, "", "0", "0123", "-123", "123_1", "123;id", "１２３"])
def test_remote_job_id_is_one_canonical_positive_scheduler_integer(tmp_path, value):
    transport = ReadonlyTransport({"job_id": value, "session_id": "history"})
    instance = backend(tmp_path, transport, local=False)
    with pytest.raises(BackendError, match="scheduler job ID"):
        instance.attach_existing()
    assert instance.job_id is None and not transport.commands


def test_missing_remote_receipt_cannot_be_inferred_from_valid_local_receipt(tmp_path):
    instance = backend(tmp_path)
    instance.transport.read = lambda *args, **kwargs: None
    with pytest.raises(BackendError, match="missing or ambiguous"):
        instance.attach_existing()
    assert instance.job_id is None and not instance.transport.commands


def test_failed_attach_retires_previously_ready_backend_instead_of_leaving_execution_access(tmp_path):
    instance = backend(tmp_path)
    instance.job_id, instance._ready = "999", True
    instance.transport_capabilities = ("exchange_v1",)
    with pytest.raises(BackendError, match="Attached and remote"):
        instance.attach_existing()
    assert instance.job_id is None and not instance._ready
    assert instance.historical_only and instance.transport_capabilities == ()


@pytest.mark.parametrize("operation", [
    lambda instance: instance.stage(), lambda instance: instance.start(),
    lambda instance: instance.require_live_worker(), lambda instance: instance.wait_until_ready(),
    lambda instance: instance.runner(),
    lambda instance: RemoteHarnessRunner(instance)._request("run", {}, {}, "never-enqueued"),
    lambda instance: RemoteHarnessRunner(instance)._exchange_request({"identity": "never-enqueued"}, {}),
])
def test_every_execution_entry_point_fails_before_transport_after_historical_attach(tmp_path, operation):
    instance = backend(tmp_path)
    instance.attach_existing()
    commands, reads = list(instance.transport.commands), list(instance.transport.reads)
    with pytest.raises(BackendError, match="Historical read-only"):
        operation(instance)
    assert instance.transport.commands == commands and instance.transport.reads == reads


def test_allow_terminal_compatibility_is_readonly_attachment_not_stage_or_submit(tmp_path, monkeypatch):
    instance = backend(tmp_path)
    monkeypatch.setattr(instance, "stage", lambda: pytest.fail("historical start must not stage"))
    assert instance.start(allow_terminal=True) == "123"
    assert instance.historical_only


@pytest.mark.parametrize("historical", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_close_already_terminal_is_noop_before_stop_or_scancel(tmp_path, historical, cancel):
    instance = backend(tmp_path)
    if historical:
        instance.attach_existing()
    else:
        instance.job_id = "123"
    instance.close(cancel=cancel)
    assert all(command[0] != "scancel" for command in instance.transport.commands)


def test_historical_running_worker_allows_only_explicit_graceful_owned_close(tmp_path):
    transport = ReadonlyTransport(state="RUNNING")
    instance = backend(tmp_path, transport)
    assert instance.attach_existing()["terminal"] is False
    writes = []
    transport.write = lambda path, data: writes.append((path, data))
    with pytest.raises(BackendError, match="forbids cancellation"):
        instance.close(cancel=True)
    assert writes == []
    instance.close()
    assert writes == [(f"{instance.remote_dir}/STOP", b"stop\n")]
    assert all(command[0] != "scancel" for command in transport.commands)


def test_close_rechecks_ownership_even_after_successful_historical_attachment(tmp_path):
    transport = ReadonlyTransport(state="RUNNING")
    instance = backend(tmp_path, transport)
    instance.attach_existing()
    transport.remote["session_id"] = "foreign"
    with pytest.raises(BackendError, match="ownership"):
        instance.close()


def test_historical_close_does_not_write_stop_when_running_state_is_unconfirmed(tmp_path, monkeypatch):
    instance = backend(tmp_path)
    instance.attach_existing()
    monkeypatch.setattr(instance, "status", lambda: {"state": "ACCOUNTING_PENDING"})
    with pytest.raises(BackendError, match="currently RUNNING"):
        instance.close()
