"""Network loss simulation only: never connect to SSH, Slurm or a GPU."""
from dataclasses import asdict
import hashlib
import json
import subprocess
import sys

import pytest

import execution_backend as module


ADDRESS = b"ssh: connect to host romeo port 22: Can't assign requested address"
PQ_BANNER = (b'** WARNING: connection is not using a post-quantum key exchange algorithm.\r\n'
             b'** This session may be vulnerable to "store now, decrypt later" attacks.\r\n'
             b'** The server may need to be upgraded. See https://openssh.com/pq.html\r\n')


@pytest.mark.parametrize("diagnostic",[
    b"ssh_dispatch_run_fatal: Connection to 194.57.105.173 port 22: Can't assign requested address\r\n",
    b"ssh_dispatch_run_fatal: Connection to 194.57.105.173 port 22: Operation timed out\r\n",
    b"Connection to 194.57.105.173 port 22 timed out\r\n",
    b"Timeout, server romeo1.univ-reims.fr not responding.\r\n",
])
@pytest.mark.parametrize("banner",[b'',PQ_BANNER])
def test_actual_allocated_bootstrap_network_diagnostics_reconnect_with_exact_known_banner(diagnostic,banner):
    assert module.transient_ssh_failure(subprocess.CompletedProcess([],255,b"",banner+diagnostic))
    assert not module.transient_ssh_failure(subprocess.CompletedProcess([],1,b"",banner+diagnostic))


@pytest.mark.parametrize("prefix",[
    b'Unknown gateway banner\r\n',
    PQ_BANNER.replace(b'may need',b'might need'),
    PQ_BANNER.rsplit(b'** The server',1)[0],
    b'Warning: Permanently added something unusual\r\n',
    b'nvrtc compilation failure:\r\n'+PQ_BANNER,
    b'Model response quoted:\r\n'+PQ_BANNER,
])
def test_unknown_or_application_banner_does_not_retry_a_quoted_network_diagnostic(prefix):
    diagnostic=b"ssh_dispatch_run_fatal: Connection to 194.57.105.173 port 22: Can't assign requested address\r\n"
    assert not module.transient_ssh_failure(subprocess.CompletedProcess([],255,b"",prefix+diagnostic))


def test_only_exact_known_host_warning_is_removed_before_transient_diagnostic():
    warning=b"Warning: Permanently added 'romeo' (ED25519) to the list of known hosts.\r\n"
    diagnostic=b"ssh_dispatch_run_fatal: Connection to 194.57.105.173 port 22: Can't assign requested address\r\n"
    for stderr in [warning+PQ_BANNER+diagnostic,PQ_BANNER+warning+diagnostic]:
        assert module.transient_ssh_failure(subprocess.CompletedProcess([],255,b"",stderr))
    assert not module.transient_ssh_failure(subprocess.CompletedProcess([],255,b"",b"'list' object has no attribute 'items'"))


@pytest.mark.parametrize("message",[
    ADDRESS,
    b"ssh: connect to host romeo port 22: Cannot assign requested address",
    b"ssh: connect to host romeo port 22: Operation timed out",
    b"ssh_dispatch_run_fatal: Connection to romeo port 22: Operation timed out",
    b"Timeout, server romeo not responding.",
])
def test_real_openssh_network_loss_signatures_are_transient_only_for_ssh_failure(message):
    assert module.transient_ssh_failure(subprocess.CompletedProcess([],255,b"",message))
    assert not module.transient_ssh_failure(subprocess.CompletedProcess([],1,b"",message))


@pytest.mark.parametrize("message",[
    b"Can't assign requested address",  # Unqualified application error.
    b"operation timed out",             # Unqualified application error.
    b"Permission denied (publickey). Connection closed by remote host.",
    b"Host key verification failed. Connection closed.",
    b"REMOTE HOST IDENTIFICATION HAS CHANGED! Connection reset.",
    b"Unable to negotiate: no matching key exchange method found. kex_exchange_identification",
    b"Bad configuration option: temporary failure in name resolution",
    b"nvrtc compilation failed",
])
def test_authentication_key_configuration_and_application_errors_are_permanent(message):
    assert not module.transient_ssh_failure(subprocess.CompletedProcess([],255,b"",message))


def test_reconnecting_read_survives_more_than_three_failures_with_bounded_backoff(tmp_path,monkeypatch):
    log = tmp_path/"transport-reconnect.jsonl"
    transport = module.SSHTransport("never-contacted",reconnect=True,reconnect_log=log)
    calls,delays = [],[]
    monkeypatch.setattr(module.time,"sleep",delays.append)
    def invoke(command,**kwargs):
        calls.append((list(command),kwargs.copy()))
        if len(calls) == 2:
            raise subprocess.TimeoutExpired(command,kwargs["timeout"])
        return subprocess.CompletedProcess(command,0 if len(calls) == 7 else 255,
            b"authoritative-ready-bytes" if len(calls) == 7 else b"",ADDRESS)
    monkeypatch.setattr(transport,"run",invoke)
    assert transport.read("/isolated/session/ready.json") == b"authoritative-ready-bytes"
    assert len(calls) == 7 and all(call == calls[0] for call in calls)
    assert delays == [1,2,5,10,30,30]
    assert calls[0][1]["timeout"] == 60
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert len({row["operation_id"] for row in events}) == 1
    assert [row["status"] for row in events] == ["waiting_for_network"]*6+["reconnected"]
    assert events[-1]["attempts"] == 7 and events[-1]["resource"] == "/isolated/session/ready.json"


def test_only_scheduler_observations_reconnect_without_repeating_a_submission(monkeypatch):
    transport = module.SSHTransport("never-contacted",reconnect=True)
    calls = []
    monkeypatch.setattr(module.time,"sleep",lambda seconds:None)
    def invoke(command,**kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command,255 if len(calls)<5 else 0,
            b"771111|RUNNING|isolated-node\n",ADDRESS)
    monkeypatch.setattr(transport,"run",invoke)
    assert transport.scheduler_read(["squeue","-h","-j","771111"]).returncode == 0
    assert len(calls) == 5 and all(command == calls[0] for command in calls)
    for command in (["sbatch","job.sh"],["scancel","771111"],["scontrol","requeue","771111"]):
        with pytest.raises(ValueError,match="read-only"):
            transport.scheduler_read(command)
    assert len(calls) == 5


def test_ended_squeue_can_fall_through_to_accounting_without_transport_retry(monkeypatch):
    transport = module.SSHTransport("never-contacted",reconnect=True)
    monkeypatch.setattr(module.time,"sleep",lambda seconds:pytest.fail("An ended job is not a transport failure"))
    monkeypatch.setattr(transport,"run",lambda command,**kw:subprocess.CompletedProcess(command,1,b"",b"Invalid job id"))
    assert transport.scheduler_read(["squeue","-j","771111"],check=False).returncode == 1


def test_reconnecting_atomic_upload_repeats_exact_bytes_after_multiple_lost_acknowledgements(tmp_path,monkeypatch):
    transport = module.SSHTransport("never-contacted",reconnect=True)
    target = tmp_path/"input.bin"
    supplied = bytearray(b"immutable expected GPU input\0"*32)
    frozen = bytes(supplied)
    calls,delays = [],[]
    monkeypatch.setattr(module.time,"sleep",delays.append)
    def invoke(command,**kwargs):
        calls.append((list(command),kwargs.copy()))
        # The atomic remote script is executed locally; each write physically
        # commits before its acknowledgement is lost. No SSH process exists.
        result = subprocess.run([sys.executable,"-c",command[2],*command[3:]],
            input=kwargs["data"],capture_output=True,timeout=5)
        assert result.returncode == 0
        if len(calls) == 1:
            supplied[:] = b"x"*len(supplied)
        return result if len(calls) == 6 else subprocess.CompletedProcess(command,255,b"",ADDRESS)
    monkeypatch.setattr(transport,"run",invoke)
    transport.write(str(target),supplied)
    assert len(calls) == 6 and all(call == calls[0] for call in calls)
    assert target.read_bytes() == frozen
    assert calls[0][0][-1] == hashlib.sha256(frozen).hexdigest()
    assert not list(tmp_path.glob("*.upload-*"))
    assert delays == [1,2,5,10,30]


def test_reconnecting_exchange_keeps_same_envelope_and_payload_beyond_three_attempts(tmp_path,monkeypatch):
    transport = module.SSHTransport("never-contacted",reconnect=True,
        reconnect_log=tmp_path/"reconnect.jsonl")
    payload = {"envelope":{"identity":"one-request-only","fingerprint":"f"*64},"blobs":{"hash":"frozen"}}
    calls,delays = [],[]
    monkeypatch.setattr(module.time,"sleep",delays.append)
    def invoke(command,**kwargs):
        calls.append((list(command),kwargs.copy()))
        payload["envelope"]["identity"] = "mutable-caller-must-not-change-the-wire"
        if len(calls) <= 4:
            return subprocess.CompletedProcess(command,255,b"",b"Timeout, server romeo not responding.")
        return subprocess.CompletedProcess(command,0,b'{"ok":false,"error_code":"pending"}',b"")
    monkeypatch.setattr(transport,"run",invoke)
    result = transport.exchange("/isolated/session",payload,wait_seconds=2,python="python3.9")
    assert result["ssh_attempts"] == 5 and len(calls) == 5
    assert all(call == calls[0] for call in calls)
    assert json.loads(calls[0][1]["data"])["envelope"]["identity"] == "one-request-only"
    assert calls[0][1]["timeout"] == 42 and calls[0][0][0] == "python3.9"
    assert delays == [1,2,5,10]
    events = [json.loads(line) for line in (tmp_path/"reconnect.jsonl").read_text().splitlines()]
    assert all(row["resource"]["identity"] == "one-request-only" for row in events)
    assert len({row["payload_sha256"] for row in events}) == 1


@pytest.mark.parametrize("operation",["read","write","exchange"])
def test_reconnect_mode_never_retries_permanent_authentication_failure(monkeypatch,operation):
    transport = module.SSHTransport("never-contacted",reconnect=True)
    calls = []
    monkeypatch.setattr(module.time,"sleep",lambda seconds:pytest.fail("Authentication failure cannot wait/retry"))
    def invoke(command,**kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command,255,b"",b"Permission denied (publickey). Connection closed.")
    monkeypatch.setattr(transport,"run",invoke)
    with pytest.raises(module.BackendError,match="Permission denied"):
        if operation == "read":
            transport.read("/isolated/receipt")
        elif operation == "write":
            transport.write("/isolated/input",b"fixed")
        else:
            transport.exchange("/isolated/session",{"envelope":{"identity":"same"}},wait_seconds=1)
    assert len(calls) == 1


@pytest.mark.parametrize("command",[
    ["sbatch","--parsable","/isolated/worker.sh"],
    ["squeue","-h","-j","771111"],
    ["tar","-xzf","-","-C","/isolated/session"],
])
def test_reconnect_mode_does_not_replay_raw_scheduler_stage_or_status_commands(monkeypatch,command):
    calls = []
    def invoke(argv,**kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv,255,b"",ADDRESS)
    monkeypatch.setattr(module.subprocess,"run",invoke)
    monkeypatch.setattr(module.time,"sleep",lambda seconds:pytest.fail("Raw commands cannot retry"))
    with pytest.raises(module.BackendError,match="assign requested address"):
        module.SSHTransport("never-contacted",reconnect=True).run(command)
    assert len(calls) == 1


def test_slurm_config_enables_reconnection_but_direct_transport_retains_legacy_default(tmp_path):
    assert module.SSHTransport("never-contacted").reconnect is False
    config = module.SlurmConfig()
    assert asdict(config)["reconnect_transport"] is True
    backend = module.SlurmBackend(config,tmp_path,tmp_path/"proof","no-submission")
    assert backend.transport.reconnect is True
    assert backend.transport.reconnect_log == tmp_path/"proof/transport-reconnect.jsonl"
    disabled = module.SlurmBackend(module.SlurmConfig(reconnect_transport=False),tmp_path,tmp_path/"off","no-submission")
    assert disabled.transport.reconnect is False
    with pytest.raises(ValueError,match="boolean"):
        module.SlurmConfig(reconnect_transport="true")
