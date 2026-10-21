"""Controller/SSH RPC regressions using only fake transports and local files."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

import execution_backend
from execution_backend import BackendError, RemoteHarnessRunner, SSHTransport, SlurmBackend, SlurmConfig
import remote_worker

REPO = Path(__file__).resolve().parents[1]
FILES = {"kernel.cu":b'extern "C" __global__ void add(){}',"input-0.bin":bytes(range(64))}
REQUEST = {"source":"kernel.cu","kernel":"add","mode":"benchmark","warmup":3,
           "iterations":3,"series":7,"grid":[1,1,1],"block":[32,1,1],"arguments":[]}


def archive_bytes(name="outputs/c.bin", content=b"independent-test-output"):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode="w:gz") as archive:
        member=tarfile.TarInfo(name)
        member.size=len(content)
        archive.addfile(member,io.BytesIO(content))
    return buffer.getvalue()


def native_response(payload, **updates):
    return {"ok":True,"stage":"complete","outputs":[],
            "request_fingerprint":payload["envelope"]["fingerprint"],
            "environment":{"uuid":"NATIVE-UUID","architecture":"sm_90"},
            "hardware_fingerprint":{"fingerprint_id":"native-hardware"},
            "latency_samples_ms":[.4,.3,.2],"series_means_ms":[.3],
            "measurement_protocol":{"warmup":3,"iterations":3,"series":7},
            "started_at_unix_ms":123456,**updates}


def reply(payload, *, artifact=None, **response_updates):
    artifact=archive_bytes() if artifact is None else artifact
    return {"ok":True,"response":native_response(payload,**response_updates),
            "artifacts_b64":base64.b64encode(artifact).decode(),
            "artifacts_sha256":hashlib.sha256(artifact).hexdigest(),
            "blob_uploads":len(payload["blobs"]),"blob_cache_hits":len(payload["files"])-len(payload["blobs"])}


class FakeExchangeTransport:
    def __init__(self, handler=None):
        self.calls=[]
        self.handler=handler
    def exchange(self,directory,payload,*,wait_seconds,python):
        self.calls.append({"directory":directory,"payload":copy.deepcopy(payload),"wait_seconds":wait_seconds,"python":python})
        return self.handler(payload,len(self.calls)) if self.handler else reply(payload)
    def read(self,*args,**kwargs):
        pytest.fail("Fast RPC must not use legacy artifact reads")
    def write(self,*args,**kwargs):
        pytest.fail("Fast RPC must not use legacy writes")
    def run(self,*args,**kwargs):
        pytest.fail("Fake transport must never invoke SSH or Slurm")


def make_runner(tmp_path,transport=None,*,mode="auto",capabilities=("exchange_v1",)):
    transport=transport or FakeExchangeTransport()
    backend=SlurmBackend(SlurmConfig(transport_mode=mode),REPO,tmp_path/"results","test-rpc-session",transport=transport)
    backend.job_id="123456"
    backend.hardware={"device_properties":{"uuid":"CACHED-UUID","architecture":"sm_90"},"fingerprint_id":"cached-hardware"}
    backend.transport_capabilities=capabilities
    backend._ready=True
    return RemoteHarnessRunner(backend),transport


@pytest.mark.parametrize("mode",["auto","exchange"])
def test_fast_requests_deduplicate_bytes_and_preserve_native_protocol_order_and_identity(tmp_path,mode):
    runner,transport=make_runner(tmp_path,mode=mode)
    first=runner._request("run",dict(REQUEST),FILES,"first")
    second=runner._request("run",dict(REQUEST),FILES,"second")
    assert len(transport.calls) == 2
    assert len(transport.calls[0]["payload"]["blobs"]) == 2
    assert transport.calls[1]["payload"]["blobs"] == {}
    assert [row["payload"]["envelope"]["identity"] for row in transport.calls] == ["first","second"]
    assert all(row["payload"]["envelope"]["request"] == REQUEST for row in transport.calls)
    for result in (first,second):
        assert result["latency_samples_ms"] == [.4,.3,.2]
        assert result["measurement_protocol"] == {"warmup":3,"iterations":3,"series":7}
        assert result["started_at_unix_ms"] == 123456
        assert result["environment"]["uuid"] == "NATIVE-UUID"
        assert result["hardware_fingerprint"]["fingerprint_id"] == "native-hardware"
        assert result["transport_receipt"]["mode"] == "exchange_v1"
    assert second["transport_receipt"]["blob_uploads"] == 0
    assert (tmp_path/"results/run-first/outputs/c.bin").read_bytes() == b"independent-test-output"


def test_missing_cached_blob_resends_only_that_blob_with_identical_request_identity(tmp_path):
    missing=hashlib.sha256(FILES["input-0.bin"]).hexdigest()
    def handle(payload,index):
        if index == 2:
            assert payload["blobs"] == {}
            return {"ok":False,"error_code":"missing_blobs","missing_blobs":[missing]}
        return reply(payload)
    runner,transport=make_runner(tmp_path,FakeExchangeTransport(handle))
    runner._request("run",dict(REQUEST),FILES,"first")
    runner._request("run",dict(REQUEST),FILES,"second")
    second,third=(row["payload"] for row in transport.calls[1:])
    assert second["envelope"] == third["envelope"] and second["files"] == third["files"]
    assert second["envelope"]["identity"] == "second"
    assert set(third["blobs"]) == {missing}
    assert base64.b64decode(third["blobs"][missing]) == FILES["input-0.bin"]


def test_missing_cached_object_retries_it_together_with_uncommitted_new_blob(tmp_path):
    cached=hashlib.sha256(FILES["kernel.cu"]).hexdigest()
    new_input=b"new bytes not committed when another blob was missing"
    fresh=hashlib.sha256(new_input).hexdigest()
    def handle(payload,index):
        if index == 2:
            assert set(payload["blobs"]) == {fresh}
            # Publication is atomic: the remote missing response committed
            # neither the old missing object nor this newly supplied object.
            return {"ok":False,"error_code":"missing_blobs","missing_blobs":[cached]}
        if index == 3:
            assert set(payload["blobs"]) == {cached,fresh}
        return reply(payload)
    runner,transport=make_runner(tmp_path,FakeExchangeTransport(handle))
    runner._request("run",dict(REQUEST),FILES,"first")
    runner._request("run",dict(REQUEST),{**FILES,"input-0.bin":new_input},"mixed")
    assert len(transport.calls) == 3
    assert transport.calls[1]["payload"]["envelope"] == transport.calls[2]["payload"]["envelope"]


@pytest.mark.parametrize("fingerprint",[None,"0"*64])
def test_fast_response_requires_exact_fingerprint_before_accepting_artifacts(tmp_path,fingerprint):
    def handle(payload,index):
        result=reply(payload)
        if fingerprint is None:
            result["response"].pop("request_fingerprint")
        else:
            result["response"]["request_fingerprint"]=fingerprint
        return result
    runner,_=make_runner(tmp_path,FakeExchangeTransport(handle))
    with pytest.raises(BackendError,match="fingerprint"):
        runner._request("run",dict(REQUEST),FILES,"not-accepted")
    assert not (tmp_path/"results/run-not-accepted/result.json").exists()


def test_poisoned_native_receipt_is_preserved_then_next_request_is_refused(tmp_path):
    def handle(payload,index):
        if index == 1:
            return reply(payload,ok=False,stage="guards",worker_poisoned=True,worker_reusable=False,error="read-only guard failed")
        return {"ok":False,"error_code":"quarantined","error":"CUDA context poisoned"}
    runner,transport=make_runner(tmp_path,FakeExchangeTransport(handle))
    bad=runner._request("run",dict(REQUEST),FILES,"poisoned")
    assert bad["ok"] is False and bad["worker_poisoned"] is True and bad["stage"] == "guards"
    assert (tmp_path/"results/run-poisoned/result.json").exists()
    with pytest.raises(BackendError,match="quarantined"):
        runner._request("run",dict(REQUEST),FILES,"blocked-next")
    assert len(transport.calls) == 2 and runner.backend._ready is False


def test_fast_archive_with_valid_sha_cannot_escape_result_directory(tmp_path):
    malicious=archive_bytes("../outside.txt",b"must not escape")
    runner,_=make_runner(tmp_path,FakeExchangeTransport(lambda payload,index:reply(payload,artifact=malicious)))
    with pytest.raises(BackendError,match="Unsafe artifact"):
        runner._request("run",dict(REQUEST),FILES,"traversal")
    assert not (tmp_path/"results/outside.txt").exists()


@pytest.mark.parametrize("mode,capabilities,reason",[
    ("auto",(),"worker_capability_unavailable"),("legacy",("exchange_v1",),"explicit_legacy")])
def test_capability_absent_or_explicit_legacy_uses_only_legacy_transport(tmp_path,monkeypatch,mode,capabilities,reason):
    class LegacyTransport:
        def __init__(self):
            self.writes=[]
        def write(self,path,data):
            self.writes.append((path,data))
        def read(self,path,optional=False):
            assert path.endswith("artifacts.tar.gz")
            return None
        def exchange(self,*args,**kwargs):
            pytest.fail("Legacy selection must not call exchange")
    runner,transport=make_runner(tmp_path,LegacyTransport(),mode=mode,capabilities=capabilities)
    monkeypatch.setattr(runner.backend,"require_live_worker",lambda:None)
    def read(name,optional=False):
        if name.endswith("envelope.json"):
            return None
        assert name.endswith("response.json")
        return {"ok":True,"outputs":[],"environment":{"uuid":"LEGACY-NATIVE"}}
    monkeypatch.setattr(runner.backend,"_json",read)
    result=runner._request("run",dict(REQUEST),FILES,"legacy")
    assert len(transport.writes) == 3 and transport.writes[-1][0].endswith("envelope.json")
    assert result["transport_receipt"]["reason"] == reason
    assert result["environment"]["uuid"] == "LEGACY-NATIVE"


def test_strict_exchange_without_capability_refuses_before_any_remote_operation(tmp_path):
    runner,transport=make_runner(tmp_path,mode="exchange",capabilities=())
    with pytest.raises(BackendError,match="does not advertise exchange_v1"):
        runner._request("run",dict(REQUEST),FILES,"strict")
    assert not transport.calls


def test_different_worker_snapshot_refuses_before_any_write_or_submission(tmp_path):
    class ExistingSnapshot:
        reads=[]
        def read(self,path,optional=False):
            self.reads.append(path)
            assert path.endswith("snapshot.json")
            return json.dumps({"sha256":"different-original-snapshot"}).encode()
        def write(self,*args,**kwargs):
            pytest.fail("An existing worker snapshot must never be overwritten")
        def run(self,*args,**kwargs):
            pytest.fail("Snapshot mismatch must not stage, submit or mutate a job")
    transport=ExistingSnapshot()
    backend=SlurmBackend(SlurmConfig(),REPO,tmp_path/"results","existing-original",transport=transport)
    with pytest.raises(BackendError,match="original controller snapshot"):
        backend.start()
    assert len(transport.reads) == 1 and backend.job_id is None


def test_ssh_lost_ack_and_timeout_retry_exact_payload_only_three_attempts(monkeypatch):
    transport=SSHTransport("never-contacted")
    calls=[]
    monkeypatch.setattr(execution_backend.time,"sleep",lambda seconds:None)
    def fake_run(command,**kwargs):
        calls.append((list(command),kwargs.copy()))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command,kwargs["timeout"])
        if len(calls) == 2:
            return subprocess.CompletedProcess(command,255,b"",b"Broken pipe after request publication")
        return subprocess.CompletedProcess(command,0,b'{"ok":false,"error_code":"pending"}',b"")
    monkeypatch.setattr(transport,"run",fake_run)
    body={"envelope":{"identity":"same-after-ack-loss","fingerprint":"f"*64},"blobs":{}}
    result=transport.exchange("/isolated/session",body,wait_seconds=2,python="python3.9")
    assert result["ssh_attempts"] == 3 and len(calls) == 3
    assert calls[0] == calls[1] == calls[2]
    assert json.loads(calls[0][1]["data"]) == body
    assert calls[0][0][0] == "python3.9" and calls[0][1]["timeout"] == 42


@pytest.mark.parametrize("failure",["timeout","network","authentication"])
def test_ssh_retries_are_bounded_and_authentication_is_not_retried(monkeypatch,failure):
    transport=SSHTransport("never-contacted")
    calls=[]
    monkeypatch.setattr(execution_backend.time,"sleep",lambda seconds:None)
    def fake_run(command,**kwargs):
        calls.append(kwargs["data"])
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command,kwargs["timeout"])
        error=b"Permission denied (publickey)." if failure == "authentication" else b"Connection reset by peer"
        return subprocess.CompletedProcess(command,255,b"",error)
    monkeypatch.setattr(transport,"run",fake_run)
    with pytest.raises(BackendError):
        transport.exchange("/isolated/session",{"envelope":{"identity":"fixed"}},wait_seconds=0)
    assert len(calls) == (1 if failure == "authentication" else 3)
    assert all(value == calls[0] for value in calls)


def filesystem_payload(identity="first",blobs=True):
    digest=hashlib.sha256(("run\0"+json.dumps(REQUEST,sort_keys=True)).encode())
    for name in sorted(FILES):
        digest.update(name.encode()+b"\0"+FILES[name])
    return {"envelope":{"identity":identity,"operation":"run","request":REQUEST,"fingerprint":digest.hexdigest()},
        "files":{name:{"sha256":hashlib.sha256(data).hexdigest(),"size":len(data)} for name,data in FILES.items()},
        "blobs":{hashlib.sha256(data).hexdigest():base64.b64encode(data).decode() for data in FILES.values()} if blobs else {}}


@pytest.mark.parametrize("damage",["bytes","symlink"])
def test_mutated_or_symlinked_cache_object_rejected_before_new_request_publication(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(remote_worker,"exchange_liveness",lambda directory,payload:None)
    assert remote_worker.exchange(tmp_path,filesystem_payload(),0)["error_code"] == "pending"
    digest=hashlib.sha256(FILES["kernel.cu"]).hexdigest()
    obj=tmp_path/"blobs/sha256"/digest
    if damage == "bytes":
        obj.chmod(0o644)
        obj.write_bytes(b"tampered after first publication")
    else:
        obj.unlink()
        outside=tmp_path/"external-source.cu"
        outside.write_bytes(FILES["kernel.cu"])
        obj.symlink_to(outside)
    result=remote_worker.exchange(tmp_path,filesystem_payload("second",blobs=False),0)
    assert result["ok"] is False and result["error_code"] == "blob_corrupt"
    assert not (tmp_path/"requests/second/envelope.json").exists()


def owned_session(tmp_path):
    directory=tmp_path/"owned-session"
    directory.mkdir()
    remote_worker.atomic_json(directory/"job_id.json",{"job_id":"123456","session_id":directory.name})
    remote_worker.atomic_json(directory/"ready.json",{"ok":True,"hardware":{"slurm_job_id":"123456"}})
    return directory,{"job_id":"123456","session_id":directory.name}


def test_liveness_accepts_only_owned_running_job_and_matching_ready_receipt(tmp_path,monkeypatch):
    directory,body=owned_session(tmp_path)
    calls=[]
    def scheduler(command,timeout):
        calls.append(command)
        assert command[0] == "squeue"
        return subprocess.CompletedProcess(command,0,"999|RUNNING\n123456|RUNNING\n","")
    monkeypatch.setattr(remote_worker,"run",scheduler)
    assert remote_worker.exchange_liveness(directory,body) is None and len(calls) == 1


def test_liveness_uses_accounting_when_job_left_queue(tmp_path,monkeypatch):
    directory,body=owned_session(tmp_path)
    calls=[]
    def scheduler(command,timeout):
        calls.append(command)
        text="" if command[0] == "squeue" else "123456|COMPLETED\n"
        return subprocess.CompletedProcess(command,0,text,"")
    monkeypatch.setattr(remote_worker,"run",scheduler)
    result=remote_worker.exchange_liveness(directory,body)
    assert result["error_code"] == "terminal" and result["state"] == "COMPLETED"
    assert [command[0] for command in calls] == ["squeue","sacct"]


@pytest.mark.parametrize("marker,code",[("STOP","not_running"),("quarantined.json","quarantined")])
def test_shutdown_or_quarantine_refused_without_scheduler_queries(tmp_path,monkeypatch,marker,code):
    directory,body=owned_session(tmp_path)
    (directory/marker).write_text("{}")
    monkeypatch.setattr(remote_worker,"run",lambda *args,**kwargs:pytest.fail("No scheduler query needed"))
    assert remote_worker.exchange_liveness(directory,body)["error_code"] == code


def test_wrong_ready_allocation_refused_before_scheduler_queries(tmp_path,monkeypatch):
    directory,body=owned_session(tmp_path)
    remote_worker.atomic_json(directory/"ready.json",{"ok":True,"hardware":{"slurm_job_id":"unrelated-job"}})
    monkeypatch.setattr(remote_worker,"run",lambda *args,**kwargs:pytest.fail("Ownership fails before scheduler query"))
    assert remote_worker.exchange_liveness(directory,body)["error_code"] == "ownership"
