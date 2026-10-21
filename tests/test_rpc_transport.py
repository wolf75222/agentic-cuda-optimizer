"""Filesystem-only exchange_v1 proofs. No SSH, Slurm, compiler or GPU runs."""
import base64
import hashlib
import json
from pathlib import Path

import pytest

import remote_worker


DATA = {"kernel.cu": b'extern "C" __global__ void add(){}',
        "input-0.bin": bytes(range(64))}


def payload(identity="rpc-first", *, blobs=True, request=None):
    request = request or {"source":"kernel.cu", "kernel":"add", "mode":"validate", "arguments":[]}
    digest = hashlib.sha256(("run\0"+json.dumps(request, sort_keys=True)).encode())
    for name in sorted(DATA):
        digest.update(name.encode()+b"\0"+DATA[name])
    return {"envelope":{"identity":identity,"operation":"run","request":request,"fingerprint":digest.hexdigest()},
            "files":{name:{"sha256":hashlib.sha256(data).hexdigest(),"size":len(data)} for name,data in DATA.items()},
            "blobs":{hashlib.sha256(data).hexdigest():base64.b64encode(data).decode() for data in DATA.values()} if blobs else {}}


@pytest.fixture
def session(tmp_path, monkeypatch):
    # Liveness itself has separate scheduler tests. These tests exercise the
    # byte/identity protocol and cannot launch a Slurm or CUDA process.
    monkeypatch.setattr(remote_worker,"exchange_liveness",lambda directory,payload:None)
    return tmp_path


def request_directory(session, identity="rpc-first"):
    return session/"requests"/identity


def respond(directory, envelope):
    response = {"ok":True,"stage":"complete","request_fingerprint":envelope["fingerprint"],
                "outputs":[],"test_execution_id":envelope["identity"]}
    (directory/"artifacts.tar.gz").write_bytes(b"opaque-test-artifact-archive")
    remote_worker.atomic_json(directory/"response.json",response)
    return response


def test_lost_ack_retry_collects_one_execution_and_envelope_is_published_last(session,monkeypatch):
    publish, execute = [], []
    atomic = remote_worker.atomic_json
    def observe(path,value):
        path=Path(path)
        if path.name == "envelope.json":
            # A worker may consume immediately at this boundary: every source
            # and input must already be complete and match its manifest.
            for name,data in DATA.items():
                assert (path.parent/name).read_bytes() == data
            publish.append(str(path))
        atomic(path,value)
        if path.name == "envelope.json" and not (path.parent/"response.json").exists():
            execute.append(value["identity"])
            respond(path.parent,value)
    monkeypatch.setattr(remote_worker,"atomic_json",observe)
    first=remote_worker.exchange(session,payload(),wait_seconds=0)
    assert first["ok"] is True
    # The first response is deliberately treated as a lost acknowledgement.
    second=remote_worker.exchange(session,payload(blobs=False),wait_seconds=0)
    assert second["ok"] is True and second["response"] == first["response"]
    assert base64.b64decode(second["artifacts_b64"]) == b"opaque-test-artifact-archive"
    assert publish == [str(request_directory(session)/"envelope.json")]
    assert execute == ["rpc-first"]


def test_new_request_reuses_session_content_addressed_blobs_without_reupload(session):
    first=remote_worker.exchange(session,payload(),wait_seconds=0)
    assert first["ok"] is False and first["error_code"] == "pending"
    first_envelope=json.loads((request_directory(session)/"envelope.json").read_text())
    respond(request_directory(session),first_envelope)
    second=remote_worker.exchange(session,payload("rpc-second",blobs=False),wait_seconds=0)
    assert second["ok"] is False and second["error_code"] == "pending"
    assert second["blob_uploads"] == 0 and second["blob_cache_hits"] == len(DATA)
    for name,data in DATA.items():
        assert (request_directory(session,"rpc-second")/name).read_bytes() == data


def test_missing_blobs_return_exact_digests_without_publishing_partial_work(session):
    missing=remote_worker.exchange(session,payload(blobs=False),wait_seconds=0)
    assert missing["ok"] is False and missing["error_code"] == "missing_blobs"
    assert set(missing["missing_blobs"]) == {hashlib.sha256(data).hexdigest() for data in DATA.values()}
    assert not (request_directory(session)/"envelope.json").exists()
    # Providing exactly the missing immutable bytes makes the SAME ID usable.
    recovered=remote_worker.exchange(session,payload(),wait_seconds=0)
    assert recovered["error_code"] == "pending"
    assert (request_directory(session)/"envelope.json").is_file()


def test_corrupt_blob_is_rejected_before_envelope_publication(session):
    request=payload()
    digest=next(iter(request["blobs"]))
    request["blobs"][digest]=base64.b64encode(b"wrong immutable bytes").decode()
    rejected=remote_worker.exchange(session,request,wait_seconds=0)
    assert rejected["ok"] is False and rejected["error_code"] == "blob_corrupt"
    assert not (request_directory(session)/"envelope.json").exists()


def test_request_identity_cannot_collect_another_fingerprint_even_after_completion(session):
    remote_worker.exchange(session,payload(),wait_seconds=0)
    directory=request_directory(session)
    original=json.loads((directory/"envelope.json").read_text())
    response=respond(directory,original)
    conflict=payload(request={"source":"kernel.cu","kernel":"add","mode":"benchmark","arguments":[]})
    rejected=remote_worker.exchange(session,conflict,wait_seconds=0)
    assert rejected["ok"] is False and rejected["error_code"] == "fingerprint_conflict"
    assert json.loads((directory/"envelope.json").read_text()) == original
    assert json.loads((directory/"response.json").read_text()) == response


def test_timeout_keeps_started_and_envelope_for_exact_request_recovery(session):
    pending=remote_worker.exchange(session,payload(),wait_seconds=0)
    assert pending["error_code"] == "pending"
    directory=request_directory(session)
    envelope=directory/"envelope.json"
    remote_worker.atomic_json(directory/"started.json",{"time":123,"job_id":"test-only"})
    before={path.name:(path.stat().st_ino,path.stat().st_mtime_ns,path.read_bytes())
            for path in (envelope,directory/"started.json")}
    again=remote_worker.exchange(session,payload(blobs=False),wait_seconds=0)
    assert again["error_code"] == "pending"
    assert before == {path.name:(path.stat().st_ino,path.stat().st_mtime_ns,path.read_bytes())
                      for path in (envelope,directory/"started.json")}
    respond(directory,json.loads(envelope.read_text()))
    recovered=remote_worker.exchange(session,payload(blobs=False),wait_seconds=0)
    assert recovered["ok"] is True
    assert recovered["response"]["test_execution_id"] == "rpc-first"


@pytest.mark.parametrize("state",["terminal","quarantined"])
def test_dead_or_quarantined_session_never_publishes_new_gpu_work(session,monkeypatch,state):
    monkeypatch.setattr(remote_worker,"exchange_liveness",lambda directory,payload:{"ok":False,"error_code":state})
    rejected=remote_worker.exchange(session,payload(),wait_seconds=0)
    assert rejected["ok"] is False and rejected["error_code"] == state
    assert not (request_directory(session)/"envelope.json").exists()


@pytest.mark.parametrize("state",["terminal","quarantined"])
def test_existing_matching_response_remains_collectible_after_session_ends(session,monkeypatch,state):
    remote_worker.exchange(session,payload(),wait_seconds=0)
    directory=request_directory(session)
    expected=respond(directory,json.loads((directory/"envelope.json").read_text()))
    def must_not_need_live_gpu(directory,payload):
        return {"ok":False,"error_code":state}
    monkeypatch.setattr(remote_worker,"exchange_liveness",must_not_need_live_gpu)
    recovered=remote_worker.exchange(session,payload(blobs=False),wait_seconds=0)
    assert recovered["ok"] is True and recovered["response"] == expected
