"""Worker-owned timing bounds, independent of wall clocks and GPFS mtimes."""
import itertools
import json
import os

import remote_worker


def test_reservations_enclose_execution_and_artifacts_on_same_worker_clock(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(remote_worker, "build_harness", lambda directory: directory / "unused")
    monkeypatch.setattr(remote_worker, "fingerprint", lambda executable: {})
    ticks = itertools.count(100, 100)
    monkeypatch.setattr(remote_worker.time, "monotonic_ns", lambda: next(ticks))
    # Wall time moves backwards; reservation evidence must still be ordered.
    wall = iter((10000, 1))
    monkeypatch.setattr(remote_worker.time, "time", lambda: next(wall))
    observations = {}
    def execute(directory, executable, envelope, timeout, hardware, native_worker=None):
        start = json.loads((directory / "started.json").read_text())
        observations[directory.name] = {"started": start, "execute": next(ticks)}
        if directory.name == "second":
            (tmp_path / "STOP").write_text("stop")
        return {"ok": True, "outputs": [], "environment": {"uuid": "native-uuid"}}
    def artifacts(directory, result):
        observations[directory.name]["artifacts"] = next(ticks)
    monkeypatch.setattr(remote_worker, "execute_request", execute)
    monkeypatch.setattr(remote_worker, "save_artifacts", artifacts)
    atomic = remote_worker.atomic_json
    def publish(path, result):
        if path.name == "response.json":
            marker = result["worker_reservation"]
            assert observations[path.parent.name]["artifacts"] < marker["end_monotonic_ns"]
            observations[path.parent.name]["published"] = next(ticks)
        atomic(path, result)
    monkeypatch.setattr(remote_worker, "atomic_json", publish)
    for name, operation in (("first", "run"), ("second", "profile")):
        request = tmp_path / "requests" / name
        request.mkdir(parents=True)
        (request / "envelope.json").write_text(json.dumps({"operation": operation, "fingerprint": name}))
    assert remote_worker.serve(tmp_path, 10) == 0
    markers = []
    for name in ("first", "second"):
        response = json.loads((tmp_path / "requests" / name / "response.json").read_text())
        marker = response["worker_reservation"]
        observed = observations[name]
        assert marker["clock"] == "time.monotonic_ns"
        assert marker["worker_pid"] == observed["started"]["worker_pid"] == os.getpid()
        assert marker["session_id"] == observed["started"]["session_id"] == tmp_path.name
        assert marker["job_id"] == observed["started"]["job_id"] == "12345"
        assert marker["start_monotonic_ns"] == observed["started"]["monotonic_ns"]
        assert marker["start_monotonic_ns"] < observed["execute"] < observed["artifacts"] < marker["end_monotonic_ns"] < observed["published"]
        assert response["environment"] == {"uuid": "native-uuid"}
        markers.append(marker)
    assert markers[0]["end_monotonic_ns"] < markers[1]["start_monotonic_ns"]


def test_legacy_interrupted_marker_cannot_invent_a_monotonic_start(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(remote_worker, "build_harness", lambda directory: directory / "unused")
    monkeypatch.setattr(remote_worker, "fingerprint", lambda executable: {})
    monkeypatch.setattr(remote_worker, "execute_request", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No replay")))
    monkeypatch.setattr(remote_worker, "save_artifacts", lambda *args: None)
    request = tmp_path / "requests" / "interrupted"
    request.mkdir(parents=True)
    (request / "envelope.json").write_text(json.dumps({"fingerprint": "original"}))
    (request / "started.json").write_text(json.dumps({"time": 10000, "job_id": "previous"}))
    assert remote_worker.serve(tmp_path, 10) == 3
    response = json.loads((request / "response.json").read_text())
    assert response["stage"] == "interrupted" and response["worker_poisoned"] is True
    assert response["worker_reservation"]["start_monotonic_ns"] is None
    assert json.loads((request / "started.json").read_text())["job_id"] == "previous"
