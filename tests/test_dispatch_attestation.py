"""CPU-only source observations; fixtures use local Git, never external tools."""
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from optimizer_agent import dispatch_attestation as attestation
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.suite_orchestrator import SuiteOrchestrator, write_json


@pytest.fixture
def source_repo(tmp_path):
    repo = tmp_path / "source"
    (repo / "optimizer_agent").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "optimizer_agent/runtime.py").write_text("def run(): return 1\n")
    (repo / "scripts/driver.py").write_text("print('fixture')\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=CPU test", "-c", "user.email=cpu@example.invalid", "commit", "-qm", "Fixture")
    return repo


def task():
    return {"id": "unit.final", "kind": "final", "session": "unit-session", "attempt": 0,
            "payload": [{"config": {"workload": "matmul", "private_literal": "DO-NOT-ARCHIVE-PAYLOAD"}}],
            "private_final_seed": "DO-NOT-ARCHIVE-SEALED-SEED"}


def read(study, reference):
    path = study / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    return json.loads(path.read_text())


def test_controller_then_task_observe_source_without_literal_configuration(source_repo, tmp_path, monkeypatch):
    study = tmp_path / "study"
    monkeypatch.setitem(sys.modules, "optimizer_agent.cpu_fixture", SimpleNamespace(__file__=str(source_repo / "optimizer_agent/runtime.py")))
    controller = attestation.capture_dispatch_attestation(study, repo=source_repo)
    reference = attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=controller, repo=source_repo)
    record = read(study, reference)
    assert record["scope"] == attestation.SCOPE
    assert record["source_identity"]["git"]["HEAD_at_dispatch"]
    assert record["source_identity"]["git"]["dirty"] is False
    assert record["controller_epoch"] == controller
    assert record["task"] == {"task_id": "unit.final", "kind": "final", "session": "unit-session", "attempt": 0,
                               "payload_sha256": attestation._hash(task()["payload"])}
    assert record["module_paths"]["optimizer_agent.cpu_fixture"]["repository_relative"] == "optimizer_agent/runtime.py"
    assert record["module_paths"]["optimizer_agent.cpu_fixture"]["on_disk_sha256"] == record["source_identity"]["python_source_hashes"]["optimizer_agent/runtime.py"]
    raw = (study / reference["path"]).read_text()
    assert "DO-NOT-ARCHIVE" not in raw and "private_final_seed" not in raw
    assert "executedSHA" not in raw and "controller_revision" not in raw
    assert any("loaded bytecode" in limitation for limitation in record["limitations"])


def test_new_version_appends_epochs_and_keeps_previous_bytes(source_repo, tmp_path):
    study = tmp_path / "study"
    first_controller = attestation.capture_dispatch_attestation(study, repo=source_repo)
    first_task = attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=first_controller, repo=source_repo)
    old_bytes = {ref["path"]: (study / ref["path"]).read_bytes() for ref in (first_controller, first_task)}
    (source_repo / "optimizer_agent/runtime.py").write_text("def run(): return 2\n")
    second_controller = attestation.capture_dispatch_attestation(study, repo=source_repo)
    second_task = attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=second_controller, repo=source_repo)
    assert read(study, second_controller)["previous_epoch"] == first_controller
    assert read(study, second_task)["previous_epoch"] == first_task
    assert read(study, second_task)["controller_epoch"] == second_controller
    assert read(study, second_task)["source_identity"]["git"]["dirty"] is True
    assert read(study, second_task)["source_identity"] != read(study, first_task)["source_identity"]
    assert all((study / name).read_bytes() == raw for name, raw in old_bytes.items())


def test_source_change_during_capture_publishes_no_epoch(source_repo, tmp_path, monkeypatch):
    original = attestation._source_hashes
    count = 0
    def changing(repo):
        nonlocal count
        count += 1
        result = original(repo)
        if count == 2:
            result["optimizer_agent/runtime.py"] = "changed"
        return result
    monkeypatch.setattr(attestation, "_source_hashes", changing)
    with pytest.raises(ValueError, match="changed during"):
        attestation.capture_dispatch_attestation(tmp_path / "study", repo=source_repo)
    assert not list((tmp_path / "study").rglob("*.json"))


def test_changed_source_cannot_reuse_current_parent_but_new_parent_is_allowed(source_repo, tmp_path):
    study = tmp_path / "study"
    parent = attestation.capture_dispatch_attestation(study, repo=source_repo)
    (source_repo / "optimizer_agent/runtime.py").write_text("def run(): return 2\n")
    with pytest.raises(ValueError, match="current controller"):
        attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=parent, repo=source_repo)
    assert not list((study / "dispatch-attestations/tasks").rglob("*.json"))
    new_parent = attestation.capture_dispatch_attestation(study, repo=source_repo)
    assert attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=new_parent, repo=source_repo)


def test_parent_comparison_ignores_pid_clock_and_module_inventory(source_repo, tmp_path, monkeypatch):
    study = tmp_path / "study"
    parent = attestation.capture_dispatch_attestation(study, repo=source_repo)
    monkeypatch.setattr(attestation.os, "getpid", lambda: 987654)
    monkeypatch.setattr(attestation.time, "time", lambda: 42)
    monkeypatch.setattr(attestation, "_module_paths", lambda repo: {"extra_import": {"path": "/observed-only"}})
    record = read(study, attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=parent, repo=source_repo))
    assert record["pid"] == 987654 and record["observed_at_unix"] == 42


def test_unrelated_git_dirty_context_does_not_change_parent_source_identity(source_repo, tmp_path, monkeypatch):
    study = tmp_path / "study"
    parent = attestation.capture_dispatch_attestation(study, repo=source_repo)
    original = attestation._git_identity(source_repo)
    monkeypatch.setattr(attestation, "_git_identity", lambda repo: {**original, "dirty": True,
        "porcelain_status": ["?? output-receipts/", " M docs/example.md"]})
    record = read(study, attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=parent, repo=source_repo))
    assert record["source_identity"]["git"]["dirty"] is True
    assert record["source_identity"]["source_identity_hash"] == read(study, parent)["source_identity"]["source_identity_hash"]


def test_git_head_change_during_capture_is_refused(source_repo, tmp_path, monkeypatch):
    original = attestation._git_identity(source_repo)
    calls = 0
    def changed(repo):
        nonlocal calls
        calls += 1
        return {**original, "HEAD_at_dispatch": original["HEAD_at_dispatch"] if calls == 1 else "changed-head"}
    monkeypatch.setattr(attestation, "_git_identity", changed)
    with pytest.raises(ValueError, match="changed during"):
        attestation.capture_dispatch_attestation(tmp_path / "study", repo=source_repo)
    assert not list((tmp_path / "study").rglob("*.json"))


@pytest.mark.parametrize("change", ["parent_sha", "parent_path", "record_digest"])
def test_parent_or_predecessor_identity_corruption_fails_closed(source_repo, tmp_path, change):
    study = tmp_path / "study"
    parent = attestation.capture_dispatch_attestation(study, repo=source_repo)
    if change == "parent_sha":
        parent = {**parent, "sha256": "bad"}
    elif change == "parent_path":
        parent = {**parent, "path": "../../outside.json"}
    else:
        path = study / parent["path"]
        record = json.loads(path.read_text())
        record["pid"] = -1
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=parent, repo=source_repo)
    assert not list((study / "dispatch-attestations/tasks").rglob("*.json"))


def test_corrupt_previous_epoch_cannot_be_silently_replaced(source_repo, tmp_path):
    study = tmp_path / "study"
    reference = attestation.capture_dispatch_attestation(study, repo=source_repo)
    path = study / reference["path"]
    raw = path.read_bytes() + b"broken"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        attestation.capture_dispatch_attestation(study, repo=source_repo)
    assert path.read_bytes() == raw
    assert len(list(path.parent.glob("*.json"))) == 1


def test_changed_older_epoch_is_detected_through_predecessor_link(source_repo, tmp_path):
    study = tmp_path / "study"
    first = attestation.capture_dispatch_attestation(study, repo=source_repo)
    second = attestation.capture_dispatch_attestation(study, repo=source_repo)
    path = study / first["path"]
    record = json.loads(path.read_text())
    record["pid"] = -1
    record["record_hash"] = attestation._hash({k:v for k,v in record.items() if k != "record_hash"})
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="chain identity"):
        attestation.capture_dispatch_attestation(study, repo=source_repo)
    assert len(list(path.parent.glob("*.json"))) == 2
    assert read(study, second)["previous_epoch"] == first


def test_task_rejects_parent_with_corrupt_older_controller_link(source_repo, tmp_path):
    study = tmp_path / "study"
    first = attestation.capture_dispatch_attestation(study, repo=source_repo)
    second = attestation.capture_dispatch_attestation(study, repo=source_repo)
    path = study / first["path"]
    changed = json.loads(path.read_text())
    changed["pid"] = -1
    # The earlier record remains internally self-consistent, but C2 retains
    # the original SHA of C1. A task must validate that predecessor link.
    changed["record_hash"] = attestation._hash({k:v for k,v in changed.items() if k != "record_hash"})
    path.write_text(json.dumps(changed))
    assert read(study, second)["previous_epoch"] == first
    with pytest.raises(ValueError, match="chain identity"):
        attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=second, repo=source_repo)
    assert not list((study / "dispatch-attestations/tasks").rglob("*.json"))


def test_valid_earlier_controller_reference_remains_allowed_after_another_epoch(source_repo, tmp_path):
    study = tmp_path / "study"
    earlier = attestation.capture_dispatch_attestation(study, repo=source_repo)
    attestation.capture_dispatch_attestation(study, repo=source_repo)
    reference = attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=earlier, repo=source_repo)
    assert read(study, reference)["controller_epoch"] == earlier


def test_first_publication_fsyncs_every_new_directory_parent_before_record(source_repo, tmp_path, monkeypatch):
    study = tmp_path / "new-parent" / "study"
    root = study / "dispatch-attestations"
    synced = []
    original_sync, original_write = attestation._fsync_directory, attestation._write_immutable
    def sync(path):
        original_sync(path)
        synced.append(Path(path))
    def write(path, raw):
        if path.parent.name == "controller":
            assert all(parent in synced for parent in (tmp_path, study.parent, study, root))
        else:
            assert root in synced and root / "tasks" in synced
        original_write(path, raw)
        assert path.parent in synced  # The final file entry is durable, too.
    monkeypatch.setattr(attestation, "_fsync_directory", sync)
    monkeypatch.setattr(attestation, "_write_immutable", write)
    parent = attestation.capture_dispatch_attestation(study, repo=source_repo)
    synced.clear()
    reference = attestation.capture_dispatch_attestation(study, task=task(), controller_epoch=parent, repo=source_repo)
    assert read(study, reference)["controller_epoch"] == parent


def test_failed_directory_parent_fsync_prevents_publication_and_retry_finishes_it(tmp_path, monkeypatch):
    target = tmp_path / "new-directory"
    original = attestation._fsync_directory
    def fail(path):
        if target.exists():
            raise OSError("CPU fixture parent fsync failure")
        original(path)
    monkeypatch.setattr(attestation, "_fsync_directory", fail)
    with pytest.raises(OSError, match="parent fsync"):
        attestation._mkdir_durable(target)
    assert target.is_dir() and not list(target.iterdir())
    calls = []
    def succeeding(path):
        original(path)
        calls.append(path)
    monkeypatch.setattr(attestation, "_fsync_directory", succeeding)
    attestation._mkdir_durable(target)
    assert calls == [tmp_path]


def test_change_in_observed_loaded_module_file_is_refused(source_repo, tmp_path, monkeypatch):
    outside = tmp_path / "loaded-external.py"
    outside.write_text("before\n")
    def modules(repo):
        old_hash = hashlib.sha256(outside.read_bytes()).hexdigest()
        outside.write_text("after\n")
        return {"optimizer_agent.external_fixture": {"path": str(outside), "on_disk_sha256": old_hash}}
    monkeypatch.setattr(attestation, "_module_paths", modules)
    with pytest.raises(ValueError, match="changed during"):
        attestation.capture_dispatch_attestation(tmp_path / "study", repo=source_repo)
    assert not list((tmp_path / "study").rglob("*.json"))


def test_immutable_atomic_publication_never_overwrites(tmp_path):
    target = tmp_path / "epoch.json"
    attestation._write_immutable(target, b"first\n")
    with pytest.raises(FileExistsError):
        attestation._write_immutable(target, b"second\n")
    assert target.read_bytes() == b"first\n"
    assert not list(tmp_path.glob(".epoch-*"))


def test_optional_archive_observation_has_explicit_limited_scope(source_repo, tmp_path):
    study = tmp_path / "study"
    folder = study / "controller-sources" / "recorded-revision"
    folder.mkdir(parents=True)
    archive = b"CPU fixture: no full TAR/Git authentication claimed"
    (folder / "source.tar.gz").write_bytes(archive)
    write_json(folder / "source.json", {"head": "recorded-revision", "dirty": [],
               "source_hashes": attestation._source_hashes(source_repo), "archive_sha256": hashlib.sha256(archive).hexdigest()})
    record = read(study, attestation.capture_dispatch_attestation(study, repo=source_repo))
    observed = record["matching_frozen_archive_observations"]
    assert len(observed) == 1
    assert "full TAR/Git authentication is separate" in observed[0]["scope"]
    assert record["source_identity"]["git"]["HEAD_at_dispatch"] != observed[0]["recorded_head"]
    (folder / "source.tar.gz").write_bytes(b"different")
    with pytest.raises(ValueError, match="archive digest"):
        attestation.capture_dispatch_attestation(study, repo=source_repo)


def test_without_git_does_not_invent_revision(tmp_path):
    repo = tmp_path / "source"
    (repo / "optimizer_agent").mkdir(parents=True)
    (repo / "optimizer_agent/runtime.py").write_text("pass\n")
    study = tmp_path / "study"
    record = read(study, attestation.capture_dispatch_attestation(study, repo=repo))
    assert record["source_identity"]["git"]["HEAD_at_dispatch"] is None
    assert record["source_identity"]["git"]["dirty"] is None


def make_suite(tmp_path):
    plan = plan_suite(CampaignConfig(run_id="attestation-unit", search_sizes=(64,), profile=False), seeds=(1, 2))
    plan["entries"] = [entry for entry in plan["entries"] if entry["workload"] == "matmul" and entry["seed"] == 1][:2]
    return SuiteOrchestrator(plan, tmp_path, backend_factory=lambda *args: pytest.fail("No real backend in CPU attestation test"))


def test_hooks_capture_controller_before_pool_and_task_before_execution(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    events = []
    parent = {"path": "CPU-controller-epoch", "sha256": "CPU-digest", "epoch_id": "CPU-epoch"}
    def capture(directory, *, task=None, controller_epoch=None):
        events.append(("task" if task else "controller", task["id"] if task else None))
        if task:
            assert controller_epoch == parent
        return parent
    class FakeExecutor:
        def __init__(self, **kwargs):
            events.append(("pool", None))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, row):
            function(row)
            result = Future()
            result.set_result(None)
            return result
    monkeypatch.setattr(attestation, "capture_dispatch_attestation", capture)
    monkeypatch.setattr("optimizer_agent.suite_orchestrator.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(suite, "_execute_task", lambda row: events.append(("execute", row["id"])))
    suite.run()
    assert events[:2] == [("controller", None), ("pool", None)]
    for offset in range(2, len(events), 2):
        assert events[offset][0] == "task" and events[offset+1] == ("execute", events[offset][1])


def test_prepared_guard_precedes_any_task_attestation(tmp_path, monkeypatch):
    suite = make_suite(tmp_path)
    row = suite.journal.tasks("generation")[0]
    write_json(suite._directory(row) / "bootstrap-recovery.json",
               {"task_id": row["id"], "previous_session": row["session"], "phase": "prepared"})
    monkeypatch.setattr(attestation, "capture_dispatch_attestation", lambda *args, **kwargs: pytest.fail("Prepared task was attested"))
    monkeypatch.setattr(suite, "_execute_task", lambda *args: pytest.fail("Prepared task executed"))
    suite._task(row)
