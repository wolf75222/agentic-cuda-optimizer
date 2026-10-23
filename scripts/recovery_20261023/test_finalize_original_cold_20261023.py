"""Cold-copy/finalizer regressions: temporary synthetic data and original graph."""
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil

import pytest

BASE = Path(__file__).parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

m = load("cold_original_operation",BASE/"finalize_original_775348_cold.py")
fixture_module = load("synthetic_original_fixture",BASE/"test_original_finalize_fixture_20261023.py")


def opaque_source(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    (source/"a").write_bytes(b"SYNTHETIC_OPAQUE_INPUT\x00\x01")
    (source/"directory").mkdir(mode=0o700)
    (source/"directory/b").write_bytes(b"SYNTHETIC_OPAQUE_CHECKPOINT")
    return source


def test_opaque_copy_and_replay_preserve_source_and_target_bytes(tmp_path):
    source = opaque_source(tmp_path)
    expected = m.inventory(source)
    target = tmp_path/"destination"
    m.copy_opaque(source,target,expected)
    assert m.inventory(target) == expected == m.inventory(source)
    before = (target/"a").stat().st_ino
    m.copy_opaque(source,target,expected)
    assert (target/"a").stat().st_ino == before and m.inventory(source) == expected


def test_interrupted_copy_resumes_without_duplicate_or_source_write(tmp_path):
    source = opaque_source(tmp_path)
    expected = m.inventory(source)
    target = tmp_path/"destination"
    def crash(phase):
        raise RuntimeError("SYNTHETIC_COPY_PROCESS_INTERRUPTION")
    with pytest.raises(RuntimeError): m.copy_opaque(source,target,expected,crash)
    m.copy_opaque(source,target,expected)
    assert m.inventory(source) == m.inventory(target) == expected


def test_copy_collision_never_overwrites_or_removes_target(tmp_path):
    source = opaque_source(tmp_path)
    expected = m.inventory(source)
    target = tmp_path/"destination"
    target.mkdir(); (target/"a").write_bytes(b"EXISTING_VALID_OTHER_ARTIFACT")
    with pytest.raises(m.Refused,match="partial_copy_collision"):
        m.copy_opaque(source,target,expected)
    assert (target/"a").read_bytes() == b"EXISTING_VALID_OTHER_ARTIFACT"
    assert m.inventory(source) == expected


def test_nested_destination_symlink_never_followed(tmp_path):
    source = opaque_source(tmp_path)
    target = tmp_path/"destination"; target.mkdir()
    elsewhere = tmp_path/"elsewhere"; elsewhere.mkdir(mode=0o700)
    (target/"directory").symlink_to(elsewhere,target_is_directory=True)
    with pytest.raises(m.Refused,match="aliased_evidence"):
        m.copy_opaque(source,target,m.inventory(source))
    assert list(elsewhere.iterdir()) == []


def test_copy_reconciles_exclusive_link_publication_crash(tmp_path):
    source = opaque_source(tmp_path)
    target = tmp_path/"destination"; target.mkdir()
    stages = tmp_path/"stages"; stages.mkdir()
    stage = stages/hashlib.sha256(b"a").hexdigest(); stage.mkdir()
    shutil.copy2(source/"a",stage/"000000.part")
    m.os.link(stage/"000000.part",target/"a")
    m.copy_opaque(source,target,m.inventory(source),staging_root=stages)
    assert (target/"a").stat().st_nlink == 1 and not list(stage.iterdir())


def test_partial_staging_bytes_retained_and_new_copy_finishes(tmp_path):
    source = opaque_source(tmp_path)
    stages = tmp_path/"stages"; stages.mkdir()
    stage = stages/hashlib.sha256(b"a").hexdigest(); stage.mkdir()
    (stage/"000000.part").write_bytes(b"SYNTHETIC_PARTIAL")
    target = tmp_path/"destination"
    m.copy_opaque(source,target,m.inventory(source),staging_root=stages)
    assert (stage/"000000.part").read_bytes() == b"SYNTHETIC_PARTIAL"
    assert m.inventory(source) == m.inventory(target)


def test_phase_records_are_immutable_and_idempotent(tmp_path):
    path = tmp_path/"prepared.json"
    m.publish(path,{"phase":"prepared"}); before=path.stat().st_mtime_ns
    m.publish(path,{"phase":"prepared"})
    assert path.stat().st_mtime_ns == before
    with pytest.raises(m.Refused,match="transaction_record_collision"):
        m.publish(path,{"phase":"something_else"})


@pytest.mark.parametrize("kind",["alias","wal"])
def test_non_cold_or_aliased_source_refuses(tmp_path,kind):
    source = opaque_source(tmp_path)
    if kind == "alias": (source/"link").symlink_to(source/"a")
    else: (source/"original-langgraph.sqlite-wal").write_bytes(b"SYNTHETIC")
    with pytest.raises(m.Refused): m.inventory(source)


def test_unknown_private_class_is_not_stringified_or_exposed():
    class Secret:
        def __repr__(self): raise AssertionError("Do not stringify a private object")
    with pytest.raises(m.Refused,match="unsupported_private_state_type"):
        m.canonical(Secret())


@pytest.fixture
def actual_original_fixture(tmp_path):
    return fixture_module.build_fixture(tmp_path)


def engine(fixture,destination,transaction,crash=None):
    return m.finalize_copy_only(fixture["generation"],destination,fixture["snapshot"],fixture["config"],
        fixture["history"],transaction,fixture["source_sha256"],crash_hook=crash)


def setup_copy(fixture,tmp_path):
    destination = tmp_path/"cold-copy"; transaction = tmp_path/"transaction"; transaction.mkdir()
    before = m.inventory(fixture["generation"])
    m.copy_opaque(fixture["generation"],destination,before,staging_root=transaction/"stage")
    return destination,transaction,before


def test_original_graph_finalizes_only_inside_copy(actual_original_fixture,tmp_path):
    fixture=actual_original_fixture
    destination,transaction,before=setup_copy(fixture,tmp_path)
    result=engine(fixture,destination,transaction)
    assert result["phase"] == "graph_end"
    assert result["cpu_finalizer_bodies_executed_this_invocation"] == 1
    assert result["method_status"] == "historical_failed_not_relabelled"
    assert result["model_calls"] == result["gpu_requests"] == result["observer_calls"] == 0
    assert m.inventory(fixture["generation"]) == before
    assert json.loads((fixture["generation"]/"summary.json").read_text())["status"] == "failed"
    assert json.loads((destination/"summary.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("window",["after_checkpoint_relocation_before_receipt","after_relocation","after_graph_end_before_commit"])
def test_durable_cpu_crash_recovery_without_historical_change(actual_original_fixture,tmp_path,window):
    fixture=actual_original_fixture
    destination,transaction,before=setup_copy(fixture,tmp_path)
    def crash(phase):
        if phase == window: raise RuntimeError("SYNTHETIC_CPU_PROCESS_INTERRUPTION")
    with pytest.raises(RuntimeError,match="SYNTHETIC_CPU_PROCESS_INTERRUPTION"):
        engine(fixture,destination,transaction,crash)
    result=engine(fixture,destination,transaction)
    assert result["phase"] == "graph_end" and m.inventory(fixture["generation"]) == before
    if window == "after_graph_end_before_commit":
        assert result["cpu_finalizer_bodies_executed_this_invocation"] == 0


def test_selected_source_mismatch_refuses_before_original_finalizer(actual_original_fixture,tmp_path):
    fixture=actual_original_fixture
    destination,transaction,before=setup_copy(fixture,tmp_path)
    with pytest.raises(m.Refused,match="checkpoint_selected_artifact_changed"):
        m.finalize_copy_only(fixture["generation"],destination,fixture["snapshot"],fixture["config"],
            fixture["history"],transaction,"0"*64)
    assert not (transaction/"node-returned.json").exists()
    assert m.inventory(fixture["generation"]) == before


def test_process_death_after_durable_node_return_replays_without_cpu_body(actual_original_fixture,tmp_path):
    fixture=actual_original_fixture
    destination,transaction,before=setup_copy(fixture,tmp_path)
    def child():
        def die(phase):
            if phase == "after_node_returned": os._exit(73)
        engine(fixture,destination,transaction,die)
    process=multiprocessing.get_context("fork").Process(target=child)
    process.start();process.join(30)
    assert process.exitcode == 73 and (transaction/"node-returned.json").exists()
    result=engine(fixture,destination,transaction)
    assert result["phase"] == "graph_end"
    assert result["cpu_finalizer_bodies_executed_this_invocation"] == 0
    assert result["durable_node_returns_replayed_this_invocation"] == 1
    assert list((transaction/"copy-crash-opaque-archives").iterdir())
    assert m.inventory(fixture["generation"]) == before


def test_committed_replay_never_opens_checkpoint_or_executes_graph(tmp_path,monkeypatch):
    target=opaque_source(tmp_path); transaction=tmp_path/"transaction";transaction.mkdir()
    m.publish(transaction/"prepared.json",{"kind":"SYNTHETIC_PREPARED"})
    m.publish(transaction/"committed.json",{"destination":str(target),
        "operational_tool_sha256":m.sha(Path(m.__file__).read_bytes()),
        "prepared_sha256":m.sha((transaction/"prepared.json").read_bytes()),
        "final_copy_inventory_sha256":m.sha(m.encoded(m.inventory(target)))})
    def no_api(*args,**kwargs): raise AssertionError("No checkpoint/model/backend API on committed replay")
    monkeypatch.setattr(m,"finalize_copy_only",no_api)
    monkeypatch.setattr(m.sqlite3,"connect",no_api)
    row=m.replay(transaction,target)
    assert row["phase"] == "committed_readonly_replay" and row["checkpoint_api_calls"] == 0


def test_staged_journal_without_owned_relocation_refuses(tmp_path):
    source=opaque_source(tmp_path); target=tmp_path/"copy";target.mkdir();transaction=tmp_path/"tx";transaction.mkdir()
    (target/"original-langgraph.sqlite-wal").write_bytes(b"SYNTHETIC_UNKNOWN_WAL")
    with pytest.raises(m.Refused,match="unowned_copy_journal_refused"):
        m.recover_owned_copy_journal(source,target,transaction)


def test_changed_original_algorithm_blob_refuses_before_import(actual_original_fixture,tmp_path,monkeypatch):
    fixture=actual_original_fixture
    fake_repo=tmp_path/"repository-parent/work/agentic-cuda-optimizer"
    fake_repo.mkdir(parents=True)
    snapshot=fake_repo.parents[1]/"work/study-source-fefd63c/work/original-1e9464da54df"
    shutil.copytree(fixture["snapshot"],snapshot)
    manifest={"upstream":dict(fixture["source_identity"],path=str(snapshot))}
    original_run=m.h.subprocess.run
    def git_read_only(command,**options):
        assert command[:2] == ["git","-C"]
        return original_run([*command[:2],str(m.REPO),*command[3:]],**options)
    monkeypatch.setattr(m.h.subprocess,"run",git_read_only)
    (snapshot/"optimizer_agent/nodes.py").write_bytes(b"MALFORMED_SYNTHETIC_ALGORITHM_BYTES")
    with pytest.raises(m.Refused,match="original_source_bytes_changed"):
        m.h.authenticate_original(fake_repo,fixture["generation"],manifest)


def test_default_plan_never_opens_real_campaign(monkeypatch,capsys):
    monkeypatch.setattr(m.sys,"argv",[m.__file__])
    def no_operation(*args,**kwargs): raise AssertionError("Plan may not touch the real campaign")
    monkeypatch.setattr(m,"operate",no_operation)
    assert m.main() == 0
    assert json.loads(capsys.readouterr().out)["actual_checkpoint_opened"] is False
