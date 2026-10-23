"""Synthetic fixtures only; this file never opens the real study checkpoint."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace as NS

import pytest

PATH = Path(__file__).with_name("inspect_original_775348_readonly.py")
SPEC = importlib.util.spec_from_file_location("trusted_original_inspection", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture():
    source = "SYNTHETIC_PUBLIC_SOURCE_ONLY"
    history = [{"iteration": i, "kernel_source": source+str(i), "case_runs": [
        {"validation_ok": i != 1} for _ in range(7)]} for i in range(4)]
    attempts = [NS(iteration=x["iteration"], kernel_source=x["kernel_source"],
        case_runs=[NS(**c) for c in x["case_runs"]]) for x in history]
    values = {"iteration": 3, "config": NS(max_iterations=3), "optimization_history": attempts,
        "best_attempt": attempts[2], "error": None,
        "messages": ["PRIVATE_MESSAGE_SENTINEL"], "RNG": "PRIVATE_RNG_SENTINEL",
        "workload": "PRIVATE_INPUT_SENTINEL", "unrelated": object()}
    snapshot = NS(values=values, next=("finalize_result",),
        tasks=(NS(name="finalize_result", error=None, interrupts=()),))
    return snapshot, history, module.hashlib.sha256(attempts[2].kernel_source.encode()).hexdigest()


def project(snapshot, history, expected):
    return module.project_snapshot(snapshot, history, expected, lambda state: "finish")


def test_projection_allows_only_finalization_and_emits_no_private_values():
    snapshot, history, expected = fixture()
    row = project(snapshot, history, expected)
    assert row["next_nodes"] == ["finalize_result"]
    assert row["phase"] == "evaluation_complete_before_finalize"
    assert row["permission_to_resume_or_relabel"] is False
    rendered = json.dumps(row)
    assert "PRIVATE_" not in rendered and "SYNTHETIC_PUBLIC_SOURCE_ONLY" not in rendered
    assert "messages" not in row and "RNG" not in row and "config" not in row


def test_projection_accepts_end_without_relabeling():
    snapshot, history, expected = fixture()
    snapshot.next, snapshot.tasks = (), ()
    row = project(snapshot, history, expected)
    assert row["phase"] == "graph_end" and row["permission_to_resume_or_relabel"] is False


@pytest.mark.parametrize("node", ["call_model", "evaluate_candidate", "propose", "tools", "UNKNOWN_PRIVATE_SENTINEL"])
def test_other_next_nodes_refused_without_echo(node):
    snapshot, history, expected = fixture()
    snapshot.next = (node,)
    with pytest.raises(module.Refused) as failure:
        project(snapshot, history, expected)
    assert node not in str(failure.value)


@pytest.mark.parametrize("change", ["iteration", "limit", "history", "source", "validity", "best", "state_error", "task_error", "interrupt", "missing_task", "extra_end_task"])
def test_changed_state_or_binding_refuses(change):
    snapshot, history, expected = fixture()
    if change == "iteration": snapshot.values["iteration"] = 2
    elif change == "limit": snapshot.values["config"].max_iterations = 4
    elif change == "history": snapshot.values["optimization_history"] = snapshot.values["optimization_history"][:-1]
    elif change == "source": snapshot.values["optimization_history"][3].kernel_source += "changed"
    elif change == "validity": snapshot.values["optimization_history"][3].case_runs[0].validation_ok = False
    elif change == "best": snapshot.values["best_attempt"] = snapshot.values["optimization_history"][0]
    elif change == "state_error": snapshot.values["error"] = "PRIVATE_ERROR_SENTINEL"
    elif change == "task_error": snapshot.tasks[0].error = "PRIVATE_TASK_ERROR_SENTINEL"
    elif change == "interrupt": snapshot.tasks[0].interrupts = ("PRIVATE_INTERRUPT_SENTINEL",)
    elif change == "missing_task": snapshot.tasks = ()
    elif change == "extra_end_task": snapshot.next = ()
    with pytest.raises(module.Refused) as failure:
        project(snapshot, history, expected)
    assert "PRIVATE_" not in str(failure.value)


def test_continuing_original_route_refused():
    snapshot, history, expected = fixture()
    with pytest.raises(module.Refused, match="original_route_would_continue"):
        module.project_snapshot(snapshot, history, expected, lambda state: "repeat")


def test_read_only_saver_reads_graph_state_without_node_or_sql_mutation(tmp_path):
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.sqlite import SqliteSaver
    path = tmp_path/"synthetic.sqlite"
    called = []
    def evaluate(state):
        called.append("evaluate")
        return {"iteration": 3, "private": "PRIVATE_TEST_FIXTURE"}
    def finalize(state):
        called.append("finalize")
        raise AssertionError("No finalization may execute in an inspection")
    builder = StateGraph(dict)
    builder.add_node("evaluate_candidate", evaluate)
    builder.add_node("finalize_result", finalize)
    builder.add_edge(START, "evaluate_candidate")
    builder.add_edge("evaluate_candidate", "finalize_result")
    builder.add_edge("finalize_result", END)
    config = {"configurable": {"thread_id": "synthetic"}}
    with SqliteSaver.from_conn_string(str(path)) as saver:
        graph = builder.compile(checkpointer=saver, interrupt_before=["finalize_result"])
        graph.invoke({"iteration": 2}, config)
    assert called == ["evaluate"]
    before = path.read_bytes()
    queries = []
    with sqlite3.connect(path.as_uri()+"?mode=ro&immutable=1", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.set_trace_callback(queries.append)
        saver = module.readonly_saver(connection)
        graph = builder.compile(checkpointer=saver)
        snapshot = graph.get_state(config)
        assert snapshot.next == ("finalize_result",)
        with pytest.raises(module.Refused, match="checkpoint_mutation_forbidden"):
            saver.put({}, {}, {}, {})
        with pytest.raises(module.Refused, match="checkpoint_mutation_forbidden"):
            saver.put_writes({}, [], "synthetic")
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM checkpoints")
    assert path.read_bytes() == before and called == ["evaluate"]
    assert not any(Path(str(path)+suffix).exists() for suffix in ("-wal", "-shm", "-journal"))
    assert not any(any(token in q.upper() for token in ("CREATE TABLE", "ALTER TABLE", "INSERT INTO", "UPDATE ")) for q in queries)


def test_existing_lock_never_creates_missing_lock(tmp_path):
    with pytest.raises(FileNotFoundError):
        with module.existing_lock(tmp_path/"missing", tmp_path):
            pass
    assert list(tmp_path.iterdir()) == []


def test_safe_evidence_refuses_symlink_and_hardlink(tmp_path):
    file = tmp_path/"public.json"
    file.write_text("{}")
    symlink = tmp_path/"link"
    symlink.symlink_to(file)
    with pytest.raises(module.Refused, match="aliased_evidence"):
        module.raw(symlink, tmp_path)
    hardlink = tmp_path/"hardlink"
    module.os.link(file, hardlink)
    with pytest.raises(module.Refused, match="unbounded_or_aliased_evidence"):
        module.raw(file, tmp_path)


def test_read_only_saver_refuses_missing_schema(tmp_path):
    with sqlite3.connect(":memory:") as connection:
        with pytest.raises(module.Refused, match="checkpoint_schema_missing"):
            module.readonly_saver(connection).setup()


def test_protocol_default_never_assesses(monkeypatch, capsys):
    monkeypatch.setattr(module.sys, "argv", [str(PATH)])
    def forbidden():
        raise AssertionError("Default protocol must never read a real checkpoint")
    monkeypatch.setattr(module, "assess", forbidden)
    assert module.main() == 0
    assert json.loads(capsys.readouterr().out)["checkpoint_opened"] is False


def synthetic_cli_result():
    return {"payload": {"kind": "final", "text": "PUBLIC_SYNTHETIC_REPLY", "tool_calls": []},
            "usage": {"input_tokens": None, "cost_usd": None},
            "events": [{"type": "turn.completed", "usage": {}}],
            "model": "gpt-6-luna", "reasoning_effort": "max", "call_id": "synthetic-call",
            "elapsed_seconds": 1.0}


def test_distinct_documented_cli_clock_samples_do_not_change_reply_identity():
    terminal = synthetic_cli_result()
    acknowledged = copy.deepcopy(terminal)
    acknowledged["elapsed_seconds"] = 1.001
    before = copy.deepcopy((acknowledged, terminal))
    row = module.compare_cli_result(acknowledged, terminal)
    assert row["ack_sample_not_earlier_than_cli"] is True
    assert row["cli_elapsed_sha256"] != row["ack_elapsed_sha256"]
    assert (acknowledged, terminal) == before


@pytest.mark.parametrize("field", ["payload", "usage", "events", "model", "reasoning_effort", "call_id"])
def test_any_scientific_or_model_identity_field_difference_still_refuses(field):
    terminal = synthetic_cli_result()
    acknowledged = copy.deepcopy(terminal)
    acknowledged[field] = "PRIVATE_CHANGED_FIELD_SENTINEL"
    with pytest.raises(module.Refused, match="original_cli_reply_identity_changed"):
        module.compare_cli_result(acknowledged, terminal)


@pytest.mark.parametrize("duration", [None, True, "1.1", -1.0, float("nan"), float("inf"), 0.999])
def test_bad_or_earlier_ack_clock_refuses(duration):
    terminal = synthetic_cli_result()
    acknowledged = copy.deepcopy(terminal)
    acknowledged["elapsed_seconds"] = duration
    with pytest.raises(module.Refused, match="original_cli_duration_semantics_changed"):
        module.compare_cli_result(acknowledged, terminal)


@pytest.mark.parametrize("target", ["ack", "terminal"])
def test_unknown_result_field_never_ignored(target):
    terminal, acknowledged = synthetic_cli_result(), synthetic_cli_result()
    (acknowledged if target == "ack" else terminal)["unexpected"] = "PRIVATE_EXTRA_SENTINEL"
    with pytest.raises(module.Refused, match="original_cli_result_fields_changed"):
        module.compare_cli_result(acknowledged, terminal)
