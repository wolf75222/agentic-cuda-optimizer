"""Future transport receipts only; no real Codex process or model is invoked."""
import hashlib
import json
import os
import signal
import subprocess
from copy import deepcopy
from unittest.mock import Mock

import pytest

import optimizer_agent.codex_backend as transport
from optimizer_agent.codex_backend import CodexBackend, CodexConfig, CodexError, schema_object


SCHEMA = schema_object({"ok": {"type": "boolean"}})
UNICODE_PROMPT = "Contexte éphémère\r\nλ → CUDA 🧪\n"


def completed_output():
    return "\n".join(map(json.dumps, [
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}},
        {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 3}},
    ]))


def saved_receipt(directory):
    paths = list(directory.glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def assert_prompt_identity(receipt, prompt=UNICODE_PROMPT):
    body = prompt.encode("utf-8", errors="strict")
    assert receipt["prompt_sha256"] == hashlib.sha256(body).hexdigest()
    assert receipt["prompt_utf8_bytes"] == len(body) > len(prompt)
    assert "prompt" not in receipt
    assert prompt not in json.dumps(receipt, ensure_ascii=False)


@pytest.mark.parametrize("streaming", [False, True])
def test_utf8_intent_is_durable_before_popen_and_pid_precedes_submission(tmp_path, monkeypatch, streaming):
    trace = tmp_path / "trace"
    backend = CodexBackend(CodexConfig(executable="fake-codex", trace_dir=trace))
    published = []
    save = backend._save_receipt

    def observe_save(call_id, receipt):
        save(call_id, receipt)
        published.append(deepcopy(receipt))

    monkeypatch.setattr(backend, "_save_receipt", observe_save)
    fsync = Mock(wraps=os.fsync)
    monkeypatch.setattr(transport.os, "fsync", fsync)

    class Process:
        pid = 73123
        returncode = 0

        def communicate(self, prompt, timeout):
            running = saved_receipt(trace)
            assert running["status"] == "running"
            assert running["cli_launch"]["child_pid"] == self.pid
            assert prompt == UNICODE_PROMPT and timeout == 600
            return completed_output(), ""

    def popen(command, **kwargs):
        prepared = saved_receipt(trace)
        assert prepared["status"] == "prepared"
        assert prepared["cli_launch"]["state"] == "prepared"
        assert "child_pid" not in prepared["cli_launch"]
        assert "started_at_unix" not in prepared["cli_launch"]
        assert "usage" not in prepared
        assert_prompt_identity(prepared)
        assert fsync.call_count >= 2  # File and directory persisted before launch.
        assert prepared["cli_launch"]["command_argv"] == command
        assert prepared["cli_launch"]["invoking_pid"] == os.getpid()
        assert kwargs["encoding"] == "utf-8" and kwargs["errors"] == "strict"
        assert kwargs["text"] is True and kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(transport.subprocess, "Popen", popen)
    if streaming:
        def communicate_stream(process, prompt, timeout, limit):
            assert limit == 1
            return process.communicate(prompt, timeout)
        monkeypatch.setattr(backend, "_communicate_search_limited", communicate_stream)
    result = backend.generate_json(UNICODE_PROMPT, SCHEMA, purpose="research" if streaming else "proposal",
                                   max_web_search_calls=1 if streaming else None)
    receipt = saved_receipt(trace)
    assert [row["status"] for row in published] == ["prepared", "running", "completed"]
    assert len({row["call_id"] for row in published}) == 1
    assert receipt["receipt_schema_version"] == 2
    assert_prompt_identity(receipt)
    assert receipt["cli_launch"]["state"] == "started"
    assert receipt["cli_launch"]["started_at_unix"] >= receipt["cli_launch"]["prepared_at_unix"]
    assert receipt["usage"]["input_tokens"] == 12 and receipt["usage"]["cost_usd"] is None
    assert result.payload == {"ok": True} and len(backend.calls) == 1


def test_launch_error_keeps_exact_prepared_identity_without_fictional_pid_or_usage(tmp_path, monkeypatch):
    backend = CodexBackend(CodexConfig(trace_dir=tmp_path))

    def fail_launch(*args, **kwargs):
        assert saved_receipt(tmp_path)["status"] == "prepared"
        raise FileNotFoundError("missing fake CLI")

    monkeypatch.setattr(transport.subprocess, "Popen", fail_launch)
    with pytest.raises(CodexError, match="missing fake CLI"):
        backend.generate_json(UNICODE_PROMPT, SCHEMA)
    receipt = saved_receipt(tmp_path)
    assert_prompt_identity(receipt)
    assert receipt["status"] == "failed" and receipt["cli_launch"]["state"] == "launch_failed"
    assert "child_pid" not in receipt["cli_launch"] and "usage" not in receipt
    assert "started_at_unix" not in receipt["cli_launch"] and backend.calls == []


@pytest.mark.parametrize("stdout,returncode,error", [
    ("[]", 0, "event must be an object"),
    ('{"type":"turn.failed","error":"provider unavailable"}', 0, "failed"),
    (completed_output(), 1, "exit 1"),
])
def test_completed_or_provider_parse_failure_retains_prompt_and_started_identity(tmp_path, monkeypatch,
                                                                              stdout, returncode, error):
    process = Mock(pid=73124, returncode=returncode)
    process.communicate.return_value = stdout, "diagnostic"
    monkeypatch.setattr(transport.subprocess, "Popen", Mock(return_value=process))
    backend = CodexBackend(CodexConfig(trace_dir=tmp_path))
    with pytest.raises(CodexError, match=error):
        backend.generate_json(UNICODE_PROMPT, SCHEMA)
    receipt = saved_receipt(tmp_path)
    assert_prompt_identity(receipt)
    assert receipt["status"] == "failed" and receipt["cli_launch"]["child_pid"] == process.pid
    assert receipt["returncode"] == returncode and "usage" not in receipt and backend.calls == []


def test_timeout_keeps_prompt_launch_identity_and_unknown_usage(tmp_path, monkeypatch):
    process = Mock(pid=73125, returncode=-signal.SIGKILL)
    process.communicate.side_effect = [subprocess.TimeoutExpired("fake", 600), ("partial JSONL", "partial stderr")]
    monkeypatch.setattr(transport.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(transport.os, "killpg", kill)
    backend = CodexBackend(CodexConfig(trace_dir=tmp_path))
    with pytest.raises(CodexError, match="timed out"):
        backend.generate_json(UNICODE_PROMPT, SCHEMA)
    kill.assert_called_once_with(process.pid, signal.SIGKILL)
    receipt = saved_receipt(tmp_path)
    assert_prompt_identity(receipt)
    assert receipt["status"] == "timeout" and receipt["cli_launch"]["state"] == "started"
    assert receipt["cli_launch"]["child_pid"] == process.pid
    assert receipt["stdout"] == "partial JSONL" and "usage" not in receipt
    assert backend.calls == []


def test_prepare_fsync_failure_never_launches_a_cli(tmp_path, monkeypatch):
    backend = CodexBackend(CodexConfig(trace_dir=tmp_path))
    monkeypatch.setattr(transport.os, "fsync", Mock(side_effect=OSError("durability unavailable")))
    popen = Mock()
    monkeypatch.setattr(transport.subprocess, "Popen", popen)
    with pytest.raises(CodexError, match="Cannot persist prepared.*durability unavailable"):
        backend.generate_json(UNICODE_PROMPT, SCHEMA)
    popen.assert_not_called()
    assert list(tmp_path.iterdir()) == [] and backend.calls == []


def test_running_receipt_failure_reaps_only_owned_child_without_submitting_prompt(tmp_path, monkeypatch):
    backend = CodexBackend(CodexConfig(trace_dir=tmp_path))
    save = backend._save_receipt

    def fail_running(call_id, receipt):
        if receipt["status"] == "running":
            raise OSError("running receipt publication failed")
        save(call_id, receipt)

    monkeypatch.setattr(backend, "_save_receipt", fail_running)
    process = Mock(pid=73126, returncode=-signal.SIGKILL)
    process.communicate.return_value = "child startup", ""
    monkeypatch.setattr(transport.subprocess, "Popen", Mock(return_value=process))
    kill = Mock()
    monkeypatch.setattr(transport.os, "killpg", kill)
    with pytest.raises(CodexError, match="running receipt publication failed"):
        backend.generate_json(UNICODE_PROMPT, SCHEMA)
    kill.assert_called_once_with(process.pid, signal.SIGKILL)
    process.communicate.assert_called_once_with()  # Never send the prompt on the failed publication.
    receipt = saved_receipt(tmp_path)
    assert_prompt_identity(receipt)
    assert receipt["status"] == "failed" and receipt["cli_launch"]["state"] == "started"
    assert receipt["cli_launch"]["reaped_after_receipt_publication_failure"] is True
    assert receipt["cli_launch"]["child_pid"] == process.pid and "usage" not in receipt
    assert backend.calls == []


def test_atomic_receipt_replace_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    backend = CodexBackend(CodexConfig(trace_dir=tmp_path))
    call_id = "a" * 32
    backend._save_receipt(call_id, {"status": "prepared"})
    previous = (tmp_path / (call_id + ".json")).read_bytes()
    monkeypatch.setattr(transport.os, "replace", Mock(side_effect=OSError("replacement unavailable")))
    with pytest.raises(OSError, match="replacement unavailable"):
        backend._save_receipt(call_id, {"status": "running"})
    assert (tmp_path / (call_id + ".json")).read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []
