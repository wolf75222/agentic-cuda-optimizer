"""Unknown counters remain distinct from a malformed protocol envelope."""
import json

import pytest

import optimizer_agent.codex_backend as transport
from optimizer_agent.codex_backend import CodexBackend, CodexConfig, schema_object


@pytest.mark.parametrize("raw_usage", [None, True, False, 7, "malformed", [1, 2], [["input_tokens", 9]]])
def test_unknown_usage_or_malformed_envelope_has_an_explicit_terminal_receipt(tmp_path, monkeypatch, raw_usage):
    class Process:
        pid = 73124
        returncode = 0

        def communicate(self, prompt, timeout):
            return "\n".join(json.dumps(row) for row in [
                {"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}},
                {"type": "turn.completed", "usage": raw_usage},
            ]), ""

    monkeypatch.setattr(transport.subprocess, "Popen", lambda *args, **kwargs: Process())
    backend = CodexBackend(CodexConfig(executable="fake-codex", trace_dir=tmp_path))
    if raw_usage is not None:
        with pytest.raises(transport.CodexError, match="usage must be an object"):
            backend.generate_json("synthetic prompt", schema_object({"ok": {"type": "boolean"}}))
    else:
        result = backend.generate_json("synthetic prompt", schema_object({"ok": {"type": "boolean"}}))
        assert result.payload == {"ok": True}
        assert result.usage == {"cost_usd": None}
    receipts = list(tmp_path.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    if raw_usage is not None:
        assert receipt["status"] == "failed"
        assert "usage must be an object" in receipt["error"]
        assert "usage" not in receipt and backend.calls == []
        return
    assert receipt["status"] == "completed"
    assert receipt["provider_usage_raw"] == raw_usage
    assert receipt["usage_observation"]["model_tokens"] is None
    assert receipt["usage_observation"]["usage_complete"] is False
    assert receipt["events"][-1]["usage"] == raw_usage
