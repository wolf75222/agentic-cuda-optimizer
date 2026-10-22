"""Replay completed local fixture receipts; no child, model, graph or GPU runs."""
import json
from pathlib import Path

import pytest

import optimizer_agent.codex_backend as transport_backend
from optimizer_agent.upstream_original import CHAT_SCHEMA, OriginalCodexTransport, OriginalTransportUncertain


def completed_before_ack(tmp_path, monkeypatch, usage, *, include_usage=True):
    directory = tmp_path / "original-model-transport"
    launches = []
    payload = {"kind": "final", "text": "one unchanged original reply", "tool_calls": []}
    class Process:
        returncode = 0
        pid = 12345
        def __init__(self, command, **kwargs):
            launches.append(command)
            self.output = Path(command[command.index("--output-last-message")+1])
        def communicate(self, prompt, timeout):
            assert prompt == "immutable original conversation"
            self.output.write_text(json.dumps(payload))
            completion = {"type": "turn.completed"}
            if include_usage:
                completion["usage"] = usage
            return json.dumps(completion), ""
    monkeypatch.setattr(transport_backend.subprocess, "Popen", Process)
    backend = transport_backend.CodexBackend(transport_backend.CodexConfig(executable="fixture-only-codex"))
    client = OriginalCodexTransport(backend, directory=directory)
    monkeypatch.setattr(client, "_ack", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt("before ACK")))
    with pytest.raises(KeyboardInterrupt, match="before ACK"):
        client._call("immutable original conversation", CHAT_SCHEMA)
    assert len(launches) == 1 and not (directory/"calls.json").exists()
    paths = list((directory/"intents").glob("*/*/cli/*.json"))
    assert len(paths) == 1
    path = paths[0]
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "completed" and receipt["returncode"] == 0
    return backend, directory, launches, path, payload


@pytest.mark.parametrize("usage,known", [
    (None, None), ({}, None), ({"input_tokens": 12}, 12),
    ({"output_tokens": 0}, 0), ({"input_tokens": True, "output_tokens": 3}, 3),
    ({"input_tokens": "12", "output_tokens": 3}, 3),
    ({"input_tokens": -1, "output_tokens": 3}, 3),
])
def test_completed_reply_with_unknown_counters_is_recovered_once_without_a_new_cli(tmp_path, monkeypatch, usage, known):
    backend, directory, launches, path, payload = completed_before_ack(tmp_path, monkeypatch, usage)
    original_receipt = path.read_bytes()
    resumed = OriginalCodexTransport(backend, directory=directory)
    result = resumed._call("immutable original conversation", CHAT_SCHEMA)
    assert result.payload == payload and result.call_id == path.stem
    assert path.read_bytes() == original_receipt and len(launches) == 1
    observed = resumed.usage()
    assert observed["model_calls"] == 1 and observed["model_tokens"] is None
    assert observed["known_model_tokens"] == known and observed["usage_complete"] is False
    assert not observed["uncertain_model_intents"]
    assert resumed._call("immutable original conversation", CHAT_SCHEMA).call_id == result.call_id
    assert len(launches) == 1 and len(json.loads((directory/"calls.json").read_text())) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_missing_usage_envelope_and_legacy_receipt_remain_replayable_and_unknown(tmp_path, monkeypatch, legacy):
    backend, directory, launches, path, payload = completed_before_ack(tmp_path, monkeypatch, None, include_usage=False)
    if legacy:
        receipt = json.loads(path.read_text())
        receipt.pop("provider_usage_raw")
        path.write_text(json.dumps(receipt))
    resumed = OriginalCodexTransport(backend, directory=directory)
    assert resumed._call("immutable original conversation", CHAT_SCHEMA).payload == payload
    assert resumed.usage()["model_tokens"] is None and len(launches) == 1


def test_actual_zero_and_provider_price_normalization_match_backend_without_inventing_billing(tmp_path, monkeypatch):
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 123}
    backend, directory, launches, path, payload = completed_before_ack(tmp_path, monkeypatch, usage)
    resumed = OriginalCodexTransport(backend, directory=directory)
    result = resumed._call("immutable original conversation", CHAT_SCHEMA)
    assert result.usage["cost_usd"] is None and result.payload == payload
    assert resumed.usage()["model_tokens"] == 0 and resumed.usage()["usage_complete"] is True
    assert len(launches) == 1


@pytest.mark.parametrize("change", ["usage", "raw", "boolean_vs_integer", "turn_nonobject", "model", "purpose", "payload", "status", "events"])
def test_unknown_usage_does_not_relax_receipt_identity_or_completion_guards(tmp_path, monkeypatch, change):
    backend, directory, launches, path, _ = completed_before_ack(tmp_path, monkeypatch, {"input_tokens": True, "output_tokens": 3})
    receipt = json.loads(path.read_text())
    if change == "usage":
        receipt["usage"]["output_tokens"] = 4
    elif change == "raw":
        receipt["provider_usage_raw"] = {"input_tokens": True, "output_tokens": 4}
    elif change == "boolean_vs_integer":
        receipt["usage"]["input_tokens"] = 1
    elif change == "turn_nonobject":
        receipt["events"][-1]["usage"] = True
    elif change == "model":
        receipt["model"] = "different-model"
    elif change == "purpose":
        receipt["purpose"] = "research"
    elif change == "payload":
        receipt["payload"] = {"unrelated": "reply"}
    elif change == "status":
        receipt["status"] = "failed"
    else:
        receipt["events"] = []
    path.write_text(json.dumps(receipt))
    resumed = OriginalCodexTransport(backend, directory=directory)
    with pytest.raises(OriginalTransportUncertain):
        resumed._call("immutable original conversation", CHAT_SCHEMA)
    assert len(launches) == 1 and not (directory/"calls.json").exists()
