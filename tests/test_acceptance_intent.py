import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from execution_backend import SlurmConfig
from scripts.acceptance_intent import persist_allocation_intent
from scripts import gpu_rpc_acceptance, gpu_contract_acceptance, gpu_autotuning_acceptance, diagnostic_loop_acceptance


def test_intent_is_exclusive_and_preserves_first_exact_ownership(tmp_path):
    backend = SimpleNamespace(session_id="owned-first", remote_dir="/isolated/sessions/owned-first", config=SlurmConfig())
    first = persist_allocation_intent(tmp_path, backend, {"HEAD": "exact-source"})
    original = (tmp_path / "allocation-intent.json").read_bytes()
    backend.session_id = "must-not-replace"
    with pytest.raises(ValueError, match="reconcile"):
        persist_allocation_intent(tmp_path, backend, {"HEAD": "different"})
    assert (tmp_path / "allocation-intent.json").read_bytes() == original
    assert first["session_id"] == "owned-first" and first["source_provenance"]["HEAD"] == "exact-source"


@pytest.mark.parametrize("module,receipt_name", [
    (gpu_rpc_acceptance, "allocation-receipt.json"),
    (gpu_contract_acceptance, "allocation-receipt.json"),
    (gpu_autotuning_acceptance, "autotuning-acceptance.json"),
    (diagnostic_loop_acceptance, "diagnostic-loop-acceptance.json"),
])
def test_lost_submission_ack_retains_intent_and_forbids_new_submission(tmp_path, monkeypatch, module, receipt_name):
    instances = []
    class Backend:
        def __init__(self, config, repo, directory, *, session_id):
            self.config, self.session_id = config, session_id
            self.remote_dir = config.remote_root + "/sessions/" + session_id
            self.job_id = None
            instances.append(self)
        def start(self):
            intent = json.loads((tmp_path / "allocation-intent.json").read_text())
            assert intent["session_id"] == self.session_id and intent["remote_dir"] == self.remote_dir
            assert intent["config"]["host"] == self.config.host
            assert intent["source_provenance"]["HEAD"] and intent["source_provenance"]["sources"]
            raise RuntimeError("fixture lost submission acknowledgement")
        def close(self, **kwargs):
            pytest.fail("Unknown job identity must never be cancelled")
    monkeypatch.setattr(module, "SlurmBackend", Backend)
    command = ["--run", "--output", str(tmp_path), "--run-id", "lost-ack-proof"]
    assert module.main(command) == 1
    receipt = json.loads((tmp_path / receipt_name).read_text())
    original = (tmp_path / "allocation-intent.json").read_bytes()
    assert receipt["job_id"] is None and receipt["session_id"] == instances[0].session_id
    assert receipt["remote_dir"] == instances[0].remote_dir
    assert receipt["allocation_intent"]["source_provenance"]["HEAD"]
    with pytest.raises(ValueError, match="fresh|intent|allocation"):
        module.main(command)
    assert len(instances) == 1 and (tmp_path / "allocation-intent.json").read_bytes() == original
