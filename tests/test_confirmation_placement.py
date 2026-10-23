"""Placement is audited separately from the exact UUID admission gate."""
from dataclasses import replace
import json
import pytest

from test_confirmation_recovery import interrupted, prepare
from optimizer_agent.suite_orchestrator import _local_session_job


def test_confirmed_recovery_uses_authenticated_previous_node_without_submission(interrupted):
    suite, task, _, _, config, *_, worker, actions = interrupted
    worker["nodes"] = "romeo-a041"
    prepare(interrupted)
    current = suite.journal.task(task["id"])
    suite.backend_factory = None
    backend = suite._backend(current, config, suite._directory(current))
    assert backend.config.nodelist == "romeo-a041"
    assert backend.config.gres == "gpu:h100:1"
    assert backend.job_id is None
    assert actions == ["authenticate_old_worker"]


def test_legacy_ack_absence_means_unpinned_and_never_adopts_new_pin(interrupted):
    suite, task, _, _, config, *_, worker, actions = interrupted
    backend = suite._backend(task, config, suite._directory(task))
    path = backend.results_dir / "slurm-session.json"
    historical = json.loads(path.read_bytes())
    historical["config"].pop("nodelist")
    path.write_text(json.dumps(historical))
    assert _local_session_job(task, backend) == "101"
    backend.config = replace(backend.config, nodelist="romeo-a041")
    with pytest.raises(ValueError, match="configuration ownership mismatch"):
        _local_session_job(task, backend)
