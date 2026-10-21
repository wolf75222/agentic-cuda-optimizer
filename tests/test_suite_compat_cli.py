"""Legacy suite commands must not relabel the improved agent as its baselines."""
import json

import pytest

from optimizer_agent import campaign, suite_orchestrator, suite_report
from optimizer_agent.campaign_config import CampaignConfig


@pytest.mark.parametrize("command,starts", [("plan-suite", False), ("report-suite", False), ("suite", True)])
def test_compatibility_cli_dispatches_real_method_adapters(tmp_path, monkeypatch, command, starts):
    config = CampaignConfig(run_id="compatibility", results_dir=str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    calls = []

    class Orchestrator:
        def __init__(self, plan, directory, concurrency):
            calls.append(("plan", plan, concurrency))
        def run(self):
            calls.append(("run",))
            return {"summary": {"complete": False}}
        def status(self):
            calls.append(("status",))
            return {"summary": {"complete": False}}

    monkeypatch.setattr(suite_orchestrator, "SuiteOrchestrator", Orchestrator)
    monkeypatch.setattr(suite_report, "write_report", lambda *_: None)
    monkeypatch.setattr(campaign, "execute", lambda *_: pytest.fail("This would silently substitute the improved optimizer"))
    result = campaign.main([command, "--config", str(path), "--include-greedy", "--concurrency", "3"])
    assert result == (1 if starts else 0)
    plan = calls[0][1]
    assert len(plan["entries"]) == 126
    assert {"true_upstream", "native_codex_vanilla", "greedy_parent"} <= set(plan["modes"])
    assert calls[0][2] == 3
    assert calls[-1][0] == ("run" if starts else "status")
