"""Helper safety/control tests; actual ROMEO evidence is produced only by --run."""
from dataclasses import asdict
from types import SimpleNamespace
import json
import pytest

from scripts import diagnostic_loop_acceptance as acceptance
from tests.test_campaign_runtime import runtime_setup, selected


def test_prepare_is_service_free_keeps_configured_partition_and_excludes_quality_cohorts(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Preparing a plan must not contact SSH, CUDA or Codex")
    monkeypatch.setattr(acceptance,"SlurmBackend",forbidden)
    monkeypatch.setattr(acceptance,"CampaignRuntime",forbidden)
    config,slurm,plan = acceptance.prepare(tmp_path,"diagnostic-plan",acceptance.REPO/"configs/romeo.json")
    configured = acceptance.SlurmConfig.load(acceptance.REPO/"configs/romeo.json")
    assert slurm.partition == configured.partition == "short"
    assert slurm.walltime == configured.walltime == "24:00:00"
    assert config.model == "gpt-6-luna" and config.reasoning_effort == "max"
    assert config.max_trials == 4 and config.actions == ("tiling",)
    assert plan["global_resource_caps"] is None and plan["holdout_exposure"] is False
    assert "excluded" in plan["scope"] and (tmp_path/"plan.json").is_file()


def test_prepare_rejects_scheduler_walltime_above_24_hours(tmp_path):
    config = asdict(acceptance.SlurmConfig.load(acceptance.REPO/"configs/romeo.json"))
    config["walltime"] = "25:00:00"
    path = tmp_path/"too-long.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="at most 24 hours"):
        acceptance.prepare(tmp_path/"plan","too-long",path)


def test_control_is_exact_parent_body_with_distinct_identity_and_no_fake_citations(runtime_setup):
    runtime,_,_,_ = runtime_setup
    state = selected(runtime)
    candidate = acceptance.install_control(runtime,state,1,SimpleNamespace(research_id="fixture-dossier"))
    source = candidate.source.read_text()
    assert source.split("\n"+acceptance.CONTROL_MARKER)[0] == runtime.trial(state)["parent"]["source"].rstrip()
    assert candidate.candidate_id != "baseline"
    proposal = acceptance.read_json(runtime.trial_dir(state)/"proposal-0-0.json")
    assert proposal["references"] == [] and proposal["techniques"] == []
    assert proposal["acceptance_control"] is True
    assert "static baseline-equivalent" in proposal["generation_mode"]
    archived = runtime.store.get_candidate_source(candidate.candidate_id,runtime.config.run_id)
    assert archived["source_parent_id"] == runtime.trial(state)["parent"]["candidate_id"]
    assert candidate.launch_rule == acceptance.LaunchRule(**runtime.trial(state)["parent"]["launch_rule"])
    state.update(repair=0,research_round=0)
    class Dossier:
        def research_ready(self,*args):
            return True
    with pytest.raises(RuntimeError,match="static acceptance control"):
        acceptance.verify_real_proposal(runtime,state,Dossier())


def test_startup_failure_preserves_incomplete_status_and_closes_only_own_session(tmp_path,monkeypatch):
    owned = []
    class Backend:
        def __init__(self,config,repo,directory,*,session_id):
            self.job_id = None
            self.config, self.session_id = config, session_id
            self.remote_dir = config.remote_root + "/sessions/" + session_id
            assert session_id.startswith("diagnostic-acceptance-")
        def start(self):
            self.job_id = "owned-job"
            return self.job_id
        def wait_until_ready(self):
            raise RuntimeError("fixture startup failure")
        def close(self,*,cancel=False):
            owned.append((self.job_id,cancel))
        def wait_until_finished(self,timeout):
            return {"job_id":self.job_id,"terminal":True,"state":"COMPLETED","exit_code":"0:0"}
    monkeypatch.setattr(acceptance,"SlurmBackend",Backend)
    receipt = acceptance.run_acceptance(tmp_path,"startup-failure",acceptance.REPO/"configs/romeo.json")
    assert receipt["ok"] is False and receipt["status"] == "incomplete"
    assert "fixture startup failure" in receipt["error"]
    assert owned == [("owned-job",False)]
    assert (tmp_path/"diagnostic-loop-acceptance.json").is_file()
