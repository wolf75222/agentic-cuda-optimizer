"""Helper safety/control tests; actual ROMEO evidence is produced only by --run."""
from dataclasses import asdict
from types import SimpleNamespace
import hashlib
import json
import pytest

from scripts import diagnostic_loop_acceptance as acceptance
from tests.test_campaign_runtime import runtime_setup, selected
from tests.test_codex_research import proposal_payload


def test_prepare_is_service_free_keeps_configured_partition_and_excludes_quality_cohorts(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Preparing a plan must not contact SSH, CUDA or Codex")
    monkeypatch.setattr(acceptance,"SlurmBackend",forbidden)
    monkeypatch.setattr(acceptance,"CampaignRuntime",forbidden)
    config,slurm,plan = acceptance.prepare(tmp_path,"diagnostic-plan",acceptance.REPO/"configs/romeo.json")
    configured = acceptance.SlurmConfig.load(acceptance.REPO/"configs/romeo.json")
    assert slurm.partition == configured.partition == "long"
    assert slurm.walltime == configured.walltime == "48:00:00"
    assert config.model == "gpt-6-luna" and config.reasoning_effort == "max"
    assert config.max_trials == 4 and config.actions == ("tiling",)
    assert plan["global_resource_caps"] is None and plan["holdout_exposure"] is False
    assert "excluded" in plan["scope"] and (tmp_path/"plan.json").is_file()


def test_prepare_preserves_long_partition_walltime_without_an_extra_helper_ceiling(tmp_path):
    config = asdict(acceptance.SlurmConfig.load(acceptance.REPO/"configs/romeo.json"))
    config["walltime"] = "72:00:00"
    path = tmp_path/"long.json"
    path.write_text(json.dumps(config))
    _,slurm,plan = acceptance.prepare(tmp_path/"plan","long-lease",path)
    assert slurm.walltime == plan["slurm"]["walltime"] == "72:00:00"
    assert plan["global_resource_caps"] is None


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


class CPURepairDossier:
    """Declared documentary service fixture, never a claimed live research run."""
    def __init__(self, value):
        self.value = value
        self.research_id = value["research_id"]
        self.queries = ["CPU fixture query"]
        self.diagnostic_ids = value.get("diagnostic_ids", [])
    @classmethod
    def from_dict(cls, value):
        return cls(value)
    def to_dict(self):
        return self.value
    def research_ready(self, contract, family, environment):
        assert (contract, family, environment) == ("fixture-contract", "tiling", "fixture-gpu")
    def validate_proposal(self, proposal):
        assert proposal["references"] == ["fixture-code"]


class CPURepairRuntime:
    """Use the real LangGraph routing with explicit non-CUDA service fixtures."""
    max_repairs = 2
    contract_hash = "fixture-contract"
    environment_hash = "fixture-gpu"

    def __init__(self, directory, *, success_at):
        self.directory = directory
        directory.mkdir(parents=True)
        (directory/"candidates").mkdir()
        self.config = SimpleNamespace(run_id="cpu-repairs", experimental_mode=None)
        self.success_at = success_at
        self.states, self.generated, self.evaluated, self.integrated = {}, [], [], []
        self.store = SimpleNamespace(archive_candidate=lambda run_id, value: self.states.update({value["candidate_id"]:value}))
        self.policy = SimpleNamespace(snapshot=lambda: {"tree":SimpleNamespace(tree=[None]*(len(self.integrated)+1))})
        self.parent = {"candidate_id":"baseline", "source":'extern "C" __global__ void matmul() {}\n',
            "kernel":"matmul", "launch_rule":asdict(acceptance.LaunchRule(
                kind="square_2d",block=(16,16,1),tile_x=16,tile_y=16)),
            "parameters":{}}
    def initialize(self, state):
        return {}
    def select(self, state):
        trial_id = "trial-"+str(state["trials_done"]+1)
        directory = self.directory/"trials"/trial_id
        directory.mkdir(parents=True)
        acceptance.write_json(directory/"trial.json", {"parent":self.parent,"action":"tiling"})
        return {"trial_id":trial_id,"repair":0,"research_round":0,"candidate_id":None,
                "valid":False,"retryable":False,"error":None,"needs_research":False}
    def trial_dir(self, state):
        return self.directory/"trials"/state["trial_id"]
    def trial(self, state):
        return acceptance.read_json(self.trial_dir(state)/"trial.json")
    def research(self, state):
        value = {"research_id":"targeted" if state["trials_done"]==3 else "initial",
                 "diagnostic_ids":["fixture-stagnation"] if state["trials_done"]==3 else []}
        acceptance.write_json(self.trial_dir(state)/"dossier.json", value)
        return {"research_ready":True,"research_round":1}
    def candidate_state(self, candidate, **extra):
        return acceptance.CampaignRuntime.candidate_state(self, candidate, **extra)
    def candidate(self, candidate_id):
        value = self.states[candidate_id]
        path = self.directory/"candidates"/(candidate_id+".cu")
        assert path.read_text() == value["source"]
        return acceptance.Candidate(candidate_id,path,value["kernel"],
                                    acceptance.LaunchRule(**value["launch_rule"]),value["parameters"])
    def generate(self, state):
        repair = state["repair"]
        previous = state["candidate_id"] if repair else "baseline"
        candidate_id = "generated-"+str(repair)
        source = self.parent["source"]+f"// independently supplied CPU fixture proposal {repair}\n"
        payload = proposal_payload()
        payload.update(parent_id="baseline",source_parent_id=previous,contract_hash=self.contract_hash,
            references=["fixture-code"],source=source,
            launch_rule=json.loads(json.dumps(self.parent["launch_rule"])),parameters={})
        proposal = asdict(acceptance.Proposal.from_dict(payload,parent_id="baseline",source_parent_id=previous,
                            contract_hash=self.contract_hash,family="tiling"))
        proposal["call_id"] = "fixture-call-"+str(repair)
        acceptance.write_json(self.trial_dir(state)/f'proposal-{repair}-{state["research_round"]}.json', proposal)
        path = self.directory/"candidates"/(candidate_id+".cu")
        path.write_text(source)
        candidate = acceptance.Candidate(candidate_id,path,"matmul",acceptance.LaunchRule(**proposal["launch_rule"]))
        self.states[candidate_id] = self.candidate_state(candidate,source_parent_id=previous)
        self.generated.append({"repair":repair,"previous_error":state.get("error"),"source_parent_id":previous})
        return {"candidate_id":candidate_id,"error":None,"needs_research":False}
    def evaluate(self, state):
        candidate_id = state["candidate_id"]
        control = candidate_id.startswith("diagnostic-control-")
        valid = control or state["repair"] >= self.success_at
        receipt = {"evaluation_id":"evaluation-"+candidate_id,"candidate_id":candidate_id,
            "ok":valid,"status":"measured" if valid else "compilation_error",
            "cases":{"fixture":{"latency_ms":1.0}} if valid else [{"case_id":"fixture","report":{
                "stage":"compile","ok":False,"error":"CPU fixture compiler error; preserve the complete receipt"}}]}
        acceptance.write_json(self.trial_dir(state)/f'evaluation-{candidate_id}.json',receipt)
        self.evaluated.append(candidate_id)
        return {"valid":valid,"retryable":not valid,"error":None if valid else json.dumps(receipt)}
    def integrate(self, state):
        self.integrated.append(state["candidate_id"])
        acceptance.write_json(self.trial_dir(state)/"confirmation.json",{
            "evaluation_id":"confirmation-"+state["candidate_id"],"status":"not_improved","ok":True})
        return {"trials_done":state["trials_done"]+1}
    def diagnose(self, state):
        return {"diagnostic_id":"fixture-stagnation"}
    def diagnostic_context(self, state):
        return {"diagnostic_id":"fixture-stagnation","triggers":["stagnation"],"evidence_references":[],
                "observations":[{"kind":"stagnation","evaluation_ids":["evaluation-"+i for i in self.integrated]}]}


def test_fourth_trial_real_langgraph_repairs_compile_failure_then_integrates_exact_final_candidate(tmp_path,monkeypatch):
    monkeypatch.setattr(acceptance,"ResearchDossier",CPURepairDossier)
    runtime = CPURepairRuntime(tmp_path/"cpu-run",success_at=1)
    result = acceptance.execute_on_runner(runtime)
    assert all(result["checks"].values())
    assert [row["repair"] for row in runtime.generated] == [0,1]
    assert runtime.generated[1]["source_parent_id"] == "generated-0"
    assert "compilation_error" in runtime.generated[1]["previous_error"]
    assert result["proposal_call_id"] == "fixture-call-1" and result["candidate_id"] == "generated-1"
    assert runtime.integrated[-1] == "generated-1" and "generated-0" not in runtime.integrated
    attempts = result["fourth_attempts"]
    assert [row["status"] for row in attempts] == ["compilation_error","measured"]
    for row in attempts:
        for field in ("proposal_reference","evaluation_reference"):
            reference = row[field]
            assert hashlib.sha256((runtime.directory/reference["path"]).read_bytes()).hexdigest() == reference["sha256"]
    archived = acceptance.read_json(runtime.directory/"trials/trial-4/acceptance-experiment.json")
    assert archived["status"] == "completed" and [e["node"] for e in archived["events"]] == [
        "generate","evaluate","repair","generate","evaluate"]


def test_fourth_trial_exhausted_repairs_remain_failed_and_preserve_all_compiler_receipts(tmp_path,monkeypatch):
    monkeypatch.setattr(acceptance,"ResearchDossier",CPURepairDossier)
    runtime = CPURepairRuntime(tmp_path/"cpu-failed",success_at=3)
    with pytest.raises(RuntimeError,match="failed independent CUDA evaluation after configured repairs"):
        acceptance.execute_on_runner(runtime)
    assert [row["repair"] for row in runtime.generated] == [0,1,2]
    assert [row["source_parent_id"] for row in runtime.generated] == ["baseline","generated-0","generated-1"]
    assert len(runtime.integrated) == 3  # Static controls only; no failed representative is accepted.
    archived = acceptance.read_json(runtime.directory/"trials/trial-4/acceptance-experiment.json")
    assert archived["status"] == "failed" and archived["max_repairs"] == 2
    assert len(archived["attempts"]) == 3
    for row in archived["attempts"]:
        reference = row["evaluation_reference"]
        path = runtime.directory/reference["path"]
        assert row["status"] == "compilation_error" and row["valid"] is False
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
        assert "complete receipt" in acceptance.read_json(path)["cases"][0]["report"]["error"]


def test_final_proposal_identity_cannot_point_to_the_first_failed_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(acceptance,"ResearchDossier",CPURepairDossier)
    runtime = CPURepairRuntime(tmp_path/"cpu-identity",success_at=1)
    state = {"run_id":runtime.config.run_id,"trials_done":3}
    state.update(runtime.select(state));state.update(runtime.research(state))
    state,attempts = acceptance.run_guarded_experiment(runtime,state)
    with pytest.raises(RuntimeError,match="exact experiment attempt receipt"):
        acceptance.verify_real_proposal(runtime,state,CPURepairDossier({"research_id":"targeted"}),
            source_parent_id="generated-0",proposal_reference=attempts[0]["proposal_reference"])
    final_path = runtime.trial_dir(state)/'proposal-1-1.json'
    altered = acceptance.read_json(final_path)
    altered["source"] += "// altered after evaluation\n"
    acceptance.write_json(final_path,altered)
    with pytest.raises(RuntimeError,match="exact candidate independently evaluated"):
        acceptance.verify_real_proposal(runtime,state,CPURepairDossier({"research_id":"targeted"}),
            source_parent_id="generated-0")
