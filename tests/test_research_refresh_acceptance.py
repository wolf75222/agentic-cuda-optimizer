"""CPU fixtures only: no model, network, SSH or GPU qualification is claimed."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer_agent import campaign
from optimizer_agent.codex_backend import CodexBackend, CodexConfig, CodexResult
from scripts import run_research_refresh_acceptance as acceptance
from tests.test_campaign_runtime import runtime_setup
from tests import test_codex_research as research_fixtures


def bootstrap_fixture(directory):
    """Declared invented CPU receipt for structural guards, never live evidence."""
    config = campaign.CampaignConfig(run_id="explicit-cpu-fixture", search_sizes=(8,), final_sizes=(11,),
        benchmark_series=1, launches_per_series=1, required_sanitizers=(), profile=False)
    props = {"name":"explicit CPU fixture", "uuid":"fixture-archived-gpu", "architecture":"sm_90",
             "driver_version":12060, "nvrtc_version":"12.6", "cuda_toolkit_version":12060,
             "harness_build_id":"explicit-fixture-build"}
    env = campaign.device_identity(props)
    suite = acceptance.WorkloadSuite(config.workload,directory/"generation/evaluator_data",config)
    candidate = suite.baseline(acceptance.REPO/"examples/naive_matmul.cu")
    source = candidate.source
    cases = suite.cases()
    semantics = acceptance.operation_spec(config.workload)
    contract = acceptance.WorkloadContract(config.run_id,1,config.workload,suite.signature,
        "Explicit CPU fixture oracle, no real GPU measurement",benchmark_series=1,launches_per_series=1,
        allowed_precision="float32",metadata_json=json.dumps({"precision":"float32","environment":env,"memory_policy":semantics["memory_policy"],
                                                              "layout":semantics["layout"]}))
    evaluator = acceptance.Evaluator(acceptance.ArchivedPropertiesOnlyRunner(props),contract,suite.oracle,
        {"short":cases[:2],"search":cases,"performance":[c for c in cases if c.metadata["n"] in config.search_sizes]},
        final_suite_factory=lambda: (),final_suite_identity=campaign.digest({"generator":"benchmark_workloads-v1",
            "sizes":config.final_sizes,"seed_commitment":campaign.digest(config.seed+1000003)}))
    contract_hash = evaluator.contract_hash
    native = {"ok":True,"slurm_job_id":"12345","environment":props,"latency_samples_ms":[1.0],
              "guard_readonly_ok":True,"guard_outputs_poisoned":True,
              "compilation_identity":{"source":source.read_text()}}
    baseline = {"evaluation_id":"explicit-fixture-measurement","candidate_id":"baseline",
                "artifact_hash":candidate.artifact_hash,"contract_hash":contract_hash,"ok":True,"status":"measured",
                "environment":props,"cases":{c.case_id:{"latency_ms":1.0,"weight":c.weight,"critical":c.critical,
                    "series":[{"mean_ms":1.0,"raw_ms":[1.0],"report":native}]}
                    for c in cases if c.metadata["n"] in config.search_sizes}}
    terminal = {"job_id":"12345","terminal":True,"state":"COMPLETED","exit_code":"0:0"}
    result = {"run_id":config.run_id,"config":config.to_dict(),"execution_complete":True,"generation_complete":True,
              "worker_terminal":terminal,"job_id":"12345","session_id":"explicit-fixture-session",
              "environment":env,"contract_hash":contract_hash,"baseline_latencies_ms":campaign.CampaignRuntime.latencies(baseline)}
    manifest = {"run_id":config.run_id,"config":config.to_dict(),"certification_status":"generation_complete",
                "environment":env,"contract_hash":contract_hash,"baseline_latencies_ms":campaign.CampaignRuntime.latencies(baseline)}
    for name,value in {"result.json":result,"generation/manifest.json":manifest,
        "generation/config.json":config.to_dict(),"generation/environment.json":env,
        "generation/contract.json":{"contract":asdict(contract),"contract_hash":contract_hash},
        "generation/baseline-measurement.json":baseline,
        "worker/hardware.json":{"slurm_job_id":"12345","device_properties":props},
        "worker/slurm-session.json":{"job_id":"12345","session_id":"explicit-fixture-session"}}.items():
        campaign.write_json(directory/name,value)
    return directory


def test_default_plan_contacts_no_services(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("Plan cannot initialize a service")
    monkeypatch.setattr(acceptance,"CampaignRuntime",forbidden)
    monkeypatch.setattr(acceptance,"CodexBackend",forbidden)
    assert acceptance.main(["--output",str(tmp_path/"plan"),"--run-id","cpu-plan"]) == 0
    plan = campaign.read_json(tmp_path/"plan/plan.json")
    assert plan["submitted"] is False and plan["model_invoked"] is False and plan["new_gpu_requests"] == 0
    assert not (tmp_path/"plan/run-intent.json").exists()


def test_archived_runner_preserves_actual_properties_and_rejects_gpu():
    props = {"uuid":"archived-identity","driver_version":12060}
    runner = acceptance.ArchivedPropertiesOnlyRunner(props)
    assert runner.get_device_properties(0) == props
    with pytest.raises(RuntimeError,match="forbids every native"):
        runner.run("not-executed",None)
    with pytest.raises(RuntimeError,match="Unexpected GPU"):
        runner.require_complete()


def test_bootstrap_requires_completed_consistent_native_artifacts_and_pinned_hash(tmp_path):
    origin = bootstrap_fixture(tmp_path/"origin")
    before = {name:acceptance.sha(origin/name) for name in acceptance.BOOTSTRAP_FILES}
    inspected = acceptance.inspect_bootstrap(origin,before["result.json"])
    assert {name:inspected["hashes"][name] for name in acceptance.BOOTSTRAP_FILES} == before
    assert inspected["authenticated_search_input_hashes"] and inspected["job_id"] == "12345"
    assert inspected["authentication"]["explicit_result_pin_checked"] is True
    assert inspected["authentication"]["recomputed_evaluator_contract_hash"] == inspected["contract"]["contract_hash"]
    with pytest.raises(ValueError,match="explicitly pinned"):
        acceptance.inspect_bootstrap(origin,"0"*64)
    result = campaign.read_json(origin/"result.json")
    result["worker_terminal"]["state"] = "RUNNING"
    campaign.write_json(origin/"result.json",result)
    with pytest.raises(ValueError,match="completed generation"):
        acceptance.inspect_bootstrap(origin)


@pytest.mark.parametrize("defect",["hardware","native_source","aggregate_only","config","artifact","suite_input",
    "contract_content","missing_case","raw_timing","readonly","missing_artifact","series_mean","case_aggregate","result_aggregate"])
def test_bootstrap_rejects_inconsistent_context_before_any_model(tmp_path,defect):
    origin = bootstrap_fixture(tmp_path/"origin")
    path = origin/"generation/baseline-measurement.json"
    data = campaign.read_json(path)
    case = next(iter(data["cases"].values()))
    if defect == "hardware":data["environment"]["uuid"] = "different"
    elif defect == "native_source":case["series"][0]["report"]["compilation_identity"]["source"] += "// altered"
    elif defect == "aggregate_only":
        for item in data["cases"].values():item["series"] = []
    elif defect == "artifact":data["artifact_hash"] = "different"
    elif defect == "missing_case":data["cases"].pop(next(iter(data["cases"])))
    elif defect == "raw_timing":case["series"][0]["raw_ms"] = [2.0]
    elif defect == "series_mean":case["series"][0]["mean_ms"] = 2.0
    elif defect == "case_aggregate":case["latency_ms"] = 2.0
    elif defect == "result_aggregate":
        path = origin/"result.json";data = campaign.read_json(path)
        data["baseline_latencies_ms"][next(iter(data["baseline_latencies_ms"]))] = 2.0
    elif defect == "readonly":case["series"][0]["report"]["guard_readonly_ok"] = False
    elif defect == "missing_artifact":
        (origin/"worker/hardware.json").unlink()
    elif defect == "config":
        data = campaign.read_json(origin/"generation/config.json")
        data["run_id"] = "changed"
        path = origin/"generation/config.json"
    elif defect == "suite_input":
        next((origin/"generation/evaluator_data").rglob("*.bin")).write_bytes(b"changed public search input")
    else:
        path = origin/"generation/contract.json"
        data = campaign.read_json(path)
        data["contract"]["oracle_provenance"] = "changed identity"
    campaign.write_json(path,data)
    with pytest.raises(ValueError):acceptance.inspect_bootstrap(origin)


def test_documentary_adapter_recomputes_original_contract_keeps_guards_and_never_charges_gpu(tmp_path):
    source = bootstrap_fixture(tmp_path/"origin")
    origin = acceptance.inspect_bootstrap(source,acceptance.sha(source/"result.json"))
    original_hashes = dict(origin["hashes"])
    runner = acceptance.ArchivedPropertiesOnlyRunner(origin["properties"])
    def forbidden(*args,**kwargs):
        raise AssertionError("Adapter bootstrap must not call a model")
    backend = SimpleNamespace(calls=[],generate_json=forbidden,config=SimpleNamespace(trace_dir=tmp_path/"copy/model"))
    config = replace(origin["config"],max_trials=1,actions=("tiling",),profile=False)
    runtime = acceptance.DocumentaryRuntime(config,runner,tmp_path/"copy",model_backend=backend,origin=origin)
    assert runtime.contract_hash == origin["contract"]["contract_hash"]
    assert runtime.baseline_measurement == origin["baseline"] and runner.gpu_attempts == []
    assert runtime.store.budget_state(config.run_id)["spent"].get("gpu_seconds",0) == 0
    assert acceptance.DocumentaryRuntime.research is campaign.CampaignRuntime.research
    assert acceptance.DocumentaryRuntime.generate is campaign.CampaignRuntime.generate
    assert acceptance.DocumentaryRuntime.diagnose is campaign.CampaignRuntime.diagnose
    for phase in (runtime.evaluate,runtime.integrate,runtime.finalize):
        with pytest.raises(RuntimeError,match="Documentary adapter forbids"):
            phase({})
    assert original_hashes == {name:acceptance.sha(source/name) for name in original_hashes}


@pytest.mark.parametrize("defect",["configuration", "source_after_inspection", "receipt_after_inspection"])
def test_documentary_adapter_refuses_context_drift_before_any_backend_call(tmp_path,defect):
    source = bootstrap_fixture(tmp_path/"origin")
    origin = acceptance.inspect_bootstrap(source,acceptance.sha(source/"result.json"))
    config = origin["config"]
    if defect == "configuration":config = replace(config,seed=config.seed+1)
    elif defect == "source_after_inspection":
        path = source/"generation/evaluator_data/baseline.cu"
        path.write_text(path.read_text()+"\n// CPU fixture mutation\n")
    else:
        path = source/"generation/baseline-measurement.json"
        value = campaign.read_json(path);value["status"] = "changed";campaign.write_json(path,value)
    def forbidden(*args,**kwargs):raise AssertionError("No model call before context authentication")
    backend = SimpleNamespace(calls=[],generate_json=forbidden,config=SimpleNamespace(trace_dir=tmp_path/"copy/model"))
    with pytest.raises(ValueError,match="changed"):
        acceptance.DocumentaryRuntime(config,acceptance.ArchivedPropertiesOnlyRunner(origin["properties"]),
            tmp_path/"copy",model_backend=backend,origin=origin)


def test_bootstrap_rejects_current_generator_baseline_drift(tmp_path,monkeypatch):
    source = bootstrap_fixture(tmp_path/"origin")
    original = acceptance.WorkloadSuite.baseline
    def changed_baseline(suite,path):
        candidate = original(suite,path)
        candidate.source.write_text(candidate.source.read_text()+"\n// CPU fixture newer baseline\n")
        return candidate
    monkeypatch.setattr(acceptance.WorkloadSuite,"baseline",changed_baseline)
    with pytest.raises(ValueError,match="baseline reconstruction differs"):
        acceptance.inspect_bootstrap(source,acceptance.sha(source/"result.json"))


def test_prompt_control_is_archived_before_backend_and_never_changes_returned_payload(tmp_path):
    payload = {"source":"unchanged provider payload","techniques":["register_blocking"]}
    result = CodexResult(payload,{},[],"gpt-6-luna","max",0.0,"fixture-call")
    class Backend:
        calls = []
        config = SimpleNamespace(trace_dir=tmp_path/"model")
        def generate_json(self,prompt,schema,**options):
            path = tmp_path/"controls/negative-prompt.txt"
            assert path.read_text() == prompt and "declared control before CLI" in prompt
            self.calls.append(result)
            return result
    controls = acceptance.PromptControls(Backend(),tmp_path/"controls")
    controls.queue("negative","declared control before CLI")
    returned = controls.generate_json("Header\nContext:\n{}",{"properties":{"source":{},"kernel":{},"techniques":{},"parent_id":{}}},purpose="proposal")
    assert returned is result and returned.payload is payload
    assert controls.invocations[0]["returned_payload_sha256"] == campaign.digest(payload)
    assert payload == {"source":"unchanged provider payload","techniques":["register_blocking"]}


def test_prompt_control_does_not_modify_research_synthesis_even_with_proposal_purpose(tmp_path):
    payload = {"source":"synthesis evidence","techniques":["covered"]}
    result = CodexResult(payload,{},[],"gpt-6-luna","max",0.0,"fixture-synthesis")
    class Backend:
        calls = []
        config = SimpleNamespace(trace_dir=tmp_path/"model")
        def generate_json(self,prompt,schema,**options):
            assert prompt == "Synthesis\nContext:\n{}"
            self.calls.append(result)
            return result
    controls = acceptance.PromptControls(Backend(),tmp_path/"controls")
    controls.queue("negative","This must reach only a candidate schema")
    returned = controls.generate_json("Synthesis\nContext:\n{}",
        {"properties":{"source":{},"kernel":{},"techniques":{}}},purpose="proposal")
    assert returned is result and controls.pending is not None
    assert controls.invocations[0]["kernel_proposal"] is False
    assert not (tmp_path/"controls/negative-prompt.txt").exists()


def protocol_runtime(runtime_setup,tmp_path,monkeypatch,*,negative_obeys=True,refresh_covers=True,documentary=False):
    """Real runtime/backend/dossier with clearly declared CPU documentary fixtures."""
    _,config,runner,_ = runtime_setup
    directory = tmp_path/"documentary-runtime"
    backend = CodexBackend(CodexConfig(executable="fixture-never-executed",trace_dir=directory/"model"))
    controls = acceptance.PromptControls(backend,directory/"acceptance-controls")
    if documentary:
        source = bootstrap_fixture(tmp_path/"origin")
        origin = acceptance.inspect_bootstrap(source,acceptance.sha(source/"result.json"))
        runner = acceptance.ArchivedPropertiesOnlyRunner(origin["properties"])
        runtime = acceptance.DocumentaryRuntime(replace(origin["config"],max_trials=1,actions=("tiling",),profile=False),
            runner,directory,model_backend=controls,origin=origin)
    else:
        runtime = campaign.CampaignRuntime(replace(config,max_trials=1,actions=("tiling",),profile=False),runner,
                                           directory,model_backend=controls,generation_only=True)
    prepared = []
    def research(workload,parent,action,evidence):
        dossier_directory = directory/"fixture-research"/str(len(prepared))
        dossier_directory.mkdir(parents=True)
        dossier = research_fixtures.ResearchTests().make_dossier(dossier_directory)
        dossier.research_id = "fixture-initial" if not prepared else "fixture-renewed"
        dossier.workload_hash = runtime.contract_hash
        dossier.environment_fingerprint = runtime.environment_hash
        dossier.strategy_family = action["strategy_family"]
        if prepared:
            dossier.diagnostic_ids = [evidence["diagnostic"]["diagnostic_id"]]
            if refresh_covers:dossier.covered_techniques.append("register_blocking")
        prepared.append(dossier)
        return dossier
    runtime.research_service.prepare = research
    prompts = []
    def popen(command,**options):
        schema = json.loads(Path(command[command.index("--output-schema")+1]).read_text())
        assert "enum" not in schema["properties"]["techniques"]["items"]
        class Process:
            returncode = 0
            def communicate(self,prompt,timeout):
                context = json.loads(prompt.split("Context:\n",1)[1]);payload = research_fixtures.proposal_payload()
                techniques = ["shared_tiling"]
                if ("explicit refusal control" in prompt and negative_obeys) or "Now propose the transformation requiring" in prompt:
                    techniques.append("register_blocking")
                payload.update(parent_id=context["parent"]["candidate_id"],
                    source_parent_id=context["effective_source_parent"]["candidate_id"],
                    contract_hash=context["contract"]["contract_hash"],strategy_family=context["action"]["strategy_family"],
                    source=context["effective_source_parent"]["source"]+f"\n// CPU fixture local variation {len(prompts)}\n",
                    kernel=context["contract"]["operation"],techniques=techniques,
                    references=list(schema["properties"]["references"]["items"]["enum"]))
                prompts.append({"prompt":prompt,"payload":payload})
                events=[{"type":"item.completed","item":{"type":"agent_message","text":json.dumps(payload)}},
                        {"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":3}}]
                return "\n".join(json.dumps(v) for v in events),""
        return Process()
    monkeypatch.setattr("optimizer_agent.codex_backend.subprocess.Popen",popen)
    return runtime,controls,prepared,prompts


def test_actual_runtime_reuses_exact_dossier_then_blocks_refreshes_and_admits(runtime_setup,tmp_path,monkeypatch):
    runtime,controls,prepared,prompts = protocol_runtime(runtime_setup,tmp_path,monkeypatch)
    result = acceptance.exercise(runtime,controls,uncovered_technique="register_blocking")
    assert all(result["checks"].values()) and len(prompts) == 4 and len(prepared) == 2
    assert result["initial_research_id"] == "fixture-initial" and result["renewed_research_id"] == "fixture-renewed"
    assert result["blocked_call_id"] == controls.calls[2].call_id
    assert controls.calls[2].payload == prompts[2]["payload"]
    assert runtime.evaluator.benchmarks == ["baseline"] and runtime.evaluator.validations == []
    assert [x["stage"] for x in controls.invocations] == ["initial-compatible","compatible-adjustment","uncovered-technique","newly-covered-proposal"]
    assert not (runtime.directory/"manifest.json").exists()
    assert len([x for x in runtime.store.events(runtime.config.run_id) if x["kind"]=="generation"]) == 3


def test_documentary_adapter_runs_unchanged_protocol_without_any_gpu_cost_or_current_hardware_claim(runtime_setup,tmp_path,monkeypatch):
    runtime,controls,prepared,prompts = protocol_runtime(runtime_setup,tmp_path,monkeypatch,documentary=True)
    result = acceptance.exercise(runtime,controls,uncovered_technique="register_blocking")
    assert all(result["checks"].values()) and len(prompts) == 4 and len(prepared) == 2
    assert runtime.runner.gpu_attempts == []
    assert runtime.store.budget_state(runtime.config.run_id)["spent"].get("gpu_seconds",0) == 0
    assert not runtime.evaluator._validated and runtime.evaluator._final_exposed is False
    context = campaign.read_json(runtime.directory/"archival-context.json")
    assert context["hardware_currently_observed"] is False and context["baseline_gpu_cost_transferred"] is False
    assert context["authentication"]["search_input_hashes"]
    assert len(context["authentication"]["checked_components"]) == 8


@pytest.mark.parametrize("negative_obeys,refresh_covers,error",[(False,True,"semantic refusal"),(True,False,"does not cover")])
def test_uncooperative_model_or_missing_real_coverage_stays_incomplete(
        runtime_setup,tmp_path,monkeypatch,negative_obeys,refresh_covers,error):
    runtime,controls,prepared,prompts = protocol_runtime(runtime_setup,tmp_path,monkeypatch,
        negative_obeys=negative_obeys,refresh_covers=refresh_covers)
    with pytest.raises(RuntimeError,match=error):
        acceptance.exercise(runtime,controls,uncovered_technique="register_blocking")
    assert len(prompts) == 3 and runtime.evaluator.benchmarks == ["baseline"]
    assert runtime.evaluator.validations == []
    assert not any(x["stage"] == "newly-covered-proposal" for x in controls.invocations)


def test_run_intent_is_durable_before_initialization_failure_and_prevents_duplicate(tmp_path,monkeypatch):
    origin = bootstrap_fixture(tmp_path/"origin");output=tmp_path/"acceptance"
    original = {name:acceptance.sha(origin/name) for name in acceptance.BOOTSTRAP_FILES}
    def fail(*args,**kwargs):
        assert (output/"run-intent.json").is_file()
        assert acceptance.sha(output/"campaign/baseline-measurement.json") == original["generation/baseline-measurement.json"]
        raise RuntimeError("fixture initialization failure before model")
    monkeypatch.setattr(acceptance,"DocumentaryRuntime",fail)
    receipt = acceptance.run_acceptance(output,"cpu-intent",origin,result_sha256=original["result.json"],executable="fixture-never-executed")
    assert receipt["ok"] is False and receipt["status"] == "incomplete"
    assert receipt["new_gpu_requests_attempted"] == 0 and receipt["bootstrap_files_unchanged"] is True
    with pytest.raises(ValueError,match="Existing intent"):
        acceptance.run_acceptance(output,"cpu-intent",origin,result_sha256=original["result.json"])
    assert original == {name:acceptance.sha(origin/name) for name in acceptance.BOOTSTRAP_FILES}
