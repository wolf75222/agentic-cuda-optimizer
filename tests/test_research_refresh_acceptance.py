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
    props = {"name":"explicit CPU fixture", "uuid":"0123456789abcdef0123456789abcdef", "architecture":"sm_90",
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
    hardware = {"environment":"romeo","slurm_job_id":"12345","device_properties":props,
        "hostname":"explicit-cpu-fixture","machine":"aarch64",
        "nvidia_smi":"GPU-01234567-89ab-cdef-0123-456789abcdef, explicit CPU fixture, 590.48.01, 0000:00:00.0",
        "gpu_uuids":["GPU-01234567-89ab-cdef-0123-456789abcdef"],
        "toolkit":"Explicit CPU fixture nvcc12.6","compiler":"Explicit CPU fixture compiler"}
    hardware["fingerprint_id"] = campaign.digest({key:hardware[key] for key in
        ("machine","device_properties","nvidia_smi","toolkit","compiler")})
    for name,value in {"result.json":result,"generation/manifest.json":manifest,
        "generation/config.json":config.to_dict(),"generation/environment.json":env,
        "generation/contract.json":{"contract":asdict(contract),"contract_hash":contract_hash},
        "generation/baseline-measurement.json":baseline,
        "worker/hardware.json":hardware,
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


def test_compile_failure_then_uncovered_repair_and_blocked_refresh_never_tells_cuda_zero(
        runtime_setup,tmp_path,monkeypatch):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from optimizer_agent.campaign_workflow import build_campaign_graph
    runtime,controls,prepared,prompts = protocol_runtime(runtime_setup,tmp_path,monkeypatch)
    prepare = runtime.research_service.prepare
    def research(*args,**kwargs):
        if prepared:
            raise campaign.ResearchBlocked("Explicit CPU fixture: refresh transport unavailable")
        return prepare(*args,**kwargs)
    runtime.research_service.prepare = research
    propose = runtime.generator.propose
    def proposal(*args,**kwargs):
        if prompts:
            controls.queue("unsupported-repair","explicit refusal control")
        return propose(*args,**kwargs)
    runtime.generator.propose = proposal
    benchmark = runtime.evaluator.benchmark
    failures = []
    def compile_failure(candidate,*args,**kwargs):
        if candidate.candidate_id == "baseline":
            return benchmark(candidate,*args,**kwargs)
        failures.append(candidate.candidate_id)
        return {"evaluation_id":"cpu-actual-compile-failure","candidate_id":candidate.candidate_id,
                "ok":False,"status":"compilation_error","error":"explicit CPU fixture compile error"}
    runtime.evaluator.benchmark = compile_failure
    with SqliteSaver.from_conn_string(str(tmp_path/"graph.sqlite")) as checkpoints:
        graph = build_campaign_graph(runtime,checkpoints)
        final = graph.invoke({"run_id":runtime.config.run_id},
            {"configurable":{"thread_id":runtime.config.run_id},"recursion_limit":100})
    assert final["finished"] and final["trials_done"] == 1 and final["repair"] == 1
    assert final["research_ready"] is False and final["retryable"] is False
    assert len(prompts) == 2 and len(failures) == 1
    assert len(runtime.policy.snapshot()["tree"].tree) == 1
    events = runtime.store.events(runtime.config.run_id)
    assert any(event["kind"] == "research_blocked" for event in events)
    assert not any(event["kind"] == "branch_decision" and
                   event["payload"].get("reason") == "repair budget exhausted" for event in events)
    trial_paths = list((runtime.directory/"trials").glob("*/result.json"))
    assert len(trial_paths) == 1
    result = campaign.read_json(trial_paths[0])
    assert "tell" not in result and result["integration_status"] == "suspended_without_performance_observation"
    assert runtime.store.incumbent(runtime.config.run_id)["candidate_id"] is None


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


def failed_resumption_fixture(tmp_path,monkeypatch):
    """All model/search activity is explicitly simulated CPU fixture data."""
    from optimizer_agent.research_service import ResearchBlocked
    bootstrap = bootstrap_fixture(tmp_path/"bootstrap")
    control = {"block_targeted":True,"prompts":[]}
    schema = {"type":"object","additionalProperties":False,
              "properties":{"explicit_cpu_fixture":{"type":"boolean"}},"required":["explicit_cpu_fixture"]}
    def research(service,workload,parent,action,evidence):
        diagnostic = evidence.get("diagnostic")
        if diagnostic and control["block_targeted"]:
            service.backend.generate_json("Explicit CPU fixture search, no network",schema,purpose="research")
            service.backend.generate_json("Explicit CPU fixture synthesis, no sources claimed live",schema,purpose="proposal")
            campaign.write_json(service.directory/"fixture-targeted-blocked.json",{
                "retrieval_status":"blocked","error":"Explicit CPU fixture missing compatibility evidence"})
            raise ResearchBlocked("Explicit CPU fixture missing compatibility evidence")
        target = service.directory/("fixture-initial" if not diagnostic else
            "fixture-renewed-"+str(len(control["prompts"])) if control.get("record_admitted_research") else "fixture-renewed")
        target.mkdir(parents=True)
        dossier = research_fixtures.ResearchTests().make_dossier(target)
        dossier.research_id = target.name
        dossier.workload_hash = workload["contract_hash"]
        dossier.environment_fingerprint = evidence["environment_fingerprint"]
        dossier.strategy_family = action["strategy_family"]
        if diagnostic:
            dossier.diagnostic_ids = [diagnostic["diagnostic_id"]]
            dossier.covered_techniques.append("Optional register reuse in explanatory prose" if control.get("omit_exact_technique") else "register_blocking")
        search=service.backend.generate_json("Explicit CPU fixture search, no network",schema,purpose="research")
        if control.get("record_admitted_research"):
            dossier.research_id=campaign.digest(str(target))[:32]
            control["next_synthesis_payload"]={key:dossier.to_dict()[key] for key in (
                "covered_techniques","reference_kernels","launch_requirements","dependencies","applicability_checks",
                "incompatible_conditions","missing_evidence","pmpp_topics","hypotheses","expected_effects",
                "validation_metrics","negative_search","original_hypothesis")}
        from optimizer_agent.research_service import synthesis_schema
        synthesis=service.backend.generate_json("Explicit CPU fixture synthesis, no sources claimed live",
            synthesis_schema(dossier.sources) if control.get("record_admitted_research") else schema,purpose="proposal")
        if control.get("record_admitted_research"):
            dossier.call_ids=[search.call_id,synthesis.call_id]
            dossier.usage={"input_tokens":4,"output_tokens":6,"cost_usd":None}
            campaign.write_json(service.directory/(dossier.research_id+".json"),dossier.to_dict())
        return dossier
    def popen(command,**options):
        current_schema = campaign.read_json(Path(command[command.index("--output-schema")+1]))
        class Process:
            returncode=0
            def communicate(self,prompt,timeout):
                if control.get("inject_source_artifact") and len(control["prompts"]) >= 7:
                    path = control.pop("inject_source_artifact")
                    if control.get("inject_symlink_target"):
                        path.symlink_to(control["inject_symlink_target"],target_is_directory=True)
                    else:path.write_text("Explicit CPU negative fixture: unexpected artifact in immutable source")
                if "parent_id" in current_schema.get("properties",{}):
                    context=json.loads(prompt.split("Context:\n",1)[1]);payload=research_fixtures.proposal_payload()
                    techniques=["shared_tiling"]
                    if "explicit refusal control" in prompt or "Now propose the transformation requiring" in prompt:
                        techniques.append("register_blocking")
                    payload.update(parent_id=context["parent"]["candidate_id"],
                        source_parent_id=context["effective_source_parent"]["candidate_id"],
                        contract_hash=context["contract"]["contract_hash"],strategy_family=context["action"]["strategy_family"],
                        source=context["effective_source_parent"]["source"]+f"\n// explicit CPU fixture variation{len(control['prompts'])}\n",
                        kernel=context["contract"]["operation"],techniques=techniques,
                        references=list(current_schema["properties"]["references"]["items"]["enum"]))
                else:payload=control.pop("next_synthesis_payload",{"explicit_cpu_fixture":True})
                control["prompts"].append(prompt)
                return "\n".join(json.dumps(row) for row in [
                    {"type":"item.completed","item":{"type":"agent_message","text":json.dumps(payload)}},
                    {"type":"turn.completed","usage":{"input_tokens":2,"output_tokens":3}}]),""
        return Process()
    monkeypatch.setattr(acceptance.ResearchService,"prepare",research)
    # This intentionally incomplete invented baseline is not a live native
    # archive. Only the documentary fixtures use this explicit CPU stand-in.
    monkeypatch.setattr(acceptance,"resume_runtime_context",lambda origin:{
        "kind":"explicit_cpu_fixture_not_native_evidence", "hardware_currently_observed":False,
        "job_id":origin["job_id"], "session_id":origin["session_id"]})
    monkeypatch.setattr(acceptance,"controller_provenance",lambda *args:{"HEAD":"explicit-cpu-fixture", "dirty_state":[],
        "sources":{"scripts/run_research_refresh_acceptance.py":acceptance.sha(Path(acceptance.__file__))}})
    monkeypatch.setattr("optimizer_agent.codex_backend.subprocess.Popen",popen)
    source=tmp_path/"incomplete"
    old=acceptance.run_acceptance(source,"explicit-cpu-original",bootstrap,
        result_sha256=acceptance.sha(bootstrap/"result.json"),executable="explicit-fixture-never-executed")
    assert old["ok"] is False and old["error"].startswith("Actual targeted research blocked:")
    assert len(old["model_calls"])==7 and len(control["prompts"])==7
    return source,control


def resume_pins(source):
    return {"receipt_sha256":acceptance.sha(source/acceptance.RECEIPT),
            "manifest_sha256":acceptance.sha(source/"proof-sha256.json")}


def test_resumption_replays_guards_and_only_runs_fresh_research_and_final_proposal(tmp_path,monkeypatch):
    (tmp_path/".git").mkdir()  # Evidence checkout differs from acceptance.REPO/controller.
    source,control=failed_resumption_fixture(tmp_path,monkeypatch)
    pins=resume_pins(source);old_hashes={str(p.relative_to(source)):acceptance.sha(p) for p in source.rglob('*') if p.is_file()}
    control["block_targeted"]=False
    result=acceptance.run_resumption_acceptance(tmp_path/"resumed","explicit-cpu-resume",source,
        **pins,executable="explicit-fixture-never-executed")
    assert result["ok"] is True, result.get("error")
    assert result["status"]=="documentary_research_refresh_resumption_acceptance_passed"
    assert result["resumption"]["evidence_repository"]==str(tmp_path)
    assert result["resumption"]["source_receipt"]["path_base"]=="repository"
    assert result["resumption"]["source_receipt"]["path"]=="incomplete/"+acceptance.RECEIPT
    assert len(result["checks"])==17 and all(result["checks"].values())
    assert len(result["inherited_model_calls"])==7 and len(result["model_calls"])==3
    assert len(control["prompts"])==10
    assert result["new_usage_delta"]["model_calls"]==3 and result["new_usage_delta"]["model_tokens"]==15
    assert result["new_gpu_requests_attempted"]==0 and result["bootstrap_files_unchanged"] is True
    assert result["resume_source_unchanged"] is True
    assert old_hashes=={str(p.relative_to(source)):acceptance.sha(p) for p in source.rglob('*') if p.is_file()}
    before=campaign.read_json(source/"events.json");after=campaign.read_json(tmp_path/"resumed/events.json")
    assert after[:len(before)]==before
    assert [e["kind"] for e in after[len(before):]]==["model","model","research","model","generation"]
    assert acceptance.sha(tmp_path/"resumed/source-evidence/previous-receipt.json")==pins["receipt_sha256"]
    with pytest.raises(ValueError,match="Existing resume intent"):
        acceptance.run_resumption_acceptance(tmp_path/"resumed","no-repeat",source,**pins)
    assert len(control["prompts"])==10


def test_resumption_blocked_again_stays_incomplete_without_repeating_initial_stages(tmp_path,monkeypatch):
    source,control=failed_resumption_fixture(tmp_path,monkeypatch)
    result=acceptance.run_resumption_acceptance(tmp_path/"resumed","explicit-cpu-resume-blocked",source,
        **resume_pins(source),executable="explicit-fixture-never-executed")
    assert result["ok"] is False and result["status"]=="incomplete"
    assert result["error"].startswith("Actual resumed targeted research blocked:")
    assert len(result["model_calls"])==2 and len(control["prompts"])==9
    assert result["resume_source_unchanged"] is True and result["new_gpu_requests_attempted"]==0


@pytest.mark.parametrize("defect",["receipt_pin","manifest_pin","source_file","extra_file","wrong_phase"])
def test_resumption_refuses_missing_or_changed_evidence_before_any_new_call(tmp_path,monkeypatch,defect):
    source,control=failed_resumption_fixture(tmp_path,monkeypatch);pins=resume_pins(source)
    if defect=="receipt_pin":pins["receipt_sha256"]="0"*64
    elif defect=="manifest_pin":pins["manifest_sha256"]="0"*64
    elif defect=="source_file":
        p=source/"events.json";p.write_text(p.read_text()+" ")
    elif defect=="extra_file":(source/"unmanifested.json").write_text('{}')
    else:
        p=source/acceptance.RECEIPT;d=campaign.read_json(p);d["error"]="Different unobserved failure";campaign.write_json(p,d)
        pins["receipt_sha256"]=acceptance.sha(p)
    with pytest.raises(ValueError):
        acceptance.run_resumption_acceptance(tmp_path/"refused","no-new-call",source,**pins)
    assert len(control["prompts"])==7 and not (tmp_path/"refused/run-intent.json").exists()


def test_resumption_plan_validates_without_invoking_a_service(tmp_path,monkeypatch):
    source,control=failed_resumption_fixture(tmp_path,monkeypatch);pins=resume_pins(source)
    def forbidden(*args,**kwargs):raise AssertionError("Plan must not construct a backend")
    monkeypatch.setattr(acceptance,"CodexBackend",forbidden)
    assert acceptance.main(["--plan","--resume-from",str(source),"--resume-receipt-sha256",pins["receipt_sha256"],
        "--resume-proof-manifest-sha256",pins["manifest_sha256"],"--output",str(tmp_path/"plan")])==0
    assert len(control["prompts"])==7 and not (tmp_path/"plan/run-intent.json").exists()


@pytest.mark.parametrize("symlink",[False,True])
def test_resumption_source_inventory_drift_prevents_success_claim(tmp_path,monkeypatch,symlink):
    source,control=failed_resumption_fixture(tmp_path,monkeypatch);pins=resume_pins(source)
    control["block_targeted"]=False
    control["inject_source_artifact"]=source/"unexpected-during-model"
    if symlink:
        target=tmp_path/"explicit-fixture-target";target.mkdir()
        control["inject_symlink_target"]=target
    result=acceptance.run_resumption_acceptance(tmp_path/"resumed","explicit-cpu-source-drift",source,
        **pins,executable="explicit-fixture-never-executed")
    assert result["ok"] is False and result["status"]=="incomplete"
    assert result["resume_source_unchanged"] is False and result["checks"]["resume_source_unchanged"] is False
    assert result["new_gpu_requests_attempted"]==0


def twice_blocked_fixture(tmp_path,monkeypatch):
    (tmp_path/".git").mkdir()
    initial,control=failed_resumption_fixture(tmp_path,monkeypatch)
    source=tmp_path/"once-resumed"
    result=acceptance.run_resumption_acceptance(source,"explicit-cpu-first-resume-blocked",initial,
        **resume_pins(initial),executable="explicit-fixture-never-executed")
    assert result["ok"] is False and len(result["model_calls"])==2
    assert len(campaign.read_json(source/"events.json"))==19 and len(control["prompts"])==9
    return initial,source,control


def rewrite_fixture_manifest(source):
    """Re-pin explicitly invented negative fixture data, never real receipts."""
    manifest=campaign.read_json(source/"proof-sha256.json")
    manifest["files"]={str(p.relative_to(source)):acceptance.sha(p) for p in source.rglob("*")
        if p.is_file() and str(p.relative_to(source)) not in (acceptance.RECEIPT,"proof-sha256.json")}
    campaign.write_json(source/"proof-sha256.json",manifest)
    receipt=campaign.read_json(source/acceptance.RECEIPT)
    receipt["proof_manifest"]["sha256"]=acceptance.sha(source/"proof-sha256.json")
    campaign.write_json(source/acceptance.RECEIPT,receipt)


def test_second_resumption_derives_failed_round3_and_researches_round4_without_replaying_nine_calls(tmp_path,monkeypatch):
    initial,source,control=twice_blocked_fixture(tmp_path,monkeypatch)
    snapshots={p:{str(f.relative_to(p)):acceptance.sha(f) for f in p.rglob("*") if f.is_file()} for p in (initial,source)}
    prepared=acceptance.inspect_resume_source(source,**resume_pins(source))
    assert len(prepared["calls"])==9 and prepared["state"]["repair"]==2
    assert prepared["state"]["research_round"]==3
    assert campaign.read_json(source/"resume-state.json")["state"]["research_round"]==2
    assert prepared["research_round_derivation"]=={
        "initial_anchor_round":2,"completed_failed_round":3,"next_fresh_round":4}
    control["block_targeted"]=False
    output=tmp_path/"twice-resumed"
    result=acceptance.run_resumption_acceptance(output,"explicit-cpu-second-resume",source,
        **resume_pins(source),executable="explicit-fixture-never-executed")
    assert result["ok"] is True,result.get("error")
    assert len(result["inherited_model_calls"])==9 and len(result["model_calls"])==3
    assert len(control["prompts"])==12 and result["new_usage_delta"]["model_calls"]==3
    assert result["ledger_usage"]["spent"]["model_calls"]==12
    assert result["resumption"]["controls_directory"]=="acceptance-controls-resume-2"
    assert (output/"campaign/acceptance-controls-resume-2/invocations.json").is_file()
    assert acceptance.sha(output/"campaign/acceptance-controls-resume/invocations.json")==snapshots[source]["campaign/acceptance-controls-resume/invocations.json"]
    assert (output/"campaign/trials"/prepared["state"]["trial_id"]/"proposal-2-4.json").is_file()
    assert campaign.read_json(output/"resume-state.json")["state"]["research_round"]==3
    assert result["new_gpu_requests_attempted"]==0 and result["bootstrap_files_unchanged"] is True
    assert result["resume_source_unchanged"] is True
    events=campaign.read_json(output/"events.json")
    assert events[:19]==campaign.read_json(source/"events.json")
    assert [e["kind"] for e in events[19:]]==["model","model","research","model","generation"]
    for archive,hashes in snapshots.items():
        assert hashes=={str(f.relative_to(archive)):acceptance.sha(f) for f in archive.rglob("*") if f.is_file()}
    # This is deliberately bounded support for the observed first resumption,
    # not permission to recursively replay arbitrary future incomplete states.
    with pytest.raises(ValueError):acceptance.inspect_resume_source(output,**resume_pins(output))
    assert len(control["prompts"])==12


def test_second_resumption_can_stay_blocked_without_claiming_admission_or_replaying_old_calls(tmp_path,monkeypatch):
    initial,source,control=twice_blocked_fixture(tmp_path,monkeypatch)
    result=acceptance.run_resumption_acceptance(tmp_path/"twice-resumed","explicit-cpu-still-blocked",source,
        **resume_pins(source),executable="explicit-fixture-never-executed")
    assert result["ok"] is False and result["status"]=="incomplete" and "checks" not in result
    assert len(result["inherited_model_calls"])==9 and len(result["model_calls"])==2
    assert len(control["prompts"])==11 and result["ledger_usage"]["spent"]["model_calls"]==11
    assert result["resume_source_unchanged"] is True and result["new_gpu_requests_attempted"]==0
    assert campaign.read_json(initial/acceptance.RECEIPT)["ok"] is False
    assert campaign.read_json(source/acceptance.RECEIPT)["ok"] is False


@pytest.mark.parametrize("defect",["anchor_role","anchor_hash","old_call","old_campaign","new_call",
    "repeated_call","new_payload","unknown_cli","event_suffix","entry_round","hardware",
    "ledger","intent","deeper_chain","extra_candidate","private_reference","policy_snapshot"])
def test_second_resumption_rejects_re_pinned_tampered_or_ambiguous_chain_before_any_new_call(tmp_path,monkeypatch,defect):
    _,source,control=twice_blocked_fixture(tmp_path,monkeypatch)
    receipt=campaign.read_json(source/acceptance.RECEIPT)
    intent=campaign.read_json(source/"run-intent.json")
    if defect=="anchor_role":
        receipt["resumption"]["source_receipt"]["path"]="incomplete/campaign/experiments.sqlite"
    elif defect=="anchor_hash":receipt["resumption"]["source_proof_manifest"]["sha256"]="0"*64
    elif defect=="old_call":receipt["inherited_model_calls"][0]["usage"]["input_tokens"]+=1
    elif defect=="old_campaign":
        p=source/"campaign/acceptance-evidence/blocked-refusal.json";p.write_text(p.read_text()+" ")
    elif defect=="new_call":receipt["model_calls"][0]["usage"]["input_tokens"]+=1
    elif defect=="repeated_call":receipt["model_calls"][0]["call_id"]=receipt["inherited_model_calls"][0]["call_id"]
    elif defect=="new_payload":
        p=source/"campaign/model"/(receipt["model_calls"][0]["call_id"]+".json")
        data=campaign.read_json(p);data["payload"]={"explicitly_tampered":True};campaign.write_json(p,data)
    elif defect=="unknown_cli":campaign.write_json(source/"campaign/model"/("f"*32+".json"),{})
    elif defect=="event_suffix":
        p=source/"events.json";data=campaign.read_json(p);data[-1]["kind"]="diagnostic";campaign.write_json(p,data)
    elif defect=="entry_round":
        p=source/"resume-state.json";data=campaign.read_json(p);data["state"]["research_round"]=3;campaign.write_json(p,data)
    elif defect=="hardware":
        p=source/"campaign/hardware-evidence-context.json";data=campaign.read_json(p);data["job_id"]="999";campaign.write_json(p,data)
    elif defect=="ledger":receipt["new_usage_delta"]["model_calls"]=1
    elif defect=="intent":intent["run_id"]="different-run"
    elif defect=="deeper_chain":receipt["resumption"]["inherited_completed_model_calls"]=9
    elif defect=="extra_candidate":campaign.write_json(source/"campaign/trials/extra/proposal-2-4.json",{})
    elif defect=="policy_snapshot":
        with acceptance.sqlite3.connect(source/"campaign/experiments.sqlite") as db:
            db.execute("UPDATE campaigns SET tree_version=tree_version+1")
    else:receipt["resumption"]["source_receipt"]["path"]="final-groups/fake/holdout-reveal.json"
    if defect!="intent":intent["resumption"]=receipt["resumption"]
    campaign.write_json(source/acceptance.RECEIPT,receipt);campaign.write_json(source/"run-intent.json",intent)
    rewrite_fixture_manifest(source)
    with pytest.raises(ValueError):
        acceptance.run_resumption_acceptance(tmp_path/"refused-second","explicit-cpu-refused",source,
            **resume_pins(source),executable="explicit-fixture-never-executed")
    assert len(control["prompts"])==9 and not (tmp_path/"refused-second/run-intent.json").exists()


def test_second_resumption_detects_initial_anchor_inventory_drift_during_new_calls(tmp_path,monkeypatch):
    initial,source,control=twice_blocked_fixture(tmp_path,monkeypatch)
    control["block_targeted"]=False
    control["inject_source_artifact"]=initial/"unexpected-during-second-resume"
    result=acceptance.run_resumption_acceptance(tmp_path/"twice-resumed","explicit-cpu-ancestor-drift",source,
        **resume_pins(source),executable="explicit-fixture-never-executed")
    assert result["ok"] is False and result["status"]=="incomplete"
    assert result["checks"]["resume_source_unchanged"] is False
    assert result["resume_source_unchanged"] is False and result["new_gpu_requests_attempted"]==0


def test_resumption_source_directory_alias_is_rejected_before_any_archived_payload_read(tmp_path,monkeypatch):
    source,control=failed_resumption_fixture(tmp_path,monkeypatch)
    pins=resume_pins(source)
    alias=tmp_path/"source-alias";alias.symlink_to(source,target_is_directory=True)
    original=Path.read_bytes
    def forbidden_archive_read(path):
        assert not path.resolve().is_relative_to(source),"Aliased source was read before its path guard"
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",forbidden_archive_read)
    with pytest.raises(ValueError,match="symlink"):
        acceptance.run_resumption_acceptance(tmp_path/"refused-alias","explicit-cpu-source-alias",alias,**pins)
    assert len(control["prompts"])==7 and not (tmp_path/"refused-alias/run-intent.json").exists()


@pytest.mark.parametrize("target",["initial_anchor","resumed_source","bootstrap","initial_anchor_alias"])
def test_second_resumption_output_cannot_mutate_any_immutable_archive_before_rejection(tmp_path,monkeypatch,target):
    initial,source,control=twice_blocked_fixture(tmp_path,monkeypatch)
    bootstrap=tmp_path/"bootstrap"
    archives=(initial,source,bootstrap)
    snapshots={p:{str(f.relative_to(p)):acceptance.sha(f) for f in p.rglob("*") if f.is_file()} for p in archives}
    root={"initial_anchor":initial,"resumed_source":source,"bootstrap":bootstrap}.get(target)
    if target=="initial_anchor_alias":
        root=tmp_path/"ancestor-alias";root.symlink_to(initial,target_is_directory=True)
    output=root/"must-not-be-created"
    with pytest.raises(ValueError,match="immutable"):
        acceptance.run_resumption_acceptance(output,"explicit-cpu-unsafe-output",source,
            **resume_pins(source),executable="explicit-fixture-never-executed")
    assert not output.exists() and len(control["prompts"])==9
    for archive,hashes in snapshots.items():
        assert hashes=={str(f.relative_to(archive)):acceptance.sha(f) for f in archive.rglob("*") if f.is_file()}


def accepted_coverage_gap_fixture(tmp_path,monkeypatch):
    initial,source9,control=twice_blocked_fixture(tmp_path,monkeypatch)
    control.update(block_targeted=False,record_admitted_research=True,omit_exact_technique=True)
    source11=tmp_path/"accepted-gap"
    result=acceptance.run_resumption_acceptance(source11,"explicit-cpu-accepted-gap",source9,
        **resume_pins(source9),executable="explicit-fixture-never-executed")
    assert result["ok"] is False and result["error"]=="Actual resumed dossier does not cover the exact blocked technique/diagnostic"
    assert len(result["model_calls"])==2 and len(control["prompts"])==11
    assert [e["kind"] for e in campaign.read_json(source11/"events.json")][-3:]==["model","model","research"]
    return initial,source9,source11,control


def test_third_resumption_preserves_accepted_round4_then_explicitly_refreshes_round5(tmp_path,monkeypatch):
    initial,source9,source11,control=accepted_coverage_gap_fixture(tmp_path,monkeypatch)
    snapshots={p:{str(f.relative_to(p)):acceptance.sha(f) for f in p.rglob("*") if f.is_file()}
        for p in (initial,source9,source11)}
    prepared=acceptance.inspect_resume_source(source11,**resume_pins(source11))
    assert len(prepared["calls"])==11 and len(prepared["ancestor_archives"])==2
    assert prepared["state"]["research_round"]==4 and prepared["state"]["research_ready"] is True
    assert prepared["state"]["needs_research"] is False and prepared["state"]["error"] is None
    assert prepared["research_round_derivation"]=={
        "initial_anchor_round":2,"completed_failed_round":3,"completed_accepted_round":4,"next_fresh_round":5}
    accepted=prepared["coverage_refresh"]["accepted_dossier"]
    assert "register_blocking" not in campaign.read_json(source11/accepted["path"])["covered_techniques"]
    control["omit_exact_technique"]=False
    output=tmp_path/"third-resumed"
    result=acceptance.run_resumption_acceptance(output,"explicit-cpu-complement",source11,
        **resume_pins(source11),executable="explicit-fixture-never-executed")
    assert result["ok"] is True,result.get("error")
    assert all(result["checks"].values()) and len(result["checks"])==17
    assert len(result["inherited_model_calls"])==11 and len(result["model_calls"])==3
    assert result["ledger_usage"]["spent"]["model_calls"]==14 and result["new_usage_delta"]=={"model_tokens":15,"model_calls":3}
    assert result["resumption"]["phase"]=="complementary_coverage_refresh"
    assert result["resumption"]["controls_directory"]=="acceptance-controls-resume-3"
    assert len(control["prompts"])==14 and result["new_gpu_requests_attempted"]==0
    entry=campaign.read_json(output/"resume-state.json")
    assert entry["phase"]=="accepted_research_coverage_gap_reconstructed" and entry["state"]==prepared["state"]
    request=campaign.read_json(output/"coverage-refresh-request.json")
    assert request["post_research_state"]==entry["state"] and request["prior_research_succeeded"] is True
    assert request["requested_state"]==dict(entry["state"],needs_research=True,research_refresh_required=True)
    assert request["next_fresh_round"]==5 and request["missing_techniques"]==["register_blocking"]
    assert request["diagnostic_id"]==entry["state"]["diagnostic_id"] and request["requested_state"]["error"] is None
    assert acceptance.sha(output/"source-evidence/previous-accepted-dossier.json")==accepted["sha256"]
    assert (output/"campaign/trials"/entry["state"]["trial_id"]/"proposal-2-5.json").is_file()
    assert result["renewed_research_id"]!=prepared["coverage_refresh"]["accepted_research_id"]
    events=campaign.read_json(output/"events.json")
    assert events[:22]==prepared["events"] and [e["kind"] for e in events[22:]]==["model","model","research","model","generation"]
    for archive,hashes in snapshots.items():
        assert hashes=={str(f.relative_to(archive)):acceptance.sha(f) for f in archive.rglob("*") if f.is_file()}
    with pytest.raises(ValueError):acceptance.inspect_resume_source(output,**resume_pins(output))


@pytest.mark.parametrize("research_blocked",[False,True])
def test_complementary_refresh_remains_incomplete_when_fresh_evidence_still_missing(tmp_path,monkeypatch,research_blocked):
    _,_,source11,control=accepted_coverage_gap_fixture(tmp_path,monkeypatch)
    control["block_targeted"]=research_blocked
    result=acceptance.run_resumption_acceptance(tmp_path/"third-incomplete","explicit-cpu-negative-complement",source11,
        **resume_pins(source11),executable="explicit-fixture-never-executed")
    assert result["ok"] is False and result["status"]=="incomplete" and "checks" not in result
    assert len(result["inherited_model_calls"])==11 and len(result["model_calls"])==2
    assert len(control["prompts"])==13 and result["new_gpu_requests_attempted"]==0
    assert result["error"].startswith("Actual resumed targeted research blocked:" if research_blocked else "Actual resumed dossier does not cover")
    assert sum(e["kind"]=="generation" for e in campaign.read_json(tmp_path/"third-incomplete/events.json"))==2


def test_third_resumption_plan_uses_accepted_state_and_calls_no_backend(tmp_path,monkeypatch):
    _,_,source11,control=accepted_coverage_gap_fixture(tmp_path,monkeypatch)
    def forbidden(*args,**kwargs):raise AssertionError("Plan must not construct a backend")
    monkeypatch.setattr(acceptance,"CodexBackend",forbidden)
    prepared=acceptance.prepare_resumption(tmp_path/"third-plan","explicit-cpu-complement-plan",source11,**resume_pins(source11))
    plan=campaign.read_json(tmp_path/"third-plan/plan.json")
    assert plan["resume_phase"]=="complementary_coverage_refresh" and plan["inherited_completed_model_calls"]==11
    assert plan["coverage_refresh"]==prepared["coverage_refresh"] and plan["research_round_derivation"]["next_fresh_round"]==5
    assert plan["model_invoked"] is False and plan["new_gpu_requests"]==0
    assert not (tmp_path/"third-plan/run-intent.json").exists() and len(control["prompts"])==11


@pytest.mark.parametrize("defect",["ancestor_pin","prefix","old_call","new_call","usage_delta","payload","phase",
    "event_suffix","accepted_id","diagnostic","covered_exact","applicability","source_passage","extra_candidate","policy_snapshot","entry_round"])
def test_third_resumption_rejects_changed_prefix_usage_phase_or_dossier_before_new_call(tmp_path,monkeypatch,defect):
    _,_,source11,control=accepted_coverage_gap_fixture(tmp_path,monkeypatch)
    receipt=campaign.read_json(source11/acceptance.RECEIPT);intent=campaign.read_json(source11/"run-intent.json")
    if defect=="ancestor_pin":receipt["resumption"]["source_proof_manifest"]["sha256"]="0"*64
    elif defect=="old_call":receipt["inherited_model_calls"][0]["usage"]["input_tokens"]+=1
    elif defect=="new_call":receipt["model_calls"][0]["usage"]["input_tokens"]+=1
    elif defect=="usage_delta":receipt["new_usage_delta"]["model_tokens"]+=1
    elif defect=="phase":receipt["resumption"]["research_round_derivation"]["next_fresh_round"]=5
    elif defect in ("prefix","event_suffix"):
        p=source11/"events.json";data=campaign.read_json(p);data[0 if defect=="prefix" else -1]["kind"]="research_blocked";campaign.write_json(p,data)
    elif defect=="payload":
        p=source11/"campaign/model"/(receipt["model_calls"][-1]["call_id"]+".json")
        data=campaign.read_json(p);data["payload"]["covered_techniques"].append("register_blocking");campaign.write_json(p,data)
    elif defect in ("accepted_id","diagnostic","covered_exact","applicability","source_passage"):
        p=next((source11/"campaign/trials").glob("*/dossier.json"));data=campaign.read_json(p)
        if defect=="accepted_id":data["research_id"]="../unmanifested"
        elif defect=="diagnostic":data["diagnostic_ids"]=["different-diagnostic"]
        elif defect=="covered_exact":data["covered_techniques"].append("register_blocking")
        elif defect=="applicability":data["applicability_checks"][0]["satisfied"]=False
        else:data["sources"][0]["passages"][0]["text_sha256"]="0"*64
        campaign.write_json(p,data)
    elif defect=="extra_candidate":campaign.write_json(source11/"campaign/trials/extra/proposal-2-4.json",{})
    elif defect=="entry_round":
        p=source11/"resume-state.json";data=campaign.read_json(p);data["state"]["research_round"]=4;campaign.write_json(p,data)
    else:
        with acceptance.sqlite3.connect(source11/"campaign/experiments.sqlite") as db:
            db.execute("UPDATE campaigns SET tree_version=tree_version+1")
    intent["resumption"]=receipt["resumption"]
    campaign.write_json(source11/acceptance.RECEIPT,receipt);campaign.write_json(source11/"run-intent.json",intent)
    rewrite_fixture_manifest(source11)
    with pytest.raises(ValueError):
        acceptance.run_resumption_acceptance(tmp_path/"third-refused","explicit-cpu-refused-complement",source11,
            **resume_pins(source11),executable="explicit-fixture-never-executed")
    assert len(control["prompts"])==11 and not (tmp_path/"third-refused/run-intent.json").exists()


@pytest.mark.parametrize("target",["initial","nine","eleven","bootstrap"])
def test_third_resumption_rejects_destination_inside_any_ancestor_before_writes(tmp_path,monkeypatch,target):
    initial,source9,source11,control=accepted_coverage_gap_fixture(tmp_path,monkeypatch)
    archives={"initial":initial,"nine":source9,"eleven":source11,"bootstrap":tmp_path/"bootstrap"}
    output=archives[target]/"must-not-exist"
    with pytest.raises(ValueError,match="immutable"):
        acceptance.run_resumption_acceptance(output,"explicit-cpu-destination-refused",source11,**resume_pins(source11))
    assert not output.exists() and len(control["prompts"])==11


@pytest.mark.parametrize("target",["initial","nine","eleven"])
def test_third_resumption_source_inventory_drift_during_new_calls_revokes_success(tmp_path,monkeypatch,target):
    initial,source9,source11,control=accepted_coverage_gap_fixture(tmp_path,monkeypatch)
    control.update(omit_exact_technique=False,inject_source_artifact={"initial":initial,"nine":source9,"eleven":source11}[target]/"unexpected")
    result=acceptance.run_resumption_acceptance(tmp_path/"third-drift","explicit-cpu-drift",source11,
        **resume_pins(source11),executable="explicit-fixture-never-executed")
    assert result["ok"] is False and result["resume_source_unchanged"] is False
    assert result["checks"]["resume_source_unchanged"] is False and result["new_gpu_requests_attempted"]==0
