"""Exercise the pinned original graph with CPU doubles, never Slurm or a model."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from campaign_config import CampaignConfig
from optimizer_agent.frozen_launch import FrozenShapeLaunchRule, parse_launch_rule
from optimizer_agent import upstream_original as module


class CPUHarness:
    def __init__(self, directory):
        self.results_dir = Path(directory)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.calls = []

    def get_device_properties(self, device):
        return {"name":"SIMULATED_UNIT_TEST_ONLY", "uuid":"cpu-test", "architecture":"sm_90",
            "driver_version":13010, "nvrtc_version":"12.6", "harness_build_id":"cpu-double",
            "max_threads_per_block":1024, "max_shared_memory_per_block_bytes":49152,
            "max_grid_dimensions":[2147483647,65535,65535], "max_block_dimensions":[1024,1024,64],
            "multiprocessor_count":132, "free_memory_bytes":80*1024**3}

    def run(self, run_id, request):
        self.calls.append((run_id, request))
        n = int(request.arguments[-1].value)
        a,b = [np.fromfile(argument.input_file,dtype="<f4").astype(np.float64)
               for argument in request.arguments[:2]]
        if request.kernel == "matmul":
            output = (a.reshape(n,n) @ b.reshape(n,n)).astype("<f4").ravel()
        elif request.kernel == "reduction":
            output = np.asarray([a.sum()], dtype="<f4")
        elif request.kernel == "scan":
            output = np.cumsum(a).astype("<f4")
        elif request.kernel == "stencil":
            padded = np.pad(a,(1,1)); output = (padded[:-2]+padded[1:-1]+padded[2:]).astype("<f4")
        elif request.kernel == "convolution":
            padded = np.pad(a,(1,1)); output = (padded[:-2]*b[0]+padded[1:-1]*b[1]+padded[2:]*b[2]).astype("<f4")
        else:
            output = (a/np.sqrt(np.mean(a*a)+1e-5)).astype("<f4")
        source = Path(request.source).read_text()
        if "UNIT_WRONG" in source:
            output[:] = 0
        path = self.results_dir/(run_id+".bin")
        output.tofile(path)
        latency = 1.0 if "UNIT_REPAIRED" in source else 2.0
        return {"ok":True, "outputs":[{"path":str(path),"bytes":output.nbytes}],
            "latency_ms":latency,"latency_samples_ms":[latency]*request.iterations,
            "guard_readonly_ok":True,"guard_outputs_poisoned":True,"environment":self.get_device_properties(0)}


class OriginalResponses:
    """Returns only original schemas; real original nodes decide their meaning."""
    def __init__(self, sizes, operation, *, repair=False, tools=False):
        self.sizes,self.operation,self.repair,self.tools = sizes,operation,repair,tools
        self.prompts = []
        self.tool_requested = False
        self.proposals = 0

    def generate_json(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        conversation = json.loads(prompt.split("\n",1)[1])
        text = "\n".join(str(message["content"]) for message in conversation["messages"])
        if self.tools and not self.tool_requested:
            self.tool_requested = True
            payload = {"kind":"tool_calls","text":"","tool_calls":[{"name":"get_device_properties","arguments_json":"{\"device\":0}"}]}
        else:
            launches = []
            for index,n in enumerate(self.sizes):
                block = [16,16,1] if self.operation == "matmul" else [128,1,1]
                grid = [(n+15)//16,(n+15)//16,1] if self.operation == "matmul" else [(n+127)//128,1,1]
                launches.append({"case_index":index,"config":{"grid":grid,"block":block,"shared_memory_bytes":0},
                    "dynamic_shared_memory_reason":"The supplied source requires no dynamic shared allocation."})
            if "Prepare launch configurations for this existing CUDA kernel." in text:
                result = {"launches":launches}
            elif not self.repair or self.proposals >= 2:
                result = {"action":"stop","reason":"The initial kernel is correct; no further evidenced experiment remains."}
            else:
                start = self.proposals
                self.proposals += 1
                baseline = module.common_workloads.SOURCES[self.operation]
                result = {"kernel_source":baseline+"\n// "+("UNIT_WRONG" if start == 0 else "UNIT_REPAIRED"),
                    "launches":launches,"hypothesis":{"change_summary":"Repair output mapping" if start else "Change output mapping",
                        "starting_iteration":start,"supporting_measurements":[f"iteration {start} case 0 kernel_latency_ms=2.0 ms; validation is available."],
                        "proposed_change":"Repair the failed mapping." if start else "Test a changed mapping.",
                        "expected_benefit":"Predict lower kernel_latency_ms with correct outputs; a mapping error can fail validation.",
                        "case_execution_paths":[{"case_index":index,"active_path":"one unconditional path",
                            "condition_evaluation":f"n={n}; source has one path", "executes_changed_code":True}
                            for index,n in enumerate(self.sizes)],"hardware_claims":[]}}
            payload = {"kind":"final","text":json.dumps(result),"tool_calls":[]}
        return SimpleNamespace(payload=payload,usage={"input_tokens":100,"cached_input_tokens":10,"output_tokens":25},
            events=[],call_id=f"cpu-{len(self.prompts)}",model="gpt-6-luna",reasoning_effort="max",elapsed_seconds=.01)


@pytest.fixture(scope="module")
def pinned(tmp_path_factory):
    return module.snapshot_upstream(tmp_path_factory.mktemp("pinned-original"))[0]


@pytest.mark.parametrize("operation",module.common_workloads.OPERATIONS)
def test_actual_original_graph_keeps_initial_and_tool_routing(operation,tmp_path,pinned):
    config = CampaignConfig(run_id="original-"+operation,workload=operation,search_sizes=(8,),final_sizes=(8,),
        max_trials=2,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = OriginalResponses((1,17,63,8),operation,tools=True)
    runner = CPUHarness(tmp_path/"runner")
    result = module.run_upstream_campaign(config,runner,tmp_path/"original",model_backend=backend,source_snapshot=pinned)
    assert result["generation_complete"] and result["status"] == "generated", result
    assert result["method"] == "true_upstream_original"
    assert result["upstream"]["commit"] == module.UPSTREAM_COMMIT
    assert result["original_summary"]["best_iteration"] == 0
    assert result["original_summary"]["termination"] == "agent_stop"
    assert result["metrics"]["candidate_configurations_with_recorded_validity"] == 0
    assert result["metrics"]["correct_candidates"] == 0
    assert result["metrics"]["correct_candidate_fraction"] is None
    assert result["metrics"]["time_to_first_useful_gain_seconds"] is None
    assert result["search_validation"]["ok"]
    assert result["usage"]["model_calls"] == 4
    assert result["usage"]["model_tokens"] == 500
    assert set(result["candidate"]["launch_rule"]["table"]) == {"1","17","63","8"}
    from optimizer_agent.suite_orchestrator import load_candidate
    assert load_candidate(result["candidate"]).artifact_hash == result["candidate"]["artifact_hash"]
    assert list((tmp_path/"original").glob("tools-reference-*.json"))
    assert all("PMPP" in prompt for prompt in backend.prompts)
    assert not list((tmp_path/"original").rglob("final-*"))
    before = len(runner.calls)
    resumed = module.run_upstream_campaign(config,runner,tmp_path/"original",resume=True,model_backend=backend,source_snapshot=pinned)
    assert resumed == result and len(runner.calls) == before


def test_frozen_shape_rule_roundtrip_rejects_extrapolation_and_checks_device():
    data = {"17":{"grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0}}
    rule = FrozenShapeLaunchRule(data)
    assert asdict(parse_launch_rule(asdict(rule))) == asdict(rule)
    assert rule.apply({"n":17}) == data["17"]
    with pytest.raises(ValueError,match="outside"):
        rule.apply({"n":18})
    with pytest.raises(ValueError,match="thread limit"):
        rule.apply({"n":17},{"max_threads_per_block":64})


@pytest.mark.parametrize("mutation",["shape","grid"])
def test_candidate_rejects_nested_frozen_launch_mutation_before_hash_or_execution(tmp_path,mutation):
    source = tmp_path/"kernel.cu"
    source.write_text(module.common_workloads.SOURCES["scan"])
    rule = FrozenShapeLaunchRule({"17":{"grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0}})
    candidate = module.common_evaluation.Candidate("frozen",source,"scan",rule)
    original_hash = candidate.artifact_hash
    if mutation == "shape":
        rule.table["18"] = {"grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0}
    else:
        rule.table["17"]["grid"][0] = 2
    with pytest.raises(ValueError,match="launch rule changed"):
        candidate.artifact_hash
    with pytest.raises(ValueError,match="launch rule changed"):
        candidate.check_unchanged()
    assert original_hash


def test_snapshot_tamper_is_rejected_and_aliases_restored(tmp_path,pinned):
    import use_harness
    before = use_harness
    with module.original_modules(pinned) as original:
        assert Path(original.workflow.__file__).resolve().is_relative_to(pinned)
        assert original.use_harness is not before
    import use_harness
    assert use_harness is before
    snapshot,identity = module.snapshot_upstream(tmp_path/"snapshot")
    source = snapshot/"optimizer_agent"/"workflow.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == identity["source_hashes"]["optimizer_agent/workflow.py"]
    source.write_text(source.read_text()+"\n# tampered")
    with pytest.raises(RuntimeError,match="snapshot changed"):
        module.snapshot_upstream(snapshot)


def test_original_failed_candidate_is_repaired_and_geomean_selects_it(tmp_path,pinned):
    config = CampaignConfig(run_id="original-repair",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=3,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = OriginalResponses((1,17,63,8),"scan",repair=True)
    result = module.run_upstream_campaign(config,CPUHarness(tmp_path/"runner"),tmp_path/"original",
        model_backend=backend,source_snapshot=pinned)
    assert result["status"] == "generated", result
    assert result["original_summary"]["attempt_count"] == 3
    assert result["original_summary"]["best_iteration"] == 2
    assert result["original_summary"]["best_geometric_mean_ms"] == 1.0
    assert result["original_summary"]["termination"] == "agent_stop"
    assert result["metrics"]["candidate_configurations_with_recorded_validity"] == 2
    assert result["metrics"]["correct_candidates"] == 1
    assert result["metrics"]["correct_candidate_fraction"] == .5
    assert result["metrics"]["time_to_first_useful_gain_seconds"] is None
    repair_prompt = next(prompt for prompt in backend.prompts if "REPAIR REQUIRED: candidate 1" in prompt)
    assert "UNIT_WRONG" in repair_prompt
    assert "UNIT_REPAIRED" in Path(result["candidate"]["source_path"]).read_text()
    history = json.loads((tmp_path/"original"/"history.json").read_text())
    assert not any(case["validation_ok"] for case in history[1]["case_runs"])
    assert all(case["validation_ok"] for case in history[2]["case_runs"])


def test_interrupted_original_graph_resumes_real_checkpoint_and_usage(tmp_path,pinned,monkeypatch):
    config = CampaignConfig(run_id="original-resume",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=2,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = OriginalResponses((1,17,63,8),"scan")
    invoke = module.OriginalCodexTransport.invoke
    interrupted = []
    def after_completed_reply(self,messages):
        result = invoke(self,messages)
        if len(backend.prompts) == 2 and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt("simulated process interruption after durable model reply before graph checkpoint")
        return result
    monkeypatch.setattr(module.OriginalCodexTransport,"invoke",after_completed_reply)
    runner = CPUHarness(tmp_path/"runner")
    with pytest.raises(KeyboardInterrupt):
        module.run_upstream_campaign(config,runner,tmp_path/"original",model_backend=backend,source_snapshot=pinned)
    assert not (tmp_path/"original"/"manifest.json").exists()
    reference_calls = len([call for call in runner.calls if call[0].startswith("original-")])
    assert reference_calls == 4
    result = module.run_upstream_campaign(config,runner,tmp_path/"original",resume=True,model_backend=backend,source_snapshot=pinned)
    assert result["status"] == "generated", result
    assert result["usage"]["model_calls"] == 3
    assert result["usage"]["input_tokens"] == 300
    calls = json.loads((tmp_path/"original"/"transport.json").read_text())
    assert len(calls) == 8
    assert sum(call["original_run_id"].startswith("reference-") for call in calls) == 4
    assert result["original_summary"]["best_iteration"] == 0


def test_completed_original_source_tamper_cannot_reuse_generation_receipt(tmp_path,pinned):
    config = CampaignConfig(run_id="original-tamper",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=1,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = OriginalResponses((1,17,63,8),"scan")
    runner = CPUHarness(tmp_path/"runner")
    result = module.run_upstream_campaign(config,runner,tmp_path/"original",model_backend=backend,source_snapshot=pinned)
    Path(result["candidate"]["source_path"]).write_text("changed source")
    with pytest.raises(ValueError,match="changed after generation"):
        module.run_upstream_campaign(config,runner,tmp_path/"original",resume=True,model_backend=backend,source_snapshot=pinned)


def test_original_eight_call_guard_is_original_algorithm_count(tmp_path,pinned):
    class LoopingTools(OriginalResponses):
        def generate_json(self,prompt,schema,**kwargs):
            self.tool_requested = False
            return super().generate_json(prompt,schema,**kwargs)
    config = CampaignConfig(run_id="original-call-guard",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=2,max_repairs_per_trial=0,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = LoopingTools((1,17,63,8),"scan",tools=True)
    result = module.run_upstream_campaign(config,CPUHarness(tmp_path/"runner"),tmp_path/"original",
        model_backend=backend,source_snapshot=pinned)
    assert result["generation_complete"] and result["status"] == "method_failed"
    assert result["method_status"] == "failed" and result["method_error"]
    assert result["usage"]["model_calls"] == 8
    assert result["original_limits"]["max_model_calls_per_proposal"] == 8
    assert "Model/tool call budget exhausted for this proposal" in result["original_summary"]["error"]
    assert result["global_resource_caps"] is None


def test_original_reference_cannot_validate_itself_with_same_wrong_launch(tmp_path,pinned):
    class WrongReference(CPUHarness):
        def run(self,run_id,request):
            result = super().run(run_id,request)
            if request.mode == "benchmark" and Path(request.source).name == "baseline.cu":
                np.zeros(result["outputs"][0]["bytes"]//4,dtype="<f4").tofile(result["outputs"][0]["path"])
            return result
    config = CampaignConfig(run_id="original-oracle",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=2,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    result = module.run_upstream_campaign(config,WrongReference(tmp_path/"runner"),tmp_path/"original",
        model_backend=OriginalResponses((1,17,63,8),"scan"),source_snapshot=pinned)
    assert result["status"] == "method_failed" and result["candidate"] is None
    assert "independent oracle" in result["original_summary"]["error"]
    assert json.loads((tmp_path/"original"/"baseline-independent-validation.json").read_text())["ok"]


def test_private_original_imports_can_coexist_with_current_modules(pinned):
    import use_harness
    import nodes
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    originals = []
    def inspect_private(index):
        with module.original_modules(pinned,"parallel-test-"+str(index)) as original:
            originals.append(original.use_harness)
            barrier.wait(timeout=10)
            assert __import__("use_harness") is use_harness
            assert __import__("nodes") is nodes
            assert original.nodes.HarnessRuner is original.use_harness.HarnessRuner
            assert original.workflow.OptimizerState is original.state.OptimizerState
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(inspect_private,index) for index in range(2)]
        for future in futures:
            future.result()
    assert originals[0] is not originals[1]


def test_optional_research_bridge_only_exposes_actual_source_events():
    class Backend:
        def generate_json(self,prompt,schema,**kwargs):
            return SimpleNamespace(payload={"text":"Retrieved official guidance"},usage={},
                events=[{"type":"item.completed","item":{"type":"web_search","action":{"sources":[
                    {"url":"https://docs.nvidia.com/cuda/"}]}}}])
    output = module.OriginalCodexTransport(Backend()).create(input="original research prompt",tools=[{}]).model_dump()
    assert output["output"][0]["type"] == "web_search_call"
    assert output["output"][0]["action"]["sources"] == [{"url":"https://docs.nvidia.com/cuda/"}]


def test_original_transport_retry_count_survives_resume_without_reissuing_exhausted_calls(tmp_path):
    class FailingBackend:
        calls = 0
        def generate_json(self,*args,**kwargs):
            self.calls += 1
            raise module.CodexError("simulated provider failure")
    backend = FailingBackend()
    transport = module.OriginalCodexTransport(backend,max_retries=2,directory=tmp_path)
    with pytest.raises(module.CodexError):
        transport._call("original prompt",module.CHAT_SCHEMA)
    assert backend.calls == 3
    resumed = module.OriginalCodexTransport(backend,max_retries=2,directory=tmp_path)
    with pytest.raises(module.CodexError):
        resumed._call("original prompt",module.CHAT_SCHEMA)
    assert backend.calls == 3
    assert resumed.usage()["model_calls"] == 3
    assert not resumed.usage()["usage_complete"]


def fake_completed_cli(tmp_path,monkeypatch):
    """Real CodexBackend receipt path with Popen simulated, never a process."""
    from optimizer_agent import codex_backend
    calls = []
    directory = tmp_path/"model-transport"
    payload = {"kind":"final","text":"frozen original reply","tool_calls":[]}
    usage = {"input_tokens":120,"cached_input_tokens":20,"output_tokens":30}
    class Process:
        returncode = 0
        def __init__(self,command,**kwargs):
            calls.append(command)
            self.output = Path(command[command.index("--output-last-message")+1])
            # The intention physically exists before the simulated CLI starts.
            intents = list((directory/"intents").glob("*/*/intent.json"))
            assert len(intents) == 1
            intent = json.loads(intents[0].read_text())
            assert intent["request"]["prompt"] == "exact original conversation"
            assert intent["request"]["schema"] == module.CHAT_SCHEMA
            assert Path(intent["trace_directory"]) == intents[0].parent/"cli"
        def communicate(self,prompt,timeout):
            assert prompt == "exact original conversation"
            self.output.write_text(json.dumps(payload))
            return "\n".join(json.dumps(event) for event in [
                {"type":"thread.started","thread_id":"simulated-original-cli"},
                {"type":"turn.completed","usage":usage}]),""
    monkeypatch.setattr(codex_backend.subprocess,"Popen",Process)
    backend = module.CodexBackend(module.CodexConfig(executable="never-executed-codex",trace_dir=tmp_path/"unscoped-model"))
    return backend,directory,calls,payload,usage


def test_completed_cli_before_transport_ack_recovers_exact_receipt_without_a_second_prompt(tmp_path,monkeypatch):
    backend,directory,calls,payload,usage = fake_completed_cli(tmp_path,monkeypatch)
    transport = module.OriginalCodexTransport(backend,directory=directory)
    def crash_before_ack(*args,**kwargs):
        raise KeyboardInterrupt("CLI receipt completed before original calls.json acknowledgement")
    monkeypatch.setattr(transport,"_ack",crash_before_ack)
    with pytest.raises(KeyboardInterrupt):
        transport._call("exact original conversation",module.CHAT_SCHEMA)
    assert len(calls) == 1 and not (directory/"calls.json").exists()
    traces = list((directory/"intents").glob("*/*/cli/*.json"))
    assert len(traces) == 1 and json.loads(traces[0].read_text())["status"] == "completed"
    resumed = module.OriginalCodexTransport(backend,directory=directory)
    result = resumed._call("exact original conversation",module.CHAT_SCHEMA)
    assert result.payload == payload and result.call_id == traces[0].stem
    assert result.usage["input_tokens"] == usage["input_tokens"]
    assert len(calls) == 1 and resumed.usage()["model_calls"] == 1
    assert resumed.usage()["usage_complete"] is True
    assert not resumed.usage()["uncertain_model_intents"]
    assert len(json.loads((directory/"calls.json").read_text())) == 1


def test_reply_before_calls_json_commit_is_also_reconciled_without_backend(tmp_path,monkeypatch):
    class Backend:
        calls = 0
        def generate_json(self,*args,**kwargs):
            self.calls += 1
            return SimpleNamespace(payload={"kind":"final","text":"saved","tool_calls":[]},
                usage={"input_tokens":3,"output_tokens":4},events=[],call_id="already-complete")
    backend = Backend()
    transport = module.OriginalCodexTransport(backend,directory=tmp_path)
    write = module.durable_json
    def lose_ack(path,value):
        if Path(path) == tmp_path/"calls.json":
            raise KeyboardInterrupt("transport acknowledgement lost")
        write(path,value)
    monkeypatch.setattr(module,"durable_json",lose_ack)
    with pytest.raises(KeyboardInterrupt):
        transport._call("original conversation",module.CHAT_SCHEMA)
    monkeypatch.setattr(module,"durable_json",write)
    resumed = module.OriginalCodexTransport(backend,directory=tmp_path)
    assert resumed._call("original conversation",module.CHAT_SCHEMA).call_id == "already-complete"
    assert backend.calls == 1


def test_changed_durable_original_reply_cannot_be_reused_or_issue_a_new_prompt(tmp_path,monkeypatch):
    backend,directory,calls,_,_ = fake_completed_cli(tmp_path,monkeypatch)
    transport = module.OriginalCodexTransport(backend,directory=directory)
    transport._call("exact original conversation",module.CHAT_SCHEMA)
    path = directory/"calls.json"
    records = json.loads(path.read_text())
    records[0]["result"]["payload"]["text"] = "unexpected replacement"
    path.write_text(json.dumps(records))
    resumed = module.OriginalCodexTransport(backend,directory=directory)
    with pytest.raises(module.OriginalTransportUncertain):
        resumed._call("exact original conversation",module.CHAT_SCHEMA)
    assert len(calls) == 1


@pytest.mark.parametrize("case",["missing","partial","failed","wrong_model","wrong_effort","wrong_purpose","wrong_payload","wrong_usage","missing_completion","ambiguous"])
def test_unacknowledged_cli_intent_without_exact_completion_fails_closed_before_any_new_prompt(tmp_path,monkeypatch,case):
    backend,directory,calls,_,_ = fake_completed_cli(tmp_path,monkeypatch)
    transport = module.OriginalCodexTransport(backend,directory=directory)
    monkeypatch.setattr(transport,"_ack",lambda *args,**kwargs:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        transport._call("exact original conversation",module.CHAT_SCHEMA)
    trace = next((directory/"intents").glob("*/*/cli/*.json"))
    receipt = json.loads(trace.read_text())
    if case == "missing":trace.unlink()
    elif case == "partial":trace.write_text('{"status":"completed",')
    elif case == "ambiguous":trace.with_name("another-call.json").write_text(json.dumps(receipt))
    else:
        if case == "failed":receipt["status"] = "failed"
        elif case == "wrong_model":receipt["model"] = "another-model"
        elif case == "wrong_effort":receipt["reasoning_effort"] = "low"
        elif case == "wrong_purpose":receipt["purpose"] = "research"
        elif case == "wrong_payload":receipt["payload"] = {"unrelated":"payload"}
        elif case == "wrong_usage":receipt["usage"]["input_tokens"] += 1
        elif case == "missing_completion":receipt["events"] = []
        trace.write_text(json.dumps(receipt))
    resumed = module.OriginalCodexTransport(backend,directory=directory)
    # Even a different graph prompt must not bypass an uncertain older intent.
    with pytest.raises(module.OriginalTransportUncertain):
        resumed._call("new prompt must not be issued",module.CHAT_SCHEMA)
    assert len(calls) == 1 and resumed.usage()["usage_complete"] is False
    assert resumed.usage()["model_calls"] == 1


def test_model_exception_after_valid_initial_preserves_best_but_reports_method_failure(tmp_path,pinned):
    class FailureAfterInitial(OriginalResponses):
        def generate_json(self,prompt,schema,**kwargs):
            if len(self.prompts) >= 2:
                raise module.CodexError("simulated provider failure after initial kernel")
            return super().generate_json(prompt,schema,**kwargs)
    config = CampaignConfig(run_id="original-method-failed",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=2,max_repairs_per_trial=0,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = FailureAfterInitial((1,17,63,8),"scan")
    result = module.run_upstream_campaign(config,CPUHarness(tmp_path/"runner"),tmp_path/"original",
        model_backend=backend,source_snapshot=pinned)
    assert result["generation_complete"] and result["status"] == "method_failed"
    assert result["method_status"] == "failed" and "provider failure" in result["method_error"]
    assert result["method_termination"] == "external_exception"
    assert result["original_summary"]["status"] == "failed"
    assert result["candidate"] and result["retained_best_available"] and result["search_validation"]["ok"]
    assert result["candidate_fallback"] == {"retained":True,"selected_iteration":0,
        "selected_origin":"supplied_initial","reason":"method_failed"}


def test_controlled_final_compilation_failure_retains_original_successful_best_semantics(tmp_path,pinned):
    config = CampaignConfig(run_id="original-invalid-last",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=1,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    result = module.run_upstream_campaign(config,CPUHarness(tmp_path/"runner"),tmp_path/"original",
        model_backend=OriginalResponses((1,17,63,8),"scan",repair=True),source_snapshot=pinned)
    assert result["status"] == "generated" and result["method_status"] == "completed"
    assert result["method_error"] is None and result["candidate_fallback"] is None
    assert result["original_summary"]["best_iteration"] == 0
    assert result["original_summary"]["termination"] == "iteration_limit"


def test_independent_first_gain_observer_preserves_original_choices_and_messages(tmp_path,pinned):
    config = CampaignConfig(run_id="original-observer",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=3,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = OriginalResponses((1,17,63,8),"scan",repair=True)
    runner = CPUHarness(tmp_path/"runner")
    result = module.run_upstream_campaign(config,runner,tmp_path/"original",model_backend=backend,
        source_snapshot=pinned,observe_first_gain=True)
    assert result["status"] == "generated", result
    assert result["original_summary"]["best_iteration"] == 2
    assert result["metrics"]["first_useful_gain_status"] == "observed"
    assert result["metrics"]["model_tokens_to_first_useful_gain"] == 500
    assert result["usage"]["model_tokens"] == 625
    state = json.loads((tmp_path/"original/gain-observer/summary.json").read_text())
    assert len(state["observations"]) == 1  # Invalid proposal is never observed.
    receipt = next(iter(state["observations"].values()))
    assert receipt["decision"] == "promoted" and receipt["observer_gpu_requests"] > 0
    assert receipt["confirmation"]["interval"]["lower"] == 2
    assert result["usage"]["combined_method_and_observer_gpu_wall_seconds"] > result["usage"]["original_harness_elapsed_seconds"]
    assert any(run_id.startswith("observer-") for run_id,_ in runner.calls)
    assert all("original-observer-2" not in prompt and "observer_only" not in prompt
               and "paired confirmation" not in prompt for prompt in backend.prompts)
    count = len(runner.calls)
    assert module.run_upstream_campaign(config,runner,tmp_path/"original",resume=True,
        model_backend=backend,source_snapshot=pinned,observe_first_gain=True) == result
    assert len(runner.calls) == count
    with pytest.raises(ValueError,match="observation policy"):
        module.run_upstream_campaign(config,runner,tmp_path/"original",resume=True,
            model_backend=backend,source_snapshot=pinned,observe_first_gain=False)


def test_original_resume_observes_completed_checkpoint_before_next_model_action(tmp_path,pinned,monkeypatch):
    from optimizer_agent.gain_observer import GainObserver
    config = CampaignConfig(run_id="original-observer-resume",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=3,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = OriginalResponses((1,17,63,8),"scan",repair=True)
    runner = CPUHarness(tmp_path/"runner")
    observe = GainObserver.observe
    def crash_before_intent(*args,**kwargs):
        raise KeyboardInterrupt("controller stopped after completed original checkpoint")
    monkeypatch.setattr(GainObserver,"observe",crash_before_intent)
    with pytest.raises(KeyboardInterrupt):
        module.run_upstream_campaign(config,runner,tmp_path/"original",model_backend=backend,
            source_snapshot=pinned,observe_first_gain=True)
    assert len(backend.prompts) == 4
    monkeypatch.setattr(GainObserver,"observe",observe)
    result = module.run_upstream_campaign(config,runner,tmp_path/"original",resume=True,model_backend=backend,
        source_snapshot=pinned,observe_first_gain=True)
    assert result["status"] == "generated" and result["metrics"]["first_useful_gain_status"] == "observed"
    assert result["metrics"]["model_tokens_to_first_useful_gain"] == 500
    assert result["usage"]["model_tokens"] == 625


def test_original_observer_infrastructure_failure_stops_graph_and_export_gpu_validation(tmp_path,pinned,monkeypatch):
    config = CampaignConfig(run_id="original-observer-incident",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=3,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = OriginalResponses((1,17,63,8),"scan",repair=True)
    runner = CPUHarness(tmp_path/"runner")
    calls_at_failure = []
    def failure(*args,**kwargs):
        calls_at_failure.append(len(runner.calls))
        return {"ok":False,"status":"infrastructure_failure","context_reusable":False}
    monkeypatch.setattr(module.common_evaluation.Evaluator,"confirm",failure)
    result = module.run_upstream_campaign(config,runner,tmp_path/"original",model_backend=backend,
        source_snapshot=pinned,observe_first_gain=True)
    assert result["status"] == "method_failed" and result["candidate"]
    assert result["method_status"] == "failed" and result["method_termination"] == "external_exception"
    assert result["search_validation"]["status"] == "infrastructure_failure"
    assert result["metrics"]["first_useful_gain_status"] == "unknown_incident"
    assert len(backend.prompts) == 4 and len(runner.calls) == calls_at_failure[-1]
