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


def test_interrupted_original_graph_resumes_real_checkpoint_and_usage(tmp_path,pinned):
    class InterruptedResponses(OriginalResponses):
        interrupted = False
        def generate_json(self,*args,**kwargs):
            if len(self.prompts) == 1 and not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt("simulated process interruption before next original model call")
            return super().generate_json(*args,**kwargs)
    config = CampaignConfig(run_id="original-resume",workload="scan",search_sizes=(8,),final_sizes=(8,),
        max_trials=2,benchmark_series=1,confirmation_batches=2,launches_per_series=2,warmup=0,profile=False)
    backend = InterruptedResponses((1,17,63,8),"scan")
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
    assert result["generation_complete"] and result["status"] == "generation_failed"
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
    assert result["status"] == "generation_failed" and result["candidate"] is None
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
