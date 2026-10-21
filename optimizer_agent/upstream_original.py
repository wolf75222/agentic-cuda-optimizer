"""Execute the actual pinned upstream LangGraph as the original-system baseline.

The upstream Python sources are read from Git and verified byte-for-byte before
loading. Its prompts, nodes, tool routing, geometric-mean choices, early stop,
repair routing and 8*(max_retries+1) call guard are executed unchanged. Adapters
replace ChatOpenAI/OpenAI transport with local Luna Max, and the harness process
with the borrowed ROMEO runner. A checkpointer is attached at compile time.

Supplied cases/reference/initial source use the original public input interface.
Reference outputs must additionally match the independent oracle. Exported
launches remain an exact fixed-shape table, without inferred extrapolation.
Generation does not expose heldout data or certify the artifact.
"""
from __future__ import annotations

import contextlib
import builtins
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from types import ModuleType, SimpleNamespace
import uuid

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph

from .codex_backend import CodexBackend, CodexConfig, CodexError, schema_object
from .frozen_launch import FrozenShapeLaunchRule

REPO = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "1e9464da54dfc651337a97c3643bfeefec712bc1"
_MODULE_LOCK = threading.RLock()
# Upstream has absolute module imports. Capture current evaluator types before
# loading those original sources into their separate private namespace.
if str(REPO/"optimizer_agent") not in sys.path:
    sys.path.append(str(REPO/"optimizer_agent"))
import benchmark_workloads as common_workloads
import evaluation as common_evaluation
import use_harness as common_harness


def write_json(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    temporary.write_text(json.dumps(value,indent=2,default=str,allow_nan=False))
    temporary.replace(path)


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def _git(repo,*arguments):
    return subprocess.run(["git","-C",str(repo),*arguments],check=True,capture_output=True).stdout


def snapshot_upstream(destination=None,repo_dir=REPO):
    """Materialize and authenticate the original sources, never the working tree."""
    repo_dir = Path(repo_dir).resolve()
    resolved = _git(repo_dir,"rev-parse",UPSTREAM_COMMIT+"^{commit}").decode().strip()
    if resolved != UPSTREAM_COMMIT:
        raise RuntimeError("Required upstream commit identity differs")
    root = Path(destination or repo_dir/"work"/("original-"+UPSTREAM_COMMIT[:12])).resolve()
    paths = _git(repo_dir,"ls-tree","-r","--name-only",UPSTREAM_COMMIT,"optimizer_agent").decode().splitlines()
    hashes = {}
    for relative in paths:
        if not relative.endswith(".py"):
            continue
        source = _git(repo_dir,"show",UPSTREAM_COMMIT+":"+relative)
        expected = hashlib.sha256(source).hexdigest()
        target = root/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            temporary = target.with_name(target.name+"."+uuid.uuid4().hex+".tmp")
            temporary.write_bytes(source)
            try:
                # Publish a complete inode exclusively. Concurrent generators
                # cannot observe a partially written shared source snapshot.
                os.link(temporary,target)
            except FileExistsError:
                pass
            finally:
                temporary.unlink()
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Pinned upstream snapshot changed: {relative}")
        hashes[relative] = expected
    if not {"optimizer_agent/workflow.py","optimizer_agent/nodes.py"} <= set(hashes):
        raise RuntimeError("Incomplete upstream snapshot")
    identity = {"commit":UPSTREAM_COMMIT,"source_hashes":hashes,"source_identity":content_hash(hashes),"path":str(root)}
    write_json(root/"upstream-pin.json",identity)
    return root,identity


@contextlib.contextmanager
def original_modules(snapshot,namespace_key=None):
    """Execute original bytes in a private namespace, preserving absolute imports.

    A scoped Python importer resolves only the original source's own module
    names. Current module aliases and sys.path stay unchanged, including when
    other campaign modes execute in parallel threads.
    """
    source = Path(snapshot)/"optimizer_agent"
    names = {path.stem for path in source.glob("*.py")}-{"optimizer_agent","openai_test","__init__"}
    namespace = "_cuda_optimizer_original_"+content_hash(str(namespace_key or source.resolve()))[:20]
    loaded = {}
    def scoped_import(name,globals=None,locals=None,fromlist=(),level=0):
        if level == 0 and name in names:
            return load(name)
        return builtins.__import__(name,globals,locals,fromlist,level)
    def load(name):
        if name in loaded:
            return loaded[name]
        full_name = namespace+"."+name
        spec = importlib.util.spec_from_file_location(full_name,source/(name+".py"))
        module = importlib.util.module_from_spec(spec)
        module.__builtins__ = dict(vars(builtins),__import__=scoped_import)
        loaded[name] = module
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module
    with _MODULE_LOCK:
        if namespace in sys.modules:
            raise RuntimeError("This original campaign namespace is already active")
        package = ModuleType(namespace)
        package.__path__ = [str(source)]
        sys.modules[namespace] = package
    try:
        yield SimpleNamespace(**{name:load(name) for name in
            ("workflow","nodes","agent_workload","optimizer_config","state","use_harness")})
    finally:
        for name in loaded:
            sys.modules.pop(namespace+"."+name,None)
        sys.modules.pop(namespace,None)


CHAT_SCHEMA = schema_object({"kind":{"type":"string","enum":["final","tool_calls"]},
    "text":{"type":"string"},"tool_calls":{"type":"array","items":schema_object({
        "name":{"type":"string"},"arguments_json":{"type":"string"}})}})


class OriginalCodexTransport:
    """Lossless chat/tool-message bridge; decisions remain upstream JSON schemas."""
    def __init__(self,backend,max_retries=2,directory=None):
        self.backend,self.max_retries = backend,max_retries
        self.directory = Path(directory) if directory else None
        self.tools = []
        self.calls = []
        self.records = []
        if self.directory and (self.directory/"calls.json").exists():
            self.records = json.loads((self.directory/"calls.json").read_text())
            self.calls = [SimpleNamespace(**record["result"]) for record in self.records if record.get("result")]
        self.responses = self

    def bind_tools(self,tools):
        self.tools = tools
        return self

    def _call(self,prompt,schema,*,purpose="proposal",max_web_search_calls=None):
        # Message IDs are assigned in the original graph checkpoint. They make
        # reference/initial conversations distinct even when source text matches,
        # while replaying the exact interrupted graph task reuses its receipt.
        key = content_hash({"prompt":prompt,"schema":schema,"purpose":purpose,"search":max_web_search_calls})
        for record in self.records:
            if record["key"] == key and record.get("result"):
                return SimpleNamespace(**record["result"])
        previous_failures = [record for record in self.records if record["key"] == key and record.get("error")]
        if len(previous_failures) > self.max_retries:
            raise CodexError(previous_failures[-1]["error"])
        for attempt in range(len(previous_failures),self.max_retries+1):
            try:
                result = self.backend.generate_json(prompt,schema,purpose=purpose,max_web_search_calls=max_web_search_calls)
                self.calls.append(result)
                self.records.append({"key":key,"attempt":attempt,"purpose":purpose,"result":{
                    name:getattr(result,name,default) for name,default in
                    (("payload",{}),("usage",{}),("events",[]),("model",None),
                     ("reasoning_effort",None),("elapsed_seconds",None),("call_id",None))}})
                if self.directory:
                    write_json(self.directory/"calls.json",self.records)
                return result
            except CodexError as error:
                self.records.append({"key":key,"attempt":attempt,"purpose":purpose,"error":str(error),"usage":None})
                if self.directory:
                    write_json(self.directory/"calls.json",self.records)
                if attempt == self.max_retries:
                    raise

    def invoke(self,messages):
        tool_specs = [{"name":tool.name,"description":tool.description,
            "parameters":tool.args_schema.model_json_schema()} for tool in self.tools]
        prompt = ("Transport the following original chat conversation without changing its task. "
            "Return kind=tool_calls and arguments_json for calls to the listed tools, or kind=final "
            "with text containing the exact response requested by the conversation. "
            "The final text can itself be a JSON document matching the original supplied schema.\n"+
            json.dumps({"messages":[message.model_dump(mode="json") for message in messages],"available_tools":tool_specs},default=str))
        result = self._call(prompt,CHAT_SCHEMA)
        response = result.payload
        if response["kind"] == "tool_calls":
            allowed = {tool.name for tool in self.tools}
            calls = []
            for call in response["tool_calls"]:
                arguments = json.loads(call["arguments_json"])
                if call["name"] not in allowed or not isinstance(arguments,dict):
                    raise ValueError("Codex requested an unavailable original tool")
                calls.append({"id":uuid.uuid4().hex,"name":call["name"],"args":arguments,"type":"tool_call"})
            if not calls:
                raise ValueError("Original tool-call response is empty")
            return AIMessage(content=response["text"],tool_calls=calls)
        if response["tool_calls"]:
            raise ValueError("Final chat response cannot contain unexecuted tool calls")
        return AIMessage(content=response["text"])

    def create(self,**request):
        # Supplied signature/source/cases skip these original preparation model
        # calls. Optional original NVIDIA research can still use its own prompts.
        result = self._call(request["input"],schema_object({"text":{"type":"string"}}),
            purpose="research" if request.get("tools") else "proposal",
            max_web_search_calls=request.get("max_tool_calls"))
        output_items = []
        for event in result.events:
            item = event.get("item",event)
            if item.get("type") in ("web_search_call","web_search"):
                # Translate the provider event envelope only. Missing retrieved
                # sources stay missing, so the original research guard fails.
                output_items.append(dict(item,type="web_search_call"))
        output = {"output_text":result.payload["text"],"output":output_items,"codex_events":result.events}
        return SimpleNamespace(output_text=output["output_text"],model_dump=lambda **kwargs:output)

    def usage(self):
        totals = {"model_calls":len(self.records),"input_tokens":0,"cached_input_tokens":0,"output_tokens":0,
                  "usage_complete":all(record.get("result") for record in self.records)}
        for call in self.calls:
            for key in ("input_tokens","cached_input_tokens","output_tokens"):
                totals[key] += call.usage.get(key,0)
        totals["model_tokens"] = totals["input_tokens"]+totals["output_tokens"]
        return totals


class OriginalHarnessTransport:
    """Legacy request transport, with authenticated independent reference output."""
    def __init__(self,runner,directory,cases,suite,evaluator):
        self.runner,self.results_dir,self.cases,self.suite,self.evaluator = runner,Path(directory),cases,suite,evaluator
        path = self.results_dir/"transport.json"
        self.calls = json.loads(path.read_text()) if path.exists() else []

    def get_device_properties(self,device):
        return self.runner.get_device_properties(device)

    def run(self,run_id,request):
        fields = asdict(request)
        fields["arguments"] = common_harness.parse_arguments(fields["arguments"])
        adapted = common_harness.KernelRequest(**fields)
        attempt_id = content_hash({"original_run_id":run_id,"request":asdict(adapted),
            "source_hash":hashlib.sha256(Path(adapted.source).read_bytes()).hexdigest()})
        directory = self.results_dir/"transport"/attempt_id
        if (directory/"response.json").exists():
            return json.loads((directory/"response.json").read_text())
        write_json(directory/"request.json",{"original_run_id":run_id,"attempt_id":attempt_id,"request":asdict(adapted)})
        started = time.monotonic()
        report = self.runner.run("original-"+attempt_id,adapted)
        if report.get("ok") and not self.evaluator._check_environment(report):
            report = dict(report,ok=False,stage="environment",error="Original harness target identity changed")
        if run_id.startswith("reference-case-") and report.get("ok"):
            case_index = int(run_id.rsplit("-",1)[1])
            case = self.cases[case_index]
            check = self.evaluator._compare(case,report,self.suite.oracle(case))
            report = dict(report,external_reference_validation=check)
            if not check["ok"]:
                report.update(ok=False,stage="validation",error="Original reference output disagrees with independent oracle")
        write_json(directory/"response.json",report)
        self.calls.append({"original_run_id":run_id,"attempt_id":attempt_id,"elapsed_seconds":time.monotonic()-started,
            "ok":bool(report.get("ok")),"receipt":str(directory/"response.json")})
        write_json(self.results_dir/"transport.json",self.calls)
        return report


def _candidate_receipt(candidate):
    return {"candidate_id":candidate.candidate_id,"source":str(candidate.source),"source_path":str(candidate.source),
        "source_hash":hashlib.sha256(candidate.source.read_bytes()).hexdigest(),"artifact_hash":candidate.artifact_hash,
        "kernel":candidate.kernel,"launch_rule":asdict(candidate.launch_rule),"parameters":candidate.parameters}


def run_upstream_campaign(config,runner,directory,*,resume=False,finalize=False,model_backend=None,
                          source_snapshot=None,nvidia_research=False,observe_first_gain=False):
    """Run real original generation on a borrowed ready worker; no submission.

    Final certification is intentionally deferred to the common post-generation
    group evaluator. No final suite is materialized or passed to the generator.
    """
    if finalize:
        raise ValueError("Finalize all frozen campaign modes together with the common independent group evaluator")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    start_path = directory/"generation-start.json"
    started = json.loads(start_path.read_text())["unix_time"] if start_path.exists() else time.time()
    if not start_path.exists():
        write_json(start_path,{"unix_time":started})
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Original campaign already exists; use resume")
        saved = json.loads(manifest_path.read_text())
    config_data = config.to_dict()
    observation_path = directory/"observation-policy.json"
    observation_policy = {"observe_first_gain":bool(observe_first_gain),"feedback_to_method":False}
    if observation_path.exists() and json.loads(observation_path.read_text()) != observation_policy:
        raise ValueError("Original observation policy changed at resume")
    if manifest_path.exists() and not observation_path.exists() and observe_first_gain:
        raise ValueError("Cannot retrospectively enable a first-gain observer on completed generation")
    write_json(observation_path,observation_policy)
    config_path = directory/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(config_data)):
        raise ValueError("Original campaign configuration changed at resume")
    write_json(config_path,config_data)
    snapshot,identity = snapshot_upstream(source_snapshot)
    if manifest_path.exists() and saved.get("generation_complete"):
        if saved.get("upstream",{}).get("source_identity") != identity["source_identity"]:
            raise ValueError("Original source identity changed at resume")
        candidate = saved.get("candidate")
        if candidate and hashlib.sha256(Path(candidate["source_path"]).read_bytes()).hexdigest() != candidate["source_hash"]:
            raise ValueError("Frozen original source changed after generation")
        return saved
    if observe_first_gain:
        from .gain_observer import assert_observer_available
        assert_observer_available(directory/"gain-observer")
    properties = runner.get_device_properties(0)
    device_keys = ("name","architecture","uuid","gpu_uuid","compute_capability","driver_version","nvrtc_version","harness_build_id")
    environment = {key:properties[key] for key in device_keys if key in properties}
    environment_path = directory/"environment.json"
    if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
        raise ValueError("Original campaign target changed at resume")
    write_json(environment_path,environment)
    suite = common_workloads.WorkloadSuite(config.workload,directory/"inputs",config)
    baseline = suite.baseline(REPO/config.baseline_source)
    cases = suite.cases()
    performance = [case for case in cases if case.metadata["n"] in config.search_sizes]
    contract = common_evaluation.WorkloadContract(config.run_id+".original-search",1,config.workload,suite.signature,
        "Independent NumPy WorkloadSuite oracle; original search ranking remains unchanged",
        benchmark_series=config.benchmark_series,confirmation_batches=config.confirmation_batches,
        launches_per_series=config.launches_per_series,warmup=config.warmup,
        confidence=config.confidence_level,min_relative_gain=config.min_relative_gain,
        max_critical_regression=config.max_critical_regression,
        required_sanitizers=config.required_sanitizers,
        metadata_json=json.dumps({"precision":"float32","environment":environment,"domain":"fixed supplied shapes"},sort_keys=True))
    evaluator = common_evaluation.Evaluator(runner,contract,suite.oracle,{"short":cases[:2],"search":cases,"performance":performance})
    validation_cost_path = directory/"independent-validation-costs.json"
    validation_costs = json.loads(validation_cost_path.read_text()) if validation_cost_path.exists() else []
    def measured_validation(candidate,label):
        row = {"label":label,"started_at_unix":time.time(),"status":"pending"}
        validation_costs.append(row)
        write_json(validation_cost_path,validation_costs)
        clock = time.monotonic()
        try:
            return evaluator.validate(candidate,"search")
        finally:
            row.update(status="completed",wall_seconds=time.monotonic()-clock)
            write_json(validation_cost_path,validation_costs)
    baseline_check = measured_validation(baseline,"supplied-baseline-admission")
    write_json(directory/"baseline-independent-validation.json",baseline_check)
    if not baseline_check.get("ok"):
        raise RuntimeError("Original supplied reference failed the independent oracle")
    backend = model_backend or CodexBackend(CodexConfig(model="gpt-6-luna",reasoning_effort="max",
        timeout_seconds=config.model_timeout_seconds,trace_dir=directory/"model"))
    transport = OriginalCodexTransport(backend,config.max_repairs_per_trial,directory/"model-transport")
    harness = OriginalHarnessTransport(runner,directory,cases,suite,evaluator)
    observer = None
    observer_failed = False
    if observe_first_gain:
        from .gain_observer import GainObserver
        observer = GainObserver(evaluator,baseline,directory/"gain-observer",method="true_upstream_original",
            campaign_started_at_unix=started,usage_provider=lambda:{**transport.usage(),
                "method_gpu_evaluation_wall_seconds":sum(call["elapsed_seconds"] for call in harness.calls)
                    +sum(row["wall_seconds"] for row in validation_costs)
                    if all("wall_seconds" in row for row in validation_costs) else None,
                "method_harness_calls":len(harness.calls)})
    from .vanilla_gpu_service import workload_description
    description = workload_description(config.workload)
    with original_modules(snapshot,directory) as original,SqliteSaver.from_conn_string(str(directory/"original-langgraph.sqlite")) as saver:
        original_config = original.optimizer_config.OptimizerConfig(directory,0,config.max_trials,
            config.warmup,config.launches_per_series,"gpt-6-luna",config.max_repairs_per_trial,
            nvidia_research=nvidia_research,use_nsight=False,reasoning_effort="max")
        # Keep all original configuration checks. Only the provider-specific
        # effort vocabulary grows from upstream's low/medium/high to Luna Max.
        replace(original_config,reasoning_effort="high").validate()
        original_cases = []
        for case in cases:
            original_cases.append(original.agent_workload.InputCase.from_dict({
                "arguments":asdict(case.request)["arguments"],"comparisons":[asdict(c) for c in case.comparisons],
                "size_summary":f"n={case.metadata['n']}",
                "purpose":"performance" if case.metadata["n"] in config.search_sizes else "correctness"},directory))
        workload = original.agent_workload.Workload(json.dumps(description,sort_keys=True),suite.signature,original_cases,baseline.source,baseline.source)
        original.workflow.ChatOpenAI = lambda **kwargs:transport
        class DurableOriginalGraph(StateGraph):
            def compile(self,*args,**kwargs):
                kwargs["checkpointer"] = saver
                return super().compile(*args,**kwargs)
        original.workflow.StateGraph = DurableOriginalGraph
        graph = original.workflow.build_graph(transport,harness,original_config)
        graph_config = {"configurable":{"thread_id":config.run_id},
            "recursion_limit":64*(original_config.max_iterations+3)*(original_config.max_retries+1)}
        checkpoint = graph.get_state(graph_config)
        if checkpoint.values and not resume:
            raise ValueError("Original graph checkpoint exists; use resume")
        initial = None if checkpoint.values else dict(config=original_config,session_dir=directory,workload=workload,
            device_properties=properties,research_brief="",kernel_source="",kernel_role="reference",kernel_proposal=None,
            iteration=0,validation_ok=False,reference_results=[],reference_attempt=None,messages=[],optimization_history=[],
            best_attempt=None,proposal_retries=0,model_calls=0,error=None,profiling_unavailable=None,stop_reason=None)
        state = checkpoint.values or initial
        def observe_attempt(current_state):
            if not observer or not current_state["optimization_history"]:
                return
            attempt = current_state["optimization_history"][-1]
            if attempt.iteration <= 0 or original.nodes.geometric_mean_time(attempt) is None:
                return
            observed_source = directory/"observer-candidates"/(str(attempt.iteration)+".cu")
            observed_source.parent.mkdir(exist_ok=True)
            if observed_source.exists() and observed_source.read_text() != attempt.kernel_source:
                raise ValueError("Original observer source changed")
            observed_source.write_text(attempt.kernel_source)
            launches = {str(cases[case.case_index].metadata["n"]):{
                "grid":list(case.request.grid),"block":list(case.request.block),
                "shared_memory_bytes":case.request.shared_memory_bytes} for case in attempt.case_runs}
            observed_candidate = common_evaluation.Candidate("original-observer-"+str(attempt.iteration),
                observed_source,config.workload,FrozenShapeLaunchRule(launches),{})
            # No observer output is ever inserted into original state/messages,
            # history, or its selected incumbent. A checkpoint can precede this
            # callback, so examine it on resume before advancing the graph.
            observer.observe(observed_candidate,evidence_id="original-attempt-"+str(attempt.iteration))
        try:
            observe_attempt(state)
            if not checkpoint.values or checkpoint.next:
                for state in graph.stream(initial,graph_config,stream_mode="values"):
                    write_json(directory/"generation-progress.json",{"iteration":state["iteration"],"model_calls":state["model_calls"],
                        "attempt_count":len(state["optimization_history"]),"error":state["error"]})
                    observe_attempt(state)
        except Exception as error:
            if observer:
                from .gain_observer import ObserverInfrastructureError
                observer_failed = isinstance(error,ObserverInfrastructureError)
            state = dict(state,error=f"{type(error).__name__}: {error}")
            (directory/"error.log").write_text(traceback.format_exc())
            state.update(original.nodes.finalize_result(state))
        best = state.get("best_attempt")
        frozen = None
        validation = None
        if best is not None:
            source = directory/"frozen-original.cu"
            source.write_text(best.kernel_source)
            table = {str(cases[case_run.case_index].metadata["n"]):{
                "grid":list(case_run.request.grid),"block":list(case_run.request.block),
                "shared_memory_bytes":case_run.request.shared_memory_bytes} for case_run in best.case_runs}
            candidate = common_evaluation.Candidate("upstream-original-"+str(best.iteration),source,config.workload,FrozenShapeLaunchRule(table),{})
            frozen = _candidate_receipt(candidate)
            frozen["provenance"] = {"method":"true_upstream_original","upstream":identity,"selected_iteration":best.iteration,
                "selected_origin":"supplied_initial" if best.iteration == 0 else "original_generated_proposal",
                "requested_model":"gpt-6-luna","reasoning_effort":"max","launch_domain":"fixed supplied shapes; no extrapolation"}
            validation = ({"ok":False,"status":"infrastructure_failure",
                "error":"Observer infrastructure incident stopped GPU activity; frozen source retained"}
                if observer_failed else measured_validation(candidate,"selected-export-validation"))
            write_json(directory/"frozen-search-validation.json",validation)
            write_json(directory/"candidate.json",frozen)
        summary = json.loads((directory/"summary.json").read_text()) if (directory/"summary.json").exists() else {}
        generated = [attempt for attempt in state.get("optimization_history",[]) if attempt.iteration > 0]
        validations = [{"iteration":attempt.iteration,"valid":all(case.validation_ok for case in attempt.case_runs),
            "case_results":[{"case_index":case.case_index,"valid":case.validation_ok} for case in attempt.case_runs],
            "evidence":"history.json"} for attempt in generated]
        reports = [json.loads(Path(call["receipt"]).read_text()) for call in harness.calls]
        raw_series = [report["latency_samples_ms"] for report in reports if isinstance(report.get("latency_samples_ms"),list)]
        metrics = {"candidate_configurations_with_recorded_validity":len(validations),
            "correct_candidates":sum(row["valid"] for row in validations),
            "correct_candidate_fraction":sum(row["valid"] for row in validations)/len(validations) if validations else None,
            "candidate_validation_results":validations,"configuration_observations":len(validations),
            "supplied_initial_attempts":sum(attempt.iteration == 0 for attempt in state.get("optimization_history",[])),
            "configurations_independently_validated":int(validation is not None),
            "validation_evidence_basis":"Original case comparisons against independently oracle-admitted reference; selected export separately validated",
            "harness_requests":len(harness.calls),
            "cuda_event_kernel_seconds":sum(sum(series) for series in raw_series)/1000 if len(raw_series) == len(reports) and reports else None,
            "time_to_final_generation_seconds":time.time()-started,
            "time_to_first_useful_gain_seconds":None,"model_tokens_to_first_useful_gain":None,
            "first_useful_gain_status":"not_recorded_by_method",
            "unknown_metrics":["time_to_first_useful_gain_seconds","model_tokens_to_first_useful_gain"],
            "first_useful_gain_note":"Original raw-geomean selection has no common paired promotion receipt; not inferred from means"}
        observer_metrics = observer.snapshot()["metrics"] if observer else None
        if observer_metrics:
            metrics.update(observer_metrics)
            metrics["first_useful_gain_note"] = "Independent observer only; original model/control flow receives no result; elapsed includes observation overhead"
            metrics["unknown_metrics"] = [name for name in metrics["unknown_metrics"] if metrics.get(name) is None]
        manifest = {"run_id":config.run_id,"method":"true_upstream_original","generation_complete":True,
            "status":"generated" if frozen is not None and validation.get("ok") else "generation_failed",
            "certification_status":"not_yet_independently_finalized","candidate":frozen,
            "best_candidate_id":frozen["candidate_id"] if frozen else None,"upstream":identity,
            "original_feature_settings":{"nvidia_research":nvidia_research,"use_nsight":False},
            "original_limits":{"max_iterations":original_config.max_iterations,"max_retries":original_config.max_retries,
                "max_model_calls_per_proposal":8*(original_config.max_retries+1)},
            "model":"gpt-6-luna","reasoning_effort":"max","usage":{
                **transport.usage(),"local_configurations":len(generated),"original_harness_calls":len(harness.calls),
                "original_harness_elapsed_seconds":sum(call["elapsed_seconds"] for call in harness.calls),
                "gain_observer_wall_seconds":observer_metrics["observer_wall_seconds"] if observer_metrics else None,
                "combined_method_and_observer_gpu_wall_seconds":observer_metrics["combined_method_and_observer_gpu_wall_seconds"] if observer_metrics else None},"environment":environment,
            "elapsed_seconds":time.time()-started,"metrics":metrics,"termination_reason":summary.get("termination",state.get("error")),
            "original_summary":summary,"search_validation":validation,"global_resource_caps":None,
            "gain_observer":{"enabled":bool(observer),"receipt":"gain-observer/summary.json" if observer else None,
                "feedback_to_method":False,"includes_observer_overhead":bool(observer)},
            "adapter_source_hashes":{name:hashlib.sha256((REPO/"optimizer_agent"/name).read_bytes()).hexdigest()
                for name in ("upstream_original.py","frozen_launch.py","codex_backend.py","gain_observer.py")},
            "adapter_changes":["local Codex structured chat/tool transport","ROMEO legacy request transport",
                "provider reasoning effort vocabulary extended to max; other original config checks retained",
                "independent supplied reference/oracle admission","durable checkpointer attached at compile",
                "exact shape-table export; final qualification deferred",
                "optional independent gain observer outside original state/messages/selection"]}
        write_json(manifest_path,manifest)
        return manifest
