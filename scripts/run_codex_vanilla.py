#!/usr/bin/env python3
"""One ordinary local Codex optimization prompt with an owned ROMEO GPU tool.

The default is a read-only plan. --run explicitly starts one isolated service
and one native Codex CLI session. There is no campaign graph, search policy,
mandatory research, tuning controller, or global model/resource timeout here.
The resulting artifact requires an independent evaluation after generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "optimizer_agent"))
sys.path.insert(0, str(REPO))

from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.codex_backend import default_codex_executable
from optimizer_agent.vanilla_gpu_service import OPERATIONS, baseline_source, workload_description

MODEL = "gpt-6-luna"
EFFORT = "max"
LAUNCH_SCHEMA = {
    "type":"object", "additionalProperties":False,
    "properties":{
        "kind":{"type":"string","enum":["elementwise","square_2d","fixed"]},
        "block":{"type":"array","items":{"type":"integer","minimum":1},"minItems":3,"maxItems":3},
        "grid":{"type":"array","items":{"type":"integer","minimum":1},"minItems":3,"maxItems":3},
        "elements_key":{"type":"string","enum":["n"]},
        "shared_memory_bytes":{"type":"integer","minimum":0},
        "version":{"type":"integer","enum":[1]},
        "tile_x":{"anyOf":[{"type":"integer","minimum":1},{"type":"null"}]},
        "tile_y":{"anyOf":[{"type":"integer","minimum":1},{"type":"null"}]},
    },
    "required":["kind","block","grid","elements_key","shared_memory_bytes","version","tile_x","tile_y"],
}
OUTPUT_SCHEMA = {
    "type":"object","additionalProperties":False,
    "properties":{
        "source":{"type":"string"},"kernel":{"type":"string","enum":["matmul"]},
        "launch_rule":LAUNCH_SCHEMA,
        "parameters":{"type":"array","items":{"type":"object","additionalProperties":False,
            "properties":{"name":{"type":"string"},"value":{"type":"integer"}},"required":["name","value"]}},
    },
    "required":["source","kernel","launch_rule","parameters"],
}


def output_schema(operation="matmul"):
    workload_description(operation)
    schema = json.loads(json.dumps(OUTPUT_SCHEMA))
    schema["properties"]["kernel"]["enum"] = [operation]
    return schema


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary,path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def build_command(executable, workspace, protected):
    return [str(executable),"-a","never","exec","--ignore-user-config",
        "--ephemeral","--json","--skip-git-repo-check","--sandbox","workspace-write",
        "-C",str(workspace),"-m",MODEL,"-c",f'model_reasoning_effort="{EFFORT}"',
        "-c","project_doc_max_bytes=0","-c","sandbox_workspace_write.network_access=true",
        "-c","features.shell_tool=true","--output-schema",str(protected/"output-schema.json"),
        "--output-last-message",str(protected/"last-message.json"),"-"]


def plan(config, output, session_id, executable):
    if config.workload not in OPERATIONS:
        raise ValueError("Unsupported literal vanilla workload")
    root = Path(output).resolve()
    return {"mode":"plan","submitted":False,"model_started":False,
        "model":MODEL,"reasoning_effort":EFFORT,"worker_session":session_id,
        "workload":config.workload,
        "workspace":str(root/"workspace"),"protected_evidence":str(root/"evidence"),
        "command":build_command(executable,root/"workspace",root/"evidence"),
        "global_resource_caps":None,"certification_status":"not_evaluated_on_independent_final_data"}


def tool_instructions(metadata):
    command = shlex.join(metadata["tool_command"])
    description = metadata.get("workload_description") or workload_description(metadata.get("operation", "matmul"))
    launch = json.dumps(description["baseline_launch_rule"])
    return f"""# GPU tools

The owned ROMEO service accepts properties, correctness validation, and benchmark requests.
SERVICE_ENDPOINT and SERVICE_TOKEN are already set in your command environment.
Use these commands for GPU actions; the service compiles and runs your source remotely:

    {command} properties
    {command} validate --source "$PWD/kernel.cu" --launch "$PWD/launch.json"
    {command} benchmark --source "$PWD/kernel.cu" --launch "$PWD/launch.json"

The launch file is either a LaunchRule object or an object with launch_rule and
parameters. A launch matching the initial naive kernel is:

```json
{launch}
```

For square_2d, each grid axis is ceil(n / (tile_axis or block_axis)); fixed uses
the literal grid; elementwise derives grid.x from n and tile_x or block.x.
The service supplies independent inputs and references; these are outside this
workspace. Save proposed code and launch metadata in this workspace. Use the GPU
service for mutable GPU operations. Do not submit or cancel jobs through SSH or
sbatch, access other campaigns, or alter oracle/service/certification evidence.

Target properties:
```json
{json.dumps(metadata['properties'],indent=2)}
```

Fixed workload: {description['operation']}
ABI: `{description['signature']}`
Definition: {description['mathematics']}
Memory contract:
```json
{json.dumps(description['memory_policy'],indent=2)}
```
"""


def optimization_prompt(operation="matmul"):
    description = workload_description(operation)
    return f"""Optimize the naive CUDA {description['title']} in baseline.cu for the GPU described in GPU_TOOLS.md.
The signature is {description['signature']}.
The exact operation is: {description['mathematics']}.
Storage and bounds: {description['memory_policy']['sizes']}.
A, B and C are disjoint device allocations; A and B are read-only. C is output-only and must be fully overwritten without reading previous C.
Preserve full float32 arithmetic, input values, a single kernel, and all positive n including tails.
Numerical validation uses atol=2e-4 and rtol=2e-4. Double/lower-precision/tensor-core/TF32 arithmetic is outside this contract.
You have ordinary shell tools and the GPU service documented in GPU_TOOLS.md. Use them as useful to optimize and check your code.
Save the resulting kernel.cu and launch.json in this workspace, then return the complete source, kernel name, deterministic launch rule and effective integer parameters in the requested JSON format.
"""


def observed_models(value):
    models = set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key in {"model","model_id","model_slug"} and isinstance(item,str):
                models.add(item)
            elif isinstance(item,(dict,list)):
                models.update(observed_models(item))
    elif isinstance(value,list):
        for item in value:
            models.update(observed_models(item))
    return models


def normalized_candidate(payload, protected, operation="matmul"):
    if not isinstance(payload,dict) or set(payload) != set(OUTPUT_SCHEMA["required"]):
        raise ValueError("Codex final response does not match the source/launch schema")
    if not isinstance(payload["source"],str) or not payload["source"].strip() or payload["kernel"] != operation:
        raise ValueError("Codex omitted source or changed the kernel ABI")
    from evaluation import Candidate, LaunchRule
    launch = LaunchRule(**payload["launch_rule"])
    parameters = {}
    if not isinstance(payload["parameters"],list):
        raise ValueError("Effective parameters must be an array of integer name/value pairs")
    for parameter in payload["parameters"]:
        name,value = parameter["name"],parameter["value"]
        if (not isinstance(name,str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",name)
                or name in parameters or type(value) is not int):
            raise ValueError("Invalid or repeated effective parameter")
        parameters[name] = value
    source = protected/"kernel.cu"
    source.write_text(payload["source"])
    candidate = Candidate("codex-vanilla",source,operation,launch,parameters)
    return {"candidate_id":candidate.candidate_id,"source":str(source),"source_path":str(source),"source_hash":file_hash(source),
        "artifact_hash":candidate.artifact_hash,"kernel":operation,"launch_rule":payload["launch_rule"],
        "parameters":parameters,"certification_status":"not_evaluated_on_independent_final_data"}


def run_vanilla(config, output, session_id, executable, *, service_factory=None,
                popen_factory=subprocess.Popen, version_runner=subprocess.run):
    description = plan(config,output,session_id,executable)
    workspace,protected = Path(description["workspace"]),Path(description["protected_evidence"])
    if workspace.exists() or protected.exists():
        raise ValueError("Use a new output directory for each native Codex session")
    workspace.mkdir(parents=True)
    protected.mkdir(parents=True)
    (workspace/"baseline.cu").write_text(baseline_source(config, REPO))
    write_json(protected/"output-schema.json",output_schema(config.workload))
    prompt = optimization_prompt(config.workload)
    (protected/"prompt.txt").write_text(prompt)
    version = version_runner([str(executable),"--version"],capture_output=True,text=True,check=True,timeout=10)
    execution_sources = [Path(__file__),REPO/"scripts/vanilla_gpu_tool.py"]
    execution_sources += [REPO/"optimizer_agent"/name for name in
        ("vanilla_gpu_service.py","evaluation.py","benchmark_workloads.py","execution_backend.py",
         "remote_worker.py","gpu_scheduler.py","compile_cache.py","codex_backend.py")]
    execution_sources += sorted((REPO/"cuda_test_harness").glob("*.h"))
    execution_sources += sorted((REPO/"cuda_test_harness").glob("*.cpp"))
    receipt = {**description,"mode":"run","status":"starting","command":description["command"],
        "codex_version":version.stdout.strip(),"started_at":time.time(),"usage_events":[],
        "usage":{"input_tokens":0,"cached_input_tokens":0,"output_tokens":0},
        "actual_models":[],"thread_ids":[],"cost_usd":None,
        "code_hashes":{str(path.relative_to(REPO)):file_hash(path) for path in execution_sources if path.is_file()},
        "baseline_source_hash":file_hash(workspace/"baseline.cu")}
    receipt["execution_code_hash"] = hashlib.sha256(json.dumps(receipt["code_hashes"],sort_keys=True).encode()).hexdigest()
    if Path(executable).is_file():
        receipt["codex_executable_sha256"] = file_hash(executable)
    write_json(protected/"receipt.json",receipt)
    if service_factory is None:
        from optimizer_agent.vanilla_gpu_service import VanillaGPUService
        service_factory = VanillaGPUService
    service = service_factory(config,protected,workspace,session_id,repo_dir=REPO)
    process = None
    last_message = None
    try:
        receipt["submitted"] = None  # Submission may succeed before start returns.
        metadata = service.start()
        receipt.update(submitted=metadata.get("owns_worker", True),
            worker_ownership=metadata.get("worker_ownership", "owned"),
            service_endpoint=metadata["endpoint"],target_properties=metadata["properties"])
        (workspace/"GPU_TOOLS.md").write_text(tool_instructions(metadata))
        environment = dict(os.environ,SERVICE_ENDPOINT=metadata["endpoint"],SERVICE_TOKEN=metadata["token"])
        # Shell GPU access goes through the owned service. The controller's SSH
        # agent is not forwarded into the child command environment.
        environment.pop("SSH_AUTH_SOCK",None)
        with (protected/"codex.jsonl").open("w") as raw,(protected/"stderr.log").open("w") as errors:
            process = popen_factory(description["command"],cwd=workspace,env=environment,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,start_new_session=True)
            receipt.update(status="running",model_started=True,child_pid=process.pid)
            write_json(protected/"receipt.json",receipt)
            process.stdin.write(prompt)
            process.stdin.close()
            for line in process.stdout:
                raw.write(line)
                raw.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event,dict):
                    continue
                receipt["actual_models"] = sorted(set(receipt["actual_models"])|observed_models(event))
                if event.get("type") == "thread.started" and isinstance(event.get("thread_id"),str):
                    receipt["thread_ids"] = sorted(set(receipt["thread_ids"])|{event["thread_id"]})
                if event.get("type") == "turn.completed":
                    usage = dict(event.get("usage") or {})
                    receipt["usage_events"].append({"turn_id":event.get("turn_id"),"usage":usage})
                    for key in ("input_tokens","output_tokens"):
                        receipt["usage"][key] += usage.get(key,0)
                    receipt["usage"]["cached_input_tokens"] += usage.get("cached_input_tokens",usage.get("input_tokens_details",{}).get("cached_tokens",0))
                item = event.get("item",{})
                if event.get("type") == "item.completed" and item.get("type") in {"agent_message","assistant_message"}:
                    last_message = item.get("text",last_message)
                write_json(protected/"receipt.json",receipt)
            receipt["returncode"] = process.wait()
            raw.flush()
            os.fsync(raw.fileno())
        if receipt["returncode"] != 0:
            raise RuntimeError(f"Native Codex exited with code {receipt['returncode']}; inspect protected JSONL/stderr")
        if not receipt["usage_events"]:
            raise RuntimeError("Native Codex omitted completed-turn usage evidence")
        response_path = protected/"last-message.json"
        response = response_path.read_text() if response_path.is_file() else last_message
        if not response:
            raise RuntimeError("Native Codex omitted its final structured source")
        payload = json.loads(response)
        write_json(protected/"response.json",payload)
        candidate = normalized_candidate(payload,protected,config.workload)
        candidate["provenance"] = {"kind":"native_codex_one_prompt","requested_model":MODEL,
            "reasoning_effort":EFFORT,"actual_models":receipt["actual_models"],
            "worker_session":session_id,"thread_ids":receipt["thread_ids"],
            "execution_code_hash":receipt["execution_code_hash"],
            "model_usage":receipt["usage"],"receipt":str(protected/"receipt.json"),
            "raw_jsonl":str(protected/"codex.jsonl")}
        write_json(protected/"candidate.json",candidate)
        receipt.update(status="completed",candidate=candidate,ended_at=time.time(),
            model_identity_evidence="provider JSONL model fields" if receipt["actual_models"] else "requested CLI model; provider did not emit model identity")
    except BaseException as error:
        receipt.update(status="failed",error=str(error),ended_at=time.time())
        raise
    finally:
        try:
            receipt["worker_terminal"] = service.close()
            if receipt["worker_terminal"].get("job_id") and receipt["worker_terminal"].get("ownership") != "borrowed":
                receipt["submitted"] = True
        finally:
            write_json(protected/"receipt.json",receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker-session",default=None)
    parser.add_argument("--codex",default=default_codex_executable())
    parser.add_argument("--run",action="store_true",help="Explicitly start one owned GPU service and one native Codex session")
    args = parser.parse_args(argv)
    config = CampaignConfig.load(args.config)
    session_id = args.worker_session or (config.run_id[:64]+".native-vanilla-"+uuid.uuid4().hex[:8])
    if not args.run:
        print(json.dumps(plan(config,args.output,session_id,args.codex),indent=2))
        return 0
    result = run_vanilla(config,args.output,session_id,args.codex)
    print(json.dumps({"status":result["status"],"receipt":str(args.output.resolve()/"evidence/receipt.json"),
        "usage":result["usage"],"candidate":result.get("candidate")},indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
