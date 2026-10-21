"""Native CLI transport tests with fake processes/services; no GPU/model calls."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from optimizer_agent.campaign_config import CampaignConfig
from scripts import run_codex_vanilla as driver


def payload():
    return {"source":'extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n) {}',
        "kernel":"matmul","parameters":[{"name":"TILE","value":16}],
        "launch_rule":{"kind":"square_2d","block":[16,16,1],"grid":[1,1,1],"elements_key":"n",
            "shared_memory_bytes":0,"version":1,"tile_x":16,"tile_y":16}}


class Input:
    def __init__(self):
        self.value = ""
        self.closed = False
    def write(self,value):
        self.value += value
    def close(self):
        self.closed = True


class Service:
    instances = []
    def __init__(self,config,protected_dir,workspace,session_id,repo_dir):
        self.config,self.protected,self.workspace,self.session_id = config,protected_dir,workspace,session_id
        self.closed,self.starts = 0,0
        self.instances.append(self)
    def start(self):
        self.starts += 1
        self.protected.joinpath("rpc.jsonl").write_text(json.dumps({"operation":"validate","attempt_id":"real-service-fixture"})+"\n")
        return {"endpoint":"http://127.0.0.1:12345","token":"test-service-secret",
            "tool_command":["/venv/python","/readonly/vanilla_gpu_tool.py"],
            "properties":{"uuid":"test-gpu","architecture":"sm_90"},"baseline":{"ok":True},
            "operation":self.config.workload,"workload_description":driver.workload_description(self.config.workload)}
    def close(self):
        self.closed += 1
        return {"job_id":"owned-fixture-job","state":"COMPLETED"}


class Process:
    def __init__(self,events,returncode=0):
        self.stdin,self.stdout,self.pid = Input(),iter(events),12345
        self.returncode = returncode
    def wait(self):
        return self.returncode


def fake_execution(tmp_path,*,events=None,returncode=0):
    Service.instances = []
    if events is None:
        events = [
            {"type":"thread.started","thread_id":"native-thread","model":"gpt-6-luna"},
            {"type":"item.completed","item":{"type":"command_execution","command":"GPU helper validate","exit_code":0}},
            {"type":"item.completed","item":{"type":"agent_message","text":json.dumps(payload())}},
            {"type":"turn.completed","usage":{"input_tokens":3_000_000,"cached_input_tokens":2_000_000,"output_tokens":100_001}},
        ]
    process = Process([json.dumps(event)+"\n" for event in events],returncode)
    calls = []
    def popen(argv,**kwargs):
        calls.append((argv,kwargs))
        return process
    versions = []
    def version(argv,**kwargs):
        versions.append((argv,kwargs))
        return SimpleNamespace(stdout="codex-cli fixture-version\n")
    def isolation(root, *, codex_executable):
        return {"ok":True,"sensitive_root":str(Path(root).resolve()),"scope":"CPU fixture; no native isolation claim"}
    options = dict(service_factory=Service,popen_factory=popen,version_runner=version,
                   sensitive_root=tmp_path,isolation_probe=isolation)
    return process,calls,versions,options


def test_default_cli_plan_never_starts_gpu_or_model(tmp_path,monkeypatch,capsys):
    config = tmp_path/"config.json"
    config.write_text(json.dumps(CampaignConfig(run_id="native-fixture").to_dict()))
    def forbidden(*args,**kwargs):
        raise AssertionError("Planning cannot start a session")
    monkeypatch.setattr(driver,"run_vanilla",forbidden)
    output = tmp_path/"uncreated-run"
    assert driver.main(["--config",str(config),"--output",str(output)]) == 0
    described = json.loads(capsys.readouterr().out)
    assert not described["submitted"] and not described["model_started"]
    assert described["global_resource_caps"] is None
    assert not output.exists()


def test_script_entrypoint_plans_outside_repository_without_import_shadowing(tmp_path):
    config = tmp_path/"config.json"
    config.write_text(json.dumps(CampaignConfig(run_id="native-fixture").to_dict()))
    output = tmp_path/"uncreated-run"
    result = subprocess.run([sys.executable,str(Path(driver.__file__).resolve()),"--config",str(config),
        "--output",str(output),"--worker-session","plan-only"],cwd=tmp_path,
        capture_output=True,text=True,check=True)
    described = json.loads(result.stdout)
    assert described["mode"] == "plan" and not described["submitted"]
    assert not output.exists()


def test_shell_enabled_one_prompt_workspace_scope_and_real_usage_receipts(tmp_path,monkeypatch):
    monkeypatch.setenv("SSH_AUTH_SOCK","/private/parent-ssh-agent")
    process,calls,versions,options = fake_execution(tmp_path)
    root = tmp_path/"native"
    receipt = driver.run_vanilla(CampaignConfig(),root,"owned-session","/test/codex",**options)
    assert receipt["status"] == "completed"
    argv,kwargs = calls[0]
    assert argv[:4] == ["/test/codex","-a","never","exec"]
    assert argv[argv.index("-m")+1] == "gpt-6-luna"
    assert 'model_reasoning_effort="max"' in argv
    assert "features.shell_tool=true" in argv
    assert "permissions.study_native_tools.network.enabled=true" in argv
    assert not any("sandbox_workspace_write" in argument for argument in argv)
    assert 'default_permissions="study_native_tools"' in argv
    filesystem = next(argument for argument in argv if argument.startswith("permissions.study_native_tools.filesystem="))
    assert json.dumps(str(tmp_path))+'="deny"' in filesystem
    assert json.dumps(str(root/"workspace"))+'="write"' in filesystem
    assert "project_doc_max_bytes=0" in argv
    assert "--ignore-user-config" in argv and "--ignore-rules" not in argv
    assert "--ephemeral" in argv and "--json" in argv
    assert "--sandbox" not in argv
    assert argv[argv.index("--output-last-message")+1] == str(root/"evidence/last-message.json")
    assert "--add-dir" not in argv and "--dangerously-bypass-approvals-and-sandbox" not in argv
    assert "timeout" not in kwargs
    assert kwargs["cwd"] == root/"workspace"
    assert "SSH_AUTH_SOCK" not in kwargs["env"]
    assert kwargs["env"]["SERVICE_TOKEN"] == "test-service-secret"
    assert process.stdin.value == driver.optimization_prompt() and process.stdin.closed
    assert "TreeQuest" not in process.stdin.value and "PMPP" not in process.stdin.value
    assert versions[0][0] == ["/test/codex","--version"]
    assert receipt["codex_version"] == "codex-cli fixture-version"
    assert receipt["usage"] == {"input_tokens":3_000_000,"cached_input_tokens":2_000_000,"output_tokens":100_001}
    assert receipt["actual_models"] == ["gpt-6-luna"]
    assert receipt["thread_ids"] == ["native-thread"]
    assert Service.instances[0].closed == 1
    candidate = json.loads(root.joinpath("evidence/candidate.json").read_text())
    frozen = Path(candidate["source_path"])
    assert frozen.is_relative_to(root/"evidence") and not frozen.is_relative_to(root/"workspace")
    assert driver.file_hash(frozen) == candidate["source_hash"]
    assert candidate["parameters"] == {"TILE":16}
    assert candidate["certification_status"] == "not_evaluated_on_independent_final_data"
    assert candidate["provenance"]["model_usage"] == receipt["usage"]
    assert root.joinpath("evidence/rpc.jsonl").is_file()
    raw = [json.loads(line) for line in root.joinpath("evidence/codex.jsonl").read_text().splitlines()]
    assert raw[1]["item"]["type"] == "command_execution"
    assert set(path.name for path in root.joinpath("workspace").iterdir()) == {"baseline.cu","GPU_TOOLS.md"}


def test_failing_native_process_always_closes_only_owned_service(tmp_path):
    _,_,_,options = fake_execution(tmp_path,returncode=7)
    with pytest.raises(RuntimeError,match="code 7"):
        driver.run_vanilla(CampaignConfig(),tmp_path/"failed","owned-session","/test/codex",**options)
    service = Service.instances[0]
    assert service.session_id == "owned-session" and service.closed == 1
    receipt = json.loads(service.protected.joinpath("receipt.json").read_text())
    assert receipt["status"] == "failed" and receipt["worker_terminal"]["job_id"] == "owned-fixture-job"
    assert not service.protected.joinpath("candidate.json").exists()


def test_completed_turn_usage_is_required_even_with_source_output(tmp_path):
    events = [{"type":"item.completed","item":{"type":"agent_message","text":json.dumps(payload())}}]
    _,_,_,options = fake_execution(tmp_path,events=events)
    with pytest.raises(RuntimeError,match="usage evidence"):
        driver.run_vanilla(CampaignConfig(),tmp_path/"no-usage","owned-session","/test/codex",**options)
    assert Service.instances[0].closed == 1


def test_provider_omitted_model_identity_is_not_invented(tmp_path):
    events = [{"type":"item.completed","item":{"type":"agent_message","text":json.dumps(payload())}},
              {"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":10,
                  "input_tokens_details":{"cached_tokens":80}}}]
    _,_,_,options = fake_execution(tmp_path,events=events)
    receipt = driver.run_vanilla(CampaignConfig(),tmp_path/"unknown-model","owned-session","/test/codex",**options)
    assert receipt["actual_models"] == []
    assert "provider did not emit" in receipt["model_identity_evidence"]
    assert receipt["usage"]["cached_input_tokens"] == 80


def test_existing_session_directory_is_never_overwritten(tmp_path):
    root = tmp_path/"existing"
    root.joinpath("workspace").mkdir(parents=True)
    _,_,_,options = fake_execution(tmp_path)
    with pytest.raises(ValueError,match="new output directory"):
        driver.run_vanilla(CampaignConfig(),root,"owned-session","/test/codex",**options)
    assert Service.instances == []


@pytest.mark.parametrize("operation", driver.OPERATIONS)
def test_native_driver_is_one_prompt_with_exact_six_workload_contracts(tmp_path, operation):
    config = CampaignConfig(run_id="generic-" + operation, workload=operation)
    proposed = payload()
    proposed.update(source=driver.baseline_source(config), kernel=operation, parameters=[],
                    launch_rule=driver.workload_description(operation)["baseline_launch_rule"])
    events = [{"type":"item.completed","item":{"type":"agent_message","text":json.dumps(proposed)}},
              {"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":10}}]
    process,calls,_,options = fake_execution(tmp_path, events=events)
    output = tmp_path / operation
    receipt = driver.run_vanilla(config, output, "native-" + operation, "/test/codex", **options)
    assert len(calls) == 1 and process.stdin.value == driver.optimization_prompt(operation)
    assert driver.workload_description(operation)["mathematics"] in process.stdin.value
    assert receipt["candidate"]["kernel"] == operation
    assert receipt["workload"] == operation
    assert output.joinpath("workspace/baseline.cu").read_text() == driver.baseline_source(config)
    assert json.loads(output.joinpath("evidence/output-schema.json").read_text())["properties"]["kernel"]["enum"] == [operation]
    assert driver.workload_description(operation)["signature"] in output.joinpath("workspace/GPU_TOOLS.md").read_text()


def test_native_driver_does_not_claim_borrowed_worker_submission_or_termination(tmp_path):
    class Borrowed(Service):
        def start(self):
            metadata = super().start()
            return {**metadata, "owns_worker": False, "worker_ownership": "borrowed"}
        def close(self):
            self.closed += 1
            return {"job_id": "caller-job", "ownership": "borrowed", "terminal": False,
                    "state": "BORROWED_WORKER_PRESERVED", "service_closed": True}
    _,_,_,options = fake_execution(tmp_path)
    options["service_factory"] = Borrowed
    receipt = driver.run_vanilla(CampaignConfig(), tmp_path / "borrowed", "borrowed", "/test/codex", **options)
    assert receipt["status"] == "completed" and receipt["submitted"] is False
    assert receipt["worker_ownership"] == "borrowed"
    assert receipt["worker_terminal"]["state"] == "BORROWED_WORKER_PRESERVED"
