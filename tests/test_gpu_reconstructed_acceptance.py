"""CPU-only bundle/lifecycle/oracle tests; no SSH, CUDA or model invocation."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from optimizer_agent.evaluation import Candidate, LaunchRule
from scripts import gpu_reconstructed_acceptance as helper

PROPERTIES = {"uuid":"simulated-new-device", "architecture":"sm_90", "driver_version":13010,
              "nvrtc_version":"12.6", "cuda_toolkit_version":12060, "harness_build_id":"unit-native-build"}


def make_bundle(directory, *, mode="run"):
    directory.mkdir()
    source = directory / "source.cu"
    source.write_text('extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n){int r=blockIdx.y*blockDim.y+threadIdx.y;int j=blockIdx.x*blockDim.x+threadIdx.x;if(r<n&&j<n){float s=0;for(int k=0;k<n;k++)s+=a[r*n+k]*b[k*n+j];c[r*n+j]=s;}}')
    candidate = Candidate("codex-vanilla", source, "matmul", LaunchRule(kind="fixed", grid=(16,16,1), block=(16,16,1)), {"tile":32})
    diagonal = np.linspace(1, 2, 316, dtype="<f4")
    a = np.diag(diagonal)
    b = (np.arange(316 * 316, dtype="<f4") / (316 * 316)).reshape(316,316)
    expected = b * diagonal[:,None]  # Independent diagonal-matrix identity.
    prefix = "candidates/native/"
    files = {prefix + "source.cu":source.read_bytes(), prefix + "a.bin":a.tobytes(), prefix + "b.bin":b.tobytes()}
    frozen = {"candidate_id":candidate.candidate_id, "source_hash":candidate._source_hash, "artifact_hash":candidate.artifact_hash,
              "launch_rule":asdict(candidate.launch_rule), "parameters":dict(candidate.parameters)}
    original = {"candidates":{"native_codex":frozen}, "contract_hash":"original-contract", "hardware_identity":{"uuid":"archived-device"}}
    files["evidence/comparison.json"] = json.dumps(original).encode()
    rows = []
    for index, sentinel in enumerate((0,1,127,255)):
        request = {"source":prefix+"source.cu", "kernel":"matmul", "mode":mode, "device":0,
            "grid":[16,16,1], "block":[16,16,1], "shared_memory_bytes":0, "warmup":0, "iterations":1,
            "series":1, "compiler_options":[], "dependencies":[], "cache_dir":None, "compile_cache_key":None,
            "guard_outputs":True, "sentinel_byte":sentinel, "arguments":[
                {"kind":"buffer","bytes":316*316*4,"input_file":prefix+"a.bin","direction":"input"},
                {"kind":"buffer","bytes":316*316*4,"input_file":prefix+"b.bin","direction":"input"},
                {"kind":"buffer","bytes":316*316*4,"input_file":None,"direction":"output"},
                {"kind":"scalar","type":"int32","value":316}]}
        member = prefix + str(index) + ".json"
        files[member] = json.dumps(request).encode()
        options_hash = hashlib.sha256(b"[]").hexdigest()
        rows.append({"path":member,"origin":"request-receipts/"+str(index)+".json", "compiler_options":[],
                     "options_sha256":options_hash,"portable_options_sha256":options_hash,"options_relocations":[]})
    chosen = {**frozen,"id":"native", "selection":"comparison.json#native_codex", "source":prefix+"source.cu",
              "kernel":"matmul", "prefix":prefix, "requests":rows}
    manifest = {"status":"partial", "candidates":[chosen], "files":{
        name:{"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)} for name,data in files.items()}}
    bundle = directory / "study.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():
            archive.writestr(name,data)
        archive.writestr("export-manifest.json",json.dumps(manifest))
    origin = directory / "comparison.json"
    origin.write_bytes(files["evidence/comparison.json"])
    return bundle, helper.sha(bundle), candidate.artifact_hash, expected, origin


def test_default_plan_reconstructs_archived_run_and_exact_diagonal_oracle_without_submission(tmp_path, monkeypatch):
    bundle, digest, artifact, reference, origin = make_bundle(tmp_path / "archive")
    monkeypatch.setattr(helper,"SlurmBackend",lambda *a,**k:pytest.fail("Planning cannot submit"))
    output = tmp_path / "plan"
    assert helper.main(["--bundle",str(bundle),"--bundle-sha256",digest,"--artifact-hash",artifact,
                        "--origin-comparison",str(origin),"--output",str(output)]) == 0
    plan = json.loads((output / "plan.json").read_text())
    assert plan["oracle"]["expected_sha256"] == hashlib.sha256(reference.tobytes()).hexdigest()
    assert plan["request_index"] == 3 and plan["request"]["mode"] == "run" and plan["n"] == 316
    assert plan["benchmark_requests"] == 0 and plan["new_scores_created"] is False
    assert not (output / "allocation-intent.json").exists()


@pytest.mark.parametrize("damage",["bundle_hash","benchmark_request","original_pairs"])
def test_untrusted_or_wrong_execution_data_is_rejected_before_any_allocation(tmp_path, damage):
    bundle, digest, artifact, _, origin = make_bundle(tmp_path / "archive",mode="benchmark" if damage == "benchmark_request" else "run")
    if damage == "bundle_hash":
        digest = "0"*64
    if damage == "original_pairs":
        origin.write_text("changed original evidence")
    with pytest.raises(ValueError):
        helper.prepare(tmp_path / "plan",bundle,digest,expected_artifact=artifact,origin_comparison=origin)


@pytest.mark.parametrize("failure",[None,"numerical","input_mutation","uuid","sentinels","timings"])
def test_one_simulated_request_requires_oracle_guards_identity_and_unchanged_inputs(tmp_path, failure):
    bundle, digest, artifact, reference, origin = make_bundle(tmp_path / "archive")
    candidate, request, expected, plan = helper.prepare(tmp_path / "plan",bundle,digest,expected_artifact=artifact,origin_comparison=origin)
    calls = []
    class Runner:
        def get_device_properties(self,device):
            return PROPERTIES
        def run(self,run_id,req):
            calls.append(asdict(req))
            data = reference.copy()
            if failure == "numerical":
                data[0,0] = 100
            output = tmp_path / "actual.bin"; data.tofile(output)
            if failure == "input_mutation":
                Path(req.arguments[0].input_file).write_bytes(b"\0"*req.arguments[0].bytes)
            return {"ok":True,"environment":{**PROPERTIES,"uuid":"wrong-device" if failure == "uuid" else PROPERTIES["uuid"]},
                "outputs":[{"path":str(output)}], "guard_readonly_ok":True,"guard_outputs_poisoned":failure != "sentinels",
                "worker_reusable":True,"latency_samples_ms":[1] if failure == "timings" else [],
                "measurement_protocol":{"warmup":0,"iterations":1,"series":1},
                "compilation_identity":{"source":Path(req.source).read_text(),"options":req.compiler_options}}
    result = helper.execute_one(Runner(),candidate,request,expected,plan,"one-run")
    assert result["ok"] is (failure is None)
    assert len(calls) == 1 and calls[0]["mode"] == "run"
    assert result["benchmark_requests"] == 0 and result["new_scores_created"] is False


def test_lost_submission_ack_preserves_intent_and_forbids_duplicate_retry(tmp_path):
    bundle, digest, artifact, _, origin = make_bundle(tmp_path / "archive")
    output = tmp_path / "execution"; instances = []
    class Backend:
        simulation = True
        def __init__(self,config,repo,results,*,session_id):
            self.config,self.session_id,self.remote_dir,self.job_id = config,session_id,"/isolated/"+session_id,None
            instances.append(self)
        def start(self):
            saved = json.loads((output / "allocation-intent.json").read_text())
            assert saved["session_id"] == self.session_id and saved["remote_dir"] == self.remote_dir
            raise RuntimeError("Lost ACK fixture")
    kwargs = dict(run_id="ac20",selection="comparison.json#native_codex",request_index=3,expected_artifact=artifact,
                  origin_comparison=origin,backend_factory=Backend)
    result = helper.run_acceptance(output,bundle,digest,helper.REPO / "configs/romeo.json",**kwargs)
    assert result["ok"] is False and result["job_id"] is None
    assert result["evidence_kind"] == "simulated_cpu_test" and result["session_id"] == instances[0].session_id
    with pytest.raises(ValueError,match="intent"):
        helper.run_acceptance(output,bundle,digest,helper.REPO / "configs/romeo.json",**kwargs)
    assert len(instances) == 1


@pytest.mark.parametrize("terminal_state,terminal_job",[("COMPLETED","9001"),("FAILED","9001"),("COMPLETED","another-job")])
def test_only_exact_owned_terminal_can_complete_simulated_acceptance(tmp_path, terminal_state, terminal_job):
    bundle,digest,artifact,reference,origin = make_bundle(tmp_path / "archive")
    calls,closed = [],[]
    class Runner:
        def get_device_properties(self,device):
            return PROPERTIES
        def run(self,run_id,request):
            calls.append(request.mode)
            output=tmp_path / "actual.bin";reference.tofile(output)
            return {"ok":True,"slurm_job_id":"9001","environment":PROPERTIES,"worker_reusable":True,
                "guard_readonly_ok":True,"guard_outputs_poisoned":True,"outputs":[{"path":str(output)}],
                "latency_samples_ms":[],"measurement_protocol":{"warmup":0,"iterations":1,"series":1},
                "compilation_identity":{"source":Path(request.source).read_text(),"options":request.compiler_options}}
    class Backend:
        simulation=True
        def __init__(self,config,repo,results,*,session_id):
            self.config,self.session_id,self.remote_dir,self.job_id=config,session_id,"/owned/"+session_id,None
        def start(self):
            self.job_id="9001";return self.job_id
        def wait_until_ready(self):
            pass
        def runner(self):
            return Runner()
        def close(self,*,cancel=False):
            closed.append((self.job_id,cancel))
        def wait_until_finished(self,timeout):
            return {"job_id":terminal_job,"terminal":True,"state":terminal_state,"exit_code":"0:0"}
    before=helper.sha(origin)
    result=helper.run_acceptance(tmp_path / "run",bundle,digest,helper.REPO / "configs/romeo.json",run_id="one-archived-run",
        selection="comparison.json#native_codex",request_index=3,expected_artifact=artifact,origin_comparison=origin,backend_factory=Backend)
    assert result["ok"] is (terminal_state == "COMPLETED" and terminal_job == "9001")
    assert calls == ["run"] and closed == [("9001",False)]
    assert helper.sha(origin) == before and result["origin_comparison_unchanged"]
    assert result["new_scores_created"] is False and result["evidence_kind"] == "simulated_cpu_test"
