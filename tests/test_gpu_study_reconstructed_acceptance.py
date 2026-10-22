"""CPU-only fake bundles/backends; no real FINAL buffers, SSH, CUDA or model."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from optimizer_agent.evaluation import Candidate, LaunchRule, WorkloadContract
from optimizer_agent.workload_semantics import operation_spec
from scripts import gpu_study_reconstructed_acceptance as helper

OLD = {"uuid":"archived-physical-gpu","architecture":"sm_90","driver_version":13010,
       "nvrtc_version":"12.6","cuda_toolkit_version":12060,"harness_build_id":"cpu-fixture"}
NEW = {**OLD,"uuid":"different-new-physical-gpu"}


def encoded(value):
    return json.dumps(value,indent=2).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def make_bundle(directory, operation="convolution", damage=None, *, dependency=False):
    directory.mkdir()
    n, group, mode = 3, "final-groups/test-study."+operation+".1", "full"
    root = Path("/original-cold-study")
    prefix = "candidates/final-full/"
    selection = group+"/frozen-candidates.json#"+mode
    source = directory/"source.cu"
    source.write_text(f'extern "C" __global__ void {operation}(const float* a,const float* b,float* c,int n){{int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i];}}')
    candidate = Candidate("generated-method-candidate",source,operation,LaunchRule(kind="elementwise",block=(32,1,1)),{"tile":32})
    a = np.eye(n,dtype="<f4").ravel()/2 if operation=="matmul" else np.array([-.5,.25,.75],dtype="<f4")
    b = np.arange(n*n,dtype="<f4")/10 if operation=="matmul" else np.array([-.25,.5,.125],dtype="<f4")
    c_count = n*n if operation=="matmul" else (1 if operation=="reduction" else n)
    members = {prefix+"source.cu":source.read_bytes(),prefix+"a.bin":a.tobytes(),prefix+"b.bin":b.tobytes()}
    raw_source = str(root/group/"frozen/full/kernel.cu")
    members["evidence/"+group+"/frozen/full/kernel.cu"] = source.read_bytes()
    inputs = {f"sealed-inputs/final-{n}/a.bin":digest(a.tobytes()),f"sealed-inputs/final-{n}/b.bin":digest(b.tobytes())}
    metadata = {"operation_semantics":operation_spec(operation),"generation_closed":True,
                "input_hashes":inputs,"fixed_shape_domain":[n],"target_environment":OLD}
    contract = WorkloadContract("test-study."+operation+".1",1,operation,operation_spec(operation)["signature"],
        "independent CPU fixture",allowed_precision="float32",metadata_json=json.dumps(metadata,sort_keys=True),
        required_sanitizers=("memcheck","racecheck","initcheck","synccheck"))
    frozen = {"candidate_id":candidate.candidate_id,"source_path":raw_source,"source_hash":candidate._source_hash,
        "artifact_hash":candidate.artifact_hash,"kernel":operation,"launch_rule":asdict(candidate.launch_rule),"parameters":dict(candidate.parameters)}
    original = {"source":raw_source,"kernel":operation,"mode":"run","device":0,"grid":[1,1,1],"block":[32,1,1],
        "shared_memory_bytes":0,"warmup":0,"iterations":1,"series":1,"guard_outputs":True,"sentinel_byte":255,
        "compiler_options":[],"dependencies":[],"cache_dir":None,"compile_cache_key":None,
        "arguments":[{"kind":"buffer","bytes":a.nbytes,"input_file":str(root/group/f"sealed-inputs/final-{n}/a.bin"),"direction":"input"},
                     {"kind":"buffer","bytes":b.nbytes,"input_file":str(root/group/f"sealed-inputs/final-{n}/b.bin"),"direction":"input"},
                     {"kind":"buffer","bytes":c_count*4,"input_file":None,"direction":"output"},
                     {"kind":"scalar","type":"int32","value":n}]}
    portable = json.loads(json.dumps(original))
    portable["source"] = prefix+"source.cu"
    portable["arguments"][0]["input_file"],portable["arguments"][1]["input_file"] = prefix+"a.bin",prefix+"b.bin"
    source_files = {raw_source:prefix+"source.cu"}
    if dependency:
        old = str(root/group/"frozen/full/numbers.cuh")
        members[prefix+"numbers.cuh"] = b"// Declared CPU fixture dependency\n"
        source_files[old] = prefix+"numbers.cuh"
        original["dependencies"],portable["dependencies"] = ["numbers.cuh" if dependency=="relative" else old],[prefix+"numbers.cuh"]
    options_sha = digest(helper.portable_export.encoded([]))
    origin = group+"/request-receipts/authenticated-run.json"
    link = {"path":prefix+"request.json","origin":origin,"compiler_options":[],"options_sha256":options_sha,
        "portable_options_sha256":options_sha,"options_relocations":[],"dependencies":portable["dependencies"]}
    chosen = {**frozen,"id":"final-full","selection":selection,"source":prefix+"source.cu","prefix":prefix,
        "evidence_context":group+"/frozen-candidates.json","requests":[link],"reconstructable":True,
        "contract":asdict(contract),"contract_hash":"authenticated-derived-contract","environment":OLD,"source_files":source_files}
    result = {"kind":"shared_final_group","group_id":"test-study."+operation+".1","workload":operation,"shapes":[n],
        "evaluation_finished":True,"execution_complete":True,"all_modes_available":True,"job_id":"7001",
        "worker_terminal":{"terminal":True,"state":"COMPLETED","exit_code":"0:0","job_id":"7001"},
        "candidates":{"full":{**frozen,"qualified":True}},"contract":asdict(contract),
        "contract_hash":chosen["contract_hash"],"input_hashes":inputs,"environment":OLD}
    report = {"ok":True,"slurm_job_id":"7001","environment":OLD,"guard_readonly_ok":True,"guard_outputs_poisoned":True,
        "compilation_identity":{"source":source.read_text(),"options":[]}}
    manifest = {"kind":"portable_study_export","status":"complete","study_directory_observed":str(root),
        "controller_or_worker_lock_observed_active":False,"source_head_changed_during_export":False,
        "artifact_qualification_incidents":{"schema_version":1,"kind":"exported_artifact_refusal_index",
            "registry_observed":True,"registry_present":False,"incidents":[],"observations":[]},
        "active_jobs_observed":[],"inflight_search_trials":[],"errors":[],"changed_files_during_export":[],"candidates":[chosen]}
    if damage == "input":members[prefix+"a.bin"] = np.zeros_like(a).tobytes()
    elif damage == "contract":chosen["contract"]["allowed_precision"]="float64"
    elif damage == "semantic_contract":
        metadata["operation_semantics"]["numerical_tolerance"]["atol"] = 1.
        result["contract"]["metadata_json"] = chosen["contract"]["metadata_json"] = json.dumps(metadata,sort_keys=True)
    elif damage == "shape":original["arguments"][3]["value"]=4;portable["arguments"][3]["value"]=4
    elif damage == "options":original["compiler_options"]=["--use_fast_math"]
    elif damage == "portable_options":
        portable["compiler_options"]=["--use_fast_math"]
        link["portable_options_sha256"]=digest(helper.portable_export.encoded(portable["compiler_options"]))
    elif damage == "artifact":chosen["artifact_hash"]="0"*64
    elif damage == "launch":portable["block"]=[64,1,1]
    elif damage == "final_status":result["evaluation_finished"]=False
    elif damage == "terminal":result["worker_terminal"]["job_id"]="unrelated-job"
    elif damage == "partial":manifest["status"]="partial"
    elif damage == "active":manifest["controller_or_worker_lock_observed_active"]=True
    elif damage == "origin":link["origin"]="campaigns/old-search/request-receipts/one.json"
    elif damage == "dependency":chosen["source_files"].clear()
    elif damage == "abi":original["arguments"][2]["direction"]="inout";portable["arguments"][2]["direction"]="inout"
    elif damage == "benchmark":original["mode"]=portable["mode"]="benchmark"
    elif damage == "source":members[prefix+"source.cu"] += b"// mismatch\n"
    members[prefix+"request.json"] = encoded(portable)
    members["evidence/"+group+"/frozen-candidates.json"] = encoded({"full":frozen})
    members["evidence/"+group+"/result.json"] = encoded(result)
    members["evidence/"+origin] = encoded({"run_id":"authenticated-run","request":original,"report":report})
    manifest["files"] = {name:{"sha256":digest(raw),"bytes":len(raw)} for name,raw in members.items()}
    bundle = directory/"study-evidence.zip"
    with zipfile.ZipFile(bundle,"w") as archive:
        for name,raw in members.items():archive.writestr(name,raw)
        archive.writestr("export-manifest.json",encoded(manifest))
    evidence = directory/"public-terminal-observation.json"
    evidence.write_bytes(encoded({"kind":"CPU-only fixture; no actual liveness claim","terminal":True}))
    attestation = {"schema_version":1,"kind":"manual_validation_release_quiescence","bundle_sha256":helper.sha(bundle),
        "selection":selection,"attested_by":"CPU fixture","attested_at_utc":"2026-10-22T00:00:00Z",
        **{key:True for key in helper.QUIESCENT},"evidence_pins":[{"path":str(evidence),"sha256":helper.sha(evidence)}]}
    quiescence=directory/"quiescence.json";quiescence.write_bytes(encoded(attestation))
    return bundle,helper.sha(bundle),selection,quiescence,helper.sha(quiescence),a,b


def parameters(bundle):
    path,digest_value,selection,quiescence,pin,*_=bundle
    return path,digest_value,dict(selection=selection,request_index=0,quiescence=quiescence,quiescence_sha256=pin)


def test_without_quiescence_default_opens_no_bundle_and_writes_no_output(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(helper.zipfile,"ZipFile",lambda *a,**k:pytest.fail("No ZIP before quiescence"))
    monkeypatch.setattr(helper,"sha",lambda *a:pytest.fail("No opaque bundle read"))
    args=["--bundle",str(tmp_path/"does-not-exist.zip"),"--bundle-sha256","0"*64,"--selection","chosen",
          "--request-index","0","--output",str(tmp_path/"not-created")]
    assert helper.main(args)==0
    assert json.loads(capsys.readouterr().out)["bundle_opened"] is False
    assert not (tmp_path/"not-created").exists()
    with pytest.raises(SystemExit):helper.main([*args,"--run"])


@pytest.mark.parametrize("damage",["sha","missing_flag","pin","binding"])
def test_quiescence_refused_before_any_zip_or_reconstruction(tmp_path,monkeypatch,damage):
    bundle=make_bundle(tmp_path/"archive")
    path,digest_value,kwargs=parameters(bundle)
    value=json.loads(kwargs["quiescence"].read_bytes())
    if damage=="missing_flag":value[helper.QUIESCENT[0]]=False
    elif damage=="pin":value["evidence_pins"][0]["sha256"]="0"*64
    elif damage=="binding":value["bundle_sha256"]="0"*64
    kwargs["quiescence"].write_bytes(encoded(value))
    kwargs["quiescence_sha256"]="0"*64 if damage=="sha" else helper.sha(kwargs["quiescence"])
    monkeypatch.setattr(helper.zipfile,"ZipFile",lambda *a,**k:pytest.fail("Must refuse before ZIP"))
    with pytest.raises(ValueError):helper.prepare(tmp_path/"not-created",path,digest_value,**kwargs)
    assert not (tmp_path/"not-created").exists()


@pytest.mark.parametrize("operation",["matmul","reduction","scan","stencil","convolution","normalization"])
def test_six_closed_operations_reconstruct_exact_artifact_new_paths_and_independent_oracle(tmp_path,operation):
    bundle=make_bundle(tmp_path/"archive",operation,dependency=operation=="scan")
    path,digest_value,kwargs=parameters(bundle)
    candidate,request,expected,plan=helper.prepare(tmp_path/"plan",path,digest_value,**kwargs)
    assert plan["operation"]==operation and plan["n"]==3 and plan["original_environment"]["uuid"]==OLD["uuid"]
    assert candidate.artifact_hash==plan["artifact_hash"] and request.mode=="run"
    assert all(Path(a.input_file).is_relative_to(tmp_path/"plan") for a in request.arguments if getattr(a,"input_file",None))
    expected_values={"matmul":(bundle[-1]/2).ravel(),"reduction":[.5],"scan":[-.5,-.25,.5],
                     "stencil":[-.25,.5,1.],"convolution":[-.21875,.34375,.3125]}
    if operation=="normalization":
        rms=np.sqrt((.25+.0625+.5625)/3+1e-5)
        expected_values[operation]=[-.5/rms,.25/rms,.75/rms]
    np.testing.assert_allclose(expected,expected_values[operation],atol=1e-7)
    assert plan["benchmark_requests"]==plan["model_calls"]==0 and plan["new_scores_created"] is False
    if operation=="scan":assert Path(request.dependencies[0]).read_bytes()==b"// Declared CPU fixture dependency\n"


@pytest.mark.parametrize("damage",["input","contract","semantic_contract","shape","options","portable_options","artifact","launch",
    "final_status","terminal","partial","active","origin","dependency","abi","benchmark","source"])
def test_altered_contract_or_evidence_cannot_submit_or_materialize_final_data(tmp_path,damage):
    bundle=make_bundle(tmp_path/"archive",damage=damage,dependency=damage=="dependency")
    path,digest_value,kwargs=parameters(bundle)
    with pytest.raises(ValueError):helper.prepare(tmp_path/"not-created",path,digest_value,**kwargs)
    assert not (tmp_path/"not-created").exists()


def test_relative_dependency_binds_to_the_original_source_parent_without_reading_it(tmp_path):
    bundle=make_bundle(tmp_path/"archive",dependency="relative")
    path,digest_value,kwargs=parameters(bundle)
    _,request,_,_=helper.prepare(tmp_path/"prepared",path,digest_value,**kwargs)
    assert Path(request.dependencies[0]).is_relative_to(tmp_path/"prepared")
    assert Path(request.dependencies[0]).read_bytes()==b"// Declared CPU fixture dependency\n"


@pytest.mark.parametrize("options,index,prefix",[("-I/old/includes",0,"-I"),("--include-path=/old/includes",0,"--include-path="),
                                                (["-I","/old/includes"],1,""),(["--include-path","/old/includes"],1,"")])
def test_only_exact_authenticated_include_relocations_are_permitted(options,index,prefix):
    original=options if isinstance(options,list) else [options]
    target="candidates/x/source-tree/includes"
    chosen={"prefix":"candidates/x/","source_files":{"/old/includes/values.cuh":target+"/values.cuh"}}
    relocation={"index":index,"prefix":prefix,"original":original[index],"path":target}
    portable=list(original);portable[index]=prefix+target
    helper.authenticate_portable_options(original,portable,[relocation],chosen)
    for broken in (portable+["--use_fast_math"],[*portable[:-1],"changed-option"]):
        with pytest.raises(ValueError):helper.authenticate_portable_options(original,broken,[relocation],chosen)
    with pytest.raises(ValueError):helper.authenticate_portable_options(original,portable,[{**relocation,"path":target+"/other"}],chosen)


def test_two_gpu_configuration_refuses_before_constructing_or_submitting_backend(tmp_path,monkeypatch):
    bundle=make_bundle(tmp_path/"archive")
    path,digest_value,kwargs=parameters(bundle)
    config=tmp_path/"invalid-slurm.json";config.write_text(json.dumps({"gres":"gpu:h100:2"}))
    monkeypatch.setattr(helper,"SlurmBackend",lambda *a,**k:pytest.fail("One-GPU validation precedes backend construction"))
    with pytest.raises(ValueError,match="One GPU"):
        helper.run_acceptance(tmp_path/"prepared",path,digest_value,config,run_id="never-submit",**kwargs)
    assert not (tmp_path/"prepared/allocation-intent.json").exists()


def fake_report(request,expected,path,*,failure=None,job="9001"):
    actual=expected.copy()
    if failure=="numerical":actual[0]+=1
    actual.astype("<f4").tofile(path)
    if failure=="input_mutation":Path(request.arguments[0].input_file).write_bytes(b"changed")
    return {"ok":failure!="busy","status":"gpu_reservation_conflict" if failure=="busy" else "completed",
        "slurm_job_id":job,"environment":{**NEW,"uuid":"wrong" if failure=="uuid" else NEW["uuid"]},
        "outputs":[{"path":str(path)}],"guard_readonly_ok":True,"guard_outputs_poisoned":failure!="sentinels",
        "worker_reusable":failure!="poisoned","worker_poisoned":failure=="poisoned",
        "latency_samples_ms":[1.] if failure=="timings" else [],"measurement_protocol":{"warmup":0,"iterations":1,"series":1},
        "compilation_identity":{"source":Path(request.source).read_text(),"options":request.compiler_options}}


@pytest.mark.parametrize("failure",[None,"numerical","input_mutation","uuid","sentinels","timings","poisoned","busy"])
def test_one_cpu_runner_observation_keeps_new_uuid_separate_and_rejects_failed_guards(tmp_path,failure):
    bundle=make_bundle(tmp_path/"archive")
    path,digest_value,kwargs=parameters(bundle)
    candidate,request,expected,plan=helper.prepare(tmp_path/"prepared",path,digest_value,**kwargs)
    calls=[]
    class Runner:
        def get_device_properties(self,device):return NEW
        def run(self,run_id,req):
            calls.append(run_id)
            return fake_report(req,expected,tmp_path/"actual.bin",failure=failure)
    result=helper.execute_one(Runner(),candidate,request,expected,plan,"one-only")
    assert result["ok"] is (failure is None) and calls==["one-only"]
    assert result["actual_environment"]["uuid"]!=plan["original_environment"]["uuid"]
    assert result["model_calls"]==result["benchmark_requests"]==0


@pytest.mark.parametrize("changed",["source","header","launch","parameters"])
def test_reconstructed_source_tree_and_artifact_mutation_refuse_before_gpu(tmp_path,changed):
    bundle=make_bundle(tmp_path/"archive",dependency=True)
    path,digest_value,kwargs=parameters(bundle)
    candidate,request,expected,plan=helper.prepare(tmp_path/"prepared",path,digest_value,**kwargs)
    if changed=="source":candidate.source.write_text(candidate.source.read_text()+"// altered\n")
    elif changed=="header":Path(request.dependencies[0]).write_text("// altered header\n")
    elif changed=="launch":object.__setattr__(candidate.launch_rule,"block",(64,1,1))
    else:candidate.parameters["tile"]=64
    class Runner:
        def get_device_properties(self,*a):pytest.fail("Must refuse before accessing GPU")
        def run(self,*a):pytest.fail("Must refuse before accessing GPU")
    with pytest.raises(ValueError):helper.execute_one(Runner(),candidate,request,expected,plan,"must-not-run")


@pytest.mark.parametrize("failure",[None,"lost_ack","partial_start","foreign_ack","cleanup","terminal"])
def test_owned_allocation_lifecycle_preserves_partial_submission_and_never_closes_foreign_job(tmp_path,monkeypatch,failure):
    bundle=make_bundle(tmp_path/"archive")
    path,digest_value,kwargs=parameters(bundle)
    closed,calls,instances=[],[],[]
    monkeypatch.setattr(helper,"controller_provenance",lambda *a:{"HEAD":"cpu-only","sources":{"helper":"pin"}})
    class Backend:
        simulation=True
        def __init__(self,config,repo,results,*,session_id):
            self.config,self.results_dir,self.session_id=config,Path(results),session_id
            self.remote_dir,self.job_id="/fixture/"+session_id,None
            self.results_dir.mkdir(parents=True)
            instances.append(self)
        def start(self):
            assert (tmp_path/"execution/allocation-intent.json").exists()
            assert self.config.partition=="long" and self.config.walltime=="48:00:00" and self.config.gres.endswith(":1")
            if failure=="lost_ack":raise RuntimeError("Lost ACK, no local owner proof")
            (self.results_dir/"slurm-session.json").write_bytes(encoded({"session_id":"foreign" if failure=="foreign_ack" else self.session_id,
                "job_id":"9001","remote_dir":self.remote_dir,"config":asdict(self.config)}))
            if failure=="partial_start":raise RuntimeError("Failed after local ACK before in-memory adoption")
            self.job_id="9001"
            return self.job_id
        def wait_until_ready(self):calls.append("ready")
        def runner(self):
            outer=self
            class Runner:
                def get_device_properties(self,device):return NEW
                def run(self,run_id,request):
                    calls.append("one-run")
                    a,b=[np.fromfile(x.input_file,dtype="<f4") for x in request.arguments[:2]]
                    expected=helper.cpu_oracle("convolution",a,b,3)
                    return fake_report(request,expected,tmp_path/"fresh-output.bin",job=outer.job_id)
            return Runner()
        def close(self):
            closed.append(self.job_id)
            if failure=="cleanup":raise RuntimeError("Cleanup failed; preserve it")
        def wait_until_finished(self,timeout):
            return {"terminal":True,"state":"COMPLETED","exit_code":"0:0","job_id":"foreign" if failure=="terminal" else self.job_id}
    result=helper.run_acceptance(tmp_path/"execution",path,digest_value,helper.REPO/"configs/romeo.json",
        run_id="one-run",backend_factory=Backend,**kwargs)
    assert result["ok"] is (failure is None) and result["evidence_kind"]=="simulated_cpu_test"
    assert len(instances)==1 and calls.count("one-run")==int(failure not in ("lost_ack","partial_start","foreign_ack"))
    assert closed==([] if failure in ("lost_ack","foreign_ack") else ["9001"])
    if failure=="partial_start":assert result["job_id"]=="9001" and "error" in result
    if failure in ("foreign_ack","cleanup"):assert "cleanup_error" in result
    with pytest.raises(ValueError,match="fresh"):
        helper.run_acceptance(tmp_path/"execution",path,digest_value,helper.REPO/"configs/romeo.json",
            run_id="must-not-retry",backend_factory=Backend,**kwargs)
    assert len(instances)==1
