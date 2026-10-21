"""Portable reconstruction of the actual four-frozen-kernel pilot schema."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from optimizer_agent.evaluation import Candidate, LaunchRule
from optimizer_agent.portable_export import export_study, reconstruct_candidate


@pytest.fixture
def pilot(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "README.md").write_text("Committed source\n")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Unit Test", "-c", "user.email=unit@example.invalid", "commit", "-qm", "fixture"], check=True)
    study = tmp_path / "pilot"
    (study / "sources").mkdir(parents=True)
    source = study / "sources/native_codex.cu"
    source.write_text('extern "C" __global__ void matmul(const float* a,const float* b,float* c,int n){int i=threadIdx.x;if(i<n)c[i]=a[i]+b[i];}')
    candidate = Candidate("codex-vanilla", source, "matmul", LaunchRule(kind="fixed", grid=(1,1,1), block=(32,1,1)), {"tile":32})
    frozen = {"candidate_id": candidate.candidate_id, "source_path": str(source), "source_hash": candidate._source_hash,
        "artifact_hash": candidate.artifact_hash, "kernel": candidate.kernel, "launch_rule": asdict(candidate.launch_rule), "parameters": candidate.parameters}
    doc = {"kind":"independent_frozen_kernel_comparison", "ok":True, "status":"frozen_comparison_complete", "job_id":"123456",
        "candidates":{"native_codex":frozen}, "contract":{"operation":"matmul","precision":"float32"},
        "contract_hash":"exact-contract", "hardware_identity":{"uuid":"actual-measurement-device","architecture":"sm_90"},
        "terminal":{"job_id":"123456","terminal":True,"state":"COMPLETED","exit_code":"0:0"}}
    for name in ("plan.json", "comparison.json"):
        (study / name).write_text(json.dumps(doc))
    (study / "inputs").mkdir()
    for name in ("a", "b"):
        (study / "inputs" / (name + ".bin")).write_bytes(b"\x00\x00\x80\x3f" * 17)
    request = {"source":str(source), "kernel":"matmul", "mode":"run", "device":0,
        "grid":[1,1,1], "block":[32,1,1], "shared_memory_bytes":0, "warmup":0, "iterations":1,
        "compiler_options":["--fmad=true"], "arguments":[
            {"kind":"buffer","bytes":68,"input_file":str(study / "inputs/a.bin"),"direction":"input"},
            {"kind":"buffer","bytes":68,"input_file":str(study / "inputs/b.bin"),"direction":"input"},
            {"kind":"buffer","bytes":68,"input_file":None,"direction":"output"},
            {"kind":"scalar","type":"int32","value":17}]}
    (study / "request-receipts").mkdir()
    (study / "request-receipts/actual.json").write_text(json.dumps({"request":request,"report":{"ok":True}}))
    body = study / "research/sources/downloaded.txt"
    body.parent.mkdir(parents=True)
    body.write_text("EXTERNAL_BODY_NOT_REDISTRIBUTED")
    return repo, study, candidate


def test_pilot_comparison_exports_once_with_contract_inputs_and_exact_reconstruction(pilot, tmp_path):
    repo, study, candidate = pilot
    original = {str(p.relative_to(study)):hashlib.sha256(p.read_bytes()).hexdigest() for p in study.rglob("*") if p.is_file()}
    manifest = export_study(study, tmp_path / "export", repo_dir=repo)
    assert manifest["status"] == "partial" and not manifest["errors"]
    assert manifest["completion_assessment"]["pilot_qualification_authenticated"] is False
    assert len(manifest["candidates"]) == 1
    selected = manifest["candidates"][0]
    assert selected["selection"] == "comparison.json#native_codex"
    assert selected["contract_hash"] == "exact-contract"
    assert selected["environment"]["uuid"] == "actual-measurement-device"
    rebuilt, requests, _ = reconstruct_candidate(tmp_path / "export/study-evidence.zip", selected["id"], tmp_path / "reconstructed")
    assert rebuilt.artifact_hash == candidate.artifact_hash and rebuilt.source.read_bytes() == candidate.source.read_bytes()
    assert len(requests) == 1 and requests[0].compiler_options == ["--fmad=true"]
    assert Path(requests[0].arguments[0].input_file).read_bytes() == (study / "inputs/a.bin").read_bytes()
    with zipfile.ZipFile(tmp_path / "export/study-evidence.zip") as archive:
        assert archive.testzip() is None
        assert not any(b"EXTERNAL_BODY_NOT_REDISTRIBUTED" in archive.read(name) for name in archive.namelist())
    assert manifest["generation_resume"]["guard_ready"] is False
    assert original == {str(p.relative_to(study)):hashlib.sha256(p.read_bytes()).hexdigest() for p in study.rglob("*") if p.is_file()}


def test_plan_only_is_reconstructible_but_not_completed_or_newly_certified(pilot, tmp_path):
    repo, study, _ = pilot
    (study / "comparison.json").unlink()
    manifest = export_study(study, tmp_path / "export", repo_dir=repo)
    assert manifest["status"] == "partial"
    assert manifest["candidates"][0]["selection"] == "plan.json#native_codex"
    assert manifest["candidates"][0]["reconstructable"] is True


def test_embedded_terminal_for_different_job_cannot_complete_pilot(pilot, tmp_path):
    repo, study, _ = pilot
    path = study / "comparison.json"
    doc = json.loads(path.read_text()); doc["terminal"]["job_id"] = "another-job"
    path.write_text(json.dumps(doc))
    assert export_study(study, tmp_path / "export", repo_dir=repo)["status"] == "partial"
