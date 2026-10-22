"""Synthetic CPU filesystem receipts; never invoke CUDA, SSH or a model."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import pytest

from optimizer_agent import artifact_qualification_incidents as module
from optimizer_agent.compile_cache import content_hash
from optimizer_agent.evaluation import LaunchRule


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    run = "quality.synthetic.normalization.no_autotuning"
    source = b'extern "C" __global__ void normalization(const float*a,const float*b,float*c,int n){\n for(int i=threadIdx.x;i<n;i+=blockDim.x)c[i]=a[i];\n}\n'
    source_hash = hashlib.sha256(source).hexdigest()
    rule = asdict(LaunchRule(block=(256, 1, 1), tile_x=1))
    # JSON receipts contain lists, as does the native launch.
    rule = json.loads(json.dumps(rule))
    artifact = content_hash({"source":source_hash, "kernel":"normalization", "launch_rule":rule,
                             "parameters":json.dumps({}, sort_keys=True, allow_nan=False)})
    identity = {"run_id":run, "candidate_id":"c" * 24, "artifact_hash":artifact,
        "source_sha256":source_hash, "launch_rule_sha256":content_hash(rule), "controller_source_sha":"a" * 40,
        "job_id":"123", "session_id":"suite-gen-" + "b" * 32, "trial_id":"trial-1",
        "evaluation_request_id":"0" * 64, "evaluation_id":"d" * 32,
        "task_epoch_id":"000001-" + "1" * 32, "controller_epoch_id":"000001-" + "2" * 32}
    candidate = {"candidate_id":identity["candidate_id"], "source":"/historical/path/best.cu", "source_hash":source_hash,
        "artifact_hash":artifact, "kernel":"normalization", "launch_rule":rule, "parameters":{},
        "certification_status":"generation_complete"}
    request = {"schema_version":1, "run_id":run, "trial_id":identity["trial_id"], "contract_hash":"e" * 64,
        "template":{**{k:candidate[k] for k in ("candidate_id", "artifact_hash", "kernel", "launch_rule", "parameters")},
                    "source_sha256":source_hash}}
    identity["evaluation_request_id"] = hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()
    paths = module._paths(identity)
    launch = {"grid":[17, 1, 1], "block":[256, 1, 1], "shared_memory_bytes":0}
    native = {"ok":True, "slurm_job_id":"123", "environment":{"uuid":"f" * 32},
        "compilation_identity":{"source":source.decode(), "options":[]}}
    measurement = {"evaluation_id":identity["evaluation_id"], "candidate_id":identity["candidate_id"], "artifact_hash":artifact,
        "contract_hash":request["contract_hash"], "status":"measured", "ok":True, "attempt_id":"9" * 32,
        "cases":{"search-17":{"series":[{"launch":launch, "report":native, "raw_ms":[1.0]} for _ in range(2)]}}}
    result = {"run_id":run, "status":"generation_complete", "generation_complete":True, "generation_only":True,
        "best_candidate_id":identity["candidate_id"], "artifact_hash":artifact, "candidate":candidate,
        "job_id":"123", "session_id":identity["session_id"], "environment":{"uuid":"f" * 32},
        "contract_hash":request["contract_hash"],
        "worker_terminal":{"job_id":"123", "state":"COMPLETED", "exit_code":"0:0", "terminal":True}}
    review = {"kind":"independent_manual_static_artifact_review", "review_method":"manual_static_source_and_archived_launch",
        "identity":identity, "issue":module.ISSUE, "verdict":"confirmed_unsynchronized_global_writers",
        "scientific_qualification_eligible":False, "scope":"Synthetic CPU receipt, no actual GPU proof",
        "archived_launches":{"search-17":{"series":2, "launch":launch}}}
    values = {"result":result, "candidate":candidate, "best_launch":candidate, "source":source,
        "request":request, "measurement":measurement, "worker_session":{"job_id":"123", "session_id":identity["session_id"]}}
    evidence = {role:{"path":paths[role], "sha256":write(root / paths[role], value)} for role,value in values.items()}
    controller = {"kind":"controller", "epoch_id":identity["controller_epoch_id"],
        "source_identity":{"git":{"available":True, "HEAD_at_dispatch":identity["controller_source_sha"]}}}
    controller["record_hash"] = content_hash(controller)
    evidence["controller_epoch"] = {"path":paths["controller_epoch"], "sha256":write(root / paths["controller_epoch"], controller)}
    task = {"kind":"task", "epoch_id":identity["task_epoch_id"], "source_identity":controller["source_identity"],
        "task":{"task_id":run, "kind":"generation", "session":identity["session_id"]},
        "controller_epoch":{"epoch_id":identity["controller_epoch_id"], **evidence["controller_epoch"]}}
    task["record_hash"] = content_hash(task)
    evidence["task_epoch"] = {"path":paths["task_epoch"], "sha256":write(root / paths["task_epoch"], task)}
    completion = {"request_id":identity["evaluation_request_id"], "trial_id":identity["trial_id"],
        "candidate_id":identity["candidate_id"], "evaluation_id":identity["evaluation_id"], "status":"measured",
        "attempt_id":measurement["attempt_id"], "files":{"request.json":evidence["request"]["sha256"], "measurement.json":evidence["measurement"]["sha256"]}}
    evidence["completion"] = {"path":paths["completion"], "sha256":write(root / paths["completion"], completion)}
    prefix = "campaigns/" + run + "/generation/"
    trial = {"candidate_id":identity["candidate_id"], "tell":{"evaluation_id":identity["evaluation_id"], "valid":True},
        "evaluation_evidence":{"measurement":{"identity":identity["evaluation_id"], "path":paths["measurement"][len(prefix):], "sha256":evidence["measurement"]["sha256"]}}}
    evidence["trial_result"] = {"path":paths["trial_result"], "sha256":write(root / paths["trial_result"], trial)}
    review["positive_evidence"] = deepcopy(evidence)
    evidence["review"] = {"path":paths["review"], "sha256":write(root / paths["review"], review)}
    incident = {"schema_version":1, "kind":"declared_artifact_qualification_incident",
        "incident_id":module.ISSUE + "-" + artifact, "status":"confirmed_static_violation", "issue":module.ISSUE,
        "scientific_qualification_eligible":False, "identity":identity, "evidence":evidence,
        "scope":"Explicit synthetic artifact disqualification; no general source classifier"}
    path = root / "artifact-incidents" / artifact / "incident.json"
    write(path, incident)
    return root, incident, path


def load(study):
    root, incident, _ = study
    return module.load_artifact_qualification_incidents(root, [incident["identity"]["run_id"]])


def repin(study, role, mutate):
    root, incident, path = study
    file = root / incident["evidence"][role]["path"]
    value = json.loads(file.read_text())
    mutate(value)
    incident["evidence"][role]["sha256"] = write(file, value)
    if role == "measurement":
        completion_path = root / incident["evidence"]["completion"]["path"]
        completion = json.loads(completion_path.read_text())
        completion["files"]["measurement.json"] = incident["evidence"][role]["sha256"]
        incident["evidence"]["completion"]["sha256"] = write(completion_path, completion)
        trial_path = root / incident["evidence"]["trial_result"]["path"]
        trial = json.loads(trial_path.read_text())
        trial["evaluation_evidence"]["measurement"]["sha256"] = incident["evidence"][role]["sha256"]
        incident["evidence"]["trial_result"]["sha256"] = write(trial_path, trial)
    if role != "review":
        review_path = root / incident["evidence"]["review"]["path"]
        review = json.loads(review_path.read_text())
        review["positive_evidence"] = {k:deepcopy(v) for k,v in incident["evidence"].items() if k != "review"}
        incident["evidence"]["review"]["sha256"] = write(review_path, review)
    write(path, incident)


def rejected(study, unresolved=False):
    mapping, observations = load(study)
    assert not mapping
    assert len(observations) == 1 and observations[0]["status"] == "uncertain"
    assert observations[0]["artifact_hash"] == (None if unresolved else study[1]["identity"]["artifact_hash"])
    assert observations[0]["qualification_status"] == "not_admitted"
    assert "scientific_qualification_eligible" not in observations[0]


def test_authenticates_exact_declared_refusal_without_mutating_evidence(study):
    root, incident, path = study
    before = {p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}
    mapping, observations = load(study)
    artifact = incident["identity"]["artifact_hash"]
    assert mapping[artifact][0]["scientific_qualification_eligible"] is False
    assert mapping[artifact][0]["authentication"]["incident_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert observations[0]["status"] == "confirmed"
    assert before == {p.relative_to(root):p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_absence_is_not_positive_eligibility(tmp_path):
    assert module.load_artifact_qualification_incidents(tmp_path, ["run"]) == ({}, [])


@pytest.mark.parametrize("role,path", [("source", "/etc/passwd"), ("measurement", "private/experiments.sqlite"),
    ("review", "artifact-incidents/../tool.json"), ("source", "campaigns/alias/generation/best.cu")])
def test_refuses_arbitrary_alias_tool_private_or_traversal_roles(study, role, path):
    study[1]["evidence"][role]["path"] = path
    write(study[2], study[1])
    rejected(study)


@pytest.mark.parametrize("role", ["source", "measurement", "review", "result", "task_epoch"])
def test_pin_tampering_is_uncertain(study, role):
    file = study[0] / study[1]["evidence"][role]["path"]
    file.write_bytes(file.read_bytes() + b"\n")
    rejected(study)


@pytest.mark.parametrize("role,mutate", [
    ("result", lambda x:x["worker_terminal"].update(terminal=False)),
    ("worker_session", lambda x:x.update(job_id="124")),
    ("candidate", lambda x:x.update(source_hash="0" * 64)),
    ("best_launch", lambda x:x["launch_rule"].update(tile_x=256)),
    ("measurement", lambda x:x.update(evaluation_id="0" * 32)),
    ("measurement", lambda x:x.update(cases=[])),
    ("completion", lambda x:x.update(attempt_id="0" * 32)),
    ("review", lambda x:x.update(scientific_qualification_eligible=True)),
    ("review", lambda x:x["archived_launches"]["search-17"].update(series=1)),
    ("task_epoch", lambda x:x.update(record_hash="0" * 64)),
    ("measurement", lambda x:x["cases"]["search-17"]["series"][0]["report"].update(slurm_job_id="124")),
    ("measurement", lambda x:x["cases"]["search-17"]["series"][0]["report"]["environment"].update(uuid="0" * 32)),
])
def test_repinning_a_conflicting_role_does_not_authorize_it(study, role, mutate):
    repin(study, role, mutate)
    rejected(study)


@pytest.mark.parametrize("target", ["source", "review", "directory", "registry"])
def test_refuses_file_and_directory_symlinks(study, target, tmp_path):
    root, incident, path = study
    selected = (root / incident["evidence"][target]["path"] if target in {"source", "review"}
                else path.parent if target == "directory" else root / "artifact-incidents")
    moved = tmp_path / ("moved-" + target)
    selected.rename(moved)
    selected.symlink_to(moved, target_is_directory=moved.is_dir())
    mapping, observations = load(study)
    assert not mapping and observations[0]["status"] == "uncertain"


@pytest.mark.parametrize("raw", [b"[]", b"null", b'{"identity":{},"identity":{}}', b'{"value":NaN}', b'{"broken":'])
def test_malformed_incident_never_crashes_or_admits(study, raw):
    study[2].write_bytes(raw)
    rejected(study, unresolved=True)


def test_out_of_scope_does_not_follow_another_run_and_never_clears_artifact(study, monkeypatch):
    called, original = [], module._read_public
    def spy(*args, **kwargs):
        called.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "_read_public", spy)
    mapping, observations = module.load_artifact_qualification_incidents(study[0], ["different-run"])
    assert not mapping and observations[0]["status"] == "uncertain" and observations[0]["reason"] == "out_of_scope"
    assert len(called) == 1 and called[0].endswith("/incident.json")


def test_oversized_measurement_refused_without_partial_parsing(study):
    file = study[0] / study[1]["evidence"]["measurement"]["path"]
    raw = b" " * (module.MAX_ROLE_BYTES + 1)
    file.write_bytes(raw)
    study[1]["evidence"]["measurement"]["sha256"] = hashlib.sha256(raw).hexdigest()
    write(study[2], study[1])
    rejected(study)


def test_missing_identity_is_global_uncertainty_even_with_hash_named_directory(study):
    run = study[1]["identity"]["run_id"]
    artifact = study[1]["identity"]["artifact_hash"]
    study[1].pop("identity")
    write(study[2], study[1])
    mapping, observations = module.load_artifact_qualification_incidents(study[0], [run])
    assert not mapping and observations[0]["status"] == "uncertain"
    assert observations[0]["artifact_hash"] is None
    assert observations[0]["directory_artifact_hash"] == artifact


def test_unresolved_directory_is_global_uncertainty(study):
    run = study[1]["identity"]["run_id"]
    study[2].parent.rename(study[2].parent.with_name("unknown"))
    mapping, observations = module.load_artifact_qualification_incidents(study[0], [run])
    assert not mapping and observations[0]["status"] == "uncertain"
    assert observations[0]["artifact_hash"] is None


def test_rejects_string_run_list(study):
    with pytest.raises(ValueError, match="iterable"):
        module.load_artifact_qualification_incidents(study[0], study[1]["identity"]["run_id"])


@pytest.mark.parametrize("in_scope", [True, False])
def test_conflicting_directory_and_declared_artifacts_cannot_clear_either(study, in_scope):
    original = study[1]["identity"]["artifact_hash"]
    declared = "0" * 64
    study[1]["identity"]["artifact_hash"] = declared
    write(study[2], study[1])
    run = study[1]["identity"]["run_id"] if in_scope else "different-run"
    mapping, observations = module.load_artifact_qualification_incidents(study[0], [run])
    assert not mapping and observations[0]["status"] == "uncertain"
    assert observations[0]["artifact_hash"] is None
    assert observations[0]["directory_artifact_hash"] == original
    assert observations[0]["declared_artifact_hash"] == declared


def test_fifo_is_opened_nonblocking_and_refused_before_read(study, monkeypatch):
    file = study[0] / study[1]["evidence"]["source"]["path"]
    file.unlink()
    os.mkfifo(file)
    original = module.os.open
    def safe_open(path, flags, *args, **kwargs):
        if str(path) == "best.cu":
            assert flags & os.O_NONBLOCK, "FIFO open must not be allowed to block"
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(module.os, "open", safe_open)
    rejected(study)


def test_hardlink_is_refused_before_reading_aliased_bytes(study, tmp_path, monkeypatch):
    file = study[0] / study[1]["evidence"]["source"]["path"]
    alias = tmp_path / "synthetic-protected-source.cu"
    file.rename(alias)
    os.link(alias, file)
    original = module.os.read
    def safe_read(descriptor, *args, **kwargs):
        assert os.fstat(descriptor).st_nlink == 1, "Aliased bytes must not be read"
        return original(descriptor, *args, **kwargs)
    monkeypatch.setattr(module.os, "read", safe_read)
    rejected(study)
