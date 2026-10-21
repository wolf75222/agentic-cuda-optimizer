"""Actual SSH diagnostic forms and lost local submission acknowledgements.

All backends here are in-memory doubles; no model, SSH or scheduler is called.
"""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess

import pytest

from optimizer_agent.codex_backend import CodexError
from optimizer_agent.execution_backend import BackendError, SlurmConfig
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite
from optimizer_agent.suite_orchestrator import SuiteOrchestrator, infrastructure_category, write_json


@pytest.mark.parametrize("diagnostic,category",[
    ("ssh: Could not resolve hostname romeo: nodename nor servname provided, or not known", "ssh_transport_failure"),
    ("ssh: Could not resolve hostname romeo: Temporary failure in name resolution", "ssh_transport_failure"),
    ("ssh: connect to host romeo port 22: Can't assign requested address", "ssh_transport_failure"),
    ("ssh: connect to host romeo port 22: Cannot assign requested address", "ssh_transport_failure"),
    ("ssh: connect to host romeo port 22: Connection refused", "ssh_transport_failure"),
    ("ssh: connect to host romeo port 22: No route to host", "ssh_transport_failure"),
    ("ssh: connect to host romeo port 22: Network is unreachable", "ssh_transport_failure"),
    ("ssh: connect to host romeo port 22: Connection failure", "ssh_transport_failure"),
    ("kex_exchange_identification: read: Connection reset by peer", "ssh_transport_failure"),
    ("client_loop: send disconnect: Broken pipe", "ssh_transport_failure"),
    ("ssh: connect to host romeo port 22: Operation timed out", "transport_timeout"),
    ("ssh: connect to host romeo port 22: Connection timed out", "transport_timeout"),
    ("ssh_dispatch_run_fatal: Connection to romeo port 22: Operation timed out", "transport_timeout"),
    ("Timeout, server romeo not responding.", "transport_timeout"),
    ("SSH connection timed out after 15 seconds", "transport_timeout"),
    ("Atomic SSH upload timed out after 3 identical attempts", "transport_timeout"),
    ("Remote artifact read timed out after 3 identical attempts", "transport_timeout"),
    ("Remote request timed out before reply", "transport_timeout"),
    ("Worker session unit is terminal (job 771111, FAILED). Use a new session.", "worker_terminal_incomplete"),
])
def test_real_ssh_diagnostics_are_infrastructure(diagnostic,category):
    assert infrastructure_category(BackendError(diagnostic)) == category
    assert infrastructure_category(diagnostic) == category


@pytest.mark.parametrize("error",[
    TimeoutError("model timed out"),
    TimeoutError("compilation exceeded its deadline"),
    subprocess.TimeoutExpired(["codex","exec"],30),
    subprocess.TimeoutExpired(["nvcc","kernel.cu"],30),
    subprocess.CalledProcessError(255,["codex","exec"],stderr=b"ssh: Could not resolve hostname romeo: missing"),
    subprocess.CalledProcessError(1,["ssh","romeo","nvcc"],stderr=b"nvcc fatal: compilation failed"),
    CodexError("ssh: Could not resolve hostname model-provider: unknown"),
    RuntimeError("Compilation failed: ssh: connect to host romeo port 22: Connection refused"),
    RuntimeError("Model response quoted:\nssh: connect to host romeo port 22: Connection refused"),
    RuntimeError("nvrtc compilation error: Can't assign requested address"),
    RuntimeError("kernel numerical_error"),
    RuntimeError("sanitizer_qualification_failed"),
    RuntimeError("Connection failure"),
    RuntimeError("No significant gain; not_improved"),
    RuntimeError("Remote job ownership receipt session mismatch"),
])
def test_model_compiler_kernel_and_ambiguous_failures_are_not_transport(error):
    assert infrastructure_category(error) is None


def test_typed_ssh_subprocess_timeout_and_failure_retain_transport_context():
    assert infrastructure_category(subprocess.TimeoutExpired(["/usr/bin/ssh","romeo","cat","receipt"],15)) == "transport_timeout"
    assert infrastructure_category(subprocess.CalledProcessError(255,["ssh","romeo"],
        stderr=b"ssh: Could not resolve hostname romeo: unknown")) == "ssh_transport_failure"


def plan():
    value = plan_suite(CampaignConfig(run_id="ssh-classification"),seeds=(1,2))
    value["entries"] = [entry for entry in value["entries"] if entry["mode"] == "full"
        and entry["workload"] == "matmul" and entry["seed"] == 1]
    value["modes"] = ["full"]
    return value


class AcknowledgementThenFailure:
    def __init__(self,task,directory,*,mutate=lambda record:None,memory_job="771111",historical_only=False):
        self.session_id = task["session"]
        self.results_dir = Path(directory)/"worker"
        self.config = SlurmConfig()
        self.remote_dir = self.config.remote_root+"/sessions/"+self.session_id
        self.job_id = None
        self.memory_job,self.mutate = memory_job,mutate
        self.historical_only = historical_only
        self.cleanup = []

    def start(self):
        record = {"job_id":"771111","session_id":self.session_id,"remote_dir":self.remote_dir,
                  "config":json.loads(json.dumps(asdict(self.config)))}
        self.mutate(record)
        write_json(self.results_dir/"slurm-session.json",record)
        self.job_id = self.memory_job
        raise BackendError("ssh: connect to host romeo port 22: Can't assign requested address")

    def close(self):
        self.cleanup.append("close-authenticated-own-worker")

    def wait_until_finished(self,timeout):
        self.cleanup.append("wait-authenticated-own-worker")
        return {"job_id":self.job_id,"terminal":False,"state":"UNKNOWN"}


def run_start_failure(tmp_path,*,mutate=lambda record:None,memory_job="771111",journal_job=None,historical_only=False):
    instances = []
    def backend(task,config,directory):
        instance = AcknowledgementThenFailure(task,directory,mutate=mutate,memory_job=memory_job,
            historical_only=historical_only)
        instances.append(instance)
        return instance
    def no_generation(*args,**kwargs):
        pytest.fail("A failed SSH start must not run generation/model/GPU")
    value = SuiteOrchestrator(plan(),tmp_path,backend_factory=backend,generate=no_generation,qualify=no_generation)
    task = value.journal.tasks("generation")[0]
    if journal_job is not None:
        value.journal.update(task["id"],job_id=journal_job)
        task = value.journal.tasks("generation")[0]
    value._task(task)
    return value,value.journal.tasks("generation")[0],instances[0]


@pytest.mark.parametrize("memory_job",["771111",None])
def test_start_exception_keeps_authenticated_local_job_in_journal_without_attaching(tmp_path,memory_job):
    value,row,backend = run_start_failure(tmp_path,memory_job=memory_job)
    assert row["state"] == "recovery_required" and row["job_id"] == "771111"
    assert row["result"]["job_id"] == "771111"
    assert row["result"]["infrastructure_category"] == "ssh_transport_failure"
    assert row["result"]["execution_complete"] is False
    assert json.loads((value._directory(row)/"failure.json").read_text())["job_id"] == "771111"
    if memory_job is None:
        assert backend.job_id is None and backend.cleanup == []  # Ledger repair is not an attachment.


@pytest.mark.parametrize("mutate",[
    lambda row:row.update(session_id="someone-else"),
    lambda row:row.update(remote_dir="/different/owned/session"),
    lambda row:row["config"].update(host="another-host"),
    lambda row:row.pop("config"),
    lambda row:row.update(job_id=True),
    lambda row:row.update(job_id="0"),
    lambda row:row.update(job_id="771112"),
    lambda row:row.update(uncertain=True),
    lambda row:row.update(job_ids=["771111","771112"]),
])
def test_mismatched_or_ambiguous_local_receipt_is_never_adopted_or_cleaned_up(tmp_path,mutate):
    _,row,backend = run_start_failure(tmp_path,mutate=mutate)
    assert row["job_id"] is None and row["result"]["job_id"] is None
    assert row["result"]["job_ownership_error"]
    assert backend.cleanup == []


def test_existing_journal_job_cannot_be_replaced_by_different_local_acknowledgement(tmp_path):
    _,row,backend = run_start_failure(tmp_path,journal_job="771999")
    assert row["job_id"] == "771999" and row["result"]["job_id"] is None
    assert row["result"]["journal_job_id"] == "771999"
    assert "ownership mismatch" in row["result"]["job_ownership_error"]
    assert backend.cleanup == []


def test_failed_historical_attachment_cannot_be_restored_from_local_receipt_alone(tmp_path):
    _,row,backend = run_start_failure(tmp_path,memory_job=None,historical_only=True)
    assert row["job_id"] is None and row["result"]["job_ownership_error"]
    assert backend.cleanup == []
