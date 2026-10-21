"""CPU-only controls of the targeted helper; these acquire no ROMEO evidence."""
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
import pytest

from scripts import gpu_search_policy_acceptance as module


class CPUSimulatedRunner:
    def __init__(self,directory):
        self.directory=directory; self.calls=[]
    def get_device_properties(self,device):
        return {"architecture":"sm_90","uuid":"CPU_SIMULATION_NOT_GPU_EVIDENCE",
            "max_threads_per_block":1024,"max_shared_memory_per_block_bytes":49152,
            "driver_version":12000,"nvrtc_version":"12.0"}
    def run(self,run_id,request):
        self.calls.append(request)
        text=Path(request.source).read_text()
        if "CONTROL_INVALID_SYMBOL" in text:
            return {"ok":False,"stage":"compile","error":"CPU-simulated undeclared symbol",
                "environment":self.get_device_properties(0)}
        values=np.fromfile(request.arguments[0].input_file,dtype="<f4")
        output=self.directory/(run_id+".bin"); values.tofile(output)
        delay=int(re.search(r"#define CONTROL_DELAY (\d+)ULL",text)[1])
        if f"n == {module.CRITICAL_SIZE} ?" in text and len(values)==module.CRITICAL_SIZE:
            delay=module.CRITICAL_DELAY_CYCLES
        latency=(delay+1000)/1_000_000
        return {"ok":True,"guard_readonly_ok":True,"guard_outputs_poisoned":True,
            "outputs":[{"path":str(output),"bytes":values.nbytes}],
            "environment":self.get_device_properties(0),"latency_ms":latency,
            "latency_samples_ms":[latency]*request.iterations}


def test_default_plan_is_local_declared_and_frozen_without_any_runner_or_lease(tmp_path,monkeypatch):
    monkeypatch.setattr(module,"SlurmBackend",lambda *a,**k:pytest.fail("Default plan must never create a worker"))
    assert module.main(["--output",str(tmp_path),"--run-id","local-control"])==0
    plan=json.loads((tmp_path/"plan.json").read_text())
    assert not plan["submitted"] and plan["model_calls"]==plan["model_tokens"]==0
    assert plan["selection_seed"]==1729 and plan["selection_trials"]==16
    assert plan["max_repairs"]==2 and plan["planned_criteria"]==["AC-05","AC-06","AC-10"]
    assert not (tmp_path/"allocation-intent.json").exists()
    assert set(plan["static_sources"])=={"baseline","fast","slow","critical_regression"}
    assert len(plan["inputs_sha256"])==5
    assert plan["performance_cases"][0]["critical"] is True
    assert [row["weight"] for row in plan["performance_cases"]]==[.05,.95]
    for name,text in plan["static_sources"].items():
        assert str(module.DELAY_CYCLES[name])+"ULL" in text
        assert "clock64()" in text and 'asm volatile("" ::: "memory")' in text
        assert "__syncthreads" in text and "output[i] = input[i]" in text


def test_real_treequest_durable_closure_and_exact_generation_with_cpu_simulated_times(tmp_path):
    runner=CPUSimulatedRunner(tmp_path)
    evaluator,baseline,plan=module.prepare(tmp_path,"cpu-policy-control",runner=runner,
        properties=runner.get_device_properties(0))
    receipt=module.exercise(tmp_path,evaluator,baseline,plan)
    assert receipt["execution_kind"]=="cpu_simulated" and receipt["status"]=="cpu_simulated_checks_only"
    assert receipt["ok"] is False  # Never claim GPU acceptance from injected samples.
    assert receipt["treequest"]["source_pin_verified"]
    assert receipt["model_calls"]==receipt["model_tokens"]==0
    assert receipt["criteria_checks"]["AC-06"]
    assert receipt["criteria_checks"]["AC-10"]
    assert receipt["snapshot_after_closure_sha256"]==receipt["restored_snapshot_sha256"]
    assert receipt["snapshot_before_closure_sha256"]!=receipt["restored_snapshot_sha256"]
    assert receipt["checks"]["selected_source_and_parameters_received_exactly"]
    assert receipt["checks"]["confirmed_incumbent_preserved"]
    decision=receipt["branch_decision"]
    assert decision["scope"]=="parent_action" and decision["counters"]["repairs"]==2
    assert len(receipt["repair_results"])==3
    assert all(row["status"]=="compilation_error" for row in receipt["repair_results"])
    closed_parent,closed_action=decision["target_id"]
    assert all((row["parent_state"]["candidate_id"],row["action"])!=(closed_parent,closed_action)
               for row in receipt["post_closure_batch"])
    assert receipt["continued_branch"]["selected_parent"]["candidate_id"]!=closed_parent
    assert receipt["continued_branch"]["measurement"]["ok"]
    for row in receipt["generator_receipts"]:
        assert row["selected_parent"]==row["generator_input"]
        assert row["generator_input_sha256"]==module.digest(row["selected_parent"])
        assert row["variant_override"] is None
        assert row["candidate"]["parameters"]==row["selected_parent"]["parameters"]
    with sqlite3.connect(tmp_path/"experiments.sqlite") as db:
        events=[(kind,json.loads(payload)) for kind,payload in db.execute("SELECT kind,payload FROM events ORDER BY rowid")]
        closure=[index for index,row in enumerate(events) if row[0]=="branch_decision"]
        assert len(closure)==1
        assert all(row[0]!="branch_decision" for row in events[:closure[0]])
        assert db.execute("SELECT count(*) FROM attempts").fetchone()[0]==3
        assert db.execute("SELECT count(*) FROM events WHERE kind='promotion'").fetchone()[0]==1
    slow_rows=[row for row in receipt["selections"] if row["selected_slow_measured_parent"]]
    assert receipt["criteria_checks"]["AC-05"]==bool(slow_rows)
    if slow_rows:
        selected=slow_rows[0]
        assert selected["parent_latency_ms"]>selected["best_latency_ms"]
        assert selected["parent"]["parameters"]["control_variant"]=="slow"


def test_missed_slow_selection_is_honestly_partial_without_forcing_policy(tmp_path,monkeypatch):
    runner=CPUSimulatedRunner(tmp_path)
    evaluator,baseline,plan=module.prepare(tmp_path,"cpu-partial",runner=runner,
        properties=runner.get_device_properties(0),selection_trials=1)
    # The contract declaration intentionally makes the slow-labelled control
    # equally fast for this CPU test. A name alone must never prove AC-05.
    slow_source=module.source
    monkeypatch.setattr(module,"source",lambda variant:slow_source("fast" if variant=="slow" else variant))
    receipt=module.exercise(tmp_path,evaluator,baseline,plan)
    assert not receipt["checks"]["slow_control_paired_interval_below_one"]
    assert not receipt["checks"]["slow_parent_naturally_selected"]
    assert receipt["criteria_checks"]["AC-05"] is False
    assert not receipt["functional_checks_passed"] and not receipt["ok"]
    assert len(receipt["selections"])==1


def test_existing_experiment_cannot_be_replanned(tmp_path):
    (tmp_path/"experiments.sqlite").write_bytes(b"existing evidence")
    with pytest.raises(ValueError,match="replanned"):
        module.main(["--output",str(tmp_path)])


def test_critical_rejection_with_aggregate_loss_does_not_acquire_the_missing_ac10_control(tmp_path,monkeypatch):
    runner=CPUSimulatedRunner(tmp_path)
    evaluator,baseline,plan=module.prepare(tmp_path,"cpu-critical-scope",runner=runner,
        properties=runner.get_device_properties(0),selection_trials=1)
    static_source=module.source
    monkeypatch.setattr(module,"source",lambda variant:static_source("slow" if variant=="critical_regression" else variant))
    receipt=module.exercise(tmp_path,evaluator,baseline,plan)
    critical=receipt["critical_control"]["confirmation"]
    assert critical["status"]=="not_improved" and critical["critical_cases_ok"] is False
    assert critical["interval"]["upper"]<1
    assert receipt["checks"]["confirmed_incumbent_preserved"]
    assert not receipt["checks"]["critical_control_aggregate_lower_bound_above_threshold"]
    assert not receipt["criteria_checks"]["AC-10"] and not receipt["ok"]
