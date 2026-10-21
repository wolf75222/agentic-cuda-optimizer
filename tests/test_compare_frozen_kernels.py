"""Frozen comparison orchestration; workers/controls here are explicitly simulated."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts/compare_frozen_kernels.py"
SPEC = importlib.util.spec_from_file_location("compare_frozen_kernels_script", MODULE)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    directory = tmp_path / "old-pilot"
    config = helper.CampaignConfig(run_id="old-pilot", search_sizes=(128,256,512), final_sizes=(96,192,384),
        benchmark_series=2, confirmation_batches=2, launches_per_series=2)
    source = (helper.REPO / "examples/naive_matmul.cu").read_text()
    generated = {}
    for role in ("agent", "vanilla"):
        helper.write_json(directory / role / "config.json", replace(config, run_id=config.run_id+"."+role).to_dict())
        baseline = directory / role / "evaluator_data/baseline.cu"
        baseline.parent.mkdir(parents=True)
        baseline.write_text(source)
        path = directory / "direct-comparison" / (role+".cu")
        path.parent.mkdir(exist_ok=True)
        path.write_text(source + "\n// " + role)
        generated[role] = helper.Candidate(role+"-generated", path, "matmul", helper.LaunchRule(kind="square_2d",block=(16,16,1)))
    helper.write_json(directory / "comparison.json", {"ok":False,"status":"incomplete",
        "old_negative_receipt":True,"old_timing_samples":[999999,888888]})
    helper.write_json(directory / "worker-terminal.json", {"terminal":True,"state":"COMPLETED","exit_code":"0:0","job_id":"old111"})
    original_pilot_bytes = (directory / "comparison.json").read_bytes()
    native_source = tmp_path / "native-frozen/kernel.cu"
    native_source.parent.mkdir()
    native_source.write_text(source+"\n// normal Codex, one prompt, tools enabled")
    candidate = helper.Candidate("codex-vanilla",native_source,"matmul",helper.LaunchRule(kind="square_2d",block=(16,8,1)))
    native_path = native_source.parent / "candidate.json"
    helper.write_json(native_path,{**helper.candidate_receipt(candidate),
        "provenance":{"kind":"native_codex_one_prompt","requested_model":"gpt-6-luna","reasoning_effort":"max",
                      "actual_models":["unit-test-simulation"],"worker_session":"native-generation-only"}})
    def load(selected_config, selected_directory):
        assert selected_config.run_id == config.run_id and selected_directory == directory
        return generated
    monkeypatch.setattr(helper,"load_direct_candidates",load)
    return directory,native_path,config,original_pilot_bytes


def test_cpu_plan_uses_committed_seed_new_disjoint_shapes_and_no_worker(artifacts,tmp_path,monkeypatch,capsys):
    directory,native_path,config,old_bytes = artifacts
    monkeypatch.setattr(helper,"SlurmBackend",lambda *a,**k:pytest.fail("Planning may not submit or contact a worker"))
    seed = 1234567
    output = tmp_path / "new-plan"
    commitment = hashlib.sha256(str(seed).encode("ascii")).hexdigest()
    assert helper.main([str(directory),"--native-candidate",str(native_path),"--seed",str(seed),
                        "--seed-commitment",commitment,"--output",str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["submitted"] == 0
    plan = helper.read_json(output / "plan.json")
    assert plan["seed_commitment"] == commitment
    assert len(plan["input_hashes"]) == 6 and set(plan["candidates"]) == set(helper.ROLES)
    assert 65 <= plan["shapes"][0] <= 127 and 129 <= plan["shapes"][1] <= 255 and 257 <= plan["shapes"][2] <= 511
    assert not set(plan["shapes"]) & set(config.search_sizes+config.final_sizes)
    assert tuple(plan["shapes"]) == helper.select_shapes(seed,config.search_sizes,config.final_sizes)
    assert (directory / "comparison.json").read_bytes() == old_bytes
    with pytest.raises(ValueError,match="fresh"):
        helper.prepare(directory,native_path,seed,output)
    with pytest.raises(ValueError,match="commitment"):
        helper.prepare(directory,native_path,seed,tmp_path/"wrong-seed",expected_commitment="wrong")


def test_changed_native_artifact_fails_before_any_final_input_or_gpu(artifacts,tmp_path):
    directory,native_path,_,_ = artifacts
    receipt = helper.read_json(native_path)
    Path(receipt["source_path"]).write_text("changed after native generation")
    output = tmp_path/"invalid-source"
    with pytest.raises(ValueError,match="frozen"):
        helper.prepare(directory,native_path,9,output)
    assert not output.exists()


@pytest.mark.parametrize("correctness_failure",[False,True])
def test_fresh_context_all_controls_precede_new_pairs_and_failures_remain_explicit(artifacts,tmp_path,monkeypatch,correctness_failure):
    directory,native_path,_,old_bytes = artifacts
    output = tmp_path / "fresh-execution"
    prepared = helper.prepare(directory,native_path,987654321,output)
    candidates = prepared[3]
    properties = {"name":"SIMULATED UNIT TEST GPU","uuid":"new-unit-context","architecture":"sm_90",
        "driver_version":13010,"nvrtc_version":"12.6","cuda_toolkit_version":12060,"harness_build_id":"simulated-build"}
    events,instances = [],[]
    class Runner:
        results_dir = output/"simulated-worker"
        def get_device_properties(self,device):
            return properties
        def run(self,run_id,request):
            role = Path(request.source).stem
            events.append(("benchmark",role))
            latency = {"baseline":2.0,"agent":.5,"single_prompt":1.0,"native_codex":.8}[role]
            return {"ok":True,"environment":properties,"guard_readonly_ok":True,
                    "latency_samples_ms":[latency]*request.iterations,"reservation_wall_seconds":.01,
                    "compile_ms":.1,"compile_cache_hit":True}
    class Backend:
        hardware = {"device_properties":properties}
        def __init__(self,config,repo,results,*,session_id):
            assert config.walltime == "24:00:00"
            self.job_id,self.closed,self.session_id = None,False,session_id
            instances.append(self)
        def start(self):
            self.job_id = "new222"
            return self.job_id
        def wait_until_ready(self):
            pass
        def runner(self):
            return Runner()
        def close(self,*,cancel=False):
            self.closed = True
        def wait_until_finished(self,timeout):
            return {"job_id":self.job_id,"terminal":True,"state":"COMPLETED","exit_code":"0:0"}
    monkeypatch.setattr(helper,"SlurmBackend",Backend)
    real_factory = helper.make_evaluator
    def simulated_controls(config,suite,cases,runner,props):
        evaluator = real_factory(config,suite,cases,runner,props)
        def validate(candidate,bound_suite):
            events.append(("validate",candidate.candidate_id,bound_suite))
            failed = correctness_failure and candidate is candidates["native_codex"]
            if not failed:
                evaluator._validated.add((candidate.artifact_hash,bound_suite))
            return {"ok":not failed,"status":"numerical_error" if failed else "validated",
                    "evaluation_id":uuid.uuid4().hex,"cases":[{"case_id":case.case_id} for case in cases]}
        def qualify(candidate,bound_suite):
            assert bound_suite == "performance"
            events.append(("qualify",candidate.candidate_id))
            evaluator._validated.add((candidate.artifact_hash,"performance"))
            return {"ok":True,"status":"sanitizer_qualified","artifact_hash":candidate.artifact_hash,
                    "contract_hash":evaluator.contract_hash,"required_sanitizers":list(helper.TOOLS),
                    "sanitizers":[{"ok":True,"tool":tool,"case_id":case.case_id,"reservation_wall_seconds":.02}
                                  for tool in helper.TOOLS for case in cases]}
        evaluator.validate,evaluator.qualify_sanitizers = validate,qualify
        return evaluator
    monkeypatch.setattr(helper,"make_evaluator",simulated_controls)
    result = helper.execute(prepared,output,helper.REPO/"configs/romeo.json")
    assert instances[0].closed and instances[0].session_id.startswith("frozen-finals-")
    assert (directory/"comparison.json").read_bytes() == old_bytes
    assert result["job_id"] == "new222" and result["model_calls"] == 0
    if correctness_failure:
        assert not result["ok"] and result["status"] == "incomplete"
        assert not result["paired_comparisons"] and not any(event[0] == "benchmark" for event in events)
    else:
        assert result["ok"] and result["status"] == "frozen_comparison_complete"
        assert result["sanitizer_usage"]["tool_calls"] == 48
        first_benchmark = next(i for i,event in enumerate(events) if event[0] == "benchmark")
        assert sum(event[0] == "qualify" for event in events[:first_benchmark]) == 4
        assert set(result["paired_comparisons"]) == {pair[0] for pair in helper.PAIRS}
        assert result["paired_comparisons"]["agent_vs_baseline"]["interval"]["gain"] == pytest.approx(4)
        assert result["paired_comparisons"]["agent_vs_native_codex"]["interval"]["gain"] == pytest.approx(1.6)
        assert result["measurement_usage"]["benchmark_requests"] == 48
        ids = [row["evaluation_id"] for row in result["paired_comparisons"].values()]
        assert len(set(ids)) == 4
    with pytest.raises(ValueError,match="already started"):
        helper.execute(prepared,output,helper.REPO/"configs/romeo.json")
