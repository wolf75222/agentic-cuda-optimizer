"""CLI success requires a direct comparison of two valid generated kernels."""
import copy
import json

import pytest

from optimizer_agent import campaign
from optimizer_agent.campaign_config import CampaignConfig


def pilot_result(direct):
    return {"results": {mode:{"status":"validated_no_gain","candidate_id":"baseline"}
                        for mode in ("agent","vanilla")},
            "generated_agent_vs_vanilla":direct}


def measured_direct(status="not_improved"):
    tools = ["memcheck","racecheck","initcheck","synccheck"]
    artifacts = {"agent":"a"*64,"vanilla":"b"*64}
    qualification = {"ok":True,"candidates":{mode:{"ok":True,"status":"sanitizer_qualified",
        "artifact_hash":artifact_hash,"required_sanitizers":list(tools),
        "validation":{"ok":True,"cases":[{"case_id":"independent-17"}]},
        "sanitizers":[{"ok":True,"tool":tool,"case_id":"independent-17","seed":7,
                       "launch":{"grid":[2,2,1],"block":[16,16,1],"shared_memory_bytes":0}}
                      for tool in tools]} for mode,artifact_hash in artifacts.items()}}
    return {"ok":True,"status":status,"candidate_artifacts":artifacts,
            "required_sanitizers":tools,"sanitizer_qualification":qualification,
            "pairs":[{"case_id":"independent-17","series":0}],
            "interval":{"gain":.8,"lower":.7,"upper":.9}}


def invoke_compare(tmp_path,monkeypatch,result):
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(CampaignConfig(run_id="compare-fixture").to_dict()))
    monkeypatch.setattr(campaign,"compare",lambda *args,**kwargs:result)
    return campaign.main(["compare","--config",str(config_path)])


@pytest.mark.parametrize("mode",["agent","vanilla"])
def test_valid_baseline_fallback_does_not_mask_missing_generated_kernel(tmp_path,monkeypatch,mode):
    result = pilot_result({"ok":False,"status":"missing_valid_generated_kernel","mode":mode})
    assert not campaign.comparison_completed(result)
    assert invoke_compare(tmp_path,monkeypatch,result) == 1


@pytest.mark.parametrize("status",["promoted","not_improved","inconclusive"])
def test_completed_direct_comparison_succeeds_without_requiring_agent_win(tmp_path,monkeypatch,status):
    result = pilot_result(measured_direct(status))
    assert campaign.comparison_completed(result)
    assert invoke_compare(tmp_path,monkeypatch,result) == 0


@pytest.mark.parametrize("missing",["pairs","interval","candidate_artifacts"])
def test_missing_paired_receipt_is_incomplete(tmp_path,monkeypatch,missing):
    direct = measured_direct()
    direct.pop(missing)
    result = pilot_result(direct)
    assert invoke_compare(tmp_path,monkeypatch,result) == 1


@pytest.mark.parametrize("status",["numerical_error","infrastructure_failure","tool_unavailable"])
def test_failed_generated_kernel_comparison_is_nonzero(tmp_path,monkeypatch,status):
    result = pilot_result(dict(measured_direct(),ok=False,status=status))
    assert invoke_compare(tmp_path,monkeypatch,result) == 1


def test_failed_finalization_retains_nonzero_exit(tmp_path,monkeypatch):
    result = pilot_result(measured_direct())
    result["results"]["agent"]["status"] = "finalization_incomplete"
    assert invoke_compare(tmp_path,monkeypatch,result) == 1


def test_exact_artifact_identity_is_not_presented_as_a_measured_direct_comparison(tmp_path,monkeypatch):
    direct = measured_direct()
    direct.update(status="exact_artifact_identity",pairs=[],paired_measurement_performed=False)
    direct.pop("interval")
    assert invoke_compare(tmp_path,monkeypatch,pilot_result(direct)) == 1


@pytest.mark.parametrize("missing",["required_sanitizers","sanitizer_qualification"])
def test_missing_sanitizer_qualification_is_nonzero(tmp_path,monkeypatch,missing):
    direct = measured_direct()
    direct.pop(missing)
    assert invoke_compare(tmp_path,monkeypatch,pilot_result(direct)) == 1


@pytest.mark.parametrize("mode",["agent","vanilla"])
@pytest.mark.parametrize("defect",["missing_candidate","failed_wrapper","failed_candidate","wrong_artifact",
                                   "missing_reports","missing_tool","failed_report","wrong_case","wrong_required_tools",
                                   "failed_validation","missing_validation_cases","missing_measured_case"])
def test_each_generated_candidate_requires_all_configured_tools_on_each_case(tmp_path,monkeypatch,mode,defect):
    direct = measured_direct()
    qualification = direct["sanitizer_qualification"]
    candidate = qualification["candidates"][mode]
    if defect == "missing_candidate":
        qualification["candidates"].pop(mode)
    elif defect == "failed_wrapper":
        qualification["ok"] = False
    elif defect == "failed_candidate":
        candidate["ok"] = False
    elif defect == "wrong_artifact":
        candidate["artifact_hash"] = "c"*64
    elif defect == "missing_reports":
        candidate["sanitizers"] = []
    elif defect == "missing_tool":
        candidate["sanitizers"].pop()
    elif defect == "failed_report":
        candidate["sanitizers"][0]["ok"] = False
    elif defect == "wrong_case":
        candidate["sanitizers"][0]["case_id"] = "unmeasured-shape"
    elif defect == "wrong_required_tools":
        candidate["required_sanitizers"] = ["memcheck"]
    elif defect == "failed_validation":
        candidate["validation"]["ok"] = False
    elif defect == "missing_validation_cases":
        candidate["validation"]["cases"] = []
    elif defect == "missing_measured_case":
        direct["pairs"].append({"case_id":"independent-63","series":0})
    assert invoke_compare(tmp_path,monkeypatch,pilot_result(direct)) == 1


def test_configured_tool_subset_is_enforced_without_inventing_extra_requirements(tmp_path,monkeypatch):
    direct = measured_direct()
    direct["required_sanitizers"] = ["memcheck"]
    for candidate in direct["sanitizer_qualification"]["candidates"].values():
        candidate["required_sanitizers"] = ["memcheck"]
        candidate["sanitizers"] = [candidate["sanitizers"][0]]
    assert invoke_compare(tmp_path,monkeypatch,pilot_result(direct)) == 0


def test_all_tools_must_cover_each_validated_shape(tmp_path,monkeypatch):
    direct = measured_direct()
    direct["pairs"].append({"case_id":"independent-63","series":0})
    for candidate in direct["sanitizer_qualification"]["candidates"].values():
        candidate["validation"]["cases"].append({"case_id":"independent-63"})
        extra = copy.deepcopy(candidate["sanitizers"])
        for report in extra:
            report["case_id"] = "independent-63"
        candidate["sanitizers"].extend(extra)
    assert invoke_compare(tmp_path,monkeypatch,pilot_result(direct)) == 0
    direct["sanitizer_qualification"]["candidates"]["vanilla"]["sanitizers"].pop()
    assert invoke_compare(tmp_path,monkeypatch,pilot_result(direct)) == 1
