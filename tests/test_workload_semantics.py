"""A research guard must receive the real operation, including boundaries."""
import json

import pytest

from optimizer_agent import campaign
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.vanilla_gpu_service import workload_description
from tests.test_campaign_runtime import EvaluatorFixture, ModelFixture, RunnerFixture


@pytest.mark.parametrize("operation,required", [
    ("matmul", ("a[row*n+k]*b[k*n+col]",)),
    ("reduction", ("c[0]", "sum(i=0..n-1",)),
    ("scan", ("sum(j=0..i", "inclusive")),
    ("stencil", ("a[i-1]", "a[i+1]", "j<0 or j>=n")),
    ("convolution", ("a[i-1]*b[0]", "a[i+1]*b[2]", "not reversed", "outside [0,n)")),
    ("normalization", ("1e-5", "RMS", "not mean subtraction")),
])
def test_normal_agent_and_native_tools_share_immutable_exact_operation(tmp_path, monkeypatch, operation, required):
    monkeypatch.setattr(campaign, "Evaluator", EvaluatorFixture)
    config = CampaignConfig(run_id="semantic-"+operation, workload=operation, search_sizes=(64,), final_sizes=(96,))
    model = ModelFixture()
    runtime = campaign.CampaignRuntime(config, RunnerFixture(), tmp_path, model_backend=model)
    target = runtime.workload_description
    contract = json.loads(runtime.evaluator.contract.metadata_json)
    for text in required:
        assert text in target["mathematics"]
        assert text in contract["mathematics"]
    assert target["mathematics"] == workload_description(operation)["mathematics"]
    assert target["memory_policy"] == workload_description(operation)["memory_policy"]
    assert contract["memory_policy"] == target["memory_policy"]
    assert not model.calls
    if operation == "reduction":
        assert "one element" in target["memory_policy"]["sizes"]
    if operation == "convolution":
        assert "B exactly 3" in target["memory_policy"]["sizes"]
