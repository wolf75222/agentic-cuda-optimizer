import pytest
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import MODES, plan_suite, summarize_suite


def test_default_108_true_upstream_native_and_isolated_ablations():
    plan = plan_suite(CampaignConfig(run_id="q"))
    assert len(plan["entries"]) == 108
    modes = {e["mode"]: e["config"] for e in plan["entries"] if e["workload"] == "matmul" and e["seed"] == 1729}
    assert set(modes) == set(MODES)
    assert modes["full"]["experimental_mode"] is None
    assert "greedy_parent" not in modes
    assert not modes["no_autotuning"]["autotuning_enabled"]
    assert modes["no_autotuning"]["research_required"]
    assert not modes["no_research"]["research_required"] and modes["no_research"]["autotuning_enabled"]
    assert modes["no_pmpp"]["research_required"] and modes["no_pmpp"]["autotuning_enabled"]
    assert all(e["config"]["search_sizes"] == e["config"]["final_sizes"] for e in plan["entries"])
    assert all(set(e["config"]["required_sanitizers"]) == {"memcheck","racecheck","initcheck","synccheck"} for e in plan["entries"])
    assert plan["global_resource_caps"] is None
    assert len(plan_suite(CampaignConfig(), include_greedy=True)["entries"]) == 126


def test_long_ids_unique_and_invalid_seed_replication_rejected():
    plan = plan_suite(CampaignConfig(run_id="q"*79), seeds=(1,2))
    identifiers = [e["config"]["run_id"] for e in plan["entries"]]
    assert len(set(identifiers)) == len(identifiers)
    assert all(len(value) <= 80 for value in identifiers)
    with pytest.raises(ValueError):
        plan_suite(CampaignConfig(), seeds=(1,1))


def test_report_retains_measured_regressions_and_unknown_denominator():
    plan = plan_suite(CampaignConfig(run_id="q"), seeds=(1,2))
    entry = plan["entries"][0]
    manifests = {entry["config"]["run_id"]: {"status":"generation_complete", "final":{"confirmation":{"interval":{"gain":1000}}}}}
    # Per-generation scores are not accepted as §16 final evidence.
    assert summarize_suite(plan, manifests)["rows"][0]["gain"] is None
    groups = {entry["group_id"]: {"candidates": {entry["mode"]: {
        "status":"validated_gain_uncertified", "qualified":True,
        "confirmation":{"interval":{"gain":.8,"lower":.7,"upper":.9,"case_gains":{"final-64":{"gain":.8}}}}}}}}
    result = summarize_suite(plan, manifests, groups)
    assert not result["complete"]
    assert result["rows"][0]["gain"] == .8
    row = result["aggregates"][entry["mode"]]
    assert row["certified"] == 0 and row["final_measured"] == 1
    assert row["regressions"] == 1 and row["unmeasured_or_failed"] == 11
    assert row["geometric_mean_all_measured_gain"] == .8
