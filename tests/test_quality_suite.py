import pytest
from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import MODES, plan_suite, summarize_suite


def test_default_126_true_upstream_native_and_isolated_ablations():
    plan = plan_suite(CampaignConfig(run_id="q"))
    assert len(plan["entries"]) == 126
    modes = {e["mode"]: e["config"] for e in plan["entries"] if e["workload"] == "matmul" and e["seed"] == 1729}
    assert set(modes) == set(MODES)
    assert modes["full"]["experimental_mode"] is None
    assert "greedy_parent" not in modes
    assert not modes["no_autotuning"]["autotuning_enabled"]
    assert modes["no_autotuning"]["research_required"]
    assert not modes["no_research"]["research_required"] and modes["no_research"]["autotuning_enabled"]
    assert modes["no_pmpp"]["research_required"] and modes["no_pmpp"]["autotuning_enabled"]
    assert modes["pmpp_permanent_only"]["experimental_mode"] == "ablation_pmpp_permanent_only"
    assert modes["pmpp_permanent_only"]["research_required"] and modes["pmpp_permanent_only"]["web_search_required"]
    assert all(e["config"]["search_sizes"] == e["config"]["final_sizes"] for e in plan["entries"])
    assert all(set(e["config"]["required_sanitizers"]) == {"memcheck","racecheck","initcheck","synccheck"} for e in plan["entries"])
    assert plan["global_resource_caps"] is None
    expanded = plan_suite(CampaignConfig(), include_greedy=True)
    assert len(expanded["entries"]) == 144 and len(expanded["modes"]) == 8
    assert len({entry["group_id"] for entry in expanded["entries"]}) == 18
    assert plan["ablation_contrasts"]["targeted_pmpp_consultation"]["reference"] == "pmpp_permanent_only"
    assert plan["ablation_contrasts"]["permanent_pmpp_guidance"]["reference"] == "no_pmpp"
    with pytest.raises(ValueError, match="retains mandatory"):
        CampaignConfig(experimental_mode="ablation_pmpp_permanent_only", research_required=False)


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
