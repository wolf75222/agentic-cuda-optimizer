from optimizer_agent.campaign_config import CampaignConfig
from optimizer_agent.quality_suite import plan_suite, summarize_suite


def test_six_families_repeat_seeds_and_isolated_ablation_flags():
    plan = plan_suite(CampaignConfig(run_id="q"), seeds=(1,2))
    assert len(plan["entries"]) == 60
    modes = {e["mode"]: e["config"] for e in plan["entries"] if e["workload"] == "matmul" and e["seed"] == 1}
    assert modes["full"]["experimental_mode"] is None
    assert modes["greedy_parent"]["experimental_mode"] == "ablation_no_tree"
    assert modes["no_autotuning"]["research_required"] and not modes["no_autotuning"]["autotuning_enabled"]
    assert modes["no_research"]["autotuning_enabled"] and not modes["no_research"]["research_required"]
    assert modes["no_pmpp"]["research_required"] and modes["no_pmpp"]["autotuning_enabled"]
    assert all("model_tokens" not in e["config"] for e in plan["entries"])
    assert plan["global_resource_caps"] is None


def test_report_exposes_missing_uncertified_and_regressions():
    plan = plan_suite(CampaignConfig(run_id="q"), seeds=(1,2))
    entry = plan["entries"][0]
    result = summarize_suite(plan, {entry["config"]["run_id"]: {
        "certification_status": "validated_gain_uncertified",
        "final": {"confirmation": {"interval": {"gain": .8, "lower": .7, "upper": .9}}}}})
    assert not result["complete"]
    assert result["rows"][0]["gain"] == .8
    assert result["aggregates"][entry["mode"]]["certified"] == 0
    assert result["aggregates"][entry["mode"]]["missing"] == 11
