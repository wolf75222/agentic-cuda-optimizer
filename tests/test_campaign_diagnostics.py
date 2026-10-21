"""Real diagnostic/research/generator services with explicit transport/CUDA mocks."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from optimizer_agent import campaign
from optimizer_agent.codex_backend import CodexResult
from optimizer_agent.diagnostics import build_diagnostic, diagnostic_digest, verify_diagnostic
from optimizer_agent.research_service import diagnostic_research_context, ResearchBlocked
from tests.test_campaign_runtime import runtime_setup, selected, archive_candidate
from tests.test_codex_research import proposal_payload


def complete_proposal(runtime, state):
    value = proposal_payload()
    trial = runtime.trial(state)
    value.update(parent_id=trial["parent"]["candidate_id"], source_parent_id=trial["parent"]["candidate_id"],
                 contract_hash=runtime.contract_hash, strategy_family=trial["action"],
                 source=runtime.baseline.source.read_text(), kernel=runtime.config.workload)
    return value


def diagnostic_fixture(**overrides):
    arguments = dict(run_id="run", trial_id="trial", candidate_id="candidate", strategy_family="tiling",
        contract_hash="contract", environment_fingerprint="gpu", stage="result", error=None, needs_research=False,
        proposal={"hypothesis":"tile reuse", "cuda_constraint":"block barrier", "expected_measurement":"lower latency",
                  "references":["fixture-code"], "techniques":["tiling"]},
        measurement=None, confirmation=None, profile=None, previous_results=[], evidence_references=[], candidate=None,
        pmpp_guidance=None)
    arguments.update(overrides)
    return build_diagnostic(**arguments)


def test_stagnation_requires_three_valid_same_family_trials_and_no_promotion():
    results = [dict(trial_id=str(i), strategy_family="tiling", valid=True, promoted=False,
                    evaluation_id=f"evaluation-{i}") for i in range(3)]
    diagnostic = diagnostic_fixture(previous_results=results)
    assert diagnostic["triggers"] == ["stagnation"]
    observation = next(x for x in diagnostic["observations"] if x["kind"] == "stagnation")
    assert observation["evaluation_ids"] == ["evaluation-0", "evaluation-1", "evaluation-2"]
    assert not diagnostic_fixture(previous_results=results[:2])["requires_targeted_research"]
    results[-1]["promoted"] = True
    assert "stagnation" not in diagnostic_fixture(previous_results=results)["triggers"]
    results[-1].update(promoted=False, strategy_family="coalescing")
    assert "stagnation" not in diagnostic_fixture(previous_results=results)["triggers"]


def test_diagnostic_identity_and_unavailable_profile_never_invent_measurements(tmp_path):
    diagnostic = diagnostic_fixture(needs_research=True,
        profile={"ok":False,"error":"ERR_NVGPUCTRPERM", "execution_environment":"romeo"})
    assert diagnostic["requires_targeted_research"]
    assert diagnostic_research_context({"diagnostic":diagnostic})["diagnostic_id"] == diagnostic["diagnostic_id"]
    profile = next(x["value"] for x in diagnostic["observations"] if x["kind"] == "profile")
    assert "metrics" not in profile and profile["ok"] is False
    altered = dict(diagnostic, triggers=[])
    with pytest.raises(ResearchBlocked, match="immutable identity"):
        diagnostic_research_context({"diagnostic":altered})
    with pytest.raises(ValueError, match="contract/environment"):
        verify_diagnostic(diagnostic, tmp_path, contract_hash="different", environment_fingerprint="gpu")


def test_measured_diagnostic_drives_real_targeted_service_and_next_proposal(runtime_setup):
    runtime, config, _, _ = runtime_setup
    first = selected(runtime)
    candidate = archive_candidate(runtime, first)
    first.update(candidate_id=candidate.candidate_id, valid=True)
    first_proposal = complete_proposal(runtime, first)
    campaign.write_json(runtime.trial_dir(first)/"proposal-0-0.json", first_proposal)
    measurement = runtime.evaluator.benchmark(candidate)
    campaign.write_json(runtime.trial_dir(first)/"evaluation-fast.json", measurement)
    runtime.evaluator.confirm = lambda candidate, incumbent: dict(evaluation_id="paired-not-improved",
        status="not_improved", ok=True, candidate_id=candidate.candidate_id, incumbent_id=incumbent.candidate_id,
        interval=[.9, 1.1])
    runtime.integrate(first)
    frozen_tree_size = len(runtime.policy.snapshot()["tree"].tree)
    campaign.write_json(runtime.trial_dir(first)/"profile.json", dict(ok=True, execution_environment="local",
        metrics={"dram__throughput.avg.pct_of_peak_sustained_elapsed":{"value":12.5,"unit":"percent"}},
        hardware_fingerprint={"uuid":"fixture-local-GPU"}, diagnostic_only=True, ranking_eligible=False,
        hardware_transfer_note="Local counters are not target measurements"))
    diagnostic_update = runtime.diagnose(first)
    diagnostic = runtime.diagnostic_context(diagnostic_update, runtime.trial(first))
    assert diagnostic["triggers"] == ["unexplained_result"]
    assert any(reference["identity"] == measurement["evaluation_id"] for reference in diagnostic["evidence_references"])
    assert runtime.diagnose(first) == diagnostic_update  # Replay does not profile or change identity.
    assert len(runtime.policy.snapshot()["tree"].tree) == frozen_tree_size
    second = {"run_id":config.run_id, "trials_done":1, **diagnostic_update,
              **runtime.select({"trials_done":1, **diagnostic_update})}
    assert second["trial_id"] != first["trial_id"]

    class Transport:
        def __init__(self):
            self.calls = []
        def generate_json(self, prompt, schema, **options):
            self.calls.append((prompt, schema, options))
            assert diagnostic["diagnostic_id"] in prompt
            assert "paired-not-improved" in prompt and "12.5" in prompt
            if options["purpose"] == "research":
                payload = {"sources":[],"rejected_examples":[]}
                events = [{"type":"item.completed","item":{"id":"fixture-search","type":"web_search",
                    "query":"fixture diagnostic tiled GEMM", "action":{"type":"search","queries":["fixture diagnostic tiled GEMM"]}}}]
            elif "covered_techniques" in schema["properties"]:
                sources = schema["properties"]["hypotheses"]["items"]["properties"]["source_ids"]["items"]["enum"]
                code = schema["properties"]["reference_kernels"]["items"]["enum"]
                payload = dict(covered_techniques=["shared_tiling"],reference_kernels=code,
                    launch_requirements=["guard tails and synchronize"],dependencies=[],
                    applicability_checks=[dict(condition="fixture ABI",evidence="mock contract and sources",satisfied=True)],
                    incompatible_conditions=[],missing_evidence=[],pmpp_topics=["tiling"],
                    hypotheses=[dict(documented_fact="mock documentary barrier semantics",
                        observed_code_property="mock shared tile",expected_effect_to_measure="explain paired-not-improved",
                        source_ids=sources)],expected_effects=["measure latency"],validation_metrics=["paired target timing"],
                    negative_search=[],original_hypothesis="")
                events = []
            else:
                payload = complete_proposal(runtime, second)
                payload["references"] = schema["properties"]["references"]["items"]["enum"]
                events = []
            return CodexResult(payload, {"input_tokens":20,"output_tokens":10}, events,
                               "gpt-6-luna","max",.01,uuid.uuid4().hex)

    transport = Transport()
    runtime.backend.backend = transport
    runtime.research_service.fetcher = lambda url, timeout, limit: (
        (b"// mock source\n"+b"__global__ void mock() {}\n"*8) if "raw.githubusercontent.com" in url else
        b'<section id="shared-memory"><p>mock shared memory documentation</p></section><section id="synchronization-functions"><p>mock __syncthreads semantics</p></section>', url)
    update = runtime.research(second)
    assert update["research_ready"], update
    second.update(update)
    dossier = campaign.ResearchDossier.from_dict(campaign.read_json(runtime.trial_dir(second)/"dossier.json"))
    assert dossier.diagnostic_ids == [diagnostic["diagnostic_id"]]
    generated = runtime.generate(second)
    assert generated["candidate_id"] and generated["error"] is None
    assert len(transport.calls) == 3  # Search, actual-source synthesis, then proposal.
    assert "Local counters are not target measurements" in transport.calls[-1][0]
    assert len(runtime.policy.snapshot()["tree"].tree) == frozen_tree_size
    assert runtime.store.incumbent(config.run_id)["candidate_id"] is None


def test_diagnostic_source_hash_tampering_blocks_next_generation(runtime_setup):
    runtime, _, _, _ = runtime_setup
    state = selected(runtime)
    candidate = archive_candidate(runtime, state)
    state.update(candidate_id=candidate.candidate_id, valid=False, needs_research=True, error="uncovered technique")
    update = runtime.diagnose(state)
    archived = runtime.candidate(candidate.candidate_id).source
    archived.write_text(archived.read_text()+"\n// changed evidence\n")
    with pytest.raises(ValueError, match="evidence hash changed"):
        runtime.diagnostic_context(update, runtime.trial(state))
