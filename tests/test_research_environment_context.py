"""Prompt metadata and strict research guards using explicit CPU-only fixtures."""
from copy import deepcopy
import hashlib
import json

import pytest

from optimizer_agent.codex_backend import CodexResult
from optimizer_agent.research_service import (
    ResearchBlocked, ResearchConfig, ResearchService, environment_field_semantics,
)
from tests import test_codex_research as research_fixtures


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class MetadataBackend:
    """Returns simulated search/synthesis data, never starts a CLI or network."""
    def __init__(self, *, missing=False, failed_check=False):
        self.missing, self.failed_check = missing, failed_check
        self.contexts, self.payloads = [], []

    def generate_json(self, prompt, schema, **options):
        raw_context = prompt.split("Context: ", 1)[1].split("\nActually read source content:", 1)[0]
        self.contexts.append(json.loads(raw_context))
        if options["purpose"] == "research":
            payload = {"sources": [], "rejected_examples": [], "queries": ["declared fixture query"]}
            events = [{"type": "item.completed", "item": {"id": "simulated-search", "type": "web_search",
                "action": {"type": "search", "queries": ["explicit simulated CUDA research"]}}}]
        else:
            contents = json.loads(prompt.split("Actually read source content:\n", 1)[1])
            references = [entry["source"]["source_id"] for entry in contents if entry["source"]["kind"] == "code"]
            payload = dict(covered_techniques=["shared_tiling"], reference_kernels=references,
                launch_requirements=["Validate all tails after compilation"], dependencies=[],
                applicability_checks=[dict(condition="Required driver/API compatibility is documented",
                    evidence="Missing documentary mapping" if self.failed_check else "Explicit fixture documentary evidence",
                    satisfied=not self.failed_check)],
                incompatible_conditions=[],
                missing_evidence=["Required driver compatibility documentation was not read"] if self.missing else [],
                pmpp_topics=[], hypotheses=[dict(documented_fact="Fixture block barrier semantics",
                    observed_code_property="Fixture cooperative tile", expected_effect_to_measure="Measure target latency",
                    source_ids=references)], expected_effects=["Measure latency"],
                validation_metrics=["Independent CPU oracle"], negative_search=[], original_hypothesis="")
            events = []
        self.payloads.append(deepcopy(payload))
        return CodexResult(payload, {"input_tokens": 1, "output_tokens": 1}, events,
                           "simulated-model", "max", 0.01, f"simulated-{len(self.contexts)}")


def fetch_fixture(url, timeout, limit):
    if "raw.githubusercontent.com" in url:
        return (b"// Explicit simulated source, not a downloaded NVIDIA implementation.\n"
                b"struct BlockReduce {};\n__global__ void gemm_device() {}\n"), url
    return (b'<section id="shared-memory"><p>Explicit simulated shared memory evidence</p></section>'
            b'<section id="synchronization-functions"><p>Explicit simulated barrier evidence</p></section>'), url


def prepare_fixture(tmp_path, backend):
    workload = {"contract_hash": "immutable-contract", "environment": {"uuid": "simulated-target",
        "driver_version": 13010, "nvrtc_version": "12.6", "cuda_toolkit_version": 12060,
        "harness_build_id": "simulated-build"}}
    evidence = {"environment_fingerprint": "immutable-environment"}
    parent, action = {"candidate_id": "baseline"}, {"strategy_family": "tiling"}
    arguments = [workload, parent, action, evidence]
    frozen = deepcopy(arguments)
    frozen_hashes = [digest(value) for value in arguments]
    persisted_environment = tmp_path / "environment.json"
    persisted_environment.write_text(json.dumps(workload["environment"]))
    persisted_hash = hashlib.sha256(persisted_environment.read_bytes()).hexdigest()
    service = ResearchService(backend, tmp_path / "research", fetcher=fetch_fixture,
        config=ResearchConfig(experimental=True, include_pmpp=False))
    return service, arguments, frozen, frozen_hashes, persisted_environment, persisted_hash


def assert_prompt_only(arguments, frozen, frozen_hashes, persisted_environment, persisted_hash, backend):
    assert arguments == frozen
    assert [digest(value) for value in arguments] == frozen_hashes
    assert hashlib.sha256(persisted_environment.read_bytes()).hexdigest() == persisted_hash
    assert len(backend.contexts) == 2
    for context in backend.contexts:
        assert context["workload"] == frozen[0]
        assert context["evidence"] == frozen[3]
        note = context["environment_field_semantics"]["driver_version"]
        assert note["reported_value"] == 13010 and type(note["reported_value"]) is int
        assert note["producer_api"] == "cuDriverGetVersion(int*)"
        assert "repository" in note["definition_scope"] and "other backends" in note["definition_scope"]
        assert "not an NVIDIA driver package/release" in note["repository_representation"]
        assert "None supplied" in note["compatibility_evidence"]
        assert "consult official documentation" in note["research_requirement"]
        assert "satisfied" not in note and "source_id" not in note


def test_annotation_reaches_search_and_synthesis_without_mutating_values_identities_or_receipts(tmp_path):
    backend = MetadataBackend()
    service, arguments, frozen, hashes, environment_path, environment_hash = prepare_fixture(tmp_path, backend)
    dossier = service.prepare(*arguments)
    assert_prompt_only(arguments, frozen, hashes, environment_path, environment_hash, backend)
    assert dossier.workload_hash == "immutable-contract"
    assert dossier.environment_fingerprint == "immutable-environment"
    assert dossier.experimental_mode == "ablation_no_pmpp" and not dossier.pmpp_topics
    assert "pmpp" not in {source.kind for source in dossier.sources}
    assert "environment_field_semantics" not in dossier.to_dict()
    assert dossier.research_ready("immutable-contract", "tiling", "immutable-environment")


@pytest.mark.parametrize("missing,failed", [(True, True), (True, False), (False, True)])
def test_annotation_never_certifies_missing_compatibility_or_overrides_false_conditions(tmp_path, missing, failed):
    backend = MetadataBackend(missing=missing, failed_check=failed)
    service, arguments, frozen, hashes, environment_path, environment_hash = prepare_fixture(tmp_path, backend)
    with pytest.raises(ResearchBlocked) as caught:
        service.prepare(*arguments)
    assert_prompt_only(arguments, frozen, hashes, environment_path, environment_hash, backend)
    if missing:
        assert "missing_evidence" in str(caught.value)
        assert "Required driver compatibility documentation" in str(caught.value)
    if failed:
        assert "Required driver/API compatibility is documented" in str(caught.value)
    [blocked_path] = (tmp_path / "research").glob("*-blocked.json")
    blocked = json.loads(blocked_path.read_text())
    assert blocked_path.name == f"{blocked['research_id']}-blocked.json"
    dossier_path = blocked_path.with_name(f"{blocked['research_id']}.json")
    saved = json.loads(dossier_path.read_text())
    assert saved["research_id"] == blocked["research_id"]
    assert saved["missing_evidence"] == backend.payloads[1]["missing_evidence"]
    assert saved["applicability_checks"] == backend.payloads[1]["applicability_checks"]
    assert saved["environment_fingerprint"] == "immutable-environment"
    assert "environment_field_semantics" not in saved
    assert blocked["retrieval_status"] == "research_blocked" and blocked["error"] in str(caught.value)


def test_definition_is_scoped_to_repository_producer_and_never_coerces_another_backends_release():
    workload = {"environment": {"driver_version": "590.48.01", "backend": "unspecified-external-producer"}}
    frozen = deepcopy(workload)
    note = environment_field_semantics(workload)["driver_version"]
    assert note["reported_value"] == "590.48.01"
    assert "producer only" in note["definition_scope"] and "verify producer provenance" in note["definition_scope"]
    assert workload == frozen
    assert environment_field_semantics({"environment": {"nvrtc_version": "12.6"}}) == {}


def test_missing_evidence_and_failed_conditions_are_actionable_but_bounded(tmp_path):
    dossier = research_fixtures.ResearchTests().make_dossier(tmp_path)
    dossier.missing_evidence = [f"missing-{index}:" + "x" * 10000 for index in range(10)]
    dossier.applicability_checks = [dict(condition=f"condition-{index}:" + "y" * 10000,
                                       evidence="not established", satisfied=False) for index in range(10)]
    frozen = deepcopy(dossier.to_dict())
    with pytest.raises(ResearchBlocked) as caught:
        dossier.research_ready("hash", "tiling", "gpu-fingerprint")
    error = str(caught.value)
    assert "missing-0:" in error and "condition-0:" in error and "(+7 more)" in error
    assert "missing-3:" not in error and "condition-3:" not in error
    assert len(error) < 1300
    assert dossier.to_dict() == frozen
