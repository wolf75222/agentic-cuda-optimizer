"""CPU documentary fixtures distinguish negative findings from web failure.

No web, model or GPU execution is claimed by these structural guard tests.
"""
from dataclasses import asdict

import pytest

from optimizer_agent.research_service import ResearchBlocked, ResearchDossier
from tests import test_codex_research as research_fixtures


def negative_dossier(tmp_path):
    dossier = research_fixtures.ResearchTests().make_dossier(tmp_path)
    dossier.reference_kernels = []
    dossier.negative_search = [
        source.source_id + ": consulted CPU fixture implementation requires a multi-launch host API, outside the single-kernel contract"
        for source in dossier.sources if source.kind == "code"]
    dossier.original_hypothesis = "Original unmeasured hypothesis: adapt the documented cooperation principle to a single self-contained kernel"
    return dossier


def test_documented_negative_search_allows_original_hypothesis_without_inventing_code_reference(tmp_path):
    dossier = negative_dossier(tmp_path)
    restored = ResearchDossier.from_dict(asdict(dossier))
    assert restored.research_ready(restored.workload_hash, restored.strategy_family, restored.environment_fingerprint)
    payload = research_fixtures.proposal_payload()
    payload["hypothesis"] = restored.original_hypothesis
    payload["references"] = [source.source_id for source in restored.sources if source.kind != "code"]
    restored.validate_proposal(payload)
    assert not restored.reference_kernels
    assert all("outside the single-kernel contract" in entry for entry in restored.negative_search)
    payload["references"].append("invented-unread-implementation")
    with pytest.raises(ResearchBlocked, match="unread sources"):
        restored.validate_proposal(payload)


@pytest.mark.parametrize("negative,hypothesis", [([],"Original hypothesis"), ([" "],"Original hypothesis"),
                                                (["Rejected consulted example"]," "),
                                                ("Rejected consulted example","Original hypothesis"),
                                                ([3],"Original hypothesis")])
def test_empty_negative_findings_cannot_replace_an_applicable_reference(tmp_path,negative,hypothesis):
    dossier = negative_dossier(tmp_path)
    dossier.negative_search,dossier.original_hypothesis = negative,hypothesis
    with pytest.raises(ResearchBlocked, match="documented negative finding"):
        dossier.research_ready(dossier.workload_hash, dossier.strategy_family, dossier.environment_fingerprint)


def test_negative_finding_cannot_convert_failed_web_retrieval_to_success(tmp_path):
    dossier = negative_dossier(tmp_path)
    dossier.retrieval_status = "research_blocked"
    with pytest.raises(ResearchBlocked):
        dossier.research_ready(dossier.workload_hash, dossier.strategy_family, dossier.environment_fingerprint)

