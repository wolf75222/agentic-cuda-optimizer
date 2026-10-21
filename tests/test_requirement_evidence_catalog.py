"""Durable audit catalogs must actually reach the rendered requirement report."""
import pytest

from optimizer_agent.suite_report import build_requirement_status,requirement_registry


def test_catalog_from_full_audit_keeps_implementation_cpu_gpu_claims_separate():
    catalog = {"schema_version":1,"evidence_catalog":{"receipt":{"kind":"cpu_test"}},
        "requirements":{"SEC-5.2":{
            "implemented":{"ok":True,"evidence":["source:diagnostics.py"]},
            "local_tested":{"ok":True,"evidence":["receipt"]},
            "romeo_executed":{"ok":False,"evidence":["historical-partial"],"status":"partial"},
            "remaining":["Fresh diagnostic loop on corrected source"]}}}
    row = next(r for r in build_requirement_status(catalog) if r["id"] == "SEC-5.2")
    assert row["implemented"]["ok"] and row["local_tested"]["ok"]
    assert not row["romeo_executed"]["ok"]
    assert row["remaining"] == ["Fresh diagnostic loop on corrected source"]


def test_all_mandatory_decisions_sections_and_acceptance_criteria_are_reported():
    ids = [r["id"] for r in requirement_registry()]
    assert len(ids) == len(set(ids)) == 78
    assert {"ARCH-01","TREE-01","RES-01","RES-02","PMPP-01","EVAL-01","TUNE-01",
            "EXEC-01","STATE-01","QUALITY-01"} <= set(ids)
    assert {"SEC-"+str(i) for i in range(1,20)} <= set(ids)
    assert {"AC-%02d"%i for i in range(1,21)} <= set(ids)


def test_a_catalog_with_ambiguous_requirement_shape_is_rejected():
    with pytest.raises(ValueError,match="mapping"):
        build_requirement_status({"requirements":[{"id":"AC-20"}]})
