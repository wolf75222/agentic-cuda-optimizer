"""Local visualization of real search persistence; no model/GPU/SSH calls."""
import hashlib
import json
from pathlib import Path
import sqlite3
from xml.etree import ElementTree

import pytest

from optimizer_agent.experiment_store import ExperimentStore
from optimizer_agent.search_policy import SearchPolicy
from optimizer_agent.greedy_search_policy import GreedySearchPolicy
from scripts import render_search_tree as module


BASELINE={"candidate_id":"baseline","source":"// FULL_KERNEL_SOURCE_NOT_FOR_SVG\nvoid baseline(){}",
    "parameters":{"tile":32},"artifact_hash":"baseline-artifact"}


def ledger(tmp_path,*,greedy=False):
    source=tmp_path/"generation"/"experiments.sqlite"
    store=ExperimentStore(source)
    policy=(GreedySearchPolicy if greedy else SearchPolicy)(store,"render-test",BASELINE,{"case":10.},
        actions=["tile","vector"],contract_fingerprint="contract",environment_fingerprint="cpu-test-only",seed=7)
    trial=policy.ask()
    state={"candidate_id":"fast<&>","source":"// FULL_KERNEL_SOURCE_NOT_FOR_SVG\nvoid faster(){}",
        "parameters":{"tile":64},"artifact_hash":"fast-artifact","tree_parent_id":"baseline"}
    store.archive_candidate("render-test",state)
    policy.tell_once(trial.trial_id,{"state":state,"valid":True,"evaluation_id":"measured-fast",
        "purpose":"tree_reward","fresh":True,"latencies":{"case":8.}})
    store.promote("render-test",state["candidate_id"],0,{"status":"improved","evidence":["cpu-test-only-promotion"]})
    policy.close("parent_action",(state["candidate_id"],"tile"),"Incompatible alignment <&>; see the retained source",
        author="cpu_test",evidence=["cpu-only"],counters={"repairs":2})
    pending=policy.ask()
    cancelled=policy.ask()
    policy.suspend_or_cancel(cancelled.trial_id,"Network unavailable <&>; not a numerical reward",consumption={"trials":1})
    return source,store,policy,pending


def inventory(directory):
    return {str(path.relative_to(directory)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()}


def test_tree_json_svg_preserve_real_scores_parents_pending_incumbent_and_closures_without_source_writes(tmp_path):
    source,store,policy,pending=ledger(tmp_path)
    before=inventory(source.parent)
    destination=tmp_path/"separate-render"
    result=module.render_search_tree(source,destination,trusted_local=True)
    assert inventory(source.parent)==before
    assert result["source_files_unchanged"] and result["gpu_requests"]==result["model_calls"]==result["ssh_calls"]==0
    assert set(path.name for path in destination.iterdir())=={"search-tree.json","search-tree.svg"}
    document=json.loads((destination/"search-tree.json").read_text())
    nodes=document["nodes"]
    baseline=next(node for node in nodes if node.get("candidate_id")=="baseline")
    fast=next(node for node in nodes if node.get("candidate_id")=="fast<&>")
    trial=next(node for node in nodes if node["kind"]=="pending_trial")
    assert baseline["score"]==.5 and fast["score"]==pytest.approx(10/18)
    assert fast["parent_key"]==baseline["key"]
    assert fast["search_incumbent"] and fast["eligible_actions"]==["vector"]
    assert fast["closure_reasons"][0]["reason"].startswith("Incompatible alignment")
    assert fast["evaluation_references"]==["measured-fast"]
    assert trial["trial_id"]==pending.trial_id and trial["score"] is None
    assert document["search_state"]["status"]=="incomplete_pending_trials"
    assert document["certification"]["status"]=="not_inferred_from_search_registry"
    assert any(row["kind"]=="trial_cancelled" for row in document["abandonment_and_incident_events"])
    assert document["capture"]["backup"]["integrity_check"]=="ok"
    svg=(destination/"search-tree.svg").read_text()
    tree=ElementTree.fromstring(svg)
    assert "Score de RECHERCHE ≠ gain final" in svg
    assert "FULL_KERNEL_SOURCE_NOT_FOR_SVG" not in svg
    assert "fast&lt;&amp;&gt;" in svg and "Incompatible alignment &lt;&amp;&gt;" in svg
    assert "<script" not in svg and "<foreignObject" not in svg and "href=" not in svg
    assert len(tree.findall(".//{http://www.w3.org/2000/svg}path"))==len(nodes)-1


def test_greedy_lineage_does_not_invent_single_scores_or_claim_an_ab_mcts_tree(tmp_path):
    source,_,_,_=ledger(tmp_path,greedy=True)
    destination=tmp_path/"greedy-render"
    module.render_search_tree(source,destination,trusted_local=True)
    document=json.loads((destination/"search-tree.json").read_text())
    assert document["algorithm"]=="greedy_parent"
    candidates=[node for node in document["nodes"] if node["kind"]=="greedy_candidate_lineage"]
    assert all(node["score"] is None for node in candidates)
    fast=next(node for node in candidates if node["candidate_id"]=="fast<&>")
    assert fast["score_observations"][0]["reward"]==pytest.approx(10/18)


def test_trust_and_private_unknown_schema_checks_precede_deserialization(tmp_path,monkeypatch):
    from optimizer_agent import portable_export
    monkeypatch.setattr(portable_export.pickle,"loads",lambda *args:pytest.fail("No pickle may be read before schema/trust qualification"))
    with pytest.raises(ValueError,match="trust-local-pickle"):
        module.render_search_tree(tmp_path/"unknown.sqlite",tmp_path/"untrusted-output")
    source=tmp_path/"private"/"suite.sqlite";source.parent.mkdir()
    with sqlite3.connect(source) as db:
        db.execute("CREATE TABLE sealed_groups(id,seed,commitment)")
        db.execute("INSERT INTO sealed_groups VALUES('group','PRIVATE_FINAL_SEED','commitment')")
    with pytest.raises(ValueError,match="suite/private/unknown"):
        module.render_search_tree(source,tmp_path/"private-output",trusted_local=True)
    assert not (tmp_path/"private-output").exists()
    source=tmp_path/"unknown"/"experiments.sqlite";store=ExperimentStore(source)
    with store.transaction() as db:db.execute("CREATE TABLE unexpected_private_table(secret)")
    with pytest.raises(ValueError,match="suite/private/unknown"):
        module.render_search_tree(source,tmp_path/"unknown-output",trusted_local=True)
    assert not (tmp_path/"unknown-output").exists()


def test_nonempty_wal_is_refused_without_checkpointing_or_touching_the_original(tmp_path):
    source,_,_,_=ledger(tmp_path)
    writer=sqlite3.connect(source)
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("INSERT INTO events(run_id,kind,payload,created_at) VALUES('render-test','active','{}',0)")
    writer.commit()
    assert Path(str(source)+"-wal").stat().st_size>0
    before=inventory(source.parent)
    try:
        with pytest.raises(ValueError,match="coherent exported copy"):
            module.render_search_tree(source,tmp_path/"active-output",trusted_local=True)
        assert inventory(source.parent)==before and not (tmp_path/"active-output").exists()
    finally:writer.close()


def test_nonempty_journal_and_overlapping_outputs_are_refused(tmp_path):
    source,_,_,_=ledger(tmp_path)
    with pytest.raises(ValueError,match="cannot overlap"):
        module.render_search_tree(source,source.parent/"render",trusted_local=True)
    with pytest.raises(ValueError,match="cannot overlap"):
        module.render_search_tree(source,source.parent.parent,trusted_local=True)
    journal=Path(str(source)+"-journal");journal.write_bytes(b"hot journal test")
    with pytest.raises(ValueError,match="WAL/journal"):
        module.render_search_tree(source,tmp_path/"journal-output",trusted_local=True)


def test_source_evolution_after_copy_refuses_before_pickle_and_before_output(tmp_path,monkeypatch):
    from optimizer_agent import portable_export
    source,store,_,_=ledger(tmp_path)
    copy=module._backup_cold
    def concurrent_write(old,new):
        result=copy(old,new)
        store.event("render-test","concurrent-writer",{})
        return result
    monkeypatch.setattr(module,"_backup_cold",concurrent_write)
    monkeypatch.setattr(portable_export.pickle,"loads",lambda *args:pytest.fail("Source drift must be detected before loading pickle"))
    with pytest.raises(ValueError,match="evolved"):
        module.render_search_tree(source,tmp_path/"raced-output",trusted_local=True)
    assert not (tmp_path/"raced-output").exists()


def test_completed_manifest_claim_stays_separate_from_search_facts_and_certification(tmp_path):
    source,store,policy,pending=ledger(tmp_path)
    policy.suspend_or_cancel(pending.trial_id,"Explicit test stop",consumption={"trials":1})
    manifest=source.parent/"manifest.json"
    manifest.write_text(json.dumps({"run_id":"render-test","status":"done","generation_complete":True,"certification_status":"certified"}))
    destination=tmp_path/"manifest-render"
    module.render_search_tree(source,destination,trusted_local=True,manifest=manifest)
    document=json.loads((destination/"search-tree.json").read_text())
    assert document["search_state"]["status"]=="snapshot_without_pending_trials"
    assert document["search_state"]["execution_completion"]=="not_inferred_from_absence_of_pending_trials"
    assert document["external_status_manifest"]["reported_certification_status"]=="certified"
    assert document["external_status_manifest"]["verified_by_renderer"] is False
    assert document["certification"]["verified_by_renderer"] is False
    assert document["certification"]["status"]=="not_inferred_from_search_registry"


def test_fresh_output_and_explicit_run_selection_prevent_overwriting_or_mixing_campaigns(tmp_path):
    source,store,_,_=ledger(tmp_path)
    SearchPolicy(store,"other",BASELINE,{"case":10.},actions=["tile"],seed=9)
    with pytest.raises(ValueError,match="multiple campaigns"):
        module.render_search_tree(source,tmp_path/"ambiguous-render",trusted_local=True)
    destination=tmp_path/"selected-render"
    module.render_search_tree(source,destination,trusted_local=True,run_id="other")
    document=json.loads((destination/"search-tree.json").read_text())
    assert document["run_id"]=="other" and all(row["run_id"]=="other" for row in document["ledger"]["tables"]["campaigns"])
    before=inventory(destination)
    with pytest.raises(ValueError,match="fresh separate"):
        module.render_search_tree(source,destination,trusted_local=True,run_id="other")
    assert inventory(destination)==before


def test_baseline_incumbent_pointer_without_promotion_is_never_called_confirmed(tmp_path):
    source,store,_,_=ledger(tmp_path)
    with store.transaction() as db:
        db.execute("DELETE FROM events WHERE kind='promotion'")
        db.execute("UPDATE campaigns SET incumbent='baseline',incumbent_version=0")
    destination=tmp_path/"baseline-pointer-render"
    module.render_search_tree(source,destination,trusted_local=True)
    document=json.loads((destination/"search-tree.json").read_text())
    assert document["search_state"]["incumbent_version"]==0
    baseline=next(node for node in document["nodes"] if node["candidate_id"]=="baseline")
    assert baseline["search_incumbent"] is True
    assert "confirmed_search_incumbent" not in baseline
    svg=(destination/"search-tree.svg").read_text()
    assert "incumbent de recherche" in svg and "incumbent confirmé" not in svg
    assert document["certification"]["verified_by_renderer"] is False
