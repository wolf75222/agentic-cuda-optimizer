#!/usr/bin/env python3
"""Render a trusted local experiments ledger to separate JSON and SVG files.

No model, CUDA, SSH or scheduler call is made. Python snapshots can execute code
while deserializing: select only an experiments database you trust, and provide
--trust-local-pickle. Private suite journals and unknown schemas are refused.
The required --output is a new directory outside the source generation tree.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import textwrap
import time
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from optimizer_agent.portable_export import encoded, file_hash, readable_database
from optimizer_agent.treequest_eligibility import BranchDecision, Eligibility

SCHEMA={
    "campaigns":{"run_id","identity","snapshot","tree_version","incumbent","incumbent_version"},
    "trials":{"run_id","trial_id","status","payload","result","evaluation_id"},
    "evaluations":{"run_id","evaluation_id","payload"},
    "reward_claims":{"run_id","evaluation_id","trial_id"},
    "integration_events":{"run_id","trial_id","processed"},
    "candidates":{"run_id","candidate_id","payload"},
    "candidate_status":{"run_id","candidate_id","valid","reason"},
    "reservations":{"run_id","reservation_id","purpose","amount","actual","created_at","expires_at"},
    "events":{"event_id","run_id","kind","payload","created_at"},
    "attempts":{"attempt_id","run_id","trial_id","status","payload"},
}
LEGEND="Score de RECHERCHE ≠ gain final. L’incumbent de recherche ne prouve pas une certification finale."


@contextmanager
def read_connection(path,*,immutable=False):
    uri=Path(path).resolve().as_uri()+"?mode=ro"+("&immutable=1" if immutable else "")
    db=sqlite3.connect(uri,uri=True)
    db.row_factory=sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        yield db
    finally:
        db.close()


def validate_schema(path,*,immutable=False):
    """Reject suite/checkpoint/private ledgers before any pickle is inspected."""
    with read_connection(path,immutable=immutable) as db:
        names={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if names-{"sqlite_sequence"}!=set(SCHEMA):
            raise ValueError("Only the known experiments schema is accepted; suite/private/unknown journals are refused")
        for table,columns in SCHEMA.items():
            if {row[1] for row in db.execute('PRAGMA table_info("'+table+'")')}!=columns:
                raise ValueError("Unknown experiments table schema: "+table)
        if {row[0] for row in db.execute("SELECT type FROM sqlite_master WHERE type IN ('view','trigger')")}:
            raise ValueError("Views/triggers are not part of the supported experiments schema")


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _selected_ledger(readable,run_id):
    campaigns=readable["tables"]["campaigns"]
    if run_id is None:
        if len(campaigns)!=1:
            raise ValueError("Select --run-id when the experiments database contains multiple campaigns")
        run_id=campaigns[0]["run_id"]
    matches=[row for row in campaigns if row["run_id"]==run_id]
    if len(matches)!=1:
        raise ValueError("The selected run_id is not uniquely present")
    return {"tables":{name:[row for row in rows if row.get("run_id")==run_id]
                      for name,rows in readable["tables"].items() if name!="sqlite_sequence"},
            "search_snapshots":[row for row in readable["search_snapshots"] if row["run_id"]==run_id]},matches[0]


def project_tree(readable,run_id=None):
    """Project serialized facts without changing scores, parents or decisions."""
    ledger,campaign=_selected_ledger(readable,run_id)
    snapshots=ledger["search_snapshots"]
    if len(snapshots)>1:
        raise ValueError("A campaign must have only one search snapshot")
    snapshot=snapshots[0] if snapshots else None
    candidates={row["candidate_id"]:row["payload"] for row in ledger["tables"]["candidates"]}
    validity={row["candidate_id"]:row for row in ledger["tables"]["candidate_status"]}
    nodes=[]; algorithm="snapshot_unavailable"; decisions=[]
    if snapshot and snapshot["algorithm"]=="treequest_ab_mcts_a":
        algorithm=snapshot["algorithm"]
        decisions=snapshot["eligibility"]["decisions"]
        for value in snapshot["nodes"]:
            nodes.append({"key":"tree:"+str(value["index"]),"parent_key":None if value["parent_index"] is None else "tree:"+str(value["parent_index"]),
                "index":value["index"],"score":value["score"],"state":value["state"],"trial_id":value["trial_id"],
                "kind":"integrated_search_node","snapshot_child_indices":value["child_indices"]})
    elif snapshot and snapshot["algorithm"]=="greedy_parent":
        value=snapshot["snapshot"]
        if value.get("algorithm")!="greedy_parent" or value.get("version")!=1:
            raise ValueError("Unknown search snapshot; no fabricated tree is rendered")
        algorithm="greedy_parent"; decisions=value["decisions"]
        for candidate_id,state in value["candidates"].items():
            parent=value["parents"][candidate_id]
            observations=[observation for observation in value["observations"].values() if observation["candidate_id"]==candidate_id]
            nodes.append({"key":"candidate:"+candidate_id,"parent_key":None if parent is None else "candidate:"+parent,
                "index":value["indices"][candidate_id],"score":None,"state":state,"trial_id":None,
                "kind":"greedy_candidate_lineage","score_observations":observations})
    elif snapshot:
        raise ValueError("Unsupported search algorithm")
    by_key={node["key"]:node for node in nodes}
    if len(by_key)!=len(nodes):
        raise ValueError("Duplicate search node identities")
    for node in nodes:
        if node["parent_key"] is not None and node["parent_key"] not in by_key:
            raise ValueError("Search snapshot has a missing parent")
        ancestors=[]; current=node
        while current is not None:
            if current["key"] in ancestors:
                raise ValueError("Search snapshot has a parent cycle")
            ancestors.append(current["key"])
            current=by_key.get(current["parent_key"])
        node["ancestor_keys"]=ancestors
        if "snapshot_child_indices" in node:
            children=[value["index"] for value in nodes if value["parent_key"]==node["key"]]
            if sorted(children)!=sorted(node["snapshot_child_indices"]):
                raise ValueError("Search child indices disagree with parent links")
    identity=campaign["identity"]
    eligibility=Eligibility([BranchDecision(**value) for value in decisions],identity.get("action_families",{}))
    active=eligibility.active_decisions()
    shims={key:SimpleNamespace(state=value["state"],parent=None) for key,value in by_key.items()}
    for key,node in by_key.items():
        shims[key].parent=shims.get(node["parent_key"])
    actions=identity.get("actions",[])
    for node in nodes:
        state=node["state"]; candidate_id=str(state.get("candidate_id","baseline"))
        status=validity.get(candidate_id)
        effective=dict(state)
        if status and status["valid"] is not None:
            effective["valid"]=bool(status["valid"])
        shims[node["key"]].state=effective
        ancestor_ids={str(shims[key].state.get("candidate_id","baseline")) for key in node["ancestor_keys"]}
        reasons=[]
        for decision in active:
            applies=(decision.scope=="candidate" and decision.target_id==candidate_id
                or decision.scope=="parent_action" and decision.target_id[0]==candidate_id
                or decision.scope=="subtree" and decision.target_id in ancestor_ids
                or decision.scope=="strategy_family" and decision.target_id in {eligibility.action_families.get(action,action) for action in actions})
            if applies: reasons.append(decision.to_dict())
        archived=candidates.get(candidate_id,{})
        node.update(candidate_id=candidate_id,artifact_hash=state.get("artifact_hash",archived.get("artifact_hash")),
            source_sha256=hashlib.sha256(state.get("source",archived.get("source","")).encode()).hexdigest(),
            tree_validity=state.get("valid"),current_validity=effective.get("valid"),validity_reason=status.get("reason") if status else None,
            search_incumbent=candidate_id==campaign["incumbent"],
            eligible_actions=[action for action in actions if eligibility.permits(shims[node["key"]],action)],
            closure_reasons=reasons,candidate_reference={"table":"candidates","candidate_id":candidate_id},
            evaluation_references=[row["evaluation_id"] for row in ledger["tables"]["evaluations"]
                if row["payload"].get("candidate_id",row["payload"].get("state",{}).get("candidate_id"))==candidate_id])
        del node["state"]
    # Pending trials are visually distinct placeholders, never scored nodes.
    trials=ledger["tables"]["trials"]
    unfinished=[row for row in trials if row["status"] in {"pending","result_ready"}]
    for trial in unfinished:
        payload=trial["payload"]
        parent_key=("tree:"+str(payload["node_to_expand"]) if algorithm=="treequest_ab_mcts_a" else
                    "candidate:"+str(payload.get("parent_state",{}).get("candidate_id")))
        if parent_key not in by_key:
            raise ValueError("A pending trial refers to an absent search parent")
        nodes.append({"key":"pending:"+trial["trial_id"],"parent_key":parent_key,"candidate_id":None,
            "kind":"pending_trial","trial_id":trial["trial_id"],"status":trial["status"],"score":None,
            "action":payload.get("action"),"artifact_hash":None,"source_sha256":None,
            "search_incumbent":False,"closure_reasons":[],"eligible_actions":[],
            "current_validity":None,"evaluation_references":[trial["evaluation_id"]] if trial["evaluation_id"] else []})
    represented={node["candidate_id"] for node in nodes if node["candidate_id"]}
    abandoned=[{"event_id":row["event_id"],"kind":row["kind"],"created_at":row["created_at"],"payload":row["payload"]}
        for row in ledger["tables"]["events"] if row["kind"] in {"trial_cancelled","research_blocked","candidate_validity","branch_decision","late_result","model_uncertain"}]
    unknown_costs=[row["reservation_id"] for row in ledger["tables"]["reservations"] if row["actual"] is None]
    return {"schema_version":1,"kind":"search_tree_visualization","run_id":campaign["run_id"],"algorithm":algorithm,
        "legend":LEGEND,"nodes":nodes,"branch_decision_history":decisions,
        "active_branch_decisions":[decision.to_dict() for decision in active],"abandonment_and_incident_events":abandoned,
        "unintegrated_candidate_ids":sorted(set(candidates)-represented),
        "search_state":{"status":"incomplete_pending_trials" if unfinished else "snapshot_without_pending_trials" if snapshot else "snapshot_unavailable",
            "execution_completion":"not_inferred_from_absence_of_pending_trials","pending_trials":len(unfinished),
            "unreconciled_cost_reservations":unknown_costs,"tree_version":campaign["tree_version"],
            "incumbent_candidate_id":campaign["incumbent"],"incumbent_version":campaign["incumbent_version"]},
        "certification":{"status":"not_inferred_from_search_registry","verified_by_renderer":False},"ledger":ledger}


def svg_tree(document):
    nodes=document["nodes"]; by_key={node["key"]:node for node in nodes}
    children={key:[] for key in by_key}
    roots=[]
    for node in nodes:
        if node["parent_key"] is None: roots.append(node["key"])
        else: children[node["parent_key"]].append(node["key"])
    positions={}; cursor=0
    def position(key,depth):
        nonlocal cursor
        xs=[position(child,depth+1) for child in children[key]]
        if xs: x=(xs[0]+xs[-1])/2
        else: x=160+cursor*270; cursor+=1
        positions[key]=(x,180+depth*155)
        return x
    for root in roots: position(root,0)
    max_y=max((value[1] for value in positions.values()),default=180)
    width=max(1100,max((value[0] for value in positions.values()),default=900)+160)
    notes=[]
    for decision in document["active_branch_decisions"]:
        notes.extend(textwrap.wrap(f"{decision['state']} {decision['scope']} {decision['target_id']}: {decision['reason']}",135))
    for event in document["abandonment_and_incident_events"]:
        if event["kind"]!="branch_decision":
            notes.extend(textwrap.wrap(f"Événement {event['event_id']} {event['kind']}: {event['payload'].get('reason',event['payload'].get('error','voir JSON'))}",135))
    height=max_y+150+max(1,len(notes))*20+90
    pieces=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Arbre de recherche local</title>',f'<desc id="desc">{escape(LEGEND)}</desc>',
        '<style>text{font-family:system-ui,sans-serif;fill:#17233b}.title{font-size:23px;font-weight:650}.small{font-size:12px}.node{font-size:13px}.edge{fill:none;stroke:#94a3b8;stroke-width:1.5}</style>',
        f'<rect width="{width}" height="{height}" fill="#f8fafc"/>',
        f'<text x="28" y="35" class="title">{escape(document["run_id"])}</text>',
        f'<text x="28" y="61" class="small">{escape(LEGEND)}</text>',
        f'<text x="28" y="83" class="small">Algorithme : {escape(document["algorithm"])} · État : {escape(document["search_state"]["status"])}</text>',
        '<text x="28" y="105" class="small">Vert : incumbent de recherche. Rouge : invalide. Gris : aucune action éligible. Pointillé : essai sans récompense intégrée.</text>',
        '<text x="28" y="126" class="small">Certification finale non déduite ; JSON contient les sources, hachages, mesures et motifs complets.</text>']
    for node in nodes:
        if node["parent_key"] is None: continue
        x,y=positions[node["key"]]; px,py=positions[node["parent_key"]]
        pieces.append(f'<path class="edge" d="M {px} {py+104} C {px} {py+129}, {x} {y-25}, {x} {y}"/>')
    for node in nodes:
        x,y=positions[node["key"]]; x-=120
        pending=node["kind"]=="pending_trial"
        fill="#fff7ed" if pending else "#fee2e2" if node["current_validity"] is False else "#e2e8f0" if not node["eligible_actions"] else "#e0f2fe"
        stroke="#15803d" if node["search_incumbent"] else "#64748b"
        dash=' stroke-dasharray="5 4"' if pending else ""
        score=node["score"]
        score_text=f"score recherche = {score:.6g}" if isinstance(score,(int,float)) and math.isfinite(score) else "score recherche non enregistré"
        title=node["candidate_id"] or "essai "+str(node["trial_id"])
        rows=[str(title),score_text,
            "incumbent de recherche" if node["search_incumbent"] else str(node.get("status") or node["kind"]),
            "source "+str(node["source_sha256"] or "—")[:20],
            "fermetures : "+str(len(node["closure_reasons"]))+" · actions : "+str(len(node["eligible_actions"]))]
        pieces.append(f'<g><title>{escape(str(title))} · {escape(str(node.get("artifact_hash") or node["trial_id"]))}</title><rect x="{x}" y="{y}" width="240" height="104" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{3 if node["search_incumbent"] else 1.5}"{dash}/>')
        for index,text in enumerate(rows):
            pieces.append(f'<text x="{x+10}" y="{y+19+index*18}" class="node">{escape(text[:32]+"…" if len(text)>33 else text)}</text>')
        pieces.append('</g>')
    notes_y=max_y+150
    pieces.append(f'<text x="28" y="{notes_y}" class="title">Fermetures et abandons enregistrés</text>')
    for index,note in enumerate(notes or ["Aucun motif de fermeture/abandon présent dans ce registre."]):
        pieces.append(f'<text x="28" y="{notes_y+30+index*20}" class="small">{escape(note)}</text>')
    pieces.append(f'<text x="28" y="{height-20}" class="small">{len(document["unintegrated_candidate_ids"])} artefacts archivés hors arbre intégré : voir search-tree.json. Aucun appel modèle/GPU/SSH.</text></svg>')
    return "\n".join(pieces)+"\n"


def _cold_fingerprint(source):
    """Read cold files only; immutable SQLite must never ignore committed WAL."""
    result={}
    for suffix in ("","-wal","-shm","-journal"):
        path=Path(str(source)+suffix)
        if not path.exists():
            result[suffix or "database"]=None
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError("Only regular trusted local database files are supported")
        before=path.stat()
        if suffix in {"-wal","-journal"} and before.st_size:
            raise ValueError("The source has a nonempty WAL/journal. Select an existing coherent exported copy; do not stop a running process for this renderer")
        fingerprint={"size":before.st_size,"mtime_ns":before.st_mtime_ns,"ctime_ns":before.st_ctime_ns,
            "inode":before.st_ino,"device":before.st_dev,"sha256":file_hash(path)}
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ctime_ns,before.st_ino,before.st_dev)!=(
                after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino,after.st_dev):
            raise ValueError("Source changed while being inspected; select a coherent offline export")
        result[suffix or "database"]=fingerprint
    if result["database"] is None:
        raise ValueError("The experiments database does not exist")
    return result


def _backup_cold(source,destination):
    """Same SQLite backup primitive as portable_export, with immutable input.

    This renderer intentionally accepts only cold snapshots. The immutable URI
    avoids even SQLite reader WAL/SHM creation in the original generation.
    """
    with read_connection(source,immutable=True) as incoming,sqlite3.connect(destination) as outgoing:
        incoming.execute("BEGIN")
        incoming.backup(outgoing)
        if [row[0] for row in outgoing.execute("PRAGMA integrity_check")]!=["ok"]:
            raise ValueError("The copied database failed SQLite integrity_check")
    return {"method":"sqlite.Connection.backup","source_uri":"mode=ro&immutable=1",
        "integrity_check":"ok","sha256":file_hash(destination)}


def render_search_tree(database,output,*,trusted_local=False,run_id=None,manifest=None):
    if not trusted_local:
        raise ValueError("Select only a trusted local experiments pickle and pass --trust-local-pickle")
    raw_source=Path(database).expanduser()
    if raw_source.is_symlink():
        raise ValueError("Select the real database path explicitly, not a symlink")
    source=raw_source.resolve(); destination=Path(output).expanduser().resolve()
    if destination==source.parent or destination.is_relative_to(source.parent) or source.parent.is_relative_to(destination):
        raise ValueError("--output must be separate from, and cannot overlap, the original generation/database directory")
    if destination.exists():
        raise ValueError("--output must be a fresh separate directory; no existing visualization is overwritten")
    before=_cold_fingerprint(source)
    validate_schema(source,immutable=True)
    with tempfile.TemporaryDirectory(prefix="render-search-tree-") as temporary:
        copied=Path(temporary)/"experiments.sqlite"
        backup=_backup_cold(source,copied)
        if _cold_fingerprint(source)!=before:
            raise ValueError("Source evolved during backup; select an existing coherent offline export")
        validate_schema(copied)
        readable=readable_database(copied)
        document=project_tree(readable,run_id)
    # No completed/empty ledger is silently described as a finished execution.
    if manifest is not None:
        manifest_path=Path(manifest).resolve()
        body=manifest_path.read_bytes(); value=json.loads(body)
        if value.get("run_id")!=document["run_id"]:
            raise ValueError("The optional status manifest belongs to another campaign")
        document["external_status_manifest"]={"path":str(manifest_path),"sha256":hashlib.sha256(body).hexdigest(),
            "reported_status":value.get("status"),"reported_method_status":value.get("method_status"),
            "reported_generation_complete":value.get("generation_complete"),
            "reported_certification_status":value.get("certification_status"),"verified_by_renderer":False}
    document["capture"]={"source_database":str(source),"source_files_before":before,
        "source_files_unchanged":True,"backup":backup,"observed_at_unix":time.time(),
        "input_scope":"Trusted cold local experiments snapshot; no suite journal or private final seed database",
        "coherence_scope":"One verified SQLite backup; no cross-file checkpoint/manifest atomicity claim",
        "pickle_policy":"Explicit trusted local input only; never load untrusted downloaded/archived pickles",
        "model_calls":0,"gpu_requests":0,"ssh_calls":0}
    svg=svg_tree(document)
    # Reject invalid XML characters in source labels before creating any output.
    from xml.etree import ElementTree
    ElementTree.fromstring(svg)
    payload=encoded(document)+b"\n"
    if _cold_fingerprint(source)!=before:
        raise ValueError("Source evolved during rendering; no visualization was written")
    destination.mkdir(parents=True,exist_ok=False)
    destination.chmod(0o700)
    for name,data in (("search-tree.json",payload),("search-tree.svg",svg.encode())):
        with (destination/name).open("xb") as file:
            file.write(data);file.flush();os.fsync(file.fileno())
        (destination/name).chmod(0o600)
    return {"json":str(destination/"search-tree.json"),"svg":str(destination/"search-tree.svg"),
        "run_id":document["run_id"],"nodes":len(document["nodes"]),
        "search_state":document["search_state"],"certification":document["certification"],
        "source_files_unchanged":True,"model_calls":0,"gpu_requests":0,"ssh_calls":0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database",required=True,type=Path,help="Trusted cold experiments.sqlite; private suite journals are refused")
    parser.add_argument("--output",required=True,type=Path,help="Fresh directory outside the original generation/database directory")
    parser.add_argument("--trust-local-pickle",action="store_true",help="Acknowledge that this selected local search pickle is trusted")
    parser.add_argument("--run-id",help="Required if the experiments database contains multiple campaigns")
    parser.add_argument("--manifest",type=Path,help="Optional same-run status declaration, recorded separately from verified search facts")
    args=parser.parse_args(argv)
    try:
        result=render_search_tree(args.database,args.output,trusted_local=args.trust_local_pickle,
            run_id=args.run_id,manifest=args.manifest)
    except (ValueError,sqlite3.DatabaseError,OSError) as error:
        parser.error(str(error))
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())

