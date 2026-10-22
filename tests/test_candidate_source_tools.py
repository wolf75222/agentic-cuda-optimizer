"""CPU-only projections and real stdio child processes; no Codex/GPU/SSH/DB."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from optimizer_agent import candidate_source_tools as tools


RUN = "test-run"
CONTRACT = "a" * 64
ENVIRONMENT = {"name":"fake CPU fixture GPU", "architecture":"sm_90", "compute_capability":{"major":9,"minor":0},
               "uuid":"cpu-fixture-only", "driver_version":1, "nvrtc_version":"test", "cuda_toolkit_version":1,
               "harness_build_id":"cpu-fixture"}
ENV = hashlib.sha256(json.dumps(ENVIRONMENT,sort_keys=True).encode()).hexdigest()
CONTEXT = dict(run_id=RUN,contract_hash=CONTRACT,environment_fingerprint=ENV)
MODULE = Path(tools.__file__).resolve()


def candidate(identifier="candidate-1"):
    source='extern "C" __global__ void kernel(float* out) { out[0] = 1.0f; }\n'
    digest=hashlib.sha256(source.encode()).hexdigest()
    launch={"kind":"fixed","block":[1,1,1],"grid":[1,1,1],"elements_key":"n",
            "shared_memory_bytes":0,"version":1,"tile_x":None,"tile_y":None}
    parameters={"TILE":16}
    artifact=tools._sha(tools._encoded({"source":digest,"kernel":"kernel","launch_rule":launch,
                                      "parameters":json.dumps(parameters,sort_keys=True)}))
    return {"candidate_id":identifier,"source":source,"source_hash":digest,"artifact_hash":artifact,
            "kernel":"kernel","launch_rule":launch,"parameters":parameters,"tree_parent_id":"baseline",
            "source_parent_id":"baseline","trial_id":"trial-1","strategy_family":"tiling",
            "references":["source-1"],"evaluations":[{"evaluation_id":"eval-1","candidate_id":identifier,
                "contract_hash":CONTRACT,"suite":"performance","status":"inconclusive","ok":True,
                "critical_cases_ok":False,"incumbent_id":"baseline","environment":ENVIRONMENT,
                "case_ids":["search-17"],"cases":{"search-17":{"latency_ms":2.0,"critical":True,
                    "input_file":"PRIVATE_INPUT_SENTINEL","report":{"oracle":"PRIVATE_ORACLE_SENTINEL"}}},
                "interval":{"gain":1.02,"lower":0.98,"upper":1.05,"confidence":0.95,"seed":919191,
                            "case_gains":{"search-17":{"gain":0.9,"lower":0.8,"upper":1.0}}},
                "pairs":[{"private":"PRIVATE_PAIR_SENTINEL"}],"request":{"arguments":"PRIVATE_ARGS_SENTINEL"}},
                {"evaluation_id":"final-private","suite":"final","cases":{"final-1":{"oracle":"PRIVATE_FINAL_SENTINEL"}}}],
            "current_validity":{"valid":False,"reason":"PRIVATE_REASON_SENTINEL /inputs/a.bin"},
            "seed":919191,"rng":"PRIVATE_RNG_SENTINEL","oracle":"PRIVATE_ORACLE_TOP_SENTINEL"}


def snapshot(tmp_path, records=None):
    return tools.create_snapshot(records if records is not None else [candidate()],tmp_path/"snapshot.json",**CONTEXT)


def initialize():
    return {"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-11-25","capabilities":{},"clientInfo":{"name":"CPU test","version":"1"}}}


def call(candidate_id="candidate-1", rpc_id=3, **arguments):
    return {"jsonrpc":"2.0","id":rpc_id,"method":"tools/call","params":{
        "name":"get_candidate_source","arguments":{"candidate_id":candidate_id,**arguments}}}


def server(tmp_path):
    info=snapshot(tmp_path)
    result=tools.CandidateSourceServer(info["path"],info["sha256"],tmp_path/"audit",
        expected_run_id=RUN,expected_contract_hash=CONTRACT,expected_environment_fingerprint=ENV)
    result.handle(initialize());result.handle({"jsonrpc":"2.0","method":"notifications/initialized"})
    return result,info


def process(info, audit, messages, *, raw=None):
    command=[sys.executable,str(MODULE),"--snapshot",info["path"],"--snapshot-sha256",info["sha256"],
             "--audit-directory",str(audit),"--run-id",RUN,"--contract-hash",CONTRACT,"--environment-fingerprint",ENV]
    data=raw if raw is not None else b"".join(json.dumps(x).encode()+b"\n" for x in messages)
    return subprocess.run(command,input=data,capture_output=True,timeout=10,cwd=MODULE.parent.parent)


def test_public_projection_exact_source_and_search_only_with_uncertainty(tmp_path):
    raw=candidate();projected=tools.project_candidate(raw,**CONTEXT)
    assert projected["source"]==raw["source"]
    assert projected["parameters"]==raw["parameters"]
    assert projected["launch_rule"]==raw["launch_rule"]
    assert projected["references"]==["source-1"]
    assert projected["validity"]=={"valid":False,"reason_sha256":tools._sha(raw["current_validity"]["reason"].encode())}
    assert projected["measurements"][0]["status"]=="inconclusive"
    assert projected["measurements"][0]["critical_cases_ok"] is False
    assert projected["measurements"][0]["interval"]["lower"]==0.98
    assert projected["omitted_evaluation_count"]==1
    text=json.dumps(projected)
    assert "PRIVATE_" not in text and "919191" not in text
    assert all(word not in text for word in ('input_file','oracle','pairs','request','rng','seed','final-private'))
    info=snapshot(tmp_path,[raw]);assert info["run_id"]==RUN and info["candidate_count"]==1
    loaded=tools.load_snapshot(info["path"],info["sha256"],expected_run_id=RUN)
    assert loaded["candidates"]==[projected]


@pytest.mark.parametrize("field,value",[("run_id","foreign"),("contract_hash","b"*64),("environment_fingerprint","c"*64)])
def test_explicit_foreign_candidate_context_refused(field,value):
    record=candidate();record[field]=value
    with pytest.raises(ValueError):tools.project_candidate(record,**CONTEXT)


@pytest.mark.parametrize("change",["source","source_hash","artifact_hash","parameters","launch_rule","candidate_id","reference"])
def test_tampered_executable_or_illegal_ids_refused(change):
    record=candidate()
    if change=="source":record["source"]+="// changed"
    elif change=="source_hash":record["source_hash"]="b"*64
    elif change=="artifact_hash":record["artifact_hash"]="b"*64
    elif change=="parameters":record["parameters"]["TILE"]=32
    elif change=="launch_rule":record["launch_rule"]["grid"][0]=2
    elif change=="candidate_id":record["candidate_id"]="../../secret"
    else:record["references"]=["/private/path"]
    with pytest.raises(ValueError):tools.project_candidate(record,**CONTEXT)


@pytest.mark.parametrize("field,value",[("candidate_id","foreign"),("run_id","foreign"),
    ("contract_hash","b"*64),("environment_fingerprint","c"*64),("environment",{**ENVIRONMENT,"uuid":"foreign"})])
def test_foreign_search_measurement_refused(field,value):
    record=candidate();record["evaluations"][0][field]=value
    with pytest.raises(ValueError):tools.project_candidate(record,**CONTEXT)


def test_tree_reward_latencies_and_unknown_validity_are_preserved():
    record=candidate();record.pop("current_validity");record["evaluations"]=[{
        "evaluation_id":"eval-tree","purpose":"tree_reward","valid":True,"latencies":{"search-17":3.0}}]
    p=tools.project_candidate(record,**CONTEXT)
    assert p["validity"]=={"valid":None}
    assert p["measurements"][0]["cases"]==[{"case_id":"search-17","latency_ms":3.0}]
    assert p["measurements"][0]["status"]=="unknown"


def test_duplicate_candidate_or_measurement_identity_refused(tmp_path):
    with pytest.raises(ValueError):snapshot(tmp_path,[candidate(),candidate()])
    record=candidate();record["evaluations"]=[record["evaluations"][0]]*2
    with pytest.raises(ValueError):tools.project_candidate(record,**CONTEXT)
    assert not (tmp_path/"snapshot.json").exists()


@pytest.mark.parametrize("where",["root","candidate","launch","measurement","case","interval","case_gain","validity"])
def test_re_pinned_snapshot_unknown_nested_fields_refused(tmp_path,where):
    info=snapshot(tmp_path);p=Path(info["path"]);data=json.loads(p.read_text());c=data["candidates"][0];m=c["measurements"][0]
    targets={"root":data,"candidate":c,"launch":c["launch_rule"],"measurement":m,"case":m["cases"][0],
             "interval":m["interval"],"case_gain":m["interval"]["case_gains"]["search-17"],"validity":c["validity"]}
    targets[where]["input_file"]="PRIVATE_FORGED_SENTINEL"
    p.write_bytes(tools._encoded(data))
    with pytest.raises(ValueError):tools.load_snapshot(p,tools._sha(p.read_bytes()))


@pytest.mark.parametrize("kind",["symlink","hardlink","changed_sha","duplicate_key","nonfinite","schema_bool"])
def test_unsafe_or_malformed_snapshot_refused_before_audit_creation(tmp_path,kind):
    info=snapshot(tmp_path);path=Path(info["path"])
    if kind=="symlink":
        other=tmp_path/"pointer.json";other.symlink_to(path);info["path"]=str(other)
    elif kind=="hardlink":os.link(path,tmp_path/"linked.json")
    elif kind=="changed_sha":path.write_bytes(path.read_bytes()+b" ")
    elif kind=="duplicate_key":path.write_bytes(b'{"schema_version":1,"schema_version":1}');info["sha256"]=tools._sha(path.read_bytes())
    elif kind=="nonfinite":path.write_bytes(b'{"x":NaN}');info["sha256"]=tools._sha(path.read_bytes())
    else:
        data=json.loads(path.read_text());data["schema_version"]=True;path.write_bytes(tools._encoded(data));info["sha256"]=tools._sha(path.read_bytes())
    audit=tmp_path/"not-created";run=process(info,audit,[])
    assert run.returncode==2 and run.stdout==b"" and not audit.exists()


def test_exclusive_snapshot_does_not_overwrite_existing_file(tmp_path):
    info=snapshot(tmp_path);before=Path(info["path"]).read_bytes()
    with pytest.raises(FileExistsError):snapshot(tmp_path)
    assert Path(info["path"]).read_bytes()==before
    assert not list(tmp_path.glob('*.tmp'))


def test_stdio_real_process_lifecycle_tool_audit_and_no_extra_capabilities(tmp_path):
    info=snapshot(tmp_path);audit=tmp_path/"tool-audit"
    messages=[initialize(),{"jsonrpc":"2.0","method":"notifications/initialized"},
        {"jsonrpc":"2.0","id":2,"method":"tools/list"},call(),call("absent",4),call(rpc_id=5,path="/private/oracle"),
        {"jsonrpc":"2.0","id":6,"method":"ping"},{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":99}}]
    run=process(info,audit,messages);assert run.returncode==0 and run.stderr==b""
    replies=[json.loads(line) for line in run.stdout.splitlines()];assert len(replies)==6
    assert replies[0]["result"]["capabilities"]=={"tools":{"listChanged":False}}
    listing=replies[1]["result"]["tools"];assert len(listing)==1 and listing[0]["name"]=="get_candidate_source"
    assert listing[0]["annotations"]=={"readOnlyHint":True,"destructiveHint":False,"idempotentHint":True,"openWorldHint":False}
    assert json.loads(replies[2]["result"]["content"][0]["text"])["candidate"]["source"]==candidate()["source"]
    assert replies[2]["result"]["isError"] is False
    assert replies[3]["result"]["isError"] is True and replies[4]["result"]["isError"] is True
    assert "PRIVATE_" not in run.stdout.decode() and "919191" not in run.stdout.decode()
    intents=list(audit.glob('*.intent.json'));results=list(audit.glob('*.result.json'));assert len(intents)==len(results)==3
    outcomes=[]
    for path in results:
        row=json.loads(path.read_text());intent_path=Path(row['intent']['path']);intent=json.loads(intent_path.read_text())
        assert tools._sha(intent_path.read_bytes())==row['intent']['sha256']
        assert row['request_sha256']==intent['request_sha256']
        assert intent['tool_name']=='get_candidate_source' and isinstance(intent['rpc_id'],int)
        matching=[reply for reply in replies if tools._sha(tools._encoded(reply))==row['response_sha256']]
        assert len(matching)==1 and (row['outcome']=='success')==(matching[0]['result']['isError'] is False)
        assert 'PRIVATE_' not in path.read_text()+intent_path.read_text()
        outcomes.append(row['outcome'])
    assert sorted(outcomes)==['refused','refused','success']


def test_zero_tools_call_means_zero_consultation_receipts(tmp_path):
    info=snapshot(tmp_path);audit=tmp_path/"audit"
    run=process(info,audit,[initialize(),{"jsonrpc":"2.0","method":"notifications/initialized"},
        {"jsonrpc":"2.0","id":2,"method":"tools/list"}])
    assert run.returncode==0 and not list(audit.iterdir())


def test_snapshot_changed_after_start_refused_not_cached_success(tmp_path):
    s,info=server(tmp_path);Path(info['path']).write_bytes(Path(info['path']).read_bytes()+b' ')
    response=s.handle(call());assert response['result']['isError'] is True
    assert json.loads(next((tmp_path/'audit').glob('*.result.json')).read_text())['outcome']=='refused'


def test_every_tool_call_including_early_or_illegal_envelope_is_audited(tmp_path):
    info=snapshot(tmp_path);s=tools.CandidateSourceServer(info['path'],info['sha256'],tmp_path/'audit')
    assert s.handle(call())['error']['code']==-32002
    s.handle(initialize());s.handle({'jsonrpc':'2.0','method':'notifications/initialized'})
    request=call();request['private_argument']='PRIVATE_SENTINEL'
    assert s.handle(request)['error']['code']==-32600
    assert len(list((tmp_path/'audit').glob('*.intent.json')))==2
    assert all('PRIVATE_SENTINEL' not in p.read_text() for p in (tmp_path/'audit').iterdir())


def test_readonly_duplicate_rpc_ids_are_distinct_observed_consultations(tmp_path):
    s,_=server(tmp_path)
    for _ in range(8):assert s.handle(call())['result']['isError'] is False
    assert len(list((tmp_path/'audit').glob('*.result.json')))==8


def test_audit_publication_failure_cannot_return_success(tmp_path,monkeypatch):
    s,_=server(tmp_path);publish=tools._publish
    def fail_result(path,value):
        if str(path).endswith('.result.json'):raise OSError('simulated disk failure')
        return publish(path,value)
    monkeypatch.setattr(tools,'_publish',fail_result)
    with pytest.raises(OSError):s.handle(call())
    assert len(list((tmp_path/'audit').glob('*.intent.json')))==1
    assert not list((tmp_path/'audit').glob('*.result.json'))


def test_intent_durable_before_lookup_and_result_durable_before_response(tmp_path,monkeypatch):
    s,_=server(tmp_path);load=tools.load_snapshot;publish=tools._publish;observed=[]
    def verify_load(*args,**kwargs):
        assert observed==['intent']
        assert len(list((tmp_path/'audit').glob('*.intent.json')))==1
        return load(*args,**kwargs)
    def verify_publish(path,value):
        result=publish(path,value);observed.append('intent' if str(path).endswith('.intent.json') else 'result')
        return result
    monkeypatch.setattr(tools,'load_snapshot',verify_load);monkeypatch.setattr(tools,'_publish',verify_publish)
    response=s.handle(call())
    assert observed==['intent','result'] and response['result']['isError'] is False


def test_unknown_tool_is_protocol_error_and_has_audited_refusal(tmp_path):
    s,_=server(tmp_path);request=call();request['params']['name']='execute_kernel'
    response=s.handle(request)
    assert response['error']['code']==-32602 and 'result' not in response
    receipt=json.loads(next((tmp_path/'audit').glob('*.result.json')).read_text())
    assert receipt['outcome']=='protocol_error'


def test_malformed_utf8_json_and_oversized_stdio_frames(tmp_path):
    info=snapshot(tmp_path);audit=tmp_path/'audit'
    run=process(info,audit,[],raw=b'\xff\n{"jsonrpc":"2.0","id":1,"id":2}\n')
    assert run.returncode==0 and all(json.loads(x)['error']['code']==-32700 for x in run.stdout.splitlines())
    run=process(info,audit,[],raw=b'x'*(tools.MAX_FRAME_BYTES+2)+b'\n')
    assert run.returncode==2 and json.loads(run.stdout)['error']['code']==-32700
    assert not list(audit.iterdir())


def test_excessively_nested_json_is_cleanly_refused(tmp_path):
    info=snapshot(tmp_path);run=process(info,tmp_path/'audit',[],raw=b'['*2000+b'0'+b']'*2000+b'\n')
    assert run.returncode==0 and run.stderr==b''
    # Parser recursion limits vary by interpreter. A parsed JSON array is still
    # not an MCP request; either parser or envelope rejection must stay clean.
    assert json.loads(run.stdout)['error']['code'] in {-32700,-32600}


@pytest.mark.parametrize('protocol',['2024-11-05','2025-03-26','2025-06-18','2025-11-25','future'])
def test_protocol_negotiation_and_pre_initialized_guard(tmp_path,protocol):
    info=snapshot(tmp_path);s=tools.CandidateSourceServer(info['path'],info['sha256'],tmp_path/'audit')
    assert 'error' in s.handle({'jsonrpc':'2.0','id':0,'method':'tools/list'})
    request=initialize();request['params']['protocolVersion']=protocol
    assert s.handle(request)['result']['protocolVersion']==(protocol if protocol in tools.PROTOCOL_VERSIONS else '2025-11-25')


def test_symlink_audit_directory_and_context_pin_refused(tmp_path):
    info=snapshot(tmp_path);actual=tmp_path/'actual';actual.mkdir();link=tmp_path/'link';link.symlink_to(actual,target_is_directory=True)
    with pytest.raises(ValueError):tools.CandidateSourceServer(info['path'],info['sha256'],link)
    with pytest.raises(ValueError):tools.load_snapshot(info['path'],info['sha256'],expected_run_id='foreign')


def test_snapshot_beneath_symlink_ancestor_refused_before_open_or_audit(tmp_path,monkeypatch):
    real=tmp_path/'real';real.mkdir();info=snapshot(real)
    alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
    audit=tmp_path/'never-created-audit'
    def refuse_open(*args,**kwargs):
        pytest.fail('Snapshot bytes must not be opened through an ancestor alias')
    monkeypatch.setattr(tools.os,'open',refuse_open)
    with pytest.raises(ValueError,match='ancestor'):
        tools.CandidateSourceServer(alias/'snapshot.json',info['sha256'],audit)
    assert not audit.exists()


def test_existing_audit_directory_beneath_alias_refused_without_writes(tmp_path):
    info=snapshot(tmp_path);real=tmp_path/'real';real.mkdir();audit=real/'existing-audit';audit.mkdir()
    sentinel=audit/'preserved.txt';sentinel.write_bytes(b'unchanged')
    alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError,match='ancestor'):
        tools.CandidateSourceServer(info['path'],info['sha256'],alias/'existing-audit')
    assert list(audit.iterdir())==[sentinel] and sentinel.read_bytes()==b'unchanged'


def test_load_missing_path_never_materializes_directories(tmp_path):
    missing=tmp_path/'missing'/'nested'/'snapshot.json'
    with pytest.raises(FileNotFoundError):tools.load_snapshot(missing,'a'*64)
    assert not (tmp_path/'missing').exists()


def test_replaced_audit_ancestor_is_rechecked_before_call_publication(tmp_path):
    info=snapshot(tmp_path);parent=tmp_path/'audit-parent';parent.mkdir()
    audit=parent/'calls';s=tools.CandidateSourceServer(info['path'],info['sha256'],audit)
    s.handle(initialize());s.handle({'jsonrpc':'2.0','method':'notifications/initialized'})
    archived=tmp_path/'original-audit-parent';parent.rename(archived)
    other=tmp_path/'other';other.mkdir();(other/'calls').mkdir();parent.symlink_to(other,target_is_directory=True)
    with pytest.raises(ValueError,match='ancestor'):s.handle(call())
    assert not list((archived/'calls').iterdir()) and not list((other/'calls').iterdir())


def test_snapshot_creation_refuses_parent_alias_and_traversal(tmp_path):
    real=tmp_path/'real';real.mkdir();alias=tmp_path/'alias';alias.symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError,match='ancestor'):
        tools.create_snapshot([candidate()],alias/'nested'/'snapshot.json',**CONTEXT)
    with pytest.raises(ValueError,match='traversal'):
        tools.create_snapshot([candidate()],real/'..'/'snapshot.json',**CONTEXT)
    assert not (real/'nested').exists() and not (tmp_path/'snapshot.json').exists()
