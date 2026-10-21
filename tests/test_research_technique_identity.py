"""CPU documentary fixtures only; no live research, model, DB or GPU claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from optimizer_agent.codex_backend import CodexResult
from optimizer_agent.research_service import (NeedsResearch, ResearchService,
    diagnostic_research_context, generate_with_context_receipt, synthesis_schema)
from tests.test_campaign_diagnostics import diagnostic_fixture
from tests import test_codex_research as fixtures


def diagnostic():
    return diagnostic_fixture(contract_hash="hash", environment_fingerprint="gpu-fingerprint",
        needs_research=True, stage="proposal", error="needs_research: proposal introduces an uncovered technique",
        proposal={"hypothesis":"128-output register reuse is an unexecuted hypothesis",
            "cuda_constraint":"ordinary block barrier", "expected_measurement":"future target timings",
            "techniques":["register_blocking"], "references":["fixture-code"]},
        candidate={"candidate_id":"accepted", "source":"// distinct 64-output accepted source"})


class Backend:
    def __init__(self, directory, coverage):
        self.directory, self.coverage, self.calls = directory, coverage, []

    def generate_json(self, prompt, schema, **options):
        phase = "search" if not self.calls else "synthesis"
        intents = list(self.directory.glob("*-"+phase+"-context-intent.json"))
        assert len(intents)==1  # Durable public intent precedes either fake call.
        intent=json.loads(intents[0].read_text())
        assert intent["status"]=="intent" and intent["prompt_sha256"]==hashlib.sha256(prompt.encode()).hexdigest()
        serialized=prompt.split("Context: ",1)[1].split("\nActually read source content:",1)[0]
        assert intent["serialized_context_sha256"]==hashlib.sha256(serialized.encode()).hexdigest()
        assert intent["public_context_complete"] is True and intent["public_context"]==json.loads(serialized)
        self.calls.append((prompt,deepcopy(schema)))
        if phase=="search":
            return CodexResult({"sources":[],"rejected_examples":[],"queries":[]},{},[
                {"type":"item.completed","item":{"id":"explicit-cpu-search","type":"web_search",
                    "action":{"type":"search","queries":["explicit CPU fixture query"]}}}],
                "gpt-6-luna","max",1,"explicit-cpu-search")
        context=json.loads(prompt.split("Context: ",1)[1].split("\nActually read source content:",1)[0])
        assert context["evidence"]["diagnostic"]["proposed_technique_names"]==["register_blocking"]
        assert "exact, case-sensitive" in prompt and "leave\nunsupported names absent" in prompt
        assert "register_blocking" in schema["properties"]["covered_techniques"]["description"]
        contents=json.loads(prompt.split("Actually read source content:\n",1)[1])
        code=[item["source"]["source_id"] for item in contents if item["source"]["kind"]=="code"]
        negative=self.coverage=="negative"
        names=["shared_tiling"]+({"supported":["register_blocking"],
            "paraphrase":["Optional register reuse across adjacent outputs"],"negative":[]}[self.coverage])
        payload=dict(covered_techniques=names,reference_kernels=[] if negative else code,
            launch_requirements=["future guarded single-kernel implementation"],dependencies=[],
            applicability_checks=[dict(condition="ordinary block semantics",evidence="explicit CPU source fixture",satisfied=True)],
            incompatible_conditions=["register reuse unsupported in this fixture"] if negative else [],
            missing_evidence=[],pmpp_topics=["tiling"],hypotheses=[dict(documented_fact="explicit CPU block semantics",
                observed_code_property="explicit CPU shared tile",expected_effect_to_measure="future GPU timings",source_ids=code)],
            expected_effects=["unmeasured"],validation_metrics=["independent oracle"],
            negative_search=["Consulted fixture code does not support the requested register-reuse path"] if negative else [],
            original_hypothesis="Alternative shared tile without the unsupported register reuse" if negative else "")
        return CodexResult(payload,{},[],"gpt-6-luna","max",1,"explicit-cpu-synthesis")


def fetcher(url,timeout,limit):
    if "raw.githubusercontent.com" in url:
        return ("// explicit CPU fixture, never compiled\nstruct BlockReduce {};\n__global__ void gemm_device() {}\n"
                +"__global__ void kernel() {}\n"*10).encode(),url
    return b"<html><p>Explicit CPU NVIDIA documentation fixture: ordinary block synchronization</p></html>",url


@pytest.mark.parametrize("coverage",["supported","paraphrase","negative"])
def test_exact_identity_guidance_never_relabels_or_forces_targeted_coverage(tmp_path,coverage):
    backend=Backend(tmp_path,coverage); original=diagnostic(); before=deepcopy(original)
    dossier=ResearchService(backend,tmp_path,fetcher=fetcher).prepare({"contract_hash":"hash"},
        {"candidate_id":"baseline"},{"strategy_family":"tiling"},
        {"environment_fingerprint":"gpu-fingerprint","diagnostic":original})
    assert original==before and dossier.diagnostic_ids==[original["diagnostic_id"]]
    assert dossier.research_ready("hash","tiling","gpu-fingerprint")
    payload=fixtures.proposal_payload();payload["references"]=[s.source_id for s in dossier.sources]
    payload["techniques"]=["register_blocking"]
    if coverage=="supported":dossier.validate_proposal(payload)
    else:
        with pytest.raises(NeedsResearch,match="uncovered technique"):dossier.validate_proposal(payload)
        assert "register_blocking" not in dossier.covered_techniques
    if coverage=="negative":
        assert dossier.documented_negative_finding() and not dossier.reference_kernels
        payload["techniques"]=["shared_tiling"];dossier.validate_proposal(payload)
    receipts=list(tmp_path.glob("*-context-result.json"));assert len(receipts)==2
    for path in receipts:
        result=json.loads(path.read_text());intent_path=Path(result["intent"]["path"])
        assert hashlib.sha256(intent_path.read_bytes()).hexdigest()==result["intent"]["sha256"]
        assert result["status"]=="completed" and result["call_id"].startswith("explicit-cpu-")


def test_context_and_schema_preserve_exact_names_without_requiring_them(tmp_path):
    value=diagnostic();context=diagnostic_research_context({"diagnostic":value})
    assert context["proposed_technique_names"]==["register_blocking"]
    context["proposed_technique_names"].append("must-not-mutate-diagnostic")
    assert value["proposal_hypothesis"]["techniques"]==["register_blocking"]
    dossier=fixtures.ResearchTests().make_dossier(tmp_path)
    schema=synthesis_schema(dossier.sources,proposed_techniques=["register_blocking"])
    assert "not a mandatory coverage checklist" in schema["properties"]["covered_techniques"]["description"]
    assert "enum" not in schema["properties"]["covered_techniques"]["items"]


def test_public_context_receipt_excludes_private_roles_and_binds_exact_components(tmp_path):
    context={"workload":{"contract_hash":"public", "runtime_evidence_context":{"job_id":"fixture","seed":234},
        "hardware_evidence_context":{"job_id":"fixture"},"config":{"seed":123},"input":[42]},
        "action":{"strategy_family":"tiling"},"evidence":{"diagnostic":{"diagnostic_id":"fixture","inputs":[9],
            "native_request":{"arguments":[{"input_file":"PRIVATE_NATIVE_INPUT","output_file":"PRIVATE_NATIVE_OUTPUT",
                "data":["PRIVATE_NATIVE_DATA"],"values":["PRIVATE_NATIVE_VALUES"]}]},
            "observed_evidence":[{"kind":"typed_failure","input_file":"PRIVATE_OBSERVED_INPUT",
                "input_values":["PRIVATE_OBSERVED_VALUES"]}]}},
        "config":{"private":"CONFIG_SENTINEL"},"seed":999,"inputs":["INPUT_SENTINEL"]}
    serialized=json.dumps(context);prompt="Context: "+serialized
    class Capture:
        def generate_json(self,*args,**kwargs):
            intent=json.loads((tmp_path/"fixture-search-context-intent.json").read_text())
            assert intent["status"]=="intent" and "call_id" not in intent
            return CodexResult({}, {}, [], "gpt-6-luna","max",1,"explicit-cpu-returned")
    generate_with_context_receipt(Capture(),tmp_path,"fixture","search",prompt,context,{})
    intent=json.loads((tmp_path/"fixture-search-context-intent.json").read_text())
    raw=(tmp_path/"fixture-search-context-intent.json").read_text()
    assert "CONFIG_SENTINEL" not in raw and "INPUT_SENTINEL" not in raw
    assert "PRIVATE_NATIVE" not in raw and "PRIVATE_OBSERVED" not in raw
    assert intent["public_context_complete"] is False
    assert intent["serialized_context_sha256"]==hashlib.sha256(serialized.encode()).hexdigest()
    assert "seed" not in intent["public_context"]["workload"]["runtime_evidence_context"]
    for key,value in {"diagnostic":context["evidence"]["diagnostic"],
        "hardware_evidence_context":context["workload"]["hardware_evidence_context"],
        "runtime_evidence_context":context["workload"]["runtime_evidence_context"]}.items():
        assert intent["component_context_sha256"][key]==hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def test_failed_call_leaves_only_intent_and_cannot_overwrite_or_repeat_it(tmp_path):
    class Failure:
        calls=0
        def generate_json(self,*args,**kwargs):
            self.calls+=1;raise RuntimeError("Explicit CPU transport failure, usage unknown")
    backend=Failure();context={"workload":{"contract_hash":"public"}}
    prompt="Context: "+json.dumps(context)
    with pytest.raises(RuntimeError):generate_with_context_receipt(backend,tmp_path,"fixture","search",prompt,context,{})
    path=tmp_path/"fixture-search-context-intent.json";before=path.read_bytes()
    assert json.loads(before)["status"]=="intent" and not list(tmp_path.glob("*-context-result.json"))
    with pytest.raises(FileExistsError):generate_with_context_receipt(backend,tmp_path,"fixture","search",prompt,context,{})
    assert backend.calls==1 and path.read_bytes()==before


def test_mismatched_context_cannot_create_intent_or_invoke_backend(tmp_path):
    class Forbidden:
        def generate_json(self,*args,**kwargs):raise AssertionError("Mismatched context reached backend")
    with pytest.raises(ValueError,match="exact outgoing prompt"):
        generate_with_context_receipt(Forbidden(),tmp_path,"fixture","search","Context: {}",
            {"workload":{"contract_hash":"different"}}, {})
    assert not list(tmp_path.iterdir())


def test_context_intent_directory_is_synced_before_call(tmp_path,monkeypatch):
    from optimizer_agent import research_service as module
    import os,stat
    original=os.fsync;synced=[]
    def fsync(descriptor):
        synced.append("directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file")
        return original(descriptor)
    monkeypatch.setattr(module.os,"fsync",fsync)
    class Backend:
        def generate_json(self,*args,**kwargs):
            assert synced==["file","directory"]
            return CodexResult({}, {}, [], "gpt-6-luna","max",1,"explicit-cpu-result")
    context={"workload":{"contract_hash":"public"}}
    generate_with_context_receipt(Backend(),tmp_path,"fixture","search","Context: "+json.dumps(context),context,{})
    assert synced==["file","directory","file","directory"]


def test_new_context_directory_ancestors_are_synced_before_first_call(tmp_path,monkeypatch):
    from optimizer_agent import research_service as module
    import os,stat
    directory=tmp_path/"new-campaign"/"research"
    original=os.fsync;directories=[]
    def fsync(descriptor):
        metadata=os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):directories.append((metadata.st_dev,metadata.st_ino))
        return original(descriptor)
    monkeypatch.setattr(module.os,"fsync",fsync)
    class Backend:
        def generate_json(self,*args,**kwargs):
            for path in (tmp_path,directory.parent,directory):
                metadata=path.stat();assert (metadata.st_dev,metadata.st_ino) in directories
            return CodexResult({}, {}, [], "gpt-6-luna","max",1,"explicit-cpu-result")
    context={"workload":{"contract_hash":"public"}}
    generate_with_context_receipt(Backend(),directory,"fixture","search","Context: "+json.dumps(context),context,{})


def test_public_complementary_request_keeps_only_named_state_and_reference_fields():
    from optimizer_agent.research_service import public_research_context
    request={"kind":"explicit_complementary_coverage_refresh","accepted_research_id":"fixture-research",
        "accepted_dossier":{"path":"campaign/trials/fixture/dossier.json","sha256":"a"*64},
        "missing_techniques":["register_blocking"],"diagnostic_id":"fixture-diagnostic","next_fresh_round":5,
        "post_research_state":{"research_round":4,"research_ready":True,"needs_research":False,
            "snapshot":"PRIVATE_SNAPSHOT","seed":123},"requested_state":{"research_round":4,"needs_research":True},
        "unknown_envelope":{"inputs":[9]},"config":{"secret":"PRIVATE_CONFIG"}}
    context={"workload":{"documentary_coverage_refresh_control":request}}
    public=public_research_context(context)["workload"]["documentary_coverage_refresh_control"]
    assert public["accepted_dossier"]==request["accepted_dossier"] and public["next_fresh_round"]==5
    assert public["post_research_state"]=={"research_round":4,"research_ready":True,"needs_research":False}
    assert "PRIVATE_" not in json.dumps(public) and "unknown_envelope" not in public
    assert public_research_context({"workload":{"documentary_coverage_refresh_control":public}})=={
        "workload":{"documentary_coverage_refresh_control":public}}
