"""Synthetic public summaries only: no incident files, study DB or GPU."""
from copy import deepcopy

import pytest

from optimizer_agent.quality_suite import summarize_suite


BAD, GOOD, OTHER = "a"*64, "b"*64, "c"*64


def fixture():
    modes=["full","native_codex_vanilla","no_pmpp"]
    entries=[{"mode":mode,"workload":"matmul","seed":1,"group_id":"group",
              "config":{"run_id":"run-"+mode}} for mode in modes]
    manifests={row["config"]["run_id"]:{"status":"generation_complete","method_status":"completed",
        "candidate":{"artifact_hash":artifact},"usage":{"model_tokens":10,"model_calls":2,"gpu_seconds":5,
            "local_configurations":3},"elapsed_seconds":7,"metrics":{"historical_search_gain":9}}
        for row,artifact in zip(entries,[BAD,GOOD,OTHER])}
    candidates={mode:{"status":"certified","qualified":True,"artifact_hash":artifact,
        "confirmation":{"interval":{"gain":gain,"lower":gain-.1,"upper":gain+.1,
            "case_gains":{"final-17":{"gain":gain}}}},"time_to_independently_confirmed_final_gain_seconds":12}
        for mode,artifact,gain in zip(modes,[BAD,GOOD,OTHER],[2.0,1.5,.8])}
    comparisons={"full_vs_native_codex_vanilla":{"ok":True,"status":"promoted","artifact_hash":BAD,
        "incumbent_artifact_hash":GOOD,"critical_cases_ok":True,"interval":{"gain":1.3,"lower":1.2,"upper":1.4}},
        "full_vs_no_pmpp":{"ok":True,"status":"promoted","artifact_hash":BAD,
        "incumbent_artifact_hash":OTHER,"interval":{"gain":2.5}},
        "native_codex_vanilla_vs_no_pmpp":{"ok":True,"status":"promoted","artifact_hash":GOOD,
        "incumbent_artifact_hash":OTHER,"interval":{"gain":1.8}},
        "no_pmpp_vs_native_codex_vanilla":{"ok":True,"status":"not_improved","artifact_hash":OTHER,
        "incumbent_artifact_hash":GOOD,"interval":{"gain":.6}}}
    groups={"group":{"execution_complete":True,"all_modes_available":True,"candidates":candidates,
                     "method_comparisons":comparisons}}
    return {"entries":entries,"modes":modes,"workloads":["matmul"]},manifests,groups


def confirmed(artifact=BAD):
    return {artifact:[{"incident_id":"incident-"+artifact,"kind":"declared_artifact_qualification_incident",
        "status":"confirmed_static_violation","scientific_qualification_eligible":False,
        "identity":{"run_id":"run-full","artifact_hash":artifact},
        "authentication":{"incident_evidence_verified":True}}]}


def test_confirmed_incident_removes_only_exact_artifact_from_qualified_statistics():
    plan,manifests,groups=fixture();before=deepcopy((plan,manifests,groups));incidents=confirmed()
    summary=summarize_suite(plan,manifests,groups,artifact_qualification_incidents=incidents)
    row=summary["rows"][0]
    assert row["status"]=="artifact_qualification_not_admitted"
    assert row["qualified"] is False and row["final_qualified"] is False
    assert row["artifact_qualification_incident_ids"]==["incident-"+BAD]
    assert row["gain"] is row["lower"] is row["upper"] is None and row["case_gains"]=={}
    assert row["observed_unqualified_status"]=="certified"
    assert row["observed_unqualified_gain"]==2.0 and row["observed_unqualified_lower"]==1.9
    assert row["observed_unqualified_case_gains"]=={"final-17":{"gain":2.0}}
    assert row["time_to_independently_confirmed_final_gain_seconds"] is None
    assert row["observed_unqualified_time_to_independently_confirmed_final_gain_seconds"]==12
    assert row["generation_method_status"]=="completed" and row["model_tokens"]==10
    assert row["gpu_reservation_wall_seconds"]==5 and row["metrics"]=={"historical_search_gain":9}
    assert summary["aggregates"]["full"]["method_completed"]==1
    assert summary["aggregates"]["full"]["method_failed"]==0
    assert summary["aggregates"]["full"]["qualified"]==0
    assert summary["aggregates"]["full"]["certified"]==0
    assert summary["aggregates"]["full"]["final_measured"]==0
    assert summary["aggregates"]["full"]["geometric_mean_all_measured_gain"] is None
    assert summary["rows"][1]["gain"]==1.5 and summary["rows"][2]["gain"]==.8
    assert summary["aggregates"]["no_pmpp"]["regressions"]==1
    assert summary["complete"] and not summary["scientific_qualified"]
    assert (plan,manifests,groups)==before


@pytest.mark.parametrize('role',["artifact_hash","candidate_artifact_hash","incumbent_artifact_hash",
                                  "left_artifact_hash","right_artifact_hash"])
def test_disqualified_artifact_filters_either_role_without_changing_unrelated_bidirectional_pairs(role):
    plan,manifests,groups=fixture();comparison=groups['group']['method_comparisons']['full_vs_native_codex_vanilla']
    comparison.pop('artifact_hash');comparison.pop('incumbent_artifact_hash');comparison[role]=BAD
    before=deepcopy(groups)
    summary=summarize_suite(plan,manifests,groups,artifact_qualification_incidents=confirmed())
    filtered=summary['method_comparisons']['group']
    for key in ['full_vs_native_codex_vanilla','full_vs_no_pmpp']:
        assert filtered[key]['status']=='artifact_qualification_not_admitted'
        assert filtered[key]['ok'] is False and filtered[key]['qualified'] is False
        assert filtered[key]['interval'] is None
        assert filtered[key]['observed_unqualified_interval']==before['group']['method_comparisons'][key]['interval']
    for key in ['native_codex_vanilla_vs_no_pmpp','no_pmpp_vs_native_codex_vanilla']:
        for field,value in before['group']['method_comparisons'][key].items():
            assert filtered[key][field]==value
        assert filtered[key]['kind']=='artifact_aware_method_comparison_summary'
    assert not summary['direct_comparisons_qualified'] and groups==before


def test_legacy_contrast_without_hashes_is_bound_by_predeclared_mode_labels():
    plan,manifests,groups=fixture()
    groups['group']['method_comparisons']['full_vs_native_codex_vanilla']={
        'ok':True,'status':'exact_artifact_identity'}
    row=summarize_suite(plan,manifests,groups,artifact_qualification_incidents=confirmed())['method_comparisons']['group']['full_vs_native_codex_vanilla']
    assert row['ok'] is False and row['observed_unqualified_status']=='exact_artifact_identity'


def test_uncertain_identified_artifact_is_not_permission_to_keep_qualified_gain():
    plan,manifests,groups=fixture();observation={'status':'uncertain','qualification_status':'not_admitted',
        'artifact_hash':GOOD,'reason':'out_of_scope'};before=deepcopy(observation)
    summary=summarize_suite(plan,manifests,groups,artifact_incident_observations=[observation])
    assert summary['rows'][0]['gain']==2
    row=summary['rows'][1];assert row['gain'] is None and row['observed_unqualified_gain']==1.5
    assert row['artifact_qualification_status']=='not_admitted_uncertain_incident'
    assert row['artifact_incident_observation_refs']==[{'observation_index':0,'status':'uncertain'}]
    assert not row['final_qualified'] and not summary['scientific_qualified']
    assert observation==before


def test_unresolved_incident_identity_prevents_global_qualification_without_inventing_a_bad_artifact():
    plan,manifests,groups=fixture();before=deepcopy(groups)
    summary=summarize_suite(plan,manifests,groups,artifact_incident_observations=[{
        'status':'uncertain','qualification_status':'not_admitted','artifact_hash':None,'error':'unreadable registry'}])
    assert summary['complete'] and not summary['scientific_qualified']
    assert summary['artifact_qualification_eligibility_known'] is False
    assert summary['artifact_evaluations_qualified'] is False
    assert all(row['artifact_qualification_status']=='eligibility_unknown' and not row['artifact_incident_blocked'] for row in summary['rows'])
    assert [row['gain'] for row in summary['rows']]==[2,1.5,.8]
    for pair,contrast in before['group']['method_comparisons'].items():
        for field,value in contrast.items():assert summary['method_comparisons']['group'][pair][field]==value
    assert groups==before


def test_aliases_of_same_artifact_all_block_without_turning_generation_methods_into_failures():
    plan,manifests,groups=fixture();groups['group']['candidates']['native_codex_vanilla']['artifact_hash']=BAD
    groups['group']['candidates']['native_codex_vanilla'].update(status='validated_no_gain',confirmation={})
    summary=summarize_suite(plan,manifests,groups,artifact_qualification_incidents=confirmed())
    row=summary['rows'][1]
    assert row['gain'] is None and row['observed_unqualified_gain']==1
    assert row['observed_unqualified_gain_basis']=='exact_baseline_artifact'
    assert row['generation_method_status']=='completed' and not row['final_qualified']


def test_other_artifact_incident_does_not_invalidate_this_completed_comparison():
    plan,manifests,groups=fixture()
    summary=summarize_suite(plan,manifests,groups,artifact_qualification_incidents=confirmed('d'*64))
    assert summary['scientific_qualified'] and summary['direct_comparisons_qualified']
    assert all(row['qualified'] for row in summary['rows'])
    assert 'does not prove absence of races' in summary['artifact_incident_qualification_scope']


def test_protocol_incident_method_failure_remains_separate_and_legacy_calls_still_work():
    plan,manifests,groups=fixture()
    baseline=summarize_suite(plan,manifests,groups)
    assert baseline['scientific_qualified']
    summary=summarize_suite(plan,manifests,groups,protocol_incidents={'run-full':[{'incident_id':'protocol'}]},
                            artifact_qualification_incidents=confirmed())
    row=summary['rows'][0]
    assert row['generation_method_status']=='failed' and row['original_generation_method_status']=='completed'
    assert summary['method_protocols_qualified'] is False and not summary['scientific_qualified']
    assert row['observed_unqualified_gain']==2 and row['model_tokens']==baseline['rows'][0]['model_tokens']


def test_lightweight_comparison_view_never_duplicates_embedded_raw_evidence():
    plan,manifests,groups=fixture();pairs=groups['group']['method_comparisons']
    for contrast in pairs.values():
        contrast.update(pairs=[{'report':{'private':'RAW_PAIR_SENTINEL'}}],request={'arguments':'RAW_REQUEST_SENTINEL'},
                        observations={'buffer':'RAW_OBSERVATION_SENTINEL'},
                        candidate_id={'request':'RAW_TYPED_ID_SENTINEL'},case_ids=[{'report':'RAW_TYPED_CASE_SENTINEL'}])
        contrast['interval'].update(report='RAW_INTERVAL_REPORT_SENTINEL',seed=123)
        contrast['interval']['case_gains']={'final-17':{'gain':1.2,'lower':1.1,'upper':1.3,
                                                      'report':'RAW_CASE_REPORT_SENTINEL'}}
    before=deepcopy(groups)
    summary=summarize_suite(plan,manifests,groups,artifact_qualification_incidents=confirmed())
    views=summary['method_comparisons']['group']
    import json
    encoded=json.dumps(views)
    assert 'SENTINEL' not in encoded and 'seed' not in encoded
    assert summary['method_comparisons_view_kind']=='artifact_aware_method_comparison_summary'
    for name,view in views.items():
        assert view['group_id']=='group' and view['pair_key']==name and view['raw_available_in_group'] is True
        assert view['candidate_id'] is None and view['case_ids'] is None
        assert all(key not in view for key in ['pairs','request','observations','report'])
    affected=views['full_vs_native_codex_vanilla']
    assert affected['interval'] is None
    assert affected['observed_unqualified_interval']['case_gains']=={'final-17':{'gain':1.2,'lower':1.1,'upper':1.3}}
    assert groups==before
