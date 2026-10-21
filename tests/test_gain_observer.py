"""CPU-only observer protocol tests; no scheduler, GPU or provider calls."""
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from optimizer_agent.gain_observer import GainObserver, ObserverInfrastructureError, read_observer_metrics


class Decisions:
    contract_hash = "frozen-common-search-contract"

    def __init__(self, directory, outcomes):
        self.directory, self.outcomes = directory, iter(outcomes)
        self.calls = 0

    def confirm(self, candidate, baseline):
        state = json.loads((self.directory/"summary.json").read_text())
        assert state["observations"][candidate.artifact_hash]["status"] == "pending"
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def candidate(artifact="kernel-A", name="candidate-A"):
    return SimpleNamespace(candidate_id=name, artifact_hash=artifact)


def observer(tmp_path, outcomes, **kwargs):
    evaluator = Decisions(tmp_path, outcomes)
    return GainObserver(evaluator,candidate("naive"),tmp_path,method="cpu-test",
        campaign_started_at_unix=time.time()-5,**kwargs),evaluator


def test_intent_precedes_measurement_and_same_artifact_has_one_paired_decision(tmp_path):
    value,evaluator = observer(tmp_path,[{"ok":True,"status":"inconclusive"},
                                         {"ok":True,"status":"promoted"}])
    first = value.observe(candidate(),evidence_id="effective-benchmark-1")
    again = value.observe(candidate(name="a-different-local-id"),evidence_id="effective-benchmark-2")
    assert first == again and evaluator.calls == 1
    assert value.snapshot()["metrics"]["time_to_first_useful_gain_seconds"] is None
    value.observe(candidate("kernel-B"),evidence_id="effective-benchmark-3")
    metrics = value.snapshot()["metrics"]
    assert evaluator.calls == 2 and metrics["first_useful_gain_status"] == "observed"
    assert metrics["time_to_first_useful_gain_seconds"] >= 5
    assert metrics["model_tokens_to_first_useful_gain"] is None
    assert metrics["model_calls_to_first_useful_gain"] is None


@pytest.mark.parametrize("failure",[KeyboardInterrupt("interrupted"),RuntimeError("infrastructure_failure")])
def test_uncertain_result_is_durable_and_cannot_be_remeasured_after_restart(tmp_path,failure):
    value,evaluator = observer(tmp_path,[failure,{"ok":True,"status":"promoted"}])
    with pytest.raises((KeyboardInterrupt,ObserverInfrastructureError)):
        value.observe(candidate(),evidence_id="benchmark")
    state = value.snapshot()
    assert state["metrics"]["first_useful_gain_status"] == "unknown_incident"
    assert state["first_useful_gain"] is None
    with pytest.raises(ObserverInfrastructureError):
        value.observe(candidate("different-artifact"),evidence_id="new-benchmark")
    with pytest.raises(ObserverInfrastructureError):
        GainObserver(evaluator,candidate("naive"),tmp_path,method="cpu-test",
            campaign_started_at_unix=state["identity"]["campaign_started_at_unix"])
    assert evaluator.calls == 1


def test_returned_quarantine_is_persisted_and_stops_observer_without_a_retry(tmp_path):
    value,evaluator = observer(tmp_path,[{"ok":False,"status":"infrastructure_failure",
        "report":{"context_reusable":False,"quarantined":True}}])
    with pytest.raises(ObserverInfrastructureError):
        value.observe(candidate(),evidence_id="benchmark")
    state = value.snapshot()
    assert state["observations"]["kernel-A"]["confirmation"]["report"]["quarantined"]
    assert state["metrics"]["first_useful_gain_status"] == "unknown_incident"
    assert evaluator.calls == 1


def test_known_method_consumption_and_unknown_native_tokens_are_not_conflated(tmp_path):
    value,_ = observer(tmp_path,[{"ok":True,"status":"promoted"}],usage_provider=lambda:{
        "model_tokens":None,"model_calls":None,"method_gpu_evaluation_wall_seconds":3.5})
    value.observe(candidate(),evidence_id="benchmark")
    metrics = value.snapshot()["metrics"]
    assert metrics["model_tokens_to_first_useful_gain"] is None
    assert metrics["method_consumption"]["method_gpu_evaluation_wall_seconds"] == 3.5
    assert metrics["combined_method_and_observer_gpu_wall_seconds"] == 3.5+metrics["observer_wall_seconds"]
    assert "no feedback" in metrics["first_useful_gain_scope"]


def test_crash_recovery_reads_first_gain_without_instantiating_evaluator_or_remeasurement(tmp_path):
    value,evaluator = observer(tmp_path,[{"ok":True,"status":"promoted"}],usage_provider=lambda:{
        "model_tokens":None,"model_calls":None,"method_gpu_evaluation_wall_seconds":1.5})
    value.observe(candidate(),evidence_id="actual-benchmark")
    # Simulate crash before parent snapshot/CLI receipt integration.
    raw = json.loads((tmp_path/"summary.json").read_text())
    assert "metrics" not in raw
    recovered = read_observer_metrics(tmp_path)
    assert recovered["first_useful_gain_status"] == "observed"
    assert recovered["model_tokens_to_first_useful_gain"] is None
    assert "subsequent method consumption is unknown" in recovered["consumption_scope"]
    assert evaluator.calls == 1
    final = value.snapshot()["metrics"]
    assert read_observer_metrics(tmp_path) == final and evaluator.calls == 1
