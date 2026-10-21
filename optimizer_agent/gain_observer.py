"""Independent first-useful-gain observer; never feeds a generator policy.

Only search inputs are used. Each artifact receives at most one fresh paired
confirmation. A durable uncertain intent is not retried until it happens to win.
Observer results live outside original graph state and native HTTP responses.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid


class ObserverInfrastructureError(RuntimeError):
    """The metric was persisted unavailable; GPU protection must stop activity."""


def _write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False,default=str))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    temporary.replace(path)
    descriptor = os.open(path.parent,os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assert_observer_available(directory):
    """CPU-only restart guard: an uncertain observation cannot resume GPU work."""
    path = Path(directory)/"summary.json"
    if path.exists() and any(row["status"] in {"pending","unknown_interrupted","metric_unavailable"}
                            for row in json.loads(path.read_text())["observations"].values()):
        raise ObserverInfrastructureError("Prior independent gain observation is uncertain; GPU activity must stop")


def _metrics(state,method_usage,has_runner):
    rows = list(state["observations"].values())
    first = state["first_useful_gain"]
    known_wall = sum(row.get("observer_wall_seconds",0) for row in rows)
    observer_wall = known_wall if all("observer_wall_seconds" in row for row in rows) else None
    method_gpu = method_usage.get("method_gpu_evaluation_wall_seconds")
    identities = sum(row.get("decision") == "exact_artifact_identity" or row["status"] == "identity_observed" for row in rows)
    unproven = sum(row.get("decision") == "artifact_identity_context_unproven" or row["status"] == "identity_context_unproven" for row in rows)
    return {
        "time_to_first_useful_gain_seconds":first["elapsed_seconds"] if first else None,
        "model_tokens_to_first_useful_gain":first["consumption"].get("model_tokens") if first else None,
        "model_calls_to_first_useful_gain":first["consumption"].get("model_calls") if first else None,
        "first_useful_gain_status":"observed" if first else "unknown_incident" if any(
            row["status"] in {"pending","unknown_interrupted","metric_unavailable"} for row in rows) else
            "unknown_identity_context" if unproven else "no_confirmed_gain",
        "observer_artifacts":len(rows),"observer_confirmations_completed":sum(row["status"] == "completed"
            and row.get("decision") not in {"exact_artifact_identity","artifact_identity_context_unproven"}
            and row.get("confirmation",{}).get("paired_measurement_performed") is not False for row in rows),
        "observer_identity_observations":identities,"observer_identity_context_unproven":unproven,
        "observer_wall_seconds":observer_wall,"observer_wall_seconds_known_lower_bound":known_wall,
        "observer_gpu_requests":sum(row["observer_gpu_requests"] for row in rows
            if row.get("observer_gpu_requests") is not None) if has_runner else None,
        "method_consumption":method_usage,
        "combined_method_and_observer_gpu_wall_seconds":method_gpu+observer_wall
            if method_gpu is not None and observer_wall is not None else None,
        "first_useful_gain_scope":"Common search validation and paired confirmation; no feedback or final certification"}


def read_observer_metrics(directory):
    """Read durable metrics after a controller crash, without runner or GPU work.

    A finalized snapshot includes all consumption observed by its parent. Without
    one, totals stop at the last durable callback; absent later cost stays unknown.
    """
    state = json.loads((Path(directory)/"summary.json").read_text())
    if "metrics" in state:
        return state["metrics"]
    rows = list(state["observations"].values())
    last = max(rows,key=lambda row:row["started_at_unix"],default={})
    usage = last.get("consumption_at_observation",last.get("consumption_before_observer",{}))
    return {**_metrics(state,usage,any(row.get("observer_gpu_requests") is not None for row in rows)),
        "consumption_scope":"Last durable observer callback; subsequent method consumption is unknown"}


def _walk(value):
    if isinstance(value,dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value,list):
        for item in value:
            yield from _walk(item)


def _infrastructure_failure(result):
    return any(row.get("context_reusable") is False or row.get("quarantined") is True
        or row.get("status") in {"infrastructure_failure","tool_unavailable","worker_unavailable","request_unavailable"}
        or row.get("stage") in {"transport","environment","harness"}
        for row in _walk(result))


class _ObservedRunner:
    def __init__(self,runner,directory):
        self.runner,self.directory = runner,Path(directory)
        self.calls = []

    def __getattr__(self,name):
        return getattr(self.runner,name)

    def run(self,run_id,request):
        identity = "observer-"+hashlib.sha256(str(self.directory).encode()).hexdigest()[:8]+"-"+str(run_id)
        path = self.directory/(identity+".json")
        record = {"observer_only":True,"run_id":identity,"started_at_unix":time.time(),"request":asdict(request)}
        _write(path,record)
        started = time.monotonic()
        try:
            result = self.runner.run(identity,request)
            record["report"] = result
            return result
        except BaseException as error:
            record["error"] = type(error).__name__+": "+str(error)
            raise
        finally:
            record["ended_at_unix"] = time.time()
            record["wall_seconds"] = time.monotonic()-started
            _write(path,record)
            self.calls.append(record)


class GainObserver:
    def __init__(self,evaluator,naive,directory,*,method,campaign_started_at_unix,usage_provider=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.directory.chmod(0o700)
        self.method,self.naive = method,naive
        self.started = campaign_started_at_unix
        self.usage_provider = usage_provider
        self._thread_lock = threading.RLock()
        self.path = self.directory/"summary.json"
        self.identity = {"method":method,"contract_hash":evaluator.contract_hash,
            "baseline_artifact_hash":naive.artifact_hash,"campaign_started_at_unix":campaign_started_at_unix}
        self.runner = None
        self.evaluator = evaluator
        if all(hasattr(evaluator,name) for name in ("runner","contract","oracle","_suites")):
            self.runner = _ObservedRunner(evaluator.runner,self.directory/"request-receipts")
            suites = {name:evaluator._suites[name] for name in ("short","search","performance")}
            self.evaluator = type(evaluator)(self.runner,evaluator.contract,evaluator.oracle,suites,
                scheduler=evaluator.scheduler)
            if self.evaluator.contract_hash != evaluator.contract_hash:
                raise ValueError("Observer requires the exact common search evaluator, without a final suite")
        with self._lock():
            if self.path.exists():
                if self._load()["identity"] != self.identity:
                    raise ValueError("Observer contract/baseline/start identity changed")
            else:
                self._save({"identity":self.identity,"observations":{},"first_useful_gain":None,
                    "observer_policy":"At most one identity or paired comparison decision per artifact; no feedback to generation or original selection",
                    "certification":"Search observer only; no final qualification"})
            assert_observer_available(self.directory)

    @contextmanager
    def _lock(self):
        with self._thread_lock,(self.directory/"observer.lock").open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)

    def _load(self):
        return json.loads(self.path.read_text())

    def _save(self,state):
        state.pop("metrics",None)  # A new intent/result invalidates an older consumption snapshot.
        _write(self.path,state)

    def _usage(self):
        if self.usage_provider is None:
            return {"model_tokens":None,"input_tokens":None,"output_tokens":None,"model_calls":None,
                "note":"Provider usage is unavailable at this observation point"}
        try:
            return self.usage_provider()
        except Exception as error:
            return {"model_tokens":None,"model_calls":None,"note":"Consumption callback unavailable: "+str(error)}

    def _request_costs(self,offset):
        calls = self.runner.calls[offset:] if self.runner else []
        reports = [call.get("report",{}) for call in calls]
        samples = [row.get("latency_samples_ms") for row in reports]
        return {"observer_gpu_requests":len(calls) if self.runner else None,
            "observer_request_wall_seconds":sum(call["wall_seconds"] for call in calls) if self.runner else None,
            "observer_cuda_event_kernel_seconds":sum(sum(series) for series in samples)/1000
                if samples and all(isinstance(series,list) for series in samples) else 0.0 if self.runner and not calls else None}

    def observe(self,candidate,*,evidence_id):
        artifact = candidate.artifact_hash
        with self._lock():
            state = self._load()
            if artifact in state["observations"]:
                previous = state["observations"][artifact]
                if previous["status"] == "pending":
                    previous.update(status="unknown_interrupted",decision=None,
                        error="A prior confirmation intent has no completed receipt; no automatic remeasurement")
                    self._save(state)
                assert_observer_available(self.directory)
                return previous
            assert_observer_available(self.directory)
            receipt = {"artifact_hash":artifact,"candidate_id":candidate.candidate_id,"evidence_id":evidence_id,
                "status":"pending","decision":None,"started_at_unix":time.time(),
                "consumption_before_observer":self._usage(),"observer_only":True}
            state["observations"][artifact] = receipt
            self._save(state)  # Intent precedes any validation/GPU work.
            started = time.monotonic()
            call_offset = len(self.runner.calls) if self.runner else 0
            try:
                result = self.evaluator.confirm(candidate,self.naive)
            except BaseException as error:
                receipt.update(status="unknown_interrupted" if not isinstance(error,Exception) else "metric_unavailable",
                    error=type(error).__name__+": "+str(error),ended_at_unix=time.time(),
                    observer_wall_seconds=time.monotonic()-started,consumption_at_observation=self._usage(),
                    **self._request_costs(call_offset))
                self._save(state)
                if isinstance(error,Exception):
                    raise ObserverInfrastructureError("Independent gain observation unavailable; GPU activity must stop: "+str(error)) from error
                raise
            observation_status = ("identity_observed" if result.get("status") == "exact_artifact_identity" else
                "identity_context_unproven" if result.get("status") == "artifact_identity_context_unproven" else
                "completed" if result.get("ok") is True else "not_qualified")
            receipt.update(status=observation_status,
                decision=result.get("status"),confirmation=result,ended_at_unix=time.time(),
                observer_wall_seconds=time.monotonic()-started,consumption_at_observation=self._usage())
            receipt.update(self._request_costs(call_offset))
            if _infrastructure_failure(result):
                receipt["status"] = "metric_unavailable"
                self._save(state)
                raise ObserverInfrastructureError("Independent gain observation encountered infrastructure failure; preserve worker protection")
            if result.get("ok") is True and result.get("status") == "promoted" and state["first_useful_gain"] is None:
                receipt["elapsed_to_useful_gain_seconds"] = receipt["ended_at_unix"]-self.started
                state["first_useful_gain"] = {"artifact_hash":artifact,"evidence_id":evidence_id,
                    "confirmed_at_unix":receipt["ended_at_unix"],
                    "elapsed_seconds":receipt["elapsed_to_useful_gain_seconds"],
                    "consumption":receipt["consumption_at_observation"],
                    "scope":"Independent search paired gain; elapsed includes observer overhead, not an equal-cost budget"}
            self._save(state)
            return receipt

    def snapshot(self):
        with self._lock():
            return self._snapshot()

    def _snapshot(self):
        state = self._load()
        state["metrics"] = _metrics(state,self._usage(),self.runner is not None)
        _write(self.path,state)
        return state
