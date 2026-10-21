"""Bounded local parameter search using the campaign's common evaluator.

CUDA source substitutions use explicit {{PARAMETER}} tokens. Launch parameters
are materialized with the same parameter dictionary; no Python/CUDA eval is used.
Every enumerated configuration (including a preflight rejection) is accounted
for, and the winner receives a new independent measurement before returning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import uuid

from .experiment_store import BudgetExceeded, canonical_json


@dataclass
class TuningObservation:
    parameters: dict
    candidate_id: str | None
    status: str
    reason: str | None = None
    validation: dict | None = None
    measurement: dict | None = None


@dataclass
class LocalSearchResult:
    best_candidate: Any | None
    evaluation: dict | None
    configurations: list[TuningObservation]
    remeasurement: dict | None
    termination_reason: str


def _latency_score(measurement):
    cases = measurement.get("cases",{})
    if not cases:
        raise ValueError("A tuning measurement must contain performance cases")
    weights = [case.get("weight",1/len(cases)) for case in cases.values()]
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Invalid tuning weights")
    total_weight = sum(weights)
    score = 0.0
    for weight,case in zip(weights,cases.values()):
        latency = case["latency_ms"]
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("Invalid tuning latency")
        score += weight/total_weight*math.log(latency)
    return math.exp(score)


class Autotuner:
    def __init__(self,evaluator,store=None,run_id=None,materialize=None,work_directory=None):
        self.evaluator,self.store,self.run_id = evaluator,store,run_id
        self.materialize = materialize or self._materialize
        # Returned candidates must outlive this adapter instance. In campaigns
        # keep every concrete source in the persistent artifact directory.
        self._workspace = Path(work_directory or (store.artifact_dir/"autotuning" if store else tempfile.mkdtemp(prefix="cuda-autotuning-")))
        self._workspace.mkdir(parents=True,exist_ok=True)

    def _materialize(self,template,parameters):
        from .evaluation import Candidate,LaunchRule
        if isinstance(template,dict):
            source = template["source"]
            source = source.read_text() if isinstance(source,Path) else source
            launch = LaunchRule(**template["launch_rule"])
            kernel = template["kernel"]
            bindings = template.get("parameter_bindings",[])
        else:
            source = template.source.read_text()
            launch,kernel = template.launch_rule,template.kernel
            bindings = []
        launch_keys = {"block_size","block_x","block_y","block_z","tile_x","tile_y","shared_memory_bytes"}
        mapped = {}
        for binding in bindings:
            parameter,target = binding["parameter"],binding["target"]
            if parameter not in parameters or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",parameter):
                raise ValueError("Binding references an unknown or unsafe parameter")
            if target == "source_macro":
                if isinstance(parameters[parameter],bool) or not isinstance(parameters[parameter],int):
                    raise ValueError("source_macro domains must contain integer values")
                pattern = re.compile(r"(?m)^(\s*#\s*define\s+"+re.escape(parameter)+r"\s+)[0-9]+(?:[uUlL]*)(\s*(?://[^\n]*)?)$")
                source,count = pattern.subn(lambda match:match.group(1)+str(parameters[parameter])+match.group(2),source)
                if count != 1:
                    raise ValueError(f"source_macro {parameter} requires one existing integer #define")
            elif target in launch_keys:
                if target in mapped and mapped[target] != parameters[parameter]:
                    raise ValueError("Launch parameter bindings disagree")
                mapped[target] = parameters[parameter]
            else:
                raise ValueError(f"Unsupported parameter binding: {target}")
        bound = {binding["parameter"] for binding in bindings}
        unknown = [key for key in parameters if "{{"+key+"}}" not in source and key not in launch_keys and key not in bound]
        if unknown:
            raise ValueError(f"Parameters lack a source/launch binding: {unknown}")
        for key,value in parameters.items():
            source = source.replace("{{"+key+"}}",str(value))
        if re.search(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}",source):
            raise ValueError("Source has unresolved parameter bindings")
        block = list(launch.block)
        effective = dict(parameters,**mapped)
        for key,value in effective.items():
            if key in launch_keys and (isinstance(value,bool) or not isinstance(value,int)):
                raise ValueError("Launch parameter bindings require integer values")
        for index,key in enumerate(("block_x","block_y","block_z")):
            if key in effective:
                block[index] = int(effective[key])
        if "block_size" in effective:
            if "block_x" in effective and effective["block_x"] != effective["block_size"]:
                raise ValueError("block_x and block_size disagree")
            block[0] = int(effective["block_size"])
        updates = {key:effective[key] for key in ("tile_x","tile_y","shared_memory_bytes") if key in effective}
        launch = replace(launch,block=tuple(block),**updates)
        candidate_id = "tune-"+hashlib.sha256((source+kernel+canonical_json(parameters)+canonical_json(asdict(launch))).encode()).hexdigest()[:24]
        source_path = self._workspace/(candidate_id+".cu")
        if source_path.exists() and source_path.read_text() != source:
            raise ValueError("Autotuning artifact identity collision")
        source_path.write_text(source)
        return Candidate(candidate_id,source_path,kernel,launch,dict(parameters))

    @staticmethod
    def _preflight(candidate,parameters,restrictions,hardware_limits):
        for restriction in restrictions:
            if not restriction(parameters):
                return "parameter restriction rejected configuration"
        launch = candidate.get("launch_rule",{}) if isinstance(candidate,dict) else candidate.launch_rule
        block = launch.get("block",(1,1,1)) if isinstance(launch,dict) else launch.block
        if any(int(dim) != dim or dim <= 0 for dim in block):
            return "block dimensions must be positive integers"
        if math.prod(block) > hardware_limits.get("max_threads_per_block",1024):
            return "thread block exceeds hardware limit"
        for dim,limit in zip(block,hardware_limits.get("max_block_dimensions",(1024,1024,64))):
            if dim > limit:
                return "block dimension exceeds hardware limit"
        shared = launch.get("shared_memory_bytes",0) if isinstance(launch,dict) else launch.shared_memory_bytes
        shared_limit = hardware_limits.get("max_shared_memory_per_block_bytes",hardware_limits.get("max_shared_memory_bytes",48*1024))
        if shared < 0 or shared > shared_limit:
            return "shared memory exceeds configured hardware limit"
        return None

    def _reserve(self,reservation_id,amount):
        if self.store:
            self.store.reserve(self.run_id,reservation_id,amount,purpose="autotuning")

    def _reconcile(self,reservation_id,actual):
        if self.store:
            self.store.reconcile(self.run_id,reservation_id,actual)

    def search(self,candidate_template,parameter_space,budget,*,restrictions=(),hardware_limits=None):
        if isinstance(budget,int):
            budget = {"configurations":budget}
        count_limit = int(budget.get("configurations",budget.get("local_configurations",0)))
        compilation_limit = int(budget.get("compilations",count_limit))
        timeout = budget.get("wall_seconds")
        if timeout is not None:
            timeout = float(timeout)
        if (count_limit < 0 or compilation_limit < 0 or
                (timeout is not None and (not math.isfinite(timeout) or timeout <= 0))):
            raise ValueError("Autotuning counts must be nonnegative; an explicit timeout must be finite and positive")
        if not parameter_space or any(not values for values in parameter_space.values()):
            raise ValueError("Parameter domains must be nonempty")
        limits = hardware_limits or {}
        start = time.monotonic() if timeout is not None else None
        observations,candidates = [],{}
        best,best_measurement,best_score = None,None,float("inf")
        compiled = 0
        termination = "domain_exhausted"
        # itertools.product is lazy in the number of combinations. A hard count
        # limit is checked before materialization or compilation.
        combinations = itertools.product(*(parameter_space[key] for key in parameter_space))
        for values in combinations:
            if len(observations) >= count_limit:
                termination = "configuration_budget"
                break
            if timeout is not None and time.monotonic()-start >= timeout:
                termination = "wall_budget"
                break
            parameters = dict(zip(parameter_space,values))
            reservation_id = "tune-"+uuid.uuid4().hex
            try:
                self._reserve(reservation_id,{"local_configurations":1})
            except BudgetExceeded:
                termination = "global_budget"
                break
            candidate = None
            try:
                candidate = self.materialize(candidate_template,parameters)
                reason = self._preflight(candidate,parameters,restrictions,limits)
            except (ValueError,TypeError) as error:
                reason = str(error)
            candidate_id = (candidate.get("candidate_id") if isinstance(candidate,dict) else candidate.candidate_id) if candidate else None
            if reason:
                observations.append(TuningObservation(parameters,candidate_id,"filtered",reason))
                self._reconcile(reservation_id,{"local_configurations":1})
                continue
            if compiled >= compilation_limit:
                self._reconcile(reservation_id,{"local_configurations":1})
                observations.append(TuningObservation(parameters,candidate_id,"budget_rejected","compilation sub-budget exhausted"))
                termination = "compilation_budget"
                break
            work_id = reservation_id+"-compile"
            try:
                self._reserve(work_id,{"compilations":1})
            except BudgetExceeded:
                self._reconcile(reservation_id,{"local_configurations":1})
                termination = "global_budget"
                break
            compiled += 1
            try:
                validation = self.evaluator.validate(candidate,"search")
                if not validation.get("ok"):
                    observations.append(TuningObservation(parameters,candidate_id,"invalid",validation.get("status"),validation))
                    continue
                measurement = self.evaluator.benchmark(candidate)
                if measurement.get("status") not in {None,"measured","ok","valid"}:
                    observations.append(TuningObservation(parameters,candidate_id,"measurement_failed",measurement.get("status"),validation,measurement))
                    continue
                score = _latency_score(measurement)
                observations.append(TuningObservation(parameters,candidate_id,"measured",validation=validation,measurement=measurement))
                candidates[candidate_id] = candidate
                if score < best_score:
                    best,best_measurement,best_score = candidate,measurement,score
            except (TimeoutError,RuntimeError,ValueError) as error:
                observations.append(TuningObservation(parameters,candidate_id,"evaluation_failed",str(error)))
            finally:
                self._reconcile(reservation_id,{"local_configurations":1})
                self._reconcile(work_id,{"compilations":1})
        if best is None:
            return LocalSearchResult(None,None,observations,None,termination)
        # The selected minimum is selection-biased. Never publish it as a tree
        # observation: independently repeat the full common measurement path.
        fresh = self.evaluator.benchmark(best)
        if fresh.get("evaluation_id") == best_measurement.get("evaluation_id"):
            raise ValueError("Winner remesurement reused the selection evaluation_id")
        _latency_score(fresh)
        return LocalSearchResult(best,best_measurement,observations,fresh,termination)
