"""Declarative constraints filter configurations before evaluator/GPU calls."""
from dataclasses import FrozenInstanceError
import pickle

import pytest

from optimizer_agent.autotuning import Autotuner
from optimizer_agent.codex_backend import validate_json_shape
from optimizer_agent.tuning_constraints import (
    CONSTRAINTS_SCHEMA,CompiledConstraint,ConstraintSpecificationError,compile_constraints,
)


def clause(op,parameters,values=()):
    return {"op":op,"parameters":list(parameters),"values":list(values)}


@pytest.mark.parametrize("declaration,accepted,rejected",[
    (clause("bounds",["P"],[8,32]),{"P":8},{"P":33}),
    (clause("bounds",["P"],[-8,0]),{"P":-4},{"P":1}),
    (clause("divisible",["P"],[8]),{"P":32},{"P":31}),
    (clause("divisible",["TILE","BLOCK"]),{"TILE":32,"BLOCK":8},{"TILE":30,"BLOCK":8}),
    (clause("divides",["VECTOR"],[64]),{"VECTOR":4},{"VECTOR":3}),
    (clause("divides",["BLOCK","TILE"]),{"TILE":32,"BLOCK":8},{"TILE":30,"BLOCK":8}),
    (clause("aligned",["STRIDE"],[16,4]),{"STRIDE":36},{"STRIDE":32}),
    (clause("forbid",["BLOCK","STAGES"],[256,8]),{"BLOCK":256,"STAGES":4},{"BLOCK":256,"STAGES":8}),
    (clause("product_le",["X","Y"],[1024]),{"X":32,"Y":32},{"X":64,"Y":32}),
    (clause("product_le",["TILE","TILE"],[1024]),{"TILE":32},{"TILE":33}),
    (clause("le",["P","Q"]),{"P":8,"Q":8},{"P":9,"Q":8}),
    (clause("lt",["P"],[8]),{"P":7},{"P":8}),
    (clause("eq",["P"],[8]),{"P":8},{"P":9}),
    (clause("ne",["P","Q"]),{"P":8,"Q":9},{"P":8,"Q":8}),
    (clause("ge",["P"],[8]),{"P":8},{"P":7}),
    (clause("gt",["P","Q"]),{"P":9,"Q":8},{"P":8,"Q":8}),
])
def test_controlled_integer_semantics(declaration,accepted,rejected):
    validate_json_shape([declaration],CONSTRAINTS_SCHEMA)
    restriction,=compile_constraints([declaration],set(accepted)|set(rejected))
    assert restriction(accepted) is True
    assert restriction(rejected) is False
    assert restriction.to_dict() == declaration


@pytest.mark.parametrize("bad",[
    {"op":"execute","parameters":["P"],"values":[1]},
    {"op":"bounds","parameters":["UNDECLARED"],"values":[1,2]},
    {"op":"bounds","parameters":["P.__class__"],"values":[1,2]},
    {"op":"bounds","parameters":["P"],"values":[1,2],"expression":"run()"},
    {"op":"bounds","parameters":["P"]},
    {"op":"bounds","parameters":"P","values":[1,2]},
    {"op":"bounds","parameters":["P"],"values":"[1,2]"},
    clause("bounds",["P"],[True,8]),
    clause("bounds",["P"],[1.0,8]),
    clause("bounds",["P"],["1",8]),
    clause("bounds",["P"],[None,8]),
    clause("bounds",["P"],[8,1]),
    clause("bounds",[],[1,8]),
    clause("bounds",["P","Q"],[1,8]),
    clause("divisible",["P"],[0]),
    clause("divisible",["P"],[-8]),
    clause("divisible",["P","Q"],[8]),
    clause("divides",["P"],[]),
    clause("aligned",["P"],[0,0]),
    clause("aligned",["P"],[16,16]),
    clause("aligned",["P"],[16,-1]),
    clause("forbid",["P","Q"],[8]),
    clause("forbid",["P","P"],[8,16]),
    clause("product_le",["P"],[]),
    clause("le",["P","Q","R"],[]),
    clause("eq",["P"],[1,2]),
])
def test_malformed_unknown_or_executable_declarations_rejected_before_search(bad):
    with pytest.raises(ConstraintSpecificationError,match=r"constraints\[0\]"):
        compile_constraints([bad],{"P","Q","R"})


@pytest.mark.parametrize("spec",[None,"P<8",{},(),["P<8"]])
def test_only_closed_json_arrays_are_accepted(spec):
    with pytest.raises(ConstraintSpecificationError):
        compile_constraints(spec,{"P"})


@pytest.mark.parametrize("allowed",["PARAM",None,["P","P"],["P;import os"],[False]])
def test_invalid_allowed_name_registry_is_rejected(allowed):
    with pytest.raises(ConstraintSpecificationError):
        compile_constraints([],allowed)


@pytest.mark.parametrize("bindings",[{},None,[],{"P":True},{"P":1.0},{"P":"8"}])
def test_missing_and_noninteger_runtime_values_fail_closed(bindings):
    restriction,=compile_constraints([clause("bounds",["P"],[1,16])],{"P"})
    assert restriction(bindings) is False


def test_zero_dynamic_divisors_are_filtered_without_division_exception():
    divisible,divides=compile_constraints([clause("divisible",["P","Q"]),clause("divides",["P","Q"])],{"P","Q"})
    assert divisible({"P":8,"Q":0}) is False
    assert divisible({"P":8,"Q":-2}) is False
    assert divides({"P":0,"Q":8}) is False
    assert divides({"P":-2,"Q":8}) is False


def test_compilation_freezes_original_declaration_and_supports_process_pickling():
    source=[clause("bounds",["P"],[8,32])]
    allowed={"P"}
    restriction,=compile_constraints(source,allowed)
    source[0]["parameters"][0]="OTHER"
    source[0]["values"][0]=1000
    allowed.clear()
    exported=restriction.to_dict()
    exported["values"][0]=2000
    assert restriction({"P":16}) is True
    assert pickle.loads(pickle.dumps(restriction)).to_dict() == clause("bounds",["P"],[8,32])
    with pytest.raises(FrozenInstanceError):
        restriction.op="gt"
    assert compile_constraints([],[]) == ()


def test_schema_rejects_unknown_properties_and_boolean_constants():
    invalid={**clause("bounds",["P"],[1,8]),"python":"P<=8"}
    with pytest.raises(ValueError,match="unexpected fields"):
        validate_json_shape([invalid],CONSTRAINTS_SCHEMA)
    with pytest.raises(ValueError,match="integer"):
        validate_json_shape([clause("bounds",["P"],[True,8])],CONSTRAINTS_SCHEMA)


def test_multiple_forbidden_combinations_are_all_enforced():
    rules=compile_constraints([clause("forbid",["P","Q"],[8,2]),clause("forbid",["P","Q"],[16,4])],{"P","Q"})
    assert not all(rule({"P":8,"Q":2}) for rule in rules)
    assert not all(rule({"P":16,"Q":4}) for rule in rules)
    assert all(rule({"P":8,"Q":4}) for rule in rules)


def test_real_autotuner_filters_constraints_before_any_evaluator_call(tmp_path):
    class Evaluator:
        def __init__(self):
            self.validations=[]
            self.measurements=[]
        def validate(self,candidate,suite):
            self.validations.append(dict(candidate["parameters"]))
            return {"ok":True}
        def benchmark(self,candidate):
            self.measurements.append(dict(candidate["parameters"]))
            return {"evaluation_id":"fresh-"+str(len(self.measurements)),
                    "cases":{"case":{"latency_ms":float(candidate["parameters"]["TILE"])}}}
    evaluator=Evaluator()
    def materialize(template,parameters):
        return {"candidate_id":str(sorted(parameters.items())),"parameters":parameters,
                "launch_rule":{"block":[32,1,1],"shared_memory_bytes":0}}
    constraints=[clause("bounds",["BLOCK"],[16,32]),clause("divisible",["TILE","BLOCK"]),
                 clause("aligned",["VECTOR"],[4,0]),clause("forbid",["BLOCK","VECTOR"],[32,4])]
    restrictions=compile_constraints(constraints,{"BLOCK","TILE","VECTOR"})
    tuner=Autotuner(evaluator,materialize=materialize,work_directory=tmp_path)
    result=tuner.search({}, {"BLOCK":[8,16,32],"TILE":[16,32],"VECTOR":[2,4]},
        {"configurations":12,"compilations":12},restrictions=restrictions)
    assert evaluator.validations == [{"BLOCK":16,"TILE":16,"VECTOR":4},{"BLOCK":16,"TILE":32,"VECTOR":4}]
    assert len(result.configurations) == 12
    assert sum(row.status == "filtered" for row in result.configurations) == 10
    assert len(evaluator.measurements) == 3  # Two valid configurations plus independent winner measurement.
    assert result.remeasurement["evaluation_id"] != result.evaluation["evaluation_id"]
    assert result.best_candidate["parameters"] == {"BLOCK":16,"TILE":16,"VECTOR":4}
