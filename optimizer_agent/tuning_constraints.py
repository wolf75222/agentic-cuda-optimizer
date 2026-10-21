"""Declarative integer restrictions for Autotuner, without expression evaluation.

Each clause is a closed JSON object with ``op``, ``parameters`` and ``values``.
The clauses are ANDed by Autotuner's existing restriction preflight. Examples::

    {"op":"bounds", "parameters":["BLOCK_X"], "values":[8,256]}
    {"op":"divisible", "parameters":["TILE","BLOCK_X"], "values":[]}
    {"op":"aligned", "parameters":["STRIDE_BYTES"], "values":[16,0]}
    {"op":"forbid", "parameters":["BLOCK_X","STAGES"], "values":[256,8]}

Single-parameter binary clauses use the integer in ``values`` as their right
operand. Two-parameter clauses use the second parameter, and ``values`` is
empty. ``divides(VEC, [N])`` means N % VEC == 0. ``aligned(p,[a,o])`` means
(p-o) % a == 0. Constants and parameter names are immutable after compilation.

These restrictions do not replace independent correctness validation or the
hardware limits already enforced by Autotuner. Workload-derived constants must
be frozen by the caller, rather than accepting expressions or metadata paths.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import operator
import re

CONSTRAINT_LANGUAGE_VERSION = 1
COMPARISONS = {"le":operator.le,"lt":operator.lt,"eq":operator.eq,
               "ne":operator.ne,"ge":operator.ge,"gt":operator.gt}
CONSTRAINT_OPERATORS = ("bounds","divisible","divides","aligned","forbid","product_le",*COMPARISONS)
CONSTRAINT_SCHEMA = {"type":"object","additionalProperties":False,
    "properties":{
        "op":{"type":"string","enum":list(CONSTRAINT_OPERATORS)},
        "parameters":{"type":"array","minItems":1,"items":{"type":"string","pattern":"^[A-Za-z_][A-Za-z0-9_]*$"}},
        "values":{"type":"array","items":{"type":"integer"}}},
    "required":["op","parameters","values"]}
CONSTRAINTS_SCHEMA = {"type":"array","items":CONSTRAINT_SCHEMA,"default":[]}
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ConstraintSpecificationError(ValueError):
    """The declaration is invalid; reject it before compiling or executing CUDA."""


def _signature(op,parameters,values):
    if op == "bounds":
        if len(parameters) != 1 or len(values) != 2 or values[0] > values[1]:
            raise ConstraintSpecificationError("bounds requires one parameter and [minimum, maximum] with minimum <= maximum")
    elif op == "aligned":
        if len(parameters) != 1 or len(values) != 2 or values[0] <= 0 or not 0 <= values[1] < values[0]:
            raise ConstraintSpecificationError("aligned requires one parameter and [positive alignment, offset in 0..alignment-1]")
    elif op == "forbid":
        if len(values) != len(parameters) or len(set(parameters)) != len(parameters):
            raise ConstraintSpecificationError("forbid requires distinct parameters and exactly one matching value per parameter")
    elif op == "product_le":
        if len(values) != 1:
            raise ConstraintSpecificationError("product_le requires one or more parameters and [maximum product]")
    else:
        if not ((len(parameters) == 1 and len(values) == 1) or (len(parameters) == 2 and not values)):
            raise ConstraintSpecificationError(op+" requires one parameter and [integer], or two parameters and []")
        if op == "divisible" and values and values[0] <= 0:
            raise ConstraintSpecificationError("divisible requires a positive constant divisor")


@dataclass(frozen=True)
class CompiledConstraint:
    """Immutable, pickleable callable accepted by ``Autotuner.search``."""
    op: str
    parameter_names: tuple[str,...]
    values: tuple[int,...]

    def __post_init__(self):
        if not isinstance(self.op,str) or self.op not in CONSTRAINT_OPERATORS:
            raise ConstraintSpecificationError("Unknown constraint operator")
        if (type(self.parameter_names) is not tuple or not self.parameter_names
                or any(not isinstance(name,str) or not _IDENTIFIER.fullmatch(name) for name in self.parameter_names)):
            raise ConstraintSpecificationError("Constraint parameters must be an immutable tuple of safe identifiers")
        if type(self.values) is not tuple or any(type(value) is not int for value in self.values):
            raise ConstraintSpecificationError("Constraint constants must be an immutable tuple of integers; booleans are not integers")
        _signature(self.op,self.parameter_names,self.values)

    def __call__(self,parameters):
        if not isinstance(parameters,Mapping):
            return False
        try:
            operands=tuple(parameters[name] for name in self.parameter_names)
        except (KeyError,TypeError):
            return False
        if any(type(value) is not int for value in operands):
            return False
        if self.op == "bounds":
            return self.values[0] <= operands[0] <= self.values[1]
        if self.op == "aligned":
            return (operands[0]-self.values[1]) % self.values[0] == 0
        if self.op == "forbid":
            return operands != self.values
        if self.op == "product_le":
            return math.prod(operands) <= self.values[0]
        left,right=operands if len(operands) == 2 else (operands[0],self.values[0])
        if self.op == "divisible":
            return right > 0 and left % right == 0
        if self.op == "divides":
            return left > 0 and right % left == 0
        return COMPARISONS[self.op](left,right)

    def to_dict(self):
        return {"op":self.op,"parameters":list(self.parameter_names),"values":list(self.values)}


def compile_constraints(spec,allowed_parameter_names):
    """Validate a closed JSON declaration and return pure boolean callables.

    Unknown references/operators, malformed arity and noninteger constants raise
    ValueError immediately. Missing or noninteger runtime bindings return False
    so Autotuner filters that configuration before its evaluator/GPU calls.
    An absent optional Proposal field should be supplied as ``[]`` by callers.
    """
    if isinstance(allowed_parameter_names,(str,bytes)):
        raise ConstraintSpecificationError("Allowed parameter names must be a collection of identifiers")
    try:
        names=tuple(allowed_parameter_names)
    except TypeError as error:
        raise ConstraintSpecificationError("Allowed parameter names must be a collection of identifiers") from error
    if any(not isinstance(name,str) or not _IDENTIFIER.fullmatch(name) for name in names):
        raise ConstraintSpecificationError("Unsafe allowed parameter identifier")
    if len(set(names)) != len(names):
        raise ConstraintSpecificationError("Duplicate allowed parameter identifier")
    allowed=frozenset(names)
    if type(spec) is not list:
        raise ConstraintSpecificationError("constraints must be a JSON array; use [] when absent")
    compiled=[]
    for index,clause in enumerate(spec):
        prefix="constraints["+str(index)+"]: "
        if type(clause) is not dict or set(clause) != {"op","parameters","values"}:
            raise ConstraintSpecificationError(prefix+"expected exactly op, parameters and values")
        if not isinstance(clause["op"],str) or clause["op"] not in CONSTRAINT_OPERATORS:
            raise ConstraintSpecificationError(prefix+"unknown operator "+repr(clause["op"]))
        if type(clause["parameters"]) is not list or type(clause["values"]) is not list:
            raise ConstraintSpecificationError(prefix+"parameters and values must be JSON arrays")
        if any(not isinstance(name,str) or not _IDENTIFIER.fullmatch(name) for name in clause["parameters"]):
            raise ConstraintSpecificationError(prefix+"unsafe parameter identifier")
        unknown=set(clause["parameters"])-allowed
        if unknown:
            raise ConstraintSpecificationError(prefix+"unknown parameter reference(s): "+", ".join(sorted(unknown)))
        try:
            compiled.append(CompiledConstraint(clause["op"],tuple(clause["parameters"]),tuple(clause["values"])))
        except ConstraintSpecificationError as error:
            raise ConstraintSpecificationError(prefix+str(error)) from error
    return tuple(compiled)
