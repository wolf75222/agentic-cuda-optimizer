from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import math
import re
import numpy as np
from use_harness import BufferArgument, BufferDirection, ScalarArgument, parse_arguments

@dataclass
class InputCase:
    arguments: list[BufferArgument | ScalarArgument]
    comparisons: list["OutputComparison"]
    size_summary: str = ""
    purpose: Literal["correctness", "performance"] = "performance"

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> "InputCase":
        arguments = parse_arguments(data["arguments"])
        comparisons = [OutputComparison(**item) for item in data["comparisons"]]
        outputs = []

        for argument in arguments:
            if not isinstance(argument, BufferArgument):
                scalar_types = {"int32", "uint32", "int64", "uint64", "float32", "float64"}
                if argument.type not in scalar_types:
                    raise ValueError(f"Unsupported scalar type: {argument.type}")
                dtype = np.dtype(argument.type)
                if dtype.kind in "iu":
                    limits = np.iinfo(dtype)
                    if type(argument.value) is not int or not limits.min <= argument.value <= limits.max:
                        raise ValueError(f"Scalar value is outside {argument.type}")
                elif type(argument.value) not in (int, float) or not math.isfinite(argument.value) or abs(argument.value) > np.finfo(dtype).max:
                    raise ValueError(f"Scalar value is outside {argument.type}")
                continue
            if type(argument.bytes) is not int or argument.bytes <= 0:
                raise ValueError("Buffer bytes must be a positive integer")
            if argument.direction == BufferDirection.Output:
                if argument.input_file is not None:
                    raise ValueError("Output buffers require input_file: null")
            else:
                if not argument.input_file:
                    raise ValueError("Input and inout buffers require an input file")
                path = (base_dir / argument.input_file).resolve()
                if not path.is_file():
                    raise ValueError(f"Input file does not exist: {path}")
                if path.stat().st_size != argument.bytes:
                    raise ValueError(f"Input file size differs from buffer bytes: {path}")
                argument.input_file = str(path)
            if argument.direction != BufferDirection.Input:
                outputs.append(argument)

        if any(type(c.output_index) is not int for c in comparisons):
            raise ValueError("Output indices must be integers")
        if not comparisons or sorted(c.output_index for c in comparisons) != list(range(len(outputs))):
            raise ValueError("Each output must have exactly one comparison rule")
        for comparison in comparisons:
            if comparison.dtype not in {"int32", "uint32", "int64", "uint64", "float32", "float64"}:
                raise ValueError(f"Unsupported output dtype: {comparison.dtype}")
            if outputs[comparison.output_index].bytes % np.dtype(comparison.dtype).itemsize:
                raise ValueError("Output allocation contains an incomplete element")
            for tolerance in (comparison.absolute_tolerance, comparison.relative_tolerance):
                if tolerance is not None and (type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0):
                    raise ValueError("Comparison tolerances must be finite and nonnegative")

        size_summary = data.get("size_summary", "")
        if not isinstance(size_summary, str):
            raise ValueError("size_summary must be a string")
        purpose = data.get("purpose", "performance")
        if purpose not in ("correctness", "performance"):
            raise ValueError("Case purpose must be correctness or performance")
        return cls(arguments=arguments, comparisons=comparisons, size_summary=size_summary, purpose=purpose)
    
@dataclass
class OutputComparison:
    output_index: int
    dtype: str
    absolute_tolerance: float | None
    relative_tolerance: float | None
    def read_output(self, report: dict) -> np.ndarray:
        supported = {
            "float32", "float64",
            "int32", "uint32",
            "int64", "uint64",
        }

        if self.dtype not in supported:
            raise ValueError(f"Unsupported output type: {self.dtype}")

        outputs = report["outputs"]

        if not 0 <= self.output_index < len(outputs):
            raise ValueError(f"Output index out of range: {self.output_index}")

        output = outputs[self.output_index]
        path = Path(output["path"])
        dtype = np.dtype(self.dtype).newbyteorder("<")
        byte_count = path.stat().st_size

        if byte_count != output["bytes"]:
            raise ValueError("Output file size differs from the harness report")

        if byte_count % dtype.itemsize != 0:
            raise ValueError("Output contains an incomplete element")

        return np.fromfile(path, dtype=dtype)
    
    def compare(self, reference_report: dict, candidate_report: dict) -> dict:
        expected = self.read_output(reference_report)
        actual = self.read_output(candidate_report)

        if len(expected) != len(actual):
            return {
                "ok": False,
                "error": "Output lengths differ",
                "expected_count": len(expected),
                "actual_count": len(actual),
            }

        atol = (
            0.0 if self.absolute_tolerance is None
            else self.absolute_tolerance
        )
        rtol = (
            0.0 if self.relative_tolerance is None
            else self.relative_tolerance
        )

        if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
            raise ValueError("Tolerances must be finite and nonnegative")

        is_float = self.dtype in ("float32", "float64")

        if is_float:
            # Compare float32 values using float64 arithmetic.
            expected = expected.astype(np.float64, copy=False)
            actual = actual.astype(np.float64, copy=False)

            with np.errstate(over="ignore", invalid="ignore"):
                matches = (
                    np.isfinite(expected)
                    & np.isfinite(actual)
                    & np.isclose(
                        actual,
                        expected,
                        atol=atol,
                        rtol=rtol,
                        equal_nan=False,
                    )
                )
        else:
            matches = actual == expected

        mismatches = ~matches
        mismatch_count = int(np.count_nonzero(mismatches))

        return {
            "ok": mismatch_count == 0,
            "mismatch_count": mismatch_count,
        }

@dataclass
class Workload:
    description: str
    signature: str | None = None
    input_cases: list[InputCase] | None = None
    reference: Path | None = None
    initial_kernel: Path | None = None

    @property
    def kernel(self) -> str:
        if self.signature is None:
            raise ValueError("Kernel signature has not been resolved yet")
        match = re.search(r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", self.signature)
        if match is None:
            raise ValueError("Expected a CUDA kernel signature containing 'void name(...)'")
        return match.group(1)
