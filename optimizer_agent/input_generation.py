import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from openai import OpenAI

from agent_workload import InputCase
from state import OptimizerState
from use_harness import HarnessRuner


def load_input_cases(path: Path) -> list[InputCase]:
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Input case manifest must contain a nonempty JSON array")
    return [InputCase.from_dict(item, path.parent) for item in data]


def prepare_inputs(
    state: OptimizerState,
    *,
    client: OpenAI,
    runner: HarnessRuner,
) -> dict:
    workload = state["workload"]
    if workload.input_cases is not None:
        if not workload.input_cases:
            raise ValueError("Supply at least one input case, or None to generate cases")
        cases = [InputCase.from_dict(asdict(case), Path.cwd()) for case in workload.input_cases]
        return finish_inputs(state, cases)

    directory = state["session_dir"] / "inputs"
    print("Generating input cases", flush=True)
    directory.mkdir(parents=True, exist_ok=False)
    device = state["device_properties"]
    existing_sources = {}
    for name, request in (
        ("reference", workload.reference),
        ("initial", workload.initial_kernel),
    ):
        if request is not None:
            existing_sources[name] = Path(request).read_text(encoding="utf-8")

    prompt = f"""
Write a standalone Python script that prepares input cases for CUDA kernels.

Operation and input constraints:
{workload.description}

Required kernel signature, with arguments in this exact order:
{workload.signature}

Existing implementations, if supplied:
{json.dumps(existing_sources, indent=2)}

Target device properties:
{json.dumps(device, indent=2)}

Use only the Python standard library and NumPy. Generate files in the current
working directory; do not use the network, install packages, or launch programs.
Do not import this project or run CUDA. Do not modify files outside this directory.

Honor all specified shapes, sizes, layouts and valid value ranges. If sizes are
unspecified, include one small correctness/tail case and at least two meaningfully
large performance cases appropriate to this operation. Do not generate only toy
shapes. Choose enough independent output work for many blocks across all device
multiprocessors. Derive data types, dimensions, and their relationships from the
workload description, actual kernel signature, and supplied implementations.
Do not assume a particular operation, data layout, or set of dimensions.
Choose two distinct performance scales based on the device's multiprocessor
count, L2 capacity, available memory, and the operation's computational cost.
When measuring device-memory bandwidth, make the actively accessed working set
several times larger than L2, within the memory budget below. Do not inflate
unused allocations to achieve this. For compute-heavy workloads, size by
operation count and available parallel work as well as memory footprint.
Exercise distinct dimension sizes and boundary conditions where the interface
allows them, while respecting all relationships required by the operation.
Estimate reference-kernel work as well as bytes:
a simple direct reference must still finish promptly, particularly under WDDM.
Use at most min(512 MiB, 20% of free GPU memory) total buffers per generated case.
Keep generated files and runtime manageable. These are sizing guidelines, not
fixed shapes: explicit user sizes and supplied cases take precedence.
Explain chosen dimensions and expected parallel work in script comments. Use reproducible NumPy random generation if randomness is needed.
Generate nontrivial data appropriate to the operation, respecting mathematical
constraints such as valid indices or positive-definite matrices.

Write raw little-endian binary input files with no header and input_cases.json.
The manifest is a nonempty JSON array. Each case has four fields:
"arguments", "comparisons", "size_summary", and "purpose".
Set purpose to "correctness" for small correctness/tail cases and "performance"
for cases used to measure optimization gains. Include at least one performance
case. All cases must pass validation, but only performance cases affect ranking.
size_summary is a concise string
listing actual shapes, layouts, dtypes and important sizes for this workload;
it must agree with the scalar values and allocations. There are no launch configuration fields.

"arguments" is ordered exactly like the kernel parameters. Each item is either:
- Buffer: "kind": "buffer", "bytes": positive allocation size,
  "input_file": relative binary file path or null,
  "direction": "input", "output", or "inout".
  Input/inout buffers require files of exactly "bytes" bytes.
  Output-only buffers require null and will be zero-initialized by the harness.
- Scalar: "kind": "scalar", "type": one of int32, uint32, int64, uint64,
  float32, float64; "value": a JSON number of the appropriate type.

"comparisons" contains one entry for each output/inout buffer, numbered in output
order starting from zero (not parameter positions). Each entry has:
"output_index": integer, "dtype": one of float32, float64, int32, uint32, int64,
uint64, "absolute_tolerance": number, "relative_tolerance": number.
Choose meaningful tolerances based on the computation; integers compare exactly.
Do not compute reference outputs or silently choose huge tolerances to pass checks.
Scalar sizes, allocations, file contents and comparison dtypes must agree.

{state.get("research_brief", "")}

Return only Python source, without Markdown fences or explanations.
"""
    failure = ""
    for retry in range(state["config"].max_retries + 1):
        print(f"Input generation attempt {retry + 1}", flush=True)
        attempt_dir = directory / f"attempt-{retry}"
        attempt_dir.mkdir()
        (attempt_dir / "generation_prompt.txt").write_text(prompt, encoding="utf-8")
        response = client.responses.create(
            model=state["config"].model,
            reasoning={"effort": state["config"].reasoning_effort}, input=prompt)
        source = response.output_text.strip()
        script_path = attempt_dir / "generate_inputs.py"
        script_path.write_text(source, encoding="utf-8")
        log_path = attempt_dir / "generator.log"
        try:
            if not source:
                raise ValueError("The model returned an empty input generator")
            compile(source, str(script_path), "exec")
            # Local Python execution, not a sandbox.
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=attempt_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=120,
                )
            if process.returncode != 0:
                raise RuntimeError(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
            cases = load_input_cases(attempt_dir / "input_cases.json")
            return finish_inputs(state, cases)
        except (ValueError, TypeError, KeyError, OSError, SyntaxError, RuntimeError, subprocess.TimeoutExpired) as error:
            failure = f"{type(error).__name__}: {error}"
            (attempt_dir / "error.txt").write_text(failure, encoding="utf-8")
            prompt += f"\nThe previous script failed:\n{failure}\nPrevious source:\n{source}\nReturn a corrected complete Python script."
    raise RuntimeError(f"Input generation retries exhausted: {failure}")


def finish_inputs(state: OptimizerState, cases: list[InputCase]) -> dict:
    if not any(case.purpose == "performance" for case in cases):
        raise ValueError("At least one input case must have purpose: performance")
    for case in cases:
        total_bytes = sum(arg.bytes for arg in case.arguments if hasattr(arg, "bytes"))
        if total_bytes > state["device_properties"]["free_memory_bytes"]:
            raise ValueError("An input case exceeds available device memory")
    path = state["session_dir"] / "input_cases.json"
    path.write_text(json.dumps([asdict(case) for case in cases], indent=2), encoding="utf-8")
    return {"workload": replace(state["workload"], input_cases=cases)}
