import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from openai import OpenAI

from agent_workload import InputCase
from kernel_proposal import (
    KernelProposal, LaunchConfig, LaunchPlan, StopOptimization,
    optimization_decision, validate_launches,
)
from optimization_history import (
    CaseRun, OptimizationAttempt, attempt_time, geometric_mean_time, geometric_mean_speedup,
    experiment_history,
)
from plot_history import plot_history
from nsight_profiler import profile_overview
from pmpp_guidance import PMPP_GUIDANCE
from state import OptimizerState
from use_harness import HarnessRuner, KernelRequest


def new_conversation(prompt: str) -> dict:
    return {
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(content=prompt)],
        "kernel_proposal": None,
        "proposal_retries": 0,
        "model_calls": 0,
    }


def validate_hardware_claims(state: OptimizerState, hypothesis) -> None:
    available = set()
    profiles = {}
    if state["config"].use_nsight:
        for path in sorted((state["session_dir"] / "profiles").glob("*/summary.json")):
            profile = json.loads(path.read_text(encoding="utf-8"))
            profiles[(profile["iteration"], profile["case_index"])] = profile_overview(profile)
    reports = list(profiles.values())
    for message in state["messages"]:
        if isinstance(message, ToolMessage) and message.name in ("read_profile_metrics", "profile_kernel"):
            if isinstance(message.content, str):
                try:
                    reports.append(json.loads(message.content))
                except json.JSONDecodeError:
                    continue
    for report in reports:
        if not isinstance(report, dict) or not report.get("ok"):
            continue
        for metric in report.get("metrics", []):
            if isinstance(metric, dict):
                available.add((report.get("iteration"), report.get("case_index"),
                               metric["name"], metric["value"], metric.get("unit", "")))
    for claim in hypothesis.hardware_claims:
        if not claim.claim.strip() or not claim.interpretation.strip():
            raise ValueError("Hardware claims need a description and interpretation")
        if claim.status == "measured" and not claim.counters:
            raise ValueError("Measured hardware claims require counter evidence; retrieve it or mark unverified")
        for counter in claim.counters:
            citation = (counter.iteration, counter.case_index, counter.metric_name, counter.value, counter.unit)
            if citation not in available:
                raise ValueError(
                    f"Counter evidence {counter.metric_name} for iteration {counter.iteration}, case "
                    f"{counter.case_index} was not received with that value/unit. Use read_profile_metrics "
                    "to retrieve it, copy the returned strings exactly, or remove the citation and mark unverified."
                )


def prepare_signature(state: OptimizerState, *, client: OpenAI) -> dict:
    workload = state["workload"]
    if workload.signature is not None:
        return {}

    sources = {}
    for name in ("reference", "initial_kernel"):
        path = getattr(workload, name)
        if path is not None:
            sources[name] = Path(path).read_text(encoding="utf-8")

    prompt = f"""
Choose the CUDA kernel signature once for this workload:
{workload.description}

Supplied implementations:
{json.dumps(sources, indent=2)}

Supplied input cases, in parameter order:
{json.dumps([asdict(case) for case in workload.input_cases], indent=2) if workload.input_cases is not None else "None"}

If an implementation is supplied, extract its entry signature without changing
the interface. Otherwise design one simple interface for the described operation.
Respect supplied input case argument order, buffer directions and scalar types.
Use independent pointers to basic numeric types and basic numeric scalar parameters.
Use descriptive parameter names so their meaning and array dimensions are clear.
Return only one declaration: extern "C" __global__ void name(...);
No body, Markdown, templates, overloads, or explanations.
This interface will be fixed for every reference and optimization candidate.
"""
    print("Resolving kernel signature", flush=True)
    for retry in range(state["config"].max_retries + 1):
        response = client.responses.create(
            model=state["config"].model,
            reasoning={"effort": state["config"].reasoning_effort}, input=prompt)
        signature = response.output_text.strip().rstrip(";").strip()
        try:
            resolved = replace(workload, signature=signature)
            _ = resolved.kernel
            if not signature.endswith(")") or any(ch in signature for ch in "{};`"):
                raise ValueError("Return only the kernel declaration")
        except ValueError as error:
            prompt += f"\nInvalid response: {signature}\nCorrection: {error}"
            continue
        (state["session_dir"] / "signature.cuh").write_text(signature + ";\n", encoding="utf-8")
        return {"workload": resolved}
    raise ValueError("Could not resolve the kernel signature")


def propose(state: OptimizerState, *, repair: bool = False) -> dict:
    workload = state["workload"]
    best = state["best_attempt"]
    latest = state["optimization_history"][-1] if state["optimization_history"] else None
    task = (
        f"REPAIR REQUIRED: candidate {latest.iteration} failed validation or execution. "
        "Fix that candidate using its compiler/execution/comparison diagnostics. "
        "Do not start another optimization experiment or request early stopping. "
        "Set hypothesis.starting_iteration to the failed candidate. Explain the bug "
        "and correction; hardware_claims may be empty for a correctness repair."
        if repair and latest is not None else
        "Decide whether there is a worthwhile next optimization experiment. "
        "If so, propose it while preserving behavior."
    )
    source = latest.kernel_source if repair and latest is not None else best.kernel_source if best else state["kernel_source"]
    profiling_guidance = ""
    if state["config"].use_nsight:
        profiles = {}
        for path in sorted((state["session_dir"] / "profiles").glob("*/summary.json")):
            profile = json.loads(path.read_text(encoding="utf-8"))
            profiles[(profile["iteration"], profile["case_index"])] = profile_overview(profile)
        profiling_guidance = f"""
Previously collected Nsight overviews (latest report for each candidate/case):
{json.dumps(list(profiles.values()), indent=2)}
The graph automatically profiles the largest input case of each valid candidate,
including the initial kernel, before the next proposal. Read those measurements
and connect your proposed change to the observed bottleneck. You may additionally
use profile_kernel(iteration, case_index) to investigate another executed case.
The overview contains only a small selection of counters. Use read_profile_metrics
to retrieve any already-collected metric by exact name or wildcard pattern, such
as '*bank_conflicts*' or 'smsp__warp_issue_stalled*'. Discover available names using
metric_pattern='*', names_only=true; follow next_offset for more matches. This reads
saved reports without executing GPU work. An omitted or unmatched counter is not
zero. Inspect relevant counters before making claims unsupported by the overview.
It returns Nsight counters for an isolated replay, not benchmark scores.
Nsight gpu__time_duration metrics measure a separate profiling replay, with the
reported metric units and profiling conditions. Never substitute these for
kernel_latency_ms, mix them with benchmark timings, or use them to rank kernels.
Use the evidence to choose a PMPP technique; do not equate high DRAM traffic with
useful bandwidth. Profiling is unavailable for an unexecuted proposal. If the tool
reports missing Nsight or denied counters, continue without profiling and do not
repeat that same environmental failure.
"""
    prompt = f"""
{task}
Operation: {workload.description}
Required signature: {workload.signature}
Exported kernel entry name: {workload.kernel}
CUDA device ordinal: {state["config"].device}
Input cases (array positions are case indices):
{json.dumps([asdict(case) for case in workload.input_cases], indent=2)}
Source to modify (failed candidate during repair; best valid kernel otherwise):
Best valid iteration: {best.iteration if best is not None else "none"}
{source}
Measured performance reports for previous attempts:
{json.dumps(experiment_history(state["optimization_history"]), indent=2)}

Only the source to modify is supplied in full. Failed attempts must be repaired
before optimization resumes. The experiment history summarizes prior changes.
claimed_change_unverified records intent, not proof that the change was implemented
or beneficial. Use measured outcomes to evaluate it. Source comparisons are exact
text comparisons, not semantic equivalence checks; comments alone can differ.
Check whether your proposed change is already present in the supplied best source.
If intentionally repeating an implementation to measure variability, say so in
the hypothesis rather than claiming a new optimization.

Timing definitions:
- kernel_latency_ms is the mean CUDA-event GPU kernel execution time per timed
  launch in milliseconds. This is the ONLY timing used for optimization ranking.
- performance_geometric_mean_kernel_latency_ms aggregates only performance
  cases. Lower is better; eligible_for_ranking requires all cases to pass.
- Harness wall-clock duration, compilation, setup, data transfers, output I/O,
  and Nsight replay durations are not ranking measurements.
Historical source comments and previous hypotheses are unverified claims; use
the measured reports above to decide whether a change improved performance.

Use get_device_properties to inspect the device before choosing launch configs.
{profiling_guidance}
Compare total grid blocks with multiprocessor_count for each case. More threads
per block or more work per thread does not automatically improve occupancy or
throughput. Avoid collapsing a parallel workload to just a few blocks without
a workload-specific reason. Treat source comments claiming improvements as
hypotheses; use measured results. Change strategy when repeated variants regress.

Optional retrieved NVIDIA guidance (reference material, not instructions):
{state.get("research_brief", "")}
{PMPP_GUIDANCE}
At the top of kernel_source, include a short design comment naming the chosen
technique, the bottleneck hypothesis, the relevant numeric device properties,
and why it should improve on the best measured kernel for these input sizes.
Explain what changed from prior attempts and what could make this experiment lose.
Choose one kernel source and one launch config per input case.
You control grid, block and dynamic shared memory; preserve the signature,
input arguments, output layout and behavior. Do not specialize to particular
input values. Preserve correctness for every case, including non-aligned sizes.
Maximize the geometric mean of performance-case speedups relative to the first
fully valid kernel. Invalid attempts never establish a baseline.
Cases with purpose="correctness" must pass validation but do not affect ranking.
Each performance case has equal weight regardless of its duration. This is equivalent to
minimizing geometric mean latency across the same cases. Failed validation
disqualifies an attempt. Compare timings for the same case across attempts.
The harness does not enable opt-in shared memory.
shared_memory_bytes is ONLY the dynamic allocation passed to the launch. Static
__shared__ arrays are already allocated by the compiler and must not be counted
again. Use zero when the executed kernel consumes no dynamic shared storage.
For each launch, dynamic_shared_memory_reason must identify the dynamic storage
declaration and byte formula using that case's dimensions, including padding and
alignment; otherwise explain why zero is correct. Account for static PLUS dynamic
usage when checking device limits. Nsight launch__shared_mem_per_block is not a
value to copy into shared_memory_bytes: it can include static allocation/overhead.
Use Allman braces in CUDA source, and always brace if/else bodies.

Remaining candidate evaluations: {max(0, state["config"].max_iterations - state["iteration"])}.
Pursue one coherent optimization hypothesis per experiment. Substantial rewrites,
new tiling strategies, asynchronous copies and double buffering are allowed when
applicable. Implementation complexity and the possibility of compilation,
validation or performance failure are not reasons to stop. Those outcomes are
part of experimentation; the best valid kernel remains available. Preserve the
workload's numerical requirements; do not silently introduce reduced precision.
Outside repair mode, you may stop early once a valid kernel exists if you cannot identify a worthwhile
new experiment within the workload constraints, or further attempts would only
repeat exhausted ideas. A few regressing variants do not exhaust other strategies.
If a promising approach remains, investigate or attempt it with the remaining
evaluations rather than deferring it merely because it is invasive or uncertain.
When bottleneck uncertainty prevents choosing an experiment and profiling is
available, use read_profile_metrics to inspect relevant collected details before
deciding to stop. If needed counters are unavailable, acknowledge the limitation.
Return {{"action": "stop", "reason": "..."}} instead
of a kernel proposal. Explain the evidence and remaining uncertainty concisely.
This means no justified next experiment within this search, not proof that the
kernel is globally optimal. Do not manufacture changes to exhaust the iteration
budget. The pipeline will retain and export the best validated kernel.
If no candidate has passed validation, repair correctness instead of stopping.

When proposing a kernel, include a non-null structured hypothesis:
- change_summary: two to eight words naming the actual change (for example,
  "Transpose shared tile layout"). Describe the modification, not an unmeasured
  improvement. For a repair, briefly name the bug fixed.
- starting_iteration: the executed kernel you are modifying. Prefer the best
  valid kernel; explain the reason in proposed_change if exploring another one.
- supporting_measurements: cite kernel_latency_ms by that exact name for benchmark
  timing evidence. Identify iteration and case for each timing or counter
  cited, with its measured value and unit. Use only supplied results or tool data.
  If profiling is unavailable, use timings/validation results and explicitly state
  which bottleneck remains uncertain. Do not invent counters or infer a bottleneck
  from one metric alone. Address evidence that contradicts your hypothesis.
- proposed_change: one coherent optimization experiment, which may require a
  substantial rewrite. Coherent does not mean a small code diff or micro-change.
  Explain why this experiment differs from previously unsuccessful attempts.
- expected_benefit: which cases and counters should improve and why, with the
  main tradeoff that could make it slower. This is a prediction to measure.
- case_execution_paths: one entry for EVERY input case. Name the path taken by
  the proposed kernel, evaluate its relevant branch conditions using actual scalar
  arguments and launch dimensions, and state whether it executes the changed code.
  Account for interior/boundary blocks if different parts of the case take different
  paths. For an unconditional kernel, explicitly say there is one path.
  Do not claim a runtime benefit from optimizing a fallback that these cases never
  execute. If investigating an indirect compiler/resource effect instead, state
  that explicitly and identify the resource measurement needed to evaluate it.
- hardware_claims: list EVERY hardware-bottleneck attribution in your explanation
  (including source comments), such as bank conflicts, occupancy limits, spills,
  bandwidth saturation or cache misses. Use [] if there are no such claims.
  Mark a claim measured only with relevant counter evidence from the supplied
  overview or a tool response. Cite iteration, case_index, metric_name, value and
  unit, copying metric strings exactly. Explain why the counters support the claim
  and any contradictory evidence in interpretation; a counter alone is not proof
  of causality. If relevant detail is absent, use read_profile_metrics to inspect
  it before calling the claim measured. Bank-conflict claims require relevant
  shared-memory/bank-conflict evidence, not merely kernel timings or DRAM traffic.
  If counters were not collected or profiling is disabled, mark the claim
  unverified, leave counters empty, and explain the uncertainty and how to test it.
  Kernel latency proves an observed timing difference, not a hardware bottleneck.
High occupancy does not prove good throughput, but it contradicts an unsupported
claim that too few active warps is the bottleneck. More grid blocks alone does not
prove improved occupancy. Relate those claims to actual measurements.
High aggregate sm__throughput alone does not establish peak useful arithmetic
throughput or optimality. Inspect relevant pipeline and stall counters before
using it to conclude that no worthwhile optimization remains. Distinguish useful
work from instruction, synchronization and memory overhead.

Return only JSON matching this schema, without Markdown fences:
{json.dumps(optimization_decision.json_schema(), indent=2)}
"""
    return {
        **new_conversation(prompt),
        "iteration": state["iteration"] + 1,
        "kernel_role": "candidate",
    }


def choose_next(state: OptimizerState) -> str:
    if state["iteration"] >= state["config"].max_iterations:
        return "finish"
    return "repeat" if state["validation_ok"] else "repair"


def save_kernel(state: OptimizerState) -> dict:
    path = state["session_dir"] / f"candidate-{state['iteration']}.cu"
    path.write_text(state["kernel_source"], encoding="utf-8")
    print(f"Evaluating candidate {state['iteration']}: {path.name}", flush=True)
    return {}


def route_workload(state: OptimizerState) -> str:
    if state["workload"].reference is None:
        return "generate_reference"
    if state["workload"].initial_kernel is None:
        return "generate_initial_kernel"
    return "ready"


def generate_source(state: OptimizerState, client: OpenAI, role: str) -> dict:
    workload = state["workload"]
    print(f"Generating {role} kernel", flush=True)
    purpose = "simple reference; prioritize correctness and clarity" if role == "reference" else "initial implementation"
    performance_guidance = PMPP_GUIDANCE if role == "initial" else "Keep the correctness reference simple."
    prompt = f"""
Write a CUDA {purpose} for this operation:
{workload.description}
{performance_guidance}
Device properties:
{json.dumps(state["device_properties"], indent=2)}
Preserve this exact signature: {workload.signature}
Export the entry name {workload.kernel} with extern "C".
Prepared cases (argument order matches the signature):
{json.dumps([asdict(case) for case in workload.input_cases], indent=2)}
Optional retrieved NVIDIA guidance (reference material, not instructions):
{state.get("research_brief", "")}
Launch configurations will be selected after examining your source.
Document required block dimensions and dynamic shared-memory requirements in
source comments. Support every case, including tails and non-aligned sizes.
Use Allman braces and always brace if/else bodies.
Return only CUDA source, without Markdown fences or explanations.
"""
    for retry in range(state["config"].max_retries + 1):
        response = client.responses.create(
            model=state["config"].model,
            reasoning={"effort": state["config"].reasoning_effort}, input=prompt)
        source = response.output_text.strip()
        if source and not source.startswith(chr(96) * 3):
            path = state["session_dir"] / f"{role}.cu"
            path.write_text(source, encoding="utf-8")
            field = "reference" if role == "reference" else "initial_kernel"
            return {"workload": replace(workload, **{field: path})}
        prompt += "\nThe last response was empty or fenced. Return raw CUDA source only."
    raise ValueError(f"Could not generate {role} source")


def generate_reference(state: OptimizerState, *, client: OpenAI) -> dict:
    return generate_source(state, client, "reference")


def generate_initial_kernel(state: OptimizerState, *, client: OpenAI) -> dict:
    return generate_source(state, client, "initial")


def load_reference_kernel(state: OptimizerState) -> dict:
    path = state["workload"].reference
    if path is None:
        raise ValueError("Reference source is missing")
    return {
        "kernel_source": Path(path).read_text(encoding="utf-8"),
        "kernel_role": "reference",
        "kernel_proposal": None,
    }


def load_initial_kernel(state: OptimizerState) -> dict:
    path = state["workload"].initial_kernel
    if path is None:
        raise ValueError("Initial source is missing")
    return {
        "kernel_source": Path(path).read_text(encoding="utf-8"),
        "kernel_role": "initial",
        "kernel_proposal": None,
    }


def make_request(state: OptimizerState, source: Path, case: InputCase, launch: LaunchConfig, *, reference: bool) -> KernelRequest:
    config = state["config"]
    return KernelRequest(
        source=str(source.resolve()),
        kernel=state["workload"].kernel,
        mode="benchmark",
        device=config.device,
        grid=list(launch.grid),
        block=list(launch.block),
        shared_memory_bytes=launch.shared_memory_bytes,
        warmup=config.warmup,
        iterations=config.iterations,
        arguments=deepcopy(case.arguments),
    )


def execute_reference(state: OptimizerState, *, runner: HarnessRuner) -> dict:
    workload = state["workload"]
    print("Benchmarking reference cases", flush=True)
    proposal = state["kernel_proposal"]
    if proposal is None or workload.reference is None:
        raise ValueError("Reference kernel and launches must be prepared")
    launches = {launch.case_index: launch.config for launch in proposal.launches}
    results = []
    case_runs = []
    for index, case in enumerate(workload.input_cases):
        request = make_request(state, workload.reference, case, launches[index], reference=True)
        results.append(runner.run(
            run_id=f"reference-case-{index}",
            request=request,
        ))
        case_runs.append(CaseRun(index, request, results[-1], [], bool(results[-1].get("ok")),
                                 size_summary=case.size_summary, purpose=case.purpose))
    reference_attempt = OptimizationAttempt(
        iteration=-1, kernel_source=state["kernel_source"], case_runs=case_runs,
        change_summary="Correctness reference",
    )
    (state["session_dir"] / "reference-benchmark.json").write_text(
        json.dumps(asdict(reference_attempt), indent=2), encoding="utf-8",
    )
    failures = [result.get("error", "Reference execution failed") for result in results if not result["ok"]]
    return {
        "reference_results": results,
        "reference_attempt": reference_attempt,
        "error": "\n".join(failures) if failures else None,
    }


def route_reference_result(state: OptimizerState) -> str:
    return "finish" if state["error"] else "continue"


def compare_outputs(input_case: InputCase, reference: dict, candidate: dict) -> dict:
    if not reference["ok"] or not candidate["ok"]:
        return {
            "validation_ok": False,
            "comparison_results": [{"ok": False, "error": candidate.get("error", "Kernel execution failed")}],
        }
    if len(candidate["outputs"]) != len(reference["outputs"]):
        return {
            "validation_ok": False,
            "comparison_results": [{"ok": False, "error": "Output counts differ"}],
        }
    comparisons = input_case.comparisons
    if not comparisons or sorted(c.output_index for c in comparisons) != list(range(len(reference["outputs"]))):
        raise ValueError("Each output needs exactly one comparison rule")
    results = []
    for comparison in comparisons:
        try:
            report = comparison.compare(reference, candidate)
        except (OSError, ValueError) as error:
            report = {"ok": False, "error": str(error)}
        results.append({"output_index": comparison.output_index, **report})
    return {"validation_ok": all(report["ok"] for report in results), "comparison_results": results}


def evaluate_candidate(state: OptimizerState, *, runner: HarnessRuner) -> dict:
    cases = state["workload"].input_cases
    proposal = state["kernel_proposal"]
    if not cases or proposal is None:
        raise ValueError("Candidate and input cases must be prepared")
    if len(state["reference_results"]) != len(cases):
        raise ValueError("Each input case needs a reference result")
    launches = {launch.case_index: launch.config for launch in proposal.launches}
    source = state["session_dir"] / f"candidate-{state['iteration']}.cu"
    case_runs = []
    for index, case in enumerate(cases):
        request = make_request(state, source, case, launches[index], reference=False)
        result = runner.run(
            run_id=f"candidate-{state['iteration']}-case-{index}",
            request=request,
        )
        comparison = compare_outputs(case, state["reference_results"][index], result)
        case_runs.append(CaseRun(
            case_index=index,
            request=request,
            harness_result=result,
            comparison_result=comparison["comparison_results"],
            validation_ok=comparison["validation_ok"],
            size_summary=case.size_summary,
            purpose=case.purpose,
        ))
    attempt = OptimizationAttempt(
        iteration=state["iteration"],
        kernel_source=state["kernel_source"],
        case_runs=case_runs,
        hypothesis=proposal.hypothesis.model_dump() if proposal.hypothesis is not None else None,
        change_summary=proposal.hypothesis.change_summary if proposal.hypothesis else "Initial implementation",
    )
    history = [*state["optimization_history"], attempt]
    (state["session_dir"] / "history.json").write_text(
        json.dumps([asdict(entry) for entry in history], indent=2), encoding="utf-8",
    )
    expected_indices = list(range(len(cases)))
    eligible = [entry for entry in history
                if geometric_mean_time(entry) is not None
                and sorted(case.case_index for case in entry.case_runs) == expected_indices]
    # The fixed baseline cancels when ranking geometric mean speedups.
    best = min(eligible, key=geometric_mean_time) if eligible else None
    valid = all(case.validation_ok for case in case_runs)
    print(f"Candidate {attempt.iteration}: {attempt.change_summary}; valid={valid}, geometric_mean_ms={geometric_mean_time(attempt)}", flush=True)
    return {
        "optimization_history": history,
        "validation_ok": valid,
        "best_attempt": best,
    }


def read_kernel_proposal(state: OptimizerState) -> dict:
    try:
        response = state["messages"][-1]
        if not isinstance(response.content, str):
            raise ValueError("Expected a JSON text response")
        if state["kernel_role"] == "candidate":
            proposal = optimization_decision.validate_json(response.content, strict=True)
            if isinstance(proposal, StopOptimization):
                if not state["validation_ok"]:
                    raise ValueError("The latest candidate failed; repair it instead of stopping")
                if not proposal.reason.strip():
                    raise ValueError("Stopping requires a nonempty reason")
                if state["best_attempt"] is None:
                    raise ValueError("No valid candidate exists yet; propose a correctness repair before stopping")
                print(f"Agent stopped optimization: {proposal.reason.strip()}", flush=True)
                return {"stop_reason": proposal.reason.strip(), "kernel_proposal": None}
            hypothesis = proposal.hypothesis
            if hypothesis is None:
                raise ValueError("Optimization candidates require a hypothesis")
            if not 2 <= len(hypothesis.change_summary.split()) <= 8:
                raise ValueError("change_summary must contain two to eight words")
            if hypothesis.starting_iteration not in {a.iteration for a in state["optimization_history"]}:
                raise ValueError("hypothesis.starting_iteration must identify an executed kernel")
            if not state["validation_ok"] and hypothesis.starting_iteration != state["optimization_history"][-1].iteration:
                raise ValueError("Repair must start from the latest failed candidate")
            if (not hypothesis.supporting_measurements
                    or any(not item.strip() for item in hypothesis.supporting_measurements)
                    or not hypothesis.proposed_change.strip() or not hypothesis.expected_benefit.strip()):
                raise ValueError("Hypothesis measurements, proposed change and expected benefit must be nonempty")
            paths = hypothesis.case_execution_paths
            if sorted(path.case_index for path in paths) != list(range(len(state["workload"].input_cases))):
                raise ValueError("Provide exactly one case_execution_paths entry per input case")
            if any(not path.active_path.strip() or not path.condition_evaluation.strip() for path in paths):
                raise ValueError("Each execution path needs a name and evaluated branch conditions")
            validate_hardware_claims(state, hypothesis)
        else:
            plan = LaunchPlan.model_validate_json(response.content, strict=True)
            proposal = KernelProposal(kernel_source=state["kernel_source"], launches=plan.launches)
        if not proposal.kernel_source.strip():
            raise ValueError("Kernel source is empty")
        validate_launches(proposal.launches, len(state["workload"].input_cases), state["device_properties"])
    except (ValueError, TypeError) as error:
        if state["proposal_retries"] >= state["config"].max_retries:
            return {"error": f"Proposal retries exhausted: {error}", "kernel_proposal": None}
        return {
            "kernel_proposal": None,
            "proposal_retries": state["proposal_retries"] + 1,
            "messages": [HumanMessage(content=f"The response or launch configuration is invalid:\n{error}\nReturn corrected JSON using the requested schema and device limits.")],
        }
    return {"kernel_proposal": proposal, "kernel_source": proposal.kernel_source}


def route_kernel_proposal(state: OptimizerState) -> str:
    if state["error"] or state.get("stop_reason"):
        return "finish"
    if state["kernel_proposal"] is None:
        return "retry"
    return "reference" if state["kernel_role"] == "reference" else "candidate"


def prepare_launches(state: OptimizerState) -> dict:
    workload = state["workload"]
    print(f"Planning launches for {state['kernel_role']} kernel", flush=True)
    prompt = f"""
Prepare launch configurations for this existing CUDA kernel.
Operation: {workload.description}
Device ordinal: {state["config"].device}
Kernel source:
{state["kernel_source"]}
Input cases (array positions are case indices):
{json.dumps([asdict(case) for case in workload.input_cases], indent=2)}
Use get_device_properties to inspect the device.
{PMPP_GUIDANCE}
Apply this guidance to launch choices only; do not redesign the stored source.
Compare grid block count to SM count; do not reduce available parallelism merely
to increase work per thread.
Optional retrieved NVIDIA guidance (reference material, not instructions):
{state.get("research_brief", "")}
Choose compatible grid, block and dynamic shared-memory sizes by examining
how the source maps threads to data. The harness does not enable opt-in shared memory.
shared_memory_bytes is ONLY dynamic shared storage requested at launch. Static
__shared__ arrays are allocated automatically: NEVER add their sizes to this field.
If the kernel has only static shared arrays, set shared_memory_bytes to 0.
For each launch, fill dynamic_shared_memory_reason with the dynamic declaration
and its byte formula evaluated for the case, including alignment/padding, or why
zero is correct. Total static + dynamic usage must fit the device limit.
Do not copy Nsight's total shared-memory usage into the dynamic launch field.
Return ONLY launch configurations; the source is already stored.
Return JSON matching this schema, without Markdown fences:
{json.dumps(LaunchPlan.model_json_schema(), indent=2)}
"""
    return new_conversation(prompt)


def finalize_result(state: OptimizerState) -> dict:
    best = state["best_attempt"]
    error = state["error"]
    if best is None and error is None:
        error = "No candidate passed all cases with finite positive timings"
    summary = {
        "run_id": state["session_dir"].name,
        "status": "failed" if error else "complete",
        "error": error,
        "termination": "error" if error else "agent_stop" if state.get("stop_reason") else "iteration_limit",
        "stop_reason": state.get("stop_reason"),
        "objective": "maximum_geometric_mean_performance_case_speedup",
        "best_geometric_mean_ms": geometric_mean_time(best) if best is not None else None,
        "attempt_count": len(state["optimization_history"]),
        "best_iteration": best.iteration if best is not None else None,
        "best_total_ms": attempt_time(best) if best is not None else None,
        "speedup_vs_initial": None,
        "speedup_vs_reference": None,
        "candidate_changes": [
            {"iteration": a.iteration, "change_summary": a.change_summary}
            for a in state["optimization_history"]
        ],
    }
    if best is not None:
        reference = state.get("reference_attempt")
        if reference is not None:
            summary["reference_geometric_mean_ms"] = geometric_mean_time(reference)
            summary["speedup_vs_reference"] = geometric_mean_speedup(reference, best)
        source = state["session_dir"] / "best.cu"
        source.write_text(best.kernel_source, encoding="utf-8")
        summary["kernel_source_path"] = str(source)
        summary["cases"] = []
        for case in best.case_runs:
            request = {**asdict(case.request), "source": str(source),
                       "id": f"best-case-{case.case_index}", "results_dir": str(state["session_dir"])}
            request_path = state["session_dir"] / f"best-case-{case.case_index}.json"
            request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
            summary["cases"].append({
                "case_index": case.case_index,
                "purpose": case.purpose,
                "latency_ms": case.harness_result["latency_ms"],
                "request_path": str(request_path),
            })
            if reference is not None:
                reference_case = next((c for c in reference.case_runs if c.case_index == case.case_index), None)
                if reference_case is not None and reference_case.harness_result.get("ok"):
                    reference_ms = reference_case.harness_result.get("latency_ms")
                    summary["cases"][-1]["reference_latency_ms"] = reference_ms
                    if isinstance(reference_ms, (int, float)) and reference_ms > 0:
                        summary["cases"][-1]["speedup_vs_reference"] = reference_ms / case.harness_result["latency_ms"]
        baseline = next((a for a in state["optimization_history"] if geometric_mean_time(a) is not None), None)
        if baseline is not None:
            summary["baseline_iteration"] = baseline.iteration
            summary["speedup_vs_baseline"] = geometric_mean_speedup(baseline, best)
            initial = next((a for a in state["optimization_history"] if a.iteration == 0), None)
            if initial is not None:
                summary["speedup_vs_initial"] = geometric_mean_speedup(initial, best)
    plot_path = state["session_dir"] / "heatmap.png"
    try:
        plot_history(state["optimization_history"], plot_path, reference=state.get("reference_attempt"))
        summary["timing_plot_path"] = str(plot_path)
    except Exception as plot_error:
        summary["plot_error"] = str(plot_error)
        print(f"Could not save timing plot: {plot_error}", flush=True)
    (state["session_dir"] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"error": error}
