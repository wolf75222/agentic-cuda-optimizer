import argparse
import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from agent_workload import InputCase, Workload
from input_generation import load_input_cases
from nodes import finalize_result
from optimizer_config import OptimizerConfig
from state import OptimizerState
from use_harness import HarnessRuner, executable
from workflow import build_graph


def load_configuration(path: Path) -> tuple[Workload, OptimizerConfig]:
    path = path.resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = data["optimizer"].copy()
    settings.pop("run_id", None)  # Older configurations supplied their own ID.
    settings["results_dir"] = (path.parent / settings["results_dir"]).resolve()
    config = OptimizerConfig(**settings)
    fields = data["workload"].copy()
    fields.pop("kernel", None)  # The signature now determines the entry name.
    for name in ("reference", "initial_kernel"):
        if fields.get(name) is not None:
            fields[name] = (path.parent / fields[name]).resolve()
    cases = fields.get("input_cases")
    if isinstance(cases, str):
        fields["input_cases"] = load_input_cases(path.parent / cases)
    elif cases is not None:
        if not isinstance(cases, list):
            raise ValueError("input_cases must be null, a manifest path, or an array")
        fields["input_cases"] = [InputCase.from_dict(item, path.parent) for item in cases]
    return Workload(**fields), config


def create_session_directory(results_dir: Path) -> Path:
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        session_dir = results_dir / f"run-{number:03d}"
        try:
            session_dir.mkdir()
            return session_dir
        except FileExistsError:
            number += 1


def optimize(
    workload: Workload,
    config: OptimizerConfig,
) -> OptimizerState:
    config.validate()
    if not workload.description.strip():
        raise ValueError("Workload description must not be empty")
    if workload.signature is not None:
        _ = workload.kernel
    if not executable.is_file():
        raise FileNotFoundError(f"Build the CUDA harness first: {executable}")
    load_dotenv()
    load_dotenv(Path(__file__).parent / ".env")
    session_dir = create_session_directory(config.results_dir)
    state = OptimizerState(
        config=config,
        session_dir=session_dir,
        workload=workload,
        device_properties={},
        research_brief="",
        kernel_source="",
        kernel_role="reference",
        kernel_proposal=None,
        iteration=0,
        validation_ok=False,
        reference_results=[],
        reference_attempt=None,
        messages=[],
        optimization_history=[],
        best_attempt=None,
        proposal_retries=0,
        model_calls=0,
        error=None,
        profiling_unavailable=None,
        stop_reason=None,
    )
    (session_dir / "config.json").write_text(
        json.dumps({"optimizer": asdict(config), "workload": asdict(workload)}, indent=2, default=str),
        encoding="utf-8",
    )
    try:
        # Snapshot supplied source files without changing the user's originals.
        sources = {}
        for name in ("reference", "initial_kernel"):
            source_path = getattr(workload, name)
            if source_path is not None:
                snapshot = session_dir / f"supplied-{name}.cu"
                snapshot.write_text(Path(source_path).read_text(encoding="utf-8"), encoding="utf-8")
                sources[name] = snapshot
        state["workload"] = replace(workload, **sources)
        runner = HarnessRuner(executable, session_dir)
        state["device_properties"] = runner.get_device_properties(config.device)
        (session_dir / "device.json").write_text(json.dumps(state["device_properties"], indent=2), encoding="utf-8")
        client = OpenAI(max_retries=config.max_retries, timeout=120)
        graph = build_graph(client, runner, config)
        step_limit = 64 * (config.max_iterations + 3) * (config.max_retries + 1)
        for current in graph.stream(state, config={"recursion_limit": step_limit}, stream_mode="values"):
            state = current
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        (session_dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        state.update(finalize_result(state))
    return state


def parse_cli(argv: list[str] | None = None) -> tuple[Workload, OptimizerConfig]:
    parser = argparse.ArgumentParser(
        usage=(
            "%(prog)s --description TEXT [options]\n"
            "       %(prog)s --config FILE [options]"
        ),
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=28),
    )
    parser.add_argument("configuration", nargs="?", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--description", metavar="TEXT", help="Workload description")
    parser.add_argument("--signature", metavar="SIGNATURE", help="CUDA signature (default: inferred once)")
    parser.add_argument("--config", type=Path, metavar="FILE", help="Load configuration JSON")
    parser.add_argument("--reference", type=Path, metavar="FILE", help="Reference CUDA source")
    parser.add_argument("--initial-kernel", type=Path, metavar="FILE", help="Initial CUDA source")
    parser.add_argument("--input-cases", type=Path, metavar="FILE", help="Input-case manifest")
    parser.add_argument("--results-dir", type=Path, metavar="DIR", help="Output directory (default: ./results)")
    parser.add_argument("--device", type=int, metavar="N", help="CUDA device (default: 0)")
    parser.add_argument("--max-iterations", type=int, metavar="N", help="Improvement attempts (default: 2)")
    parser.add_argument("--warmup", type=int, metavar="N", help="Warmup launches (default: 10)")
    parser.add_argument("--iterations", type=int, metavar="N", help="Timed launches (default: 100)")
    parser.add_argument("--model", metavar="NAME", help="OpenAI model (default: gpt-5-mini)")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), metavar="LEVEL",
                        help="low, medium, high (default: medium)")
    parser.add_argument("--max-retries", type=int, metavar="N", help="Correction retries (default: 2)")
    parser.add_argument("--nvidia-research", action="store_true", default=None,
                        help="Research NVIDIA guidance (default: off)")
    parser.add_argument("--no-nvidia-research", dest="nvidia_research", action="store_false", default=None,
                        help="Disable NVIDIA research")
    parser.add_argument("--use-nsight", action="store_true", default=None,
                        help="Enable Nsight profiling tool (default: off)")
    args = parser.parse_args(argv)
    if args.configuration is not None and args.config is not None:
        parser.error("Use either the positional configuration or --config, not both")
    configuration = args.config if args.config is not None else args.configuration
    if configuration is None:
        missing = [name for name in ("description",) if not getattr(args, name)]
        if missing:
            parser.error("Without JSON, provide " + ", ".join("--" + name.replace("_", "-") for name in missing))
    try:
        if configuration is not None:
            workload, config = load_configuration(configuration)
        else:
            workload = Workload(description=args.description, signature=args.signature)
            config = OptimizerConfig(
                results_dir=Path("results").resolve(),
                device=0,
                max_iterations=2,
                warmup=10,
                iterations=100,
                model="gpt-5-mini",
                max_retries=2,
            )
        workload_updates = {}
        for name in ("description", "signature", "reference", "initial_kernel"):
            value = getattr(args, name)
            if value is not None:
                workload_updates[name] = value.resolve() if isinstance(value, Path) else value
        if args.input_cases is not None:
            workload_updates["input_cases"] = load_input_cases(args.input_cases)
        workload = replace(workload, **workload_updates)
        config_updates = {}
        for name in ("results_dir", "device", "max_iterations", "warmup", "iterations", "model", "reasoning_effort", "max_retries", "nvidia_research", "use_nsight"):
            value = getattr(args, name)
            if value is not None:
                config_updates[name] = value.resolve() if isinstance(value, Path) else value
        config = replace(config, **config_updates)
        config.validate()
        if not workload.description.strip():
            raise ValueError("Workload description must not be empty")
        if workload.signature is not None:
            _ = workload.kernel
    except Exception as error:
        parser.error(str(error))
    return workload, config


def main(argv: list[str] | None = None) -> int:
    workload, config = parse_cli(argv)
    try:
        result = optimize(workload, config)
    except Exception as error:
        print(error)
        return 1
    print(f"Summary: {result['session_dir'] / 'summary.json'}")
    plot_path = result["session_dir"] / "heatmap.png"
    if plot_path.is_file():
        print(f"Timing plot: {plot_path}")
    if result["error"]:
        print(result["error"])
        return 1
    print(f"Best candidate: {result['best_attempt'].iteration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
