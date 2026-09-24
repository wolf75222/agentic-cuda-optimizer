import csv
from fnmatch import fnmatchcase
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

from use_harness import HarnessRuner
from state import OptimizerState


def profile_overview(profile: dict) -> dict:
    names = {
        "gpu__time_duration.sum", "dram__bytes.sum.per_second",
        "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
        "dram__cycles_elapsed.avg.per_second",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__registers_per_thread", "launch__shared_mem_per_block",
        "launch__block_size", "launch__grid_size",
        "lts__t_sector_hit_rate.pct", "l1tex__t_sector_hit_rate.pct",
    }
    return {**profile, "metrics": [m for m in profile.get("metrics", []) if m["name"] in names]}


def read_saved_metrics(runner: HarnessRuner, iteration: int, case_index: int,
                       metric_pattern: str, *, offset: int = 0, names_only: bool = False) -> dict:
    if any(type(value) is not int or value < 0 for value in (iteration, case_index, offset)):
        return {"ok": False, "error": "iteration, case_index and offset must be nonnegative integers"}
    if not isinstance(metric_pattern, str) or not metric_pattern.strip():
        return {"ok": False, "error": "Supply a metric name or wildcard pattern, such as *bank_conflicts*"}
    root = runner.results_dir / "profiles"
    directories = [p for p in root.glob(f"candidate-{iteration}-case-{case_index}-*")
                   if p.is_dir() and p.name.rsplit("-", 1)[-1].isdigit()]
    for directory in sorted(directories, key=lambda p: int(p.name.rsplit("-", 1)[-1]), reverse=True):
        summary_path = directory / "summary.json"
        metrics_path = directory / "metrics.json"
        if not summary_path.is_file() or not metrics_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("ok"):
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        matches = sorted((m for m in metrics if fnmatchcase(m["name"], metric_pattern)),
                         key=lambda m: m["name"])
        page = matches[offset:offset + 50]
        return {
            "ok": True, "iteration": iteration, "case_index": case_index,
            "profile_directory": str(directory), "metric_pattern": metric_pattern,
            "total_matches": len(matches), "offset": offset,
            "next_offset": offset + len(page) if offset + len(page) < len(matches) else None,
            "metrics": [m["name"] for m in page] if names_only else page,
            "message": "No matching metrics were collected; try another pattern or '*' to discover names." if not matches else "",
            "profiling_conditions": summary.get("profiling_conditions", {}),
            "note": "Latest successful saved profile. Diagnostic counters only; rank with benchmark kernel_latency_ms.",
        }
    return {"ok": False, "error": "No successful saved profile for this candidate/case. Use profile_kernel to collect one."}


def profile_candidate(state: OptimizerState, *, runner: HarnessRuner) -> dict:
    if state.get("profiling_unavailable") or not state["validation_ok"]:
        return {}
    attempt = state["optimization_history"][-1]
    # Select the largest allocation footprint, not the tiny correctness case.
    case = max((case for case in attempt.case_runs if case.purpose == "performance"), key=lambda case: sum(
        argument.bytes for argument in case.request.arguments if argument.kind == "buffer"
    ))
    result = profile_saved_kernel(runner, attempt.iteration, case.case_index)
    if "directory" not in result:
        directory = runner.results_dir / "profiles" / f"automatic-{attempt.iteration}"
        directory.mkdir(parents=True, exist_ok=True)
        result.update(iteration=attempt.iteration, case_index=case.case_index)
        (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["ok"]:
        print(f"Automatic profiling unavailable: {result['error']}", flush=True)
        if result.get("retryable") is False:
            return {"profiling_unavailable": result["error"]}
    return {}


def find_ncu() -> Path | None:
    found = shutil.which("ncu")
    if found:
        path = Path(found).resolve()
        if path.suffix.lower() not in (".bat", ".cmd"):
            return path
        # Invoke the executable behind NVIDIA's Windows launcher without a shell.
        executable = path.parent / "target/windows-desktop-win7-x64/ncu.exe"
        if executable.is_file():
            return executable
    if os.name == "nt":
        root = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation"
        for directory in sorted(root.glob("Nsight Compute *"), reverse=True):
            executable = directory / "target/windows-desktop-win7-x64/ncu.exe"
            if executable.is_file():
                return executable
    return None


def parse_metrics(text: str) -> list[dict]:
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines)
                  if "Metric Name" in line and "Metric Value" in line), None)
    if start is None:
        # Raw-page CSV: metric names are columns, followed by units and launches.
        start = next((index for index, line in enumerate(lines)
                      if line.startswith('"ID",') and '"Kernel Name"' in line), None)
        if start is None:
            return []
        rows = csv.reader(io.StringIO("\n".join(lines[start:])))
        names = next(rows)
        units = next(rows, [])
        metrics = []
        for row in rows:
            if not row or not row[0].isdigit():
                continue
            for name, unit, value in zip(names, units, row):
                if "__" in name and value:
                    metrics.append({"name": name, "value": value, "unit": unit})
        return metrics
    metrics = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        name = row.get("Metric Name")
        value = row.get("Metric Value")
        if name and value and name != "Metric Name":
            metrics.append({"name": name, "value": value, "unit": row.get("Metric Unit", "")})
    return metrics


def profile_saved_kernel(runner: HarnessRuner, iteration: int, case_index: int) -> dict:
    if type(iteration) is not int or type(case_index) is not int or iteration < 0 or case_index < 0:
        return {"ok": False, "error": "iteration and case_index must be nonnegative integers"}
    saved = runner.results_dir / f"run-candidate-{iteration}-case-{case_index}"
    if not (saved / "run_info.json").is_file() or not (saved / "kernel.cu").is_file():
        return {"ok": False, "error": "No executed candidate with those indices in this session"}
    ncu = find_ncu()
    if ncu is None:
        return {"ok": False, "retryable": False,
                "error": "Nsight Compute CLI was not found. Install it and add ncu to PATH."}

    root = runner.results_dir / "profiles"
    root.mkdir(exist_ok=True)
    number = 1
    while True:
        directory = root / f"candidate-{iteration}-case-{case_index}-{number:03d}"
        try:
            directory.mkdir()
            break
        except FileExistsError:
            number += 1

    request = json.loads((saved / "run_info.json").read_text(encoding="utf-8"))
    request.update(source=str((saved / "kernel.cu").resolve()), id="profile",
                   results_dir=str(directory.resolve()), mode="run", warmup=0, iterations=1)
    request_path = directory / "request.json"
    request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    report_path = directory / "profile.ncu-rep"
    csv_path = directory / "metrics.csv"
    command = [str(ncu), "--config-file", "off", "--target-processes", "application-only",
               "--kernel-name-base", "function", "--kernel-name", request["kernel"],
               "--launch-count", "1", "--replay-mode", "kernel",
               "--cache-control", "all", "--clock-control", "none",
               "--export", str(report_path), "--csv", "--page", "raw",
               "--print-units", "base", "--log-file", str(csv_path)]
    for section in ("SpeedOfLight", "LaunchStats", "Occupancy", "MemoryWorkloadAnalysis", "WarpStateStats"):
        command.extend(["--section", section])
    command.extend([str(runner.executable), "--request", str(request_path)])
    (directory / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    result = {"ok": False, "iteration": iteration, "case_index": case_index,
              "directory": str(directory), "profiling_conditions": {
                  "launches": 1, "warmup": 0, "cache_control": "all", "clock_control": "none",
                  "note": "Profile metrics, including gpu__time_duration, are diagnostic replay measurements. "
                          "Rank candidates only by kernel_latency_ms from normal benchmark performance reports; "
                          "never use profiler replay time or harness wall-clock duration."}}
    print(f"Profiling candidate {iteration}, case {case_index} with Nsight Compute", flush=True)
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", check=False, timeout=600)
        (directory / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (directory / "stderr.log").write_text(process.stderr, encoding="utf-8")
        csv_text = csv_path.read_text(encoding="utf-8", errors="replace") if csv_path.is_file() else ""
        diagnostics = "\n".join((process.stderr, process.stdout, csv_text))
        if process.returncode != 0 or "ERR_NVGPUCTRPERM" in diagnostics:
            result["error"] = diagnostics[-6000:]
            if "ERR_NVGPUCTRPERM" in diagnostics:
                instructions = (
                    "Open NVIDIA Control Panel > Desktop > Enable Developer Settings.\n"
                    "Then Developer > Manage GPU Performance Counters > Allow access to all users > Apply.\n"
                    "In newer NVIDIA App versions: System > Advanced > Developer > Manage GPU Performance Counters."
                    if os.name == "nt" else
                    "Ask your system administrator to grant your user access to NVIDIA GPU performance counters."
                )
                result["error_code"] = "ERR_NVGPUCTRPERM"
                result["retryable"] = False
                result["error"] = (
                    "Nsight profiling unavailable: GPU performance counter access is disabled for this user "
                    "(ERR_NVGPUCTRPERM).\n"
                    "Kernel compilation, validation and normal benchmarking can continue.\n"
                    + instructions + "\n"
                    "After access is enabled, profiling can be retried. Do not retry while access remains disabled.\n"
                    "Setup instructions: https://developer.nvidia.com/ERR_NVGPUCTRPERM"
                )
                print(result["error"], flush=True)
        else:
            metrics = parse_metrics(csv_text)
            harness_path = directory / "run-profile/result.json"
            harness = json.loads(harness_path.read_text(encoding="utf-8")) if harness_path.is_file() else {}
            result["ok"] = bool(metrics) and bool(harness.get("ok"))
            if not result["ok"]:
                result["error"] = harness.get("error") or "Nsight returned no metrics or the harness did not finish"
            # Keep the complete metric set on disk, and return the relevant subset to the model.
            (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            result["metrics"] = profile_overview({"metrics": metrics})["metrics"]
            result["report_path"] = str(report_path)
            result["metrics_path"] = str(directory / "metrics.json")
    except (OSError, subprocess.TimeoutExpired, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
