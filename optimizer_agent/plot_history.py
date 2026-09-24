import math
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.colors import TwoSlopeNorm
from matplotlib import colormaps

from optimization_history import OptimizationAttempt, geometric_mean_time


def valid_time(case):
    if case is None or not case.validation_ok or not case.harness_result.get("ok"):
        return None
    value = case.harness_result.get("latency_ms")
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def plot_history(history: list[OptimizationAttempt], path: Path, *, reference: OptimizationAttempt | None = None) -> None:
    history = sorted(history, key=lambda attempt: attempt.iteration)
    if reference is not None:
        history = [reference, *history]
    indices = sorted({case.case_index for attempt in history for case in attempt.case_runs})
    details = []
    labels = []
    for index in indices:
        case = next(case for attempt in history for case in attempt.case_runs if case.case_index == index)
        label = f"Case {index}"
        if case.purpose == "correctness":
            label += "\n(not scored)"
        labels.append(label)
        details.append(textwrap.fill(f"Case {index}: {case.size_summary or 'No size summary'}",
                                     width=110, subsequent_indent="    "))
    detail_height = max(0.6, 0.19 * sum(len(d.splitlines()) + 1 for d in details))
    chart_height = max(3, 0.65 * len(history))
    figure = Figure(figsize=(max(9, 2.2 * len(indices) + 2.5), chart_height + detail_height + 1),
                    layout="constrained")
    FigureCanvasAgg(figure)
    if not history or not indices:
        figure.text(0.5, 0.5, "No input-case timings recorded", ha="center", va="center")
    else:
        axis, detail_axis = figure.subplots(2, 1, gridspec_kw={"height_ratios": [chart_height, detail_height]})
        detail_axis.set_axis_off()
        detail_axis.text(0, 1, "\n\n".join(details), transform=detail_axis.transAxes, va="top", fontsize=8)
        baseline = next((attempt for attempt in history if geometric_mean_time(attempt) is not None), None)
        baseline_cases = {case.case_index: case for case in baseline.case_runs} if baseline else {}
        values = np.full((len(history), len(indices)), np.nan)
        cell_labels = {}
        for row, attempt in enumerate(history):
            cases = {case.case_index: case for case in attempt.case_runs}
            for column, index in enumerate(indices):
                case = cases.get(index)
                latency = valid_time(case)
                initial = valid_time(baseline_cases.get(index))
                if latency is None:
                    cell_labels[row, column] = "Invalid" if case and not case.validation_ok else "Unavailable"
                elif initial is None:
                    cell_labels[row, column] = f"No valid baseline\n{latency:.5g} ms"
                else:
                    speedup = initial / latency
                    if math.isfinite(speedup):
                        values[row, column] = speedup
                    cell_labels[row, column] = f"{speedup:.3f}x\n{latency:.5g} ms"
        finite = values[np.isfinite(values)]
        span = max(0.05, float(np.max(np.abs(finite - 1)))) if finite.size else 0.05
        cmap = colormaps["RdYlGn"].copy()
        cmap.set_bad("#e5e7eb")
        image = axis.imshow(values, aspect="auto", cmap=cmap,
                            norm=TwoSlopeNorm(vmin=1-span, vcenter=1, vmax=1+span))
        for (row, column), label in cell_labels.items():
            value = values[row, column]
            rgba = cmap(image.norm(value)) if math.isfinite(value) else (1, 1, 1, 1)
            brightness = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(column, row, label, ha="center", va="center", fontsize=10,
                      color="white" if brightness < 0.45 else "#17212b")
        axis.set_xticks(range(len(indices)), labels)
        axis.xaxis.tick_top()
        row_labels = []
        for attempt in history:
            label = "Reference" if attempt is reference else f"Kernel {attempt.iteration}"
            if attempt is baseline:
                label += " (baseline)"
            label += "\n" + textwrap.fill(attempt.change_summary, width=27)
            row_labels.append(label)
        axis.set_yticks(range(len(history)), row_labels, fontsize=9)
        axis.tick_params(length=0, pad=10)
        axis.set_xticks(np.arange(-0.5, len(indices), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(history), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=3)
        axis.tick_params(which="minor", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        baseline_label = "reference kernel" if baseline is reference and reference is not None else "first fully valid candidate"
        figure.suptitle(f"Speedup relative to the {baseline_label}\nEach cell: speedup and measured time", fontsize=14)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.7, pad=0.04)
        colorbar.set_label("Speedup (baseline time / kernel time)\nAbove 1 = faster")
    figure.savefig(path, dpi=160)
    figure.savefig(path.with_suffix(".svg"))
