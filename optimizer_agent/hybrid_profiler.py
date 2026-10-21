"""Conditional profiling with explicit hardware provenance and observed usage."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time


@contextmanager
def reserve_local_gpu(device=0, *, properties=None):
    """Use the evaluator's shared physical-GPU reservation for profiling."""
    import tempfile
    try:
        from .gpu_scheduler import GPUScheduler
    except ImportError:
        from gpu_scheduler import GPUScheduler
    properties = properties or {}
    # Keep the fallback identical to Evaluator._reservation as well. UUID is
    # preferred because CUDA ordinals can differ between local processes.
    device_key = str(properties.get("uuid", properties.get("architecture", "device-0")))
    scheduler = GPUScheduler(Path(tempfile.gettempdir()) / "agentic-cuda-gpu-reservations", device_key)
    with scheduler.reserve("profiling"):
        yield


class HybridProfiler:
    def __init__(self, remote_runner=None, local_runner=None, *, max_profiles=None,
                 max_seconds=None, timeout=120):
        self.remote_runner = remote_runner
        self.local_runner = local_runner
        self.max_profiles = max_profiles
        self.max_seconds = max_seconds
        self.timeout = timeout
        self.attempts = 0
        self.elapsed = 0.0
        self.remote_unavailable = None

    def profile(self, run_id, request, *, question, metrics):
        if not isinstance(question, str) or not question.strip() or not metrics:
            raise ValueError("A profile requires a question and explicit expected metrics")
        if any(not isinstance(metric, str) or not re.fullmatch(r"[A-Za-z0-9_.]+", metric) for metric in metrics):
            raise ValueError("Expected explicit Nsight metric names")
        if ((self.max_profiles is not None and self.attempts >= self.max_profiles)
                or (self.max_seconds is not None and self.elapsed >= self.max_seconds)):
            return {"ok": False, "error": "Profiling budget exhausted", "diagnostic_only": True}
        self.attempts += 1
        start = time.monotonic()
        attempts = []
        try:
            if self.remote_runner and not self.remote_unavailable:
                try:
                    result = self.remote_runner.profile(run_id + "-romeo-profile", request,
                                                        question=question, metrics=metrics)
                except Exception as error:
                    result = {"ok": False, "error": str(error), "execution_environment": "romeo"}
                result.update(diagnostic_only=True, ranking_eligible=False)
                attempts.append(result)
                if result.get("ok"):
                    return {**result, "profile_attempts": attempts}
                # Missing tools and permission denials are sticky for a campaign.
                if result.get("retryable") is False or result.get("error_code") in ("ERR_NVGPUCTRPERM", "NCU_UNAVAILABLE"):
                    self.remote_unavailable = result.get("error", "ROMEO profiling unavailable")
            result = self._local(run_id, request, question, metrics)
            result.update(diagnostic_only=True, ranking_eligible=False, execution_environment="local",
                          hardware_transfer_note="Local profiles describe the local GPU only; their times and counters are not ROMEO benchmark observations.")
            attempts.append(result.copy())
            return {**result, "remote_unavailable": self.remote_unavailable, "profile_attempts": attempts}
        finally:
            self.elapsed += time.monotonic() - start

    def _local(self, run_id, request, question, metrics):
        fingerprint = {"environment": "local", "machine": platform.machine(), "hostname": platform.node()}
        ncu = shutil.which("ncu")
        if self.local_runner is None or ncu is None:
            return {"ok": False, "retryable": False, "error": "Local CUDA harness or Nsight Compute unavailable",
                    "hardware_fingerprint": fingerprint}
        try:
            fingerprint["device_properties"] = self.local_runner.get_device_properties(request.device)
        except Exception as error:
            return {"ok": False, "retryable": False, "error": str(error), "hardware_fingerprint": fingerprint}
        directory = self.local_runner.results_dir / "profiles" / run_id
        directory.mkdir(parents=True, exist_ok=True)
        payload = asdict(request)
        payload.update(id="profile", results_dir=str(directory.resolve()), mode="run", warmup=0, iterations=1, series=1)
        path = directory / "request.json"
        path.write_text(json.dumps(payload, indent=2))
        command = [ncu, "--config-file", "off", "--target-processes", "application-only", "--launch-count", "1",
                   "--replay-mode", "kernel", "--cache-control", "all", "--clock-control", "none", "--csv",
                   "--page", "details", "--metrics", ",".join(metrics), "--log-file", str(directory / "metrics.csv"),
                   str(self.local_runner.executable), "--request", str(path)]
        try:
            with reserve_local_gpu(request.device, properties=fingerprint["device_properties"]):
                process = subprocess.run(command, capture_output=True, text=True,
                                         timeout=(min(self.timeout, max(1, self.max_seconds-self.elapsed)) if self.max_seconds is not None else self.timeout), check=False)
            (directory / "stdout.log").write_text(process.stdout)
            (directory / "stderr.log").write_text(process.stderr)
            csv_text = (directory / "metrics.csv").read_text() if (directory / "metrics.csv").exists() else ""
            try:
                from .remote_worker import parse_metrics
            except ImportError:
                from remote_worker import parse_metrics
            collected = parse_metrics(csv_text)
            result = {"ok": process.returncode == 0 and bool(collected), "metrics": collected,
                      "error": process.stderr[-6000:] if process.returncode else None,
                      "hardware_fingerprint": fingerprint, "directory": str(directory),
                      "profiling_conditions": {"question": question, "expected_metrics": metrics,
                                               "cache_control": "all", "clock_control": "none", "replay_mode": "kernel"}}
        except (OSError, subprocess.TimeoutExpired) as error:
            result = {"ok": False, "error": str(error), "hardware_fingerprint": fingerprint}
        (directory / "summary.json").write_text(json.dumps(result, indent=2))
        return result
