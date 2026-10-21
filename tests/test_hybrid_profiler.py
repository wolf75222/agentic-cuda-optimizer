from contextlib import contextmanager
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace

from evaluation import Evaluator
from hybrid_profiler import HybridProfiler, reserve_local_gpu
from use_harness import KernelRequest


def test_local_profile_shares_evaluator_physical_gpu_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    evaluator = object.__new__(Evaluator)
    evaluator.scheduler = None
    evaluator._properties = {"uuid": "physical-gpu-shared-by-two-ordinals"}
    entered = threading.Event()
    def profile():
        with reserve_local_gpu(7, properties={"uuid": evaluator._properties["uuid"]}):
            entered.set()
    with evaluator._reservation("benchmark"):
        contender = threading.Thread(target=profile)
        contender.start()
        assert not entered.wait(0.1), "local profiler overlapped an active evaluator benchmark"
    contender.join(timeout=2)
    assert entered.is_set()


def test_local_profile_uses_queried_device_uuid_for_shared_lock(tmp_path, monkeypatch):
    properties = {"uuid": "physical-gpu", "architecture": "sm_90"}
    runner = SimpleNamespace(results_dir=tmp_path, executable=tmp_path / "harness",
                             get_device_properties=lambda device: properties)
    recorded = {}
    @contextmanager
    def reserve(device, **kwargs):
        recorded.update(device=device, **kwargs)
        yield
    monkeypatch.setattr("hybrid_profiler.reserve_local_gpu", reserve)
    monkeypatch.setattr("hybrid_profiler.shutil.which", lambda name: "/fake/ncu")
    def invoke(command, **kwargs):
        csv_path = Path(command[command.index("--log-file") + 1])
        csv_path.write_text('"Metric Name","Metric Unit","Metric Value"\n"dram__bytes.sum","byte","1024"\n')
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("hybrid_profiler.subprocess.run", invoke)
    request = KernelRequest(str(tmp_path / "kernel.cu"), "kernel", "run", 3,
                            [1, 1, 1], [32, 1, 1], 0, 0, 1, [])
    result = HybridProfiler(local_runner=runner)._local("profile", request, "Memory traffic?", ["dram__bytes.sum"])
    assert result["ok"]
    assert recorded == {"device": 3, "properties": properties}
