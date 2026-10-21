"""Cross-process GPU reservations for validation, timing and profiling."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import time


class GPUInterferenceError(RuntimeError):
    pass


class GPUScheduler:
    def __init__(self, directory: Path, device_key: str, interference_probe=None, timeout: float = 1200):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / (hashlib.sha256(device_key.encode()).hexdigest() + ".lock")
        self.interference_probe = interference_probe
        self.timeout = timeout

    @contextmanager
    def reserve(self, purpose: str = "measurement"):
        import fcntl
        with self.path.open("a+") as handle:
            deadline = time.monotonic() + self.timeout
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("GPU reservation timed out")
                    time.sleep(0.05)
            try:
                if self.interference_probe and self.interference_probe():
                    raise GPUInterferenceError("External CUDA workload observed before reservation")
                handle.seek(0)
                handle.truncate()
                handle.write(f"{os.getpid()} {purpose}\n")
                handle.flush()
                yield
                if self.interference_probe and self.interference_probe():
                    raise GPUInterferenceError("External CUDA workload observed; discard these measurements")
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
