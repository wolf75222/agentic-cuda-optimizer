"""Root-owned GPU tools for a literal, single-prompt Codex CLI experiment.

The Codex child can edit CUDA and launch files in its workspace. This parent
service owns the oracle, frozen search cases, evaluator, records, and Slurm job
in a disjoint directory. No model, optimizer/search policy or holdout API runs
inside this service. HTTP requests and actual GPU measurements are serialized.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
import time
import uuid


REPO = Path(__file__).resolve().parents[1]
# The upstream workload module imports buffer/evaluation types directly. Share
# those exact classes, as campaign.py does, to preserve isinstance contracts.
if str(REPO / "optimizer_agent") not in sys.path:
    sys.path.insert(0, str(REPO / "optimizer_agent"))
from benchmark_workloads import OPERATIONS, SOURCES, WorkloadSuite
from campaign_config import CampaignConfig
from evaluation import Candidate, Evaluator, LaunchRule, WorkloadContract
from execution_backend import BackendError, SlurmBackend, SlurmConfig
SEARCH_SIZES = (128, 256, 512)
VALIDATION_SIZES = (1, 17, 63) + SEARCH_SIZES
try:
    from .workload_semantics import MEMORY_POLICY, operation_spec
except ImportError:
    from workload_semantics import MEMORY_POLICY, operation_spec


def workload_description(operation):
    """Closed semantics also supplied to normal research and generation."""
    description = operation_spec(operation)
    launch = (LaunchRule(kind="square_2d", block=(16, 16, 1)) if operation == "matmul"
              else LaunchRule(kind="elementwise", block=(128, 1, 1)))
    return {**description, "baseline_launch_rule": asdict(launch)}


def baseline_source(config, repo_dir=REPO):
    if config.workload not in OPERATIONS:
        raise ValueError(f"Unknown workload {config.workload!r}")
    return ((Path(repo_dir) / config.baseline_source).read_text() if config.workload == "matmul"
            else SOURCES[config.workload])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def device_identity(properties):
    names = ("name", "architecture", "compute_capability", "uuid", "gpu_uuid",
             "driver_version", "nvrtc_version", "cuda_toolkit_version", "harness_build_id")
    return {key: properties[key] for key in names if key in properties}


class VanillaGPUService:
    def __init__(self, config: CampaignConfig, protected_dir, child_workspace, session_id,
                 repo_dir=REPO, *, runner=None, observe_first_gain=False,
                 observer_started_at_unix=None, observer_usage_provider=None):
        self.repo_dir = Path(repo_dir).resolve()
        self.protected_dir = Path(protected_dir).resolve()
        self.child_workspace = Path(child_workspace).resolve()
        if (self.protected_dir.is_relative_to(self.child_workspace)
                or self.child_workspace.is_relative_to(self.protected_dir)):
            raise ValueError("Evaluator/proof directory and child workspace must be disjoint")
        if config.workload not in OPERATIONS or config.backend != "romeo":
            raise ValueError("The vanilla GPU experiment requires a supported, fixed ROMEO workload contract")
        # Search shapes and input seed are parent-owned. Final/holdout cases are
        # never created, admitted in a request, or exposed by an HTTP operation.
        self.config = replace(config, search_sizes=tuple(config.search_sizes))
        self.search_sizes = tuple(self.config.search_sizes)
        self.validation_sizes = tuple(dict.fromkeys((1, 17, 63) + self.search_sizes))
        self.description = workload_description(config.workload)
        self.session_id = session_id
        self.token = secrets.token_urlsafe(32)
        self.backend = None
        self._provided_runner = runner
        self.job_id = None
        self.evaluator = None
        self.baseline = None
        self.properties = {}
        self.metadata = {}
        self._server = None
        self._thread = None
        self._lock = threading.Lock()
        self._closed = False
        self._terminal = None
        self._observe_first_gain = observe_first_gain
        self._observer_started = time.time() if observer_started_at_unix is None else observer_started_at_unix
        self._observer_usage_provider = observer_usage_provider
        self.gain_observer = None
        self._observer_failure = None
        self._method_usage = {"method_gpu_evaluation_wall_seconds":0.0,"method_gpu_evaluation_calls":0}

    def _observer_usage(self):
        # The native CLI reports aggregate model usage after its complete turn.
        # During GPU callbacks those token/call counts are unknown.
        usage = {"model_tokens":None,"input_tokens":None,"output_tokens":None,"model_calls":None,
                 **self._method_usage}
        if self._observer_usage_provider is not None:
            usage.update(self._observer_usage_provider())
        return usage

    def observer_metrics(self):
        """Parent-only metrics, never part of a model-facing tool response."""
        return self.gain_observer.snapshot()["metrics"] if self.gain_observer else {}

    def _setup_evaluator(self, runner):
        suite = WorkloadSuite(self.config.workload, self.protected_dir / "search-data", self.config)
        cases = suite.cases(final=False)
        if tuple(case.metadata["n"] for case in cases) != self.validation_sizes:
            raise ValueError("The fixed vanilla search suite changed")
        contract = WorkloadContract(
            "literal-vanilla-" + hashlib.sha256(self.session_id.encode()).hexdigest()[:16],
            1, self.config.workload, suite.signature,
            "Independent NumPy CPU implementation in benchmark_workloads.py, float64 accumulation with float32 outputs",
            benchmark_series=self.config.benchmark_series,
            confirmation_batches=self.config.confirmation_batches,
            launches_per_series=self.config.launches_per_series, warmup=self.config.warmup,
            confidence=self.config.confidence_level, min_relative_gain=self.config.min_relative_gain,
            max_critical_regression=self.config.max_critical_regression,
            required_sanitizers=self.config.required_sanitizers, allowed_precision="float32",
            metadata_json=json.dumps({
                "operation": self.config.workload, "precision": "float32",
                "timed_scope": "single CUDA kernel, CUDA Events, transfers/compilation excluded",
                "cache_regime": "warm, inputs reset before every launch",
                "memory_policy": self.description["memory_policy"],
                "mathematics": self.description["mathematics"], "environment": device_identity(self.properties),
                "target_environment": device_identity(self.properties),
                "numerical_tolerance": {"atol": 2e-4, "rtol": 2e-4},
                "scope": "public search cases only; no final data API"}, sort_keys=True))
        evaluator = Evaluator(runner, contract, suite.oracle,
            {"short": cases[:2], "search": cases,
             "performance": [case for case in cases if case.metadata["n"] in self.search_sizes]})
        baseline = suite.baseline(self.repo_dir / self.config.baseline_source)
        started = time.monotonic()
        try:
            measurement = evaluator.benchmark(baseline)  # Fresh actual GPU evidence.
        finally:
            self._method_usage["method_gpu_evaluation_wall_seconds"] += time.monotonic()-started
            self._method_usage["method_gpu_evaluation_calls"] += 1
        write_json(self.protected_dir / "baseline.json", measurement)
        if measurement.get("ok") is not True:
            raise BackendError("The fixed naive baseline did not validate and benchmark successfully")
        self.evaluator, self.baseline = evaluator, measurement
        if self._observe_first_gain:
            try:
                from .gain_observer import GainObserver
            except ImportError:
                from gain_observer import GainObserver
            self.gain_observer = GainObserver(evaluator,baseline,self.protected_dir/"gain-observer",
                method="native_codex",campaign_started_at_unix=self._observer_started,
                usage_provider=self._observer_usage)
        write_json(self.protected_dir / "contract.json", {
            "contract": asdict(contract), "contract_hash": evaluator.contract_hash,
            "search_sizes": list(self.search_sizes), "validation_sizes": list(self.validation_sizes)})

    def start(self):
        if self._closed:
            raise RuntimeError("This service is closed; use a new isolated session")
        if self._server is not None:
            return self.metadata
        self.protected_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.child_workspace.mkdir(parents=True, exist_ok=True)
        if (self.protected_dir / "service.json").exists():
            raise ValueError("Protected service records already exist; use a new proof directory/session")
        self.protected_dir.chmod(0o700)
        try:
            if self._provided_runner is None:
                self.backend = SlurmBackend(SlurmConfig.load(self.repo_dir / self.config.execution_config),
                    self.repo_dir, self.protected_dir / "worker", session_id=self.session_id)
                self.backend.start()
                self.backend.wait_until_ready()
                runner = self.backend.runner()
                self.job_id = self.backend.job_id
            else:
                # Suite orchestration owns this lease. This service must never
                # submit, stop or cancel a caller-provided worker.
                runner = self._provided_runner
                parent_backend = getattr(runner, "backend", None)
                self.job_id = getattr(parent_backend, "job_id", getattr(runner, "job_id", None))
            self.properties = runner.get_device_properties(0)
            self._setup_evaluator(runner)
            service = self

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass  # Structured protected audit records replace stderr.

                def _reply(self, status, payload):
                    body = json.dumps(payload, allow_nan=False).encode()
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # The authoritative response remains in proofdir.

                def _handle(self):
                    request_id = uuid.uuid4().hex
                    supplied = self.headers.get("Authorization", "")
                    if not secrets.compare_digest(supplied, "Bearer " + service.token):
                        service._audit({"request_id": request_id, "operation": self.path,
                                        "authenticated": False, "status": "rejected"})
                        self._reply(401, {"ok": False, "error": "Authentication required"})
                        return
                    operation = self.path.removeprefix("/")
                    try:
                        if operation == "properties" and self.command == "GET":
                            request = {}
                        elif operation in {"validate", "benchmark"} and self.command == "POST":
                            length = int(self.headers.get("Content-Length", "0"))
                            if length <= 0 or length > 65536:
                                raise ValueError("A bounded JSON path request is required")
                            request = json.loads(self.rfile.read(length))
                        else:
                            raise ValueError("Only properties, validate and benchmark are available")
                        result = service.dispatch(operation, request, request_id=request_id)
                        self._reply(200, result)
                    except (ValueError, OSError, TypeError) as error:
                        service._audit({"request_id": request_id, "operation": operation,
                                        "authenticated": True, "status": "invalid_request", "error": str(error)})
                        self._reply(400, {"ok": False, "request_id": request_id, "error": str(error)})
                    except Exception as error:
                        service._audit({"request_id": request_id, "operation": operation,
                                        "authenticated": True, "status": "service_error", "error": str(error)})
                        self._reply(503, {"ok": False, "request_id": request_id, "error": str(error)})

                do_GET = _handle
                do_POST = _handle

            self._server = HTTPServer(("127.0.0.1", 0), Handler)
            endpoint = "http://127.0.0.1:" + str(self._server.server_address[1])
            self.metadata = {
                "endpoint": endpoint, "token": self.token,
                "tool_command": [sys.executable, str(self.repo_dir / "scripts/vanilla_gpu_tool.py")],
                "properties": self.properties, "baseline": self.baseline,
                "contract": asdict(self.evaluator.contract), "contract_hash": self.evaluator.contract_hash,
                "search_sizes": list(self.search_sizes), "validation_sizes": list(self.validation_sizes),
                "proof_directory": str(self.protected_dir), "slurm_job_id": self.job_id,
                "worker_ownership": "owned" if self.backend is not None else "borrowed",
                "owns_worker": self.backend is not None,
                "operation": self.config.workload, "workload_description": self.description,
                "memory_policy": self.description["memory_policy"],
            }
            write_json(self.protected_dir / "service.json", {
                **{key: value for key, value in self.metadata.items() if key != "token"},
                "token_sha256": hashlib.sha256(self.token.encode()).hexdigest(),
                "child_workspace": str(self.child_workspace),
                "scope": "One-prompt Codex CLI with genuine GPU tools; no agent search/orchestration inside service"})
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            return self.metadata
        except BaseException:
            self.close()
            raise

    def _audit(self, event):
        event = {"timestamp": time.time(), **event}
        with (self.protected_dir / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")

    def _read_child_file(self, supplied):
        if not isinstance(supplied, str) or not supplied:
            raise ValueError("A workspace source/launch path is required")
        path = Path(supplied)
        if path.is_absolute():
            try:
                relative = path.relative_to(self.child_workspace)
            except ValueError as error:
                raise ValueError("Source and launch files must be inside the child workspace") from error
        else:
            relative = path
        if not relative.parts or any(part in ("..", ".") for part in relative.parts):
            raise ValueError("Parent traversal is not a workspace file")
        # Walk directory descriptors with O_NOFOLLOW, including every parent;
        # resolving a Path once would permit a concurrent symlink substitution.
        descriptor = os.open(self.child_workspace, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in relative.parts[:-1]:
                next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            file_descriptor = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
            with os.fdopen(file_descriptor, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError("Source/launch path must identify a regular workspace file")
                return stream.read()
        finally:
            os.close(descriptor)

    def _candidate(self, request):
        if not isinstance(request, dict) or set(request) != {"source", "launch"}:
            raise ValueError("Only source and launch workspace file paths may be submitted")
        source = self._read_child_file(request["source"])
        # Validate UTF-8 before immutable materialization and source admission.
        source.decode("utf-8")
        launch = json.loads(self._read_child_file(request["launch"]))
        if not isinstance(launch, dict):
            raise ValueError("Launch file must contain a LaunchRule JSON object")
        parameters = {}
        if "launch_rule" in launch:
            if (set(launch) - {"launch_rule", "parameters", "kernel"}
                    or launch.get("kernel", self.config.workload) != self.config.workload):
                raise ValueError(f"The fixed kernel ABI is {self.config.workload}; unsupported launch metadata")
            parameters = launch.get("parameters", {})
            launch = launch["launch_rule"]
        if not isinstance(parameters, dict) or not isinstance(launch, dict):
            raise ValueError("Launch rule and parameters must be objects")
        rule = LaunchRule(**launch)
        identity = hashlib.sha256(source + json.dumps({"rule": asdict(rule), "parameters": parameters},
                                                     sort_keys=True, allow_nan=False).encode()).hexdigest()
        directory = self.protected_dir / "candidates" / identity
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "kernel.cu"
        if path.exists() and path.read_bytes() != source:
            raise ValueError("Immutable candidate collision")
        if not path.exists():
            path.write_bytes(source)
            write_json(directory / "launch.json", {"launch_rule": asdict(rule), "parameters": parameters})
        return Candidate(identity[:24], path, self.config.workload, rule, parameters)

    def dispatch(self, operation, request, *, request_id=None):
        request_id = request_id or uuid.uuid4().hex
        with self._lock:
            if self.evaluator is None or self._closed:
                raise RuntimeError("GPU service is not running")
            directory = self.protected_dir / "requests" / request_id
            write_json(directory / "request.json", {"operation": operation, "request": request})
            if operation == "properties":
                if request:
                    raise ValueError("Properties accepts no mutable configuration")
                result = {"properties": self.properties, "baseline": self.baseline,
                          "contract": asdict(self.evaluator.contract), "contract_hash": self.evaluator.contract_hash,
                          "search_sizes": list(self.search_sizes), "validation_sizes": list(self.validation_sizes),
                          "operation": self.config.workload, "workload_description": self.description,
                          "memory_policy": self.description["memory_policy"]}
                response = {"ok": True, "operation": operation, "request_id": request_id, **result}
            elif operation in {"validate", "benchmark"}:
                if self._observer_failure:
                    raise RuntimeError("GPU activity stopped after independent observer infrastructure incident")
                candidate = self._candidate(request)
                started = time.monotonic()
                try:
                    result = (self.evaluator.validate(candidate, "search") if operation == "validate"
                              else self.evaluator.benchmark(candidate))
                finally:
                    self._method_usage["method_gpu_evaluation_wall_seconds"] += time.monotonic()-started
                    self._method_usage["method_gpu_evaluation_calls"] += 1
                response = {"ok": result.get("ok") is True, "operation": operation,
                            "request_id": request_id, "candidate_id": candidate.candidate_id,
                            "artifact_hash": candidate.artifact_hash, "source_sha256": candidate._source_hash,
                            "result": result, "proof_path": str(directory / "response.json"),
                            "execution_environment": "romeo", "slurm_job_id": self.job_id}
            else:
                raise ValueError("Only properties, validate and benchmark are available")
            write_json(directory / "response.json", response)
            if operation == "benchmark" and response["ok"] and self.gain_observer:
                # Keep the genuine method measurement even if this independent
                # callback fails. Its decision is never included in the reply.
                try:
                    self.gain_observer.observe(candidate,evidence_id=result.get("evaluation_id",request_id))
                except BaseException as error:
                    self._observer_failure = str(error)
                    write_json(self.protected_dir/"observer-unavailable.json",{
                        "error":str(error),"request_id":request_id,"gpu_activity_stopped":True})
                    raise
            self._audit({"request_id": request_id, "operation": operation, "authenticated": True,
                         "status": "complete", "ok": response["ok"],
                         "artifact_hash": response.get("artifact_hash"),
                         "proof_path": str(directory / "response.json")})
            return response

    def close(self):
        if self._closed:
            return self._terminal
        self._closed = True
        if self._server:
            if self._thread and self._thread.is_alive():
                self._server.shutdown()
            self._server.server_close()
            if self._thread:
                self._thread.join(timeout=5)
        receipt = {"state": "NOT_SUBMITTED", "terminal": True}
        if self._provided_runner is not None:
            receipt = {"job_id": self.job_id, "ownership": "borrowed", "terminal": False,
                       "state": "BORROWED_WORKER_PRESERVED", "service_closed": True}
        if self.backend and self.backend.job_id:
            try:
                self.backend.close()
                receipt = self.backend.wait_until_finished(timeout=60)
                if not receipt.get("terminal"):
                    self.backend.close(cancel=True)
                    receipt = self.backend.wait_until_finished(timeout=60)
            except Exception as error:
                receipt = {"job_id": self.backend.job_id, "terminal": False, "cleanup_error": str(error)}
        self.protected_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.protected_dir / "worker-terminal.json", receipt)
        if self.gain_observer:
            self.observer_metrics()
        self._terminal = receipt
        return receipt

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args):
        self.close()
