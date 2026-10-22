#!/usr/bin/env python3
"""Validate one reconstructed §16 finalist, without timing or model calls.

Without a SHA-pinned quiescence attestation the default prints prerequisites
and does not open the bundle. With that attestation the default prepares a
local plan; --run explicitly owns one fresh long/48h/one-GPU allocation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import traceback
import uuid
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from optimizer_agent import portable_export
from optimizer_agent.campaign import device_identity, write_immutable_json
from optimizer_agent.evaluation import WorkloadContract, _source_contract_violations
from optimizer_agent.execution_backend import SlurmBackend, SlurmConfig
from optimizer_agent.suite_orchestrator import _local_session_job
from optimizer_agent.workload_semantics import operation_spec
from scripts.acceptance_intent import controller_provenance, persist_allocation_intent

QUIESCENT = ("all_relevant_model_processes_finished", "all_relevant_controllers_finished",
             "all_owned_jobs_terminal", "no_pending_submission_intents",
             "cold_component_roots_frozen", "no_unrevealed_final_inputs_to_publish")
IDENTITY_KEYS = ("uuid", "architecture", "driver_version", "nvrtc_version",
                 "cuda_toolkit_version", "harness_build_id")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _regular(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("Evidence must be a regular, unaliased local file")
    return path


def _member(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Unsafe archive/evidence member")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("Unsafe archive/evidence member")
    return value


def authenticate_quiescence(path, expected_sha, bundle_sha, selection):
    """Check a manual observation and its pins; never infer current liveness."""
    path = _regular(path)
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)) or sha(path) != expected_sha:
        raise ValueError("Quiescence attestation SHA differs")
    value = json.loads(path.read_bytes())
    if (value.get("schema_version") != 1 or value.get("kind") != "manual_validation_release_quiescence"
            or value.get("bundle_sha256") != bundle_sha or value.get("selection") != selection
            or not value.get("attested_by") or not value.get("attested_at_utc")
            or any(value.get(key) is not True for key in QUIESCENT)):
        raise ValueError("Bundle/selection quiescence is not independently attested")
    pins = value.get("evidence_pins")
    roots = value.get("roots", {})
    if not isinstance(pins, list) or not pins:
        raise ValueError("Quiescence needs independently acquired evidence pins")
    for pin in pins:
        # Absolute public receipt pins or the assembler's named-root format.
        if "root" in pin:
            root = Path(roots[pin["root"]])
            if not root.is_absolute():
                raise ValueError("Quiescence evidence root must be absolute")
            source = root / _member(pin["path"])
        else:
            source = Path(pin["path"])
            if not source.is_absolute():
                raise ValueError("Quiescence evidence path must be absolute")
        if source.suffix != ".json" or sha(_regular(source)) != pin.get("sha256"):
            raise ValueError("Quiescence evidence pin differs")
    return {"path": str(path), "sha256": expected_sha,
            "scope": "SHA-pinned manual historical observation; no scheduler/process query", "evidence_pins": pins}


def request_identity(request):
    data = asdict(request)
    data.pop("cache_dir", None)
    data.pop("compile_cache_key", None)
    data["source_sha256"] = sha(data.pop("source"))
    for argument in data["arguments"]:
        if argument.get("input_file"):
            argument["input_sha256"] = sha(argument.pop("input_file"))
    data["dependencies"] = [sha(path) for path in data.get("dependencies", [])]
    return data


def cpu_oracle(operation, a, b, n):
    """Independent float64 CPU mathematics, cast once to float32 output."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    if operation == "matmul":
        result = x.reshape(n, n) @ y.reshape(n, n)
    elif operation == "reduction":
        result = np.array([np.sum(x, dtype=np.float64)])
    elif operation == "scan":
        result = np.cumsum(x, dtype=np.float64)
    elif operation in ("stencil", "convolution"):
        p = np.pad(x, (1, 1))
        result = p[:-2] + p[1:-1] + p[2:] if operation == "stencil" else p[:-2]*y[0] + p[1:-1]*y[1] + p[2:]*y[2]
    elif operation == "normalization":
        result = x / np.sqrt(np.mean(x*x) + 1e-5)
    else:
        raise ValueError("Unsupported independent oracle")
    return np.asarray(result, dtype="<f4").ravel()


def _same_artifact(row, chosen):
    return all(row.get(key) == chosen.get(key) for key in
               ("candidate_id", "source_hash", "artifact_hash", "kernel", "launch_rule", "parameters"))


def authenticate_portable_options(original, portable, relocations, chosen):
    """Allow only the exporter's declared include-directory relocation.

    Portable option hashes alone do not authenticate the transformation from
    the options actually measured. All other bytes/order must remain exact.
    """
    from optimizer_agent.compile_cache import validate_dependency_options
    validate_dependency_options(original)
    expected, include_indices, paired = list(original), set(), False
    for index, option in enumerate(original):
        if paired:
            include_indices.add(index)
            paired = False
        elif option in ("-I", "--include-path"):
            paired = True
        elif option.startswith("-I") or option.startswith("--include-path="):
            include_indices.add(index)
    if paired:
        raise ValueError("Original include option has no directory")
    used = set()
    for row in relocations:
        index = row.get("index")
        if type(index) is not int or index not in include_indices or index in used or row.get("original") != original[index]:
            raise ValueError("Compiler option relocation is not bound to an original include option")
        used.add(index)
        prefix = row.get("prefix")
        option = original[index]
        if prefix == "" and index and original[index-1] in ("-I", "--include-path"):
            old_directory = option
        elif prefix in ("-I", "--include-path=") and option.startswith(prefix):
            old_directory = option[len(prefix):]
        else:
            raise ValueError("Only the four literal include-path forms may be relocated")
        old = Path(old_directory)
        target = _member(row.get("path"))
        if not old.is_absolute() or ".." in old.parts or not target.startswith(chosen["prefix"]):
            raise ValueError("Include relocation is outside the archived compilation tree")
        covered = 0
        for location, member in chosen["source_files"].items():
            source = Path(location)
            if source.is_relative_to(old):
                covered += 1
                if str(PurePosixPath(target) / source.relative_to(old).as_posix()) != member:
                    raise ValueError("Include relocation changes the authenticated source-tree layout")
        if not covered:
            raise ValueError("Include relocation has no authenticated source-tree files")
        expected[index] = prefix + target
    if used != include_indices or expected != portable:
        raise ValueError("Portable compiler options differ from the measured options and verified relocations")


def _archive_authority(archive, manifest, chosen, request_index):
    selection = chosen["selection"]
    match = re.fullmatch(r"(final-groups/[^/]+)/frozen-candidates\.json#([A-Za-z0-9_]+)", selection)
    if not match or chosen.get("evidence_context") != selection.split("#")[0]:
        raise ValueError("Select an actual common-final frozen candidate, not a pilot/search receipt")
    group, mode = match.groups()
    if mode == "baseline":
        raise ValueError("Select a retained method artifact rather than the external baseline entry")
    def document(name):
        key = "evidence/" + _member(name)
        if key not in manifest["files"]:
            raise ValueError("Missing archived final authority: " + name)
        return json.loads(archive.read(key))
    frozen = document(group + "/frozen-candidates.json")
    result = document(group + "/result.json")
    terminal = result.get("worker_terminal", {})
    if (result.get("kind") != "shared_final_group" or result.get("evaluation_finished") is not True
            or result.get("execution_complete") is not True or result.get("all_modes_available") is not True
            or terminal.get("terminal") is not True or terminal.get("state") != "COMPLETED"
            or terminal.get("exit_code") != "0:0" or not result.get("job_id")
            or str(terminal.get("job_id")) != str(result["job_id"])):
        raise ValueError("Selected final experiment lacks a successful owned terminal/closed result")
    outcome = result.get("candidates", {}).get(mode, {})
    if (not _same_artifact(frozen.get(mode, {}), chosen) or not _same_artifact(outcome, chosen)
            or outcome.get("qualified") is not True or chosen.get("contract") != result.get("contract")
            or chosen.get("contract_hash") != result.get("contract_hash")
            or chosen.get("environment") != result.get("environment")):
        raise ValueError("Selected artifact, final contract or qualification differs from archived authority")
    contract = WorkloadContract(**result["contract"])
    semantics = operation_spec(contract.operation)
    metadata = json.loads(contract.metadata_json)
    if (contract.signature != semantics["signature"] or chosen["kernel"] != contract.operation
            or contract.allowed_precision != "float32" or contract.nonfinite_policy != "reject"
            or metadata.get("operation_semantics") != semantics or metadata.get("generation_closed") is not True
            or metadata.get("input_hashes") != result.get("input_hashes")
            or metadata.get("target_environment") != result.get("environment")):
        raise ValueError("Unsupported or inconsistent closed six-operation float32 contract")
    if type(request_index) is not int or not 0 <= request_index < len(chosen["requests"]):
        raise ValueError("An exact archived request index is required")
    link = chosen["requests"][request_index]
    origin = _member(link["origin"])
    if not origin.startswith(group + "/request-receipts/"):
        raise ValueError("Selected execution is not from the chosen common-final group")
    receipt = document(origin)
    request, report = receipt.get("request", {}), receipt.get("report", {})
    if (report.get("ok") is not True or str(report.get("slurm_job_id")) != str(result["job_id"])
            or report.get("guard_readonly_ok") is not True or report.get("guard_outputs_poisoned") is not True
            or not all(report.get("environment", {}).get(key) == result.get("environment", {}).get(key)
                       for key in IDENTITY_KEYS)):
        raise ValueError("Archived request lacks a successful source/ownership/guard observation")
    return group, mode, result, contract, metadata, link, request, report


def prepare(directory, bundle, bundle_sha256, *, selection, request_index, quiescence, quiescence_sha256):
    # This guard deliberately precedes even ZipFile construction or bundle hashing.
    observation = authenticate_quiescence(quiescence, quiescence_sha256, bundle_sha256, selection)
    directory, bundle = Path(directory).resolve(), _regular(bundle)
    if directory.exists() or directory == bundle.parent or bundle.is_relative_to(directory):
        raise ValueError("Use a fresh destination disjoint from the bundle and its inputs")
    if sha(bundle) != bundle_sha256:
        raise ValueError("Bundle SHA differs from its independently pinned bytes")
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("export-manifest.json"))
        if (manifest.get("kind") != "portable_study_export" or manifest.get("status") != "complete"
                or manifest.get("controller_or_worker_lock_observed_active") is not False
                or any(manifest.get(key) for key in ("active_jobs_observed", "inflight_search_trials", "errors", "changed_files_during_export"))
                or manifest.get("source_head_changed_during_export") is not False):
            raise ValueError("Only a complete quiescent study export is accepted")
        observed_root = Path(manifest["study_directory_observed"])
        if not observed_root.is_absolute() or directory.is_relative_to(observed_root) or observed_root.is_relative_to(directory):
            raise ValueError("Reconstruction output must be disjoint from the original study")
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]) | {"export-manifest.json"}:
            raise ValueError("Duplicate, missing or unexpected archive members")
        for name, identity in manifest["files"].items():
            raw = archive.read(_member(name))
            if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
                raise ValueError("Archived member SHA differs: " + name)
        choices = [row for row in manifest["candidates"] if selection in (row["id"], row["selection"])]
        if len(choices) != 1:
            raise ValueError("Candidate selection is missing or ambiguous")
        chosen = choices[0]
        group, mode, result, contract, metadata, link, original, report = _archive_authority(archive, manifest, chosen, request_index)
        source = archive.read(chosen["source"])
        if (hashlib.sha256(source).hexdigest() != chosen["source_hash"]
                or report.get("compilation_identity", {}).get("source") != source.decode()
                or report.get("compilation_identity", {}).get("options") != original.get("compiler_options")
                or link["compiler_options"] != original.get("compiler_options")):
            raise ValueError("Source/options differ from the actual archived execution")
        old_root = Path(manifest["study_directory_observed"])
        original_source = Path(original["source"])
        if not original_source.is_relative_to(old_root / group):
            raise ValueError("Archived source is outside the final group")
        if archive.read("evidence/" + str(original_source.relative_to(old_root))) != source:
            raise ValueError("Executed source bytes differ from selected artifact")
        portable = json.loads(archive.read(link["path"]))
        authenticate_portable_options(original["compiler_options"],portable.get("compiler_options",[]),
                                      link.get("options_relocations",[]),chosen)
        path_fields = {"source", "arguments", "compiler_options", "dependencies", "cache_dir", "compile_cache_key"}
        if {k:v for k,v in portable.items() if k not in path_fields} != {k:v for k,v in original.items() if k not in path_fields}:
            raise ValueError("Portable request changed non-path execution fields")
        args = original.get("arguments", [])
        if (len(args) != 4 or [(a.get("kind"), a.get("direction")) for a in args[:3]] !=
                [("buffer", "input"), ("buffer", "input"), ("buffer", "output")]
                or args[3].get("kind") != "scalar" or args[3].get("type") != "int32"
                or type(args[3].get("value")) is not int or not 0 < args[3]["value"] < 2**31):
            raise ValueError("Unsupported disjoint read-only float32 A/B/output-C/n ABI")
        n, operation = args[3]["value"], contract.operation
        if n not in result.get("shapes", []) or metadata.get("fixed_shape_domain") != result.get("shapes"):
            raise ValueError("Request shape is outside the frozen final contract")
        counts = (n*n, n*n, n*n) if operation == "matmul" else (n, 3 if operation == "convolution" else n, 1 if operation == "reduction" else n)
        if ([a.get("bytes") for a in args[:3]] != [v*4 for v in counts] or args[2].get("input_file") is not None
                or original.get("mode") != "run" or original.get("device") != 0
                or any(original.get(k) != v for k,v in {"warmup":0,"iterations":1,"series":1,"guard_outputs":True}.items())
                or original.get("sentinel_byte") not in (127, 255)):
            raise ValueError("Select one guarded archived run, without warmup/benchmark or ABI changes")
        for index, name in enumerate(("a", "b")):
            relative = f"sealed-inputs/final-{n}/{name}.bin"
            argument = portable["arguments"][index]
            raw = archive.read(argument["input_file"])
            if (Path(args[index]["input_file"]) != old_root / group / relative
                    or hashlib.sha256(raw).hexdigest() != result["input_hashes"].get(relative)
                    or len(raw) != args[index]["bytes"]
                    or {k:v for k,v in argument.items() if k != "input_file"} != {k:v for k,v in args[index].items() if k != "input_file"}):
                raise ValueError("Final input bytes/ABI are not bound to the archived contract")
        if portable["arguments"][2:] != args[2:]:
            raise ValueError("Output/scalar ABI changed during relocation")
        original_dependencies = original.get("dependencies", [])
        if len(original_dependencies) != len(portable.get("dependencies", [])):
            raise ValueError("Dependency relocation changed the compilation inputs")
        for old, new in zip(original_dependencies, portable["dependencies"]):
            original_dependency = Path(old) if Path(old).is_absolute() else original_source.parent / old
            if chosen["source_files"].get(os.path.normpath(str(original_dependency))) != new:
                raise ValueError("Dependency is not bound to the archived source-tree map")
    authenticate_quiescence(quiescence, quiescence_sha256, bundle_sha256, selection)
    if sha(bundle) != bundle_sha256:
        raise ValueError("Bundle changed before reconstruction")
    candidate, requests, _ = portable_export.reconstruct_candidate(bundle, selection, directory / "reconstructed")
    if (sha(bundle) != bundle_sha256 or candidate.artifact_hash != chosen["artifact_hash"]
            or candidate._source_hash != chosen["source_hash"] or candidate.candidate_id != chosen["candidate_id"]):
        raise ValueError("Reconstruction differs from the authenticated final artifact")
    request = requests[request_index]
    if candidate.kernel != operation or _source_contract_violations(candidate.source.read_text(), operation, contract.signature, "float32"):
        raise ValueError("Reconstructed source violates the closed ABI/precision contract")
    launch = candidate.launch_rule.apply({"n": n})
    if any(getattr(request, key) != value for key, value in launch.items()):
        raise ValueError("Reconstructed launch differs from the selected frozen rule")
    arrays = [np.fromfile(arg.input_file, dtype="<f4") for arg in request.arguments[:2]]
    if any(not np.isfinite(x).all() or np.any(np.abs(x)>1) for x in arrays):
        raise ValueError("Archived inputs violate the finite dense float32 input domain")
    expected = cpu_oracle(operation, *arrays, n)
    if expected.size != counts[2] or not np.isfinite(expected).all():
        raise ValueError("Independent CPU oracle failed")
    plan = {"kind":"reconstructed_study_finalist_acceptance_plan", "bundle":str(bundle), "bundle_sha256":bundle_sha256,
        "selection":chosen["selection"], "request_index":request_index, "operation":operation, "n":n,
        "group_id":result["group_id"], "mode":mode, "artifact_hash":candidate.artifact_hash, "source_hash":candidate._source_hash,
        "launch_rule":asdict(candidate.launch_rule), "parameters":dict(candidate.parameters), "signature":contract.signature,
        "original_contract_hash":result["contract_hash"], "workload_contract_hash":contract.contract_hash,
        "original_environment":result["environment"], "original_job_id":result["job_id"],
        "authority_member":"evidence/"+group+"/result.json", "origin_request":link["origin"],
        "quiescence":observation, "request":asdict(request), "request_identity":request_identity(request),
        "source_tree_sha256":{str(directory/"reconstructed"/member):manifest["files"][member]["sha256"]
                              for member in chosen["source_files"].values()},
        "oracle":{"implementation":"Independent NumPy float64 mathematics; float32 output", "numpy_version":np.__version__,
            "expected_sha256":hashlib.sha256(expected.tobytes()).hexdigest(), **operation_spec(operation)["numerical_tolerance"]},
        "harness_requests":1, "benchmark_requests":0, "model_calls":0, "new_scores_created":False,
        "scope":"One fresh correctness observation of exact reconstructed inputs/source/options/launch. No inherited qualification, benchmark, comparison, gain or model invocation."}
    write_immutable_json(directory / "plan.json", plan)
    return candidate, request, expected, plan


def execute_one(runner, candidate, request, expected, plan, run_id):
    candidate.check_unchanged()
    before = request_identity(request)
    if before != plan["request_identity"] or any(sha(path)!=pin for path,pin in plan["source_tree_sha256"].items()):
        raise ValueError("Frozen reconstructed execution inputs changed before the run")
    properties = runner.get_device_properties(0)
    if any(properties.get(key) in (None, "") for key in IDENTITY_KEYS):
        raise ValueError("Actual GPU/toolchain identity is incomplete")
    launch = candidate.launch_rule.apply({"n":plan["n"]}, properties)
    if any(getattr(request,key) != value for key,value in launch.items()):
        raise ValueError("Launch changed or is incompatible with the new GPU")
    report = runner.run(run_id, request)
    outputs = report.get("outputs", [])
    actual = np.fromfile(outputs[0]["path"],dtype="<f4") if len(outputs)==1 else np.array([],dtype="<f4")
    numerical = actual.size == expected.size and bool(np.isfinite(actual).all()) and bool(np.all(np.isclose(
        actual.astype(np.float64),expected.astype(np.float64),rtol=plan["oracle"]["rtol"],atol=plan["oracle"]["atol"])))
    candidate.check_unchanged()
    checks = {"independent_cpu_oracle":numerical, "native_success":report.get("ok") is True,
        "new_gpu_identity":all(report.get("environment",{}).get(k)==properties[k] for k in IDENTITY_KEYS),
        "inputs_readonly":report.get("guard_readonly_ok") is True, "outputs_poisoned":report.get("guard_outputs_poisoned") is True,
        "frozen_execution_inputs":before == request_identity(request)
            and all(sha(path)==pin for path,pin in plan["source_tree_sha256"].items()),
        "compiled_source_and_options":report.get("compilation_identity",{}).get("source")==candidate.source.read_text()
            and report.get("compilation_identity",{}).get("options")==request.compiler_options,
        "one_run_without_timing":report.get("latency_samples_ms")==[] and report.get("measurement_protocol",{})==
            {**report.get("measurement_protocol",{}),"warmup":0,"iterations":1,"series":1},
        "clean_worker":report.get("worker_reusable") is True and not report.get("worker_poisoned")}
    return {"ok":all(checks.values()),"checks":checks,"report":report,"actual_environment":device_identity(properties),
        "output_sha256":hashlib.sha256(actual.tobytes()).hexdigest(),"harness_requests":1,"benchmark_requests":0,
        "model_calls":0,"new_scores_created":False}


def run_acceptance(directory, bundle, bundle_sha256, execution_config, *, selection, request_index,
                   quiescence, quiescence_sha256, run_id, backend_factory=None):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError("Use a fresh output; reconcile any existing allocation intent rather than retry")
    candidate, request, expected, plan = prepare(directory,bundle,bundle_sha256,selection=selection,request_index=request_index,
        quiescence=quiescence,quiescence_sha256=quiescence_sha256)
    config = replace(SlurmConfig.load(execution_config),partition="long",walltime="48:00:00")
    backend = (backend_factory or SlurmBackend)(config,REPO,directory/"worker",session_id="study-reconstructed-"+uuid.uuid4().hex)
    provenance = controller_provenance(REPO,__file__,execution_config)
    receipt = {"ok":False,"job_id":None,"session_id":backend.session_id,"remote_dir":backend.remote_dir,
        "plan":plan,"source_provenance":provenance,"model_calls":0,"new_scores_created":False,
        "evidence_kind":"simulated_cpu_test" if getattr(backend,"simulation",False) else "romeo_execution"}
    receipt["allocation_intent"] = persist_allocation_intent(directory,backend,provenance)
    try:
        authenticate_quiescence(quiescence,quiescence_sha256,bundle_sha256,selection)
        if sha(bundle)!=bundle_sha256:
            raise ValueError("Bundle changed before submission")
        submitted = str(backend.start())
        owned = _local_session_job({"session":backend.session_id,"job_id":submitted},backend)
        if owned != submitted:
            raise ValueError("New allocation lacks its exact local submission acknowledgement")
        receipt["job_id"] = owned
        backend.wait_until_ready()
        receipt["execution"] = execute_one(backend.runner(),candidate,request,expected,plan,run_id)
        receipt["execution"]["checks"]["owned_slurm_job"] = str(receipt["execution"]["report"].get("slurm_job_id")) == owned
        receipt["execution"]["ok"] = all(receipt["execution"]["checks"].values())
    except Exception as error:
        receipt.update(error=str(error),traceback=traceback.format_exc())
    finally:
        try:
            owned = _local_session_job({"session":backend.session_id,"job_id":receipt["job_id"]},backend)
            if owned:
                if backend.job_id is None:
                    backend.job_id = owned  # Cleanup only; close() reauthenticates the remote ownership receipt.
                receipt["job_id"] = owned
                backend.close()
                receipt["terminal"] = backend.wait_until_finished(timeout=120)
            elif getattr(backend,"job_id",None):
                raise ValueError("Cleanup refused: in-memory job has no authenticated local acknowledgement")
        except Exception as error:
            receipt["cleanup_error"] = str(error)
        try:
            receipt["source_provenance_after"] = controller_provenance(REPO,__file__,execution_config)
            receipt["bundle_unchanged"] = sha(bundle)==bundle_sha256
            authenticate_quiescence(quiescence,quiescence_sha256,bundle_sha256,selection)
            ended = receipt.get("terminal",{})
            receipt["ok"] = bool(receipt.get("execution",{}).get("ok") and not receipt.get("error") and not receipt.get("cleanup_error")
                and receipt["bundle_unchanged"] and receipt["source_provenance_after"]["sources"]==provenance["sources"]
                and ended.get("terminal") is True and ended.get("state")=="COMPLETED" and ended.get("exit_code")=="0:0"
                and str(ended.get("job_id"))==receipt["job_id"])
        except Exception as error:
            receipt["verification_error"] = str(error)
        write_immutable_json(directory/"allocation-receipt.json",receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--bundle-sha256",required=True)
    parser.add_argument("--selection",required=True)
    parser.add_argument("--request-index",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--quiescence",type=Path)
    parser.add_argument("--quiescence-sha256")
    parser.add_argument("--execution-config",type=Path,default=REPO/"configs/romeo.json")
    parser.add_argument("--run-id",default="study-reconstructed-"+uuid.uuid4().hex[:12])
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args(argv)
    if not args.quiescence or not args.quiescence_sha256:
        if args.run:
            parser.error("--run requires an independently acquired --quiescence and its exact --quiescence-sha256")
        print(json.dumps({"submitted":False,"bundle_opened":False,"materialized":False,
            "status":"awaiting_pinned_quiescence","required_attestation_flags":QUIESCENT,
            "binding_fields":["schema_version=1","kind=manual_validation_release_quiescence","bundle_sha256","selection",
                "attested_by","attested_at_utc","evidence_pins[{path:absolute_public_json,sha256}]"],
            "scope":"No bundle, final input, oracle, model, GPU or scheduler accessed"}))
        return 0
    kwargs = dict(selection=args.selection,request_index=args.request_index,quiescence=args.quiescence,
                  quiescence_sha256=args.quiescence_sha256)
    if not args.run:
        prepare(args.output,args.bundle,args.bundle_sha256,**kwargs)
        print(json.dumps({"submitted":False,"plan":str(args.output.resolve()/"plan.json")}))
        return 0
    result = run_acceptance(args.output,args.bundle,args.bundle_sha256,args.execution_config,run_id=args.run_id,**kwargs)
    print(json.dumps({"ok":result["ok"],"job_id":result["job_id"],"receipt":str(args.output.resolve()/"allocation-receipt.json")}))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
