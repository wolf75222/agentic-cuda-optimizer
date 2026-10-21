"""Portable study evidence and candidate reconstruction, without GPU execution.

SQLite files are captured through the online backup API. Tree pickles are read
only from the explicitly selected trusted local study, never from an archive.
An active export is a protected partial snapshot, not a certification.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import zipfile

REPO = Path(__file__).resolve().parents[1]
# Existing workload helpers retain upstream absolute sibling imports. Append
# their directory after the package root so a fresh CLI can resolve them without
# shadowing the optimizer_agent package with its legacy optimizer_agent.py file.
if str(REPO/"optimizer_agent") not in sys.path:
    sys.path.append(str(REPO/"optimizer_agent"))
UPSTREAM_COMMIT = "1e9464da54dfc651337a97c3643bfeefec712bc1"
TERMINAL_TASK_STATES = {"completed", "failed", "recovery_required"}
ORIGINAL_BRIDGE_FILES = ("upstream_original.py", "frozen_launch.py", "codex_backend.py", "gain_observer.py")


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _plain(value):
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else {"nonfinite":str(value)}
    if isinstance(value,bytes):
        return {"bytes":len(value),"sha256":digest(value)}
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,dict):
        return {str(key):_plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple,set)):
        return [_plain(item) for item in value]
    if hasattr(value,"tolist"):
        return _plain(value.tolist())
    if is_dataclass(value):
        return {field.name:_plain(getattr(value,field.name)) for field in fields(value)}
    return {"python_type":type(value).__module__+"."+type(value).__qualname__,"repr":repr(value)}


def _safe_member(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in name:
        raise ValueError("Unsafe archive path: "+name)
    return path


def sqlite_backup(source,destination):
    """Capture one transactionally coherent DB, including committed WAL pages."""
    source,destination = Path(source).resolve(),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(source.as_uri()+"?mode=ro",uri=True) as incoming:
        with sqlite3.connect(destination) as outgoing:
            incoming.backup(outgoing)
            checks = [row[0] for row in outgoing.execute("PRAGMA integrity_check")]
            if checks != ["ok"]:
                raise ValueError("SQLite backup integrity check failed: "+str(checks))
    return {"method":"sqlite.Connection.backup","integrity_check":"ok","sha256":file_hash(destination)}


def readable_database(path):
    """Render durable ledger rows and the trusted local tree without cycles."""
    result = {"tables":{},"search_snapshots":[]}
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for table in tables:
            if not table.replace("_","").isalnum():
                raise ValueError("Unsupported SQLite table name")
            rows = []
            for row in db.execute('SELECT * FROM "'+table+'"'):
                values = dict(row)
                for key,value in values.items():
                    if isinstance(value,str) and value.lstrip().startswith(("{","[")):
                        try:
                            values[key] = json.loads(value)
                        except ValueError:
                            pass
                rows.append(_plain(values))
            result["tables"][table] = rows
        if "campaigns" in tables:
            for row in db.execute("SELECT run_id,snapshot FROM campaigns WHERE snapshot IS NOT NULL"):
                # The input is a trusted, local campaign selected by the user.
                # Reconstruction never deserializes these Python pickles.
                snapshot = pickle.loads(row["snapshot"])
                state = snapshot.get("tree")
                if state is not None and hasattr(state,"tree"):
                    nodes = [{"index":node.expand_idx,"parent_index":node.parent.expand_idx if node.parent else None,
                        "child_indices":[child.expand_idx for child in node.children],"trial_id":node.trial_id,
                        "score":node.score,"state":_plain(node.state)} for node in state.tree.get_nodes()]
                    readable = {"algorithm":"treequest_ab_mcts_a","nodes":nodes,
                        "pending_trials":_plain(state.trial_store.running_trials),
                        "finished_trials":_plain(state.trial_store.finished_trials),
                        "eligibility":_plain(snapshot["eligibility"]),"rng":_plain(snapshot.get("rng"))}
                else:
                    readable = {"algorithm":"greedy_parent","snapshot":_plain(snapshot)}
                result["search_snapshots"].append({"run_id":row["run_id"],**readable})
    return result


def _git(repo,*args):
    return subprocess.check_output(["git","-C",str(repo),*args],stderr=subprocess.PIPE)


class _Archive:
    def __init__(self,archive):
        self.archive,self.files = archive,{}

    def add(self,name,data):
        _safe_member(name)
        if name in self.files:
            if self.files[name]["sha256"] != digest(data):
                raise ValueError("Conflicting archive member: "+name)
            return name
        self.archive.writestr(name,data)
        self.files[name] = {"sha256":digest(data),"bytes":len(data)}
        return name

    def add_file(self,name,path):
        return self.add(name,Path(path).read_bytes())


def _source_snapshot(writer,repo):
    commit = _git(repo,"rev-parse","HEAD").decode().strip()
    status = _git(repo,"status","--porcelain=v1","-z").decode()
    writer.add("source/HEAD.tar",_git(repo,"archive","--format=tar",commit))
    writer.add("source/working-tree.patch",_git(repo,"diff","--binary","HEAD"))
    source = {"head":commit,"dirty":bool(status),"porcelain_status":status.split("\0"),
        "head_archive":"source/HEAD.tar","working_tree_patch":"source/working-tree.patch",
        "working_tree_patch_sha256":writer.files["source/working-tree.patch"]["sha256"],
        "claim":"HEAD archive is exact; working tree cleanliness is reported separately","bridge":{},"untracked_sources":{},
        "bridge_scope":"Local bridge bytes at export; recorded execution hashes are checked separately"}
    for relative in _git(repo,"ls-files","--others","--exclude-standard","-z").decode().split("\0"):
        path = Path(relative)
        if (not relative or not path.parts or path.parts[0] not in {"optimizer_agent","scripts","tests","docs","configs"}
                or path.suffix not in {".py",".md",".json",".toml",".txt",".yaml",".yml",".sh",".cu",".cuh",".h"}):
            continue
        actual = repo/path
        if actual.is_file() and not actual.is_symlink():
            member = writer.add_file("source/untracked/"+relative,actual)
            source["untracked_sources"][relative] = {"path":member,"sha256":writer.files[member]["sha256"]}
    for name in (*ORIGINAL_BRIDGE_FILES,"evaluation.py",
                 "benchmark_workloads.py","use_harness.py","portable_export.py"):
        path = repo/"optimizer_agent"/name
        if path.is_file():
            member = writer.add_file("source/bridge-working-tree/optimizer_agent/"+name,path)
            source["bridge"][name] = {"path":member,"sha256":writer.files[member]["sha256"]}
    try:
        actual = _git(repo,"rev-parse",UPSTREAM_COMMIT+"^{commit}").decode().strip()
        if actual != UPSTREAM_COMMIT:
            raise ValueError("Upstream commit mismatch")
        paths = _git(repo,"ls-tree","-r","--name-only",UPSTREAM_COMMIT,"optimizer_agent").decode().splitlines()
        hashes = {}
        for path in paths:
            if path.endswith(".py"):
                data = _git(repo,"show",UPSTREAM_COMMIT+":"+path)
                hashes[path] = digest(data)
                writer.add("source/upstream-original/"+path,data)
        source["upstream_original"] = {"commit":actual,"authentication":"Git commit object and per-file SHA-256",
            "source_hashes":hashes,"path":"source/upstream-original"}
    except (subprocess.CalledProcessError,ValueError) as error:
        source["upstream_original"] = {"commit":UPSTREAM_COMMIT,"status":"unavailable","error":str(error)}
    return source


def _historical_adapter_archives(writer,repo):
    """Authenticate captured historical bytes, without inferring their execution.

    source.json is an observed receipt, not an externally precommitted digest.
    Its complete file map and the TAR bytes must agree with one local pinned
    Git commit. No archive members are extracted or executed.
    """
    prefixes = sorted({name.rsplit("/",1)[0] for name in writer.files
        if name.startswith("evidence/controller-sources/") and name.endswith(("/source.json","/source.tar.gz"))})
    validated,rejected = [],[]
    for prefix in prefixes:
        try:
            parts = PurePosixPath(prefix).parts
            if len(parts) != 3 or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}",parts[-1]):
                raise ValueError("Historical source directory is not a full pinned Git revision")
            revision = parts[-1]
            metadata_member,archive_member = prefix+"/source.json",prefix+"/source.tar.gz"
            if metadata_member not in writer.files or archive_member not in writer.files:
                raise ValueError("Historical source requires both source.json and source.tar.gz")
            metadata_bytes = writer.archive.read(metadata_member)
            archive_bytes = writer.archive.read(archive_member)
            metadata = json.loads(metadata_bytes)
            if not isinstance(metadata,dict) or metadata.get("head") != revision or metadata.get("dirty") != []:
                raise ValueError("Historical source head/directory mismatch or checkout not recorded clean")
            if metadata.get("archive_sha256") != digest(archive_bytes):
                raise ValueError("Historical source TAR checksum mismatch")
            if _git(repo,"rev-parse",revision+"^{commit}").decode().strip() != revision:
                raise ValueError("Historical source Git commit mismatch")
            object_format = _git(repo,"rev-parse","--show-object-format").decode().strip()
            if object_format not in {"sha1","sha256"}:
                raise ValueError("Unsupported historical Git object format")
            git_files = {}
            for entry in _git(repo,"ls-tree","-r","-z",revision).split(b"\0"):
                if not entry:
                    continue
                identity,name = entry.split(b"\t",1)
                mode,kind,object_id = identity.decode().split()
                if mode not in {"100644","100755"} or kind != "blob":
                    raise ValueError("Historical Git tree contains a non-regular source member")
                git_files[name.decode()] = object_id
            hashes = metadata.get("source_hashes")
            if (not isinstance(hashes,dict) or set(hashes) != set(git_files)
                    or any(not isinstance(value,str) or not re.fullmatch(r"[0-9a-f]{64}",value) for value in hashes.values())):
                raise ValueError("Historical source manifest does not cover the exact pinned Git tree")
            seen,actual_hashes = set(),{}
            with tarfile.open(fileobj=io.BytesIO(archive_bytes),mode="r:gz") as incoming:
                for member in incoming:
                    path = _safe_member(member.name)
                    name = str(path)
                    if name in seen:
                        raise ValueError("Duplicate historical TAR member: "+name)
                    seen.add(name)
                    if member.isdir():
                        if not any(item.startswith(name+"/") for item in git_files):
                            raise ValueError("Unexpected historical TAR directory: "+name)
                        continue
                    if not member.isreg() or name not in git_files:
                        raise ValueError("Historical TAR contains a link or unexpected non-regular member: "+name)
                    data = incoming.extractfile(member).read()
                    blob_id = hashlib.new(object_format,b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
                    if blob_id != git_files[name] or digest(data) != hashes[name]:
                        raise ValueError("Historical TAR/manifest/pinned Git bytes mismatch: "+name)
                    actual_hashes[name] = digest(data)
            if set(actual_hashes) != set(git_files):
                raise ValueError("Historical TAR does not contain the complete pinned Git tree")
            bridge_hashes = {name:actual_hashes.get("optimizer_agent/"+name) for name in ORIGINAL_BRIDGE_FILES}
            if not all(bridge_hashes.values()):
                raise ValueError("Historical source is missing a required original bridge file")
            validated.append({"git_revision":revision,"source_metadata":metadata_member,
                "source_metadata_sha256":digest(metadata_bytes),"archive":archive_member,
                "archive_sha256":digest(archive_bytes),"adapter_source_hashes":bridge_hashes,
                "verified_git_file_count":len(actual_hashes),"recorded_status":metadata.get("status"),
                "authentication":"Complete captured TAR and observed source manifest agree with the pinned local Git tree",
                "metadata_digest_scope":"SHA-256 of the source.json bytes included in this export; no external precommitment claimed"})
        except (OSError,ValueError,KeyError,TypeError,UnicodeDecodeError,tarfile.TarError,subprocess.CalledProcessError) as error:
            rejected.append({"path":prefix.removeprefix("evidence/"),"error":str(error)})
    return {"validated":validated,"rejected":rejected}


def _recorded_bridge_sources(writer,repo,study,documents,source,errors):
    """Resolve only the recorded adapter files; never choose an execution SHA."""
    source["recorded_bridge_hash_checks"] = []
    source["recorded_bridge_source_bindings"] = []
    historical = None
    for path,document in documents:
        if not isinstance(document,dict) or document.get("method") != "true_upstream_original":
            continue
        relative = str(path.relative_to(study))
        receipt_sha = writer.files["evidence/"+relative]["sha256"]
        expected = document.get("adapter_source_hashes")
        valid = (isinstance(expected,dict) and set(expected) == set(ORIGINAL_BRIDGE_FILES)
            and all(isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) for value in expected.values()))
        binding = {"receipt":relative,"receipt_sha256":receipt_sha,"controller_revision":None,
            "scope":"Exact recorded adapter bytes only; task-to-controller revision is not established",
            "current_adapter_bytes_match":False,"historical_adapter_bytes_resolved":False,
            "adapter_bytes_resolved":False,"matching_historical_archives":[]}
        source["recorded_bridge_source_bindings"].append(binding)
        if not valid:
            errors.append({"path":relative,"error":"Original source receipt requires the exact four SHA-256 adapter_source_hashes"})
            continue
        checks = []
        for name,recorded in expected.items():
            actual = source["bridge"].get(name,{}).get("sha256")
            checks.append({"receipt":relative,"receipt_sha256":receipt_sha,"file":name,
                "recorded_sha256":recorded,"archived_sha256":actual,"matches":actual == recorded,
                "current_matches":actual == recorded})
        binding["current_adapter_bytes_match"] = all(check["current_matches"] for check in checks)
        if not binding["current_adapter_bytes_match"]:
            if historical is None:
                historical = _historical_adapter_archives(writer,repo)
                source["historical_controller_source_archives"] = historical
                errors.extend(historical["rejected"])
            binding["matching_historical_archives"] = [item for item in historical["validated"]
                if item["adapter_source_hashes"] == expected]
            binding["historical_adapter_bytes_resolved"] = bool(binding["matching_historical_archives"])
        binding["adapter_bytes_resolved"] = binding["current_adapter_bytes_match"] or binding["historical_adapter_bytes_resolved"]
        for check in checks:
            check["historical_adapter_bytes_resolved"] = binding["historical_adapter_bytes_resolved"]
        source["recorded_bridge_hash_checks"].extend(checks)
        if not binding["adapter_bytes_resolved"]:
            errors.append({"path":relative,"error":"Recorded original adapter bytes differ from the current bridge and have no authenticated historical source archive"})


def _walk(value):
    if isinstance(value,dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value,list):
        for item in value:
            yield from _walk(item)


def _reference_file_kind(path):
    if "research" not in path.parts:
        return None
    if "sources" in path.parts:
        return "downloaded_body_cache"
    if "passages" in path.parts:
        return "supplied_passage_text"
    if path.name.endswith("-supplied-content.json"):
        return "synthesis_input_with_reference_text"
    return None


def _reference_metadata(document,origin):
    fields = ("source_id","kind","url","repository","commit_or_version","file","symbol",
        "section_or_locator","provenance","license_if_code_reused","content_sha256","cache_path",
        "retrieved_at","read_characters","read_receipt_version","launcher_status")
    passage_fields = ("passage_id","role","line_start","line_end","fragment","characters","sha256","text_sha256",
        "content_sha256","cache_path","truncated")
    for row in _walk(document):
        if not {"source_id","content_sha256","kind"} <= row.keys():
            continue
        yield {"origin_receipt":origin,"source":{key:row[key] for key in fields if key in row},
            "passages":[{key:part[key] for key in passage_fields if key in part}
                for part in row.get("passages",[]) if isinstance(part,dict)]}


def _candidate_records(documents):
    records = []
    pilot_comparisons = {path.parent.resolve() for path,document in documents
        if path.name == "comparison.json" and isinstance(document,dict)
        and document.get("kind") == "independent_frozen_kernel_comparison"}
    for path,document in documents:
        if path.name == "frozen-candidates.json":
            records.extend((path,name,receipt) for name,receipt in document.items() if isinstance(receipt,dict))
        elif path.name in ("candidate.json","best-launch.json") and isinstance(document,dict):
            records.append((path,document.get("candidate_id","selected"),document))
        elif (path.name in ("plan.json","comparison.json") and isinstance(document,dict)
                and document.get("kind") == "independent_frozen_kernel_comparison"):
            # The completed comparison is the authoritative copy of the same
            # predeclared frozen candidates; avoid exporting its plan twice.
            if path.name == "plan.json" and path.parent.resolve() in pilot_comparisons:
                continue
            records.extend((path,name,receipt) for name,receipt in document.get("candidates",{}).items()
                           if isinstance(receipt,dict))
    return records


def _source_path(receipt,path):
    value = receipt.get("source_path",receipt.get("source"))
    if isinstance(value,str) and "\n" not in value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = path.parent/candidate
        if candidate.is_file():
            return candidate.resolve()
    for name in ("best.cu","frozen-original.cu","kernel.cu"):
        candidate = path.parent/name
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Selected candidate source is unavailable")


def _context(path,documents):
    indexed = {item.resolve():value for item,value in documents}
    for parent in (path.parent,*path.parents):
        for name in ("comparison.json","result.json","manifest.json","contract.json","group-result.pending.json"):
            value = indexed.get((parent/name).resolve())
            if isinstance(value,dict) and (value.get("contract") or value.get("config") or value.get("environment")):
                context = dict(value)
                if "environment" not in context:
                    context["environment"] = value.get("hardware_identity") or value.get("hardware",{}).get("device_properties")
                return context
    return {}


def _include_options(options,base):
    result = []
    previous = None
    for index,option in enumerate(options):
        prefix,value = None,None
        if previous is not None:
            prefix,value = "",option
            previous = None
        elif option in ("-I","--include-path"):
            previous = option
            continue
        elif option.startswith("--include-path="):
            prefix,value = "--include-path=",option.split("=",1)[1]
        elif option.startswith("-I") and len(option)>2:
            prefix,value = "-I",option[2:]
        if value is not None:
            directory = Path(value)
            if not directory.is_absolute():
                raise ValueError("Relative compiler include paths require an authenticated compilation working directory")
            result.append({"index":index,"prefix":prefix,"directory":directory.resolve()})
    if previous is not None:
        raise ValueError("Compiler include-path option has no argument")
    return result


def _portable_candidate(writer,study,path,name,receipt,documents,requests):
    from .evaluation import Candidate
    from .frozen_launch import parse_launch_rule
    source = _source_path(receipt,path)
    context = _context(path,documents)
    kernel = receipt.get("kernel") or context.get("config",{}).get("workload") or context.get("contract",{}).get("operation")
    if not kernel:
        raise ValueError("Candidate kernel entry name is missing")
    candidate = Candidate(receipt["candidate_id"],source,kernel,parse_launch_rule(receipt["launch_rule"]),receipt.get("parameters",{}))
    if receipt.get("source_hash",candidate._source_hash) != candidate._source_hash or receipt.get("artifact_hash") != candidate.artifact_hash:
        raise ValueError("Selected source/launch/parameters do not match frozen hashes")
    selection = str(path.relative_to(study))+"#"+str(name)
    identifier = hashlib.sha256(selection.encode()).hexdigest()[:20]
    prefix = "candidates/"+identifier+"/"
    related = []
    for location,request in requests:
        if request.get("kernel") != kernel:
            continue
        original = Path(request["source"])
        if not original.is_absolute():
            original = location.parent/original
        if not original.is_file() or file_hash(original) != candidate._source_hash:
            continue
        scalars = [item for item in request["arguments"] if item.get("kind") == "scalar"]
        n = scalars[-1].get("value") if scalars else None
        try:
            launch = candidate.launch_rule.apply({"n":n})
        except ValueError:
            continue
        if any(request.get(key) != value for key,value in launch.items()):
            continue
        related.append((location,request,original))
    source_files = {source}
    dependency_paths = set()
    include_directories = set()
    from .compile_cache import compilation_manifest, validate_dependency_options
    for location,request,original in related:
        validate_dependency_options(request.get("compiler_options",[]))
        include_directories.update(row["directory"] for row in _include_options(request.get("compiler_options",[]),original.parent))
        for value in request.get("dependencies",[]):
            dependency = Path(value)
            if not dependency.is_absolute():
                dependency = original.parent/dependency
            if not dependency.is_file():
                raise ValueError("Declared compilation dependency is unavailable: "+str(dependency))
            dependency_paths.add(dependency.resolve())
    # Reuse local quoted/angle discovery. Unresolved toolkit angle includes
    # remain represented by the immutable toolchain/environment metadata.
    compilation = compilation_manifest(source,dependencies=dependency_paths,
        options=["--include-path="+str(path) for path in sorted(include_directories)],architecture="archive",
        toolchain={"scope":"dependency discovery only; actual options retained per request"})
    source_files.update(Path(value) for value in compilation["files"])
    common = Path(os.path.commonpath([str(item.parent) for item in source_files]+[str(path) for path in include_directories]))
    mapping = {}
    for item in sorted(source_files):
        member = prefix+"source-tree/"+str(item.relative_to(common))
        mapping[str(item.resolve())] = writer.add_file(member,item)
    portable_requests = []
    seen = set()
    for location,request,original in related:
        data = json.loads(json.dumps(request))
        data["source"] = mapping[str(source)]
        data["cache_dir"],data["compile_cache_key"] = None,None
        for index,argument in enumerate(data["arguments"]):
            if argument.get("input_file"):
                input_path = Path(argument["input_file"])
                if not input_path.is_absolute():
                    input_path = location.parent/input_path
                payload = input_path.read_bytes()
                if len(payload) != argument.get("bytes"):
                    raise ValueError("Input bytes differ from request allocation")
                argument["input_file"] = writer.add(prefix+"inputs/"+digest(payload)+".bin",payload)
        data["dependencies"] = [mapping[str((original.parent/dep).resolve() if not Path(dep).is_absolute() else Path(dep).resolve())]
            for dep in data.get("dependencies",[])]
        relocations = []
        original_options = request.get("compiler_options",[])
        for option in _include_options(original_options,original.parent):
            relative_directory = prefix+"source-tree/"+str(option["directory"].relative_to(common))
            data["compiler_options"][option["index"]] = option["prefix"]+relative_directory
            relocations.append({"index":option["index"],"prefix":option["prefix"],"path":relative_directory,
                "original":original_options[option["index"]]})
        identity = digest(encoded(data))
        if identity in seen:
            continue
        seen.add(identity)
        member = writer.add(prefix+"requests/"+str(len(portable_requests))+".json",encoded(data))
        portable_requests.append({"path":member,"origin":str(location.relative_to(study)),
            "compiler_options":original_options,"options_sha256":digest(encoded(original_options)),
            "portable_options_sha256":digest(encoded(data.get("compiler_options",[]))),"options_relocations":relocations,
            "dependencies":data["dependencies"]})
    result = {"id":identifier,"selection":selection,"candidate_id":candidate.candidate_id,"kernel":kernel,
        "source":mapping[str(source)],"source_hash":candidate._source_hash,"artifact_hash":candidate.artifact_hash,
        "launch_rule":asdict(candidate.launch_rule),"parameters":candidate.parameters,
        "requests":portable_requests,"reconstructable":bool(portable_requests),
        "contract":context.get("contract"),"contract_hash":context.get("contract_hash"),
        "environment":context.get("environment"),"evidence_context":str(path.relative_to(study)),
        "certification":"Evidence copied without granting a new certification",
        "source_files":mapping,"prefix":prefix}
    writer.add(prefix+"candidate.json",encoded(result))
    return result


def export_study(directory,output,*,repo_dir=REPO):
    """Export a complete or explicitly partial local study; submit no work."""
    study,output,repo = Path(directory).resolve(),Path(output).resolve(),Path(repo_dir).resolve()
    if not study.is_dir() or output == study or output.is_relative_to(study):
        raise ValueError("Output must be outside the existing study directory")
    output.mkdir(parents=True,exist_ok=True)
    os.chmod(output,0o700)
    archive_path = output/"study-evidence.zip"
    temporary_archive = output/"study-evidence.zip.tmp"
    errors,documents,requests,databases,observed_jobs,inflight_trials = [],[],[],[],[],[]
    omitted_references,reference_metadata = [],[]
    files = sorted(path for path in study.rglob("*") if path.is_file() and not path.is_symlink())
    before = {str(path): (path.stat().st_size,path.stat().st_mtime_ns) for path in files}
    with ExitStack() as stack,zipfile.ZipFile(temporary_archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        controller_active = False
        for path in [study/"controller.lock",*sorted((study/"task-locks").glob("*.lock"))]:
            if path.exists():
                handle = stack.enter_context(path.open("r"))
                try:
                    fcntl.flock(handle,fcntl.LOCK_SH|fcntl.LOCK_NB)
                except BlockingIOError:
                    controller_active = True
        writer = _Archive(archive)
        source = _source_snapshot(writer,repo)
        for path in files:
            relative = path.relative_to(study)
            if path.name.endswith(("-wal","-shm")) or path.suffix == ".lock" or path.name == "controller.lock":
                continue
            reference_kind = _reference_file_kind(relative)
            if reference_kind:
                data = path.read_bytes()
                omitted_references.append({"original_path":str(relative),"sha256":digest(data),"bytes":len(data),
                    "kind":reference_kind,"disposition":"body omitted; original receipt/hash remains unchanged"})
                if path.suffix == ".json":
                    try:
                        reference_metadata.extend(_reference_metadata(json.loads(data),str(relative)))
                    except (ValueError,UnicodeDecodeError):
                        pass
                continue
            if path.suffix in (".sqlite",".db"):
                with tempfile.TemporaryDirectory(prefix="cuda-export-db-") as temporary:
                    backup = Path(temporary)/path.name
                    try:
                        metadata = sqlite_backup(path,backup)
                        member = writer.add_file("evidence/"+str(relative),backup)
                        readable = readable_database(backup)
                        writer.add("observability/"+str(relative)+".json",encoded(readable))
                        databases.append({"path":member,**metadata})
                        for snapshot in readable["search_snapshots"]:
                            pending = snapshot.get("pending_trials",snapshot.get("snapshot",{}).get("pending",{}))
                            if pending:
                                inflight_trials.append({"database":str(relative),"run_id":snapshot["run_id"],"pending_trial_ids":list(pending)})
                        for task in readable["tables"].get("tasks",[]):
                            if task.get("job_id") and task.get("state") not in TERMINAL_TASK_STATES:
                                observed_jobs.append({"job_id":task["job_id"],"task_id":task.get("id"),"state":task.get("state"),
                                    "source":"transactional local suite journal; no scheduler query"})
                    except Exception as error:
                        errors.append({"path":str(relative),"error":str(error)})
                continue
            try:
                data = path.read_bytes()
                writer.add("evidence/"+str(relative),data)
                if path.suffix == ".json":
                    try:
                        document = json.loads(data)
                    except (ValueError,UnicodeDecodeError):
                        continue
                    documents.append((path,document))
                    reference_metadata.extend(_reference_metadata(document,str(relative)))
                    for row in _walk(document):
                        if {"source","kernel","arguments","grid","block"} <= row.keys() and isinstance(row["arguments"],list):
                            requests.append((path,row))
            except (OSError,ValueError) as error:
                errors.append({"path":str(relative),"error":str(error)})
        candidates = []
        for path,name,receipt in _candidate_records(documents):
            try:
                candidates.append(_portable_candidate(writer,study,path,name,receipt,documents,requests))
                if not candidates[-1]["reconstructable"]:
                    errors.append({"path":str(path.relative_to(study)),"candidate":str(name),"error":"No matching real request/input receipt was available"})
            except Exception as error:
                errors.append({"path":str(path.relative_to(study)),"candidate":str(name),"error":str(error)})
        suite_rows = []
        for database in databases:
            if database["path"].endswith("suite.sqlite"):
                readable = json.loads(archive.read("observability/"+database["path"].removeprefix("evidence/")+".json"))
                suite_rows.extend(readable["tables"].get("tasks",[]))
        terminal_ids = {str(row["job_id"]) for _,document in documents for row in _walk(document)
            if row.get("job_id") and row.get("state") in {"COMPLETED","FAILED","CANCELLED","TIMEOUT","NODE_FAIL","OUT_OF_MEMORY","PREEMPTED","BOOT_FAIL"}}
        for path,document in documents:
            if path.name == "slurm-session.json" and isinstance(document,dict) and document.get("job_id"):
                job_id = str(document["job_id"])
                if job_id not in terminal_ids and not any(str(row["job_id"]) == job_id for row in observed_jobs):
                    observed_jobs.append({"job_id":job_id,"state":"no_cached_terminal_receipt",
                        "source":str(path.relative_to(study)),"live_scheduler_state":"not_queried"})
        complete = bool(suite_rows) and all(row.get("state") == "completed" for row in suite_rows) and any(row.get("kind") == "final" for row in suite_rows)
        if not suite_rows:
            summaries = [value for path,value in documents if path.name == "comparison.json"]
            terminals = [value for path,value in documents if path.name == "worker-terminal.json"]
            complete = any(value.get("ok") is True and value.get("status") == "completed" and any(
                terminal.get("terminal") and terminal.get("state") == "COMPLETED" and terminal.get("exit_code") == "0:0"
                and str(terminal.get("job_id")) == str(value.get("job_id")) for terminal in terminals) for value in summaries)
        pilot_format = any(isinstance(value,dict) and value.get("kind") == "independent_frozen_kernel_comparison"
                           for path,value in documents if path.name == "comparison.json")
        if pilot_format:
            # Byte-exact reconstruction is supported independently of export
            # completion. An ok flag and embedded terminal are not sufficient
            # to authenticate the pilot's full sanitizer/measurement receipts.
            complete = False
        if complete and not candidates:
            errors.append({"error":"Completed journal has no portable frozen candidate receipts"})
        after = {str(path):(path.stat().st_size,path.stat().st_mtime_ns) for path in study.rglob("*") if path.is_file() and not path.is_symlink()}
        changed = [name for name in set(before)|set(after) if before.get(name) != after.get(name) and not name.endswith(("-wal","-shm",".lock"))]
        requires_original = any(row.get("method") == "true_upstream_original" for _,value in documents for row in _walk(value))
        if requires_original and source["upstream_original"].get("status") == "unavailable":
            errors.append({"error":"Pinned original source object is unavailable"})
        _recorded_bridge_sources(writer,repo,study,documents,source,errors)
        source["head_after_export"] = _git(repo,"rev-parse","HEAD").decode().strip()
        source["dirty_after_export"] = bool(_git(repo,"status","--porcelain=v1","-z"))
        source_changed = source["head_after_export"] != source["head"] or digest(_git(repo,"diff","--binary","HEAD")) != source["working_tree_patch_sha256"]
        for name,identity in source["bridge"].items():
            if not (repo/"optimizer_agent"/name).is_file() or file_hash(repo/"optimizer_agent"/name) != identity["sha256"]:
                source_changed = True
        reference_index = {"schema_version":1,"sources":list({digest(encoded(row)):row for row in reference_metadata}.values()),
            "omitted_files":omitted_references,"retrieval_performed_by_exporter":False,
            "generation_guard_ready":False,"automatic_generation_resume_available":False,
            "kernel_reconstruction_requires_reference_download":False,
            "rehydration_required_for_generation":bool(omitted_references or reference_metadata),
            "instructions":["Obtain exact source versions from the recorded URLs/commits and authenticate body and passage hashes.",
                "Restore the referenced cache and exact supplied-content receipts under a controlled relocation workflow before rerunning research_ready.",
                "This archive preserves historical receipt metadata unchanged. Export and kernel reconstruction perform no retrieval, rereading or generation-guard approval."]}
        reference_member = writer.add("reference-rehydration.json",encoded(reference_index))
        partial = not complete or controller_active or bool(observed_jobs) or bool(inflight_trials) or bool(errors) or bool(changed) or source_changed
        manifest = {"schema_version":1,"kind":"portable_study_export","status":"partial" if partial else "complete",
            "created_at":datetime.now(timezone.utc).isoformat(),"study_directory_observed":str(study),"source":source,
            "database_snapshots":databases,"database_consistency":"Each backup is transactionally coherent; no cross-database atomicity claim during active execution",
            "controller_or_worker_lock_observed_active":controller_active,"active_jobs_observed":observed_jobs,
            "inflight_search_trials":inflight_trials,"source_head_changed_during_export":source_changed,
            "scheduler_observation":"Cached local receipts/journal only; scheduler was not queried",
            "changed_files_during_export":changed,"errors":errors,"candidates":candidates,"files":writer.files,
            "certification":"No certification is created by exporting; inspect original final qualification/confirmation receipts",
            "completion_assessment":{"pilot_comparison_detected":pilot_format,
                "pilot_qualification_authenticated":False if pilot_format else None,
                "note":"Pilot artifacts/requests are reconstructible; full qualification authentication is not implemented by this exporter, so the snapshot remains partial" if pilot_format else None},
            "private_partial_snapshot":partial,"holdout_state":"SQLite backups can contain unrevealed seeds; keep partial exports protected",
            "reference_policy":"Downloaded bodies, supplied passage text and synthesis source-text inputs excluded; exact metadata receipts/hashes retained",
            "reference_rehydration":reference_member,"reference_files_omitted":omitted_references,
            "generation_resume":{"guard_ready":reference_index["generation_guard_ready"],
                "rehydration_required":reference_index["rehydration_required_for_generation"],
                "automatic_resume_available":False,"retrieval_performed":False}}
        archive.writestr("export-manifest.json",encoded(manifest))
    temporary_archive.replace(archive_path)
    os.chmod(archive_path,0o600)
    (output/"export-manifest.json").write_bytes(encoded(manifest))
    os.chmod(output/"export-manifest.json",0o600)
    return manifest


def reconstruct_candidate(bundle,selection,destination):
    """Verify/rebase an archived candidate and real requests; never execute it."""
    from .evaluation import Candidate
    from .frozen_launch import parse_launch_rule
    from .use_harness import KernelRequest
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Reconstruction requires a fresh directory")
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("export-manifest.json"))
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"])|{"export-manifest.json"}:
            raise ValueError("Unexpected, duplicate or missing archive members")
        for name in names:
            _safe_member(name)
        choices = [item for item in manifest["candidates"] if selection in (item["id"],item["selection"])]
        if len(choices) != 1:
            raise ValueError("Candidate selection is missing or ambiguous")
        chosen = choices[0]
        if not chosen["requests"]:
            raise ValueError("Candidate has no archived executable request")
        def verified_member(name):
            _safe_member(name)
            if name not in manifest["files"] or not name.startswith(chosen["prefix"]):
                raise ValueError("Request path is outside the verified candidate files")
            return destination/name
        verified_member(chosen["source"])
        for row in chosen["requests"]:
            verified_member(row["path"])
        for name,identity in manifest["files"].items():
            if name.startswith(chosen["prefix"]):
                data = archive.read(name)
                if digest(data) != identity["sha256"] or len(data) != identity["bytes"]:
                    raise ValueError("Archive checksum mismatch: "+name)
                target = destination/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data)
        candidate = Candidate(chosen["candidate_id"],destination/chosen["source"],chosen["kernel"],
            parse_launch_rule(chosen["launch_rule"]),chosen["parameters"])
        if candidate._source_hash != chosen["source_hash"] or candidate.artifact_hash != chosen["artifact_hash"]:
            raise ValueError("Reconstructed source/launch/parameters differ from frozen artifact")
        reconstructed = []
        for index,row in enumerate(chosen["requests"]):
            path = destination/row["path"]
            data = json.loads(path.read_text())
            if data["source"] != chosen["source"]:
                raise ValueError("Request source differs from selected frozen artifact")
            if (digest(encoded(row["compiler_options"])) != row["options_sha256"] or
                    digest(encoded(data.get("compiler_options",[]))) != row.get("portable_options_sha256",row["options_sha256"])):
                raise ValueError("Reconstructed compiler options differ")
            for option in row.get("options_relocations",[]):
                _safe_member(option["path"])
                if not option["path"].startswith(chosen["prefix"]):
                    raise ValueError("Compiler include path escaped the selected candidate")
                directory = destination/option["path"]
                directory.mkdir(parents=True,exist_ok=True)
                data["compiler_options"][option["index"]] = option["prefix"]+str(directory)
            for key in ("source",):
                data[key] = str(verified_member(data[key]))
            for argument in data["arguments"]:
                if argument.get("input_file"):
                    member = argument["input_file"]
                    if member not in manifest["files"] or not member.startswith(chosen["prefix"]):
                        raise ValueError("Request input is outside the verified candidate files")
                    argument["input_file"] = str(verified_member(member))
            data["dependencies"] = [str(verified_member(value)) for value in data.get("dependencies",[])]
            reconstructed.append(KernelRequest.from_dict(data))
            path.write_bytes(encoded(dict(data,id="reconstructed-"+chosen["id"]+"-"+str(index),results_dir=str(destination/"results"))))
        (destination/"reconstruction.json").write_bytes(encoded({"candidate":chosen,"export_status":manifest["status"],
            "executed":False,"qualification":"Archived qualification does not qualify a new execution environment",
            "generation_resume":manifest.get("generation_resume"),
            "research_note":"Kernel reconstruction does not rehydrate external research sources or approve research_ready"}))
    return candidate,reconstructed,manifest
