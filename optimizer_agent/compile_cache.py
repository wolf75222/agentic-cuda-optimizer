"""Content-addressed compilation manifests. Measurement evidence is never cached here."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_dependency_options(options):
    """Reject forced includes until their compiler-specific lookup is modeled."""
    for option in options:
        if not isinstance(option, str):
            raise ValueError("Compiler options must be strings")
        if option.startswith(("--pre-include", "-include")):
            raise ValueError("Forced-include compiler options are unsupported by dependency discovery")


def compilation_manifest(source: Path, *, dependencies=(), options=(), architecture: str,
                         toolchain: dict, abi: str = "cuda-driver-pointer64-cxx17-v1") -> dict:
    """Hash explicit and recursively found local quoted/angle dependencies.

    Unresolved angle headers belong to the recorded compiler/toolkit identity;
    local include-path headers must not silently become system dependencies.
    Ambiguous local resolution, missing quoted includes and nonliteral include
    directives fail closed. This is not a preprocessor. The caller supplies
    the execution worker's toolchain, never local SSH-host data.
    """
    if not architecture or not toolchain:
        raise ValueError("A target architecture and execution toolchain identity are required")
    options = tuple(options)
    validate_dependency_options(options)
    visited: dict[str, str] = {}
    roots = [Path(source).resolve(), *(Path(p).resolve() for p in dependencies)]
    include_dirs = [p.parent for p in roots]
    pending_include = False
    for option in options:
        value = None
        if pending_include:
            value, pending_include = option, False
        elif option in ("-I", "--include-path"):
            pending_include = True
        elif option.startswith("--include-path="):
            value = option.split("=", 1)[1]
        elif option.startswith("-I") and len(option) > 2:
            value = option[2:]
        if value is not None:
            if not value or value.startswith("-"):
                raise ValueError("Compiler include-path option has no directory")
            include_dirs.append(Path(value).resolve())
    if pending_include:
        raise ValueError("Compiler include-path option has no argument")

    def visit(path: Path):
        if str(path) in visited:
            return
        data = path.read_bytes()
        visited[str(path)] = hashlib.sha256(data).hexdigest()
        for operand in re.findall(r'^\s*#\s*include\b([^\n]*)', data.decode(errors="replace"), re.MULTILINE):
            literal = re.fullmatch(r'\s*(?:"([^"\n]+)"|<([^>\n]+)>)\s*(?://.*|/\*.*\*/\s*)?', operand)
            if literal is None:
                raise ValueError("Nonliteral or unsupported include directive in compilation dependency discovery")
            quoted, angled = literal.groups()
            name = quoted or angled
            matches = set()
            for directory in (path.parent, *include_dirs):
                candidate = directory / name
                if candidate.is_symlink() or candidate.exists():
                    try:
                        resolved = candidate.resolve(strict=True)
                    except (OSError, RuntimeError) as error:
                        raise ValueError(f"Unresolvable local compilation dependency: {name}") from error
                    if not resolved.is_file():
                        raise ValueError(f"Local compilation dependency is not a file: {name}")
                    matches.add(resolved)
            if len(matches) > 1:
                raise ValueError(f"Ambiguous local compilation dependency: {name}")
            if not matches and quoted:
                raise ValueError(f"Unresolved compilation dependency: {name}")
            if matches:
                visit(matches.pop())

    for root in roots:
        visit(root)
    manifest = {"schema_version": 1, "source_hash": visited[str(roots[0])], "files": visited,
                "options": list(options), "architecture": architecture, "toolchain": toolchain, "abi": abi}
    return {**manifest, "key": content_hash(manifest)}


class CompilationCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def lookup(self, manifest: dict) -> Path | None:
        target = self.directory / (manifest["key"] + ".cubin")
        receipt = target.with_suffix(".json")
        if not target.is_file() or not receipt.is_file():
            return None
        metadata = json.loads(receipt.read_text())
        if metadata.get("manifest") != manifest or metadata.get("sha256") != hashlib.sha256(target.read_bytes()).hexdigest():
            return None
        return target

    def store(self, manifest: dict, binary: bytes) -> Path:
        import os
        import uuid
        target = self.directory / (manifest["key"] + ".cubin")
        temporary = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_bytes(binary)
        os.replace(temporary, target)
        receipt = target.with_suffix(".json")
        temporary = receipt.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temporary.write_text(json.dumps({"manifest": manifest, "sha256": hashlib.sha256(binary).hexdigest()}, sort_keys=True))
        os.replace(temporary, receipt)
        return target
