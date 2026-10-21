"""Content-addressed compilation manifests. Measurement evidence is never cached here."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compilation_manifest(source: Path, *, dependencies=(), options=(), architecture: str,
                         toolchain: dict, abi: str = "cuda-driver-pointer64-cxx17-v1") -> dict:
    """Hash explicit and recursively quoted dependencies; missing includes fail closed.

    System headers belong to the recorded compiler/toolkit identity. The caller must
    supply that identity from the execution worker, never the local host for SSH jobs.
    """
    if not architecture or not toolchain:
        raise ValueError("A target architecture and execution toolchain identity are required")
    visited: dict[str, str] = {}
    roots = [Path(source).resolve(), *(Path(p).resolve() for p in dependencies)]
    include_dirs = [p.parent for p in roots]
    include_dirs.extend(Path(option.split("=", 1)[1]).resolve()
                        for option in options if option.startswith("--include-path="))

    def visit(path: Path):
        if str(path) in visited:
            return
        data = path.read_bytes()
        visited[str(path)] = hashlib.sha256(data).hexdigest()
        for name in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', data.decode(errors="replace"), re.MULTILINE):
            found = next((p / name for p in [path.parent, *include_dirs] if (p / name).is_file()), None)
            if found is None:
                raise ValueError(f"Unresolved compilation dependency: {name}")
            visit(found.resolve())

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
