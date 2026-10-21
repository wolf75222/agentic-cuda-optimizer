"""Fail-closed filesystem isolation for native Codex model tools.

The CLI/controller are trusted transports (auth, schema and output handling).
The named permissions profile confines tool reads of the study tree. This is
not a sandbox around the trusted CLI itself, nor a promise about files outside
the explicitly supplied sensitive root. A fresh controller-created workspace
is mandatory: an existing hardlink supplied by a trusted controller is an
already disclosed file, which pathname policies cannot retroactively hide.
"""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
from xml.etree import ElementTree


PROFILE = "study_native_tools"
SCOPE = "model tool filesystem isolation; trusted CLI/controller can read schema/auth"


class NativeReadIsolationError(RuntimeError):
    """The native tool read boundary could not be established before launch."""


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def permission_config(sensitive_root, workspace, output_schema):
    """Return per-invocation TOML overrides, without any legacy sandbox key.

An inline table is deliberate: quoted dotted CLI keys are interpreted as
literal quotation marks by some CLI releases, whereas this table is parsed
as TOML. More-specific workspace/schema rules reopen only those paths.
    """
    if sensitive_root is None:
        raise NativeReadIsolationError("An explicit sensitive_root is required for native tool isolation")
    root = Path(sensitive_root).resolve()
    workspace, schema = Path(workspace).resolve(), Path(output_schema).resolve()
    if not root.is_dir() or workspace == root or not workspace.is_relative_to(root):
        raise NativeReadIsolationError("The fresh native workspace must be strictly inside an existing sensitive_root")
    if not schema.is_relative_to(root) or schema.is_relative_to(workspace):
        raise NativeReadIsolationError("The output schema must be inside the denied root and outside workspace")
    if schema.name != "output-schema.json":
        raise NativeReadIsolationError("Only the literal output-schema.json transport exception is supported")
    filesystem = "{" + ",".join(json.dumps(str(path)) + "=" + json.dumps(access)
        for path, access in ((root, "deny"), (workspace, "write"), (schema, "read"))) + "}"
    return [f'default_permissions="{PROFILE}"', f'permissions.{PROFILE}.extends=":workspace"',
        f"permissions.{PROFILE}.filesystem={filesystem}", f"permissions.{PROFILE}.network.enabled=true"]


def permission_arguments(sensitive_root, workspace, output_schema):
    return [argument for setting in permission_config(sensitive_root, workspace, output_schema)
            for argument in ("-c", setting)]


def _rendered_permissions_match(rendered, root, workspace, schema):
    """Check actual resolved default permissions, not an echoed instruction."""
    try:
        messages = json.loads(rendered)
        for message in messages:
            for content in message.get("content", []):
                text = content.get("text", "")
                if not text.startswith("<environment_context>"):
                    continue
                context = ElementTree.fromstring(text)
                filesystem = context.find("./filesystem/permission_profile/file_system")
                if filesystem is None or filesystem.get("type") != "restricted":
                    continue
                entries = {(item.get("access"), item.findtext("path"), item.get("escalatable"))
                           for item in filesystem.findall("entry")}
                return (("deny", str(root), "false") in entries
                        and any(access == "write" and path == str(workspace) for access, path, _ in entries)
                        and any(access == "read" and path == str(schema) for access, path, _ in entries))
    except (ValueError, TypeError, AttributeError, ElementTree.ParseError):
        pass
    return False


def probe_read_isolation(sensitive_root, *, codex_executable=None, timeout_seconds=20):
    """Exercise the real installed native sandbox, with synthetic data only.

This CPU-only preflight is safe to call BEFORE a worker is submitted. It
creates a temporary tree under sensitive_root and removes it before return;
no workspace/evidence from a real generation, model or GPU is opened. Failure
raises rather than falling back to a broad legacy workspace-write policy.
    """
    if sensitive_root is None:
        raise NativeReadIsolationError("An explicit sensitive_root is required before worker/model launch")
    root = Path(sensitive_root).resolve()
    if not root.is_dir():
        raise NativeReadIsolationError("sensitive_root must already exist for the CPU isolation probe")
    if platform.system() != "Darwin":
        raise NativeReadIsolationError("Native read isolation has only been qualified on macOS; this OS is unsupported")
    if codex_executable is None:
        from .codex_backend import default_codex_executable
        codex_executable = default_codex_executable()
    executable = Path(codex_executable).resolve()
    if not executable.is_file():
        raise NativeReadIsolationError("The requested native Codex executable is unavailable")
    started = time.time()
    checks = {}
    def run(command, **options):
        try:
            return subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds,
                                  check=False, **options)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise NativeReadIsolationError(f"Native read-isolation CPU probe failed: {type(error).__name__}") from error
    version = run([str(executable), "--version"])
    if version.returncode:
        raise NativeReadIsolationError("Native Codex version probe failed")
    with tempfile.TemporaryDirectory(prefix=".native-isolation-probe-", dir=root) as temporary:
        probe = Path(temporary).resolve()
        workspace, evidence = probe / "workspace", probe / "evidence"
        workspace.mkdir()
        evidence.mkdir()
        public, private = "synthetic-public-workspace", "synthetic-private-study-sentinel"
        allowed, database = workspace / "allowed.txt", probe / "suite.sqlite"
        sibling, schema = evidence / "private.json", evidence / "output-schema.json"
        allowed.write_text(public)
        database.write_text(private)
        sibling.write_text(private)
        schema.write_text('{"synthetic_schema":true}')
        context_marker = "SYNTHETIC_PARENT_DOCUMENT_MUST_NOT_BECOME_CONTEXT"
        (probe / "AGENTS.md").write_text(context_marker)
        overrides = permission_arguments(root, workspace, schema)
        # prompt-input renders local context without a model request. Do not
        # print its contents or alter the user's config/auth environment. The
        # actual exec command additionally uses --ignore-user-config.
        rendered = run([str(executable), "-C", str(workspace), *overrides, "-c", "project_doc_max_bytes=0",
                        "debug", "prompt-input", "Synthetic isolation CPU prompt"])
        context_ok = (rendered.returncode == 0 and context_marker not in rendered.stdout
                      and "Synthetic isolation CPU prompt" in rendered.stdout)
        checks["parent_auto_context_excluded"] = {"ok": context_ok, "returncode": rendered.returncode}
        if not context_ok:
            raise NativeReadIsolationError("Native parent auto-context CPU probe failed")
        resolved = _rendered_permissions_match(rendered.stdout, root, workspace, schema)
        checks["default_profile_resolves_read_denial"] = {"ok": resolved, "returncode": rendered.returncode}
        if not resolved:
            raise NativeReadIsolationError("The default native permission profile did not resolve the required read denial")
        base = [str(executable), *overrides, "-c", "project_doc_max_bytes=0",
                "sandbox", "-P", PROFILE, "--include-managed-config", "-C", str(workspace)]
        def check(name, arguments, *, expected=None, denied=False):
            result = run(base + arguments)
            # Never copy command stdout/stderr (including a private sentinel)
            # to a receipt. A nonzero code alone is not proof of enforcement.
            permission_error = any(word in result.stderr.lower()
                for word in ("operation not permitted", "permission denied"))
            ok = (result.returncode != 0 and permission_error and private not in result.stdout) if denied else (
                result.returncode == 0 and (expected is None or result.stdout == expected))
            checks[name] = {"ok": ok, "returncode": result.returncode,
                            "permission_error": permission_error}
            if not ok:
                raise NativeReadIsolationError(f"Native read-isolation check failed: {name}")
        child_cat = ["/bin/sh", "-c", 'exec /bin/cat "$1"', "isolation-probe"]
        check("workspace_read_descendant", child_cat + [str(allowed)], expected=public)
        check("parent_database_read_denied", child_cat + [str(database)], denied=True)
        check("parent_traversal_read_denied", child_cat + [str(workspace / ".." / "suite.sqlite")], denied=True)
        check("evidence_read_denied", child_cat + [str(sibling)], denied=True)
        check("literal_schema_read", child_cat + [str(schema)], expected=schema.read_text())
        check("workspace_write_descendant", ["/bin/sh", "-c", 'printf public > "$1" && /bin/cat "$1"',
              "isolation-probe", str(workspace / "written.txt")], expected="public")
        check("parent_write_denied", ["/bin/sh", "-c", 'printf denied > "$1"',
              "isolation-probe", str(probe / "forbidden-write")], denied=True)
        check("symlink_read_escape_denied", ["/bin/sh", "-c", '/bin/ln -s "$1" "$2" && exec /bin/cat "$2"',
              "isolation-probe", str(database), str(workspace / "symbolic-alias")], denied=True)
        check("symlink_write_escape_denied", ["/bin/sh", "-c", 'printf denied > "$1"',
              "isolation-probe", str(workspace / "symbolic-alias")], denied=True)
        check("hardlink_creation_escape_denied", ["/bin/sh", "-c", '/bin/ln "$1" "$2" && exec /bin/cat "$2"',
              "isolation-probe", str(database), str(workspace / "hard-alias")], denied=True)
        if database.read_text() != private:
            raise NativeReadIsolationError("The synthetic protected database was modified by a sandboxed tool")
        requests = []
        rpc_payload = {"ok": True, "properties": {"synthetic_cpu_probe": True}}
        marker = json.dumps(rpc_payload).encode()
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append((self.path, self.headers.get("Authorization")))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(marker)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            check("loopback_http_gpu_transport", ["/bin/sh", "-c",
                  'exec /usr/bin/curl --silent --fail --max-time 5 "$1"', "isolation-probe",
                  f"http://127.0.0.1:{server.server_port}/properties"], expected=marker.decode())
            helper = Path(__file__).resolve().parents[1] / "scripts" / "vanilla_gpu_tool.py"
            check("actual_gpu_client_loopback_transport", ["/bin/sh", "-c", 'exec "$@"', "isolation-probe",
                  sys.executable, str(helper), "properties", "--endpoint",
                  f"http://127.0.0.1:{server.server_port}", "--token", "synthetic-probe-token"],
                  expected=json.dumps(rpc_payload, indent=2) + "\n")
            if requests != [("/properties", None), ("/properties", "Bearer synthetic-probe-token")]:
                raise NativeReadIsolationError("The synthetic loopback GPU endpoint did not receive the sandboxed request")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    return {"ok": True, "method": "codex_named_permission_profile", "scope": SCOPE,
        "sensitive_root": str(root), "profile": PROFILE, "platform": platform.system(),
        "codex_executable": str(executable), "codex_executable_sha256": _hash(executable),
        "codex_version": version.stdout.strip(), "checks": checks,
        "fresh_workspace_required": True, "temporary_probe_removed": not probe.exists(),
        "model_started": False, "gpu_submitted": False, "elapsed_seconds": time.time() - started,
        "source_sha256": _hash(__file__), "gpu_client_sha256": _hash(helper)}
