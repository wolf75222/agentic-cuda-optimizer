"""Filesystem boundary tests. The real macOS probe uses no model or GPU."""
import json
from pathlib import Path
import platform
from xml.sax.saxutils import escape

import pytest

from optimizer_agent import native_read_isolation as isolation
from optimizer_agent.campaign_config import CampaignConfig
from scripts import run_codex_vanilla as driver


def test_named_profile_reopens_only_literal_workspace_and_schema_inside_sensitive_tree(tmp_path):
    workspace = tmp_path / "trial" / "workspace"
    schema = tmp_path / "trial" / "evidence" / "output-schema.json"
    settings = isolation.permission_config(tmp_path, workspace, schema)
    assert settings == [
        'default_permissions="study_native_tools"',
        'permissions.study_native_tools.extends=":workspace"',
        'permissions.study_native_tools.filesystem={' +
        json.dumps(str(tmp_path)) + '="deny",' + json.dumps(str(workspace)) + '="write",' +
        json.dumps(str(schema)) + '="read"}',
        'permissions.study_native_tools.network.enabled=true',
    ]
    command = driver.build_command("codex", workspace, schema.parent, sensitive_root=tmp_path)
    assert "--ignore-user-config" in command and "project_doc_max_bytes=0" in command
    assert "--sandbox" not in command
    assert not any("sandbox_mode" in item or "sandbox_workspace_write" in item for item in command)
    assert command[command.index("--output-schema") + 1] == str(schema)
    assert command[command.index("--output-last-message") + 1] == str(schema.parent / "last-message.json")


@pytest.mark.parametrize("bad", ["missing-root", "outside-workspace", "schema-inside-workspace", "broad-schema"])
def test_bad_scope_fails_closed_before_creating_any_session_directory(tmp_path, bad):
    root, workspace = tmp_path, tmp_path / "workspace"
    schema = tmp_path / "evidence" / "output-schema.json"
    if bad == "missing-root":
        root = None
    elif bad == "outside-workspace":
        workspace = tmp_path.parent / "outside"
    elif bad == "schema-inside-workspace":
        schema = workspace / "output-schema.json"
    else:
        schema = tmp_path / "evidence"
    with pytest.raises(isolation.NativeReadIsolationError):
        isolation.permission_config(root, workspace, schema)
    assert list(tmp_path.iterdir()) == []


def test_unsupported_os_refuses_before_any_command_or_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation.platform, "system", lambda: "UnsupportedOS")
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported platform must fail before starting any command")
    monkeypatch.setattr(isolation.subprocess, "run", forbidden)
    with pytest.raises(isolation.NativeReadIsolationError, match="unsupported"):
        isolation.probe_read_isolation(tmp_path, codex_executable="/test/codex")
    assert list(tmp_path.iterdir()) == []


def test_default_profile_probe_requires_resolved_machine_permission_entries(tmp_path):
    workspace, schema = tmp_path / "workspace", tmp_path / "evidence" / "output-schema.json"
    def rendered(root_access="deny", escalatable="false"):
        text = ('<environment_context><filesystem><permission_profile type="managed">'
                '<file_system type="restricted">'
                f'<entry access="{root_access}" escalatable="{escalatable}"><path>{escape(str(tmp_path))}</path></entry>'
                f'<entry access="write"><path>{escape(str(workspace))}</path></entry>'
                f'<entry access="read"><path>{escape(str(schema))}</path></entry>'
                '</file_system></permission_profile></filesystem></environment_context>')
        return json.dumps([{"content": [{"type": "input_text", "text": text}]}])
    assert isolation._rendered_permissions_match(rendered(), tmp_path, workspace, schema)
    assert not isolation._rendered_permissions_match(rendered("read"), tmp_path, workspace, schema)
    assert not isolation._rendered_permissions_match(rendered(escalatable="true"), tmp_path, workspace, schema)
    assert not isolation._rendered_permissions_match(json.dumps([{"content": [{"text": "deny " + str(tmp_path)}]}]),
                                                       tmp_path, workspace, schema)


def test_failed_isolation_precedes_service_construction_model_and_session_files(tmp_path):
    calls = []
    def refused(*args, **kwargs):
        calls.append("probe")
        raise isolation.NativeReadIsolationError("Synthetic read-deny unavailable")
    def forbidden(*args, **kwargs):
        raise AssertionError("No model, service or version command may follow failed isolation")
    with pytest.raises(isolation.NativeReadIsolationError, match="unavailable"):
        driver.run_vanilla(CampaignConfig(), tmp_path / "native", "cpu-test", "/test/codex",
            sensitive_root=tmp_path, isolation_probe=refused, service_factory=forbidden,
            popen_factory=forbidden, version_runner=forbidden)
    assert calls == ["probe"] and list(tmp_path.iterdir()) == []


def test_missing_explicit_root_cannot_start_native_session(tmp_path):
    with pytest.raises(isolation.NativeReadIsolationError, match="explicit sensitive_root"):
        driver.run_vanilla(CampaignConfig(), tmp_path / "native", "cpu-test", "/test/codex")
    assert list(tmp_path.iterdir()) == []


def test_mismatched_preflight_scope_is_rejected_before_gpu(tmp_path):
    with pytest.raises(isolation.NativeReadIsolationError, match="explicit sensitive_root"):
        driver.run_vanilla(CampaignConfig(), tmp_path / "native", "cpu-test", "/test/codex",
            sensitive_root=tmp_path,
            isolation_probe=lambda *args, **kwargs: {"ok": True, "sensitive_root": str(tmp_path.parent)})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(platform.system() != "Darwin", reason="Real CPU sandbox qualified on macOS only")
def test_real_native_sandbox_denies_parent_aliases_preserves_child_reads_and_loopback(tmp_path):
    executable = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")
    if not executable.is_file():
        pytest.skip("Qualified desktop bundled CLI is unavailable")
    before = set(tmp_path.iterdir())
    receipt = isolation.probe_read_isolation(tmp_path, codex_executable=executable)
    assert receipt["ok"] and receipt["temporary_probe_removed"]
    assert not receipt["model_started"] and not receipt["gpu_submitted"]
    assert receipt["scope"] == isolation.SCOPE
    assert receipt["fresh_workspace_required"]
    assert all(check["ok"] for check in receipt["checks"].values())
    assert {"parent_database_read_denied", "parent_traversal_read_denied", "evidence_read_denied",
        "symlink_read_escape_denied", "symlink_write_escape_denied", "hardlink_creation_escape_denied", "workspace_write_descendant",
        "loopback_http_gpu_transport", "actual_gpu_client_loopback_transport",
        "parent_auto_context_excluded"} <= receipt["checks"].keys()
    assert set(tmp_path.iterdir()) == before
    assert "synthetic-private-study-sentinel" not in json.dumps(receipt)
    assert len(receipt["source_sha256"]) == len(receipt["codex_executable_sha256"]) == 64
