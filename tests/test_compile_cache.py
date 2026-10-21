"""Real temporary header bytes, no compiler, GPU, model or network."""
from pathlib import Path

import pytest

from optimizer_agent.compile_cache import compilation_manifest


def manifest(source, options=()):
    return compilation_manifest(source, options=options, architecture="sm_90",
                                toolchain={"nvrtc_version": "explicit CPU fixture"})


@pytest.mark.parametrize("form", ["joined_short", "split_short", "joined_long", "split_long"])
def test_local_angle_header_and_nested_dependency_content_invalidate_the_key(tmp_path, form):
    source = tmp_path / "kernel.cu"
    source.write_text("#include <gain.cuh>\n")
    include = tmp_path / "headers"
    include.mkdir()
    header, nested = include / "gain.cuh", include / "nested.h"
    header.write_text('#include "nested.h"\n#define GAIN VALUE\n')
    nested.write_text("#define VALUE 1\n")
    options = {"joined_short": ["-I" + str(include)], "split_short": ["-I", str(include)],
               "joined_long": ["--include-path=" + str(include)], "split_long": ["--include-path", str(include)]}[form]
    original = manifest(source, options)
    assert set(original["files"]) == {str(source), str(header), str(nested)}
    assert manifest(source, options) == original
    nested.write_text("#define VALUE 2\n")
    assert manifest(source, options)["key"] != original["key"]
    nested.write_text("#define VALUE 1\n")
    header.write_text('#include "nested.h"\n#define GAIN (VALUE+1)\n')
    assert manifest(source, options)["key"] != original["key"]


def test_guarded_header_cycle_is_finite_and_binds_every_header(tmp_path):
    source, first, second = (tmp_path / name for name in ("kernel.cu", "first.h", "second.h"))
    source.write_text("#include <first.h>\n")
    first.write_text('#pragma once\n#include "second.h"\n')
    second.write_text('#pragma once\n#include "first.h"\n#define VALUE 1\n')
    original = manifest(source)
    assert len(original["files"]) == 3
    second.write_text(second.read_text() + "#define OTHER 2\n")
    assert manifest(source)["key"] != original["key"]


def test_multiple_distinct_local_include_candidates_are_rejected(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text("#include <gain.cuh>\n")
    directories = [tmp_path / "one", tmp_path / "two"]
    for index, directory in enumerate(directories):
        directory.mkdir()
        (directory / "gain.cuh").write_text(f"#define GAIN {index}\n")
    with pytest.raises(ValueError, match="Ambiguous local compilation dependency"):
        manifest(source, ["-I" + str(path) for path in directories])


def test_symlink_header_cycle_fails_closed_instead_of_becoming_a_system_header(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text("#include <gain.cuh>\n")
    (tmp_path / "gain.cuh").symlink_to("other.cuh")
    (tmp_path / "other.cuh").symlink_to("gain.cuh")
    with pytest.raises(ValueError, match="Unresolvable local compilation dependency"):
        manifest(source)


@pytest.mark.parametrize("options", [["-I"], ["--include-path"], ["--include-path="], ["-I", "--fmad=true"]])
def test_incomplete_include_options_are_rejected(tmp_path, options):
    source = tmp_path / "kernel.cu"
    source.write_text("#include <cuda_runtime.h>\n")
    with pytest.raises(ValueError, match="include-path option"):
        manifest(source, options)


def test_unavailable_system_headers_still_use_the_recorded_execution_toolchain(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text("#include <cuda_runtime.h>\n")
    original = manifest(source)
    assert set(original["files"]) == {str(source)}
    changed = compilation_manifest(source, architecture="sm_90", toolchain={"nvrtc_version": "different fixture"})
    assert changed["key"] != original["key"]


@pytest.mark.parametrize("form", ["long_joined", "long_split", "short_split", "short_joined"])
def test_forced_include_options_are_explicitly_unsupported_before_header_read(tmp_path, monkeypatch, form):
    source, forced = tmp_path / "kernel.cu", tmp_path / "forced.h"
    source.write_text("// explicit CPU fixture\n")
    forced.write_text("#define VALUE 1\n")
    options = {"long_joined": ["--pre-include=" + str(forced)], "long_split": ["--pre-include", str(forced)],
               "short_split": ["-include", str(forced)], "short_joined": ["-include=" + str(forced)]}[form]
    original = Path.read_bytes
    def guarded(path):
        assert path != forced, "Unsupported forced include must not be read"
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    with pytest.raises(ValueError, match="Forced-include compiler options are unsupported"):
        manifest(source, options)


@pytest.mark.parametrize("nested", [False, True])
def test_macro_include_never_creates_a_cache_identity_even_when_header_changes(tmp_path, nested):
    source, header = tmp_path / "kernel.cu", tmp_path / "gain.cuh"
    macro = '#define HEADER <gain.cuh>\n#include HEADER\n'
    if nested:
        source.write_text('#include "wrapper.h"\n')
        (tmp_path / "wrapper.h").write_text(macro)
    else:
        source.write_text(macro)
    for value in (1, 2):
        header.write_text(f"#define GAIN {value}\n")
        with pytest.raises(ValueError, match="Nonliteral or unsupported include directive"):
            manifest(source, ["-I" + str(tmp_path)])
