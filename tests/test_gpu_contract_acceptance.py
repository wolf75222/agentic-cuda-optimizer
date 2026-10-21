from dataclasses import replace
import json
from pathlib import Path
import zipfile
from types import SimpleNamespace

import pytest

from scripts import gpu_contract_acceptance as helper


PROPERTIES = {"uuid": "unit-only", "architecture": "sm_90", "driver_version": 13010,
              "nvrtc_version": "12.6", "cuda_toolkit_version": 12060, "harness_build_id": "fixture"}


class NoExecutionRunner:
    def __init__(self):
        self.calls = []
    def get_device_properties(self, device):
        return PROPERTIES
    def run(self, *args):
        self.calls.append(args)
        raise AssertionError("Precision contract rejection must happen before the harness")


def test_preparation_never_contacts_scheduler_or_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(helper, "SlurmBackend", lambda *args, **kwargs: pytest.fail("plan cannot allocate"))
    assert helper.main(["--output", str(tmp_path / "plan"), "--run-id", "cpu-plan"]) == 0
    plan = json.loads((tmp_path / "plan/plan.json").read_text())
    assert plan["submitted"] is False and plan["destructive_job_needed"] is False
    assert plan["model_calls"] == 0 and len(plan["planned_checks"]) == 5


@pytest.mark.parametrize("precision", ["double", "tf32", "half"])
def test_precision_variants_are_rejected_before_any_harness_call(tmp_path, precision):
    runner = NoExecutionRunner()
    evaluator, _, variants, _, _ = helper.prepare(tmp_path, "precision-unit", runner=runner, properties=PROPERTIES)
    result = evaluator.benchmark(variants[precision])
    assert not result["ok"] and result["status"] == "contract_violation"
    assert not runner.calls


def make_bundle(tmp_path):
    evaluator, candidate, _, cases, _ = helper.prepare(tmp_path / "source", "bundle-unit", properties=PROPERTIES)
    request = replace(cases[-1].request, source=str(candidate.source), mode="run", compiler_options=["--fmad=true"],
                      **candidate.launch_rule.apply(cases[-1].metadata))
    path = tmp_path / "candidate.zip"
    manifest = helper.create_bundle(path, candidate, request, PROPERTIES)
    return path, candidate, request, manifest


def test_bundle_reconstructs_exact_source_launch_options_and_input_bytes(tmp_path):
    path, original, request, manifest = make_bundle(tmp_path)
    candidate, rebuilt, actual = helper.reconstruct_bundle(path, tmp_path / "rebuilt")
    assert candidate.artifact_hash == original.artifact_hash
    assert candidate.source.read_bytes() == original.source.read_bytes()
    assert rebuilt.compiler_options == request.compiler_options == ["--fmad=true"]
    assert candidate.launch_rule == original.launch_rule
    assert rebuilt.grid == request.grid and rebuilt.block == request.block
    assert actual == manifest
    for old, new in zip(request.arguments, rebuilt.arguments):
        if getattr(old, "input_file", None):
            assert Path(old.input_file).read_bytes() == Path(new.input_file).read_bytes()
    _, old_manifest = helper.bind_cache(request, PROPERTIES)
    _, new_manifest = helper.bind_cache(rebuilt, PROPERTIES)
    assert old_manifest["source_hash"] == new_manifest["source_hash"]
    assert old_manifest["options"] == new_manifest["options"]
    assert old_manifest["key"] != new_manifest["key"], "absolute source paths intentionally affect compile cache identity"


def rewrite_zip(path, update):
    with zipfile.ZipFile(path) as archive:
        values = {name: archive.read(name) for name in archive.namelist()}
    update(values)
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in values.items():
            archive.writestr(name, value)


def test_bundle_corruption_is_rejected_before_materialization(tmp_path):
    path, _, _, _ = make_bundle(tmp_path)
    rewrite_zip(path, lambda values: values.update({"source.cu": b"corrupted"}))
    destination = tmp_path / "rebuilt"
    with pytest.raises(ValueError, match="checksum mismatch"):
        helper.reconstruct_bundle(path, destination)
    assert not destination.exists()


def test_bundle_extra_files_and_traversal_are_rejected(tmp_path):
    path, _, _, _ = make_bundle(tmp_path)
    rewrite_zip(path, lambda values: values.update({"../outside": b"no"}))
    with pytest.raises(ValueError, match="Unexpected or missing"):
        helper.reconstruct_bundle(path, tmp_path / "rebuilt")


def test_reconstruction_never_overwrites_an_existing_directory(tmp_path):
    path, _, _, _ = make_bundle(tmp_path)
    destination = tmp_path / "exists"
    destination.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        helper.reconstruct_bundle(path, destination)


def test_additional_domain_measures_one_launch_and_hardware_filters_the_other(tmp_path):
    _, baseline, _, _, _ = helper.prepare(tmp_path / "input", "invalid-domain-unit")
    recording = SimpleNamespace(calls=[])
    class Evaluator:
        sequence = 0
        def validate(self, candidate, suite):
            recording.calls.append({"request": {"block": list(candidate.launch_rule.block), "mode": "run"}})
            return {"ok": True, "status": "validated"}
        def benchmark(self, candidate):
            self.sequence += 1
            recording.calls.append({"request": {"block": list(candidate.launch_rule.block), "mode": "benchmark"}})
            return {"ok": True, "status": "measured", "evaluation_id": str(self.sequence),
                    "cases": {"case": {"latency_ms": 1.0}}}
    result = helper.invalid_launch_acceptance(Evaluator(), baseline, recording, tmp_path,
        {**PROPERTIES, "max_threads_per_block": 1024, "max_block_dimensions": [1024, 1024, 64]})
    assert result["ok"]
    assert result["counters"] == {"measured_configurations": 1, "filtered_configurations": 1,
                                   "native_requests": 3, "native_requests_with_2048_threads_x": 0}
    assert [row["parameters"] for row in result["observations"]] == [{"BLOCK_X": 32}, {"BLOCK_X": 2048}]
