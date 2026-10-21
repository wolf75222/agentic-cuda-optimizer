import json
import hashlib

from scripts import gpu_rpc_acceptance as acceptance


def test_predeclared_order_preserves_requests_and_raw_sample_counts():
    plan = acceptance.protocol()
    assert len(plan["order"]) == 16
    assert [row["index"] for row in plan["order"]] == list(range(16))
    assert plan["model_calls"] == 0
    for mode in ("legacy", "exchange"):
        rows = [row for row in plan["order"] if row["transport"] == mode]
        assert len(rows) == plan["expected_requests_per_transport"] == 8
        benchmarks = [row for row in rows if row["mode"] == "benchmark"]
        assert len(benchmarks) == plan["expected_benchmark_requests_per_transport"] == 4
        assert len(benchmarks) * plan["launches_per_benchmark"] == plan["expected_raw_timings_per_transport"] == 8
    for start in range(0, 16, 2):
        a, b = plan["order"][start:start + 2]
        assert (a["n"], a["mode"], a["repetition"]) == (b["n"], b["mode"], b["repetition"])
        assert {a["transport"], b["transport"]} == {"legacy", "exchange"}


def test_default_only_writes_plan_without_ssh_or_slurm(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Planning must not construct a Slurm controller")
    monkeypatch.setattr(acceptance, "SlurmBackend", forbidden)
    assert acceptance.main(["--output", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "protocol.json").read_text()) == acceptance.protocol()
    assert not (tmp_path / "allocation-receipt.json").exists()


def test_provenance_hashes_effective_config_and_campaign_sources(tmp_path):
    alternate = tmp_path / "selected-execution.json"
    alternate.write_text('{"host":"alternate-romeo"}')
    sources = acceptance.provenance(alternate)["sources"]
    assert sources[str(alternate)] == hashlib.sha256(alternate.read_bytes()).hexdigest()
    assert "configs/romeo.json" not in sources
    assert "optimizer_agent/campaign.py" in sources
    assert "optimizer_agent/campaign_config.py" in sources
