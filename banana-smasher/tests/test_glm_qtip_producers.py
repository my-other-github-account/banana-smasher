"""Read-only integration tests; fixtures are not producer liveness evidence."""

import hashlib
import json

import pytest


def source(tmp_path):
    config = {
        "text_config": {
            "num_hidden_layers": 45,
            "first_k_dense_replace": 3,
            "n_routed_experts": 288,
        }
    }
    prefix = "model.language_model.layers."
    names = {
        f"{prefix}{layer}.mlp.experts.{expert}.{projection}_proj.weight": "weights.safetensors"
        for layer in range(3, 46)
        for expert in range(288)
        for projection in ("gate", "up", "down")
    }
    names["model.language_model.layers.3.mlp.shared_experts.gate_proj.weight"] = (
        "native.safetensors"
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": names}))
    return hashlib.sha256(index.read_bytes()).hexdigest()


def test_existing_hosts_have_complete_disjoint_rosters_and_preserved_roots(tmp_path):
    from banana_smasher.glm_qtip_producers import producer_plan

    basis = source(tmp_path)
    union = set()
    for host, start, stop in [
        ("spark-3", 3, 17),
        ("spark-5-work", 17, 31),
        ("spark-7", 31, 45),
    ]:
        plan = producer_plan(host, tmp_path, intended_basis=basis)
        cells = {(r["layer"], r["expert"], r["projection"]) for r in plan["cells"]}
        assert len(cells) == 8064
        assert {x[0] for x in cells} == set(range(start, stop))
        assert {x[1] for x in cells} == set(range(288))
        assert not union & cells
        union |= cells
        assert (
            plan["producer_root"]
            == f"/home/dnola/missions/GLM_Q1_CHAMPION_t_024b05d4_{host.replace('-', '_')}"
        )
        assert plan["launch_authorized"] is False
        assert plan["historical_label"] == "partial K1/256-expert-plan diagnostic"
        assert plan["target"]["scope"] == "routed_only"
    assert len(union) == 24192
    from banana_smasher.glm_qtip_producers import validate_fanin_roster

    rows = [
        {"host": host, "id": cell["id"]}
        for host in ("spark-3", "spark-5-work", "spark-7")
        for cell in producer_plan(host, tmp_path, intended_basis=basis)["cells"]
    ]
    assert validate_fanin_roster(tmp_path, basis, rows)["expected_cells"] == 24192
    with pytest.raises(ValueError, match="fan-in"):
        validate_fanin_roster(tmp_path, basis, rows[:-1])
    with pytest.raises(ValueError, match="fan-in"):
        validate_fanin_roster(tmp_path, basis, rows + [rows[0]])
    wrong_host = [dict(row) for row in rows]
    wrong_host[0]["host"] = "spark-7"
    with pytest.raises(ValueError, match="fan-in"):
        validate_fanin_roster(tmp_path, basis, wrong_host)


def test_roster_rejects_wrong_basis_and_missing_source_projection(tmp_path):
    from banana_smasher.glm_qtip_producers import producer_plan

    basis = source(tmp_path)
    with pytest.raises(ValueError, match="basis"):
        producer_plan("spark-3", tmp_path, intended_basis="0" * 64)
    index = tmp_path / "model.safetensors.index.json"
    document = json.loads(index.read_text())
    del document["weight_map"][
        "model.language_model.layers.3.mlp.experts.287.up_proj.weight"
    ]
    index.write_text(json.dumps(document))
    basis = hashlib.sha256(index.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="missing.*up_proj"):
        producer_plan("spark-3", tmp_path, intended_basis=basis)


def test_adoption_rejects_copied_shards_dead_pid_and_wrong_host(tmp_path):
    from banana_smasher.glm_qtip_producers import producer_plan, validate_adoption

    basis = source(tmp_path)
    plan = producer_plan("spark-5-work", tmp_path, intended_basis=basis)
    process = {
        "alive": True,
        "pid": 123,
        "start_ticks": 456,
        "host": plan["host"],
        "root": plan["producer_root"],
        "layers": plan["layers"],
    }
    claim = {
        "task_id": "t_024b05d4",
        "host": plan["host"],
        "workload_pid": 123,
        "start_ticks": 456,
        "expiry_unix": 200,
        "source_model_index_sha256": basis,
    }
    shards = {
        "intended_basis": basis,
        "rows": [
            {
                "layer": layer,
                "owner": "t_024b05d4",
                "pid": 123,
                "startticks": 456,
                "range": [0, 288],
                "basis": basis,
                "state": "CLAIMED",
                "projections": ["fused13", "down"],
            }
            for layer in plan["layers"]
        ],
    }
    result = validate_adoption(plan, claim, shards, process, now=100)
    assert result["status"] == "ADOPT_EXISTING"
    assert result["launch_authorized"] is False
    for changed in (
        {**process, "alive": False},
        {**process, "start_ticks": 457},
        {**process, "host": "spark-3"},
    ):
        with pytest.raises(ValueError, match="identity|reconciliation"):
            validate_adoption(plan, claim, shards, changed, now=100)
    with pytest.raises(ValueError, match="claim"):
        validate_adoption(plan, claim, shards, process, now=201)
    shards["rows"][0]["layer"] = 3
    with pytest.raises(ValueError, match="shards"):
        validate_adoption(plan, claim, shards, process, now=100)
