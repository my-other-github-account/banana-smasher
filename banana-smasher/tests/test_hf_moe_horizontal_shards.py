from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


def _model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"n_routed_experts": 2, "num_hidden_layers": 1}) + "\n"
    )
    tensors = {
        "layers.0.experts.0.down_proj.weight": np.arange(16, dtype=np.float16).reshape(2, 8),
        "layers.0.experts.1.down_proj.weight": (np.arange(16, dtype=np.float16) + 3).reshape(2, 8),
        "layers.0.router.weight": np.arange(8, dtype=np.float16).reshape(2, 4),
    }
    shard = model / "model-00001-of-00001.safetensors"
    save_file(tensors, shard)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}}, sort_keys=True) + "\n"
    )
    return model


def _build(model: Path, output: Path, start: int, end: int):
    from banana_smasher import build_hf_moe_uniform_shard

    return build_hf_moe_uniform_shard(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        tier="q2",
        scope="routed_only",
        native_rest=True,
        routed_ordinal_start=start,
        routed_ordinal_end=end,
        output=output,
    )


def test_public_horizontal_shards_union_disjoint_ranges_and_reopen(tmp_path: Path) -> None:
    from banana_smasher import open_hf_moe_uniform, open_hf_moe_uniform_shard
    from banana_smasher import union_hf_moe_uniform_shards

    model = _model(tmp_path)
    shard0 = _build(model, tmp_path / "shard0", 0, 1)
    shard1 = _build(model, tmp_path / "shard1", 1, 2)

    assert shard0["shard"]["routed_ordinals"] == [0, 1]
    assert shard1["shard"]["routed_ordinals"] == [1, 2]
    assert len(shard0["native_tensors"]) == 1
    assert shard1["native_tensors"] == []
    assert open_hf_moe_uniform_shard(tmp_path / "shard0") == shard0

    merged = union_hf_moe_uniform_shards(
        [tmp_path / "shard1", tmp_path / "shard0"], output=tmp_path / "merged"
    )
    reopened = open_hf_moe_uniform(tmp_path / "merged")

    assert reopened == merged
    assert [row["name"] for row in merged["routed_tensors"]] == sorted(
        row["name"] for row in merged["routed_tensors"]
    )
    assert merged["accounting"]["routed_tensor_count"] == 2
    assert merged["accounting"]["native_tensor_count"] == 1
    assert merged["union"]["input_ranges"] == [[0, 1], [1, 2]]
    worked = (Path(__file__).parents[2] / "WORKED_EXAMPLE.md").read_text()
    assert "build_hf_moe_uniform_shard(" in worked
    assert "union_hf_moe_uniform_shards(" in worked
    member_hashes = [
        row["wire"][kind]["sha256"]
        for row in merged["routed_tensors"]
        for kind in ("trellis", "scales")
    ] + [row["artifact_sha256"] for row in merged["native_tensors"]]
    assert merged["union"]["ordered_member_sha256"] == hashlib.sha256(
        "".join(member_hashes).encode()
    ).hexdigest()


def test_horizontal_union_refuses_overlap_and_gap_before_output(tmp_path: Path) -> None:
    from banana_smasher import union_hf_moe_uniform_shards

    model = _model(tmp_path)
    shard0 = _build(model, tmp_path / "shard0", 0, 1)
    assert shard0["status"] == "PASS"

    with pytest.raises(ValueError, match="gap-free"):
        union_hf_moe_uniform_shards([tmp_path / "shard0"], output=tmp_path / "gap")
    assert not (tmp_path / "gap").exists()

    duplicate = _build(model, tmp_path / "duplicate", 0, 1)
    assert duplicate["status"] == "PASS"
    with pytest.raises(ValueError, match="disjoint"):
        union_hf_moe_uniform_shards(
            [tmp_path / "shard0", tmp_path / "duplicate"], output=tmp_path / "overlap"
        )
    assert not (tmp_path / "overlap").exists()
