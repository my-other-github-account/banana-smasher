from __future__ import annotations

import hashlib
import json
import numpy as np
import torch
from pathlib import Path

from safetensors.torch import save_file


PROJECTIONS = ("w1", "w2", "w3")


def _batch10_units() -> list[dict[str, object]]:
    return [
        {"expert": expert, "projection": projection}
        for expert in reversed(range(10))
        for projection in reversed(PROJECTIONS)
    ]


def test_batch10_groups_ten_experts_per_projection_deterministically() -> None:
    from banana_smasher.qtip_v7_batch import group_v7_batch10

    groups = group_v7_batch10(_batch10_units())

    assert tuple(groups) == PROJECTIONS
    for projection in PROJECTIONS:
        assert [unit["expert"] for unit in groups[projection]] == list(range(10))
        assert {unit["projection"] for unit in groups[projection]} == {projection}


def test_cross_unit_ldlq_is_exactly_equal_to_independent_unit_fixture() -> None:
    from banana_smasher.qtip_v7_batch import buffered_ldlq_cross_unit

    class FixtureQ2:
        @staticmethod
        def tensor_core_permutation() -> np.ndarray:
            return np.arange(256, dtype=np.int64)

        @staticmethod
        def quantize_k2_tiles(
            tiles: torch.Tensor,
            parent_lut: torch.Tensor,
            *,
            chunk_tiles: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del parent_lut, chunk_tiles
            quantized = torch.round(tiles * 4) / 4
            states = torch.round(quantized * 4).to(torch.int16)
            return quantized, states

    generator = torch.Generator().manual_seed(20260811)
    prepared = []
    for expert in range(2):
        target = torch.randn((128, 128), generator=generator)
        lower = torch.tril(
            torch.randn((128, 128), generator=generator) * 0.002,
            diagonal=-1,
        )
        prepared.append(
            {
                "expert": expert,
                "projection": "w1",
                "lower": lower,
                "transformed": {"target_inner": target},
            }
        )
    parent_lut = torch.zeros(1024, dtype=torch.float16)

    quantized, states, counters = buffered_ldlq_cross_unit(
        FixtureQ2, prepared, parent_lut
    )
    serial = [
        buffered_ldlq_cross_unit(FixtureQ2, [item], parent_lut)
        for item in prepared
    ]

    assert all(torch.equal(quantized[index], serial[index][0][0]) for index in range(2))
    assert all(torch.equal(states[index], serial[index][1][0]) for index in range(2))
    assert counters["cuda_tiles"] == sum(row[2]["cuda_tiles"] for row in serial)
    assert counters["fallback_calls"] == 0


def test_prepare_v7_unit_reuses_exact_shared_hessian_factor(
    monkeypatch,
) -> None:
    from banana_smasher import qtip_k2
    from banana_smasher.qtip_v7_batch import prepare_v7_unit

    def quantize_fixture(
        tiles: torch.Tensor,
        parent_lut: torch.Tensor,
        *,
        chunk_tiles: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del parent_lut, chunk_tiles
        return torch.round(tiles), torch.zeros_like(tiles, dtype=torch.int16)

    monkeypatch.setattr(qtip_k2, "quantize_k2_tiles", quantize_fixture)
    generator = torch.Generator().manual_seed(31)
    raw_h = torch.eye(128, dtype=torch.float32) * 512_000
    units = [
        {
            "expert": expert,
            "projection": "w1",
            "source": torch.randn((128, 128), generator=generator),
            "raw_h": raw_h.clone(),
            "raw_h_count": 512_000,
            "input_identity": {"raw_hessian_data_sha256": "a" * 64},
        }
        for expert in range(2)
    ]
    parent_lut = torch.linspace(-1, 1, 1024, dtype=torch.float16)
    factor_cache = {}

    first = prepare_v7_unit(qtip_k2, units[0], parent_lut, factor_cache)
    second = prepare_v7_unit(qtip_k2, units[1], parent_lut, factor_cache)

    assert first["hessian_cache_hit"] is False
    assert second["hessian_cache_hit"] is True
    assert len(factor_cache) == 1
    assert second["lower"] is first["lower"]
    assert second["hessian_sha256"] == first["hessian_sha256"]
    assert second["prepare_counters"]["fallback_calls"] == 0


def test_finalize_batch_unit_packs_deterministically_and_reports_no_fallback() -> None:
    from banana_smasher import qtip_k2
    from banana_smasher.qtip_v7_batch import finalize_batch_unit

    states = torch.arange(8 * 8 * 256, dtype=torch.int32).to(torch.int16).reshape(
        8, 8, 256
    )
    quantized = torch.zeros((128, 128), dtype=torch.float32)
    controls = torch.ones(128, dtype=torch.float32)
    item = {
        "expert": 7,
        "projection": "w3",
        "source": torch.zeros((128, 128), dtype=torch.float32),
        "proxy_hessian": torch.eye(128, dtype=torch.float32),
        "hessian_sha256": "1" * 64,
        "lower_sha256": "2" * 64,
        "transformed": {
            "target_inner": quantized.clone(),
            "su": controls,
            "sv": controls,
            "suh": controls.half(),
            "svh": controls.half(),
            "global_scale": 1.0,
        },
    }

    first = finalize_batch_unit(qtip_k2, item, quantized, states)
    second = finalize_batch_unit(qtip_k2, item, quantized, states)

    assert torch.equal(first["packed_codes"], qtip_k2.pack_k2(states))
    assert torch.equal(first["packed_codes"], second["packed_codes"])
    assert first["boundaries"]["packed_sha256"] == second["boundaries"]["packed_sha256"]
    assert first["solver_counters"]["fallback_calls"] == 0
    assert first["source_only"] is True


def test_producer_preserves_prepare_grouped_solve_finalize_invocation(monkeypatch) -> None:
    from banana_smasher import qtip_v7_batch

    trace = []
    units = [
        {"expert": expert, "projection": projection}
        for expert in range(10)
        for projection in PROJECTIONS
    ]

    def prepare(q2, unit, parent_lut, factor_cache):
        del q2, parent_lut
        trace.append(("prepare", unit["expert"], unit["projection"]))
        factor_cache.setdefault(("shared", 1, 128), (object(), "h", object(), "l"))
        return {**unit, "proxy_hessian": object()}

    def solve(q2, group, parent_lut, *, chunk_tiles=4096):
        del q2, parent_lut, chunk_tiles
        trace.append(("solve", group[0]["projection"], tuple(x["expert"] for x in group)))
        return [object()] * 10, [object()] * 10, {
            "qfn_calls": 8,
            "extension_calls": 8,
            "cuda_tiles": 320,
            "fallback_calls": 0,
            "chunk_tiles": 4096,
        }

    def finalize(q2, item, quantized, states):
        del q2, quantized, states
        trace.append(("finalize", item["expert"], item["projection"]))
        return {
            "member": f"E{item['expert']:03d}/{item['projection']}",
            "solver_counters": {"fallback_calls": 0},
        }

    monkeypatch.setattr(qtip_v7_batch, "prepare_v7_unit", prepare)
    monkeypatch.setattr(qtip_v7_batch, "buffered_ldlq_cross_unit", solve)
    monkeypatch.setattr(qtip_v7_batch, "finalize_batch_unit", finalize)

    results, receipt = qtip_v7_batch.produce_qtip2_v7_batch10(
        units, object(), q2=object()
    )

    assert trace[:30] == [
        ("prepare", unit["expert"], unit["projection"]) for unit in units
    ]
    solve_rows = [row for row in trace if row[0] == "solve"]
    assert solve_rows == [
        ("solve", projection, tuple(range(10))) for projection in PROJECTIONS
    ]
    assert len(results) == 30
    assert receipt["schema"] == "banana-smasher-qtip2-v7-batch10-producer-v1"
    assert receipt["group_sizes"] == {projection: 10 for projection in PROJECTIONS}
    assert receipt["factor_cache_entries"] == 1
    assert receipt["counters"]["fallback_calls"] == 0


def test_k2_kernel_uses_attempt9_dp4a_half2_packed_branch_path() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src/banana_smasher/qtip_k2_kernel.cu"
    ).read_text()

    assert "__dp4a(" in source
    assert "__hfma2(" in source
    assert "constexpr int kThreads = 1024;" in source
    assert "constexpr int kPackedEdges = kEdges / 4;" in source
    assert "uint8_t* all_branches" in source


def test_batch10_producer_is_exported_from_public_api() -> None:
    import banana_smasher

    assert callable(banana_smasher.produce_qtip2_v7_batch10)
    assert "produce_qtip2_v7_batch10" in banana_smasher.__all__


def test_public_source_materializer_binds_model_pre_and_layer_range(tmp_path: Path) -> None:
    import banana_smasher

    model = tmp_path / "authenticated-model"
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    tensors = {}
    weight_map = {}
    for expert in range(10):
        for projection in PROJECTIONS:
            prefix = f"layers.0.ffn.experts.{expert}.{projection}"
            tensors[f"{prefix}.weight"] = torch.full(
                (128, 64), 0x22 + expert, dtype=torch.uint8
            )
            tensors[f"{prefix}.scale"] = torch.full(
                (128, 4), 127, dtype=torch.uint8
            )
            weight_map[f"{prefix}.weight"] = shard.name
            weight_map[f"{prefix}.scale"] = shard.name
    save_file(tensors, shard)
    index = model / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}, sort_keys=True))
    basis = hashlib.sha256(index.read_bytes()).hexdigest()

    hessian_root = model / "hessians"
    hessian_root.mkdir()
    closures = []
    paths = {}
    for label in ["shared_w1_w3", *(f"e{expert:03d}_w2" for expert in range(10))]:
        path = hessian_root / f"L000_{label}_H_sum.fp32.npy"
        values = np.eye(128, dtype=np.float32) * 512_000
        np.save(path, values, allow_pickle=False)
        paths[label] = path
    for expert in range(10):
        for projection in PROJECTIONS:
            label = "shared_w1_w3" if projection in {"w1", "w3"} else f"e{expert:03d}_w2"
            path = paths[label]
            closures.append({
                "expert": expert,
                "member": projection,
                "count": 512_000,
                "raw_sum_path": path.name,
                "raw_sum_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "raw_sum_data_sha256": hashlib.sha256(
                    np.load(path, allow_pickle=False).tobytes(order="C")
                ).hexdigest(),
            })
    (hessian_root / "L000_STANDARD250_H_PLANE.json").write_text(json.dumps({
        "basis": basis,
        "layer": 0,
        "closures": closures,
    }, sort_keys=True))

    checkpoint = tmp_path / "PRE.pt"
    parent = torch.linspace(-1, 1, 1024, dtype=torch.float32)
    torch.save({
        "identity": {"model_index_sha256": basis},
        "state": {"luts": {"layers.0.qtip2_v7.layer_lut": parent}},
    }, checkpoint)

    materialized = banana_smasher.materialize_qtip2_v7_sources(
        model,
        checkpoint,
        range(0, 1),
        experts=range(10),
        device="cpu",
    )

    assert tuple(materialized) == (0,)
    assert torch.equal(materialized[0]["parent_lut"], parent.half())
    assert len(materialized[0]["units"]) == 30
    assert {unit["projection"] for unit in materialized[0]["units"]} == set(PROJECTIONS)
    assert all(unit["raw_h_count"] == 512_000 for unit in materialized[0]["units"])
    assert all(unit["input_identity"]["model_index_sha256"] == basis for unit in materialized[0]["units"])
    assert callable(banana_smasher.produce_qtip2_v7_batch10)
    assert "materialize_qtip2_v7_sources" in banana_smasher.__all__
