from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from banana_smasher.hf_deepseek_v4_d4_adapter import (
    DeepseekV4D4Runtime,
    _decode_bs_pack_expert,
    _open_bs_pack_projection,
    _unpack_le12_values,
    _unpack_le_values,
)


def _pack_le12(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.uint16)
    assert values.ndim == 1 and values.size % 2 == 0
    first = values[0::2]
    second = values[1::2]
    packed = np.empty((values.size // 2, 3), dtype=np.uint8)
    packed[:, 0] = first & 0xFF
    packed[:, 1] = ((first >> 8) & 0x0F) | ((second & 0x0F) << 4)
    packed[:, 2] = second >> 4
    return packed.tobytes()


def _pack_le(values: np.ndarray, *, bits: int, experts: int) -> bytes:
    rows = np.asarray(values, dtype=np.uint16).reshape(experts, -1)
    bit_rows = (
        (rows[..., None] >> np.arange(bits, dtype=np.uint16)) & 1
    ).astype(np.uint8)
    return np.packbits(
        bit_rows.reshape(experts, -1), axis=1, bitorder="little"
    ).tobytes()


def test_unpack_le12_values_round_trips_boundaries() -> None:
    expected = np.asarray([0, 1, 15, 16, 255, 256, 4094, 4095], dtype=np.uint16)
    packed = np.frombuffer(_pack_le12(expected), dtype=np.uint8)

    actual = _unpack_le12_values(packed)

    np.testing.assert_array_equal(actual, expected)


def test_load_vq3u_experts_falls_back_to_bound_bs_pack(tmp_path: Path) -> None:
    runtime = DeepseekV4D4Runtime.__new__(DeepseekV4D4Runtime)
    runtime.planes_dir = tmp_path / "planes"
    runtime.model_root = tmp_path
    calls: list[tuple[str, int]] = []
    runtime._load_vq3u_pt_experts = lambda layer: calls.append(("pt", layer)) or "pt"
    runtime._load_bs_pack_experts = lambda layer: calls.append(("pack", layer)) or "pack"

    assert runtime._load_vq3u_experts(7) == "pack"
    assert calls == [("pack", 7)]

    runtime.planes_dir.mkdir()
    (runtime.planes_dir / "vq3u_layer_007.pt").touch()
    calls.clear()

    assert runtime._load_vq3u_experts(7) == "pt"
    assert calls == [("pt", 7)]


def test_open_bs_pack_projection_binds_and_decodes_exact_members(tmp_path: Path) -> None:
    layer_root = tmp_path / "planes" / "layers" / "layer_003" / "truevq_d4"
    layer_root.mkdir(parents=True)
    codebook = np.arange(8, dtype="<f2").reshape(4, 2)
    codes = np.asarray([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.uint16)
    scales = np.asarray([127, 128, 129, 130], dtype=np.uint8)
    members = {
        "codebooks": (layer_root / "cb.bin", codebook.tobytes(), "fp16", "<f2"),
        "codes": (layer_root / "codes.bin", _pack_le12(codes), "le12", "|u1"),
        "scales": (layer_root / "scales.bin", scales.tobytes(), "e8m0", "|u1"),
    }
    index: dict[str, dict[str, object]] = {}
    for role, (path, payload, encoding, dtype) in members.items():
        path.write_bytes(payload)
        index[f"layers.3.truevq_d4.d4_k4.fused13.{role}"] = {
            "path": str(path.relative_to(tmp_path)),
            "data_bytes": len(payload),
            "encoding": encoding,
            "dtype": dtype,
            "subtier": 4,
        }
    manifest = {"tensor_index": index}

    projection = _open_bs_pack_projection(
        tmp_path,
        manifest,
        layer=3,
        projection="fused13",
        experts=2,
        rows=2,
        columns=4,
        k=4,
        d=2,
        scale_group_size=4,
    )

    np.testing.assert_array_equal(projection["codebook"], codebook)
    np.testing.assert_array_equal(projection["scales"], scales.reshape(2, 2, 1))
    np.testing.assert_array_equal(
        _unpack_le12_values(projection["codes"][0]), codes[:4]
    )
    np.testing.assert_array_equal(
        _unpack_le12_values(projection["codes"][1]), codes[4:]
    )


def test_open_bs_pack_projection_decodes_canonical_k2048_le11(tmp_path: Path) -> None:
    layer_root = tmp_path / "planes" / "layers" / "layer_003" / "truevq_d4"
    layer_root.mkdir(parents=True)
    codebook = np.arange(4096, dtype="<f2").reshape(2048, 2)
    codes = np.asarray([0, 1, 2046, 2047, 17, 18, 19, 20], dtype=np.uint16)
    scales = np.asarray([127, 128, 129, 130], dtype=np.uint8)
    members = {
        "codebooks": (layer_root / "cb.bin", codebook.tobytes(), "fp16", "<f2"),
        "codes": (
            layer_root / "codes.bin",
            _pack_le(codes, bits=11, experts=2),
            "le11",
            "|u1",
        ),
        "scales": (layer_root / "scales.bin", scales.tobytes(), "e8m0", "|u1"),
    }
    index: dict[str, dict[str, object]] = {}
    for role, (path, payload, encoding, dtype) in members.items():
        path.write_bytes(payload)
        index[f"layers.3.truevq_d4.d4_k2048.fused13.{role}"] = {
            "path": str(path.relative_to(tmp_path)),
            "data_bytes": len(payload),
            "encoding": encoding,
            "dtype": dtype,
            "subtier": 2048,
        }

    projection = _open_bs_pack_projection(
        tmp_path,
        {"tensor_index": index},
        layer=3,
        projection="fused13",
        experts=2,
        rows=2,
        columns=4,
        k=2048,
        d=2,
        scale_group_size=4,
    )

    assert projection["code_bits"] == 11
    np.testing.assert_array_equal(
        _unpack_le_values(projection["codes"][0], bits=11, value_count=4),
        codes[:4],
    )
    np.testing.assert_array_equal(
        _unpack_le_values(projection["codes"][1], bits=11, value_count=4),
        codes[4:],
    )


def test_decode_bs_pack_expert_reconstructs_scaled_dense_matrix() -> None:
    import torch

    codebook = np.asarray([[1, 2], [3, 4], [5, 6], [7, 8]], dtype="<f2")
    codes = np.asarray([0, 1, 2, 3], dtype=np.uint16)
    scales = np.asarray([[127], [128]], dtype=np.uint8)

    actual = _decode_bs_pack_expert(
        torch,
        codebook=codebook,
        packed_codes=np.frombuffer(_pack_le12(codes), dtype=np.uint8),
        scales=scales,
        rows=2,
        columns=4,
        d=2,
        scale_group_size=4,
        device="cpu",
        dtype=torch.float32,
    )

    expected = torch.tensor([[1, 2, 3, 4], [10, 12, 14, 16]], dtype=torch.float32)
    torch.testing.assert_close(actual, expected)


def test_load_bs_pack_experts_streams_both_projections_without_compat_file(
    tmp_path: Path,
) -> None:
    import json
    import torch

    tensor_index: dict[str, dict[str, object]] = {}
    layer_root = tmp_path / "planes" / "layers" / "layer_000" / "truevq_d4"
    layer_root.mkdir(parents=True)

    def add_projection(
        projection: str,
        columns: int,
        codes: np.ndarray,
    ) -> None:
        codebook = np.asarray([[1, 1], [2, 2], [3, 3], [4, 4]], dtype="<f2")
        scales = np.full((2, 2, columns // 2), 127, dtype=np.uint8)
        values = {
            "codebooks": (codebook.tobytes(), "fp16", "<f2"),
            "codes": (_pack_le12(codes), "le12", "|u1"),
            "scales": (scales.tobytes(), "e8m0", "|u1"),
        }
        for role, (payload, encoding, dtype) in values.items():
            path = layer_root / f"{projection}.{role}.bin"
            path.write_bytes(payload)
            tensor_index[f"layers.0.truevq_d4.d4_k4.{projection}.{role}"] = {
                "path": str(path.relative_to(tmp_path)),
                "data_bytes": len(payload),
                "encoding": encoding,
                "dtype": dtype,
                "subtier": 4,
            }

    add_projection(
        "fused13",
        4,
        np.asarray([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.uint16),
    )
    add_projection(
        "down",
        2,
        np.asarray([0, 1, 2, 3], dtype=np.uint16),
    )
    (tmp_path / "BANANA_PACK_MANIFEST.json").write_text(
        json.dumps({"tensor_index": tensor_index})
    )

    runtime = DeepseekV4D4Runtime.__new__(DeepseekV4D4Runtime)
    runtime.torch = torch
    runtime.device = "cpu"
    runtime.model_root = tmp_path
    runtime.config = SimpleNamespace(
        hidden_size=2,
        moe_intermediate_size=2,
        n_routed_experts=2,
    )
    runtime._d4_k = 4096
    runtime._d4_d = 2
    runtime._d4_scale_group_size = 2
    runtime._counted_paths = set()
    runtime._bytes_read = 0
    runtime._ensure_materialization_memory = lambda required, layer: None

    gate_up, down = runtime._load_bs_pack_experts(0)

    assert tuple(gate_up.shape) == (2, 2, 4)
    assert tuple(down.shape) == (2, 2, 2)
    torch.testing.assert_close(
        gate_up[0].float(),
        torch.tensor([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        down[0].float(),
        torch.tensor([[1, 1], [2, 2]], dtype=torch.float32),
    )
    assert not list(tmp_path.rglob("vq3u_layer_*.pt"))
