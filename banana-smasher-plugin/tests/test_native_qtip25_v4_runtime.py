from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from banana_smasher.backpack import _materialize_native_v4_plane_source
from banana_smasher.banana_v1 import banana_v1_gaussian_codebook
from banana_smasher.contract import export_pack
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4 import (
    decode_native_v4,
    expand_native_v4_tlut,
    pack_native_v4_states,
    states_from_native_v4_packed,
)
from banana_smasher_plugin.native_qtip25_v4 import (
    NATIVE_QTIP25_V4_RUNTIME_FAMILY,
    dequantize_native_v4_blocks,
    native_v4_decode_counters,
    reset_native_v4_decode_counters,
)
from banana_smasher_plugin.native_planes import (
    NativePlaneLayer,
    NativePlanePack,
    NativePlanePrerequisiteError,
    _canonical_specialized_tier,
)
from banana_smasher_plugin.v4_acceleration import runtime_sentinel


def test_native_v4_installed_consumer_matches_reference_and_counts_no_fallback() -> None:
    rng = np.random.default_rng(57101415)
    raw = rng.integers(0, 2, size=(2, 640), dtype=np.uint8)
    packed = np.packbits(raw, axis=1, bitorder="big")
    states = states_from_native_v4_packed(packed, steps=64)
    packed = pack_native_v4_states(states)
    tlut = gaussian_tlut(bits=9, columns=2)

    reset_native_v4_decode_counters()
    observed = dequantize_native_v4_blocks(
        torch.from_numpy(packed).reshape(1, 2, 80), torch.from_numpy(tlut)
    )
    expected = decode_native_v4(
        packed,
        np.ones(2, dtype=np.float32),
        positions=256,
        tlut=tlut,
    ).reshape(1, 2, 16, 16)

    assert NATIVE_QTIP25_V4_RUNTIME_FAMILY == "qtip25_native_v4"
    assert torch.equal(observed, torch.from_numpy(expected))
    assert native_v4_decode_counters() == {
        "decode_calls": 1,
        "decode_blocks": 2,
        "decode_code_bytes": 160,
        "cuda_decode_calls": 0,
        "fallback_calls": 0,
    }


def test_native_v5_installed_consumer_decodes_pr31_codebook_without_fallback() -> None:
    packed = np.arange(80, dtype=np.uint8).reshape(1, 80)
    codebook = banana_v1_gaussian_codebook()

    reset_native_v4_decode_counters()
    observed = dequantize_native_v4_blocks(
        torch.from_numpy(packed), torch.from_numpy(codebook)
    )
    expected = decode_native_v4(
        packed,
        np.ones(1, dtype=np.float32),
        positions=256,
        tlut=codebook,
    ).reshape(1, 16, 16)

    assert torch.equal(observed, torch.from_numpy(expected))
    assert native_v4_decode_counters()["fallback_calls"] == 0



def test_b8_b10_b12_selected_pack_loads_and_executes_on_cpu(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "n_routed_experts": 256,
                "hidden_size": 16,
                "moe_intermediate_size": 16,
            }
        )
    )
    descriptors = {
        f"native-b{bits}": {
            "id": f"native-b{bits}",
            "family": "qtip_native_v4",
            "bpw": bits / 4,
        }
        for bits in (8, 10, 12)
    }
    cells = []
    selected = {}
    roots = {}
    for projection, output_width in (("fused13", 32), ("down", 16)):
        for start, stop, bits in ((0, 86, 8), (86, 171, 10), (171, 256, 12)):
            ids = list(range(start, stop))
            cell_id = f"{projection}-b{bits}"
            root = tmp_path / "candidates" / cell_id
            root.mkdir(parents=True)
            rows = len(ids) * output_width
            blocks = rows * 16 // 256
            (root / "wire.bin").write_bytes(bytes(blocks * 8 * bits))
            np.save(root / "SU.npy", np.ones(16, dtype=np.float16), allow_pickle=False)
            np.save(root / "SV.npy", np.ones(rows, dtype=np.float16), allow_pickle=False)
            np.save(root / "Wscale.npy", np.asarray(1.0, dtype=np.float32), allow_pickle=False)
            (root / "CELL_RECEIPT.json").write_text(
                json.dumps({"source": {"shape": [rows, 16]}})
            )
            cells.append(
                {
                    "cell_id": cell_id,
                    "layer": 0,
                    "projection": projection,
                    "expert_ids": ids,
                }
            )
            selected[cell_id] = {"tier": f"native-b{bits}"}
            roots[cell_id] = root
    _materialize_native_v4_plane_source(
        source,
        cells=cells,
        selected=selected,
        tier_descriptors=descriptors,
        artifact_roots=roots,
    )
    pack_root = tmp_path / "pack"
    export_pack(
        source_root=source,
        output=pack_root,
        model_id="fixture/native-v4",
        instance_id="native-v4-cpu",
        runtime_floor_bytes=0,
    )

    pack = NativePlanePack.from_model_root(pack_root)
    layer = NativePlaneLayer(pack, 0, device="cpu")
    experts = torch.tensor([0, 100, 200], dtype=torch.int64)
    state = layer._states["down"]

    assert len(set(state.families.tolist())) == 3
    assert [
        int(state.pointer_tables["native_v4_transition_bits"][expert])
        for expert in experts.tolist()
    ] == [8, 10, 12]
    assert set(state.pointer_tables["native_v4_codes"].tolist()) - {0}


def test_native_v4_runtime_family_rejects_mismatched_declared_geometry() -> None:
    with pytest.raises(NativePlanePrerequisiteError, match="family/geometry mismatch"):
        _canonical_specialized_tier(
            "native-b7",
            {
                "family": "qtip_native_v4_b7",
                "geometry": {"L": 16, "B": 9, "V": 4, "tlut_bits": 9},
            },
        )


def test_native_v4_runtime_rejects_code_bytes_that_disagree_with_declared_b() -> None:
    with pytest.raises(NativePlanePrerequisiteError, match="code geometry mismatch"):
        _canonical_specialized_tier(
            "native-b7",
            {
                "family": "qtip_native_v4_b7",
                "geometry": {"L": 16, "B": 7, "V": 4, "tlut_bits": 9},
                "input_width": 16,
                "output_width": 16,
                "tensors": {
                    "codes": {"dtype": "|u1", "shape": [1, 1, 72]},
                    "expert_ids": {"dtype": "<i2", "shape": [1]},
                },
            },
        )


def test_native_v4_runtime_families_append_without_aliasing_legacy_ids() -> None:
    counters = runtime_sentinel()["family_counters"]

    assert counters[:4] == ("qtip2", "qtip3", "d4", "native_mxfp4")
    assert counters[4:] == tuple(
        f"qtip_native_v4_b{bits}" for bits in range(4, 17)
    )
    assert len(set(counters)) == 17


def test_native_v4_cuda_source_matches_reference_state_values_and_receipt_order() -> None:
    table = np.arange(1024, dtype=np.float32).reshape(512, 2)
    states = np.asarray([0x0000, 0x0001, 0x1234, 0x8000, 0xFFFF], dtype=np.uint64)
    expanded = expand_native_v4_tlut(table)

    for state in states.tolist():
        rotated = ((state << 8) | (state >> 8)) & 0xFFFF
        expected = []
        for selected in (state, rotated):
            hashed = (selected + 1) * selected
            index = (hashed >> 6) & 511
            sign = 1.0 - 2.0 * ((hashed >> 15) & 1)
            expected.extend((table[index, 0] * sign, table[index, 1]))
        assert expanded[state].tolist() == expected

    csrc = (
        Path(__file__).parents[1]
        / "src"
        / "banana_smasher_plugin"
        / "csrc"
        / "native_v4_gemv.cu"
    ).read_text()
    assert "(static_cast<uint32_t>(selected) + 1u) * selected" in csrc
    assert "(hash >> 6) & 511u" in csrc
    assert "(hash >> 15) & 1u" in csrc
    assert "lane == 0 || lane == 2" in csrc
    assert csrc.index("native_v4_fused_gemv_kernel<<<") < csrc.index(
        "native_v4_receipt_kernel<<<"
    )


def test_native_v4_compaction_requires_full_counter_receipt_capacity() -> None:
    csrc = (
        Path(__file__).parents[1]
        / "src"
        / "banana_smasher_plugin"
        / "csrc"
        / "route_compaction.cu"
    ).read_text()
    assert "physical_counters.numel() >= 153" in csrc
