from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

from banana_smasher.contract import (
    TIER_CODES,
    _finish_selected_array,
    _npy_metadata,
    _open_selected_array,
    export_pack,
    materialize_selected_wire,
    unpack_index_rows,
    verify_pack,
)
from banana_smasher.loader import PackLoader
from banana_smasher.repack import repack_to_safetensors, verify_repack_roundtrip
from banana_smasher.repair import CodebookRepair, RepairBundle


def _write_file(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_banana_smasher_layer(root: Path, *, layer: int = 0) -> Path:
    root.mkdir(parents=True)
    tiers = {
        "d4_k256": list(range(3)),
        "d4_k1024": list(range(3, 91)),
        "d4_k2048": list(range(91, 251)),
        "d4_k4096": list(range(251, 256)),
    }
    rows: list[dict[str, object]] = []
    for tier_index, (tier, experts) in enumerate(tiers.items(), start=1):
        bits = int(tier.removeprefix("d4_k")).bit_length() - 1
        for projection in ("down", "fused13"):
            marker = tier_index + (0 if projection == "down" else 16)
            rows.extend(
                [
                    _write_file(
                        root / f"{tier}.{projection}.codebook.fp16.bin",
                        np.arange(4 * (tier_index + 1), dtype=np.float16).tobytes(),
                    ),
                    _write_file(
                        root / f"{tier}.{projection}.codes.le{bits}.bin",
                        bytes([marker]) * (len(experts) * (tier_index + 2)),
                    ),
                    _write_file(
                        root / f"{tier}.{projection}.expert_ids.i16.bin",
                        np.asarray(experts, dtype="<i2").tobytes(),
                    ),
                    _write_file(
                        root / f"{tier}.{projection}.scales.e8m0.bin",
                        bytes([marker + 1]) * (len(experts) * (tier_index + 1)),
                    ),
                ]
            )
    receipt = {
        "schema": "banana_smasher-materialized-layer-v1",
        "status": "PASS",
        "layer": layer,
        "files": rows,
    }
    (root / "LAYER_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return root


def _write_mixed_substrate_layer(root: Path, *, layer: int = 0) -> Path:
    root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    families = {
        "d4_k256": {
            "experts": {"down": [0], "fused13": [1]},
            "planes": {
                "codebook.fp16": np.zeros((256, 4), dtype="<f2").tobytes(),
                "codes.le8": b"\x01" * 3,
                "scales.e8m0": b"\x02" * 2,
            },
        },
        "d4_k1024": {
            "experts": {"down": [1], "fused13": [0]},
            "planes": {
                "codebook.fp16": np.zeros((1024, 4), dtype="<f2").tobytes(),
                "codes.le10": b"\x01" * 3,
                "scales.e8m0": b"\x02" * 2,
            },
        },
        "d4_k2048": {
            "experts": [2],
            "planes": {
                "codebook.fp16": np.zeros((2048, 4), dtype="<f2").tobytes(),
                "codes.le11": b"\x01" * 3,
                "scales.e8m0": b"\x02" * 2,
            },
        },
        "d4_k4096": {
            "experts": [3],
            "planes": {
                "codebook.fp16": np.zeros((4096, 4), dtype="<f2").tobytes(),
                "codes.le12": b"\x01" * 3,
                "scales.e8m0": b"\x02" * 2,
            },
        },
        "d8_k256": {
            "experts": [4],
            "planes": {
                "codebook.fp16": np.zeros((256, 8), dtype="<f2").tobytes(),
                "codes.le8": b"\x03" * 5,
                "scales.e8m0": b"\x04" * 2,
            },
        },
        "native_mxfp4": {
            "experts": list(range(5, 256)),
            "planes": {
                "weights.mxfp4": b"\x05" * (251 * 7),
                "scales.e8m0": b"\x06" * (251 * 2),
            },
        },
    }
    for tier, spec in families.items():
        for projection in ("down", "fused13"):
            experts = spec["experts"]
            if isinstance(experts, dict):
                experts = experts[projection]
            for role, payload in spec["planes"].items():
                rows.append(_write_file(root / f"{tier}.{projection}.{role}.bin", payload))
            rows.append(
                _write_file(
                    root / f"{tier}.{projection}.expert_ids.i16.bin",
                    np.asarray(experts, dtype="<i2").tobytes(),
                )
            )
    (root / "LAYER_RECEIPT.json").write_text(
        json.dumps(
            {
                "schema": "banana_smasher-materialized-layer-v1",
                "status": "PASS",
                "layer": layer,
                "files": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root


def _write_serving_root(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "hidden_size": 16,
                "moe_intermediate_size": 8,
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (root / name).write_text("{}\n")
    return root


def _wire_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_selected_array_helpers_close_memory_maps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "selected.npy"
    writable = _open_selected_array(
        output,
        count=1,
        row_shape=(4,),
        dtype=np.dtype("int16"),
    )
    writable[0] = np.arange(4, dtype=np.int16)
    _finish_selected_array(output, writable)
    assert getattr(writable, "_mmap").closed

    opened: list[np.memmap] = []
    load = np.load

    def capture_load(*args: object, **kwargs: object) -> np.ndarray:
        # NumPy's public typing cannot express a transparent test wrapper.
        assert args
        array = load(*args, **kwargs)
        assert isinstance(array, np.memmap)
        opened.append(array)
        return array

    monkeypatch.setattr("banana_smasher.contract.np.load", capture_load)
    assert _npy_metadata(output)["shape"] == [1, 4]
    assert opened and getattr(opened[0], "_mmap").closed


def test_materialized_wire_root_exports_all_layers_with_bound_routes_and_repair(
    tmp_path: Path,
) -> None:
    source = tmp_path / "wire"
    first = _write_banana_smasher_layer(source / "layer_000", layer=0)
    _write_banana_smasher_layer(source / "layer_001", layer=1)
    assignment = {
        "assignment": {
            str(layer): {
                str(expert): {
                    "down": next(
                        tier
                        for tier, members in {
                            "d4_k256": range(3),
                            "d4_k1024": range(3, 91),
                            "d4_k2048": range(91, 251),
                            "d4_k4096": range(251, 256),
                        }.items()
                        if expert in members
                    ),
                    "fused13": next(
                        tier
                        for tier, members in {
                            "d4_k256": range(3),
                            "d4_k1024": range(3, 91),
                            "d4_k2048": range(91, 251),
                            "d4_k4096": range(251, 256),
                        }.items()
                        if expert in members
                    ),
                }
                for expert in range(256)
            }
            for layer in range(2)
        }
    }
    assignment_path = tmp_path / "ASSIGNMENT.json"
    assignment_path.write_text(json.dumps(assignment, sort_keys=True))
    overlay_path = tmp_path / "ACTIVE_OVERLAY.json"
    overlay_path.write_text("{}\n")
    old_path = first / "d4_k256.down.codebook.fp16.bin"
    old = np.fromfile(old_path, dtype="<f2").reshape(-1, 4)
    replacement = np.full(old.shape, 7, dtype=np.float16)
    repair = RepairBundle(
        checkpoint_path=tmp_path / "UPDATE_004.pt",
        checkpoint_sha256="1" * 64,
        active_overlay_path=overlay_path,
        active_overlay_sha256="2" * 64,
        assignment_path=assignment_path,
        assignment_sha256=hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
        checkpoint_format="bs-basic-repair-v1",
        mechanism="physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains",
        update=4,
        codebooks={
            _wire_sha(old): CodebookRepair(
                checkpoint_key=f"L0/d4_k256_down_{_wire_sha(old)}",
                source_wire_sha256=_wire_sha(old),
                array=replacement,
            )
        },
        dense_tensors={
            "norms/model.norm": np.ones(4, dtype=np.float32),
            "outputs/model.layers.0.self_attn.o_b_proj.output_log_gain": np.asarray(
                0.0, dtype=np.float32
            ),
        },
        norm_count=1,
        output_count=1,
    )

    pack = tmp_path / "pack"
    manifest = export_pack(
        source_root=source,
        output=pack,
        model_id="fixture",
        instance_id="u004-fixture",
        link_mode="hardlink",
        repair=repair,
        serving_model_root=_write_serving_root(tmp_path / "serving"),
        runtime_floor_bytes=123,
    )

    assert manifest["source_format"] == "banana_smasher-materialized-wire-v1"
    assert manifest["layers"] == [0, 1]
    assert set(manifest["provenance"]["source_layer_receipt_sha256"]) == {"0", "1"}
    assert set(manifest["selected_payloads"]["layers"]) == {"0", "1"}
    selected = manifest["selected_payloads"]["layers"]["0"]["down"]
    assert selected["tiers"][:4] == ["d4_k256", "d4_k256", "d4_k256", "d4_k1024"]
    assert selected["slots"][:4] == [0, 1, 2, 0]
    codebook = selected["payloads"]["d4_k256"]["tensors"]["codebooks"]
    assert codebook["storage_kind"] == "raw"
    repaired = np.memmap(
        pack / "planes" / codebook["file"],
        mode="r",
        dtype=codebook["dtype"],
        shape=tuple(codebook["shape"]),
    )
    assert np.all(repaired == 7)
    assert verify_pack(pack)["repair"]["codebook_checkpoint_keys"] == 1
    linked_source = source / "layer_001/d4_k2048.down.codes.le11.bin"
    linked_pack = pack / "planes/layers/layer_001/truevq_d4/d4_k2048.down.codes.le11.bin"
    assert os.stat(linked_source).st_ino == os.stat(linked_pack).st_ino


def test_banana_smasher_wire_export_and_safetensors_roundtrip_are_byte_exact(
    tmp_path: Path,
) -> None:
    source = _write_banana_smasher_layer(tmp_path / "layer_000")
    pack = tmp_path / "pack"

    manifest = export_pack(
        source_root=source,
        output=pack,
        model_id="banana_smasher-fixture",
        instance_id="bs-pack-0001-banana_smasher",
        link_mode="hardlink",
    )
    receipt = verify_pack(pack)

    source_codes = source / "d4_k2048.down.codes.le11.bin"
    packed_codes = pack / "planes/layers/layer_000/truevq_d4" / source_codes.name
    assert os.stat(source_codes).st_ino == os.stat(packed_codes).st_ino
    assert manifest["source_format"] == "banana_smasher-materialized-layer-v1"
    assert (
        manifest["provenance"]["source_layer_receipt_sha256"]
        == hashlib.sha256((source / "LAYER_RECEIPT.json").read_bytes()).hexdigest()
    )
    assert receipt["tensor_count"] == 34

    tier_map = np.load(
        pack / "planes/layers/layer_000/experts/tier_map.npy",
        mmap_mode="r",
    )
    subtier_map = np.load(
        pack / "planes/layers/layer_000/experts/subtier_map.npy",
        mmap_mode="r",
    )
    assert np.all(tier_map == TIER_CODES["truevq_d4"])
    assert subtier_map[91] == 2048
    assert subtier_map[250] == 2048
    assert subtier_map[251] == 4096

    with PackLoader(pack, verify=True).open_layer(0, framework="np") as layer_view:
        raw_codes = layer_view.get("layers.0.truevq_d4.d4_k2048.down.codes")
        assert isinstance(raw_codes, np.memmap)
        assert raw_codes.tobytes() == source_codes.read_bytes()

    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.glob("*.bin")
    }
    repack_receipt = repack_to_safetensors(pack, drop_planes=True)
    roundtrip = verify_repack_roundtrip(pack)
    after = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.glob("*.bin")
    }

    assert before == after
    assert repack_receipt["status"] == "PASS"
    assert roundtrip["byte_exact_tensors"] == 34
    with safe_open(pack / "bs-pack.safetensors", framework="np") as handle:
        key = "layers.0.truevq_d4.d4_k2048.down.codes"
        assert handle.get_tensor(key).tobytes() == source_codes.read_bytes()


def test_mixed_substrate_receipt_exports_d8_and_native_mxfp4_planes(
    tmp_path: Path,
) -> None:
    source = _write_mixed_substrate_layer(tmp_path / "layer_000")

    manifest = export_pack(
        source_root=source,
        output=tmp_path / "pack",
        model_id="mixed-substrate-fixture",
        instance_id="mixed-substrate-0001",
        link_mode="hardlink",
    )

    tensors = manifest["tensor_index"]
    d8 = tensors["layers.0.truevq_d8.d8_k256.down.codebooks"]
    assert d8["shape"] == [256, 8]
    assert d8["storage"]["path"].startswith(
        "planes/layers/layer_000/truevq_d8/"
    )
    native = tensors["layers.0.native_mxfp4.native_mxfp4.down.packed"]
    assert native["shape"] == [251, 7]
    assert native["storage"]["path"].startswith(
        "planes/layers/layer_000/native_mxfp4/"
    )
    tier_map = np.load(
        tmp_path / "pack/planes/layers/layer_000/experts/tier_map.npy"
    )
    subtier_map = np.load(
        tmp_path / "pack/planes/layers/layer_000/experts/subtier_map.npy"
    )
    assert tier_map.tolist() == [
        TIER_CODES["truevq_d4"],
        TIER_CODES["truevq_d4"],
        TIER_CODES["truevq_d4"],
        TIER_CODES["truevq_d4"],
        TIER_CODES["truevq_d8"],
        *([TIER_CODES["native_mxfp4"]] * 251),
    ]
    assert subtier_map.tolist() == [0, 0, 2048, 4096, *([0] * 252)]


def test_banana_smasher_wire_export_refuses_receipt_drift(tmp_path: Path) -> None:
    source = _write_banana_smasher_layer(tmp_path / "layer_000")
    (source / "d4_k2048.down.codes.le11.bin").write_bytes(b"drift")

    import pytest

    from banana_smasher.contract import PackValidationError

    with pytest.raises(PackValidationError, match="banana_smasher source byte count mismatch"):
        export_pack(
            source_root=source,
            output=tmp_path / "pack",
            model_id="banana_smasher-fixture",
            instance_id="bs-pack-0001-banana_smasher",
            link_mode="copy",
        )


def _write_selected_wire_source(root: Path) -> Path:
    layer = root / "layer_000"
    layer.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    partitions = {
        "d4_k256": [0],
        "d4_k1024": [1],
        "d4_k2048": [2],
        "d4_k4096": [3],
        "d8_k256": [4],
        "native_mxfp4": list(range(5, 256)),
    }
    for tier, experts in partitions.items():
        for projection in ("down", "fused13"):
            output_width = 32 if projection == "down" else 64
            if tier.startswith("d4_k"):
                bits = int(tier.removeprefix("d4_k")).bit_length() - 1
                row_bytes = (8 * bits + 7) // 8
                planes = {
                    "codebook.fp16": np.zeros(
                        (int(tier.removeprefix("d4_k")), 4), dtype="<f2"
                    ).tobytes(),
                    f"codes.le{bits}": bytes([bits])
                    * (len(experts) * output_width * row_bytes),
                    "scales.e8m0": b"\x02" * (len(experts) * output_width),
                }
            elif tier == "d8_k256":
                planes = {
                    "codebook.fp16": np.zeros((256, 8), dtype="<f2").tobytes(),
                    "codes.le8": b"\x03" * (len(experts) * 32 * 8),
                    "scales.e8m0": b"\x04" * (len(experts) * 32),
                }
            else:
                planes = {
                    "weights.mxfp4": b"\x05" * (len(experts) * 7),
                    "scales.e8m0": b"\x06" * (len(experts) * 2),
                }
            for role, payload in planes.items():
                rows.append(_write_file(layer / f"{tier}.{projection}.{role}.bin", payload))
            rows.append(
                _write_file(
                    layer / f"{tier}.{projection}.expert_ids.i16.bin",
                    np.asarray(experts, dtype="<i2").tobytes(),
                )
            )
    (layer / "LAYER_RECEIPT.json").write_text(
        json.dumps(
            {
                "schema": "banana_smasher-materialized-layer-v1",
                "status": "PASS",
                "layer": 0,
                "files": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return root


def test_selected_wire_materializer_merges_rebased_qtip_d4_and_base_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_selected_wire_source(tmp_path / "wire")
    assignment = {
        "assignment": {
            "0": {
                str(expert): {
                    projection: (
                        "qtip3_3.0117"
                        if expert == 4
                        else "d4_k1024"
                        if expert == 0 and projection == "down"
                        else "d4_k256"
                        if expert == 0
                        else "d4_k1024"
                        if expert == 1
                        else "d4_k2048"
                        if expert == 2
                        else "d4_k4096"
                        if expert == 3
                        else "native_mxfp4"
                    )
                    for projection in ("down", "fused13")
                }
                for expert in range(256)
            }
        }
    }
    assignment_path = tmp_path / "ASSIGNMENT.json"
    assignment_path.write_text(json.dumps(assignment, sort_keys=True))
    assignment_sha = hashlib.sha256(assignment_path.read_bytes()).hexdigest()

    declared = tmp_path / "declared"
    replica = tmp_path / "replica"
    replica.mkdir()
    artifact_payloads: dict[str, dict[str, object]] = {}
    rows = []
    for projection in ("down", "fused13"):
        name = f"qtip-{projection}.pt"
        payload = f"qtip-{projection}".encode()
        (replica / name).write_bytes(payload)
        artifact_payloads[name] = {
            "schema": "qtip-hyb-wire-unit-v1",
            "identity": {"layer": 0, "expert": 4, "projection": projection},
            "trellis": np.arange(6, dtype=np.int16).reshape(2, 3),
            "SU": np.ones(32, dtype=np.float16),
            "SV": np.ones(32 if projection == "down" else 64, dtype=np.float16)
            * 2,
            "Wscale": np.asarray(3, dtype=np.float32),
        }
        rows.append(
            {
                "layer": 0,
                "expert": 4,
                "projection": projection,
                "new": "qtip3_3.0117",
                "artifact": str(declared / name),
                "artifact_bytes": len(payload),
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    d4_payload = b"d4-down"
    (replica / "d4-down.pt").write_bytes(d4_payload)
    d4_codes = np.arange(32 * 8, dtype=np.int16).reshape(32, 8) % 1024
    artifact_payloads["d4-down.pt"] = {
        "codes": d4_codes,
        "scales": np.arange(32, dtype=np.uint8).reshape(32, 1),
        "meta": {"layer": 0, "expert": 0, "projection": "down", "k": 1024},
    }
    codebook = np.arange(1024 * 4, dtype=np.float16).reshape(1024, 4)
    (replica / "d4-codebook.bin").write_bytes(codebook.astype("<f2").tobytes())
    rows.append(
        {
            "layer": 0,
            "expert": 0,
            "projection": "down",
            "new": "d4_k1024",
            "artifact": str(declared / "d4-down.pt"),
            "artifact_bytes": len(d4_payload),
            "artifact_sha256": hashlib.sha256(d4_payload).hexdigest(),
            "codebook": str(declared / "d4-codebook.bin"),
            "codebook_bytes": int(codebook.nbytes),
            "codebook_sha256": hashlib.sha256(
                codebook.astype("<f2").tobytes()
            ).hexdigest(),
        }
    )
    overlay = {
        "schema": "p885-wire-c-active-overlay-v1",
        "status": "PASS_EXACT_ACTIVE_LAYERS",
        "stale": False,
        "active_assignment_sha256": assignment_sha,
        "final_assignment_sha256": assignment_sha,
        "rows": rows,
    }
    overlay_path = tmp_path / "ACTIVE_OVERLAY.json"
    overlay_path.write_text(json.dumps(overlay, sort_keys=True))
    overlay_sha = hashlib.sha256(overlay_path.read_bytes()).hexdigest()

    def fake_load(path: Path) -> dict[str, object]:
        return artifact_payloads[path.name]

    monkeypatch.setattr("banana_smasher.contract._load_selected_artifact", fake_load)
    output = tmp_path / "selected"
    receipt = materialize_selected_wire(
        source_root=source,
        output=output,
        assignment_path=assignment_path,
        assignment_sha256=assignment_sha,
        active_overlay_path=overlay_path,
        active_overlay_sha256=overlay_sha,
        artifact_rebases=[(declared, replica)],
        hidden_size=32,
        moe_intermediate_size=32,
    )

    assert receipt["status"] == "PASS"
    assert receipt["family_counts"] == {
        "d4": 8,
        "native": 502,
        "qtip2": 0,
        "qtip3": 2,
    }
    meta = json.loads((output / "layer_000.meta.json").read_text())
    assert meta["tier2"][4] == "qtip3_3.0117"
    assert meta["family2"][4] == TIER_CODES["qtip3"]
    qtip = meta["payloads"]["down"]["qtip3_3.0117"]["tensors"]
    assert np.load(output / qtip["trellis"]["file"]).shape == (1, 2, 3)
    d4 = meta["payloads"]["down"]["d4_k1024"]["tensors"]
    packed = np.load(output / d4["codes"]["file"])
    assert d4["codes"]["encoding"] == "little-endian-packed-index-rows-v1"
    assert np.array_equal(
        unpack_index_rows(packed[0], bits=10, values_per_row=8), d4_codes
    )
    native_ids = np.load(
        output
        / meta["payloads"]["down"]["native_mxfp4"]["tensors"]["expert_ids"][
            "file"
        ]
    )
    assert native_ids.tolist() == list(range(5, 256))

    serving = _write_serving_root(tmp_path / "serving-selected")
    serving_config = json.loads((serving / "config.json").read_text())
    serving_config["hidden_size"] = 32
    serving_config["moe_intermediate_size"] = 32
    (serving / "config.json").write_text(json.dumps(serving_config))
    pack = tmp_path / "selected-pack"
    manifest = export_pack(
        source_root=output,
        output=pack,
        model_id="selected-fixture",
        instance_id="selected-fixture-1",
        link_mode="hardlink",
        serving_model_root=serving,
        runtime_floor_bytes=0,
    )
    assert manifest["provenance"]["selected_wire_materialization"]["sha256"] == (
        hashlib.sha256((output / "SELECTED_WIRE_RECEIPT.json").read_bytes()).hexdigest()
    )
    packed_pack = pack / "planes" / d4["codes"]["file"]
    assert os.stat(output / d4["codes"]["file"]).st_ino == os.stat(packed_pack).st_ino
    assert np.array_equal(np.load(packed_pack)[0], packed[0])
    assert verify_pack(pack)["source_format"] == "p1016-true-c-native-planes-v1"
