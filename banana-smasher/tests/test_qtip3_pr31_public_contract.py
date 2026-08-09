from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


PROVIDER_ID = "periodic-qtip3@3.00"
LEGACY_PROVIDER_ID = "qtip-native-v6@3.00"
LUT_ID = "pr31-affine-gaussian-edge-v1"
BASIS_SHA = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"


def _qtip3_fixed_module():
    spec = importlib.util.find_spec("banana_smasher.qtip3_fixed")
    assert spec is not None, "public fixed-assignment QTIP3 module is missing"
    return importlib.import_module("banana_smasher.qtip3_fixed")


def test_qtip3_manifest_runtime_smoke_is_exported_from_public_package() -> None:
    package = importlib.import_module("banana_smasher")

    assert package.smoke_qtip3_fixed_manifest is _qtip3_fixed_module().smoke_qtip3_fixed_manifest


def _write_member(
    tmp_path: Path, *, provider_id: str = PROVIDER_ID
) -> tuple[Path, Path]:
    lut = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float16)
    lut_path = tmp_path / "pr31-lut.pt"
    torch.save(lut, lut_path)
    member = {
        "schema": "banana-smasher-qtip3-fixed-member-v1",
        "codec_provider_id": provider_id,
        "basis_index_sha256": BASIS_SHA,
        "source_weight_sha256": "1" * 64,
        "hessian_sha256": "2" * 64,
        "geometry": {
            "L": 16,
            "B": 12,
            "V": 4,
            "layout": "homogeneous",
            "phase_widths": [3, 3, 3, 3],
        },
        "lut": {
            "identity": LUT_ID,
            "tensor_sha256": hashlib.sha256(lut.numpy().tobytes()).hexdigest(),
            "data_bytes": lut.numel() * lut.element_size(),
        },
        "codes": torch.arange(96, dtype=torch.uint8).reshape(1, 96),
        "SU": torch.ones(16, dtype=torch.float16),
        "SV": torch.ones(16, dtype=torch.float16),
        "Wscale": torch.tensor(0.75, dtype=torch.float32),
    }
    member_path = tmp_path / "QTIP3_MEMBER.pt"
    torch.save(member, member_path)
    return member_path, lut_path


def test_qtip3_member_binds_owned_pr31_lut_and_exact_fixed_geometry(tmp_path: Path) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path)

    loaded = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)

    assert loaded.codec_provider_id == PROVIDER_ID
    assert loaded.geometry == {
        "L": 16,
        "B": 12,
        "V": 4,
        "layout": "homogeneous",
        "phase_widths": [3, 3, 3, 3],
    }
    assert loaded.lut_identity == LUT_ID
    assert loaded.lut_tensor_sha256 == hashlib.sha256(loaded.lut.numpy().tobytes()).hexdigest()
    assert loaded.codes.dtype == torch.uint8
    assert loaded.codes.requires_grad is False
    assert loaded.Wscale.dtype == torch.float32

    tampered = torch.load(lut_path, weights_only=True)
    tampered[0] += 1
    torch.save(tampered, lut_path)
    with pytest.raises(ValueError, match="LUT tensor SHA-256 mismatch"):
        qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)


def test_materialize_qtip3_fixed_member_consumes_physical_cell_receipt(
    tmp_path: Path,
) -> None:
    qtip3 = _qtip3_fixed_module()
    cell_root = tmp_path / "L007" / "E003_down"
    cell_root.mkdir(parents=True)
    arrays = {
        "codes": np.arange(96, dtype=np.uint8).reshape(1, 96),
        "SU": np.ones(16, dtype=np.float16),
        "SV": np.ones(16, dtype=np.float16),
        "Wscale": np.asarray(0.75, dtype=np.float32),
    }
    artifacts = {}
    for name, value in arrays.items():
        path = cell_root / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        artifacts[name] = {
            "path": f"/sealed/L007/E003_down/{name}.npy",
            "bytes": path.stat().st_size,
            "data_bytes": value.nbytes,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    lut = np.linspace(-1.0, 1.0, 1024, dtype=np.float16)
    lut_source = tmp_path / "pr31-codebook.npy"
    np.save(lut_source, lut, allow_pickle=False)
    receipt = cell_root / "CELL_RECEIPT.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-periodic-qtip3-pr31-cell-v1",
                "status": "PASS",
                "task_id": "t_064d19ef",
                "layer": 7,
                "expert": 3,
                "projection": "down",
                "basis_index_sha256": BASIS_SHA,
                "source_model_shard": {"tensor_sha256": "1" * 64},
                "control": {"sha256": "2" * 64},
                "geometry": {
                    "L": 16,
                    "B": 12,
                    "V": 4,
                    "rate_num": 3,
                    "rate_den": 1,
                    "phase_count": 1,
                    "alternation": False,
                    "member_averaging": False,
                },
                "tlut": {
                    "identity": LUT_ID,
                    "tensor_sha256": hashlib.sha256(lut.tobytes()).hexdigest(),
                },
                "accounting": {"exact_code_bpw": 3.0},
                "artifacts": artifacts,
            }
        )
        + "\n"
    )
    member_path = tmp_path / "QTIP3_MEMBER.pt"
    lut_path = tmp_path / "PR31_LUT.pt"

    result = qtip3.materialize_qtip3_fixed_member(
        cell_receipt=receipt,
        lut_source=lut_source,
        member_output=member_path,
        lut_output=lut_path,
        intended_basis_sha256=BASIS_SHA,
        artifact_root=cell_root,
    )
    loaded = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)

    assert result["status"] == "PASS"
    assert result["layer"] == 7
    assert result["expert"] == 3
    assert result["projection"] == "down"
    assert loaded.codec_provider_id == PROVIDER_ID
    assert loaded.source_weight_sha256 == "1" * 64
    assert loaded.hessian_sha256 == "2" * 64
    assert torch.equal(loaded.codes, torch.from_numpy(arrays["codes"]))


def test_materialize_qtip3_fixed_manifest_streams_cells_and_deduplicates_lut_wire(
    tmp_path: Path,
) -> None:
    qtip3 = _qtip3_fixed_module()
    lut = np.linspace(-1.0, 1.0, 1024, dtype=np.float16)
    lut_source = tmp_path / "pr31-codebook.npy"
    np.save(lut_source, lut, allow_pickle=False)
    lut_sha = hashlib.sha256(lut.tobytes()).hexdigest()
    receipts = []
    for projection in ("fused13", "down"):
        cell_root = tmp_path / "L007" / f"E003_{projection}"
        cell_root.mkdir(parents=True)
        arrays = {
            "codes": np.arange(96, dtype=np.uint8).reshape(1, 96),
            "SU": np.ones(16, dtype=np.float16),
            "SV": np.ones(16, dtype=np.float16),
            "Wscale": np.asarray(0.75, dtype=np.float32),
        }
        artifacts = {}
        for name, value in arrays.items():
            path = cell_root / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            artifacts[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "data_bytes": value.nbytes,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        receipt = cell_root / "CELL_RECEIPT.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "banana-smasher-periodic-qtip3-pr31-cell-v1",
                    "status": "PASS",
                    "task_id": "t_064d19ef",
                    "layer": 7,
                    "expert": 3,
                    "projection": projection,
                    "basis_index_sha256": BASIS_SHA,
                    "source_model_shard": {"tensor_sha256": "1" * 64},
                    "control": {"sha256": "2" * 64},
                    "geometry": {
                        "L": 16,
                        "B": 12,
                        "V": 4,
                        "rate_num": 3,
                        "rate_den": 1,
                        "phase_count": 1,
                        "alternation": False,
                        "member_averaging": False,
                    },
                    "tlut": {"path": str(lut_source), "tensor_sha256": lut_sha},
                    "accounting": {
                        "weights": 256,
                        "exact_code_bits": 768,
                        "exact_code_bpw": 3.0,
                        "code_data_bytes": 96,
                        "transform_bytes": 64,
                        "Wscale_bytes": 4,
                        "shared_tlut_bytes": 2048,
                    },
                    "artifacts": artifacts,
                }
            )
            + "\n"
        )
        receipts.append(receipt)
    manifest_path = tmp_path / "FIXED_QTIP3_MEMBERS.jsonl"
    terminal_path = tmp_path / "FIXED_QTIP3_MANIFEST.json"

    from banana_smasher import backpack_provider_from_declaration

    terminal = backpack_provider_from_declaration(PROVIDER_ID).materialize(
        cell_receipts=receipts,
        lut_source=lut_source,
        manifest_output=manifest_path,
        terminal_output=terminal_path,
        intended_basis_sha256=BASIS_SHA,
        expected_identities=[(7, 3, "fused13"), (7, 3, "down")],
    )

    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert terminal["status"] == "PASS"
    assert terminal["member_count"] == 2
    assert terminal["coverage"] == {"layers": [7], "members": 2}
    assert terminal["wire"] == {
        "logical_weights": 512,
        "code_data_bytes": 192,
        "transform_data_bytes": 128,
        "wscale_data_bytes": 8,
        "shared_lut_data_bytes": 2048,
        "full_routed_wire_bytes": 2376,
        "exact_code_bpw": 3.0,
        "full_routed_wire_bpw": 37.125,
    }
    assert terminal["shared_lut"]["tensor_sha256"] == lut_sha
    assert terminal["shared_lut"]["deduplicated_instances"] == 1
    assert terminal["manifest"]["sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert [row["projection"] for row in rows] == ["fused13", "down"]
    assert sum(row["cell_payload_bytes"] for row in rows) == 328
    assert len({row["activation_artifacts"][0]["id"] for row in rows}) == 1
    assert all(row["activation_artifacts"][0]["bytes"] == 2048 for row in rows)
    assert all(Path(row["artifacts"]["codes"]["path"]).is_file() for row in rows)

    loaded = tuple(
        qtip3.iter_qtip3_fixed_manifest(manifest_path, lut_path=lut_source)
    )
    assert [member.artifact_path.name for member in loaded] == [
        "CELL_RECEIPT.json",
        "CELL_RECEIPT.json",
    ]
    assert all(member.codes.shape == (1, 96) for member in loaded)
    assert all(member.lut_tensor_sha256 == lut_sha for member in loaded)
    assert loaded[0].lut.data_ptr() == loaded[1].lut.data_ptr()

    smoke_path = tmp_path / "QTIP3_RUNTIME_SMOKE.json"
    smoke = qtip3.smoke_qtip3_fixed_manifest(
        manifest_path=manifest_path,
        lut_path=lut_source,
        output=smoke_path,
        intended_basis_sha256=BASIS_SHA,
        expected_identities=[(7, 3, "fused13"), (7, 3, "down")],
        device="cpu",
    )
    assert smoke["status"] == "PASS"
    assert smoke["coverage"] == {"members_loaded": 2, "runtime_groups_executed": 2}
    assert [row["projection"] for row in smoke["runtime_groups"]] == [
        "down",
        "fused13",
    ]
    assert all(row["output_finite"] is True for row in smoke["runtime_groups"])
    assert smoke_path.is_file()


def test_materialize_qtip3_retained_weights_builds_sparse_shipping_shards(
    tmp_path: Path,
) -> None:
    qtip3 = _qtip3_fixed_module()
    source_root = tmp_path / "source"
    source_root.mkdir()
    keep = torch.arange(8, dtype=torch.float32)
    routed = torch.arange(16, dtype=torch.float32)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(
        {
            "embed.weight": keep,
            "layers.0.ffn.experts.0.w1.weight": routed,
        },
        source_root / shard_name,
    )
    source_index = source_root / "model.safetensors.index.json"
    source_index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": keep.nbytes + routed.nbytes},
                "weight_map": {
                    "embed.weight": shard_name,
                    "layers.0.ffn.experts.0.w1.weight": shard_name,
                },
            }
        )
        + "\n"
    )
    basis = hashlib.sha256(source_index.read_bytes()).hexdigest()
    retained_index = tmp_path / "retained.json"
    retained_index.write_text(
        json.dumps(
            {
                "basis_model_index_sha256": basis,
                "retained_tensors": [
                    {
                        "name": "embed.weight",
                        "shape": list(keep.shape),
                        "dtype": "F32",
                        "bytes": keep.nbytes,
                        "sha256": hashlib.sha256(keep.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    }
                ],
            }
        )
        + "\n"
    )
    output_root = tmp_path / "pack" / "retained"

    terminal = qtip3.materialize_qtip3_retained_weights(
        source_model_root=source_root,
        retained_index=retained_index,
        output_root=output_root,
        intended_basis_sha256=basis,
    )

    output_index = json.loads((output_root / "model.safetensors.index.json").read_text())
    assert terminal["status"] == "PASS"
    assert terminal["tensor_count"] == 1
    assert terminal["data_bytes"] == keep.nbytes
    assert output_index["weight_map"] == {"embed.weight": shard_name}
    with safe_open(output_root / shard_name, framework="pt", device="cpu") as handle:
        assert handle.keys() == ["embed.weight"]
        assert torch.equal(handle.get_tensor("embed.weight"), keep)


def test_qtip3_public_provider_verifies_complete_shipping_pack(tmp_path: Path) -> None:
    from banana_smasher import backpack_provider_from_declaration

    lut = np.linspace(-1.0, 1.0, 1024, dtype=np.float16)
    lut_path = tmp_path / "pr31-codebook.npy"
    np.save(lut_path, lut, allow_pickle=False)
    lut_tensor_sha = hashlib.sha256(lut.tobytes()).hexdigest()
    rows = []
    for projection in ("fused13", "down"):
        root = tmp_path / projection
        root.mkdir()
        arrays = {
            "codes": np.arange(96, dtype=np.uint8).reshape(1, 96),
            "SU": np.ones(16, dtype=np.float16),
            "SV": np.ones(16, dtype=np.float16),
            "Wscale": np.asarray(0.75, dtype=np.float32),
        }
        artifacts = {}
        for name, value in arrays.items():
            path = root / f"{name}.npy"
            np.save(path, value, allow_pickle=False)
            artifacts[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "data_bytes": value.nbytes,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "dtype": value.dtype.str,
                "shape": list(value.shape),
            }
        rows.append(
            {
                "schema": "banana-smasher-qtip3-fixed-manifest-member-v1",
                "status": "PASS",
                "codec_provider_id": PROVIDER_ID,
                "basis_index_sha256": BASIS_SHA,
                "layer": 7,
                "expert": 3,
                "projection": projection,
                "source_weight_sha256": "1" * 64,
                "hessian_sha256": "2" * 64,
                "geometry": {
                    "L": 16,
                    "B": 12,
                    "V": 4,
                    "layout": "homogeneous",
                    "phase_widths": [3, 3, 3, 3],
                },
                "receipt": {"path": str(root / "CELL_RECEIPT.json")},
                "artifacts": artifacts,
                "cell_payload_bytes": 164,
                "physical_bytes": 164,
                "logical_weights": 256,
                "activation_artifacts": [
                    {
                        "id": f"periodic-qtip3-pr31-lut-{lut_tensor_sha[:16]}",
                        "bytes": 2048,
                        "sha256": lut_tensor_sha,
                    }
                ],
            }
        )
    manifest = tmp_path / "FIXED_QTIP3_MEMBERS.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    routed_terminal = tmp_path / "FIXED_QTIP3_MANIFEST.json"
    routed_terminal.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip3-fixed-manifest-v1",
                "status": "PASS",
                "codec_provider_id": PROVIDER_ID,
                "basis_index_sha256": BASIS_SHA,
                "member_count": 2,
                "coverage": {"layers": [7], "members": 2},
                "manifest": {
                    "path": str(manifest),
                    "bytes": manifest.stat().st_size,
                    "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                },
                "shared_lut": {
                    "path": str(lut_path),
                    "storage_bytes": lut_path.stat().st_size,
                    "storage_sha256": hashlib.sha256(lut_path.read_bytes()).hexdigest(),
                    "data_bytes": 2048,
                    "tensor_sha256": lut_tensor_sha,
                    "deduplicated_instances": 1,
                },
                "wire": {
                    "logical_weights": 512,
                    "code_data_bytes": 192,
                    "transform_data_bytes": 128,
                    "wscale_data_bytes": 8,
                    "shared_lut_data_bytes": 2048,
                    "full_routed_wire_bytes": 2376,
                    "exact_code_bpw": 3.0,
                    "full_routed_wire_bpw": 37.125,
                },
            }
        )
        + "\n"
    )
    retained_index = tmp_path / "model.safetensors.index.json"
    retained_index.write_text(json.dumps({"metadata": {"total_size": 32}, "weight_map": {}}) + "\n")
    retained_shard = tmp_path / "model-00001-of-00001.safetensors"
    retained_shard.write_bytes(b"retained-shipping-payload")
    retained_terminal = tmp_path / "RETAINED_SHIPPING.json"
    retained_terminal.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip3-retained-shipping-v1",
                "status": "PASS",
                "basis_model_index_sha256": BASIS_SHA,
                "tensor_count": 1,
                "data_bytes": 32,
                "index": {
                    "path": str(retained_index),
                    "bytes": retained_index.stat().st_size,
                    "sha256": hashlib.sha256(retained_index.read_bytes()).hexdigest(),
                },
                "shards": [
                    {
                        "path": str(retained_shard),
                        "bytes": retained_shard.stat().st_size,
                        "sha256": hashlib.sha256(retained_shard.read_bytes()).hexdigest(),
                        "tensor_count": 1,
                    }
                ],
            }
        )
        + "\n"
    )
    output = tmp_path / "QTIP3_SHIPPING_PACK.json"

    terminal = backpack_provider_from_declaration(PROVIDER_ID).verify(
        routed_terminal=routed_terminal,
        retained_terminal=retained_terminal,
        output=output,
        intended_basis_sha256=BASIS_SHA,
        expected_layers=[7],
        expected_member_count=2,
        base_model_parameters=600,
    )

    assert terminal["status"] == "PASS"
    assert terminal["coverage"] == {"layers": 1, "members": 2, "retained_tensors": 1}
    assert terminal["wire"]["routed_bytes"] == 2376
    assert terminal["wire"]["retained_bytes"] == 32
    assert terminal["wire"]["whole_model_bytes"] == 2408
    assert terminal["wire"]["whole_model_bpw"] == pytest.approx(2408 * 8 / 600)
    assert terminal["shared_lut"]["tensor_sha256"] == lut_tensor_sha
    assert output.is_file()


def test_build_qtip3_retained_index_excludes_only_routed_expert_tensors(
    tmp_path: Path,
) -> None:
    qtip3 = _qtip3_fixed_module()
    source_root = tmp_path / "source"
    source_root.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    tensors = {
        "embed.weight": torch.arange(8, dtype=torch.float32),
        "layers.0.attn.wkv.weight": torch.arange(8, dtype=torch.float32).to(
            torch.float8_e4m3fn
        ),
        "layers.0.attn.wkv.scale": torch.ones(8, dtype=torch.float32).to(
            torch.float8_e8m0fnu
        ),
        "layers.0.ffn.experts.0.w1.weight": torch.arange(4, dtype=torch.float32),
        "layers.0.ffn.experts.0.w1.scale": torch.arange(2, dtype=torch.float32),
        "layers.0.ffn.shared_experts.w1.weight": torch.arange(6, dtype=torch.float32),
        "mtp.0.ffn.experts.0.w1.weight": torch.arange(10, dtype=torch.float32),
    }
    save_file(tensors, source_root / shard_name)
    source_index = source_root / "model.safetensors.index.json"
    source_index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.nbytes for tensor in tensors.values())},
                "weight_map": {name: shard_name for name in tensors},
            }
        )
        + "\n"
    )
    basis = hashlib.sha256(source_index.read_bytes()).hexdigest()
    output = tmp_path / "retained-index.json"

    terminal = qtip3.build_qtip3_retained_index(
        source_model_root=source_root,
        output=output,
        intended_basis_sha256=basis,
    )

    retained = json.loads(output.read_text())
    expected_names = {
        "embed.weight",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wkv.scale",
        "layers.0.ffn.shared_experts.w1.weight",
        "mtp.0.ffn.experts.0.w1.weight",
    }
    assert terminal["status"] == "PASS"
    assert terminal["retained_tensor_count"] == 5
    assert terminal["excluded_routed_tensor_count"] == 2
    assert {row["name"] for row in retained["retained_tensors"]} == expected_names
    dtype_by_name = {row["name"]: row["dtype"] for row in retained["retained_tensors"]}
    assert dtype_by_name["layers.0.attn.wkv.weight"] == "F8_E4M3"
    assert dtype_by_name["layers.0.attn.wkv.scale"] == "F8_E8M0"
    assert all(row["sha256"] == qtip3._tensor_sha256(tensors[row["name"]]) for row in retained["retained_tensors"])


def test_qtip3_legacy_identity_is_accepted_without_reinterpretation(tmp_path: Path) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path, provider_id=LEGACY_PROVIDER_ID)
    payload = torch.load(member_path, weights_only=True)
    payload["geometry"].pop("layout")
    torch.save(payload, member_path)

    loaded = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)

    assert loaded.codec_provider_id == LEGACY_PROVIDER_ID
    assert "layout" not in loaded.geometry


def test_qtip3_member_accepts_exact_three_bit_packed_code_geometry(tmp_path: Path) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path)
    payload = torch.load(member_path, weights_only=True)
    payload["codes"] = torch.arange(96, dtype=torch.uint8).reshape(1, 96)
    torch.save(payload, member_path)

    loaded = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)

    assert loaded.codes.shape == (1, 96)
    assert loaded.codes.numel() * 8 == loaded.SU.numel() * loaded.SV.numel() * 3


def test_public_backpack_registry_exposes_qtip3_fixed_assignment_lifecycle() -> None:
    _qtip3_fixed_module()
    from banana_smasher import (
        backpack_provider_from_declaration,
        builtin_backpack_family_providers,
    )

    bindings = builtin_backpack_family_providers()
    assert PROVIDER_ID in bindings
    assert LEGACY_PROVIDER_ID in bindings
    binding = bindings[PROVIDER_ID]
    assert binding.provider_id == PROVIDER_ID
    assert binding.kind == "fixed_qtip"
    assert binding.runtime_family == "periodic_qtip3"
    assert backpack_provider_from_declaration(PROVIDER_ID) == binding
    assert (
        backpack_provider_from_declaration(LEGACY_PROVIDER_ID).provider_id
        == LEGACY_PROVIDER_ID
    )
    assert all(
        callable(value)
        for value in (
            binding.generate,
            binding.materialize,
            binding.price,
            binding.predict,
            binding.verify,
        )
    )


def test_qtip3_public_repair_microdose_changes_only_authorized_continuous_state(
    tmp_path: Path,
) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path)
    member = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)
    runtime = qtip3.Qtip3FixedRepairRuntime(
        members=[member], learning_rate=0.05, device="cpu"
    )
    codes_before = member.codes.clone()
    geometry_before = dict(member.geometry)
    lut_before = runtime.shared_lut.detach().clone()

    receipt = runtime.microdose(
        activation_inputs=torch.ones((1, 4, 16), dtype=torch.float32),
        teacher_targets=torch.zeros((1, 4, 16), dtype=torch.float32),
        teacher_mask=torch.ones((1, 4), dtype=torch.bool),
    )

    assert receipt["status"] == "PASS_UPDATE"
    assert receipt["finite_nonzero_gradients"] is True
    assert receipt["authorized_parameter_delta"] > 0
    assert receipt["packed_codes_unchanged"] is True
    assert receipt["geometry_unchanged"] is True
    assert receipt["acceleration_counters"]["periodic_qtip3_lut_vjp_calls"] > 0
    assert receipt["acceleration_counters"]["fallback_calls"] == 0
    assert torch.equal(member.codes, codes_before)
    assert member.geometry == geometry_before
    assert not torch.equal(runtime.shared_lut.detach(), lut_before)

    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert '[project.entry-points."banana_smasher.update_backends"]' in project
    assert "periodic-qtip3" in project


def test_qtip3_public_decode_executes_fixed_member_runtime(tmp_path: Path) -> None:
    from banana_smasher import decode_qtip3_fixed_member

    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path)
    member = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)

    weight = decode_qtip3_fixed_member(member, device="cpu")

    assert weight.shape == (16, 16)
    assert weight.dtype == torch.float32
    assert weight.device.type == "cpu"
    assert torch.isfinite(weight).all()

    parity = qtip3.compare_qtip3_fixed_member_devices(
        member,
        reference_device="cpu",
        candidate_device="cpu",
        atol=0.0,
        rtol=0.0,
    )
    assert parity["status"] == "PASS"
    assert parity["max_abs_error"] == 0.0
    assert parity["reference_sha256"] == parity["candidate_sha256"]
    assert parity["fallback"] == {"used": False}


def test_qtip3_public_update_backend_accumulates_exact_segments_with_real_lut_vjp(
    tmp_path: Path,
) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path)
    teacher_batch = tmp_path / "teacher-batch.pt"
    torch.save(
        {
            "activation_inputs": torch.ones((1, 4, 16), dtype=torch.float32),
            "teacher_targets": torch.zeros((1, 4, 16), dtype=torch.float32),
            "teacher_mask": torch.ones((1, 4), dtype=torch.bool),
        },
        teacher_batch,
    )
    request = tmp_path / "request.json"
    request.write_text(
        __import__("json").dumps(
            {
                "schema": qtip3.QTIP3_UPDATE_SCHEMA,
                "members": [
                    {
                        "artifact": member_path.name,
                        "lut": lut_path.name,
                    }
                ],
                "teacher_batch": teacher_batch.name,
                "learning_rate": 0.05,
                "device": "cpu",
            }
        )
        + "\n"
    )
    output = tmp_path / "repair.pt"
    receipt_path = tmp_path / "repair.receipt.json"

    receipt = qtip3.run_qtip3_fixed_update(
        request=request,
        output=output,
        receipt=receipt_path,
        identity={
            "content_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "assignment_sha256": "3" * 64,
            "aot_sha256": "4" * 64,
            "runtime_sha256": "5" * 64,
            "code_sha256": "6" * 64,
        },
        requested_tokens=2,
        physical_tokens=2,
        segments=2,
        batch_size=1,
        memory_sizing={"physical_tokens": 2},
        resume=True,
        restart=False,
    )

    assert receipt["completed_segments"] == 2
    assert receipt["forward_count"] == 2
    assert receipt["backward_count"] == 2
    assert receipt["optimizer_steps"] == 1
    assert receipt["observed_input_shape"] == [1, 2]
    assert receipt["fixed_qtip3"]["packed_codes_unchanged"] is True
    assert receipt["fixed_qtip3"]["transforms_unchanged"] is True
    assert receipt["fixed_qtip3"]["geometry_unchanged"] is True
    counters = receipt["fixed_qtip3"]["acceleration_counters"]
    assert counters["periodic_qtip3_lut_gather_calls"] == 2
    assert counters["periodic_qtip3_lut_vjp_calls"] == 2
    assert counters["fallback_calls"] == 0
    assert output.is_file()
    assert receipt_path.is_file()
