from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import banana_smasher.backpack as backpack_module

from banana_smasher.backpack import (
    BackpackPlan,
    BackpackFamilyBinding,
    BackpackPlanError,
    anchor_backpack,
    anchor_backpack_candidates,
    price_backpack_selection,
    build_backpack,
    list_backpack_family_bindings,
    generate_backpack_candidates,
    export_backpack_lifecycle,
    generate_qtip_backpack_candidate,
    generate_vector_vq_backpack_candidate,
    inspect_backpack,
    materialize_backpack_source,
    pack_indices,
    predict_backpack,
    quantize_vector_cell,
    repair_backpack,
    resolve_backpack_family,
    reuse_backpack_receipts,
    score_backpack,
    solve_backpack,
    status_backpack,
)
from banana_smasher.cli import _parser, main
from banana_smasher.contract import (
    KERNEL_MANIFEST_NAME,
    PackValidationError,
    export_pack,
    layout_sha256,
    load_manifest,
    verify_pack,
)
from banana_smasher.loader import PackLoader
from banana_smasher.repack import repack_to_safetensors
from banana_smasher.repair import (
    CodebookRepair,
    REPAIR_STATE_PATH,
    RepairBundle,
    write_repair_payload,
)


CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _fixture_plan(tmp_path: Path, *, exact_bytes: int = 53344) -> dict[str, object]:
    model = tmp_path / "model"
    model.mkdir(parents=True)
    rng = np.random.default_rng(7)
    cells = []
    weights = []
    for expert_ids in (range(128), range(128, 256)):
        for projection in ("fused13", "down"):
            index = len(cells)
            value = rng.normal(size=(len(expert_ids), 16)).astype(np.float32)
            np.save(model / f"cell{index}.npy", value, allow_pickle=False)
            weights.append(value)
            feature_start = sum(array.size for array in weights[:-1])
            cells.append(
                {
                    "cell_id": f"cell{index}",
                    "path": f"cell{index}.npy",
                    "feature_slice": [feature_start, feature_start + value.size],
                    "layer": 0,
                    "projection": projection,
                    "expert_ids": list(expert_ids),
                }
            )
    (model / "BACKPACK_MODEL.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-backpack-model-v1",
                "revision": "fixture-r1",
                "weight_count": sum(value.size for value in weights),
                "dense_bytes": 0,
                "metadata_bytes": 0,
                "repair_bytes": 0,
                "cells": cells,
            }
        )
        + "\n"
    )
    features = rng.normal(size=(64, sum(value.size for value in weights))).astype(np.float32)
    classes = np.asarray([CLASSES[index % len(CLASSES)] for index in range(64)])
    bank = tmp_path / "anchor64.npz"
    np.savez(bank, features=features, classes=classes)
    return {
        "schema": "banana-smasher-backpack-plan-v1",
        "model": {"root": str(model), "revision": "fixture-r1"},
        "target": {"exact_bytes": exact_bytes},
        "tiers": [
            {"id": "d4-k4", "family": "vector_vq", "dimension": 4, "bits": 2},
            {"id": "d8-k4", "family": "vector_vq", "dimension": 8, "bits": 2},
            {
                "id": "qtip-2.0",
                "family": "qtip",
                "bpw": 2.0,
                "backend": "fixture_reference",
            },
        ],
        "anchor": {"bank": str(bank), "teacher": "model"},
        "prediction": {"class_caps": {name: 100.0 for name in CLASSES}},
        "repair": {"method": "fixture_residual", "strength": 0.5},
        "output": {
            "pack": str(tmp_path / "final-pack"),
            "model_id": "tiny-backpack",
            "instance_id": "tiny-backpack-1",
        },
    }


def _decode_strings(array: np.ndarray) -> list[str]:
    values = np.asarray(array)
    if values.dtype == np.uint8 and values.ndim == 2:
        return [bytes(row).rstrip(b"\0").decode("utf-8") for row in values]
    return [
        value.decode("utf-8") if isinstance(value, (bytes, bytearray, np.bytes_)) else str(value)
        for value in values.reshape(-1)
    ]


def _serving_model(tmp_path: Path) -> Path:
    from safetensors.numpy import save_file

    root = tmp_path / "serving-model"
    root.mkdir(parents=True)
    shard = "model-00001-of-00001.safetensors"
    tensors = {
        "model.norm.weight": np.ones(4, dtype=np.float32),
        "model.layers.0.self_attn.o_b_proj.weight": np.ones((4, 4), dtype=np.float32),
    }
    save_file(tensors, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
                "weight_map": {name: shard for name in tensors},
            },
            sort_keys=True,
        )
        + "\n"
    )
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "model_type": "deepseek_v3",
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "scale_fmt": "float32",
                    "weight_block_size": [128, 128],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (root / name).write_text("{}\n")
    return root


def _kernel_cache(tmp_path: Path) -> Path:
    root = tmp_path / "kernel-cache"
    root.mkdir()
    adapter = root / "runtime_adapter.py"
    adapter.write_text(
        "class RuntimeAdapter:\n"
        "    API_VERSION = 1\n"
        "    def build_layer(self, **kwargs): return kwargs\n"
        "    def forward(self, state, **kwargs): return state\n"
    )
    (root / KERNEL_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema": "bs-kernel-cache",
                "schema_version": 1,
                "quant_method": "banana_smasher",
                "pack_schema": "bs-pack",
                "pack_schema_version": 1,
                "tensor_layout_sha256": layout_sha256(),
                "families": ["qtip2", "truevq_d4", "truevq_d8"],
                "architectures": ["sm_120"],
                "runtime_adapter": {
                    "path": adapter.name,
                    "class": "RuntimeAdapter",
                    "api_version": 1,
                },
                "files": [
                    {
                        "path": adapter.name,
                        "bytes": adapter.stat().st_size,
                        "sha256": hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return root


def test_plan_rejects_impossible_or_ambiguous_tiers(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    plan["target"] = {"exact_bytes": 70, "whole_model_bpw": 2.0}
    with pytest.raises(BackpackPlanError, match="exactly one"):
        BackpackPlan.from_mapping(plan)

    plan = _fixture_plan(tmp_path / "dimension")
    plan["tiers"][0]["dimension"] = 6  # type: ignore[index]
    with pytest.raises(BackpackPlanError, match="dimension must be 4 or 8"):
        BackpackPlan.from_mapping(plan)

    plan = _fixture_plan(tmp_path / "qtip")
    plan["tiers"][2]["bpw"] = 2.1  # type: ignore[index]
    with pytest.raises(BackpackPlanError, match="0.25"):
        BackpackPlan.from_mapping(plan)

    plan = _fixture_plan(tmp_path / "empty-codebook")
    plan["tiers"][0] = {  # type: ignore[index]
        "id": "d4-k1",
        "family": "vector_vq",
        "dimension": 4,
        "codebook_size": 1,
    }
    with pytest.raises(BackpackPlanError, match="2..65536"):
        BackpackPlan.from_mapping(plan)

    plan = _fixture_plan(tmp_path / "wide-index")
    plan["tiers"][0]["bits"] = 17  # type: ignore[index]
    with pytest.raises(BackpackPlanError, match="1..16"):
        BackpackPlan.from_mapping(plan)

    plan = _fixture_plan(tmp_path / "unknown")
    plan["typo"] = True
    with pytest.raises(BackpackPlanError, match="unknown fields: typo"):
        BackpackPlan.from_mapping(plan)

    plan = _fixture_plan(tmp_path / "packaged-qtip")
    plan["tiers"][2] = {"id": "qtip-2.0", "family": "qtip", "bpw": 2.0}  # type: ignore[index]
    with pytest.raises(BackpackPlanError, match="source_root"):
        BackpackPlan.from_mapping(plan)


def test_plan_schema_matches_parser_surface(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schema/banana-smasher-backpack-plan-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(plan, schema)

    empty_reuse = _fixture_plan(tmp_path / "empty-reuse")
    empty_reuse["reuse_receipts"] = []
    BackpackPlan.from_mapping(empty_reuse)
    jsonschema.validate(empty_reuse, schema)

    duplicate_tier = _fixture_plan(tmp_path / "duplicate-tier")
    duplicate_tier["tiers"].append(dict(duplicate_tier["tiers"][0]))  # type: ignore[union-attr]
    with pytest.raises(BackpackPlanError, match="duplicate tier id"):
        BackpackPlan.from_mapping(duplicate_tier)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(duplicate_tier, schema)

    invalid = _fixture_plan(tmp_path / "invalid")
    invalid["reuse_receipts"] = [
        {
            "role": "candidate_stage",
            "path": str(tmp_path / "candidate-stage.json"),
            "sha256": "0" * 64,
            "schema": "wrong-stage-schema",
            "stage": "candidates",
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)

    missing_stage_schema = _fixture_plan(tmp_path / "missing-stage-schema")
    missing_stage_schema["reuse_receipts"] = [
        {
            "role": "candidate_stage",
            "path": str(tmp_path / "candidate-stage.json"),
            "sha256": "0" * 64,
            "stage": "candidates",
        }
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_stage_schema, schema)
    with pytest.raises(BackpackPlanError, match="schema is required"):
        BackpackPlan.from_mapping(missing_stage_schema)


@pytest.mark.parametrize("tier_id", ["../escape", "nested/tier", "nested\\tier", "."])
def test_plan_rejects_tier_ids_that_are_not_safe_path_components(
    tmp_path: Path, tier_id: str
) -> None:
    plan = _fixture_plan(tmp_path)
    plan["tiers"][0]["id"] = tier_id  # type: ignore[index]

    with pytest.raises(BackpackPlanError, match="safe path component"):
        BackpackPlan.from_mapping(plan)


def test_plan_rejects_tier_ids_that_collide_on_case_insensitive_filesystems(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    plan["tiers"][1]["id"] = "D4-K4"  # type: ignore[index]

    with pytest.raises(BackpackPlanError, match="colliding tier id"):
        BackpackPlan.from_mapping(plan)


def test_inspect_rejects_cell_ids_that_are_not_safe_path_components(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    manifest_path = Path(str(plan["model"]["root"])) / "BACKPACK_MODEL.json"  # type: ignore[index]
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["cell_id"] = "../escape"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(BackpackPlanError, match="safe path component"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_inspect_rejects_cell_ids_that_collide_on_case_insensitive_filesystems(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    manifest_path = Path(str(plan["model"]["root"])) / "BACKPACK_MODEL.json"  # type: ignore[index]
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][1]["cell_id"] = "CELL0"
    manifest_path.write_text(json.dumps(manifest) + "\n")

    with pytest.raises(BackpackPlanError, match="colliding model cell"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_inspect_rejects_direct_symlink_model_cell(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    model_root = Path(str(plan["model"]["root"]))  # type: ignore[index]
    cell = model_root / "cell0.npy"
    target = model_root / "cell0-target.npy"
    cell.rename(target)
    cell.symlink_to(target)

    with pytest.raises(BackpackPlanError, match="regular NPY file"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_inspect_requires_complete_expert_partition_per_projection(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    model_root = Path(str(plan["model"]["root"]))  # type: ignore[index]
    manifest_path = model_root / "BACKPACK_MODEL.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][1]["expert_ids"] = list(range(255))
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(BackpackPlanError, match="projection down"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_inspect_requires_matching_expert_partitions_across_projections(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    model_root = Path(str(plan["model"]["root"]))  # type: ignore[index]
    manifest_path = model_root / "BACKPACK_MODEL.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][1]["expert_ids"] = list(range(64)) + list(range(128, 192))
    manifest["cells"][3]["expert_ids"] = list(range(64, 128)) + list(range(192, 256))
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(BackpackPlanError, match="identical expert partitions"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_inspect_rejects_explicit_direct_symlink_model_manifest(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    model_root = Path(str(plan["model"]["root"]))  # type: ignore[index]
    manifest = model_root / "BACKPACK_MODEL.json"
    target = model_root / "BACKPACK_MODEL.target.json"
    manifest.rename(target)
    manifest.symlink_to(target)
    plan["model"]["manifest"] = str(manifest)  # type: ignore[index]

    with pytest.raises(BackpackPlanError, match="manifest must be a regular file"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_inspect_rejects_direct_symlink_anchor_bank(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    bank = Path(str(plan["anchor"]["bank"]))  # type: ignore[index]
    target = tmp_path / "anchor64-target.npz"
    bank.rename(target)
    bank.symlink_to(target)

    with pytest.raises(BackpackPlanError, match="anchor.bank must be a regular NPZ"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_inspect_rejects_direct_symlink_custom_teacher(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    target = tmp_path / "teacher-target.npy"
    np.save(target, np.zeros(64, dtype=np.float32), allow_pickle=False)
    teacher = tmp_path / "teacher.npy"
    teacher.symlink_to(target)
    plan["anchor"]["teacher"] = str(teacher)  # type: ignore[index]

    with pytest.raises(BackpackPlanError, match="anchor.teacher must be a regular NPY"):
        inspect_backpack(plan, run_root=tmp_path / "run")


def test_candidate_generation_rejects_nested_run_output_symlink(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    inspect_backpack(plan, run_root=run_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    tier_root = run_root / "candidates" / "d4-k4"
    tier_root.parent.mkdir()
    tier_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BackpackPlanError, match="candidate output path must not be a symlink"):
        generate_backpack_candidates(plan, run_root=run_root)

    assert list(outside.iterdir()) == []


def test_build_rejects_direct_run_root_symlink(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    run_root = tmp_path / "run-link"
    run_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BackpackPlanError, match="run root must not be a direct symlink"):
        build_backpack(plan, run_root=run_root)

    assert list(outside.iterdir()) == []


def test_materializer_rejects_symlinked_destination_parent_without_touching_target(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    parsed = BackpackPlan.from_mapping(plan)
    run_root = tmp_path / "run"
    inspect_backpack(parsed, run_root=run_root)
    candidates = generate_backpack_candidates(parsed, run_root=run_root)
    _manifest, cells = backpack_module._load_cells(parsed)
    assignment = [
        {"cell_id": cell["cell_id"], "tier": "d4-k4"}
        for cell in cells
    ]
    artifact_roots = {
        str(cell["cell_id"]): backpack_module.candidate_artifact_root(
            candidates,
            tier="d4-k4",
            cell_id=str(cell["cell_id"]),
        )
        for cell in cells
    }
    outside = tmp_path / "outside"
    source = outside / "pre-repair-source"
    source.mkdir(parents=True)
    sentinel = source / "keep.txt"
    sentinel.write_text("do not delete\n")
    materialized = tmp_path / "materialized"
    materialized.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BackpackPlanError, match="materializer destination.*symlink"):
        materialize_backpack_source(
            materialized / "pre-repair-source",
            plan=parsed,
            cells=cells,
            assignment=assignment,
            artifact_roots=artifact_roots,
        )

    assert sentinel.read_text() == "do not delete\n"


def test_d4_and_d8_use_real_vector_geometry_and_bit_packing() -> None:
    weights = np.arange(16, dtype=np.float32)
    d4 = quantize_vector_cell(weights, dimension=4, bits=2)
    d8 = quantize_vector_cell(weights, dimension=8, bits=2)

    assert d4["vectors"].shape == (4, 4)
    assert d8["vectors"].shape == (2, 8)
    assert d4["codebook"].shape[1] == 4
    assert d8["codebook"].shape[1] == 8
    assert d4["decoded"].shape == d8["decoded"].shape == (16,)
    assert len(pack_indices(np.asarray([0, 1, 2, 3]), bits=2)) == 1
    assert len(pack_indices(np.asarray([0, 3, 1]), bits=2)) == 1


@pytest.mark.parametrize("family", ["vector_vq", "qtip"])
def test_direct_candidate_generation_materializes_expert_specific_weights(
    tmp_path: Path,
    family: str,
) -> None:
    first = np.linspace(-4.0, -1.0, 16, dtype=np.float32)
    second = np.linspace(1.0, 7.0, 16, dtype=np.float32)
    cell = {
        "cell_id": "cell",
        "layer": 0,
        "projection": "down",
        "expert_ids": [3, 9],
        "weights": np.concatenate([first, second]),
    }
    if family == "vector_vq":
        tier = {"id": "d4", "family": "vector_vq", "dimension": 4, "bits": 2}
        receipt = generate_vector_vq_backpack_candidate(
            tmp_path,
            tier=tier,
            cell=cell,
        )
    else:
        tier = {
            "id": "qtip",
            "family": "qtip",
            "bpw": 2.0,
            "backend": "fixture_reference",
        }
        receipt = generate_qtip_backpack_candidate(
            tmp_path,
            tier=tier,
            cell=cell,
            geometry_by_identity={(0, 3, "down"): (16, 2, 2), (0, 9, "down"): (16, 2, 2)},
        )

    root = Path(receipt["wire"]["path"]).parent
    wire = (root / "wire.bin").read_bytes()
    offsets = np.load(root / "tensor_offsets.npy", allow_pickle=False)
    codebooks = np.load(root / "codebooks.npy", allow_pickle=False).view(np.uint8).reshape(-1)
    first_payload = (
        wire[int(offsets[0, 0]) : int(offsets[1, 0])],
        bytes(codebooks[int(offsets[0, 2]) : int(offsets[1, 2])]),
    )
    second_payload = (
        wire[int(offsets[1, 0]) : int(offsets[2, 0])],
        bytes(codebooks[int(offsets[1, 2]) : int(offsets[2, 2])]),
    )
    decoded = np.load(root / "decoded.npy", allow_pickle=False)

    assert first_payload != second_payload
    assert decoded.shape == (32,)
    assert not np.array_equal(decoded[:16], decoded[16:])


def test_packaged_qtip_candidate_consumes_hash_bound_unit_artifacts(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from banana_smasher.qtip_rings import qtip_ring_manifest, resolve_qtip_ring

    source_root = tmp_path / "packaged-qtip"
    source_root.mkdir()
    ring = resolve_qtip_ring(2.0)
    (source_root / "QTIP_RUN_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-run-manifest-v1",
                "status": "PASS",
                "tiers": [{"name": ring.tier, "ring": qtip_ring_manifest(ring)}],
            },
            sort_keys=True,
        )
        + "\n"
    )
    expert_ids = [3, 9]
    decoded_rows = []
    for offset, expert in enumerate(expert_ids):
        config = source_root / "L000" / f"E{expert:03d}_down.json"
        config.parent.mkdir(exist_ok=True)
        config.write_text(
            json.dumps(
                {
                    "layer": 0,
                    "expert": expert,
                    "projection": "down",
                    "geometry": {"L": 16, "K": 2, "V": 2},
                },
                sort_keys=True,
            )
            + "\n"
        )
        unit_root = source_root / "solve" / "L000" / f"E{expert:03d}_down"
        unit_root.mkdir(parents=True)
        artifact = unit_root / "QTIP_UNIT.pt"
        trellis = torch.tensor([[expert, offset]], dtype=torch.uint16)
        reconstructed = torch.arange(16, dtype=torch.float16) + 20 * offset
        decoded_rows.append(reconstructed.float().numpy())
        torch.save(
            {
                "schema": "banana-smasher-qtip-unit-v1",
                "shape": [4, 4],
                "trellis": trellis,
                "SU": torch.full((4,), offset + 1, dtype=torch.float16),
                "SV": torch.full((4,), offset + 2, dtype=torch.float16),
                "Wscale": torch.tensor(float(offset + 1)),
                "tlut": torch.full((4, 2), offset + 3, dtype=torch.float16),
                "reconstructed_weight": reconstructed,
                "geometry": {
                    "L": 16,
                    "K": 2,
                    "V": 2,
                    "tlut_bits": 9,
                    "decode_mode": "quantlut_sym",
                    "td_x": 16,
                    "td_y": 16,
                },
            },
            artifact,
        )
        receipt = unit_root / "QTIP_SOLVE_RECEIPT.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "banana-smasher-qtip-solve-v1",
                    "status": "PASS",
                    "layer": 0,
                    "expert": expert,
                    "projection": "down",
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                    "artifact": artifact.name,
                    "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "assignment_sha256": hashlib.sha256(
                        trellis.numpy().tobytes(order="C")
                    ).hexdigest(),
                },
                sort_keys=True,
            )
            + "\n"
        )

    tier = {
        "id": "qtip",
        "family": "qtip",
        "bpw": 2.0,
        "backend": "packaged_qtip",
        "source_root": str(source_root),
    }
    cell = {
        "cell_id": "cell",
        "layer": 0,
        "projection": "down",
        "expert_ids": expert_ids,
        "weights": np.zeros(32, dtype=np.float32),
    }
    receipt = generate_qtip_backpack_candidate(
        tmp_path / "run",
        tier=tier,
        cell=cell,
        geometry_by_identity={(0, 3, "down"): (16, 2, 2), (0, 9, "down"): (16, 2, 2)},
    )

    candidate_root = Path(receipt["wire"]["path"]).parent
    assert receipt["algorithm"] == "qtip-packaged-v1"
    assert len(receipt["source_units"]) == 2
    assert np.array_equal(
        np.load(candidate_root / "decoded.npy", allow_pickle=False),
        np.concatenate(decoded_rows),
    )
    assert backpack_module._validate_candidate_receipt(
        str(candidate_root / "RECEIPT.json"),
        tier=tier,
        cell=cell,
        geometry_by_identity={(0, 3, "down"): (16, 2, 2), (0, 9, "down"): (16, 2, 2)},
    )
    source_artifact = source_root / "solve/L000/E003_down/QTIP_UNIT.pt"
    source_artifact.write_bytes(source_artifact.read_bytes() + b"tamper")
    assert not backpack_module._validate_candidate_receipt(
        str(candidate_root / "RECEIPT.json"),
        tier=tier,
        cell=cell,
        geometry_by_identity={(0, 3, "down"): (16, 2, 2), (0, 9, "down"): (16, 2, 2)},
    )


def test_custom_teacher_is_hash_bound_and_used(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    teacher = tmp_path / "teacher.npy"
    np.save(teacher, np.zeros(8192, dtype=np.float32), allow_pickle=False)
    plan["anchor"]["teacher"] = str(teacher)  # type: ignore[index]

    inspected = inspect_backpack(plan, run_root=tmp_path / "run")

    assert inspected["teacher"]["kind"] == "npy"
    assert inspected["teacher"]["sha256"] == hashlib.sha256(teacher.read_bytes()).hexdigest()


def test_qtip_fractional_tier_routes_through_explicit_fixture_ring(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=1000)
    plan["tiers"] = [
        {
            "id": "qtip-2.25",
            "family": "qtip",
            "bpw": 2.25,
            "backend": "fixture_reference",
        }
    ]
    run_root = tmp_path / "run"
    inspect_backpack(plan, run_root=run_root)

    generated = generate_backpack_candidates(plan, run_root=run_root)

    receipt = json.loads(
        Path(generated["candidate_tiers"][0]["cells"][0]["receipt"]).read_text()
    )
    assert receipt["algorithm"] == "qtip-fixture-reference"
    assert receipt["backend"] == "fixture_reference"
    assert receipt["ring"]["tier"] == "qtip@2.25"
    assert {
        tuple(row["geometry"][key] for key in ("L", "K", "V"))
        for row in receipt["records"]
    } == {
        (16, 2, 2),
        (16, 3, 2),
    }


def test_qtip_fractional_repair_preserves_global_ring_assignment(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=49760)
    plan["tiers"] = [
        {
            "id": "qtip-2.25",
            "family": "qtip",
            "bpw": 2.25,
            "backend": "fixture_reference",
        }
    ]

    result = build_backpack(plan, run_root=tmp_path / "run")

    assert result["status"] == "PASS"
    assert verify_pack(Path(result["final_pack"]))["status"] == "PASS"


def test_qtip_materialization_preserves_offsets_and_exact_ring_geometry(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=2000)
    plan["tiers"] = [
        {
            "id": "qtip-2.0",
            "family": "qtip",
            "bpw": 2.0,
            "backend": "fixture_reference",
        },
        {
            "id": "qtip-2.25",
            "family": "qtip",
            "bpw": 2.25,
            "backend": "fixture_reference",
        },
    ]
    run_root = tmp_path / "run"
    inspect_backpack(plan, run_root=run_root)
    parsed = BackpackPlan.from_mapping(plan)
    _manifest, cells = backpack_module._load_cells(parsed)
    geometry_by_tier = {
        str(tier["id"]): backpack_module._exact_qtip_geometries(tier, cells)
        for tier in parsed.tiers
    }

    assignment = []
    artifact_roots = {}
    for index, cell in enumerate(cells):
        tier = parsed.tiers[0 if index < 2 else 1]
        geometry = geometry_by_tier[str(tier["id"])]
        receipt = generate_qtip_backpack_candidate(
            run_root,
            tier=tier,
            cell=cell,
            geometry_by_identity=geometry,
        )
        cell_id = str(cell["cell_id"])
        assignment.append({"cell_id": cell_id, "tier": tier["id"]})
        artifact_roots[cell_id] = Path(receipt["wire"]["path"]).parent  # type: ignore[index]
    source = tmp_path / "materialized-qtip"
    materialize_backpack_source(
        source,
        plan=parsed,
        cells=cells,
        assignment=assignment,
        artifact_roots=artifact_roots,
    )
    pack = tmp_path / "qtip-pack"
    export_pack(
        source_root=source,
        output=pack,
        model_id="fixture-model",
        instance_id="qtip-mixed-records",
        link_mode="copy",
    )
    manifest = load_manifest(pack)

    for field in (
        "expert_ids",
        "tensor_offsets",
        "record_tiers",
        "record_geometry",
        "record_projections",
        "record_boundaries",
    ):
        assert f"layers.0.qtip2.{field}" in manifest["tensor_index"]
    expert_ids = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.expert_ids"]["path"],
        allow_pickle=False,
    )
    offsets = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.tensor_offsets"]["path"],
        allow_pickle=False,
    )
    codes = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.codes"]["path"],
        allow_pickle=False,
    )
    scales = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.scales"]["path"],
        allow_pickle=False,
    )
    codebooks = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.codebooks"]["path"],
        allow_pickle=False,
    )
    tiers = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.record_tiers"]["path"],
        allow_pickle=False,
    )
    geometry = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.record_geometry"]["path"],
        allow_pickle=False,
    )
    projections = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.record_projections"]["path"],
        allow_pickle=False,
    )
    boundaries = np.load(
        pack / manifest["tensor_index"]["layers.0.qtip2.record_boundaries"]["path"],
        allow_pickle=False,
    )

    assert offsets.dtype == np.int64
    assert offsets.shape == (expert_ids.size + 1, 3)
    assert offsets[0].tolist() == [0, 0, 0]
    assert offsets[-1].tolist() == [codes.nbytes, scales.nbytes, codebooks.nbytes]
    assert np.all(np.diff(offsets, axis=0) >= 0)
    assert tiers.shape[0] == projections.shape[0] == expert_ids.size
    assert tiers.dtype == np.uint8 and tiers.shape == (expert_ids.size, 32)
    assert projections.dtype == np.uint8 and projections.shape == (expert_ids.size, 8)
    assert geometry.dtype == np.int32
    assert set(_decode_strings(tiers)) == {"qtip@2.00", "qtip@2.25"}
    assert {tuple(int(value) for value in row) for row in geometry} >= {
        (16, 2, 2),
        (16, 3, 2),
    }
    assert _decode_strings(projections)[0] == "fused13"
    assert _decode_strings(projections)[-1] == "down"
    assert {int(value) for value in np.diff(offsets[:, 0])} >= {4, 6}
    assert boundaries.tolist() == [[128, 128, 128], [256, 256, 256], [384, 384, 384]]
    repack_pack = tmp_path / "qtip-pack-repack"
    shutil.copytree(pack, repack_pack)
    assert repack_to_safetensors(repack_pack)["status"] == "PASS"

    tier_labels = _decode_strings(tiers)
    swap_index = next(
        index
        for index, (tier_label, row) in enumerate(zip(tier_labels, geometry, strict=True))
        if tier_label == "qtip@2.25" and int(row[1]) == 2
    )
    geometry[swap_index, 1] = 3
    geometry_name = "layers.0.qtip2.record_geometry"
    geometry_tensor = manifest["tensor_index"][geometry_name]
    geometry_path = pack / geometry_tensor["path"]
    np.save(geometry_path, geometry, allow_pickle=False)
    geometry_file = next(
        row for row in manifest["files"] if row["path"] == geometry_tensor["path"]
    )
    geometry_file["bytes"] = geometry_path.stat().st_size
    geometry_file["sha256"] = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    geometry_tensor["data_sha256"] = hashlib.sha256(
        geometry.tobytes(order="C")
    ).hexdigest()
    (pack / "BANANA_PACK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    with pytest.raises(PackValidationError, match="QTIP ring assignment mismatch"):
        verify_pack(pack)


def test_vector_candidate_preserves_k65536_geometry_without_int16_overflow(
    tmp_path: Path,
) -> None:
    receipt = generate_vector_vq_backpack_candidate(
        tmp_path,
        tier={
            "id": "d4-k65536",
            "family": "vector_vq",
            "dimension": 4,
            "bits": 16,
        },
        cell={
            "cell_id": "cell0",
            "layer": 0,
            "projection": "fused13",
            "expert_ids": [0],
            "weights": np.arange(16, dtype=np.float32),
        },
    )
    geometry = np.load(
        Path(receipt["wire"]["path"]).parent / "record_geometry.npy",
        allow_pickle=False,
    )

    assert geometry.dtype == np.int32
    assert geometry.tolist() == [[4, 16, 65536]]


def test_synthetic_end_to_end_builds_mixed_verified_pack_and_resumes(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"

    result = build_backpack(plan, run_root=run_root)

    assert result["status"] == "PASS"
    assert result["stages"] == [
        "inspect",
        "candidates",
        "candidate_anchor",
        "pred",
        "solve_materialize",
        "pre_repair_anchor",
        "repair",
        "final_score",
    ]
    assert {row["family"] for row in result["candidate_tiers"]} == {
        "vector_vq",
        "qtip",
    }
    assert {row["dimension"] for row in result["candidate_tiers"] if row["family"] == "vector_vq"} == {4, 8}
    assert len({row["tier"] for row in result["assignment"]}) >= 2
    selected = {row["cell_id"]: row["tier"] for row in result["assignment"]}
    assert selected["cell0"] == selected["cell1"]
    assert selected["cell2"] == selected["cell3"]
    assert result["byte_accounting"]["whole_model_bytes"] == plan["target"]["exact_bytes"]
    assert result["pre_repair_anchor"]["windows"] == 64
    assert set(result["pre_repair_anchor"]["by_class"]) == set(CLASSES)
    assert result["final_anchor"]["overall"]["kld"] <= result["pre_repair_anchor"]["overall"]["kld"]
    assert verify_pack(tmp_path / "final-pack")["status"] == "PASS"
    pack_manifest = load_manifest(tmp_path / "final-pack")
    assert pack_manifest["layers"] == [0]
    tier_map_name = "layers.0.experts.tier_map"
    assert tier_map_name in pack_manifest["tensor_index"]
    tier_map = np.load(
        tmp_path / "final-pack" / pack_manifest["tensor_index"][tier_map_name]["path"],
        allow_pickle=False,
    )
    assigned_families = {
        "truevq_d4" if row["tier"].startswith("d4-") else
        "truevq_d8" if row["tier"].startswith("d8-") else
        "qtip2"
        for row in result["assignment"]
    }
    assert {pack_manifest["tier_codes"][family] for family in assigned_families} == {
        int(value) for value in np.unique(tier_map)
    }
    assert not any(name.endswith("_weights") for name in pack_manifest["tensor_index"])
    assert sum(
        row["data_bytes"] for row in pack_manifest["tensor_index"].values()
    ) == result["byte_accounting"]["whole_model_bytes"]

    status = status_backpack(run_root)
    assert status["status"] == "PASS"
    assert status["first_incomplete_stage"] is None
    assert status["completed_stages"] == 8

    resumed = build_backpack(plan, run_root=run_root)
    assert resumed["resumed_stages"] == result["stages"]
    assert resumed["final_receipt_sha256"] == result["final_receipt_sha256"]


def test_deleted_final_pack_invalidates_repair_and_final_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    final_pack = tmp_path / "final-pack"
    shutil.rmtree(final_pack)

    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "repair"

    rebuilt = build_backpack(plan, run_root=run_root)
    assert verify_pack(final_pack)["status"] == "PASS"
    assert "repair" not in rebuilt["resumed_stages"]


def test_tampered_final_pack_accounting_invalidates_repair_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    manifest_path = tmp_path / "final-pack" / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["backpack_byte_accounting"]["whole_model_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    status = status_backpack(run_root)

    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "repair"


def test_hash_adjusted_final_pack_invalidates_repair_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    result = build_backpack(plan, run_root=run_root)
    pack = Path(result["final_pack"])
    manifest_path = pack / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    name = "layers.0.truevq_d4.codebooks"
    tensor = manifest["tensor_index"][name]
    codebooks_path = pack / tensor["path"]
    codebooks = np.load(codebooks_path, allow_pickle=False)
    codebooks.reshape(-1)[0] += np.float16(1.0)
    np.save(codebooks_path, codebooks, allow_pickle=False)
    file_row = next(row for row in manifest["files"] if row["path"] == tensor["path"])
    file_row["bytes"] = codebooks_path.stat().st_size
    file_row["sha256"] = hashlib.sha256(codebooks_path.read_bytes()).hexdigest()
    tensor["data_sha256"] = hashlib.sha256(codebooks.tobytes(order="C")).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assert verify_pack(pack)["status"] == "PASS"

    status = status_backpack(run_root)

    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "repair"


def test_deleted_final_receipt_invalidates_final_stage_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    (run_root / "FINAL_RECEIPT.json").unlink()

    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "final_score"

    rebuilt = build_backpack(plan, run_root=run_root)
    assert Path(rebuilt["final_receipt"]).is_file()
    assert "final_score" not in rebuilt["resumed_stages"]


def test_tampered_candidate_artifact_invalidates_candidate_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    decoded = run_root / "candidates" / "d4-k4" / "cell0" / "decoded.npy"
    decoded.write_bytes(decoded.read_bytes() + b"tamper")

    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "candidates"


def test_tampered_candidate_anchor_invalidates_anchor_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    anchor_path = run_root / "anchors" / "d4-k4.json"
    anchor = json.loads(anchor_path.read_text())
    anchor["overall"]["kld"] += 1.0
    anchor_path.write_text(json.dumps(anchor, indent=2, sort_keys=True) + "\n")

    status = status_backpack(run_root)

    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "candidate_anchor"


def test_tampered_pre_repair_anchor_invalidates_pre_repair_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    anchor_path = run_root / "anchors" / "pre-repair-backpack.json"
    anchor = json.loads(anchor_path.read_text())
    anchor["overall"]["kld"] += 1.0
    anchor_path.write_text(json.dumps(anchor, indent=2, sort_keys=True) + "\n")

    status = status_backpack(run_root)

    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "pre_repair_anchor"


def test_tampered_prediction_rows_invalidate_prediction_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    pred_path = run_root / "pred" / "rows.json"
    pred = json.loads(pred_path.read_text())
    pred["rows"][0]["physical_bytes"] += 1
    pred_path.write_text(json.dumps(pred, indent=2, sort_keys=True) + "\n")

    status = status_backpack(run_root)

    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "pred"


def test_relabelled_candidate_receipt_invalidates_candidate_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    receipt_path = run_root / "candidates" / "qtip-2.0" / "cell0" / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["tier"] = "d4-k4"
    receipt["algorithm"] = "nearest-vector-codeword"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "candidates"


def test_fixture_qtip_payload_cannot_be_relabelled_as_packaged_qtip(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    parsed = BackpackPlan.from_mapping(plan)
    run_root = tmp_path / "run"
    build_backpack(parsed, run_root=run_root)
    _manifest, cells = backpack_module._load_cells(parsed)
    tier = dict(parsed.tiers[2])
    tier["backend"] = "packaged_qtip"
    tier["source_root"] = str(tmp_path / "packaged-source")
    receipt_path = run_root / "candidates" / "qtip-2.0" / "cell0" / "RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["algorithm"] = "qtip-packaged-v1"
    receipt["backend"] = "packaged_qtip"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    assert not backpack_module._validate_candidate_receipt(
        str(receipt_path),
        tier=tier,
        cell=cells[0],
        geometry_by_identity=backpack_module._exact_qtip_geometries(tier, cells),
    )


def test_pack_verifier_rejects_hash_adjusted_nonmonotonic_record_offsets(
    tmp_path: Path,
) -> None:
    result = build_backpack(_fixture_plan(tmp_path), run_root=tmp_path / "run")
    pack = Path(result["final_pack"])
    manifest_path = pack / "BANANA_PACK_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    name = "layers.0.qtip2.tensor_offsets"
    tensor = manifest["tensor_index"][name]
    offsets_path = pack / tensor["path"]
    offsets = np.load(offsets_path, allow_pickle=False)
    offsets[1, 0] = offsets[2, 0] + 1
    np.save(offsets_path, offsets, allow_pickle=False)
    file_row = next(row for row in manifest["files"] if row["path"] == tensor["path"])
    file_row["bytes"] = offsets_path.stat().st_size
    file_row["sha256"] = hashlib.sha256(offsets_path.read_bytes()).hexdigest()
    tensor["data_sha256"] = hashlib.sha256(offsets.tobytes(order="C")).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(PackValidationError, match="tensor_offsets must be monotonic"):
        verify_pack(pack)


def test_mutated_custom_teacher_invalidates_inspect_resume(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    teacher = tmp_path / "teacher.npy"
    np.save(teacher, np.zeros(8192, dtype=np.float32), allow_pickle=False)
    plan["anchor"]["teacher"] = str(teacher)  # type: ignore[index]
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    np.save(teacher, np.ones(8192, dtype=np.float32), allow_pickle=False)

    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "inspect"


def test_inspect_emits_required_byte_bindings_and_rejects_missing_bytes(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    inspected = inspect_backpack(plan, run_root=run_root)

    assert isinstance(inspected["model_manifest"]["bytes"], int)
    assert isinstance(inspected["anchor_bank"]["bytes"], int)
    assert all(isinstance(row["bytes"], int) for row in inspected["cell_artifacts"])

    stage_path = run_root / "stages" / "01-inspect.json"
    stage = json.loads(stage_path.read_text())
    stage["result"]["model_manifest"].pop("bytes")
    stage_path.write_text(json.dumps(stage, indent=2, sort_keys=True) + "\n")

    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "inspect"


def test_fixed_model_artifacts_are_hash_bound_and_materialized(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=53349)
    model = plan["model"]
    assert isinstance(model, dict)
    model_root = Path(model["root"])
    fixed = model_root / "dense.bin"
    fixed.write_bytes(b"dense")
    manifest_path = model_root / "BACKPACK_MODEL.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dense_bytes"] = fixed.stat().st_size
    manifest["fixed_artifacts"] = [
        {
            "role": "dense",
            "path": fixed.name,
            "bytes": fixed.stat().st_size,
            "sha256": hashlib.sha256(fixed.read_bytes()).hexdigest(),
        }
    ]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    result = build_backpack(plan, run_root=tmp_path / "run")
    pack_manifest = load_manifest(result["final_pack"])
    accounting = pack_manifest["backpack_byte_accounting"]

    assert accounting["fixed_bytes"] == 5
    assert accounting["whole_model_bytes"] == 53349
    fixed_rows = [
        row
        for row in pack_manifest["files"]
        if row["role"] == "backpack_fixed_dense"
    ]
    assert len(fixed_rows) == 1
    assert (Path(result["final_pack"]) / fixed_rows[0]["path"]).read_bytes() == b"dense"


def test_existing_unrelated_output_pack_is_not_deleted(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    output = tmp_path / "final-pack"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not delete\n")

    with pytest.raises(Exception):
        build_backpack(plan, run_root=tmp_path / "run")

    assert sentinel.read_text() == "do not delete\n"


def test_failure_receipt_stops_at_solve_boundary(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=590)
    run_root = tmp_path / "run"

    with pytest.raises(ValueError, match="envelope"):
        build_backpack(plan, run_root=run_root)

    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "solve_materialize"
    assert status["failed_stage"] == "solve_materialize"
    assert not (run_root / "stages" / "06-pre-repair-anchor.json").exists()


def test_public_stage_apis_compose_without_private_orchestrator(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "composed-run"
    for stage_api in (
        inspect_backpack,
        generate_backpack_candidates,
        anchor_backpack_candidates,
        predict_backpack,
        solve_backpack,
        anchor_backpack,
        repair_backpack,
        score_backpack,
    ):
        assert stage_api(plan, run_root=run_root)["status"] == "PASS"
    assert status_backpack(run_root)["status"] == "PASS"


def test_orchestrator_calls_public_stage_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _fixture_plan(tmp_path)
    observed: list[str] = []
    original = backpack_module.inspect_backpack

    def observed_inspect(plan: object, *, run_root: str | Path) -> dict[str, object]:
        observed.append("inspect_backpack")
        return original(plan, run_root=run_root)  # type: ignore[arg-type]

    monkeypatch.setattr(backpack_module, "inspect_backpack", observed_inspect)
    result = build_backpack(plan, run_root=tmp_path / "instrumented-run")
    assert result["status"] == "PASS"
    assert observed == ["inspect_backpack"]


def test_public_candidate_and_materializer_apis_are_importable_and_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import banana_smasher

    assert callable(generate_qtip_backpack_candidate)
    assert callable(generate_vector_vq_backpack_candidate)
    assert callable(materialize_backpack_source)
    assert callable(banana_smasher.generate_qtip_backpack_candidate)
    assert callable(banana_smasher.generate_vector_vq_backpack_candidate)
    assert callable(banana_smasher.materialize_backpack_source)
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "public-producers"
    inspect_backpack(plan, run_root=run_root)

    calls: list[str] = []
    original_vq = backpack_module.generate_vector_vq_backpack_candidate
    original_qtip = backpack_module.generate_qtip_backpack_candidate
    original_materialize = backpack_module.materialize_backpack_source

    def observed_vq(*args, **kwargs):
        calls.append("vq")
        return original_vq(*args, **kwargs)

    def observed_qtip(*args, **kwargs):
        calls.append("qtip")
        return original_qtip(*args, **kwargs)

    def observed_materialize(*args, **kwargs):
        calls.append("materialize")
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(backpack_module, "generate_vector_vq_backpack_candidate", observed_vq)
    monkeypatch.setattr(backpack_module, "generate_qtip_backpack_candidate", observed_qtip)
    monkeypatch.setattr(backpack_module, "materialize_backpack_source", observed_materialize)

    candidates = generate_backpack_candidates(plan, run_root=run_root)
    assert candidates["status"] == "PASS"
    assert {"vq", "qtip"} <= set(calls)

    anchor_backpack_candidates(plan, run_root=run_root)
    predict_backpack(plan, run_root=run_root)
    solve = solve_backpack(plan, run_root=run_root)
    assert solve["status"] == "PASS"
    assert "materialize" in calls


def test_public_family_registry_exposes_builtin_generate_materialize_price_predict_bindings(
) -> None:
    bindings = list_backpack_family_bindings()

    assert bindings
    assert all(isinstance(binding, BackpackFamilyBinding) for binding in bindings)
    assert {
        binding.provider
        for binding in bindings
    } >= {
        "native_mxfp4",
        "qtip@2.00",
        "qtip@2.50",
        "qtip@3.00",
        "d4_k2048",
        "d4_k4096",
    }
    assert all(callable(binding.generate) for binding in bindings)
    assert all(callable(binding.materialize) for binding in bindings)
    assert all(callable(binding.price) for binding in bindings)
    assert all(callable(binding.predict) for binding in bindings)

    qtip = resolve_backpack_family(
        {
            "id": "qtip-2.5",
            "family": "qtip",
            "provider": "qtip@2.50",
            "bpw": 2.5,
            "backend": "fixture_reference",
        }
    )
    native = resolve_backpack_family(
        {"id": "native-mxfp4", "family": "native_mxfp4"}
    )

    assert qtip.provider == "qtip@2.50"
    assert native.provider == "native_mxfp4"


def test_uniform_selection_receipt_backed_price_matches_materialized_payload_bytes(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=100000)
    parsed = BackpackPlan.from_mapping(plan)
    run_root = tmp_path / "run"
    inspected = inspect_backpack(parsed, run_root=run_root)
    generated = generate_backpack_candidates(parsed, run_root=run_root)
    _manifest, cells = backpack_module._load_cells(parsed)
    assignment = [
        {"cell_id": str(cell["cell_id"]), "tier": "d4-k4"}
        for cell in cells
    ]
    priced = price_backpack_selection(
        parsed,
        assignment=assignment,
        candidates=generated,
    )

    artifact_roots = {
        str(cell["cell_id"]): backpack_module.candidate_artifact_root(
            generated,
            tier="d4-k4",
            cell_id=str(cell["cell_id"]),
        )
        for cell in cells
    }
    source = tmp_path / "uniform-source"
    materialize_backpack_source(
        source,
        plan=parsed,
        cells=cells,
        assignment=assignment,
        artifact_roots=artifact_roots,
    )
    pack = tmp_path / "uniform-pack"
    export_pack(
        source_root=source,
        output=pack,
        model_id="uniform-fixture",
        instance_id="uniform-fixture-1",
        link_mode="copy",
    )
    manifest = load_manifest(pack)
    tensor_bytes = sum(
        int(row["data_bytes"]) for row in manifest["tensor_index"].values()
    )

    assert priced["provider_counts"] == {"vector_vq:d4:b2": 4}
    assert priced["materialized_payload_bytes"] == tensor_bytes - int(
        inspected["fixed_bytes"]["routing_bytes"]
    )


def test_qtip_1_5_provider_declaration_builds_and_appears_in_family_counts(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=100000)
    plan["tiers"] = [
        {
            "id": "qtip-1.5",
            "family": "qtip",
            "provider": "qtip@1.50",
            "bpw": 1.5,
            "backend": "fixture_reference",
        }
    ]
    parsed = BackpackPlan.from_mapping(plan)
    inspect_root = tmp_path / "inspect-run"
    inspect_backpack(parsed, run_root=inspect_root)
    generated = generate_backpack_candidates(parsed, run_root=inspect_root)
    _manifest, cells = backpack_module._load_cells(parsed)
    assignment = [
        {"cell_id": str(cell["cell_id"]), "tier": "qtip-1.5"}
        for cell in cells
    ]
    priced = price_backpack_selection(
        parsed,
        assignment=assignment,
        candidates=generated,
    )
    plan["target"]["exact_bytes"] = (
        int(json.loads((inspect_root / "stages" / "01-inspect.json").read_text())["result"]["fixed_total_bytes"])
        + int(priced["materialized_payload_bytes"])
    )

    result = build_backpack(plan, run_root=tmp_path / "build-run")

    assert result["status"] == "PASS"
    assert result["family_counts"]["qtip@1.50"] == 4
    assert result["candidate_tiers"][0]["provider"] == "qtip@1.50"
    assert verify_pack(Path(result["final_pack"]))["status"] == "PASS"


def test_stage_reuse_resumes_candidate_and_anchor_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_plan = _fixture_plan(tmp_path)
    baseline_run = tmp_path / "baseline-run"
    baseline = build_backpack(baseline_plan, run_root=baseline_run)
    assert baseline["status"] == "PASS"

    candidate_stage = baseline_run / "stages" / "02-candidates.json"
    anchor_stage = baseline_run / "stages" / "03-candidate-anchor.json"
    reused_plan = json.loads(json.dumps(baseline_plan))
    reused_plan["reuse_receipts"] = [
        {
            "role": "candidate_stage",
            "path": str(candidate_stage),
            "sha256": hashlib.sha256(candidate_stage.read_bytes()).hexdigest(),
            "schema": "banana-smasher-backpack-stage-receipt-v1",
            "stage": "candidates",
        },
        {
            "role": "candidate_anchor_stage",
            "path": str(anchor_stage),
            "sha256": hashlib.sha256(anchor_stage.read_bytes()).hexdigest(),
            "schema": "banana-smasher-backpack-stage-receipt-v1",
            "stage": "candidate_anchor",
        },
    ]

    monkeypatch.setitem(
        backpack_module._STAGE_RUNNERS,
        "candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("candidate replayed")),
    )
    monkeypatch.setitem(
        backpack_module._STAGE_RUNNERS,
        "candidate_anchor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("anchor replayed")),
    )

    reused_run = tmp_path / "reused-run"
    result = build_backpack(reused_plan, run_root=reused_run)

    assert result["status"] == "PASS"
    assert {"candidates", "candidate_anchor"} <= set(result["resumed_stages"])
    assert status_backpack(reused_run)["status"] == "PASS"


def test_repair_bundle_reaches_export_pack_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _fixture_plan(tmp_path)
    plan["repair"] = {
        "method": "repair_bundle",
        "checkpoint": str(tmp_path / "UPDATE.pt"),
        "checkpoint_sha256": "1" * 64,
        "active_overlay": str(tmp_path / "ACTIVE.json"),
        "active_overlay_sha256": "2" * 64,
        "assignment": str(tmp_path / "ASSIGNMENT.json"),
        "assignment_sha256": "3" * 64,
        "update": 12,
    }
    observed: list[dict[str, object]] = []
    bundle = RepairBundle(
        checkpoint_path=Path("/sealed/UPDATE.pt"),
        checkpoint_sha256="1" * 64,
        active_overlay_path=Path("/sealed/ACTIVE.json"),
        active_overlay_sha256="2" * 64,
        assignment_path=Path("/sealed/ASSIGNMENT.json"),
        assignment_sha256="3" * 64,
        checkpoint_format="bs-basic-repair-v1",
        mechanism="physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains",
        update=12,
        codebooks={},
        dense_tensors={
            "norms/model.norm": np.arange(4, dtype=np.float32),
            "outputs/model.layers.0.self_attn.o_b_proj.output_log_gain": np.asarray(
                0.125, dtype=np.float32
            ),
        },
        norm_count=1,
        output_count=1,
    )
    sizing_root = tmp_path / "repair-sizing"
    write_repair_payload(sizing_root, bundle, [])
    repair_state_bytes = (sizing_root / REPAIR_STATE_PATH).stat().st_size
    model_manifest_path = Path(str(plan["model"]["root"])) / "BACKPACK_MODEL.json"  # type: ignore[index]
    model_manifest = json.loads(model_manifest_path.read_text())
    model_manifest["repair_bytes"] = repair_state_bytes
    model_manifest_path.write_text(json.dumps(model_manifest, sort_keys=True) + "\n")
    plan["target"] = {"exact_bytes": 53344 + repair_state_bytes}

    def fake_load_repair_bundle(**kwargs):
        observed.append(kwargs)
        return bundle

    monkeypatch.setattr(backpack_module, "load_repair_bundle", fake_load_repair_bundle)
    result = build_backpack(plan, run_root=tmp_path / "repair-bundle-run")
    manifest = load_manifest(Path(result["final_pack"]))

    assert observed and observed[0]["checkpoint_sha256"] == "1" * 64
    assert manifest["repair"]["checkpoint_sha256"] == bundle.checkpoint_sha256
    assert manifest["repair"]["active_overlay_sha256"] == bundle.active_overlay_sha256
    assert manifest["repair"]["assignment_sha256"] == bundle.assignment_sha256
    assert manifest["backpack_byte_accounting"]["repair_state_bytes"] == repair_state_bytes
    assert manifest["backpack_byte_accounting"]["whole_model_bytes"] == plan["target"]["exact_bytes"]
    assert verify_pack(Path(result["final_pack"]))["repair"]["norms"] == 1


def test_final_score_uses_repair_bundle_materialized_codebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _fixture_plan(tmp_path)
    plan["repair"] = {
        "method": "repair_bundle",
        "checkpoint": str(tmp_path / "UPDATE.pt"),
        "checkpoint_sha256": "1" * 64,
        "active_overlay": str(tmp_path / "ACTIVE.json"),
        "active_overlay_sha256": "2" * 64,
        "assignment": str(tmp_path / "ASSIGNMENT.json"),
        "assignment_sha256": "3" * 64,
        "update": 12,
    }
    model_root = Path(str(plan["model"]["root"]))  # type: ignore[index]
    original = np.load(model_root / "cell2.npy", allow_pickle=False)
    first_expert = original.reshape(128, -1)[0]
    source_codebook = np.asarray(
        quantize_vector_cell(first_expert, dimension=4, bits=2)["codebook"]
    )
    source_sha = hashlib.sha256(
        np.ascontiguousarray(source_codebook).tobytes(order="C")
    ).hexdigest()
    replacement = np.zeros_like(source_codebook, dtype=np.float16)
    sizing_bundle = RepairBundle(
        checkpoint_path=Path("/sealed/UPDATE.pt"),
        checkpoint_sha256="1" * 64,
        active_overlay_path=Path("/sealed/ACTIVE.json"),
        active_overlay_sha256="2" * 64,
        assignment_path=Path("/sealed/ASSIGNMENT.json"),
        assignment_sha256="3" * 64,
        checkpoint_format="bs-basic-repair-v1",
        mechanism="physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains",
        update=12,
        codebooks={},
        dense_tensors={},
        norm_count=0,
        output_count=0,
    )
    sizing_root = tmp_path / "repair-sizing"
    write_repair_payload(sizing_root, sizing_bundle, [])
    repair_state_bytes = (sizing_root / REPAIR_STATE_PATH).stat().st_size
    model_manifest_path = model_root / "BACKPACK_MODEL.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    model_manifest["repair_bytes"] = repair_state_bytes
    model_manifest_path.write_text(json.dumps(model_manifest, sort_keys=True) + "\n")
    plan["target"] = {"exact_bytes": 53344 + repair_state_bytes}
    bundle = RepairBundle(
        **{
            **sizing_bundle.__dict__,
            "codebooks": {
                source_sha: CodebookRepair(
                    checkpoint_key=f"L0/d4_k4__2_{source_sha}",
                    source_wire_sha256=source_sha,
                    array=replacement,
                )
            },
        }
    )
    monkeypatch.setattr(backpack_module, "load_repair_bundle", lambda **_kwargs: bundle)

    run_root = tmp_path / "repair-bundle-run"
    result = build_backpack(plan, run_root=run_root)

    assert next(row for row in result["assignment"] if row["cell_id"] == "cell2")[
        "tier"
    ] == "d4-k4"
    manifest = load_manifest(Path(result["final_pack"]))
    codebook_row = manifest["tensor_index"]["layers.0.truevq_d4.codebooks"]
    materialized = np.load(
        Path(result["final_pack"]) / codebook_row["path"], allow_pickle=False
    )
    assert materialized.shape == (256, 4, 4)
    assert np.array_equal(materialized[0], np.zeros((4, 4), dtype=np.float16))

    pre_export_cells = [
        np.load(run_root / "repair" / "cells" / f"cell{index}.npy", allow_pickle=False)
        for index in range(4)
    ]
    artifact_cells = list(pre_export_cells)
    artifact_cells[2] = artifact_cells[2].copy()
    artifact_cells[2][: first_expert.size] = 0
    with np.load(plan["anchor"]["bank"], allow_pickle=False) as bank:  # type: ignore[index]
        features = np.asarray(bank["features"], dtype=np.float32)
        classes = np.asarray(bank["classes"]).astype(str)
    teacher = np.concatenate(
        [
            np.load(model_root / f"cell{index}.npy", allow_pickle=False).reshape(-1)
            for index in range(4)
        ]
    )
    expected = backpack_module._anchor_metrics(
        features,
        classes,
        teacher,
        np.concatenate(artifact_cells).astype(np.float32),
    )
    stale = backpack_module._anchor_metrics(
        features,
        classes,
        teacher,
        np.concatenate(pre_export_cells).astype(np.float32),
    )
    assert result["final_anchor"] != stale
    assert result["final_anchor"]["overall"]["top1"] == expected["overall"]["top1"]
    assert result["final_anchor"]["overall"]["kld"] == pytest.approx(
        expected["overall"]["kld"], abs=1e-3
    )


def test_packaged_qtip_final_scoring_requires_exported_payload_identity(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    plan["repair"] = {"method": "none"}
    run_root = tmp_path / "run"
    result = build_backpack(plan, run_root=run_root)
    parsed = BackpackPlan.from_mapping(plan)
    packaged = replace(
        parsed,
        tiers=tuple(
            {
                **tier,
                "backend": "packaged_qtip",
                "source_root": str(tmp_path / "packaged-qtip"),
            }
            if tier["family"] == "qtip"
            else tier
            for tier in parsed.tiers
        ),
    )
    _manifest, cells = backpack_module._load_cells(packaged)
    candidates = {"candidate_tiers": result["candidate_tiers"]}

    weights = backpack_module._final_pack_weights(
        packaged,
        Path(result["final_pack"]),
        result["assignment"],
        cells,
        candidates,
    )
    assert weights.shape == (8192,)

    qtip_row = next(row for row in result["assignment"] if row["tier"] == "qtip-2.0")
    candidate_root = backpack_module.candidate_artifact_root(
        candidates,
        tier="qtip-2.0",
        cell_id=qtip_row["cell_id"],
    )
    codebooks_path = candidate_root / "codebooks.npy"
    codebooks = np.load(codebooks_path, allow_pickle=False)
    codebooks.reshape(-1)[0] += np.float16(1.0)
    np.save(codebooks_path, codebooks, allow_pickle=False)

    with pytest.raises(BackpackPlanError, match="does not match exported record"):
        backpack_module._final_pack_weights(
            packaged,
            Path(result["final_pack"]),
            result["assignment"],
            cells,
            candidates,
        )


def test_receipt_reuse_binds_evidence_without_replay(tmp_path: Path) -> None:
    admitted = tmp_path / "qtip-anchor.json"
    admitted.write_text(
        json.dumps({"schema": "qtip-anchor-v1", "status": "PASS"}) + "\n"
    )
    quarantined = tmp_path / "d4-anchor.json"
    quarantined.write_text(
        json.dumps({"schema": "d4-anchor-v1", "status": "QUARANTINED"}) + "\n"
    )

    result = reuse_backpack_receipts(
        [
            {
                "role": "qtip2_anchor",
                "path": str(admitted),
                "sha256": hashlib.sha256(admitted.read_bytes()).hexdigest(),
                "admission": "admitted",
            },
            {
                "role": "d4_anchor_diagnostic",
                "path": str(quarantined),
                "sha256": hashlib.sha256(quarantined.read_bytes()).hexdigest(),
                "admission": "evidence_only",
            },
        ],
        output=tmp_path / "reuse.json",
    )
    assert result["execution"] == {
        "transformer_replay": False,
        "candidate_generation": False,
        "anchor_replay": False,
    }
    assert result["admitted"] == 1
    assert result["evidence_only"] == 1


def test_plan_integrates_sealed_receipt_reuse_into_build_dag(tmp_path: Path) -> None:
    plan = _fixture_plan(tmp_path)
    completed = tmp_path / "completed-qtip-anchor.json"
    completed.write_text(
        json.dumps({"schema": "qtip-anchor-v1", "status": "PASS"}) + "\n"
    )
    plan["reuse_receipts"] = [
        {
            "role": "qtip_anchor",
            "path": str(completed),
            "sha256": hashlib.sha256(completed.read_bytes()).hexdigest(),
        }
    ]
    run_root = tmp_path / "run"

    result = build_backpack(plan, run_root=run_root)
    inspect_receipt = json.loads(
        (run_root / "stages" / "01-inspect.json").read_text()
    )["result"]

    assert result["status"] == "PASS"
    assert inspect_receipt["receipt_reuse"]["execution"] == {
        "transformer_replay": False,
        "candidate_generation": False,
        "anchor_replay": False,
    }
    assert inspect_receipt["receipt_reuse"]["receipts"][0]["role"] == "qtip_anchor"

    completed.write_text(
        json.dumps({"schema": "qtip-anchor-v1", "status": "FAIL"}) + "\n"
    )
    status = status_backpack(run_root)
    assert status["status"] == "INCOMPLETE"
    assert status["first_incomplete_stage"] == "inspect"


def test_receipt_reuse_rejects_direct_symlink(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"schema": "anchor-v1", "status": "PASS"}) + "\n")
    link = tmp_path / "receipt-link.json"
    link.symlink_to(receipt)

    with pytest.raises(BackpackPlanError, match="regular file"):
        reuse_backpack_receipts(
            [
                {
                    "role": "anchor",
                    "path": str(link),
                    "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                }
            ],
            output=tmp_path / "reuse.json",
        )


def test_receipt_reuse_rejects_direct_output_symlink(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"schema": "anchor-v1", "status": "PASS"}) + "\n")
    outside = tmp_path / "outside.json"
    outside.write_text("ORIGINAL\n")
    output = tmp_path / "reuse.json"
    output.symlink_to(outside)

    with pytest.raises(BackpackPlanError, match="output must not be a direct symlink"):
        reuse_backpack_receipts(
            [
                {
                    "role": "anchor",
                    "path": str(receipt),
                    "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                }
            ],
            output=output,
        )

    assert outside.read_text() == "ORIGINAL\n"


def test_backpack_cli_build_and_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan = _fixture_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan) + "\n")
    parser = _parser()
    parsed = parser.parse_args(
        ["backpack", "build", "--plan", str(plan_path), "--run-root", str(tmp_path / "run")]
    )
    assert parsed.command == "backpack"
    assert parsed.backpack_command == "build"

    assert main(
        ["backpack", "build", "--plan", str(plan_path), "--run-root", str(tmp_path / "run")]
    ) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["status"] == "PASS"

    assert main(["backpack", "status", "--run-root", str(tmp_path / "run")]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["completed_stages"] == 8


def test_lifecycle_exports_use_one_pack_abi_and_preserve_pre_post_geometry(
    tmp_path: Path,
) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    serving_model = _serving_model(tmp_path)
    kernel_cache = _kernel_cache(tmp_path)

    exports = {
        stage: export_backpack_lifecycle(
            run_root,
            lifecycle=stage,
            output=tmp_path / f"{stage}-model",
            serving_model_root=serving_model,
            kernel_cache_root=kernel_cache,
            tier="d4-k4" if stage == "uniform-anchor" else None,
        )
        for stage in ("uniform-anchor", "pre-repair", "post-repair")
    }

    manifests = {
        stage: load_manifest(result["model"])
        for stage, result in exports.items()
    }
    assert {manifest["quant_method"] for manifest in manifests.values()} == {
        "banana_smasher"
    }
    assert {
        json.loads((Path(result["model"]) / "config.json").read_text())[
            "quantization_config"
        ]["quant_method"]
        for result in exports.values()
    } == {"banana_smasher"}
    assert exports["pre-repair"]["assignment_sha256"] == exports["post-repair"][
        "assignment_sha256"
    ]
    assert exports["pre-repair"]["expert_wire_layout_sha256"] == exports[
        "post-repair"
    ]["expert_wire_layout_sha256"]
    assert exports["pre-repair"]["whole_model_shape_sha256"] == exports[
        "post-repair"
    ]["whole_model_shape_sha256"]
    assert exports["uniform-anchor"]["assignment"] == {
        row["cell_id"]: "d4-k4"
        for row in exports["uniform-anchor"]["assignment_rows"]
    }
    assert exports["uniform-anchor"]["expert_plane_bytes"] != exports["pre-repair"][
        "expert_plane_bytes"
    ]
    assert exports["post-repair"]["repair"]["dense_application"][
        "norms_materialized"
    ] == 1
    assert exports["post-repair"]["repair"]["dense_application"][
        "outputs_folded"
    ] == 1
    codebook_name = "layers.0.truevq_d4.codebooks"
    pre_codebook = np.load(
        Path(exports["pre-repair"]["model"])
        / manifests["pre-repair"]["tensor_index"][codebook_name]["path"],
        allow_pickle=False,
    )
    post_codebook = np.load(
        Path(exports["post-repair"]["model"])
        / manifests["post-repair"]["tensor_index"][codebook_name]["path"],
        allow_pickle=False,
    )
    assert pre_codebook.shape == post_codebook.shape
    assert not np.array_equal(pre_codebook, post_codebook)
    from safetensors.numpy import load_file

    shard = "model-00001-of-00001.safetensors"
    pre_dense = load_file(Path(exports["pre-repair"]["model"]) / shard)
    post_dense = load_file(Path(exports["post-repair"]["model"]) / shard)
    assert np.allclose(post_dense["model.norm.weight"], pre_dense["model.norm.weight"] * 1.5)
    assert np.allclose(
        post_dense["model.layers.0.self_attn.o_b_proj.weight"],
        pre_dense["model.layers.0.self_attn.o_b_proj.weight"] * 1.5,
    )
    post_quant = json.loads(
        (Path(exports["post-repair"]["model"]) / "config.json").read_text()
    )["quantization_config"]
    assert post_quant["repair_application"] == "export-folded-v1"
    assert post_quant["runtime_output_gain"] is False
    for result in exports.values():
        model = Path(result["model"])
        assert verify_pack(model)["status"] == "PASS"
        loader = PackLoader(
            model,
            kernel_cache_root=model / "kernel-cache",
            architecture="sm_120",
        )
        assert loader.layers == [0]
        assert loader.runtime_adapter_class().API_VERSION == 1
        assert (model / "model.safetensors.index.json").is_file()
        assert result["metadata_bytes"] > 0

    moved = tmp_path / "moved-post-repair-model"
    shutil.copytree(exports["post-repair"]["model"], moved)
    assert PackLoader(
        moved,
        kernel_cache_root=moved / "kernel-cache",
        architecture="sm_120",
    ).serve_receipt["status"] == "PASS"


def test_backpack_cli_exports_lifecycle_from_run_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _fixture_plan(tmp_path)
    run_root = tmp_path / "run"
    build_backpack(plan, run_root=run_root)
    serving_model = _serving_model(tmp_path)
    kernel_cache = _kernel_cache(tmp_path)
    output = tmp_path / "pre-model"

    assert main(
        [
            "backpack",
            "export",
            "--run-root",
            str(run_root),
            "--lifecycle",
            "pre-repair",
            "--output",
            str(output),
            "--serving-model-root",
            str(serving_model),
            "--kernel-cache-root",
            str(kernel_cache),
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["lifecycle"] == "pre-repair"
    assert receipt["model"] == str(output.resolve())
    assert verify_pack(output)["status"] == "PASS"
    assert PackLoader(
        output,
        kernel_cache_root=output / "kernel-cache",
        architecture="sm_120",
    ).serve_receipt["status"] == "PASS"


def test_public_backpack_family_proof_script_runs_from_source(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "tests" / "backpack_family_api_proof.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--workdir",
            str(tmp_path / "proof"),
        ],
        cwd=root.parent,
        env={
            **os.environ,
            "PYTHONPATH": str(root / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "PASS"
