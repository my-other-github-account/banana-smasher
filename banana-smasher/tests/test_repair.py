from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.cli import _parser
from banana_smasher.contract import (
    PackValidationError,
    export_pack,
    verify_pack,
)
from banana_smasher.repair import (
    CodebookRepair,
    REPAIR_FORMAT,
    RepairBundle,
    _normalize_checkpoint_format,
    materialize_codebook_plane,
    validate_repair_state,
)
from test_contract import _write_qtip2_source


def _wire_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def test_sealed_pre_rename_repair_format_is_normalized_fail_closed() -> None:
    sealed_format = bytes.fromhex(
        "67656e657369732d62617369632d7265706169722d7631"
    ).decode("utf-8")
    assert _normalize_checkpoint_format(sealed_format) == REPAIR_FORMAT
    assert _normalize_checkpoint_format(REPAIR_FORMAT) == REPAIR_FORMAT
    with pytest.raises(ValueError, match="not an approved sealed format"):
        _normalize_checkpoint_format("unknown-repair-format")


def _fixture_bundle(old: np.ndarray, replacement: np.ndarray) -> RepairBundle:
    old_sha = _wire_sha(old)
    return RepairBundle(
        checkpoint_path=Path("/sealed/UPDATE_012.pt"),
        checkpoint_sha256="1" * 64,
        active_overlay_path=Path("/sealed/ACTIVE_OVERLAY.json"),
        active_overlay_sha256="2" * 64,
        assignment_path=Path("/sealed/ASSIGNMENT.json"),
        assignment_sha256="3" * 64,
        checkpoint_format="bs-basic-repair-v1",
        mechanism="physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains",
        update=12,
        codebooks={
            old_sha: CodebookRepair(
                checkpoint_key=f"L0/d4_k4__2_{old_sha}",
                source_wire_sha256=old_sha,
                array=np.ascontiguousarray(replacement, dtype=np.float16),
            )
        },
        dense_tensors={
            "norms/model.norm": np.arange(4, dtype=np.float32),
            "outputs/model.layers.0.self_attn.o_b_proj.output_log_gain": np.asarray(
                0.125, dtype=np.float32
            ),
        },
        norm_count=1,
        output_count=1,
    )


def _metadata_only_serving(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "scale_fmt": "float32",
                    "weight_block_size": [128, 128],
                },
            }
        )
        + "\n"
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (root / name).write_text("{}\n")
    return root


def test_repair_export_requires_serving_model_for_dense_state(
    tmp_path: Path,
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    codebook = source / "layers/layer_000/qtip2/codebooks.npy"
    old = np.load(codebook, allow_pickle=False)
    replacement = np.full(old.shape, 7.0, dtype=np.float16)
    bundle = _fixture_bundle(old, replacement)
    pack = tmp_path / "pack"

    with pytest.raises(PackValidationError, match="requires serving_model_root"):
        export_pack(
            source_root=source,
            output=pack,
            model_id="fixture-model",
            instance_id="bs-pack-repair-0001",
            link_mode="hardlink",
            repair=bundle,
        )
    assert not pack.exists()


def test_repair_export_fails_closed_when_checkpoint_codebook_has_no_plane(
    tmp_path: Path,
) -> None:
    source = _write_qtip2_source(tmp_path / "source")
    old = np.arange(16, dtype=np.float16).reshape(4, 4)
    bundle = _fixture_bundle(old + 100, old + 200)

    with pytest.raises(ValueError, match="checkpoint codebooks were not materialized"):
        export_pack(
            source_root=source,
            output=tmp_path / "pack",
            model_id="fixture-model",
            instance_id="bs-pack-repair-missing",
            link_mode="copy",
            repair=bundle,
            serving_model_root=_metadata_only_serving(tmp_path / "serving"),
        )
    assert not (tmp_path / "pack").exists()


def test_materializer_replaces_each_index_in_multi_codebook_plane(
    tmp_path: Path,
) -> None:
    base = np.stack(
        [
            np.arange(8, dtype=np.float16).reshape(4, 2),
            np.arange(8, 16, dtype=np.float16).reshape(4, 2),
        ]
    )
    source = tmp_path / "layer_000.d4_k4.2.codebooks.npy"
    destination = tmp_path / "pack/planes/layer_000.d4_k4.2.codebooks.npy"
    np.save(source, base, allow_pickle=False)
    repairs = {}
    for index in range(2):
        source_sha = _wire_sha(base[index])
        repairs[source_sha] = CodebookRepair(
            checkpoint_key=f"L0/codebook_{index}_{source_sha}",
            source_wire_sha256=source_sha,
            array=np.full((4, 2), index + 20, dtype=np.float16),
        )

    rows = materialize_codebook_plane(source, destination, repairs)

    assert rows is not None
    assert [row["codebook_index"] for row in rows] == [0, 1]
    repaired = np.load(destination, allow_pickle=False)
    assert np.all(repaired[0] == 20)
    assert np.all(repaired[1] == 21)


def test_validate_repair_state_rejects_nonfinite_and_surface_drift() -> None:
    valid = {
        "codebooks": {
            "L0": {
                "cb_" + "a" * 64: np.ones((2, 2), dtype=np.float32),
            }
        },
        "norms": {"model.norm": np.ones(2, dtype=np.float32)},
        "outputs": {
            "model.layers.0.self_attn.o_b_proj.output_log_gain": np.asarray(
                0.0, dtype=np.float32
            )
        },
    }
    result = validate_repair_state(valid, expected_counts=(1, 1, 1))
    assert len(result["codebooks"]) == 1

    valid["outputs"][
        "model.layers.0.self_attn.o_b_proj.output_log_gain"
    ] = np.asarray(np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        validate_repair_state(valid, expected_counts=(1, 1, 1))


def test_smash_export_parser_exposes_bound_repair_inputs() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "export",
            "--source-root",
            "/planes",
            "--output",
            "/pack",
            "--model-id",
            "DeepSeek-V4-Flash-BQ3",
            "--instance-id",
            "u012-v5",
            "--repair-checkpoint",
            "/sealed/UPDATE_012.pt",
            "--repair-checkpoint-sha256",
            "1" * 64,
            "--active-overlay",
            "/sealed/ACTIVE.json",
            "--active-overlay-sha256",
            "2" * 64,
            "--assignment",
            "/sealed/ASSIGNMENT.json",
            "--assignment-sha256",
            "3" * 64,
            "--repair-update",
            "12",
        ]
    )
    assert args.repair_checkpoint == Path("/sealed/UPDATE_012.pt")
    assert args.repair_update == 12


def test_serving_export_materializes_norms_and_folds_output_log_gain(
    tmp_path: Path,
) -> None:
    from safetensors.numpy import load_file, save_file

    source = _write_qtip2_source(tmp_path / "source")
    codebook = np.load(
        source / "layers/layer_000/qtip2/codebooks.npy", allow_pickle=False
    )
    serving = tmp_path / "serving"
    serving.mkdir()
    shard = "model-00001-of-00001.safetensors"
    base = {
        "model.norm.weight": np.ones(2, dtype=np.float16),
        "model.layers.0.self_attn.o_b_proj.weight": np.ones(
            (2, 2), dtype=np.float32
        ),
    }
    save_file(base, serving / shard)
    (serving / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.nbytes for value in base.values())},
                "weight_map": {name: shard for name in base},
            },
            sort_keys=True,
        )
        + "\n"
    )
    (serving / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV3ForCausalLM"],
                "quantization_config": {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "scale_fmt": "float32",
                    "weight_block_size": [128, 128],
                },
            }
        )
        + "\n"
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        (serving / name).write_text("{}\n")
    bundle = _fixture_bundle(codebook, codebook + np.float16(1))
    bundle = RepairBundle(
        **{
            **bundle.__dict__,
            "dense_tensors": {
                "norms/model.norm.weight": np.asarray([2.0, 3.0], dtype=np.float32),
                "outputs/model.layers.0.self_attn.o_b_proj.output_log_gain": np.asarray(
                    np.log(2.0), dtype=np.float32
                ),
            },
        }
    )
    output = tmp_path / "post-repair-model"

    manifest = export_pack(
        source_root=source,
        output=output,
        model_id="fixture-model",
        instance_id="post-repair",
        link_mode="hardlink",
        repair=bundle,
        serving_model_root=serving,
    )

    repaired = load_file(output / shard)
    unchanged = load_file(serving / shard)
    assert np.array_equal(repaired["model.norm.weight"], np.asarray([2.0, 3.0]))
    assert repaired["model.norm.weight"].dtype == np.float32
    assert np.allclose(
        repaired["model.layers.0.self_attn.o_b_proj.weight"], 2.0
    )
    assert np.allclose(
        unchanged["model.layers.0.self_attn.o_b_proj.weight"], 1.0
    )
    assert (output / shard).stat().st_ino != (serving / shard).stat().st_ino
    assert json.loads((output / "model.safetensors.index.json").read_text())[
        "metadata"
    ]["total_size"] == sum(value.nbytes for value in repaired.values())
    assert repaired["model.norm.weight"].shape == unchanged["model.norm.weight"].shape
    assert repaired["model.layers.0.self_attn.o_b_proj.weight"].shape == unchanged[
        "model.layers.0.self_attn.o_b_proj.weight"
    ].shape
    quant = json.loads((output / "config.json").read_text())["quantization_config"]
    assert quant["repair_application"] == "export-folded-v1"
    assert quant["runtime_output_gain"] is False
    assert manifest["repair"]["dense_application"]["outputs_folded"] == 1
    assert manifest["repair"]["dense_application"]["norms_materialized"] == 1

    duplicate = tmp_path / "duplicate-fold"
    with pytest.raises(PackValidationError, match="already has export-folded repair"):
        export_pack(
            source_root=source,
            output=duplicate,
            model_id="repair",
            instance_id="repair-duplicate",
            repair=bundle,
            serving_model_root=output,
            link_mode="copy",
        )
    assert not duplicate.exists()
    assert next(row for row in manifest["links"] if row["path"] == shard)["mode"] == (
        "generated"
    )
    assert verify_pack(output)["status"] == "PASS"
