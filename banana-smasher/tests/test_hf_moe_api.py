from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_public_hf_source_admission_pins_revision_and_index(tmp_path: Path) -> None:
    from banana_smasher import admit_hf_source

    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text(json.dumps({"model_type": "fixture_moe"}) + "\n")
    index = model / "model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "layers.0.experts.0.down_proj.weight": "model-00001-of-00002.safetensors",
                    "embed_tokens.weight": "model-00002-of-00002.safetensors",
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    revision = "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"
    (model / "model-00001-of-00002.safetensors").write_bytes(b"routed")
    (model / "model-00002-of-00002.safetensors").write_bytes(b"native")
    receipt_path = tmp_path / "SOURCE_ADMISSION.json"

    receipt = admit_hf_source(
        model,
        revision=revision,
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "PASS"
    assert receipt["model_root"] == str(model.resolve())
    assert receipt["revision"] == revision
    assert receipt["config_sha256"] == _sha(config)
    assert receipt["model_index_sha256"] == _sha(index)
    assert receipt["tensor_count"] == 2
    assert receipt["shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert json.loads(receipt_path.read_text()) == receipt


def test_generic_hf_moe_plan_serializes_routed_and_native_inventories(
    tmp_path: Path,
) -> None:
    from banana_smasher import plan_hf_moe_uniform

    model = tmp_path / "numeric-experts-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture_numeric_moe",
                "text_config": {
                    "n_routed_experts": 1,
                    "n_shared_experts": 1,
                    "num_hidden_layers": 1,
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    shard = model / "model-00001-of-00001.safetensors"
    tensors = {
        "layers.0.experts.0.down_proj.weight": np.arange(8, dtype=np.float16).reshape(2, 4),
        "layers.0.experts.0.down_proj.weight_scale_inv": np.ones(2, dtype=np.float32),
        "layers.0.shared_experts.down_proj.weight": np.arange(8, dtype=np.float16).reshape(2, 4),
        "layers.1.experts.0.down_proj.weight": np.arange(2, dtype=np.float16),
        "embed_tokens.weight": np.arange(6, dtype=np.float16).reshape(3, 2),
    }
    save_file(tensors, shard)
    index = {
        "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
        "weight_map": {name: shard.name for name in tensors},
    }
    (model / "model.safetensors.index.json").write_text(
        json.dumps(index, sort_keys=True) + "\n"
    )
    receipt_path = tmp_path / "UNIFORM_PLAN.json"

    plan = plan_hf_moe_uniform(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        tier="q2",
        scope="routed_only",
        native_rest=True,
        receipt_path=receipt_path,
    )

    assert plan["status"] == "PASS"
    assert plan["adapter"]["id"] == "hf-numeric-experts-v1"
    assert plan["routed_tensors"] == [
        {
            "dtype": "F16",
            "name": "layers.0.experts.0.down_proj.weight",
            "parameters": 8,
            "shape": [2, 4],
            "shard": shard.name,
            "source_bytes": 16,
        }
    ]
    assert {row["name"] for row in plan["native_tensors"]} == {
        "layers.0.experts.0.down_proj.weight_scale_inv",
        "layers.0.shared_experts.down_proj.weight",
        "layers.1.experts.0.down_proj.weight",
        "embed_tokens.weight",
    }
    assert plan["accounting"]["routed_tensor_count"] == 1
    assert plan["accounting"]["native_tensor_count"] == 4
    assert plan["accounting"]["source_tensor_count"] == 5
    assert plan["geometry"] == {
        "auxiliary_layer_ids": [1],
        "expected_model_layers": 1,
        "model_layer_gaps": [],
        "model_layer_ids": [0],
        "routed_layer_ids": [0],
    }
    assert plan["coverage"] == {"duplicates": [], "gaps": []}
    assert plan["mechanisms"] == {"fallback": 0}
    assert json.loads(receipt_path.read_text()) == plan


def test_public_docs_show_the_general_hf_moe_plan_call() -> None:
    repository = Path(__file__).parents[2]
    readme = (repository / "README.md").read_text(encoding="utf-8")
    worked = (repository / "WORKED_EXAMPLE.md").read_text(encoding="utf-8")

    assert "Python 3.11 or newer" in readme
    assert "Start with `CODEBASE_MAP.md`" not in readme
    assert "WORKED_EXAMPLE.md" in readme
    assert "plan_hf_moe_uniform(" in worked
    assert "preflight_hf_moe_output_fit(" in worked
    assert "estimate_hf_moe_uniform(" in worked
    assert "ResidentRepairAPI.build_uniform(" in worked
    assert "open_hf_moe_uniform(" in worked
    assert 'reopened["artifact_root"]' in worked
    assert 'scope="routed_only"' in worked
    assert "native_rest=True" in worked


def test_public_hf_moe_build_materializes_one_q2_tensor_and_reopens_native_bytes(
    tmp_path: Path,
) -> None:
    from banana_smasher import ResidentRepairAPI, open_hf_moe_uniform

    model = tmp_path / "numeric-experts-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture_numeric_moe",
                "n_routed_experts": 1,
                "num_hidden_layers": 1,
            },
            sort_keys=True,
        )
        + "\n"
    )
    shard = model / "model-00001-of-00001.safetensors"
    tensors = {
        "layers.0.experts.0.down_proj.weight": np.arange(16, dtype=np.float16).reshape(2, 8),
        "layers.0.router.weight": np.arange(8, dtype=np.float16).reshape(2, 4),
    }
    save_file(tensors, shard)
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
                "weight_map": {name: shard.name for name in tensors},
            },
            sort_keys=True,
        )
        + "\n"
    )
    output = tmp_path / "artifact"

    built = ResidentRepairAPI.build_uniform(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        tier="q2",
        scope="routed_only",
        native_rest=True,
        output=output,
    )
    reopened = open_hf_moe_uniform(output)

    assert reopened == built
    assert reopened["artifact_root"] == str(output.resolve())
    assert built["status"] == "PASS"
    assert built["accounting"]["routed_tensor_count"] == 1
    assert built["accounting"]["planned_routed_tensor_count"] == 1
    assert built["accounting"]["native_tensor_count"] == 1
    assert built["accounting"]["planned_native_tensor_count"] == 1
    assert built["reload_verified"] is True
    assert built["accounting"]["routed_parameters"] == 16
    assert built["accounting"]["native_parameters"] == 8
    assert built["routed_tensors"][0]["wire"]["geometry"]["K"] == 2
    assert built["routed_tensors"][0]["wire"]["code_bpw"] == 2.0
    assert built["native_tensors"][0]["representation"] == "exact-source-data-bytes"
    assert built["native_tensors"][0]["source_sha256"] == built["native_tensors"][0]["artifact_sha256"]
    assert built["coverage"] == {"duplicates": [], "gaps": []}
    assert built["mechanisms"] == {
        "fallback": 0,
        "reconstruction": 0,
        "relay": 0,
        "streaming": 0,
    }


def test_public_output_fit_preflight_uses_measured_plan_bytes_and_positive_reserve(
    tmp_path: Path,
) -> None:
    from banana_smasher import preflight_hf_moe_output_fit

    plan = {
        "status": "PASS",
        "intent": {"tier": "q2", "scope": "routed_only", "native_rest": True},
        "routed_tensors": [
            {"name": "layers.0.experts.0.down_proj.weight", "shape": [2, 8]}
        ],
        "accounting": {
            "native_source_bytes": 16,
            "routed_parameters": 16,
        },
    }
    receipt_path = tmp_path / "OUTPUT_FIT.json"

    receipt = preflight_hf_moe_output_fit(
        plan,
        free_bytes=10_000,
        reserve_bytes=128,
        receipt_path=receipt_path,
    )

    assert receipt["status"] == "PASS"
    assert receipt["free_bytes"] == 10_000
    assert receipt["native_payload_bytes"] == 16
    assert receipt["q2_code_bytes"] == 4
    assert receipt["q2_scale_bytes"] == 8
    assert receipt["reserve_bytes"] == 128
    assert receipt["required_bytes"] > 156
    assert json.loads(receipt_path.read_text()) == receipt


def test_public_output_fit_can_admit_local_native_spill(tmp_path: Path) -> None:
    from banana_smasher import preflight_hf_moe_output_fit

    plan = {
        "status": "PASS",
        "intent": {"tier": "q2", "scope": "routed_only", "native_rest": True},
        "routed_tensors": [
            {"name": "layers.0.experts.0.down_proj.weight", "shape": [20, 80]}
        ],
        "accounting": {"native_source_bytes": 20_000, "routed_parameters": 1600},
    }

    receipt = preflight_hf_moe_output_fit(
        plan,
        free_bytes=10_000,
        reserve_bytes=128,
        native_spill_root=tmp_path / "native-spill",
        native_spill_free_bytes=30_000,
        native_spill_reserve_bytes=128,
        receipt_path=tmp_path / "OUTPUT_FIT_SPLIT.json",
    )

    assert receipt["status"] == "PASS"
    assert receipt["storage_mode"] == "split-native-local-v1"
    assert receipt["primary_required_bytes"] <= receipt["free_bytes"]
    assert receipt["native_spill_required_bytes"] <= receipt["native_spill_free_bytes"]
    assert receipt["native_payload_bytes"] == 20_000


def test_public_hf_moe_build_reopens_split_native_storage(tmp_path: Path) -> None:
    from banana_smasher import ResidentRepairAPI, open_hf_moe_uniform

    model = tmp_path / "numeric-experts-split-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"n_routed_experts": 1, "num_hidden_layers": 1}) + "\n"
    )
    shard = model / "model-00001-of-00001.safetensors"
    tensors = {
        "layers.0.experts.0.down_proj.weight": np.arange(16, dtype=np.float16).reshape(2, 8),
        "layers.0.router.weight": np.arange(8, dtype=np.float16).reshape(2, 4),
    }
    save_file(tensors, shard)
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
                "weight_map": {name: shard.name for name in tensors},
            },
            sort_keys=True,
        )
        + "\n"
    )
    output = tmp_path / "artifact"
    spill = tmp_path / "native-spill"

    built = ResidentRepairAPI.build_uniform(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        tier="q2",
        scope="routed_only",
        native_rest=True,
        output=output,
        native_spill_root=spill,
    )
    reopened = open_hf_moe_uniform(output)

    assert reopened == built
    assert built["storage"]["mode"] == "split-native-local-v1"
    assert Path(built["storage"]["native_root"]).is_dir()
    native = built["native_tensors"][0]
    assert native["storage_root"] == "native"
    assert not (output / native["path"]).exists()
    assert (Path(built["storage"]["native_root"]) / native["path"]).is_file()


def test_public_bounded_canary_is_diagnostic_and_projects_complete_build(
    tmp_path: Path,
) -> None:
    from banana_smasher import estimate_hf_moe_uniform

    model = tmp_path / "numeric-experts-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture_numeric_moe",
                "n_routed_experts": 1,
                "num_hidden_layers": 1,
            },
            sort_keys=True,
        )
        + "\n"
    )
    shard = model / "model-00001-of-00001.safetensors"
    tensors = {
        "layers.0.experts.0.down_proj.weight": np.arange(16, dtype=np.float16).reshape(2, 8),
        "layers.0.router.weight": np.arange(8, dtype=np.float16).reshape(2, 4),
    }
    save_file(tensors, shard)
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.nbytes for value in tensors.values())},
                "weight_map": {name: shard.name for name in tensors},
            },
            sort_keys=True,
        )
        + "\n"
    )
    receipt_path = tmp_path / "BUILD_ESTIMATE.json"

    estimate = estimate_hf_moe_uniform(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        tier="q2",
        scope="routed_only",
        native_rest=True,
        receipt_path=receipt_path,
    )

    assert estimate["status"] == "PASS_DIAGNOSTIC"
    assert estimate["artifact_admissible"] is False
    assert estimate["artifact_created"] is False
    assert estimate["canary"]["routed_tensor_count"] == 1
    assert estimate["canary"]["wall_seconds"] > 0
    assert estimate["canary"]["peak_memory_bytes"] > 0
    assert estimate["projection"]["complete_routed_tensor_count"] == 1
    assert estimate["projection"]["complete_wall_seconds"] > 0
    assert estimate["projection"]["complete_payload_bytes"] > 0
    assert json.loads(receipt_path.read_text()) == estimate


def test_public_bounded_canary_reads_safetensors_float8_e4m3(tmp_path: Path) -> None:
    from banana_smasher import estimate_hf_moe_uniform

    model = tmp_path / "float8-numeric-experts-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"n_routed_experts": 1, "num_hidden_layers": 1}) + "\n"
    )
    shard = model / "model-00001-of-00001.safetensors"
    routed_name = "layers.0.experts.0.down_proj.weight"
    native_name = "layers.0.router.weight"
    header = {
        routed_name: {"dtype": "F8_E4M3", "shape": [2, 8], "data_offsets": [0, 16]},
        native_name: {"dtype": "F16", "shape": [2, 4], "data_offsets": [16, 32]},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    shard.write_bytes(
        struct.pack("<Q", len(header_bytes))
        + header_bytes
        + bytes([0x38]) * 16
        + np.arange(8, dtype=np.float16).tobytes()
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {routed_name: shard.name, native_name: shard.name}}) + "\n"
    )

    estimate = estimate_hf_moe_uniform(
        model,
        revision="3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
        tier="q2",
        scope="routed_only",
        native_rest=True,
        receipt_path=tmp_path / "FLOAT8_ESTIMATE.json",
    )

    assert estimate["status"] == "PASS_DIAGNOSTIC"
    assert estimate["canary"]["source_dtype"] == "F8_E4M3"
    assert estimate["canary"]["parameters"] == 16
