from __future__ import annotations

import hashlib
import json
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
        "layers.1.mtp.weight": np.arange(2, dtype=np.float16),
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
            "shard": shard.name,
            "source_bytes": 16,
        }
    ]
    assert {row["name"] for row in plan["native_tensors"]} == {
        "layers.0.experts.0.down_proj.weight_scale_inv",
        "layers.0.shared_experts.down_proj.weight",
        "layers.1.mtp.weight",
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
    assert "plan_hf_moe_uniform(" in worked
    assert 'scope="routed_only"' in worked
    assert "native_rest=True" in worked
