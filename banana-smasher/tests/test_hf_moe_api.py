from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
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


def test_source_admission_reclaims_hashed_pages_when_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import banana_smasher.hf_moe as hf_moe

    calls: list[tuple[int, int, int, int]] = []
    monkeypatch.setattr(hf_moe.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    monkeypatch.setattr(
        hf_moe.os,
        "posix_fadvise",
        lambda fd, offset, length, advice: calls.append((fd, offset, length, advice)),
        raising=False,
    )
    source = tmp_path / "member.bin"
    source.write_bytes(b"x" * ((8 << 20) + 3))

    assert hf_moe._sha256(source) == hashlib.sha256(source.read_bytes()).hexdigest()
    assert [(offset, length, advice) for _, offset, length, advice in calls] == [
        (0, 8 << 20, 4),
        (8 << 20, 3, 4),
    ]


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
    assert plan["geometry"]["auxiliary_layer_ids"] == [1]
    assert plan["geometry"]["expected_model_layers"] == 1
    assert plan["geometry"]["model_layer_gaps"] == []
    assert plan["geometry"]["model_layer_ids"] == [0]
    assert plan["geometry"]["routed_layer_ids"] == [0]
    # G15: the auxiliary-layer semantics are stated inline in the receipt, not implied.
    assert "num_hidden_layers" in plan["geometry"]["auxiliary_layer_rule"]
    assert "num_hidden_layers" in plan["geometry"]["auxiliary_layer_deciding_config_keys"]
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


def test_public_hf_moe_build_batches_equal_width_routed_tensors_exactly(
    tmp_path: Path,
) -> None:
    from banana_smasher import build_hf_moe_uniform, build_hf_moe_uniform_shard

    model = tmp_path / "numeric-experts-model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fixture_numeric_moe",
                "n_routed_experts": 2,
                "num_hidden_layers": 1,
            },
            sort_keys=True,
        )
        + "\n"
    )
    shard = model / "model-00001-of-00001.safetensors"
    tensors = {
        f"layers.0.experts.{expert}.{projection}_proj.weight": (
            np.arange(2 * width, dtype=np.float16).reshape(2, width)
            + expert
            + projection_ordinal
        )
        for expert in range(2)
        for projection_ordinal, (projection, width) in enumerate(
            (("down", 8), ("gate", 16), ("up", 16))
        )
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
    revision = "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"

    batched = build_hf_moe_uniform(
        model,
        revision=revision,
        tier="q2",
        scope="routed_only",
        native_rest=True,
        output=tmp_path / "batched",
    )
    isolated = [
        build_hf_moe_uniform_shard(
            model,
            revision=revision,
            tier="q2",
            scope="routed_only",
            native_rest=True,
            routed_ordinal_start=ordinal,
            routed_ordinal_end=ordinal + 1,
            output=tmp_path / f"isolated-{ordinal}",
        )
        for ordinal in range(6)
    ]

    assert batched["acceleration"] == {
        "routed_encode_batches": 2,
        "routed_tensors_batched": 6,
        "max_batch_tensors": 10,
        "same_width_batching": True,
    }
    assert [
        (
            row["wire"]["trellis"]["sha256"],
            row["wire"]["scales"]["sha256"],
        )
        for row in batched["routed_tensors"]
    ] == [
        (
            result["routed_tensors"][0]["wire"]["trellis"]["sha256"],
            result["routed_tensors"][0]["wire"]["scales"]["sha256"],
        )
        for result in isolated
    ]


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


def test_q2_gpu_encoder_memory_plan_is_bounded_before_allocation() -> None:
    from banana_smasher.qtip1 import plan_qtip2_cuda_chunks

    plan = plan_qtip2_cuda_chunks(
        rows=2048,
        width=4096,
        free_bytes=16 << 30,
        reserve_bytes=4 << 30,
    )

    assert plan["status"] == "PASS"
    assert plan["chunk_rows"] == 2048
    assert plan["chunk_count"] == 1
    assert plan["peak_memory_bytes"] <= (16 << 30) - (4 << 30)
    assert plan["full_batch_backpointer_bytes"] == 8_589_934_592
    assert plan["bounded_backpointer_bytes"] == plan["full_batch_backpointer_bytes"]
    assert int(plan["chunk_rows"]) <= 8192
    assert int(plan["exact_max_chunk_rows"]) == 8192
    assert int(plan["backpointer_address_bits"]) == 64
    assert int(plan["backpointer_bits_per_prefix_step"]) == 4
    assert plan["encoder"] == "full-row-packed-backpointer-cuda"


def test_q2_full_row_cuda_source_preserves_exact_arithmetic_and_tie_contract() -> None:
    source = (
        Path(__file__).parents[1]
        / "src/banana_smasher/trellis_v2/csrc/trellis_v2_exact.cu"
    ).read_text()

    assert "full_row_k2_viterbi" in source
    assert "#include <math_constants.h>" in source
    assert "__fmul_rn" in source
    assert "__fadd_rn" in source
    assert "candidate < best" in source
    assert "q00 | (q01 << 4)" in source
    assert "q10 | (q11 << 4)" in source
    assert "float2" not in source


def test_q2_gpu_encoder_memory_plan_refuses_before_allocation() -> None:
    from banana_smasher.qtip1 import plan_qtip2_cuda_chunks

    with pytest.raises(RuntimeError, match="QTIP2 CUDA peak-memory admission failed"):
        plan_qtip2_cuda_chunks(
            rows=2048,
            width=4096,
            free_bytes=(4 << 30) + (1 << 20),
            reserve_bytes=4 << 30,
        )


def test_q2_gpu_encoder_matches_known_current_output_hash() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("known-control output-hash gate requires CUDA")
    from banana_smasher.qtip1 import (
        QTIP2_GEOMETRY,
        encode_qtip2_bounded_cuda,
        gaussian_tlut,
    )

    matrix = (np.arange(64, dtype=np.float32).reshape(4, 16) - 31.5) / 7
    encoded, report = encode_qtip2_bounded_cuda(
        matrix,
        geometry=QTIP2_GEOMETRY,
        tlut=gaussian_tlut(bits=9, columns=2),
    )

    assert hashlib.sha256(encoded.packed.tobytes()).hexdigest() == (
        "fe136b4f2340f5e462cbff8198248fb51ca3806a847d1bab700cded2ab1f74b5"
    )
    assert hashlib.sha256(encoded.states.tobytes()).hexdigest() == (
        "ae445063d0c47ebdaf812935b70497fc065b2721d131c54e42751ccffa543935"
    )
    assert hashlib.sha256(encoded.scales.tobytes()).hexdigest() == (
        "ab9166e7feddc0384435d16d4c9dae7f8dcf2d6683e2484d0ba771a6c2bfe54b"
    )
    assert report["fallback"] == 0
    assert report["memory_plan"]["peak_memory_bytes"] <= (
        report["memory_plan"]["free_bytes"] - report["memory_plan"]["reserve_bytes"]
    )
