from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import banana_smasher.backpack_runtime_exact64 as backpack_runtime_exact64
from banana_smasher import materialize_provenance_virtual_backpack
from banana_smasher.backpack_runtime_exact64 import _validate_whole_model_accounting
from banana_smasher.backpack_virtual import verify_virtual_backpack
from banana_smasher.d4_wire import decode_d4_expert, unpack_d4_codes
from banana_smasher.hf_deepseek_v4_backpack_adapter import (
    DeepseekV4BackpackRuntime,
    _available_materialization_bytes,
    _fwht,
    bind_recovered_qtip3_split_payload,
)


def test_runtime_accepts_full_closure_recovered_qtip3_receipt(tmp_path) -> None:
    artifact = tmp_path / "QTIP_UNIT.pt"
    artifact.write_bytes(b"sealed-qtip3")
    receipt = {
        "schema": "banana-smasher-recovered-public-api-qtip-unit-v1",
        "status": "PASS",
        "basis_sha256": "a" * 64,
        "cell_id": "L000:E000:down",
        "tier": "qtip3",
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }

    backpack_runtime_exact64.DeepseekV4BackpackRuntime._validate_native_qtip3_receipt(
        receipt,
        layer=0,
        expert=0,
        projection="down",
        basis_sha256="a" * 64,
        artifact_path=artifact,
    )


def test_bind_recovered_qtip3_split_payload_uses_public_api_source(tmp_path) -> None:
    materialized = tmp_path / "materialized" / "L000_E000_down"
    source = tmp_path / "incoming" / "L000_E000_down"
    materialized.mkdir(parents=True)
    source.mkdir(parents=True)
    codes_path = source / "codes.npy"
    np.save(codes_path, np.zeros((16,), dtype=np.uint8), allow_pickle=False)
    control = source / "QTIP_CONTROL.pt"
    torch.save({"shape": [8, 16], "geometry": {"B": 12}}, control)
    codes_sha = hashlib.sha256(codes_path.read_bytes()).hexdigest()
    control_sha = hashlib.sha256(control.read_bytes()).hexdigest()
    cell = {
        "schema": "banana-smasher-qtip-native-v4-cell-v1",
        "status": "PASS",
        "provider": "qtip-native-v6@3.00",
        "basis_sha256": "a" * 64,
        "artifacts": {"codes": {"sha256": codes_sha}},
        "control": {"path": str(control), "sha256": control_sha},
    }
    cell_path = source / "CELL_RECEIPT.json"
    cell_path.write_text(json.dumps(cell, sort_keys=True) + "\n")
    cell_sha = hashlib.sha256(cell_path.read_bytes()).hexdigest()
    public = {
        "schema": "banana-smasher-qtip3-v7-public-api-producer-v1-cell",
        "status": "PASS",
        "basis_sha256": "a" * 64,
        "cell": "L000/E000_down",
        "api_receipt_sha256": cell_sha,
    }
    public_path = source / "PUBLIC_CELL_RECEIPT.json"
    public_path.write_text(json.dumps(public, sort_keys=True) + "\n")
    recovered = {
        "schema": "banana-smasher-recovered-public-api-qtip-unit-v1",
        "status": "PASS",
        "basis_sha256": "a" * 64,
        "cell_id": "L000:E000:down",
        "tier": "qtip3",
        "artifact_bytes": codes_path.stat().st_size,
        "artifact_sha256": codes_sha,
        "public_api_source": {
            "cell_receipt_sha256": cell_sha,
            "codes_sha256": codes_sha,
            "public_receipt_sha256": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        },
    }
    receipt_path = materialized / "QTIP_SOLVE_RECEIPT.json"
    receipt_path.write_text(json.dumps(recovered, sort_keys=True) + "\n")

    updated = bind_recovered_qtip3_split_payload(
        receipt_path=receipt_path, source_unit_root=source, source_host="spark-8"
    )

    assert updated["closure_split_payload"] == {
        "closure_receipt_sha256": cell_sha,
        "control_path": str(control.resolve()),
        "control_sha256": control_sha,
        "codes_path": str(codes_path.resolve()),
        "codes_sha256": codes_sha,
        "source_host": "spark-8",
    }


def test_exact64_accepts_one_window_sanity_bank() -> None:
    backpack_runtime_exact64._validate_bank_window_count(
        expected_windows=1, observed_windows=1
    )


def test_exact64_whole_model_accounting_includes_sealed_envelope_padding() -> None:
    accounting = {
        "expert_physical_wire_bytes": 92_967_396_864,
        "dense_nonrouted_bytes": 9_017_356_608,
        "repair_bytes": 0,
        "metadata_bytes": 14_756_006,
        "fixed_nonexpert_bytes": 9_032_112_614,
        "padding_bytes": 490_522,
        "whole_shipping_bytes": 102_000_000_000,
        "shipping_bytes_cap": 102_000_000_000,
        "shipping_slack_bytes": 0,
        "logical_base_parameters": 236_000_000_000,
        "whole_model_bpw_numerator_bits": 816_000_000_000,
        "whole_model_bpw_exact_ratio": "816000000000/236000000000",
        "whole_model_bpw_decimal": (
            "3.4576271186440677966101694915254237288135593220338983050847457627118644067796610"
        ),
    }

    assert _validate_whole_model_accounting(
        {"whole_model_accounting": accounting}
    ) == accounting


def test_gb10_materialization_admission_uses_reclaimable_host_memory(tmp_path) -> None:
    class Cuda:
        @staticmethod
        def mem_get_info():
            return 1 << 30, 1 << 30

        @staticmethod
        def get_device_properties(device):
            assert device == "cuda"
            return type("Properties", (), {"name": "NVIDIA GB10"})()

    torch = type("Torch", (), {"cuda": Cuda()})()
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   115754600 kB\n")

    assert _available_materialization_bytes(
        torch, "cuda", meminfo_path=meminfo
    ) == 115754600 * 1024


def test_qtip_cuda_fwht_uses_required_fused_quack_backend(monkeypatch) -> None:
    calls = []
    sentinel = object()

    class FakeCudaValue:
        shape = (2, 8)
        is_cuda = True

        def contiguous(self):
            return self

    hadamard = types.ModuleType("quack.hadamard")

    def hadamard_transform(value, *, scale):
        calls.append((value, scale))
        return sentinel

    setattr(hadamard, "hadamard_transform", hadamard_transform)
    quack = types.ModuleType("quack")
    quack.__path__ = []
    monkeypatch.setitem(sys.modules, "quack", quack)
    monkeypatch.setitem(sys.modules, "quack.hadamard", hadamard)

    value = FakeCudaValue()
    assert _fwht(torch, value) is sentinel
    assert calls == [(value, 1 / math.sqrt(8))]


def test_unpack_d4_codes_round_trips_fixed_d4_wire() -> None:
    for bits in (11, 12):
        values = np.asarray([0, 1, (1 << bits) - 1, 17, 513], dtype=np.uint16)
        wire = np.packbits(
            (
                (values[:, None] >> np.arange(bits, dtype=np.uint16)) & 1
            ).astype(np.uint8).reshape(-1),
            bitorder="little",
        )

        actual = unpack_d4_codes(wire, bits=bits, count=values.size)

        np.testing.assert_array_equal(actual, values)
        np.testing.assert_array_equal(
            unpack_d4_codes(wire.tobytes(), bits=bits, count=values.size), values
        )


def test_decode_d4_expert_applies_codebook_and_e8m0_scales() -> None:
    rows, columns, bits = 2, 32, 11
    codes = np.arange(rows * columns // 4, dtype=np.uint16)
    packed = np.packbits(
        (
            (codes[:, None] >> np.arange(bits, dtype=np.uint16)) & 1
        ).astype(np.uint8).reshape(-1),
        bitorder="little",
    )
    codebook = np.zeros((1 << bits, 4), dtype=np.float16)
    codebook[codes] = np.stack(
        [
            codes,
            codes + 1,
            codes + 2,
            codes + 3,
        ],
        axis=1,
    )
    scales = np.asarray([[127], [128]], dtype=np.uint8)

    actual = decode_d4_expert(
        packed,
        scales,
        codebook,
        bits=bits,
        rows=rows,
        columns=columns,
        torch=torch,
        device="cpu",
    )

    expected = codebook[codes].reshape(rows, columns).astype(np.float32)
    expected[1] *= 2.0
    np.testing.assert_array_equal(actual.float().numpy(), expected)


def test_runtime_selects_exact_expert_slice_from_d4_pack(monkeypatch) -> None:
    rows, columns, bits = 4096, 2048, 11
    code_bytes = rows * columns // 4 * bits // 8
    scale_bytes = rows * columns // 32
    arrays = {
        "layers.0.truevq_d4.d4_k2048.down.expert_ids": np.asarray(
            [17, 29], dtype=np.int16
        ),
        "layers.0.truevq_d4.d4_k2048.down.codes": np.concatenate(
            (
                np.full(code_bytes, 3, dtype=np.uint8),
                np.full(code_bytes, 7, dtype=np.uint8),
            )
        ),
        "layers.0.truevq_d4.d4_k2048.down.scales": np.concatenate(
            (
                np.full(scale_bytes, 127, dtype=np.uint8),
                np.full(scale_bytes, 128, dtype=np.uint8),
            )
        ),
        "layers.0.truevq_d4.d4_k2048.down.codebooks": np.zeros(
            (2048, 4), dtype=np.float16
        ),
    }

    class View:
        def get(self, name):
            return arrays[name]

    captured = {}

    def fake_decode(codes, scales, codebook, **kwargs):
        captured.update(
            codes=np.asarray(codes),
            scales=np.asarray(scales),
            codebook=np.asarray(codebook),
            kwargs=kwargs,
        )
        return "decoded"

    monkeypatch.setattr(
        "banana_smasher.hf_deepseek_v4_backpack_adapter.decode_d4_expert",
        fake_decode,
    )
    runtime = DeepseekV4BackpackRuntime.__new__(DeepseekV4BackpackRuntime)
    runtime.torch = object()
    runtime.device = "cuda"

    actual = runtime._decode_d4("d4_k2048", 0, 29, "down", View())

    assert actual == "decoded"
    assert captured["codes"].shape == (code_bytes,)
    assert captured["scales"].shape == (scale_bytes,)
    assert np.all(captured["codes"] == 7)
    assert np.all(captured["scales"] == 128)
    assert captured["codebook"].shape == (2048, 4)
    assert captured["kwargs"] == {
        "bits": 11,
        "rows": rows,
        "columns": columns,
        "torch": runtime.torch,
        "device": "cuda",
    }


def test_runtime_batches_native_qtip3_payloads_through_one_public_decode(monkeypatch) -> None:
    runtime = DeepseekV4BackpackRuntime.__new__(DeepseekV4BackpackRuntime)
    runtime.torch = torch  # type: ignore[assignment]
    runtime.device = "cpu"
    calls = []

    def decode(packed, scales, *, positions, tlut, geometry):
        calls.append((tuple(packed.shape), tuple(scales.shape), positions, tuple(tlut.shape)))
        return torch.arange(packed.shape[0] * positions, dtype=torch.float32).reshape(
            packed.shape[0], positions
        )

    monkeypatch.setattr(
        "banana_smasher.hf_deepseek_v4_backpack_adapter.decode_native_v4_torch",
        decode,
    )
    monkeypatch.setattr(
        "banana_smasher.hf_deepseek_v4_backpack_adapter._fwht",
        lambda _torch, value: value,
    )
    tlut = torch.zeros(512, 2)
    payloads = [
        {
            "shape": [2, 8],
            "trellis": torch.full((2, 3), index, dtype=torch.uint8),
            "tlut": tlut.clone(),
            "Wscale": torch.tensor(1.0),
            "SV": torch.ones(2),
            "SU": torch.ones(8),
        }
        for index in range(2)
    ]

    observed = runtime._decode_native_qtip3_payloads(payloads)

    assert calls == [((4, 3), (4,), 8, (512, 2))]
    assert len(observed) == 2
    assert tuple(observed[0].shape) == (2, 8)
    assert torch.equal(observed[0], torch.arange(16).reshape(2, 8).to(torch.bfloat16))
    assert torch.equal(
        observed[1], torch.arange(16, 32).reshape(2, 8).to(torch.bfloat16)
    )


def test_runtime_composes_closure_bound_legacy_qtip2_split_payload(tmp_path) -> None:
    control_path = tmp_path / "control.pt"
    codes_path = tmp_path / "codes.npy"
    torch.save(
        {
            "schema": "banana-smasher-qtip2-public-unit-v1",
            "shape": [16, 8],
            "trellis": torch.zeros(1, dtype=torch.int16),
            "SU": torch.ones(8),
            "SV": torch.ones(16),
            "Wscale": torch.tensor(1.0),
            "tlut": torch.zeros(512, 2),
            "geometry": {
                "L": 16,
                "K": 2,
                "V": 2,
                "tlut_bits": 9,
                "decode_mode": "quantlut_sym",
            },
        },
        control_path,
    )
    np.save(codes_path, np.arange(32, dtype=np.uint8))
    observed = []
    runtime = DeepseekV4BackpackRuntime.__new__(DeepseekV4BackpackRuntime)
    runtime.torch = torch
    runtime._record_path = lambda path: observed.append(path)  # type: ignore[method-assign]
    receipt = {
        "closure_split_payload": {
            "closure_receipt_sha256": "1" * 64,
            "control_path": str(control_path),
            "control_sha256": "2" * 64,
            "codes_path": str(codes_path),
            "codes_sha256": "3" * 64,
            "source_host": "fixture",
        }
    }

    payload = runtime._load_qtip_payload(
        receipt=receipt,
        artifact_path=tmp_path / "absent-monolith.pt",
        source_key="qtip2",
    )

    assert observed == [control_path, codes_path]
    assert payload["schema"] == "banana-smasher-qtip-unit-v1"
    assert payload["geometry"]["K"] == 2
    assert payload["trellis"].dtype == torch.uint8
    assert payload["trellis"].numel() == 32
    assert payload["trellis"].tolist() == list(range(32))


def test_runtime_composes_native_v6_qtip3_split_with_bound_shared_tlut(
    tmp_path,
) -> None:
    basis = "a" * 64
    root = tmp_path / "qtip3"
    cell_root = root / "outputs" / "full_api" / "L002_E000_down"
    cell_root.mkdir(parents=True)
    inputs = root / "inputs"
    inputs.mkdir()
    control_path = tmp_path / "control.pt"
    torch.save(
        {
            # The source control can be inherited from an older QTIP2 solve;
            # native-v6 QTIP3 identity comes from the public cell receipt.
            "schema": "banana-smasher-qtip2-public-unit-v1",
            "shape": [16, 8],
            "trellis": torch.zeros(1, dtype=torch.int16),
            "SU": torch.ones(8),
            "SV": torch.ones(16),
            "Wscale": torch.tensor(1.0),
            "tlut": torch.zeros(512, 2),
            "geometry": {
                "L": 16,
                "K": 2,
                "V": 2,
                "tlut_bits": 9,
                "decode_mode": "quantlut_sym",
            },
        },
        control_path,
    )
    codes_path = cell_root / "codes.npy"
    np.save(codes_path, np.arange(48, dtype=np.uint8).reshape(16, 3))
    tlut_path = inputs / "qtip_tlut.npy"
    tlut = np.arange(1024, dtype=np.float32).reshape(512, 2)
    np.save(tlut_path, tlut)
    control_sha = hashlib.sha256(control_path.read_bytes()).hexdigest()
    codes_sha = hashlib.sha256(codes_path.read_bytes()).hexdigest()
    tlut_sha = hashlib.sha256(tlut_path.read_bytes()).hexdigest()
    tlut_tensor_sha = hashlib.sha256(tlut.tobytes(order="C")).hexdigest()
    geometry = {
        "L": 16,
        "B": 12,
        "V": 4,
        "rate_num": 3,
        "rate_den": 1,
        "phase_count": 1,
        "unique_transition_bits_per_payload": 1,
        "alternation": False,
        "member_averaging": False,
        "tlut_bits": 9,
        "decode_mode": "paired_quantlut_sym",
    }
    (cell_root / "CELL_RECEIPT.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-native-v4-cell-v1",
                "status": "PASS",
                "provider": "qtip-native-v6@3.00",
                "codec_version": "v6",
                "basis_sha256": basis,
                "geometry": geometry,
                "artifacts": {"codes": {"sha256": codes_sha}},
                "control": {"sha256": control_sha},
                "tlut": {
                    "sha256": tlut_sha,
                    "tensor_sha256": tlut_tensor_sha,
                },
            }
        )
        + "\n"
    )
    observed = []
    runtime = DeepseekV4BackpackRuntime.__new__(DeepseekV4BackpackRuntime)
    runtime.torch = torch  # type: ignore[assignment]
    runtime.basis_sha256 = basis
    runtime._record_path = lambda path: observed.append(path)  # type: ignore[method-assign]

    payload = runtime._load_qtip_payload(
        receipt={
            "closure_split_payload": {
                "closure_receipt_sha256": "1" * 64,
                "control_path": str(control_path),
                "control_sha256": control_sha,
                "codes_path": str(codes_path),
                "codes_sha256": codes_sha,
                "source_host": "fixture",
            }
        },
        artifact_path=tmp_path / "absent-monolith.pt",
        source_key="qtip3",
    )

    assert observed == [
        control_path,
        codes_path,
        cell_root / "CELL_RECEIPT.json",
        tlut_path,
    ]
    assert payload["schema"] == "banana-smasher-qtip3-native-v6-unit-v1"
    assert payload["geometry"] == geometry
    assert payload["trellis"].shape == (16, 3)
    assert payload["tlut"].shape == (512, 2)


def test_materialize_provenance_assignment_for_exact64(tmp_path) -> None:
    basis = "1" * 64
    bank = "2" * 64
    teacher = "3" * 64
    scorer = "4" * 64
    model_id = "deepseek-ai/DeepSeek-V4-Flash-0731"
    model_revision = "0731"
    producers = {}
    sources = {}
    for tier in ("native_mxfp4", "d4_k2048"):
        root = tmp_path / tier
        root.mkdir()
        identity = root / "IDENTITY.json"
        identity.write_text(json.dumps({"tier": tier}) + "\n")
        digest = hashlib.sha256(identity.read_bytes()).hexdigest()
        producer_receipt = tmp_path / f"{tier}.producer.json"
        producer_receipt.write_text(json.dumps({"tier": tier, "status": "PASS"}) + "\n")
        producer_digest = hashlib.sha256(producer_receipt.read_bytes()).hexdigest()
        producers[tier] = {
            "path": str(producer_receipt),
            "sha256": producer_digest,
            "artifact_sha256": producer_digest,
        }
        sources[tier] = {
            "root": str(root),
            "identity": identity.name,
            "identity_sha256": digest,
            "basis_sha256": basis,
        }
    source_bindings = tmp_path / "sources.json"
    source_bindings.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-backpack-virtual-sources-v1",
                "sources": sources,
            }
        )
        + "\n"
    )
    common = {
        "schema": "banana-smasher-provenance-option-row-v1",
        "model_id": model_id,
        "model_revision": model_revision,
        "basis_sha256": basis,
        "bank_sha256": bank,
        "teacher_sha256": teacher,
        "scorer_sha256": scorer,
        "prediction_by_class": {"chat": 0.1},
        "activation_ids": [],
        "prediction_producer": producers["native_mxfp4"],
    }
    ledger_rows = [
        {
            **common,
            "layer": 0,
            "expert": 0,
            "projection": "down",
            "cell_id": "L000:E000:down",
            "tier": "native_mxfp4",
            "physical_bytes": 11,
            "physical_producer": producers["native_mxfp4"],
        },
        {
            **common,
            "layer": 0,
            "expert": 0,
            "projection": "fused13",
            "cell_id": "L000:E000:fused13",
            "tier": "d4_k2048",
            "physical_bytes": 13,
            "physical_producer": producers["d4_k2048"],
            "activation_ids": ["sha256:" + "5" * 64],
        },
    ]
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows))
    ledger_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
    whole = {
        "shipping_bytes_cap": 34,
        "expert_envelope_bytes": 27,
        "selected_expert_bytes": 27,
        "dense_nonrouted_bytes": 5,
        "repair_bytes": 0,
        "metadata_bytes": 2,
        "fixed_nonexpert_bytes": 7,
        "whole_shipping_bytes": 34,
        "shipping_slack_bytes": 0,
    }
    assignment = tmp_path / "assignment.json"
    assignment.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-provenance-weighted-assignment-v1",
                "status": "PASS_PREDICTION_ONLY",
                "model_id": model_id,
                "model_revision": model_revision,
                "basis_sha256": basis,
                "bank_sha256": bank,
                "teacher_sha256": teacher,
                "scorer_sha256": scorer,
                "whole_model_accounting": whole,
                "activation_artifacts": [
                    {
                        "id": "sha256:" + "5" * 64,
                        "bytes": 3,
                        "role": "d4_codebook",
                    }
                ],
                "assignments": [
                    {
                        "cell_id": row["cell_id"],
                        "tier": row["tier"],
                        "bytes": row["physical_bytes"],
                        "prediction_by_class": row["prediction_by_class"],
                    }
                    for row in ledger_rows
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    assignment_sha = hashlib.sha256(assignment.read_bytes()).hexdigest()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-provenance-weighted-solve-receipt-v1",
                "status": "PASS",
                "assignment": {
                    "path": str(assignment),
                    "sha256": assignment_sha,
                    "bytes": assignment.stat().st_size,
                },
                "option_ledger": {
                    "path": str(ledger),
                    "sha256": ledger_sha,
                    "bytes": ledger.stat().st_size,
                },
                "whole_model_accounting": whole,
            },
            sort_keys=True,
        )
        + "\n"
    )
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    for producer in producers.values():
        Path(producer["path"]).unlink()

    result = materialize_provenance_virtual_backpack(
        assignment,
        receipt,
        ledger,
        source_bindings,
        tmp_path / "virtual",
        expected_assignment_sha256=assignment_sha,
        expected_solve_receipt_sha256=receipt_sha,
        logical_base_parameters=100,
    )

    assert result["status"] == "PASS"
    assert result["tier_counts"] == {"d4_k2048": 1, "native_mxfp4": 1}
    assert result["byte_accounting"]["assigned_package_bytes"] == 34
    manifest = json.loads(
        (tmp_path / "virtual" / "BACKPACK_VIRTUAL_MANIFEST.json").read_text()
    )
    assert manifest["whole_model_accounting"]["whole_model_bpw_exact_ratio"] == "272/100"
    assert manifest["byte_accounting"]["activation_bytes"] == 3
    assert manifest["source_assignment"]["sha256"] == assignment_sha


def test_verify_virtual_backpack_charges_explicit_whole_model_padding(tmp_path) -> None:
    test_materialize_provenance_assignment_for_exact64(tmp_path)
    manifest_path = tmp_path / "virtual" / "BACKPACK_VIRTUAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["byte_accounting"]["fixed_nonexpert_bytes"] -= 1
    manifest["whole_model_accounting"]["fixed_nonexpert_bytes"] -= 1
    manifest["whole_model_accounting"]["metadata_bytes"] -= 1
    manifest["whole_model_accounting"]["padding_bytes"] = 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    result = verify_virtual_backpack(tmp_path / "virtual")

    assert result["logical_materialized_bytes"] == 34


def test_verify_virtual_backpack_rejects_missing_sampled_qtip_payload(tmp_path) -> None:
    test_materialize_provenance_assignment_for_exact64(tmp_path)
    virtual = tmp_path / "virtual"
    index_path = virtual / "MATERIALIZATION_INDEX.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    for row, tier in zip(rows, ("qtip2", "qtip3"), strict=True):
        unit = tmp_path / tier / f"L{row['layer']:03d}" / f"E{row['expert']:03d}_{row['projection']}"
        unit.mkdir(parents=True, exist_ok=True)
        artifact = unit / "QTIP_UNIT.pt"
        artifact.write_bytes(f"physical-{tier}".encode())
        artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
        receipt = unit / "QTIP_SOLVE_RECEIPT.json"
        receipt.write_text(
            json.dumps({"status": "PASS", "artifact_sha256": artifact_sha}, sort_keys=True)
            + "\n"
        )
        row.update(
            tier=tier,
            source_key=tier,
            physical_receipt_path=str(receipt),
            physical_receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
            physical_artifact_sha256=artifact_sha,
        )
    index_raw = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
    index_path.write_bytes(index_raw)
    manifest_path = virtual / "BACKPACK_VIRTUAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["materialization_index"].update(
        sha256=hashlib.sha256(index_raw).hexdigest(), bytes=len(index_raw)
    )
    manifest["tier_counts"] = {"qtip2": 1, "qtip3": 1}
    manifest["source_component_counts"] = {"qtip2": 1, "qtip3": 1}
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    (tmp_path / "qtip3/L000/E000_fused13/QTIP_UNIT.pt").unlink()

    with pytest.raises(ValueError, match="content spot-check"):
        verify_virtual_backpack(virtual)
