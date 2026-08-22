from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from banana_smasher import materialize_provenance_virtual_backpack
from banana_smasher.d4_wire import decode_d4_expert, unpack_d4_codes
from banana_smasher.hf_deepseek_v4_backpack_adapter import (
    DeepseekV4BackpackRuntime,
    _available_materialization_bytes,
)


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


def test_runtime_composes_closure_bound_split_qtip_payload(tmp_path) -> None:
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
    np.save(codes_path, np.arange(48, dtype=np.uint8))
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
        source_key="qtip3",
    )

    assert observed == [control_path, codes_path]
    assert payload["schema"] == "banana-smasher-qtip-unit-v1"
    assert payload["geometry"]["K"] == 3
    assert payload["trellis"].dtype == torch.uint8
    assert payload["trellis"].numel() == 48
    assert payload["trellis"].tolist() == list(range(48))


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
            "shape": [16, 8],
            "SU": torch.ones(8),
            "SV": torch.ones(16),
            "Wscale": torch.tensor(1.0),
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
