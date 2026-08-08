from __future__ import annotations

from pathlib import Path

import numpy as np

from banana_smasher import (
    anchor_qtip25_native_v4_cell,
    build_qtip25_native_v4_cell,
    build_qtip_native_v4_cell,
)
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.cli import main
from banana_smasher import qtip25_native_v4_cuda_cell


CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.npy"
    control = tmp_path / "control.npz"
    tlut = tmp_path / "tlut.npy"
    rng = np.random.default_rng(17)
    np.save(source, rng.normal(size=(16, 16)).astype(np.float32), allow_pickle=False)
    np.savez(
        control,
        SU=np.ones(16, dtype=np.float16),
        SV=np.ones(16, dtype=np.float16),
        Wscale=np.asarray(1.0, dtype=np.float32),
        shape=np.asarray([16, 16], dtype=np.int64),
        qtip_k=np.asarray(2, dtype=np.int64),
    )
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)
    return source, control, tlut


def test_build_and_anchor_native_v4_cell_through_public_api(tmp_path: Path) -> None:
    source, control, tlut = _fixture(tmp_path)
    candidate_root = tmp_path / "candidate"
    receipt = build_qtip25_native_v4_cell(
        source,
        control,
        tlut,
        candidate_root,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        backend="reference",
    )

    assert receipt["schema"] == "banana-smasher-qtip25-native-v4-cell-v1"
    assert receipt["status"] == "PASS"
    assert receipt["backend"] == "reference"
    assert receipt["geometry"] == {
        "L": 16,
        "B": 10,
        "V": 4,
        "rate_num": 5,
        "rate_den": 2,
        "phase_count": 1,
        "unique_transition_bits_per_payload": 1,
        "alternation": False,
        "member_averaging": False,
        "tlut_bits": 9,
        "decode_mode": "paired_quantlut_sym",
    }
    assert receipt["accounting"]["exact_code_bpw"] == 2.5
    assert receipt["accounting"]["code_data_bytes"] == 80
    assert receipt["accounting"]["transform_bytes"] == 64
    assert receipt["artifacts"]["codes"]["sha256"]
    assert np.load(candidate_root / "SU.npy", allow_pickle=False).dtype == np.float16
    decoded = np.load(candidate_root / "decoded.npy", allow_pickle=False)
    assert decoded.dtype == np.float32
    assert decoded.shape == (16, 16)

    rng = np.random.default_rng(23)
    bank = tmp_path / "anchor64.npz"
    np.savez(
        bank,
        features=rng.normal(size=(64, 256)).astype(np.float32),
        classes=np.asarray([CLASSES[index % len(CLASSES)] for index in range(64)]),
    )
    anchor = anchor_qtip25_native_v4_cell(
        candidate_root,
        anchor_bank=bank,
        teacher=source,
        output=tmp_path / "ANCHOR.json",
    )

    assert anchor["schema"] == "banana-smasher-qtip25-native-v4-anchor-v1"
    assert anchor["status"] == "PASS"
    assert anchor["same_instrument"] is True
    assert anchor["windows"] == 64
    assert anchor["candidate_receipt_sha256"] == receipt["receipt_sha256"]
    assert set(anchor["metrics"]["by_class"]) == set(CLASSES)
    assert Path(anchor["receipt"]).is_file()


def test_build_native_v4_cell_can_skip_decoded_file_for_packed_product(
    tmp_path: Path,
) -> None:
    source, control, tlut = _fixture(tmp_path)
    candidate_root = tmp_path / "packed-candidate"

    receipt = build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        candidate_root,
        bpw=3.0,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        backend="reference",
        materialize_decoded=False,
    )

    assert receipt["accounting"]["exact_code_bpw"] == 3.0
    assert receipt["decode_validation"]["tensor_sha256"]
    assert receipt["decode_validation"]["materialized"] is False
    assert "decoded" not in receipt["artifacts"]
    assert not (candidate_root / "decoded.npy").exists()
    assert receipt["direct_error"]["mse"] >= 0.0


def test_shared_rate_parameterized_api_accepts_compact_pr31_lut_at_qtip3_rate(
    tmp_path: Path,
) -> None:
    source, control, tlut = _fixture(tmp_path)
    np.save(
        tlut,
        np.linspace(-2.0, 2.0, 1024, dtype=np.float16),
        allow_pickle=False,
    )

    receipt = build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        tmp_path / "shared-qtip3-candidate",
        bpw=3.0,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        backend="reference",
        codec_version="v6",
        materialize_decoded=False,
        scale_factors=(1.0,),
        ldlq_scale_semantics="rms_ratio",
        feedback_mode="off",
        trellis_objective="lexicographic_l4",
    )

    assert receipt["codec_version"] == "v6"
    assert receipt["geometry"]["rate_num"] == 3
    assert receipt["tlut"]["shape"] == [1024]
    assert receipt["tlut"]["dtype"] == "float16"
    assert receipt["optimization"]["feedback_mode"] == "off"
    assert receipt["optimization"]["scale_factors"] == [1.0]
    assert receipt["optimization"]["trellis_objective"] == "lexicographic_l4"


def test_cuda_cell_uses_selected_scale_from_cuda_optimization(
    tmp_path: Path, monkeypatch
) -> None:
    source, control, tlut = _fixture(tmp_path)
    output = tmp_path / "cuda-candidate"

    def fake_run_cuda_cell(*_args, **_kwargs):
        np.save(output / "codes.npy", np.zeros((1, 96), dtype=np.uint8), allow_pickle=False)
        (output / "NATIVE_V4_CELL_RECEIPT.json").write_text("{}\n")
        return {
            "encode": {"wall_seconds": 1.0},
            "optimization": {
                "method": "rms_only_no_feedback",
                "base_scale": 1.0,
                "selected_factor": 1.0,
                "selected_scale": 1.0,
                "scale_factors": [1.0],
                "feedback_nonzero_count": 0,
            },
            "installed_cuda_decode": {"counters": {"fallback_calls": 0}},
            "cuda": {"device": "test"},
        }

    monkeypatch.setattr(qtip25_native_v4_cuda_cell, "run_cuda_cell", fake_run_cuda_cell)

    receipt = build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        output,
        bpw=3.0,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        backend="cuda",
        materialize_decoded=False,
    )

    assert receipt["optimization"]["selected_scale"] == 1.0
    assert receipt["decode_validation"]["materialized"] is False


def test_native_v4_cli_builds_and_anchors_a_cell(tmp_path: Path, capsys) -> None:
    source, control, tlut = _fixture(tmp_path)
    candidate = tmp_path / "candidate-cli"
    assert main(
        [
            "qtip-native-v4",
            "build-cell",
            "--source",
            str(source),
            "--control",
            str(control),
            "--tlut",
            str(tlut),
            "--output",
            str(candidate),
            "--intended-basis-sha256",
            "9" * 64,
            "--observed-basis-sha256",
            "9" * 64,
            "--backend",
            "reference",
        ]
    ) == 0
    build_output = capsys.readouterr().out
    assert '"exact_code_bpw": 2.5' in build_output

    rng = np.random.default_rng(29)
    bank = tmp_path / "anchor64-cli.npz"
    np.savez(
        bank,
        features=rng.normal(size=(64, 256)).astype(np.float32),
        classes=np.asarray([CLASSES[index % len(CLASSES)] for index in range(64)]),
    )
    anchor_path = tmp_path / "ANCHOR_CLI.json"
    assert main(
        [
            "qtip-native-v4",
            "anchor-cell",
            "--candidate",
            str(candidate),
            "--anchor-bank",
            str(bank),
            "--teacher",
            str(source),
            "--output",
            str(anchor_path),
        ]
    ) == 0
    anchor_output = capsys.readouterr().out
    assert '"same_instrument": true' in anchor_output
    assert anchor_path.is_file()
