from __future__ import annotations

from pathlib import Path

import numpy as np

from banana_smasher import (
    anchor_qtip25_native_v4_cell,
    build_qtip25_native_v4_cell,
)
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.cli import main


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
    assert receipt["optimization"]["method"] == "rms_only_no_feedback"
    assert receipt["optimization"]["selected_scale"] == 1.0
    assert receipt["optimization"]["feedback_nonzero_count"] == 0
    assert receipt["artifacts"]["codes"]["sha256"]
    assert np.load(candidate_root / "SU.npy", allow_pickle=False).dtype == np.float16
    assert np.load(candidate_root / "Wscale.npy", allow_pickle=False) == np.float32(1.0)
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


def test_hessian_requires_explicit_feedback_and_preserves_a_c_d_controls(
    tmp_path: Path,
) -> None:
    source, control, tlut = _fixture(tmp_path)
    with np.load(control, allow_pickle=False) as payload:
        np.savez(
            tmp_path / "absolute-control.npz",
            **{name: payload[name] for name in payload.files if name != "Wscale"},
            Wscale=np.asarray(0.75, dtype=np.float32),
        )
    hessian = np.eye(16, dtype=np.float32)
    hessian[1:, :-1] += np.eye(15, dtype=np.float32) * np.float32(0.1)
    hessian[:-1, 1:] += np.eye(15, dtype=np.float32) * np.float32(0.1)
    hessian_path = tmp_path / "hessian.npy"
    np.save(hessian_path, hessian, allow_pickle=False)

    conservative = tmp_path / "conservative-a"
    a_receipt = build_qtip25_native_v4_cell(
        source,
        tmp_path / "absolute-control.npz",
        tlut,
        conservative,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        backend="reference",
        hessian=hessian_path,
        scale_factors=(0.5, 2.0),
    )

    assert a_receipt["optimization"]["method"] == "rms_only_no_feedback"
    assert a_receipt["optimization"]["scale_semantics"] == "absolute_unit"
    assert a_receipt["optimization"]["selected_scale"] == 1.0
    assert a_receipt["optimization"]["scale_factors"] == [1.0]
    assert a_receipt["optimization"]["feedback_nonzero_count"] == 0
    assert np.load(conservative / "Wscale.npy", allow_pickle=False) == np.float32(0.75)

    absolute_feedback = tmp_path / "explicit-c"
    c_receipt = build_qtip25_native_v4_cell(
        source,
        tmp_path / "absolute-control.npz",
        tlut,
        absolute_feedback,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        backend="reference",
        hessian=hessian_path,
        feedback_mode="reverse_16",
        scale_factors=(0.5, 2.0),
    )

    assert c_receipt["optimization"]["method"] == "qtip_batch_block_ldl_reverse_16"
    assert c_receipt["optimization"]["scale_semantics"] == "absolute_unit"
    assert c_receipt["optimization"]["selected_scale"] == 1.0
    assert c_receipt["optimization"]["scale_factors"] == [1.0]
    assert c_receipt["optimization"]["feedback_mode"] == "reverse_16"
    assert np.load(absolute_feedback / "Wscale.npy", allow_pickle=False) == np.float32(
        0.75
    )

    relative_feedback = tmp_path / "explicit-d"
    d_receipt = build_qtip25_native_v4_cell(
        source,
        tmp_path / "absolute-control.npz",
        tlut,
        relative_feedback,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        backend="reference",
        hessian=hessian_path,
        feedback_mode="reverse_16",
        ldlq_scale_semantics="relative_search",
        scale_factors=(0.5, 2.0),
    )

    assert d_receipt["optimization"]["method"] == "qtip_batch_block_ldl_reverse_16"
    assert d_receipt["optimization"]["scale_semantics"] == "relative_search"
    assert d_receipt["optimization"]["scale_factors"] == [0.5, 2.0]
    assert d_receipt["optimization"]["feedback_mode"] == "reverse_16"


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
