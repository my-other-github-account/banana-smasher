from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from banana_smasher import (
    anchor_qtip25_native_v4_cell,
    build_qtip25_native_v4_cell,
)
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.cli import main
from banana_smasher.qtip25_native_v4_api import (
    build_qtip_native_transform_control,
    qtip_transform_seed,
)


CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def test_source_transform_control_is_seeded_basis_bound_and_idempotent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npy"
    output = tmp_path / "L004/E007_down/QTIP_UNIT.pt"
    np.save(source, np.ones((32, 16), dtype=np.float32), allow_pickle=False)
    seed = qtip_transform_seed("domain-v1", "sealed-material", 4, 7, "down")
    assert seed == qtip_transform_seed("domain-v1", "sealed-material", 4, 7, "down")
    assert qtip_transform_seed(
        "qtip-rht-bounded36-v1",
        "4fa7b1213db1d6a4670b534785edb1681d1538bb6d12a90222e33c30251c2462"
        "|t_782dc70e|heldout-experts-v1",
        39,
        0,
        "down",
    ) == 3662494846445047602
    receipt = build_qtip_native_transform_control(
        source,
        output,
        transform_seed=seed,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        device="cpu",
    )
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert receipt["status"] == "PASS"
    assert payload["shape"] == [32, 16]
    assert payload["SU"].dtype == torch.float16
    assert payload["SV"].dtype == torch.float16
    assert set(payload["SU"].tolist()) <= {-1.0, 1.0}
    assert set(payload["SV"].tolist()) <= {-1.0, 1.0}
    assert build_qtip_native_transform_control(
        source,
        output,
        transform_seed=seed,
        intended_basis_sha256="9" * 64,
        observed_basis_sha256="9" * 64,
        device="cpu",
    )["sha256"] == receipt["sha256"]
    with pytest.raises(ValueError, match="basis mismatch"):
        build_qtip_native_transform_control(
            source,
            tmp_path / "bad.pt",
            transform_seed=seed,
            intended_basis_sha256="9" * 64,
            observed_basis_sha256="8" * 64,
            device="cpu",
        )


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
