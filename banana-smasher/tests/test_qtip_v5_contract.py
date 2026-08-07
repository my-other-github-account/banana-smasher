from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from banana_smasher import (
    DEFAULT_QTIP_V5_MENU,
    build_qtip_native_v4_cell,
)
from banana_smasher.backpack_providers import backpack_provider_from_declaration
from banana_smasher.cli import _parser
from banana_smasher.measured_backpack_spsa import run_measured_spsa
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4 import native_v4_geometry, native_v4_wire_accounting


def test_qtip_v5_menu_declares_all_native_quarter_steps_in_order() -> None:
    expected_ids = (
        "native_mxfp4",
        *(f"native_v4_b{bits}" for bits in range(4, 17)),
        "d4_k2048",
        "d4_k4096",
    )

    assert DEFAULT_QTIP_V5_MENU.tier_ids == expected_ids
    native = DEFAULT_QTIP_V5_MENU.declarations[1:14]
    assert [(tier.family, tier.B, tier.code_bpw) for tier in native] == [
        ("qtip_native_v4", bits, bits / 4) for bits in range(4, 17)
    ]
    for bits in range(4, 17):
        bpw = bits / 4
        geometry = native_v4_geometry(bpw)
        provider = backpack_provider_from_declaration(
            {"family": "qtip_native_v4", "bpw": bpw}
        )
        accounting = native_v4_wire_accounting(
            position_count=256,
            geometry=geometry,
        )

        assert geometry.B == bits
        assert geometry.rate_num / geometry.rate_den == bpw
        assert provider.provider_id == f"qtip-native-v4@{bpw:.2f}"
        assert accounting["code_payload_bytes"] == 256 * bits // 32
        assert accounting["phase_count"] == 1
        assert accounting["alternation"] is False
        assert accounting["member_averaging"] is False


def _cell_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(57)
    source = tmp_path / "source.npy"
    control = tmp_path / "control.npz"
    tlut = tmp_path / "tlut.npy"
    hessian = tmp_path / "hessian.npy"
    np.save(source, rng.normal(size=(32, 32)).astype(np.float32), allow_pickle=False)
    np.savez(
        control,
        SU=np.ones(32, dtype=np.float16),
        SV=np.ones(32, dtype=np.float16),
        Wscale=np.asarray(0.75, dtype=np.float32),
        shape=np.asarray([32, 32], dtype=np.int64),
    )
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)
    calibration = rng.normal(size=(32, 48)).astype(np.float32)
    np.save(
        hessian,
        calibration @ calibration.T + np.eye(32, dtype=np.float32),
        allow_pickle=False,
    )
    return source, control, tlut, hessian


def test_qtip_v5_defaults_to_a_and_keeps_reverse_16_as_explicit_opt_in(
    tmp_path: Path,
) -> None:
    source, control, tlut, hessian = _cell_inputs(tmp_path)
    common = {
        "bpw": 1.0,
        "intended_basis_sha256": "a" * 64,
        "observed_basis_sha256": "a" * 64,
        "backend": "reference",
        "hessian": hessian,
    }

    default_receipt = build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        tmp_path / "default-a",
        **common,
    )
    diagnostic_receipt = build_qtip_native_v4_cell(
        source,
        control,
        tlut,
        tmp_path / "explicit-c",
        feedback_mode="reverse_16",
        **common,
    )

    assert default_receipt["optimization"]["feedback_mode"] == "off"
    assert default_receipt["optimization"]["feedback_nonzero_count"] == 0
    assert default_receipt["optimization"]["selected_scale"] == 1.0
    assert np.load(tmp_path / "default-a" / "Wscale.npy") == np.float32(0.75)
    assert diagnostic_receipt["optimization"]["feedback_mode"] == "reverse_16"
    assert diagnostic_receipt["optimization"]["feedback_nonzero_count"] > 0


def test_qtip_v5_feedback_mode_is_declaration_and_cli_bound(tmp_path: Path) -> None:
    source, control, tlut, hessian = _cell_inputs(tmp_path)
    args = _parser().parse_args(
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
            str(tmp_path / "candidate"),
            "--intended-basis-sha256",
            "b" * 64,
            "--observed-basis-sha256",
            "b" * 64,
            "--hessian",
            str(hessian),
            "--feedback-mode",
            "reverse_16",
        ]
    )

    assert args.feedback_mode == "reverse_16"


def test_measured_search_resume_rejects_ordered_menu_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    root.mkdir()
    (root / "STATE.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-measured-spsa-state-v2",
                "ordered_tier_ids": [],
                "tier_menu": [],
                "tier_menu_sha256": "0" * 64,
            }
        )
    )
    train_slice = {
        "slice_id": "train-0",
        "window_ids": [f"w{index}" for index in range(8)],
        "holdout_used": False,
    }

    with pytest.raises(ValueError, match="tier menu mismatch"):
        run_measured_spsa(
            {},
            {},
            {},
            [train_slice],
            lambda *_args: {},
            root,
            coarse_iterations=0,
            refine_iterations=0,
        )
