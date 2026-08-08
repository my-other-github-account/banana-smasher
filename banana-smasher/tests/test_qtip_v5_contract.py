from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import banana_smasher as banana
import banana_smasher.qtip25_native_v4_api as native_qtip_api
from banana_smasher import (
    DEFAULT_QTIP_V5_MENU,
    build_qtip_native_v4_cell,
)
from banana_smasher.backpack_providers import (
    backpack_provider_from_declaration,
    generate_backpack_candidate,
)
from banana_smasher.banana_v1 import (
    banana_v1_gaussian_codebook,
    decode_banana_v1,
    expand_banana_v1_codebook,
)
from banana_smasher.cli import _parser
from banana_smasher.measured_backpack_spsa import run_measured_spsa
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4 import (
    decode_native_v4,
    decode_native_v4_torch,
    native_v4_geometry,
    native_v4_wire_accounting,
    native_v5_edge_states,
    native_v5_phase_widths,
    solve_native_v4,
    states_from_native_v4_packed,
)



def test_native_analysis_inverts_nonunit_output_scales() -> None:
    rng = np.random.default_rng(288275)
    source = rng.normal(size=(32, 32)).astype(np.float32)
    control = {
        "SU": np.linspace(0.5, 1.5, 32, dtype=np.float32),
        "SV": np.linspace(0.75, 1.25, 32, dtype=np.float32),
        "Wscale": np.float32(0.8),
        "shape": source.shape,
    }

    blocks = native_qtip_api._to_normalized_blocks(source, control)
    reconstructed = native_qtip_api._from_normalized_blocks(blocks, control)

    np.testing.assert_allclose(reconstructed, source, rtol=2e-6, atol=2e-6)


def test_historical_analysis_uses_blockwise_hadamard_128_semantics() -> None:
    source = np.arange(32 * 32, dtype=np.float32).reshape(32, 32)
    control = {
        "SU": np.ones(32, dtype=np.float32),
        "SV": np.ones(32, dtype=np.float32),
        "Wscale": np.float32(1.0),
        "shape": source.shape,
        "hadamard_block": 16,
    }

    blocks = native_qtip_api._to_normalized_blocks(source, control)
    observed = (
        blocks.reshape(2, 2, 16, 16).transpose(0, 2, 1, 3).reshape(32, 32)
    )
    expected = np.concatenate(
        [native_qtip_api._fwht(part) for part in np.split(source, 2, axis=1)],
        axis=1,
    )
    expected = np.concatenate(
        [native_qtip_api._fwht(part.T).T for part in np.split(expected, 2, axis=0)],
        axis=0,
    )

    np.testing.assert_array_equal(observed, expected)



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
    assert [tier.lut_identity for tier in native] == [
        "pr31-affine-gaussian-edge-v1"
    ] * 13
    assert [tier.provider for tier in native] == [
        f"qtip-native-v5@{bits / 4:.2f}" for bits in range(4, 17)
    ]
    assert [
        backpack_provider_from_declaration(tier.as_mapping()).provider_id for tier in native
    ] == [f"qtip-native-v5@{bits / 4:.2f}" for bits in range(4, 17)]
    for bits in range(4, 17):
        bpw = bits / 4
        geometry = native_v4_geometry(bpw)
        declaration = native[bits - 4].as_mapping()
        provider = backpack_provider_from_declaration(
            {"family": "qtip_native_v4", "bpw": bpw}
        )
        accounting = native_v4_wire_accounting(
            position_count=256,
            geometry=geometry,
        )

        assert geometry.B == bits
        expected_widths = [
            ((lane + 1) * bits) // 4 - (lane * bits) // 4
            for lane in range(4)
        ]
        assert declaration["phase_widths"] == expected_widths
        assert declaration["lut_multiplier"] == 48917
        assert declaration["lut_offset"] == 50631
        assert geometry.rate_num / geometry.rate_den == bpw
        assert provider.provider_id == f"qtip-native-v4@{bpw:.2f}"
        assert accounting["code_payload_bytes"] == 256 * bits // 32
        assert accounting["phase_count"] == 1
        assert accounting["alternation"] is False
        assert accounting["member_averaging"] is False


def test_qtip_v6_menu_is_the_default_native_ladder() -> None:
    menu = banana.DEFAULT_QTIP_V6_MENU
    native = menu.declarations[1:14]

    assert [tier.B for tier in native] == list(range(4, 17))
    assert [tier.provider for tier in native] == [
        f"qtip-native-v6@{bits / 4:.2f}" for bits in range(4, 17)
    ]
    assert [tier.scale_semantics for tier in native] == ["rms_ratio"] * 13
    assert [
        backpack_provider_from_declaration(tier.as_mapping()).provider_id for tier in native
    ] == [f"qtip-native-v6@{bits / 4:.2f}" for bits in range(4, 17)]


def test_native_v5_b8_decode_exactly_matches_four_sequential_banana_v1_b2_steps() -> None:
    codebook = banana_v1_gaussian_codebook()
    packed = np.arange(64, dtype=np.uint8).reshape(1, -1)
    scales = np.ones(1, dtype=np.float32)

    # Native V4 records each edge's final state at lane 3.  The equivalent
    # circular B2 stream therefore starts three scalar emissions before state 0.
    expected = np.roll(
        decode_banana_v1(
            packed,
            scales,
            positions=256,
            codebook=codebook,
        ),
        3,
        axis=1,
    )
    observed = decode_native_v4(
        packed,
        scales,
        positions=256,
        tlut=codebook,
        geometry=native_v4_geometry(2.0),
    )

    np.testing.assert_array_equal(observed, expected)


def test_native_v5_edge_solver_recovers_a_zero_distortion_b8_path() -> None:
    geometry = native_v4_geometry(2.0)
    codebook = banana_v1_gaussian_codebook()
    packed = np.arange(64, dtype=np.uint8).reshape(1, -1)
    states = states_from_native_v4_packed(packed, steps=64, geometry=geometry)
    edge_states = native_v5_edge_states(
        np.roll(states, 1, axis=1),
        states & (geometry.branches - 1),
        geometry=geometry,
    )
    target = expand_banana_v1_codebook(codebook)[edge_states]

    solved = solve_native_v4(
        target,
        tlut=codebook,
        scales=np.ones(1, dtype=np.float32),
        geometry=geometry,
    )

    assert solved.distortion == 0.0
    np.testing.assert_array_equal(
        decode_native_v4(
            solved.packed,
            solved.scales,
            positions=256,
            tlut=codebook,
            geometry=geometry,
        ).reshape(target.shape),
        target,
    )


def test_qtip_v6_two_cycle_warmup_improves_fixed_b8_tail_biting_path() -> None:
    geometry = native_v4_geometry(2.0)
    target = np.random.default_rng(1).normal(size=(1, 64, 4)).astype(np.float32)

    solved = solve_native_v4(
        target,
        tlut=banana_v1_gaussian_codebook(),
        scales=np.ones(1, dtype=np.float32),
        geometry=geometry,
        cyclic_warmup_cycles=2,
    )

    assert solved.distortion == pytest.approx(14.981470282964779)
    assert (solved.states[0, -1] & (geometry.prefixes - 1)) == (
        solved.states[0, 0] >> geometry.B
    )


def test_native_v5_edges_close_to_the_native_successor_at_every_rate() -> None:
    predecessors = np.asarray([0x0000, 0x0001, 0x1234, 0x8000, 0xFFFF])[:, None]

    for bits in range(4, 17):
        geometry = native_v4_geometry(bits / 4)
        widths = native_v5_phase_widths(geometry=geometry)
        branches = np.asarray([0, 1, (1 << bits) // 3, (1 << bits) - 1])[None, :]
        edge_states = native_v5_edge_states(
            predecessors,
            branches,
            geometry=geometry,
        )

        expected_widths = tuple(
            ((lane + 1) * bits) // 4 - (lane * bits) // 4
            for lane in range(4)
        )
        assert widths == expected_widths
        assert sum(widths) == bits
        np.testing.assert_array_equal(
            edge_states[..., -1],
            ((predecessors << bits) | branches) & 0xFFFF,
        )


def test_native_v5_reference_and_torch_decode_match_at_every_rate() -> None:
    torch = pytest.importorskip("torch")
    codebook = banana_v1_gaussian_codebook()
    scales = np.asarray([0.75], dtype=np.float32)

    for bits in range(4, 17):
        geometry = native_v4_geometry(bits / 4)
        packed = np.arange(8 * bits, dtype=np.uint8).reshape(1, -1)
        expected = decode_native_v4(
            packed,
            scales,
            positions=256,
            tlut=codebook,
            geometry=geometry,
        )
        observed = decode_native_v4_torch(
            torch.from_numpy(packed),
            torch.from_numpy(scales),
            positions=256,
            tlut=torch.from_numpy(codebook),
            geometry=geometry,
        )

        assert torch.equal(observed, torch.from_numpy(expected))


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


def test_qtip_v6_limits_single_cycle_warmup_to_periodic_b10(tmp_path: Path) -> None:
    rng = np.random.default_rng(4286)
    source = tmp_path / "source.npy"
    control = tmp_path / "control.npz"
    codebook = tmp_path / "pr31-codebook.npy"
    source_weights = rng.normal(size=(16, 16)).astype(np.float32)
    table = banana_v1_gaussian_codebook()
    np.save(source, source_weights, allow_pickle=False)
    np.savez(
        control,
        SU=np.ones(16, dtype=np.float16),
        SV=np.ones(16, dtype=np.float16),
        Wscale=np.asarray(1.0, dtype=np.float32),
        shape=np.asarray([16, 16], dtype=np.int64),
    )
    np.save(codebook, table, allow_pickle=False)
    compact, _ = native_qtip_api._load_control(control)
    blocks = native_qtip_api._to_normalized_blocks(source_weights, compact)
    source_rms = float(np.sqrt(np.mean(blocks.astype(np.float64) ** 2)))
    lut_rms = float(
        np.sqrt(
            np.mean(
                expand_banana_v1_codebook(table).astype(np.float64) ** 2
            )
        )
    )
    scale = source_rms / lut_rms

    for bits, expected_cycles in ((8, 2), (10, 1), (12, 2)):
        geometry = native_v4_geometry(bits / 4)
        output = tmp_path / f"b{bits}"
        receipt = banana.build_qtip_native_cell(
            source,
            control,
            codebook,
            output,
            bpw=bits / 4,
            codec_version="v6",
            intended_basis_sha256="4" * 64,
            observed_basis_sha256="4" * 64,
            backend="reference",
        )
        expected = solve_native_v4(
            blocks,
            tlut=table,
            scales=np.full(len(blocks), scale, dtype=np.float32),
            geometry=geometry,
            cyclic_warmup_cycles=expected_cycles,
        )

        assert receipt["optimization"]["cyclic_warmup_cycles"] == expected_cycles
        np.testing.assert_array_equal(
            np.load(output / "codes.npy", allow_pickle=False), expected.packed
        )


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


def test_qtip_v6_cell_build_consumes_compact_pr31_lut_identity(tmp_path: Path) -> None:
    source, control, _old_tlut, _hessian = _cell_inputs(tmp_path)
    codebook = tmp_path / "pr31-codebook.npy"
    np.save(codebook, banana_v1_gaussian_codebook(), allow_pickle=False)

    receipt = banana.build_qtip_native_cell(
        source,
        control,
        codebook,
        tmp_path / "v5-cell",
        bpw=2.5,
        intended_basis_sha256="c" * 64,
        observed_basis_sha256="c" * 64,
        backend="reference",
    )

    assert receipt["tlut"]["shape"] == [1024]
    assert receipt["tlut"]["identity"] == "pr31-affine-gaussian-edge-v1"
    assert receipt["tlut"]["phase_widths"] == [2, 3, 2, 3]
    assert receipt["tlut"]["multiplier"] == 48917
    assert receipt["tlut"]["offset"] == 50631
    assert receipt["accounting"]["shared_tlut_bytes"] == 2048


def test_qtip_v5_cell_build_uses_only_the_compact_pr31_lut(tmp_path: Path) -> None:
    source, control, _old_tlut, _hessian = _cell_inputs(tmp_path)
    lut = tmp_path / "pr31-codebook.npy"
    np.save(lut, banana_v1_gaussian_codebook(), allow_pickle=False)

    receipt = banana.build_qtip_native_cell(
        source,
        control,
        lut,
        tmp_path / "independent-v5-cell",
        bpw=2.0,
        codec_version="v5",
        intended_basis_sha256="3" * 64,
        observed_basis_sha256="3" * 64,
        backend="reference",
        ldlq_scale_semantics="absolute_unit",
    )

    assert receipt["codec_version"] == "v5"
    assert receipt["tlut"]["identity"] == "pr31-affine-gaussian-edge-v1"
    assert receipt["tlut"]["shape"] == [1024]
    assert receipt["accounting"]["shared_tlut_bytes"] == 2048

    forbidden = tmp_path / "external-codec-sized-lut.npy"
    np.save(forbidden, np.zeros(65536, dtype=np.float16), allow_pickle=False)
    with pytest.raises(ValueError, match="compact PR31"):
        banana.build_qtip_native_cell(
            source,
            control,
            forbidden,
            tmp_path / "forbidden-v5-cell",
            bpw=2.0,
            codec_version="v5",
            intended_basis_sha256="3" * 64,
            observed_basis_sha256="3" * 64,
            backend="reference",
        )


def test_qtip_v6_rms_ratio_scale_is_one_encode_without_feedback(tmp_path: Path) -> None:
    source, control, _old_tlut, _hessian = _cell_inputs(tmp_path)
    codebook = tmp_path / "pr31-codebook.npy"
    np.save(codebook, banana_v1_gaussian_codebook(), allow_pickle=False)

    receipt = banana.build_qtip_native_cell(
        source,
        control,
        codebook,
        tmp_path / "v6-rms-cell",
        bpw=2.5,
        intended_basis_sha256="d" * 64,
        observed_basis_sha256="d" * 64,
        backend="reference",
        ldlq_scale_semantics="rms_ratio",
        feedback_mode="off",
    )

    assert receipt["optimization"]["scale_semantics"] == "rms_ratio"
    assert receipt["optimization"]["scale_factors"] == [1.0]
    assert receipt["optimization"]["selected_scale"] != 1.0
    assert receipt["optimization"]["feedback_nonzero_count"] == 0


def test_qtip_v6_b10_refines_the_fixed_path_to_least_squares_scale(tmp_path: Path) -> None:
    source, control, _old_tlut, _hessian = _cell_inputs(tmp_path)
    codebook = tmp_path / "pr31-codebook.npy"
    table = banana_v1_gaussian_codebook()
    np.save(codebook, table, allow_pickle=False)

    receipt = banana.build_qtip_native_cell(
        source,
        control,
        codebook,
        tmp_path / "v6-b10-ls-scale",
        bpw=2.5,
        intended_basis_sha256="5" * 64,
        observed_basis_sha256="5" * 64,
        backend="reference",
    )

    compact, _ = native_qtip_api._load_control(control)
    normalized = native_qtip_api._to_normalized_blocks(
        np.load(source, allow_pickle=False), compact
    )
    packed = np.load(tmp_path / "v6-b10-ls-scale" / "codes.npy", allow_pickle=False)
    unit = decode_native_v4(
        packed,
        np.ones(len(packed), dtype=np.float32),
        positions=256,
        tlut=table,
        geometry=native_v4_geometry(2.5),
    ).reshape(normalized.shape)
    expected = float(
        np.vdot(unit.astype(np.float64), normalized.astype(np.float64))
        / np.vdot(unit.astype(np.float64), unit.astype(np.float64))
    )

    assert receipt["optimization"]["scale_refinement"] == "least_squares_fixed_path"
    assert receipt["optimization"]["selected_scale"] == pytest.approx(expected)


def test_qtip_v6_b10_keeps_the_physical_hessian_refinement_on_the_fixed_path(
    tmp_path: Path,
) -> None:
    source, control, _old_tlut, hessian = _cell_inputs(tmp_path)
    codebook = tmp_path / "pr31-codebook.npy"
    table = banana_v1_gaussian_codebook()
    np.save(codebook, table, allow_pickle=False)

    receipt = banana.build_qtip_native_cell(
        source,
        control,
        codebook,
        tmp_path / "v6-b10-hessian-scale",
        bpw=2.5,
        intended_basis_sha256="7" * 64,
        observed_basis_sha256="7" * 64,
        backend="reference",
        hessian=hessian,
        feedback_mode="off",
    )

    compact, _ = native_qtip_api._load_control(control)
    normalized = native_qtip_api._to_normalized_blocks(
        np.load(source, allow_pickle=False), compact
    )
    packed = np.load(tmp_path / "v6-b10-hessian-scale" / "codes.npy", allow_pickle=False)
    unit = decode_native_v4(
        packed,
        np.ones(len(packed), dtype=np.float32),
        positions=256,
        tlut=table,
        geometry=native_v4_geometry(2.5),
    ).reshape(normalized.shape)
    target_matrix = np.load(source, allow_pickle=False).astype(np.float64)
    unit_matrix = native_qtip_api._from_normalized_blocks(unit, compact).astype(
        np.float64
    )
    hessian_value = np.load(hessian, allow_pickle=False).astype(np.float64)
    expected = float(
        np.sum((unit_matrix @ hessian_value) * target_matrix, dtype=np.float64)
        / np.sum((unit_matrix @ hessian_value) * unit_matrix, dtype=np.float64)
    )

    assert receipt["optimization"]["scale_refinement"] == (
        "physical_hessian_weighted_least_squares_fixed_path"
    )
    assert receipt["optimization"]["selected_scale"] == pytest.approx(expected)
    assert "path_refinement_candidates" not in receipt["optimization"]
    assert "path_refinement_changed_codes" not in receipt["optimization"]


def test_qtip_v6_b10_fits_a_symmetric_monotone_rate_specific_pr31_codebook() -> None:
    prior = banana_v1_gaussian_codebook()
    level_ids = np.repeat(np.arange(1024, dtype=np.int32), 3)
    magnitudes = np.linspace(0.01, 2.0, 512, dtype=np.float64) ** 1.35
    target_table = np.concatenate((-magnitudes[::-1], magnitudes))
    target_table /= np.sqrt(np.mean(target_table * target_table, dtype=np.float64))
    targets = (np.float64(2.3) * target_table[level_ids]).reshape(32, 96)

    fitted = native_qtip_api.fit_compact_pr31_codebook(
        level_ids.reshape(32, 96),
        targets,
        prior=prior,
    )

    assert fitted.dtype == np.float16
    assert fitted.shape == (1024,)
    np.testing.assert_array_equal(fitted, -fitted[::-1])
    assert np.all(np.diff(fitted.astype(np.float32)) >= 0)
    assert np.sqrt(np.mean(fitted.astype(np.float64) ** 2)) == pytest.approx(
        1.0, abs=5e-4
    )

    def fixed_assignment_sse(table: np.ndarray) -> float:
        decoded = table.astype(np.float64)[level_ids]
        scale = float(np.vdot(decoded, targets.reshape(-1)) / np.vdot(decoded, decoded))
        delta = decoded * scale - targets.reshape(-1)
        return float(np.vdot(delta, delta))

    assert fixed_assignment_sse(fitted) < fixed_assignment_sse(prior) * 1e-3


def test_qtip_v6_b10_exposes_the_frozen_disjoint_bank_rate_specific_codebook() -> None:
    table = native_qtip_api.periodic_b10_rate_specific_codebook()

    assert table.dtype == np.float16
    assert table.shape == (1024,)
    assert hashlib.sha256(table.tobytes()).hexdigest() == (
        "3ca6e37fbdbc2b5bbc69ecf51e47eda09d69f5f68e85c36e3e69b02e1b14f634"
    )
    np.testing.assert_array_equal(table, -table[::-1])
    assert np.all(np.diff(table.astype(np.float32)) >= 0)


def test_qtip_v6_rms_ratio_supports_single_pass_reverse_16_feedback(
    tmp_path: Path,
) -> None:
    source, control, _old_tlut, hessian = _cell_inputs(tmp_path)
    codebook = tmp_path / "pr31-codebook.npy"
    np.save(codebook, banana_v1_gaussian_codebook(), allow_pickle=False)

    receipt = banana.build_qtip_native_cell(
        source,
        control,
        codebook,
        tmp_path / "v6-rms-feedback-cell",
        bpw=2.0,
        intended_basis_sha256="e" * 64,
        observed_basis_sha256="e" * 64,
        backend="reference",
        hessian=hessian,
        scale_factors=(1.0,),
        ldlq_scale_semantics="rms_ratio",
        feedback_mode="reverse_16",
    )

    assert receipt["optimization"]["scale_semantics"] == "rms_ratio"
    assert receipt["optimization"]["scale_factors"] == [1.0]
    assert receipt["optimization"]["selected_scale"] != 1.0
    assert receipt["optimization"]["feedback_nonzero_count"] > 0


def test_qtip_v6_feedback_uses_robust_hessian_diagonal_loading() -> None:
    assert native_qtip_api._hessian_regularization_sigma(
        codec_version="v6", hadamard_block=None
    ) == pytest.approx(1.0)
    assert native_qtip_api._hessian_regularization_sigma(
        codec_version="v4", hadamard_block=None
    ) == pytest.approx(0.01)
    assert native_qtip_api._hessian_regularization_sigma(
        codec_version="v4", hadamard_block=128
    ) == pytest.approx(0.025)


def test_qtip_native_build_defaults_v6_and_keeps_v4_explicit(tmp_path: Path) -> None:
    source, control, old_tlut, _hessian = _cell_inputs(tmp_path)
    codebook = tmp_path / "pr31-codebook.npy"
    np.save(codebook, banana_v1_gaussian_codebook(), allow_pickle=False)
    common = {
        "bpw": 2.5,
        "intended_basis_sha256": "e" * 64,
        "observed_basis_sha256": "e" * 64,
        "backend": "reference",
    }

    default_receipt = banana.build_qtip_native_cell(
        source,
        control,
        codebook,
        tmp_path / "default-v6",
        **common,
    )
    compatibility_receipt = banana.build_qtip_native_cell(
        source,
        control,
        old_tlut,
        tmp_path / "explicit-v4",
        codec_version="v4",
        **common,
    )

    assert default_receipt["codec_version"] == "v6"
    assert default_receipt["provider"] == "qtip-native-v6@2.50"
    assert default_receipt["tlut"]["identity"] == "pr31-affine-gaussian-edge-v1"
    assert default_receipt["optimization"]["scale_semantics"] == "rms_ratio"
    assert default_receipt["optimization"]["cyclic_warmup_cycles"] == 1
    assert compatibility_receipt["codec_version"] == "v4"
    assert compatibility_receipt["provider"] == "qtip-native-v4@2.50"
    assert compatibility_receipt["tlut"]["identity"] == "q9-v2-v4"
    assert compatibility_receipt["optimization"]["scale_semantics"] == "absolute_unit"
    assert compatibility_receipt["optimization"]["cyclic_warmup_cycles"] == 1


def test_qtip_v6_backpack_provider_generates_v6_cell(tmp_path: Path) -> None:
    source, control, _old_tlut, _hessian = _cell_inputs(tmp_path)
    control_root = tmp_path / "controls"
    control_root.mkdir()
    control.rename(control_root / "cell0.npz")
    codebook = tmp_path / "pr31-codebook.npy"
    np.save(codebook, banana_v1_gaussian_codebook(), allow_pickle=False)
    weights = np.load(source, allow_pickle=False)
    tier = {
        "id": "native_v6_b8",
        "family": "qtip_native_v4",
        "provider": "qtip-native-v6@2.00",
        "codec_version": "v6",
        "bpw": 2.0,
        "B": 8,
        "phase_widths": [2, 2, 2, 2],
        "lut_identity": "pr31-affine-gaussian-edge-v1",
        "lut_multiplier": 48917,
        "lut_offset": 50631,
        "control_root": str(control_root),
        "tlut": str(codebook),
        "basis_sha256": "9" * 64,
        "backend": "reference",
        "solve_batch": 2048,
        "decode_batch": 2048,
        "decode_repeats": 1,
    }

    candidate = generate_backpack_candidate(
        tmp_path / "run",
        tier=tier,
        cell={
            "cell_id": "cell0",
            "layer": 0,
            "projection": "down",
            "expert_ids": [0],
            "weights": weights.reshape(-1),
        },
    )
    cell_receipt = json.loads(Path(candidate["native_v4_cell_receipt"]).read_text())

    assert cell_receipt["codec_version"] == "v6"
    assert cell_receipt["provider"] == "qtip-native-v6@2.00"
    assert cell_receipt["optimization"]["scale_semantics"] == "rms_ratio"
    assert candidate["codec_version"] == "v6"
    assert candidate["provider"] == "qtip-native-v6@2.00"
    assert candidate["algorithm"] == "qtip-native-v6"


def test_backpack_plan_accepts_native_v6_declaration(tmp_path: Path) -> None:
    codebook = tmp_path / "pr31-codebook.npy"
    np.save(codebook, banana_v1_gaussian_codebook(), allow_pickle=False)
    declaration = {
        "id": "native_v6_b8",
        "family": "qtip_native_v4",
        "provider": "qtip-native-v6@2.00",
        "codec_version": "v6",
        "bpw": 2.0,
        "B": 8,
        "phase_widths": [2, 2, 2, 2],
        "lut_identity": "pr31-affine-gaussian-edge-v1",
        "lut_multiplier": 48917,
        "lut_offset": 50631,
        "scale_semantics": "rms_ratio",
        "control_root": str(tmp_path / "controls"),
        "tlut": str(codebook),
        "tlut_sha256": hashlib.sha256(codebook.read_bytes()).hexdigest(),
        "basis_sha256": "8" * 64,
        "backend": "reference",
    }

    plan = banana.BackpackPlan.from_mapping(
        {
            "schema": "banana-smasher-backpack-plan-v1",
            "model": {"root": str(tmp_path / "model"), "revision": "v6-smoke"},
            "target": {"exact_bytes": 1},
            "tiers": [declaration],
            "anchor": {"bank": str(tmp_path / "anchor.npz"), "teacher": "model"},
            "prediction": {
                "class_caps": {
                    name: 1.0
                    for name in (
                        "agentic",
                        "chat",
                        "code",
                        "multilingual",
                        "prose",
                        "reasoning",
                    )
                }
            },
            "repair": {"method": "none"},
            "output": {
                "pack": str(tmp_path / "pack"),
                "model_id": "v6-smoke",
                "instance_id": "v6-smoke-v1",
            },
        }
    )

    parsed = plan.tiers[0]
    assert parsed["codec_version"] == "v6"
    assert parsed["provider"] == "qtip-native-v6@2.00"
    assert parsed["scale_semantics"] == "rms_ratio"
    assert parsed["activation_artifacts"][0]["id"].startswith("qtip-native-v6-tlut-")


def test_qtip_native_cli_defaults_v6_and_accepts_explicit_v4(tmp_path: Path) -> None:
    source, control, tlut, _hessian = _cell_inputs(tmp_path)
    common = [
        "qtip-native",
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
        "f" * 64,
        "--observed-basis-sha256",
        "f" * 64,
    ]

    default_args = _parser().parse_args(common)
    compatibility_args = _parser().parse_args([*common, "--codec-version", "v4"])

    assert default_args.codec_version == "v6"
    assert compatibility_args.codec_version == "v4"


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
