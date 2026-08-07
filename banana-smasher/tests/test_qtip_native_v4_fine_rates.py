from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from banana_smasher import (
    BackpackPlan,
    backpack_provider_from_declaration,
    build_qtip_native_v4_anchor_set,
    build_qtip_native_v4_cell,
    generate_backpack_candidate,
    export_pack,
    verify_backpack_candidate,
)
from banana_smasher.backpack import _materialize_native_v4_plane_source
from banana_smasher.qtip1 import gaussian_tlut
from banana_smasher.qtip25_native_v4 import (
    decode_native_v4,
    ldlq_native_v4_matrix,
    native_v4_lower_from_hessian,
    native_v4_geometry,
    solve_native_v4,
)


def test_native_v4_geometry_supports_homogeneous_quarter_rates() -> None:
    rng = np.random.default_rng(41)
    tlut = gaussian_tlut(bits=9, columns=2)

    for bpw, transition_bits, packed_bytes in (
        (1.75, 7, 7),
        (2.25, 9, 9),
        (2.75, 11, 11),
    ):
        geometry = native_v4_geometry(bpw)
        target = rng.normal(size=(1, 8, 4)).astype(np.float32)
        encoded = solve_native_v4(target, tlut=tlut, geometry=geometry)
        decoded = decode_native_v4(
            encoded.packed,
            encoded.scales,
            positions=32,
            tlut=tlut,
            geometry=geometry,
        )

        assert geometry.B == transition_bits
        assert geometry.rate_num / geometry.rate_den == bpw
        assert encoded.code_bpw == bpw
        assert encoded.packed.shape == (1, packed_bytes)
        assert decoded.shape == (1, 32)
        assert np.isfinite(decoded).all()


def test_native_v4_scale_search_uses_nonzero_reverse_16_ldlq() -> None:
    rng = np.random.default_rng(47)
    source = rng.normal(size=(32, 32)).astype(np.float32)
    calibration = rng.normal(size=(32, 48)).astype(np.float32)
    hessian = calibration @ calibration.T + np.eye(32, dtype=np.float32)
    lower = native_v4_lower_from_hessian(hessian)
    unregularized_lower = native_v4_lower_from_hessian(
        hessian, regularization_sigma=0.0
    )
    tlut = gaussian_tlut(bits=9, columns=2)

    fixed = ldlq_native_v4_matrix(
        source,
        lower,
        tlut=tlut,
        scale_factors=(1.0,),
        scale_semantics="relative_search",
    )
    searched = ldlq_native_v4_matrix(
        source,
        lower,
        tlut=tlut,
        scale_factors=(0.9, 1.0, 1.1),
        scale_semantics="relative_search",
    )
    decoded_tiles = decode_native_v4(
        searched.packed,
        searched.scales,
        positions=256,
        tlut=tlut,
    ).reshape(2, 2, 16, 16)
    decoded = decoded_tiles.transpose(0, 2, 1, 3).reshape(32, 32)

    assert searched.feedback_nonzero_count == np.count_nonzero(lower)
    assert searched.feedback_nonzero_count > 0
    assert not np.array_equal(lower, unregularized_lower)
    assert searched.scale_factors == (0.9, 1.0, 1.1)
    assert searched.packed.shape == (4, 80)
    assert searched.distortion <= fixed.distortion
    assert np.unique(searched.scales).tolist() == [searched.scales[0]]
    assert searched.scales[0] > 0
    np.testing.assert_allclose(decoded, searched.decoded, rtol=0.0, atol=0.0)


def test_public_cell_builder_accepts_arbitrary_quarter_rates(tmp_path) -> None:
    rng = np.random.default_rng(43)
    source = tmp_path / "source.npy"
    control = tmp_path / "control.npz"
    tlut = tmp_path / "tlut.npy"
    np.save(source, rng.normal(size=(16, 16)).astype(np.float32), allow_pickle=False)
    np.savez(
        control,
        SU=np.ones(16, dtype=np.float16),
        SV=np.ones(16, dtype=np.float16),
        Wscale=np.asarray(1.0, dtype=np.float32),
        shape=np.asarray([16, 16], dtype=np.int64),
    )
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)

    for bpw, transition_bits, code_bytes in (
        (1.75, 7, 56),
        (2.25, 9, 72),
        (2.75, 11, 88),
    ):
        receipt = build_qtip_native_v4_cell(
            source,
            control,
            tlut,
            tmp_path / f"candidate-{bpw}",
            bpw=bpw,
            intended_basis_sha256="a" * 64,
            observed_basis_sha256="a" * 64,
            backend="reference",
        )

        assert receipt["status"] == "PASS"
        assert receipt["geometry"]["B"] == transition_bits
        assert receipt["accounting"]["exact_code_bpw"] == bpw
        assert receipt["accounting"]["code_data_bytes"] == code_bytes


def test_public_cell_builder_scales_code_bytes_with_physical_cell_size(tmp_path) -> None:
    rng = np.random.default_rng(45)
    tlut = tmp_path / "tlut.npy"
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)

    for rows, columns in ((16, 16), (32, 64)):
        source = tmp_path / f"source-{rows}x{columns}.npy"
        control = tmp_path / f"control-{rows}x{columns}.npz"
        np.save(
            source,
            rng.normal(size=(rows, columns)).astype(np.float32),
            allow_pickle=False,
        )
        np.savez(
            control,
            SU=np.ones(columns, dtype=np.float16),
            SV=np.ones(rows, dtype=np.float16),
            Wscale=np.asarray(1.0, dtype=np.float32),
            shape=np.asarray([rows, columns], dtype=np.int64),
        )
        receipt = build_qtip_native_v4_cell(
            source,
            control,
            tlut,
            tmp_path / f"candidate-{rows}x{columns}",
            bpw=2.25,
            intended_basis_sha256="e" * 64,
            observed_basis_sha256="e" * 64,
            backend="reference",
        )
        weights = rows * columns

        assert receipt["accounting"]["weights"] == weights
        assert receipt["accounting"]["code_data_bytes"] == weights * 9 // 32
        assert np.load(receipt["artifacts"]["decoded"]["path"]).shape == (rows, columns)


def test_public_anchor_set_builds_an_arbitrary_declared_rate_set(tmp_path) -> None:
    rng = np.random.default_rng(47)
    source = tmp_path / "source.npy"
    control = tmp_path / "control.npz"
    tlut = tmp_path / "tlut.npy"
    bank = tmp_path / "anchor64.npz"
    np.save(source, rng.normal(size=(16, 16)).astype(np.float32), allow_pickle=False)
    np.savez(
        control,
        SU=np.ones(16, dtype=np.float16),
        SV=np.ones(16, dtype=np.float16),
        Wscale=np.asarray(1.0, dtype=np.float32),
        shape=np.asarray([16, 16], dtype=np.int64),
    )
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)
    classes = np.asarray(
        [("agentic", "chat", "code", "multilingual", "prose", "reasoning")[i % 6] for i in range(64)]
    )
    np.savez(
        bank,
        features=rng.normal(size=(64, 256)).astype(np.float32),
        classes=classes,
    )

    result = build_qtip_native_v4_anchor_set(
        source,
        control,
        tlut,
        tmp_path / "anchor-set",
        bpws=(1.75, 2.25, 2.75),
        anchor_bank=bank,
        teacher=source,
        intended_basis_sha256="b" * 64,
        observed_basis_sha256="b" * 64,
        backend="reference",
    )

    assert result["status"] == "PASS"
    assert [row["bpw"] for row in result["tiers"]] == [1.75, 2.25, 2.75]
    assert [row["geometry"]["B"] for row in result["tiers"]] == [7, 9, 11]
    assert all(row["anchor"]["windows"] == 64 for row in result["tiers"])


def test_backpack_provider_is_declaration_driven_for_any_quarter_rate() -> None:
    for bpw, transition_bits in ((1.25, 5), (1.75, 7), (2.25, 9), (2.75, 11), (3.75, 15)):
        provider = backpack_provider_from_declaration(
            {"family": "qtip_native_v4", "bpw": bpw}
        )

        assert provider.provider_id == f"qtip-native-v4@{bpw:.2f}"
        assert provider.kind == "qtip_native_v4"
        assert provider.runtime_family == "qtip_native_v4"
        assert provider.rate_num / provider.rate_den == bpw
        assert provider.transition_bits == transition_bits


def test_backpack_candidate_generation_executes_declared_native_v4_rate(tmp_path) -> None:
    rng = np.random.default_rng(53)
    cell_id = "layer0-down"
    control_root = tmp_path / "controls"
    control_root.mkdir()
    np.savez(
        control_root / f"{cell_id}.npz",
        SU=np.ones(16, dtype=np.float16),
        SV=np.ones(16, dtype=np.float16),
        Wscale=np.asarray(1.0, dtype=np.float32),
        shape=np.asarray([16, 16], dtype=np.int64),
    )
    tlut = tmp_path / "tlut.npy"
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)
    tlut_sha256 = hashlib.sha256(tlut.read_bytes()).hexdigest()
    tier = {
        "id": "native-v4-225",
        "family": "qtip_native_v4",
        "provider": "qtip-native-v4@2.25",
        "bpw": 2.25,
        "backend": "reference",
        "control_root": str(control_root),
        "tlut": str(tlut),
        "basis_sha256": "c" * 64,
        "solve_batch": 1,
        "decode_batch": 1,
        "decode_repeats": 1,
        "activation_artifacts": [
            {
                "id": f"qtip-native-v4-tlut-{tlut_sha256[:16]}",
                "bytes": tlut.stat().st_size,
                "sha256": tlut_sha256,
                "path": str(tlut),
            }
        ],
    }
    cell = {
        "cell_id": cell_id,
        "layer": 0,
        "projection": "down",
        "expert_ids": [0],
        "weights": rng.normal(size=256).astype(np.float32),
    }

    receipt = generate_backpack_candidate(tmp_path / "run", tier=tier, cell=cell)

    assert receipt["status"] == "PASS"
    assert receipt["bpw"] == 2.25
    assert receipt["geometry"]["B"] == 9
    assert receipt["wire"]["bytes"] == 72
    assert verify_backpack_candidate(receipt, tier=tier, cell=cell)


def test_backpack_plan_accepts_an_arbitrary_native_v4_anchor_menu(tmp_path) -> None:
    controls = tmp_path / "controls"
    controls.mkdir()
    tlut = tmp_path / "tlut.npy"
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)
    tlut_sha256 = hashlib.sha256(tlut.read_bytes()).hexdigest()
    rates = (1.75, 2.25, 2.75)
    plan = BackpackPlan.from_mapping(
        {
            "schema": "banana-smasher-backpack-plan-v1",
            "model": {"root": "model", "revision": "fixture"},
            "target": {"whole_model_bpw": 2.25},
            "tiers": [
                {
                    "id": f"native-v4-{int(bpw * 100)}",
                    "family": "qtip_native_v4",
                    "bpw": bpw,
                    "backend": "reference",
                    "control_root": str(controls),
                    "tlut": str(tlut),
                    "tlut_sha256": tlut_sha256,
                    "basis_sha256": "d" * 64,
                }
                for bpw in rates
            ],
            "anchor": {"bank": "anchor64.npz", "teacher": "model"},
            "prediction": {
                "class_caps": {
                    name: 1.0
                    for name in ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
                }
            },
            "repair": {"method": "none"},
            "output": {
                "pack": "pack",
                "model_id": "fixture/model",
                "instance_id": "native-v4-fine-rates",
            },
        },
        base_dir=tmp_path,
    )

    assert [tier["bpw"] for tier in plan.tiers] == list(rates)
    assert [tier["provider"] for tier in plan.tiers] == [
        "qtip-native-v4@1.75",
        "qtip-native-v4@2.25",
        "qtip-native-v4@2.75",
    ]
    assert len({tier["activation_artifacts"][0]["id"] for tier in plan.tiers}) == 1


def test_native_v4_provider_materializes_declared_geometry(tmp_path) -> None:
    rng = np.random.default_rng(47)
    control_root = tmp_path / "controls"
    control_root.mkdir()
    tlut = tmp_path / "tlut.npy"
    np.save(tlut, gaussian_tlut(bits=9, columns=2), allow_pickle=False)
    tlut_sha256 = hashlib.sha256(tlut.read_bytes()).hexdigest()
    cell = {
        "cell_id": "layer-0-expert-0-down",
        "layer": 0,
        "projection": "down",
        "expert_ids": [0],
        "weights": rng.normal(size=256).astype(np.float32),
    }
    np.savez(
        control_root / f"{cell['cell_id']}.npz",
        SU=np.ones(16, dtype=np.float16),
        SV=np.ones(16, dtype=np.float16),
        Wscale=np.asarray(1.0, dtype=np.float32),
        shape=np.asarray([16, 16], dtype=np.int64),
    )

    for bpw, transition_bits in ((1.75, 7), (2.25, 9), (2.5, 10)):
        tier = {
            "id": f"native-v4-b{transition_bits}",
            "family": "qtip_native_v4",
            "bpw": bpw,
            "backend": "reference",
            "control_root": str(control_root),
            "tlut": str(tlut),
            "tlut_sha256": tlut_sha256,
            "basis_sha256": "e" * 64,
            "solve_batch": 1,
            "decode_batch": 1,
            "decode_repeats": 1,
            "activation_artifacts": [],
        }
        root = tmp_path / f"run-b{transition_bits}"
        receipt = generate_backpack_candidate(root, tier=tier, cell=cell)
        provider = backpack_provider_from_declaration(tier)
        payloads = {}
        provider.materialize(
            payloads,
            tier=tier,
            cell=cell,
            artifact_root=Path(receipt["receipt"]).parent,
        )

        fields = payloads[(0, "qtip_native_v4")]
        assert set(fields) == {
            "codes",
            "SU",
            "SV",
            "Wscale",
            "expert_ids",
            "record_tiers",
            "record_geometry",
            "record_projections",
            "record_boundaries",
        }
        assert fields["record_geometry"][0].tolist() == [[16, transition_bits, 4]]
        assert sum(array.nbytes for array in fields["codes"]) == receipt["wire"]["bytes"]


def test_selected_b7_b9_b10_materialize_as_native_planes(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "n_routed_experts": 256,
                "hidden_size": 16,
                "moe_intermediate_size": 8,
            }
        )
    )
    descriptors = {
        f"native-b{transition_bits}": {
            "id": f"native-b{transition_bits}",
            "family": "qtip_native_v4",
            "bpw": transition_bits / 4,
        }
        for transition_bits in (7, 9, 10)
    }
    partitions = ((0, 86, 7), (86, 171, 9), (171, 256, 10))
    cells = []
    selected = {}
    roots = {}
    for projection in ("fused13", "down"):
        for start, stop, transition_bits in partitions:
            expert_ids = list(range(start, stop))
            cell_id = f"{projection}-b{transition_bits}"
            root = tmp_path / "candidates" / cell_id
            root.mkdir(parents=True)
            rows = len(expert_ids) * 16
            blocks = rows * 16 // 256
            (root / "wire.bin").write_bytes(bytes(blocks * 8 * transition_bits))
            np.save(root / "SU.npy", np.ones(16, dtype=np.float16), allow_pickle=False)
            np.save(root / "SV.npy", np.ones(rows, dtype=np.float16), allow_pickle=False)
            np.save(root / "Wscale.npy", np.asarray(1.0, dtype=np.float32), allow_pickle=False)
            (root / "CELL_RECEIPT.json").write_text(
                json.dumps({"source": {"shape": [rows, 16]}})
            )
            cells.append(
                {
                    "cell_id": cell_id,
                    "layer": 0,
                    "projection": projection,
                    "expert_ids": expert_ids,
                }
            )
            selected[cell_id] = {"tier": f"native-b{transition_bits}"}
            roots[cell_id] = root

    _materialize_native_v4_plane_source(
        source,
        cells=cells,
        selected=selected,
        tier_descriptors=descriptors,
        artifact_roots=roots,
    )
    output = tmp_path / "pack"
    manifest = export_pack(
        source_root=source,
        output=output,
        model_id="fixture/native-v4",
        instance_id="b7-b9-b10",
        runtime_floor_bytes=0,
    )

    assert manifest["source_format"] == "p1016-true-c-native-planes-v1"
    meta = json.loads((output / "planes" / "layer_000.meta.json").read_text())
    assert set(meta["family13"]) == {4, 5, 6}
    assert set(meta["family2"]) == {4, 5, 6}
    assert {
        payload["geometry"]["B"]
        for payloads in meta["payloads"].values()
        for payload in payloads.values()
    } == {7, 9, 10}
