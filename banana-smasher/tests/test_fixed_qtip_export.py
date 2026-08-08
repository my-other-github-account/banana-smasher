from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from banana_smasher.cli import main
from banana_smasher.contract import PackValidationError, verify_pack
from banana_smasher.fixed_qtip_export import (
    _validate_contract,
    account_fixed_qtip_wire,
    build_periodic_parity_members,
)
from banana_smasher_plugin.native_planes import NativePlaneLayer, NativePlanePack


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _write_model(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeForCausalLM"],
                "hidden_size": 4,
                "moe_intermediate_size": 2,
                "quantization_config": {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "scale_fmt": "ue8m0",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    (root / "generation_config.json").write_text("{}")


def _write_unit(path: Path, *, k: int, projection: str) -> dict[str, object]:
    input_width, output_width = ((3, 4) if projection == "fused13" else (2, 3))
    trellis = torch.arange(4 * k, dtype=torch.int16).reshape(4, k)
    payload = {
        "schema": "ds4-qtip-hyb-bounded36-unit-v1",
        "geometry": {
            "L": 16,
            "K": k,
            "V": 2,
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
            "td_x": 16,
            "td_y": 16,
        },
        "shape": [output_width, input_width],
        "trellis": trellis,
        "SU": torch.arange(input_width, dtype=torch.float16) + 1,
        "SV": torch.arange(output_width, dtype=torch.float16) + 1,
        "Wscale": torch.tensor(1.0, dtype=torch.float32),
        "tlut": torch.full((512, 2), float(k), dtype=torch.float32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "canonical_packed_sha256": _tensor_sha256(trellis),
    }


def test_periodic_parity_builder_selects_only_predeclared_two_cell_orientation(
    tmp_path: Path,
    capsys,
) -> None:
    manifests = {}
    for k in (2, 3):
        path = tmp_path / f"qtip{k}.jsonl"
        rows = [
            {
                "layer": 0,
                "expert": expert,
                "projection": projection,
                "tier": f"qtip{k}",
                "geometry": {"L": 16, "K": k, "V": 2},
                "canonical_packed_sha256": str(k) * 64,
                "artifact": {
                    "path": f"/sealed/qtip{k}/E{expert:03d}_{projection}.pt",
                    "bytes": k,
                    "sha256": str(k) * 64,
                },
            }
            for expert in (0, 1)
            for projection in ("down", "fused13")
        ]
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
        )
        manifests[k] = path

    output = tmp_path / "periodic-parity.jsonl"
    receipt = build_periodic_parity_members(
        qtip2_members=manifests[2],
        qtip2_members_sha256=_sha256(manifests[2]),
        qtip3_members=manifests[3],
        qtip3_members_sha256=_sha256(manifests[3]),
        output=output,
    )
    selected = [json.loads(line) for line in output.read_text().splitlines()]

    assert [(row["expert"], row["geometry"]["K"]) for row in selected] == [
        (0, 2),
        (0, 2),
        (1, 3),
        (1, 3),
    ]
    assert {row["tier"] for row in selected} == {"qtip@2.50"}
    assert {(row["expert"], row["source_tier"]) for row in selected} == {
        (0, "qtip2"),
        (1, "qtip3"),
    }
    assert receipt["qtip2_experts"] == 1
    assert receipt["qtip3_experts"] == 1
    assert receipt["nominal_code_bpw"] == 2.5
    assert receipt["members_sha256"] == _sha256(output)

    cli_output = tmp_path / "periodic-parity-cli.jsonl"
    assert (
        main(
            [
                "build-periodic-parity-members",
                "--qtip2-members",
                str(manifests[2]),
                "--qtip2-members-sha256",
                _sha256(manifests[2]),
                "--qtip3-members",
                str(manifests[3]),
                "--qtip3-members-sha256",
                _sha256(manifests[3]),
                "--output",
                str(cli_output),
            ]
        )
        == 0
    )
    cli_receipt = json.loads(capsys.readouterr().out)
    assert cli_receipt["members_sha256"] == _sha256(cli_output)
    assert cli_output.read_bytes() == output.read_bytes()


def test_periodic_parity_builder_rejects_projection_incomplete_payload_rows(
    tmp_path: Path,
) -> None:
    manifests = {}
    for k in (2, 3):
        path = tmp_path / f"qtip{k}.jsonl"
        rows = [
            {
                "layer": 0,
                "expert": expert,
                "projection": projection,
                "tier": f"qtip{k}",
                "geometry": {"L": 16, "K": k, "V": 2},
                "canonical_packed_sha256": str(k) * 64,
                "artifact": {
                    "path": f"/sealed/qtip{k}/E{expert:03d}_{projection}.pt",
                    "bytes": k,
                    "sha256": str(k) * 64,
                },
            }
            for expert, projections in ((0, ("down",)), (1, ("down", "fused13")))
            for projection in projections
        ]
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            )
        )
        manifests[k] = path

    with pytest.raises(PackValidationError, match="projection-complete"):
        build_periodic_parity_members(
            qtip2_members=manifests[2],
            qtip2_members_sha256=_sha256(manifests[2]),
            qtip3_members=manifests[3],
            qtip3_members_sha256=_sha256(manifests[3]),
            output=tmp_path / "periodic-parity.jsonl",
        )


def test_existing_projection_mixed_fixed_qtip_contract_remains_supported() -> None:
    members_sha = "a" * 64
    rows = [
        {
            "layer": 0,
            "expert": expert,
            "projection": projection,
            "tier": "qtip@2.50",
            "geometry": {
                "L": 16,
                "K": 2 if (expert + (projection == "fused13")) % 2 == 0 else 3,
                "V": 2,
            },
            "canonical_packed_sha256": "b" * 64,
            "artifact": {
                "path": "/sealed/QTIP_UNIT.pt",
                "bytes": 1,
                "sha256": "c" * 64,
            },
        }
        for expert in range(256)
        for projection in ("down", "fused13")
    ]
    grouped, assignment = _validate_contract(
        rows,
        {
            "schema": "banana-smasher-qtip25-pack-admission-v1",
            "status": "PASS",
            "tier": "qtip@2.50",
            "repair_status": "absent",
            "physical_payload": {
                "artifact_count": len(rows),
                "members_manifest_sha256": members_sha,
            },
        },
        members_sha256=members_sha,
    )

    assert assignment is None
    assert len(grouped) == 4
    assert grouped[(0, "down", 2)][0]["expert"] == 0
    assert grouped[(0, "fused13", 3)][0]["expert"] == 0


def test_fixed_qtip_manifest_survives_cli_export_verify_and_runtime_load(
    tmp_path: Path,
) -> None:
    member_root = tmp_path / "members"
    templates = {}
    for projection in ("down", "fused13"):
        for k in (2, 3):
            path = tmp_path / "templates" / f"{projection}-k{k}.pt"
            templates[(projection, k)] = (path, _write_unit(path, k=k, projection=projection))

    rows = []
    for layer_index in (0, 1):
        for expert in range(256):
            for projection in ("down", "fused13"):
                k = 2 if expert % 2 == 0 else 3
                source, binding = templates[(projection, k)]
                staged = (
                    member_root
                    / f"L{layer_index:03d}"
                    / f"E{expert:03d}_{projection}"
                    / "QTIP_UNIT.pt"
                )
                staged.parent.mkdir(parents=True)
                os.link(source, staged)
                rows.append(
                    {
                        "layer": layer_index,
                        "expert": expert,
                        "projection": projection,
                        "tier": "qtip@2.50",
                        "geometry": {"L": 16, "K": k, "V": 2},
                        "canonical_packed_sha256": binding["canonical_packed_sha256"],
                        "artifact": {
                            "path": "/sealed/source/QTIP_UNIT.pt",
                            "ssh": "sealed-source",
                            "bytes": binding["bytes"],
                            "sha256": binding["sha256"],
                        },
                    }
                )
    members = tmp_path / "QTIP25_MEMBERS.jsonl"
    members.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    members_sha = _sha256(members)
    admission = tmp_path / "QTIP25_PACK_ADMISSION.json"
    admission.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip25-pack-admission-v1",
                "status": "PASS",
                "tier": "qtip@2.50",
                "repair_status": "absent",
                "assignment_rule": {
                    "schema": "banana-smasher-periodic-qtip-parity-v1",
                    "global_expert_ordinal": {
                        "even": "qtip2",
                        "odd": "qtip3",
                    },
                    "assignment_payload_bytes": 0,
                },
                "physical_payload": {
                    "artifact_count": 1024,
                    "members_manifest_sha256": members_sha,
                },
            },
            sort_keys=True,
        )
    )
    model = tmp_path / "model"
    _write_model(model)
    output = tmp_path / "pack"

    assert (
        main(
            [
                "export",
                "--source-root",
                str(member_root),
                "--fixed-members-manifest",
                str(members),
                "--fixed-members-manifest-sha256",
                members_sha,
                "--fixed-pack-admission",
                str(admission),
                "--fixed-pack-admission-sha256",
                _sha256(admission),
                "--serving-model-root",
                str(model),
                "--runtime-floor-bytes",
                "0",
                "--output",
                str(output),
                "--model-id",
                "test/fixed-qtip25",
                "--instance-id",
                "fixed-qtip25-test",
            ]
        )
        == 0
    )
    receipt = verify_pack(output)
    assert receipt["status"] == "PASS"
    pack = NativePlanePack.from_model_root(output)
    layer = NativePlaneLayer(
        pack,
        0,
        device="cpu",
        dispatch=lambda **kwargs: torch.zeros(
            (kwargs["x"].shape[0], kwargs["state"].output_width),
            dtype=torch.float32,
        ),
    )
    for projection in ("down", "fused13"):
        state = layer.state(projection)
        assert set(state.tiers) == {"qtip25k2", "qtip25k3"}
        assert torch.bincount(state.families.to(torch.int64), minlength=4).tolist() == [128, 128, 0, 0]
        assert state.families.tolist() == [expert % 2 for expert in range(256)]
        assert set(state.payloads) == {"qtip25k2", "qtip25k3"}
        assert all("expert_ids" not in payload for payload in state.payloads.values())
        assert torch.unique(state.lut2[:, 1]).tolist() == [2.0]
        assert torch.unique(state.lut3[:, 1]).tolist() == [3.0]
    manifest = json.loads((output / "BANANA_PACK_MANIFEST.json").read_text())
    assert manifest["fixed_assignment"]["repair_status"] == "absent"
    assert manifest["fixed_assignment"]["members_manifest_sha256"] == members_sha
    assert manifest["fixed_assignment"]["assignment_rule"] == {
        "schema": "banana-smasher-periodic-qtip-parity-v1",
        "global_expert_ordinal": {"even": "qtip2", "odd": "qtip3"},
        "assignment_payload_bytes": 0,
        "qtip2_experts": 256,
        "qtip3_experts": 256,
        "nominal_code_bpw": 2.5,
    }
    for layer_index in (0, 1):
        meta = json.loads(
            (output / f"planes/layer_{layer_index:03d}.meta.json").read_text()
        )
        assert not {
            "tier13",
            "slot13",
            "family13",
            "tier2",
            "slot2",
            "family2",
        } & meta.keys()
        for projection in ("down", "fused13"):
            route = manifest["selected_payloads"]["layers"][str(layer_index)][
                projection
            ]
            assert set(route) == {"assignment_rule", "payloads"}
            assert route["assignment_rule"] == manifest["fixed_assignment"][
                "assignment_rule"
            ]
    assert "repair" not in manifest
    accounting = account_fixed_qtip_wire(output)
    assert accounting["complete_wire_bytes"] == sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    assert accounting["assignment_payload_bytes"] == 0
    assert accounting["routing_index_file_bytes"] == 0
    assert accounting["provider_rule_metadata_bytes"] == (
        output / "planes/layer_000.meta.json"
    ).stat().st_size + (output / "planes/layer_001.meta.json").stat().st_size
    assert accounting["qtip2_experts"] == 256
    assert accounting["qtip3_experts"] == 256
    assert accounting["nominal_code_bpw"] == 2.5
    assert accounting["complete_wire_bytes"] == sum(
        accounting[key]
        for key in (
            "tensor_file_bytes",
            "provider_rule_metadata_bytes",
            "provenance_bytes",
            "serving_and_pack_bytes",
        )
    )
