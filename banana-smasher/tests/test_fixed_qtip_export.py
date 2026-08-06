from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from banana_smasher.cli import main
from banana_smasher.contract import verify_pack
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
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "canonical_packed_sha256": _tensor_sha256(trellis),
    }


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
    for expert in range(256):
        for projection in ("down", "fused13"):
            k = 2 if (expert + (projection == "fused13")) % 2 == 0 else 3
            source, binding = templates[(projection, k)]
            staged = member_root / "L000" / f"E{expert:03d}_{projection}" / "QTIP_UNIT.pt"
            staged.parent.mkdir(parents=True)
            os.link(source, staged)
            rows.append(
                {
                    "layer": 0,
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
                "physical_payload": {
                    "artifact_count": 512,
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
        assert set(state.payloads) == {"qtip25k2", "qtip25k3"}
    manifest = json.loads((output / "BANANA_PACK_MANIFEST.json").read_text())
    assert manifest["fixed_assignment"]["repair_status"] == "absent"
    assert manifest["fixed_assignment"]["members_manifest_sha256"] == members_sha
    assert manifest["fixed_assignment"]["codec"] == {
        "public_name": "QTIP2.5-AVG-MEMBER",
        "machine_id": "qtip25_avg_member",
        "codec_form": "avg_member_50_50",
        "requested_id": "qtip@2.50",
        "compatibility_alias": True,
        "runtime_family": "qtip2",
        "runtime_payload_families": ["qtip2", "qtip3"],
        "nominal_code_bpw": "2.5",
        "rate_num": 5,
        "rate_den": 2,
    }
    assert "repair" not in manifest
