from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import pytest

from banana_smasher.backpack import _build_backpack as build_backpack
from banana_smasher.contract import PackValidationError, verify_pack
from banana_smasher.fixed_qtip_export import _load_member


CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _kernel_swizzle(
    canonical: torch.Tensor,
    *,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    paired = (
        canonical.contiguous()
        .view(torch.uint8)
        .flatten()
        .view(-1, 2)
        .flip((-1,))
        .contiguous()
    )
    return (
        paired.reshape(m // 32, 2, n // 32, 2, 32, k)
        .permute(0, 2, 4, 3, 1, 5)
        .contiguous()
        .flip((-1,))
        .flatten()
        .view(torch.uint16)
        .reshape(canonical.shape)
    )


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
    input_width = output_width = 32
    canonical = (
        torch.arange(output_width * input_width * k // 16, dtype=torch.int32)
        .to(torch.uint16)
        .reshape(output_width, input_width * k // 16)
    )
    trellis = _kernel_swizzle(
        canonical,
        m=output_width,
        n=input_width,
        k=k,
    )
    if k == 3:
        # The sealed producer stores K3 kernel words in a signed int16 container;
        # canonical recovery reinterprets the exact 16-bit payload losslessly.
        trellis = trellis.view(torch.int16)
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
        "tlut": torch.arange(1024, dtype=torch.float32).reshape(512, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "canonical_packed_sha256": _tensor_sha256(canonical),
        "kernel_packed_sha256": _tensor_sha256(trellis),
        "kernel_packed_bytes": trellis.numel() * trellis.element_size(),
        "selectable_data_bytes": sum(
            payload[name].numel() * payload[name].element_size()
            for name in ("trellis", "SU", "SV", "Wscale")
        ),
        "shared_tlut_bytes": payload["tlut"].numel() * payload["tlut"].element_size(),
    }


def test_fixed_qtip_member_recomputes_canonical_hash_from_kernel_wire(
    tmp_path: Path,
) -> None:
    path = tmp_path / "QTIP_UNIT.pt"
    binding = _write_unit(path, k=2, projection="down")
    row = {
        "layer": 0,
        "expert": 0,
        "projection": "down",
        "geometry": {"L": 16, "K": 2, "V": 2},
        "canonical_pack_roundtrip_exact": True,
        "canonical_packed_sha256": binding["canonical_packed_sha256"],
        "kernel_packed_sha256": binding["kernel_packed_sha256"],
        "kernel_packed_bytes": binding["kernel_packed_bytes"],
        "artifact": {
            "path": str(path),
            "bytes": binding["bytes"],
            "sha256": binding["sha256"],
        },
    }

    loaded = _load_member(row, None)
    assert _tensor_sha256(loaded["trellis"]) == binding["kernel_packed_sha256"]

    row["canonical_packed_sha256"] = "0" * 64
    with pytest.raises(PackValidationError, match="canonical payload drift"):
        _load_member(row, None)


def test_fixed_qtip_manifest_survives_public_build_backpack(
    tmp_path: Path,
) -> None:
    member_root = tmp_path / "members"
    templates = {}
    for projection in ("down", "fused13"):
        for k in (2, 3):
            path = tmp_path / "templates" / f"{projection}-k{k}.pt"
            templates[(projection, k)] = (
                path,
                _write_unit(path, k=k, projection=projection),
            )

    rows = []
    selectable_expert_bytes = 0
    for expert in range(256):
        for projection in ("down", "fused13"):
            k = 2 if (expert + (projection == "fused13")) % 2 == 0 else 3
            source, binding = templates[(projection, k)]
            selectable_expert_bytes += int(binding["selectable_data_bytes"])
            staged = (
                member_root / "L000" / f"E{expert:03d}_{projection}" / "QTIP_UNIT.pt"
            )
            staged.parent.mkdir(parents=True)
            os.link(source, staged)
            rows.append(
                {
                    "layer": 0,
                    "expert": expert,
                    "projection": projection,
                    "tier": "qtip@2.50",
                    "geometry": {"L": 16, "K": k, "V": 2},
                    "canonical_pack_roundtrip_exact": True,
                    "canonical_packed_sha256": binding["canonical_packed_sha256"],
                    "kernel_packed_sha256": binding["kernel_packed_sha256"],
                    "kernel_packed_bytes": binding["kernel_packed_bytes"],
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
    bank = tmp_path / "authority.json"
    bank.write_text("{}")
    result = build_backpack(
        {
            "schema": "banana-smasher-backpack-plan-v1",
            "model": {"root": str(model), "revision": "fixture-model-r1"},
            "target": {
                "exact_bytes": selectable_expert_bytes
                + int(templates[("down", 2)][1]["shared_tlut_bytes"])
            },
            "tiers": [
                {
                    "id": "qtip25-fixed",
                    "family": "qtip",
                    "provider": "packaged_qtip",
                    "runtime_family": "qtip2",
                    "bpw": 2.5,
                    "backend": "packaged_qtip",
                    "fixed_assignment": {
                        "path": str(members),
                        "sha256": members_sha,
                        "member_root": str(member_root),
                    },
                }
            ],
            "anchor": {"bank": str(bank), "teacher": "model"},
            "prediction": {"class_caps": {name: 1.0 for name in CLASSES}},
            "repair": {"method": "none"},
            "output": {
                "pack": str(output),
                "model_id": "test/fixed-qtip25",
                "instance_id": "fixed-qtip25-test",
            },
            "reuse_receipts": [
                {
                    "role": "fixed-qtip-pack-admission",
                    "path": str(admission),
                    "sha256": _sha256(admission),
                    "admission": "admitted",
                }
            ],
        },
        run_root=tmp_path / "run",
    )
    assert result["status"] == "PASS"
    assert result["execution"] == "fixed_assignment_streaming"
    assert result["expert_plane_bytes"] == selectable_expert_bytes + 4096
    assert result["shared_tlut_bytes"] == 4096
    assert result["routing_index_bytes"] == 1024
    receipt = verify_pack(output)
    assert receipt["status"] == "PASS"
    manifest = json.loads((output / "BANANA_PACK_MANIFEST.json").read_text())
    assert manifest["fixed_assignment"]["repair_status"] == "absent"
    assert manifest["fixed_assignment"]["members_manifest_sha256"] == members_sha
    assert "repair" not in manifest
    resumed = build_backpack(
        json.loads((tmp_path / "run" / "PLAN.json").read_text()),
        run_root=tmp_path / "run",
    )
    assert resumed["resumed_stages"] == ["fixed_assignment_export"]
