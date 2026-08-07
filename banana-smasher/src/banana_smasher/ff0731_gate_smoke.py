from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .ff0731_gate_runtime import FF0731GateRuntimeAdapter
from .gate_only_trainer import (
    EXPERT_ENVELOPE_BYTES,
    EXPERT_ENVELOPE_PADDING_BYTES,
    FIXED_DENSE_METADATA_BYTES,
    TIERS,
    WHOLE_MODEL_TARGET_BYTES,
    GateOnlyModel,
    final_logit_teacher_kld,
    frozen_state_digest,
    one_cell_sign_step,
    project_exact_budget,
    straight_through_categorical,
)

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-value for value in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]],
    dtype=torch.float32,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    if value.ndim == 0:
        value = value.reshape(1)
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mem_available_bytes() -> int:
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _e8m0(value: torch.Tensor) -> torch.Tensor:
    return torch.exp2(value.view(torch.uint8).to(torch.float32) - 127.0)


def _dequant_native_mxfp4(weight: torch.Tensor, scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    packed = weight.view(torch.uint8).to(device)
    scale_value = _e8m0(scale.to(device))
    byte_values = torch.arange(256, device=device)
    lut = torch.stack((_E2M1.to(device)[byte_values & 15], _E2M1.to(device)[byte_values >> 4]), -1)
    values = lut[packed.long()].flatten(-2)
    expanded_scale = scale_value.repeat_interleave(32, -1)
    if values.shape != expanded_scale.shape:
        raise RuntimeError(
            f"native MXFP4 value/scale shape mismatch: {values.shape} != {expanded_scale.shape}"
        )
    return (values * expanded_scale).to(torch.bfloat16)


def _fwht(value: torch.Tensor) -> torch.Tensor:
    size = value.shape[-1]
    if size <= 0 or size & (size - 1):
        raise ValueError(f"FWHT requires a power-of-two width, got {size}")
    result = value.contiguous()
    width = 1
    while width < size:
        shaped = result.reshape(*result.shape[:-1], size // (2 * width), 2, width)
        left, right = shaped[..., 0, :], shaped[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape(
            *result.shape[:-1], size
        )
        width *= 2
    return result / math.sqrt(size)


def _decode_compressed(
    length: int,
    tlut_bits: int,
    rate: int,
    vector_log2: int,
    rows: int,
    columns: int,
    compressed: torch.Tensor,
    expanded_lut: torch.Tensor,
) -> torch.Tensor:
    if compressed.dtype != torch.uint16:
        compressed = compressed.view(torch.uint16)
    if compressed.shape != (rate * rows * columns // 16,):
        raise ValueError(f"compressed QTIP shape mismatch: {compressed.shape}")
    block_size = 16 * 16
    bits_per_block = rate * block_size
    compressed = (
        compressed.view(torch.uint8)
        .reshape(rows // 32, columns // 32, block_size // 8, 2, 2, rate)
        .permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(rows // 16, columns // 16, bits_per_block // 16, 2)
        .flip((-1,))
        .view(torch.uint16)
        .reshape(rows // 16, columns // 16, bits_per_block // 16)
    )
    blocked = compressed.reshape(rate * rows * columns // bits_per_block, bits_per_block // 16, 1)
    rolled = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    blocked32 = (
        torch.cat((rolled, blocked), dim=-1)
        .reshape(blocked.shape[0], -1)
        .contiguous()
        .view(torch.uint32)
    )
    expanded32 = blocked32.reshape(*blocked32.shape, 1).expand(*blocked32.shape, 16).view(
        torch.int32
    )
    shifts = torch.arange(16, dtype=torch.int32, device=blocked.device).reshape(1, 1, -1)
    shifted = expanded32 >> (16 - shifts.expand(expanded32.shape))
    indices = torch.bitwise_and(
        shifted.reshape(shifted.shape[0], -1)[:, 16 - length :: rate << vector_log2],
        (1 << length) - 1,
    )
    mma_swizzled = expanded_lut[indices]
    return (
        mma_swizzled.reshape(rows // 16, columns // 16, 16, 16)
        .reshape(rows // 16, columns // 16, 8, 4, 2, 2, 2)
        .permute(0, -2, 2, 1, -3, 3, -1)
        .reshape(rows, columns)
    )


def _decode_qtip(unit_path: Path, expected_components: dict[str, Any], device: torch.device) -> torch.Tensor:
    unit = torch.load(unit_path, map_location="cpu", weights_only=False)
    geometry = unit.get("geometry")
    if not isinstance(geometry, dict):
        raise RuntimeError(f"QTIP unit lacks geometry: {unit_path}")
    if (geometry.get("L"), geometry.get("V"), geometry.get("tlut_bits"), geometry.get("decode_mode")) != (
        16,
        2,
        9,
        "quantlut_sym",
    ):
        raise RuntimeError(f"inadmissible QTIP geometry: {geometry}")
    rate = int(geometry["K"])
    if rate not in (2, 3):
        raise RuntimeError(f"smoke requires QTIP2 or QTIP3, got K={rate}")
    for name, expected in expected_components.items():
        tensor = unit.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"QTIP unit is missing tensor {name}: {unit_path}")
        if list(tensor.shape) != expected.get("shape", list(tensor.shape)):
            raise RuntimeError(f"QTIP {name} shape mismatch: {list(tensor.shape)}")
        if _sha256_tensor(tensor) != expected["sha256"]:
            raise RuntimeError(f"QTIP {name} physical SHA-256 mismatch: {unit_path}")
    rows, columns = (int(value) for value in unit["shape"])
    index = torch.arange(1 << 16, device=device)
    quadratic = (index + 1) * index
    sign_flip = 1 - ((quadratic >> 15) & 1) * 2
    lut_index = (quadratic >> (16 - 9 - 1)) & ((1 << 9) - 1)
    expanded = unit["tlut"].float().to(device)[lut_index]
    expanded[:, 0] *= sign_flip
    raw = _decode_compressed(
        16,
        9,
        rate,
        1,
        rows,
        columns,
        unit["trellis"].to(device).reshape(-1),
        expanded,
    )
    quantized = raw * unit["Wscale"].to(device)
    quantized = _fwht(quantized.T).T * unit["SV"].float().to(device)[:, None]
    quantized = _fwht(quantized) * unit["SU"].float().to(device)
    return quantized.to(torch.bfloat16)


class _ExactCellLayer(torch.nn.Module):
    native: torch.Tensor
    qtip2: torch.Tensor
    qtip3: torch.Tensor

    def __init__(self, native: torch.Tensor, qtip2: torch.Tensor, qtip3: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("native", native)
        self.register_buffer("qtip2", qtip2)
        self.register_buffer("qtip3", qtip3)

    def forward(
        self,
        activation: torch.Tensor,
        gates: torch.Tensor,
        hard_tiers: torch.Tensor,
    ) -> torch.Tensor:
        branch_outputs = torch.stack(
            (
                F.linear(activation, self.native),
                F.linear(activation, self.qtip2),
                F.linear(activation, self.qtip3),
            ),
            dim=1,
        )
        return torch.sum(branch_outputs * gates.reshape(1, 3, 1), dim=1) + (
            hard_tiers.sum() * 0
        )


class _ExactHead(torch.nn.Module):
    weight: torch.Tensor

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return F.linear(activation, self.weight)


def _load_manifest_cell(
    path: Path, cell_id: str
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    payload = path.read_bytes()
    if path.suffix == ".gz":
        import gzip

        document = json.loads(gzip.decompress(payload))
    else:
        document = json.loads(payload)
    if document.get("basis_sha256") is None:
        raise RuntimeError("physical manifest lacks basis")
    matches = [row for row in document.get("cells", []) if row.get("cell_id") == cell_id]
    if len(matches) != 1:
        raise RuntimeError(f"physical manifest must contain one {cell_id} row")
    return matches[0], str(document["basis_sha256"]), document


def _manifest_component_expectations(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(name): component
        for name, component in value.items()
        if isinstance(component, dict) and "sha256" in component
    }


def _tensor_from_model(
    model_root: Path, weight_map: dict[str, str], name: str
) -> torch.Tensor:
    from safetensors import safe_open

    shard = model_root / weight_map[name]
    with safe_open(shard, framework="pt") as handle:
        return handle.get_tensor(name)


def run_smoke(
    *,
    model_root: Path,
    qtip2_unit: Path,
    qtip3_unit: Path,
    physical_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("FF0731 gate runtime smoke requires CUDA")
    available = _mem_available_bytes()
    peak_estimate = 5 * 4096 * 2048 * 2 + 2 * 1024**3
    if peak_estimate > available - 4 * 1024**3:
        raise MemoryError(
            f"smoke peak estimate {peak_estimate} exceeds MemAvailable-4GiB {available - 4 * 1024**3}"
        )
    model_root = model_root.resolve()
    index_path = model_root / "model.safetensors.index.json"
    basis_sha256 = _sha256_file(index_path)
    cell, manifest_basis, physical_document = _load_manifest_cell(
        physical_manifest, "L000.E000.down"
    )
    if manifest_basis != basis_sha256:
        raise RuntimeError(f"basis mismatch: manifest={manifest_basis}, model={basis_sha256}")
    weight_map = json.loads(index_path.read_text())["weight_map"]

    weight = _tensor_from_model(model_root, weight_map, "layers.0.ffn.experts.0.w2.weight")
    scale = _tensor_from_model(model_root, weight_map, "layers.0.ffn.experts.0.w2.scale")
    native = _dequant_native_mxfp4(weight, scale, device)
    del weight, scale
    qtip2_artifact = cell["tiers"]["qtip2"]["artifacts"][0]
    if (
        qtip2_unit.stat().st_size != qtip2_artifact["container_bytes"]
        or _sha256_file(qtip2_unit) != qtip2_artifact["container_sha256"]
    ):
        raise RuntimeError("QTIP2 physical container does not match the all-cell manifest")
    qtip2_expected = _manifest_component_expectations(qtip2_artifact.get("components", {}))
    qtip3_expected = dict(cell["tiers"]["qtip3"]["artifacts"][0]["components"])
    qtip3_expected["tlut"] = physical_document["global_artifacts"]["qtip3_shared_tlut"]
    if not qtip2_expected:
        qtip2_unit_value = torch.load(qtip2_unit, map_location="cpu", weights_only=False)
        qtip2_expected = {
            name: {
                "sha256": _sha256_tensor(qtip2_unit_value[name]),
                "shape": list(qtip2_unit_value[name].shape),
            }
            for name in ("SU", "SV", "Wscale", "trellis")
        }
    qtip2 = _decode_qtip(qtip2_unit, qtip2_expected, device)
    qtip3 = _decode_qtip(qtip3_unit, qtip3_expected, device)
    if native.shape != qtip2.shape or native.shape != qtip3.shape:
        raise RuntimeError(
            f"three-tier branch shape mismatch: {native.shape}, {qtip2.shape}, {qtip3.shape}"
        )

    head = _tensor_from_model(model_root, weight_map, "head.weight")[:128].to(
        device=device, dtype=torch.bfloat16
    )
    activation = torch.linspace(-1.0, 1.0, native.shape[1], device=device).reshape(1, -1).to(
        torch.bfloat16
    )
    layer_module = torch.jit.trace(
        _ExactCellLayer(native, qtip2, qtip3),
        (activation, torch.tensor([[1.0, 0.0, 0.0]], device=device), torch.tensor([0], device=device)),
    )
    head_module = torch.jit.trace(
        _ExactHead(head), torch.zeros((1, native.shape[0]), device=device, dtype=torch.bfloat16)
    )
    layer_path = output / "L000.E000.down.layer.pt"
    head_path = output / "head-128.pt"
    torch.jit.save(layer_module, str(layer_path))
    torch.jit.save(head_module, str(head_path))

    with torch.no_grad():
        native_hidden = F.linear(activation, native)
        teacher_logits = F.linear(native_hidden, head)
    activation_path = output / "activation.pt"
    teacher_path = output / "teacher_logits.pt"
    torch.save(activation.detach().cpu(), activation_path)
    torch.save(teacher_logits.detach().cpu(), teacher_path)

    smoke_physical = {
        "schema": "banana-smasher-ff0731-three-tier-cells-v1",
        "status": "PASS",
        "basis_sha256": basis_sha256,
        "tiers": list(TIERS),
        "cells": [
            {
                "cell_id": "L000.E000.down",
                "layer": 0,
                "expert": 0,
                "projection": "down",
                "tiers": {
                    "native_mxfp4": {
                        "wire_bytes": cell["tiers"]["native_mxfp4"]["wire_bytes"],
                        "artifacts": [
                            {
                                "path": str(index_path),
                                "bytes": index_path.stat().st_size,
                                "sha256": basis_sha256,
                            }
                        ],
                    },
                    "qtip2": {
                        "wire_bytes": cell["tiers"]["qtip2"]["wire_bytes"],
                        "artifacts": [
                            {
                                "path": str(qtip2_unit),
                                "bytes": qtip2_unit.stat().st_size,
                                "sha256": _sha256_file(qtip2_unit),
                            }
                        ],
                    },
                    "qtip3": {
                        "wire_bytes": cell["tiers"]["qtip3"]["wire_bytes"],
                        "artifacts": [
                            {
                                "path": str(qtip3_unit),
                                "bytes": qtip3_unit.stat().st_size,
                                "sha256": _sha256_file(qtip3_unit),
                            }
                        ],
                    },
                },
            }
        ],
    }
    smoke_manifest_path = output / "smoke-physical.json"
    _atomic_json(smoke_manifest_path, smoke_physical)
    runtime_config = {
        "schema": "banana-smasher-ff0731-torchscript-gate-runtime-v1",
        "strict_geometry": False,
        "device": "cuda",
        "verify_payloads": True,
        "physical_manifest": {
            "path": str(smoke_manifest_path),
            "sha256": _sha256_file(smoke_manifest_path),
        },
        "layers": [
            {
                "layer": 0,
                "cell_ids": ["L000.E000.down"],
                "module": {"path": str(layer_path), "sha256": _sha256_file(layer_path)},
            }
        ],
        "final_head": {"path": str(head_path), "sha256": _sha256_file(head_path)},
        "data_sidecars": [
            {"path": str(activation_path), "sha256": _sha256_file(activation_path)},
            {"path": str(teacher_path), "sha256": _sha256_file(teacher_path)},
        ],
    }
    runtime = FF0731GateRuntimeAdapter(
        model_root=model_root,
        basis_sha256=basis_sha256,
        parameters=runtime_config,
    )
    frozen_before = frozen_state_digest(runtime.frozen_state())
    batch = {
        "window_id": "physical-canary-L000.E000.down",
        "activation": {"path": str(activation_path), "sha256": _sha256_file(activation_path)},
        "teacher_logits": {"path": str(teacher_path), "sha256": _sha256_file(teacher_path)},
    }
    branch_kld = []
    for tier_index in range(3):
        gates = F.one_hot(torch.tensor([tier_index], device=device), num_classes=3).to(torch.float32)
        with runtime.layer_stage(0) as forward:
            hidden = forward(
                runtime.initial(batch),
                gates=gates,
                hard_tiers=torch.tensor([tier_index], device=device),
                window_id=batch["window_id"],
            )
        loss = final_logit_teacher_kld(
            runtime.final_logits(hidden, window_id=batch["window_id"]),
            runtime.teacher_logits(batch),
        )
        branch_kld.append(float(loss))
    if not all(math.isfinite(value) and value >= 0.0 for value in branch_kld):
        raise RuntimeError(f"non-finite three-branch KLD: {branch_kld}")

    model = GateOnlyModel(len(physical_document["cells"])).to(device)
    with torch.no_grad():
        model.tier_logits[0, 1] = 2.0
    optimizer = torch.optim.SGD([model.tier_logits], lr=1.0)
    optimizer_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if any(parameter is candidate for group in optimizer.param_groups for candidate in group["params"])
    ]
    if list(dict(model.named_parameters())) != ["tier_logits"] or optimizer_parameter_names != [
        "tier_logits"
    ]:
        raise RuntimeError("physical smoke optimizer allowlist is not exactly tier_logits")
    gates, _ = straight_through_categorical(model.tier_logits[:1], temperature=1.0)
    hard_tiers = gates.detach().argmax(dim=-1)
    optimizer.zero_grad(set_to_none=True)
    with runtime.layer_stage(0) as forward:
        hidden = forward(
            runtime.initial(batch),
            gates=gates,
            hard_tiers=hard_tiers,
            window_id=batch["window_id"],
        )
    autograd_loss = final_logit_teacher_kld(
        runtime.final_logits(hidden, window_id=batch["window_id"]),
        runtime.teacher_logits(batch),
    )
    autograd_loss.backward()
    if model.tier_logits.grad is None or not torch.isfinite(model.tier_logits.grad).all():
        raise RuntimeError("physical gate smoke produced no finite tier-logit gradient")
    raw_gradient_norm = float(torch.linalg.vector_norm(model.tier_logits.grad.detach()))
    if raw_gradient_norm <= 0.0:
        raise RuntimeError("physical gate smoke produced a zero tier-logit gradient")
    optimizer.step()
    before, after, direction_gradient = one_cell_sign_step(
        branch_kld=torch.tensor(branch_kld),
        initial_logits=torch.zeros(3),
        learning_rate=1.0,
    )
    best = min(range(3), key=branch_kld.__getitem__)
    if not float(after[best]) > float(before[best]):
        raise RuntimeError("lower-KLD physical branch probability did not rise")

    padding = physical_document.get("global_artifacts", {}).get("expert_envelope_padding")
    if not isinstance(padding, dict) or padding.get("bytes") != EXPERT_ENVELOPE_PADDING_BYTES:
        raise RuntimeError("physical manifest lacks canonical expert-envelope padding")
    padding_path = output / "expert-envelope-padding.bin"
    padding_path.write_bytes(bytes.fromhex(str(padding.get("content_hex", ""))))
    if (
        padding_path.stat().st_size != EXPERT_ENVELOPE_PADDING_BYTES
        or _sha256_file(padding_path) != padding.get("sha256")
    ):
        raise RuntimeError("physical expert-envelope padding identity mismatch")

    projection_started = time.time()
    projection = project_exact_budget(
        tier_logits=model.tier_logits.detach(),
        cell_ids=[str(row["cell_id"]) for row in physical_document["cells"]],
        tier_bytes=torch.tensor(
            [
                [int(row["tiers"][tier]["wire_bytes"]) for tier in TIERS]
                for row in physical_document["cells"]
            ],
            dtype=torch.int64,
        ),
        expert_envelope_bytes=EXPERT_ENVELOPE_BYTES,
        expert_envelope_padding_bytes=EXPERT_ENVELOPE_PADDING_BYTES,
    )
    projection_seconds = time.time() - projection_started
    hard_whole_model_bytes = FIXED_DENSE_METADATA_BYTES + projection.hard_expert_bytes
    if (
        projection.hard_expert_bytes != EXPERT_ENVELOPE_BYTES
        or hard_whole_model_bytes != WHOLE_MODEL_TARGET_BYTES
    ):
        raise RuntimeError("physical smoke exact projection missed the canonical byte target")
    frozen_after = frozen_state_digest(runtime.frozen_state())
    if frozen_after != frozen_before:
        raise RuntimeError("physical smoke changed frozen runtime state")

    receipt = {
        "schema": "banana-smasher-ff0731-gate-runtime-smoke-v1",
        "status": "PASS",
        "basis_sha256": basis_sha256,
        "cell_id": "L000.E000.down",
        "device": torch.cuda.get_device_name(),
        "branch_kld": dict(zip(TIERS, branch_kld, strict=True)),
        "autograd_loss": float(autograd_loss.detach()),
        "trainable_parameter_names": list(dict(model.named_parameters())),
        "trainable_parameter_shape": list(model.tier_logits.shape),
        "optimizer_parameter_names": optimizer_parameter_names,
        "tier_logit_gradient_first_cell": model.tier_logits.grad.detach().cpu()[0].tolist(),
        "raw_gradient_l2_norm": raw_gradient_norm,
        "direction_gradient": direction_gradient.tolist(),
        "best_tier": TIERS[best],
        "best_probability_before": float(before[best]),
        "best_probability_after": float(after[best]),
        "execution_trace": runtime.execution_trace,
        "hard_selected_tiers_executed": all(
            row["hard_forward"] for row in runtime.execution_trace
        ),
        "exact_projection": {
            "cell_payload_bytes": projection.hard_cell_payload_bytes,
            "expert_envelope_padding_bytes": projection.expert_envelope_padding_bytes,
            "expert_bytes": projection.hard_expert_bytes,
            "fixed_dense_metadata_bytes": FIXED_DENSE_METADATA_BYTES,
            "whole_model_bytes": hard_whole_model_bytes,
            "tier_counts": projection.tier_counts,
            "solver": projection.solver,
            "wall_seconds": projection_seconds,
        },
        "frozen_state_digest_before": frozen_before,
        "frozen_state_digest_after": frozen_after,
        "branch_tensor_sha256": {
            "native_mxfp4": _sha256_tensor(native),
            "qtip2": _sha256_tensor(qtip2),
            "qtip3": _sha256_tensor(qtip3),
        },
        "artifacts": {
            "layer_module": {"path": str(layer_path), "sha256": _sha256_file(layer_path)},
            "head_module": {"path": str(head_path), "sha256": _sha256_file(head_path)},
            "qtip2_unit": {"path": str(qtip2_unit), "sha256": _sha256_file(qtip2_unit)},
            "qtip3_unit": {"path": str(qtip3_unit), "sha256": _sha256_file(qtip3_unit)},
            "expert_envelope_padding": {
                "path": str(padding_path),
                "bytes": padding_path.stat().st_size,
                "sha256": _sha256_file(padding_path),
            },
        },
        "mem_available_preflight_bytes": available,
        "peak_estimate_bytes": peak_estimate,
        "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.time() - started,
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    receipt_path = output / "SMOKE_RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    receipt["receipt"] = {"path": str(receipt_path), "sha256": _sha256_file(receipt_path)}
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--qtip2-unit", type=Path, required=True)
    parser.add_argument("--qtip3-unit", type=Path, required=True)
    parser.add_argument("--physical-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(
        model_root=args.model_root,
        qtip2_unit=args.qtip2_unit,
        qtip3_unit=args.qtip3_unit,
        physical_manifest=args.physical_manifest,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
