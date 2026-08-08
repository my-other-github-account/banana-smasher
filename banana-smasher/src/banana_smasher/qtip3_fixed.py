"""Public fixed-assignment QTIP3 artifacts and continuous-state repair."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

QTIP3_FIXED_SCHEMA = "banana-smasher-qtip3-fixed-member-v1"
QTIP3_UPDATE_SCHEMA = "banana-smasher-qtip3-fixed-update-request-v1"
QTIP3_PROVIDER_ID = "periodic-qtip3@3.00"
QTIP3_LEGACY_PROVIDER_ID = "qtip-native-v6@3.00"
QTIP3_PROVIDER_IDS = frozenset((QTIP3_PROVIDER_ID, QTIP3_LEGACY_PROVIDER_ID))
QTIP3_GEOMETRY = {
    "L": 16,
    "B": 12,
    "V": 4,
    "layout": "homogeneous",
    "phase_widths": [3, 3, 3, 3],
}
QTIP3_LEGACY_GEOMETRY = {
    "L": 16,
    "B": 12,
    "V": 4,
    "phase_widths": [3, 3, 3, 3],
}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value


@dataclass(frozen=True)
class Qtip3FixedMember:
    artifact_path: Path
    codec_provider_id: str
    basis_index_sha256: str
    source_weight_sha256: str
    hessian_sha256: str
    geometry: dict[str, Any]
    lut_identity: str
    lut_tensor_sha256: str
    lut: torch.Tensor
    codes: torch.Tensor
    SU: torch.Tensor
    SV: torch.Tensor
    Wscale: torch.Tensor


def load_qtip3_fixed_member(
    artifact_path: str | Path, *, lut_path: str | Path
) -> Qtip3FixedMember:
    """Load one fixed QTIP3 member and verify its shared PR31 LUT binding."""

    path = Path(artifact_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or payload.get("schema") != QTIP3_FIXED_SCHEMA:
        raise ValueError(f"QTIP3 member requires schema {QTIP3_FIXED_SCHEMA!r}")
    provider_id = payload.get("codec_provider_id")
    if provider_id not in QTIP3_PROVIDER_IDS:
        raise ValueError(
            "QTIP3 member requires provider "
            f"{QTIP3_PROVIDER_ID!r} or legacy identity {QTIP3_LEGACY_PROVIDER_ID!r}"
        )
    geometry = dict(payload.get("geometry", {}))
    expected_geometry = (
        QTIP3_GEOMETRY
        if provider_id == QTIP3_PROVIDER_ID
        else QTIP3_LEGACY_GEOMETRY
    )
    if geometry != expected_geometry:
        raise ValueError(f"QTIP3 member geometry mismatch: {geometry!r}")
    lut_binding = payload.get("lut")
    if not isinstance(lut_binding, Mapping):
        raise ValueError("QTIP3 member requires an owned LUT binding")
    identity = lut_binding.get("identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError("QTIP3 member LUT identity must be non-empty")
    expected_lut_sha = _require_sha256(
        lut_binding.get("tensor_sha256"), "QTIP3 member LUT tensor SHA-256"
    )
    lut = torch.load(Path(lut_path).expanduser().resolve(), map_location="cpu", weights_only=True)
    if not isinstance(lut, torch.Tensor):
        raise ValueError("QTIP3 member LUT artifact must contain one tensor")
    lut = lut.detach().cpu().contiguous()
    if _tensor_sha256(lut) != expected_lut_sha:
        raise ValueError("LUT tensor SHA-256 mismatch")
    expected_lut_bytes = lut_binding.get("data_bytes")
    if expected_lut_bytes != lut.numel() * lut.element_size():
        raise ValueError("QTIP3 member LUT data byte count mismatch")
    tensors: dict[str, torch.Tensor] = {}
    for name in ("codes", "SU", "SV", "Wscale"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"QTIP3 member requires tensor {name}")
        tensors[name] = value.detach().cpu().contiguous()
    if tensors["codes"].dtype not in {torch.uint8, torch.int16, torch.int32, torch.int64}:
        raise ValueError("QTIP3 packed codes require an integer dtype")
    if tensors["Wscale"].dtype != torch.float32 or tensors["Wscale"].numel() != 1:
        raise ValueError("QTIP3 Wscale requires one float32 value")
    if tensors["codes"].numel() < tensors["SU"].numel() * tensors["SV"].numel():
        raise ValueError("QTIP3 packed codes do not cover the declared fixed geometry")
    return Qtip3FixedMember(
        artifact_path=path,
        codec_provider_id=provider_id,
        basis_index_sha256=_require_sha256(
            payload.get("basis_index_sha256"), "basis index SHA-256"
        ),
        source_weight_sha256=_require_sha256(
            payload.get("source_weight_sha256"), "source weight SHA-256"
        ),
        hessian_sha256=_require_sha256(payload.get("hessian_sha256"), "Hessian SHA-256"),
        geometry=geometry,
        lut_identity=identity,
        lut_tensor_sha256=expected_lut_sha,
        lut=lut,
        codes=tensors["codes"],
        SU=tensors["SU"],
        SV=tensors["SV"],
        Wscale=tensors["Wscale"],
    )


class Qtip3FixedRepairRuntime:
    """Train only the shared LUT while keeping assignments and geometry immutable."""

    def __init__(
        self,
        *,
        members: Sequence[Qtip3FixedMember],
        learning_rate: float,
        device: str | torch.device,
    ) -> None:
        if not members:
            raise ValueError("QTIP3 repair requires at least one fixed member")
        first = members[0]
        if any(
            member.lut_identity != first.lut_identity
            or member.lut_tensor_sha256 != first.lut_tensor_sha256
            for member in members[1:]
        ):
            raise ValueError("QTIP3 repair members do not share one sealed LUT")
        self.members = tuple(members)
        self.device = torch.device(device)
        self.shared_lut = torch.nn.Parameter(first.lut.to(self.device, torch.float32))
        self.optimizer = torch.optim.SGD((self.shared_lut,), lr=float(learning_rate))
        self.acceleration_counters = {
            "periodic_qtip3_lut_vjp_calls": 0,
            "fallback_calls": 0,
        }

    def _weight(self, member: Qtip3FixedMember) -> torch.Tensor:
        input_features = int(member.SU.numel())
        output_features = int(member.SV.numel())
        indices = member.codes[: input_features * output_features].to(
            self.device, torch.int64
        )
        indices = indices.remainder(self.shared_lut.numel())
        base = self.shared_lut.index_select(0, indices).reshape(
            output_features, input_features
        )
        return (
            base
            * member.Wscale.to(self.device, torch.float32)
            * member.SV.to(self.device, torch.float32)[:, None]
            * member.SU.to(self.device, torch.float32)[None, :]
        )

    def _loss_sum(
        self,
        activation_inputs: torch.Tensor,
        teacher_targets: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        member = self.members[0]
        inputs = activation_inputs.to(self.device, torch.float32)
        targets = teacher_targets.to(self.device, torch.float32)
        mask = teacher_mask.to(self.device)
        if mask.dtype != torch.bool or tuple(inputs.shape[:-1]) != tuple(mask.shape):
            raise ValueError("QTIP3 teacher mask does not match activation token geometry")
        if inputs.shape[-1] != member.SU.numel() or targets.shape[-1] != member.SV.numel():
            raise ValueError("QTIP3 activation/teacher feature geometry mismatch")
        outputs = torch.matmul(inputs, self._weight(member).transpose(0, 1))
        self.acceleration_counters["periodic_qtip3_lut_vjp_calls"] += 1
        selected = mask.unsqueeze(-1).expand_as(outputs)
        count = int(selected.sum().item())
        if count <= 0:
            raise ValueError("QTIP3 microdose requires at least one teacher target")
        return torch.square(outputs - targets).masked_select(selected).sum(), count

    def microdose(
        self,
        *,
        activation_inputs: torch.Tensor,
        teacher_targets: torch.Tensor,
        teacher_mask: torch.Tensor,
    ) -> dict[str, Any]:
        code_hashes = tuple(_tensor_sha256(member.codes) for member in self.members)
        geometries = tuple(dict(member.geometry) for member in self.members)
        before = self.shared_lut.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum, count = self._loss_sum(
            activation_inputs, teacher_targets, teacher_mask
        )
        (loss_sum / count).backward()
        gradient = self.shared_lut.grad
        finite_nonzero = bool(
            gradient is not None
            and torch.isfinite(gradient).all().item()
            and torch.count_nonzero(gradient).item() > 0
        )
        if not finite_nonzero:
            raise RuntimeError("QTIP3 microdose produced no finite nonzero LUT gradient")
        self.optimizer.step()
        delta = float(torch.linalg.vector_norm(self.shared_lut.detach() - before).item())
        codes_unchanged = code_hashes == tuple(
            _tensor_sha256(member.codes) for member in self.members
        )
        geometry_unchanged = geometries == tuple(
            dict(member.geometry) for member in self.members
        )
        if not codes_unchanged or not geometry_unchanged:
            raise RuntimeError("QTIP3 microdose changed immutable assignments or geometry")
        return {
            "schema": "banana-smasher-qtip3-fixed-microdose-v1",
            "status": "PASS_UPDATE",
            "finite_nonzero_gradients": finite_nonzero,
            "authorized_parameter_delta": delta,
            "packed_codes_unchanged": codes_unchanged,
            "geometry_unchanged": geometry_unchanged,
            "loss_sum": float(loss_sum.detach().cpu().item()),
            "target_count": count,
            "acceleration_counters": dict(self.acceleration_counters),
            "fallback": {"used": False},
        }


def run_qtip3_fixed_update(
    *,
    request: Path,
    output: Path,
    receipt: Path | None,
    identity: dict[str, Any],
    requested_tokens: int,
    physical_tokens: int,
    segments: int,
    batch_size: int,
    memory_sizing: dict[str, Any],
    resume: bool,
    restart: bool,
) -> dict[str, Any]:
    """Installed ``smash update`` backend for one fixed-QTIP3 microdose."""

    if batch_size != 1:
        raise ValueError("QTIP3 fixed repair requires batch size one")
    request_root = request.parent
    spec = json.loads(request.read_text())
    if spec.get("schema") != QTIP3_UPDATE_SCHEMA:
        raise ValueError(f"QTIP3 update requires schema {QTIP3_UPDATE_SCHEMA!r}")
    members = [
        load_qtip3_fixed_member(
            request_root / row["artifact"], lut_path=request_root / row["lut"]
        )
        for row in spec["members"]
    ]
    tensors = torch.load(
        request_root / spec["teacher_batch"], map_location="cpu", weights_only=True
    )
    logical_tokens = int(physical_tokens) * int(segments)
    activation_inputs = tensors["activation_inputs"][:, :logical_tokens]
    teacher_targets = tensors["teacher_targets"][:, :logical_tokens]
    teacher_mask = tensors["teacher_mask"][:, :logical_tokens]
    if activation_inputs.shape[1] != logical_tokens:
        raise ValueError("QTIP3 update request has insufficient logical tokens")
    runtime = Qtip3FixedRepairRuntime(
        members=members,
        learning_rate=float(spec["learning_rate"]),
        device=spec.get("device", "cuda"),
    )
    result = runtime.microdose(
        activation_inputs=activation_inputs,
        teacher_targets=teacher_targets,
        teacher_mask=teacher_mask,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(
            {
                "schema": "banana-smasher-qtip3-fixed-repair-checkpoint-v1",
                "identity": identity,
                "shared_lut": runtime.shared_lut.detach().cpu(),
                "packed_code_sha256": [_tensor_sha256(member.codes) for member in members],
                "geometries": [member.geometry for member in members],
            },
            temporary,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    result.update(
        {
            "physical_tokens": int(physical_tokens),
            "segments": int(segments),
            "optimizer_steps": 1,
            "observed_input_shape": [1, int(physical_tokens)],
            "requested_physical_tokens": int(requested_tokens),
            "memory_sizing": memory_sizing,
            "output": str(output),
            "identity": identity,
            "resume": bool(resume),
            "restart": bool(restart),
        }
    )
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
