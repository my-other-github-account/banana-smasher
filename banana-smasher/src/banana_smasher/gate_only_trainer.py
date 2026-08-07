from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
TIERS = ("native_mxfp4", "qtip2", "qtip3")
FF0731_MODEL_ROOT = "DeepSeek-V4-Flash-0731"
FF0731_CELL_COUNT = 22_016
WHOLE_MODEL_TARGET_BYTES = 102_000_000_000
FIXED_DENSE_METADATA_BYTES = 9_032_112_614
EXPERT_ENVELOPE_BYTES = 92_967_887_386
EXPERT_ENVELOPE_PADDING_BYTES = 2
REPAIR_BUDGET_BYTES = 0


@dataclass(frozen=True)
class GateDataManifest:
    path: Path
    sha256: str
    split: str
    class_counts: dict[str, int]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class GateTrainingConfig:
    cell_count: int = FF0731_CELL_COUNT
    whole_model_target_bytes: int = WHOLE_MODEL_TARGET_BYTES
    fixed_dense_metadata_bytes: int = FIXED_DENSE_METADATA_BYTES
    expert_envelope_bytes: int = EXPERT_ENVELOPE_BYTES
    expert_envelope_padding_bytes: int = EXPERT_ENVELOPE_PADDING_BYTES
    repair_budget_bytes: int = REPAIR_BUDGET_BYTES
    steps: int = 20
    learning_rate: float = 0.1
    temperature: float = 1.0
    dev_every: int = 1
    byte_dual_learning_rate: float = 0.1

    def validate(self) -> None:
        integer_fields = {
            "cell_count": self.cell_count,
            "whole_model_target_bytes": self.whole_model_target_bytes,
            "fixed_dense_metadata_bytes": self.fixed_dense_metadata_bytes,
            "expert_envelope_bytes": self.expert_envelope_bytes,
            "expert_envelope_padding_bytes": self.expert_envelope_padding_bytes,
            "repair_budget_bytes": self.repair_budget_bytes,
            "steps": self.steps,
            "dev_every": self.dev_every,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_fields.values()
        ):
            raise ValueError("gate training integer configuration fields must be non-negative")
        if self.cell_count == 0 or self.steps == 0 or self.dev_every == 0:
            raise ValueError("cell_count, steps, and dev_every must be positive")
        if self.repair_budget_bytes != 0:
            raise ValueError("gate-only training requires repair_budget_bytes=0")
        if self.expert_envelope_padding_bytes > self.expert_envelope_bytes:
            raise ValueError("expert envelope padding cannot exceed the expert envelope")
        if (
            self.fixed_dense_metadata_bytes
            + self.expert_envelope_bytes
            + self.repair_budget_bytes
            != self.whole_model_target_bytes
        ):
            raise ValueError("gate-only whole-model byte equation mismatch")
        for label, value in (
            ("learning_rate", self.learning_rate),
            ("temperature", self.temperature),
            ("byte_dual_learning_rate", self.byte_dual_learning_rate),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")


@dataclass(frozen=True)
class HardProjection:
    tier_indices: torch.Tensor
    hard_expert_bytes: int
    hard_cell_payload_bytes: int
    expert_envelope_padding_bytes: int
    tier_counts: dict[str, int]
    assignments: tuple[dict[str, Any], ...]
    solver: dict[str, Any]


@dataclass
class GateTrainingResult:
    model: "GateOnlyModel"
    receipt: dict[str, Any]
    optimizer_parameter_names: list[str]
    frozen_state_digest_before: str
    frozen_state_digest_after: str
    final_projection: HardProjection


class GateOnlyModel(nn.Module):
    """The complete trainable model: one categorical logit triplet per expert cell."""

    def __init__(self, cell_count: int, *, initial_logits: torch.Tensor | None = None) -> None:
        super().__init__()
        if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count <= 0:
            raise ValueError("cell_count must be a positive integer")
        if initial_logits is None:
            value = torch.zeros((cell_count, len(TIERS)), dtype=torch.float32)
        else:
            value = torch.as_tensor(initial_logits, dtype=torch.float32).detach().clone()
            if value.shape != (cell_count, len(TIERS)) or not torch.isfinite(value).all():
                raise ValueError(f"initial_logits must have shape [{cell_count},{len(TIERS)}]")
        self.tier_logits = nn.Parameter(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _path_mentions_holdout(path: Path) -> bool:
    return any("HOLDOUT" in part.upper() for part in path.parts)


def _load_data_manifest(path: str | Path, *, expected_split: str) -> GateDataManifest:
    resolved = Path(path).expanduser().resolve()
    if _path_mentions_holdout(resolved):
        raise ValueError(f"HOLDOUT paths are forbidden during gate training: {resolved}")
    payload = resolved.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid gate data manifest {resolved}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != "banana-smasher-gate-data-v1":
        raise ValueError("gate data manifest schema must be banana-smasher-gate-data-v1")
    split = document.get("split")
    if split == "HOLDOUT" or split != expected_split:
        raise ValueError(f"gate data manifest split must be {expected_split}, not {split!r}")
    raw_counts = document.get("class_counts")
    if not isinstance(raw_counts, dict) or set(raw_counts) != set(CLASSES):
        raise ValueError("gate data manifest class_counts must cover exactly six classes")
    counts: dict[str, int] = {}
    for name in CLASSES:
        value = raw_counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"gate data manifest class count for {name} must be positive")
        counts[name] = value
    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("gate data manifest rows must be non-empty")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    observed = {name: 0 for name in CLASSES}
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"gate data manifest row {index} must be an object")
        row = dict(raw)
        window_id = row.get("window_id")
        class_name = row.get("class")
        if not isinstance(window_id, str) or not window_id or window_id in seen:
            raise ValueError("gate data manifest window_id values must be non-empty and unique")
        if class_name not in observed:
            raise ValueError(f"gate data manifest row class is invalid: {class_name!r}")
        for field, value in row.items():
            if "path" in field.lower() and isinstance(value, str) and "HOLDOUT" in value.upper():
                raise ValueError("HOLDOUT paths are forbidden during gate training")
        seen.add(window_id)
        observed[str(class_name)] += 1
        rows.append(row)
    if observed != counts:
        raise ValueError(f"gate data manifest class_counts mismatch: declared={counts}, observed={observed}")
    return GateDataManifest(
        path=resolved,
        sha256=hashlib.sha256(payload).hexdigest(),
        split=expected_split,
        class_counts=counts,
        rows=tuple(rows),
    )


def load_training_manifests(
    train_manifest: str | Path, dev_manifest: str | Path
) -> tuple[GateDataManifest, GateDataManifest]:
    train = _load_data_manifest(train_manifest, expected_split="TRAIN")
    dev = _load_data_manifest(dev_manifest, expected_split="DEV")
    train_ids = {str(row["window_id"]) for row in train.rows}
    dev_ids = {str(row["window_id"]) for row in dev.rows}
    overlap = sorted(train_ids & dev_ids)
    if overlap:
        raise ValueError(f"TRAIN/DEV window overlap is forbidden: {overlap[:3]}")
    return train, dev


def frozen_state_digest(state: Mapping[str, Any]) -> str:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("runtime frozen_state() must return a non-empty mapping")
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise ValueError("frozen state must map non-empty names to tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(_canonical(list(tensor.shape)))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def straight_through_categorical(
    logits: torch.Tensor,
    *,
    temperature: float,
    hard_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits / temperature, dim=-1)
    indices = probabilities.argmax(dim=-1) if hard_indices is None else hard_indices.to(logits.device)
    hard = F.one_hot(indices, num_classes=len(TIERS)).to(probabilities.dtype)
    return hard + probabilities - probabilities.detach(), probabilities


def final_logit_teacher_kld(candidate_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    candidate = torch.as_tensor(candidate_logits, dtype=torch.float32)
    teacher = torch.as_tensor(teacher_logits, dtype=torch.float32, device=candidate.device)
    if candidate.shape != teacher.shape or candidate.ndim < 1 or candidate.shape[-1] < 2:
        raise ValueError("candidate and teacher final logits must have the same [..., vocab] shape")
    return F.kl_div(
        F.log_softmax(candidate, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    )


def one_cell_sign_step(
    *,
    branch_kld: torch.Tensor,
    initial_logits: torch.Tensor,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    losses = torch.as_tensor(branch_kld, dtype=torch.float32)
    if losses.shape != (len(TIERS),) or not torch.isfinite(losses).all():
        raise ValueError("branch_kld must contain three finite measured KLD values")
    model = GateOnlyModel(1, initial_logits=torch.as_tensor(initial_logits).reshape(1, -1))
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    before = torch.softmax(model.tier_logits.detach()[0], dim=-1)
    gates, _ = straight_through_categorical(model.tier_logits, temperature=1.0)
    loss = torch.sum(gates[0] * losses)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if model.tier_logits.grad is None:
        raise RuntimeError("one-cell sign step produced no tier-logit gradient")
    gradient = model.tier_logits.grad.detach()[0].clone()
    optimizer.step()
    after = torch.softmax(model.tier_logits.detach()[0], dim=-1)
    return before, after, gradient


def _tier_byte_tensor(value: Any, *, cell_count: int) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.int64).cpu()
    if result.shape != (cell_count, len(TIERS)) or torch.any(result < 0):
        raise ValueError(f"runtime tier_bytes must have shape [{cell_count},{len(TIERS)}]")
    return result


def _project_two_repeated_shapes(
    *,
    negative_log_probabilities: torch.Tensor,
    cell_ids: Sequence[str],
    byte_rows: torch.Tensor,
    cell_payload_envelope: int,
    expert_envelope_padding_bytes: int,
) -> HardProjection | None:
    """Fast exact-count projection for FF0731's two repeated projection shapes."""
    import numpy as np

    patterns: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(byte_rows.tolist()):
        patterns.setdefault(tuple(int(value) for value in row), []).append(index)
    if len(patterns) != 2 or len(cell_ids) <= 512:
        return None
    groups = sorted(patterns.items())
    group_counts = [len(indices) for _, indices in groups]
    bases = [min(range(len(TIERS)), key=pattern.__getitem__) for pattern, _ in groups]
    alternatives = [
        tuple(index for index in range(len(TIERS)) if index != base) for base in bases
    ]
    baseline_bytes = sum(
        pattern[base] * count
        for (pattern, _), base, count in zip(groups, bases, group_counts, strict=True)
    )
    remaining = cell_payload_envelope - baseline_bytes
    if remaining < 0:
        return None
    deltas = [
        tuple(pattern[index] - pattern[base] for index in alternatives[group_index])
        for group_index, ((pattern, _), base) in enumerate(zip(groups, bases, strict=True))
    ]
    divisor = math.gcd(*[value for pair in deltas for value in pair])
    if divisor <= 0 or remaining % divisor:
        return None
    normalized = [[value // divisor for value in pair] for pair in deltas]
    target = remaining // divisor
    a, b = normalized[0]
    c, d = normalized[1]
    equation_gcd = math.gcd(a, b)
    modulus = a // equation_gcd
    if modulus <= group_counts[0] or math.gcd(b // equation_gcd, modulus) != 1:
        return None
    inverse = pow(b // equation_gcd, -1, modulus)
    costs = negative_log_probabilities.detach().cpu().numpy()
    group_means = [
        costs[np.asarray(indices, dtype=np.int64)].mean(axis=0) for _, indices in groups
    ]
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for first_alternative_group_1 in range(group_counts[1] + 1):
        second_alternative_group_1 = np.arange(
            group_counts[1] - first_alternative_group_1 + 1, dtype=np.int64
        )
        residual = (
            target
            - c * first_alternative_group_1
            - d * second_alternative_group_1
        )
        valid = residual >= 0
        valid &= residual % equation_gcd == 0
        if not np.any(valid):
            continue
        second_alternative_group_0 = (
            (residual // equation_gcd) * inverse
        ) % modulus
        first_alternative_group_0 = (
            residual - b * second_alternative_group_0
        ) // a
        valid &= first_alternative_group_0 >= 0
        valid &= second_alternative_group_0 >= 0
        valid &= (
            first_alternative_group_0 + second_alternative_group_0
            <= group_counts[0]
        )
        for offset in np.flatnonzero(valid):
            counts = (
                int(first_alternative_group_0[offset]),
                int(second_alternative_group_0[offset]),
                first_alternative_group_1,
                int(second_alternative_group_1[offset]),
            )
            objective = 0.0
            for group_index, (left, right) in enumerate((counts[:2], counts[2:])):
                base_count = group_counts[group_index] - left - right
                objective += float(group_means[group_index][bases[group_index]]) * base_count
                objective += (
                    float(group_means[group_index][alternatives[group_index][0]]) * left
                )
                objective += (
                    float(group_means[group_index][alternatives[group_index][1]]) * right
                )
            candidate = (objective, counts)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None

    tier_indices = torch.empty(len(cell_ids), dtype=torch.int64)
    count_vector = best[1]
    for group_index, (_, indices) in enumerate(groups):
        quotas = {
            bases[group_index]: group_counts[group_index]
            - count_vector[group_index * 2]
            - count_vector[group_index * 2 + 1],
            alternatives[group_index][0]: count_vector[group_index * 2],
            alternatives[group_index][1]: count_vector[group_index * 2 + 1],
        }
        unassigned = set(indices)
        ordered_tiers = sorted(range(len(TIERS)), key=lambda index: (quotas[index], TIERS[index]))
        for tier_index in ordered_tiers[:-1]:
            other_tiers = [index for index in ordered_tiers if index != tier_index]
            ranked = sorted(
                unassigned,
                key=lambda index: (
                    float(
                        costs[index, tier_index]
                        - min(costs[index, other] for other in other_tiers)
                    ),
                    str(cell_ids[index]),
                ),
            )
            selected = ranked[: quotas[tier_index]]
            for index in selected:
                tier_indices[index] = tier_index
            unassigned.difference_update(selected)
        final_tier = ordered_tiers[-1]
        if len(unassigned) != quotas[final_tier]:
            raise RuntimeError("repeated-shape projection quota assignment failed")
        for index in unassigned:
            tier_indices[index] = final_tier

    assigned_cell_bytes = sum(
        int(byte_rows[index, int(tier_indices[index])]) for index in range(len(cell_ids))
    )
    if assigned_cell_bytes != cell_payload_envelope:
        raise RuntimeError("repeated-shape projection violated exact cell payload envelope")
    assignments = tuple(
        {
            "cell_id": str(cell_id),
            "tier": TIERS[int(tier_indices[index])],
            "bytes": int(byte_rows[index, int(tier_indices[index])]),
            "prediction_by_class": {
                "gate_nll": float(
                    negative_log_probabilities[index, int(tier_indices[index])]
                )
            },
        }
        for index, cell_id in enumerate(cell_ids)
    )
    counts = {
        tier: int(torch.sum(tier_indices == index)) for index, tier in enumerate(TIERS)
    }
    return HardProjection(
        tier_indices=tier_indices,
        hard_expert_bytes=assigned_cell_bytes + expert_envelope_padding_bytes,
        hard_cell_payload_bytes=assigned_cell_bytes,
        expert_envelope_padding_bytes=expert_envelope_padding_bytes,
        tier_counts=counts,
        assignments=assignments,
        solver={
            "status": "aggregate-count-exact",
            "optimality_proven": False,
            "aggregate_count_optimality_proven": True,
            "exact_envelope": True,
            "shape_count": 2,
            "aggregate_objective": best[0],
        },
    )


def project_exact_budget(
    *,
    tier_logits: torch.Tensor,
    cell_ids: Sequence[str],
    tier_bytes: torch.Tensor,
    expert_envelope_bytes: int,
    expert_envelope_padding_bytes: int = 0,
) -> HardProjection:
    logits = torch.as_tensor(tier_logits, dtype=torch.float64).detach().cpu()
    if logits.shape != (len(cell_ids), len(TIERS)) or not torch.isfinite(logits).all():
        raise ValueError("tier_logits/cell_ids projection geometry mismatch")
    if len(cell_ids) == 0 or len(set(cell_ids)) != len(cell_ids):
        raise ValueError("projection cell_ids must be non-empty and unique")
    byte_rows = _tier_byte_tensor(tier_bytes, cell_count=len(cell_ids))
    if (
        isinstance(expert_envelope_padding_bytes, bool)
        or not isinstance(expert_envelope_padding_bytes, int)
        or expert_envelope_padding_bytes < 0
        or expert_envelope_padding_bytes > expert_envelope_bytes
    ):
        raise ValueError("expert_envelope_padding_bytes must fit inside the expert envelope")
    cell_payload_envelope = expert_envelope_bytes - expert_envelope_padding_bytes
    negative_log_probabilities = -torch.log_softmax(logits, dim=-1)
    repeated_shape_projection = _project_two_repeated_shapes(
        negative_log_probabilities=negative_log_probabilities,
        cell_ids=cell_ids,
        byte_rows=byte_rows,
        cell_payload_envelope=cell_payload_envelope,
        expert_envelope_padding_bytes=expert_envelope_padding_bytes,
    )
    if repeated_shape_projection is not None:
        return repeated_shape_projection
    from .knapsack import solve_class_balanced_options

    bytes_by_option = {
        (str(cell), tier): int(byte_rows[cell_index, tier_index])
        for cell_index, cell in enumerate(cell_ids)
        for tier_index, tier in enumerate(TIERS)
    }
    costs_by_option = {
        (str(cell), tier): {"gate_nll": float(negative_log_probabilities[cell_index, tier_index])}
        for cell_index, cell in enumerate(cell_ids)
        for tier_index, tier in enumerate(TIERS)
    }
    cap = math.fsum(row["gate_nll"] for row in costs_by_option.values()) + 1.0
    solved = solve_class_balanced_options(
        cells=[str(cell) for cell in cell_ids],
        tiers=list(TIERS),
        bytes_by_option=bytes_by_option,
        class_costs_by_option=costs_by_option,
        envelope_bytes=cell_payload_envelope,
        class_caps={"gate_nll": cap},
        exact_envelope=True,
    )
    by_name = {name: index for index, name in enumerate(TIERS)}
    indices = torch.tensor(
        [by_name[str(row["tier"])] for row in solved["assignments"]], dtype=torch.long
    )
    counts = {tier: int(torch.sum(indices == index)) for index, tier in enumerate(TIERS)}
    return HardProjection(
        tier_indices=indices,
        hard_expert_bytes=int(solved["assigned_bytes"]) + expert_envelope_padding_bytes,
        hard_cell_payload_bytes=int(solved["assigned_bytes"]),
        expert_envelope_padding_bytes=expert_envelope_padding_bytes,
        tier_counts=counts,
        assignments=tuple(dict(row) for row in solved["assignments"]),
        solver=dict(solved["solver"]),
    )


def _runtime_geometry(runtime: Any, *, cell_count: int) -> tuple[list[str], list[int], list[int], torch.Tensor]:
    cell_ids = [str(value) for value in runtime.cell_ids]
    cell_layers = [int(value) for value in runtime.cell_layers]
    layers = [int(value) for value in runtime.layers]
    if len(cell_ids) != cell_count or len(set(cell_ids)) != cell_count:
        raise ValueError(f"runtime must declare exactly {cell_count} unique cells")
    if len(cell_layers) != cell_count or layers != sorted(set(layers)):
        raise ValueError("runtime cell_layers/layers geometry is invalid")
    if set(cell_layers) != set(layers):
        raise ValueError("runtime cell_layers must cover every declared layer")
    tier_bytes = _tier_byte_tensor(runtime.tier_bytes, cell_count=cell_count)
    return cell_ids, cell_layers, layers, tier_bytes


def _manifest_loss(
    runtime: Any,
    model: GateOnlyModel,
    manifest: GateDataManifest,
    *,
    cell_layers: Sequence[int],
    layers: Sequence[int],
    mode: str,
    temperature: float,
    hard_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    batches = list(runtime.batches(manifest))
    if not batches:
        raise ValueError(f"runtime produced no {manifest.split} batches")
    activations = [runtime.initial(batch) for batch in batches]
    for layer in layers:
        positions = [index for index, value in enumerate(cell_layers) if value == layer]
        layer_logits = model.tier_logits[positions]
        if mode == "soft":
            gates = torch.softmax(layer_logits / temperature, dim=-1)
            selected = gates.argmax(dim=-1)
        elif mode == "projected":
            if hard_indices is None:
                raise ValueError("projected evaluation requires hard_indices")
            selected = hard_indices[positions].to(layer_logits.device)
            gates = F.one_hot(selected, num_classes=len(TIERS)).to(layer_logits.dtype)
        elif mode == "straight_through":
            gates, _ = straight_through_categorical(layer_logits, temperature=temperature)
            selected = gates.detach().argmax(dim=-1)
        else:
            raise ValueError(f"unsupported gate forward mode {mode!r}")
        with runtime.layer_stage(layer) as forward:
            activations = [
                forward(
                    activation,
                    gates=gates,
                    hard_tiers=selected,
                    window_id=batch["window_id"],
                )
                for activation, batch in zip(activations, batches, strict=True)
            ]
    losses = [
        final_logit_teacher_kld(
            runtime.final_logits(activation, window_id=batch["window_id"]),
            runtime.teacher_logits(batch),
        )
        for activation, batch in zip(activations, batches, strict=True)
    ]
    return torch.stack(losses).mean()


def _entropy(probabilities: torch.Tensor) -> float:
    value = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1).mean()
    return float(value.detach())


def train_gate_only(
    runtime: Any,
    train_manifest: GateDataManifest,
    dev_manifest: GateDataManifest,
    config: GateTrainingConfig | None = None,
) -> GateTrainingResult:
    configuration = GateTrainingConfig() if config is None else config
    configuration.validate()
    cell_ids, cell_layers, layers, tier_bytes = _runtime_geometry(
        runtime, cell_count=configuration.cell_count
    )
    initial = runtime.initial_tier_logits() if hasattr(runtime, "initial_tier_logits") else None
    model = GateOnlyModel(configuration.cell_count, initial_logits=initial)
    gate_device = getattr(runtime, "gate_device", None)
    if gate_device is not None:
        model.to(gate_device)
    parameter_names = list(dict(model.named_parameters()))
    if parameter_names != ["tier_logits"]:
        raise RuntimeError(f"optimizer parameter allowlist mismatch: {parameter_names}")
    optimizer = torch.optim.Adam([model.tier_logits], lr=configuration.learning_rate)
    optimizer_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if any(parameter is candidate for group in optimizer.param_groups for candidate in group["params"])
    ]
    if optimizer_parameter_names != ["tier_logits"]:
        raise RuntimeError(f"optimizer parameter allowlist mismatch: {optimizer_parameter_names}")

    frozen_before = frozen_state_digest(runtime.frozen_state())
    checkpoints: list[dict[str, Any]] = []
    previous_projection: HardProjection | None = None
    byte_dual = 0.0
    last_gradient_norm = 0.0
    last_st_loss: float | None = None

    def checkpoint(step: int) -> HardProjection:
        nonlocal previous_projection
        projection = project_exact_budget(
            tier_logits=model.tier_logits,
            cell_ids=cell_ids,
            tier_bytes=tier_bytes,
            expert_envelope_bytes=configuration.expert_envelope_bytes,
            expert_envelope_padding_bytes=configuration.expert_envelope_padding_bytes,
        )
        with torch.no_grad():
            probabilities = torch.softmax(model.tier_logits / configuration.temperature, dim=-1)
            expected_expert = float(torch.sum(probabilities.cpu() * tier_bytes))
            soft_train = float(
                _manifest_loss(
                    runtime,
                    model,
                    train_manifest,
                    cell_layers=cell_layers,
                    layers=layers,
                    mode="soft",
                    temperature=configuration.temperature,
                )
            )
            hard_train = float(
                _manifest_loss(
                    runtime,
                    model,
                    train_manifest,
                    cell_layers=cell_layers,
                    layers=layers,
                    mode="projected",
                    temperature=configuration.temperature,
                    hard_indices=projection.tier_indices,
                )
            )
            hard_dev = float(
                _manifest_loss(
                    runtime,
                    model,
                    dev_manifest,
                    cell_layers=cell_layers,
                    layers=layers,
                    mode="projected",
                    temperature=configuration.temperature,
                    hard_indices=projection.tier_indices,
                )
            )
            moved = (
                0
                if previous_projection is None
                else int(torch.sum(projection.tier_indices != previous_projection.tier_indices))
            )
            checkpoints.append(
                {
                    "step": step,
                    "train_st_kld": last_st_loss,
                    "soft_train_kld": soft_train,
                    "projected_hard_train_kld": hard_train,
                    "projected_hard_dev_kld": hard_dev,
                    "moved_cells": moved,
                    "tier_counts": projection.tier_counts,
                    "gradient_l2_norm": last_gradient_norm,
                    "logit_entropy": _entropy(probabilities),
                    "byte_dual": byte_dual,
                    "expected_expert_bytes": expected_expert,
                    "expected_whole_model_bytes": configuration.fixed_dense_metadata_bytes
                    + expected_expert,
                    "hard_expert_bytes": projection.hard_expert_bytes,
                    "hard_whole_model_bytes": configuration.fixed_dense_metadata_bytes
                    + projection.hard_expert_bytes,
                    "soft_to_hard_relaxation_gap": hard_train - soft_train,
                }
            )
        previous_projection = projection
        return projection

    final_projection = checkpoint(0)
    for step in range(1, configuration.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        st_loss = _manifest_loss(
            runtime,
            model,
            train_manifest,
            cell_layers=cell_layers,
            layers=layers,
            mode="straight_through",
            temperature=configuration.temperature,
        )
        probabilities = torch.softmax(model.tier_logits / configuration.temperature, dim=-1)
        expected_expert = torch.sum(probabilities.cpu() * tier_bytes).to(st_loss.device)
        normalized_byte_gap = (
            expected_expert - configuration.expert_envelope_bytes
        ) / configuration.expert_envelope_bytes
        objective = st_loss + byte_dual * normalized_byte_gap
        objective.backward()
        if model.tier_logits.grad is None or not torch.isfinite(model.tier_logits.grad).all():
            raise RuntimeError("gate-only trainer produced an invalid tier_logits gradient")
        last_gradient_norm = float(torch.linalg.vector_norm(model.tier_logits.grad.detach()))
        last_st_loss = float(st_loss.detach())
        optimizer.step()
        byte_dual = max(
            0.0,
            byte_dual
            + configuration.byte_dual_learning_rate * float(normalized_byte_gap.detach()),
        )
        if step % configuration.dev_every == 0 or step == configuration.steps:
            final_projection = checkpoint(step)

    for manifest in (train_manifest, dev_manifest):
        if _sha256_file(manifest.path) != manifest.sha256:
            raise RuntimeError(f"immutable {manifest.split} manifest changed during training")
    frozen_after = frozen_state_digest(runtime.frozen_state())
    if frozen_after != frozen_before:
        raise RuntimeError("frozen runtime state changed during gate-only training")
    receipt = {
        "schema": "banana-smasher-ff0731-gate-only-training-v1",
        "status": "PASS",
        "objective": "frozen-own-base-final-logit-teacher-kld",
        "tiers": list(TIERS),
        "cell_count": configuration.cell_count,
        "trainable_parameter_names": ["tier_logits"],
        "trainable_parameter_shape": [configuration.cell_count, len(TIERS)],
        "optimizer_parameter_names": optimizer_parameter_names,
        "frozen_state_digest_before": frozen_before,
        "frozen_state_digest_after": frozen_after,
        "whole_model_target_bytes": configuration.whole_model_target_bytes,
        "fixed_dense_metadata_bytes": configuration.fixed_dense_metadata_bytes,
        "expert_envelope_bytes": configuration.expert_envelope_bytes,
        "expert_envelope_padding_bytes": configuration.expert_envelope_padding_bytes,
        "repair_budget_bytes": configuration.repair_budget_bytes,
        "train_manifest": {
            "path": str(train_manifest.path),
            "sha256": train_manifest.sha256,
            "class_counts": train_manifest.class_counts,
        },
        "dev_manifest": {
            "path": str(dev_manifest.path),
            "sha256": dev_manifest.sha256,
            "class_counts": dev_manifest.class_counts,
        },
        "checkpoints": checkpoints,
        "final_assignment": list(final_projection.assignments),
        "final_solver": final_projection.solver,
        "hard_kld_trace_semantics": "projected assignments; flat rows before a tier flip are not a convergence trace",
    }
    return GateTrainingResult(
        model=model,
        receipt=receipt,
        optimizer_parameter_names=optimizer_parameter_names,
        frozen_state_digest_before=frozen_before,
        frozen_state_digest_after=frozen_after,
        final_projection=final_projection,
    )


def _load_runtime_adapter(specification: str, expected_sha256: str) -> type[Any]:
    module_name, separator, class_name = specification.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError("runtime adapter must use MODULE:CLASS spelling")
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None or module_spec.origin is None:
        raise ValueError(f"cannot locate gate runtime adapter module {module_name}")
    path = Path(module_spec.origin).resolve()
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(
            f"gate runtime adapter SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    module = importlib.import_module(module_name)
    runtime = getattr(module, class_name, None)
    required = (
        "batches",
        "initial",
        "layer_stage",
        "final_logits",
        "teacher_logits",
        "frozen_state",
    )
    if (
        not isinstance(runtime, type)
        or getattr(runtime, "API_VERSION", None) != 1
        or any(not callable(getattr(runtime, name, None)) for name in required)
    ):
        raise ValueError("gate runtime adapter does not implement API v1")
    return runtime


def run_gate_training_cli(
    *,
    model_root: str | Path,
    basis_sha256: str,
    train_manifest: str | Path,
    dev_manifest: str | Path,
    runtime_adapter: str,
    runtime_adapter_sha256: str,
    runtime_config: str | Path,
    output: str | Path,
    steps: int,
    learning_rate: float,
    temperature: float,
    dev_every: int,
) -> dict[str, Any]:
    root = Path(model_root).expanduser().resolve()
    index_path = root / "model.safetensors.index.json"
    actual_basis = _sha256_file(index_path)
    if actual_basis != basis_sha256:
        raise ValueError(f"FF0731 basis mismatch: expected {basis_sha256}, got {actual_basis}")
    config_path = Path(runtime_config).expanduser().resolve()
    config_payload = config_path.read_bytes()
    runtime_parameters = json.loads(config_payload)
    if not isinstance(runtime_parameters, dict):
        raise ValueError("gate runtime config must contain a JSON object")
    runtime_class = _load_runtime_adapter(runtime_adapter, runtime_adapter_sha256)
    runtime = runtime_class(
        model_root=root,
        basis_sha256=basis_sha256,
        parameters=runtime_parameters,
    )
    train, dev = load_training_manifests(train_manifest, dev_manifest)
    result = train_gate_only(
        runtime,
        train,
        dev,
        GateTrainingConfig(
            steps=steps,
            learning_rate=learning_rate,
            temperature=temperature,
            dev_every=dev_every,
        ),
    )
    output_root = Path(output).expanduser().resolve()
    receipt = {
        **result.receipt,
        "basis_sha256": basis_sha256,
        "model_root": str(root),
        "runtime_adapter": runtime_adapter,
        "runtime_adapter_sha256": runtime_adapter_sha256,
        "runtime_config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(config_payload).hexdigest(),
        },
    }
    _atomic_bytes(output_root / "RECEIPT.json", _canonical(receipt))
    _atomic_bytes(
        output_root / "ASSIGNMENT.json",
        _canonical(
            {
                "schema": "banana-smasher-ff0731-gate-assignment-v1",
                "status": "PASS",
                "basis_sha256": basis_sha256,
                "tiers": list(TIERS),
                "byte_accounting": {
                    "expert_physical_wire_bytes": result.final_projection.hard_expert_bytes,
                    "expert_cell_payload_bytes": result.final_projection.hard_cell_payload_bytes,
                    "expert_envelope_padding_bytes": result.final_projection.expert_envelope_padding_bytes,
                    "fixed_dense_metadata_bytes": FIXED_DENSE_METADATA_BYTES,
                    "repair_bytes": REPAIR_BUDGET_BYTES,
                    "whole_model_bytes": FIXED_DENSE_METADATA_BYTES
                    + result.final_projection.hard_expert_bytes,
                    "whole_model_target_bytes": WHOLE_MODEL_TARGET_BYTES,
                },
                "assignments": list(result.final_projection.assignments),
            }
        ),
    )
    torch.save(
        {"tier_logits": result.model.tier_logits.detach().cpu()},
        output_root / "TIER_LOGITS.pt",
    )
    return {
        "status": "PASS",
        "command": "train-gates",
        "receipt": str(output_root / "RECEIPT.json"),
        "assignment": str(output_root / "ASSIGNMENT.json"),
        "tier_logits": str(output_root / "TIER_LOGITS.pt"),
        "basis_sha256": basis_sha256,
        "hard_whole_model_bytes": FIXED_DENSE_METADATA_BYTES
        + result.final_projection.hard_expert_bytes,
    }
