from __future__ import annotations

import hashlib
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .gate_only_trainer import FF0731_CELL_COUNT, FF0731_MODEL_ROOT, TIERS

_PHYSICAL_SCHEMA = "banana-smasher-ff0731-three-tier-cells-v1"
_RUNTIME_SCHEMA = "banana-smasher-ff0731-torchscript-gate-runtime-v1"
_PROJECTIONS = ("down", "fused13")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    path = Path(value).expanduser().resolve()
    if any("HOLDOUT" in part.upper() for part in path.parts):
        raise ValueError(f"HOLDOUT paths are forbidden in the FF0731 gate runtime: {path}")
    return path


def _verified_file(reference: object, *, label: str, verify_bytes: bool = True) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} must be a file reference object")
    path = _resolved_path(reference.get("path"), label=label)
    expected_sha = reference.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"{label} must declare SHA-256")
    if not path.is_file():
        raise ValueError(f"{label} file is missing: {path}")
    expected_bytes = reference.get("bytes")
    if expected_bytes is not None:
        if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
            raise ValueError(f"{label} bytes must be an integer")
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{label} byte mismatch: expected {expected_bytes}, got {actual_bytes}: {path}"
            )
    if verify_bytes:
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}: {path}"
            )
    return path


def _digest_tensor(sha256: str) -> torch.Tensor:
    return torch.tensor(list(bytes.fromhex(sha256)), dtype=torch.uint8)


class FF0731GateRuntimeAdapter:
    """Concrete staged TorchScript adapter for exact-current FF0731 gate training.

    Each immutable layer module owns the real native-MXFP4/QTIP2/QTIP3 branch
    implementation for that layer and accepts the API-v1 `(activation, gates,
    hard_tiers)` call. The adapter binds those executable modules to the complete
    per-cell physical manifest, immutable input sidecars, and final head.
    """

    API_VERSION = 1

    def __init__(
        self,
        model_root: Path,
        basis_sha256: str,
        parameters: dict[str, Any],
    ) -> None:
        if not isinstance(parameters, dict) or parameters.get("schema") != _RUNTIME_SCHEMA:
            raise ValueError(f"runtime config schema must be {_RUNTIME_SCHEMA}")
        self.model_root = Path(model_root).expanduser().resolve()
        self.basis_sha256 = basis_sha256
        self.strict_geometry = parameters.get("strict_geometry", True)
        if not isinstance(self.strict_geometry, bool):
            raise ValueError("strict_geometry must be boolean")
        if self.strict_geometry and self.model_root.name != FF0731_MODEL_ROOT:
            raise ValueError(f"production FF0731 runtime requires sole model root {FF0731_MODEL_ROOT}")
        if not isinstance(basis_sha256, str) or len(basis_sha256) != 64:
            raise ValueError("basis_sha256 must contain 64 hexadecimal characters")
        try:
            bytes.fromhex(basis_sha256)
        except ValueError as exc:
            raise ValueError("basis_sha256 must be hexadecimal") from exc
        if self.strict_geometry:
            index_path = self.model_root / "model.safetensors.index.json"
            if not index_path.is_file():
                raise ValueError(f"FF0731 model index is missing: {index_path}")
            actual_basis = _sha256_file(index_path)
            if actual_basis != basis_sha256:
                raise ValueError(
                    f"FF0731 basis mismatch: expected {basis_sha256}, got {actual_basis}"
                )

        verify_payloads = parameters.get("verify_payloads", False)
        if not isinstance(verify_payloads, bool):
            raise ValueError("verify_payloads must be boolean")
        physical_reference = parameters.get("physical_manifest")
        self.physical_manifest_path = _verified_file(
            physical_reference, label="physical manifest"
        )
        physical_payload = self.physical_manifest_path.read_bytes()
        physical_json = (
            gzip.decompress(physical_payload)
            if self.physical_manifest_path.suffix == ".gz"
            else physical_payload
        )
        physical = json.loads(physical_json)
        if (
            not isinstance(physical, dict)
            or physical.get("schema") != _PHYSICAL_SCHEMA
            or physical.get("status") != "PASS"
        ):
            raise ValueError(f"physical manifest must be a PASS {_PHYSICAL_SCHEMA} document")
        if physical.get("basis_sha256") != basis_sha256:
            raise ValueError("physical manifest basis does not match the runtime basis")
        if physical.get("tiers") != list(TIERS):
            raise ValueError(f"physical manifest tiers must be exactly {list(TIERS)}")
        cells = physical.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("physical manifest cells must be a non-empty array")
        if self.strict_geometry and len(cells) != FF0731_CELL_COUNT:
            raise ValueError("production physical manifest must contain exactly 22,016 cells")

        cell_ids: list[str] = []
        cell_layers: list[int] = []
        tier_byte_rows: list[list[int]] = []
        frozen_hashes: dict[str, str] = {
            "physical_manifest": hashlib.sha256(physical_payload).hexdigest()
        }
        observed_geometry: set[tuple[int, int, str]] = set()
        for index, row in enumerate(cells):
            if not isinstance(row, dict):
                raise ValueError(f"physical manifest cell {index} must be an object")
            cell_id = row.get("cell_id")
            layer = row.get("layer")
            expert = row.get("expert")
            projection = row.get("projection")
            if (
                not isinstance(cell_id, str)
                or isinstance(layer, bool)
                or not isinstance(layer, int)
                or isinstance(expert, bool)
                or not isinstance(expert, int)
                or projection not in _PROJECTIONS
            ):
                raise ValueError(f"physical manifest cell {index} geometry is invalid")
            expected_id = f"L{layer:03d}.E{expert:03d}.{projection}"
            if cell_id != expected_id:
                raise ValueError(
                    f"physical manifest cell ID mismatch: expected {expected_id}, got {cell_id}"
                )
            geometry = (layer, expert, str(projection))
            if cell_id in cell_ids or geometry in observed_geometry:
                raise ValueError(f"duplicate physical manifest cell: {cell_id}")
            observed_geometry.add(geometry)
            raw_tiers = row.get("tiers")
            if not isinstance(raw_tiers, dict) or list(raw_tiers) != list(TIERS):
                raise ValueError(f"cell {cell_id} tiers must be ordered exactly as {list(TIERS)}")
            byte_row: list[int] = []
            for tier in TIERS:
                tier_row = raw_tiers[tier]
                if not isinstance(tier_row, dict):
                    raise ValueError(f"cell {cell_id} tier {tier} must be an object")
                wire_bytes = tier_row.get("wire_bytes")
                if isinstance(wire_bytes, bool) or not isinstance(wire_bytes, int) or wire_bytes <= 0:
                    raise ValueError(f"cell {cell_id} tier {tier} wire_bytes must be positive")
                artifacts = tier_row.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    raise ValueError(f"cell {cell_id} tier {tier} must name physical artifacts")
                for artifact_index, artifact in enumerate(artifacts):
                    label = f"{cell_id} {tier} artifact {artifact_index}"
                    if not isinstance(artifact, dict):
                        raise ValueError(f"{label} must be an object")
                    identity_sha = artifact.get("identity_sha256", artifact.get("sha256"))
                    if not isinstance(identity_sha, str) or len(identity_sha) != 64:
                        raise ValueError(f"{label} must declare identity_sha256")
                    try:
                        bytes.fromhex(identity_sha)
                    except ValueError as exc:
                        raise ValueError(f"{label} identity_sha256 must be hexadecimal") from exc
                    path_value = artifact.get("path")
                    if path_value is not None:
                        path = _verified_file(
                            artifact,
                            label=label,
                            verify_bytes=verify_payloads,
                        )
                        artifact_name = str(path)
                    else:
                        artifact_id = artifact.get("artifact_id")
                        if not isinstance(artifact_id, str) or not artifact_id:
                            raise ValueError(f"{label} without a path must declare artifact_id")
                        if verify_payloads:
                            raise ValueError(
                                f"{label} cannot verify_payloads without a local physical path"
                            )
                        artifact_name = artifact_id
                    assert isinstance(artifact, dict)
                    frozen_hashes[
                        f"artifact:{cell_id}:{tier}:{artifact_index}:{artifact_name}"
                    ] = identity_sha
                byte_row.append(wire_bytes)
            cell_ids.append(cell_id)
            cell_layers.append(layer)
            tier_byte_rows.append(byte_row)

        if self.strict_geometry:
            expected_geometry = {
                (layer, expert, projection)
                for layer in range(43)
                for expert in range(256)
                for projection in _PROJECTIONS
            }
            if observed_geometry != expected_geometry:
                raise ValueError("production physical manifest does not cover exact FF0731 geometry")

        self.cell_ids = tuple(cell_ids)
        self.cell_layers = tuple(cell_layers)
        self.layers = tuple(sorted(set(cell_layers)))
        self.tier_bytes = torch.tensor(tier_byte_rows, dtype=torch.int64)
        configured_device = parameters.get("device", "cpu")
        self.gate_device = torch.device(configured_device)
        self._module_device = self.gate_device

        raw_layers = parameters.get("layers")
        if not isinstance(raw_layers, list) or len(raw_layers) != len(self.layers):
            raise ValueError("runtime config must declare exactly one executable module per layer")
        self._layer_modules: dict[int, Path] = {}
        for row in raw_layers:
            if not isinstance(row, dict) or isinstance(row.get("layer"), bool):
                raise ValueError("runtime layer declaration is invalid")
            layer = row.get("layer")
            if not isinstance(layer, int) or layer in self._layer_modules:
                raise ValueError("runtime layer IDs must be unique integers")
            expected_ids = [
                cell_id
                for cell_id, cell_layer in zip(self.cell_ids, self.cell_layers, strict=True)
                if cell_layer == layer
            ]
            if row.get("cell_ids") != expected_ids:
                raise ValueError(f"runtime layer {layer} cell IDs do not match the physical manifest")
            module_reference = row.get("module")
            module_path = _verified_file(module_reference, label=f"layer {layer} module")
            assert isinstance(module_reference, dict)
            frozen_hashes[f"layer_module:{layer}"] = str(module_reference["sha256"])
            self._layer_modules[layer] = module_path
        if tuple(sorted(self._layer_modules)) != self.layers:
            raise ValueError("runtime executable layer set does not match physical manifest layers")

        final_reference = parameters.get("final_head")
        self._final_head_path = _verified_file(final_reference, label="final head module")
        assert isinstance(final_reference, dict)
        frozen_hashes["final_head"] = str(final_reference["sha256"])
        self._final_head = torch.jit.load(
            str(self._final_head_path), map_location=self._module_device
        ).eval()
        self._final_head_dtype = next(
            (
                value.dtype
                for value in self._final_head.state_dict().values()
                if value.is_floating_point()
            ),
            None,
        )

        sidecars = parameters.get("data_sidecars", [])
        if not isinstance(sidecars, list):
            raise ValueError("data_sidecars must be an array")
        if self.strict_geometry and not sidecars:
            raise ValueError("production runtime must predeclare immutable data_sidecars")
        for index, reference in enumerate(sidecars):
            _verified_file(reference, label=f"data sidecar {index}")
            assert isinstance(reference, dict)
            frozen_hashes[f"data_sidecar:{index}"] = str(reference["sha256"])

        initial_reference = parameters.get("initial_tier_logits")
        self._initial_tier_logits: torch.Tensor | None = None
        if initial_reference is not None:
            initial_path = _verified_file(
                initial_reference, label="initial tier logits"
            )
            assert isinstance(initial_reference, dict)
            frozen_hashes["initial_tier_logits"] = str(initial_reference["sha256"])
            value = torch.load(initial_path, map_location="cpu", weights_only=True)
            self._initial_tier_logits = torch.as_tensor(value, dtype=torch.float32)
            if self._initial_tier_logits.shape != (len(self.cell_ids), len(TIERS)):
                raise ValueError("initial tier logits shape does not match physical cells")

        self._frozen_hashes = frozen_hashes
        self.execution_trace: list[dict[str, Any]] = []

    def initial_tier_logits(self) -> torch.Tensor | None:
        return (
            None
            if self._initial_tier_logits is None
            else self._initial_tier_logits.detach().clone()
        )

    def batches(self, manifest: Any) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in manifest.rows)

    def _load_sidecar(self, reference: object, *, label: str) -> torch.Tensor:
        path = _verified_file(reference, label=label)
        value = torch.load(path, map_location=self._module_device, weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{label} must contain one tensor")
        return value.detach()

    def initial(self, batch: dict[str, Any]) -> torch.Tensor:
        return self._load_sidecar(
            batch.get("activation"), label=f"activation for {batch.get('window_id')}"
        )

    @contextmanager
    def layer_stage(self, layer: int):
        if layer not in self._layer_modules:
            raise ValueError(f"runtime layer is not declared: {layer}")
        module = torch.jit.load(
            str(self._layer_modules[layer]), map_location=self._module_device
        ).eval()
        positions = [
            index for index, cell_layer in enumerate(self.cell_layers) if cell_layer == layer
        ]
        expected_cells = len(positions)

        def forward(
            activation: torch.Tensor,
            *,
            gates: torch.Tensor,
            hard_tiers: torch.Tensor,
            window_id: str,
        ) -> torch.Tensor:
            if gates.shape != (expected_cells, len(TIERS)):
                raise ValueError(
                    f"layer {layer} gates must have shape [{expected_cells},{len(TIERS)}]"
                )
            hard_tiers = hard_tiers.to(device=gates.device, dtype=torch.long)
            if hard_tiers.shape != (expected_cells,):
                raise ValueError(f"layer {layer} hard_tiers must have shape [{expected_cells}]")
            if torch.any(hard_tiers < 0) or torch.any(hard_tiers >= len(TIERS)):
                raise ValueError(f"layer {layer} hard_tiers are out of range")
            hard = F.one_hot(hard_tiers, num_classes=len(TIERS)).to(gates.dtype)
            hard_forward = bool(torch.equal(gates.detach(), hard))
            counts = {
                tier: int(torch.sum(hard_tiers.detach().cpu() == tier_index))
                for tier_index, tier in enumerate(TIERS)
            }
            self.execution_trace.append(
                {
                    "layer": layer,
                    "window_id": str(window_id),
                    "hard_forward": hard_forward,
                    "tier_counts": counts,
                }
            )
            result = module(
                activation.to(self._module_device),
                gates.to(self._module_device),
                hard_tiers.to(self._module_device),
            )
            if not isinstance(result, torch.Tensor):
                raise RuntimeError(f"layer {layer} TorchScript module did not return a tensor")
            return result

        try:
            yield forward
        finally:
            if self._module_device.type == "cuda":
                torch.cuda.empty_cache()

    def final_logits(self, activation: torch.Tensor, *, window_id: str) -> torch.Tensor:
        del window_id
        result = self._final_head(
            activation.to(device=self._module_device, dtype=self._final_head_dtype)
        )
        if not isinstance(result, torch.Tensor):
            raise RuntimeError("final head TorchScript module did not return a tensor")
        return result

    def teacher_logits(self, batch: dict[str, Any]) -> torch.Tensor:
        return self._load_sidecar(
            batch.get("teacher_logits"),
            label=f"teacher logits for {batch.get('window_id')}",
        )

    def frozen_state(self) -> dict[str, torch.Tensor]:
        return {
            name: _digest_tensor(sha256)
            for name, sha256 in sorted(self._frozen_hashes.items())
        }
