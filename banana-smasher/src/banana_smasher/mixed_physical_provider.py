from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .production_rails import _ArtifactBinding, _ProvenSession
from .resident_balanced64 import RepairArtifact

CANONICAL_BASIS_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CANONICAL_FACTORY = "banana_smasher.mixed_physical_provider:open_provider"
CANONICAL_LAYER_SPLIT = {0: (0, 20), 1: (21, 42)}
CANONICAL_TIERS = frozenset({"native_mxfp4", "qtip2", "qtip3"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _layer_split(value: Any) -> dict[int, tuple[int, int]]:
    if not isinstance(value, Mapping):
        raise ValueError("physical mixed provider requires exact two-rank layer_split")
    try:
        result = {int(rank): tuple(int(item) for item in bounds) for rank, bounds in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("physical mixed provider requires exact two-rank layer_split") from exc
    if result != CANONICAL_LAYER_SPLIT:
        raise ValueError("physical mixed provider requires rank0 [0,20] and rank1 [21,42]")
    return result


class MixedPhysicalProvider:
    """Canonical physical bridge from a sealed mixed Backpack to resident repair.

    The mixed quantized cells stay immutable. The existing resident engine owns
    score, optimizer updates, checkpoint persistence, and phase restoration; its
    expert module is replaced only at the authenticated provider seam with the
    in-tree mixed-cell materializer.
    """

    physical_mixed_provider = True

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        identity_sha256: str,
        basis_sha256: str,
        checkpoint: str,
        checkpoint_sha256: str,
        virtual_manifest: str | Path,
        materialization_index: str | Path,
        rank: int,
        run_root: str | Path,
        config: Mapping[str, Any],
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.virtual_manifest = Path(virtual_manifest).resolve()
        self.materialization_index = Path(materialization_index).resolve()
        self.run_root = Path(run_root).resolve()
        self.identity_sha256 = str(identity_sha256)
        self.basis_sha256 = str(basis_sha256)
        self.checkpoint = str(checkpoint)
        self.checkpoint_sha256 = str(checkpoint_sha256)
        self.rank = int(rank)
        self.config = dict(config)
        self.layer_split = _layer_split(self.config.get("layer_split"))
        if self.rank not in self.layer_split:
            raise ValueError("physical mixed provider rank must be 0 or 1")
        if self.basis_sha256 != CANONICAL_BASIS_SHA256:
            raise ValueError("physical mixed provider basis identity mismatch")
        if self.virtual_manifest != self.artifact_root / "BACKPACK_VIRTUAL_MANIFEST.json":
            raise ValueError("physical mixed provider virtual manifest path mismatch")
        if self.materialization_index != self.artifact_root / "MATERIALIZATION_INDEX.jsonl":
            raise ValueError("physical mixed provider materialization index path mismatch")
        manifest = json.loads(self.virtual_manifest.read_text())
        index_sha = _sha256(self.materialization_index)
        if (
            manifest.get("basis_sha256") != self.basis_sha256
            or manifest.get("materialization_index", {}).get("sha256") != index_sha
        ):
            raise ValueError("physical mixed provider virtual/index identity mismatch")
        rows = [
            json.loads(line)
            for line in self.materialization_index.read_text().splitlines()
            if line.strip()
        ]
        tiers = {str(row.get("source_key")) for row in rows}
        cells: set[tuple[int, int, str]] = set()
        for row in rows:
            try:
                cell = (
                    int(row["layer"]),
                    int(row["expert"]),
                    str(row["projection"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "physical mixed provider requires an exact 43x256x2 cell roster"
                ) from exc
            if cell in cells:
                raise ValueError(
                    "physical mixed provider requires an exact 43x256x2 cell roster"
                )
            cells.add(cell)
        expected_cells = {
            (layer, expert, projection)
            for layer in range(43)
            for expert in range(256)
            for projection in ("down", "fused13")
        }
        if tiers != CANONICAL_TIERS or cells != expected_cells:
            raise ValueError(
                "physical mixed provider requires an exact 43x256x2 "
                "native_mxfp4+qtip2+qtip3 cell roster"
            )

        resident_root = self.config.get("resident_artifact_root")
        model_root = self.config.get("model_root")
        runtime = self.config.get("backpack_runtime")
        if not isinstance(resident_root, str) or not isinstance(model_root, str) or not isinstance(runtime, Mapping):
            raise ValueError(
                "physical mixed provider requires resident_artifact_root, model_root, and backpack_runtime"
            )
        expert_source = Path(__file__).with_name("resident_mixed_experts.py").resolve()
        self.config.update(
            {
                "basis_sha256": self.basis_sha256,
                "rank": self.rank,
                "world_size": 2,
                "authorized_api": True,
                "local_only": True,
                "layer_split": {str(key): list(value) for key, value in self.layer_split.items()},
                "resident_expert_source": str(expert_source),
                "resident_expert_source_sha256": _sha256(expert_source),
                "mixed_backpack_runtime": {
                    **dict(runtime),
                    "basis_sha256": self.basis_sha256,
                    "virtual_manifest": str(self.virtual_manifest),
                    "materialization_index": str(self.materialization_index),
                    "model_root": str(Path(model_root).resolve()),
                },
            }
        )
        self.resident_artifact = RepairArtifact.open(Path(resident_root).resolve())
        self.session = self._open_session()

    def _open_session(self) -> _ProvenSession:
        checkpoint_path = self.resident_artifact.checkpoint_path(self.checkpoint)
        if _sha256(checkpoint_path) != self.checkpoint_sha256:
            raise ValueError("mixed resident state checkpoint does not match sealed identity")
        binding = _ArtifactBinding(
            identity_sha256=self.identity_sha256,
            basis_sha256=self.basis_sha256,
            checkpoint=self.checkpoint,
            score_checkpoints={"pre": self.checkpoint, "post": self.checkpoint},
            artifact_manifest_sha256="",
            checkpoint_sha256=self.checkpoint_sha256,
            artifact_mode="mixed-backpack-virtual-v1",
            virtual_manifest_sha256=_sha256(self.virtual_manifest),
            materialization_index_sha256=_sha256(self.materialization_index),
        )
        return _ProvenSession(
            self.resident_artifact,
            binding,
            continuation_config=self.config,
            receipt_root=self.run_root,
        )

    def score(self, phase: str) -> Mapping[str, Any]:
        result = dict(self.session.score(phase))
        result["physical_provider"] = "mixed-backpack-resident-v1"
        result["rank_layer_range"] = list(self.layer_split[self.rank])
        return result

    def train(self, updates: int) -> Mapping[str, Any]:
        result = dict(self.session.train(updates))
        self.checkpoint = str(result["checkpoint"])
        self.checkpoint_sha256 = str(result["checkpoint_sha256"])
        return result

    def restore_pre_score(self, pre: Mapping[str, Any]) -> None:
        self.session._pre_kld = float(pre["mean_kld"])

    def restore_training(
        self, pre: Mapping[str, Any], training: Mapping[str, Any]
    ) -> None:
        self.checkpoint = str(training["checkpoint"])
        self.checkpoint_sha256 = str(training["checkpoint_sha256"])
        self.session = self._open_session()
        self.session._pre_kld = float(pre["mean_kld"])


def open_provider(**kwargs: Any) -> MixedPhysicalProvider:
    return MixedPhysicalProvider(**kwargs)


__all__ = [
    "CANONICAL_BASIS_SHA256",
    "CANONICAL_FACTORY",
    "CANONICAL_LAYER_SPLIT",
    "MixedPhysicalProvider",
    "open_provider",
]
