from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.mixed_physical_provider import (
    CANONICAL_BASIS_SHA256,
    MixedPhysicalProvider,
)


def _write_incomplete_chain(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    index = root / "MATERIALIZATION_INDEX.jsonl"
    index.write_text(
        "".join(
            json.dumps(
                {
                    "layer": layer,
                    "expert": 0,
                    "projection": "down",
                    "source_key": ("native_mxfp4", "qtip2", "qtip3")[layer % 3],
                },
                sort_keys=True,
            )
            + "\n"
            for layer in range(43)
        )
    )
    manifest = root / "BACKPACK_VIRTUAL_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "basis_sha256": CANONICAL_BASIS_SHA256,
                "materialization_index": {
                    "sha256": hashlib.sha256(index.read_bytes()).hexdigest()
                },
            },
            sort_keys=True,
        )
    )
    return manifest, index


def test_physical_provider_rejects_incomplete_cell_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "mixed"
    manifest, index = _write_incomplete_chain(root)
    monkeypatch.setattr(
        "banana_smasher.mixed_physical_provider.RepairArtifact.open",
        lambda _root: object(),
    )
    monkeypatch.setattr(MixedPhysicalProvider, "_open_session", lambda self: object())

    with pytest.raises(ValueError, match="43x256x2"):
        MixedPhysicalProvider(
            artifact_root=root,
            identity_sha256="1" * 64,
            basis_sha256=CANONICAL_BASIS_SHA256,
            checkpoint="UPDATE_000",
            checkpoint_sha256="2" * 64,
            virtual_manifest=manifest,
            materialization_index=index,
            rank=0,
            run_root=tmp_path / "run",
            config={
                "layer_split": {"0": [0, 20], "1": [21, 42]},
                "resident_artifact_root": str(tmp_path / "resident"),
                "model_root": str(tmp_path / "model"),
                "backpack_runtime": {},
            },
        )
