from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.artifact_identity import ArtifactIdentity
from banana_smasher.resident_admission import ADMISSION_SPEC_SCHEMA, admit_resident_artifact


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _spec(tmp_path: Path, checkpoint: Path, authenticated: Path) -> Path:
    spec = {
        "schema": ADMISSION_SPEC_SCHEMA,
        "artifact_id": "fixture-u0",
        "basis_sha256": _sha(b"basis"),
        "corpora": {
            "builder_eval_sha256": _sha(b"builder"),
            "train_score_sha256": _sha(b"score"),
            "teacher_inventory_sha256": _sha(b"teacher"),
        },
        "checkpoint": {
            "name": "UPDATE_000",
            "identity_sha256": _sha(b"checkpoint-identity"),
            "next_update": 0,
            "lock_sha256": _sha(b"lock"),
            "trajectory_sha256": _sha(b"trajectory"),
        },
        "composition": {
            "kind": "uniform-qtip-v7",
            "layers": [{"layer": layer, "tiers": {"qtip2_v7": 512}} for layer in range(43)],
        },
        "canary": {
            "reference": {"kld": 0.25, "top1": 7},
            "tolerance": {"kld_abs": 0.0, "top1_abs": 0},
        },
        "score": {"window_ids": list(range(64))},
        "authenticated_inputs": [{"path": str(authenticated), "sha256": _sha(authenticated.read_bytes())}],
        "continuations": {
            str(rank): {
                "rank": rank,
                "authorized_api": True,
                "world_size": 2,
                "local_only": True,
            }
            for rank in (0, 1)
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec, sort_keys=True))
    return path


def test_admission_generates_one_identity_and_two_verified_rank_configs(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    authenticated = tmp_path / "roster.json"
    authenticated.write_bytes(b"roster")
    spec = _spec(tmp_path, checkpoint, authenticated)
    output = tmp_path / "artifact"
    receipt = admit_resident_artifact(
        spec,
        output,
        checkpoint=checkpoint,
        checkpoint_sha256=_sha(checkpoint.read_bytes()),
    )
    identity = ArtifactIdentity.load(output)
    assert receipt["status"] == "PASS"
    assert receipt["checkpoint_path"] == str(checkpoint.resolve())
    assert receipt["checkpoint_sha256"] == _sha(checkpoint.read_bytes())
    assert receipt["artifact_identity_sha256"] == identity.sha256
    assert set(receipt["rank_configs"]) == {"0", "1"}
    for rank in (0, 1):
        config = json.loads((output / f"production-rails.rank{rank}.json").read_text())
        assert config["continuation"]["rank"] == rank
        assert list(config["allowed_artifacts"]) == [identity.sha256]
    assert not list(output.glob(".verify-rank*"))


def test_admission_binds_joint_admission_sha_from_authenticated_asset(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    asset_root = tmp_path / "asset"
    admission = asset_root / "code" / "JOINT_REPAIR_ADMISSION.json"
    admission.parent.mkdir(parents=True)
    admission.write_bytes(b'{"schema":"joint"}\n')
    spec_path = _spec(tmp_path, checkpoint, admission)
    spec = json.loads(spec_path.read_text())
    for rank in (0, 1):
        spec["continuations"][str(rank)]["asset_root"] = str(asset_root)
    spec_path.write_text(json.dumps(spec, sort_keys=True))

    output = tmp_path / "artifact"
    admit_resident_artifact(
        spec_path,
        output,
        checkpoint=checkpoint,
        checkpoint_sha256=_sha(checkpoint.read_bytes()),
    )

    expected = _sha(admission.read_bytes())
    for rank in (0, 1):
        config = json.loads((output / f"production-rails.rank{rank}.json").read_text())
        assert config["continuation"]["admission_sha256"] == expected


def test_admission_refuses_explicit_checkpoint_sha_mismatch(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    authenticated = tmp_path / "roster.json"
    authenticated.write_bytes(b"roster")
    spec = _spec(tmp_path, checkpoint, authenticated)
    with pytest.raises(ValueError, match="explicit checkpoint SHA"):
        admit_resident_artifact(
            spec,
            tmp_path / "artifact",
            checkpoint=checkpoint,
            checkpoint_sha256=_sha(b"wrong"),
        )
