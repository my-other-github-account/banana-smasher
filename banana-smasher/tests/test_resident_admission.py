from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

from banana_smasher.artifact_identity import ArtifactIdentity
from banana_smasher.resident_admission import (
    ADMISSION_SPEC_SCHEMA,
    MIXED_ADMISSION_SPEC_SCHEMA,
    admit_mixed_resident_artifact,
    admit_resident_artifact,
    provider_binding,
)


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


def test_provider_binding_changes_when_balanced64_window_roster_changes(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    authenticated = tmp_path / "roster.json"
    authenticated.write_bytes(b"roster")
    spec_path = _spec(tmp_path, checkpoint, authenticated)
    spec = json.loads(spec_path.read_text())

    _, original = provider_binding(spec)
    spec["score"]["window_ids"] = list(range(64, 128))
    _, changed = provider_binding(spec)

    assert changed != original


def test_admission_rejects_rank_scientific_runtime_digest_drift(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    authenticated = tmp_path / "roster.json"
    authenticated.write_bytes(b"roster")
    spec_path = _spec(tmp_path, checkpoint, authenticated)
    spec = json.loads(spec_path.read_text())
    spec["continuations"]["0"]["resident_expert_source_sha256"] = _sha(b"official-runtime")
    spec["continuations"]["1"]["resident_expert_source_sha256"] = _sha(b"different-runtime")
    spec_path.write_text(json.dumps(spec, sort_keys=True))

    with pytest.raises(ValueError, match="continuations scientific binding mismatch"):
        admit_resident_artifact(
            spec_path,
            tmp_path / "artifact",
            checkpoint=checkpoint,
            checkpoint_sha256=_sha(checkpoint.read_bytes()),
        )


def test_admission_auto_binds_authenticated_grouped_wrapper_next_to_expert(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    expert = tmp_path / "runtime" / "fast_v7_expert_base.py"
    wrapper = expert.with_name("fast_k2_grouped.py")
    expert.parent.mkdir()
    expert.write_bytes(b"# expert\n")
    wrapper.write_bytes(b"# grouped wrapper\n")
    spec_path = _spec(tmp_path, checkpoint, expert)
    spec = json.loads(spec_path.read_text())
    spec["authenticated_inputs"].append(
        {"path": str(wrapper), "sha256": _sha(wrapper.read_bytes())}
    )
    for rank in (0, 1):
        spec["continuations"][str(rank)].update(
            resident_expert_source=str(expert),
            resident_expert_source_sha256=_sha(expert.read_bytes()),
        )
    spec_path.write_text(json.dumps(spec, sort_keys=True))

    output = tmp_path / "artifact"
    admit_resident_artifact(
        spec_path,
        output,
        checkpoint=checkpoint,
        checkpoint_sha256=_sha(checkpoint.read_bytes()),
    )

    for rank in (0, 1):
        continuation = json.loads(
            (output / f"production-rails.rank{rank}.json").read_text()
        )["continuation"]
        assert continuation["fast_k2_wrapper_source"] == str(wrapper.resolve())
        assert continuation["fast_k2_wrapper_source_sha256"] == _sha(wrapper.read_bytes())


def test_admission_refuses_unstaged_grouped_wrapper_next_to_expert(tmp_path: Path):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    expert = tmp_path / "runtime" / "fast_v7_expert_base.py"
    expert.parent.mkdir()
    expert.write_bytes(b"# expert\n")
    spec_path = _spec(tmp_path, checkpoint, expert)
    spec = json.loads(spec_path.read_text())
    for rank in (0, 1):
        spec["continuations"][str(rank)].update(
            resident_expert_source=str(expert),
            resident_expert_source_sha256=_sha(expert.read_bytes()),
        )
    spec_path.write_text(json.dumps(spec, sort_keys=True))

    with pytest.raises(ValueError, match="authenticated grouped-K2 wrapper"):
        admit_resident_artifact(
            spec_path,
            tmp_path / "artifact",
            checkpoint=checkpoint,
            checkpoint_sha256=_sha(checkpoint.read_bytes()),
        )


def _mixed_chain(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "mixed-chain"
    root.mkdir()
    provider = tmp_path / "mixed_fixture_provider.py"
    provider.write_text(
        """events = []
class Provider:
    def __init__(self, checkpoint, checkpoint_sha256, config):
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.event_path = config.get("fixture_event_path")
        events.append(("open", checkpoint))
        self.record("open", checkpoint)
    def record(self, *row):
        if self.event_path:
            with open(self.event_path, "a") as stream:
                stream.write("|".join(str(value) for value in row) + "\\n")
    def score(self, phase):
        events.append(("score", phase, self.checkpoint))
        self.record("score", phase, self.checkpoint)
        return {
            "mean_kld": 0.25 if phase == "pre" else 0.24,
            "top1_matches": 7,
            "positions": 65536,
            "support": 8192,
            "execution_mode": "resident_model_in_memory",
            "runtime_counters": {
                "windows": 64,
                "checkpoint_loads_during_score": 0,
                "candidate_file_reads_during_score": 0,
            },
            "checkpoint": self.checkpoint,
        }
    def train(self, updates):
        events.append(("train", updates, self.checkpoint))
        self.record("train", updates, self.checkpoint)
        self.checkpoint = f"UPDATE_{updates:03d}"
        self.checkpoint_sha256 = "a" * 64
        return {"updates": updates, "checkpoint": self.checkpoint,
                "checkpoint_sha256": self.checkpoint_sha256}
    def restore_pre_score(self, pre):
        events.append(("restore_pre_score", pre["mean_kld"]))
        self.record("restore_pre_score", pre["mean_kld"])
    def restore_training(self, pre, training):
        self.checkpoint = training["checkpoint"]
        self.checkpoint_sha256 = training["checkpoint_sha256"]
        events.append(("restore_training", self.checkpoint))
        self.record("restore_training", self.checkpoint)
def open_provider(**kwargs):
    return Provider(kwargs["checkpoint"], kwargs["checkpoint_sha256"], kwargs["config"])
"""
    )
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
    virtual = {
        "schema": "banana-smasher-backpack-virtual-assignment-v1",
        "status": "PASS_LOGICAL_FULL_WIRE",
        "basis_sha256": _sha(b"basis"),
        "materialization_index": {
            "file": index.name,
            "bytes": index.stat().st_size,
            "sha256": _sha(index.read_bytes()),
        },
    }
    virtual_path = root / "BACKPACK_VIRTUAL_MANIFEST.json"
    virtual_path.write_text(json.dumps(virtual, sort_keys=True))
    identity = {
        "schema": "banana-smasher-artifact-identity-v1",
        "basis": {"model_index_sha256": _sha(b"basis")},
        "corpora": {
            "builder_eval_sha256": _sha(b"builder"),
            "train_score_sha256": _sha(b"score"),
            "u0_lock_sha256": _sha(b"lock"),
            "teacher_inventory_sha256": _sha(b"teacher"),
        },
        "checkpoints": {
            "UPDATE_000": {
                "sha256": _sha(b"checkpoint"),
                "identity_sha256": _sha(b"checkpoint-identity"),
            }
        },
        "composition": {
            "kind": "mixed-per-layer-per-expert",
            "layers": [
                {
                    "layer": layer,
                    "tiers": {
                        ("native_mxfp4", "qtip2", "qtip3")[layer % 3]: 1
                    },
                }
                for layer in range(43)
            ],
        },
        "canary": {
            "reference": {"kld": 0.25, "top1": 7},
            "tolerance": {"kld_abs": 0.02, "top1_abs": 0},
        },
        "runtime": {},
    }
    identity_path = root / "identity.json"
    identity_path.write_text(json.dumps(identity, sort_keys=True))
    spec = {
        "schema": MIXED_ADMISSION_SPEC_SCHEMA,
        "allow_test_mixed_provider": True,
        "identity_sha256": _sha(identity_path.read_bytes()),
        "virtual_manifest_sha256": _sha(virtual_path.read_bytes()),
        "materialization_index_sha256": _sha(index.read_bytes()),
        "checkpoint": "UPDATE_000",
        "score": {
            "window_ids": list(range(64)),
            "positions": 64 * 1024,
            "support": 8192,
        },
        "continuations": {
            str(rank): {
                "rank": rank,
                "authorized_api": True,
                "world_size": 2,
                "local_only": True,
                "mixed_provider_factory": "mixed_fixture_provider:open_provider",
                "mixed_provider_source": str(provider),
                "mixed_provider_source_sha256": _sha(provider.read_bytes()),
            }
            for rank in (0, 1)
        },
    }
    spec_path = tmp_path / "mixed-admission.json"
    spec_path.write_text(json.dumps(spec, sort_keys=True))
    return root, spec_path


def test_canonical_mixed_provider_drives_public_api_and_preserves_rank_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banana_smasher.mixed_physical_provider import MixedPhysicalProvider
    from banana_smasher.resident_repair_api import ResidentRepairAPI

    root, spec_path = _mixed_chain(tmp_path)
    identity = json.loads((root / "identity.json").read_text())
    identity["basis"]["model_index_sha256"] = (
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
    )
    (root / "identity.json").write_text(json.dumps(identity, sort_keys=True))
    virtual = json.loads((root / "BACKPACK_VIRTUAL_MANIFEST.json").read_text())
    virtual["basis_sha256"] = identity["basis"]["model_index_sha256"]
    index_path = root / "MATERIALIZATION_INDEX.jsonl"
    index_path.write_text(
        "".join(
            json.dumps(
                {
                    "layer": layer,
                    "expert": expert,
                    "projection": projection,
                    "source_key": ("native_mxfp4", "qtip2", "qtip3")[
                        (layer + expert) % 3
                    ],
                },
                sort_keys=True,
            )
            + "\n"
            for layer in range(43)
            for expert in range(256)
            for projection in ("down", "fused13")
        )
    )
    identity["composition"]["layers"] = [
        {
            "layer": layer,
            "tiers": {
                tier: sum(
                    2
                    for expert in range(256)
                    if ("native_mxfp4", "qtip2", "qtip3")[(layer + expert) % 3]
                    == tier
                )
                for tier in ("native_mxfp4", "qtip2", "qtip3")
            },
        }
        for layer in range(43)
    ]
    (root / "identity.json").write_text(json.dumps(identity, sort_keys=True))
    virtual["materialization_index"] = {
        "file": index_path.name,
        "bytes": index_path.stat().st_size,
        "sha256": _sha(index_path.read_bytes()),
    }
    virtual["source_bindings"] = {
        tier: {
            "basis_sha256": identity["basis"]["model_index_sha256"],
            "identity_sha256": _sha(f"{tier}-source".encode()),
        }
        for tier in ("native_mxfp4", "qtip2", "qtip3")
    }
    virtual["source_component_counts"] = {
        tier: sum(row["tiers"][tier] for row in identity["composition"]["layers"])
        for tier in ("native_mxfp4", "qtip2", "qtip3")
    }
    virtual["tier_counts"] = dict(virtual["source_component_counts"])
    (root / "BACKPACK_VIRTUAL_MANIFEST.json").write_text(
        json.dumps(virtual, sort_keys=True)
    )
    spec = json.loads(spec_path.read_text())
    spec["identity_sha256"] = _sha((root / "identity.json").read_bytes())
    spec["virtual_manifest_sha256"] = _sha(
        (root / "BACKPACK_VIRTUAL_MANIFEST.json").read_bytes()
    )
    spec["materialization_index_sha256"] = _sha(index_path.read_bytes())
    spec.pop("allow_test_mixed_provider")
    identity_manifest = tmp_path / "authenticated-mixed-identity.json"
    identity_manifest.write_bytes((root / "identity.json").read_bytes())
    (root / "identity.json").unlink()
    spec.pop("identity_sha256")
    spec["identity_manifest"] = {
        "path": str(identity_manifest),
        "sha256": _sha(identity_manifest.read_bytes()),
    }
    for rank, continuation in spec["continuations"].items():
        continuation.pop("mixed_provider_factory")
        continuation.pop("mixed_provider_source")
        continuation.pop("mixed_provider_source_sha256")
        continuation.update(
            {
                "layer_split": {"0": [0, 20], "1": [21, 42]},
                "resident_artifact_root": str(tmp_path / "resident-state"),
                "model_root": str(tmp_path / "model"),
                "backpack_runtime": {},
            }
        )
    spec_path.write_text(json.dumps(spec, sort_keys=True))

    events: list[tuple[object, ...]] = []

    class PhysicalSession:
        def score(self, phase):
            events.append(("score", phase))
            return {
                "mean_kld": 0.25 if phase == "pre" else 0.24,
                "top1_matches": 7,
                "positions": 65536,
                "support": 8192,
                "execution_mode": "resident_model_in_memory",
                "runtime_counters": {
                    "windows": 64,
                    "checkpoint_loads_during_score": 0,
                    "candidate_file_reads_during_score": 0,
                },
                "checkpoint": "UPDATE_000" if phase == "pre" else "UPDATE_045",
            }

        def train(self, updates):
            events.append(("train", updates))
            return {
                "updates": updates,
                "checkpoint": "UPDATE_045",
                "checkpoint_sha256": "a" * 64,
            }

        def restore_pre_score(self, pre):
            events.append(("restore_pre_score", pre["mean_kld"]))

        def restore_training(self, pre, training):
            events.append(("restore_training", training["checkpoint"]))

    monkeypatch.setattr(
        MixedPhysicalProvider,
        "_open_session",
        lambda self: PhysicalSession(),
    )
    monkeypatch.setattr(
        "banana_smasher.mixed_physical_provider.RepairArtifact.open",
        lambda _root: object(),
    )
    receipt = admit_mixed_resident_artifact(spec_path, root)
    assert (root / "identity.json").read_bytes() == identity_manifest.read_bytes()
    assert receipt["artifact_identity_sha256"] == _sha(identity_manifest.read_bytes())
    assert receipt["identity_manifest_source"] == str(identity_manifest.resolve())
    for rank in (0, 1):
        config = json.loads((root / f"production-rails.rank{rank}.json").read_text())
        continuation = config["continuation"]
        assert continuation["mixed_provider_factory"] == (
            "banana_smasher.mixed_physical_provider:open_provider"
        )
        assert continuation["layer_split"] == {"0": [0, 20], "1": [21, 42]}

    monkeypatch.setenv("RANK", "0")
    api = ResidentRepairAPI.build_uniform(
        root,
        tier="q2",
        checkpoint_sha=receipt["checkpoint_sha256"],
        run_root=tmp_path / "run",
    )
    assert api.score_pre()["mean_kld"] == 0.25
    assert api.repair_train(updates=45)["checkpoint"] == "UPDATE_045"
    assert api.score_post()["mean_kld"] == 0.24
    assert events == [("score", "pre"), ("train", 45), ("score", "post")]


def test_production_mixed_admission_rejects_fixture_only_provider(tmp_path: Path) -> None:
    root, spec_path = _mixed_chain(tmp_path)
    spec = json.loads(spec_path.read_text())
    spec.pop("allow_test_mixed_provider")
    spec_path.write_text(json.dumps(spec, sort_keys=True))

    with pytest.raises(ValueError, match="canonical physical mixed provider"):
        admit_mixed_resident_artifact(spec_path, root)


def test_mixed_admission_consumes_sealed_chain_and_generates_exact_rank_configs(
    tmp_path: Path,
) -> None:
    root, spec = _mixed_chain(tmp_path)
    identity_before = (root / "identity.json").read_bytes()
    virtual_before = (root / "BACKPACK_VIRTUAL_MANIFEST.json").read_bytes()
    index_before = (root / "MATERIALIZATION_INDEX.jsonl").read_bytes()

    receipt = admit_mixed_resident_artifact(spec, root)

    assert receipt["status"] == "PASS"
    assert receipt["artifact_mode"] == "mixed-backpack-virtual-v1"
    assert set(receipt["rank_configs"]) == {"0", "1"}
    for rank in (0, 1):
        config = json.loads((root / f"production-rails.rank{rank}.json").read_text())
        binding = next(iter(config["allowed_artifacts"].values()))
        assert config["continuation"]["rank"] == rank
        assert binding["artifact_mode"] == "mixed-backpack-virtual-v1"
        assert binding["physical_tiers"] == ["native_mxfp4", "qtip2", "qtip3"]
        assert binding["materialization_index_sha256"] == _sha(index_before)
    assert (root / "identity.json").read_bytes() == identity_before
    assert (root / "BACKPACK_VIRTUAL_MANIFEST.json").read_bytes() == virtual_before
    assert (root / "MATERIALIZATION_INDEX.jsonl").read_bytes() == index_before


def test_mixed_admission_rejects_mismatched_chain_identity(tmp_path: Path) -> None:
    root, spec = _mixed_chain(tmp_path)
    value = json.loads(spec.read_text())
    value["materialization_index_sha256"] = _sha(b"wrong")
    spec.write_text(json.dumps(value, sort_keys=True))

    with pytest.raises(ValueError, match="materialization index identity mismatch"):
        admit_mixed_resident_artifact(spec, root)
    assert not (root / "production-rails.rank0.json").exists()


def test_identityless_mixed_admission_rejects_unauthenticated_identity_manifest(
    tmp_path: Path,
) -> None:
    root, spec_path = _mixed_chain(tmp_path)
    identity_manifest = tmp_path / "authenticated-mixed-identity.json"
    identity_manifest.write_bytes((root / "identity.json").read_bytes())
    (root / "identity.json").unlink()
    spec = json.loads(spec_path.read_text())
    spec.pop("identity_sha256")
    spec["identity_manifest"] = {
        "path": str(identity_manifest),
        "sha256": _sha(b"different identity bytes"),
    }
    spec_path.write_text(json.dumps(spec, sort_keys=True))

    with pytest.raises(ValueError, match="identity manifest identity mismatch"):
        admit_mixed_resident_artifact(spec_path, root)
    assert not (root / "identity.json").exists()
    assert not (root / "production-rails.rank0.json").exists()


def test_mixed_admission_runtime_uses_one_physical_provider_for_all_phases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banana_smasher.resident_repair_api import ResidentRepairAPI

    root, spec = _mixed_chain(tmp_path)
    receipt = admit_mixed_resident_artifact(spec, root)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("RANK", "0")
    sys.modules.pop("mixed_fixture_provider", None)

    api = ResidentRepairAPI.build_uniform(
        root,
        tier="q2",
        checkpoint_sha=receipt["checkpoint_sha256"],
        run_root=tmp_path / "run",
    )
    assert isinstance(api, ResidentRepairAPI)
    assert api.score_pre()["mean_kld"] == 0.25
    assert api.repair_train(updates=45)["checkpoint"] == "UPDATE_045"
    assert api.score_post()["mean_kld"] == 0.24

    provider = importlib.import_module("mixed_fixture_provider")
    assert provider.events == [
        ("open", "UPDATE_000"),
        ("score", "pre", "UPDATE_000"),
        ("train", 45, "UPDATE_000"),
        ("score", "post", "UPDATE_045"),
    ]
    sys.modules.pop("mixed_fixture_provider", None)


def test_canonical_improve_cli_reaches_mixed_provider_in_each_fresh_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banana_smasher.improve import run_improve

    root, spec_path = _mixed_chain(tmp_path)
    event_path = tmp_path / "provider-events.log"
    spec = json.loads(spec_path.read_text())
    for continuation in spec["continuations"].values():
        continuation["fixture_event_path"] = str(event_path)
    spec_path.write_text(json.dumps(spec, sort_keys=True))
    receipt = admit_mixed_resident_artifact(spec_path, root)
    monkeypatch.setenv("RANK", "0")
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(tmp_path), source_root, os.environ.get("PYTHONPATH", ""))),
    )

    result = run_improve(
        root,
        receipt["checkpoint_sha256"],
        tmp_path / "cli-run",
        updates=45,
    )

    assert result["status"] == "PASS"
    assert event_path.read_text().splitlines() == [
        "open|UPDATE_000",
        "score|pre|UPDATE_000",
        "open|UPDATE_000",
        "restore_pre_score|0.25",
        "train|45|UPDATE_000",
        "open|UPDATE_000",
        "restore_training|UPDATE_045",
        "score|post|UPDATE_045",
    ]


def test_public_resident_admit_mixed_cli_generates_rank_configs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from banana_smasher.cli import main

    root, spec = _mixed_chain(tmp_path)

    assert main(
        [
            "resident",
            "admit-mixed",
            "--spec",
            str(spec),
            "--artifact-root",
            str(root),
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["command"] == "resident admit-mixed"
    assert set(result["rank_configs"]) == {"0", "1"}
