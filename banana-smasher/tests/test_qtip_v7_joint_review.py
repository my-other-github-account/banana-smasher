from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.qtip_v7_joint_workflow import (
    inspect_joint_inputs,
    launch_balanced64_shards,
    materialize_joint,
    _train_joint as train_joint,
    verify_joint_checkpoint,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "inputs"
    root.mkdir(parents=True)
    layers = []
    external = []
    for layer in range(43):
        lut = root / f"L{layer:03d}.tlut.f16"
        np.zeros(1024, dtype="<f2").tofile(lut)
        layers.append(
            {
                "layer": layer,
                "path": lut.name,
                "bytes": lut.stat().st_size,
                "sha256": _sha(lut),
                "dtype": "float16",
                "shape": [1024],
            }
        )
        wire = root / f"L{layer:03d}.wire"
        wire.write_bytes(bytes([layer]))
        external.append(
            {
                "layer": layer,
                "member_count": 768,
                "complete_wire_bytes": 1,
                "identity_sha256": _sha(wire),
                "provider": "review-fixture",
                "path": wire.name,
                "bytes": 1,
                "sha256": _sha(wire),
                "members": [
                    {"expert": expert, "projection": projection}
                    for expert in range(256)
                    for projection in ("w1", "w2", "w3")
                ],
            }
        )
    manifest = root / "QTIP_V7_MANIFEST.json"
    manifest.write_bytes(
        _canonical(
            {
                "schema": "banana-smasher-qtip-v7-artifact-v1",
                "rate": 2,
                "members": [],
                "external_layers": external,
                "layer_luts": layers,
                "joint_trainable_surface": {
                    "layer_luts": {f"L{i:03d}": [1024] for i in range(43)},
                    "norms": {f"rmsnorm_{i:03d}": [2] for i in range(235)},
                    "outputs": {f"output_gain_L{i:03d}": [] for i in range(43)},
                },
            }
        )
    )
    windows = [
        {"ordinal": ordinal, "teacher_logits": [0.25 + ordinal / 1000, -0.25]}
        for ordinal in range(64)
    ]
    bank = root / "teacher-bank.json"
    bank.write_bytes(
        _canonical(
            {
                "schema": "banana-smasher-qtip-v7-teacher-bank-v1",
                "bank_id": "BALANCED64_V1",
                "teacher_sha256": "f" * 64,
                "teacher_logits_sha256": hashlib.sha256(
                    _canonical(windows)
                ).hexdigest(),
                "windows": windows,
            }
        )
    )
    return manifest, bank, tmp_path / "run"


def _external_trainer(tmp_path: Path) -> Path:
    path = tmp_path / "external-trainer.py"
    path.write_text("""#!/usr/bin/env python3
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
freeze = Path(os.environ["QTIP_V7_FREEZE"])
bank = json.loads(Path(os.environ["QTIP_V7_TEACHER_BANK"]).read_text())
out = Path(os.environ["QTIP_V7_CHECKPOINT"])
update = int(os.environ["QTIP_V7_TARGET_UPDATE"])
resume = os.environ["QTIP_V7_RESUME_FROM"]
surface = json.loads(freeze.read_text())["trainable_surface"]
if resume:
    parent = torch.load(resume, map_location="cpu", weights_only=True)
    state = parent["state"]
    parent_row = {"path": str(Path(resume).resolve()), "sha256": hashlib.sha256(Path(resume).read_bytes()).hexdigest(), "update": int(parent["update"])}
else:
    state = {group: {name: torch.full(shape, 0.1, dtype=torch.float32) for name, shape in rows.items()} for group, rows in surface.items()}
    parent_row = None
shift = 0.1 / (update + 1)
predictions = [[float(value) + (shift if i == 0 else -shift) for i, value in enumerate(row["teacher_logits"])] for row in bank["windows"]]
def softmax(values):
    a = np.asarray(values, dtype=np.float64); a -= a.max(); a = np.exp(a); return a / a.sum()
losses = []
for row, prediction in zip(bank["windows"], predictions, strict=True):
    teacher = softmax(row["teacher_logits"]); predicted = softmax(prediction)
    losses.append(float(np.sum(teacher * (np.log(teacher) - np.log(predicted)))))
continuity = {"optimizer": {"step": update}, "scheduler": {"update": update}, "rng_state": torch.Generator().manual_seed(update).get_state(), "trainer_identity": os.environ["QTIP_V7_TRAINER_SHA256"], "parent": parent_row}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"format": "banana-smasher-qtip-v7-joint-checkpoint-v1", "update": update, "objective": "teacher_kld", "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(), "teacher_kld": sum(losses) / len(losses), "predictions": predictions, "state": state, "continuity": continuity}, out)
""")
    path.chmod(0o644)
    return path


def _external_scorer(tmp_path: Path) -> Path:
    path = tmp_path / "external-scorer.py"
    path.write_text("""#!/usr/bin/env python3
import json, os
from pathlib import Path
start = int(os.environ["QTIP_V7_SHARD_START"]); end = int(os.environ["QTIP_V7_SHARD_END"])
rows = [{"ordinal": n, "positions": 1024, "support": 8192, "kld_sum_binary64": float(n + 1), "top1_matches": 900, "fallback_calls": 0, "pass_through_bytes": 0, "hidden_fp32_control_bytes": 0} for n in range(start, end + 1)]
Path(os.environ["QTIP_V7_SHARD_RECEIPT"]).write_text(json.dumps({"schema": "banana-smasher-qtip-v7-balanced64-shard-v1", "status": "PASS", "candidate_sha256": os.environ["QTIP_V7_CANDIDATE_SHA256"], "teacher_bank_sha256": os.environ["QTIP_V7_TEACHER_BANK_SHA256"], "ordinal_start": start, "ordinal_end": end, "rows": rows}, sort_keys=True) + "\\n")
""")
    path.chmod(0o755)
    return path


def test_inspect_rejects_malformed_teacher_digest_and_declaration_only_wire(
    tmp_path: Path,
) -> None:
    manifest, bank, run = _fixture(tmp_path)
    value = json.loads(bank.read_text())
    value["teacher_sha256"] = "g" * 64
    bank.write_bytes(_canonical(value))
    with pytest.raises(ValueError, match="teacher SHA-256"):
        inspect_joint_inputs(
            manifest=manifest,
            teacher_bank=bank,
            run_root=run,
            trainer_host="trainer",
        )

    manifest, bank, run = _fixture(tmp_path / "second")
    value = json.loads(manifest.read_text())
    value["external_layers"][0].pop("path")
    manifest.write_bytes(_canonical(value))
    with pytest.raises(ValueError, match="physical readback"):
        inspect_joint_inputs(
            manifest=manifest,
            teacher_bank=bank,
            run_root=run,
            trainer_host="trainer",
        )


def test_external_trainer_authenticates_kld_surface_and_resume(tmp_path: Path) -> None:
    manifest, bank, run = _fixture(tmp_path)
    frozen = inspect_joint_inputs(
        manifest=manifest,
        teacher_bank=bank,
        run_root=run,
        trainer_host="trainer",
    )
    freeze = Path(frozen["freeze"]["path"])
    trainer = _external_trainer(tmp_path)
    u0 = run / "U0.pt"
    first = train_joint(freeze=freeze, checkpoint=u0, target_update=0, trainer=trainer)
    assert first["trainer"] == "external"
    assert first["authenticated_kld_windows"] == 64

    u5 = run / "U5.pt"
    resumed = train_joint(
        freeze=freeze,
        checkpoint=u5,
        target_update=5,
        trainer=trainer,
        resume_from=u0,
    )
    assert resumed["resumed_from_update"] == 0
    assert resumed["teacher_kld"] < first["teacher_kld"]
    verified = verify_joint_checkpoint(freeze=freeze, checkpoint=u5)
    assert verified["authenticated_kld_windows"] == 64

    import torch

    u5.chmod(0o644)
    payload = torch.load(u5, map_location="cpu", weights_only=True)
    payload["state"]["norms"]["rmsnorm_000"] = torch.ones(3)
    bad = run / "bad-shape.pt"
    torch.save(payload, bad)
    bad.chmod(0o444)
    with pytest.raises(ValueError, match="shape"):
        verify_joint_checkpoint(freeze=freeze, checkpoint=bad)

    value = torch.load(u5, map_location="cpu", weights_only=True)
    value["teacher_kld"] = float(value["teacher_kld"]) + 1
    bad_kld = run / "bad-kld.pt"
    torch.save(value, bad_kld)
    bad_kld.chmod(0o444)
    with pytest.raises(RuntimeError, match="self-reported teacher_kld"):
        verify_joint_checkpoint(freeze=freeze, checkpoint=bad_kld)

    value = torch.load(u5, map_location="cpu", weights_only=True)
    value["continuity"]["parent"]["sha256"] = "0" * 64
    bad_resume = run / "bad-resume.pt"
    torch.save(value, bad_resume)
    bad_resume.chmod(0o444)
    with pytest.raises(ValueError, match="parent continuity"):
        verify_joint_checkpoint(freeze=freeze, checkpoint=bad_resume)

    value = torch.load(u5, map_location="cpu", weights_only=True)
    value["continuity"]["optimizer"]["step"] = 4
    bad_optimizer = run / "bad-optimizer.pt"
    torch.save(value, bad_optimizer)
    bad_optimizer.chmod(0o444)
    with pytest.raises(ValueError, match="optimizer continuity"):
        verify_joint_checkpoint(freeze=freeze, checkpoint=bad_optimizer)


def test_external_scorer_and_wire_accounting_distinguish_physical_from_referenced(
    tmp_path: Path,
) -> None:
    manifest, bank, run = _fixture(tmp_path)
    frozen = inspect_joint_inputs(
        manifest=manifest,
        teacher_bank=bank,
        run_root=run,
        trainer_host="trainer",
    )
    freeze = Path(frozen["freeze"]["path"])
    checkpoint = run / "U0.pt"
    train_joint(
        freeze=freeze,
        checkpoint=checkpoint,
        target_update=0,
        trainer=_external_trainer(tmp_path),
    )
    scorer = _external_scorer(tmp_path)
    launched = launch_balanced64_shards(
        candidate=checkpoint,
        freeze=freeze,
        teacher_bank=bank,
        output=run / "scores",
        workers=[f"local-a={scorer}", f"local-b={scorer}"],
    )
    assert len(launched["shards"]) == 2

    result = materialize_joint(
        freeze=freeze,
        manifest=manifest,
        checkpoint=checkpoint,
        output=run / "materialized",
    )
    assert result["physical_accounting"] == (
        "requires qtip-v7-wire verified layer receipts"
    )
    assert "referenced_wire_bytes" not in result
    assert "logical_wire_bytes" not in result
    assert "physical_stored_bytes" not in result


def test_trainer_alias_is_refused_and_failed_peer_cancels_other_workers(
    tmp_path: Path,
) -> None:
    manifest, bank, run = _fixture(tmp_path)
    frozen = inspect_joint_inputs(
        manifest=manifest,
        teacher_bank=bank,
        run_root=run,
        trainer_host="trainer",
        trainer_aliases=["127.0.0.9"],
    )
    freeze = Path(frozen["freeze"]["path"])
    checkpoint = run / "U0.pt"
    train_joint(
        freeze=freeze,
        checkpoint=checkpoint,
        target_update=0,
        trainer=_external_trainer(tmp_path),
    )
    with pytest.raises(ValueError, match="live trainer host"):
        launch_balanced64_shards(
            candidate=checkpoint,
            freeze=freeze,
            teacher_bank=bank,
            output=run / "alias-refused",
            workers=[f"side@127.0.0.9:/remote/scorer={_external_scorer(tmp_path)}"],
        )

    failing = tmp_path / "fail.py"
    failing.write_text("raise SystemExit(7)\n")
    sleeper = tmp_path / "sleep.py"
    marker = tmp_path / "must-not-exist"
    sleeper.write_text(
        "import time\nfrom pathlib import Path\ntime.sleep(5)\n"
        f"Path({str(marker)!r}).write_text('uncancelled')\n"
    )
    with pytest.raises(RuntimeError, match="exited with status 7"):
        launch_balanced64_shards(
            candidate=checkpoint,
            freeze=freeze,
            teacher_bank=bank,
            output=run / "cancel-peers",
            workers=[f"local-fail={failing}", f"local-sleep={sleeper}"],
        )
    assert not marker.exists()

    marker = tmp_path / "reverse-order-must-not-exist"
    sleeper.write_text(
        "import time\nfrom pathlib import Path\ntime.sleep(5)\n"
        f"Path({str(marker)!r}).write_text('uncancelled')\n"
    )
    with pytest.raises(RuntimeError, match="exited with status 7"):
        launch_balanced64_shards(
            candidate=checkpoint,
            freeze=freeze,
            teacher_bank=bank,
            output=run / "cancel-earlier-peer",
            workers=[f"local-sleep={sleeper}", f"local-fail={failing}"],
        )
    assert not marker.exists()


def test_recipe_has_public_u0_and_no_private_tmp_identifier() -> None:
    recipe = (
        Path(__file__).parents[2]
        / "archive"
        / "notes"
        / "qtip-v7-joint-repair-one-line-workflow.md"
    )
    text = recipe.read_text()
    assert "--target-update 0" in text
    assert "/tmp/" not in text
    assert "t_" not in text


@pytest.mark.skipif(
    "QTIP_V7_SSH_FIXTURE" not in os.environ,
    reason="requires an authenticated SSH fixture",
)
def test_authenticated_ssh_fixture_stages_hashes_and_scores(tmp_path: Path) -> None:
    manifest, bank, run = _fixture(tmp_path)
    frozen = inspect_joint_inputs(
        manifest=manifest,
        teacher_bank=bank,
        run_root=run,
        trainer_host="trainer-not-this-fixture",
    )
    freeze = Path(frozen["freeze"]["path"])
    checkpoint = run / "U0.pt"
    train_joint(
        freeze=freeze,
        checkpoint=checkpoint,
        target_update=0,
        trainer=_external_trainer(tmp_path),
    )
    result = launch_balanced64_shards(
        candidate=checkpoint,
        freeze=freeze,
        teacher_bank=bank,
        output=run / "ssh-scores",
        workers=[
            f"{os.environ['QTIP_V7_SSH_EXPECTED']}@{os.environ['QTIP_V7_SSH_FIXTURE']}:"
            f"{os.environ['QTIP_V7_SSH_ROOT']}={_external_scorer(tmp_path)}"
        ],
        remote_python=os.environ.get("QTIP_V7_SSH_PYTHON", "python3"),
    )
    assert (
        result["route_identities"][os.environ["QTIP_V7_SSH_FIXTURE"]].split(".")[0]
        == (os.environ["QTIP_V7_SSH_EXPECTED"].split(".")[0])
    )
    assert result["shards"][0]["ordinal_start"] == 0
    assert result["shards"][0]["ordinal_end"] == 63
