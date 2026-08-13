from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from banana_smasher.cli import main
from banana_smasher.qtip_v7_joint_workflow import _same_host


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _v7_manifest(tmp_path: Path) -> Path:
    root = tmp_path / "v7"
    root.mkdir()
    layers = []
    external = []
    for layer in range(43):
        lut = root / f"L{layer:03d}.tlut.f16"
        np.linspace(-1, 1, 1024, dtype=np.float16).tofile(lut)
        layers.append({
            "layer": layer,
            "path": lut.name,
            "bytes": lut.stat().st_size,
            "sha256": _sha(lut),
            "dtype": "float16",
            "shape": [1024],
        })
        external.append({
            "layer": layer,
            "member_count": 768,
            "complete_wire_bytes": 768 * 2_109_444,
            "identity_sha256": f"{layer + 1:064x}",
            "provider": "fixture-external-wire",
        })
    return _json(root / "QTIP_V7_MANIFEST.json", {
        "schema": "banana-smasher-qtip-v7-artifact-v1",
        "rate": 2,
        "members": [],
        "external_layers": external,
        "layer_luts": layers,
    })


def _teacher_bank(tmp_path: Path) -> Path:
    return _json(tmp_path / "teacher-bank.json", {
        "schema": "banana-smasher-qtip-v7-teacher-bank-v1",
        "bank_id": "BALANCED64_V1",
        "teacher_sha256": "f" * 64,
        "windows": list(range(64)),
    })


def _trainer(tmp_path: Path) -> Path:
    path = tmp_path / "trainer.py"
    path.write_text("""#!/usr/bin/env python3
import os
from pathlib import Path
import torch
update = int(os.environ['QTIP_V7_TARGET_UPDATE'])
out = Path(os.environ['QTIP_V7_CHECKPOINT'])
out.parent.mkdir(parents=True, exist_ok=True)
state = {
    'layer_luts': {f'L{i:03d}': torch.full((1024,), i + update / 100, dtype=torch.float32) for i in reversed(range(43))},
    'norms': {f'norm_{i:03d}': torch.ones(2, dtype=torch.float32) for i in range(235)},
    'outputs': {f'gain_{i:03d}': torch.tensor(0.0, dtype=torch.float32) for i in range(43)},
}
freeze_sha = __import__('hashlib').sha256(Path(os.environ['QTIP_V7_FREEZE']).read_bytes()).hexdigest()
torch.save({'format': 'banana-smasher-qtip-v7-joint-checkpoint-v1', 'update': update,
            'objective': 'teacher_kld', 'freeze_sha256': freeze_sha,
            'teacher_kld': 0.25 / (update + 1), 'state': state}, out)
""")
    path.chmod(0o755)
    return path


def _shard_worker(tmp_path: Path) -> Path:
    path = tmp_path / "shard-worker.py"
    path.write_text("""#!/usr/bin/env python3
import json, os
from pathlib import Path
start = int(os.environ['QTIP_V7_SHARD_START'])
end = int(os.environ['QTIP_V7_SHARD_END'])
out = Path(os.environ['QTIP_V7_SHARD_RECEIPT'])
out.parent.mkdir(parents=True, exist_ok=True)
rows = [{'ordinal': n, 'mean_kld': 0.01 + n / 100000, 'top1_match': 1} for n in range(start, end + 1)]
out.write_text(json.dumps({'schema': 'banana-smasher-qtip-v7-balanced64-shard-v1',
    'status': 'PASS', 'candidate_sha256': os.environ['QTIP_V7_CANDIDATE_SHA256'],
    'teacher_bank_sha256': os.environ['QTIP_V7_TEACHER_BANK_SHA256'],
    'ordinal_start': start, 'ordinal_end': end, 'rows': rows}, sort_keys=True) + '\\n')
""")
    path.chmod(0o755)
    return path


def _joint_checkpoint(
    path: Path, update: int = 5, *, freeze: Path | None = None
) -> Path:
    state = {
        "layer_luts": {
            f"L{i:03d}": torch.full((1024,), i + update / 100, dtype=torch.float32)
            for i in range(43)
        },
        "norms": {
            f"norm_{i:03d}": torch.ones(2, dtype=torch.float32) for i in range(235)
        },
        "outputs": {
            f"gain_{i:03d}": torch.tensor(0.0, dtype=torch.float32)
            for i in range(43)
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "banana-smasher-qtip-v7-joint-checkpoint-v1",
            "update": update,
            "objective": "teacher_kld",
            "freeze_sha256": _sha(freeze) if freeze is not None else "0" * 64,
            "teacher_kld": 1.0 / (update + 1),
            "state": state,
        },
        path,
    )
    path.chmod(0o444)
    return path


def test_public_joint_workflow_end_to_end(tmp_path: Path, capsys) -> None:
    manifest = _v7_manifest(tmp_path)
    bank = _teacher_bank(tmp_path)
    run = tmp_path / "run"

    assert main(["qtip-v7-joint-repair", "inspect", "--manifest", str(manifest),
                 "--teacher-bank", str(bank), "--run-root", str(run),
                 "--trainer-host", "192.168.200.9"]) == 0
    frozen = json.loads(capsys.readouterr().out)
    assert frozen["status"] == "PASS"
    assert frozen["inventory"] == {
        "layers": 43, "layer_luts": 43, "rmsnorm_masters": 235, "output_gains": 43
    }
    assert frozen["teacher_bank"]["windows"] == 64

    trainer = _trainer(tmp_path)
    checkpoint5 = run / "checkpoints" / "UPDATE_005.pt"
    assert main(["qtip-v7-joint-repair", "train", "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--checkpoint", str(checkpoint5), "--target-update", "5", "--trainer", str(trainer)]) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained["status"] == "PASS"
    assert trained["update"] == 5
    assert trained["objective"] == "teacher_kld"
    receipt5 = Path(trained["receipt"])
    assert receipt5.is_file()
    assert checkpoint5.stat().st_mode & 0o222 == 0
    assert receipt5.stat().st_mode & 0o222 == 0

    assert main(["qtip-v7-joint-repair", "verify", "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--checkpoint", str(checkpoint5), "--receipt", str(receipt5)]) == 0
    assert json.loads(capsys.readouterr().out)["checkpoint_sha256"] == _sha(checkpoint5)

    checkpoint8 = run / "checkpoints" / "UPDATE_008.pt"
    assert main(["qtip-v7-joint-repair", "train", "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--checkpoint", str(checkpoint8), "--resume-from", str(checkpoint5),
                 "--target-update", "8", "--trainer", str(trainer)]) == 0
    assert json.loads(capsys.readouterr().out)["resumed_from_update"] == 5

    worker = _shard_worker(tmp_path)
    shard_root = run / "balanced64"
    assert main(["qtip-v7-joint-repair", "shard-launch", "--candidate", str(checkpoint5),
                 "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--teacher-bank", str(bank), "--output", str(shard_root),
                 "--worker", f"local-a={worker}", "--worker", f"local-b={worker}"]) == 0
    launched = json.loads(capsys.readouterr().out)
    assert launched["status"] == "PASS"
    assert [(row["ordinal_start"], row["ordinal_end"]) for row in launched["shards"]] == [(0, 31), (32, 63)]

    aggregate = run / "candidate.aggregate.json"
    assert main(["qtip-v7-joint-repair", "aggregate", "--shards", str(shard_root),
                 "--output", str(aggregate)]) == 0
    measured = json.loads(capsys.readouterr().out)
    assert measured["windows"] == 64
    assert measured["top1_matches"] == 64

    baseline = _json(run / "baseline.aggregate.json", {
        **measured,
        "candidate_sha256": "0" * 64,
        "rows": [
            {
                **row,
                "mean_kld": row["mean_kld"] + 0.1,
                "top1_match": int(row["ordinal"] != 0),
            }
            for row in measured["rows"]
        ],
        "mean_kld": measured["mean_kld"] + 0.1,
        "top1_matches": 63,
    })
    champion = run / "champion.json"
    assert main(["qtip-v7-joint-repair", "compare", "--baseline", str(baseline),
                 "--candidate", str(aggregate), "--output", str(champion)]) == 0
    assert json.loads(capsys.readouterr().out)["champion"] == "candidate"

    materialized = run / "materialized-u5"
    assert main(["qtip-v7-joint-repair", "materialize", "--manifest", str(manifest),
                 "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--checkpoint", str(checkpoint5), "--output", str(materialized)]) == 0
    wire = json.loads(capsys.readouterr().out)
    assert wire["status"] == "PASS"
    assert wire["stored_wire_bytes"] == wire["qtip_wire_bytes"] + wire["dense_repair_bytes"]
    assert wire["wire_size_delta"] == 0
    layer0 = np.fromfile(materialized / "L000.tlut.f16", dtype="<f2")
    layer42 = np.fromfile(materialized / "L042.tlut.f16", dtype="<f2")
    assert np.all(layer0 == np.float16(0.05))
    assert np.all(layer42 == np.float16(42.05))
    assert (materialized / "repair_state.safetensors").is_file()


def test_joint_workflow_help_names_copy_pasteable_commands(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["qtip-v7-joint-repair", "--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("inspect", "train", "verify", "shard-launch", "aggregate", "compare", "materialize"):
        assert command in help_text
    assert "43 LUTs" in help_text
    assert "235 RMSNorm" in help_text
    assert "teacher KLD" in help_text


def test_joint_checkpoint_requires_teacher_kld(tmp_path: Path, capsys) -> None:
    manifest = _v7_manifest(tmp_path)
    bank = _teacher_bank(tmp_path)
    run = tmp_path / "run"
    assert main(["qtip-v7-joint-repair", "inspect", "--manifest", str(manifest),
                 "--teacher-bank", str(bank), "--run-root", str(run),
                 "--trainer-host", "192.168.200.9"]) == 0
    capsys.readouterr()
    bad = tmp_path / "bad.pt"
    torch.save({"format": "banana-smasher-qtip-v7-joint-checkpoint-v1", "update": 1,
                "state": {"layer_luts": {}, "norms": {}, "outputs": {}}}, bad)
    bad.chmod(0o444)
    assert main(["qtip-v7-joint-repair", "verify", "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--checkpoint", str(bad)]) == 2
    assert "teacher_kld" in json.loads(capsys.readouterr().err)["error"]


def test_shard_launch_refuses_live_trainer_fabric_host_dot9(
    tmp_path: Path, capsys
) -> None:
    manifest = _v7_manifest(tmp_path)
    bank = _teacher_bank(tmp_path)
    run = tmp_path / "run"
    assert main([
        "qtip-v7-joint-repair", "inspect",
        "--manifest", str(manifest),
        "--teacher-bank", str(bank),
        "--run-root", str(run),
        "--trainer-host", "192.168.200.9",
    ]) == 0
    capsys.readouterr()
    checkpoint = _joint_checkpoint(
        run / "UPDATE_005.pt", freeze=run / "FROZEN_INPUTS.json"
    )
    worker = _shard_worker(tmp_path)

    assert main([
        "qtip-v7-joint-repair", "shard-launch",
        "--candidate", str(checkpoint),
        "--freeze", str(run / "FROZEN_INPUTS.json"),
        "--teacher-bank", str(bank),
        "--output", str(run / "must-not-exist"),
        "--worker", f"spark-8@192.168.200.9:/dev/shm/forbidden={worker}",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_type"] == "ValueError"
    assert error["error"] == (
        "refusing BALANCED64 shard worker on live trainer host 192.168.200.9"
    )
    assert not (run / "must-not-exist").exists()


def test_host_identity_does_not_alias_distinct_canonical_fabric_addresses() -> None:
    assert _same_host("192.168.200.9", "192.168.200.9")
    assert not _same_host("192.168.200.1", "192.168.200.9")
    assert not _same_host("192.168.200.3", "192.168.200.9")
    assert not _same_host("192.168.200.4", "192.168.200.9")
    assert not _same_host("192.168.200.8", "192.168.200.9")


def test_shard_launch_rejects_shell_unsafe_remote_roots(tmp_path: Path) -> None:
    from banana_smasher.qtip_v7_joint_workflow import _parse_worker

    worker = _shard_worker(tmp_path)
    for root in ("/tmp/a b", "/tmp/a;id", "/tmp/a$(id)", "/tmp/../x"):
        with pytest.raises(ValueError, match="shell-safe absolute REMOTE_ROOT"):
            _parse_worker(f"spark-1@192.168.200.1:{root}={worker}")


def test_shard_launch_refuses_trainer_hostname_via_expected_route_identity(
    tmp_path: Path, capsys
) -> None:
    manifest = _v7_manifest(tmp_path)
    bank = _teacher_bank(tmp_path)
    run = tmp_path / "run"
    assert main([
        "qtip-v7-joint-repair", "inspect",
        "--manifest", str(manifest),
        "--teacher-bank", str(bank),
        "--run-root", str(run),
        "--trainer-host", "spark-8",
    ]) == 0
    capsys.readouterr()
    checkpoint = _joint_checkpoint(
        run / "UPDATE_005.pt", freeze=run / "FROZEN_INPUTS.json"
    )
    worker = _shard_worker(tmp_path)

    assert main([
        "qtip-v7-joint-repair", "shard-launch",
        "--candidate", str(checkpoint),
        "--freeze", str(run / "FROZEN_INPUTS.json"),
        "--teacher-bank", str(bank),
        "--output", str(run / "must-not-exist"),
        "--worker", f"spark-8@192.168.200.9:/dev/shm/forbidden={worker}",
    ]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error_type"] == "ValueError"
    assert error["error"] == (
        "refusing BALANCED64 shard worker on live trainer host spark-8"
    )
    assert not (run / "must-not-exist").exists()
