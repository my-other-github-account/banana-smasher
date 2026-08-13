from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from banana_smasher.cli import main
from banana_smasher.qtip_v7_joint_workflow import _same_host
from banana_smasher.qtip_v7_plan import run_joint_plan


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
            "complete_wire_bytes": 1,
            "identity_sha256": "",
            "provider": "fixture-external-wire",
            "path": f"L{layer:03d}.wire",
            "bytes": 1,
            "sha256": "",
            "members": [
                {"expert": expert, "projection": projection}
                for expert in range(256)
                for projection in ("w1", "w2", "w3")
            ],
        })
        wire = root / external[-1]["path"]
        wire.write_bytes(bytes([layer]))
        external[-1]["identity_sha256"] = _sha(wire)
        external[-1]["sha256"] = _sha(wire)
    return _json(root / "QTIP_V7_MANIFEST.json", {
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
    })


def _teacher_bank(tmp_path: Path) -> Path:
    windows = [
        {"ordinal": ordinal, "teacher_logits": [0.25 + ordinal / 1000, -0.25]}
        for ordinal in range(64)
    ]
    return _json(tmp_path / "teacher-bank.json", {
        "schema": "banana-smasher-qtip-v7-teacher-bank-v1",
        "bank_id": "BALANCED64_V1",
        "teacher_sha256": "f" * 64,
        "positions_per_window": 1024,
        "support": 8192,
        "teacher_logits_sha256": hashlib.sha256(
            (json.dumps(windows, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest(),
        "windows": windows,
    })


def test_plan_refuses_missing_teacher_target_before_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets = tmp_path / "source" / "teacher-targets"
    targets.mkdir(parents=True)
    expected = []
    for ordinal in range(64):
        name = f"t8192_win{ordinal:02d}.pt"
        path = targets / name
        path.write_bytes(f"teacher-{ordinal}".encode())
        expected.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    (targets / "t8192_win20.pt").unlink()
    trainer_called = False

    def forbidden_trainer(**_: object) -> dict[str, object]:
        nonlocal trainer_called
        trainer_called = True
        return {}

    monkeypatch.setattr(
        "banana_smasher.qtip_v7_joint_workflow.train_joint", forbidden_trainer
    )
    plan = _json(tmp_path / "plan.json", {
        "schema": "banana-smasher-qtip-v7-repair-plan-v1",
        "trainer_host": "trainer.example",
        "target_update": 5,
        "checkpoint": "checkpoints/UPDATE_005.pt",
        "inputs": [{
            "name": "teacher_targets",
            "source": str(targets),
            "destination": "inputs/teacher-targets",
            "expected": {"count": 64, "files": expected},
        }],
    })

    with pytest.raises(FileNotFoundError, match="teacher-targets/t8192_win20.pt"):
        run_joint_plan(plan=plan, run_root=tmp_path / "run")

    assert trainer_called is False
    assert not (tmp_path / "run" / "receipts" / "INPUTS_READY.json").exists()


def _plan_input(name: str, source: Path, destination: str) -> dict[str, object]:
    if source.is_dir():
        files = [
            {
                "path": str(path.relative_to(source)),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
            }
            for path in sorted(source.iterdir())
        ]
        expected: dict[str, object] = {"count": len(files), "files": files}
    else:
        expected = {"count": 1, "bytes": source.stat().st_size, "sha256": _sha(source)}
    return {"name": name, "source": str(source), "destination": destination, "expected": expected}


def test_plan_run_stages_regular_inputs_and_resumes_sealed_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    manifest = _v7_manifest(tmp_path)
    bank = _teacher_bank(tmp_path)
    trainer = _trainer(tmp_path)
    targets = tmp_path / "teacher-targets"
    targets.mkdir()
    for ordinal in range(64):
        (targets / f"t8192_win{ordinal:02d}.pt").write_bytes(f"teacher-{ordinal}".encode())
    inputs = [
        _plan_input("teacher_targets", targets, "inputs/teacher-targets"),
        _plan_input("teacher_bank", bank, "inputs/teacher-bank.json"),
        _plan_input("manifest", manifest.parent, "inputs/v7"),
        _plan_input("trainer", trainer, "inputs/trainer.py"),
    ]
    for name in ("corpus", "model", "runtime", "admission", "inventory", "roster", "planes"):
        source = tmp_path / f"{name}.json"
        source.write_text(f'{{"fixture":"{name}"}}\n')
        inputs.append(_plan_input(name, source, f"inputs/{name}.json"))
    plan = _json(tmp_path / "plan.json", {
        "schema": "banana-smasher-qtip-v7-repair-plan-v1",
        "trainer_host": "trainer.example",
        "target_update": 1,
        "checkpoint": "checkpoints/UPDATE_001.pt",
        "inputs": inputs,
        "workflow": {
            "manifest": {"input": "manifest", "path": "QTIP_V7_MANIFEST.json"},
            "teacher_bank": {"input": "teacher_bank"},
            "trainer": {"input": "trainer"},
        },
    })
    run = tmp_path / "run"

    assert main(["qtip-v7-joint-repair", "run", "--plan", str(plan), "--run-root", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    ready = json.loads((run / "receipts" / "INPUTS_READY.json").read_text())
    assert ready["inputs"]["teacher_targets"]["accepted"] == 64
    assert ready["accepted"] == ready["total"] == 160
    assert all(
        Path(row["path"]).is_file() and not Path(row["path"]).is_symlink()
        for value in ready["inputs"].values()
        for row in value["files"]
    )

    monkeypatch.setattr(
        "banana_smasher.qtip_v7_joint_workflow.train_joint",
        lambda **_: pytest.fail("sealed plan run invoked trainer again"),
    )
    assert main(["qtip-v7-joint-repair", "run", "--plan", str(plan), "--run-root", str(run)]) == 0
    capsys.readouterr()
    assert main(["qtip-v7-joint-repair", "status", "--run-root", str(run), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["first_incomplete_stage"] is None
    assert status["stages"]["INPUTS"]["accepted"] == 160
    assert status["stages"]["PRE"]["accepted"] == 1
    assert status["stages"]["TRAIN"]["accepted"] == 1


def _trainer(tmp_path: Path) -> Path:
    path = tmp_path / "trainer.py"
    path.write_text("""#!/usr/bin/env python3
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
freeze = Path(os.environ['QTIP_V7_FREEZE'])
bank = json.loads(Path(os.environ['QTIP_V7_TEACHER_BANK']).read_text())
out = Path(os.environ['QTIP_V7_CHECKPOINT'])
update = int(os.environ['QTIP_V7_TARGET_UPDATE'])
resume = os.environ['QTIP_V7_RESUME_FROM']
surface = json.loads(freeze.read_text())['trainable_surface']
if resume:
    parent = torch.load(resume, map_location='cpu', weights_only=True)
    state = parent['state']
    parent_row = {'path': str(Path(resume).resolve()), 'sha256': hashlib.sha256(Path(resume).read_bytes()).hexdigest(), 'update': int(parent['update'])}
else:
    state = {group: {name: torch.full(shape, 0.1, dtype=torch.float32) for name, shape in rows.items()} for group, rows in surface.items()}
    parent_row = None
for entries in state.values():
    for tensor in entries.values():
        tensor.add_(-0.01 * (update - (0 if parent_row is None else parent_row["update"])))
shift = 0.1 / (update + 1)
predictions = [[float(value) + (shift if i == 0 else -shift) for i, value in enumerate(row['teacher_logits'])] for row in bank['windows']]
def softmax(values):
    a = np.asarray(values, dtype=np.float64); a -= a.max(); a = np.exp(a); return a / a.sum()
losses = []
for row, prediction in zip(bank['windows'], predictions):
    teacher = softmax(row['teacher_logits']); predicted = softmax(prediction)
    losses.append(float(np.sum(teacher * (np.log(teacher) - np.log(predicted)))))
continuity = {'optimizer': {'step': update}, 'scheduler': {'update': update}, 'rng_state': torch.Generator().manual_seed(update).get_state(), 'trainer_identity': os.environ['QTIP_V7_TRAINER_SHA256'], 'parent': parent_row}
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({'format': 'banana-smasher-qtip-v7-joint-checkpoint-v1', 'update': update, 'objective': 'teacher_kld', 'freeze_sha256': hashlib.sha256(freeze.read_bytes()).hexdigest(), 'teacher_kld': sum(losses) / len(losses), 'predictions': predictions, 'state': state, 'continuity': continuity}, out)
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
rows = [{'ordinal': n, 'positions': 1024, 'support': 8192,
         'kld_sum_binary64': (0.01 + n / 100000) * 1024,
         'top1_matches': 1024, 'fallback_calls': 0,
         'pass_through_bytes': 0, 'hidden_fp32_control_bytes': 0}
        for n in range(start, end + 1)]
out.write_text(json.dumps({'schema': 'banana-smasher-qtip-v7-balanced64-shard-v1',
    'status': 'PASS', 'candidate_sha256': os.environ['QTIP_V7_CANDIDATE_SHA256'],
    'teacher_bank_sha256': os.environ['QTIP_V7_TEACHER_BANK_SHA256'],
    'ordinal_start': start, 'ordinal_end': end, 'rows': rows}, sort_keys=True) + '\\n')
""")
    path.chmod(0o755)
    return path


def _pair_worker(tmp_path: Path) -> Path:
    path = tmp_path / "pair-worker.py"
    path.write_text("""#!/usr/bin/env python3
import hashlib, json, os
from pathlib import Path
start = int(os.environ['QTIP_V7_PAIR_ORDINAL_START'])
end = int(os.environ['QTIP_V7_PAIR_ORDINAL_END'])
out = Path(os.environ['QTIP_V7_PAIR_RECEIPT'])
out.parent.mkdir(parents=True, exist_ok=True)
rows = [{'ordinal': n, 'positions': 1024, 'support': 8192,
         'kld_sum_binary64': (0.02 + n / 100000) * 1024,
         'top1_matches': 900 + n % 100, 'fallback_calls': 0,
         'pass_through_bytes': 0, 'hidden_fp32_control_bytes': 0}
        for n in range(start, end + 1)]
out.write_text(json.dumps({
    'schema': 'banana-smasher-qtip-v7-balanced64-shard-v1',
    'status': 'PASS', 'execution_mode': 'two-node-layer-major',
    'candidate_sha256': os.environ['QTIP_V7_CANDIDATE_SHA256'],
    'teacher_bank_sha256': os.environ['QTIP_V7_TEACHER_BANK_SHA256'],
    'ordinal_start': start, 'ordinal_end': end,
    'stage_a_layers': [0, 21], 'stage_b_layers': [22, 42],
    'frontier_sha256': hashlib.sha256(f'{start}:{end}'.encode()).hexdigest(),
    'rows': rows,
}, sort_keys=True) + '\\n')
""")
    path.chmod(0o755)
    return path


def _joint_checkpoint(
    path: Path, update: int = 5, *, freeze: Path | None = None
) -> Path:
    assert freeze is not None
    from banana_smasher.qtip_v7_joint_workflow import train_joint

    train_joint(
        freeze=freeze,
        checkpoint=path,
        target_update=update,
        trainer=_trainer(path.parent),
    )
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

    checkpoint5 = run / "checkpoints" / "UPDATE_005.pt"
    assert main(["qtip-v7-joint-repair", "train", "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--checkpoint", str(checkpoint5), "--target-update", "5",
                 "--trainer", str(_trainer(tmp_path))]) == 0
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
                 "--target-update", "8", "--trainer", str(_trainer(tmp_path))]) == 0
    assert json.loads(capsys.readouterr().out)["resumed_from_update"] == 5

    shard_root = run / "balanced64"
    assert main(["qtip-v7-joint-repair", "shard-launch", "--candidate", str(checkpoint5),
                 "--freeze", str(run / "FROZEN_INPUTS.json"),
                 "--teacher-bank", str(bank), "--output", str(shard_root),
                 "--worker", f"local-a={_shard_worker(tmp_path)}",
                 "--worker", f"local-b={_shard_worker(tmp_path)}"]) == 0
    launched = json.loads(capsys.readouterr().out)
    assert launched["status"] == "PASS"
    assert [(row["ordinal_start"], row["ordinal_end"]) for row in launched["shards"]] == [(0, 31), (32, 63)]

    aggregate = run / "candidate.aggregate.json"
    assert main(["qtip-v7-joint-repair", "aggregate", "--shards", str(shard_root),
                 "--output", str(aggregate)]) == 0
    measured = json.loads(capsys.readouterr().out)
    assert measured["windows"] == 64
    assert measured["positions"] == 65_536
    assert measured["support"] == 8192
    assert measured["top1_matches"] == 65_536

    baseline = _json(run / "baseline.aggregate.json", {
        **measured,
        "candidate_sha256": "0" * 64,
        "rows": [
            {
                **row,
                "kld_sum_binary64": row["kld_sum_binary64"] + 102.4,
                "top1_matches": 1023,
            }
            for row in measured["rows"]
        ],
        "mean_kld": measured["mean_kld"] + 0.1,
        "top1_matches": 65_472,
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
    assert wire["physical_accounting"] == "requires qtip-v7-wire verified layer receipts"
    assert "stored_wire_bytes" not in wire
    layer0 = np.fromfile(materialized / "L000.tlut.f16", dtype="<f2")
    layer42 = np.fromfile(materialized / "L042.tlut.f16", dtype="<f2")
    assert np.all(layer0 < np.float16(0.1))
    assert np.array_equal(layer0, layer42)
    assert (materialized / "repair_state.safetensors").is_file()


def test_joint_workflow_help_names_copy_pasteable_commands(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["qtip-v7-joint-repair", "--help"])
    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("prepare", "run", "status", "inspect", "train", "verify", "shard-launch", "aggregate", "compare", "materialize"):
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
