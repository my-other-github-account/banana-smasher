from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from banana_smasher.cli import main
from banana_smasher.improve import _execute_phase, _initialize_distributed_from_env, run_improve


CHECKPOINT_SHA = "f" * 64


def test_phase_initializes_distributed_pair_from_documented_launcher_env(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    class FakeDistributed:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def is_initialized() -> bool:
            return False

        @staticmethod
        def init_process_group(*, backend: str, init_method: str) -> None:
            events.append((backend, init_method))

    monkeypatch.setenv("WORLD_SIZE", "2")
    _initialize_distributed_from_env(FakeDistributed())
    assert events == [("nccl", "env://")]


def test_two_process_gloo_is_initialized_during_api_and_destroyed_after(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    assert torch.distributed.is_available()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    script = textwrap.dedent(
        """
        import json, os, pathlib, torch.distributed as dist
        from banana_smasher.improve import _execute_phase
        class API:
            def score_pre(self):
                assert dist.is_initialized()
                return {"mean_kld": 0.2284983253897188}
        root = pathlib.Path(os.environ["PHASE_ROOT"])
        _execute_phase("score_pre", root / "artifact", "f" * 64, root, 1,
                       api_factory=lambda *args, **kwargs: API())
        assert not dist.is_initialized()
        print(json.dumps({"rank": int(os.environ["RANK"]), "destroyed": True}))
        """
    )
    processes = []
    for rank in range(2):
        env = dict(os.environ)
        env.update(
            RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2",
            MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
            BANANA_SMASHER_DISTRIBUTED_BACKEND="gloo",
            PHASE_ROOT=str(tmp_path / f"rank{rank}"),
        )
        processes.append(subprocess.Popen([sys.executable, "-c", script], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    rows = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        rows.append(json.loads(stdout))
    assert rows == [{"rank": 0, "destroyed": True}, {"rank": 1, "destroyed": True}]


def _write_phase(path: Path, phase: str, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "banana-smasher-improve-phase-v1", "status": "PASS", "phase": phase, "result": value}))


def test_run_improve_uses_three_fresh_processes_and_seals_verdict(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, *, check, env):
        calls.append(list(command))
        phase = command[command.index("--phase") + 1]
        run_root = Path(command[command.index("--run-root") + 1])
        values = {
            "score_pre": {"mean_kld": 0.2284983253897188, "top1_matches": 56533},
            "repair_train": {"updates": 45, "checkpoint": "UPDATE_049", "checkpoint_sha256": "a" * 64},
            "score_post": {"mean_kld": 0.211277616743619, "top1_matches": 56508},
        }
        _write_phase(run_root / f"{phase}.json", phase, values[phase])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    receipt = run_improve(tmp_path / "artifact", CHECKPOINT_SHA, tmp_path / "run", updates=45)

    assert [row[row.index("--phase") + 1] for row in calls] == ["score_pre", "repair_train", "score_post"]
    assert all(row[:3] == [sys.executable, "-m", "banana_smasher.improve"] for row in calls)
    assert receipt["status"] == "PASS"
    assert receipt["improvement"]["post_kld"] < receipt["improvement"]["pre_kld"]
    assert json.loads((tmp_path / "run" / "IMPROVE_RESULT.json").read_text()) == receipt


def test_run_improve_fails_nonzero_when_post_does_not_improve(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, *, check, env):
        phase = command[command.index("--phase") + 1]
        run_root = Path(command[command.index("--run-root") + 1])
        value = {"score_pre": {"mean_kld": 0.2}, "repair_train": {"updates": 45}, "score_post": {"mean_kld": 0.2}}[phase]
        _write_phase(run_root / f"{phase}.json", phase, value)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="did not improve"):
        run_improve(tmp_path / "artifact", CHECKPOINT_SHA, tmp_path / "run", updates=45)
    assert json.loads((tmp_path / "run" / "IMPROVE_RESULT.json").read_text())["status"] == "FAILED"


def test_execute_score_probe_forwards_only_explicit_windows(tmp_path: Path) -> None:
    calls: list[tuple[int, ...]] = []

    class API:
        def score_probe(self, windows):
            calls.append(tuple(windows))
            return {"mean_kld": 0.09936928004026413, "positions": 1024}

    result = _execute_phase(
        "score_probe",
        tmp_path / "artifact",
        CHECKPOINT_SHA,
        tmp_path / "run",
        45,
        (28,),
        api_factory=lambda *args, **kwargs: API(),
    )

    assert calls == [(28,)]
    assert result["positions"] == 1024
    receipt = json.loads((tmp_path / "run" / "score_probe.json").read_text())
    assert receipt["phase"] == "score_probe"
    assert receipt["result"]["mean_kld"] == 0.09936928004026413


def test_smash_improve_is_one_documented_command(tmp_path: Path, monkeypatch, capsys) -> None:
    expected = {"status": "PASS", "improvement": {"improved": True}}
    monkeypatch.setattr("banana_smasher.improve.run_improve", lambda *args, **kwargs: expected)
    assert main(["improve", str(tmp_path / "artifact"), "--checkpoint-sha", CHECKPOINT_SHA, "--run-root", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_execute_phase_restores_only_sealed_prior_phase_state(tmp_path: Path) -> None:
    events: list[tuple] = []

    class FakeAPI:
        def score_pre(self):
            events.append(("score_pre",))
            return {"mean_kld": 0.2284983253897188}

        def restore_pre_score(self, pre):
            events.append(("restore_pre_score", pre["mean_kld"]))

        def repair_train(self, *, updates):
            events.append(("repair_train", updates))
            return {"updates": updates, "checkpoint": "UPDATE_049", "checkpoint_sha256": "a" * 64}

        def restore_training(self, pre, training):
            events.append(("restore_training", pre["mean_kld"], training["updates"]))

        def score_post(self):
            events.append(("score_post",))
            return {"mean_kld": 0.211277616743619}

    def factory(artifact_root, *, tier, checkpoint_sha, run_root):
        events.append(("build_uniform", Path(artifact_root), tier, checkpoint_sha, Path(run_root)))
        return FakeAPI()

    root = tmp_path / "run"
    assert _execute_phase("score_pre", tmp_path / "artifact", CHECKPOINT_SHA, root, 45, api_factory=factory)["mean_kld"] == 0.2284983253897188
    assert _execute_phase("repair_train", tmp_path / "artifact", CHECKPOINT_SHA, root, 45, api_factory=factory)["updates"] == 45
    assert _execute_phase("score_post", tmp_path / "artifact", CHECKPOINT_SHA, root, 45, api_factory=factory)["mean_kld"] == 0.211277616743619
    assert ("restore_pre_score", 0.2284983253897188) in events
    assert ("restore_training", 0.2284983253897188, 45) in events
    assert json.loads((root / "score_post.json").read_text())["status"] == "PASS"
