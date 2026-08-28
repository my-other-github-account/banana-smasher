from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

_PHASES = ("score_pre", "repair_train", "score_post")
_SCHEMA = "banana-smasher-improve-phase-v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _read_phase(run_root: Path, phase: str) -> dict[str, Any]:
    path = run_root / f"{phase}.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"missing or invalid {phase} receipt: {path}") from exc
    if value.get("schema") != _SCHEMA or value.get("status") != "PASS" or value.get("phase") != phase:
        raise RuntimeError(f"{phase} receipt is not a matching PASS: {path}")
    result = value.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{phase} receipt result is not an object: {path}")
    return result


def run_improve(
    artifact_root: str | Path,
    checkpoint_sha: str,
    run_root: str | Path,
    *,
    updates: int = 45,
) -> dict[str, Any]:
    artifact = Path(artifact_root).expanduser().resolve()
    root = Path(run_root).expanduser().resolve()
    if len(checkpoint_sha) != 64:
        raise ValueError("checkpoint SHA must contain 64 hexadecimal characters")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("updates must be a positive integer")
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for phase in _PHASES:
        command = [
            sys.executable,
            "-m",
            "banana_smasher.improve",
            "--phase",
            phase,
            "--artifact-root",
            str(artifact),
            "--checkpoint-sha",
            checkpoint_sha,
            "--run-root",
            str(root),
            "--updates",
            str(updates),
        ]
        subprocess.run(command, check=True, env=env)
    pre = _read_phase(root, "score_pre")
    training = _read_phase(root, "repair_train")
    post = _read_phase(root, "score_post")
    pre_kld = float(pre["mean_kld"])
    post_kld = float(post["mean_kld"])
    improvement = {
        "pre_kld": pre_kld,
        "post_kld": post_kld,
        "delta_kld": post_kld - pre_kld,
        "improved": post_kld < pre_kld,
    }
    result = {
        "schema": "banana-smasher-improve-result-v1",
        "status": "PASS" if improvement["improved"] else "FAILED",
        "input_checkpoint_sha256": checkpoint_sha,
        "processes": list(_PHASES),
        "pre": pre,
        "training": training,
        "post": post,
        "improvement": improvement,
    }
    _atomic_json(root / "IMPROVE_RESULT.json", result)
    if not improvement["improved"]:
        raise ValueError(
            f"resident KLD did not improve: pre={pre_kld:.17g} post={post_kld:.17g}; "
            f"receipt={root / 'IMPROVE_RESULT.json'}"
        )
    return result


def _execute_phase(
    phase: str,
    artifact_root: str | Path,
    checkpoint_sha: str,
    run_root: str | Path,
    updates: int,
    *,
    api_factory=None,
) -> dict[str, Any]:
    root = Path(run_root).expanduser().resolve()
    if api_factory is None:
        from .resident_repair_api import ResidentRepairAPI

        api_factory = ResidentRepairAPI.build_uniform
    api = api_factory(
        Path(artifact_root).expanduser().resolve(),
        tier="q2",
        checkpoint_sha=checkpoint_sha,
        run_root=root,
    )
    if phase == "score_pre":
        result = dict(api.score_pre())
    elif phase == "repair_train":
        pre = _read_phase(root, "score_pre")
        api.restore_pre_score(pre)
        result = dict(api.repair_train(updates=updates))
    elif phase == "score_post":
        pre = _read_phase(root, "score_pre")
        training = _read_phase(root, "repair_train")
        api.restore_training(pre, training)
        result = dict(api.score_post())
    else:
        raise ValueError(f"unsupported improve phase: {phase}")
    receipt = {
        "schema": _SCHEMA,
        "status": "PASS",
        "phase": phase,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "input_checkpoint_sha256": checkpoint_sha,
        "result": result,
    }
    _atomic_json(root / f"{phase}.json", receipt)
    return result


def _phase_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m banana_smasher.improve")
    parser.add_argument("--phase", choices=_PHASES, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=45)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _phase_parser().parse_args(argv)
    _execute_phase(
        args.phase,
        args.artifact_root,
        args.checkpoint_sha,
        args.run_root,
        args.updates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
