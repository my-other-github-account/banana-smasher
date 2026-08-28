#!/usr/bin/env python3
"""Launch the joint all-43 trainer with a sealed official-qtip-k2 expert binding."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

OFFICIAL_SOURCE_SHA256 = "00c0d888b017f7d93a0f5c214673c7579f296d3ba23274ac6355c1f80511a16b"
CANONICAL_REFERENCE = (
    "419790fad2cc5370fc2b9ec4c9b0b96652862b94:"
    "runtime/v7/runner/joint_v7_expert_base.py:67-152"
)
EXPERT_SYMBOL = "JointV7ExpertBase"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_exec(source: Path, trainer: Path, trainer_args: list[str]) -> tuple[str, list[str], dict[str, str]]:
    source = source.resolve()
    trainer = trainer.resolve()
    observed = sha256_path(source)
    if observed != OFFICIAL_SOURCE_SHA256:
        raise RuntimeError(
            f"official-qtip-k2 source mismatch {observed} != {OFFICIAL_SOURCE_SHA256}"
        )
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    env = dict(os.environ)
    env["JOINT_V7_EXPERT_BASE"] = f"{source}:{EXPERT_SYMBOL}"
    env["JOINT_ALL43_MECHANISM"] = "official-qtip-k2-1d1024-joint-all43"
    env["JOINT_ALL43_CANONICAL_REFERENCE"] = CANONICAL_REFERENCE
    argv = [sys.executable, str(trainer), *trainer_args]
    return sys.executable, argv, env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--trainer", required=True, type=Path)
    args, trainer_args = parser.parse_known_args()
    executable, argv, env = build_exec(args.source, args.trainer, trainer_args)
    os.execvpe(executable, argv, env)
    raise AssertionError("execvpe returned")


if __name__ == "__main__":
    raise SystemExit(main())
