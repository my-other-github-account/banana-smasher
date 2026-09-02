#!/usr/bin/env python3
"""Run an immutable proven scorer under current task authority."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if path.read_bytes() != payload:
        raise RuntimeError("deployment receipt readback mismatch")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--deployment-receipt", required=True, type=Path)
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_sha):
        raise SystemExit("canonical git SHA must be 40 lowercase hex characters")
    observed = sha256(args.source)
    if observed != args.source_sha256:
        raise SystemExit(
            f"sealed rail source SHA drift: {observed} != {args.source_sha256}"
        )
    wrapper = Path(__file__).resolve()
    receipt = {
        "schema": "banana-smasher-sealed-rail-deployment-v1",
        "status": "PINNED",
        "task_id": args.task_id,
        "run_id": args.run_id,
        "canonical_git_sha": args.git_sha,
        "wrapper_path": str(wrapper),
        "wrapper_sha256": sha256(wrapper),
        "source_path": str(args.source.resolve()),
        "source_sha256": observed,
        "forwarded_argv": forwarded,
        "pid": os.getpid(),
        "created_unix": time.time(),
    }
    receipt["receipt_sha256"] = atomic_json(args.deployment_receipt, receipt)
    spec = importlib.util.spec_from_file_location("banana_smasher_proven_sealed_rail", args.source)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import sealed rail source: {args.source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    setattr(module, "TASK", args.task_id)
    setattr(module, "RUN", args.run_id)
    old_argv = sys.argv
    try:
        sys.argv = [str(args.source), *forwarded]
        return int(module.main() or 0)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
