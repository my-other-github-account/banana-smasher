"""Fork two public score ranks after one ordinary CPU checkpoint load."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping

from .balanced64 import ArtifactError


def prepare_ordinary_checkpoint_payload(path: str | Path, expected_sha256: str) -> Mapping[str, Any]:
    """Hash-bind and ordinarily materialize one payload before rank fork."""
    from .official_k2_resident_score import _install_ordinary_fork_payload

    return _install_ordinary_fork_payload(Path(path), expected_sha256)


def run_forked_rank_children(
    rank_main: Callable[[int], int], *, ranks: Iterable[int] = (0, 1)
) -> dict[str, Any]:
    """Run rank callables in fork children; on failure terminate and reap peers."""
    selected = tuple(int(rank) for rank in ranks)
    if selected != (0, 1):
        raise ArtifactError("ordinary-load fork broker requires exact ranks (0, 1)")
    pids: dict[int, int] = {}
    rank_by_pid: dict[int, int] = {}
    for rank in selected:
        pid = os.fork()
        if pid == 0:
            try:
                code = int(rank_main(rank))
            except BaseException:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                code = 125
            os._exit(code if 0 <= code <= 255 else 125)
        pids[rank] = pid
        rank_by_pid[pid] = rank

    remaining = set(rank_by_pid)
    exit_codes: dict[int, int] = {}
    failed = False
    while remaining:
        pid, status = os.waitpid(-1, 0)
        if pid not in remaining:
            continue
        remaining.remove(pid)
        code = os.waitstatus_to_exitcode(status)
        exit_codes[rank_by_pid[pid]] = code
        if code != 0 and not failed:
            failed = True
            for peer in tuple(remaining):
                try:
                    os.kill(peer, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    return {
        "status": "FAIL" if failed else "PASS",
        "pids": pids,
        "exit_codes": exit_codes,
        "all_children_reaped": True,
    }


def run_same_process_dual_shard(rank_main: Callable[[int], int]) -> dict[str, Any]:
    """Run the local-CUDA dual shard scorer without creating rank processes."""
    code = int(rank_main(1))
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "pids": {0: os.getpid(), 1: os.getpid()},
        "exit_codes": {0: code, 1: code},
        "all_children_reaped": True,
        "same_process_dual_shard": True,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _parse_checkpoint(value: str) -> tuple[Path, str]:
    try:
        path, sha256 = value.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint must be PATH=SHA256") from exc
    if not path or len(sha256) != 64:
        raise argparse.ArgumentTypeError("checkpoint must be PATH=SHA256")
    return Path(path), sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=_parse_checkpoint)
    parser.add_argument("--rank-argv-json", action="append", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--same-process-dual-shard", action="store_true")
    args = parser.parse_args(argv)
    if len(args.rank_argv_json) != 2:
        parser.error("exactly two --rank-argv-json values are required")
    rank_argv = [json.loads(value) for value in args.rank_argv_json]
    if any(not isinstance(value, list) or not all(isinstance(item, str) for item in value) for value in rank_argv):
        parser.error("each --rank-argv-json must encode a string array")

    prepared = [dict(prepare_ordinary_checkpoint_payload(path, sha256)) for path, sha256 in args.checkpoint]
    started = time.time()

    def rank_main(rank: int) -> int:
        from .cli import main as public_main

        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        if args.same_process_dual_shard:
            os.environ["BANANA_SMASHER_SAME_PROCESS_DUAL_SHARD"] = "1"
        return int(public_main(rank_argv[rank]))

    result = (
        run_same_process_dual_shard(rank_main)
        if args.same_process_dual_shard
        else run_forked_rank_children(rank_main)
    )
    receipt = {
        "schema": "banana-smasher-ordinary-load-fork-broker-v1",
        "status": result["status"],
        "broker_pid": os.getpid(),
        "checkpoint_mmap": False,
        "materializations": prepared,
        "rank_pids": result["pids"],
        "rank_exit_codes": result["exit_codes"],
        "all_children_reaped": result["all_children_reaped"],
        "same_process_dual_shard": bool(result.get("same_process_dual_shard", False)),
        "started_unix": started,
        "finished_unix": time.time(),
    }
    _atomic_json(args.receipt, receipt)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
