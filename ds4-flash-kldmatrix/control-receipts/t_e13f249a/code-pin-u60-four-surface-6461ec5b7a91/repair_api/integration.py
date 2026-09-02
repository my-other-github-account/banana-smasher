"""Run one real-root resident API integration and persist an honest receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
import time

from . import ArtifactError, ResidentRepairAPI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="resident-api-integration")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--checkpoint", default="UPDATE_016")
    parser.add_argument("--windows", default=None)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    selected = None if args.windows is None else [int(value) for value in args.windows.split(",") if value]
    command = " ".join(shlex.quote(value) for value in [sys.executable, "-m", "repair_api.integration", *sys.argv[1:]])
    started = time.perf_counter()
    receipt: dict[str, object] = {
        "schema": "resident-api-integration-v1",
        "status": "FAILED",
        "command": command,
        "artifact_root": str(args.artifact_root.expanduser()),
        "checkpoint": args.checkpoint,
        "windows": selected,
        "resident_score_wall_seconds": None,
    }
    try:
        api = ResidentRepairAPI.open(args.artifact_root)
        result = api.score(args.checkpoint, windows=selected)
        receipt.update({"status": "PASS", "score": result.as_dict()})
        receipt["resident_score_wall_seconds"] = result.timed_wall_seconds
    except (ArtifactError, OSError, ValueError) as exc:
        receipt.update({
            "status": "BLOCKED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
    receipt["integration_wall_seconds"] = time.perf_counter() - started
    ResidentRepairAPI.write_receipt(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
