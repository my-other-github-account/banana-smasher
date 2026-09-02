"""Exercise the resident API against a remote real artifact without staging payloads."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time


PACKAGE_FILES = ("__init__.py", "balanced64.py", "api.py")


def _remote_program(package: dict[str, str], artifact_root: str, checkpoint: str, windows: list[int]) -> str:
    return f'''import base64, json, sys, tempfile
from pathlib import Path
files = {package!r}
with tempfile.TemporaryDirectory(prefix="repair-api-integration-") as td:
    package = Path(td) / "repair_api"
    package.mkdir()
    for name, payload in files.items():
        (package / name).write_bytes(base64.b64decode(payload))
    sys.path.insert(0, td)
    from repair_api import ResidentRepairAPI
    api = ResidentRepairAPI.open(Path({artifact_root!r}))
    result = api.score({checkpoint!r}, windows={windows!r})
    print(json.dumps(result.as_dict(), sort_keys=True))
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="repair-api-remote-integration")
    parser.add_argument("--remote", default="spark-7")
    parser.add_argument("--python", default="/home/dnola/humming_env/bin/python")
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--checkpoint", default="UPDATE_016")
    parser.add_argument("--windows", default="28")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    windows = [int(value) for value in args.windows.split(",") if value]
    package = {
        name: base64.b64encode((Path(__file__).parent / name).read_bytes()).decode()
        for name in PACKAGE_FILES
    }
    remote = _remote_program(package, args.artifact_root, args.checkpoint, windows)
    payload = base64.b64encode(remote.encode()).decode()
    command = [
        "printf", "%s", "<base64-remote-program>", "|", "base64", "-d", "|",
        "ssh", args.remote, args.python, "-",
    ]
    started = time.perf_counter()
    receipt: dict[str, object] = {
        "schema": "resident-api-integration-v1",
        "status": "FAILED",
        "command": " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
        "remote_command": " ".join(shlex.quote(value) for value in command),
        "artifact_root": args.artifact_root,
        "remote": args.remote,
        "checkpoint": args.checkpoint,
        "windows": windows,
        "resident_score_wall_seconds": None,
    }
    try:
        completed = subprocess.run(
            f"printf %s {shlex.quote(payload)} | base64 -d | ssh {shlex.quote(args.remote)} {shlex.quote(args.python)} -",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        receipt["ssh_returncode"] = completed.returncode
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-2000:] or completed.stdout[-2000:])
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        if result.get("status") != "PASS":
            raise RuntimeError(f"remote API returned {result.get('status')}")
        receipt.update({
            "status": "PASS",
            "score": result,
            "resident_score_wall_seconds": result.get("timed_wall_seconds"),
        })
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        receipt.update({"error_type": type(exc).__name__, "error": str(exc)})
    receipt["integration_wall_seconds"] = time.perf_counter() - started
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(f".{args.receipt.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
