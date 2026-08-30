from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_sensitivity_probe_worker_cli_imports_from_pinned_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, str(root / "tools/sensitivity_probe_worker.py"), "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--probe-manifest" in result.stdout
    assert "--start" in result.stdout and "--end" in result.stdout
    assert "--reproduce-baseline" in result.stdout


def test_projection_granularity_cli_imports_from_pinned_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    result = subprocess.run(
        [sys.executable, str(root / "tools/projection_granularity.py"), "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--baseline-assignment" in result.stdout
    assert "--output" in result.stdout
