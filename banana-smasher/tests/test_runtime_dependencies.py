from __future__ import annotations

import os
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace

from banana_smasher.resident_repair_api import _ensure_ninja_available


def test_solve_extra_installs_ninja_for_runtime_cuda_extensions() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    solve = project["project"]["optional-dependencies"]["solve"]

    assert "ninja==1.13.0" in solve


def test_admitted_api_restores_ninja_to_path_from_installed_solve_extra(
    tmp_path: Path, monkeypatch,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ninja_executable = bin_dir / "ninja"
    ninja_executable.write_text("#!/bin/sh\nexit 0\n")
    ninja_executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setitem(sys.modules, "ninja", SimpleNamespace(BIN_DIR=str(bin_dir)))

    resolved = _ensure_ninja_available()

    assert resolved == ninja_executable
    assert os.environ["PATH"].split(os.pathsep)[0] == str(bin_dir)
