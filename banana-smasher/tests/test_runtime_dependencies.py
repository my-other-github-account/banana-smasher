from __future__ import annotations

from pathlib import Path
import tomllib


def test_solve_extra_installs_ninja_for_runtime_cuda_extensions() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    solve = project["project"]["optional-dependencies"]["solve"]

    assert "ninja==1.13.0" in solve
