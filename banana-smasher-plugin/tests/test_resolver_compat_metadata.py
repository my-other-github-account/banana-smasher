from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]


def _project(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def test_stock_vllm_exact_flashinfer_pins_accept_compat_distributions() -> None:
    cubin = _project(ROOT / "packaging" / "flashinfer-cubin-tombstone" / "pyproject.toml")
    compat = Version(str(cubin["version"]))

    assert compat in SpecifierSet("==0.6.12")
    assert compat not in SpecifierSet("==0.6.17")

    module = (
        ROOT
        / "packaging"
        / "flashinfer-cubin-tombstone"
        / "src"
        / "flashinfer_cubin"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    assert '__version__ = "0.6.17"' in module


def test_runtime_packages_preserve_stock_numpy_range() -> None:
    for pyproject in (
        ROOT / "banana-smasher" / "pyproject.toml",
        ROOT / "banana-smasher-plugin" / "pyproject.toml",
    ):
        project = _project(pyproject)
        numpy_requirement = next(
            str(item) for item in project["dependencies"] if str(item).startswith("numpy")
        )
        specifier = SpecifierSet(numpy_requirement.removeprefix("numpy"))
        assert Version("2.2.6") in specifier
        assert Version("2.3.5") not in specifier
