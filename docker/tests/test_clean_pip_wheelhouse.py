from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PLUGIN_PYPROJECT = ROOT / "banana-smasher-plugin" / "pyproject.toml"
PACKAGE_PYPROJECT = ROOT / "banana-smasher" / "pyproject.toml"
DOCKERFILE = ROOT / "docker" / "Dockerfile"
INSTALL = ROOT / "docker" / "clean-pip" / "INSTALL.txt"


def _normalized_dependencies() -> set[str]:
    text = PLUGIN_PYPROJECT.read_text()
    dependency_block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    return {
        re.sub(r"\s+", "", dependency).lower()
        for dependency in re.findall(r'"([^"\n]+)"', dependency_block)
    }


def test_plugin_runtime_dependencies_match_stock_vllm_0240() -> None:
    dependencies = _normalized_dependencies()

    assert "flashinfer-python==0.6.12;sys_platform=='linux'andplatform_machine=='aarch64'" in dependencies
    assert "flashinfer-jit-cache==0.6.12+cu130;sys_platform=='linux'andplatform_machine=='aarch64'" in dependencies
    assert "deep-gemm==2.6.1;sys_platform=='linux'andplatform_machine=='aarch64'" in dependencies


def test_normal_package_pulls_runtime_plugin_on_linux_aarch64() -> None:
    package = PACKAGE_PYPROJECT.read_text()

    assert (
        "banana-smasher-plugin==0.2.0; sys_platform == 'linux' and platform_machine == 'aarch64'"
        in package
    )


def test_normal_package_preserves_stock_vllm_numpy_constraint() -> None:
    package = PACKAGE_PYPROJECT.read_text()

    assert '"numpy>=2.2,<2.3"' in package
    assert '"numpy==2.3.5"' not in package


def test_clean_pip_wheelhouse_stage_exports_only_resolver_inputs() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "FROM package-builder AS clean-pip-wheelhouse-builder" in dockerfile
    assert "COPY --from=package-builder /wheels/banana_smasher-1.0.0-py3-none-any.whl /wheelhouse/" in dockerfile
    assert "COPY --from=deepgemm-builder /wheels/deep_gemm-2.6.1-cp312-cp312-linux_aarch64.whl /wheelhouse/" in dockerfile
    clean_stage = dockerfile.split("FROM package-builder AS clean-pip-wheelhouse-builder", 1)[1].split("FROM scratch AS clean-pip-wheelhouse", 1)[0]
    assert "flashinfer_python" not in clean_stage
    assert "flashinfer_jit_cache" not in clean_stage
    assert "FROM scratch AS clean-pip-wheelhouse" in dockerfile
    assert "COPY --from=clean-pip-wheelhouse-builder /wheelhouse/ /wheelhouse/" in dockerfile


def test_clean_pip_install_is_one_ordinary_resolver_command() -> None:
    assert INSTALL.read_text() == (
        "python3 -m pip install --no-index --find-links /wheelhouse "
        "banana-smasher==1.0.0\n"
    )
