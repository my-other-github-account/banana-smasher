from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "docker/Dockerfile"


def test_public_dockerfile_pins_and_records_vllm_release_source() -> None:
    dockerfile = DOCKERFILE.read_text()
    digest = "sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b"
    commit = "ee0da84ab9e04ac7610e28580af62c365e898389"
    base = f"vllm/vllm-openai:v0.24.0@{digest}"

    assert "ARG VLLM_IMAGE" not in dockerfile
    assert "FROM ${VLLM_IMAGE}" not in dockerfile
    assert [line for line in dockerfile.splitlines() if line.startswith("FROM ")] == [
        f"FROM {base} AS package-builder",
        f"FROM {base} AS flashinfer-builder",
        f"FROM {base} AS deepgemm-builder",
        f"FROM {base} AS runtime",
    ]
    assert "ARG VLLM_UPSTREAM_TAG=refs/tags/v0.24.0" in dockerfile
    assert f"ARG VLLM_UPSTREAM_COMMIT={commit}" in dockerfile
    assert "ARG VLLM_UPSTREAM_REV=ee0da84a" in dockerfile
    assert 'io.banana-smasher.vllm.upstream-tag="refs/tags/v0.24.0"' in dockerfile
    assert f'io.banana-smasher.vllm.upstream-commit="{commit}"' in dockerfile
    assert f'"vllm_upstream_commit":"{commit}"' in dockerfile
    assert '"vllm_upstream_tag":"refs/tags/v0.24.0"' in dockerfile


def test_public_image_builds_local_wheels_only_from_copied_repository_source() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "COPY banana-smasher /src/banana-smasher" in dockerfile
    assert "COPY banana-smasher-plugin /src/banana-smasher-plugin" in dockerfile
    assert "python3 -m build --wheel --outdir /wheels /src/banana-smasher" in dockerfile
    assert "python3 -m build --wheel --outdir /wheels /src/banana-smasher-plugin" in dockerfile
    assert "COPY dist" not in dockerfile
    assert "COPY build" not in dockerfile
    assert "COPY .venv" not in dockerfile
    assert "PYTHONPATH" not in dockerfile
    assert "--mount=type=bind" not in dockerfile
    assert "/Users/" not in dockerfile
    assert "/home/" not in dockerfile


def test_public_image_bakes_tilelang_linkage_and_runtime_defaults() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "tilelang" in dockerfile
    assert "libcudart_stub.so" in dockerfile
    assert "libcudart.so.13" in dockerfile
    assert 'COPY docker/runtime_defaults.json /opt/banana-smasher/runtime_defaults.json' in dockerfile
    assert "BANANA_SMASHER_AOT_ROOT=/opt/banana-smasher/aot" in dockerfile
    assert 'CMD ["vllm", "serve", "/model"' in dockerfile


def test_public_build_context_excludes_non_source_and_private_surfaces() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    required = {
        ".git",
        ".gitignore",
        ".worktrees/",
        "/.hermes/",
        "/.cursorrules",
        "/AGENTS.md",
        "/CLAUDE.md",
        "/notes/",
        ".env",
        ".env.*",
        "**/.env",
        "**/.env.*",
        ".netrc",
        "**/.netrc",
        ".venv/",
        ".venv-*/",
        "**/.venv/",
        "**/.venv-*/",
        ".pytest_cache/",
        "**/.pytest_cache/",
        ".ruff_cache/",
        "**/.ruff_cache/",
        ".mypy_cache/",
        "**/.mypy_cache/",
        "__pycache__/",
        "**/__pycache__/",
        "/artifacts/",
        "/models/",
        "/model-data/",
        "/packs/",
        "/frozen/",
        "/private/",
        "/runtime-private/",
        "/patches/",
        "/wheels/",
        "/wheelhouse/",
        "*.whl",
        "**/*.whl",
        "*.gguf",
        "**/*.gguf",
        "*.safetensors",
        "**/*.safetensors",
        "*.pt",
        "**/*.pt",
        "**/*.bin",
        "**/*.onnx",
    }

    assert required <= patterns, f"missing .dockerignore patterns: {sorted(required - patterns)}"
    assert "*.patch" not in patterns
    assert "**/*.patch" not in patterns


def test_active_public_container_path_has_no_stale_runtime_or_codename() -> None:
    active_paths = (
        ROOT / "docker/Dockerfile",
        ROOT / "docker/runtime_defaults.json",
        ROOT / "examples/build_image.sh",
        ROOT / "examples/serve.sh",
        ROOT / "README.md",
    )
    content = "\n".join(path.read_text() for path in active_paths)

    assert re.search(r"genesis", content, flags=re.IGNORECASE) is None
    assert re.search(r"v?llm[^\n]*0\.20(?:-dev)?", content, flags=re.IGNORECASE) is None
    for path in active_paths[:4]:
        runtime_text = path.read_text()
        assert "PYTHONPATH" not in runtime_text
        assert re.search(r"(?:--mount|-v)[^\n]*(?:\.venv|/venv)", runtime_text) is None
