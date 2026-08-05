from __future__ import annotations

import json
from pathlib import Path

import pytest

from banana_smasher.serving import (
    RUNTIME_ENV_DEFAULTS,
    build_serve_command,
    inspect_model_pack,
)
from banana_smasher.cli import main


def _model(root: Path) -> Path:
    root.mkdir()
    (root / "BANANA_PACK_MANIFEST.json").write_text('{"schema":"bs-pack"}\n')
    (root / "config.json").write_text(
        json.dumps(
            {"quantization_config": {"quant_method": "banana_smasher"}}
        )
        + "\n"
    )
    return root


def test_local_serve_command_is_stock_vllm_with_boot10_defaults(tmp_path: Path) -> None:
    model = _model(tmp_path / "model")
    command, environment = build_serve_command(model, port=8123)

    assert command[:3] == ["vllm", "serve", str(model.resolve())]
    assert ["--kv-cache-memory-bytes", "3221225472"] == command[
        command.index("--kv-cache-memory-bytes") : command.index("--kv-cache-memory-bytes") + 2
    ]
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8123"]
    for name, value in RUNTIME_ENV_DEFAULTS.items():
        assert environment[name] == value


def test_container_serve_mounts_only_model_and_cache_then_runs_stock_vllm(
    tmp_path: Path,
) -> None:
    model = _model(tmp_path / "model")
    command, _ = build_serve_command(
        model,
        container_image="banana-smasher-runtime:test",
        host="127.0.0.1",
        port=8001,
    )

    assert command[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert "127.0.0.1:8001:8001" in command
    assert f"{model.resolve()}:/model:ro" in command
    image_index = command.index("banana-smasher-runtime:test")
    assert command[image_index + 1 : image_index + 4] == ["vllm", "serve", "/model"]
    assert command[-4:] == ["--host", "0.0.0.0", "--port", "8001"]


def test_model_identity_check_rejects_non_banana_config(tmp_path: Path) -> None:
    model = _model(tmp_path / "model")
    (model / "config.json").write_text(
        '{"quantization_config":{"quant_method":"fp8"}}\n'
    )
    with pytest.raises(ValueError, match="quant_method"):
        inspect_model_pack(model)


def test_smash_serve_dry_run_exposes_exact_container_transaction(
    tmp_path: Path, capsys
) -> None:
    model = _model(tmp_path / "model")
    assert (
        main(
            [
                "serve",
                str(model),
                "--container-image",
                "banana-smasher-runtime:test",
                "--dry-run",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "READY"
    assert receipt["backend"] == "container"
    assert receipt["argv"][0:3] == ["docker", "run", "--rm"]
    assert "banana-smasher-runtime:test" in receipt["argv"]
