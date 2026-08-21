from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[2] / "tools/w328_recovery/t8192_ds4_build_v3.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("w328_builder_checkpoint_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_atomic_hidden_checkpoint_round_trip(tmp_path: Path) -> None:
    builder = load_builder()
    hidden = [torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 2, 2)]
    path = builder.atomic_hidden_checkpoint(
        tmp_path,
        layer=5,
        wins=[328],
        hidden=hidden,
    )

    assert path == tmp_path / "hidden_after_L005.pt"
    assert not list(tmp_path.glob(".*.tmp"))
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "banana-smasher.w328.hidden_checkpoint.v1"
    assert payload["completed_layer"] == 5
    assert payload["next_layer"] == 6
    assert payload["window_order"] == [328]
    assert torch.equal(payload["hidden"][0], hidden[0])


def test_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    builder = load_builder()
    hidden = [torch.zeros((1, 1, 1, 1), dtype=torch.bfloat16)]
    builder.atomic_hidden_checkpoint(tmp_path, layer=0, wins=[328], hidden=hidden)

    try:
        builder.atomic_hidden_checkpoint(tmp_path, layer=0, wins=[328], hidden=hidden)
    except FileExistsError as error:
        assert "hidden_after_L000.pt" in str(error)
    else:
        raise AssertionError("checkpoint overwrite was not refused")
