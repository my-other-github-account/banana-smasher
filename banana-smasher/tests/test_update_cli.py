from __future__ import annotations

import json
from pathlib import Path

import banana_smasher.update as update_module
import pytest
from banana_smasher.cli import _parser, main
from banana_smasher.token_sizing import MemoryBudget


def _identity() -> dict[str, str]:
    return {
        "content_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "assignment_sha256": "3" * 64,
        "aot_sha256": "4" * 64,
        "runtime_sha256": "5" * 64,
        "code_sha256": "6" * 64,
    }


def _argv(tmp_path: Path) -> list[str]:
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(_identity()))
    request = tmp_path / "request.json"
    request.write_text('{"schema":"fixture"}\n')
    return [
        "update",
        "--backend",
        "fixture",
        "--request",
        str(request),
        "--identity",
        str(identity),
        "--output",
        str(tmp_path / "out.pt"),
        "--tokens",
        "1024",
        "--segments",
        "3",
        "--available-bytes",
        str(16 * 1024**3),
        "--resident-frozen-bytes",
        str(2 * 1024**3),
        "--trainable-bytes",
        "1024",
        "--optimizer-bytes",
        "2048",
        "--staging-bytes",
        "4096",
        "--activation-bytes-per-token",
        "1048576",
    ]


def test_update_cli_keeps_physical_tokens_distinct_from_segments(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(_argv(tmp_path))


def test_update_cli_dispatches_memory_sized_physical_tokens(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    observed: dict = {}

    def fake_run_registered_update(**kwargs):
        observed.update(kwargs)
        return {
            "schema": "banana-smasher-update-receipt-v4",
            "status": "PASS_UPDATE",
            "physical_tokens": kwargs["requested_tokens"],
            "logical_tokens": kwargs["requested_tokens"] * kwargs["segments"],
            "segments": kwargs["segments"],
            "observed_input_shape": [1, kwargs["requested_tokens"]],
            "optimizer_steps": 1,
        }

    monkeypatch.setattr(
        update_module, "run_registered_update", fake_run_registered_update
    )
    with pytest.raises(SystemExit):
        main(_argv(tmp_path))
    assert observed == {}


def _registered_kwargs(tmp_path: Path) -> dict:
    request = tmp_path / "registered-request.json"
    request.write_text('{"schema":"fixture"}\n')
    return {
        "backend_name": "fixture",
        "request": request,
        "output": tmp_path / "registered-output.pt",
        "receipt": None,
        "identity": _identity(),
        "requested_tokens": 2,
        "segments": 3,
        "batch_size": 1,
        "memory_budget": MemoryBudget(
            available_bytes=8 * 1024**3,
            resident_frozen_bytes=0,
            trainable_bytes=0,
            optimizer_bytes=0,
            staging_bytes=0,
            calibrated_activation_bytes_per_token=1,
        ),
    }


def _backend_receipt() -> dict:
    return {
        "status": "PASS_UPDATE",
        "physical_tokens": 2,
        "logical_tokens": 6,
        "logical_items": 6,
        "segments": 3,
        "optimizer_steps": 1,
        "observed_input_shape": [1, 2],
        "teacher_geometry": {
            "target_shape": [1, 2],
            "mask_shape": [1, 2],
            "position_shape": [1, 2],
        },
        "forward_count": 3,
        "backward_count": 3,
        "peak_memory_bytes": 1024,
        "finite_required_trainable_gradients": True,
        "immutable_identity": _identity(),
        "timing": {"started_unix": 1.0, "completed_unix": 2.0, "segments": []},
        "fallback": {"used": False, "reason": None},
    }


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("status", "PASS_ANYTHING", "passing receipt"),
        ("observed_input_shape", [999, 2], "batch-1 physical token shape"),
    ],
)
def test_registered_update_rejects_malformed_backend_receipts(
    tmp_path: Path, monkeypatch, field: str, invalid: object, message: str
) -> None:
    receipt = _backend_receipt()
    receipt[field] = invalid

    class Entry:
        @staticmethod
        def load():
            return lambda **_kwargs: receipt

    monkeypatch.setattr(update_module, "_update_entry_point", lambda _name: Entry())
    with pytest.raises(RuntimeError, match=message):
        update_module.run_registered_update(**_registered_kwargs(tmp_path))


def test_registered_update_requires_batch_one_before_backend_load(
    tmp_path: Path, monkeypatch
) -> None:
    calls = 0

    def forbidden(_name: str):
        nonlocal calls
        calls += 1
        raise AssertionError("backend loaded")

    monkeypatch.setattr(update_module, "_update_entry_point", forbidden)
    kwargs = _registered_kwargs(tmp_path)
    kwargs["batch_size"] = 2
    with pytest.raises(ValueError, match="requires batch_size=1"):
        update_module.run_registered_update(**kwargs)
    assert calls == 0
