from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._body.splitlines(keepends=True))


def _load_smoke_api() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "banana_smasher_smoke_api", ROOT / "examples/smoke_api.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_rejects_nonempty_role_template_echo(monkeypatch: pytest.MonkeyPatch) -> None:
    smoke_api = _load_smoke_api()
    replies = iter(
        (
            b"",
            json.dumps({"data": [{"id": "banana-smasher-v5"}]}).encode(),
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '<|im_start|>user\\nReply with OK.<|im_end|>'
                            }
                        }
                    ]
                }
            ).encode(),
        )
    )

    def fake_urlopen(*args: Any, **kwargs: Any) -> _Response:
        return _Response(next(replies))

    monkeypatch.setattr(smoke_api.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit, match="semantic OK"):
        smoke_api.main()


def test_main_requires_semantic_ok_from_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    smoke_api = _load_smoke_api()
    stream = (
        'data: {"choices":[{"delta":{"content":"<|im_start|>assistant"}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    replies = iter(
        (
            b"",
            json.dumps({"data": [{"id": "banana-smasher-v5"}]}).encode(),
            json.dumps({"choices": [{"message": {"content": "OK."}}]}).encode(),
            stream,
        )
    )

    def fake_urlopen(*args: Any, **kwargs: Any) -> _Response:
        return _Response(next(replies))

    monkeypatch.setattr(smoke_api.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(SystemExit, match="streaming chat"):
        smoke_api.main()
