from __future__ import annotations

import importlib.util
import io
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "tools/w328_recovery/complete_provider_recovery4_local30.py"


def load_module():
    spec = importlib.util.spec_from_file_location("w328_l034_provider_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, rc: int) -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self._rc = rc

    def wait(self) -> int:
        return self._rc


def test_initial_four_byte_short_stream_retries() -> None:
    module = load_module()
    processes = [
        FakeProcess(b"", b"ssh: banner timeout\n", 255),
        FakeProcess(b"\x00\x00\x00\x02{}", b"", 0),
    ]
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return processes.pop(0)

    proc, prefix = module.open_framed_stream(
        ["ssh", "host"], b"[]\n", attempts=3, popen=popen, retry_delay=0
    )

    assert prefix == b"\x00\x00\x00\x02"
    assert proc.stdout.read() == b"{}"
    assert len(calls) == 2
