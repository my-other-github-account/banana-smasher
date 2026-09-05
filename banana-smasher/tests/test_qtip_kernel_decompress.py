"""The builder decoder must be in the canonical pin, not a mission patch."""
import types

import pytest

torch = pytest.importorskip("torch")
from banana_smasher import qtip_runner


def test_builder_loads_canonical_decoder_and_compiles_k1(monkeypatch):
    external = []

    def source(name, path):
        external.append(path.name)
        return types.SimpleNamespace()

    # Unused upstream builder modules are isolated; the actual decoder is real.
    monkeypatch.setattr(qtip_runner, "load_source_module", source)
    modules_before = dict(__import__("sys").modules)
    try:
        decoder = qtip_runner.load_official_qtip()[3]
    finally:
        import sys
        for key in list(sys.modules):
            if key == "lib" or key.startswith("lib.") or key == "glog":
                if key in modules_before:
                    sys.modules[key] = modules_before[key]
                else:
                    sys.modules.pop(key, None)
    assert "kernel_decompress.py" not in external
    assert decoder.__name__ == "banana_smasher.qtip_kernel_decompress"
    # K1 requires the byte-unswizzle to become contiguous before dtype-view.
    words = torch.arange(64, dtype=torch.int32).to(torch.uint16)
    lut = torch.arange(65536 * 2, dtype=torch.float32).reshape(65536, 2)
    eager = decoder.decode_compressed.__wrapped__
    expected = eager(16, 9, 1, 1, 32, 32, words, lut)
    compiled = torch.compile(eager, backend="aot_eager")
    actual = compiled(16, 9, 1, 1, 32, 32, words, lut)
    assert actual.shape == (32, 32)
    assert torch.equal(actual, expected)
