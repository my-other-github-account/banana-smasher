from __future__ import annotations

from pathlib import Path

from importlib.metadata import version


def test_tombstone_shadows_stale_namespace_with_jit_fallback() -> None:
    import flashinfer_cubin

    assert version("flashinfer-cubin") == "0.6.12+banana.fi0617"
    assert flashinfer_cubin.__version__ == "0.6.17"
    assert Path(flashinfer_cubin.get_cubin_dir()) == Path.home() / ".cache" / "flashinfer" / "cubins"
