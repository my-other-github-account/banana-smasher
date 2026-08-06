"""Compatibility shim that selects FlashInfer's JIT fallback cache."""

from pathlib import Path

__version__ = "0.6.17"


def get_cubin_dir() -> str:
    """Return the same cache path FlashInfer uses when no cubin wheel exists."""
    return str(Path.home() / ".cache" / "flashinfer" / "cubins")
