"""Marker installed after removing the stale flashinfer-cubin distribution.

FlashInfer 0.6.17 uses its matching JIT cache on this runtime. This package
intentionally does not provide the old ``flashinfer_cubin`` import namespace.
"""

FLASHINFER_CUBIN_REMOVED = True
