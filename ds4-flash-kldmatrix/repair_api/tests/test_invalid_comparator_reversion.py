from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_invalid_mode_w2_repair_is_fully_removed_from_product_path() -> None:
    """Restore the exact pre-hypothesis product bytes after comparator invalidation."""
    resident = ROOT / "modern_green_resident.py"
    runner = ROOT / "resident_full64_accept.py"

    assert _sha(resident) == "fddf8136d4fe5a4e9c41eb664a774070f39e48c0091e14e92922b5cacf28a199"
    assert _sha(runner) == "996a313cc1f3bab20deec6811df3d1a69d1162412b980e69ef8605150b1b5433"

    resident_source = resident.read_text()
    runner_source = runner.read_text()
    assert "_sealed_builder_native_down_projection" not in resident_source
    assert "SEALED_NATIVE_BF16_W2_HELPER_MISSING" not in resident_source
    assert "FAST_K2_GLOBAL_BF16_FLINEAR_SCOPE" not in runner_source
    assert "_install_r20_full_weight_projection(engine, layers=" not in runner_source
