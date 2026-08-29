"""Regression: the resident engine must rebase reboot-volatile LUT provenance.

A reboot destroys the ``/dev/shm`` LUT parent tree whose absolute paths are
recorded inside the sealed admission document.  ``_rebase_admission_lut_sources``
already exists and is applied by the scorer engine, but the resident training
engine read the admission raw, so a resident continuation still died with
``L000 manifest missing: /dev/shm/...``.  These tests pin the rebase at the
resident seam and pin that digests remain the sealed admission's authority.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from repair_api.balanced64 import ArtifactError
from repair_api.official_k2_resident_score import _rebase_admission_lut_sources


STALE_ROOT = "/dev/shm/V7_CODEBOOK_t_0c44dcc6_s6/lut_parents"


def _write(path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _admission(tmp_path, *, layers=range(43)):
    """Build a sealed-shaped admission plus an identity-equal durable mirror."""
    durable = tmp_path / "lut_parents"
    rows = []
    for layer in layers:
        parent = durable / f"L{layer:03d}" / "parent"
        manifest_sha = _write(
            parent / "QTIP_V7_MANIFEST.json",
            json.dumps({"members": [{"expert": 0, "projection": "w1", "sha256": "0" * 64}]}).encode(),
        )
        wire_sha = _write(parent / f"L{layer:03d}.tlut.f16", bytes([layer]) * 2048)
        rows.append(
            {
                "layer": layer,
                "source_manifest": {
                    "path": f"{STALE_ROOT}/L{layer:03d}/parent/QTIP_V7_MANIFEST.json",
                    "sha256": manifest_sha,
                },
                "wire": {
                    "path": f"{STALE_ROOT}/L{layer:03d}/parent/L{layer:03d}.tlut.f16",
                    "sha256": wire_sha,
                },
            }
        )
    return {"framework": "banana-smasher", "trainable_roster": {"luts": rows}}, durable


def test_rebases_every_stale_shm_lut_source_to_the_durable_mirror(tmp_path):
    admission, durable = _admission(tmp_path)

    rebound = _rebase_admission_lut_sources(admission, durable)

    rows = rebound["trainable_roster"]["luts"]
    assert len(rows) == 43
    for row in rows:
        layer = int(row["layer"])
        assert not row["source_manifest"]["path"].startswith("/dev/shm")
        assert row["source_manifest"]["path"] == str(
            durable / f"L{layer:03d}" / "parent/QTIP_V7_MANIFEST.json"
        )
        assert row["wire"]["path"] == str(
            durable / f"L{layer:03d}" / "parent" / f"L{layer:03d}.tlut.f16"
        )
    # the sealed admission itself is never mutated
    assert admission["trainable_roster"]["luts"][0]["source_manifest"]["path"].startswith(
        "/dev/shm"
    )


def test_rebase_preserves_sealed_digests_as_the_scientific_authority(tmp_path):
    admission, durable = _admission(tmp_path)
    before = [
        (row["source_manifest"]["sha256"], row["wire"]["sha256"])
        for row in admission["trainable_roster"]["luts"]
    ]

    rebound = _rebase_admission_lut_sources(admission, durable)

    after = [
        (row["source_manifest"]["sha256"], row["wire"]["sha256"])
        for row in rebound["trainable_roster"]["luts"]
    ]
    assert after == before


def test_rebase_refuses_a_mirror_whose_bytes_drifted(tmp_path):
    admission, durable = _admission(tmp_path)
    (durable / "L007" / "parent" / "L007.tlut.f16").write_bytes(b"drifted")

    with pytest.raises(ArtifactError, match="durable LUT source SHA mismatch"):
        _rebase_admission_lut_sources(admission, durable)


def test_rebase_refuses_an_incomplete_mirror(tmp_path):
    admission, durable = _admission(tmp_path)
    (durable / "L011" / "parent" / "QTIP_V7_MANIFEST.json").unlink()

    with pytest.raises(ArtifactError, match="durable LUT source is missing"):
        _rebase_admission_lut_sources(admission, durable)


def test_resident_engine_applies_the_rebase_at_its_admission_seam():
    """The resident seam must call the rebase, not read the admission raw."""
    import inspect

    from repair_api import modern_green_resident

    source = inspect.getsource(modern_green_resident)
    seam = source.split('admission_path.read_text()')[1][:400]
    assert "lut_parent_root" in seam
    assert "_rebase_admission_lut_sources" in seam
