from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from banana_smasher import BananaV1All43Adapter as PublicBananaV1All43Adapter
from banana_smasher.banana_v1 import (
    BANANA_V1_GEOMETRY,
    BananaV1BuildResult,
    banana_v1_inverse_transform,
    banana_v1_wire_accounting,
    decode_banana_v1,
    pack_banana_v1_states,
    write_banana_v1_candidate,
)
from banana_smasher.banana_v1_runtime_adapter import BananaV1All43Adapter


BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"


class _Weight:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    def clone(self):
        return _Weight(self.values.copy())

    def new_tensor(self, values):
        return np.asarray(values, dtype=np.float32)

    def __setitem__(self, key, value):
        self.values[key] = value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    expert: int = 0,
    projection: str = "w1",
    row_start: int = 0,
    column_start: int = 0,
) -> tuple[Path, str, np.ndarray]:
    codebook = np.linspace(-2.0, 2.0, 1024, dtype=np.float16)
    states = np.zeros((1, 256), dtype=np.int32)
    packed = pack_banana_v1_states(states)
    scales = np.asarray([0.5], dtype=np.float32)
    transformed = decode_banana_v1(
        packed, scales, positions=256, codebook=codebook
    ).reshape(16, 16)
    signs = np.ones(16, dtype=np.float32)
    decoded = banana_v1_inverse_transform(transformed, su=signs, sv=signs)
    result = BananaV1BuildResult(
        source_shape=(16, 16),
        decoded=decoded,
        states=states,
        packed=packed,
        scales=scales,
        codebook=codebook,
        su=signs,
        sv=signs,
        distortion=0.0,
        scale_factor=1.0,
        scale_factors=(1.0,),
        accounting=banana_v1_wire_accounting(
            position_count=256,
            sequence_count=1,
            scale_bytes=4,
            transform_bytes=64,
            shared_codebook_bytes=2048,
        ),
    )
    template = tmp_path / "template"
    write_banana_v1_candidate(template, result)
    members = tmp_path / "members"
    rows = []
    for layer in range(43):
        root = members / f"L{layer:03d}_E{expert:03d}_{projection}_tile000"
        shutil.copytree(template, root)
        rows.append(
            {
                "id": (
                    f"L{layer:03d}/E{expert:03d}/{projection}/"
                    f"tile-r{row_start:03d}-r{row_start + 15:03d}-"
                    f"c{column_start:03d}-c{column_start + 15:03d}"
                ),
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "row_start": row_start,
                "column_start": column_start,
                "member_root": str(root),
                "receipt_sha256": _sha(root / "BANANA_V1_RECEIPT.json"),
            }
        )
    shared = tmp_path / "shared_codebook.fp16"
    shared.write_bytes(codebook.tobytes())
    terminal = tmp_path / "ALL43_SHARED_TRAIN_TERMINAL.json"
    terminal.write_text(
        json.dumps(
            {
                "basis_sha256": BASIS,
                "model_index_sha256": BASIS,
                "roster_count": 43,
                "layers": list(range(43)),
                "single_shared_codebook": {
                    "path": str(shared),
                    "sha256": _sha(shared),
                    "shape": [1024],
                    "dtype": "float16",
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    manifest = tmp_path / "BANANA_V1_ALL43_ADAPTER.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-banana-v1-all43-runtime-adapter-v1",
                "basis_sha256": BASIS,
                "terminal": {"path": str(terminal), "sha256": _sha(terminal)},
                "shared_codebook": {
                    "path": str(shared),
                    "sha256": _sha(shared),
                    "shape": [1024],
                    "dtype": "float16",
                },
                "members": rows,
            },
            sort_keys=True,
        )
        + "\n"
    )
    return manifest, _sha(terminal), decoded


def test_all43_adapter_decodes_known_member_exact_and_has_no_roster_gap(
    tmp_path: Path,
) -> None:
    manifest, terminal_sha, expected = _fixture(tmp_path)

    adapter = BananaV1All43Adapter.open(
        manifest,
        expected_basis_sha256=BASIS,
        expected_terminal_sha256=terminal_sha,
    )
    assert PublicBananaV1All43Adapter is BananaV1All43Adapter

    observed = adapter.decode_member(0)
    assert np.array_equal(observed, expected)
    assert adapter.layers == tuple(range(43))
    assert len(adapter.members) == 43
    assert adapter.shared_codebook.shape == (BANANA_V1_GEOMETRY.codebook_levels,)


def test_joint_runtime_binds_all43_manifest_before_expert_construction(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, terminal_sha, _expected = _fixture(tmp_path)
    runtime_path = (
        Path(__file__).resolve().parents[2]
        / "runtime/v7/runner/joint_v7_runtime_adapter.py"
    )
    spec = importlib.util.spec_from_file_location("joint_v7_runtime_adapter_test", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    device = SimpleNamespace(type="cuda", index=0)
    sources = {}
    for layer in range(43):
        master = SimpleNamespace(shape=(1024,), dtype="torch.float32", device=device)
        sources[layer] = SimpleNamespace(
            layer=layer,
            master=master,
            wire_lut=lambda: None,
            member_path=lambda *_args: tmp_path,
        )
    monkeypatch.setenv("BANANA_V1_ALL43_MANIFEST", str(manifest))
    monkeypatch.setenv("BANANA_V1_ALL43_MANIFEST_SHA256", _sha(manifest))
    monkeypatch.setenv("BANANA_V1_ALL43_BASIS_SHA256", BASIS)
    monkeypatch.setenv("BANANA_V1_ALL43_TERMINAL_SHA256", terminal_sha)

    observed = runtime._exact_sources(sources, device)

    assert all(
        isinstance(source._banana_v1_all43_adapter, BananaV1All43Adapter)
        for source in observed.values()
    )


def test_adapter_patches_only_the_declared_physical_tile(tmp_path: Path) -> None:
    manifest, terminal_sha, expected = _fixture(tmp_path)
    adapter = BananaV1All43Adapter.open(
        manifest,
        expected_basis_sha256=BASIS,
        expected_terminal_sha256=terminal_sha,
    )

    original = _Weight(np.full((32, 32), -7.0, dtype=np.float32))

    patched = adapter.patch_weight(0, 0, "w1", original)

    assert np.array_equal(patched.values[:16, :16], expected)
    assert np.all(patched.values[16:, :] == -7.0)
    assert np.all(patched.values[:16, 16:] == -7.0)
    assert np.all(original.values == -7.0)
    assert adapter.patch_weight(0, 1, "w1", original) is original

    fresh = _Weight(np.full((32, 32), -7.0, dtype=np.float32))
    observed = adapter.patch_fresh_weight(0, 0, "w1", fresh)
    assert observed is fresh
    assert np.array_equal(fresh.values[:16, :16], expected)


def test_adapter_authenticates_and_patches_an_alternate_declared_surface(
    tmp_path: Path,
) -> None:
    manifest, terminal_sha, expected = _fixture(
        tmp_path,
        expert=7,
        projection="w2",
        row_start=8,
        column_start=16,
    )
    manifest_sha = _sha(manifest)
    adapter = BananaV1All43Adapter.open(
        manifest,
        expected_basis_sha256=BASIS,
        expected_terminal_sha256=terminal_sha,
        expected_manifest_sha256=manifest_sha,
        expert=7,
        projection="w2",
        row_start=8,
        column_start=16,
    )

    original = _Weight(np.full((40, 48), -7.0, dtype=np.float32))
    patched = adapter.patch_weight(0, 7, "w2", original)
    changed = np.zeros((40, 48), dtype=bool)
    changed[8:24, 16:32] = True

    assert adapter.manifest_sha256 == manifest_sha
    assert adapter.members[0].id == "L000/E007/w2/tile-r008-r023-c016-c031"
    assert np.array_equal(patched.values[8:24, 16:32], expected)
    assert np.all(patched.values[~changed] == -7.0)
    assert np.all(original.values == -7.0)
    assert adapter.patch_weight(0, 0, "w2", original) is original
    assert adapter.patch_weight(0, 7, "w1", original) is original


def test_joint_expert_dispatches_decoded_weight_through_bound_adapter() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "runtime/v7/runner/joint_v7_expert_base.py"
    ).read_text()

    assert "adapter.patch_fresh_weight(self.layer, self.expert, self.projection, weight)" in source
    assert "_banana_v1_adapter_from_env()" in source
