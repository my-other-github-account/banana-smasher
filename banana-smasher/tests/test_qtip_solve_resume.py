from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from banana_smasher import solver_qtip_profile as qtip
from banana_smasher import qtip_materialize


MODEL_INDEX_BYTES = b'{"weight_map":{}}'
BASIS = hashlib.sha256(MODEL_INDEX_BYTES).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trust_test_runner(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(qtip, "_TRUSTED_PUBLIC_QTIP_RUNNER_SHA256", _sha256(path))


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "SHARDS.json").write_text(
        json.dumps({"intended_basis": {"index_sha256": BASIS}}) + "\n"
    )
    return run_root


def _config(path: Path, expert: int, projection: str) -> dict[str, object]:
    model_root = path.parent.parent / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "model.safetensors.index.json").write_bytes(MODEL_INDEX_BYTES)
    value: dict[str, object] = {
        "layer": 9,
        "expert": expert,
        "projection": projection,
        "tier": "qtip3",
        "layer_census": {"qtip3": 512},
        "geometry": {"L": 16, "K": 3, "V": 2},
        "model_root": str(model_root),
        "input_identity": {"model_index": {"sha256": BASIS}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value


def _sealed_unit(
    root: Path, config_path: Path, expert: int, projection: str
) -> tuple[Path, Path]:
    """Write one canonical-format PASS unit: payload + hash-bound receipt."""
    out = root / "solve" / "L009" / f"E{expert:03d}_{projection}"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "QTIP_UNIT.pt"
    trellis = torch.tensor(
        [[expert, 0 if projection == "fused13" else 1]], dtype=torch.int64
    )
    config = json.loads(config_path.read_text())
    geometry = config["geometry"]
    torch.save(
        {
            "schema": "banana-smasher-qtip-unit-v1",
            "shape": [2, 2],
            "trellis": trellis,
            "SU": torch.ones(2, dtype=torch.float16),
            "SV": torch.ones(2, dtype=torch.float16),
            "Wscale": torch.tensor(1.0),
            "tlut": torch.ones((512, 2), dtype=torch.float16),
            "geometry": {
                "L": geometry["L"],
                "K": geometry["K"],
                "V": geometry["V"],
                "tlut_bits": 9,
                "decode_mode": "quantlut_sym",
                "td_x": 16,
                "td_y": 16,
            },
        },
        artifact,
    )
    _fsync_file(artifact)
    receipt_path = out / "QTIP_SOLVE_RECEIPT.json"
    receipt = {
        "schema": "banana-smasher-qtip-solve-v1",
        "status": "PASS",
        "layer": 9,
        "expert": expert,
        "projection": projection,
        "config_sha256": _sha256(config_path),
        "basis_gate": {
            "schema": "banana-smasher-qtip-basis-gate-v1",
            "status": "PASS",
            "index_sha256": BASIS,
            "intended_basis": BASIS,
        },
        "artifact": str(artifact),
        "artifact_sha256": _sha256(artifact),
        "assignment_sha256": qtip._tensor_sha256(trellis),
        "total_wall_seconds": 1.0,
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    _fsync_file(receipt_path)
    return artifact, receipt_path


def test_resident_batch_preserves_479_pass_units_and_resumes_at_unit_480(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """479 valid PASS units keep identical bytes AND mtimes; unit 480 onward runs."""
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    paths: list[Path] = []
    sealed: list[tuple[Path, Path, bytes, bytes, int, int]] = []
    for unit in range(512):
        expert, projection_index = divmod(unit, 2)
        projection = ("fused13", "down")[projection_index]
        path = config_root / f"E{expert:03d}_{projection}.json"
        _config(path, expert, projection)
        paths.append(path)
        if unit < 479:
            artifact, receipt = _sealed_unit(run_root, path, expert, projection)
            sealed.append(
                (
                    artifact,
                    receipt,
                    artifact.read_bytes(),
                    receipt.read_bytes(),
                    artifact.stat().st_mtime_ns,
                    receipt.stat().st_mtime_ns,
                )
            )
    assert len(sealed) == 479

    calls: list[Path] = []

    def fake_main(path: Path, root: Path, layer: int, *, profile_mode: bool):
        assert root == run_root
        assert layer == 9
        assert profile_mode is False
        calls.append(path)
        value = json.loads(path.read_text())
        return {
            "status": "PASS",
            "layer": 9,
            "expert": value["expert"],
            "projection": value["projection"],
            "assignment_sha256": hashlib.sha256(path.name.encode()).hexdigest(),
            "total_wall_seconds": 2.0,
        }

    monkeypatch.setattr(qtip, "main", fake_main)
    batch = qtip.main_many(
        config_root,
        run_root,
        9,
        tier="qtip3",
        all_cells=True,
        profile_mode=False,
    )

    # Execution begins with unit 480 (index 479) and continues thereafter.
    assert calls == paths[479:]
    assert len(calls) == 33
    assert batch["status"] == "PASS"
    assert batch["units"] == 512
    assert batch["resumed_units"] == 479
    assert batch["computed_units"] == 33
    # Every ordered assignment survives, resumed and computed alike.
    assert len(batch["ordered_assignments"]) == 512
    # Sealed units are byte-for-byte and mtime-for-mtime untouched.
    for (
        artifact,
        receipt,
        artifact_bytes,
        receipt_bytes,
        artifact_mtime,
        receipt_mtime,
    ) in sealed:
        assert artifact.read_bytes() == artifact_bytes
        assert receipt.read_bytes() == receipt_bytes
        assert artifact.stat().st_mtime_ns == artifact_mtime
        assert receipt.stat().st_mtime_ns == receipt_mtime


def test_resident_batch_validates_every_existing_unit_before_first_new_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    missing = config_root / "E000_fused13.json"
    corrupt = config_root / "E000_down.json"
    _config(missing, 0, "fused13")
    _config(corrupt, 0, "down")
    artifact, _receipt = _sealed_unit(run_root, corrupt, 0, "down")
    artifact.write_bytes(b"corrupt")

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError(
            "preflight must reject corruption before computing a missing unit"
        )

    monkeypatch.setattr(qtip, "main", must_not_compute)
    with pytest.raises(RuntimeError, match="existing QTIP unit payload hash drift"):
        qtip.main_many(config_root, run_root, 9, profile_mode=False)


@pytest.mark.parametrize(
    "corruption",
    ["missing-receipt", "missing-payload", "payload-hash", "receipt-json"],
)
def test_existing_unit_fails_loudly_on_partial_or_corrupt_state(
    tmp_path: Path, corruption: str
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    artifact, receipt = _sealed_unit(run_root, config_path, 0, "fused13")
    if corruption == "missing-receipt":
        receipt.unlink()
    elif corruption == "missing-payload":
        artifact.unlink()
    elif corruption == "payload-hash":
        artifact.write_bytes(b"drift")
    else:
        receipt.write_text("{not json")

    with pytest.raises(RuntimeError, match="existing QTIP unit"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        ("status", "identity drift"),
        ("expert", "identity drift"),
        ("config", "config hash drift"),
        ("artifact-path", "artifact path drift"),
        ("assignment", "assignment digest drift"),
        ("basis-gate", "basis drift"),
    ],
)
def test_existing_unit_fails_loudly_on_divergence(
    tmp_path: Path, mutate: str, error: str
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    artifact, receipt_path = _sealed_unit(run_root, config_path, 0, "fused13")
    receipt = json.loads(receipt_path.read_text())
    if mutate == "status":
        receipt["status"] = "FAIL"
    elif mutate == "expert":
        receipt["expert"] = 1
    elif mutate == "config":
        config = json.loads(config_path.read_text())
        config["rht_seed"] = 7
        config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    elif mutate == "artifact-path":
        receipt["artifact"] = str(artifact.parent / "elsewhere.pt")
    elif mutate == "assignment":
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        payload["trellis"] = torch.tensor([[99, 99]], dtype=torch.int64)
        torch.save(payload, artifact)
        receipt["artifact_sha256"] = _sha256(artifact)
    else:
        receipt["basis_gate"]["index_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match=error):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


def test_existing_unit_rehashes_live_model_index_against_run_basis(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    config = _config(config_path, 0, "fused13")
    _sealed_unit(run_root, config_path, 0, "fused13")
    model_index = Path(str(config["model_root"])) / "model.safetensors.index.json"
    model_index.write_bytes(b"drift")

    with pytest.raises(RuntimeError, match="live model basis drift"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


@pytest.mark.parametrize("drift", ["payload-geometry", "payload-tensor"])
def test_existing_unit_rejects_incomplete_or_inconsistent_payload(
    tmp_path: Path, drift: str
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    artifact, receipt_path = _sealed_unit(run_root, config_path, 0, "fused13")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if drift == "payload-geometry":
        payload["geometry"]["K"] = 2
    else:
        del payload["SU"]
    torch.save(payload, artifact)
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact_sha256"] = _sha256(artifact)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="payload schema is invalid"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


@pytest.mark.parametrize("timing", [True, -1.0, float("nan"), float("inf"), "1.0"])
def test_existing_unit_rejects_invalid_timing(tmp_path: Path, timing: object) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    _artifact, receipt_path = _sealed_unit(run_root, config_path, 0, "fused13")
    receipt = json.loads(receipt_path.read_text())
    receipt["total_wall_seconds"] = timing
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    with pytest.raises(RuntimeError, match="timing is invalid"):
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)


def test_missing_unit_returns_none_and_profile_mode_never_resumes(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    # Nothing durable exists: fresh compute is required.
    assert (
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=False)
        is None
    )
    # A sealed solve unit never short-circuits profiling.
    _sealed_unit(run_root, config_path, 0, "fused13")
    assert (
        qtip._validated_existing_unit(config_path, run_root, 9, profile_mode=True)
        is None
    )


def test_manifest_bound_rht_seed_has_no_package_campaign_material() -> None:
    material = "model-index-sha-and-run-manifest-sha"
    expected = qtip._canonical_rht_seed(material, 3, 17, "down")
    config = {
        "rht_seed_policy": "qtip-rht-manifest-v1",
        "rht_seed_material": material,
        "rht_seed": expected,
    }

    assert qtip._resolve_rht_seed(
        config,
        {},
        layer=3,
        expert=17,
        projection="down",
    ) == (expected, "qtip-rht-manifest-v1")


def test_explicit_source_policy_requires_hash_bound_materialization() -> None:
    with pytest.raises(ValueError, match="lacks hash-bound materialization"):
        qtip._resolve_rht_seed(
            {
                "rht_seed_policy": "qtip-rht-explicit-seed-v1",
                "rht_seed": 6134856253202741598,
            },
            {},
            layer=35,
            expert=0,
            projection="fused13",
        )


@pytest.mark.parametrize("seed", [None, -1, 1 << 63, "6134856253202741598"])
def test_explicit_source_policy_rejects_unbounded_or_non_integer_seed(seed) -> None:
    with pytest.raises(ValueError, match="explicit RHT seed"):
        qtip._resolve_rht_seed(
            {
                "rht_seed_policy": "qtip-rht-explicit-seed-v1",
                "rht_seed": seed,
                "materialization": {
                    "run_manifest_sha256": "1" * 64,
                    "source_config_sha256": "2" * 64,
                },
            },
            {},
            layer=35,
            expert=0,
            projection="fused13",
        )


def test_campaign_named_explicit_seed_policy_is_not_whitelisted() -> None:
    with pytest.raises(ValueError, match="unsupported RHT seed policy"):
        qtip._resolve_rht_seed(
            {"rht_seed_policy": "qtip-rht-bounded36-v1", "rht_seed": 7},
            {},
            layer=35,
            expert=0,
            projection="fused13",
        )


@pytest.mark.parametrize(
    "projection",
    ["", "../down", "fused13/../../escape", "/tmp/down", "other"],
)
def test_qtip_projection_rejects_output_path_control(projection: str) -> None:
    with pytest.raises(ValueError, match="unsupported QTIP projection"):
        qtip_materialize.validate_qtip_projection(projection)


@pytest.mark.parametrize("projection", ["fused13", "down"])
def test_qtip_projection_accepts_only_public_projection_names(projection: str) -> None:
    assert qtip_materialize.validate_qtip_projection(projection) == projection


def test_hessian_binding_accepts_hash_identical_prefetched_capture_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-captures"
    staged = tmp_path / "prefetched-captures"
    source.mkdir()
    staged.mkdir()
    members = []
    for window in range(2):
        capture_name = f"xmoe_L035_win{window:04d}.pt"
        done_name = capture_name + ".DONE.json"
        capture_raw = f"capture-{window}".encode()
        done_raw = (json.dumps({"window": window}) + "\n").encode()
        for root in (source, staged):
            (root / capture_name).write_bytes(capture_raw)
            (root / done_name).write_bytes(done_raw)
        members.append(
            {
                "window": window,
                "capture": {
                    "path": str(source / capture_name),
                    "bytes": len(capture_raw),
                    "sha256": hashlib.sha256(capture_raw).hexdigest(),
                },
                "capture_done": {
                    "path": str(source / done_name),
                    "bytes": len(done_raw),
                    "sha256": hashlib.sha256(done_raw).hexdigest(),
                },
            }
        )
    manifest_path = tmp_path / "HESSIAN_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-hessian-layer-manifest-v1",
                "status": "PASS",
                "layer": 35,
                "windows": 2,
                "capture_root": str(source),
                "members": members,
            },
            sort_keys=True,
        )
        + "\n"
    )
    config = {
        "fit_capture_root": str(staged),
        "fit_windows": 2,
        "hessian_layer_manifest": str(manifest_path),
        "hessian_layer_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }

    bound_root, windows, receipt = qtip._bind_hessian_layer_manifest(
        config, layer=35
    )

    assert bound_root == staged.resolve()
    assert windows == 2
    assert receipt["relocated_capture_root"] is True
    assert receipt["manifest_capture_root"] == str(source.resolve())


def test_hessian_binding_rejects_digestless_members_at_original_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "captures"
    source.mkdir()
    capture = source / "xmoe_L035_win0000.pt"
    done = source / "xmoe_L035_win0000.pt.DONE.json"
    capture.write_bytes(b"capture")
    done.write_text('{"md5":"fixture"}\n')
    manifest_path = tmp_path / "HESSIAN_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-hessian-layer-manifest-v1",
                "status": "PASS",
                "layer": 35,
                "windows": 1,
                "capture_root": str(source),
                "members": [
                    {
                        "window": 0,
                        "capture": {
                            "path": str(capture),
                            "bytes": capture.stat().st_size,
                        },
                        "capture_done": {
                            "path": str(done),
                            "bytes": done.stat().st_size,
                        },
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    config = {
        "fit_capture_root": str(source),
        "fit_windows": 1,
        "hessian_layer_manifest": str(manifest_path),
        "hessian_layer_manifest_sha256": _sha256(manifest_path),
    }

    with pytest.raises(ValueError, match="lacks SHA-256"):
        qtip._bind_hessian_layer_manifest(config, layer=35)


def test_load_captures_verifies_done_receipt_md5(tmp_path: Path) -> None:
    capture = tmp_path / "xmoe_L003_win0000.pt"
    torch.save(
        {
            "layer": 3,
            "win": 0,
            "x": torch.zeros((1, 2)),
            "topk": torch.zeros((1, 1), dtype=torch.int64),
            "w": torch.ones((1, 1)),
        },
        capture,
    )
    (tmp_path / "xmoe_L003_win0000.pt.DONE.json").write_text(
        json.dumps({"md5": "0" * 32}) + "\n"
    )

    with pytest.raises(RuntimeError, match="capture MD5 mismatch"):
        qtip._load_captures(tmp_path, 3, 1)


def test_tlut_cache_does_not_retain_unreferenced_tensor(tmp_path: Path) -> None:
    source = tmp_path / "tlut.pt"
    torch.save({"tlut": torch.arange(8, dtype=torch.float32)}, source)
    tensor = qtip._load_tlut(source)
    reference = weakref.ref(tensor)
    key = source.resolve()
    assert qtip._TLUT_CACHE[key] is tensor

    del tensor
    gc.collect()

    assert reference() is None
    assert key not in qtip._TLUT_CACHE


def test_public_receipt_removes_host_and_absolute_local_paths(tmp_path: Path) -> None:
    receipt = qtip._public_receipt(
        {
            "host": "private-host",
            "artifact": str(tmp_path / "QTIP_UNIT.pt"),
            "nested": [{"path": str(tmp_path / "capture.pt")}],
        }
    )

    assert "host" not in receipt
    assert receipt["artifact"] == "QTIP_UNIT.pt"
    assert receipt["nested"] == [{"path": "capture.pt"}]


def test_load_weight_derives_shapes_from_model_metadata(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text(
        json.dumps({"hidden_size": 64, "moe_intermediate_size": 32}) + "\n"
    )
    tensors = {}
    weight_map = {}
    for name, packed_shape, scale_shape in (
        ("w1", (32, 32), (32, 2)),
        ("w3", (32, 32), (32, 2)),
        ("w2", (64, 16), (64, 1)),
    ):
        weight_key = f"layers.0.ffn.experts.0.{name}.weight"
        scale_key = f"layers.0.ffn.experts.0.{name}.scale"
        tensors[weight_key] = torch.zeros(packed_shape, dtype=torch.uint8)
        tensors[scale_key] = torch.full(scale_shape, 127, dtype=torch.uint8)
        weight_map[weight_key] = "model.safetensors"
        weight_map[scale_key] = "model.safetensors"
    save_file(tensors, model_root / "model.safetensors")
    (model_root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}) + "\n"
    )

    fused, _ = qtip._load_weight(model_root, 0, 0, "fused13")
    down, _ = qtip._load_weight(model_root, 0, 0, "down")

    assert tuple(fused.shape) == (64, 64)
    assert tuple(down.shape) == (64, 32)


def test_qtip2_canonical_pack_geometry_is_bound_to_manifest_contract() -> None:
    rings = __import__("banana_smasher.qtip_rings", fromlist=["resolve_qtip_ring"])
    ring = rings.resolve_qtip_ring("2.00")

    assert ring.geometries == ((16, 2, 2),)
    assert rings.canonical_qtip_packed_shape(
        codebook=ring.codebook,
        geometry=ring.geometries[0],
        matrix_shape=(4096, 4096),
    ) == (65536, 32)
    contract = ring.codebook["pack_contract"]
    assert "packed_shape" not in contract
    malformed = dict(ring.codebook)
    malformed["pack_contract"] = {**contract, "output_rows": "matrix_rows"}
    with pytest.raises(ValueError, match="canonical pack contract"):
        rings.canonical_qtip_packed_shape(
            codebook=malformed,
            geometry=(16, 2, 2),
            matrix_shape=(4096, 4096),
        )


def test_sha_declared_public_runner_physically_packs_qtip2_at_runtime(
    tmp_path: Path,
) -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "src/banana_smasher/qtip_runner.py"
    )
    declared_sha = _sha256(runner)
    manifest = tmp_path / "QTIP_RUN_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-run-manifest-v1",
                "status": "PASS",
                "tiers": [
                    {
                        "name": "qtip@2.00",
                        "bindings": {
                            "qtip_runner": {
                                "path": str(runner),
                                "bytes": runner.stat().st_size,
                                "sha256": declared_sha,
                            }
                        },
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    config = {
        "tier": "qtip@2.00",
        "qtip_runner": str(runner),
        "materialization": {
            "run_manifest": str(manifest),
            "run_manifest_sha256": _sha256(manifest),
        },
    }

    resolved_path, resolved_sha = qtip._declared_public_qtip_runner(config)
    with pytest.raises(ValueError, match="runner SHA mismatch"):
        qtip._load_public_qtip_runner(resolved_path, "0" * 64)
    loaded = qtip._load_public_qtip_runner(resolved_path, resolved_sha)

    rings = __import__("banana_smasher.qtip_rings", fromlist=["resolve_qtip_ring"])
    ring = rings.resolve_qtip_ring("2.00")
    source = torch.empty((4_096, 4_096), device="meta")
    codebook = SimpleNamespace(L=16, K=2, V=2)
    qtip._bind_public_runner_pack_contract(
        codebook,
        {
            "geometry": dict(
                zip(("L", "K", "V"), ring.geometries[0], strict=True)
            ),
            "codebook": dict(ring.codebook),
        },
        source,
    )
    packed = torch.empty((65_536, 32), dtype=torch.uint16)

    assert Path(loaded.__file__).resolve() == runner.resolve()
    assert "_bind_public_runner_pack_contract" in qtip.main.__code__.co_names
    assert loaded.validate_manifest_packed_layout(
        codebook, packed, *source.shape
    ) == (65_536, 32)
    with pytest.raises(RuntimeError, match="canonical packed shape/dtype mismatch"):
        loaded.validate_manifest_packed_layout(
            codebook, torch.empty((65_536, 31), dtype=torch.uint16), *source.shape
        )
    assert (
        loaded.pack_kernel_layout.__globals__["validate_manifest_packed_layout"]
        is loaded.validate_manifest_packed_layout
    )
    assert "validate_manifest_packed_layout" in loaded.pack_kernel_layout.__code__.co_names


def test_public_runner_loader_accepts_manifest_selected_python_with_exact_sha(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "untrusted_runner.py"
    runner.write_text(
        "def pack_kernel_layout(cb, states, m, k):\n"
        "    return states\n"
        "def build_qtip(cb, states, m, k):\n"
        "    return pack_kernel_layout(cb, states, m, k)\n"
    )

    loaded = qtip._load_public_qtip_runner(runner, _sha256(runner))

    assert loaded.__file__ is not None
    assert Path(loaded.__file__).resolve() == runner.resolve()
    with pytest.raises(ValueError, match="runner SHA mismatch"):
        qtip._load_public_qtip_runner(runner, "0" * 64)


def test_public_runner_loader_owns_manifest_shape_for_generic_inline_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "public_runner.py"
    runner.write_text(
        "def pack_kernel_layout(cb, states, m, k):\n"
        "    raise RuntimeError('legacy runner hardcodes a packed shape')\n"
        "def build_qtip(cb, states, m, k):\n"
        "    return pack_kernel_layout(cb, states, m, k)\n"
    )
    declared_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    _trust_test_runner(monkeypatch, runner)
    loaded = qtip._load_public_qtip_runner(runner, declared_sha)
    cb = SimpleNamespace(
        L=2,
        K=1,
        V=1,
        pack_trellis=lambda tiled: tiled.to(torch.uint16),
        unpack_trellis=lambda packed, _tile_size: packed,
        _banana_smasher_public_runner_pack_contract={
            "schema": "banana-smasher-public-runner-pack-contract-v1",
            "geometry": (2, 1, 1),
            "matrix_shape": (4, 4),
            "input_tile": (2, 2),
            "dtype": "uint16",
            "packed_words_per_tile_per_k": 4,
            "output_rows": "input_tile_grid",
            "expected_shape": (4, 4),
        }
    )
    states = torch.arange(16).reshape(4, 4)

    packed, receipt = loaded.build_qtip(cb, states, 4, 4)
    assert tuple(packed.shape) == (4, 4)
    assert receipt["kernel_swizzle"] == "manifest-canonical-direct"
    cb.pack_trellis = lambda tiled: tiled[:, :-1].to(torch.uint16)
    with pytest.raises(RuntimeError, match="manifest packed shape/dtype mismatch"):
        loaded.build_qtip(cb, states, 4, 4)
    cb.pack_trellis = lambda tiled: tiled.to(torch.uint16)
    cb.unpack_trellis = lambda packed, _tile_size: packed.to(torch.int64) + 1
    with pytest.raises(RuntimeError, match="manifest pack roundtrip mismatch"):
        loaded.build_qtip(cb, states, 4, 4)


def test_generic_inline_pack_requires_manifest_binding_and_exact_state_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "public_runner.py"
    runner.write_text(
        "def pack_kernel_layout(cb, states, m, k):\n"
        "    return 'legacy-bypass'\n"
        "def build_qtip(cb, states, m, k):\n"
        "    return pack_kernel_layout(cb, states, m, k)\n"
    )
    _trust_test_runner(monkeypatch, runner)
    loaded = qtip._load_public_qtip_runner(
        runner, hashlib.sha256(runner.read_bytes()).hexdigest()
    )
    cb = SimpleNamespace(
        L=2,
        K=1,
        V=1,
        pack_trellis=lambda tiled: tiled.to(torch.uint16),
        unpack_trellis=lambda packed, _tile_size: packed,
    )
    states = torch.arange(16).reshape(4, 4)

    with pytest.raises(RuntimeError, match="manifest pack binding missing"):
        loaded.build_qtip(cb, states, 4, 4)

    cb._banana_smasher_public_runner_pack_contract = {
        "expected_shape": (4, 4),
        "output_rows": "input_tile_grid",
        "packed_words_per_tile_per_k": 4,
        "dtype": "uint16",
        "input_tile": (2, 2),
        "matrix_shape": (4, 4),
        "geometry": (2, 1, 1),
        "schema": "banana-smasher-public-runner-pack-contract-v1",
    }
    with pytest.raises(RuntimeError, match="input state shape mismatch"):
        loaded.build_qtip(cb, states.reshape(2, 8), 4, 4)

    packed, _ = loaded.build_qtip(cb, states, 4, 4)
    assert tuple(packed.shape) == (4, 4)

    loaded_again = qtip._load_public_qtip_runner(
        runner, hashlib.sha256(runner.read_bytes()).hexdigest()
    )
    assert loaded_again is not loaded
    assert (
        loaded.build_qtip.__globals__["pack_kernel_layout"]
        is loaded.pack_kernel_layout
    )
    assert (
        loaded_again.build_qtip.__globals__["pack_kernel_layout"]
        is loaded_again.pack_kernel_layout
    )
    assert loaded_again.pack_kernel_layout is not loaded.pack_kernel_layout

    cb.unpack_trellis = lambda packed, _tile_size: torch.zeros_like(packed)
    with pytest.raises(RuntimeError, match="manifest pack roundtrip mismatch"):
        loaded.build_qtip(cb, states, 4, 4)


def test_manifest_pack_accepts_vectorized_states_and_rejects_source_matrix_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "public_runner.py"
    runner.write_text(
        "def pack_kernel_layout(cb, states, m, k):\n"
        "    return 'legacy-bypass'\n"
        "def build_qtip(cb, states, m, k):\n"
        "    return pack_kernel_layout(cb, states, m, k)\n"
    )
    _trust_test_runner(monkeypatch, runner)
    loaded = qtip._load_public_qtip_runner(
        runner, hashlib.sha256(runner.read_bytes()).hexdigest()
    )
    cb = SimpleNamespace(
        L=2,
        K=1,
        V=2,
        pack_trellis=lambda tiled: tiled.to(torch.uint16),
        unpack_trellis=lambda packed, _tile_size: packed,
        _banana_smasher_public_runner_pack_contract={
            "schema": "banana-smasher-public-runner-pack-contract-v1",
            "geometry": (2, 1, 2),
            "matrix_shape": (4, 4),
            "input_tile": (2, 2),
            "dtype": "uint16",
            "packed_words_per_tile_per_k": 2,
            "output_rows": "input_tile_grid",
            "expected_shape": (4, 2),
        },
    )
    vectorized_states = torch.arange(8).reshape(4, 2)

    packed, receipt = loaded.build_qtip(cb, vectorized_states, 4, 4)

    assert tuple(packed.shape) == (4, 2)
    assert receipt["canonical_pack_roundtrip_exact"] is True
    with pytest.raises(RuntimeError, match="input state shape mismatch"):
        loaded.build_qtip(cb, torch.arange(16).reshape(4, 4), 4, 4)


def test_public_runner_loader_rejects_build_path_that_does_not_own_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "public_runner.py"
    runner.write_text(
        "def pack_kernel_layout(states):\n"
        "    return states\n"
        "def build_qtip(states):\n"
        "    return states\n"
    )
    declared_sha = hashlib.sha256(runner.read_bytes()).hexdigest()
    _trust_test_runner(monkeypatch, runner)

    with pytest.raises(RuntimeError, match="does not own canonical pack path"):
        qtip._load_public_qtip_runner(runner, declared_sha)


def test_unit_payload_geometry_binds_selected_qtip2_codebook() -> None:
    candidate = {
        "geometry": {"L": 16, "K": 3, "V": 2, "tlut_bits": 9},
    }
    config = {
        "geometry": {"L": 16, "K": 2, "V": 2},
        "codebook": {
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
            "td_x": 16,
            "td_y": 16,
        },
    }

    qtip._bind_candidate_geometry(candidate, config)

    assert candidate["geometry"] == {
        "L": 16,
        "K": 2,
        "V": 2,
        "tlut_bits": 9,
        "decode_mode": "quantlut_sym",
        "td_x": 16,
        "td_y": 16,
    }


def test_builder_memory_contract_covers_preallocated_and_final_contiguous_outputs() -> None:
    cb = SimpleNamespace(V=2, idx_dtype=torch.int32)
    source = torch.empty((65_536, 6_144), device="meta")

    contract = qtip._bind_builder_memory_contract(cb, source)

    state_elements = source.numel() // cb.V
    assert contract == {
        "schema": "banana-smasher-qtip-builder-memory-v2",
        "state_elements": state_elements,
        "state_storage_bytes": state_elements * 4,
        "retained_output_bytes": source.numel() * 4,
    }
    assert cb._banana_smasher_memory_contract is contract
    assert cb._banana_smasher_observed_state_elements == 0
    with pytest.raises(TypeError):
        contract["state_elements"] = 0
    with pytest.raises(RuntimeError, match="state output closure mismatch"):
        qtip._verify_builder_memory_contract(cb)
    cb._banana_smasher_observed_state_elements = state_elements
    assert qtip._verify_builder_memory_contract(cb) is contract


def test_builder_memory_contract_rejects_replaced_derived_fields() -> None:
    cb = SimpleNamespace(V=2, idx_dtype=torch.int32)
    source = torch.empty((65_536, 6_144), device="meta")
    contract = qtip._bind_builder_memory_contract(cb, source)
    replacement = dict(contract)
    replacement["state_storage_bytes"] = 0
    replacement["retained_output_bytes"] = 0
    cb._banana_smasher_memory_contract = replacement
    cb._banana_smasher_observed_state_elements = contract["state_elements"]

    with pytest.raises(RuntimeError, match="builder memory contract drift"):
        qtip._verify_builder_memory_contract(cb)


def test_qtip2_candidate_physically_matches_manifest_pack_shape() -> None:
    from banana_smasher.qtip_rings import resolve_qtip_ring

    ring = resolve_qtip_ring("2.00")
    config = {
        "geometry": dict(zip(("L", "K", "V"), ring.geometries[0], strict=True)),
        "codebook": dict(ring.codebook),
    }
    source = torch.empty((4096, 4096), device="meta")
    candidate = {
        "trellis": torch.empty((65536, 32), dtype=torch.uint16, device="meta")
    }

    assert qtip._validate_candidate_packed_shape(candidate, config, source) == (
        65536,
        32,
    )
    candidate["trellis"] = torch.empty(
        (65536, 31), dtype=torch.uint16, device="meta"
    )
    with pytest.raises(RuntimeError, match="packed shape/dtype mismatch"):
        qtip._validate_candidate_packed_shape(candidate, config, source)


@pytest.mark.parametrize(("bpw", "K"), [("1.25", 1), ("1.50", 2), ("3.25", 4)])
def test_existing_pass_unit_resumes_unchanged_for_arbitrary_qtip_bpw(
    tmp_path: Path, bpw: str, K: int
) -> None:
    config_path = tmp_path / "configs" / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    config = _config(config_path, 0, "fused13")
    config.update({"bpw": bpw, "geometry": {"L": 16, "K": K, "V": 2}})
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n")
    artifact, receipt = _sealed_unit(run_root, config_path, 0, "fused13")
    before = (
        artifact.read_bytes(), receipt.read_bytes(),
        artifact.stat().st_mtime_ns, receipt.stat().st_mtime_ns,
    )

    resumed = qtip._validated_existing_unit(
        config_path, run_root, 9, profile_mode=False
    )

    assert resumed is not None
    assert resumed["status"] == "PASS"
    assert (
        artifact.read_bytes(), receipt.read_bytes(),
        artifact.stat().st_mtime_ns, receipt.stat().st_mtime_ns,
    ) == before


def test_fsynced_pass_unit_returns_public_fast_path_receipt_without_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_root = tmp_path / "configs"
    config_path = config_root / "E000_fused13.json"
    run_root = _write_run_root(tmp_path)
    _config(config_path, 0, "fused13")
    artifact, unit_receipt = _sealed_unit(
        run_root, config_path, 0, "fused13"
    )
    before = (
        artifact.read_bytes(),
        unit_receipt.read_bytes(),
        artifact.stat().st_mtime_ns,
        unit_receipt.stat().st_mtime_ns,
    )

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError("a hash-valid fsynced PASS unit must resume")

    monkeypatch.setattr(qtip, "main", must_not_compute)
    batch = qtip.main_many(config_root, run_root, 9, profile_mode=False)

    captured = capsys.readouterr()
    receipt_path = run_root / "solve" / "L009" / "QTIP_BATCH_RECEIPT.json"
    durable = json.loads(receipt_path.read_text())
    assert captured.out == ""
    assert batch == durable
    assert batch["schema"] == "banana-smasher-qtip-resident-batch-v1"
    assert batch["status"] == "PASS"
    assert batch["mode"] == "solve"
    assert batch["units"] == batch["resumed_units"] == 1
    assert batch["computed_units"] == 0
    assert batch["ordered_assignments"] == [
        {
            "layer": 9,
            "expert": 0,
            "projection": "fused13",
            "assignment_sha256": json.loads(unit_receipt.read_text())[
                "assignment_sha256"
            ],
        }
    ]
    assert len(batch["ordered_assignment_sha256"]) == 64
    assert all(
        math.isfinite(value) and value >= 0.0
        for value in batch["phase_seconds"].values()
    )
    assert (
        artifact.read_bytes(),
        unit_receipt.read_bytes(),
        artifact.stat().st_mtime_ns,
        unit_receipt.stat().st_mtime_ns,
    ) == before


@pytest.mark.parametrize("bpw", ["1.25", "1.50", "3.25"])
def test_ordered_fractional_ring_accepts_manifest_declared_mixed_geometries(
    tmp_path: Path, bpw: str
) -> None:
    from banana_smasher.qtip_rings import (
        assign_ring_geometries,
        qtip_ring_manifest,
        resolve_qtip_ring,
    )

    ring = resolve_qtip_ring(bpw)
    config_root = tmp_path / "configs"
    manifest = config_root / "QTIP_RUN_MANIFEST.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-run-manifest-v1",
                "status": "PASS",
                "tiers": [
                    {
                        "name": ring.tier,
                        "ring": qtip_ring_manifest(ring),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    manifest_sha = _sha256(manifest)
    paths: list[Path] = []
    identities = [(9, expert, "fused13") for expert in range(4)]
    assignments = assign_ring_geometries(ring, identities)
    for _layer, expert, projection in identities:
        geometry = assignments[(9, expert, projection)]
        path = config_root / f"E{expert:03d}_fused13.json"
        config = _config(path, expert, projection)
        config.update(
            {
                "tier": ring.tier,
                "bpw": ring.canonical_bpw,
                "geometry": dict(zip(("L", "K", "V"), geometry, strict=True)),
                "backend": ring.backend_for(geometry),
                "codebook": dict(ring.codebook),
                "aot": dict(ring.aot),
                "materialization": {
                    "schema": "banana-smasher-qtip-config-materialization-v1",
                    "run_manifest": str(manifest),
                    "run_manifest_sha256": manifest_sha,
                    "qtip_ring_bpw": ring.canonical_bpw,
                },
            }
        )
        path.write_text(json.dumps(config, sort_keys=True) + "\n")
        paths.append(path)

    (config_root / "QTIP_CONFIG_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-config-manifest-v1",
                "status": "PASS",
                "tier": ring.tier,
                "run_manifest_sha256": manifest_sha,
                "ring": {"bpw": ring.canonical_bpw},
                "member_records": [
                    {
                        "layer": 9,
                        "expert": json.loads(path.read_text())["expert"],
                        "projection": json.loads(path.read_text())["projection"],
                        "path": str(path),
                        "sha256": _sha256(path),
                    }
                    for path in paths
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )

    assert qtip._ordered_qtip_configs(
        config_root, 9, tier=ring.tier, all_cells=False
    ) == paths

    first = json.loads(paths[0].read_text())
    different_path = next(
        path
        for path in paths[1:]
        if json.loads(path.read_text())["geometry"] != first["geometry"]
    )
    different = json.loads(different_path.read_text())
    first["geometry"], different["geometry"] = (
        different["geometry"], first["geometry"]
    )
    first["backend"], different["backend"] = (
        different["backend"], first["backend"]
    )
    paths[0].write_text(json.dumps(first, sort_keys=True) + "\n")
    different_path.write_text(json.dumps(different, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="ring assignment mismatch"):
        qtip._ordered_qtip_configs(
            config_root, 9, tier=ring.tier, all_cells=False
        )


def test_resident_batch_skips_39_valid_units_and_starts_at_first_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    paths: list[Path] = []
    sealed: list[tuple[Path, Path, bytes, bytes, int, int]] = []
    for unit in range(40):
        expert, projection_index = divmod(unit, 2)
        projection = ("fused13", "down")[projection_index]
        path = config_root / f"E{expert:03d}_{projection}.json"
        _config(path, expert, projection)
        paths.append(path)
        if unit < 39:
            artifact, receipt = _sealed_unit(run_root, path, expert, projection)
            sealed.append(
                (
                    artifact,
                    receipt,
                    artifact.read_bytes(),
                    receipt.read_bytes(),
                    artifact.stat().st_mtime_ns,
                    receipt.stat().st_mtime_ns,
                )
            )

    calls: list[Path] = []
    kernel_cache_root = tmp_path / "sealed-kernel-cache"

    def fake_main(
        path: Path,
        root: Path,
        layer: int,
        *,
        profile_mode: bool,
        kernel_cache_root: Path | None = None,
    ):
        assert kernel_cache_root == tmp_path / "sealed-kernel-cache"
        calls.append(path)
        value = json.loads(path.read_text())
        return {
            "status": "PASS",
            "layer": layer,
            "expert": value["expert"],
            "projection": value["projection"],
            "assignment_sha256": hashlib.sha256(path.name.encode()).hexdigest(),
            "total_wall_seconds": 1.0,
        }

    monkeypatch.setattr(qtip, "main", fake_main)
    batch = qtip.main_many(
        config_root,
        run_root,
        9,
        profile_mode=False,
        resume=True,
        resume_flag_explicit=True,
        kernel_cache_root=kernel_cache_root,
    )

    assert calls == [paths[39]]
    assert batch["resumed_units"] == 39
    assert batch["computed_units"] == 1
    assert batch["resume"] == {
        "enabled": True,
        "explicit_flag": True,
        "policy": "hash-validate-pass-skip",
    }
    assert batch["kernel_cache_root"] == kernel_cache_root.name
    assert set(batch["phase_seconds"]) == {
        "resume_preflight",
        "unit_dispatch",
        "batch_boundary_overhead",
        "batch_receipt_fsync",
    }
    assert batch["phase_seconds"]["batch_boundary_overhead"] >= 0.0
    assert batch["phase_seconds"]["batch_receipt_fsync"] >= 0.0
    for artifact, receipt, artifact_bytes, receipt_bytes, artifact_mtime, receipt_mtime in sealed:
        assert artifact.read_bytes() == artifact_bytes
        assert receipt.read_bytes() == receipt_bytes
        assert artifact.stat().st_mtime_ns == artifact_mtime
        assert receipt.stat().st_mtime_ns == receipt_mtime


def test_resident_batch_skips_a_complete_layer_without_rewriting_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = tmp_path / "configs"
    run_root = _write_run_root(tmp_path)
    sealed: list[tuple[Path, Path, bytes, bytes, int, int]] = []
    for expert, projection in ((0, "fused13"), (0, "down")):
        path = config_root / f"E{expert:03d}_{projection}.json"
        _config(path, expert, projection)
        artifact, receipt = _sealed_unit(run_root, path, expert, projection)
        sealed.append(
            (
                artifact,
                receipt,
                artifact.read_bytes(),
                receipt.read_bytes(),
                artifact.stat().st_mtime_ns,
                receipt.stat().st_mtime_ns,
            )
        )

    def must_not_compute(*_args, **_kwargs):
        raise AssertionError("a complete hash-valid layer must skip all unit compute")

    monkeypatch.setattr(qtip, "main", must_not_compute)
    batch = qtip.main_many(config_root, run_root, 9, profile_mode=False)

    assert batch["resumed_units"] == 2
    assert batch["computed_units"] == 0
    for artifact, receipt, artifact_bytes, receipt_bytes, artifact_mtime, receipt_mtime in sealed:
        assert artifact.read_bytes() == artifact_bytes
        assert receipt.read_bytes() == receipt_bytes
        assert artifact.stat().st_mtime_ns == artifact_mtime
        assert receipt.stat().st_mtime_ns == receipt_mtime


@pytest.mark.parametrize(
    ("bpw", "K"),
    [("1.25", 1), ("1.50", 1), ("3.00", 3), ("3.25", 4)],
)
def test_configured_viterbi_accepts_every_packaged_kernel_geometry(
    bpw: str,
    K: int,
) -> None:
    cb = SimpleNamespace(L=16, K=K, V=2)
    exact = SimpleNamespace(
        geometry=lambda value: {
            "implementation": "test-aot",
            "L": value.L,
            "K": value.K,
            "V": value.V,
            "branches_per_prefix": 1 << (value.K * value.V),
            "steps": 128,
        },
        exact_prefix_viterbi=lambda _cb, x, _overlap=None: x,
    )

    metadata = qtip._install_configured_viterbi(
        cb,
        exact,
        qtip._ExactTimers(),
        {"bpw": bpw, "geometry": {"L": 16, "K": K, "V": 2}},
        profile_mode=False,
    )

    assert metadata["L"] == 16
    assert metadata["K"] == K
    assert metadata["V"] == 2
    assert metadata["production_default"] is True


def test_configured_viterbi_unknown_geometry_names_compiled_set_producer() -> None:
    cb = SimpleNamespace(L=17, K=3, V=2)
    exact = SimpleNamespace()

    with pytest.raises(
        ValueError,
        match=r"geometry .* not in compiled set.*smash kernels build --tier qtip --bpw 9\.00",
    ):
        qtip._install_configured_viterbi(
            cb,
            exact,
            qtip._ExactTimers(),
            {"bpw": "9.00", "geometry": {"L": 17, "K": 3, "V": 2}},
            profile_mode=False,
        )


def test_configured_viterbi_dispatch_is_backend_data_not_geometry(monkeypatch) -> None:
    cb = SimpleNamespace(L=16, K=2, V=2)
    calls: list[str] = []

    def exact_prefix(_cb, x, _overlap=None):
        calls.append("generic")
        return x

    exact = SimpleNamespace(
        geometry=lambda value: {
            "implementation": "persistent-prefix-generic-aot-v1",
            "L": value.L,
            "K": value.K,
            "V": value.V,
            "branches_per_prefix": 1 << (value.K * value.V),
            "steps": 128,
        },
        exact_prefix_viterbi=exact_prefix,
    )
    monkeypatch.setattr(
        qtip,
        "backend_for_geometry",
        lambda geometry: "persistent-prefix-generic-aot-v1",
    )

    metadata = qtip._install_configured_viterbi(
        cb,
        exact,
        qtip._ExactTimers(),
        {
            "bpw": "2.00",
            "backend": "persistent-prefix-generic-aot-v1",
            "geometry": {"L": 16, "K": 2, "V": 2},
        },
        profile_mode=False,
    )
    fake_x = SimpleNamespace(is_cuda=True, ndim=2, shape=(256, 1))
    assert cb.quantize_seq(fake_x) is fake_x

    assert metadata["implementation"] == "persistent-prefix-generic-aot-v1"
    assert calls == ["generic"]


def test_resume_preflight_accepts_new_packaged_geometry_before_compute(tmp_path: Path) -> None:
    run_root = _write_run_root(tmp_path)
    config_path = tmp_path / "configs" / "E000_down.json"
    value = _config(config_path, 0, "down")
    value["geometry"] = {"L": 16, "K": 4, "V": 2}
    value["bpw"] = "3.25"
    config_path.write_text(json.dumps(value, sort_keys=True) + "\n")

    assert qtip._validated_existing_unit(
        config_path,
        run_root,
        9,
        profile_mode=False,
    ) is None
