from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace
from typing import cast

import pytest

from banana_smasher.backpack import BackpackPlan, BackpackPlanError, build_backpack
from banana_smasher.backpack_providers import backpack_provider_from_declaration
from banana_smasher.cli import _parser, main


def _plan(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": "banana-smasher-backpack-plan-v1",
        "model": {
            "root": "model",
            "manifest": "model/model.json",
            "revision": "flash-revision",
        },
        "target": {"exact_bytes": 1},
        "tiers": [
            {
                "id": "qtip2",
                "family": "qtip",
                "bpw": 2.0,
                "calibration": "calibration/qtip-v7.json",
            }
        ],
        "anchor": {"bank": "anchor.npz", "teacher": "model"},
        "prediction": {
            "class_caps": {
                "agentic": 1.0,
                "chat": 1.0,
                "code": 1.0,
                "multilingual": 1.0,
                "prose": 1.0,
                "reasoning": 1.0,
            }
        },
        "repair": {"method": "none"},
        "output": {
            "pack": "pack",
            "model_id": "deepseek-v4-flash",
            "instance_id": "fresh-qtip2-v7",
        },
    }


def test_ordinary_qtip2_declaration_resolves_to_native_v7_without_legacy_source_root(
    tmp_path: Path,
) -> None:
    parsed = BackpackPlan.from_mapping(_plan(tmp_path), base_dir=tmp_path)

    tier = parsed.tiers[0]
    provider = backpack_provider_from_declaration(tier)

    assert tier["backend"] == "native_v7"
    assert "source_root" not in tier
    assert provider.provider_id == "qtip2-v7"
    assert provider.kind == "qtip_v7"
    assert provider.runtime_family == "qtip2_v7"
    assert provider.generate.__name__ == "generate_qtip_v7_backpack_candidates"
    assert provider.materialize.__name__ == "materialize_qtip_v7_backpack_layer"


def test_explicit_qtip2_v7_provider_matches_schema_and_parser(tmp_path: Path) -> None:
    document = _plan(tmp_path)
    document["tiers"][0]["provider"] = "qtip2-v7"

    parsed = BackpackPlan.from_mapping(document, base_dir=tmp_path)

    assert parsed.tiers[0]["provider"] == "qtip2-v7"
    assert backpack_provider_from_declaration(parsed.tiers[0]).provider_id == "qtip2-v7"


def test_qtip_200_alias_matches_schema_and_native_v7_parser(tmp_path: Path) -> None:
    document = _plan(tmp_path)
    tiers = document["tiers"]
    assert isinstance(tiers, list) and isinstance(tiers[0], dict)
    tiers[0]["provider"] = "qtip@2.00"
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "schema"
            / "banana-smasher-backpack-plan-v1.schema.json"
        ).read_text()
    )
    jsonschema = pytest.importorskip("jsonschema")

    jsonschema.validate(document, schema)
    parsed = BackpackPlan.from_mapping(document, base_dir=tmp_path)

    assert parsed.tiers[0]["backend"] == "native_v7"
    assert backpack_provider_from_declaration(parsed.tiers[0]).provider_id == "qtip2-v7"


def test_native_v7_candidate_resume_requires_current_model_index_binding(
    tmp_path: Path,
) -> None:
    import banana_smasher.backpack as backpack

    model = tmp_path / "model"
    model.mkdir()
    index = model / "model.safetensors.index.json"
    index.write_text("{}\n")
    identity = hashlib.sha256(index.read_bytes()).hexdigest()
    plan = cast(
        BackpackPlan,
        SimpleNamespace(model={"root": str(model), "revision": identity}),
    )
    binding = {
        "role": "model_index",
        "path": str(index),
        "bytes": index.stat().st_size,
        "sha256": identity,
    }

    assert backpack._validate_native_v7_model_index_binding([binding], plan=plan)
    assert not backpack._validate_native_v7_model_index_binding(
        [{**binding, "role": "calibration_manifest"}], plan=plan
    )
    index.write_text('{"drift": true}\n')
    assert not backpack._validate_native_v7_model_index_binding([binding], plan=plan)


def test_ordinary_qtip2_cannot_select_legacy_packaged_backend(tmp_path: Path) -> None:
    document = _plan(tmp_path)
    document["tiers"][0]["backend"] = "packaged_qtip"
    document["tiers"][0].pop("calibration")
    document["tiers"][0]["source_root"] = "legacy-units"

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).parents[1] / "schema" / "banana-smasher-backpack-plan-v1.schema.json").read_text()
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, schema)
    with pytest.raises(BackpackPlanError, match="ordinary QTIP2.*legacy packaged_qtip"):
        BackpackPlan.from_mapping(document, base_dir=tmp_path)
    with pytest.raises(ValueError, match="ordinary QTIP2.*legacy packaged_qtip"):
        backpack_provider_from_declaration(document["tiers"][0])


def test_explicit_qtip2_v7_provider_cannot_select_packaged_backend(tmp_path: Path) -> None:
    document = _plan(tmp_path)
    document["tiers"][0].update({"provider": "qtip2-v7", "backend": "packaged_qtip"})
    document["tiers"][0].pop("calibration")
    document["tiers"][0]["source_root"] = "legacy-units"

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).parents[1] / "schema" / "banana-smasher-backpack-plan-v1.schema.json").read_text()
    )
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, schema)
    with pytest.raises(BackpackPlanError, match="ordinary QTIP2.*legacy packaged_qtip"):
        BackpackPlan.from_mapping(document, base_dir=tmp_path)
    with pytest.raises(ValueError, match="qtip2-v7.*native_v7"):
        backpack_provider_from_declaration(document["tiers"][0])


def test_build_through_pre_repair_anchor_stops_before_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import banana_smasher.backpack as backpack

    parsed = object()
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(backpack, "_bind_run", lambda *_args, **_kwargs: (parsed, root, "p" * 64))
    monkeypatch.setattr(backpack, "_fixed_assignment_admission", lambda _plan: None)
    monkeypatch.setattr(backpack, "_load_verified_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(backpack, "_sha_file", lambda _path: "s" * 64)
    calls: list[str] = []

    def stage(name: str):
        def run(_plan: object, *, run_root: Path) -> dict[str, object]:
            calls.append(name)
            index = backpack.STAGES.index(name) + 1
            path = backpack._stage_path(run_root, index, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"result": {"method": "qtip2-v7-native"}}))
            return {"status": "PASS", "stage": name, "method": "qtip2-v7-native"}

        return run

    for name, public_name in (
        ("inspect", "inspect_backpack"),
        ("candidates", "generate_backpack_candidates"),
        ("candidate_anchor", "anchor_backpack_candidates"),
        ("pred", "predict_backpack"),
        ("solve_materialize", "solve_backpack"),
        ("pre_repair_anchor", "anchor_backpack"),
        ("repair", "repair_backpack"),
        ("final_score", "score_backpack"),
    ):
        monkeypatch.setattr(backpack, public_name, stage(name))

    result = build_backpack(parsed, run_root=root, through="pre_repair_anchor")

    assert calls == list(backpack.STAGES[:6])
    assert result["stage"] == "pre_repair_anchor"
    assert result["method"] == "qtip2-v7-native"
    assert result["stages"] == list(backpack.STAGES[:6])
    assert not backpack._stage_path(root, 7, "repair").exists()


def test_cli_exposes_same_pre_repair_boundary() -> None:
    args = _parser().parse_args(
        [
            "backpack",
            "build",
            "--plan",
            "plan.json",
            "--run-root",
            "run",
            "--through",
            "pre-repair-anchor",
        ]
    )

    assert args.through == "pre-repair-anchor"


def test_fresh_v7_generator_calls_shipped_producer_and_wire_not_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    import banana_smasher.backpack_qtip_v7 as v7

    def save(name: str, value: np.ndarray) -> tuple[str, str]:
        path = tmp_path / name
        np.save(path, value, allow_pickle=False)
        return path.name, hashlib.sha256(path.read_bytes()).hexdigest()

    lut_path, lut_sha = save("lut.npy", np.arange(1024, dtype=np.float16))
    hessian_path, hessian_sha = save("hessian.npy", np.eye(2, dtype=np.float32))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-v7-calibration-v1",
                "model_basis_sha256": "b" * 64,
                "layers": [
                    {
                        "layer": 0,
                        "lut": {"path": lut_path, "sha256": lut_sha},
                        "hessians": {
                            projection: {
                                "path": hessian_path,
                                "sha256": hessian_sha,
                                "count": 64,
                            }
                            for projection in ("w1", "w2", "w3")
                        },
                    }
                ],
            }
        )
    )
    cells = [
        {
            "cell_id": projection,
            "layer": 0,
            "projection": projection,
            "expert_ids": [0],
            "matrix_shape": [2, 2],
            "weights": np.arange(4, dtype=np.float32) + index,
        }
        for index, projection in enumerate(("w1", "w2", "w3"))
    ]
    calls = {"producer": 0, "wire": 0, "legacy": 0}

    def producer(units, parent_lut, *, output_root):
        calls["producer"] += 1
        assert len(units) == 3
        assert parent_lut.dtype == np.float16
        return [
            {
                "layer": unit["layer"],
                "expert": unit["expert"],
                "projection": unit["projection"],
                "packed_codes": bytes([index + 1, index + 2]),
                "suh": np.ones(2, dtype=np.float16),
                "svh": np.ones(2, dtype=np.float16),
                "global_scale": 1.0,
                "decoded": np.asarray(unit["weight"], dtype=np.float32).reshape(-1),
            }
            for index, unit in enumerate(units)
        ], {
            "method": "qtip2-v7-batch",
            "qfn_calls": 3,
            "extension_calls": 3,
            "cuda_tiles": 3,
            "generic_fallback_calls": 0,
        }

    def wire(*, source_root, lut, layer, output, receipt):
        calls["wire"] += 1
        output.write_bytes(b"native-v7-wire")
        payload = {
            "schema": "banana-smasher-qtip-v7-wire-v1",
            "status": "PASS",
            "layer": layer,
            "wire": str(output),
            "complete_wire_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "generic_fallback_calls": 0,
        }
        receipt.write_text(json.dumps(payload))
        return payload

    def account(*, receipts, output, weight_denominator, weight_denominator_label):
        calls["accounting"] = calls.get("accounting", 0) + 1
        assert len(receipts) == 1
        payload = {
            "schema": "banana-smasher-qtip-v7-model-accounting-v1",
            "status": "PASS",
            "verified_layer_receipts": 1,
            "qtip_routed_stored_bytes": 14,
            "stored_wire_bpw": {"weight_denominator": weight_denominator},
        }
        output.write_text(json.dumps(payload) + "\n")
        return payload

    monkeypatch.setattr(v7, "_produce_native_v7_batch", producer)
    monkeypatch.setattr(
        v7,
        "_materialize_native_v7_layer",
        lambda **_kwargs: pytest.fail("provider materialize seam was bypassed"),
    )
    monkeypatch.setattr(v7, "_account_native_v7_model", account)
    monkeypatch.setattr(
        v7,
        "_load_legacy_packaged_unit",
        lambda *_args, **_kwargs: calls.__setitem__("legacy", calls["legacy"] + 1),
    )

    tier = {
        "id": "qtip2",
        "family": "qtip",
        "bpw": 2.0,
        "backend": "native_v7",
        "calibration": str(calibration),
    }
    run_root = tmp_path / "run"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("preserve")
    run_root.mkdir()
    (run_root / "qtip-v7").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="qtip-v7.*symlink"):
        v7.generate_qtip_v7_backpack_candidates(
            run_root,
            tier=tier,
            cells=cells,
            model_basis_sha256="b" * 64,
            weight_denominator=12,
            materialize=wire,
        )
    assert sentinel.read_text() == "preserve"
    assert set(outside.iterdir()) == {sentinel}
    (run_root / "qtip-v7").unlink()

    result = v7.generate_qtip_v7_backpack_candidates(
        run_root,
        tier=tier,
        cells=cells,
        model_basis_sha256="b" * 64,
        weight_denominator=12,
        materialize=wire,
    )

    assert calls == {"producer": 1, "wire": 1, "legacy": 0, "accounting": 1}
    assert result["method"] == "qtip2-v7-native"
    assert result["producer_calls"] == 1
    assert result["wire_calls"] == 1
    assert result["qfn_calls"] == 3
    assert result["extension_calls"] == 3
    assert result["cuda_tiles"] == 3
    assert result["legacy_packaged_loader_calls"] == 0
    assert result["generic_fallback_calls"] == 0
    assert result["model_accounting"]["status"] == "PASS"
    assert len(result["cells"]) == 3
    assert all(row["algorithm"] == "qtip2-v7-native" for row in result["cells"])


def test_native_v7_layer_materialization_packs_then_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import banana_smasher.backpack_qtip_v7 as v7
    import banana_smasher.qtip_v7_wire as wire

    output = tmp_path / "layer.q2v7layer"
    receipt = tmp_path / "WIRE_RECEIPT.json"
    calls: list[str] = []

    def pack(**kwargs):
        calls.append("pack")
        output.write_bytes(b"native-v7-wire")
        receipt.write_text(json.dumps({"schema": "wire"}) + "\n")
        return {
            "status": "PASS",
            "wire": str(output),
            "wire_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "receipt": str(receipt),
        }

    def verify(**kwargs):
        calls.append("verify")
        assert kwargs == {"wire": output, "receipt": receipt}
        return {
            "status": "PASS",
            "wire": str(output),
            "wire_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "receipt": str(receipt),
            "physical_bytes_authenticated": True,
        }

    monkeypatch.setattr(wire, "pack_qtip_v7_layer", pack)
    monkeypatch.setattr(wire, "verify_qtip_v7_layer", verify)

    result = v7._materialize_native_v7_layer(
        source_root=tmp_path / "members",
        lut=tmp_path / "lut.npy",
        layer=0,
        output=output,
        receipt=receipt,
    )

    assert calls == ["pack", "verify"]
    assert result["physical_bytes_authenticated"] is True


def test_native_v7_batch_transfers_only_one_ten_expert_chunk_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import banana_smasher.backpack_qtip_v7 as v7
    import banana_smasher.qtip_v7_batch as batch

    tracker = {
        "cuda_moves": 0,
        "moves_at_call": [],
        "sizes": [],
        "live_results": 0,
        "live_at_call": [],
    }

    class ResultToken:
        def __init__(self):
            tracker["live_results"] += 1

        def __del__(self):
            tracker["live_results"] -= 1

    class FakeTensor:
        def __init__(self, value):
            self.value = value

        def to(self, device=None):
            if device == "cuda":
                tracker["cuda_moves"] += 1
            return self

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        device=lambda value: value,
        from_numpy=FakeTensor,
    )

    def producer(units, _lut):
        tracker["moves_at_call"].append(tracker["cuda_moves"])
        tracker["sizes"].append(len(units))
        tracker["live_at_call"].append(tracker["live_results"])
        return [
            {
                "expert": unit["expert"],
                "projection": unit["projection"],
                "packed_codes": b"wire",
                "suh": np.ones(2, dtype=np.float16),
                "svh": np.ones(2, dtype=np.float16),
                "global_scale": 1.0,
                "decoded": np.ones(4, dtype=np.float32),
                "_cuda_lifetime_token": ResultToken(),
            }
            for unit in units
        ], {
            "counters": {"qfn_calls": 1, "extension_calls": 1, "cuda_tiles": 1}
        }

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(batch, "produce_qtip2_v7_batch", producer)
    units = [
        {
            "expert": expert,
            "projection": projection,
            "source": np.ones((2, 2), dtype=np.float32),
            "raw_h": np.eye(2, dtype=np.float32),
        }
        for expert in range(12)
        for projection in ("w1", "w2", "w3")
    ]

    v7._produce_native_v7_batch(
        units,
        np.arange(1024, dtype=np.float16),
        output_root=Path("unused"),
    )

    assert tracker["sizes"] == [30, 6]
    assert tracker["moves_at_call"] == [31, 37]
    assert tracker["live_at_call"] == [0, 0]
    assert tracker["live_results"] == 0


def test_native_v7_materialization_rejects_corrupt_post_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import banana_smasher.backpack as backpack

    root = tmp_path / "run"
    source = root / "qtip-v7" / "qtip2" / "L000" / "L000.q2v7layer"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"authenticated-wire")
    wire_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = cast(
        BackpackPlan,
        SimpleNamespace(
            output={"model_id": "flash", "instance_id": "selected-v7"}
        ),
    )
    prior = {
        "inspect": {"target_whole_model_bytes": source.stat().st_size},
        "candidates": {
            "candidate_tiers": [
                {
                    "tier": "qtip2",
                    "method": "qtip2-v7-native",
                    "producer_calls": 1,
                    "wire_calls": 1,
                    "qfn_calls": 1,
                    "extension_calls": 1,
                    "cuda_tiles": 1,
                    "v7_layers": [
                        {
                            "layer": 0,
                            "wire": str(source),
                            "wire_bytes": source.stat().st_size,
                            "wire_sha256": wire_sha,
                        }
                    ],
                }
            ]
        },
    }
    solved = {
        "activated_artifacts": [
            {
                "id": "qtip2-v7-layer-0",
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": wire_sha,
            }
        ],
        "assigned_bytes": source.stat().st_size,
        "cell_payload_bytes": 0,
        "activation_bytes": source.stat().st_size,
        "solver": {},
    }

    def corrupt_copy(_source: Path, destination: Path) -> None:
        destination.write_bytes(source.read_bytes() + b"-corrupt")

    monkeypatch.setattr(backpack.shutil, "copyfile", corrupt_copy)

    with pytest.raises(BackpackPlanError, match="copy identity drift"):
        backpack._materialize_native_v7_pre_repair(
            plan,
            root,
            prior,
            assignment=[],
            solved=solved,
            fixed_bytes=0,
        )


def test_normative_spec_and_schema_valid_fresh_plan_are_linked_and_shipped() -> None:
    project = Path(__file__).parents[1]
    spec = project / "QTIP_V7_DEFAULT_API_SPEC.md"
    example = project / "examples" / "fresh-flash-qtip2-v7.json"
    readme = (project / "README.md").read_text()
    pyproject = tomllib.loads((project / "pyproject.toml").read_text())
    schema = json.loads(
        (project / "schema" / "banana-smasher-backpack-plan-v1.schema.json").read_text()
    )

    assert spec.read_text().startswith("# Native QTIP2-V7 Default API Specification")
    assert "[Native QTIP2-V7 default API specification](QTIP_V7_DEFAULT_API_SPEC.md)" in readme
    assert "[fresh Flash QTIP2-V7 example plan](examples/fresh-flash-qtip2-v7.json)" in readme
    assert '"family": "qtip", "bpw": 2.0, "source_root"' not in readme
    assert "smash verify ./backpack-run/pre-repair-pack" not in readme
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include["QTIP_V7_DEFAULT_API_SPEC.md"] == (
        "banana_smasher/QTIP_V7_DEFAULT_API_SPEC.md"
    )
    assert force_include["examples/fresh-flash-qtip2-v7.json"] == (
        "banana_smasher/examples/fresh-flash-qtip2-v7.json"
    )

    document = json.loads(example.read_text())
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(document, schema)
    native_schema = schema["$defs"]["qtip_native_v7"]
    assert native_schema["required"] == ["id", "family", "bpw", "calibration"]
    assert schema["$defs"]["qtip_packaged_default"]["not"] == {
        "properties": {"bpw": {"const": 2.0}}
    }
    assert document["tiers"] == [
        {
            "id": "qtip2",
            "family": "qtip",
            "bpw": 2.0,
            "calibration": "inputs/qtip-v7-calibration.json",
        }
    ]
    parsed = BackpackPlan.from_mapping(document, base_dir=example.parent)
    assert backpack_provider_from_declaration(parsed.tiers[0]).provider_id == "qtip2-v7"


def test_fresh_fixture_python_and_cli_reach_same_v7_pre_repair_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np
    import banana_smasher.backpack_qtip_v7 as v7

    model = tmp_path / "model"
    model.mkdir()
    cells = []
    weights = []
    for projection_index, projection in enumerate(("w1", "w2", "w3")):
        value = (np.arange(4, dtype=np.float32) + projection_index).reshape(1, 2, 2)
        path = model / f"{projection}.npy"
        np.save(path, value, allow_pickle=False)
        start = sum(array.size for array in weights)
        weights.append(value)
        cells.append(
            {
                "cell_id": projection,
                "path": path.name,
                "feature_slice": [start, start + value.size],
                "layer": 0,
                "projection": projection,
                "expert_ids": [0],
                "matrix_shape": [2, 2],
            }
        )
    index = model / "model.safetensors.index.json"
    index.write_text("{}\n")
    basis = hashlib.sha256(index.read_bytes()).hexdigest()
    manifest = model / "BACKPACK_MODEL.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-backpack-model-v1",
                "revision": basis,
                "expert_count": 1,
                "weight_count": sum(value.size for value in weights),
                "dense_bytes": 0,
                "metadata_bytes": 0,
                "repair_bytes": 0,
                "cells": cells,
            }
        )
        + "\n"
    )
    lut_path = tmp_path / "lut.npy"
    hessian_path = tmp_path / "hessian.npy"
    np.save(lut_path, np.arange(1024, dtype=np.float16), allow_pickle=False)
    np.save(hessian_path, np.eye(2, dtype=np.float32), allow_pickle=False)
    calibration = tmp_path / "qtip-v7-calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-v7-calibration-v1",
                "model_basis_sha256": basis,
                "layers": [
                    {
                        "layer": 0,
                        "lut": {
                            "path": lut_path.name,
                            "sha256": hashlib.sha256(lut_path.read_bytes()).hexdigest(),
                        },
                        "hessians": {
                            projection: {
                                "path": hessian_path.name,
                                "sha256": hashlib.sha256(hessian_path.read_bytes()).hexdigest(),
                                "count": 64,
                            }
                            for projection in ("w1", "w2", "w3")
                        },
                    }
                ],
            }
        )
        + "\n"
    )
    bank = tmp_path / "anchor64.npz"
    classes = np.asarray(
        [("agentic", "chat", "code", "multilingual", "prose", "reasoning")[i % 6]
         for i in range(64)]
    )
    np.savez(
        bank,
        features=np.ones((64, 12), dtype=np.float32),
        classes=classes,
    )
    plan_document = {
        "schema": "banana-smasher-backpack-plan-v1",
        "model": {"root": str(model), "manifest": str(manifest), "revision": basis},
        "target": {"exact_bytes": 272},
        "tiers": [
            {
                "id": "qtip2",
                "family": "qtip",
                "bpw": 2.0,
                "calibration": str(calibration),
            }
        ],
        "anchor": {"bank": str(bank), "teacher": "model"},
        "prediction": {
            "class_caps": {
                name: 1.0
                for name in ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
            }
        },
        "repair": {"method": "none"},
        "output": {
            "pack": str(tmp_path / "unused-final-pack"),
            "model_id": "DeepSeek-V4-Flash-0731",
            "instance_id": "fresh-v7-fixture",
        },
    }
    calls = {"producer": 0, "wire": 0, "runtime_decode": 0, "legacy": 0}

    def producer(units, parent_lut, *, output_root):
        calls["producer"] += 1
        return [
            {
                "layer": unit["layer"],
                "expert": unit["expert"],
                "projection": unit["projection"],
                "packed_codes": b"\x01\x02",
                "suh": np.ones(2, dtype=np.float16),
                "svh": np.ones(2, dtype=np.float16),
                "global_scale": 1.0,
                # Deliberately differs from the selected wire runtime result. The
                # pre-repair anchor must not score this producer sidecar.
                "decoded": np.asarray(unit["weight"], dtype=np.float32).reshape(-1) + 100,
            }
            for unit in units
        ], {
            "method": "qtip2-v7-batch",
            "qfn_calls": 3,
            "extension_calls": 3,
            "cuda_tiles": 3,
            "generic_fallback_calls": 0,
        }

    def wire(*, source_root, lut, layer, output, receipt):
        calls["wire"] += 1
        output.write_bytes(b"0123456789abcdef")
        payload = {
            "schema": "banana-smasher-qtip-v7-wire-v1",
            "status": "PASS",
            "layer": layer,
            "wire": str(output),
            "complete_wire_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "generic_fallback_calls": 0,
        }
        receipt.write_text(json.dumps(payload) + "\n")
        return payload

    def account(*, receipts, output, weight_denominator, weight_denominator_label):
        payload = {
            "schema": "banana-smasher-qtip-v7-model-accounting-v1",
            "status": "PASS",
            "verified_layer_receipts": len(receipts),
            "qtip_routed_stored_bytes": 16,
            "stored_wire_bpw": {"weight_denominator": weight_denominator},
        }
        output.write_text(json.dumps(payload) + "\n")
        return payload

    def runtime_decode(*, selected_layers, cells):
        calls["runtime_decode"] += 1
        assert len(selected_layers) == 1
        selected_wire = Path(selected_layers[0]["path"])
        assert selected_wire.read_bytes() == b"0123456789abcdef"
        return np.concatenate(
            [np.asarray(cell["weights"], dtype=np.float32).reshape(-1) for cell in cells]
        )

    monkeypatch.setattr(v7, "_produce_native_v7_batch", producer)
    monkeypatch.setattr(v7, "_materialize_native_v7_layer", wire)
    monkeypatch.setattr(v7, "_account_native_v7_model", account)
    monkeypatch.setattr(v7, "decode_selected_qtip_v7_backpack_weights", runtime_decode)
    monkeypatch.setattr(
        v7,
        "_load_legacy_packaged_unit",
        lambda *_args, **_kwargs: calls.__setitem__("legacy", calls["legacy"] + 1),
    )

    python_root = tmp_path / "python-run"
    parsed = BackpackPlan.from_mapping(plan_document)
    python_result = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    resumed = build_backpack(parsed, run_root=python_root, through="pre_repair_anchor")

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan_document) + "\n")
    cli_root = tmp_path / "cli-run"
    assert main(
        [
            "backpack", "build", "--plan", str(plan_path), "--run-root", str(cli_root),
            "--through", "pre-repair-anchor",
        ]
    ) == 0
    cli_result = json.loads(capsys.readouterr().out)

    assert calls == {"producer": 2, "wire": 2, "runtime_decode": 2, "legacy": 0}
    assert python_result["method"] == cli_result["method"] == "qtip2-v7-native"
    assert python_result["producer_calls"] == cli_result["producer_calls"] == 1
    assert python_result["wire_calls"] == cli_result["wire_calls"] == 1
    assert python_result["legacy_packaged_loader_calls"] == 0
    assert python_result["generic_fallback_calls"] == 0
    assert python_result["runtime_wire_decode_calls"] == 1
    assert python_result["metrics"]["overall"]["kld"] == 0.0
    assert python_result["completed_stage"] == "pre_repair_anchor"
    assert python_result["repair_executed"] is False
    assert python_result["plan_sha256"] == cli_result["plan_sha256"]
    assert python_result["model_revision"] == basis
    assert python_result["anchor_bank"]["sha256"] == hashlib.sha256(bank.read_bytes()).hexdigest()
    assert python_result["teacher"]["kind"] == "model"
    assert len(python_result["selected_wires"]) == 1
    assert python_result["selected_wires"][0]["bytes"] == 16
    assert python_result["selected_assignment_sha256"] == hashlib.sha256(
        Path(python_result["selected_assignment_receipt"]).read_bytes()
    ).hexdigest()
    assert python_result["stored_wire_bytes"] == 16
    assert python_result["weight_denominator"] == 12
    assert resumed["resumed_stages"] == list((
        "inspect", "candidates", "candidate_anchor", "pred",
        "solve_materialize", "pre_repair_anchor",
    ))
    assignment_path = Path(python_result["selected_assignment_receipt"])
    drifted_assignment = json.loads(assignment_path.read_text())
    drifted_assignment["unexpected_drift"] = True
    assignment_path.write_text(json.dumps(drifted_assignment) + "\n")
    after_drift = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    assert after_drift["resumed_stages"] == [
        "inspect", "candidates", "candidate_anchor", "pred", "pre_repair_anchor"
    ]
    assert "solve_materialize" not in after_drift["resumed_stages"]
    assert "unexpected_drift" not in json.loads(assignment_path.read_text())
    assert after_drift["selected_assignment_sha256"] == hashlib.sha256(
        assignment_path.read_bytes()
    ).hexdigest()

    calibration_document = json.loads(calibration.read_text())
    calibration_document["integrity_probe"] = "drift"
    calibration.write_text(json.dumps(calibration_document) + "\n")
    after_calibration_drift = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    assert after_calibration_drift["resumed_stages"] == ["inspect"]
    assert calls == {"producer": 3, "wire": 3, "runtime_decode": 3, "legacy": 0}

    candidate_stage_path = python_root / "stages" / "02-candidates.json"
    candidate_stage = json.loads(candidate_stage_path.read_text())
    foreign_wire = tmp_path / "foreign.q2v7layer"
    foreign_wire.write_bytes(b"foreign-not-a-produced-v7-wire")
    foreign_row = candidate_stage["result"]["candidate_tiers"][0]["v7_layers"][0]
    foreign_row["wire"] = str(foreign_wire)
    foreign_row["wire_sha256"] = hashlib.sha256(foreign_wire.read_bytes()).hexdigest()
    candidate_stage_path.write_text(json.dumps(candidate_stage) + "\n")
    after_foreign_wire = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    assert "candidates" not in after_foreign_wire["resumed_stages"]
    assert calls == {"producer": 4, "wire": 4, "runtime_decode": 3, "legacy": 0}
    selected_wire = Path(after_foreign_wire["selected_wires"][0]["path"])
    assert selected_wire.read_bytes() != foreign_wire.read_bytes()

    anchor_stage_path = python_root / "stages" / "06-pre-repair-anchor.json"
    anchor_stage = json.loads(anchor_stage_path.read_text())
    anchor_stage["result"]["selected_assignment_sha256"] = "0" * 64
    anchor_stage_path.write_text(json.dumps(anchor_stage) + "\n")
    after_anchor_drift = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    assert "pre_repair_anchor" not in after_anchor_drift["resumed_stages"]
    assert after_anchor_drift["selected_assignment_sha256"] != "0" * 64

    anchor_stage = json.loads(anchor_stage_path.read_text())
    arbitrary_wire = tmp_path / "arbitrary-selected.q2v7layer"
    arbitrary_wire.write_bytes(b"arbitrary-selected-wire")
    arbitrary_row = {
        **anchor_stage["result"]["selected_wires"][0],
        "path": str(arbitrary_wire),
        "bytes": arbitrary_wire.stat().st_size,
        "sha256": hashlib.sha256(arbitrary_wire.read_bytes()).hexdigest(),
    }
    anchor_stage["result"]["selected_wires"] = [arbitrary_row]
    anchor_stage["result"]["anchor_input"]["selected_wires"] = [arbitrary_row]
    anchor_stage_path.write_text(json.dumps(anchor_stage) + "\n")
    after_selected_wire_drift = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    assert "pre_repair_anchor" not in after_selected_wire_drift["resumed_stages"]
    assert after_selected_wire_drift["selected_wires"][0]["path"] != str(arbitrary_wire)

    solve_stage_path = python_root / "stages" / "05-solve-materialize.json"
    solve_stage = json.loads(solve_stage_path.read_text())
    assignment_receipt = json.loads(assignment_path.read_text())
    extra_wire = tmp_path / "extra-activated.q2v7layer"
    extra_wire.write_bytes(b"extra-activated-wire")
    extra_activation = {
        "id": "qtip2-v7-layer-999",
        "path": str(extra_wire),
        "bytes": extra_wire.stat().st_size,
        "sha256": hashlib.sha256(extra_wire.read_bytes()).hexdigest(),
    }
    solve_stage["result"]["activated_artifacts"].append(extra_activation)
    assignment_receipt["activated_artifacts"].append(extra_activation)
    assignment_path.write_text(json.dumps(assignment_receipt) + "\n")
    solve_stage["result"]["assignment_receipt_sha256"] = hashlib.sha256(
        assignment_path.read_bytes()
    ).hexdigest()
    solve_stage_path.write_text(json.dumps(solve_stage) + "\n")
    after_activation_drift = build_backpack(
        parsed, run_root=python_root, through="pre_repair_anchor"
    )
    assert "solve_materialize" not in after_activation_drift["resumed_stages"]
    regenerated_solve = json.loads(solve_stage_path.read_text())["result"]
    assert extra_activation not in regenerated_solve["activated_artifacts"]

    index.write_text('{"drift": true}\n')
    with pytest.raises(BackpackPlanError, match="model revision must equal the source index"):
        build_backpack(parsed, run_root=python_root, through="pre_repair_anchor")
    failed_candidates = json.loads(candidate_stage_path.read_text())
    assert failed_candidates["status"] == "FAIL"
    assert failed_candidates["stage"] == "candidates"

    assert not (python_root / "stages" / "07-repair.json").exists()
    assert not (python_root / "FINAL_RECEIPT.json").exists()
