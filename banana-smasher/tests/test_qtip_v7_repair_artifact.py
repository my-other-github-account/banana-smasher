from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from banana_smasher.cli import main
from banana_smasher.qtip_v7_repair import (
    build_qtip_v7_repair_bundle,
    export_qtip_v7_artifact,
    load_qtip_v7_artifact,
)
from banana_smasher.token_sizing import MemoryBudget
from banana_smasher.update_backends.physical_repair import PhysicalRepairBackend


MEMBER_BYTES = 2_109_444


def _identity() -> dict[str, str]:
    names = (
        "content_sha256", "config_sha256", "assignment_sha256",
        "aot_sha256", "runtime_sha256", "code_sha256",
    )
    return {name: str(index) * 64 for index, name in enumerate(names, start=1)}


def _physical_context(tmp_path: Path) -> dict[str, object]:
    return {
        "output": tmp_path / "update.pt",
        "receipt": tmp_path / "update.receipt.json",
        "identity": _identity(),
        "requested_tokens": 1,
        "physical_tokens": 1,
        "segments": 1,
        "batch_size": 1,
        "memory_sizing": {"physical_tokens": 1},
        "memory_budget": MemoryBudget(
            available_bytes=8 * 1024**3, resident_frozen_bytes=0,
            trainable_bytes=4096, optimizer_bytes=0, staging_bytes=0,
            calibrated_activation_bytes_per_token=1,
        ),
        "resume": True,
        "restart": False,
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _genuine_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    # The immutable V7 member wire is codes + FP16 SUH/SVH + billed FP32 scalar.
    member = root / "L033.E000.w1.q2v7wire"
    member.write_bytes(bytes((index * 29 + 7) & 255 for index in range(MEMBER_BYTES)))
    lut = root / "L033.tlut.f16"
    np.linspace(-2, 2, 1024, dtype=np.float16).astype("<f2").tofile(lut)
    manifest = root / "QTIP_V7_MANIFEST.json"
    manifest.write_text(json.dumps({
        "schema": "banana-smasher-qtip-v7-artifact-v1",
        "rate": 2,
        "members": [{
            "layer": 33, "expert": 0, "projection": "w1",
            "path": member.name, "bytes": MEMBER_BYTES, "sha256": _sha(member),
            "packed_code_bytes": 2_097_152,
        }],
        "layer_luts": [{
            "layer": 33, "path": lut.name, "bytes": 2048, "sha256": _sha(lut),
            "dtype": "float16", "shape": [1024],
        }],
    }, sort_keys=True))
    return manifest


def test_update0_export_preserves_complete_v7_wire(tmp_path: Path) -> None:
    manifest = _genuine_fixture(tmp_path)
    source = load_qtip_v7_artifact(manifest)
    update0 = tmp_path / "update0"
    receipt0 = export_qtip_v7_artifact(manifest=manifest, output=update0)
    read0 = load_qtip_v7_artifact(update0 / "QTIP_V7_MANIFEST.json")

    assert receipt0["update"] == 0
    assert receipt0["complete_wire_bytes"] == MEMBER_BYTES + 2048
    assert read0.member_wire_sha256 == source.member_wire_sha256
    assert read0.layer_luts[33].tobytes() == source.layer_luts[33].tobytes()

    assert not (update0 / "L033.E000.w1.q2v7wire").exists()


def test_public_cli_exports_v7_update0(tmp_path: Path, capsys) -> None:
    manifest = _genuine_fixture(tmp_path)
    output = tmp_path / "cli-update0"

    assert main([
        "qtip-v7-export", "--manifest", str(manifest), "--output", str(output)
    ]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS"
    assert result["command"] == "qtip-v7-export"
    assert load_qtip_v7_artifact(output / "QTIP_V7_MANIFEST.json").complete_wire_bytes == MEMBER_BYTES + 2048


def test_real_update1_trains_shared_lut_and_exports_same_complete_wire(
    tmp_path: Path,
) -> None:
    manifest = _genuine_fixture(tmp_path)
    document = json.loads(manifest.read_text())
    root = manifest.parent
    # A real V7 member is raw packed codes, FP16 SUH/SVH, then one FP32 scale.
    for projection in ("w1", "w2", "w3"):
        path = root / f"L033.E000.{projection}.q2v7wire"
        k, m = ((4096, 2048) if projection != "w2" else (2048, 4096))
        with path.open("wb") as handle:
            handle.write(bytes(2_097_152))
            handle.write(np.ones(k, dtype="<f2").tobytes())
            handle.write(np.ones(m, dtype="<f2").tobytes())
            handle.write(np.array([0.25], dtype="<f4").tobytes())
        document["members"].append({
            "layer": 33, "expert": 0, "projection": projection,
            "path": path.name, "bytes": MEMBER_BYTES, "sha256": _sha(path),
            "packed_code_bytes": 2_097_152,
        })
    document["members"] = document["members"][1:]
    manifest.write_text(json.dumps(document, sort_keys=True))
    training = tmp_path / "training.pt"
    torch.save({
        "input_ids": torch.zeros(1, 1, dtype=torch.int64),
        "activation_inputs": torch.ones(1, 1, 4096) / 4096,
        "teacher_targets": torch.linspace(-1, 1, 4096).reshape(1, 1, 4096),
        "teacher_mask": torch.ones(1, 1, dtype=torch.bool),
        "positions": torch.zeros(1, 1, dtype=torch.int64),
    }, training)
    output = tmp_path / "repair-bundle.pt"

    receipt = build_qtip_v7_repair_bundle(
        manifest=manifest, training=training, output=output,
        learning_rate=1e-6,
        members=["33:0:w1", "33:0:w2", "33:0:w3"],
    )
    bundle = torch.load(output, weights_only=False)

    assert receipt["rate"] == 2
    assert receipt["trainable_layer_luts"] == 1
    assert receipt["fixed_members"] == 3
    assert [row["projection"] for row in bundle["layers"]] == ["w1", "w2", "w3"]
    assert all(row["layer_lut"] == 33 for row in bundle["layers"])
    assert bundle["layer_luts"][33].shape == (1024,)
    assert bundle["layers"][0]["trellis"].dtype == torch.uint16

    request = {
        "schema": "banana-smasher-physical-repair-request-v1",
        "bundle": str(output),
        "bundle_sha256": _sha(output),
        "device": "cpu",
    }
    backend = PhysicalRepairBackend(request, _physical_context(tmp_path))
    worker = backend.initialize()
    assert len(worker["runtime"].layers) == 1
    expert = worker["runtime"].layers[0]
    assert len({id(expert.w1.tlut), id(expert.w2.tlut), id(expert.w3.tlut)}) == 1
    packed_before = [value.detach().clone() for value in worker["packed_indices"]]
    decoded_update0 = expert.w1._weight().detach().clone()
    activation = bundle["activation_inputs"]
    with torch.no_grad():
        gate = expert.w1(activation)
        up = expert.w3(activation)
        product = torch.nn.functional.silu(gate) * up
        expected = expert.w2(product)
        observed = expert(activation)
    assert gate.shape == up.shape == product.shape == (1, 1, 2048)
    assert observed.shape == expected.shape == (1, 1, 4096)
    assert torch.equal(observed, expected)
    result = backend.cycle(worker, request)

    assert result["gradient"]["nonzero"] is True
    assert result["gradient"]["max_abs"] > 0
    assert result["parameter"]["max_abs_diff"] > 0
    assert result["physical_repair"]["train_objective_improved"] is True
    assert result["physical_repair"]["train_objective_after"] < result["physical_repair"]["train_objective_before"]
    assert all(torch.equal(before, after) for before, after in zip(packed_before, worker["packed_indices"]))
    assert not torch.equal(decoded_update0, expert.w1._weight().detach())

    source = load_qtip_v7_artifact(manifest)
    update1 = tmp_path / "update1"
    receipt1 = export_qtip_v7_artifact(
        manifest=manifest, output=update1, update_artifact=tmp_path / "update.pt"
    )
    read1 = load_qtip_v7_artifact(update1 / "QTIP_V7_MANIFEST.json")
    assert receipt1["packed_identity"] is True
    assert receipt1["wire_size_delta"] == 0
    assert read1.member_wire_sha256 == source.member_wire_sha256
    assert read1.layer_luts[33].tobytes() != source.layer_luts[33].tobytes()
    assert read1.complete_wire_bytes == source.complete_wire_bytes