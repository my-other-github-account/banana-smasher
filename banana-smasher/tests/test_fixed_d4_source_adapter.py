from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from banana_smasher.cli import main


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for name, (dtype, shape, value) in tensors.items():
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(len(raw_header).to_bytes(8, "little") + raw_header + payload)


def test_public_prepare_solve_streams_native_mxfp4_source_into_bound_config(
    tmp_path: Path, capsys
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    weight_map: dict[str, str] = {}
    # One 32-value MXFP4 block per weight: packed low/high E2M1 nibbles are
    # 1.0 and 2.0, with a raw E8M0 scale byte. The real model uses the same
    # dtypes and simply has larger matrix shapes.
    for expert in range(256):
        for weight in ("w1", "w2", "w3"):
            prefix = f"layers.0.ffn.experts.{expert}.{weight}"
            tensors[f"{prefix}.weight"] = ("I8", [1, 16], bytes([0x42]) * 16)
            tensors[f"{prefix}.scale"] = ("F8_E8M0", [1, 1], bytes([127]))
            weight_map[f"{prefix}.weight"] = shard.name
            weight_map[f"{prefix}.scale"] = shard.name
    _write_safetensors(shard, tensors)
    basis_index = model / "model.safetensors.index.json"
    basis_index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    basis = hashlib.sha256(basis_index.read_bytes()).hexdigest()

    prepared = tmp_path / "prepared"

    assert (
        main(
            [
                "fixed-d4",
                "prepare-solve",
                "--model",
                str(model),
                "--tier",
                "d4_k2048",
                "--layer",
                "0",
                "--output",
                str(prepared),
                "--basis-sha256",
                basis,
                "--chunk-vectors",
                "32",
                "--reserve-bytes",
                "0",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "PASS"
    assert receipt["basis_sha256"] == basis
    assert receipt["source_dtype"] == "packed-mxfp4-e2m1-with-e8m0-scales"
    assert receipt["codebook_source"] == "source-frequency-top-k"
    config_path = Path(receipt["config"])
    config = json.loads(config_path.read_text())
    assert config["schema"] == "banana-smasher-fixed-d4-exact-solve-v1"
    assert config["vector_domain"] == "mxfp4_e2m1"
    assert config["basis_index"] == "model.safetensors.index.json"

    down = np.load(
        prepared / config["projections"]["down"]["normalized_vectors"]["path"]
    )
    fused13 = np.load(
        prepared / config["projections"]["fused13"]["normalized_vectors"]["path"]
    )
    assert down.shape == (256, 8, 4)
    assert fused13.shape == (256, 16, 4)
    assert np.all(down == [1.0, 2.0, 1.0, 2.0])
    assert np.all(fused13 == [1.0, 2.0, 1.0, 2.0])

    solve = tmp_path / "solve"
    assert (
        main(
            [
                "fixed-d4",
                "solve",
                "--config",
                str(config_path),
                "--output",
                str(solve),
                "--basis-sha256",
                basis,
            ]
        )
        == 0
    )
    solved = json.loads(capsys.readouterr().out)
    manifest = json.loads(Path(solved["manifest"]).read_text())
    for projection in ("down", "fused13"):
        assignments = np.load(
            Path(solved["manifest"]).parent
            / manifest["projections"][projection]["assignments"]["path"]
        )
        assert np.all(assignments == 0)
