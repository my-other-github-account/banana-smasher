from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from banana_smasher.anchor import build_bank_manifest, materialize_bank, register_bank
from banana_smasher.cli import main
from banana_smasher.contract import export_pack


def _bound_array(root: Path, name: str, value: np.ndarray) -> dict[str, object]:
    path = root / name
    np.save(path, value, allow_pickle=False)
    payload = path.read_bytes()
    return {
        "path": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_public_exact_solver_materializes_real_vllm_bank_logits(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    basis_index = tmp_path / "model.safetensors.index.json"
    basis_index.write_text('{"weight_map": {}}')
    basis_sha256 = hashlib.sha256(basis_index.read_bytes()).hexdigest()

    inputs = tmp_path / "solver-inputs"
    inputs.mkdir()
    codebook = np.zeros((2048, 4), dtype=np.float32)
    codebook[7] = [1.0, 2.0, 3.0, 4.0]
    vectors = np.broadcast_to(codebook[7], (256, 2, 4)).copy()
    projections = {}
    for projection in ("down", "fused13"):
        projections[projection] = {
            "normalized_vectors": _bound_array(
                inputs, f"{projection}.vectors.npy", vectors
            ),
            "scales": _bound_array(
                inputs,
                f"{projection}.scales.npy",
                np.full((256, 1), 127, dtype=np.uint8),
            ),
            "codebook": _bound_array(
                inputs, f"{projection}.codebook.npy", codebook
            ),
        }
    solver_config = inputs / "solve.json"
    solver_config.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-fixed-d4-exact-solve-v1",
                "tier": "d4_k2048",
                "layer": 0,
                "basis_index": str(basis_index),
                "basis_sha256": basis_sha256,
                "chunk_vectors": 32,
                "projections": projections,
            }
        )
    )
    solve_root = tmp_path / "solve"
    assert main(
        [
            "fixed-d4",
            "solve",
            "--config",
            str(solver_config),
            "--output",
            str(solve_root),
            "--basis-sha256",
            basis_sha256,
        ]
    ) == 0
    solve_receipt = json.loads(capsys.readouterr().out)
    manifest = Path(solve_receipt["manifest"])
    for projection in ("down", "fused13"):
        binding = json.loads(manifest.read_text())["projections"][projection][
            "assignments"
        ]
        assignments = np.load(manifest.parent / binding["path"], allow_pickle=False)
        assert assignments.shape == (256, 2)
        assert np.all(assignments == 7)

    wire = tmp_path / "wire"
    assert main(
        [
            "fixed-d4",
            "materialize",
            "--manifest",
            str(manifest),
            "--output",
            str(wire),
            "--basis-sha256",
            basis_sha256,
        ]
    ) == 0
    capsys.readouterr()
    model = tmp_path / "model"
    export_pack(
        source_root=wire,
        output=model,
        model_id="real-fixed-d4",
        instance_id="real-fixed-d4-1",
        runtime_floor_bytes=0,
        link_mode="copy",
    )

    parent_rows = [
        {"window_id": index, "class": "fixture", "tokens": [index + 1, index + 2]}
        for index in range(64)
    ]
    parent = tmp_path / "balanced64.jsonl"
    parent.write_text("".join(json.dumps(row) + "\n" for row in parent_rows))
    parent_identity = {
        "status": "resolved",
        "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "uri": "fixture://balanced64",
    }
    bank = build_bank_manifest(
        bank_id="balanced64",
        role="train_balanced64",
        windows=[
            {"id": row["window_id"], "class": row["class"]}
            for row in parent_rows
        ],
        parent_corpus=parent_identity,
        identities={
            "corpus": parent_identity,
            "tokenizer": {
                "status": "resolved",
                "sha256": "1" * 64,
                "uri": "fixture://tokenizer",
            },
            "teacher": {"status": "unresolved", "reason": "not needed"},
            "scorer": {
                "status": "resolved",
                "sha256": "2" * 64,
                "uri": "fixture://scorer",
            },
        },
        split_lineage={"split": "train", "parent_bank_id": None},
        creation={"method": "fixture", "config": {}},
        relationships=[],
    )
    run_root = tmp_path / "anchor-run"
    register_bank(run_root, bank)
    materialize_bank(bank, parent, run_root / "banks" / "balanced64.jsonl")

    expected_model = model
    vllm = ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FlatLogprobs(Sequence):
        """Match vLLM's non-list SampleLogprobs sequence shape."""

        def __init__(self, position):
            self.position = position

        def __len__(self):
            return 1

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.position

    class LLM:
        def __init__(self, *, model: str, **kwargs):
            assert Path(model) == expected_model
            assert kwargs == {
                "enforce_eager": True,
                "max_logprobs": -1,
                "tensor_parallel_size": 1,
            }

        def generate(self, prompts, sampling_params, *, use_tqdm=False):
            assert sampling_params.kwargs == {
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": -1,
            }
            assert use_tqdm is False
            results = []
            for prompt in prompts:
                token_ids = prompt["prompt_token_ids"]
                final_token = token_ids[-1]
                generated = {
                    0: SimpleNamespace(logprob=-float(final_token)),
                    1: SimpleNamespace(logprob=-float(final_token + 1)),
                    2: SimpleNamespace(logprob=-float(final_token + 2)),
                }
                results.append(
                    SimpleNamespace(
                        outputs=[SimpleNamespace(logprobs=FlatLogprobs(generated))],
                    )
                )
            return results

    setattr(vllm, "LLM", LLM)
    setattr(vllm, "SamplingParams", SamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    producer_config = (
        Path(__file__).parents[1] / "producer_configs" / "fixed_d4_vllm.json"
    )
    public_config = json.loads(producer_config.read_text())
    assert public_config["parameters"]["engine"]["max_logprobs"] == -1
    assert main(
        [
            "anchor",
            "materialize-candidate",
            "--run-root",
            str(run_root),
            "--bank",
            "balanced64",
            "--candidate-id",
            "d4-k2048-real",
            "--model",
            str(model),
            "--config",
            str(producer_config),
            "--basis-sha256",
            basis_sha256,
        ]
    ) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["coverage"] == "64/64"
    rows = [
        json.loads(line)
        for line in (run_root / receipt["relative_path"]).read_text().splitlines()
    ]
    assert len(rows) == 64
    assert rows[0]["logits"] == [-2.0, -3.0, -4.0]
    assert rows[1]["logits"] == [-3.0, -4.0, -5.0]
    assert len({tuple(row["logits"]) for row in rows}) == 64
