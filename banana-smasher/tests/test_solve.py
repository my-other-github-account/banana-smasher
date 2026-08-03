from __future__ import annotations

import hashlib
import builtins
import json
from pathlib import Path

import numpy as np

from banana_smasher.cli import _parser, main


def test_solve_defaults_to_exact_accelerated_backend_without_runtime_proof() -> None:
    args = _parser().parse_args(
        ["solve", "--source-root", "/tmp/source", "--output", "/tmp/output"]
    )
    assert args.backend == "exact-gemm"
    assert args.reference_search is False
    assert args.verbose_receipts is False
    assert not hasattr(args, "self_check_rows")
    assert not hasattr(args, "reference_fallback")


def test_solve_hides_developer_flags_from_user_help() -> None:
    parser = _parser()
    command_action = next(
        action for action in parser._actions if getattr(action, "choices", None)
    )
    solve = command_action.choices["solve"]
    help_text = solve.format_help()
    assert "--reference-search" not in help_text
    assert "--verbose-receipts" not in help_text
    assert "--implementation" not in help_text
    assert "--self-check" not in help_text


def _bound_file(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_solve_fixture(root: Path) -> None:
    root.mkdir()
    np.save(
        root / "vectors.npy",
        np.asarray([[0.0, 0.0, 0.0, 0.0], [0.9, 1.0, 1.1, 1.0]], dtype=np.float32),
    )
    np.save(
        root / "codebook.npy",
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0],
            ],
            dtype=np.float32,
        ),
    )
    (root / "solve.json").write_text(
        json.dumps(
            {
                "schema": "banana-smasher-solve-input-v1",
                "layer": 23,
                "cells": [
                    {
                        "cell": "L023.E000.P13",
                        "vectors": _bound_file(root / "vectors.npy"),
                        "codebook": _bound_file(root / "codebook.npy"),
                    }
                ],
            }
        )
    )


def _add_frozen_bucket(root: Path) -> None:
    arrays = {
        "weights": np.zeros((1, 32), dtype=np.float32),
        "h": np.ones((32,), dtype=np.float32),
        "codes": np.zeros((1, 1, 8), dtype=np.int64),
        "scales": np.full((1, 1, 1), 127, dtype=np.int64),
        "codebooks": np.zeros((1, 4), dtype=np.float32),
        "codebook_offsets": np.zeros((1,), dtype=np.int64),
    }
    for name, value in arrays.items():
        np.save(root / f"{name}.npy", value)
    manifest_path = root / "solve.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["frozen_bucket"] = {
        "options": ["candidate-a"],
        "vector_width": 4,
        **{
            name: _bound_file(root / f"{name}.npy")
            for name in arrays
        },
    }
    manifest_path.write_text(json.dumps(manifest))


def test_reference_search_is_explicit_and_writes_slim_receipt(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary) == {
        "artifact",
        "backend",
        "command",
        "elapsed_seconds",
        "receipt",
        "status",
    }
    assert summary["status"] == "PASS"
    assert summary["backend"] == "reference-search"
    assert np.load(output / "winners.npz", allow_pickle=False)[
        "L023.E000.P13"
    ].tolist() == [0, 1]
    receipt = json.loads((output / "SOLVE_RECEIPT.json").read_text())
    assert receipt["shape"] == {"candidates": 3, "cells": 1, "rows": 2}
    manifest_payload = (source / "solve.json").read_bytes()
    assert receipt["input_manifest"] == {
        "path": "solve.json",
        "bytes": len(manifest_payload),
        "sha256": hashlib.sha256(manifest_payload).hexdigest(),
    }
    assert receipt["inputs"] == [
        {"cell": "L023.E000.P13", "field": "vectors", **_bound_file(source / "vectors.npy")},
        {"cell": "L023.E000.P13", "field": "codebook", **_bound_file(source / "codebook.npy")},
    ]
    artifact_payload = (output / "winners.npz").read_bytes()
    assert receipt["artifact_bytes"] == len(artifact_payload)
    assert receipt["artifact_sha256"] == hashlib.sha256(artifact_payload).hexdigest()
    assert "self_check" not in json.dumps(receipt)
    assert "fallback" not in json.dumps(receipt)


def test_receipt_paths_remain_valid_after_output_directory_is_moved(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    _add_frozen_bucket(source)
    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 0
    capsys.readouterr()

    receipt = json.loads((output / "SOLVE_RECEIPT.json").read_text())
    assert receipt["artifact"] == "winners.npz"
    assert receipt["receipt"] == "SOLVE_RECEIPT.json"
    assert receipt["bucket_artifact"] == "bucket_scores.npz"

    moved_output = tmp_path / "moved-output"
    output.rename(moved_output)
    reopened_receipt_path = moved_output / receipt["receipt"]
    reopened_receipt = json.loads(reopened_receipt_path.read_text())
    artifact_payload = (moved_output / reopened_receipt["artifact"]).read_bytes()
    assert reopened_receipt["artifact_bytes"] == len(artifact_payload)
    assert reopened_receipt["artifact_sha256"] == hashlib.sha256(artifact_payload).hexdigest()
    bucket_payload = (moved_output / reopened_receipt["bucket_artifact"]).read_bytes()
    assert reopened_receipt["bucket_artifact_bytes"] == len(bucket_payload)
    assert reopened_receipt["bucket_artifact_sha256"] == hashlib.sha256(bucket_payload).hexdigest()


def test_default_solve_fails_loud_without_accelerator_and_publishes_nothing(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["status"] == "FAIL"
    assert "no reference fallback" in failure["error"]
    assert not output.exists()


def test_all_input_bindings_validate_before_compute_or_publish(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from banana_smasher import exact_codebook

    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    late_vectors = source / "late-vectors.npy"
    np.save(late_vectors, np.ones((1, 4), dtype=np.float32))
    manifest_path = source / "solve.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"].append(
        {
            "cell": "L023.E001.P13",
            "vectors": _bound_file(late_vectors),
            "codebook": _bound_file(source / "codebook.npy"),
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    late_vectors.write_bytes(late_vectors.read_bytes() + b"tamper")

    calls = 0
    original = exact_codebook.exhaustive_reference_winners

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(exact_codebook, "exhaustive_reference_winners", counted)
    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert "cells[1].vectors byte count mismatch" in failure["error"]
    assert calls == 0
    assert not output.exists()


def test_same_length_input_tamper_is_rejected_by_sha256(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    vectors = source / "vectors.npy"
    payload = bytearray(vectors.read_bytes())
    payload[-1] ^= 1
    vectors.write_bytes(payload)

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "cells[0].vectors sha256 mismatch"
    assert not output.exists()


def test_compute_failure_publishes_no_partial_output(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from banana_smasher import exact_codebook

    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)

    def fail_compute(*_args, **_kwargs):
        raise RuntimeError("injected solve failure")

    monkeypatch.setattr(exact_codebook, "exhaustive_reference_winners", fail_compute)
    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "injected solve failure"
    assert not output.exists()
    assert list(tmp_path.glob(".output.*")) == []


def test_boolean_layer_is_rejected(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    manifest_path = source / "solve.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["layer"] = True
    manifest_path.write_text(json.dumps(manifest))

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "solve-input layer must be a non-negative integer"
    assert not output.exists()


def test_boolean_frozen_vector_width_is_rejected(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    manifest_path = source / "solve.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["frozen_bucket"] = {
        "options": ["candidate-a"],
        "vector_width": True,
    }
    manifest_path.write_text(json.dumps(manifest))

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "cells[0].frozen_bucket.vector_width must be positive"
    assert not output.exists()


def test_zero_row_vectors_are_rejected(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    vectors = source / "vectors.npy"
    np.save(vectors, np.empty((0, 4), dtype=np.float32))
    manifest_path = source / "solve.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["vectors"] = _bound_file(vectors)
    manifest_path.write_text(json.dumps(manifest))

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "cells[0].vectors must contain at least one row"
    assert not output.exists()


def test_non_float32_vectors_are_rejected_instead_of_silently_narrowed(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    vectors = source / "vectors.npy"
    np.save(vectors, np.ones((1, 4), dtype=np.float64))
    manifest_path = source / "solve.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["vectors"] = _bound_file(vectors)
    manifest_path.write_text(json.dumps(manifest))

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "cells[0].vectors must have dtype float32"
    assert not output.exists()


def test_bound_input_symlink_is_rejected(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    vectors_link = source / "vectors-link.npy"
    vectors_link.symlink_to(source / "vectors.npy")
    manifest_path = source / "solve.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cells"][0]["vectors"] = _bound_file(vectors_link)
    manifest_path.write_text(json.dumps(manifest))

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "cells[0].vectors.path must not contain a symlink"
    assert not output.exists()


def test_missing_triton_error_names_supported_dependencies_not_a_dead_extra(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    import torch

    from banana_smasher import exact_codebook

    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(exact_codebook, "triton", None)

    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cuda",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert "install torch and Triton" in failure["error"]
    assert "banana-smasher[solve]" not in failure["error"]
    assert not output.exists()


def test_missing_torch_dependency_fails_as_structured_solve_error(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_solve_fixture(source)
    real_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    assert main(
        [
            "solve",
            "--source-root",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--reference-search",
        ]
    ) == 2
    failure = json.loads(capsys.readouterr().err)
    assert failure["error"] == "solve requires torch; install torch on a supported host"
    assert failure["error_type"] == "RuntimeError"
    assert not output.exists()
