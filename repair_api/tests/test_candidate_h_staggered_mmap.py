import hashlib
import inspect
from pathlib import Path
import tempfile

import pytest
import torch

from repair_api.balanced64 import ArtifactError
from repair_api.official_k2_resident_score import (
    OfficialK2ResidentScorer,
    _load_hash_bound_torch_mmap,
    _load_score_checkpoint,
)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().numpy().tobytes()


def test_hash_bound_read_only_mmap_preserves_checkpoint_tensor_bytes(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        expected = torch.arange(4096, dtype=torch.int64).reshape(64, 64)
        torch.save({"state": {"weight": expected}}, checkpoint)
        checkpoint.chmod(0o444)
        expected_file_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        calls = []
        real_load = torch.load

        def recording_load(path, **kwargs):
            calls.append((Path(path), dict(kwargs)))
            return real_load(path, **kwargs)

        monkeypatch.setattr(torch, "load", recording_load)
        payload = _load_hash_bound_torch_mmap(checkpoint, expected_file_sha)

        assert calls == [
            (checkpoint.resolve(), {"map_location": "cpu", "mmap": True, "weights_only": True}),
        ]
        observed = payload["state"]["weight"]
        assert hashlib.sha256(_tensor_bytes(observed)).hexdigest() == hashlib.sha256(
            _tensor_bytes(expected)
        ).hexdigest()

        with pytest.raises(ArtifactError, match="checkpoint mmap source SHA mismatch"):
            _load_hash_bound_torch_mmap(checkpoint, "0" * 64)


def test_public_score_serializes_rank_construction_before_hash_bound_mmap() -> None:
    source = inspect.getsource(OfficialK2ResidentScorer.score)

    gate = source.index("_wait_for_cold_load_gate")
    mmap_load = source.index("_load_score_checkpoint")
    engine = source.index("OfficialK2ResidentRankEngine(")
    assert gate < mmap_load < engine
    assert "_load_score_checkpoint(checkpoint_path, checkpoint_sha, self.config)" in source


def test_runtime_can_explicitly_isolate_candidate_from_mmap_loader(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        torch.save({"state": {"weight": torch.arange(8)}}, checkpoint)
        expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        calls = []

        def ordinary_load(path):
            calls.append(Path(path))
            return {"state": {"weight": torch.arange(8)}}

        monkeypatch.setattr(
            "repair_api.official_k2_resident_score._load_torch", ordinary_load
        )
        payload = _load_score_checkpoint(
            checkpoint, expected_sha, {"checkpoint_mmap": False}
        )

        assert calls == [checkpoint]
        assert torch.equal(payload["state"]["weight"], torch.arange(8))
