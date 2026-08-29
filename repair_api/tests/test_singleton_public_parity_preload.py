import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import ModernGreenResidentEngine


def test_public_parity_tap_w28_uses_explicit_singleton_fixture(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "repair_api.modern_green_resident._require_static_w28_teacher",
        lambda _root, expected: expected,
    )
    monkeypatch.setattr(
        "repair_api.modern_green_resident._sha256_file",
        lambda _path: "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd",
    )
    corpus = [{"token_ids": [1] * 1024, "real_len": 1024} for _ in range(57)]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus))
    teacher_root = tmp_path / "teacher"
    teacher_root.mkdir()

    class FakeTensor:
        shape = (1024, 8192)

        def __getitem__(self, _key):
            return self

        def to(self, **_kwargs):
            return self

        def contiguous(self):
            return self

    monkeypatch.setattr(
        "repair_api.balanced64._load_torch",
        lambda _path: {"idx": FakeTensor(), "logprob": FakeTensor()},
    )

    class FakeIds:
        def __setitem__(self, _key, _value):
            pass

    torch = SimpleNamespace(
        long="long", int64="int64", float16="float16",
        full=lambda *_args, **_kwargs: FakeIds(),
        tensor=lambda value, **_kwargs: tuple(value),
    )
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.published_pre_recipe = True
    engine.rank = 1
    engine.torch = torch
    engine.student = SimpleNamespace(device="cuda")
    engine.corpus_path = corpus_path
    engine.config = {
        "resident_validation_proof": True,
        "validation_corpus": str(corpus_path),
        "sealed_builder_window_microbatch": 1,
        "singleton_public_parity_tap_only": True,
    }

    prepared = engine.preload_validation((28,), teacher_root)

    assert prepared["windows"] == (28,)
    assert prepared["physical_windows"] == (28,)
    assert prepared["physical_batch_size"] == 1
    assert set(prepared["ids"]) == {28}
    assert set(prepared["teachers"]) == {28}

    engine.config.pop("singleton_public_parity_tap_only")
    with pytest.raises(ArtifactError, match="requires sealed mb=2 microbatch"):
        engine.preload_validation((28,), teacher_root)
