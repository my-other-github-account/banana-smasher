from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib

import banana_smasher.resident_continuation as continuation_module
from banana_smasher.resident_proven_api import ResidentRepairAPI as ProvenResidentRepairAPI

from banana_smasher.resident_continuation import (
    ModernGreenResidentEngine,
    _checkpoint_cursor,
    _construct_shard_student,
    _score_group_logits,
)


class _Student:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _call(trainer):
    return _construct_shard_student(
        trainer,
        torch="torch",
        np="numpy",
        base="base",
        official_k2="k2",
        model_root=Path("model"),
        admission={"framework": "banana-smasher"},
        parent_root=Path("parent"),
        member_roster_path=Path("roster.json"),
        member_roster_sha256="a" * 64,
        payload={"state": {}},
        rank=0,
        first=0,
        last=20,
        status_cb=lambda **_: None,
    )


def test_construct_shard_student_uses_authenticated_legacy_parent_abi():
    student = _call(SimpleNamespace(ShardStudent=_Student))
    assert student.kwargs["parent_root"] == Path("parent")
    assert student.kwargs["l034_roster"] == Path("roster.json")


def test_construct_shard_student_uses_all_layer_roster_abi_when_available():
    trainer = SimpleNamespace(
        ShardStudent=_Student,
        load_member_roster=lambda path, digest: {"path": path, "sha": digest},
    )
    student = _call(trainer)
    assert student.kwargs["member_roster"] == {
        "path": Path("roster.json"),
        "sha": "a" * 64,
    }



class _FakeTensor:
    def __init__(self, source):
        self.source = source

    def unsqueeze(self, _dim):
        return self

    def to(self, _device):
        return self


def test_training_and_balanced64_score_inputs_remain_separate_when_windows_overlap():
    class T:
        CORPUS = "original-corpus"
        TEACH = "original-teacher"

        @classmethod
        def load_corpus(cls):
            return cls.CORPUS

        @staticmethod
        def window_ids(corpus, window):
            return _FakeTensor((corpus, window)), 1024

        @classmethod
        def teacher_rows(cls, window):
            return (cls.TEACH, window)

    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.config = {
        "score_windows": [28],
        "train_corpus": "/inputs/train.json",
        "train_teacher_root": "/inputs/train-teacher",
        "score_corpus": "/inputs/score.json",
        "score_teacher_root": "/inputs/score-teacher",
    }
    engine.base = SimpleNamespace(T=T)
    engine.student = SimpleNamespace(device="cuda")
    engine.rank = 1

    engine._load_training_data()

    assert engine.ids_cache[28].source == ("/inputs/train.json", 28)
    assert engine.score_ids_cache[28].source == ("/inputs/score.json", 28)
    assert engine.teacher_cache[28] == ("/inputs/train-teacher", 28)
    assert engine.score_teacher_cache[28] == ("/inputs/score-teacher", 28)
    assert T.CORPUS == "original-corpus"
    assert T.TEACH == "original-teacher"


def test_canonical_u0_checkpoint_cursor_is_admitted():
    assert _checkpoint_cursor({"next_update": 0}) == 0


def test_score_group_projects_the_pipeline_microbatch_with_one_head_call():
    calls = []

    class Final:
        def to(self, dtype):
            calls.append(("to", dtype))
            return "batched-final"

    def lm_head(value):
        calls.append(("lm_head", value))
        return "batched-logits"

    torch = SimpleNamespace(bfloat16="bf16")
    assert _score_group_logits(lm_head, Final(), torch) == "batched-logits"
    assert calls == [("to", "bf16"), ("lm_head", "batched-final")]


def test_score_only_checkpoint_does_not_require_optimizer_or_scheduler_state():
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.score_only = True
    engine.payload = {"state": {"luts": {}, "norms": {}, "outputs": {}}}

    engine._load_optimizer_scheduler_state()


def test_public_resident_score_engine_loads_exact_state_without_training_lineage(tmp_path, monkeypatch):
    checkpoint = tmp_path / "SERIALIZED_PRE.pt"
    checkpoint.write_bytes(b"exact-pre")
    payload = {"state": {"luts": {}, "norms": {}, "outputs": {}}}
    artifact = SimpleNamespace(windows=tuple(range(64)))
    api = ProvenResidentRepairAPI(artifact, loader=lambda path: payload)
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(continuation_module, "ModernGreenResidentEngine", FakeEngine)
    config = {
        "authorized_api": True, "world_size": 2, "rank": 0, "local_only": True,
        "basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
        "layer_split": {"0": [0, 20], "1": [21, 42]},
        "trainer_source": "trainer.py", "base_source_sha256": "a" * 64,
        "model_root": "model", "asset_root": "assets", "member_roster": "roster",
        "member_roster_sha256": "b" * 64, "teacher_root": "teachers",
        "corpus": "corpus", "master_addr": "127.0.0.1", "master_port": 1234,
        "manifest": "manifest", "delta_dir": "delta", "vq3b_dir": "vq",
    }

    engine = api.construct_resident_score_engine(
        checkpoint, hashlib.sha256(b"exact-pre").hexdigest(), config=config
    )

    assert isinstance(engine, FakeEngine)
    assert captured["payload"] is payload
    assert captured["config"]["score_only"] is True
    assert captured["config"]["score_windows"] == list(range(64))
    assert captured["layer_ranges"] == {0: (0, 20), 1: (21, 42)}
