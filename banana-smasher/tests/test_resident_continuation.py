from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from banana_smasher.resident_continuation import (
    ModernGreenResidentEngine,
    _checkpoint_cursor,
    _construct_shard_student,
    _score_group_logits,
    _score_window_groups,
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


def test_construct_shard_student_uses_modern_green_legacy_roster_abi():
    student = _call(SimpleNamespace(ShardStudent=_Student))
    assert student.kwargs["parent_root"] == Path("parent")
    assert student.kwargs["l034_roster"] == Path("roster.json")
    assert "member_roster" not in student.kwargs


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
    assert "l034_roster" not in student.kwargs


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


def test_physical_score_uses_four_ordered_sixteen_window_groups():
    groups = _score_window_groups(tuple(range(64)))
    assert groups == [list(range(start, start + 16)) for start in range(0, 64, 16)]
