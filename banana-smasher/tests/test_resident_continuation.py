from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from banana_smasher.resident_continuation import _construct_shard_student


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
