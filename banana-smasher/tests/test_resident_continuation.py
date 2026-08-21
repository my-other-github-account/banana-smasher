from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import hashlib
import sys

import banana_smasher.resident_continuation as continuation_module
from banana_smasher.resident_proven_api import ResidentRepairAPI as ProvenResidentRepairAPI

from banana_smasher.resident_continuation import (
    OFFICIAL_PHYSICAL_LAYER_SHA256,
    ModernGreenResidentEngine,
    _checkpoint_cursor,
    _construct_shard_student,
    _enqueue_rank_send,
    _flush_rank_sends,
    _official_expert_source_path,
    _score_group_logits,
    _score_window_groups,
    _select_trainer_fwht,
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


def test_rank_send_pipeline_preserves_tensor_lifetime_and_waits_fifo():
    events = []

    class Work:
        def __init__(self, value):
            self.value = value

        def wait(self):
            events.append(("wait", self.value))

    class Dist:
        def isend(self, value, *, dst):
            events.append(("isend", value, dst))
            return Work(value)

    pending = []
    _enqueue_rank_send(Dist(), pending, "group0")
    assert pending[0][1] == "group0"
    _enqueue_rank_send(Dist(), pending, "group1")
    assert events == [
        ("isend", "group0", 1),
        ("isend", "group1", 1),
        ("wait", "group0"),
    ]
    _flush_rank_sends(pending)
    assert events[-1] == ("wait", "group1")
    assert pending == []


def test_resident_score_selects_authenticated_trainer_quack_backend():
    calls = []
    trainer = SimpleNamespace(set_fwht_backend=calls.append)

    _select_trainer_fwht(trainer)

    assert calls == ["quack"]


def test_resident_binds_the_sealed_parity_expert_implementation():
    source = _official_expert_source_path()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == OFFICIAL_PHYSICAL_LAYER_SHA256
    text = source.read_text()
    assert ".clamp(" not in text
    assert "torch.argsort(top_k_index, dim=1, stable=True)" in text


def test_resident_import_paths_do_not_shadow_trainer_fwht_selector(tmp_path):
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.trainer_path = tmp_path / "trainer.py"
    engine.asset_root = tmp_path / "assets"
    original = list(sys.path)
    try:
        engine._prepare_import_paths()
        repository = Path(continuation_module.__file__).resolve().parents[3]
        assert str(repository / "runtime" / "v7" / "runner") not in sys.path
    finally:
        sys.path[:] = original


def test_physical_score_uses_ordered_four_window_groups():
    groups = _score_window_groups(tuple(range(64)))
    assert groups == [list(range(start, start + 4)) for start in range(0, 64, 4)]


def test_layer_stack_reuses_one_dynamic_cache(monkeypatch):
    caches = []

    class Cache:
        def __init__(self, *, config):
            self.config = config
            caches.append(self)

    transformers = ModuleType("transformers")
    cache_utils = ModuleType("transformers.cache_utils")
    cache_utils.DynamicCache = Cache
    transformers.cache_utils = cache_utils
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", cache_utils)
    seen = []

    class Layer:
        def __call__(self, hidden, **kwargs):
            seen.append(kwargs["past_key_values"])
            return hidden

    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.student = SimpleNamespace(
        config="config",
        model=SimpleNamespace(model=SimpleNamespace(layers=[Layer(), Layer()])),
    )
    engine.first = 0
    engine.last = 1
    engine._positional = lambda ids, template, cache: ("pos", "pe", "mask")

    engine._run_layers(SimpleNamespace(ndim=3), "ids", False)

    assert len(caches) == 1
    assert seen == [caches[0], caches[0]]


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
