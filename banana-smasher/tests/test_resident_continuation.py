from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import hashlib
import os
import sys

import pytest

import banana_smasher.resident_continuation as continuation_module
from banana_smasher.resident_balanced64 import ArtifactError
from banana_smasher.resident_proven_api import ResidentRepairAPI as ProvenResidentRepairAPI

from banana_smasher.resident_continuation import (
    OFFICIAL_PHYSICAL_LAYER_SHA256,
    ModernGreenResidentEngine,
    _checkpoint_cursor,
    _checkpoint_lut_admission,
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


def test_construct_shard_student_omits_l034_for_parent_only_rank0_abi():
    class ParentOnlyStudent:
        def __init__(self, *, parent_root, **kwargs):
            self.parent_root = parent_root
            self.kwargs = kwargs

    student = _call(SimpleNamespace(ShardStudent=ParentOnlyStudent))
    assert student.parent_root == Path("parent")
    assert "l034_roster" not in student.kwargs


def test_construct_shard_student_passes_l034_for_rank1_legacy_abi():
    class L034Student:
        def __init__(self, *, parent_root, l034_roster, **kwargs):
            self.parent_root = parent_root
            self.l034_roster = l034_roster
            self.kwargs = kwargs

    student = _call(SimpleNamespace(ShardStudent=L034Student))
    assert student.parent_root == Path("parent")
    assert student.l034_roster == Path("roster.json")


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


def test_checkpoint_derived_lut_is_admitted_only_when_wire_matches_loaded_state(tmp_path):
    import numpy as np

    original = np.zeros(1024, dtype="<f2")
    checkpoint = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
    wire = tmp_path / "L021.tlut.f16"
    wire.write_bytes(checkpoint.astype("<f2").tobytes())
    admission = {
        "trainable_roster": {
            "luts": [{
                "layer": 21,
                "name": "layers.21.experts.tlut",
                "wire": {
                    "source_path": str(wire),
                    "sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                },
            }]
        }
    }

    rebound, rows = _checkpoint_lut_admission(
        admission, {"luts": {"layers.21.experts.tlut": checkpoint}}
    )

    observed = hashlib.sha256(wire.read_bytes()).hexdigest()
    assert rebound["trainable_roster"]["luts"][0]["wire"]["sha256"] == observed
    assert rows == [{
        "layer": 21,
        "name": "layers.21.experts.tlut",
        "source": "checkpoint_state_float16_wire",
        "sha256": observed,
    }]
    assert admission["trainable_roster"]["luts"][0]["wire"]["sha256"] != observed


def test_original_provider_lut_remains_admitted_when_checkpoint_lut_differs(tmp_path):
    import numpy as np

    original = np.zeros(1024, dtype="<f2")
    checkpoint = np.ones(1024, dtype=np.float32)
    wire = tmp_path / "L021.tlut.f16"
    wire.write_bytes(original.tobytes())
    digest = hashlib.sha256(wire.read_bytes()).hexdigest()
    admission = {
        "trainable_roster": {
            "luts": [{
                "layer": 21,
                "name": "layers.21.experts.tlut",
                "wire": {"source_path": str(wire), "sha256": digest},
            }]
        }
    }

    rebound, rows = _checkpoint_lut_admission(
        admission, {"luts": {"layers.21.experts.tlut": checkpoint}}
    )

    assert rebound == admission
    assert rows == []


def test_checkpoint_derived_lut_rebinding_rejects_bytes_not_in_loaded_state(tmp_path):
    import numpy as np

    wire = tmp_path / "L021.tlut.f16"
    wire.write_bytes(np.ones(1024, dtype="<f2").tobytes())
    admission = {
        "trainable_roster": {
            "luts": [{
                "layer": 21,
                "name": "layers.21.experts.tlut",
                "wire": {
                    "source_path": str(wire),
                    "sha256": hashlib.sha256(np.zeros(1024, dtype="<f2").tobytes()).hexdigest(),
                },
            }]
        }
    }

    with pytest.raises(ArtifactError, match="L021 checkpoint-derived LUT SHA mismatch"):
        _checkpoint_lut_admission(
            admission,
            {"luts": {"layers.21.experts.tlut": np.full(1024, 2.0, dtype=np.float32)}},
        )



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
        @staticmethod
        def isend(value, dst):
            return value, dst

        @staticmethod
        def P2POp(op, value, dst, *, group):
            return (op, value, dst, group)

        @staticmethod
        def batch_isend_irecv(ops):
            op, value, dst, group = ops[0]
            assert group is None
            op(value, dst)
            events.append(("isend", value, dst))
            return [Work(value)]

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


def test_score_configuration_forces_a1_eager_attention(monkeypatch):
    monkeypatch.setenv("BR_ATTN_IMPL", "sdpa")
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.config = {}
    engine.corpus_path = Path("corpus.json")
    engine.teacher_root = Path("teachers")
    engine.model_root = Path("model")
    engine.base = SimpleNamespace(T=SimpleNamespace(CKPT=None, DEV=None))
    engine.__dict__["torch"] = SimpleNamespace(
        manual_seed=lambda _seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda _seed: None),
    )

    engine._configure_base()

    assert os.environ["BR_ATTN_IMPL"] == "eager"


def test_physical_score_uses_eight_ordered_eight_window_groups():
    groups = _score_window_groups(tuple(range(64)))
    assert groups == [list(range(offset, offset + 8)) for offset in range(0, 64, 8)]


def test_rank_receive_uses_batched_p2p_and_waits():
    events = []

    class Work:
        def wait(self):
            events.append("wait")

    class Dist:
        @staticmethod
        def irecv(value, src):
            return value, src

        @staticmethod
        def P2POp(op, value, src, *, group):
            return (op, value, src, group)

        @staticmethod
        def batch_isend_irecv(ops):
            op, value, src, group = ops[0]
            assert group is None
            op(value, src)
            events.append(("irecv", value, src))
            return [Work()]

    continuation_module._recv_rank_activation(Dist(), "activation")

    assert events == [("irecv", "activation", 0), "wait"]


def test_layer_stack_uses_a1_equivalent_fresh_cache_per_layer(monkeypatch):
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

    assert len(caches) == 3
    assert seen == [caches[1], caches[2]]
    assert seen[0] is not seen[1]


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
