from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from banana_smasher.hf_sharded_balanced64_runtime import ShardedHFBalanced64Runtime


def _source(tmp_path: Path) -> dict:
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["FixtureForConditionalGeneration"],
                "num_hidden_layers": 2,
                "n_routed_experts": 8,
                "hc_mult": 2,
                "layer_types": ["linear_attention", "deepseek_sparse_attention"],
                "mlp_layer_types": ["dense", "sparse"],
                "vocab_size": 9000,
            }
        )
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-1.safetensors",
                    "model.layers.0.self_attn.q_proj.weight": "model-1.safetensors",
                    "model.layers.1.linear_attn.in_proj.weight": "model-2.safetensors",
                    "model.layers.1.mlp.experts.0.gate_proj.weight": "model-2.safetensors",
                    "lm_head.weight": "model-2.safetensors",
                    "vision_tower.layers.0.weight": "vision.safetensors",
                }
            }
        )
    )
    for name in ("model-1.safetensors", "model-2.safetensors", "vision.safetensors"):
        (root / name).write_bytes(name.encode())
    return {
        "schema": "banana-smasher-hf-source-admission-v1",
        "status": "PASS",
        "model_root": str(root),
        "model_index_sha256": hashlib.sha256(
            (root / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "shards": ["model-1.safetensors", "model-2.safetensors", "vision.safetensors"],
    }


def test_capability_selection_uses_config_index_and_role_not_model_name(tmp_path: Path) -> None:
    runtime = ShardedHFBalanced64Runtime(executor_factory=lambda **_: None)
    source = _source(tmp_path)

    assert runtime.supports(subject=source, role="teacher") is True

    artifact = _artifact(tmp_path, source)
    assert runtime.supports(subject=artifact, role="candidate_pre") is True

    config_path = Path(source["model_root"]) / "config.json"
    config = json.loads(config_path.read_text())
    config["num_nextn_predict_layers"] = 1
    config_path.write_text(json.dumps(config))
    source["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    index_path = Path(source["model_root"]) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["model.language_model.layers.2.mtp.weight"] = "model-2.safetensors"
    index_path.write_text(json.dumps(index))
    source["model_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    assert runtime.supports(subject=source, role="teacher") is True

    malformed = dict(artifact, artifact_root=None)
    assert runtime.supports(subject=malformed, role="candidate_pre") is False
    assert runtime.supports(subject=source, role="candidate_pre") is False
    assert runtime.supports(subject=artifact, role="teacher") is False


def test_project_registers_balanced64_runtime_entry_point() -> None:
    project = Path(__file__).parents[1] / "pyproject.toml"
    text = project.read_text()
    assert '[project.entry-points."banana_smasher.balanced64_runtimes"]' in text
    assert "hf-sharded" in text


def test_runtime_capability_matches_package_executor_hybrid_contract(tmp_path: Path) -> None:
    runtime = ShardedHFBalanced64Runtime(executor_factory=lambda **_: None)
    source = _source(tmp_path)
    config_path = Path(source["model_root"]) / "config.json"
    config = json.loads(config_path.read_text())
    config["hc_mult"] = 0
    config_path.write_text(json.dumps(config))
    source["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()

    assert runtime.supports(subject=source, role="teacher") is False


def test_runtime_capability_ignores_vision_layer_ordinals(tmp_path: Path) -> None:
    runtime = ShardedHFBalanced64Runtime(executor_factory=lambda **_: None)
    source = _source(tmp_path)
    index_path = Path(source["model_root"]) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"]["vision_tower.layers.99.weight"] = "vision.safetensors"
    index_path.write_text(json.dumps(index))
    source["model_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()

    assert runtime.supports(subject=source, role="teacher") is True


class _Session:
    def __init__(self, role: str, events: list[tuple]) -> None:
        self.role = role
        self.events = events
        self.ready = False

    def teacher_window(self, *, window, token_ids):
        self.events.append(("teacher", window["ordinal"], len(token_ids)))
        support = np.arange(8192, dtype=np.int32)[None, :]
        logits = np.linspace(-4.0, 4.0, 8192, dtype=np.float32)[None, :]
        return {
            "support_token_ids": support,
            "support_logits": logits,
            "position_map": np.zeros(1024, dtype=np.uint16),
            "top1_token_ids": np.full(1024, 8191, dtype=np.int32),
        }

    def candidate_window(self, *, window, token_ids, support_token_ids, position_map):
        self.events.append(("candidate", window["ordinal"], len(token_ids)))
        assert support_token_ids.shape == (1, 8192)
        assert position_map.shape == (1024,)
        return {
            "support_logits": np.linspace(-4.0, 4.0, 8192, dtype=np.float32)[None, :],
            "position_map": np.zeros(1024, dtype=np.uint16),
            "top1_token_ids": np.full(1024, 8191, dtype=np.int32),
        }

    def finish_setup(self):
        self.events.append(("resident-ready", self.role))
        self.ready = True

    def resident_ready(self):
        return self.ready

    def counters(self):
        counters = {
            "setup_model_reads": 3,
            "setup_payload_reads": 64 if self.role == "candidate_pre" else 0,
            "working_set_loads": 2,
            "fallback": 0,
            "relay": 0,
            "reconstruction": 0,
            "streaming": 0,
        }
        if self.role == "candidate_pre":
            counters.update(timed_payload_reads=0, timed_model_reads=0)
        return counters


def _suite_and_corpus(tmp_path: Path, *, model_index_sha256: str = "a" * 64) -> tuple[dict, Path]:
    windows = [
        {"ordinal": ordinal, "window_id": 1000 + ordinal, "source_class": "fixture"}
        for ordinal in range(64)
    ]
    corpus = tmp_path / "model-token-ledger.json"
    ledger = {
        "schema": "banana-smasher.balanced64-token-ledger.v1",
        "window_population_sha256": "d" * 64,
        "source_provenance_sha256": "e" * 64,
        "item_roster_sha256": "f" * 64,
        "model_index_sha256": model_index_sha256,
        "revision": "1" * 40,
        "tokenizer": {"id": "fixture-model-tokenizer-v1"},
        "positions_per_window": 1024,
        "rows": [
            {
                **row,
                "item_id": f"item-{row['ordinal']:02d}",
                "text_sha256": f"{row['ordinal']:064x}",
                "token_count": 1025,
                "token_ids": list(range(1025)),
            }
            for row in windows
        ],
    }
    corpus.write_text(json.dumps(ledger, sort_keys=True) + "\n")
    ledger_sha256 = hashlib.sha256(corpus.read_bytes()).hexdigest()
    suite = {
        "windows": windows,
        "suite_lock_sha256": "c" * 64,
        "window_population_sha256": ledger["window_population_sha256"],
        "source_provenance_sha256": ledger["source_provenance_sha256"],
        "source_windows_sha256": ledger_sha256,
        "token_ledger": {
            "schema": ledger["schema"],
            "sha256": ledger_sha256,
            "model_index_sha256": model_index_sha256,
            "tokenizer_id": ledger["tokenizer"]["id"],
            "row_count": 64,
        },
    }
    return suite, corpus


def test_teacher_capture_preserves_order_geometry_and_durable_compact_rows(tmp_path: Path) -> None:
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(tmp_path, model_index_sha256=source["model_index_sha256"])
    events: list[tuple] = []
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kw: _Session(kw["role"], events)
    )
    output = tmp_path / "teacher"

    result = runtime.capture_teacher(
        source=source, suite_lock=suite, corpus=corpus, output=output
    )

    assert events == [("teacher", ordinal, 1025) for ordinal in range(64)]
    assert [row["ordinal"] for row in result["rows"]] == list(range(64))
    assert all(row["positions"] == 1024 and row["support"] == 8192 for row in result["rows"])
    assert result["runtime_counters"] == {
        "setup_model_reads": 3,
        "setup_payload_reads": 0,
        "working_set_loads": 2,
        "fallback": 0,
        "relay": 0,
        "reconstruction": 0,
        "streaming": 0,
    }
    first = result["rows"][0]
    ledger_sha256 = hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert first["input_policy"] == (
        "model-specific-token-ledger-v1:no-retokenization:"
        f"ledger-sha256={ledger_sha256}:tokenizer-id=fixture-model-tokenizer-v1"
    )
    row_path = Path(first["output_root"]) / first["path"]
    assert row_path.is_file()
    with np.load(row_path, allow_pickle=False) as payload:
        assert payload["support_token_ids"].shape == (1, 8192)
        assert payload["support_logits"].shape == (1, 8192)
        assert payload["position_map"].shape == (1024,)
        assert payload["top1_token_ids"].shape == (1024,)


def test_teacher_capture_supports_ordered_one_window_diagnostic(tmp_path: Path) -> None:
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(tmp_path, model_index_sha256=source["model_index_sha256"])
    events: list[tuple] = []
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kw: _Session(kw["role"], events)
    )

    result = runtime.capture_teacher(
        source=source,
        suite_lock=suite,
        corpus=corpus,
        output=tmp_path / "teacher-canary",
        windows=[suite["windows"][28]],
    )

    assert events == [("teacher", 28, 1025)]
    assert [row["ordinal"] for row in result["rows"]] == [28]


@pytest.mark.parametrize("mutation", ["missing", "bad-json", "duplicate-id", "short-tokens"])
def test_teacher_capture_rejects_ordinary_bad_corpus(tmp_path: Path, mutation: str) -> None:
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(tmp_path, model_index_sha256=source["model_index_sha256"])
    if mutation == "missing":
        corpus.unlink()
    elif mutation == "bad-json":
        corpus.write_text("not-json\n")
    else:
        ledger = json.loads(corpus.read_text())
        if mutation == "duplicate-id":
            ledger["rows"][1]["window_id"] = ledger["rows"][0]["window_id"]
        else:
            ledger["rows"][0]["token_ids"] = [1, 2]
            ledger["rows"][0]["token_count"] = 2
        corpus.write_text(json.dumps(ledger, sort_keys=True) + "\n")
    runtime = ShardedHFBalanced64Runtime(executor_factory=lambda **kw: _Session(kw["role"], []))

    with pytest.raises(ValueError, match="corpus"):
        runtime.capture_teacher(source=source, suite_lock=suite, corpus=corpus, output=tmp_path / "out")


def _artifact(tmp_path: Path, source: dict) -> dict:
    root = tmp_path / "artifact"
    root.mkdir(exist_ok=True)
    routed_dir = root / "routed"
    native_dir = root / "native"
    routed_dir.mkdir(exist_ok=True)
    native_dir.mkdir(exist_ok=True)
    trellis = routed_dir / "fixture.trellis.npy"
    scales = routed_dir / "fixture.scales.npy"
    np.save(trellis, np.zeros((1, 1), dtype=np.uint16), allow_pickle=False)
    np.save(scales, np.ones((1,), dtype=np.float32), allow_pickle=False)

    def member(path: Path) -> dict:
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    routed_name = "model.layers.1.mlp.experts.0.gate_proj.weight"
    index = json.loads(
        (Path(source["model_root"]) / "model.safetensors.index.json").read_text()
    )
    native_rows = []
    for slot, name in enumerate(sorted(set(index["weight_map"]) - {routed_name})):
        native = native_dir / f"fixture-{slot:02d}.native.bin"
        native.write_bytes(f"native:{name}".encode())
        native_sha = hashlib.sha256(native.read_bytes()).hexdigest()
        native_rows.append(
            {
                "name": name,
                "representation": "exact-source-data-bytes",
                "path": native.relative_to(root).as_posix(),
                "source_sha256": native_sha,
                "artifact_sha256": native_sha,
            }
        )
    return {
        "schema": "banana-smasher-hf-moe-uniform-artifact-v1",
        "status": "PASS",
        "reload_verified": True,
        "artifact_root": str(root),
        "source": source,
        "intent": {"tier": "q2", "scope": "routed_only", "native_rest": True},
        "routed_tensors": [
            {
                "name": routed_name,
                "wire": {
                    "geometry": {
                        "L": 16,
                        "K": 2,
                        "V": 2,
                        "tlut_bits": 9,
                        "decode_mode": "quantlut_sym",
                    },
                    "trellis": member(trellis),
                    "scales": member(scales),
                },
            }
        ],
        "native_tensors": native_rows,
        "coverage": {"duplicates": [], "gaps": []},
        "accounting": {
            "routed_tensor_count": 1,
            "planned_routed_tensor_count": 1,
            "native_tensor_count": len(native_rows),
            "planned_native_tensor_count": len(native_rows),
        },
    }


def test_score_pre_finishes_setup_before_timing_and_uses_identical_support(tmp_path: Path) -> None:
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(tmp_path, model_index_sha256=source["model_index_sha256"])
    events: list[tuple] = []
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kw: _Session(kw["role"], events)
    )
    captured = runtime.capture_teacher(
        source=source, suite_lock=suite, corpus=corpus, output=tmp_path / "teacher"
    )
    events.clear()

    result = runtime.score_pre(
        artifact=_artifact(tmp_path, source),
        teacher_capture={"rows": captured["rows"]},
        suite_lock=suite,
        corpus=corpus,
    )

    assert events[:-1] == [("candidate", ordinal, 1025) for ordinal in range(64)]
    assert events[-1] == ("resident-ready", "candidate_pre")
    assert result["resident_ready"] is True
    assert result["timed_wall_seconds"] >= 0
    assert result["runtime_counters"] == {
        "fallback": 0,
        "reconstruction": 0,
        "relay": 0,
        "setup_model_reads": 3,
        "setup_payload_reads": 64,
        "streaming": 0,
        "timed_model_reads": 0,
        "timed_payload_reads": 0,
        "working_set_loads": 2,
    }
    assert [row["ordinal"] for row in result["rows"]] == list(range(64))
    assert all(row["kld_values"] == ["0.0"] * 1024 for row in result["rows"])
    assert all(row["top1_matches"] == 1024 for row in result["rows"])


def test_score_pre_rejects_corrupt_teacher_row_before_executor_setup(tmp_path: Path) -> None:
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(tmp_path, model_index_sha256=source["model_index_sha256"])
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kw: _Session(kw["role"], [])
    )
    captured = runtime.capture_teacher(
        source=source, suite_lock=suite, corpus=corpus, output=tmp_path / "teacher"
    )
    path = Path(captured["rows"][0]["output_root"]) / captured["rows"][0]["path"]
    path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="hash"):
        runtime.score_pre(
            artifact=_artifact(tmp_path, source),
            teacher_capture={"rows": captured["rows"]},
            suite_lock=suite,
            corpus=corpus,
        )


def test_score_pre_rejects_corrupt_candidate_member_before_executor_setup(tmp_path: Path) -> None:
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(tmp_path, model_index_sha256=source["model_index_sha256"])
    events: list[tuple] = []
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kw: _Session(kw["role"], events)
    )
    captured = runtime.capture_teacher(
        source=source, suite_lock=suite, corpus=corpus, output=tmp_path / "teacher"
    )
    events.clear()
    artifact = _artifact(tmp_path, source)
    member = Path(artifact["artifact_root"]) / artifact["routed_tensors"][0]["wire"]["trellis"]["path"]
    member.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="artifact member"):
        runtime.score_pre(
            artifact=artifact,
            teacher_capture={"rows": captured["rows"]},
            suite_lock=suite,
            corpus=corpus,
        )
    assert events == []


def test_synthetic_end_to_end_through_auto_discovered_public_apis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from banana_smasher import capture_balanced64_teacher, score_balanced64_pre
    from banana_smasher import hf_balanced64

    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(tmp_path, model_index_sha256=source["model_index_sha256"])
    index = Path(source["model_root"]) / "model.safetensors.index.json"
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    suite.update(
        {
            "schema": "banana-smasher.balanced64-suite-lock.v1",
            "positions": 65536,
            "positions_per_window": 1024,
            "support": 8192,
            "window_count": 64,
            "window_population_sha256": "d" * 64,
            "source_windows_sha256": corpus_sha,
            "teacher_bank": "FIXTURE_OWN_BASE_FP8",
            "teacher_source_model_index_sha256": index_sha,
        }
    )
    suite.pop("suite_lock_sha256")
    canonical = json.dumps(
        suite, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    suite["suite_lock_sha256"] = hashlib.sha256(canonical).hexdigest()
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kw: _Session(kw["role"], [])
    )

    class _Point:
        @staticmethod
        def load():
            return lambda: runtime

    class _Points:
        @staticmethod
        def select(*, group):
            assert group == "banana_smasher.balanced64_runtimes"
            return [_Point()]

    monkeypatch.setattr(hf_balanced64.importlib_metadata, "entry_points", _Points)
    teacher = capture_balanced64_teacher(
        source["model_root"],
        revision="1" * 40,
        suite_lock=suite,
        corpus=corpus,
        output=tmp_path / "teacher-public",
        receipt_path=tmp_path / "TEACHER.json",
    )
    admitted_source = teacher["source"]
    artifact = _artifact(tmp_path, admitted_source)
    pre = score_balanced64_pre(
        artifact,
        teacher_capture=teacher,
        suite_lock=suite,
        corpus=corpus,
        receipt_path=tmp_path / "PRE.json",
    )

    assert teacher["runtime"]["id"] == "hf-sharded-balanced64-v1"
    assert teacher["row_count"] == 64
    assert pre["runtime"]["id"] == "hf-sharded-balanced64-v1"
    assert pre["rows_sealed"] == 64
    assert pre["positions"] == 65536
    assert pre["mean_kld"] == 0.0
    assert pre["top1_matches"] == 65536
    assert pre["runtime_counters"] == {
        "timed_payload_reads": 0,
        "timed_model_reads": 0,
        "fallback": 0,
        "relay": 0,
        "reconstruction": 0,
        "streaming": 0,
    }
