from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from banana_smasher.artifact_identity import ArtifactIdentity
from banana_smasher.hf_uniform_physical_provider import (
    HFUniformPhysicalProvider,
    _load_bound_inputs,
)
from banana_smasher.hf_sharded_balanced64_runtime import ShardedHFBalanced64Runtime
from banana_smasher.production_rails import (
    PIPELINE_MICROBATCH,
    PRODUCTION_RAILS_SCHEMA,
    ProductionRails,
)
from banana_smasher.qtip1 import QTIP2_GEOMETRY, encode_qtip, gaussian_tlut
from banana_smasher.resident_repair_api import BackpackArtifact, ResidentRepairAPI
from banana_smasher.resident_training import checkpoint_info


FIXTURE_PRE_KLD = 0.10354898407586514


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixture_uniform_builder(**kwargs):
    return kwargs["output"]


def fixture_backpack_mixer(**kwargs):
    return kwargs["output"]


class _FixtureHFExecutor:
    def __init__(self, *, role: str) -> None:
        self.role = role
        self.ready = False

    @staticmethod
    def _base_logits(scale: float) -> np.ndarray:
        return (np.linspace(-1.0, 1.0, 8192, dtype=np.float32) * scale)[None, :]

    def teacher_window(self, *, window, token_ids):
        del window, token_ids
        return {
            "support_token_ids": np.arange(8192, dtype=np.int32)[None, :],
            "support_logits": self._base_logits(2.0),
            "position_map": np.zeros(1024, dtype=np.uint16),
            "top1_token_ids": np.full(1024, 8191, dtype=np.int32),
        }

    def candidate_window(self, *, window, token_ids, support_token_ids, position_map):
        del window, token_ids, support_token_ids, position_map
        return {
            "support_logits": self._base_logits(1.0),
            "position_map": np.zeros(1024, dtype=np.uint16),
            "top1_token_ids": np.full(1024, 8191, dtype=np.int32),
        }

    def finish_setup(self) -> None:
        self.ready = True

    def resident_ready(self) -> bool:
        return self.ready

    def counters(self) -> dict[str, int]:
        counters = {
            "setup_model_reads": 3,
            "setup_payload_reads": 45 if self.role == "candidate_pre" else 0,
            "working_set_loads": 45,
            "fallback": 0,
            "relay": 0,
            "reconstruction": 0,
            "streaming": 0,
        }
        if self.role == "candidate_pre":
            counters.update(timed_payload_reads=0, timed_model_reads=0)
        return counters


def open_fixture_executor(**kwargs):
    del kwargs["subject"], kwargs["suite_lock"], kwargs["corpus_rows"]
    return _FixtureHFExecutor(role=kwargs["role"])


class _FixtureHFUniformBackend:
    PARAMETER_ID = "fixture.routed_logit_weight"

    def __init__(
        self,
        *,
        basis_sha256: str,
        suite_lock_path: str | Path,
        teacher_capture_path: str | Path,
        corpus_path: str | Path,
        **_: object,
    ) -> None:
        import torch

        self.torch = torch
        self.parameter = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        (
            _lock,
            _windows,
            self.corpus_rows_by_ordinal,
            self.teacher_rows_by_ordinal,
        ) = _load_bound_inputs(
            suite_lock_path=Path(suite_lock_path).expanduser().resolve(),
            teacher_capture_path=Path(teacher_capture_path).expanduser().resolve(),
            corpus_path=Path(corpus_path).expanduser().resolve(),
            basis_sha256=basis_sha256,
        )

    @staticmethod
    def _base_logits() -> np.ndarray:
        return np.linspace(-1.0, 1.0, 8192, dtype=np.float32)

    def resident_parameters(self):
        from banana_smasher.resident_training import ParameterDescriptor

        return [
            (
                ParameterDescriptor(
                    self.PARAMETER_ID,
                    "routed_q2",
                    self.PARAMETER_ID,
                ),
                self.parameter,
            )
        ]

    def residency_metadata(self) -> dict[str, object]:
        return {
            "resident_bytes": 4,
            "payload_disk_reads": 0,
            "model_disk_reads": 0,
            "trainable_count": 1,
        }

    def _candidate_logits(self):
        return self.parameter * self.torch.as_tensor(
            self._base_logits(),
            dtype=self.torch.float32,
            device=self.parameter.device,
        )

    def candidate_rows(self, teachers, corpus_rows):
        del corpus_rows
        logits = self._candidate_logits().detach().cpu().numpy()[None, :]
        return [
            {
                "support_logits": logits,
                "position_map": np.zeros(1024, dtype=np.uint16),
                "top1_token_ids": np.full(1024, 8191, dtype=np.int32),
            }
            for _teacher in teachers
        ]

    def loss_for_windows(self, windows, *, tokens):
        del tokens
        losses = []
        candidate = self._candidate_logits()
        candidate_logprob = candidate - self.torch.logsumexp(candidate, dim=-1, keepdim=True)
        for window in windows:
            teacher = self.teacher_rows_by_ordinal[int(window)]
            teacher_row = self.torch.as_tensor(
                np.asarray(teacher["support_logits"], dtype=np.float32)[0],
                dtype=self.torch.float32,
                device=self.parameter.device,
            )
            teacher_logprob = teacher_row - self.torch.logsumexp(
                teacher_row, dim=-1, keepdim=True
            )
            losses.append(
                (teacher_logprob.exp() * (teacher_logprob - candidate_logprob)).sum()
            )
        return self.torch.stack(losses).mean()

    def post_optimizer_step(self, names):
        del names
        return None

    def trainable_state_dict(self):
        return {self.PARAMETER_ID: self.parameter.detach().cpu().clone()}

    def load_trainable_state_dict(self, state):
        with self.torch.no_grad():
            self.parameter.copy_(
                self.torch.as_tensor(
                    state[self.PARAMETER_ID],
                    dtype=self.parameter.dtype,
                    device=self.parameter.device,
                )
            )

    def parameter_digests(self) -> dict[str, str]:
        array = self.parameter.detach().cpu().numpy()
        return {self.PARAMETER_ID: _sha_bytes(np.ascontiguousarray(array).tobytes())}

    def runtime_counters(self) -> dict[str, int]:
        return {
            "setup_model_reads": 0,
            "setup_payload_reads": 0,
            "working_set_loads": 1,
            "fallback": 0,
            "relay": 0,
            "reconstruction": 0,
            "streaming": 0,
            "timed_payload_reads": 0,
            "timed_model_reads": 0,
        }


def open_fixture_hf_uniform_backend(**kwargs):
    return _FixtureHFUniformBackend(**kwargs)


def _suite_and_corpus(
    tmp_path: Path,
    *,
    model_index_sha256: str,
) -> tuple[dict[str, object], Path]:
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
    corpus.write_text(json.dumps(ledger, sort_keys=True) + "\n", encoding="utf-8")
    ledger_sha256 = _sha_bytes(corpus.read_bytes())
    suite = {
        "schema": "banana-smasher.balanced64-suite-lock.v1",
        "positions": 64 * 1024,
        "positions_per_window": 1024,
        "support": 8192,
        "window_count": 64,
        "windows": windows,
        "window_population_sha256": ledger["window_population_sha256"],
        "source_provenance_sha256": ledger["source_provenance_sha256"],
        "source_windows_sha256": ledger_sha256,
        "teacher_bank": "FIXTURE_OWN_BASE_FP8",
        "teacher_source_model_index_sha256": model_index_sha256,
        "token_ledger": {
            "schema": ledger["schema"],
            "sha256": ledger_sha256,
            "model_index_sha256": model_index_sha256,
            "tokenizer_id": ledger["tokenizer"]["id"],
            "row_count": 64,
        },
    }
    canonical = json.dumps(
        suite, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    suite["suite_lock_sha256"] = _sha_bytes(canonical)
    return suite, corpus


def _source(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "model"
    root.mkdir()
    layer_types = ["linear_attention"] * 23 + ["deepseek_sparse_attention"] * 22
    mlp_types = ["dense"] * 2 + ["sparse"] * 43
    config = {
        "architectures": ["FixtureForConditionalGeneration"],
        "model_type": "glm5_next",
        "num_hidden_layers": 45,
        "n_routed_experts": 2,
        "hc_mult": 2,
        "layer_types": layer_types,
        "mlp_layer_types": mlp_types,
        "vocab_size": 9000,
    }
    (root / "config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n", encoding="utf-8"
    )
    weight_map = {
        "model.embed_tokens.weight": "model-1.safetensors",
        "lm_head.weight": "model-2.safetensors",
    }
    for layer in range(45):
        if layer == 22:
            name = f"model.layers.{layer}.mlp.experts.0.gate_proj.weight"
            shard = "model-2.safetensors"
        else:
            name = f"model.layers.{layer}.self_attn.q_proj.weight"
            shard = "model-1.safetensors" if layer < 23 else "model-2.safetensors"
        weight_map[name] = shard
    index = {"weight_map": weight_map}
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name in ("model-1.safetensors", "model-2.safetensors"):
        (root / name).write_bytes(name.encode())
    return {
        "schema": "banana-smasher-hf-source-admission-v1",
        "status": "PASS",
        "model_root": str(root),
        "model_index_sha256": _sha_bytes((root / "model.safetensors.index.json").read_bytes()),
        "config_sha256": _sha_bytes((root / "config.json").read_bytes()),
        "shards": ["model-1.safetensors", "model-2.safetensors"],
    }


def _artifact(
    tmp_path: Path,
    *,
    source: dict[str, object],
    provider_binding_sha256: str,
    checkpoint_sha256: str,
) -> BackpackArtifact:
    root = tmp_path / "artifact"
    root.mkdir()
    routed_dir = root / "routed"
    native_dir = root / "native"
    routed_dir.mkdir()
    native_dir.mkdir()
    matrix = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    encoded = encode_qtip(
        matrix,
        geometry=QTIP2_GEOMETRY,
        tlut=gaussian_tlut(bits=QTIP2_GEOMETRY.tlut_bits, columns=QTIP2_GEOMETRY.V),
    )
    trellis = routed_dir / "fixture.trellis.npy"
    scales = routed_dir / "fixture.scales.npy"
    np.save(trellis, encoded.packed, allow_pickle=False)
    np.save(scales, encoded.scales, allow_pickle=False)

    def member(path: Path) -> dict[str, object]:
        return {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha_bytes(path.read_bytes()),
        }

    weight_map = json.loads(
        (Path(str(source["model_root"])) / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )["weight_map"]
    routed_name = "model.layers.22.mlp.experts.0.gate_proj.weight"
    native_rows = []
    for slot, name in enumerate(sorted(set(weight_map) - {routed_name})):
        native = native_dir / f"fixture-{slot:02d}.native.bin"
        native.write_bytes(f"native:{name}".encode())
        digest = _sha_bytes(native.read_bytes())
        native_rows.append(
            {
                "name": name,
                "representation": "exact-source-data-bytes",
                "path": native.relative_to(root).as_posix(),
                "source_sha256": digest,
                "artifact_sha256": digest,
            }
        )
    receipt = {
        "schema": "banana-smasher-hf-moe-uniform-artifact-v1",
        "status": "PASS",
        "reload_verified": True,
        "source": source,
        "intent": {"tier": "q2", "scope": "routed_only", "native_rest": True},
        "geometry": {
            "expected_model_layers": 45,
            "dense_prefix_layers": 0,
            "routed_experts": 2,
            "model_layer_ids": list(range(45)),
            "auxiliary_layer_ids": [],
            "routed_layer_ids": [22],
            "model_layer_gaps": [],
            "routed_auxiliary_layers": [],
        },
        "routed_tensors": [
            {
                "name": routed_name,
                "wire": {
                    "geometry": QTIP2_GEOMETRY.as_mapping(),
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
    (root / "ARTIFACT.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    identity = {
        "schema": "banana-smasher-artifact-identity-v1",
        "basis": {"model_index_sha256": source["model_index_sha256"]},
        "corpora": {
            "builder_eval_sha256": "1" * 64,
            "train_score_sha256": "2" * 64,
            "u0_lock_sha256": "3" * 64,
            "teacher_inventory_sha256": "4" * 64,
        },
        "checkpoints": {
            "UPDATE_000": {
                "sha256": checkpoint_sha256,
                "identity_sha256": "5" * 64,
            }
        },
        "composition": {
            "kind": "uniform-qtip-v7",
            "layers": [
                {"layer": layer, "tiers": {"qtip2_v7": 1, "native": 1}}
                for layer in range(45)
            ],
        },
        "canary": {
            "reference": {"kld": FIXTURE_PRE_KLD, "top1": 64 * 1024},
            "tolerance": {"kld_abs": 0.0, "top1_abs": 0},
        },
        "runtime": {
            "production_rails": {
                "provider_binding_sha256": provider_binding_sha256
            }
        },
    }
    (root / "identity.json").write_text(
        json.dumps(identity, sort_keys=True) + "\n", encoding="utf-8"
    )
    return BackpackArtifact(
        root=root,
        identity=ArtifactIdentity.load(root),
        checkpoint_sha256=checkpoint_sha256,
    )


def _provider_source_sha() -> str:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "banana_smasher"
        / "hf_uniform_physical_provider.py"
    )
    return _sha_bytes(source.read_bytes()) if source.is_file() else "0" * 64


def _config(
    tmp_path: Path,
    *,
    teacher_capture: Path,
    suite_lock: Path,
    corpus: Path,
    artifact: BackpackArtifact,
) -> Path:
    config = {
        "schema": PRODUCTION_RAILS_SCHEMA,
        "pipeline_microbatch": PIPELINE_MICROBATCH,
        "model_layer_count": 45,
        "layers": list(range(45)),
        "uniform_builder": "test_hf_uniform_physical_provider:fixture_uniform_builder",
        "backpack_mixer": "test_hf_uniform_physical_provider:fixture_backpack_mixer",
        "score_contract": {
            "positions_per_window": 1024,
            "support": 8192,
            "window_ids": list(range(64)),
        },
        "allowed_artifacts": {
            artifact.identity.sha256: {
                "basis_sha256": artifact.identity.basis_sha256,
                "checkpoint": "UPDATE_000",
                "artifact_manifest_sha256": _sha_bytes(
                    (artifact.root / "ARTIFACT.json").read_bytes()
                ),
                "checkpoint_sha256": artifact.checkpoint_sha256,
                "artifact_mode": "hf-uniform-q2-native-rest-v1",
            }
        },
        "continuation": {
            "rank": 0,
            "world_size": 2,
            "layer_split": {"0": [0, 22], "1": [23, 44]},
            "hf_uniform_provider_factory": (
                "banana_smasher.hf_uniform_physical_provider:open_provider"
            ),
            "hf_uniform_provider_source_sha256": _provider_source_sha(),
            "hf_uniform_backend_factory": (
                "test_hf_uniform_physical_provider:open_fixture_hf_uniform_backend"
            ),
            "suite_lock_path": str(suite_lock),
            "teacher_capture_path": str(teacher_capture),
            "corpus_path": str(corpus),
            "executor_factory": (
                "test_hf_uniform_physical_provider:open_fixture_executor"
            ),
        },
    }
    path = tmp_path / "production-rails.rank0.json"
    path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _provider_binding_sha(config: dict[str, object]) -> str:
    binding_fields = {
        key: config.get(key)
        for key in (
            "schema",
            "pipeline_microbatch",
            "model_layer_count",
            "layers",
            "uniform_builder",
            "backpack_mixer",
            "score_contract",
            "continuation_science",
        )
    }
    return _sha_bytes(
        json.dumps(binding_fields, sort_keys=True, separators=(",", ":")).encode()
    )


def _provider(
    tmp_path: Path,
    *,
    teacher_capture: Path,
    suite_lock: Path,
    corpus: Path,
    artifact: BackpackArtifact,
    run_root: Path | None = None,
) -> HFUniformPhysicalProvider:
    return HFUniformPhysicalProvider(
        artifact_root=artifact.root,
        identity_sha256=artifact.identity.sha256,
        basis_sha256=artifact.identity.basis_sha256,
        checkpoint="UPDATE_000",
        checkpoint_sha256=artifact.checkpoint_sha256,
        rank=0,
        run_root=tmp_path / "provider-run" if run_root is None else run_root,
        config={
            "model_layer_count": 45,
            "layer_split": {"0": [0, 22], "1": [23, 44]},
            "hf_uniform_backend_factory": (
                "test_hf_uniform_physical_provider:open_fixture_hf_uniform_backend"
            ),
            "suite_lock_path": str(suite_lock),
            "teacher_capture_path": str(teacher_capture),
            "corpus_path": str(corpus),
            "executor_factory": (
                "test_hf_uniform_physical_provider:open_fixture_executor"
            ),
        },
    )


def test_hf_uniform_provider_uses_checkpointed_parameter_state_not_posthoc_logit_scaling(
    tmp_path: Path,
) -> None:
    checkpoint_sha256 = "a" * 64
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(
        tmp_path, model_index_sha256=str(source["model_index_sha256"])
    )
    suite_lock_path = tmp_path / "suite-lock.json"
    suite_lock_path.write_text(json.dumps(suite, sort_keys=True) + "\n", encoding="utf-8")
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kwargs: open_fixture_executor(**kwargs)
    )
    teacher = runtime.capture_teacher(
        source=source,
        suite_lock=suite,
        corpus=corpus,
        output=tmp_path / "teacher-rows",
    )
    teacher_path = tmp_path / "teacher-capture.json"
    teacher_path.write_text(json.dumps(teacher, sort_keys=True) + "\n", encoding="utf-8")
    provider_binding_sha256 = _provider_binding_sha(
        {
            "schema": PRODUCTION_RAILS_SCHEMA,
            "pipeline_microbatch": PIPELINE_MICROBATCH,
            "model_layer_count": 45,
            "layers": list(range(45)),
            "uniform_builder": "test_hf_uniform_physical_provider:fixture_uniform_builder",
            "backpack_mixer": "test_hf_uniform_physical_provider:fixture_backpack_mixer",
            "score_contract": {
                "positions_per_window": 1024,
                "support": 8192,
                "window_ids": list(range(64)),
            },
        }
    )
    artifact = _artifact(
        tmp_path,
        source=source,
        provider_binding_sha256=provider_binding_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    provider = _provider(
        tmp_path,
        teacher_capture=teacher_path,
        suite_lock=suite_lock_path,
        corpus=corpus,
        artifact=artifact,
        run_root=tmp_path / "provider-run",
    )

    parameter_ids = provider.trainable_parameter_ids()
    before = provider.trainable_parameter_digests()
    pre = dict(provider.score("pre"))
    training = dict(provider.train(1))

    assert "support_logit_scale" not in pre
    assert "support_logit_scale" not in training
    assert parameter_ids
    assert pre["checkpoint"] == "UPDATE_000"
    assert training["checkpoint"] == "UPDATE_001"
    assert provider.trainable_parameter_digests() != before

    info = checkpoint_info(training["checkpoint_path"])
    assert info["format"] == "banana-smasher-resident-checkpoint-v1"
    assert info["parameter_groups"]["routed_q2"] == list(parameter_ids)
    assert info["trainable_count"] == len(parameter_ids)

    restored = _provider(
        tmp_path,
        teacher_capture=teacher_path,
        suite_lock=suite_lock_path,
        corpus=corpus,
        artifact=artifact,
        run_root=tmp_path / "provider-post",
    )
    restored.restore_training(pre, training)
    post = dict(restored.score("post"))

    assert restored.trainable_parameter_ids() == parameter_ids
    assert restored.trainable_parameter_digests() == provider.trainable_parameter_digests()
    assert "support_logit_scale" not in post
    assert post["checkpoint"] == "UPDATE_001"


def test_hf_uniform_provider_persists_trained_checkpoint_for_fresh_post_score(
    tmp_path: Path,
) -> None:
    checkpoint_sha256 = "a" * 64
    source = _source(tmp_path)
    suite, corpus = _suite_and_corpus(
        tmp_path, model_index_sha256=str(source["model_index_sha256"])
    )
    suite_lock_path = tmp_path / "suite-lock.json"
    suite_lock_path.write_text(json.dumps(suite, sort_keys=True) + "\n", encoding="utf-8")
    runtime = ShardedHFBalanced64Runtime(
        executor_factory=lambda **kwargs: open_fixture_executor(**kwargs)
    )
    teacher = runtime.capture_teacher(
        source=source,
        suite_lock=suite,
        corpus=corpus,
        output=tmp_path / "teacher-rows",
    )
    teacher_path = tmp_path / "teacher-capture.json"
    teacher_path.write_text(json.dumps(teacher, sort_keys=True) + "\n", encoding="utf-8")
    provisional_config = {
        "schema": PRODUCTION_RAILS_SCHEMA,
        "pipeline_microbatch": PIPELINE_MICROBATCH,
        "model_layer_count": 45,
        "layers": list(range(45)),
        "uniform_builder": "test_hf_uniform_physical_provider:fixture_uniform_builder",
        "backpack_mixer": "test_hf_uniform_physical_provider:fixture_backpack_mixer",
        "score_contract": {
            "positions_per_window": 1024,
            "support": 8192,
            "window_ids": list(range(64)),
        },
    }
    provider_binding_sha256 = _provider_binding_sha(provisional_config)
    artifact = _artifact(
        tmp_path,
        source=source,
        provider_binding_sha256=provider_binding_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    config_path = _config(
        tmp_path,
        teacher_capture=teacher_path,
        suite_lock=suite_lock_path,
        corpus=corpus,
        artifact=artifact,
    )

    pre_api = ResidentRepairAPI(
        rails=ProductionRails.from_file(config_path, run_root=tmp_path / "pre-run"),
        run_root=tmp_path / "pre-facade",
    )
    pre = pre_api.score_pre(artifact, checkpoint_sha=checkpoint_sha256)

    train_api = ResidentRepairAPI(
        rails=ProductionRails.from_file(config_path, run_root=tmp_path / "train-run"),
        run_root=tmp_path / "train-facade",
    )
    train_api.restore_pre_score(pre, artifact, checkpoint_sha=checkpoint_sha256)
    training = train_api.repair_train(
        artifact, updates=45, checkpoint_sha=checkpoint_sha256
    )

    post_api = ResidentRepairAPI(
        rails=ProductionRails.from_file(config_path, run_root=tmp_path / "post-run"),
        run_root=tmp_path / "post-facade",
    )
    post_api.restore_training(pre, training, artifact, checkpoint_sha=checkpoint_sha256)
    post = post_api.score_post(artifact, checkpoint_sha=checkpoint_sha256)

    checkpoint_info_row = checkpoint_info(training["checkpoint_path"])

    assert pre["checkpoint"] == "UPDATE_000"
    assert training["checkpoint"] == "UPDATE_045"
    assert Path(str(training["checkpoint_path"])).is_file()
    assert training["checkpoint_sha256"] == _sha_bytes(
        Path(str(training["checkpoint_path"])).read_bytes()
    )
    assert checkpoint_info_row["format"] == "banana-smasher-resident-checkpoint-v1"
    assert checkpoint_info_row["parameter_groups"]["routed_q2"] == training["trainable_parameter_ids"]
    assert checkpoint_info_row["trainable_count"] == len(training["trainable_parameter_ids"])
    assert training["objective"] == "teacher_kl"
    assert training["changed_parameter_ids"] == training["trainable_parameter_ids"]
    assert post["checkpoint"] == "UPDATE_045"
    assert post["rank_layer_range"] == [0, 22]
    assert post["mean_kld"] != pre["mean_kld"]
