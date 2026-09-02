from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from banana_smasher.hf_sharded_balanced64_executor import (
    ArtifactTensorStore,
    LayerStreamedHFSession,
    PackageHFShardedExecutor,
    require_hf_runtime,
    top_support,
)
from banana_smasher.qtip1 import QTIP2_GEOMETRY, encode_qtip, gaussian_tlut


def _member(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_package_executor_capability_uses_nested_hybrid_moe_config(tmp_path: Path) -> None:
    root = tmp_path / "model"
    root.mkdir()
    config = {
        "architectures": ["FixtureForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 4,
            "n_routed_experts": 8,
            "hc_mult": 2,
            "layer_types": [
                "linear_attention",
                "deepseek_sparse_attention",
                "linear_attention",
                "deepseek_sparse_attention",
            ],
            "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
            "vocab_size": 9000,
        },
        "vision_config": {"depth": 2},
    }
    (root / "config.json").write_text(json.dumps(config))
    subject = {"source": {"model_root": str(root)}}

    assert PackageHFShardedExecutor.supports(subject=subject, role="candidate_pre") is True
    config["text_config"]["hc_mult"] = 0
    (root / "config.json").write_text(json.dumps(config))
    assert PackageHFShardedExecutor.supports(subject=subject, role="candidate_pre") is False


def test_solve_runtime_pins_native_glm5_next_transformers_revision() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    project_dependencies = project["project"]["dependencies"]
    solve = project["project"]["optional-dependencies"]["solve"]

    assert project["tool"]["hatch"]["metadata"]["allow-direct-references"] is True
    assert "tokenizers==0.23.1" in project_dependencies
    assert (
        "transformers @ git+https://github.com/huggingface/transformers.git"
        "@b6c0bfe04c823a7b2ca48f91b8b91b2a7741f309" in solve
    )


def test_dependency_preflight_accepts_exact_pinned_native_runtime(monkeypatch) -> None:
    import types
    import banana_smasher.hf_sharded_balanced64_executor as executor

    torch = pytest.importorskip("torch")
    pinned_transformers = types.SimpleNamespace(__version__="5.16.0.dev0")

    def available(name: str):
        if name == "torch":
            return torch
        if name == "transformers":
            return pinned_transformers
        raise ModuleNotFoundError(name)

    class _Distribution:
        def __init__(self, version: str, direct_url: str | None = None) -> None:
            self.version = version
            self._direct_url = direct_url

        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return self._direct_url

    def distribution(name: str):
        if name == "transformers":
            return _Distribution(
                "5.16.0.dev0",
                json.dumps(
                    {
                        "url": "https://github.com/huggingface/transformers.git",
                        "vcs_info": {
                            "vcs": "git",
                            "commit_id": "b6c0bfe04c823a7b2ca48f91b8b91b2a7741f309",
                        },
                    }
                ),
            )
        if name == "tokenizers":
            return _Distribution("0.23.1")
        raise executor.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(executor.importlib, "import_module", available)
    monkeypatch.setattr(executor.importlib_metadata, "distribution", distribution)
    resolved_torch, resolved_transformers = require_hf_runtime()
    assert resolved_torch is torch
    assert resolved_transformers is pinned_transformers


@pytest.mark.parametrize(
    ("repository_url", "vcs", "commit_id", "tokenizers_version"),
    [
        ("https://github.com/huggingface/transformers.git", "git", "0" * 40, "0.23.1"),
        (
            "https://github.com/huggingface/transformers.git",
            "git",
            "b6c0bfe04c823a7b2ca48f91b8b91b2a7741f309",
            "0.23.2",
        ),
        (
            "https://example.invalid/transformers.git",
            "git",
            "b6c0bfe04c823a7b2ca48f91b8b91b2a7741f309",
            "0.23.1",
        ),
        (
            "https://github.com/huggingface/transformers.git",
            "hg",
            "b6c0bfe04c823a7b2ca48f91b8b91b2a7741f309",
            "0.23.1",
        ),
    ],
)
def test_dependency_preflight_rejects_runtime_identity_drift(
    monkeypatch,
    repository_url: str,
    vcs: str,
    commit_id: str,
    tokenizers_version: str,
) -> None:
    import types
    import banana_smasher.hf_sharded_balanced64_executor as executor

    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        executor.importlib,
        "import_module",
        lambda name: torch if name == "torch" else types.SimpleNamespace(__version__="5.16.0.dev0"),
    )

    class _Distribution:
        def __init__(self, version: str, direct_url: str | None = None) -> None:
            self.version = version
            self._direct_url = direct_url

        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return self._direct_url

    def distribution(name: str):
        if name == "transformers":
            return _Distribution(
                "5.16.0.dev0",
                json.dumps(
                    {
                        "url": repository_url,
                        "vcs_info": {"vcs": vcs, "commit_id": commit_id},
                    }
                ),
            )
        return _Distribution(tokenizers_version)

    monkeypatch.setattr(executor.importlib_metadata, "distribution", distribution)
    with pytest.raises(RuntimeError, match="runtime identity"):
        require_hf_runtime()


def test_dependency_preflight_is_explicit_when_native_hf_runtime_is_missing(monkeypatch) -> None:
    import banana_smasher.hf_sharded_balanced64_executor as executor

    real_import = executor.importlib.import_module

    def missing(name: str):
        if name == "transformers":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(executor.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match=r"transformers>=5\.16"):
        require_hf_runtime()


def test_meta_model_uses_declared_installed_architecture_without_remote_code(tmp_path: Path) -> None:
    import types

    torch = pytest.importorskip("torch")
    config = types.SimpleNamespace(architectures=["FixtureConditional"])

    class _AutoConfig:
        @staticmethod
        def from_pretrained(root, *, local_files_only, trust_remote_code):
            assert Path(root) == tmp_path
            assert local_files_only is True
            assert trust_remote_code is False
            return config

    class _FixtureConditional(torch.nn.Module):
        def __init__(self, received):
            super().__init__()
            assert received is config
            self.weight = torch.nn.Parameter(torch.ones(1))

    session = object.__new__(LayerStreamedHFSession)
    session.torch = torch
    session.source = {"model_root": str(tmp_path)}
    session.transformers = types.SimpleNamespace(
        AutoConfig=_AutoConfig,
        FixtureConditional=_FixtureConditional,
        PreTrainedModel=torch.nn.Module,
    )

    model = session._meta_model()

    assert isinstance(model, _FixtureConditional)
    assert model.weight.is_meta
    assert model.training is False


def test_artifact_store_decodes_real_q2_wire(monkeypatch, tmp_path: Path) -> None:
    import banana_smasher.hf_sharded_balanced64_executor as executor

    class _SourceStore:
        def __init__(self, source):
            assert source == {"model_root": "/model"}

        def names(self):
            return set()

    monkeypatch.setattr(executor, "SourceTensorStore", _SourceStore)
    root = tmp_path / "artifact"
    routed_dir = root / "routed"
    routed_dir.mkdir(parents=True)
    matrix = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    tlut = gaussian_tlut(bits=QTIP2_GEOMETRY.tlut_bits, columns=QTIP2_GEOMETRY.V)
    encoded = encode_qtip(matrix, geometry=QTIP2_GEOMETRY, tlut=tlut)
    trellis = routed_dir / "trellis.npy"
    scales = routed_dir / "scales.npy"
    np.save(trellis, encoded.packed, allow_pickle=False)
    np.save(scales, encoded.scales, allow_pickle=False)
    artifact = {
        "artifact_root": str(root),
        "source": {"model_root": "/model"},
        "geometry": {"routed_layer_ids": [3, 4, 5, 6, 7, 8]},
        "routed_tensors": [
            {
                "name": "model.language_model.layers.8.mlp.experts.0.gate_proj.weight",
                "shape": [2, 32],
                "wire": {
                    "geometry": QTIP2_GEOMETRY.as_mapping(),
                    "trellis": _member(trellis, root),
                    "scales": _member(scales, root),
                },
            }
        ],
        "native_tensors": [],
    }
    store = ArtifactTensorStore(artifact)

    decoded = store.tensor("model.language_model.layers.8.mlp.experts.0.gate_proj.weight")

    assert decoded.shape == matrix.shape
    assert np.isfinite(decoded.numpy()).all()
    assert store.payload_reads == 2


def test_candidate_source_prefix_includes_first_routed_boundary(monkeypatch, tmp_path: Path) -> None:
    import banana_smasher.hf_sharded_balanced64_executor as executor

    torch = pytest.importorskip("torch")
    tensors = {
        "model.embed_tokens.weight": torch.tensor([[1.0, 2.0]]),
        "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": torch.tensor([[3.0, 4.0]]),
    }

    class _SourceStore:
        def __init__(self, source):
            assert source == {"model_root": "/model"}

        def names(self):
            return set(tensors)

        def tensor(self, name):
            return tensors[name]

    monkeypatch.setattr(executor, "SourceTensorStore", _SourceStore)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    first_routed = "model.language_model.layers.3.mlp.experts.0.gate_proj.weight"
    artifact = {
        "artifact_root": str(artifact_root),
        "source": {"model_root": "/model"},
        "geometry": {"routed_layer_ids": [3, 4]},
        "routed_tensors": [{
            "name": first_routed,
            "path": "routed/must-not-read.npy",
            "shape": [1, 2],
            "wire": {},
        }],
        "native_tensors": [{
            "name": "model.embed_tokens.weight",
            "path": "native/stale-copy.bin",
            "shape": [1, 2],
            "dtype": "F32",
        }],
    }

    store = ArtifactTensorStore(artifact)

    assert store.names() == set(tensors)
    assert store.tensor("model.embed_tokens.weight") is tensors["model.embed_tokens.weight"]
    assert store.tensor(first_routed) is tensors[first_routed]
    assert store.payload_reads == 0
    assert store.model_reads == 2


def test_candidate_source_prefix_includes_second_routed_boundary(monkeypatch, tmp_path: Path) -> None:
    import banana_smasher.hf_sharded_balanced64_executor as executor

    torch = pytest.importorskip("torch")
    second_routed = "model.language_model.layers.4.mlp.experts.0.gate_proj.weight"
    tensors = {
        second_routed: torch.tensor([[5.0, 6.0]]),
    }

    class _SourceStore:
        def __init__(self, source):
            assert source == {"model_root": "/model"}

        def names(self):
            return set(tensors)

        def tensor(self, name):
            return tensors[name]

    monkeypatch.setattr(executor, "SourceTensorStore", _SourceStore)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = {
        "artifact_root": str(artifact_root),
        "source": {"model_root": "/model"},
        "geometry": {"routed_layer_ids": [3, 4, 5]},
        "routed_tensors": [{
            "name": second_routed,
            "path": "routed/must-not-read.npy",
            "shape": [1, 2],
            "wire": {},
        }],
        "native_tensors": [],
    }

    store = ArtifactTensorStore(artifact)

    assert store.tensor(second_routed) is tensors[second_routed]
    assert store.payload_reads == 0
    assert store.model_reads == 1


def test_candidate_source_prefix_includes_third_routed_boundary(monkeypatch, tmp_path: Path) -> None:
    import banana_smasher.hf_sharded_balanced64_executor as executor

    torch = pytest.importorskip("torch")
    third_routed = "model.language_model.layers.5.mlp.experts.0.gate_proj.weight"
    tensors = {third_routed: torch.tensor([[7.0, 8.0]])}

    class _SourceStore:
        def __init__(self, source):
            assert source == {"model_root": "/model"}

        def names(self):
            return set(tensors)

        def tensor(self, name):
            return tensors[name]

    monkeypatch.setattr(executor, "SourceTensorStore", _SourceStore)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = {
        "artifact_root": str(artifact_root),
        "source": {"model_root": "/model"},
        "geometry": {"routed_layer_ids": [3, 4, 5, 6]},
        "routed_tensors": [{
            "name": third_routed,
            "path": "routed/must-not-read.npy",
            "shape": [1, 2],
            "wire": {},
        }],
        "native_tensors": [],
    }

    store = ArtifactTensorStore(artifact)

    assert store.tensor(third_routed) is tensors[third_routed]
    assert store.payload_reads == 0
    assert store.model_reads == 1


def test_candidate_source_prefix_includes_fourth_routed_boundary(monkeypatch, tmp_path: Path) -> None:
    import banana_smasher.hf_sharded_balanced64_executor as executor

    torch = pytest.importorskip("torch")
    fourth_routed = "model.language_model.layers.6.mlp.experts.0.gate_proj.weight"
    tensors = {fourth_routed: torch.tensor([[9.0, 10.0]])}

    class _SourceStore:
        def __init__(self, source):
            assert source == {"model_root": "/model"}

        def names(self):
            return set(tensors)

        def tensor(self, name):
            return tensors[name]

    monkeypatch.setattr(executor, "SourceTensorStore", _SourceStore)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = {
        "artifact_root": str(artifact_root),
        "source": {"model_root": "/model"},
        "geometry": {"routed_layer_ids": [3, 4, 5, 6, 7]},
        "routed_tensors": [{
            "name": fourth_routed,
            "path": "routed/must-not-read.npy",
            "shape": [1, 2],
            "wire": {},
        }],
        "native_tensors": [],
    }

    store = ArtifactTensorStore(artifact)

    assert store.tensor(fourth_routed) is tensors[fourth_routed]
    assert store.payload_reads == 0
    assert store.model_reads == 1


def test_candidate_source_prefix_includes_fifth_routed_boundary(monkeypatch, tmp_path: Path) -> None:
    import banana_smasher.hf_sharded_balanced64_executor as executor

    torch = pytest.importorskip("torch")
    fifth_routed = "model.language_model.layers.7.mlp.experts.0.gate_proj.weight"
    tensors = {fifth_routed: torch.tensor([[11.0, 12.0]])}

    class _SourceStore:
        def __init__(self, source):
            assert source == {"model_root": "/model"}

        def names(self):
            return set(tensors)

        def tensor(self, name):
            return tensors[name]

    monkeypatch.setattr(executor, "SourceTensorStore", _SourceStore)
    artifact_root = tmp_path / "artifact"
    artifact_root.mkdir()
    artifact = {
        "artifact_root": str(artifact_root),
        "source": {"model_root": "/model"},
        "geometry": {"routed_layer_ids": [3, 4, 5, 6, 7, 8]},
        "routed_tensors": [{
            "name": fifth_routed,
            "path": "routed/must-not-read.npy",
            "shape": [1, 2],
            "wire": {},
        }],
        "native_tensors": [],
    }

    store = ArtifactTensorStore(artifact)

    assert store.tensor(fifth_routed) is tensors[fifth_routed]
    assert store.payload_reads == 0
    assert store.model_reads == 1


def test_descaled_q2_payload_is_not_scaled_twice() -> None:
    from banana_smasher.hf_sharded_balanced64_executor import ArtifactTensorStore

    name = "model.language_model.layers.8.mlp.experts.0.gate_proj.weight"
    store = ArtifactTensorStore.__new__(ArtifactTensorStore)
    store.routed = {
        name: {"source_transform": {"output_quantity": "descaled_weight"}}
    }
    store.source_routed_layers = frozenset()

    assert store.requires_source_scale(name) is False


def test_working_set_materialization_casts_fp_weights_and_allows_parameterless_modules() -> None:
    torch = pytest.importorskip("torch")

    class _Store:
        def names(self):
            return {"linear.weight"}

        def load_many(self, names):
            assert names == ["linear.weight"]
            return {"linear.weight": torch.ones((2, 3), dtype=torch.float32)}

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.linear = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
            self.identity = torch.nn.Identity()

    session = object.__new__(LayerStreamedHFSession)
    session.torch = torch
    session.device = "cpu"
    session.store = _Store()
    session._model = _Model()
    session._working_set_loads = 0

    session._materialize(session._model.identity)
    session._materialize(session._model.linear)

    assert session._model.linear.weight.device.type == "cpu"
    assert session._model.linear.weight.dtype == torch.bfloat16
    assert session._working_set_loads == 1


def test_working_set_fuses_numeric_experts_and_applies_exact_scales() -> None:
    torch = pytest.importorskip("torch")
    tensors = {
        "experts.0.gate_proj.weight": torch.ones((2, 3)),
        "experts.0.gate_proj.weight_scale_inv": torch.tensor(2.0),
        "experts.0.up_proj.weight": torch.ones((2, 3)) * 2,
        "experts.0.up_proj.weight_scale_inv": torch.tensor(3.0),
        "experts.0.down_proj.weight": torch.ones((3, 2)) * 3,
        "experts.0.down_proj.weight_scale_inv": torch.tensor(4.0),
        "experts.1.gate_proj.weight": torch.ones((2, 3)) * 4,
        "experts.1.gate_proj.weight_scale_inv": torch.tensor(5.0),
        "experts.1.up_proj.weight": torch.ones((2, 3)) * 5,
        "experts.1.up_proj.weight_scale_inv": torch.tensor(6.0),
        "experts.1.down_proj.weight": torch.ones((3, 2)) * 6,
        "experts.1.down_proj.weight_scale_inv": torch.tensor(7.0),
    }

    class _Store:
        def names(self):
            return set(tensors)

        def load_many(self, names):
            return {name: tensors[name] for name in names}

    class _Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.gate_up_proj = torch.nn.Parameter(torch.empty((2, 4, 3)))
                self.down_proj = torch.nn.Parameter(torch.empty((2, 3, 2)))

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = _Experts()

    session = object.__new__(LayerStreamedHFSession)
    session.torch = torch
    session.device = "cpu"
    session.store = _Store()
    session._model = _Model()
    session._working_set_loads = 0

    session._materialize(session._model.experts)

    gate_up = session._model.experts.gate_up_proj.detach()
    down = session._model.experts.down_proj.detach()
    assert gate_up[:, :2].tolist() == [
        [[2.0] * 3, [2.0] * 3],
        [[20.0] * 3, [20.0] * 3],
    ]
    assert gate_up[:, 2:].tolist() == [
        [[6.0] * 3, [6.0] * 3],
        [[30.0] * 3, [30.0] * 3],
    ]
    assert down[:, 0, 0].tolist() == [12.0, 42.0]


def test_working_set_maps_split_convolution_forget_gate_and_hyperconnection_state() -> None:
    torch = pytest.importorskip("torch")
    tensors = {
        "layer.self_attn.q_conv1d.weight": torch.ones((2, 2)),
        "layer.self_attn.k_conv1d.weight": torch.ones((2, 2)) * 2,
        "layer.self_attn.v_conv1d.weight": torch.ones((2, 2)) * 3,
        "layer.self_attn.dt_bias": torch.tensor([4.0, 5.0]),
        "layer.hc_attn_fn": torch.tensor([6.0, 7.0]),
    }

    class _Store:
        def names(self):
            return set(tensors)

        def load_many(self, names):
            return {name: tensors[name] for name in names}

    class _ForgetGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.dt_bias = torch.nn.Parameter(torch.empty(2))

    class _Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.conv1d = torch.nn.Conv1d(6, 6, 2, groups=6, bias=False)
            self.forget_gate = _ForgetGate()

    class _Hyper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                self.fn = torch.nn.Parameter(torch.empty(2))

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attention()
            self.attn_hc = _Hyper()

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = _Layer()

    session = object.__new__(LayerStreamedHFSession)
    session.torch = torch
    session.device = "cpu"
    session.store = _Store()
    session._model = _Model()
    session._working_set_loads = 0

    session._materialize(session._model.layer)

    assert session._model.layer.self_attn.conv1d.weight[:, 0, 0].tolist() == [
        1.0,
        1.0,
        2.0,
        2.0,
        3.0,
        3.0,
    ]
    assert session._model.layer.self_attn.forget_gate.dt_bias.tolist() == [4.0, 5.0]
    assert session._model.layer.attn_hc.fn.tolist() == [6.0, 7.0]


def test_native_glm5_next_hybrid_tensor_path_produces_finite_nonconstant_logits() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model_class = getattr(transformers, "Glm5NextTextModel", None)
    config_class = getattr(transformers, "Glm5NextTextConfig", None)
    if model_class is None or config_class is None:
        pytest.skip("installed solve runtime does not provide native glm5_next")
    config = config_class(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        kv_lora_rank=4,
        q_lora_rank=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        qk_nope_head_dim=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=1,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -5.0,
        },
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        mlp_layer_types=["dense"] * 3 + ["sparse"],
        indexer_types=["full"] * 4,
        pad_token_id=None,
    )
    torch.manual_seed(7)
    model = model_class(config).eval()
    head = torch.nn.Linear(8, 32, bias=False).eval()

    with torch.no_grad():
        hidden = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False).last_hidden_state[0]
        result = top_support(hidden, head.weight, support_token_ids=None, support=8)

    assert hidden.shape == (4, 8)
    assert result["support_logits"].shape == (4, 8)
    assert np.isfinite(result["support_logits"]).all()
    assert float(np.std(result["support_logits"])) > 0.0


def test_top_support_projects_real_logits_on_identical_ordered_support() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.arange(18, dtype=torch.float32).reshape(9, 2)

    full = top_support(hidden, weight, support_token_ids=None, support=4)
    same = top_support(
        hidden,
        weight,
        support_token_ids=full["support_token_ids"],
        support=4,
    )

    assert full["support_logits"].shape == (2, 4)
    assert same["support_token_ids"].tolist() == full["support_token_ids"].tolist()
    assert np.allclose(same["support_logits"], full["support_logits"])
    assert same["top1_token_ids"].tolist() == full["top1_token_ids"].tolist()
