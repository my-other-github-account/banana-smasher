from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
import gc
import hashlib
import inspect
import sys
import tempfile
import weakref

import pytest
import torch
from unittest.mock import patch

from repair_api.official_k2_resident_score import (
    OFFICIAL_PHYSICAL_LAYER_SHA256,
    OfficialK2ResidentRankEngine,
    OfficialK2ResidentScorer,
    _effective_score_window_batch_size,
)
from repair_api.balanced64 import ArtifactError, RepairArtifact
from repair_api import cli
from repair_api.modern_green_resident import (
    SEALED_GROUPED_EXPERT_SHA256,
    SEALED_GROUPED_WRAPPER_SHA256,
    STATIC_W28_GROUPED_EXPERT_SHA256,
    STATIC_W28_GROUPED_WRAPPER_SHA256,
)


def test_resident_trainer_module_uses_distinct_hash_bound_source() -> None:
    source = (Path(__file__).parents[1] / "official_k2_resident_score.py").read_text()

    assert 'self.official_expert_source = Path(str(config["official_expert_source"])' in source
    assert 'self.expert_source = Path(str(config["resident_expert_source"])' in source
    assert 'self._load_module("fast_v7_expert_base", self.expert_source)' in source
    assert 'self.official_expert_source: expert_source_sha256' in source
    assert 'self.expert_source: resident_expert_source_sha256' in source
    assert 'self.expert_source = Path(str(config["official_expert_source"])' not in source


def test_resident_expert_asset_hash_matches_runtime_identity_without_external_receipts() -> None:
    source = Path(__file__).resolve().parents[1] / "assets" / "fast_v7_expert_base.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == OFFICIAL_PHYSICAL_LAYER_SHA256


def test_modern_green_sealed_hashes_match_committed_asset_bytes() -> None:
    assets = Path(__file__).resolve().parents[1] / "assets"

    assert hashlib.sha256((assets / "fast_k2_grouped.py").read_bytes()).hexdigest() == (
        SEALED_GROUPED_WRAPPER_SHA256
    )
    assert hashlib.sha256((assets / "fast_v7_expert_base.py").read_bytes()).hexdigest() == (
        SEALED_GROUPED_EXPERT_SHA256
    )

    assert hashlib.sha256((assets / "static_w28_fast_k2_grouped.py").read_bytes()).hexdigest() == (
        STATIC_W28_GROUPED_WRAPPER_SHA256
    )
    assert hashlib.sha256((assets / "static_w28_fast_v7_expert_base.py").read_bytes()).hexdigest() == (
        STATIC_W28_GROUPED_EXPERT_SHA256
    )


def test_checkpoint_rebind_reuses_exposed_local_dense_surfaces() -> None:
    exposed: list[object] = []
    loaded: list[tuple[object, object]] = []

    class Trainer:
        @staticmethod
        def expose_local_dense(torch, student, admission):
            exposed.append(admission)
            return {"lut": 1}, {"norm": 2}, {"output": 3}

        @staticmethod
        def load_local_state(rows, saved, device):
            loaded.append((rows, saved))

    engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
    engine.trainer = Trainer()
    engine.torch = object()
    engine.student = SimpleNamespace(device="cuda")
    payload = {"state": {"luts": {"lut": 10}, "norms": {"norm": 20}, "outputs": {"output": 30}}}
    resident_admission = {"trainable_roster": {"luts": [], "rmsnorms": [], "output_gains": []}}

    engine._bind_checkpoint_state(payload, resident_admission)
    engine._bind_checkpoint_state(payload, {"scope": "ROUTED_K2_POST"})

    assert exposed == [resident_admission]
    assert loaded == [
        ({"lut": 1}, {"lut": 10}),
        ({"norm": 2}, {"norm": 20}),
        ({"output": 3}, {"output": 30}),
        ({"lut": 1}, {"lut": 10}),
        ({"norm": 2}, {"norm": 20}),
        ({"output": 3}, {"output": 30}),
    ]


def test_full_manifest_batching_is_admitted_without_changing_window_order() -> None:
    source = inspect.getsource(OfficialK2ResidentRankEngine.score)
    assert '_effective_score_window_batch_size' in source
    assert _effective_score_window_batch_size(4, 64) == 4
    assert 'self.windows[start : start + batch_size]' in source
    assert 'range(completed_before, len(self.windows), batch_size)' in source
    assert 'batch_size > 4' not in source


def test_public_score_cli_forwards_the_requested_receipt_path() -> None:
    source = inspect.getsource(cli.main)
    assert 'receipt_path=args.receipt' in source


def test_public_checkpoint_key_normalizes_raw_u0_label() -> None:
    repaired = RepairArtifact(Path("."), {"checkpoints": {"UPDATE_000": {}}})

    assert repaired.checkpoint_key("U0") == "UPDATE_000"


def test_hash_bound_source_loads_sibling_modules_without_global_path_leak() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "sibling.py").write_text("VALUE = 41\n")
        source = root / "bound_source.py"
        source.write_text("import sibling\nVALUE = sibling.VALUE + 1\n")
        before = list(sys.path)

        module = OfficialK2ResidentRankEngine._load_module("bound_source_fixture", source)

        assert module.VALUE == 42
        assert sys.path == before


def test_hash_bound_base_trainer_is_distinct_from_resident_trainer() -> None:
    source = inspect.getsource(OfficialK2ResidentRankEngine.__init__)
    path_source = inspect.getsource(OfficialK2ResidentRankEngine._prepare_import_paths)

    assert 'self.lp4_train_path: LP4_TRAIN_SOURCE_SHA256' in source
    assert 'sys.modules["lp4_train"] = self.trainer' not in source
    assert "self.lp4_train_path.parent" in path_source


def test_direct_layer_forward_preserves_sealed_cache_and_batch4() -> None:
    caches: list[object] = []
    batches: list[int] = []

    class FakeCache:
        def __init__(self, config):
            self.config = config
            self.layers = [SimpleNamespace(is_initialized=False) for _ in range(2)]

    class Layer:
        pass

    engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
    engine.first = 0
    engine.last = 1
    engine.torch = torch
    engine.config = {}
    engine.student = SimpleNamespace(
        config=object(), model=SimpleNamespace(model=SimpleNamespace(layers=[Layer(), Layer()]))
    )
    engine._positional = lambda ids, template, cache: (
        object(), object(), torch.zeros((4, 1, 8, 8))
    )
    def streamed(layer, hidden, ids, pe, pos, mask, cache, scratch):
        del layer, pe, pos, mask, scratch
        caches.append(cache)
        batches.append(int(hidden.shape[0]))
        assert ids.shape[0] == hidden.shape[0]
        return hidden + 1
    engine._streamed_decoder_layer = streamed
    hidden = torch.zeros((4, 8, 2, 4), dtype=torch.float32)
    ids = torch.zeros((4, 8))
    transformers = ModuleType("transformers")
    cache_utils = ModuleType("transformers.cache_utils")
    setattr(cache_utils, "DynamicCache", FakeCache)
    setattr(transformers, "cache_utils", cache_utils)

    with patch.dict(
        sys.modules,
        {"transformers": transformers, "transformers.cache_utils": cache_utils},
    ):
        result = engine._run_layers(hidden, ids)

    assert torch.equal(result, hidden + 2)
    assert batches == [4, 4]
    assert len(caches) == 2 and caches[0] is caches[1]


def test_exact_alternate_pre_is_admitted_only_for_sealed_reference_diagnostic() -> None:
    from repair_api.balanced64 import ArtifactError
    from repair_api.official_k2_resident_score import (
        ALTERNATE_PRE_CHECKPOINT_SHA256,
        authorize_production_score,
    )

    with pytest.raises(ArtifactError, match="quarantine-only"):
        authorize_production_score(
            0, checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
        )
    admitted = authorize_production_score(
        0,
        checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
        allow_alternate_pre_diagnostic=True,
    )
    assert admitted["scope"] == "ALTERNATE_PRE_DIAGNOSTIC_ONLY"
    assert admitted["checkpoint_sha256"] == ALTERNATE_PRE_CHECKPOINT_SHA256
    with pytest.raises(ArtifactError, match="parentless update 0"):
        authorize_production_score(
            1,
            checkpoint_sha256=ALTERNATE_PRE_CHECKPOINT_SHA256,
            checkpoint_parent_sha256="parent",
            allow_alternate_pre_diagnostic=True,
        )

    source = inspect.getsource(OfficialK2ResidentScorer.parity_tap)
    assert 'self.config.get("parity_tap_mode") == "sealed_reference"' in source


def test_public_parity_tap_can_bind_only_the_exact_routed_sealed_reference() -> None:
    api_source = inspect.getsource(cli.main)
    method_source = inspect.getsource(cli.ResidentRepairAPI.parity_tap)
    scorer_source = inspect.getsource(OfficialK2ResidentScorer.parity_tap)

    assert 'route=json.loads(args.route.read_text()) if args.route is not None else None' in api_source
    assert 'if mode != "sealed_reference"' in method_source
    assert "validate_routed_k2_closure(route)" in method_source
    assert 'backend_config["route_kind"] = ROUTED_K2_ROUTE_KIND' in method_source
    assert 'Path(str(route["official_source_package"])) / "joint_v7_expert_base.py"' in method_source
    assert '"resident_expert_source_sha256": route["official_class_sha256"]' in method_source
    assert 'if self.config.get("route_kind") == ROUTED_K2_ROUTE_KIND' in scorer_source
    assert "authorize_routed_k2_score(" in scorer_source

    from repair_api.official_k2_resident_score import ROUTED_K2_CLOSURE
    assert ROUTED_K2_CLOSURE["official_class_sha256"] == (
        "7687e39fc5b6bb34b30e8d4a79771affb472497f4d2f323adbe1e8e277746729"
    )


def test_routed_pre_class_is_adapted_only_for_sealed_diagnostic_mode() -> None:
    from repair_api.official_k2_resident_score import (
        ROUTED_K2_ROUTE_KIND,
        _bind_resident_expert_class,
    )

    class Routed:
        def __init__(self, layer: int, pilot: bool, *, plane_source: object) -> None:
            self.layer = layer
            self.pilot = pilot
            self.plane_source = plane_source

    diagnostic = SimpleNamespace(JointV7ExpertBase=Routed)
    assert _bind_resident_expert_class(
        diagnostic, ROUTED_K2_ROUTE_KIND, "sealed_reference"
    ) is diagnostic
    adapted = diagnostic.FullyResidentGroupedV7Experts(7, True, plane_source=object())
    assert isinstance(adapted, Routed)
    assert type(adapted).__name__ == "FullyResidentGroupedV7Experts"
    assert adapted.resident_bytes == 0

    resident = object()
    production = SimpleNamespace(
        JointV7ExpertBase=Routed,
        FullyResidentGroupedV7Experts=resident,
    )
    assert _bind_resident_expert_class(production, ROUTED_K2_ROUTE_KIND, None) is production
    assert production.FullyResidentGroupedV7Experts is resident

    with pytest.raises(ArtifactError, match="missing FullyResidentGroupedV7Experts"):
        _bind_resident_expert_class(SimpleNamespace(JointV7ExpertBase=Routed), ROUTED_K2_ROUTE_KIND, None)

    ordinary = SimpleNamespace(FullyResidentGroupedV7Experts=resident)
    assert _bind_resident_expert_class(ordinary, None, None) is ordinary
    assert ordinary.FullyResidentGroupedV7Experts is resident


def test_support_mass_diagnostic_matches_sealed_kld_score_arithmetic() -> None:
    from repair_api.official_k2_resident_score import _support_mass_diagnostic

    ref_lp = torch.tensor([[0.0, -1.0], [-2.0, -3.0]], dtype=torch.float16)
    q_lp = torch.tensor([[-0.5, -1.5], [-2.5, -3.5]], dtype=torch.float16)
    diagnostic = _support_mass_diagnostic(torch, ref_lp, q_lp)

    expected_p = ref_lp.float().exp().sum(-1)
    expected_q = q_lp.float().exp().sum(-1)
    assert diagnostic == {
        "mass_p_mean": expected_p.mean().item(),
        "mass_p_sum": expected_p.sum().item(),
        "mass_q_mean": expected_q.mean().item(),
        "mass_q_sum": expected_q.sum().item(),
    }


def test_routed_diagnostic_terminal_allows_only_source_reads() -> None:
    from repair_api.official_k2_resident_score import _validate_parity_terminal

    terminals = [{
        "timed_model_payload_reads": 7,
        "fallback_calls": 0,
        "reconstruction_calls": 0,
        "reference_fwht_calls": 0,
        "cpu_relay_bytes": 0,
        "layer_streaming_calls": 0,
    }]
    _validate_parity_terminal(terminals, allow_source_reads=True)
    with pytest.raises(ArtifactError, match="terminal closure"):
        _validate_parity_terminal(terminals, allow_source_reads=False)
    terminals[0]["fallback_calls"] = 1
    with pytest.raises(ArtifactError, match="terminal closure"):
        _validate_parity_terminal(terminals, allow_source_reads=True)


def test_streamed_decoder_workspace_is_bitwise_and_never_holds_four_full_tensors() -> None:
    """Product-first residuals ping-pong exactly between hidden and one scratch bank."""
    full_ptrs: set[int] = set()

    class HyperConnection:
        @staticmethod
        def __call__(hidden):
            batch, seq, streams, _ = hidden.shape
            post = torch.full((batch, seq, streams), 0.75, dtype=hidden.dtype)
            comb = torch.tensor([[0.75, 0.25], [0.125, 0.875]], dtype=hidden.dtype)
            comb = comb.expand(batch, seq, streams, streams).clone()
            return post, comb, hidden.sum(dim=2)

    class Attention:
        @staticmethod
        def __call__(hidden, **kwargs):
            del kwargs
            return hidden * torch.tensor(0.375, dtype=hidden.dtype), None

    class Mlp:
        @staticmethod
        def __call__(hidden, input_ids=None):
            assert input_ids is not None
            return hidden * torch.tensor(-0.625, dtype=hidden.dtype)

    layer = SimpleNamespace(
        attn_hc=HyperConnection(),
        ffn_hc=HyperConnection(),
        self_attn=Attention(),
        mlp=Mlp(),
        input_layernorm=lambda value: value,
        post_attention_layernorm=lambda value: value,
    )
    engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
    engine.torch = torch
    engine._call_chunked_self_attention = lambda attention, hidden, **kwargs: attention(hidden, **kwargs)
    engine._release_attention_output_workspace = lambda attention_output: None
    hidden = torch.arange(4 * 8 * 2 * 4, dtype=torch.bfloat16).reshape(4, 8, 2, 4)
    original = hidden.clone()
    with patch.object(torch, "empty_like", wraps=torch.empty_like) as empty_like:
        scratch = engine._decoder_workspace_for(hidden, stream_key="fixture")
        assert engine._decoder_workspace_for(hidden, stream_key="fixture") is scratch
    assert empty_like.call_count == 1
    full_ptrs.update((hidden.data_ptr(), scratch.data_ptr()))
    ids = torch.zeros((4, 8), dtype=torch.long)

    # Reproduce the immutable public decoder expression exactly: product first,
    # matmul second, then elementwise add at each residual boundary.
    post, comb, collapsed = layer.attn_hc(original)
    attention, _ = layer.self_attn(layer.input_layernorm(collapsed))
    expected = post.to(original.dtype).unsqueeze(-1) * attention.unsqueeze(-2) + torch.matmul(
        comb.to(original.dtype).transpose(-1, -2), original
    )
    post, comb, collapsed = layer.ffn_hc(expected)
    mlp = layer.mlp(layer.post_attention_layernorm(collapsed), input_ids=ids)
    expected = post.to(original.dtype).unsqueeze(-1) * mlp.unsqueeze(-2) + torch.matmul(
        comb.to(original.dtype).transpose(-1, -2), expected
    )

    result = engine._streamed_decoder_layer(
        layer, hidden, ids, object(), object(), object(), object(), scratch
    )

    assert torch.equal(result, expected)
    assert result.data_ptr() == hidden.data_ptr()
    assert len(full_ptrs) == 2


def test_chunked_eager_attention_is_bitwise_and_bounds_query_workspace() -> None:
    """Query streaming keeps batch4 and exact per-row eager arithmetic."""
    engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
    engine.torch = torch
    module = SimpleNamespace(num_key_value_groups=2, sinks=torch.tensor([0.25, -0.5]))
    query = torch.randn((4, 2, 9, 5), dtype=torch.bfloat16)
    key = torch.randn((4, 1, 11, 5), dtype=torch.bfloat16)
    value = torch.randn((4, 1, 11, 5), dtype=torch.bfloat16)
    mask = torch.zeros((4, 1, 9, 11), dtype=torch.bfloat16)

    repeated_key = key[:, :, None, :, :].expand(4, 1, 2, 11, 5).reshape(4, 2, 11, 5)
    repeated_value = value[:, :, None, :, :].expand(4, 1, 2, 11, 5).reshape(4, 2, 11, 5)
    weights = torch.matmul(query, repeated_key.transpose(2, 3)) * (5**-0.5)
    weights = weights + mask
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(4, -1, 9, -1)
    logits = torch.cat([weights, sinks], dim=-1)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(logits, dim=-1, dtype=logits.dtype)[..., :-1]
    expected = torch.matmul(probs.to(repeated_value.dtype), repeated_value).transpose(1, 2).contiguous()

    observed_batches: list[int] = []
    observed_workspaces: list[tuple[int, int, int]] = []
    engine._attention_workspace = None
    result, returned_weights = engine._chunked_eager_attention_forward(
        module, query, key, value, mask, scaling=5**-0.5,
        query_chunk_size=3, _chunk_observer=lambda rows: observed_batches.append(rows),
        _official_k2_workspace_factory=engine._attention_workspace_for,
        _workspace_observer=lambda output, weights, logits: observed_workspaces.append(
            (output.data_ptr(), weights.data_ptr(), logits.data_ptr())
        ),
    )

    assert torch.equal(result, expected)
    assert returned_weights is None
    assert observed_batches == [3, 3, 3]
    assert result.shape[0] == 4
    assert len(set(observed_workspaces)) == 1

    second, _ = engine._chunked_eager_attention_forward(
        module, query, key, value, mask, scaling=5**-0.5,
        query_chunk_size=3,
        _official_k2_workspace_factory=engine._attention_workspace_for,
        _workspace_observer=lambda output, weights, logits: observed_workspaces.append(
            (output.data_ptr(), weights.data_ptr(), logits.data_ptr())
        ),
    )
    assert torch.equal(second, expected)
    assert len(set(observed_workspaces)) == 1


def test_chunked_attention_seam_registers_public_interface_and_restores_config(monkeypatch) -> None:
    def original_backend(*args, **kwargs):
        del args, kwargs
        raise AssertionError("registered eager backend must not run")

    class Interface:
        mapping = {"eager": original_backend}

        @classmethod
        def register(cls, key, value):
            cls.mapping[key] = value

        @classmethod
        def get_interface(cls, key, default):
            return cls.mapping.get(key, default)

    monkeypatch.setitem(globals(), "ALL_ATTENTION_FUNCTIONS", Interface)

    class Attention:
        def __init__(self):
            self.config = SimpleNamespace(_attn_implementation="eager")
            self.num_key_value_groups = 1
            self.sinks = torch.zeros(1)
            self.training = False

        def forward(self, hidden, attention_mask=None, **kwargs):
            backend = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, original_backend
            )
            query = hidden.transpose(1, 2)
            return backend(self, query, query, query, attention_mask, scaling=1.0, **kwargs)

        __call__ = forward

    engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
    engine.torch = torch
    engine.config = {"attention_query_chunk_size": 3}
    attention = Attention()
    hidden = torch.randn((4, 8, 1, 3), dtype=torch.bfloat16)
    observed_batches: list[int] = []
    output, weights = engine._call_chunked_self_attention(
        attention, hidden, _chunk_observer=lambda rows: observed_batches.append(rows)
    )

    assert output.shape == hidden.shape
    assert weights is None
    assert observed_batches == [3, 3, 2]
    assert attention.config._attn_implementation == "eager"
    assert Interface.mapping["official_k2_chunked_eager"] == engine._chunked_eager_attention_forward
    assert engine._attention_workspace is not None


def test_attention_output_workspace_retires_only_after_public_attention() -> None:
    """The output bank dies at the public-attention seam; scratch banks remain."""
    engine = OfficialK2ResidentRankEngine.__new__(OfficialK2ResidentRankEngine)
    engine.torch = torch
    query = torch.randn((2, 2, 7, 3), dtype=torch.bfloat16)
    key = torch.randn((2, 1, 7, 3), dtype=torch.bfloat16)

    output, weights, logits = engine._attention_workspace_for(
        query, key, 4, torch.bfloat16
    )
    output_ptr = output.data_ptr()
    weights_ptr = weights.data_ptr()
    logits_ptr = logits.data_ptr()
    public_attention_output = output.clone()
    public_attention_bytes = public_attention_output.view(torch.uint8).clone()

    engine._release_attention_output_workspace(public_attention_output)

    workspace = next(iter(engine._attention_workspaces.values()))
    assert workspace[1] is None
    assert workspace[2].data_ptr() == weights_ptr
    assert workspace[3].data_ptr() == logits_ptr
    assert torch.equal(public_attention_output.view(torch.uint8), public_attention_bytes)

    del output
    next_output, next_weights, next_logits = engine._attention_workspace_for(
        query, key, 4, torch.bfloat16
    )
    assert next_output.data_ptr() != public_attention_output.data_ptr()
    assert next_weights.data_ptr() == weights_ptr
    assert next_logits.data_ptr() == logits_ptr
    assert next_output.numel() * next_output.element_size() == query.numel() * query.element_size()
    assert output_ptr != public_attention_output.data_ptr()
