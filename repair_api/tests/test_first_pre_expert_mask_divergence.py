from __future__ import annotations

import hashlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import torch

from repair_api.modern_green_resident import ModernGreenResidentEngine


KNOWN_EQUAL_EMBEDDING = {
    "dtype": "torch.bfloat16",
    "shape": [1, 2048, 4096],
    "sha256": "928bbdbba1d6f7bf65fb5cef01518cab09e5d360d5c3a04edbb9f672551dd26a",
}
EXACT_PLAIN_MASK = {
    "dtype": "torch.bool",
    "shape": [1, 1, 8192, 8192],
    "true_count": 1040448,
    "sha256": "ff130261db6b248e95768545c36b2c0fa8e424a4f7b7663ecf0a11c4a27e66fc",
}


def _bytes_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_registered_sdpa_builds_exact_layer0_mask_without_mutating_known_equal_frontier(
    monkeypatch,
) -> None:
    """Regress the sealed first divergence: custom mask dispatch returned None."""
    config = SimpleNamespace(
        _attn_implementation="official_k2_sink_corrected_sdpa",
        _attn_implementation_internal="official_k2_sink_corrected_sdpa",
    )
    calls: list[str] = []

    def fake_mask(*, config: Any, inputs_embeds: torch.Tensor, **_kwargs: Any) -> Any:
        calls.append(str(config._attn_implementation))
        if config._attn_implementation == "sdpa":
            return dict(EXACT_PLAIN_MASK)
        if config._attn_implementation == "eager":
            return {"dtype": "torch.bfloat16", "kind": "compressor-additive"}
        return None

    transformers = ModuleType("transformers")
    masking_utils = ModuleType("transformers.masking_utils")
    setattr(masking_utils, "create_sliding_window_causal_mask", fake_mask)
    setattr(transformers, "masking_utils", masking_utils)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking_utils)

    template = torch.arange(32, dtype=torch.bfloat16).reshape(1, 8, 4)
    template_sha_before = _bytes_sha256(template)
    frontier_before = dict(KNOWN_EQUAL_EMBEDDING)
    rotary = lambda value, **kwargs: (value, kwargs["layer_type"])
    engine = cast(Any, ModernGreenResidentEngine.__new__(ModernGreenResidentEngine))
    engine.torch = torch
    engine.student = SimpleNamespace(
        device=torch.device("cpu"),
        config=config,
        model=SimpleNamespace(
            config=config,
            model=SimpleNamespace(rotary_emb=rotary),
        ),
    )

    _position_ids, _position_embeddings, masks = engine._positional(
        torch.zeros((1, 8), dtype=torch.long), template, object()
    )

    assert isinstance(masks, dict), "unregistered custom mask dispatch returned None"
    assert masks["plain"] == EXACT_PLAIN_MASK
    assert calls == ["sdpa", "eager"]
    assert config._attn_implementation == "official_k2_sink_corrected_sdpa"
    assert config._attn_implementation_internal == "official_k2_sink_corrected_sdpa"
    assert _bytes_sha256(template) == template_sha_before
    assert KNOWN_EQUAL_EMBEDDING == frontier_before
