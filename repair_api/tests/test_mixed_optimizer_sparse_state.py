from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_ASSET_ROOT = Path(__file__).parents[1] / "assets"
sys.path.insert(0, str(_ASSET_ROOT))
_SOURCE = _ASSET_ROOT / "modern_green_clean_u0.py"
_SPEC = importlib.util.spec_from_file_location("modern_green_clean_u0_sparse_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _group(params: list[int], name: str) -> dict[str, object]:
    return {
        "params": params,
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "group_name": name,
    }


def test_merge_optimizer_state_derives_sparse_template_from_mixed_trainables() -> None:
    pure_lut = "model.layers.0.self_attn.compressor.indexer.lut"
    mixed_lut = "layers.1.qtip2_v7.layer_lut"
    dormant_norm = "model.layers.2.self_attn.compressor.indexer.kv_norm"
    output = "model.layers.1.self_attn.o_proj.output_log_gain"
    ordered_state = {
        "luts": {pure_lut: object(), mixed_lut: object()},
        "norms": {dormant_norm: object()},
        "outputs": {output: object()},
    }
    rows = [
        {
            "param_names": {
                "luts": [pure_lut, mixed_lut],
                "norms": [],
                "outputs": [],
            },
            "optimizer": {
                "state": {0: {"step": 1}},
                "param_groups": [
                    _group([0, 1], "luts"),
                    _group([], "norms"),
                    _group([], "outputs"),
                ],
            },
        },
        {
            "param_names": {
                "luts": [],
                "norms": [dormant_norm],
                "outputs": [output],
            },
            "optimizer": {
                "state": {1: {"step": 1}},
                "param_groups": [
                    _group([], "luts"),
                    _group([0], "norms"),
                    _group([1], "outputs"),
                ],
            },
        },
    ]

    assert dormant_norm in _MODULE.DORMANT_NORMS
    assert mixed_lut not in _MODULE.DORMANT_NORMS

    merged = _MODULE.merge_optimizer_state(rows, ordered_state)

    assert merged["state"] == {0: {"step": 1}, 3: {"step": 1}}
    assert merged["param_groups"][0]["params"] == [0, 1]
    assert merged["param_groups"][1]["params"] == [2]
    assert merged["param_groups"][2]["params"] == [3]
