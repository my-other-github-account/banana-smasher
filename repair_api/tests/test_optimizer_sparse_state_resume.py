from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest


ASSETS = (
    Path("repair_api/assets/modern_green_clean_u0.py"),
    Path("repair_api/assets/static_w28_modern_green_clean_u0.py"),
    Path("ds4-flash-kldmatrix/repair_api/assets/modern_green_clean_u0.py"),
    Path("ds4-flash-kldmatrix/repair_api/assets/static_w28_modern_green_clean_u0.py"),
)
DORMANT = "model.layers.10.self_attn.compressor.indexer.kv_norm"
ACTIVE = "model.layers.11.input_layernorm.weight"


def _load_merge(asset: Path) -> Callable[[list[dict[str, Any]], dict[str, dict[str, object]]], dict[str, Any]]:
    source = asset.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "merge_optimizer_state"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            ast.Assign(
                targets=[ast.Name(id="DORMANT_NORMS", ctx=ast.Store())],
                value=ast.Set(elts=[ast.Constant(DORMANT)]),
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(asset), "exec"), namespace)
    return cast(
        Callable[[list[dict[str, Any]], dict[str, dict[str, object]]], dict[str, Any]],
        namespace["merge_optimizer_state"],
    )


def _rows(*, include_dormant: bool, include_active: bool):
    rank0_state = {0: {"step": 21}}
    if include_dormant:
        rank0_state[1] = {"step": 20}
    if include_active:
        rank0_state[2] = {"step": 21}
    return [
        {
            "param_names": {
                "luts": ["lut"],
                "norms": [DORMANT, ACTIVE],
                "outputs": [],
            },
            "optimizer": {
                "state": rank0_state,
                "param_groups": [
                    {"params": [0], "lr": 0.5},
                    {"params": [1, 2], "lr": 0.5},
                    {"params": [], "lr": 0.5},
                ],
            },
        },
        {
            "param_names": {"luts": [], "norms": [], "outputs": ["out"]},
            "optimizer": {
                "state": {0: {"step": 21}},
                "param_groups": [
                    {"params": [], "lr": 0.5},
                    {"params": [], "lr": 0.5},
                    {"params": [0], "lr": 0.5},
                ],
            },
        },
    ]


ORDERED = {
    "luts": {"lut": object()},
    "norms": {DORMANT: object(), ACTIVE: object()},
    "outputs": {"out": object()},
}


@pytest.mark.parametrize("relative_asset", ASSETS)
def test_merge_accepts_resumed_state_for_dormant_norm(relative_asset: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    merge = _load_merge(repo / relative_asset)

    merged = merge(_rows(include_dormant=True, include_active=True), ORDERED)

    assert set(merged["state"]) == {0, 1, 2, 3}


@pytest.mark.parametrize("relative_asset", ASSETS)
def test_merge_allows_only_dormant_state_to_be_missing(relative_asset: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    merge = _load_merge(repo / relative_asset)

    merged = merge(_rows(include_dormant=False, include_active=True), ORDERED)
    assert set(merged["state"]) == {0, 2, 3}

    with pytest.raises(RuntimeError, match="sparse-state coverage drift"):
        merge(_rows(include_dormant=True, include_active=False), ORDERED)
