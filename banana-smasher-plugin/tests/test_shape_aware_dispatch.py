from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/banana_smasher_plugin/dispatch_policy.py"
)


def _load_policy():
    spec = importlib.util.spec_from_file_location("banana_smasher_dispatch_policy", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shape_policy_covers_decision_rows_and_prefill() -> None:
    policy = _load_policy()
    expected = {
        6: ("vq_scalar_m1", 1, 1),
        12: ("vq_scalar_m2", 2, 1),
        24: ("vq_vector_m4", 4, 1),
        48: ("vq_vector_m4_x2", 4, 2),
        96: ("vq_vector_m4_x4", 4, 4),
        192: ("mixed_packed_mblock16", 16, 2),
    }
    for rows, wanted in expected.items():
        decision = policy.shape_policy(rows)
        assert (
            decision["kernel"],
            decision["chunk_tokens"],
            decision["chunks"],
        ) == wanted
        assert decision["route_rows"] == rows
        assert decision["tokens"] == rows // 6
        assert decision["zero_dequant"] is True
        assert decision["graph_reuse"] is (rows <= 96)
        assert decision["activation"] == "active"

    for rows in (378, 384, 390, 3072, 12000, 49152):
        decision = policy.shape_policy(rows)
        assert decision["mblock"] == 16
        assert decision["zero_dequant"] is True
    with pytest.raises(NotImplementedError, match="at most 8192"):
        policy.shape_policy(8193 * 6)


def test_shape_policy_fails_closed_on_unreachable_route_shapes() -> None:
    policy = _load_policy()
    for rows in (0, -6, 1, 7, 97):
        with pytest.raises(ValueError):
            policy.shape_policy(rows)


@pytest.mark.parametrize(
    ("tokens", "kernel", "chunks"),
    (
        (5, "vq_vector_m4_x2", 2),
        (7, "vq_vector_m4_x2", 2),
        (9, "vq_vector_m4_x4", 3),
        (31, "vq_vector_m4_x4", 8),
    ),
)
def test_intermediate_scheduler_shapes_use_four_token_graph_chunks(
    tokens: int, kernel: str, chunks: int
) -> None:
    decision = _load_policy().shape_policy(tokens * 6)
    assert decision["kernel"] == kernel
    assert decision["chunk_tokens"] == 4
    assert decision["chunks"] == chunks


def test_shape_policy_has_no_environment_driven_product_switches() -> None:
    source = SOURCE.read_text()
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "fallback" not in source
    assert policy_labels(source) == {
        "vq_scalar_m1",
        "vq_scalar_m2",
        "vq_vector_m4",
        "vq_vector_m4_x2",
        "vq_vector_m4_x4",
        "mixed_packed_mblock16",
    }


def policy_labels(source: str) -> set[str]:
    return {
        label
        for label in (
            "vq_scalar_m1",
            "vq_scalar_m2",
            "vq_vector_m4",
            "vq_vector_m4_x2",
            "vq_vector_m4_x4",
            "mixed_packed_mblock16",
        )
        if label in source
    }
