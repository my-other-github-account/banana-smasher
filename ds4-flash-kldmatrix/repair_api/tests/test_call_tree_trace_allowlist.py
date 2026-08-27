import json

import torch

from repair_api.call_tree_trace import (
    FullCallTreeTrace,
    _semantic_line_boundary,
    _semantic_python_boundary,
    _tensor_metadata,
)


def test_compact_allowlist_keeps_route_weight_boundary_and_seals_small_terminal(tmp_path) -> None:
    provider = "/sealed/repair_api/assets/fast_v7_expert_base.py"
    assert _semantic_python_boundary(provider, "forward") == "provider.forward"
    assert _semantic_python_boundary("/site-packages/torch/_ops.py", "__call__") is None
    assert _semantic_line_boundary(provider, "forward", 302) == (
        "route_weight_multiply_inputs", ("routed_output", "route_weight")
    )
    assert _semantic_line_boundary(
        "/site-packages/transformers/integrations/moe.py", "grouped_mm_experts_forward", 465,
    ) == ("route_weight_multiply_inputs", ("proj_out", "sample_weights_g"))

    path = tmp_path / "compact.jsonl"
    trace = FullCallTreeTrace(
        torch.nn.Identity(),
        path,
        rail="compact_fixture",
        basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
    ).start()
    trace._event({
        "kind": "torch_dispatch",
        "operator": "aten.mul.Tensor",
        "callsite": {"file": provider, "function": "forward", "line": 302},
        "semantic_boundary": "route_weight_multiply",
        "inputs": [{"shape": [38, 4096], "dtype": "torch.bfloat16", "device": "cuda:0", "stride": [4096, 1]}],
        "outputs": [{"shape": [38, 4096], "dtype": "torch.bfloat16", "device": "cuda:0", "stride": [4096, 1]}],
    })
    terminal = trace.stop()
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["kind"] for row in rows] == ["header", "torch_dispatch", "footer"]
    assert rows[1]["semantic_boundary"] == "route_weight_multiply"
    assert rows[-1]["status"] == "PASS"
    assert terminal["event_count"] == 3
    assert terminal["event_count"] < 100


def test_named_boundary_fingerprint_reproduces_known_route_weight_dtype_and_hash_difference() -> None:
    down = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    bf16_weight = torch.tensor([0.5, 0.25, 0.75, 1.0], dtype=torch.bfloat16).reshape(-1, 1)
    fp32_weight = bf16_weight.float()
    accepted = down * bf16_weight
    product = down.float() * fp32_weight

    accepted_meta = _tensor_metadata(accepted)
    product_meta = _tensor_metadata(product)

    assert accepted_meta["dtype"] == "torch.bfloat16"
    assert product_meta["dtype"] == "torch.float32"
    assert accepted_meta["sample_sha256"] != product_meta["sample_sha256"]
    assert accepted_meta["sample_numel"] == product_meta["sample_numel"] == 16


def test_generic_torch_dispatch_noise_is_not_recorded(tmp_path) -> None:
    path = tmp_path / "named-only.jsonl"
    with FullCallTreeTrace(
        torch.nn.Identity(), path, rail="named_only", basis_sha256="b" * 64,
        canonical_code_commit="c" * 40,
    ):
        _ = torch.ones((1, 4096)) + 1
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["header", "footer"]


def test_named_source_fixture_records_bounded_route_weight_hash_difference(tmp_path) -> None:
    lines = [""] * 314
    lines[295] = "def forward(routed_output, route_weight, hidden_states, top_k_index):"
    lines[296] = "    marker = 0"
    lines[301] = "    routed_output = routed_output * route_weight.to(dtype=routed_output.dtype)"
    lines[302] = "    marker += 1"
    lines[308] = "    final = routed_output.view(hidden_states.shape[0], top_k_index.shape[1], hidden_states.shape[-1]).sum(dim=1)"
    lines[311] = "    marker += 1"
    lines[313] = "    return final.to(hidden_states.dtype)"
    namespace = {}
    exec(compile("\n".join(lines), "/sealed/fast_v7_expert_base.py", "exec"), namespace)
    fixture = namespace["forward"]

    def capture(label, routed_output, route_weight):
        path = tmp_path / f"{label}.jsonl"
        with FullCallTreeTrace(
            torch.nn.Identity(), path, rail=label, basis_sha256="b" * 64,
            canonical_code_commit="c" * 40,
        ):
            fixture(
                routed_output, route_weight,
                torch.zeros((2, 4), dtype=torch.bfloat16),
                torch.zeros((2, 2), dtype=torch.int64),
            )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) <= 8
        return {row.get("semantic_boundary"): row for row in rows}

    down = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    weight = torch.tensor([0.5, 0.25, 0.75, 1.0], dtype=torch.bfloat16).reshape(-1, 1)
    accepted = capture("accepted", down, weight)
    product = capture("product", down.float(), weight.float())

    accepted_route = accepted["route_weight_multiply_output"]["tensors"][0]
    product_route = product["route_weight_multiply_output"]["tensors"][0]
    assert accepted_route["dtype"] == "torch.bfloat16"
    assert product_route["dtype"] == "torch.float32"
    assert accepted_route["sample_sha256"] != product_route["sample_sha256"]
