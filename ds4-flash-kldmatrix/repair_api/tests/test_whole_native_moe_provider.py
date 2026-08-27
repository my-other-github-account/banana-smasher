from types import SimpleNamespace

import torch
import torch.nn.functional as F


def test_native_provider_binding_replaces_the_complete_custom_forward(monkeypatch):
    import repair_api.modern_green_resident as resident

    calls = []
    monkeypatch.setattr(
        resident,
        "_sealed_builder_native_moe_return",
        lambda module, hidden, indices, weights, *, full_weight_builder: calls.append(
            (module, hidden, indices, weights, full_weight_builder)
        ) or "native-return",
    )

    class FrozenCustomProvider:
        def forward(self, *_args):
            return "custom-return"

    builder = lambda *_args: None
    bound = resident._bind_sealed_native_moe_return_provider(
        FrozenCustomProvider,
        {
            "resident_native_moe_return": "accepted_deepseek_v4_expert_loop_v1",
            "resident_native_moe_provider_sha256": resident.ACCEPTED_ROUTED_RETURN_PROVIDER_SHA256,
        },
        full_weight_builder=builder,
    )
    instance = bound()
    args = (object(), object(), object())
    assert instance.forward(*args) == "native-return"
    assert calls == [(instance, *args, builder)]


def test_whole_native_provider_matches_accepted_expert_loop_byte_exact():
    from repair_api.modern_green_resident import _sealed_builder_native_moe_return

    torch.manual_seed(5947)
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    top_k_index = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int64)
    top_k_weights = torch.tensor(
        [[0.625, 0.375], [0.5, 0.5], [0.25, 0.75]], dtype=torch.float32
    )
    gate_up = torch.randn(2, 6, 4, dtype=torch.bfloat16)
    down = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    module = SimpleNamespace(
        num_experts=2,
        intermediate_dim=3,
        packed_w1=torch.tensor([10, 11]),
        packed_w3=torch.tensor([20, 21]),
        packed_w2=torch.tensor([30, 31]),
        su_w1=torch.zeros(2, 1), sv_w1=torch.zeros(2, 1),
        su_w3=torch.zeros(2, 1), sv_w3=torch.zeros(2, 1),
        su_w2=torch.zeros(2, 1), sv_w2=torch.zeros(2, 1),
        plane_source=SimpleNamespace(wire_lut=lambda: torch.zeros(1)),
        act=F.silu,
        limit=100.0,
    )

    def build(packed, _lut, _su, _sv):
        code = int(packed.item())
        expert = code % 10
        if code // 10 == 1:
            return gate_up[expert, :3].transpose(0, 1).contiguous()
        if code // 10 == 2:
            return gate_up[expert, 3:].transpose(0, 1).contiguous()
        return down[expert].transpose(0, 1).contiguous()

    expected = torch.zeros_like(hidden)
    mask = F.one_hot(top_k_index, num_classes=2).permute(2, 1, 0)
    for expert_row in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        expert = expert_row[0]
        slot, token = torch.where(mask[expert])
        routed = hidden[token]
        gate, up = F.linear(routed, gate_up[expert]).chunk(2, dim=-1)
        current = module.act(gate.clamp(max=module.limit)) * up.clamp(
            min=-module.limit, max=module.limit
        )
        current = F.linear(current, down[expert]) * top_k_weights[token, slot, None]
        expected.index_add_(0, token, current.to(expected.dtype))

    observed = _sealed_builder_native_moe_return(
        module, hidden, top_k_index, top_k_weights, full_weight_builder=build
    )
    assert observed.dtype == hidden.dtype
    assert torch.equal(observed, expected)
