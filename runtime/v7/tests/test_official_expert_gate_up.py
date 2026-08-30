from pathlib import Path


def test_official_expert_gate_up_matches_native_fused_clamped_path():
    source = (Path(__file__).parents[1] / "runner/joint_v7_expert_base.py").read_text()
    start = source.index("    def _gate_up(self, hidden: torch.Tensor")
    end = source.index("\n    def forward(self, hidden_states", start)
    body = source[start:end]

    assert "torch.cat((w1._weight(), w3._weight()), dim=0)" in body
    assert "F.linear(hidden.to(torch.bfloat16), gate_up_weight)" in body
    assert "gate, up = gate_up.chunk(2, dim=-1)" in body
    assert "gate = gate.clamp(max=self.swiglu_limit)" in body
    assert "up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)" in body
