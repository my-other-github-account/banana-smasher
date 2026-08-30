from pathlib import Path


def test_official_expert_projection_matches_builder_bf16_linear():
    source = (Path(__file__).parents[1] / "runner/joint_v7_expert_base.py").read_text()
    start = source.index("    def _forward(self, hidden: torch.Tensor)")
    end = source.index("\n    def forward(self, hidden: torch.Tensor)", start)
    body = source[start:end]

    assert "weight = self._weight()" in body
    assert "F.linear(hidden.to(torch.bfloat16), weight).float()" in body
    assert "hidden.float()" not in body
    assert "torch.matmul" not in body
