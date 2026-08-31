from pathlib import Path


def test_static_w28_fills_bounded_host_arena_before_cuda_migration() -> None:
    source = (
        Path(__file__).parents[1]
        / "assets"
        / "static_w28_fast_v7_expert_base.py"
    ).read_text()

    constructor = source[source.index("class FullyResidentGroupedV7Experts"):]
    assert "arena_cpu_tensor = torch.empty(" in constructor
    assert "dtype=torch.int16, pin_memory=True" in constructor
    assert "arena_pointer, arena_owner" not in constructor
    assert "packed_cuda = packed.to(device=device, non_blocking=True)" in source
