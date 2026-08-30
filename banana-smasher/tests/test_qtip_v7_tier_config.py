from __future__ import annotations

import inspect

import pytest

from banana_smasher import qtip25_native_v4
from banana_smasher import qtip25_native_v4_api
from banana_smasher.qtip3_api_producer import Qtip3ApiConfig, _valid_cuda_receipt


@pytest.mark.parametrize(
    ("bpw", "provider", "geometry"),
    [
        (1.0, "qtip-native-v6@1.00", (4, 16, 4)),
        (3.0, "qtip-native-v6@3.00", (12, 16, 4)),
        (4.0, "qtip-native-v6@4.00", (16, 16, 4)),
    ],
)
def test_v7_tier_config_resolves_q1_q3_q4(
    bpw: float, provider: str, geometry: tuple[int, int, int]
) -> None:
    config = Qtip3ApiConfig.for_bpw(bpw)
    assert config.bpw == bpw
    assert config.provider == provider
    assert config.geometry == geometry

    _valid_cuda_receipt(
        {
            "status": "PASS",
            "backend": "cuda",
            "codec_version": "v6",
            "provider": provider,
            "geometry": {"B": geometry[0], "L": geometry[1], "V": geometry[2]},
            "installed_cuda_decode": {
                "counters": {"cuda_decode_calls": 1, "fallback_calls": 0}
            },
        },
        config,
    )


def test_v7_tier_config_refuses_non_ladder_rate() -> None:
    with pytest.raises(ValueError, match="QTIP V7 ladder"):
        Qtip3ApiConfig.for_bpw(2.5)


def test_cross_cell_api_emits_rate_derived_provider_and_accounting() -> None:
    source = inspect.getsource(qtip25_native_v4_api.build_qtip_native_cells)
    assert 'f"qtip-native-v6@{rate:.2f}"' in source
    assert '"exact_code_bpw": rate' in source
    assert "(geometry.B, geometry.L, geometry.V) not in {(4, 16, 4), (12, 16, 4), (16, 16, 4)}" in source


def test_cuda_kernel_source_is_specialized_for_q1_q3_q4_geometry() -> None:
    for bpw, branch_bits, prefixes in ((1.0, 4, 4096), (3.0, 12, 16), (4.0, 16, 1)):
        geometry = qtip25_native_v4.native_v4_geometry(bpw)
        source = qtip25_native_v4._native_v4_cuda_source(geometry)
        assert f"constexpr int PREFIXES = {prefixes};" in source
        assert f"constexpr int BRANCH_BITS = {branch_bits};" in source

    for function in (
        qtip25_native_v4._native_v4_cuda_pass,
        qtip25_native_v4._native_v4_cuda_warmup_overlap,
    ):
        source = inspect.getsource(function)
        assert "_load_native_v4_cuda_extension(geometry)" in source


def test_q1_cuda_kernel_strides_prefixes_beyond_one_block() -> None:
    source = qtip25_native_v4._native_v4_cuda_source(
        qtip25_native_v4.native_v4_geometry(1.0)
    )
    assert "if (PREFIXES > THREADS)" in source
    assert "for (int prefix = tid; prefix < PREFIXES; prefix += THREADS)" in source
    assert "for (int branch = 0; branch < (1 << BRANCH_BITS); ++branch)" in source
    assert source.count("static_cast<int64_t>(step - capture_step)") >= 2


def test_regeneration_entrypoint_uses_one_tier_config_for_admission_smoke_and_run() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1]
        / "src"
        / "banana_smasher"
        / "qtip3_regenerate.py"
    ).read_text()
    assert 'TIER_CONFIG = Qtip3ApiConfig.for_bpw(float(os.environ.get("QTIP3_BPW", "3.0")))' in source
    assert "admit_host_and_shard(new_plan, gpu_probe=gpu_probe, config=TIER_CONFIG)" in source
    assert "config = TIER_CONFIG" in source
    assert "bpw=TIER_CONFIG.bpw" in source
