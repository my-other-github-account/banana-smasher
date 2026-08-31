from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from banana_smasher import qtip25_native_v4
from banana_smasher import qtip25_native_v4_api
from banana_smasher import qtip3_api_producer as producer_module
from banana_smasher.qtip3_api_producer import (
    CellSpec,
    Qtip3ApiConfig,
    Qtip3ApiPlan,
    _valid_cuda_receipt,
    build_clean102_option_row,
    run_cells_batched,
)


@pytest.mark.parametrize(
    ("bpw", "tier", "provider", "geometry"),
    [
        (1.0, "qtip1_v7", "qtip-native-v6@1.00", (4, 16, 4)),
        (3.0, "qtip3_v7", "qtip-native-v6@3.00", (12, 16, 4)),
        (4.0, "qtip4_v7", "qtip-native-v6@4.00", (16, 16, 4)),
    ],
)
def test_v7_tier_config_resolves_q1_q3_q4(
    bpw: float, tier: str, provider: str, geometry: tuple[int, int, int]
) -> None:
    config = Qtip3ApiConfig.for_bpw(bpw)
    assert config.bpw == bpw
    assert config.tier == tier
    assert config.provider == provider
    assert config.geometry == geometry
    assert Qtip3ApiConfig.for_tier(tier) == config

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
    with pytest.raises(ValueError, match="tier must be"):
        Qtip3ApiConfig.for_tier("qtip4")
    with pytest.raises(ValueError, match="tier/provider/geometry"):
        Qtip3ApiConfig(bpw=1.0, tier="qtip4_v7", provider="qtip-native-v6@1.00", geometry=(4, 16, 4))


@pytest.mark.parametrize("tier", ["qtip1_v7", "qtip4_v7"])
def test_clean102_row_is_sha_bound_to_public_receipt(tmp_path, monkeypatch, tier: str) -> None:
    monkeypatch.setattr(producer_module, "LAYERS", (2,))
    config = Qtip3ApiConfig.for_tier(tier)
    basis_file = tmp_path / "model.index"
    basis_file.write_bytes(b"representative DeepSeek-V4-Flash-0731 index")
    basis = hashlib.sha256(basis_file.read_bytes()).hexdigest()
    authority = tmp_path / "authority"
    allocation = "HOST_ALLOCATION t_dry spark-dry qtip-v7-dry-run"
    authority.write_text(allocation + "\n")
    source = tmp_path / "source.npy"
    source.write_bytes(b"representative-source")
    control = tmp_path / "control.pt"
    control.write_bytes(b"representative-control")
    tlut = tmp_path / "qtip_tlut.npy"
    tlut.write_bytes(b"representative-tlut")
    mission = tmp_path / "mission"
    (mission / "receipts").mkdir(parents=True)
    (mission / "receipts" / "ADMISSION.json").write_text("{}\n")
    output = mission / "outputs" / "full_api" / "L002_E007_down"
    plan = Qtip3ApiPlan(
        task_id="t_dry", board_run_id=1, host="spark-dry", allocation=allocation,
        intended_basis_sha256=basis, driver_goals_path=authority,
        driver_goals_sha256=hashlib.sha256(authority.read_bytes()).hexdigest(),
        claim_path=tmp_path / "claim.json", shards_path=mission / "SHARDS.json",
        mission_root=mission, model_index_path=basis_file, tlut_path=tlut,
        layers=(2,), cell_roster=((2, 7, "down"),),
    )
    cell = CellSpec(layer=2, expert=7, projection="down", source=source,
                    control=control, output=output)
    calls = []

    def dry_batch(rows, *_args, **kwargs):
        calls.append((len(rows), kwargs["bpw"]))
        result = []
        for row in rows:
            row_output = __import__("pathlib").Path(row["output"])
            row_output.mkdir(parents=True, exist_ok=True)
            (row_output / "codes.npy").write_bytes(b"canonical-code-plane")
            api_receipt = row_output / "CELL_RECEIPT.json"
            value = {
                "status": "PASS", "backend": "cuda", "codec_version": "v6",
                "provider": config.provider, "geometry": {
                    "B": config.geometry[0], "L": config.geometry[1], "V": config.geometry[2]
                }, "installed_cuda_decode": {
                    "counters": {"cuda_decode_calls": 1, "fallback_calls": 0}
                }, "receipt": str(api_receipt), "receipt_sha256": "e" * 64,
            }
            api_receipt.write_text(json.dumps(value, sort_keys=True) + "\n")
            result.append(value)
        return result

    terminal = run_cells_batched(plan, config, [cell], batch_api=dry_batch, batch_size=1)
    assert terminal["status"] == "PASS"
    assert terminal["provider"] == config.provider
    assert calls == [(1, config.bpw)]
    receipt_path = output / "PUBLIC_CELL_RECEIPT.json"
    assert (output / "codes.npy").is_file()
    assert (output / "CELL_RECEIPT.json").is_file()
    assert receipt_path.is_file()
    assert (mission / "receipts" / "PRODUCER_TERMINAL.json").is_file()
    resumed = run_cells_batched(
        plan, config, [cell], batch_api=lambda *_args, **_kwargs: pytest.fail("resume replayed")
    )
    assert resumed["new_cells"] == 0
    prediction_receipt = mission / "CLEAN102_PREDICTION.json"
    prediction_receipt.write_text(json.dumps({
        "schema": "clean102-prediction-v1", "status": "PASS",
        "basis_sha256": basis, "cell_id": "L002:E007:down",
    }, sort_keys=True) + "\n")
    row = build_clean102_option_row(
        receipt_path, prediction_receipt_path=prediction_receipt, config=config,
        physical_bytes=(output / "codes.npy").stat().st_size,
        prediction_by_class={"chat": 0.125, "reasoning": 0.25},
        bank_sha256="b" * 64, teacher_sha256="c" * 64, scorer_sha256="d" * 64,
    )
    assert row["schema"] == "banana-smasher-provenance-option-row-v1"
    assert row["model_id"] == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert row["cell_id"] == "L002:E007:down"
    assert row["tier"] == tier
    assert row["activation_ids"] == []
    assert row["physical_producer"]["sha256"] != row["prediction_producer"]["sha256"]
    assert row["physical_producer"]["artifact_sha256"] == hashlib.sha256(
        (output / "codes.npy").read_bytes()
    ).hexdigest()


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
    assert 'TIER_CONFIG = Qtip3ApiConfig.for_tier(os.environ.get("QTIP3_TIER", "qtip3_v7"))' in source
    assert "QTIP3_TIER and QTIP3_BPW select inconsistent V7 geometry" in source
    assert "admit_host_and_shard(new_plan, gpu_probe=gpu_probe, config=TIER_CONFIG)" in source
    assert "config = TIER_CONFIG" in source
    assert "bpw=TIER_CONFIG.bpw" in source
