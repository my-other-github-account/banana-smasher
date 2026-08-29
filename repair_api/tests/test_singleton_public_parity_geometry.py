import pytest

from repair_api.resident_full64_accept import _admit_initial_w28_geometry


def test_public_parity_diagnostic_admits_explicit_singleton_only() -> None:
    config = {
        "score_window_batch_size": 1,
        "sealed_builder_window_microbatch": 1,
    }

    status = _admit_initial_w28_geometry(
        config,
        singleton_public_parity_tap_only=True,
    )

    assert status == "SINGLETON_PUBLIC_PARITY_TAP_ONLY"


@pytest.mark.parametrize(
    "config",
    [
        {"score_window_batch_size": 2, "sealed_builder_window_microbatch": 1},
        {"score_window_batch_size": 1, "sealed_builder_window_microbatch": 2},
    ],
)
def test_public_parity_diagnostic_rejects_non_singleton_geometry(config) -> None:
    with pytest.raises(RuntimeError, match="PARITY_TAP_REQUIRES_SINGLETON_GEOMETRY"):
        _admit_initial_w28_geometry(
            config,
            singleton_public_parity_tap_only=True,
        )


def test_full64_path_still_rejects_singleton_geometry() -> None:
    config = {
        "score_window_batch_size": 1,
        "sealed_builder_window_microbatch": 1,
    }

    with pytest.raises(RuntimeError, match="FULL64_REQUIRES_ADMITTED_BATCH_GEOMETRY"):
        _admit_initial_w28_geometry(
            config,
            singleton_public_parity_tap_only=False,
        )
