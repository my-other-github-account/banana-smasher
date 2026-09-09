"""Avoid optional historical-output advice when live capacity already passes.

This is NOT an artifact validator, claim gate, or replacement for per-batch
memory admission. Callers own exclusive-seat checks, authentic input/output
validation and a bounded reclaim callback; no filesystem scan is done here.
"""
from collections.abc import Callable
import time


def capacity_advice(
    probe_free_bytes: Callable[[], int],
    peak_estimate_bytes: int,
    reserve_bytes: int,
    reclaim: Callable[[], None],
) -> dict:
    """Probe, optionally advise once, then independently enforce the same gate.

    The strict free > estimate + reserve comparison matches producer admission.
    Probe/callback failures propagate; there is no weaker or slower fallback.
    Persist the returned receipt and retain the ordinary per-batch gate.
    """
    if any(type(value) is not int or value <= 0
           for value in (peak_estimate_bytes, reserve_bytes)):
        raise ValueError('capacity estimate and reserve must be positive integer bytes')
    start = time.perf_counter()
    before = probe_free_bytes()
    if type(before) is not int or before < 0:
        raise ValueError('capacity probe must return nonnegative integer bytes')
    action = 'skipped_sufficient_capacity'
    if before <= peak_estimate_bytes + reserve_bytes:
        reclaim()
        action = 'advised_once'
    after = probe_free_bytes()
    if type(after) is not int or after < 0:
        raise ValueError('capacity probe must return nonnegative integer bytes')
    if after <= peak_estimate_bytes + reserve_bytes:
        raise RuntimeError('CAPACITY_REFUSAL_AFTER_ADVICE_GATE')
    return dict(schema='banana-smasher-capacity-advice-v1', action=action,
                before_free_bytes=before, after_free_bytes=after,
                peak_estimate_bytes=peak_estimate_bytes, reserve_bytes=reserve_bytes,
                elapsed_seconds=time.perf_counter() - start)
