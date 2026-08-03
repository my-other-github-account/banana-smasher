from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .token_sizing import require_integer


def backward_logical_mean(loss_sum: Any, logical_items: int) -> None:
    """Backpropagate a scalar summed loss against the full logical item count."""
    logical_items = require_integer("logical_items", logical_items)
    if logical_items <= 0:
        raise ValueError(f"logical_items must be positive, got {logical_items}")
    if getattr(loss_sum, "ndim", None) != 0:
        raise ValueError("loss_sum must return a scalar summed loss")
    (loss_sum / logical_items).backward()


def exact_accumulation_step(
    *,
    optimizer: Any,
    segments: Sequence[Any],
    item_count: Callable[[Any], int],
    loss_sum: Callable[[Any], Any],
) -> dict[str, Any]:
    """Apply one exact logical-mean update over one or more physical segments.

    ``loss_sum`` returns a summed scalar for each segment. Dividing every
    backward root by the total logical extent makes unequal slices exact.
    Exactly one optimizer mutation is made after every backward succeeds.
    """
    work = list(segments)
    counts = [
        require_integer(f"segment_items[{index}]", item_count(segment))
        for index, segment in enumerate(work)
    ]
    if not counts or any(count <= 0 for count in counts):
        raise ValueError(f"accumulation segments must be non-empty, got {counts}")

    logical_items = sum(counts)
    optimizer.zero_grad(set_to_none=True)
    detached_loss_sum = 0.0
    for segment in work:
        current = loss_sum(segment)
        backward_logical_mean(current, logical_items)
        detached_loss_sum += float(current.detach())
    optimizer.step()
    return {
        "segments": len(work),
        "segment_items": counts,
        "logical_items": logical_items,
        "logical_mean_loss": detached_loss_sum / logical_items,
        "forward_count": len(work),
        "backward_count": len(work),
        "optimizer_steps": 1,
    }
