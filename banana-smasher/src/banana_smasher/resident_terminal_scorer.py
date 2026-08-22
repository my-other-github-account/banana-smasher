from __future__ import annotations

import math
from typing import Any


def score_terminal_hidden(
    hidden: Any,
    support_token_ids: Any,
    lm_head: Any,
    *,
    chunk_size: int = 128,
    q_lp_out: Any | None = None,
    q_argmax_out: Any | None = None,
    compute_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """Score one terminal activation without synchronizing device outputs.

    The caller may retain and pass the output buffers across windows.  This
    keeps head/logit chunks temporary while q_lp and q_argmax remain resident
    until the caller deliberately persists or reduces them.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    torch = __import__("torch")
    positions = int(hidden.shape[0])
    support_width = int(support_token_ids.shape[1])
    if tuple(support_token_ids.shape) != (positions, support_width):
        raise ValueError("terminal support rows must match hidden positions")
    device = hidden.device
    if q_lp_out is None:
        q_lp_out = torch.empty(
            (positions, support_width), dtype=torch.float16, device=device
        )
    if q_argmax_out is None:
        q_argmax_out = torch.empty((positions,), dtype=torch.int32, device=device)
    if tuple(q_lp_out.shape) != (positions, support_width):
        raise ValueError("q_lp output buffer shape mismatch")
    if tuple(q_argmax_out.shape) != (positions,):
        raise ValueError("q_argmax output buffer shape mismatch")
    if q_lp_out.device != device or q_argmax_out.device != device:
        raise ValueError("terminal output buffers must share the hidden device")

    for start in range(0, positions, chunk_size):
        stop = min(start + chunk_size, positions)
        chunk = hidden[start:stop]
        if compute_dtype is not None:
            chunk = chunk.to(compute_dtype)
        logits = lm_head(chunk).float()
        support = support_token_ids[start:stop].long()
        q_lp_out[start:stop].copy_(logits.gather(1, support))
        q_argmax_out[start:stop].copy_(logits.argmax(-1))
    return q_lp_out, q_argmax_out


class ResidentScoreAccumulator:
    """Reduce paired KLD/Top-1 on-device and finalize through one D2H copy."""

    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self._window_ids: list[object] = []
        self._rows: list[Any] = []

    def add(
        self,
        window_id: object,
        teacher_idx: Any,
        teacher_logprob: Any,
        q_lp_at_ref: Any,
        q_argmax: Any,
    ) -> None:
        count = min(
            int(teacher_idx.shape[0]),
            int(teacher_logprob.shape[0]),
            int(q_lp_at_ref.shape[0]),
            int(q_argmax.shape[0]),
        )
        if count < 1 or teacher_logprob.shape[1] != q_lp_at_ref.shape[1]:
            raise ValueError("candidate/teacher terminal score shape mismatch")
        ref = teacher_logprob[:count].float()
        cand = q_lp_at_ref[:count].float()
        lp_n = ref - ref.logsumexp(-1, keepdim=True)
        lq_n = cand - cand.logsumexp(-1, keepdim=True)
        kld = (lp_n.exp() * (lp_n - lq_n)).sum(-1)
        matches = q_argmax[:count].long() == teacher_idx[:count, 0].long()
        row = self.torch.stack(
            (
                kld.sum(dtype=self.torch.float32),
                matches.sum(dtype=self.torch.float32),
                self.torch.as_tensor(count, dtype=self.torch.float32, device=kld.device),
                self.torch.isfinite(kld).all().to(dtype=self.torch.float32),
            )
        )
        self._window_ids.append(window_id)
        self._rows.append(row)

    def finalize(self) -> dict[str, Any]:
        if not self._rows:
            raise ValueError("cannot finalize an empty terminal score")
        rows = (
            self.torch.stack(self._rows)
            .to(device="cpu")
            .to(dtype=self.torch.float64)
            .numpy()
        )
        if not all(row[3] == 1.0 for row in rows):
            raise ValueError("terminal KLD is non-finite")
        per_window = []
        for window_id, row in zip(self._window_ids, rows, strict=True):
            kld_sum = float(row[0])
            matches = int(row[1])
            positions = int(row[2])
            per_window.append(
                {
                    "window_id": window_id,
                    "positions": positions,
                    "kld_sum": kld_sum,
                    "mean_kld": kld_sum / positions,
                    "top1_matches": matches,
                    "top1_agreement": matches / positions,
                }
            )
        kld_sum = math.fsum(row["kld_sum"] for row in per_window)
        positions = sum(row["positions"] for row in per_window)
        top1_matches = sum(row["top1_matches"] for row in per_window)
        return {
            "schema": "banana-smasher-anchor-sidecar-score-v1",
            "status": "PASS",
            "positions": positions,
            "kld_sum": kld_sum,
            "mean_kld": kld_sum / positions,
            "top1_matches": top1_matches,
            "top1_agreement": top1_matches / positions,
            "per_window": per_window,
        }
