from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from banana_smasher.resident_terminal_scorer import (
    ResidentScoreAccumulator,
    score_terminal_hidden,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def _stamp(callable_):
    _sync()
    started = time.perf_counter()
    with torch.no_grad():
        value = callable_()
    _sync()
    return value, time.perf_counter() - started


def _metric(idx, teacher_lp, q_lp, q_argmax):
    ref = teacher_lp.float()
    cand = q_lp.float()
    lp_n = ref - ref.logsumexp(-1, keepdim=True)
    lq_n = cand - cand.logsumexp(-1, keepdim=True)
    kld = (lp_n.exp() * (lp_n - lq_n)).sum(-1)
    return float(kld.sum(dtype=torch.float64)), int((q_argmax.long() == idx[:, 0].long()).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(7)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    windows, positions, hidden_width, vocab, support_width = 4, 128, 256, 4096, 512
    forward = torch.nn.Linear(hidden_width, hidden_width, bias=False, device=device)
    head = torch.nn.Linear(hidden_width, vocab, bias=False, device=device)
    forward.requires_grad_(False)
    head.requires_grad_(False)
    inputs = [torch.randn(positions, hidden_width, device=device) for _ in range(windows)]
    teacher_idx = [
        torch.randint(vocab, (positions, support_width), dtype=torch.int64, device=device)
        for _ in range(windows)
    ]
    teacher_lp = [
        torch.randn(positions, support_width, dtype=torch.float16, device=device)
        for _ in range(windows)
    ]
    rows = {"baseline": [], "candidate": []}
    equality = {"kld_max_abs": float("inf"), "top1_exact": False}
    for iteration in range(args.repeats + 2):
        baseline_parts = {"forward": 0.0, "head_log_softmax": 0.0, "d2h_reduction": 0.0, "allocator_cache": 0.0}
        candidate_parts = dict(baseline_parts)
        baseline_result = []
        candidate_result = []
        baseline_total = time.perf_counter()
        for slot in range(windows):
            hidden, elapsed = _stamp(lambda slot=slot: forward(inputs[slot]))
            baseline_parts["forward"] += elapsed
            chunks = []
            argmax = []
            for start in range(0, positions, 32):
                head_started = time.perf_counter()
                logits = head(hidden[start : start + 32]).float()
                gathered = logits.gather(1, teacher_idx[slot][start : start + 32]).half()
                predicted = logits.argmax(-1).int()
                _sync()
                baseline_parts["head_log_softmax"] += time.perf_counter() - head_started
                transfer_started = time.perf_counter()
                chunks.append(gathered.cpu())
                argmax.append(predicted.cpu())
                baseline_parts["d2h_reduction"] += time.perf_counter() - transfer_started
            reduction_started = time.perf_counter()
            q_lp = torch.cat(chunks)
            q_argmax = torch.cat(argmax)
            baseline_result.append(_metric(teacher_idx[slot].cpu(), teacher_lp[slot].cpu(), q_lp, q_argmax))
            baseline_parts["d2h_reduction"] += time.perf_counter() - reduction_started
            allocator_started = time.perf_counter()
            del chunks, argmax, q_lp, q_argmax, hidden
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            baseline_parts["allocator_cache"] += time.perf_counter() - allocator_started
        baseline_wall = time.perf_counter() - baseline_total

        accumulator = ResidentScoreAccumulator(torch)
        q_lp_out = torch.empty((positions, support_width), dtype=torch.float16, device=device)
        q_argmax_out = torch.empty((positions,), dtype=torch.int32, device=device)
        candidate_total = time.perf_counter()
        for slot in range(windows):
            hidden, elapsed = _stamp(lambda slot=slot: forward(inputs[slot]))
            candidate_parts["forward"] += elapsed
            (q_lp, q_argmax), elapsed = _stamp(
                lambda slot=slot, hidden=hidden: score_terminal_hidden(
                    hidden,
                    teacher_idx[slot],
                    head,
                    chunk_size=32,
                    q_lp_out=q_lp_out,
                    q_argmax_out=q_argmax_out,
                    compute_dtype=torch.float32,
                )
            )
            candidate_parts["head_log_softmax"] += elapsed
            reduction_started = time.perf_counter()
            accumulator.add(str(slot), teacher_idx[slot], teacher_lp[slot], q_lp, q_argmax)
            candidate_parts["d2h_reduction"] += time.perf_counter() - reduction_started
            del hidden
        finalize_started = time.perf_counter()
        finalized = accumulator.finalize()
        candidate_parts["d2h_reduction"] += time.perf_counter() - finalize_started
        candidate_wall = time.perf_counter() - candidate_total
        candidate_result = [(row["kld_sum"], row["top1_matches"]) for row in finalized["per_window"]]
        equality = {
            "kld_max_abs": max(abs(a[0] - b[0]) for a, b in zip(baseline_result, candidate_result, strict=True)),
            "top1_exact": [a[1] for a in baseline_result] == [b[1] for b in candidate_result],
        }
        if iteration < 2:
            continue
        rows["baseline"].append({"wall": baseline_wall, **baseline_parts})
        rows["candidate"].append({"wall": candidate_wall, **candidate_parts})

    medians = {
        arm: {key: statistics.median(row[key] for row in arm_rows) for key in arm_rows[0]}
        for arm, arm_rows in rows.items()
    }
    source = Path(__file__).parents[1] / "src/banana_smasher/resident_terminal_scorer.py"
    kld_tolerance = 1e-4
    receipt = {
        "schema": "banana-smasher-resident-terminal-scorer-microbench-v1",
        "status": "PASS" if equality["top1_exact"] and equality["kld_max_abs"] <= kld_tolerance else "FAIL",
        "surface": "claim-free-code-microbench",
        "device": str(device),
        "same_work": {"windows": windows, "positions": positions, "hidden_width": hidden_width, "vocab": vocab, "support_width": support_width, "repeats": args.repeats},
        "decomposition_seconds_median": medians,
        "speedup": medians["baseline"]["wall"] / medians["candidate"]["wall"],
        "paired_metrics": equality,
        "paired_kld_abs_tolerance": kld_tolerance,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cache_note": "baseline allocator_cache includes per-window empty_cache when the accelerator exposes it",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
