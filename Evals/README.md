# DeepSeek-V4-Flash-0731 quant results

This page compares quality and size for four 0731 quants. Exact receipts and the
full protocol are linked below the readable summary.

## Results

Every model below ran the same 64 windows and 65,536 scored positions against the
same FP8 copy of DeepSeek-V4-Flash-0731.

| Quant | Top-1 ↑ | KLD ↓ | Size | bpw | FP |
|---|---:|---:|---:|---:|---|
| **Unsloth IQ4** | **92.44%** | **0.068** | 136.7 GB | 3.85 | FP8 e4m3 own-base |
| **Unsloth IQ3** | **87.95%** | **0.178** | 104.2 GB | 2.93 | FP8 e4m3 own-base |
| **Unsloth IQ2** | **84.57%** | **0.277** | 90.9 GB | 2.56 | FP8 e4m3 own-base |
| **DwarfStar Q2** | **83.69%** | **0.310** | 93.7 GB | 2.64 | FP8 e4m3 own-base |

IQ4 keeps the most quality. IQ3 is the middle option. IQ2 is the strongest
compact result here: it is smaller than DwarfStar and scores better on both
quality metrics.

Top-1 is the easiest number to read: it is how often the quant picks the same
next token as FP8. Higher is better. KLD measures how much the full token
probability distribution moved; lower is better.

The table is rounded for humans. The [machine-readable result](results/deepseek-v4-flash-0731-balanced64-v1.json)
contains the exact bytes, ratios, KLD values, Top-1 matches, and full decimals.
All four measurements are complete. Artifact download metadata in the JSON is
about future replay, not measurement completeness.

## What makes these apples to apples

- Same model family: DeepSeek-V4-Flash-0731
- Same 64 ordered windows
- Same 1,024 positions per window, 65,536 total
- Same FP8 e4m3 own-base teacher
- Same teacher top-8,192 token support
- Same KLD and Top-1 definitions
- Same packed-wire denominator: 284,334,567,511 parameters

The class mix is agentic/chat/code/multilingual/prose/reasoning =
`19/7/9/10/10/9`.

## Adding Banana Smasher

There is no Banana Smasher row yet. We will add it when the final pack exists and
has passed this exact test. Internal `train_balanced64` Anchor scores do not count
because they use different windows.

The admission steps are:

1. Freeze the final candidate and count every byte shipped with it.
2. Run the exact 64 windows from the [BALANCED64 lock](configs/balanced64-v1.json).
3. Score the candidate against the FP8 own-base teacher on the same 65,536 positions.
4. Save all 64 per-window receipts and aggregate them with the repository tool.
5. Add the row only when Top-1, KLD, GB, packed-wire bpw, FP, and all 64 receipts are complete.

No partial run, different window bank, fallback output, or HOLDOUT result can be
substituted into this table.

## Check the published result

From the repository root:

```bash
python3 -m Evals.tools.receipts verify \
  Evals/results/deepseek-v4-flash-0731-balanced64-v1.json \
  --suite-lock Evals/configs/balanced64-v1.json
```

Expected order for both metrics:

```text
IQ4 > IQ3 > IQ2 > DwarfStar
```

## Aggregate a new 64-window result

Put the 64 completed row receipts in one directory, then run:

```bash
python3 -m Evals.tools.receipts aggregate work/balanced64-windows \
  --suite-lock Evals/configs/balanced64-v1.json \
  --output work/balanced64-aggregate.json
```

The aggregator rejects missing or duplicate windows, changed classes, wrong
position counts, negative or non-finite KLD, basis drift, and invalid Top-1
counts.

## Files

- [Exact result JSON](results/deepseek-v4-flash-0731-balanced64-v1.json)
- [Frozen BALANCED64 lock](configs/balanced64-v1.json)
- [Full measurement protocol](protocols/balanced64-v1.md)
- [One-window receipt template](templates/balanced64-window-v1.json)
- [Verifier and aggregator](tools/receipts.py)
- [JSON schemas](schemas/)

The full protocol contains the exact math, reduction order, receipt schema, CLI
producer commands, and replay limits. Most readers only need the table above.
