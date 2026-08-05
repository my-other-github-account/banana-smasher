# DeepSeek-V4-Flash-0731 quant results

This page compares quality and exact whole-model shipping size for six quants measured on the frozen competitive `BALANCED64_V1` population.

## Results

Every model below ran the same 64 windows and 65,536 scored positions against the same FP8 copy of DeepSeek-V4-Flash-0731. The table is ordered by Top-1 agreement; both global metrics produce the same ranking.

| Quant | Top-1 ↑ | KLD ↓ | Exact decimal GB | Normalized packed-wire bpw | FP basis |
|---|---:|---:|---:|---:|---|
| **Unsloth IQ4** | **92.4438%** (60,584/65,536) | **0.0683488486737012** | 136.662446656 | 3.84511662728346850440505038646609313779223476049666531535789 | FP8 e4m3 dynamic own-base |
| **QTIP3 uniform exact** | **91.6809%** (60,084/65,536) | **0.11022678823825564** | 123.934682354 | 3.48700992465027274894687927721480550531597653683639523743380 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ3** | **87.9486%** (57,638/65,536) | **0.17770788160865483** | 104.207848032 | 2.93197830834883710932601535266166267708804597088615155566779 | FP8 e4m3 dynamic own-base |
| **QTIP2 corrected all-43** | **87.1124%** (57,090/65,536) | **0.24085164613260832** | 89.296314458 | 2.5124293606557819496666946714231865550935692962972950439475838776323831162246700 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ2** | **84.5673%** (55,422/65,536) | **0.2767474104898907** | 90.860736928 | 2.55644574554192780938968595190844480957809362062402409152428 | FP8 e4m3 dynamic own-base |
| **DwarfStar Q2** | **83.6868%** (54,845/65,536) | **0.30952134732070036** | 93.691352992 | 2.63608758687774759058919129311816402968907463449170730541087 | FP8 e4m3 dynamic own-base |

Top-1 is how often the quant selects the same next token as FP8 on the common ordered support. KLD measures movement of the full supported token distribution. Higher Top-1 and lower KLD are better.

The [machine-readable result](results/deepseek-v4-flash-0731-balanced64-v1.json) contains exact bytes, full decimal ratios, candidate/teacher/scorer/population identities, component-byte ledgers, source hashes, replay limits, and the six-category breakdowns for all six quants.

## What makes these apples to apples

- Same model family: DeepSeek-V4-Flash-0731
- Same 64 ordered windows and 65,536 total positions
- Same FP8 e4m3 dynamic own-base teacher
- Same teacher top-8,192 token support
- Same KLD and Top-1 definitions
- Same packed-wire denominator: 284,334,567,511 parameters
- Same corrected class mix: agentic/chat/code/multilingual/prose/reasoning = `19/7/9/10/10/9`

No partial run, different window bank, fallback output, or HOLDOUT result is admitted to this ranking.

## Category breakdowns

These category rows are derived from the exact same 64-window competitive aggregates as the global table. Top-1 percentages are rounded for readability, with exact integer counts beside them. KLD is the position-weighted class mean; exact decimals remain in the machine-readable result.

### Top-1 agreement

| Quant | Agentic | Chat | Code | Multilingual | Prose | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| **Unsloth IQ4** | 92.12% (17,922/19,456) | 94.45% (6,770/7,168) | 94.61% (8,719/9,216) | 89.90% (9,206/10,240) | 89.05% (9,119/10,240) | 96.01% (8,848/9,216) |
| **QTIP3 exact** | 91.75% (17,851/19,456) | 95.37% (6,836/7,168) | 94.34% (8,694/9,216) | 87.47% (8,957/10,240) | 86.48% (8,856/10,240) | 96.46% (8,890/9,216) |
| **Unsloth IQ3** | 87.68% (17,059/19,456) | 91.35% (6,548/7,168) | 91.46% (8,429/9,216) | 83.54% (8,555/10,240) | 82.17% (8,414/10,240) | 93.67% (8,633/9,216) |
| **QTIP2 all-43** | 87.78% (17,079/19,456) | 91.62% (6,567/7,168) | 91.29% (8,413/9,216) | 80.11% (8,203/10,240) | 79.01% (8,091/10,240) | 94.80% (8,737/9,216) |
| **Unsloth IQ2** | 84.43% (16,426/19,456) | 88.85% (6,369/7,168) | 89.08% (8,210/9,216) | 78.76% (8,065/10,240) | 76.81% (7,865/10,240) | 92.09% (8,487/9,216) |
| **DwarfStar Q2** | 83.13% (16,174/19,456) | 88.38% (6,335/7,168) | 88.10% (8,119/9,216) | 77.75% (7,962/10,240) | 76.12% (7,795/10,240) | 91.80% (8,460/9,216) |

### KLD

| Quant | Agentic | Chat | Code | Multilingual | Prose | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| **Unsloth IQ4** | 0.1061 | 0.0256 | 0.0332 | 0.0941 | 0.0823 | 0.0131 |
| **QTIP3 exact** | 0.1513 | 0.0285 | 0.0528 | 0.1792 | 0.1563 | 0.0168 |
| **Unsloth IQ3** | 0.2507 | 0.0736 | 0.0894 | 0.2688 | 0.2279 | 0.0360 |
| **QTIP2 all-43** | 0.2896 | 0.0683 | 0.1223 | 0.4477 | 0.3549 | 0.0341 |
| **Unsloth IQ2** | 0.3770 | 0.1123 | 0.1441 | 0.4302 | 0.3623 | 0.0601 |
| **DwarfStar Q2** | 0.4198 | 0.1250 | 0.1745 | 0.4674 | 0.4150 | 0.0625 |

## Internal anchors are separate

The internal `train_balanced64` QTIP anchors use a different population with **0/64 competitive window-ID overlap**. They are not rows in the table above. Their exact calculations and machine result now live on the [Backpack calculation page](../Backpack/README.md).

## Check the published result

From the repository root:

```bash
python3 -m Evals.tools.receipts verify \
  Evals/results/deepseek-v4-flash-0731-balanced64-v1.json \
  --suite-lock Evals/configs/balanced64-v1.json
```

Expected order for both metrics:

```text
IQ4 > QTIP3 > IQ3 > QTIP2 > IQ2 > DwarfStar
```

## Aggregate a new 64-window result

Put the 64 completed row receipts in one directory, then run:

```bash
python3 -m Evals.tools.receipts aggregate work/balanced64-windows \
  --suite-lock Evals/configs/balanced64-v1.json \
  --output work/balanced64-aggregate.json
```

The aggregator rejects missing or duplicate windows, changed classes, wrong position counts, negative or non-finite KLD, basis drift, and invalid Top-1 counts.

## Files

- [Exact competitive result JSON](results/deepseek-v4-flash-0731-balanced64-v1.json)
- [Frozen BALANCED64 lock](configs/balanced64-v1.json)
- [Full measurement protocol](protocols/balanced64-v1.md)
- [One-window receipt template](templates/balanced64-window-v1.json)
- [Verifier and aggregator](tools/receipts.py)
- [JSON schemas](schemas/)
- [Separate internal Backpack anchor calculations](../Backpack/README.md)
