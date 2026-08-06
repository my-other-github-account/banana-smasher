# DeepSeek-V4-Flash-0731 quant results

This page compares quality and exact whole-model shipping size for seven quants measured on the frozen competitive `BALANCED64_V1` population.

## Results

Every model below ran the same 64 windows and 65,536 scored positions against the same FP8 copy of DeepSeek-V4-Flash-0731. The table is ordered by Top-1 agreement; the KLD order differs only for QTIP2.5 versus IQ3.

| Quant | Top-1 ↑ | KLD ↓ | Exact decimal GB | Comparison BPW | FP basis |
|---|---:|---:|---:|---:|---|
| **Unsloth IQ4** | **92.44%** (60,584/65,536) | **0.068349** | 136.662 | 3.845 | FP8 e4m3 dynamic own-base |
| **QTIP3 uniform exact** | **91.68%** (60,084/65,536) | **0.110227** | 123.935 | 3.487 | FP8 e4m3 dynamic own-base |
| **QTIP2.5 deterministic mixed ring** | **89.09%** (58,389/65,536) | **0.181971** | 106.623 | 3.000 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ3** | **87.95%** (57,638/65,536) | **0.177708** | 104.208 | 2.932 | FP8 e4m3 dynamic own-base |
| **QTIP2 corrected all-43** | **87.11%** (57,090/65,536) | **0.240852** | 89.296 | 2.512 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ2** | **84.57%** (55,422/65,536) | **0.276747** | 90.861 | 2.556 | FP8 e4m3 dynamic own-base |
| **DwarfStar Q2** | **83.69%** (54,845/65,536) | **0.309521** | 93.691 | 2.636 | FP8 e4m3 dynamic own-base |

Top-1 is how often the quant selects the same next token as FP8 on the common ordered support. KLD measures movement of the full supported token distribution. Higher Top-1 and lower KLD are better.

Comparison BPW is exact total artifact weight bytes × 8 divided by the canonical 284,334,567,511 base-model logical parameters for every row. This fixed denominator is the apples-to-apples value used in the table and in public quant labels. DwarfStar additionally ships a 19,845,850,983-parameter drafter; its auxiliary-inclusive BPW is 2.464 and remains available as the separately labeled `total_model_bpw` field in the machine receipt, but it is not a comparable quant label. `UD-IQ3_XXS` is a dynamic mixed quant—not a uniform three-bit model—so its exact comparison value is 2.932 BPW.

The [machine-readable result](results/deepseek-v4-flash-0731-balanced64-v1.json) contains exact bytes, full decimal ratios, candidate/teacher/scorer/population identities, component-byte ledgers, source hashes, replay limits, and the six-category breakdowns for all seven quants.

## What makes these apples to apples

- Same model family: DeepSeek-V4-Flash-0731
- Same 64 ordered windows and 65,536 total positions
- Same FP8 e4m3 dynamic own-base teacher
- Same teacher top-8,192 token support
- Same KLD and Top-1 definitions
- Same 284,334,567,511-parameter base-model denominator for comparable/publication BPW; auxiliary-inclusive BPW is reported separately
- Same corrected class mix: agentic/chat/code/multilingual/prose/reasoning = `19/7/9/10/10/9`

No partial run, different window bank, fallback output, or HOLDOUT result is admitted to this ranking.

## Category breakdowns

These category rows are derived from the exact same 64-window competitive aggregates as the global table. Top-1 percentages are rounded for readability, with exact integer counts beside them. KLD is the position-weighted class mean; exact decimals remain in the machine-readable result.

### Top-1 agreement

| Quant | Agentic | Chat | Code | Multilingual | Prose | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| **Unsloth IQ4** | 92.12% (17,922/19,456) | 94.45% (6,770/7,168) | 94.61% (8,719/9,216) | 89.90% (9,206/10,240) | 89.05% (9,119/10,240) | 96.01% (8,848/9,216) |
| **QTIP3 exact** | 91.75% (17,851/19,456) | 95.37% (6,836/7,168) | 94.34% (8,694/9,216) | 87.47% (8,957/10,240) | 86.48% (8,856/10,240) | 96.46% (8,890/9,216) |
| **QTIP2.5 mixed** | 89.27% (17,368/19,456) | 93.22% (6,682/7,168) | 93.09% (8,579/9,216) | 83.30% (8,530/10,240) | 82.26% (8,423/10,240) | 95.56% (8,807/9,216) |
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
| **QTIP2.5 mixed** | 0.2331 | 0.0496 | 0.0836 | 0.3233 | 0.2647 | 0.0264 |
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

Expected Top-1 order:

```text
IQ4 > QTIP3 > QTIP2.5 > IQ3 > QTIP2 > IQ2 > DwarfStar
```

Expected KLD order:

```text
IQ4 > QTIP3 > IQ3 > QTIP2.5 > QTIP2 > IQ2 > DwarfStar
```

## Standard HumanEval tooling

`HUMANEVAL_0731_V1` provides one frozen HumanEval/HumanEval+ path for any DeepSeek-V4-Flash-0731 artifact exposed through an OpenAI-compatible endpoint. It fixes the historical false-cap bug by binding a real 4,096-token completion budget that excludes the prompt, preserves semantic null responses as failures, and enforces four disjoint resumable shards with exactly one sample per task.

Inspect the frozen config:

```bash
python3 -m Evals.tools.humaneval show-config
```

The CLI supports `generate`, `merge`, `audit`, and `score`. Generated code must be scored in the provided network-isolated Docker environment. See the [HumanEval 0731 protocol](protocols/humaneval-0731-v1.md) for the complete commands.

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
- [Standard HumanEval CLI](tools/humaneval.py)
- [Frozen HumanEval 0731 lock](configs/humaneval-0731-v1.json)
- [HumanEval 0731 protocol](protocols/humaneval-0731-v1.md)
- [Pinned HumanEval container](docker/humaneval/Dockerfile)
- [JSON schemas](schemas/)
- [Separate internal Backpack anchor calculations](../Backpack/README.md)
