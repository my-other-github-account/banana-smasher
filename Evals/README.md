# DeepSeek-V4-Flash-0731 quant results

This page compares quality and declared shipping-accounting size for eleven quants measured on the frozen competitive `BALANCED64_V1` population. Every row states whether its byte numerator excludes MTP, includes the native MTP checkpoint, or includes a separate drafter.

## Results

Every model below ran the same 64 windows and 65,536 scored positions against the same FP8 copy of DeepSeek-V4-Flash-0731. The table is ordered by Top-1 agreement; by KLD, EXL3 K3 moves ahead of both IQ3 and QTIP2.5, while IQ3 also moves ahead of QTIP2.5. The EXL3 routed-only K2 row replaces only routed experts and preserves every shared and non-routed tensor in the exact native source representation; it is distinct from both the all-eligible-linears EXL3 K2 row and the EXL3 K2.5 physical-alternating control.

| Quant | Top-1 ↑ | KLD ↓ | Exact accounting GB | Shipped auxiliary scope | Base-equivalent BPW | Matched physical BPW | FP basis |
|---|---:|---:|---:|---|---:|---:|---|
| **Unsloth IQ4** | **92.44%** (60,584/65,536) | **0.068349** | 136.662 | MTP excluded | 3.845 | 3.845 | FP8 e4m3 dynamic own-base |
| **QTIP3 uniform exact** | **91.68%** (60,084/65,536) | **0.110227** | **123.969** | MTP included | **3.488** | 3.367 | FP8 e4m3 dynamic own-base |
| **QTIP2.5 deterministic mixed ring** | **89.09%** (58,389/65,536) | **0.181971** | **106.657** | MTP included | **3.001** | 2.897 | FP8 e4m3 dynamic own-base |
| **EXL3 K3 uniform exact** | **88.30%** (57,870/65,536) | **0.136015** | 113.260 | MTP included | 3.187 | 3.076 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ3** | **87.95%** (57,638/65,536) | **0.177708** | 104.208 | MTP excluded | 2.932 | 2.932 | FP8 e4m3 dynamic own-base |
| **QTIP2 corrected all-43** | **87.11%** (57,090/65,536) | **0.240852** | **89.330** | MTP included | **2.513** | 2.426 | FP8 e4m3 dynamic own-base |
| **EXL3 K2 routed-only + native rest** | **86.33%** (56,579/65,536) | **0.234288** | 89.371 | MTP included | 2.515 | 2.427 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ2** | **84.57%** (55,422/65,536) | **0.276747** | 90.861 | MTP excluded | 2.556 | 2.556 | FP8 e4m3 dynamic own-base |
| **DwarfStar Q2** | **83.69%** (54,845/65,536) | **0.309521** | 93.691 | Stock MTP excluded; separate drafter included | 2.636 | 2.464 | FP8 e4m3 dynamic own-base |
| **EXL3 K2.5 physical alternating** | **83.29%** (54,585/65,536) | **0.299604** | 94.833 | MTP included | 2.668 | 2.576 | FP8 e4m3 dynamic own-base |
| **EXL3 K2 uniform exact** | **81.78%** (53,593/65,536) | **0.366820** | 77.862 | MTP included | 2.191 | 2.115 | FP8 e4m3 dynamic own-base |

Top-1 is how often the quant selects the same next token as FP8 on the common ordered support. KLD measures movement of the full supported token distribution. Higher Top-1 and lower KLD are better.

**Base-equivalent BPW** is every shipped artifact byte × 8 divided by the fixed 284,334,567,511-parameter base-model denominator. Native MTP and separate-drafter bytes receive no denominator credit, so this is the conservative apples-to-apples value used for public quant labels. **Matched physical BPW** divides the same bytes by the parameters represented by that row's declared payload scope: 284,334,567,511 for base-only artifacts, 294,550,374,339 for base plus the 10,215,806,828-parameter native MTP checkpoint, and 304,180,418,494 for DwarfStar's base plus separate 19,845,850,983-parameter drafter. Because those payload scopes differ, matched physical BPW is disclosure—not a cross-row ranking.

The EXL3 K2.5 physical-alternating control has `2.499913678623607` BPW over EXL3-eligible optimized weights. That optimizer-scope rate is recorded in the accounting receipt but is **not** used as its public quant label. Its apples-to-apples base-equivalent value is `2.668206220359224…` BPW because the public numerator includes all 94,832,907,712 shipped bytes, including native MTP and metadata.

The corrected QTIP accounting restores ten omitted MTP tensors totaling 33,843,220 payload bytes and includes the deterministic index-length increase. That raises QTIP2 from 89.296 to **89.330 GB**, QTIP2.5 from 106.623 to **106.657 GB**, and QTIP3 from 123.935 to **123.969 GB**; quality metrics do not change. The source index hashes remain recorded in the machine receipt. Corrected index byte lengths are exact reconstructions, but new corrected index content hashes are explicitly unmaterialized rather than fabricated.

The [machine-readable result](results/deepseek-v4-flash-0731-balanced64-v1.json) contains exact bytes, full decimal ratios, per-row payload scope, both BPW conventions, candidate/teacher/scorer/population identities, available component-byte ledgers, source hashes, replay limits, and the six-category breakdowns for all eleven quants. The SHA-bound [MTP size-accounting correction](results/deepseek-v4-flash-0731-mtp-size-accounting-v1.json) records the omitted tensor names, old and corrected QTIP byte totals, source-index hashes, corrected index lengths, denominator policy, and scope evidence. The routed-only EXL3 K2 row binds the protected K2 source tree/index, overlay and selected-payload proofs, 64-window measurement, independent recomputation, functional readback, exact 69,662,278,656-byte routed payload, and exact 19,708,797,688-byte native-rest payload. Its separately materialized composite tree and manifest remain explicitly unavailable rather than inferred. The all-eligible-linears EXL3 K2 and physical-alternating EXL3 K2.5 rows remain unchanged. The older EXL3 K3 source revision, code lineage, weight-component ledger, and runtime measurements also remain explicitly unavailable rather than inferred.

## What makes these apples to apples

- Same model family: DeepSeek-V4-Flash-0731
- Same 64 ordered windows and 65,536 total positions
- Same FP8 e4m3 dynamic own-base teacher
- Same teacher top-8,192 token support
- Same KLD and Top-1 definitions
- Same 284,334,567,511-parameter base-model denominator for conservative base-equivalent/publication BPW
- Explicit per-row payload scope; matched physical BPW uses 294,550,374,339 parameters for native-MTP rows and 304,180,418,494 for DwarfStar's separate drafter
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
| **EXL3 K3 exact** | 88.87% (17,291/19,456) | 90.11% (6,459/7,168) | 91.44% (8,427/9,216) | 84.84% (8,688/10,240) | 82.76% (8,475/10,240) | 92.56% (8,530/9,216) |
| **Unsloth IQ3** | 87.68% (17,059/19,456) | 91.35% (6,548/7,168) | 91.46% (8,429/9,216) | 83.54% (8,555/10,240) | 82.17% (8,414/10,240) | 93.67% (8,633/9,216) |
| **QTIP2 all-43** | 87.78% (17,079/19,456) | 91.62% (6,567/7,168) | 91.29% (8,413/9,216) | 80.11% (8,203/10,240) | 79.01% (8,091/10,240) | 94.80% (8,737/9,216) |
| **EXL3 K2 routed + native rest** | 86.43% (16,815/19,456) | 89.30% (6,401/7,168) | 90.08% (8,302/9,216) | 81.48% (8,344/10,240) | 80.18% (8,210/10,240) | 92.31% (8,507/9,216) |
| **Unsloth IQ2** | 84.43% (16,426/19,456) | 88.85% (6,369/7,168) | 89.08% (8,210/9,216) | 78.76% (8,065/10,240) | 76.81% (7,865/10,240) | 92.09% (8,487/9,216) |
| **DwarfStar Q2** | 83.13% (16,174/19,456) | 88.38% (6,335/7,168) | 88.10% (8,119/9,216) | 77.75% (7,962/10,240) | 76.12% (7,795/10,240) | 91.80% (8,460/9,216) |
| **EXL3 K2.5 physical alternating** | 83.83% (16,310/19,456) | 85.57% (6,134/7,168) | 87.89% (8,100/9,216) | 78.34% (8,022/10,240) | 76.18% (7,801/10,240) | 89.17% (8,218/9,216) |
| **EXL3 K2 exact** | 82.56% (16,063/19,456) | 84.15% (6,032/7,168) | 86.74% (7,994/9,216) | 75.98% (7,780/10,240) | 74.20% (7,598/10,240) | 88.17% (8,126/9,216) |

### KLD

| Quant | Agentic | Chat | Code | Multilingual | Prose | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| **Unsloth IQ4** | 0.1061 | 0.0256 | 0.0332 | 0.0941 | 0.0823 | 0.0131 |
| **QTIP3 exact** | 0.1513 | 0.0285 | 0.0528 | 0.1792 | 0.1563 | 0.0168 |
| **EXL3 K3 exact** | 0.1870 | 0.0665 | 0.0733 | 0.1900 | 0.1745 | 0.0425 |
| **Unsloth IQ3** | 0.2507 | 0.0736 | 0.0894 | 0.2688 | 0.2279 | 0.0360 |
| **QTIP2.5 mixed** | 0.2331 | 0.0496 | 0.0836 | 0.3233 | 0.2647 | 0.0264 |
| **EXL3 K2 routed + native rest** | 0.3232 | 0.0989 | 0.1223 | 0.3636 | 0.2960 | 0.0515 |
| **QTIP2 all-43** | 0.2896 | 0.0683 | 0.1223 | 0.4477 | 0.3549 | 0.0341 |
| **Unsloth IQ2** | 0.3770 | 0.1123 | 0.1441 | 0.4302 | 0.3623 | 0.0601 |
| **EXL3 K2.5 physical alternating** | 0.3856 | 0.1540 | 0.1626 | 0.4383 | 0.3981 | 0.1046 |
| **DwarfStar Q2** | 0.4198 | 0.1250 | 0.1745 | 0.4674 | 0.4150 | 0.0625 |
| **EXL3 K2 exact** | 0.4630 | 0.1931 | 0.1964 | 0.5621 | 0.4817 | 0.1246 |

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
IQ4 > QTIP3 > QTIP2.5 > EXL3 K3 > IQ3 > QTIP2 > EXL3 K2 routed + native rest > IQ2 > DwarfStar > EXL3 K2.5 > EXL3 K2
```

Expected KLD order:

```text
IQ4 > QTIP3 > EXL3 K3 > IQ3 > QTIP2.5 > EXL3 K2 routed + native rest > QTIP2 > IQ2 > EXL3 K2.5 > DwarfStar > EXL3 K2
```

## Standard HumanEval tooling

`HUMANEVAL_0731_V1` provides one frozen HumanEval/HumanEval+ path for any DeepSeek-V4-Flash-0731 artifact exposed through an OpenAI-compatible endpoint. It fixes the historical false-cap bug by binding a real 4,096-token completion budget that excludes the prompt, preserves semantic null responses as failures, and enforces four disjoint resumable shards with exactly one sample per task.

Inspect the frozen config:

```bash
python3 -m Evals.tools.humaneval show-config
```

The CLI supports `generate`, `merge`, `audit`, and `score`. Generated code must be scored in the provided network-isolated Docker environment. See the [HumanEval 0731 protocol](protocols/humaneval-0731-v1.md) for the complete commands and the [published FF0731 result table](../notes/humaneval/deepseek-v4-flash-0731-results.md) for Official, IQ4, IQ3, and DwarfStar scores.

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
- [MTP size-accounting correction receipt](results/deepseek-v4-flash-0731-mtp-size-accounting-v1.json)
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
