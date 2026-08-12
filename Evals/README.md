# DeepSeek-V4-Flash-0731 quant results

This page compares quality and declared shipping-accounting size for fourteen quants measured on the frozen competitive `BALANCED64_V1` population. Every row states whether its byte numerator excludes MTP, includes the native MTP checkpoint, or includes a separate drafter.

The separate [MMLU-500 capability-density table](../notes/benchmarks/mmlu-density/mmlu500-v1/four-row-results.md) scores twelve sealed rows on one immutable zero-shot question bank: the Official native MXFP4 reference, the four previously published Unsloth/DwarfStar rows, the two routed-only EXL3 K2/K3 native-rest measurements, QTIP2 corrected all-43, EXL3 K3 uniform exact, and the newly completed QTIP3, QTIP2.5, and full EXL3 K2 measurements. The original four per-question result sets remain unchanged. Every later row carries compact public-safe measurement provenance; the completed full EXL3 K2.5 search is separately recorded as `ARTIFACT_UNAVAILABLE`, with no inferred score.

## Results

Every quant row with Top-1/KLD values ran the same 64 windows and 65,536 scored positions against the same FP8 copy of DeepSeek-V4-Flash-0731. Those measured rows are ordered by Top-1 agreement; the Official native MMLU/accounting reference is pinned above them and has no inferred BALANCED64 values. `EXL3 K2.5` denotes the measured greedy optimizer assignment: the full row applies it to all eligible weights, while the routed-only row applies that exact assignment only to routed experts and preserves every shared and non-routed tensor in the exact native source representation. The routed K2 and K3 rows use the same native-rest scope at homogeneous endpoints. The in-house physical alternating K2/K3 comparator is a separate control outside the EXL matrix and is not labeled EXL K2.5.

| Quant | Top-1 ↑ | KLD ↓ | MMLU ↑ | Above-Chance MMLU/BPW (within model) ↑ | Raw MMLU/BPW ↑ | Above-Chance MMLU/GB ↑ | Exact accounting GB | Shipped auxiliary scope | Comparison BPW | FP basis |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| **Official native MXFP4** |  |  | **84.60%** (423/500) | **13.576** | **19.270** | **0.382** | 156.035 | MTP/drafter excluded; base-only reference | 4.390 | Native MXFP4; BALANCED64 not measured |
| **Unsloth IQ4** | **92.44%** (60,584/65,536) | **0.068349** | **83.40%** (417/500) | **15.188** | **21.690** | **0.427** | 136.662 | MTP excluded | 3.845 | FP8 e4m3 dynamic own-base |
| **EXL3 K3 routed-only + native rest** | **92.23%** (60,447/65,536) | **0.076868** | **85.20%** (426/500) | **17.255** | **24.420** | **0.485** | 123.999 | MTP included | 3.489 | FP8 e4m3 dynamic own-base |
| **QTIP3 uniform exact** | **91.68%** (60,084/65,536) | **0.110227** | **84.20%** (421/500) | **16.973** | **24.140** | **0.478** | **123.969** | MTP included | **3.488** | FP8 e4m3 dynamic own-base |
| **QTIP2.5 deterministic mixed ring** | **89.09%** (58,389/65,536) | **0.181971** | **82.80%** (414/500) | **19.261** | **27.592** | **0.542** | **106.657** | MTP included | **3.001** | FP8 e4m3 dynamic own-base |
| **EXL3 K2.5 greedy-upcast routed-only + native rest** | **88.33%** (57,885/65,536) | **0.174604** |  |  |  |  | 106.283 | MTP included | 2.990 | FP8 e4m3 dynamic own-base; MMLU run superseded before acceptance |
| **EXL3 K3 uniform exact** | **88.30%** (57,870/65,536) | **0.136015** | **84.80%** (424/500) | **18.766** | **26.611** | **0.528** | 113.260 | MTP included | 3.187 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ3** | **87.95%** (57,638/65,536) | **0.177708** | **83.20%** (416/500) | **19.850** | **28.377** | **0.558** | 104.208 | MTP excluded | 2.932 | FP8 e4m3 dynamic own-base |
| **QTIP2 corrected all-43** | **87.11%** (57,090/65,536) | **0.240852** | **82.40%** (412/500) | **22.838** | **32.785** | **0.643** | **89.330** | MTP included | **2.513** | FP8 e4m3 dynamic own-base |
| **EXL3 K2 routed-only + native rest** | **86.33%** (56,579/65,536) | **0.234288** | **83.60%** (418/500) | **23.305** | **33.247** | **0.656** | 89.371 | MTP included | 2.515 | FP8 e4m3 dynamic own-base |
| **Unsloth IQ2** | **84.57%** (55,422/65,536) | **0.276747** | **81.80%** (409/500) | **22.218** | **31.998** | **0.625** | 90.861 | MTP excluded | 2.556 | FP8 e4m3 dynamic own-base |
| **DwarfStar Q2** | **83.69%** (54,845/65,536) | **0.309521** | **80.60%** (403/500) | **21.092** | **30.576** | **0.593** | 93.691 | Stock MTP excluded; separate drafter included | 2.636 | FP8 e4m3 dynamic own-base |
| **EXL3 K2.5 greedy optimizer full** | **83.51%** (54,732/65,536) | **0.302775** |  |  |  |  | 94.833 | MTP included | 2.668 | FP8 e4m3 dynamic own-base; MMLU artifact unavailable |
| **Physical alternating K2/K3 2.5-BPW comparator** | **83.29%** (54,585/65,536) | **0.299604** |  |  |  |  | 94.833 | MTP included | 2.668 | FP8 e4m3 dynamic own-base |
| **EXL3 K2 uniform exact** | **81.78%** (53,593/65,536) | **0.366820** | **73.80%** (369/500) | **22.276** | **33.688** | **0.627** | **77.862** | MTP included | **2.191** | FP8 e4m3 dynamic own-base |

`Above-Chance MMLU/BPW (within model)` is `(MMLU percentage - 25) / comparison BPW`; `Raw MMLU/BPW` is `MMLU percentage / comparison BPW`. BPW density is comparable only for variants sharing the same base-model parameter denominator and is not a cross-model-family ranking. `Above-Chance MMLU/GB` is `(MMLU percentage - 25) / complete decimal artifact GB` and is the storage-normalized cross-model metric. Every metric uses exact machine-readable denominators rather than rounded display values. Blank cells mean no sealed MMLU result is available; no score or density is projected. The full EXL3 K2.5 artifact was retired before a score terminal could be sealed, and the routed-only K2.5 MMLU run was superseded with zero accepted rows; both remain blank. The Official native row is the base-only MMLU/accounting reference: 156,035,165,824 complete bytes with native MTP and any drafter excluded. No BALANCED64 Top-1/KLD terminal is sealed for that artifact, so those cells remain blank rather than being inferred from its reference role.

## EXL 2×3 scope/rate matrix

These are scope-matched physical `BALANCED64_V1` cells. Each measured cell shows Top-1 agreement, KLD, exact shipped GB, and base-equivalent BPW. The K2.5 column uses the exact measured greedy optimizer assignment in both scopes; it is not an average or parity interpolation.

| EXL scope | K2 | EXL optimizer K2.5 | K3 |
|---|---|---|---|
| **Full / all eligible** | 53,593/65,536; KLD 0.366820; 77.862 GB; 2.191 BPW | 54,732/65,536; KLD 0.302775; 94.833 GB; 2.668 BPW — greedy optimizer | 57,870/65,536; KLD 0.136015; 113.260 GB; 3.187 BPW |
| **Routed experts only + exact native rest** | 56,579/65,536; KLD 0.234288; 89.371 GB; 2.515 BPW | **57,885/65,536; KLD 0.174604; 106.283 GB; 2.990 BPW — exact greedy assignment** | 60,447/65,536; KLD 0.076868; 123.999 GB; 3.489 BPW |

The **Physical alternating K2/K3 2.5-BPW comparator** is an in-house 68-K2/61-K3 control. It remains in the global comparison table with every original metric and receipt preserved, but it is outside this EXL matrix and is not the EXL optimizer K2.5 cell. Its separately sealed routed-native control scored 58,047/65,536 with KLD 0.16150331034095772; that matched-scope comparison is labeled strict alternating rather than EXL K2.5.

Top-1 is how often the quant selects the same next token as FP8 on the common ordered support. KLD measures movement of the full supported token distribution. Higher Top-1 and lower KLD are better.

**Base-equivalent BPW** is every shipped artifact byte × 8 divided by the fixed 284,334,567,511-parameter base-model denominator. Native MTP and separate-drafter bytes receive no denominator credit, so this is the conservative apples-to-apples value used for public quant labels. **Matched physical BPW** divides the same bytes by the parameters represented by that row's declared payload scope: 284,334,567,511 for base-only artifacts, 294,550,374,339 for base plus the 10,215,806,828-parameter native MTP checkpoint, and 304,180,418,494 for DwarfStar's base plus separate 19,845,850,983-parameter drafter. Because those payload scopes differ, matched physical BPW is disclosure—not a cross-row ranking.

The physical alternating comparator has `2.499913678623607` BPW over EXL3-eligible optimized weights. That optimizer-scope rate is recorded in the accounting receipt but is **not** used as an EXL K2.5 label. Its apples-to-apples base-equivalent value is `2.668206220359224…` BPW because the public numerator includes all 94,832,907,712 shipped bytes, including native MTP and metadata. The distinct full EXL K2.5 greedy optimizer ships 94,832,865,520 bytes and has base-equivalent BPW `2.6682050332506607541984592911089396395596594633708887538020512532588125649342761`.

The routed-native greedy cell selects 86,573,712,384 routed-optimizer bytes (35,641,165,824 from K2 and 50,932,546,560 from K3) plus 19,708,797,688 exact native-rest bytes, for 106,282,510,072 shipped payload bytes. Its protected per-key selection manifest adds 58,031,468 bytes to the virtual container and is disclosed separately rather than charged twice to the established public payload convention. The exact solution has 66 K2 and 63 K3 groups; the routed namespace resolves 42 K2 and 44 K3 groups after non-routed groups remain native. Native checkpoint leaves are byte-identical in storage, while the protected EXL3 runtime dequantizes non-EXL3 leaves to FP16 for execution; no native MXFP4 execution kernel is claimed. The accepted attempt took 480.7423007488251 seconds (`0.13353952831692165` Spark-hours) with zero accepted-attempt failures. One earlier pre-score namespace mismatch failed closed before candidate rows and was recovered without replay. Independent recomputation, an 84-file durable mirror, and physical host/shard release postimages were sealed before publication.

The corrected QTIP accounting restores ten omitted MTP tensors totaling 33,843,220 payload bytes and includes the deterministic index-length increase. That raises QTIP2 from 89.296 to **89.330 GB** and QTIP3 from 123.935 to **123.969 GB**. QTIP2.5's later physical MMLU terminal materialized an exact **106.657444992 GB** artifact, superseding the earlier 106.657097796 GB reconstruction by 347,196 bytes. Quality metrics do not change. Source index hashes and the exact physical-terminal receipt remain recorded in the machine receipts; no unmaterialized content hash is fabricated.

The [machine-readable result](results/deepseek-v4-flash-0731-balanced64-v1.json) contains exact bytes, full decimal ratios, per-row payload scope, both BPW conventions, candidate identities, measurement bindings, protected source hashes, replay limits, and six-category breakdowns for all fourteen quants. The SHA-bound [MTP size-accounting correction](results/deepseek-v4-flash-0731-mtp-size-accounting-v1.json) records denominator policy and scope evidence for the complete row population. The full greedy K2.5 row binds its exact-rate solution, corrected optimizer measurement, artifact identity, payload manifest, physical provenance, 64-window bindings, capture manifest, and independent recomputation. The routed-native greedy row additionally binds its fresh overlay identity, routed tensor-source manifest, runtime-selected payload proof, Exact64 capture and independent recomputation, functional readback, durable mirror, and release terminal. The routed K3 row binds its protected source tree/index, overlay and selected-payload proofs, 64-window measurement, raw rows, independent recomputation, functional readback, exact 104,290,452,480-byte routed payload, and exact 19,708,797,688-byte native-rest payload. Separately materialized routed composite tree and publicly distributed manifest digests remain explicitly unavailable rather than inferred.

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
| **EXL3 K3 routed + native rest** | 91.85% (17,871/19,456) | 94.21% (6,753/7,168) | 94.64% (8,722/9,216) | 89.46% (9,161/10,240) | 88.76% (9,089/10,240) | 96.04% (8,851/9,216) |
| **QTIP3 exact** | 91.75% (17,851/19,456) | 95.37% (6,836/7,168) | 94.34% (8,694/9,216) | 87.47% (8,957/10,240) | 86.48% (8,856/10,240) | 96.46% (8,890/9,216) |
| **QTIP2.5 mixed** | 89.27% (17,368/19,456) | 93.22% (6,682/7,168) | 93.09% (8,579/9,216) | 83.30% (8,530/10,240) | 82.26% (8,423/10,240) | 95.56% (8,807/9,216) |
| **EXL3 K2.5 greedy routed + native rest** | 87.93% (17,107/19,456) | 91.16% (6,534/7,168) | 91.80% (8,460/9,216) | 84.14% (8,616/10,240) | 83.10% (8,509/10,240) | 93.96% (8,659/9,216) |
| **EXL3 K3 exact** | 88.87% (17,291/19,456) | 90.11% (6,459/7,168) | 91.44% (8,427/9,216) | 84.84% (8,688/10,240) | 82.76% (8,475/10,240) | 92.56% (8,530/9,216) |
| **Unsloth IQ3** | 87.68% (17,059/19,456) | 91.35% (6,548/7,168) | 91.46% (8,429/9,216) | 83.54% (8,555/10,240) | 82.17% (8,414/10,240) | 93.67% (8,633/9,216) |
| **QTIP2 all-43** | 87.78% (17,079/19,456) | 91.62% (6,567/7,168) | 91.29% (8,413/9,216) | 80.11% (8,203/10,240) | 79.01% (8,091/10,240) | 94.80% (8,737/9,216) |
| **EXL3 K2 routed + native rest** | 86.43% (16,815/19,456) | 89.30% (6,401/7,168) | 90.08% (8,302/9,216) | 81.48% (8,344/10,240) | 80.18% (8,210/10,240) | 92.31% (8,507/9,216) |
| **Unsloth IQ2** | 84.43% (16,426/19,456) | 88.85% (6,369/7,168) | 89.08% (8,210/9,216) | 78.76% (8,065/10,240) | 76.81% (7,865/10,240) | 92.09% (8,487/9,216) |
| **DwarfStar Q2** | 83.13% (16,174/19,456) | 88.38% (6,335/7,168) | 88.10% (8,119/9,216) | 77.75% (7,962/10,240) | 76.12% (7,795/10,240) | 91.80% (8,460/9,216) |
| **EXL3 K2.5 greedy optimizer full** | 84.06% (16,355/19,456) | 85.92% (6,159/7,168) | 88.03% (8,113/9,216) | 78.32% (8,020/10,240) | 76.61% (7,845/10,240) | 89.41% (8,240/9,216) |
| **Physical alternating K2/K3 comparator** | 83.83% (16,310/19,456) | 85.57% (6,134/7,168) | 87.89% (8,100/9,216) | 78.34% (8,022/10,240) | 76.18% (7,801/10,240) | 89.17% (8,218/9,216) |
| **EXL3 K2 exact** | 82.56% (16,063/19,456) | 84.15% (6,032/7,168) | 86.74% (7,994/9,216) | 75.98% (7,780/10,240) | 74.20% (7,598/10,240) | 88.17% (8,126/9,216) |

### KLD

| Quant | Agentic | Chat | Code | Multilingual | Prose | Reasoning |
|---|---:|---:|---:|---:|---:|---:|
| **Unsloth IQ4** | 0.1061 | 0.0256 | 0.0332 | 0.0941 | 0.0823 | 0.0131 |
| **EXL3 K3 routed + native rest** | 0.1217 | 0.0288 | 0.0313 | 0.1076 | 0.0912 | 0.0149 |
| **QTIP3 exact** | 0.1513 | 0.0285 | 0.0528 | 0.1792 | 0.1563 | 0.0168 |
| **EXL3 K3 exact** | 0.1870 | 0.0665 | 0.0733 | 0.1900 | 0.1745 | 0.0425 |
| **EXL3 K2.5 greedy routed + native rest** | 0.2567 | 0.0654 | 0.0845 | 0.2675 | 0.2119 | 0.0317 |
| **Unsloth IQ3** | 0.2507 | 0.0736 | 0.0894 | 0.2688 | 0.2279 | 0.0360 |
| **QTIP2.5 mixed** | 0.2331 | 0.0496 | 0.0836 | 0.3233 | 0.2647 | 0.0264 |
| **EXL3 K2 routed + native rest** | 0.3232 | 0.0989 | 0.1223 | 0.3636 | 0.2960 | 0.0515 |
| **QTIP2 all-43** | 0.2896 | 0.0683 | 0.1223 | 0.4477 | 0.3549 | 0.0341 |
| **Unsloth IQ2** | 0.3770 | 0.1123 | 0.1441 | 0.4302 | 0.3623 | 0.0601 |
| **Physical alternating K2/K3 comparator** | 0.3856 | 0.1540 | 0.1626 | 0.4383 | 0.3981 | 0.1046 |
| **EXL3 K2.5 greedy optimizer full** | 0.3965 | 0.1512 | 0.1640 | 0.4481 | 0.3914 | 0.1017 |
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
IQ4 > EXL3 K3 routed + native rest > QTIP3 > QTIP2.5 > EXL3 K2.5 greedy routed + native rest > EXL3 K3 full > IQ3 > QTIP2 > EXL3 K2 routed + native rest > IQ2 > DwarfStar > EXL3 K2.5 greedy full > physical alternating comparator > EXL3 K2
```

Expected KLD order:

```text
IQ4 > EXL3 K3 routed + native rest > QTIP3 > EXL3 K3 full > EXL3 K2.5 greedy routed + native rest > IQ3 > QTIP2.5 > EXL3 K2 routed + native rest > QTIP2 > IQ2 > physical alternating comparator > EXL3 K2.5 greedy full > DwarfStar > EXL3 K2
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
