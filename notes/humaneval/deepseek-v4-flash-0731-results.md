# DeepSeek-V4-Flash-0731 HumanEval results

| Model | HumanEval pass@1 | HumanEval+ pass@1 | Base-model weight size (decimal GB) | Effective BPW |
|---|---:|---:|---:|---:|
| **Official, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 156.016 | 4.390 |
| **UD-IQ4_XS, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 136.662 | 3.845 |
| **UD-IQ3_XXS, Thinking ON** | **94.51%** (155/164) | **90.24%** (148/164) | 104.208 | 2.932 |
| **UD-IQ2_XXS, Thinking ON** | **95.73%** (157/164) | **92.07%** (151/164) | 90.861 | 2.556 |
| **DwarfStar/DS4 asymmetric Q2, Thinking ON** | **93.90%** (154/164) | **87.80%** (144/164) | 86.720 | 2.440 |

Each score is actual benchmark accuracy from one generated solution per task. It is not token Top-1 agreement or best-of-n.

All size and BPW rows exclude MTP and drafter weights. Effective BPW is base-model weight bytes × 8 divided by the common 284,334,567,511-parameter base-model denominator.

## DwarfStar artifact and runtime identity

The DwarfStar score is not the direct Unsloth `UD-IQ2_XXS` model under another name. It was generated from `antirez/deepseek-v4-gguf@1cd7b564460821938add0475a60b942c409295e0`, file `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`. The single base-model GGUF is 86,720,111,488 bytes. It uses `IQ2_XXS` routed-expert gate/up tensors, `Q2_K` routed-expert down tensors, and higher-precision non-routed tensors.

That artifact ran target-only with no MTP through the Entrpi DwarfStar/DS4 engine, commit `72ae4bfa43d47b53f1ac781e673a088c6051af33`. Its runtime reported compressed FP8 primary KV plus a packed FP4 indexer cache. The direct Unsloth IQ2 and IQ3 rows instead used llama.cpp with F16 K/V. The DwarfStar row is therefore a measured model-plus-runtime result, not an isolated weight-quantization comparison.

## Direct Unsloth IQ2/IQ3 4K-cap audit

All 164 requests completed for both direct Unsloth rows; no timeout, dropped-request, or HTTP-error row entered either score. IQ3 produced eight 4,096-token length stops before any visible solution, while IQ2 produced six. This two-task difference accounts for the full two-task HumanEval inversion; the single-run ordering is not evidence that IQ2 is intrinsically higher quality.
