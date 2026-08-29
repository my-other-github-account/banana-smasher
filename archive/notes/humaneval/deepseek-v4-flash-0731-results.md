# DeepSeek-V4-Flash-0731 HumanEval results

| Model | HumanEval pass@1 | HumanEval+ pass@1 | Base-model weight size (decimal GB) | Effective BPW |
|---|---:|---:|---:|---:|
| **Official, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 156.016 | 4.390 |
| **UD-IQ4_XS, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 136.662 | 3.845 |
| **UD-IQ3_XXS, Thinking ON** | **94.51%** (155/164) | **90.24%** (148/164) | 104.208 | 2.932 |
| **UD-IQ2_XXS, Thinking ON** | **95.73%** (157/164) | **89.63%** (147/164) | 90.861 | 2.556 |
| **DwarfStar/DS4 asymmetric Q2, Thinking ON** | **93.90%** (154/164) | **87.80%** (144/164) | 86.720 | 2.440 |
| **Mia/0xSero REAP-K216 EXL3 3.0, Thinking OFF** | **95.12%** (156/164) | **91.46%** (150/164) | 106.817 | 3.005 |

Each score is actual benchmark accuracy from one generated solution per task. It is not token Top-1 agreement or best-of-n.

All size and BPW rows exclude MTP and drafter weights. Effective BPW is base-model weight bytes × 8 divided by the common 284,334,567,511-parameter base-model denominator.

## DwarfStar artifact and runtime identity

The DwarfStar score is not the direct Unsloth `UD-IQ2_XXS` model under another name. It was generated from `antirez/deepseek-v4-gguf@1cd7b564460821938add0475a60b942c409295e0`, file `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`. The single base-model GGUF is 86,720,111,488 bytes. It uses `IQ2_XXS` routed-expert gate/up tensors, `Q2_K` routed-expert down tensors, and higher-precision non-routed tensors.

That artifact ran target-only with no MTP through the Entrpi DwarfStar/DS4 engine, commit `72ae4bfa43d47b53f1ac781e673a088c6051af33`. Its runtime reported compressed FP8 primary KV plus a packed FP4 indexer cache. The direct Unsloth IQ2 and IQ3 rows instead used llama.cpp with F16 K/V. The DwarfStar row is therefore a measured model-plus-runtime result, not an isolated weight-quantization comparison.

## Direct Unsloth IQ2/IQ3 4K-cap audit

All 164 requests completed for both direct Unsloth rows; no unresolved timeout, dropped-request, HTTP-error, or context-error row entered either score. The matched IQ2 run used `--parallel 4 --ctx-size 32768` (8,192 tokens per slot, F16/F16 KV) and produced seven 4,096-token length stops before any visible solution: `HumanEval/1`, `HumanEval/76`, `HumanEval/116`, `HumanEval/129`, `HumanEval/130`, `HumanEval/132`, and `HumanEval/145`. Those seven are exactly its HumanEval failures. Ten additional tasks failed only the strict HumanEval+ tests: `HumanEval/32`, `HumanEval/39`, `HumanEval/86`, `HumanEval/91`, `HumanEval/99`, `HumanEval/124`, `HumanEval/125`, `HumanEval/134`, `HumanEval/141`, and `HumanEval/151`.

The replaced IQ2 run used 34,816 total context tokens (8,704 per slot) and produced six length stops: `HumanEval/32`, `HumanEval/47`, `HumanEval/116`, `HumanEval/129`, `HumanEval/132`, and `HumanEval/145`. Moving IQ2 to the common 8,192-token-per-slot rail preserved its 157/164 HumanEval count but changed the task-level outcomes and reduced strict HumanEval+ from 151/164 to 147/164. IQ3 remains unchanged at 155/164 HumanEval and 148/164 HumanEval+, with eight 4,096-token length stops before any visible solution. These single-run orderings are not evidence that either quant is intrinsically higher quality.

## Mia/0xSero REAP-K216 EXL3 3.0 audit

The Mia row is a physical 164/164 generation and frozen EvalPlus terminal, not a projected or borrowed score. It used exact artifact index `b7a450f88c99aee7f6d44ecb127e91e45ab5ccb1a0dad49ca9eabb90b400c304`, target-only on one DGX Spark, with MTP and speculative decoding disabled. The known-coherent MMLU-era serving seam used the pinned runtime image digest `sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`, `VLLM_DSV4_PADDED_NVFP4=0`, and thinking disabled at the server chat-template default.

Before HumanEval generation, the serve passed a free-text English coherence canary and a frozen three-task gate at 3/3 HumanEval and 3/3 HumanEval+. The accepted-row request audit matched the published DwarfStar target-only row on every non-model field: one user message, one greedy sample, temperature 0, top-p 0.95, 4,096 completion-token cap, no stop field, and no chat-template override. All 164 full-run requests completed, all four request-audit shards passed, and all 164 sanitized solutions were non-empty and syntax-valid. Frozen EvalPlus measured 156/164 HumanEval and 150/164 HumanEval+. The retracted 0/164 row came from an incoherent serve-side decode path and is not a model-capability result. The public-safe machine receipt is [`mia-0xsero-reap-k216-exl3-0731.json`](mia-0xsero-reap-k216-exl3-0731.json).
