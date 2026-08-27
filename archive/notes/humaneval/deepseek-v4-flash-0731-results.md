# DeepSeek-V4-Flash-0731 HumanEval results

| Model | HumanEval pass@1 | HumanEval+ pass@1 | Base-model weight size (decimal GB) | Effective BPW |
|---|---:|---:|---:|---:|
| **Official, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 156.016 | 4.390 |
| **UD-IQ4_XS, Thinking ON** | **96.95%** (159/164) | **92.68%** (152/164) | 136.662 | 3.845 |
| **UD-IQ3_XXS, Thinking ON** | **94.51%** (155/164) | **90.24%** (148/164) | 104.208 | 2.932 |
| **UD-IQ2_XXS, Thinking ON** | **95.73%** (157/164) | **89.63%** (147/164) | 90.861 | 2.556 |
| **DwarfStar/DS4 asymmetric Q2, Thinking ON** | **93.90%** (154/164) | **87.80%** (144/164) | 86.720 | 2.440 |
| **Mia/0xSero REAP-K216 EXL3 3.0, Thinking ON** | **0.00%** (0/164) | **0.00%** (0/164) | 106.817 | 3.005 |

Each score is actual benchmark accuracy from one generated solution per task. It is not token Top-1 agreement or best-of-n.

All size and BPW rows exclude MTP and drafter weights. Effective BPW is base-model weight bytes × 8 divided by the common 284,334,567,511-parameter base-model denominator.

## DwarfStar artifact and runtime identity

The DwarfStar score is not the direct Unsloth `UD-IQ2_XXS` model under another name. It was generated from `antirez/deepseek-v4-gguf@1cd7b564460821938add0475a60b942c409295e0`, file `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, SHA-256 `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`. The single base-model GGUF is 86,720,111,488 bytes. It uses `IQ2_XXS` routed-expert gate/up tensors, `Q2_K` routed-expert down tensors, and higher-precision non-routed tensors.

That artifact ran target-only with no MTP through the Entrpi DwarfStar/DS4 engine, commit `72ae4bfa43d47b53f1ac781e673a088c6051af33`. Its runtime reported compressed FP8 primary KV plus a packed FP4 indexer cache. The direct Unsloth IQ2 and IQ3 rows instead used llama.cpp with F16 K/V. The DwarfStar row is therefore a measured model-plus-runtime result, not an isolated weight-quantization comparison.

## Direct Unsloth IQ2/IQ3 4K-cap audit

All 164 requests completed for both direct Unsloth rows; no unresolved timeout, dropped-request, HTTP-error, or context-error row entered either score. The matched IQ2 run used `--parallel 4 --ctx-size 32768` (8,192 tokens per slot, F16/F16 KV) and produced seven 4,096-token length stops before any visible solution: `HumanEval/1`, `HumanEval/76`, `HumanEval/116`, `HumanEval/129`, `HumanEval/130`, `HumanEval/132`, and `HumanEval/145`. Those seven are exactly its HumanEval failures. Ten additional tasks failed only the strict HumanEval+ tests: `HumanEval/32`, `HumanEval/39`, `HumanEval/86`, `HumanEval/91`, `HumanEval/99`, `HumanEval/124`, `HumanEval/125`, `HumanEval/134`, `HumanEval/141`, and `HumanEval/151`.

The replaced IQ2 run used 34,816 total context tokens (8,704 per slot) and produced six length stops: `HumanEval/32`, `HumanEval/47`, `HumanEval/116`, `HumanEval/129`, `HumanEval/132`, and `HumanEval/145`. Moving IQ2 to the common 8,192-token-per-slot rail preserved its 157/164 HumanEval count but changed the task-level outcomes and reduced strict HumanEval+ from 151/164 to 147/164. IQ3 remains unchanged at 155/164 HumanEval and 148/164 HumanEval+, with eight 4,096-token length stops before any visible solution. These single-run orderings are not evidence that either quant is intrinsically higher quality.

## Mia/0xSero REAP-K216 EXL3 3.0 audit

The Mia row is a physical 164/164 generation and EvalPlus terminal, not a projected or borrowed score. It used the exact K216 artifact manifest `ea8522d22abbbb91f9bb992884e5b1e546ff86336d17b2a64fe95b00157ed6d4` and HF revision `22f28d32b9b29b4352eaa380ff8c2c170b2847ab` target-only on one DGX Spark, with thinking enabled through the model's standard chat-template default. DSpark, speculative decoding, and ABLATE were disabled.

All 164 audited responses were retained exactly once, but `message.content` was null or empty for every task; the reasoning field was never scored. There were 128 length stops. EvalPlus therefore measured 0/164 HumanEval and 0/164 HumanEval+, which is the row reported above. The public-safe machine receipt, including canonical generation, raw EvalPlus, runtime, scorer, and artifact hashes, is [`mia-0xsero-reap-k216-exl3-0731.json`](mia-0xsero-reap-k216-exl3-0731.json).
