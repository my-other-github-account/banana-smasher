# Fixed MMLU-500 capability-density bank

This directory freezes the permanent zero-shot MMLU subset used for Banana
Smasher capability-density comparisons beginning 2026-08-10. The bank was
sealed before any model was scored against it.

## Frozen identity

- Dataset: `cais/mmlu`
- Revision: `c30699e8356da336a370243923dbaf21066bb9fe`
- Config/split: `all/test`
- Population: 14,042 questions
- Sample: 500 questions
- Sampling seed: `banana-smasher-mmlu500-v1-2026-08-10`
- Sampling method: rank every pinned test row by
  `SHA256(seed + NUL + canonical_row_sha256)` and take the lowest 500
- Source Parquet SHA-256:
  `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`
- `items.jsonl` SHA-256:
  `df6704c4d02550b9155e106bc9a9e1bfe1164a663d509e41a76736bb60d01ded`
- `manifest.json` SHA-256:
  `2325d58687a0b5def7b48979a5886a9f7c5089c294445e885e0867101b07482d`
- Coverage: all 57 MMLU subjects
- Gold letters A/B/C/D: 116 / 131 / 127 / 126

The ordered items, prompts, choices, answers, source row indices, row hashes and
selection hashes are stored in `items.jsonl`. Do not redraw or reorder this bank
for future variants. If a protocol change becomes necessary, create a separately
versioned bank rather than modifying `mmlu500-v1`.

## Scoring

Each question is one zero-shot prompt forward. Score the single-token answer
choices A/B/C/D from the logits at the final prompt position. No generation and
no fourfold prompt repetition are required.

Every result reports at least:

- MMLU accuracy as `correct/500` and a percentage;
- complete model artifact bytes, decimal GB, and exact base-equivalent comparison BPW;
- MMLU intelligence density;
- density relative to the official DeepSeek-V4-Flash-0731 reference scored on
  this identical bank;
- model repository revision and complete artifact member hashes.

### Public intelligence-density metric

```text
MMLU intelligence density = (MMLU percentage - 25) / exact base-equivalent comparison BPW
```

The unit is **MMLU percentage points above random guessing per base-equivalent BPW**. Compute with the exact machine-readable BPW, never the rounded three-decimal display value.
MMLU has four choices, so the random baseline is 25%.

```text
relative density = candidate capability density / reference capability density
```

The reference is `deepseek-ai/DeepSeek-V4-Flash-0731`, measured through the same
prompt, tokenizer and scoring contract. Unsloth Q8 is a comparator, not a silent
replacement for the official reference.

Always publish absolute MMLU and complete size next to density. Density alone can
favor a very small model despite unacceptable absolute capability.

## Unsloth source surface

`unsloth-0731-variants-v1.json` freezes all 13 quantized variant directories in
`unsloth/DeepSeek-V4-Flash-0731-GGUF` revision
`fbbb5b93fb787c21338159b0af3318bb3f4d9768`, including every GGUF member's byte
count and LFS SHA-256, complete decimal GB and common-denominator BPW.

## Rebuild

Download the pinned Parquet file and run in an isolated environment containing
DuckDB 1.4.3:

```bash
python tools/mmlu_density/build_mmlu500_manifest.py \
  --parquet /path/to/all-test.parquet \
  --output-dir notes/benchmarks/mmlu-density/mmlu500-v1
```

A valid rebuild must reproduce both frozen file hashes above byte-for-byte.
