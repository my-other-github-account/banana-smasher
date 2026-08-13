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
- above-chance MMLU per BPW and raw MMLU per BPW;
- above-chance MMLU per complete decimal artifact GB;
- density relative to the official DeepSeek-V4-Flash-0731 reference scored on
  this identical bank;
- model repository revision and complete artifact member hashes.

### Public density metrics

```text
Above-Chance MMLU per BPW (within model) = (MMLU percentage - 25) / exact base-equivalent comparison BPW
Raw MMLU per BPW = MMLU percentage / exact base-equivalent comparison BPW
Above-Chance MMLU per GB = (MMLU percentage - 25) / complete decimal artifact GB
```

The primary BPW unit is **MMLU percentage points above random guessing per base-equivalent BPW**. Compute with exact machine-readable denominators, never rounded display values. BPW density is comparable only within variants sharing the same base-model parameter denominator; use complete bytes/decimal GB and Above-Chance MMLU per GB for cross-model storage comparisons.
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

## Kimi-K3 IQ1S one-file layerwise reproductions

The two pinned standalone runners reproduce the published Kimi rows in
`kimi-iq1s-results.json`:

```bash
python reproduce_unsloth_layerwise.py \
  --model-dir /models/Kimi-K3-GGUF/UD-IQ1_S \
  --binary-dir /path/to/sparkinfer-k3-mmlu-bin \
  --output-dir ./mmlu500-unsloth

python reproduce_neuron_layerwise.py \
  --model-dir /models/Kimi-K3-Neuron-IQ1S-GGUF \
  --binary-dir /path/to/sparkinfer-k3-mmlu-bin \
  --output-dir ./mmlu500-neuron
```

Run them in an environment containing exactly `tiktoken==0.12.0`. Each file
fetches and authenticates this frozen 500-row bank and the pinned Kimi tokenizer,
checks every local GGUF member against its pinned size and SHA-256, carries the
hidden state plus residual-checkpoint bank through layers 0–92 one resident
range at a time, scores only token IDs 32–35, and independently aggregates the
result. Full runs pass only at the published `412/500` (Unsloth) or `342/500`
(Neuron) score. Use `--prepare-only` for a weight-free basis/tokenizer check and
`--limit N` for a compute smoke prefix.

`--binary-dir` must contain the three CUDA executables used by the sealed runs:
`kimi_k3_prefix_dump`, `kimi_k3_boundary_advance`, and
`kimi_k3_boundary_score`, built against the K3 SparkInfer runtime. They are kept
external because embedding the native CUDA/C++ runtime would turn each small
runner into a cosmetic single-file archive rather than a readable script.
