# MMLU-500 Evals capability density

All four rows use the immutable `mmlu500-v1` bank: 500 ordered zero-shot literal prompts, no chat template or answer generation, and final-position A/B/C/D logits normalized over the four choices. Aggregates below were independently recomputed from the published per-question rows.

| Evals row | MMLU | MMLU % | Gold CE (bits) | Complete bytes | Decimal GB | Base-eq BPW | Capability density | Density vs Unsloth IQ4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unsloth IQ4 | 417/500 | 83.40% | 0.701727 | 136662446656 | 136.662446656 | 3.8451166272834685 | 0.42733026832895527 | 1.0000x |
| Unsloth IQ3 | 416/500 | 83.20% | 0.749950 | 104207848032 | 104.207848032 | 2.931978308348837 | 0.5584992023069897 | 1.3069x |
| Unsloth IQ2 | 409/500 | 81.80% | 0.842842 | 90860736928 | 90.860736928 | 2.556445745541928 | 0.6251325040981072 | 1.4629x |
| DwarfStar Q2 0731 | 403/500 | 80.60% | 0.809176 | 93691352992 | 93.691352992 | 2.6360875868777476 | 0.5934379024790847 | 1.3887x |

Capability density is `(MMLU percentage - 25) / complete decimal GB`. Relative density uses `Unsloth IQ4` as the fixed 1.0x reference. DwarfStar's denominator is the complete base-plus-drafter Evals payload even though the measured next-token logits come from the target/base model.

Machine-readable aggregates and evidence hashes are in [`results.json`](results.json). The exact model basis is [`four-row-mission-basis.json`](four-row-mission-basis.json), and the frozen prompts are [`items.jsonl`](items.jsonl).

Public-safe per-question records, tokenizer receipts, and complete physical model identities are in [`evidence/`](evidence/).

The public basis records source-scoring basis `83ace3f25a4f77325479690a47e7b86f7dee5ef44513996b551f24145ff88f8e` and its non-scientific public-copy transform; bank, model, and scoring fields are unchanged.
