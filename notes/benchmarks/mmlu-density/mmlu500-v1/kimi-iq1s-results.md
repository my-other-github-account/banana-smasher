# Kimi-K3 IQ1S on the immutable MMLU-500 bank

Both rows use the same frozen 500-question zero-shot literal-prompt bank as the physical Unsloth IQ4 reference. Each question is scored from final-position A/B/C/D logits with Kimi token IDs A=32, B=33, C=34, and D=35; no chat template, generation, drafter, or MTP result is included.

| Model / revision / variant | MMLU | MMLU % | Gold CE (bits) | Complete bytes | Decimal GB | Base-eq BPW | Complete-size intelligence density | Density vs Unsloth IQ4 | Scope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kimi-K3 Neuron IQ1S<br>`vcruz305/Kimi-K3-Neuron-IQ1S-GGUF` @ `a2d6283870dd97d2f177c69d94fb18120e79fe65`<br>`Neuron-IQ1S` | 342/500 | 68.40% | 1.165030 | 330167807328 | 330.167807328 | 0.95014648319565004035456873418826900283791328926667634504843132688331761993971133 | 0.1314483091226546 | 0.3076x | base model only / no drafter-MTP claim |
| Kimi-K3 Unsloth UD-IQ1_S<br>`unsloth/Kimi-K3-GGUF` @ `a0836360ce58dfec088d966a97f2ddc8a606279b`<br>`UD-IQ1_S` | 412/500 | 82.40% | 0.752219 | 594040923616 | 594.040923616 | 1.7095121993142056149365169369264937425174447366771137735818560164433767348427323 | 0.0966263395636098 | 0.2261x | base model only / no drafter-MTP claim |

Complete-size intelligence density is `(MMLU percentage - 25) / complete decimal GB`, in MMLU percentage points above chance per complete decimal GB. Relative density divides that quantity by the sealed same-bank physical Unsloth `UD-IQ4_XS` density, `0.42733026832895527`; the IQ4 reference remains `1.0x` and the Kimi rows are not normalized to each other.

Base-equivalent BPW uses one common exact denominator: `2,779,931,837,184` parameters from the pinned `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, authoritative Hugging Face model metadata field `safetensors.total`. BPW is `8 * complete_bytes / base_parameter_count`; active parameters and sparse resident bytes are not used.

The aggregates were independently recomputed directly from 500 qrows per model. Validation required ordinals 0–499 exactly once, frozen row order and answers, exact A/B/C/D candidate mappings, finite four-choice logits and probabilities, probability/softmax agreement, and the frozen Kimi token ledger. The Neuron and Unsloth qrows SHA-256 values are `bd3e7ee3006dc2120ec1e5cee09aff52c9995dcd0c333e8a0f7d572453ed5258` and `8d2514d7ee5f71a6c551d280b39e95a1bd4ae99afc42093b69cfcfde99391124`, respectively.

Machine-readable results are in [`kimi-iq1s-results.json`](kimi-iq1s-results.json), with public-safe source bindings in [`kimi-iq1s-evidence-manifest.json`](kimi-iq1s-evidence-manifest.json). This companion does not modify or replace the binding multi-row result table.
