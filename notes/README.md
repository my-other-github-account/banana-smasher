# Reports and evidence

This directory is the public reporting surface for Banana Smasher. It contains curated human-readable reports, comparison tables, migration notes, and compact scrubbed receipt summaries. Keep each artifact under `notes/`; do not place reporting material beside product code.

## Evidence rules

- Bind each claim to a model/artifact basis and same-work receipt hashes.
- Separate measured results from estimates and pending hardware gates.
- Prefer aggregate metrics and reproducible commands over raw logs.
- Record failures and rejected comparisons; do not present a missing gate as a pass.
- Mark newly added measurements as `NEW`, and leave the owner as `TBD` until a named public maintainer accepts it.
- Every comparison row must state method, exact basis, KLD, top-1, GB, packed bpw, floating-point format, instrument, sample count (`n`), status, and source. Every row with a sealed MMLU score must also report `MMLU/GB`; pending MMLU rows use a blank cell, never a projection. Record FP8 as 8 bits.
- Keep raw private receipts, machine paths, host identities, task identifiers, credentials, and internal orchestration outside Git.
- Do not add model weights, generated packs, checkpoints, or other large binary payloads.

Use `report-template.md` for new performance or quality reports.

## OpenRouter HumanEval

**Owner:** TBD · **Status:** independently verified same-contract measurement

**NEW** GLM-5.2 through Z.AI's pinned FP8 OpenRouter endpoint scored **163/164 (99.39%) HumanEval** and **157/164 (95.73%) HumanEval+** under the frozen 164-task, medium-reasoning, 16,384-token EvalPlus contract. The historical FF0731 provider rail scored 160/164 and 153/164 respectively. Provider and wall-clock matching were intentionally not required; see [`humaneval/glm-5.2-openrouter-zai-fp8.md`](humaneval/glm-5.2-openrouter-zai-fp8.md) and its adjacent machine-readable receipt for the exact scope and hash bindings.

## Kimi K3

**Owner:** TBD · **Status:** independently verified same-bank measurements

Both rows use the immutable `mmlu500-v1` bank: 500 ordered literal zero-shot prompts, no chat or thinking wrapper, no generation, and final-position A/B/C/D logits for token IDs 32/33/34/35. KLD and separate Top-1 agreement were not measured for these MMLU-only rows.

| Complete Kimi-K3 artifact | MMLU | Gold CE (bits) | Complete bytes | Decimal GB | Base-eq BPW | MMLU/bit | MMLU/GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NEW** Neuron IQ1S | 342/500 (68.4%) | 1.1650300319915994 | 330167807328 | 330.167807328 | 0.9501464831956500 | 45.67716743425893 | 0.1314483091226546 |
| **NEW** Unsloth UD-IQ1_S | 412/500 (82.4%) | 0.7522193724299207 | 594040923616 | 594.040923616 | 1.709512199314206 | 33.57682970792885 | 0.0966263395636098 |

`MMLU/bit` is the repository's chance-adjusted intelligence-density metric, `(MMLU percentage - 25) / exact base-equivalent BPW`. `MMLU/GB` is `(MMLU percentage - 25) / complete decimal GB`. Base-equivalent BPW is `complete artifact bytes × 8 / 2,779,931,837,184`, using the official Kimi-K3 parameter total from `moonshotai/Kimi-K3@9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; it is an artifact-wide accounting ratio, not a claim that every tensor has one physical bit width.

Artifact revisions: Neuron `vcruz305/Kimi-K3-Neuron-IQ1S-GGUF@a2d6283870dd97d2f177c69d94fb18120e79fe65`; Unsloth `unsloth/Kimi-K3-GGUF@a0836360ce58dfec088d966a97f2ddc8a606279b`. Bank items SHA-256: `df6704c4d02550b9155e106bc9a9e1bfe1164a663d509e41a76736bb60d01ded`; literal ledger SHA-256: `7cdb6a0a93a3d613212ac9960666ee9a26256de86709aaa9fa5765cd1c91e8b4`.

Evidence bindings: Neuron final seal `a2af318426147736bb9836fd41b3322abbaa833402878375d12d5f386ecef694` and independent verifier `fba0e780ea11d3f6c14c176edce854d933fa246c846f19ef5e8d7624174f7960`; Unsloth public evidence `0fad0b7d591c8afc885c5a1d8e04ad50e5aba153653b4beff72f352d13166d7b`, independent verifier `111974faafaffaf204b33d04f32299822473243161f4e5ae5bc8b5c2addb2909`, and final pass `14b9d3e08fc3922f68b70e3dd95a5c090e31421d6cd6010b7ba950c82683b3c9`.
