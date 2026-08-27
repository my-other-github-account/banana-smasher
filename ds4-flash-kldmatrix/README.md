# DS4 Flash KLD matrix

This repository records a measured quantization, repair, evaluation, and serving
campaign for DeepSeek V4 Flash. It keeps evidence classes that are easy to
conflate separate: paired teacher-rail quality, predicted quality, artifact
footprint, synchronized kernel timings, request-level served throughput, and
upstream packaging status.

Evidence cut: **2026-07-18 10:00 PDT**. Sections labeled “2026-07-17 cutover”
are historical snapshots; later sealed rows are called out separately rather
than silently rewriting the cutover narrative.

## Headline results

At the 2026-07-17 cutover, the repaired IQ3-size COMBO artifact measured
`KL(ref || candidate) = 0.077061` (receipt-exact `0.07706103515625`) at
`101.95 GB` on the paired 512-window rail. A separate V2 fine-tuning run
measured `0.076285939453125`, but inherited an evaluation-trained warm start
and is therefore confirmation rather than a zero-leakage claim.

A post-cutover R5 rail measured a lower KLD, `0.07506409375`, with top-1
agreement `0.916774736328125` over the same 512 windows and 524,288 positions.
Its receipt reports `193.063787137 GB` for the runtime-composed pack, so it is
**not** an IQ3-size/T1 fit claim. “Lowest measured KLD in this campaign” and
“best quality at the 101.95 GB envelope” are different statements.

Quality is also not speed. The learned-VQ serve progressed from `2.1439`
token/s on the generic path to `6.5891` after grouped batching at the cutover.
After cutover, an opt-in CUDA warp-GEMV candidate measured `14.13` token/s over
a full 4,096-token stream (`14.93` median on 5×64). The actual V4-step32
product overlay then measured `13.91` token/s sustained over 4K and `13.96`
median on 5×64, clearing the campaign's `10 token/s` gate. The kernel has a
hash-pinned wheel/source and independent validation, but it was not published
as an upstream package in the audited evidence set.

## Architecture at a glance

```text
source corpus + tokenizer
        │
        ├── DS4 eval/calibration manifests
        │
source teacher ──> top-8192 teacher rail
        │                    │
quantize / solve backpack    │
        │                    │
repair or code reassignment  │
        └──> candidate rail ─┴──> paired KLD + task checks
                       │
                       └──> packed serving wire
                                  │
                                  ├── kernel microbench
                                  ├── served decode A/B
                                  └── ToolBench parity gate
```

The package builder, quality rail, serving runtime, and behavioral evaluator
are separate systems with separate receipts. A pass in one class is never
promoted into another class without a corresponding measurement.

## Quick receipt check

The small public metadata can be syntax-checked without model weights:

```bash
python -m json.tool out/DS4_CORPUS_META.json >/dev/null
python -m json.tool out/DS4_CALIB_META.json >/dev/null
python -m json.tool out/MMLU_QUESTION_SET_DS4.json >/dev/null
```

This verifies file integrity/shape only. Recomputing KLD still requires the
licensed model assets, omitted teacher rail, scorer, and exact candidate pack
described in `docs/PIPELINE.md`.

## Documentation

- [End-to-end artifact and measurement pipeline](docs/PIPELINE.md)
- [Measured results ladder and receipt map](docs/RESULTS_LADDER.md)
- [Evaluation and hosted-baseline parity](docs/EVAL_PARITY.md)
- [Serving performance investigation and packaging gap](docs/SERVING.md)
- [Operational failures and prevention rules](docs/RUNBOOK_PITFALLS.md)

The historical matrix remains in `SCOREBOARD.md`; the original cutover summary
is `RESULTS.md`, and later operational seals are indexed in `LIVE_STATE.md`.
Large model payloads and several campaign receipts are not mirrored here. The
documentation names those omissions explicitly rather than claiming that a
fresh checkout can reproduce every forward pass.

## Evidence policy

A publishable model-quality row needs immutable candidate identity, common
corpus and tokenizer hashes, teacher/scorer pins, window coverage, and a
receipt path. A publishable speed row additionally needs exact server/kernel
revisions, launch flags, request shape, warmup, dispatch proof, and repeated
served timing. Predictions, small gates, and kernel microbenchmarks remain
useful, but are never relabeled as full-corpus quality or served throughput.

The audited publication set does not contain a full teacher-payload digest;
that is a provenance gap, not a hash to infer from the corpus or scorer.
Receipt filenames and hashes for non-public campaign assets are retained in the
results and serving documents so the evidence can be reconciled without
publishing private host paths.

## Boundary with Atlas AutoKernel

This DS4 campaign ran on a different model/runtime surface from the Atlas
AutoKernel target. Its VQ warp-GEMV and serving lessons are transferable
engineering evidence, but none of the DS4 token/s rows count toward Atlas's
spark-6 success gate. The governing Atlas state at this evidence cut remains:
`27.975698` DFlash token/s versus a `30 token/s` gate, with zero of three target
HF-kernel publications completed. Atlas success still requires measured served
throughput on spark-6 and packageable upstream kernels in the target runtime.
