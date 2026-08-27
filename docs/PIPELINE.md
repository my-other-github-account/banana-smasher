# Reproducing the DS4/KLD pipeline

This document describes the artifact and evaluation contracts used by this
repository. It is intentionally stricter than a one-line benchmark recipe:
results are comparable only when the corpus, tokenizer, teacher support,
position cutoff, candidate bytes, and scorer are all identified.

The pipeline produces **quality evidence**. It does not by itself prove that a
candidate fits a deployment envelope, uses a fast kernel, or improves served
tokens per second.

Evidence cut: **2026-07-18 10:00 PDT**. The pipeline was assembled from the
2026-07-17 campaign state and reconciled against later R4/R5 and serving seals.
`$CAMPAIGN_RECEIPTS` used by this documentation is a sanitized logical name for
the internal hash-bound receipt archive; it is not an environment variable
supplied by this checkout.

## Evidence classes

Keep these classes separate in receipts and result tables:

1. **Paired KLD quality** — teacher-forced `KL(ref ∥ candidate)` on the same
   windows and scored positions.
2. **Estimated or predicted quality** — solver output or a projection from a
   smaller gate. It is useful for ranking candidates, not for sealing a row.
3. **Model footprint** — artifact bytes or effective bits per weight. This is
   not a speed measurement.
4. **Kernel microbenchmark** — synchronized time for a named kernel and tensor
   shape. This is not end-to-end serving throughput.
5. **Served throughput** — request-level prefill/decode tokens per second from
   a running server, with the complete launch and request configuration.

## Repository artifact contract

The following repository-local receipts define the common evaluation surface.
They are source evidence for this documentation; a standalone publication must
include them at the same relative paths:

| Contract | Receipt | Sealed value |
| --- | --- | --- |
| Evaluation corpus | `out/DS4_CORPUS_META.json` | 512 windows; MD5 `1701920b4ba96dea0b18fe9df0151876` |
| Calibration corpus | `out/DS4_CALIB_META.json` | 1,024 windows; MD5 `d09b006997b1843f041bf70c72ab695d` |
| Evaluation tokenizer | `out/DS4_CORPUS_META.json` | DS4 tokenizer MD5 `3f75dbea81fe67dd8c07843bdf9ce36e` |
| MMLU question set | `out/MMLU_QUESTION_SET_DS4.json` | 500 questions; SHA-256 `24d60b46aa7d0268b5f230760f3caa1391211fdd2893c9073c9e037135b4443a` |
| MMLU source archive | `out/MMLU_QUESTION_SET_DS4.json` | MD5 `20bb207676c1f58dc70afc9267cd206c` |
| Position convention | `out/DS4_CORPUS_META.json` | first `min(1024, real_len - 1)` positions per window; 524,288 positions total |
| Teacher support | `SCOREBOARD.md` | reference top-8,192 support, renormalized |
| Sealed scorer | `SCOREBOARD.md` | `kld_score.py` MD5 prefix `8011368c`; `KL(ref ∥ candidate)`, reference support, both sides renormalized |
| Teacher payload digest | not present in the publication set | **PROVENANCE GAP:** the audited rows pin corpus, support convention, scorer, and per-row ledgers, but no immutable full teacher-rail MD5 was recoverable. Add the teacher-ledger digest before claiming turnkey reproduction. |

Historical harnesses in `src/` still assume that their source corpus and
model assets have been staged outside this repository. They are evidence of
the procedure, but they are not yet a portable downloader. External users
must supply those licensed/source assets and replace the local staging lookup
before running the builders.

## Stage 1 — build immutable corpus splits

The evaluation and calibration builders decode the source windows with the
source tokenizer and re-encode the resulting text with the DS4 tokenizer.
They do not add special tokens.

```bash
python src/build_ds4_corpus.py
python src/build_ds4_calib_corpus.py
```

Expected outputs:

```text
out/windows_ds4_eval.json
out/DS4_CORPUS_MANIFEST.jsonl
out/DS4_CORPUS_META.json
out/windows_ds4_calib.json
out/DS4_CALIB_MANIFEST.jsonl
out/DS4_CALIB_META.json
out/CALIB_SELECTION.json
```

Required gates:

- evaluation has exactly 512 windows;
- calibration has exactly 1,024 windows;
- the metadata hashes match the table above;
- calibration/evaluation disjointness is inherited from the document-level
  source split and is recorded in `DS4_CALIB_META.json`;
- the DS4 tokenizer hash matches the live candidate tokenizer;
- every candidate and teacher sees the same token IDs.

Caveat: conversational windows retain the source model's chat-template surface
forms as literal text. This is uniform across the paired KLD rows, so deltas
remain valid, but it is not a DS4-native dialogue re-render.

## Stage 2 — build the top-8,192 teacher rail

For each evaluation window, run the source teacher in teacher-forced mode and
store the top-8,192 token IDs and log-probabilities for every scored position.
A generic invocation has this shape:

```bash
python t8192_ds4_build_v3.py \
  --mode bf16 \
  --local-dir "$MODEL_DIR" \
  --meta-dir "$MODEL_DIR" \
  --corpus out/windows_ds4_eval.json \
  --out "$RAIL_DIR/t8192_eval" \
  --start 0 --count 512 --chunk 64 --mb 4 \
  --tag teacher_eval
```

`t8192_ds4_build_v3.py` is part of the serving/evaluation stack referenced by
`SCOREBOARD.md`; it is not currently included in this repository. A fully
standalone release must vendor or replace it before claiming one-command
reproduction.

The teacher rail is valid only if:

- `DONE.jsonl` covers every window exactly once;
- every payload hash matches its ledger row;
- the teacher readback scores `KL = 0` and top-1 agreement `= 1` against itself;
- the mean teacher NLL is lower than the quantized serve anchor;
- the corpus and scorer identities are present in the receipt.

## Stage 3 — measure anchors, then solve the backpack

Measure each candidate tier on the common teacher/corpus surface before asking
a solver to choose a package. An anchor row should bind:

- layer, projection, and expert/unit identity;
- quantization tier and codebook/scalar-grid identity;
- packed bytes and effective bpw;
- measured local damage or a clearly labeled proxy;
- corpus, teacher, tokenizer, and scorer identities.

The budgeted solver selects one tier per unit under the byte envelope and emits
an assignment manifest. Its aggregate KLD is **PREDICTED**: it assumes a damage
model and cannot include every cross-unit interaction. Use the prediction to
rank packages and decide which one to build; only a paired rail can seal
quality.

A candidate is not defined by a nickname such as “IQ3” or “GPTQ.” It is the
combination of:

- an immutable base manifest;
- per-layer/per-expert tier assignments;
- packed codes;
- codebooks or scalar lookup tables;
- scales;
- any dense-side repair parameters;
- a target manifest hash and payload hashes.

## Stage 4 — materialize the exact target pack

The incremental/fastbuild path is an identity-preserving acceleration, not a
new artifact type. A delta contains only assignments that differ from a named
base manifest; unchanged rows remain hard-linked or mmap-backed by the base.
The completion receipt must bind both base and target manifest hashes, enumerate
every changed row, and prove the materialized pack matches the target.

**Never overlay a changed-row delta onto a pack that is already complete.** A
valid delta is applied exactly once to its named base. Reapplying it to a
complete target, applying it to a different base, or exporting from the base
without materializing the delta mixes assignment identities. That failure mode
produced an invalid prerepair artifact and a meaningless KLD regression during
the campaign.

Required fastbuild gates:

1. base-manifest hash equals the delta receipt's base hash;
2. delta target hash equals the solver's target manifest hash;
3. changed-row count and per-tier counts match the delta ledger;
4. unchanged rows are byte-identical to the base;
5. changed rows are byte-identical to the registered target payloads;
6. a `COMPLETE` sentinel is written only after full hash readback;
7. downstream export asserts the **target** identity, never just the base.

## Stage 5 — apply repair or calibrated-code overlays

### Repair after solve (COMBO)

The COMBO mechanism trained codebooks, all RMSNorm parameters, and attention
output gains while preserving the package envelope. The historical workflow
under `combo_v2_*/code/` demonstrates the guarded sequence:

1. verify the host/resource claim and reject GPU co-tenancy;
2. verify source manifests, checkpoints, corpora, teacher rows, and hashes;
3. prepare the requested train/eval corpus view;
4. train without modifying the immutable base artifact;
5. export into a new directory and record checkpoint ancestry;
6. re-read exported bytes and write an atomic completion receipt.

The checked-in launcher is a historical campaign launcher with task-specific
paths. Treat it as a reference implementation of the guards, not as a public
copy-and-run command.

### Repair before solve (tier re-encode)

A learned codebook is valid only with the codes and scale convention for which
it was trained. The tier-repair arms trained codebooks in their own code space;
only 15.8% of sampled assignments overlapped the target-bin codes. A
codebook-only swap onto foreign codes is therefore not semantics-preserving.

The valid transplant is:

1. load the original higher-precision weights;
2. install the selected repaired codebook checkpoint;
3. re-encode codes against that fixed codebook;
4. refit scales using the same reconstruction convention;
5. verify checkpoint step/hash and target-manifest identity;
6. run a small paired gate before a full solve or rail.

The early two-tier and three-tier gates translated about 47–49% of the local
repair prediction into backpack KLD improvement. The campaign rounded this to
an empirical **0.48 translation coefficient** when pricing repaired tiers in
pre-reanchor solves. It is a measured planning coefficient for this regime,
not a universal codebook constant and not a substitute for a rail.

### GPTQ/code-reassignment overlay

The R2 calibrated-code path changed assignments while preserving the existing
wire format, codebook/scalar values, group geometry, and package envelope. Its
receipt must identify the exact units covered, calibration corpus, Hessian or
error-feedback solver settings, source/target manifests, and overlay hashes.

The measured 64-window result originated as a five-layer pilot; the later
sealed overlay counted 2,026 recalibrated units. It never became a complete
43-layer/full-bin canonical package with a 512-window rail. Preserve those two
facts together rather than promoting the gate into a full-model row.

## Stage 6 — candidate forward and paired KLD

Run the candidate over the same 512 windows and produce the same top-8,192
payload shape as the teacher. Then score:

```text
KL(ref || candidate)
```

on the reference top-8,192 support after renormalizing both distributions.
Aggregate by scored positions, not by an unweighted mean of chunk means.

A sealed row must record at least:

```json
{
  "row_id": "stable descriptive name",
  "measurement_status": "MEASURED",
  "corpus_md5": "...",
  "n_windows": 512,
  "n_positions": 524288,
  "pos_cutoff": 1024,
  "support_size": 8192,
  "kl_vs_teacher": 0.0,
  "top1_agree": 0.0,
  "manifest_md5": "...",
  "candidate_ledger_md5": "...",
  "score_ledger_md5": "...",
  "bytes_used": 0
}
```

Use `MEASURED` only after the full candidate payloads have been generated and
scored. A 16- or 64-window gate remains a gate even if its number is favorable.

Post-cutover examples make the size binding requirement concrete:

- the R4 three-tier runtime composition sealed at `0.091723068359375` over 512
  windows, but its audited result row did not seal a package footprint, so it
  cannot by itself claim the `≤101.95 GB` target;
- the R5 step-35 composition sealed at `0.07506409375` over the same 512-window
  surface, but its receipt reports `193.063787137 GB`; it is a measured quality
  win at a different operating point, not an IQ3-size win.

The corresponding result receipts are identified by SHA-256 in
`docs/RESULTS_LADDER.md`. They are not mirrored in the public checkout.

## Stage 7 — task-quality cross-checks

KLD and task evaluation answer different questions. For MMLU-500, use the
protocol in `docs/EVAL_PARITY.md` and preserve per-question rows. Checked-in
examples include the top-level R1 row and the MMLU ladder files under `out/`:

```text
out/R1_MMLU500_ROW.json
out/mmlu_ladder_*/MMLU_LADDER.json
```

Do not replace a paired KLD row with MMLU, and do not infer task parity from a
small KLD change. Report both when both exist.

## Stage 8 — serving and throughput

Only after candidate byte identity and quality are established should the
artifact be loaded into a serving stack. Serving requires a separate receipt:

- server and kernel commits;
- complete launch flags and environment;
- tokenizer/template/tool-parser configuration;
- context, concurrency, KV-cache, speculative-decoding, and memory settings;
- request prompt and output-token counts;
- warmup policy and synchronized timing;
- prefill and decode throughput separately;
- proof that the requested fast kernel path executed.

See `docs/SERVING.md`. A good KLD result is not a served-throughput result.

The serving sequence eventually produced a hash-pinned CUDA warp-GEMV wheel
and a served 4K A/B, but that does not retroactively make every quality pack a
served artifact. Bind a serving row to the exact wire/overlay and runtime
hashes. In particular, the post-cutover 13.91 token/s product row belongs to the
V4-step32 serving overlay, not to the 193 GB R5 quality pack.

## Minimal verification commands

These commands validate the repository-local metadata without recomputing the
model forwards. They require the small receipt bundle named above; documentation
alone is insufficient:

```bash
python -m json.tool out/DS4_CORPUS_META.json >/dev/null
python -m json.tool out/DS4_CALIB_META.json >/dev/null
python -m json.tool out/MMLU_QUESTION_SET_DS4.json >/dev/null
python -m json.tool out/R1_NLL_ROW.json >/dev/null
python -m json.tool sealed_rows/VQ3_MIXED_100G_MEASURED_ROW.json >/dev/null
```

On macOS, verify the immutable corpus with:

```bash
md5 -q out/windows_ds4_eval.json
```

On Linux, use `md5sum` instead. The expected digest is
`1701920b4ba96dea0b18fe9df0151876`.

## What is still missing for a turnkey public reproduction

This repository preserves many receipts and historical harnesses, but it does
not yet include every licensed/model asset or every teacher/candidate forward
helper. Until those components are vendored or replaced, the honest claim is
“documented and receipt-backed campaign pipeline,” not “single-command public
reproduction from a fresh checkout.”

## Packaging and target-runtime boundary

For an upstream kernel deliverable, add one more stage after served A/B:

1. freeze a public ABI and supported shape/dtype matrix;
2. publish source and deterministic build metadata;
3. run independent correctness, fallback, and edge-shape tests;
4. integrate opt-in dispatch with an on-path sentinel;
5. run same-artifact served A/B from the packaged build;
6. land review/CI in the target runtime or kernel registry.

The DS4 CUDA warp candidate reached steps 1–5 in campaign form but the audited
evidence contains no merged/published upstream artifact. It also targets a
different runtime from Atlas AutoKernel. Atlas's success condition remains a
measured spark-6 served win and packageable upstream kernels on that target;
DS4 quality rows, DS4 token/s, and this documentation package do not satisfy
that gate.
