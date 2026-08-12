# BALANCED64 V1 validation and reproduction protocol

## Scope

BALANCED64 compares quantized DeepSeek-V4-Flash-0731 candidates with their own
FP8 teacher on a frozen set of 64 windows. The compact competitive receipt covers
`UD-IQ2_XXS`, `UD-IQ3_XXS`, `UD-IQ4_XS`, `DwarfStar-Q2-0731`, corrected all-43
QTIP2, deterministic mixed QTIP2.5, and exact uniform QTIP3.

The paired global metrics are:

- **KLD:** `KL(teacher || candidate)`, lower is better.
- **Top-1 agreement:** teacher and candidate argmax token IDs agree on the same
  ordered support, higher is better.

It is not a generative benchmark and it is not HOLDOUT512.

## Frozen public basis

The executable authority is
[`../configs/balanced64-v1.json`](../configs/balanced64-v1.json).

| Field | Value |
|---|---|
| Public suite-lock SHA-256 | `d5610f11c23b75f81e196e74407cb7e642a4f4a2e12f55925e13e5a7fe43ffb9` |
| Historical source-suite SHA-256 | `7f756b898aea80cb4dd9320da4cd0c855f258d055f62ef6c37151d27857fa0ad` |
| Historical source-window SHA-256 | `5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd` |
| Canonical 512-window provenance SHA-256 | `facae84e7fe7744a424b8978b697770a2642ae368c667427c03fcfa9fc143bbe` |
| Public 64-window population SHA-256 | `24089eea1b3e5650265b971930571dbf249aba0b2f62e954a9628dcbfd182f09` |
| Corrected public class-map SHA-256 | `c9ccf14df02d8f5d41508bb2e6e9c9525f1ea1d16533f2d65398d708a6ac9aaa` |
| Teacher bank | `TEACHER_0731_BALANCED64_V2` |
| Teacher source-model index SHA-256 | `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` |
| Windows | 64 ordered windows |
| Positions | 1,024 per window; 65,536 total |
| Support | teacher top 8,192 token IDs per position |
| FP | `FP8 e4m3 dynamic own-base teacher` |
| Base-model parameter count | `284334567511` parameters |
| Comparison/publication BPW denominator | Canonical base-model logical parameter count (`284334567511`) for every artifact |
| Auxiliary-inclusive BPW denominator | Base plus separately shipped auxiliary-model logical parameters; secondary accounting only |
| Classes | agentic 19, chat 7, code 9, multilingual 10, prose 10, reasoning 9 |

The tracked lock contains all 64 `(ordinal, window_id, source_class)` triples.
Its digest is recomputed from UTF-8 canonical JSON and also hard-pinned in the
helper. Passing a different but self-consistent lock fails.

### Retired class map

The protected `610e13dd…` map assigned 52 of the 64 windows to stale classes.
It is retained only as an invalid historical identity. It must not be used for
subgroup results. The corrected classes above come from projecting the same
window IDs through the canonical 512-window provenance. Global 65,536-position
KLD and Top-1 values are unchanged by regrouping.

## Metric definition

For each scored position, let `S` be the ordered 8,192-token support selected
by the teacher. Gather teacher and candidate log-probabilities on exactly `S`,
then renormalize both vectors on `S`:

```text
log_p = teacher_logprob[S] - logsumexp(teacher_logprob[S])
log_q = candidate_logprob[S] - logsumexp(candidate_logprob[S])
p = exp(log_p)
KLD = sum_i p_i * (log_p_i - log_q_i)
```

The standardized row contract stores every per-position KLD as the shortest
round-trip decimal representation of an IEEE-754 binary64 value. Values are
visited in ascending window ordinal and then position order. Global KLD is:

```text
math.fsum(all 65,536 ordered binary64 values) / 65,536
```

Negative or non-finite KLD values are rejected; there is no clamp. This exact
reduction applies to new standardized rows. The compact historical KLD fields
remain protected-source claims until their per-position rows are supplied.

Top-1 agreement is:

```text
argmax_first(log_p) == argmax_first(log_q)
```

Both arms use deterministic first-index tie breaking. This is agreement on the
shared teacher support, not unconstrained candidate-vocabulary Top-1.

## A. Validate the compact published result

```bash
git clone https://github.com/my-other-github-account/banana-smasher.git
cd banana-smasher
python3 -m Evals.tools.receipts verify \
  Evals/results/deepseek-v4-flash-0731-balanced64-v1.json \
  --suite-lock Evals/configs/balanced64-v1.json
```

Expected KLD ranking, from lower to higher:

```text
UD-IQ4_XS > QTIP3-uniform-exact > UD-IQ3_XXS > QTIP2.5-all43-FF0731 > UD-IQ2_XXS > DwarfStar-Q2-0731
```

Expected Top-1 ranking, from higher to lower:

```text
UD-IQ4_XS > QTIP3-uniform-exact > QTIP2.5-all43-FF0731 > UD-IQ3_XXS > UD-IQ2_XXS > DwarfStar-Q2-0731
```

This validates tracked structure, suite-lock consistency, Top-1/GB arithmetic,
comparison and separately labeled auxiliary-inclusive BPW arithmetic,
denominator/FP consistency, SHA-256 syntax, replay-status honesty,
and rankings. For the QTIP rows it also checks six-class position and Top-1 sums,
integer-derived class rates, weighted class KLD, exact component-byte sums, and
candidate/teacher/scorer/population bindings. It does not authenticate or
retrieve protected source receipts, and it does not recompute KLD from protected
per-position payloads.

## B. Reaggregate standardized per-window receipts

This path is runnable only if all 64 row receipts are supplied independently.
Each file must match
[`../schemas/balanced64-window-v1.schema.json`](../schemas/balanced64-window-v1.schema.json)
and the exact public lock.

For every frozen window:

1. Bind `suite_lock_sha256`, teacher source index, and one stable candidate
   artifact manifest/tree SHA-256.
2. Preserve the exact ordinal, window ID, corrected class, and 1,024 positions.
3. Emit 1,024 `kld_values`, each as Python's shortest round-trip binary64
   `repr`; do not emit a pre-summed window KLD.
4. Emit the integer Top-1 match count for the window.
5. Reject missing, negative, non-finite, truncated, or fallback-produced rows.

Start from [`../templates/balanced64-window-v1.json`](../templates/balanced64-window-v1.json).
The one-position template is illustrative; a valid BALANCED64 row has exactly
1,024 values and `positions: 1024`.

Place only the 64 rows in one directory, then run:

```bash
python3 -m Evals.tools.receipts aggregate work/balanced64-windows \
  --suite-lock Evals/configs/balanced64-v1.json \
  --output work/balanced64-aggregate.json
```

The helper rejects duplicate JSON keys, missing/duplicate ordinals, window or
class drift, suite/teacher/candidate basis drift, wrong position counts,
noncanonical binary64 strings, negative/non-finite KLD, and invalid Top-1
numerators. It reports corrected-class and global aggregates.

## C. Full GPU measurement replay boundary

The seven published measurements are **not** currently end-to-end
replayable from a clean public clone:

- protected teacher-bank payloads and corpus text are not distributed;
- protected source-receipt hashes are identifiers, not authenticated downloads;
- the historical scorer implementation and numeric environment are not present
  on canonical `main`;
- IQ3 and IQ4 lack public candidate repository revisions and artifact
  manifest/tree digests.

The machine receipt marks every row `replay.status: blocked` and names its
blockers. Do not describe compact validation or row reaggregation as physical
model replay.

For a future evaluation to claim full replay, publish or provide retrievable:

1. Candidate and teacher repository revisions plus artifact/tree manifests.
2. Corpus/window payloads or a lawful deterministic fetch/build procedure.
3. Scorer source revision and runnable command.
4. Numeric dtype, logsumexp implementation, support construction, tie policy,
   KLD serialization/reduction, and negative-value policy.
5. Hardware/software environment and fail-closed capability checks.

When the canonical `smash bank` / `smash evaluate` producer lands, add its exact
revision and commands in a new protocol/result version. Do not retrofit an
unversioned producer onto this historical receipt.
