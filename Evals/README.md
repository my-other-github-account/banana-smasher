# Banana Smasher evaluations

This folder is the public entry point for frozen evaluation contracts, executable
receipt checks, and sealed result tables.

## DeepSeek-V4-Flash-0731 BALANCED64

All rows below are complete 64-window measurements under the same contract:

- 64 ordered windows × 1,024 positions = 65,536 scored positions
- FP8 e4m3 dynamic own-base teacher
- teacher top-8,192 support
- `KL(teacher || candidate)`; lower is better
- teacher/candidate Top-1 agreement on the same ordered support; higher is better
- packed-wire size with one denominator: `284,334,567,511` parameters
- corrected class split: agentic/chat/code/multilingual/prose/reasoning = `19/7/9/10/10/9`

| Quant | FP | Decimal GB | Packed-wire bpw | KLD ↓ | Top-1 ↑ |
|---|---|---:|---:|---:|---:|
| Unsloth UD-IQ4_XS | FP8 e4m3 own-base | 136.662446656 | 3.845116627283469 | 0.0683488486737012 | 60,584/65,536 (92.44384765625%) |
| Unsloth UD-IQ3_XXS | FP8 e4m3 own-base | 104.207848032 | 2.931978308348837 | 0.17770788160865483 | 57,638/65,536 (87.9486083984375%) |
| Unsloth UD-IQ2_XXS | FP8 e4m3 own-base | 90.860736928 | 2.556445745541928 | 0.2767474104898907 | 55,422/65,536 (84.5672607421875%) |
| DwarfStar Q2 0731 | FP8 e4m3 own-base | 93.691352992 | 2.636087586877748 | 0.30952134732070036 | 54,845/65,536 (83.68682861328125%) |

These four BALANCED64 measurements are complete. Artifact-download metadata in
the machine receipt is replay provenance only; it does not make any measurement
row partial.

There is no admitted Banana Smasher row yet. A Banana row is added only after the
final pack is measured on this exact frozen population—not the separate
`train_balanced64` Anchor bank.

Machine result:
[`results/deepseek-v4-flash-0731-balanced64-v1.json`](results/deepseek-v4-flash-0731-balanced64-v1.json)

Frozen suite lock:
[`configs/balanced64-v1.json`](configs/balanced64-v1.json)

Full metric and row contract:
[`protocols/balanced64-v1.md`](protocols/balanced64-v1.md)

## Validate the published table

From the repository root:

```bash
python3 -m Evals.tools.receipts verify \
  Evals/results/deepseek-v4-flash-0731-balanced64-v1.json \
  --suite-lock Evals/configs/balanced64-v1.json
```

This recomputes the suite-lock identities, Top-1 rates, decimal GB, normalized
packed-wire bpw, shared denominator, and rankings. Historical KLD cannot be
recomputed without the protected per-position source rows.

## Apples-to-apples steps for a new 0731 quant

1. **Freeze identity and size.** Bind the exact candidate artifact revision/tree,
   every shipped quant payload byte, and the common parameter denominator. Never
   derive VQ bpw from an in-memory integer container.
2. **Use the exact frozen population.** Resolve the 64 ordered window IDs and
   corrected classes in `configs/balanced64-v1.json`. Reject missing, reordered,
   substituted, truncated, or fallback windows. Do not use `train_balanced64`.
3. **Use the same teacher and support.** Run the DeepSeek-V4-Flash-0731 FP8 e4m3
   dynamic own-base teacher and preserve its ordered top-8,192 token IDs and
   log-probabilities for the first 1,024 scored positions of every window.
4. **Score the candidate on the same positions.** Gather candidate
   log-probabilities on exactly the teacher support, renormalize both arms on that
   support, calculate `KL(teacher || candidate)`, and compare deterministic
   first-index argmax token IDs for Top-1.
5. **Emit all 64 row receipts.** Each row must bind suite lock, candidate and
   teacher identities, ordinal, window ID, corrected class, 1,024 binary64 KLD
   values, and integer Top-1 matches. Start from
   `templates/balanced64-window-v1.json`.
6. **Aggregate without hand editing.** Put only the 64 receipts in one directory:

   ```bash
   python3 -m Evals.tools.receipts aggregate work/balanced64-windows \
     --suite-lock Evals/configs/balanced64-v1.json \
     --output work/balanced64-aggregate.json
   ```

7. **Add the size columns.** Report decimal GB from packed wire bytes and compute
   normalized bpw as `packed_wire_bytes × 8 / 284334567511`. FP remains FP8, so
   its logical basis is 8 bpw—not 16.
8. **Admit only a complete row.** Require 64/64 windows, 65,536/65,536 positions,
   KLD, Top-1, FP, GB, bpw, and all source identities. HOLDOUT results are reported
   separately and never substituted into BALANCED64.

## Banana Smasher CLI producer surface

The repository also exposes manifest-bound paired evaluation commands:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./banana-smasher

smash bank \
  --model-root "$FP8_MODEL_ROOT" \
  --corpus "$CORPUS_ROOT" \
  --windows-manifest "$BALANCED64_WINDOWS_MANIFEST" \
  --instrument-profile "$BALANCED64_INSTRUMENT_PROFILE" \
  --output "$TEACHER_BANK"

smash evaluate \
  --model-root "$MODEL_ROOT" \
  --candidate "$CANDIDATE_PACK" \
  --reference "$FP8_REFERENCE_PACK" \
  --bank "$TEACHER_BANK" \
  --output "$EVALUATION_ROOT"
```

`smash evaluate` is required to run paired candidate/reference arms and seals an
`EVALUATION_COMPLETE` marker plus `evaluation.json`. A result is comparable to
the table above only if the supplied manifests bind the exact BALANCED64 lock,
teacher basis, support, population, and metric semantics listed here.

## Layout

- `configs/` — frozen suite lock and all 64 ordered window IDs
- `protocols/` — exact metric, aggregation, and replay procedure
- `results/` — machine-readable sealed comparison
- `schemas/` — closed JSON contracts
- `templates/` — one-window producer template
- `tools/receipts.py` — pure-Python fail-closed verifier and aggregator
