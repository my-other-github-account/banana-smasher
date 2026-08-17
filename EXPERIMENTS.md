# Experiment specifications

Banana Smasher experiments use versioned JSON with separate `scientific` and
`execution` sections. Operators author the scientific intent once instead of
reconstructing parent, windows, optimizer, data, and scoring details at launch
time.

## Workflow

1. **Author** `experiments/NAME.json` with logical artifact IDs and exact SHA-256
   values. If a source hash was not recorded, mark it `identity_status:
   "unavailable"` and explain why; never invent it.
2. **Explain** before launch:
   `smash experiment explain experiments/NAME.json`.
3. **Diff** against the intended reference:
   `smash experiment diff REFERENCE.json CANDIDATE.json`. Scientific drift is
   classified `SCIENTIFIC` and exits nonzero. Placement/kernel-only changes are
   `EXECUTION_ONLY` and preserve the scientific identity.
4. **Lock** the accepted config:
   `smash experiment lock experiments/NAME.json --output EXPERIMENT.lock.json`.
   `mode=reproduce` fails if `reproduction_of` has different scientific fields;
   `mode=extend` emits `NOT A REPRODUCTION` and a new identity.
5. **Run** only after `smash experiment validate-runtime LOCK OBSERVED` passes.
   The public QTIP V7 joint trainer also accepts `--experiment-lock` together
   with `--experiment-observed` and performs this check before trainer/model/GPU
   work.
6. **Receipt** the successful run with the lock's
   `scientific_identity_sha256`, source config SHA, and produced artifacts.

## Why scientific and execution fields are separate

The all-64 reconstruction changed parent lineage, optimizer grouping, warmup,
and first-applied LR while looking operationally similar to the Green run.
Those changes now alter the scientific identity and fail reproduction before
compute. Moving the same science to another device or selecting another kernel
changes only `execution`; it remains explainable without pretending that
placement is experimental meaning.

## Evaluation suite-lock fan-in

`EvaluationSpec` references the existing `BALANCED64_V1` suite lock by logical
config ID and `suite_lock_sha256`; it does not duplicate windows, metric
reduction, support, or Top-1 semantics. This branch remains based on current
`main`. During later branch fan-in, merge the compatible public files from
`origin/feat/evaluation-reproducibility` (including
`evaluations/configs/balanced64-v1.json` and its receipt verifier) rather than
copying or redefining that protocol in experiment specs.
