# Paired real-axis bank and evaluation API

This note documents the portable API contract added from source commit `009b41befb9f559805fe2b4148818684f2232ba7`. It contains no model results, corpus records, machine paths, credentials, or operational lineage.

## Public commands

Build a manifest-bound teacher bank:

```text
smash bank \
  --model-root MODEL_ROOT \
  --corpus CORPUS_ROOT \
  --windows-manifest WINDOWS_JSON \
  --instrument-profile INSTRUMENT_JSON \
  --output BANK_ROOT
```

Run a required two-arm evaluation:

```text
smash evaluate \
  --model-root MODEL_ROOT \
  --candidate CANDIDATE_PACK \
  --reference REFERENCE_PACK \
  --bank BANK_ROOT \
  --output EVALUATION_ROOT
```

`--reference` is mandatory. The persisted evaluation mode is exactly `paired_real_axis`; there is no single-arm compatibility mode.

## Portable contracts

The package ships these draft-2020-12 schemas:

- `bs-teacher-bank-v1`
- `bs-real-axis-windows-v1`
- `bs-real-axis-runtime-v1`
- `bs-real-axis-instrument-v1`
- `bs-paired-real-axis-evaluation-v1`

`bs-pack-v1` retains its canonical required fields and permits an optional `real_axis` identity object. Existing export and repair packs remain valid when that optional object is absent.

Bank members, arm artifacts, and layer checkpoints use relative paths and bind byte counts plus SHA-256 identities. Completion markers bind their manifests. Pair checkpoint markers bind both arm manifests and the previous pair checkpoint, so resume selects only the greatest contiguous, jointly verified boundary. Missing markers, missing members, extra files, unsafe paths, symlinks, byte drift, digest drift, population drift, and checkpoint-chain drift fail closed.

## Interpretation boundary

The teacher bank stores declared support log-probabilities from a manifest-driven numerical layer walk. Evaluation reports `KL(teacher || candidate)`, top-1 agreement, per-window paired deltas, and a confidence interval over the declared paired windows.

These measurements are reusable numerical artifact comparisons. They do not establish causal-context equivalence, same-work language-model equivalence, or equivalence to an undeclared serving workload.
