# Evaluations

This directory is the executable entry point for Banana Smasher evaluations.
It contains frozen suite locks, metric protocols, schemas, templates, and
pure-stdlib validation/aggregation helpers. Public result claims and tables
live under [`notes/evaluations/`](../notes/evaluations/), as required by the
repository reporting policy.

## Quick validation

From the repository root:

```bash
python3 -m evaluations.tools.receipts verify \
  notes/evaluations/results/deepseek-v4-flash-0731-balanced64-v1.json \
  --suite-lock evaluations/configs/balanced64-v1.json
python3 -m pytest -q tests/test_evaluation_receipts.py
```

The first command recomputes Top-1 ratios, decimal-GB values, normalized BPW,
the shared denominator, and rankings. It also recomputes the public suite-lock,
window-population, and corrected class-map digests and requires the exact
published lock.

It does **not** authenticate protected source receipts, recompute historical
KLD without per-position rows, download a model, or execute a scorer. SHA-256
values for protected sources are identifiers only.

## Layout

- `configs/` — immutable suite locks: ordered windows, corrected classes, metric semantics, and basis hashes.
- `protocols/` — metric, validation, reaggregation, and replay-boundary procedures.
- `schemas/` — closed machine-readable contracts (`additionalProperties: false`).
- `templates/` — minimal standardized producer output examples.
- `tools/receipts.py` — fail-closed validator and ordered per-position aggregator.
- `results/` — index only; evidence and result tables are stored in `notes/evaluations/`.

## Reproducibility levels

1. **Compact receipt validation** is cheap and available in a clean clone. It
   proves internal arithmetic and exact agreement with the tracked suite lock.
2. **Standardized row reaggregation** requires all 64 per-window receipts. The
   helper verifies the frozen ordinal/window/class population and reduces all
   65,536 binary64 KLD values once with `math.fsum` in canonical order.
3. **Full GPU measurement replay** additionally requires exact candidate and
   teacher artifacts, corpus payloads, and a versioned scorer. The historical
   four-model bundle is explicitly blocked at this level: protected inputs are
   not distributed, the historical scorer is absent from canonical `main`, and
   IQ3/IQ4 lack complete public candidate identities.

See [`protocols/balanced64-v1.md`](protocols/balanced64-v1.md) for the exact
contract and limitations.

## Class-map correction

A protected historical class map (`610e13dd…`) is retired and invalid for
subgroup reporting. Projecting the same 64 global windows through the canonical
512-window provenance produces the valid counts:

- Agentic 19
- Chat 7
- Code 9
- Multilingual 10
- Prose 10
- Reasoning 9

This regrouping does not change any global KLD or Top-1 value. No historical
class-level metric from the retired map is published here.

## Adding a new evaluation

1. Freeze the population and metric before loading a candidate.
2. Record exact input revisions, byte counts, SHA-256 digests, FP basis, one
   parameter denominator, and public retrieval instructions.
3. Pin the scorer source revision, numeric dtype, logsumexp implementation,
   serialization, reduction order, and negative-value policy.
4. Emit one closed-schema, ordinal-bound receipt per test case. For BALANCED64,
   preserve every per-position KLD as its shortest round-trip binary64 string.
5. Aggregate with the repository helper; do not hand-edit derived rates.
6. Store the machine result under `notes/evaluations/results/` and the readable
   report under `notes/evaluations/`.
7. Add negative tests for denominator, basis, population, metric, source, and
   duplicate-key drift.
8. State whether the run is compact validation, row reaggregation, synthetic
   testing, or physical model replay. Never promote one category into another.

Existing versioned results are immutable after publication. Corrections create
a new receipt/report that identify the superseded basis and explain the change.
