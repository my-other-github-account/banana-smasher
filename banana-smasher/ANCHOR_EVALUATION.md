# Anchor evaluation API

Status: versioned product surface (`banana-smasher-bank-manifest-v1`)

The `smash anchor` command family implements a reproducible evaluation rail for
four distinct bank roles. It is generic: panel size, class names, quotas,
ranking fields, tier menus, parent weights, and policy thresholds are manifest
or caller data. The role names preserve the binding train/holdout separation,
but neither `64`, `512`, six classes, nor any model family is hardcoded into
selection or aggregation.

## Bank roles and default authority

| Role | Intended use | Solver/fitting default |
|---|---|---|
| `train_balanced64` | Cheap training anchor-pricing panel | Allowed |
| `train512` | Full training parent and escalation rail | Allowed |
| `holdout_balanced64` | Quick final-result check | Rejected |
| `holdout512` | Final/ship measurement | Rejected |

`smash anchor solver-row` rejects either holdout role. The
`--diagnostic-override` flag is deliberately non-default and emits
`diagnostic_only: true`; it does not turn a holdout result into training data.

The wheel includes `anchor_banks/FOUR_BANK_PROVENANCE.json`, four recovered
bank manifests, a historical train-panel calibration receipt, and an explicit
unresolved-provenance record. Unknown tokenizer, teacher, or historical scorer
hashes remain `status: unresolved` with a reason. Exact scoring requires the
caller to supply resolved same-work producer and candidate identities; no hash
is inferred from another bank or run.

## Manifest and producer formats

`schema/anchor-bank-manifest-v1.schema.json` defines the public manifest. Window
order is semantic. The manifest binds:

- role and parent corpus identity;
- ordered `{id, class}` rows and exact class counts;
- corpus, tokenizer, teacher, and scorer identities;
- parent dataset field names;
- split lineage, deterministic creation configuration, and relationships;
- membership, class-map, and manifest-payload SHA-256 values.

A resolved identity has `status`, `sha256`, and a scrubbed URI. An unresolved
identity has only `status` and `reason`. Validation rejects a guessed hash or
URI attached to an unresolved identity.

Parent datasets may be JSONL objects or one JSON array of objects.
`dataset_fields.window_id` names the required source ID field and
`dataset_fields.class` names the materialized class field. If the parent rows
carry that class field, materialization verifies it; otherwise it attaches the
manifest's hash-bound class label. Other parent payload fields are preserved.
Teacher and candidate producer JSONL rows have `window_id` plus exactly one of
`logits` or `probabilities`. Each value may be a single vocabulary vector (one
position) or a rectangular `[positions, vocab]` matrix:

```json
{"window_id": 17, "logits": [2.0, 0.0]}
```

```json
{"window_id": 17, "probabilities": [0.88, 0.12]}
```

```json
{"window_id": 17, "logits": [[2.0, 0.0], [0.1, 1.9]]}
```

The scorer computes `KL(teacher || candidate)` independently at every position
and stores the arithmetic mean as `kld`, with the evaluated `position_count` in
each raw row. It rejects duplicate, missing, or unexpected IDs; mixed position
or vocabulary dimensions; ragged matrices; non-finite values; invalid
probability mass; identity drift; and resume rows from another basis, bank,
teacher, candidate, producer, or scorer.

## One-run-root workflow

Set immutable inputs explicitly. The examples use shell placeholders without
defaults so automation fails before invoking the CLI when a binding is absent.

```bash
export RUN_ROOT=/path/to/new-or-existing-anchor-run
export BANK_MANIFEST=/path/to/train_balanced64.bank.json
export RESOLVED_BANK_MANIFEST=/path/to/run-specific-train_balanced64.bank.json
export IDENTITY_UPDATES=/path/to/exact-identity-updates.json
export PARENT_JSONL=/path/to/declared-parent.jsonl
export TEACHER_JSONL=/path/to/exact-teacher.jsonl
export CANDIDATE_JSONL=/path/to/exact-candidate.jsonl
export BANK_ID=train_balanced64
export CANDIDATE_ID=uniform-tier-a
export TEACHER_SHA256=EXACT_TEACHER_ARTIFACT_SHA256
export CANDIDATE_PRODUCER_SHA256=EXACT_PRODUCER_SHA256
export CANDIDATE_ARTIFACT_SHA256=EXACT_ARTIFACT_SHA256
export BASIS_SHA256=EXACT_BASIS_SHA256

smash anchor validate --manifest "$BANK_MANIFEST"
smash anchor resolve --manifest "$BANK_MANIFEST" --identities "$IDENTITY_UPDATES" --output "$RESOLVED_BANK_MANIFEST"
smash anchor register --run-root "$RUN_ROOT" --manifest "$RESOLVED_BANK_MANIFEST"
smash anchor materialize --run-root "$RUN_ROOT" --bank "$BANK_ID" --parent "$PARENT_JSONL"
smash anchor import-producer --run-root "$RUN_ROOT" --bank "$BANK_ID" --kind teacher --source "$TEACHER_JSONL" --sha256 "$TEACHER_SHA256"
smash anchor import-producer --run-root "$RUN_ROOT" --bank "$BANK_ID" --kind candidate --candidate-id "$CANDIDATE_ID" --source "$CANDIDATE_JSONL" --sha256 "$CANDIDATE_PRODUCER_SHA256"
smash anchor score --run-root "$RUN_ROOT" --bank "$BANK_ID" --candidate-id "$CANDIDATE_ID" --teacher-sha256 "$TEACHER_SHA256" --teacher-uri producer://exact-teacher --candidate-sha256 "$CANDIDATE_ARTIFACT_SHA256" --candidate-uri candidate://uniform-tier-a --basis-sha256 "$BASIS_SHA256"
smash anchor aggregate --run-root "$RUN_ROOT" --bank "$BANK_ID" --candidate-id "$CANDIDATE_ID"
smash anchor solver-row --run-root "$RUN_ROOT" --bank "$BANK_ID" --candidate-id "$CANDIDATE_ID"
smash anchor status --run-root "$RUN_ROOT" --format human
smash anchor status --run-root "$RUN_ROOT" --format json
```

Register and materialize each role in the same run root. Pass
`--disjoint-bank OTHER_ID` to `materialize` for every requested membership
non-overlap. Producer imports preserve exact bytes and write relative paths;
large logits and model artifacts belong in the run root or an external store,
not Git.

All writes are atomic. Repeating register, materialize, import, or score with
identical bytes is an idempotent resume. A conflicting existing artifact fails
closed instead of being overwritten.

The identity-update object may contain `corpus`, `tokenizer`, `teacher`,
`scorer`, or `parent_corpus`; every supplied value must be a resolved identity
with an exact SHA-256 and URI. `resolve` preserves the ordered membership and
creation data while recomputing the manifest payload hash. Exact scoring
requires resolved corpus, tokenizer, scorer, and parent-corpus identities. A
run-specific teacher identity may be bound by the scoring arguments; if the
manifest already resolves teacher, the hashes must match.
Teacher/candidate artifact identities and producer-file identities are distinct:
the score receipt binds both. `import-producer --sha256` admits the JSONL bytes;
`score --teacher-sha256` identifies the teacher artifact that produced them.

## Deterministic balanced selection

A selection config is ordinary JSON. Class names, quotas, ranking, seed, and
tier menus are caller data:

```json
{
  "bank_id": "train-panel-v2",
  "role": "train_balanced64",
  "quotas": {"class-a": 8, "class-b": 8},
  "seed": "panel-v2",
  "ranking_field": "selection_loss",
  "tier_menus": {"uniform": ["tier-a", "tier-b"]}
}
```

```bash
smash anchor select --run-root "$RUN_ROOT" --parent-bank train512 --parent "$PARENT_JSONL" --config /path/to/selection.json
```

Rows are selected per class by ascending finite `ranking_field`, with a stable
seed-and-window hash tie-break. Omit `ranking_field` for deterministic hash
selection. The generated manifest records the complete config and the parent
relationship.

## Calibration and escalation policy

Pass a `banana-smasher-anchor-calibration-v1` object to aggregation:

```bash
smash anchor aggregate --run-root "$RUN_ROOT" --bank train_balanced64 --candidate-id "$CANDIDATE_ID" --calibration /path/to/calibration.json
```

Measured bank means remain under `measured` with label `measured_on_bank`.
Corrected class values and the parent-count-weighted global value appear only
under `parent_estimates` with label `estimated_parent_not_measured` and the
calibration hash. The API never relabels an estimate as a full-parent
measurement.

After the same candidate has measured aggregates on both training rails, put
caller policy in a JSON file:

```json
{
  "max_abs_class_relative_pct": 4.0,
  "max_abs_global_relative_pct": 2.0
}
```

```bash
smash anchor compare --run-root "$RUN_ROOT" --panel-bank train_balanced64 --parent-bank train512 --candidate-id "$CANDIDATE_ID" --thresholds /path/to/thresholds.json
```

The receipt includes absolute and relative overall/per-class errors and one
explicit `policy_result`: `retain_panel` or `escalate_to_full_parent`. The
thresholds are never embedded campaign policy.

## Run-root layout and status

```text
RUN_ROOT/
  manifests/BANK.json
  banks/BANK.jsonl
  producers/teacher/BANK.jsonl
  producers/candidate/CANDIDATE/BANK.jsonl
  imports/*.json
  scores/CANDIDATE/BANK/raw.jsonl
  aggregates/CANDIDATE/BANK.json
  comparisons/CANDIDATE.json
  solver_rows/CANDIDATE--BANK.json
```

`status --format human` prints a completion grid for bank production, teacher
coverage, candidate coverage, scoring, aggregation, and provenance.
`status --format json` emits `banana-smasher-anchor-status-v1` for automation.

## Python API

The package root exports `build_bank_manifest`, `validate_bank_manifest`,
`resolve_bank_identities`, `create_balanced_subset`, `materialize_bank`, `register_bank`,
`import_producer`, `score_bank`, `aggregate_scores`,
`compare_training_rails`, `emit_solver_row`, and `status_report`.
`AnchorEvaluationError` is the fail-closed contract exception.

## Reproducibility gates

The CPU fixture in `tests/test_anchor_evaluation.py` proves deterministic
membership and hash stability, parent/class/disjointness validation, producer
admission, resume, actionable missing-producer errors, class/global
aggregation, calibration labeling, train-panel escalation policy, holdout
rejection, status reporting, and the complete CLI chain. Release verification
builds the wheel, installs it into a clean environment, runs that chain, and
compares source-tree and installed-wheel `smash` command surfaces.
