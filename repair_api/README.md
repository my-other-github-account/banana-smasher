# Repair artifact API

The public scoring surface is intentionally small:

```python
from repair_api import RepairArtifact

artifact = RepairArtifact.open("/path/to/repair-artifact")
trend = artifact.trend([0, 3, 16])
```

For resident experiment work, use one API object. It caches each selected
checkpoint/window load, times only the in-memory reduction, and carries the
basis/corpus/teacher/window/checkpoint identity into every receipt:

```python
from repair_api import ResidentRepairAPI

api = ResidentRepairAPI.open("/path/to/repair-artifact")
post = api.score("UPDATE_016", receipt_path="receipts/U16_RESIDENT.json")
pair = api.resume_compare("UPDATE_016", "UPDATE_003")
step = api.continue_to("UPDATE_016", "U64")
```

`resume_compare` requires shared scientific identity and an immediate-parent
checkpoint binding. `continue_to` requires an exact declared milestone; it does
not guess or silently substitute a partial checkpoint.

The two new experiment paths are API-owned and execute without raw launchers.
A clean-U0 replay constructs model, optimizer, and scheduler objects from
factories, runs exactly 16 update callbacks, authenticates the resulting state
against the loaded target checkpoint (not a caller-declared SHA or fixture), and
writes an immutable receipt:

```python
replay = {
    "model_factory": make_model,
    "optimizer_factory": make_optimizer,
    "scheduler_factory": make_scheduler,
    "update_fn": update_once,
    "geometry": {"layers": 43, "hidden": 4096},
    "basis_sha256": BASIS_SHA,
    "corpus_sha256": CORPUS_SHA,
    "seed": 1701,
}
api.construct_clean_u0("UPDATE_000", "UPDATE_016", replay=replay,
                       receipt_path="receipts/CLEAN_U0_REPLAY.json")
```

`construct_clean_u0` rejects `expected_target_state` and
`target_state_sha256`; the target state must be read from the sealed U16
checkpoint. It also rejects checkpoint-loaded factories, callbacks, and raw
command substitutes.

U16 continuation has one public execution verb: `continue_two_spark_real`. It
constructs the accepted `ShardStudent` from the immutable trainer/source model
and admission assets, routes the grouped-K2 resident student through the exact
rank-0/rank-1 layer split, evaluates the real teacher KL objective, and runs
Adam plus LambdaLR internally. The config must declare `world_size=2`, rank,
a disjoint full 43-layer `layer_split`, one shared optimizer/scheduler lineage,
local-only execution, the authenticated U16/basis SHAs, and immutable paths for
`trainer_source`, `model_root`, `asset_root`, `parent_root`, `l034_roster`,
`teacher_root`, `corpus`, `master_addr`, and `master_port`:

```python
api.continue_two_spark_real(
    "UPDATE_016", [20, 32, 48, 64], config={
        "authorized_api": True, "world_size": 2, "rank": RANK,
        "local_only": True, "layer_split": {0: [0, 20], 1: [21, 42]},
        "shared_optimizer_scheduler_lineage": LINEAGE,
        "basis_sha256": EXPECTED_BASIS_SHA,
        "checkpoint_sha256": U16_CHECKPOINT_SHA,
        "trainer_source": "/home/dnola/missions/MODERN_GREEN_t_6bc398da/source/modern_green_clean_u0.py",
        "model_root": MODEL_ROOT, "asset_root": ASSET_ROOT,
        "parent_root": PARENT_ROOT, "l034_roster": L034_ROSTER,
        "teacher_root": TEACHER_ROOT, "corpus": CORPUS,
        "master_addr": MASTER_ADDR, "master_port": MASTER_PORT,
    }, receipt_path=RECEIPT,
)
```

Caller-supplied `advance_fn`, resident state/model mappings, interpolation,
state-only hashes, raw launchers, and single-device substitutes are rejected.
Each U20/U32/U48/U64 milestone records nonzero gradients and parameter deltas,
CUDA-synchronized forward/backward/optimizer timings, process/GPU evidence,
and immutable checkpoint path/SHA plus parent binding. The implementation
never treats checkpoint tensors alone as a model or a loss.

Or from the CLI:

```bash
python -m repair_api /path/to/repair-artifact --trend 0,3,16
```

Candidate generation is also API-owned. It binds the declared checkpoint SHA,
identity, and `next_update` into a derived copy of the sealed official builder,
then emits the canonical `q_lp_at_ref`/`q_argmax` rows before the fixed reducer
scores them:

For a real U16→U64 continuation, use the single resident materialization verb.
It requires both rank receipts from the loaded two-Spark continuation, validates
all four immediate-parent checkpoint/state SHA bindings, rejects fixture,
unloaded, and sub-second state markers, and scores the generated files rather
than any legacy row-metric shortcut:

```python
aggregate = api.materialize_candidates(
    ["UPDATE_020", "UPDATE_032", "UPDATE_048", "UPDATE_064"],
    builder_template=BUILDER, ref_dir=TEACHER_DIR, corpus=CORPUS,
    meta_dir=MODEL_DIR, continuation_receipts=[RANK0, RANK1],
    receipt_dir="receipts", windows=WINDOWS_64,
)
```

The method emits one `U20/U32/U48/U64_CANDIDATE_BALANCED64.json` receipt and
one `U16_U64_CANDIDATE_BALANCED64_AGGREGATE.json`; every score uses 64 ordered
windows, 1,024 positions/window, support 8,192, `KL(teacher||candidate)`,
binary64 and `math.fsum`, with zero timed payload reads. A mechanical
`PASS_4_OF_4` aggregate retains `quality_status=RED_UNPROMOTED` until the
independent product quality gate is improved.

CLI equivalent:

```bash
python -m repair_api /path/to/repair-artifact \
  --materialize UPDATE_020,UPDATE_032,UPDATE_048,UPDATE_064 \
  --continuation-receipt /path/to/RANK0.json \
  --continuation-receipt /path/to/RANK1.json \
  --builder-template /path/to/builder_update_template.py \
  --ref-dir /path/to/BALANCED64_TEACHER --corpus /path/to/windows_ds4_eval.json \
  --meta-dir /path/to/model-metadata --receipt-dir /path/to/receipts
```

```bash
python -m repair_api /path/to/repair-artifact \
  --generate --checkpoint 16 \
  --builder-template /path/to/builder_update_template.py \
  --ref-dir /path/to/BALANCED64_TEACHER \
  --corpus /path/to/windows_ds4_eval.json \
  --meta-dir /path/to/model-metadata \
  --python-executable /path/to/runtime/python \
  --remote user@qsfp-host:model-shards
```

## Artifact layout

The artifact is self-contained and contains one `ARTIFACT.json` at its root:

```text
repair-artifact/
  ARTIFACT.json
  checkpoints/
    UPDATE_000.pt
    UPDATE_003.pt
    UPDATE_016.pt
  score/
    teacher/
      t8192_win28.pt
      ...
    candidates/
      0/
        q8192_win28.pt
        ...
      3/
        q8192_win28.pt
        ...
      16/
        q8192_win28.pt
        ...
```

`ARTIFACT.json` declares only relative paths, checkpoint names, and the ordered
window set. The scoring function is fixed in `repair_api.balanced64`; checkpoint
selection never changes it. The only per-call choice is the checkpoint and an
optional subset of the artifact's declared windows.

The standardized score is fixed at:

- 64 declared ordered windows by default;
- 1,024 positions per window;
- support 8,192;
- `KL(teacher || candidate)`;
- float64 support renormalization;
- Python `math.fsum` in window/position order;
- full-vocabulary candidate argmax for Top-1.

Missing files, absolute paths, root escapes, undeclared windows, fewer than
1,024 rows, or incomplete candidate rows fail loudly. Real roots that carry
extra page padding are reduced to the canonical first 1,024 rows. Legacy roots
without an explicit teacher inventory derive a deterministic file inventory
from the selected teacher files; omitted checkpoint parent lineage is read from
the checkpoint identity payload. There is no legacy scorer fallback.

For a current remote artifact root, exercise the exact resident implementation
without copying the score tensors:

```bash
python repair_api/remote_integration.py \
  --remote spark-7 \
  --python /home/dnola/humming_env/bin/python \
  --artifact-root /home/dnola/missions/MODERN_GREEN_SCORE_t_632e9474_s7/api_stage_t_f5d2415c/artifact_v10 \
  --checkpoint UPDATE_016 --windows 28 \
  --receipt repair_api/receipts/U16_REMOTE_RESIDENT.json
```
