# Resident repair training guide

This is the canonical production path for a routed-only Q2 artifact. The public facade is `banana_smasher.ResidentRepairAPI`; callers do not select sampling, optimizer dtypes, learning rates, or held-out policy.

## Inputs

You need:

- one admitted routed-only Q2 artifact directory;
- the exact 64-character SHA-256 of the selected checkpoint;
- two Linux CUDA ranks with the artifact available at the same logical path;
- `RANK`, `LOCAL_RANK`, `WORLD_SIZE=2`, `MASTER_ADDR`, and `MASTER_PORT` set by the launcher.

The admitted artifact must contain `identity.json`, its authenticated checkpoint and manifest, and `production-rails.rank0.json` / `production-rails.rank1.json`. `ResidentRepairAPI.build_uniform(...)` verifies artifact identity, checkpoint bytes, basis, routed-only Q2 composition, rank geometry, and the artifact-owned provider config. It fails closed rather than selecting another tier or route.

The production launcher must reserve the two hosts before starting either rank. On systemd hosts use `MemoryMax=80G`, `MemorySwapMax=16G`, and `LimitMEMLOCK=infinity`; these are process limits, not scientific recipe controls. Start rank 0 before rank 1 and use the same `MASTER_ADDR` / `MASTER_PORT` on both.

## Straight API sequence

Run the following program on both ranks with the launcher environment above:

```python
import os
import torch.distributed as dist
from banana_smasher import ResidentRepairAPI

artifact_root = "/absolute/path/to/admitted-q2-artifact"
checkpoint_sha = "<64-character-checkpoint-sha256>"
run_root = "/absolute/path/to/resident-repair-run"

# WORLD_SIZE=2 uses the standard env:// distributed contract.
dist.init_process_group(backend="nccl", init_method="env://")
try:
    api = ResidentRepairAPI.build_uniform(
        artifact_root,
        tier="q2",
        checkpoint_sha=checkpoint_sha,
        run_root=run_root,
        scope="routed_only",
        native_rest=True,
    )
    pre = api.score_pre()
    training = api.repair_train(updates=45)
    post = api.score_post()
finally:
    dist.destroy_process_group()

assert post["mean_kld"] < pre["mean_kld"]
```

`score_post()` writes `facade/rankN/RESIDENT_ARM_RESULT.json` and raises if `post_kld >= pre_kld`; a normal uncaught exception therefore makes the launcher exit nonzero. Do not catch that exception and convert it to success.

## Canonical one-command phase runner

For production recovery and lower peak memory, run the public command on both ranks:

```console
smash improve /absolute/path/to/admitted-q2-artifact \
  --checkpoint-sha <64-character-checkpoint-sha256> \
  --run-root /absolute/path/to/resident-repair-run \
  --updates 45
```

The command executes `score_pre`, `repair_train`, and `score_post` in three fresh phase processes while preserving only authenticated phase receipts. It writes:

- `score_pre.json`;
- `repair_train.json`;
- `score_post.json`;
- `IMPROVE_RESULT.json`.

`IMPROVE_RESULT.json` is `PASS` only when `post.mean_kld < pre.mean_kld`; otherwise the command exits nonzero.

## Package-owned U45 recipe

`repair_train(updates=45)` always receives the package-owned `u45_validated_v1` recipe:

- broad rotation (`broad_rotation_v1`), 16 windows per update;
- pipeline microbatch 4;
- FP32 loss reduction / backward-safe reductions;
- FP64 optimizer moments;
- base learning rates: LUTs `1e-2`, norms `1e-4`, outputs `1e-2`, with scale `0.1`;
- every accepted update durable;
- held-out validation every four updates;
- halt after two consecutive flat/rising held-out boundaries;
- per-update loss instrumentation and a pre-score-derived loss guard.

These values are overwritten by `ProductionRails`; callers cannot replace them through artifact config.

## Expected acceptance range

The sealed U45 lineage is a scale check, not a substitute for the current receipt:

- PRE: `0.2292069946743951`-class routed-only Balanced64 KLD (some admitted frames report `0.2284983253897188`);
- U45 lineage POST: `0.211277616743619`-class KLD;
- required verdict: `POST < PRE`.

Always publish the exact current `score_pre.json`, `repair_train.json`, `score_post.json`, and `IMPROVE_RESULT.json` values. Never copy the example numbers into a receipt.

## Failure rules

Fail nonzero on missing rank environment, uninitialized/wrong world size, unknown artifact identity, checkpoint or basis drift, absent rank config, non-finite/excess loss, held-out kill, phase timeout, or `POST >= PRE`. There is no fallback, offline scorer, replay route, or alternate tier.
