# Worked example: routed-only Q2 PRE → U45 repair → POST

This example uses only the public API. Replace the three path/SHA placeholders; do not edit the recipe.

## 1. Reserve and launch the two ranks

Make the admitted artifact visible at the same absolute path on both hosts. Launch rank 0 first and rank 1 immediately after it, with one shared master address and port. A systemd launch should enforce:

```text
MemoryMax=80G
MemorySwapMax=16G
LimitMEMLOCK=infinity
```

Each rank needs:

```text
WORLD_SIZE=2
RANK=0 or 1
LOCAL_RANK=0 or 1
MASTER_ADDR=<rank-0 fabric address>
MASTER_PORT=<reserved port>
```

## 2. Run these exact API calls on both ranks

```python
import json
from pathlib import Path

import torch.distributed as dist
from banana_smasher import ResidentRepairAPI

ARTIFACT = Path("/absolute/path/to/admitted-q2-artifact")
RUN_ROOT = Path("/absolute/path/to/resident-repair-run")
CHECKPOINT_SHA = "<64-character-checkpoint-sha256>"


def main() -> None:
    dist.init_process_group("nccl", init_method="env://")
    try:
        api = ResidentRepairAPI.build_uniform(
            ARTIFACT,
            tier="q2",
            checkpoint_sha=CHECKPOINT_SHA,
            run_root=RUN_ROOT,
            scope="routed_only",
            native_rest=True,
        )
        pre = dict(api.score_pre())
        training = dict(api.repair_train(updates=45))
        post = dict(api.score_post())
    finally:
        dist.destroy_process_group()

    summary = {
        "pre_kld": float(pre["mean_kld"]),
        "updates": int(training["updates"]),
        "post_kld": float(post["mean_kld"]),
        "improved": float(post["mean_kld"]) < float(pre["mean_kld"]),
    }
    print(json.dumps(summary, sort_keys=True))
    if not summary["improved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

`build_uniform(...)` opens and admits the artifact; callers do not supply teacher paths, corpus hashes, per-rank rail configs, layer geometry, optimizer dtypes, sampling policy, or learning rates. Those identities and the validated U45 recipe are artifact/package owned and fail closed when incomplete.

## 3. Expected output shape

A sealed acceptance-lineage example is:

```json
{
  "improved": true,
  "post_kld": 0.211277616743619,
  "pre_kld": 0.2292069946743951,
  "updates": 45
}
```

The current run's exact values may differ. Acceptance is mechanical: the real current POST must be lower than the real current PRE. `score_post()` also publishes `facade/rankN/RESIDENT_ARM_RESULT.json` and raises when the inequality fails.

## 4. Equivalent public command

For the production fresh-phase runner, invoke this same command on both ranks under the environment above:

```console
smash improve /absolute/path/to/admitted-q2-artifact \
  --checkpoint-sha <64-character-checkpoint-sha256> \
  --run-root /absolute/path/to/resident-repair-run \
  --updates 45
```

Require all four receipts in `RUN_ROOT`: `score_pre.json`, `repair_train.json`, `score_post.json`, and `IMPROVE_RESULT.json`. A successful shell exit without `IMPROVE_RESULT.json.status == "PASS"` is not acceptance.

See `REPAIR_TRAINING_GUIDE.md` for the fail-closed inputs, recipe, receipts, and launch contract.
