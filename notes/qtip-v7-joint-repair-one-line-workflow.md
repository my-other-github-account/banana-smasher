# QTIP V7 joint repair: frozen inputs → U5-equivalent score → longer resume

This is the public, one-line workflow for the exact all-43 QTIP V7 repair surface: 43 layer LUTs, 235 RMSNorm masters, 43 attention output gains, and a required teacher-KLD objective. Replace only the parameterized paths/hosts below; no mission-local Python is invoked by these commands.

## Inputs and executables

```bash
export V7_MANIFEST=/path/to/QTIP_V7_MANIFEST.json
export TEACHER_BANK=/path/to/BALANCED64_V1.teacher-bank.json
export RUN=/path/to/qtip-v7-joint-run
export TRAINER=/path/to/public-joint-trainer
export SHARD_WORKER=/path/to/public-balanced64-worker
export TRAINER_HOST=spark-8
```

`$TRAINER` is launched with the frozen paths and requested horizon in `QTIP_V7_*` environment variables. It must emit `banana-smasher-qtip-v7-joint-checkpoint-v1`, including `objective=teacher_kld`, `freeze_sha256=SHA256(QTIP_V7_FREEZE)`, a finite `teacher_kld`, 43 `L000..L042` LUT tensors, 235 RMSNorm tensors, and 43 output-gain tensors. `$SHARD_WORKER` receives its candidate, teacher bank, exact ordinal range, and receipt path through the same public `QTIP_V7_*` environment contract.

## 1. Inspect and freeze the exact inventory and teacher bank

```bash
smash qtip-v7-joint-repair inspect --manifest "$V7_MANIFEST" --teacher-bank "$TEACHER_BANK" --run-root "$RUN" --trainer-host "$TRAINER_HOST"
```

This creates immutable `$RUN/FROZEN_INPUTS.json` and fails unless the physical inventory is exactly layers 0..42, 768 members per layer, one LUT per layer, and 64 unique teacher-bank windows.

## 2. Launch/resume full joint training to U5

Fresh launch:

```bash
smash qtip-v7-joint-repair train --freeze "$RUN/FROZEN_INPUTS.json" --trainer "$TRAINER" --target-update 5 --checkpoint "$RUN/checkpoints/UPDATE_005.pt"
```

Resume an interrupted/shorter accepted checkpoint instead:

```bash
smash qtip-v7-joint-repair train --freeze "$RUN/FROZEN_INPUTS.json" --trainer "$TRAINER" --resume-from "$RUN/checkpoints/UPDATE_003.pt" --target-update 5 --checkpoint "$RUN/checkpoints/UPDATE_005.pt"
```

The command validates the complete trainable surface and teacher KLD, then writes `$RUN/checkpoints/UPDATE_005.pt.PASS.json` only after checkpoint readback and SHA-256 verification.

## 3. Independently verify the immutable U5 checkpoint/PASS receipt

```bash
smash qtip-v7-joint-repair verify --freeze "$RUN/FROZEN_INPUTS.json" --checkpoint "$RUN/checkpoints/UPDATE_005.pt" --receipt "$RUN/checkpoints/UPDATE_005.pt.PASS.json"
```

## 4. Copy and launch disjoint BALANCED64 side-Spark shards

The trainer is not signaled, paused, or inspected. All workers are staged/launched before any result is collected. This example splits exact ordinals 0..63 across four side Sparks over their direct fabric/QSFP addresses:

```bash
smash qtip-v7-joint-repair shard-launch --candidate "$RUN/checkpoints/UPDATE_005.pt" --freeze "$RUN/FROZEN_INPUTS.json" --teacher-bank "$TEACHER_BANK" --output "$RUN/balanced64/u5" --worker "spark-1@192.168.200.1:/dev/shm/qtip-v7-u5-o00-15=$SHARD_WORKER" --worker "spark-3@192.168.200.3:/dev/shm/qtip-v7-u5-o16-31=$SHARD_WORKER" --worker "spark-4@192.168.200.4:/dev/shm/qtip-v7-u5-o32-47=$SHARD_WORKER" --worker "spark-7@192.168.200.8:/dev/shm/qtip-v7-u5-o48-63=$SHARD_WORKER"
```

Before staging any files, each `EXPECTED_HOST@ROUTE` is preflighted with `ssh ROUTE hostname`; a route/identity mismatch fails closed. The command also refuses any route, expected identity, or observed identity matching `trainer_host` sealed by `inspect`, so the canonical Spark-8 trainer (`192.168.200.9`) cannot become a shard worker.

For a local fixture/smoke, use repeated `local-N=$SHARD_WORKER` bindings instead.

## 5. Aggregate, compare, and select the champion

```bash
smash qtip-v7-joint-repair aggregate --shards "$RUN/balanced64/u5" --output "$RUN/scores/u5.json"
```

```bash
smash qtip-v7-joint-repair compare --baseline "$RUN/scores/u0.json" --candidate "$RUN/scores/u5.json" --output "$RUN/CHAMPION_U5.json"
```

Candidate selection is fail-closed: mean KLD must be non-worse and Top-1 matches must be non-worse; otherwise the baseline remains champion.

## 6. Continue the accepted checkpoint to a longer horizon

```bash
smash qtip-v7-joint-repair train --freeze "$RUN/FROZEN_INPUTS.json" --trainer "$TRAINER" --resume-from "$RUN/checkpoints/UPDATE_005.pt" --target-update 256 --checkpoint "$RUN/checkpoints/UPDATE_256.pt"
```

Score U256 with the same shard-launch → aggregate → compare sequence, using a fresh output namespace.

## 7. Materialize and account exact stored wire

```bash
smash qtip-v7-joint-repair materialize --freeze "$RUN/FROZEN_INPUTS.json" --manifest "$V7_MANIFEST" --checkpoint "$RUN/checkpoints/UPDATE_005.pt" --output "$RUN/materialized/UPDATE_005"
```

The receipt reports `qtip_wire_bytes`, `dense_repair_bytes`, and their exact sum `stored_wire_bytes`. Packed member wires remain hash-bound and are not duplicated; all 43 FP16 LUTs and `repair_state.safetensors` are physically materialized. The command requires packed identity and a zero QTIP wire-size delta.

## Discoverability

```bash
smash qtip-v7-joint-repair --help
smash qtip-v7-joint-repair shard-launch --help
```
