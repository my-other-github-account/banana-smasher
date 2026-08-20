# QTIP V7 joint repair: frozen inputs → U5-equivalent score → longer resume

This is the public, one-line workflow for the exact all-43 QTIP V7 repair surface: 43 layer LUTs, 235 RMSNorm masters, 43 attention output gains, and a required teacher-KLD objective. Replace only the parameterized paths/hosts below; no mission-local Python is invoked by these commands.

## Inputs and executables

```bash
export V7_MANIFEST=/path/to/QTIP_V7_MANIFEST.json
export TEACHER_BANK=/path/to/BALANCED64_V1.teacher-bank.json
export RUN=/path/to/qtip-v7-joint-run
export TRAINER_HOST=spark-8
```

The package owns the complete joint trainer and BALANCED64 scorer. Checkpoints bind the exact frozen keys/shapes, authenticated teacher logits/KLD, optimizer/scheduler/RNG state, packaged trainer identity, and resume parent.

## 1. Inspect and freeze the exact inventory and teacher bank

```bash
smash qtip-v7-joint-repair inspect --manifest "$V7_MANIFEST" --teacher-bank "$TEACHER_BANK" --run-root "$RUN" --trainer-host "$TRAINER_HOST" --trainer-alias 192.168.200.9
```

This creates immutable `$RUN/FROZEN_INPUTS.json` and fails unless the physical inventory is exactly layers 0..42, 768 members per layer, one LUT per layer, and 64 unique teacher-bank windows.

## 2. Materialize the U0 baseline, then train/resume to U5

```bash
smash qtip-v7-joint-repair train --freeze "$RUN/FROZEN_INPUTS.json" --target-update 0 --checkpoint "$RUN/checkpoints/UPDATE_000.pt"
smash qtip-v7-joint-repair shard-launch --candidate "$RUN/checkpoints/UPDATE_000.pt" --freeze "$RUN/FROZEN_INPUTS.json" --teacher-bank "$TEACHER_BANK" --output "$RUN/balanced64/u0" --worker local-a=builtin
smash qtip-v7-joint-repair aggregate --shards "$RUN/balanced64/u0" --output "$RUN/scores/u0.json"
```

Fresh launch:

```bash
smash qtip-v7-joint-repair train --freeze "$RUN/FROZEN_INPUTS.json" --target-update 5 --checkpoint "$RUN/checkpoints/UPDATE_005.pt"
```

Resume an interrupted/shorter accepted checkpoint instead:

```bash
smash qtip-v7-joint-repair train --freeze "$RUN/FROZEN_INPUTS.json" --resume-from "$RUN/checkpoints/UPDATE_003.pt" --target-update 5 --checkpoint "$RUN/checkpoints/UPDATE_005.pt"
```

The command validates the complete trainable surface and teacher KLD, then writes `$RUN/checkpoints/UPDATE_005.pt.PASS.json` only after checkpoint readback and SHA-256 verification.

## 3. Independently verify the immutable U5 checkpoint/PASS receipt

```bash
smash qtip-v7-joint-repair verify --freeze "$RUN/FROZEN_INPUTS.json" --checkpoint "$RUN/checkpoints/UPDATE_005.pt" --receipt "$RUN/checkpoints/UPDATE_005.pt.PASS.json"
```

## 4. Copy and launch disjoint BALANCED64 side-Spark shards

The trainer is not signaled, paused, or inspected. All workers are staged/launched before any result is collected. This example splits exact ordinals 0..63 across four side Sparks over their direct fabric/QSFP addresses:

```bash
smash qtip-v7-joint-repair shard-launch --candidate "$RUN/checkpoints/UPDATE_005.pt" --freeze "$RUN/FROZEN_INPUTS.json" --teacher-bank "$TEACHER_BANK" --output "$RUN/balanced64/u5" --worker "spark-1@192.168.200.1:/dev/shm/qtip-v7/u5/o00-15=builtin" --worker "spark-3@192.168.200.3:/dev/shm/qtip-v7/u5/o16-31=builtin" --worker "spark-4@192.168.200.4:/dev/shm/qtip-v7/u5/o32-47=builtin" --worker "spark-7@192.168.200.8:/dev/shm/qtip-v7/u5/o48-63=builtin"
```

Before staging any files, each `EXPECTED_HOST@ROUTE` is preflighted with `ssh ROUTE hostname`; a route/identity mismatch fails closed. The command also refuses any route, expected identity, or observed identity matching `trainer_host` sealed by `inspect`, so the canonical Spark-8 trainer (`192.168.200.9`) cannot become a shard worker.

For a local fixture/smoke, use repeated `local-N=builtin` bindings instead.

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
smash qtip-v7-joint-repair train --freeze "$RUN/FROZEN_INPUTS.json" --resume-from "$RUN/checkpoints/UPDATE_005.pt" --target-update 256 --checkpoint "$RUN/checkpoints/UPDATE_256.pt"
```

Score U256 with the same shard-launch → aggregate → compare sequence, using a fresh output namespace.

## 7. Materialize trained state

```bash
smash qtip-v7-joint-repair materialize --freeze "$RUN/FROZEN_INPUTS.json" --manifest "$V7_MANIFEST" --checkpoint "$RUN/checkpoints/UPDATE_005.pt" --output "$RUN/materialized/UPDATE_005"
```

Packed member wires remain hash-bound and are not duplicated; all 43 FP16 LUTs and `repair_state.safetensors` are physically materialized. This command deliberately does not claim a logical/referenced byte total. Physical accounting comes only from readback of fixed-envelope `qtip-v7-wire` layer receipts.

## 8. Pack and verify each physical QTIP V7 layer

For each layer, point `WIRE_ROOT` at the exact directory containing 768 files named `E000_w1.q2v7wire` through `E255_w3.q2v7wire`, in expert-major `w1,w2,w3` order, and point `LUT` at the exact learned FP16 `[1024]` LUT:

```bash
smash qtip-v7-wire pack-layer --source-root "$WIRE_ROOT" --lut "$LUT" --layer 37 --output "$RUN/wire/L037.qtip-v7-wire"
```

```bash
smash qtip-v7-wire verify-layer --wire "$RUN/wire/L037.qtip-v7-wire" --receipt "$RUN/wire/L037.qtip-v7-wire.receipt.json"
```

The encoder streams all 768 unchanged 2,097,152-byte trellis/index planes, losslessly compresses the 9,440,256-byte concatenated control plane, embeds the exact 2,048-byte LUT, and pads the stored file to exactly 1,620,052,992 bytes. Verification authenticates the physical file, exact roster, embedded LUT, and the byte-for-byte reconstructed member stream without buffering the 1.62 GB wire.

After all exact layers `L000..L042` verify, derive the canonical full-model comparison directly from their immutable receipts (repeat `--receipt` exactly 43 times):

```bash
smash qtip-v7-wire account-model \
  --receipt "$RUN/wire/L000.qtip-v7-wire.receipt.json" \
  ... \
  --receipt "$RUN/wire/L042.qtip-v7-wire.receipt.json" \
  --weight-denominator 671000000000 \
  --weight-denominator-label "declared total model weight parameters" \
  --output "$RUN/wire/QTIP_V7_MODEL_ACCOUNTING.json"
```

The receipt-derived result is QTIP routed stored bytes 69,662,278,656, EXL K2 routed stored bytes 69,662,278,656, routed gap 0, native/base bytes 19,708,797,688 included once per full model, QTIP and EXL full stored bytes 89,371,076,344 each, and full gap 0. Stored-wire BPW is emitted as an exact fraction and decimal against the explicitly declared weight denominator; it is not a decoded-dtype claim.

## Discoverability

```bash
smash qtip-v7-joint-repair --help
smash qtip-v7-joint-repair shard-launch --help
smash qtip-v7-wire --help
```
