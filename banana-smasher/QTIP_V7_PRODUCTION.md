# Canonical QTIP1/QTIP4 V7 production

QTIP1 V7 and QTIP4 V7 use `banana_smasher.qtip3_regenerate` unchanged: the
same public `build_qtip_native_cells` codebook/plane path proven for QTIP3 at
the `d2c844d8` lineage pin. `QTIP3_TIER` selects only the validated immutable
geometry and receipt identity:

| tier | provider | `(B,L,V)` | exact code BPW |
| --- | --- | --- | --- |
| `qtip1_v7` | `qtip-native-v6@1.00` | `(4,16,4)` | 1.00 |
| `qtip3_v7` | `qtip-native-v6@3.00` | `(12,16,4)` | 3.00 |
| `qtip4_v7` | `qtip-native-v6@4.00` | `(16,16,4)` | 4.00 |

Unknown names and mismatched legacy `QTIP3_BPW` values fail before admission.

## DeepSeek-V4-Flash-0731 invocation

Set the existing Q3 authority, host, roster, layer, control-map, and claim
variables exactly as for the proven Q3 run, plus one of the tier names:

```sh
export QTIP3_TIER=qtip1_v7                 # or qtip4_v7
export QTIP3_ROOT=/mission/qtip1-v7        # task-private durable mission root
export QTIP3_MODEL_INDEX=/home/dnola/models/hf/DeepSeek-V4-Flash-0731/model.safetensors.index.json
export QTIP3_TASK_ID=t_REPLACE
export QTIP3_BOARD_RUN_ID=REPLACE
export QTIP3_HOST=spark-REPLACE
export QTIP3_ALLOCATION='HOST_ALLOCATION t_REPLACE spark-REPLACE qtip1-v7'
export QTIP3_DRIVER_SHA=REPLACE_64_HEX
export QTIP3_EXPECTED_CLAIM=REPLACE_64_HEX
export QTIP3_LAYERS=2
export QTIP3_CONTROL_ROOT=/mission/qtip-controls
export QTIP3_CONTROL_MAP='{"2":"qtip3-control-prefix"}'
export QTIP3_CELL_ROSTER_PATH=/mission/QTIP1_MISSING.json
export QTIP3_CELL_ROSTER_EXPECTED_COUNT=REPLACE
python -m banana_smasher.qtip3_regenerate
```

For a bounded production proof set `QTIP3_MAX_NEW_BATCHES=1`. For the existing
bitwise baseline-versus-batched CUDA smoke path set `QTIP3_SMOKE_COUNT=20`.
QTIP4 uses the same command with `QTIP3_TIER=qtip4_v7`, a separate mission
root/allocation/roster, and no producer fork.

## Output, resume, and verification

Each selected cell is written below
`$QTIP3_OUTPUT_ROOT/Llll_Eeee_{down|fused13}/` (default
`$QTIP3_ROOT/outputs/full_api/`). Production cleanup retains `codes.npy`,
`CELL_RECEIPT.json`, and `PUBLIC_CELL_RECEIPT.json`. Controller receipts are
under `$QTIP3_ROOT/receipts/`; monotone progress is also mirrored to
`$QTIP3_ROOT/PROGRESS.json`.

Resume uses the same command and mission root. A cell is skipped only when its
PUBLIC receipt is PASS and matches task, basis, cell, tier-derived provider,
zero fallback calls, and a positive CUDA decode count. A sealed API receipt is
adopted after source/control/TLUT SHA verification if the wrapper stopped
before writing PUBLIC_CELL. Other partial cells are regenerated.

Verify exact bytes with:

```sh
find "$QTIP3_ROOT/outputs/full_api" -name PUBLIC_CELL_RECEIPT.json -print0 \
  | xargs -0 shasum -a 256
shasum -a 256 "$QTIP3_ROOT/receipts/PRODUCER_TERMINAL.json"
```

`PUBLIC_CELL_RECEIPT.json` uses
`banana-smasher-qtip3-v7-public-api-producer-v1-cell` and records tier,
provider, `(B,L,V)`, BPW, basis, cell identity, API-receipt SHA, and CUDA/fallback
counters. Its file SHA is the resume/terminal authority.

## CLEAN102 predicted-KLD option rows

After CLEAN102 scoring supplies a prediction receipt, non-negative per-class
predicted KLD, and exact retained `codes.npy` bytes, call
`build_clean102_option_row(...)` with the PUBLIC receipt, CLEAN102 prediction
receipt, `Qtip3ApiConfig.for_tier(...)`, bank/teacher/scorer SHA-256 values, and
those measurements. Emit the returned object as one canonical JSONL line. It uses
`banana-smasher-provenance-option-row-v1`, model
`deepseek-ai/DeepSeek-V4-Flash-0731`, revision `0731`, the canonical V7 tier,
no activation charge, and independently SHA-bound `prediction_producer` and
`physical_producer` descriptors; the latter also binds the exact `codes.npy`
artifact SHA. The helper refuses a receipt whose tier,
provider, geometry, BPW, basis, identity, hashes, byte count, or KLD values are
inconsistent.