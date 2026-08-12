# QTIP V7 public API repair proof

Implementation commit: `e325e39c6eb09f9efedd70a98528c26b42de1168`

Branch: `feat/qtip-v7-api-repair-20260811`

Focused proof command:

`PYTHONPATH=banana-smasher/src python -m pytest -q banana-smasher/tests/test_qtip_v7_repair_artifact.py --basetemp=/tmp/t_4d1cd140-qtip-v7-receipts`

Result: `3 passed in 5.61s`.

The focused proof uses the genuine QTIP2 V7 raw member geometry: each projection member is a 2,109,444-byte opaque wire consisting of 2,097,152 packed-code bytes, FP16 SUH/SVH arrays, and one FP32 scale. The three L033 E000 w1/w2/w3 members share exactly one external 1024-value FP16 layer LUT.

## Update 0

`smash qtip-v7-export --manifest $MANIFEST --output $UPDATE0_DIR`

The export/readback receipt reported:

- update: 0
- packed identity: true
- complete logical wire bytes: 2,111,492 for the one-member update-0 fixture
- layer LUT bytes: 2,048
- wire-size delta: 0
- update-0 LUT readback: byte-identical

The overlay directory contains no copied member wire. Its manifest references the immutable source member by path and SHA-256 and materializes only the already-billed layer LUT slot.

## Update 1 public invocation

Build a physical repair bundle using only TRAIN tensors:

`smash qtip-v7-bundle --manifest $MANIFEST --training $TRAIN_PT --output $BUNDLE_PT --learning-rate $LR --member 33:0:w1 --member 33:0:w2 --member 33:0:w3`

Create a `banana-smasher-physical-repair-request-v1` request binding the bundle path, bundle SHA-256, and device, then execute one existing public physical update:

`smash update --backend physical-repair --request $REQUEST --identity $IDENTITY --output $UPDATE_PT --receipt $UPDATE_RECEIPT --tokens 1 --segments 1 --batch-size 1 --available-bytes $AVAILABLE --resident-frozen-bytes $FROZEN --trainable-bytes 4096 --optimizer-bytes 0 --staging-bytes 0 --activation-bytes-per-token 1`

Export and read back the repaired overlay:

`smash qtip-v7-export --manifest $MANIFEST --update-artifact $UPDATE_PT --output $UPDATE1_DIR`

Focused update-1 receipt values:

- fixed members: 3 (w1/w2/w3)
- trainable layer LUTs: 1
- finite nonzero gradient: true
- gradient max absolute value: 1,097,340,420,096.0
- LUT max absolute delta: 1.0973403453826904
- TRAIN objective before: 547,611,475,968.0
- TRAIN objective after: 13,318,833,152.0
- TRAIN objective improved: true
- packed indices frozen: true
- decoded once through the installed layer graph: true
- update-1 packed member identity: true
- update-1 complete logical wire bytes: 6,330,380
- update-1 layer LUT bytes: 2,048
- update-1 wire-size delta: 0

The backend owns one FP32 `[1024]` trainable per selected layer and aliases it across that layer's causal w1/w2/w3 consumers. Trellis/codes, transforms, geometry, scales, assignments, and member wire bytes remain buffers or immutable referenced files. Export casts the learned layer parameter back into the same external 1024-value FP16 slot; it adds no member bytes or side channel.
