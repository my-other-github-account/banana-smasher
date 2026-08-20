# Unified V7 in-memory training

This folder is the complete V7 execution unit. The only runtime command is:

```bash
V7_RANK=0 V7_MODEL_ROOT=/local/model_rank0 ./run.sh   # Spark-1, L000-L020
V7_RANK=1 V7_MODEL_ROOT=/local/model_rank1 ./run.sh   # Spark-3, L021-L042
```

`run.sh` performs one bounded closure check and then launches the actual two-node resident trainer. The closure check reads the model index and `stat`s its shards; it does **not** materialize the model. It also verifies the vendored Python/C++/CUDA consumer closure and compiles the grouped native extension before the expensive model load.

## Required external inputs

The code, native sources, decoder package, LP4 helpers, admission, and output layout live here. Large immutable inputs remain external and are passed explicitly:

- `V7_MODEL_ROOT`: local regular model directory for that rank; no SSHFS/JIT path.
- `V7_PARENT_ROOT`: full-parent V7 wire root.
- `V7_LP4_PACK`: LP4 pack directory.
- `V7_LP4_MANIFEST`: LP4 manifest JSON.
- `V7_DELTA_DIR`: directory containing `DELTA_PACK.COMPLETE`.
- `V7_CORPUS`: official TRAIN corpus.
- `V7_TEACH`: published teacher bank.

The default Green paths are documented in `config.env.example`; model roots remain rank-specific because each host owns its local resident copy.

## Output contract

All rank-local output goes below `output/`:

- `closure-rankN.json`: fast closure receipt;
- `rankN/`: trainer status, logs, checkpoints, and receipts;
- `extensions/`: local native extension cache.

No output is written into mission-specific staging directories. A run is accepted only after both rank receipts show `TRAINING_ENTERED` and the requested checkpoint/metric receipt is present.

## Scientific identity

The launcher hard-binds the Green derivative to:

- full-parent basis `98efab45…`;
- admission `76d0674e…`;
- fresh U0 only;
- rank split `0–20 / 21–42`;
- official grouped K2 `[1024]` LUT geometry;
- batch/pipeline microbatch 4;
- 64 TRAIN windows `20..83`;
- no PRE, U23, or lower-LR continuation.

The launcher does not contain a fallback host, one-node mode, continuation checkpoint, or alternate geometry.
