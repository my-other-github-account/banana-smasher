# Simple deployment API PoC

Date: 2026-08-04

## Scope

This slice starts from the exact live-service source revision
`d7bda11b4057f0a256ac4635623135238fda5601` and adds one user-facing serving
transaction:

```bash
smash serve /path/to/model-pack
```

The Python API is `banana_smasher.serving.serve`. It performs only a lightweight
model identity check, applies the working runtime process defaults, and replaces
itself with stock `vllm serve`. It is not a second serving implementation.

The currently dependency-complete path is:

```bash
smash serve /path/to/model-pack \
  --container-image banana-smasher-runtime:local
```

The intended wheelhouse path is:

```bash
python -m pip install --find-links /path/to/banana-wheelhouse \
  -r requirements-serve.txt
python -m pip install --find-links /path/to/banana-wheelhouse \
  banana-smasher-plugin==0.2.0
smash serve /path/to/model-pack
```

The plugin depends on `banana-smasher==1.0.0`, so that second install supplies
both the vLLM general plugin and the `smash` command.

## Dependency boundary

The public package index does not currently expose the exact proven dependency
closure: vLLM 0.24.0, patched FlashInfer 0.6.17 plus its SM121 AOT cache wheel,
and pinned DeepGEMM 2.6.1. `requirements-serve.txt` therefore describes the
Banana Smasher wheelhouse contract; it is not represented as a working public
index install. The pinned image remains the honest executable PoC closure until
those companion wheels are published.

## Verification

- Targeted API, CLI, and source-closure tests: `29 passed`.
- `banana_smasher-1.0.0-py3-none-any.whl`: built and ZIP-verified.
- `banana_smasher_plugin-0.2.0.tar.gz`: built without `CUDA_HOME`, contains
  the required CUDA/C++ sources and `banana-smasher==1.0.0` dependency metadata;
  test artifact SHA-256
  `a1c9ef7e92be2d3ec73b6704551b426356d073af274c35774fae9f21abe4b82a`.
  The platform wheel intentionally remains a CUDA-host build.
- Installed `smash` executable: replaced itself with a fake `vllm` binary; the
  binary received the model path, all serving defaults, port override, and the
  four required FlashInfer/DeepGEMM process defaults.
- Live Spark-4 comparison: the generated default `vllm serve /model` argv
  matches the running Boot10 process argv argument-for-argument.
- Published-package real-host boundary: commit `ddf07fceaec8e20308af29507902384e162f16ec`
  was cloned and installed on Spark-4, then run against the actual read-only model
  mount and live image reference. Receipt
  `/home/dnola/missions/SMASH_API_DDF07FC/REAL_SPARK4_DEPLOYMENT_BOUNDARY.json`
  has SHA-256
  `9f5c782328502844541dc2aa223b40a9c18a909feeebc96d5dec1bdc13524d2f`.
  It records exact generated/live argv equality, plugin distribution `0.2.0`
  with `banana_smasher_plugin:register`, and unchanged service PID/startticks
  with health `200 -> 200`.

A second real GPU boot through this wrapper remains the final deployment proof.
At verification time Spark-8 was already owned by an active `run514-a32` image
build, so neither it nor the healthy Spark-4 endpoint was interrupted.
