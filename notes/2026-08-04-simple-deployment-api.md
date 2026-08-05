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

- Targeted API and CLI tests: `5 passed`.
- `banana_smasher-1.0.0-py3-none-any.whl`: built and ZIP-verified.
- Installed `smash` executable: replaced itself with a fake `vllm` binary; the
  binary received the model path, all serving defaults, port override, and the
  four required FlashInfer/DeepGEMM process defaults.
- Live Spark-4 comparison: the generated default `vllm serve /model` argv
  matches the running Boot10 process argv argument-for-argument.

A second real GPU boot through this wrapper remains the final deployment proof;
the healthy Spark-4 endpoint was not interrupted for this API slice.
