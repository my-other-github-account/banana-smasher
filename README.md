# Banana Smasher runtime

This standalone repository owns one path only:

materialized quant source -> `smash export` -> `smash verify` -> self-contained `/model` pack -> pinned stock vLLM image with `vllm.general_plugins` -> OpenAI-compatible API.

It does not contain training, solver orchestration, benchmark ledgers, or historical run artifacts.

## Clone

```bash
git clone https://github.com/my-other-github-account/banana-smasher.git
cd banana-smasher
```

The repository contains two installable Python distributions: `banana-smasher` for export, verification, and pack development, and `banana-smasher-plugin` for stock-vLLM serving integration.

## Simple deployment API

The user-facing serving contract is one model artifact plus one installed vLLM
plugin. Banana Smasher remains a thin launcher around stock vLLM:

```bash
python -m pip install --find-links /path/to/banana-wheelhouse -r requirements-serve.txt
python -m pip install --find-links /path/to/banana-wheelhouse banana-smasher-plugin==0.2.0
smash serve /path/to/model-pack
```

`banana-smasher-plugin` depends on `banana-smasher`, so installing the plugin also
installs the `smash` API. `smash serve` checks the model identity, applies the
known-working Boot10 arguments and process defaults, then `exec`s ordinary
`vllm serve`. It accepts `--host`, `--port`, and `--served-model-name`; use
`--dry-run` to print the exact launch transaction.

The public Python index does not yet carry the patched vLLM 0.24, FlashInfer
0.6.17 AOT, and DeepGEMM 2.6.1 wheel closure named by
`requirements-serve.txt`. The modestly functional deployment path available
today is therefore the same API backed by the pinned image:

```bash
IMAGE=banana-smasher-runtime:local examples/build_image.sh
python -m pip install ./banana-smasher
smash serve /path/to/model-pack --container-image banana-smasher-runtime:local
```

This mounts only the model artifact read-only (plus the persistent FlashInfer
cache volume) and runs the same stock `vllm serve /model` command. There is no
campaign launcher, model-path environment variable, or alternate serving stack
in the user flow.

## Build and test both Python packages on a development host

The following is the non-GPU static development gate. It builds, inspects, installs,
and tests both wheels; installing only the exporter package is not a complete runtime
setup.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip build
python -m pip install pytest==8.4.2 ruff==0.12.7 numpy==2.3.5 safetensors==0.8.0 torch
python -m build --wheel --outdir dist ./banana-smasher
python -m build --wheel --outdir dist ./banana-smasher-plugin
python -m zipfile -t dist/banana_smasher-1.0.0-py3-none-any.whl
python -m zipfile -t dist/banana_smasher_plugin-0.2.0-py3-none-any.whl
python -m pip install --no-deps --force-reinstall \
  dist/banana_smasher-1.0.0-py3-none-any.whl \
  dist/banana_smasher_plugin-0.2.0-py3-none-any.whl
python -m pytest -q banana-smasher/tests banana-smasher-plugin/tests docker/tests tests
```

These host tests do not install stock vLLM, FlashInfer, DeepGEMM, CUDA, or the
GPU kernels. The complete runtime install is the pinned Linux ARM64 image below;
only an SM120/SM121 GPU boot can prove its accelerator paths.

## Export and verify a model pack

`RUNTIME_FLOOR_BYTES` is mandatory for the P1016 path. Set it to the measured value from the caller-provided runtime receipt; this repository deliberately provides no default.

```bash
export QUANT_SOURCE=/path/to/materialized-quant-source
export SERVING_MODEL=/path/to/serveable-base-model
export MODEL_OUT=/path/to/model-pack
export MODEL_ID=your-model-id
export INSTANCE_ID=your-pack-instance
export RUNTIME_FLOOR_BYTES=MEASURED_RECEIPT_VALUE

smash export --source-root "$QUANT_SOURCE" --runtime-floor-bytes "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}" --serving-model-root "$SERVING_MODEL" --output "$MODEL_OUT" --model-id "$MODEL_ID" --instance-id "$INSTANCE_ID" --link-mode copy
smash verify "$MODEL_OUT"
```

`examples/export_model.sh` provides the same fail-closed command. The resulting directory is self-contained and is the only directory mounted at `/model` for serving.

## Build and serve the pinned Linux ARM64 image

The release helper uses Docker Buildx with `--platform linux/arm64` and
`--no-cache`; it never reuses an unproven release layer.

```bash
IMAGE=banana-smasher-runtime:local examples/build_image.sh
MODEL_DIR="$MODEL_OUT" IMAGE=banana-smasher-runtime:local examples/serve.sh
```

`examples/serve.sh` creates and mounts the named volume
`banana-smasher-flashinfer-cache` at
`/root/.cache/vllm/flashinfer_autotune_cache` by default, so a valid generated
FlashInfer 0.6.17 cache survives container restarts. Set
`FLASHINFER_CACHE_VOLUME=''` only for an intentionally ephemeral run. The image's
exact `CMD` runs stock `vllm serve /model` with the pinned runtime defaults; do
not replace it with an alternate launcher.

## OpenAI API smoke test

```bash
python examples/smoke_api.py
```

The smoke script checks `/health`, proves `/v1/models` contains the expected
served model, and only then requires a non-empty `/v1/chat/completions` response.

## Capture a future FlashInfer 0.6.17 SM121 cache

After a full GPU warmup has generated the cache in a running container, capture
and validate it without renaming or editing any cache JSON:

```bash
CONTAINER=banana-smasher-runtime \
CACHE_CAPTURE_DIR=/path/to/new-empty-capture \
examples/capture_flashinfer_cache.sh
```

The validator rejects a path other than `0.6.17/121a`, mismatched
`_metadata.flashinfer_version`, malformed members, symlinks, and unexpected
files. No current-version cache is baked in this repository; regeneration,
capture, and image admission remain an explicit Linux ARM64 SM121 hardware gate.

## Verification surfaces

- `ACCELERATIONS.md` explains every retained acceleration.
- `runtime/ACCELERATION_MANIFEST.json` is the machine-readable source/build/activation/test map.
- `PROVENANCE.md` and `provenance/SOURCE_INVENTORY.json` bind retained files to source commit `c00714c6803f7e2de7a95d103dbe172236b22adf`.
- `python -m pytest -q` runs package, plugin-feasible, Docker-static, and extraction-contract tests.

The CUDA image build and GPU boot require a Linux ARM64 CUDA host with an SM120/SM121 GPU. They are a separate hardware gate; a macOS test pass is not evidence of a GPU boot.
