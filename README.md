# Banana Smasher runtime

This standalone repository owns two reusable product paths:

materialized quant source -> `smash export` -> `smash verify` -> self-contained `/model` pack -> pinned stock vLLM image with `vllm.general_plugins` -> OpenAI-compatible API.

bank manifest -> materialized evaluation bank -> exact teacher/candidate producer rows -> resumable KLD -> measured aggregate -> optional parent estimate -> training-only solver row.

It does not contain campaign orchestration, private benchmark ledgers, raw
historical run artifacts, model weights, or logits. The generic anchor API,
schemas, public-safe four-bank provenance bundle, and commands are documented in
`banana-smasher/ANCHOR_EVALUATION.md`. Public frozen competitive evaluation
contracts and result tables live under [`Evals/`](Evals/); separate internal
QTIP family-anchor calculations live under [`Backpack/`](Backpack/).

The public Evals page includes a dedicated EXL 2×3 scope/rate matrix covering
full/all-eligible and routed-experts-only + exact-native-rest K2, greedy K2.5,
and K3 cells. Five cells are measured; the routed greedy K2.5 measurement is
explicitly reserved in progress rather than assigned a guessed score.

## Clone

```bash
git clone https://github.com/my-other-github-account/banana-smasher.git
cd banana-smasher
```

The repository contains two installable Python distributions: `banana-smasher` for export, verification, pack development, and anchor evaluation, and `banana-smasher-plugin` for stock-vLLM serving integration.

## Five-minute Backpack quickstart

Install the ordinary wheel, inspect the built-in tier menu, and run the small
end-to-end fixture. The smoke calls the public generate, candidate-anchor,
predict, exact-solve, materialize, pack-verify, and one-plan build APIs; it also
checks the installed `smash backpack status` CLI.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install ./banana-smasher
smash backpack providers
python banana-smasher/tests/backpack_provider_flat_smoke.py --root /tmp/banana-backpack-smoke
```

For a real model, create a `banana-smasher-backpack-plan-v1` JSON document using
`banana-smasher/schema/banana-smasher-backpack-plan-v1.schema.json`, then run:

```bash
smash backpack build --plan plan.json --run-root ./backpack-run
smash backpack status --run-root ./backpack-run
smash verify ./backpack-run/final-pack
```

Plans can compose native MXFP4, QTIP2/QTIP2.5/QTIP3, and fixed D4 K2048/K4096
providers. Authentic QTIP1/QTIP1.5 declarations are public extension points but
do not yet claim stock-vLLM K1/V1 execution. Python users can independently call
`generate_backpack_candidate`, `anchor_backpack_candidates`,
`predict_backpack_candidate`, `solve_backpack`,
`materialize_backpack_assignment`, and `verify_backpack_candidate`; the
`build_backpack` orchestration path calls those same provider operations rather
than a private duplicate implementation.

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

For model-size reports and public names, use `smash bpw` rather than dividing by
container tensor counts. It emits the versioned whole-model accounting record,
uses canonical base-model logical parameters for the comparable/publication BPW,
and reports any auxiliary-model denominator separately. See
[`banana-smasher/README.md`](banana-smasher/README.md#standard-whole-model-bpw-accounting).

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
