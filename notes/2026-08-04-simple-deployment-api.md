# Native vLLM deployment API PoC

Date: 2026-08-04

## Product contract

The wrapper-based API from the earlier `ddf07fc` slice is superseded. The serving
contract is now:

```bash
# Export environment
smash export ... --output /path/to/model-pack
smash verify /path/to/model-pack

# Serving environment; no repository checkout or exporter package required
python -m pip install \
  --extra-index-url https://YOUR-BANANA-WHEELHOUSE/simple \
  banana-smasher-plugin==0.2.0
vllm serve /path/to/model-pack
```

There is no `smash serve` command, Python serving wrapper, Banana launcher,
model-path environment variable, or alternate API server.

## Native integration

vLLM 0.24 loads `vllm.general_plugins` while constructing its CLI parser. The
Banana plugin installs an idempotent classmethod hook on
`EngineArgs.from_cli_args`. The hook:

1. resolves vLLM's positional `model_tag` using the same precedence as stock
   `ServeSubcommand`;
2. recognizes only a local export whose `config.json` has
   `quantization_config.quant_method=banana_smasher` and whose
   `BANANA_PACK_MANIFEST.json` has `quant_method=banana_smasher`;
3. reads the export's versioned `banana_smasher_runtime` profile, with a
   compatibility profile for existing pre-profile exports;
4. fills only fields that still equal stock vLLM defaults;
5. preserves explicit user CLI overrides;
6. sets required FlashInfer/DeepGEMM process defaults before engine creation;
7. returns control to unmodified vLLM argument and engine construction.

New exports stamp `banana-smasher-vllm-runtime-v1` /
`sm121-single-gpu-v1` into `config.json`. The current V5/U12 artifact predates
that field, so the plugin derives its served-model identity from the manifest.

## Package boundary

`banana-smasher-plugin` no longer depends on `banana-smasher`. Its runtime
metadata declares the exact serving closure: vLLM 0.24.0, FlashInfer Python and
SM121 AOT cache 0.6.17, DeepGEMM 2.6.1, Quack, and safetensors. The serving host
therefore needs the plugin wheel and its wheelhouse dependencies, not an export
repository checkout.

The public Python index does not currently contain the complete patched Linux
ARM64 closure. Until those companion wheels are published, the canonical image
is the dependency-complete install proof. Its default command is now exactly:

```text
vllm serve /model
```

The Docker image no longer bakes Banana-specific vLLM flags or Banana runtime
environment variables; the installed plugin owns them.

## Verification completed

- Plugin runtime-profile tests: `7 passed`.
- Exporter/CLI tests: `2 passed`, `1` macOS-only metadata-refresh skip.
- Docker/source native-contract tests: `6 passed`.
- Real vLLM 0.24 parser smoke inside the live dependency closure: PASS.
- Automatic `vllm.general_plugins` entry-point discovery in a separate CPU-only
  process: PASS.
- Exact tested argv: `vllm serve /model`.
- Plugin source loaded through entry-point metadata, not a direct wrapper call.
- Hook installed exactly once.
- Current real export recognized as profile `sm121-single-gpu-v1` and served name
  `DeepSeek-V4-Flash-BQ3`.
- Real `AsyncEngineArgs` received the Boot10 engine defaults.
- The same real parsed namespace received reasoning, chat-template, and tool-use
  defaults.
- Required runtime environment was produced by the plugin with no external
  environment input.
- No engine or CUDA context was created for the parser smoke; the live Spark-4
  service remained unchanged.

## Remaining physical proof

A fresh GPU boot from a newly built package/image must still run the exact plain
command and reach the OpenAI-compatible health and generation gates. It must be
performed only at an owner-selected safe service boundary; the healthy Boot10
foundation is not restarted solely for this API proof.
