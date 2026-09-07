# Runtime accelerations

## Unpromoted single-Spark QTIP build experiments (2026-09-07)

These are bounded same-input diagnostics, not production quality acceptance.
The DS4 panel is four distinct authentic K1 cells: L013 E084/E085, each down
and fused13, with unchanged source/calibration/geometry/seeds. Decode conformance
is included in build wall; first-use consumer compilation is not producer speed.

- Existing cross-unit batching at `dca8699d2539917c7b1259952eefd7d3649de007`:
  warm four-cell singleton 38.194566 s versus grouped 36.810715 s (1.037594x).
  Grouped numerical reconstruction differs: three cells exceed the strict
  baseline-repeat diagnostic limit, one improves. This is neither a byte-based
  rejection nor proof of meaningful held-out regression.
- Opt-in `viterbi_backpointer_dtype="uint16"` at
  `b5f59663e009d663d2618c460dc46fb6cd95f1e6`: lossless internal K1 backpointer
  compression, int32 returned states and full recurrence unchanged. Warm default
  int32 four-cell repeats 37.961138/38.300984 s versus uint16
  48.923851/48.165708 s (0.775923x/0.795192x): **negative speed result**.
  Final warm arm peak allocated/reserved bytes were 2677950464/4624220160
  (int32) and 1595801600/2468347904 (uint16). The storage reduction did not
  accelerate this panel. Packed assignments match on all four cells as a
  diagnostic only. Default remains int32; no production activation.

Provenance: task `t_ebcba52e`, dedicated Spark-6; immutable experiment roots
`/run/user/1000/t_ebcba52e_ds4_a1`, `t_ebcba52e_quality_ds4_a1`, and
`t_ebcba52e_bp16_a1` on that host. Compact terminal receipts are retained in the
task workspace as `DS4_A1_TERMINAL.json` and `BP16_A1_TERMINAL.json`.
Held-out output KLD remains unmeasured; neither variant is promoted. Next causal
step is a bounded default-K1 device/CPU cost profile, not another blind storage
or batch-size variant. A profile's instrumented wall is never a speed denominator.

`runtime/ACCELERATION_MANIFEST.json` is the exact machine-readable inventory. This document is its concise operator view.

| ID | Development source | Image build input | Runtime activation | Principal test |
|---|---|---|---|---|
| `bs-pack-export-verify` | exporter, schemas, repair/repack/materialized-wire code | `banana-smasher` wheel | `smash export`, `smash verify`, `smash serve-check` | exporter CLI/contract/materialized-wire tests |
| `stock-vllm-general-plugin` | plugin entry point and `register()` | plugin wheel built from checkout | stock vLLM discovers `vllm.general_plugins` | plugin contract tests |
| `native-plane-p1016` | native-plane loader and quantization config | NumPy, safetensors, plugin wheel | `quant_method=banana_smasher` selects native routed experts behind a `cudagraph_unsafe` opaque custom op; breakable `PIECEWISE` capture is mandatory and other capture modes fail closed | native-plane runtime, compile-boundary, and image-default tests |
| `p1016-cutedsl-tlut` | P1016 kernels plus packaged QTIP TLUT | quack-kernels and `qtip_tlut.npy` | fail-closed `mixed_exact_gemv` dispatch | CuteDSL and native-plane tests |
| `sm121-deepgemm-dense-e8m0` | SM12x O-projection and dense preflight hooks | official DeepGEMM 2.6.1 `nv_dev_f8e8fb5` source | stock SM100+ packed-scale O-projection with `VLLM_USE_DEEP_GEMM=1` and `VLLM_USE_DEEP_GEMM_E8M0=1` | SM121 dense/V4 tests |
| `deepgemm-ue8m0-warmup` | dense/grouped warmup scale initializer | same pinned DeepGEMM source wheel | initializes every dummy UE8M0 activation scale to one before DeepGEMM warmup | dense-capability tests |
| `sm121-deepgemm-sparse-indexer` | external DeepGEMM registration hook | same pinned DeepGEMM source wheel | boot-time SM12x lazy symbol registration | sparse-indexer and dense-capability tests |
| `sm121-persistent-topk` | TopK correction hook | stock vLLM persistent TopK op | replaces unsupported cooperative TopK on SM12x only | sparse-indexer TopK tests |
| `sm121-v4-attention-flashinfer` | V4 attention selector hook | source-built FlashInfer with pinned fixes | SM12x FlashMLA request routes to FlashInfer sparse MLA | attention tests |
| `stock-deepgemm-mhc` | plugin preserves stock MHC dispatch | pinned DeepGEMM source wheel | no plugin override; stock public backend remains active | SM121 MHC tests |
| `flashinfer-sparse-decode-compat` | sparse-decode signature adapter | source-built FlashInfer | one-time API variant adapter during plugin registration | FlashInfer compatibility tests |
| `sm120-aot-cubins` | 26 SM120 cubins plus immutable producer map | exact names, bytes, and SHA-256 admitted from `runtime/ASSET_MANIFEST.json` | AOT root and MoE W2 environment paths | exact source/image admission and native runtime tests |
| `e43-aot-cubins` | 6 E43 cubins plus immutable producer map | exact names, bytes, and SHA-256 admitted from `runtime/ASSET_MANIFEST.json` | MoE W3 cubin environment path | exact source/image admission tests |
| `flashinfer-autotune-cache` | version-aware validator and excluded 0.6.14 provenance only | no cache is baked; 0.6.17/121a must be generated on a GPU | persistent named volume at the versioned vLLM cache root | mismatch rejection and capture tests |
| `real-libcudart-link` | checked-in FlashInfer patch | real CUDA 13 runtime link replaces TileLang stub | image-build verification imports FlashInfer against real runtime | Docker static/image verification tests |

## Stage coverage

Development includes both package sources, all package/plugin tests, JSON schemas, repair/repack/materialized-wire handling, and every AOT asset consumed by the image. Image build compiles both local wheels, source-builds pinned FlashInfer and DeepGEMM revisions, verifies package imports, links real `libcudart`, writes package provenance, and preserves the exact stock-vLLM `CMD`. Serving mounts only a verified pack at `/model`; plugin registration activates fail-closed runtime hooks before model load.

`runtime/KERNEL_PRODUCERS.json` and `archive/KERNEL_DEVELOPMENT.md` distinguish shipped/hash-gated cubins from exact-source-rebuild seals. The SM120 set remains unsealed because cubit short identity `5912400` is unresolved. The E43 recipe has a sealed independent 6/6 byte-identical rebuild receipt.
