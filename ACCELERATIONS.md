# Runtime accelerations

## Exact sparse consumer staging (2026-09-07, unpromoted)

`ArtifactTensorStore` accepts an explicit `native_payload_reads: true` on the
admitted subject. This reads exact exported native bytes with path, source/hash,
size and dtype checks instead of staging whole original source shards. The
default remains source-backed; selected corrupt native payloads never fall back.
The focused suite reports 46 passed, one skipped on the Mac (transformers absent).
This is consumer infrastructure, not a measured producer speed improvement.

The dedicated GLM diagnostic resumes three immutable L004 windows (ordinals
0/1/3) and substitutes four sealed L005 fused experts. Baseline singleton repeats
and grouped candidate share the fixed K2 prefix and suffix. It is explicitly a
paired sparse diagnostic, not native-prefix or uniform-K3 equivalence evidence.
The first physical L005 materialization took 269.6501306 seconds, with 1759 payload
reads and zero source-model reads. Downstream output-KLD remains pending; no
candidate or production default has been promoted. Cold consumer setup and
validation wall are not hidden inside the producer speedup denominator.

The first mixed-K2/K3 replacement consumer then stalled inside Inductor/SymPy
symbolic stride simplification and was stopped with no replacement/forward
sealed. Canonical `73e04f9fd8d02a7e70bc4edd59595ad626a08d3c` keeps
`torch.compile(dynamic=False)` for the packed decoder, without changing its
arithmetic. The physical mixed-tier canary subsequently completed in
1.693785015 seconds and produced finite 4096-by-4096 K3 weights. This is a
mechanical canary, not an independent numerical or held-out quality gate.
The recovered L005 materialization took 155.398136120 seconds; all twelve
replacement bindings and nine L005 arm/window forwards sealed with zero
source-model reads. These sequential cold/warm consumer costs are not a
matched producer speed comparison.

The diagnostic now checkpoints every arm/window/layer. Its unchanged
40-GiB estimated-peak plus 4-GiB reserve gate stopped before L010 allocation
at MemAvailable 47,160,709,120 bytes. All L005--L009 forwards were preserved.
After the source owner reclaimed only three hash-verified duplicate input
shards (16,089,895,160 bytes), the next attempt resumed all nine slots from
L009 and allocated L010 with MemAvailable 66,324,426,752 bytes. No sealed
forward, baseline build, or source scorer was replayed; no resource gate was
weakened.

The suffix has now sealed through L044 and readout. An independent CPU verifier
recomputed binary64 KLD from all nine hash-authenticated readouts and the exact
teacher-selected 8192-token supports (1024 positions/window). This is conditional
teacher-support KLD, not full-vocabulary KL. Singleton repeats were identical;
the predeclared repeat-derived per-window and pooled limits were each 1e-7.
Grouped-minus-singleton deltas for windows 28, 56, and 71 were respectively
-0.014696289583460309, -0.001827300157524414, and +0.0020095746115649515.
The pooled delta improved by -0.004838005043139924, but window 71 failed the
frozen per-window gate. Therefore this grouped candidate is NOT promoted.
This is a held-out sparse diagnostic failure, not a byte-equality rejection or
full-model/uniform-K3 conclusion; do not weaken tolerances after observing it.
The retained independent receipt is
`/dev/shm/t_ebcba52e_glm_suffix_a5/INDEPENDENT_VERIFY_run8290.json` on Spark-6,
with verifier SHA256
`4707a831d4f0aea9d49520a3e189e1967e1173a6fd56af156cbbd27013f83240`.
Producer timing remains separate from this full suffix validation cost.
The next causal direction is to isolate batched linear-algebra numerical changes
from execution-scheduling gains, preserving this failed candidate and all sealed
frontiers rather than replaying them or fitting to these frozen windows.

## Unitwise factorization with resident LDLQ batching (experimental)

The same-input four-cell GLM causal profile isolated a numerical batch-axis
change in block-LDL: max_abs 0.0012226838152855635, 33,338,796 changed entries
among 67,108,864. With exactly the same lower factors and transformed weights,
batched versus singleton LDLQ returned identical quantized values and states for
all four cells. Profile repetitions are instrumentation, never speed evidence.

The opt-in public batch config `block_ldl_unitwise: true` retains singleton
block-LDL factorization while keeping cross-unit LDLQ and resident staging.
Default remains false; mixed/nonboolean settings fail admission. Build receipts
record the selected factorization axis. CPU tests cover actual factorization,
input preservation, configuration admission and controller/builder wiring.
This isolates the measured mechanism rather than changing frozen output gates;
physical speed, paired reconstruction and held-out acceptance are still required.

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
Held-out output KLD remains unmeasured; neither variant is promoted.

Further same-panel results (all opt-in, production defaults unchanged):
- Device profile: generic Viterbi dominated two real cells (768 launches,
  16.006374 s device). CPU synchronization is waiting, not additive opportunity.
  Instrumented wall is never a speed denominator.
- Warp8 vs warp16: warm four-cell 35.272042/35.248076 s vs
  37.844731/38.561673 s (1.072938x/1.094008x). Canonical paired reconstruction
  passes 8/8 comparisons, decoded max_abs 0, validation 8.310000 s.
  That is a bounded numerical result, not measured output KLD or full-model proof.
- Full branch-loop unroll4 vs rolled1, both warp8: warm four-cell
  34.629889/34.678049 s vs 34.893328/35.640234 s (1.007607x/1.027746x),
  24/24 builds pass canonical decode conformance. Near-neutral, not substantial.
  Setup arms 40.015426/37.233308 s include different first-use compilation;
  they do not establish a matched cold-build speedup.
- Compiler metadata still reports 65,536 shared bytes for the generic gather
  in all six compiled warp/unroll variants. Next isolated causal experiment is
  `viterbi_structured_gather=true` for K1 only: select a contiguous predecessor
  row, then fourfold broadcast, rather than a general full-prefix gather.
  Matched physical result: generic8 warm four-cell 34.769903/35.412527 s
  versus structured8 31.155750/31.166651 s (1.116003x/1.136231x), all24 builds
  pass canonical decode conformance. Subsequent canonical paired reconstruction
  passes8/8, decoded max_abs0, separately charged validation7.828695 s.
  Compiler entries for the new movement report4096 shared bytes; this is not a
  measured occupancy claim. Output KLD is still unmeasured, so no promotion.
  Structured16 was then measured under the changed shared-resource envelope:
  warm 32.018977/31.943393 s versus structured8 31.001402/31.182052 s
  (0.968220x/0.976166x). All24 builds pass; retain structured8, not16.
- Structured8 grouped2 vs singleton on the same four cells at `624c210f`:
  warm 28.204494/28.214014 s vs31.105050/31.288451 s
  (1.102840x/1.108968x). All24 builds pass. Final warm producer-core totals
  are26.130852 vs27.864315 s, conformance0.364080 vs0.351104 s,
  staging1.682423 vs3.025813 s; these phases are included in wall.
  Peak allocated/reserved bytes are5213002240/9240051712 grouped,
  2665339904/4611637248 singleton. Setup walls28.948990/33.555770 s
  have first-use/order confounds, not matched cold-JIT evidence.
  Separate canonical reconstruction validator took8.543186 s: only2/8
  numerical comparisons meet baseline-repeat limits. E084 down NMSE
  0.499220566 exceeds0.499113813; fused E0840.597373550 exceeds0.597026913,
  fused E0850.562766462 exceeds0.562458819. E085 down improves.
  This is a numerical diagnostic, not byte-based rejection or measured output
  KLD harm. Preserve candidate for decisive held-out adjudication; no promotion.
- `e782a5fe` adds opt-in K1 `viterbi_backpointer_dtype="uint8"` branch-index
  storage, reconstructing the full state using the implicit prefix column.
  Full branch recurrence and returned int32 wire remain unchanged; default
  int32 and conservative memory admission remain. CPU tests exercise actual
  store/restore expressions over all65536 states plus public installer/launch
  wiring. Physical result: warm byte-q34.983784/35.026989 s vs
  int32 31.160164/31.181123 s (0.890703x/0.890203x), all24 builds pass.
  Final warm peak allocated bytes1058930688 vs2677950464. Canonical
  reconstruction passes8/8 with decoded max_abs0, validation7.999939 s.
  The byte table saves memory but is slower; retain structured8/int32.
  This still does not measure held-out output KLD.

Exact receipts: `t_ebcba52e_k1warps_a1`, `t_ebcba52e_quality_k1warps_a1`,
`t_ebcba52e_unroll_a1`; local terminal copies retain individual phases and peaks.

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
