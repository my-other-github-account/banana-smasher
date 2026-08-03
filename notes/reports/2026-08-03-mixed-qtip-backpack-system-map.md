# Mixed-QTIP Backpack system map and deployment reconstruction

**Date:** 2026-08-03
**Status:** CANONICAL SYSTEM MAP; GPU REPRODUCTION STILL OPEN
**Static baseline audited:** `6614d1199ca87b70f74d67f9a636793c7ff6807f`
**Product:** Banana Smasher mixed-QTIP Backpack

This report records the public-safe system knowledge needed to build, repair, export, serve, and improve the product without depending on a private campaign tree or an earlier deployed artifact.

## 1. Product identity

The production Backpack mixes these expert-plane families on one fixed 43-layer, 256-expert basis:

- QTIP2;
- QTIP3;
- D4 with explicit codebook cardinalities, including k1024, k2048, and k4096;
- native MXFP4.

The assignment authority shared by V4 and V5 is:

```text
F521 assignment SHA-256
f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39
```

The sealed active-overlay identity used by the V4 lineage is:

```text
9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62
```

Repair identities:

| Product point | Update | Checkpoint SHA-256 | Role |
|---|---:|---|---|
| V4 | 4 | `082d6d8cf64e83f478c637428779487a3248d0559a1b4a27abfb7f681baa5d41` | accepted U004 repair authority |
| V5 | 12 | `429b644b22920012e8e6be1da07617b4b2ff68aad4a2b2baca5602882cb94fdf` | U012 repair authority being reproduced |

V5 pack identities currently verified at the pack-materialization gate:

| Surface | SHA-256 |
|---|---|
| outer pack manifest | `2d47c50a43450fac17727987beebee84faf90140815cdc54adbb3b0bc4104098` |
| repair manifest | `fb01855311d3cbae507e01e40e890512fcde8af9dfa347fbe54b5df7cedbbcba` |
| base model index | `58c9d59dfe8fd1e7e833be131043f4b45bfa27064fc19b9fa4fffa6475f2d0fc` |
| assignment | `f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39` |
| repair checkpoint | `429b644b22920012e8e6be1da07617b4b2ff68aad4a2b2baca5602882cb94fdf` |

The verified V5 selection contains 22,016 expert projections:

| Family | Projection count |
|---|---:|
| QTIP2 | 2,266 |
| QTIP3 | 14,979 |
| D4 k1024 | 197 |
| D4 k2048 | 2,610 |
| D4 k4096 | 1,910 |
| native MXFP4 | 54 |
| **Total** | **22,016** |

The verified V5 materialized model surface contains 2,063 declared files and 110,364,099,133 manifest-covered bytes. This is a pack-integrity result only; it is not yet a boot, coherence, or performance result.

## Immediate V5 deployment policy

The existing verified U012/F521 result is the current deployment input. Do not regenerate its candidates, Backpack assignment, pre-repair measurements, or repair. The current project begins at handoff verification, final pack binding/export, plugin/image boot, and performance restoration.

The future-facing constraint is narrower: the U012 deployment path must not require a hidden patch, side artifact, or manual mutation that a future end-to-end pipeline would not emit. The future pipeline must produce an equivalent handoff; it does not need to be rerun to prove U012 serving now.

## 2. Canonical repository map

| Responsibility | Canonical location |
|---|---|
| exporter/validator CLI | `banana-smasher/src/banana_smasher/cli.py` |
| pack contract and atomic export | `banana-smasher/src/banana_smasher/contract.py` |
| repair checkpoint validation/materialization | `banana-smasher/src/banana_smasher/repair.py` |
| safetensors repack | `banana-smasher/src/banana_smasher/repack.py` |
| exact-byte Backpack solver | `banana-smasher/src/banana_smasher/knapsack.py` |
| explicit dynamic dimensions | `banana-smasher/src/banana_smasher/backpack_dimensions.py` |
| validation ceremony | `banana-smasher/src/banana_smasher/validation.py` |
| pack schemas | `banana-smasher/schema/` |
| pack format specification | `banana-smasher/PACK_FORMAT.md` |
| stock-vLLM plugin registration | `banana-smasher-plugin/src/banana_smasher_plugin/__init__.py` |
| quantization config and routed experts | `banana-smasher-plugin/src/banana_smasher_plugin/quantization.py` |
| plane loading and dispatch | `banana-smasher-plugin/src/banana_smasher_plugin/native_planes.py` |
| QTIP/D4 kernels | `banana-smasher-plugin/src/banana_smasher_plugin/p1016_kernels.py` |
| packaged QTIP table | `banana-smasher-plugin/src/banana_smasher_plugin/qtip_tlut.npy` |
| dense runtime repair application | `banana-smasher-plugin/src/banana_smasher_plugin/repair.py` |
| AOT cubins | `banana-smasher/kernels/cubins-sm120/`, `banana-smasher/kernels/cubins-e43/` |
| machine-readable acceleration map | `runtime/ACCELERATION_MANIFEST.json` |
| exact AOT asset admission | `runtime/ASSET_MANIFEST.json` |
| kernel producer provenance | `runtime/KERNEL_PRODUCERS.json` |
| pinned image build | `docker/Dockerfile` |
| serve defaults | `docker/runtime_defaults.json` |
| image verification | `docker/scripts/verify_public_image.py` |
| release examples | `examples/` |
| decisions, reports, and tables | `notes/` |

The canonical repository is the only place where new product source or documentation lands. Protected campaign material is evidence for semantic ports; it is not an importable build dependency.

## 3. Deployment handoff inputs

Current V5 deployment may begin with the existing hash-verified U012 handoff:

1. F521 assignment and active overlay;
2. selected QTIP2/QTIP3/D4/native-MXFP4 planes and layer manifests;
3. U012 repair checkpoint/state and repair manifest;
4. exact base-model shards, index, config, tokenizer, and generation metadata;
5. outer pack manifest and all bound file hashes;
6. canonical Banana Smasher source revision;
7. supported Linux ARM64 SM120/SM121 hardware for the physical gate.

A future fresh run begins with these upstream authorities:

1. exact base-model revision and complete model files;
2. tokenizer and generation metadata from that same revision;
3. calibration/trace inputs required to generate candidate payloads;
4. own-base teacher logits and evaluation-bank identities;
5. routing/class features used by Backpack dimensions;
6. a pipeline configuration declaring tier menu, byte envelope, class ceilings, repair configuration, and performance protocol;
7. a canonical Banana Smasher source revision.

That future run must emit a handoff equivalent to the accepted U012 structure. The same deployment/export/plugin path must consume both. Current U012 serving must not be delayed by recomputing the upstream pipeline.

## 4. Deployment handoff and future run-root layout

The target portable future run root is:

```text
RUN_ROOT/
  RUN_MANIFEST.json
  inputs/
    base-model.json
    data-authorities.json
    pipeline-config.json
  candidates/
    ledger.jsonl
    dimensions.jsonl
    class-ceilings.json
    anchors/
    payloads/
  backpack/
    dimensions.complete.jsonl
    assignment.json
    active-overlay.json
    selection-receipt.json
  pre-repair/
    model/
    verification.json
    evaluation.json
  repair/
    checkpoint.pt
    checkpoint.json
    training-receipt.json
  final/
    model/
    verification.json
    serve-check.json
  release/
    exporter-wheel.json
    plugin-wheel.json
    image.json
    api.json
    performance.json
```

All manifest paths are relative to `RUN_ROOT`. Every file descriptor includes byte count, SHA-256, producer revision, producer command/API, and status. A path that escapes the run root is fatal.

For current V5, a handoff manifest should bind the already-existing U012 `backpack/`, `repair/`, model, and identity surfaces into the subset needed by final export and serving. Materializing that manifest is verification/packaging work, not a rerun of upstream production.

## 5. Current deployment start and future full-pipeline stages

Current V5 starts at section 5.8 after verifying the existing U012 handoff. Sections 5.1–5.7 specify what a future full pipeline must produce; they are not current U012 prerequisites.

### 5.1 Freeze base and instruments

- Hash the complete model index, config, tokenizer metadata, and teacher identity.
- Freeze the evaluation bank before candidate selection.
- Use the model's own base as KLD teacher.
- Freeze the six classes: agentic, chat, code, multilingual, prose, and reasoning.
- Freeze the integer byte envelope and runtime-residency floor.

Output: `inputs/*.json` plus `RUN_MANIFEST.json`.

### 5.2 Generate the complete candidate menu

For every eligible layer/expert/projection cell, produce every intended tier candidate and its physical payload from the declared base and calibration inputs.

Each candidate row must bind:

- basis SHA-256;
- layer, global expert, and projection;
- family and D4 cardinality where applicable;
- packed wire bytes, not an unpacked integer container size;
- source payload hashes;
- producer revision and command;
- explicit status.

Incomplete tier/layer coverage remains preliminary and cannot enter allocation.

### 5.3 Bind dynamic Backpack dimensions

`smash backpack-dimensions` joins only explicit per-candidate inputs:

- six-class predictions;
- routing importance;
- projection weight/correction;
- exact physical bytes;
- source sidecar hashes;
- class ceilings.

Aggregate-to-cell inference is forbidden. A missing candidate dimension blocks allocation.

### 5.4 Solve the exact-byte Backpack

`smash knapsack --run-root ... --envelope-bytes ...` validates the run-root-local anchor and damage descriptors, exact basis, tier coverage, and integer byte counts before solving.

Selection must be:

- class-aware;
- code-favoring according to the frozen policy;
- exact-integer, not floating-point approximate;
- one assignment per cell;
- within the declared byte envelope;
- published once with a deterministic receipt.

Outputs: assignment, active overlay, selected candidate set, exact byte accounting, and selection receipt.

### 5.5 Materialize selected mixed-QTIP payloads

Materialize only the selected QTIP2, QTIP3, D4, and native MXFP4 payloads into a canonical quant source. Preserve global expert IDs and fused13/down boundaries. Generate layer metadata, tier/subtier maps, and source receipts.

This stage is the physical source for both pre-repair and final export. It is never mutated by repair or pack publication.

### 5.6 Export and evaluate pre-repair

Run `smash export` without repair inputs against the selected quant source and complete serving-model root. Verify immediately with `smash verify`.

The pre-repair pack is used to measure:

- global and six-class KLD;
- top-1 agreement;
- GB and packed bpw;
- any train/repair losses required by the repair producer.

The pre-repair pack is an immutable measured output, not a hidden base that final export edits in place.

### 5.7 Train repair

Train repair from the same run's selected source, pre-repair behavior, teacher, and frozen repair configuration. Emit a weights-only `bs-basic-repair-v1` checkpoint.

The current repair contract expects exactly:

- 196 repaired codebooks;
- 235 RMSNorm tensors;
- 43 attention output gains.

The checkpoint header binds the update number and physical repair mechanism. Non-finite values, shape drift, wrong counts, wrong source-wire hashes, or a partial state fail closed.

### 5.8 Export the final repaired pack

For current V5, final export reads the existing verified U012 handoff. For a future run it reads the equivalent newly generated handoff. In both cases the declared inputs are:

- repair checkpoint and checkpoint SHA-256;
- active overlay and SHA-256;
- assignment and SHA-256;
- repair update;
- complete serving-model root;
- measured runtime floor.

`smash export` materializes repaired codebooks into the new output, writes dense repair state, binds all identities, writes `PACK_COMPLETE`, publishes the manifest last, and runs `smash verify`.

Final export must not edit the pre-repair pack or selected source. Hardlinks are allowed only as an output-space optimization on the same filesystem; they do not change authority or permit source mutation.

### 5.9 Build the release image

Build both wheels from one canonical revision. Verify wheel ZIP contents and SHA-256. Build the pinned Linux ARM64 image without cache.

Pinned runtime surfaces currently include:

- stock vLLM 0.24 image by digest;
- source-built FlashInfer at the pinned revision/fix set;
- DeepGEMM 2.6.1 at pinned public source revision;
- packaged QTIP TLUT;
- exact admitted SM120 and E43 cubins;
- real CUDA runtime linkage;
- runtime defaults from `docker/runtime_defaults.json`.

The final image contains the plugin wheel and accelerators. The only model mount is the verified final pack at `/model`.

### 5.10 Serve and measure

Stock vLLM discovers `banana_smasher_plugin:register` via `vllm.general_plugins`. The plugin registers `quant_method=banana_smasher`, validates the local model root, runs dense weight-map preflight, loads native mixed planes, applies dense repair state, and admits only physically supported fast paths.

The API gate runs before performance:

1. `/health`;
2. `/v1/models` exact served model;
3. coherent non-streaming completion;
4. coherent streaming completion;
5. restart and repeat without artifact mutation.

Then run the frozen warmup and performance protocol in `notes/tables/2026-08-03-runtime-performance-gates.md`.

## 6. Previously accepted serving mechanics to preserve semantically

The accepted mixed-QTIP runtime combined:

- dense-all prefill behavior;
- grouped plane loading and consolidation;
- packed QTIP2/QTIP3/D4/native plane execution;
- a singleton decode hot path;
- scalar row handling for partially populated compact blocks;
- vector-M4 handling for full four-row compact blocks;
- concurrency capture sizes 1, 2, 4, 8, and 16;
- prefill `mblock=16` kernels;
- `mc4` and fragment-major `mc4afrag` prefill variants where exact assets are admitted;
- pinned vLLM, FlashInfer, DeepGEMM, AOT cubins, and warm caches in one image;
- `/model` as the single mounted model root.

The historical container proved these mechanics could coexist, but its delivery depended on image-internal runtime modules and specialized setup. The modern target preserves the behavior through reviewed package source, an ordinary plugin wheel, declared image dependencies, and a self-contained pack.

### Decode/concurrency dispatch

The retained acceptance contract is:

- compact decode block: `mblock=4`;
- scalar path: `valid_m < 4`;
- vector-M4 path: `valid_m == 4`;
- C8/C16 rely on routing compaction into full expert-local M4 blocks rather than a single global M8/M16 quant kernel;
- CUDA graph capture sizes: 1, 2, 4, 8, 16.

### Prefill dispatch

The retained acceptance contract is:

- prefill block: `mblock=16`;
- `mc4` amortizes plane reads across full tiles;
- `mc4afrag` uses fragment-major activations where its exact cubins are admitted;
- missing required accelerated assets fail closed for release rather than silently becoming an accepted slow path.

The current clean plugin ships the relevant cubins but does not yet fully expose this accepted scalar/vector and prefill dispatch policy through the modern native-plane implementation. That semantic port and its physical dispatch evidence remain release blockers.

## 7. Current API and closure gaps

| Surface | Implemented now | Required for easy handoff deployment |
|---|---|---|
| explicit dimensions | `smash backpack-dimensions` | future pipeline generates all ledgers and calls it automatically; not rerun for U012 deployment |
| exact selection | `smash knapsack` | future pipeline seals assignment automatically; current U012 uses accepted F521 |
| pre-repair export | `smash export` | future pipeline owns this stage; current U012 does not repeat it |
| repair export | all seven repair arguments are fail-closed | deployment API resolves them from the verified U012/equivalent handoff; no manual path discovery |
| pack verification | `smash verify` | included automatically after each export and before publication |
| serve compatibility | `smash serve-check` | release operation resolves packaged kernel ABI automatically |
| plugin | ordinary `vllm.general_plugins` entry point | GPU-proven full mixed-QTIP dispatch and performance receipts |
| receipts | hashes and statuses exist | remove private absolute paths; use handoff-relative paths/content IDs |
| performance | static tests and retained historical evidence | one command produces coherent API plus C1–C16 and prefill tables |

## 8. Fail-closed rules

The deployment path must stop if any of these occur:

- basis, assignment, overlay, checkpoint, or update mismatch;
- missing candidate, tier, layer, projection, or explicit dimension;
- floating-point byte assignment or inferred physical bytes;
- missing selected payload or source sidecar;
- repair codebook not consumed exactly once;
- source artifact changes during export;
- declared input escapes the handoff root or fails its content identity;
- extra, missing, symlinked, resized, or hash-drifted pack file;
- pack/kernel architecture or ABI mismatch;
- stock-vLLM plugin not discovered;
- dense weight map does not resolve completely;
- mixed plane family or fast kernel unavailable;
- API response empty or incoherent;
- required performance cell missing or outside the predeclared tolerance.

## 9. Evidence boundary

This public report intentionally omits operational host paths, hostnames, IP addresses, and raw receipts. Exact private evidence remains outside the repository. Public release evidence must be reconstructed as scrubbed, revision-bound summaries with the artifact identities above.
