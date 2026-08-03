# V4/V5 live reproduction status

**Date:** 2026-08-03
**Status:** IN PROGRESS
**Canonical static baseline:** `6614d1199ca87b70f74d67f9a636793c7ff6807f`

This note separates static repository readiness, model-pack verification, API coherence, and performance. A PASS in one column does not imply another.

## Product identities

| Product point | Assignment | Repair update | Checkpoint SHA-256 |
|---|---|---:|---|
| V4 | F521 `f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39` | 4 | `082d6d8cf64e83f478c637428779487a3248d0559a1b4a27abfb7f681baa5d41` |
| V5 | F521 `f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39` | 12 | `429b644b22920012e8e6be1da07617b4b2ff68aad4a2b2baca5602882cb94fdf` |

## Gate summary

| Gate | V4 U004 | V5 U012 |
|---|---|---|
| exact assignment/checkpoint authority located | PASS | PASS |
| public-safe source canonicalized | IN PROGRESS | IN PROGRESS |
| selected mixed-QTIP payload source verified | PASS as retained authority; clean re-export pending | PASS at current pack-materialization gate |
| final repaired pack from deployment API | PENDING | PENDING export from existing verified U012 handoff |
| both wheels built from final revision | PENDING | PENDING |
| pinned stock-vLLM image built | PENDING | PENDING |
| model boot | PENDING | PENDING |
| coherent non-stream completion | PENDING | PENDING |
| coherent streaming completion | PENDING | PENDING |
| restart/no-mutation | PENDING | PENDING |
| fast-path/dispatch evidence | PENDING | PENDING |
| C1–C16 | PENDING | PENDING |
| prefill | PENDING | PENDING |
| release decision | BLOCKED | BLOCKED |

## Immediate priority order

1. Use the existing verified U012/F521 artifact result; do not rerun Backpack construction, pre-repair evaluation, or repair.
2. Build/install the current plugin and pinned image, then make that exact U012 artifact boot and answer coherently.
3. Restore the accepted mixed-QTIP fast paths and measure decode, C1–C16 concurrency, and prefill immediately.
4. Only after performant serving is proven, consolidate the handoff/export API and clean-box user experience.

The API constraint is not a requirement to recompute U012. It means the working deployment must not depend on a hidden patch or side artifact that would be absent when a future full pipeline emits an equivalent handoff.

## V4 findings

- The exact U004 checkpoint identity was recovered from an immutable replay copy and independently matched by another retained copy.
- A mutable path previously associated with U004 no longer matched the accepted checkpoint hash and was rejected.
- The preserved container/runtime source binds the selected mixed-QTIP pack before repair; it does not provide a correct direct runtime hook for `UPDATE_004.pt`.
- Correct reconstruction therefore requires the canonical exporter to materialize U004-repaired codebooks and dense repair state into a new task-owned pack.
- No source pack should be edited, and no full duplicate is required when immutable files can be hardlinked on the same filesystem.
- A candidate source change for breakable piecewise CUDA graphs exists on the isolated V4 branch, but it is not accepted until tests, image build, boot, coherence, and performance complete.

V4 has not yet been reproduced live in the clean stock-vLLM plugin path. Historical performance remains reference evidence, not a current acceptance claim.

## V5 pack-materialization findings

Current verified identities:

| Surface | Value |
|---|---|
| outer manifest SHA-256 | `2d47c50a43450fac17727987beebee84faf90140815cdc54adbb3b0bc4104098` |
| repair manifest SHA-256 | `fb01855311d3cbae507e01e40e890512fcde8af9dfa347fbe54b5df7cedbbcba` |
| base index SHA-256 | `58c9d59dfe8fd1e7e833be131043f4b45bfa27064fc19b9fa4fffa6475f2d0fc` |
| assignment SHA-256 | `f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39` |
| checkpoint SHA-256 | `429b644b22920012e8e6be1da07617b4b2ff68aad4a2b2baca5602882cb94fdf` |
| declared files | 2,063 |
| declared bytes | 110,364,099,133 |
| projections | 22,016 |

Projection coverage:

- QTIP2: 2,266
- QTIP3: 14,979
- D4 k1024: 197
- D4 k2048: 2,610
- D4 k4096: 1,910
- native MXFP4: 54

The source pack's config lacked one manifest-bound final newline. A task-owned serving tree was materialized with the exact expected config bytes while leaving the source pack untouched. Expert projections are intentionally represented by the native-plane manifests rather than dense expert tensors in the model index.

This is a pack gate only. No V5 API or throughput acceptance has been made.

## Current canonical implementation state

Implemented:

- fail-closed pack export and verification;
- complete serving metadata/base-shard materialization;
- repair checkpoint validation and codebook/dense repair materialization;
- explicit Backpack dimensions;
- exact-byte knapsack inputs/solve;
- stock-vLLM general-plugin registration;
- native mixed-plane loader and routed-expert method;
- dense weight-map preflight and dense repair application;
- pinned image/dependency/asset admission;
- static package, plugin, Docker, and extraction tests.

Still required, in priority order:

1. export or bind the existing verified U012 handoff without rerunning upstream construction or repair;
2. full Linux ARM64 SM120/SM121 image build and first boot;
3. coherent API and restart evidence;
4. modern plugin port of the accepted scalar/vector-M4 decode and `mc4`/`mc4afrag` prefill behavior;
5. authoritative mixed-QTIP C2/C4/C8/C16 reference recovery;
6. live C1–C16 and prefill measurements on the exact final V5 artifact;
7. automatic resolution of handoff assignment, overlay, repair checkpoint, update, and selected source;
8. content-addressed portable receipts instead of private absolute input paths;
9. a future full-pipeline producer that emits the same handoff contract;
10. clean-box reproduction with no undeclared artifact or cache.

## Release blocker

Neither V4 nor V5 is releasable while deployment depends on an undeclared authority outside its verified handoff or while any required API/performance cell remains pending. The existing verified U012 result is itself an allowed handoff and should be used now.

The binding workflow and measurement rules are in:

- `notes/decisions/2026-08-03-mixed-qtip-standalone-deployment.md`
- `notes/reports/2026-08-03-mixed-qtip-backpack-system-map.md`
- `notes/tables/2026-08-03-runtime-performance-gates.md`
