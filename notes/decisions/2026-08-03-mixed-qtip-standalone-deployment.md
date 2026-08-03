# Decision: mixed-QTIP Backpack and standalone deployment closure

**Date:** 2026-08-03
**Status:** BINDING
**Scope:** product, export, serving, release, and reporting

## Decision

Banana Smasher has one product path: a class-aware mixed-QTIP Backpack composed of QTIP2, QTIP3, D4 subtiers, and native MXFP4, exported as a verified `bs-pack` and served through pinned stock vLLM plus the installable `banana-smasher-plugin`.

V4 is the containerized U004 Backpack on the F521 assignment. V5 is U012 on the same assignment and pipeline basis.

No unrelated quantization format is a compatibility target, deployment input, preservation target, benchmark substitute, or reporting topic for this repository.

## Immediate V5 priority

Use the existing hash-verified U012/F521 result. Do not rerun candidate generation, Backpack selection, pre-repair evaluation, or U012 repair as a condition of deployment.

The immediate order is:

1. make existing U012 boot through the stock-vLLM plugin;
2. obtain coherent streaming and non-streaming responses;
3. restore and prove decode, C1–C16 concurrency, and prefill performance;
4. consolidate the export/handoff API and clean-box presentation after the performance proof.

## Deployment handoff law

The existing U012 result is an allowed, authoritative deployment handoff. A future fresh pipeline must emit the same handoff contract; the deployment path must not care whether that contract came from the already-completed U012 run or a future end-to-end run.

The final serving pack may consume:

1. the existing verified U012 assignment, overlay, selected planes, repair checkpoint/state, base-model files, and manifests;
2. the source revision of this repository and its built exporter/plugin wheels;
3. declared packaged runtime assets and measured runtime settings;
4. for a future fresh run, the equivalent hash-bound handoff emitted by that run.

The final serving pack must not consume:

- an assignment, overlay, checkpoint, plane, or model file outside the declared U012/future-pipeline handoff;
- a previously patched service image;
- out-of-tree Python overlays or host-side monkey patches;
- a manually edited `config.json`, tokenizer, manifest, plane, or repair file;
- an undeclared cache whose contents affect model bytes or dispatch behavior;
- private receipts or cluster paths as runtime inputs.

An optional machine cache is permitted only when it is content-addressed, validated, reproducible from declared inputs, explicitly non-authoritative, and safe to delete before a clean-box reproduction.

## Required handoff lineage

For current V5 work, deployment begins at the completed U012 handoff:

```text
existing hash-verified U012/F521 handoff
  -> final serving-pack binding/export
  -> pack and kernel compatibility verification
  -> stock-vLLM image + plugin wheel
  -> coherent API evidence
  -> decode, concurrency, and prefill evidence
```

For future models, the full pipeline must produce the same deployment input:

```text
base model + pipeline config + data authorities
  -> candidate generation
  -> explicit dimensions
  -> exact-byte Backpack selection
  -> selected payload materialization
  -> pre-repair evaluation
  -> repair
  -> hash-verified deployment handoff
  -> the same export, plugin, serve, and performance path used by U012 now
```

Every deployment arrow requires producer revision, command/API arguments, input SHA-256 identities, output SHA-256 identities, status, and a fail-loud remedy when an input is absent.

The final exporter may reconstruct or bind the deployable pack from the existing verified U012 handoff. It must not require an extra artifact or manual mutation outside that handoff. The same exporter must later accept an equivalent handoff from a future fresh pipeline.

## API contract

### Existing public surfaces

The current repository exposes:

- `smash backpack-dimensions`
- `smash knapsack`
- `smash export`
- `smash verify`
- `smash serve-check`
- `smash validate`
- the `vllm.general_plugins` entry point in `banana-smasher-plugin`
- `examples/build_image.sh`
- `examples/serve.sh`
- `examples/smoke_api.py`

These are product primitives, not yet the complete handoff-to-serving workflow.

### Required closure

The public API must gain one deployment-handoff operation that:

1. reads one versioned handoff manifest, including the existing U012 handoff now;
2. resolves only handoff-relative or content-addressed producer outputs;
3. verifies every input hash before use;
4. proves all Backpack cells and dimensions are complete;
5. exports or binds the final pack without modifying source artifacts;
6. binds the repair checkpoint, assignment, active overlay, selected source, and update automatically;
7. emits a portable receipt containing no private absolute path;
8. returns the exact next command and remedy on failure;
9. can be resumed idempotently after interruption;
10. produces an artifact that a clean stock-vLLM container can serve using only the plugin wheel and `/model`.

The immediate target is one U012 handoff invocation, one verification invocation, and one standard stock-vLLM serve invocation. Future end-to-end runs should terminate by emitting the same handoff. The implementation may keep separate internal stages, but users must not manually discover or patch authority files.

## Serving contract

The supported production boundary is:

- pinned `vllm/vllm-openai:v0.24.0` by digest;
- both wheels built from one canonical repository revision;
- `banana-smasher-plugin` discovered through `vllm.general_plugins`;
- one verified self-contained model directory mounted at `/model`;
- exact runtime defaults from `docker/runtime_defaults.json`;
- fail-closed accelerator and architecture admission;
- no required `PYTHONPATH`, source-tree mount, alternate vLLM fork, or host patch.

Runtime environment values baked into the image may select a packaged accelerator, but model identity and pack location must come from the pack/config contract rather than hidden environment state.

## Performance is part of correctness

A release is not accepted merely because it imports, boots, or returns text. The exact same final artifact and image must pass:

1. `/health` and `/v1/models`;
2. coherent non-streaming completion;
3. coherent streaming completion;
4. restart without artifact mutation;
5. required mixed-QTIP fast-path and kernel-dispatch receipts;
6. decode C1;
7. concurrency C2, C4, C8, and C16;
8. prefill at the frozen prompt lengths;
9. the full 3-by-5 warmup shape matrix before timing;
10. the tolerance and sample rules in `notes/tables/2026-08-03-runtime-performance-gates.md`.

Missing performance cells block release. Results from a different artifact, basis, image, instrument, or workload are not substitutes.

## Enforcement requirements

The repository must add or retain tests that prove:

- schemas and runtime code agree on `quant_method=banana_smasher`;
- repair export refuses a partial input set;
- every repair codebook is consumed exactly once;
- handoff paths cannot escape the handoff root;
- final export refuses undeclared authorities outside the handoff;
- pack publication is manifest-last and atomic;
- source artifacts are not modified;
- source and wheel command surfaces match;
- plugin installation is sufficient for stock-vLLM discovery;
- missing kernels or incompatible hardware fail closed;
- release reporting cannot mark performance PASS with missing C1–C16 or prefill cells.

## Publication boundary

Public notes may contain repository-relative paths, public dependency revisions, artifact hashes, schemas, counts, and scrubbed measurements. They must not contain private host paths, hostnames, IP addresses, credentials, or raw operational receipts.
