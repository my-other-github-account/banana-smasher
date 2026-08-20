# Production resident rail

`banana_smasher.ResidentRepairAPI` is the only public scoring/training facade. The concrete provider is `banana_smasher.production_rails.ProductionRails`; the CLI keeps the complete pre-score → continuation → post-score arm in one process:

```bash
smash resident arm \
  --artifact-root /local/repair-artifact \
  --rails-config /local/production-rails.json \
  --run-root /local/run \
  --updates 4
```

Do not split these phases across separate CLI processes. `RESIDENT_LIFECYCLE.json` must finish with exactly one `model_constructions` and one `resident_loads`, two scores/canary passes, one training call, four updates, and in-memory checkpoint swaps between phases.

## Provider config contract

The config schema is `banana-smasher-production-resident-rails-v1`. It must declare:

- `pipeline_microbatch: 4` (other geometry fails closed);
- `layers: [0, ..., 42]` in exact order; no per-layer exceptions;
- `uniform_builder` and `backpack_mixer` as `module:callable` package hooks;
- no production `session_factory` hook (the config rejects it); production constructs one `ModernGreenResidentEngine`/`ShardStudent` for the entire arm;
- `allowed_artifacts`, keyed by the exact SHA-256 of `identity.json`, with `basis_sha256`, `checkpoint`, `artifact_manifest_sha256`, and `checkpoint_sha256`; both the manifest and selected checkpoint bytes are verified before model construction;
- the official continuation configuration when using the proven two-Spark engine.

The artifact's `identity.json` must contain `runtime.production_rails.provider_binding_sha256`. This is the canonical SHA-256 of the provider ABI fields (schema, microbatch, ordered layers, and builder/mixer hook references). The full config SHA and provider-binding SHA are both recorded in the lifecycle receipt. Artifact identity must also cover all 43 layers in generic order and be admitted by `allowed_artifacts`; unknown identity, provider drift, basis drift, manifest/checkpoint-byte drift, malformed composition, or canary mismatch is rejected before score publication.

The default session scores directly through the live two-rank ShardStudent. Training calls `advance_to` on that same object, persists recovery bytes, and advances the in-memory binding to the returned checkpoint without loading it back. Post-score therefore measures the just-trained resident parameters; it never selects pre-existing candidate rows or reconstructs a model.

Each rank writes `RESIDENT_LIFECYCLE.rankN.json` and rank-qualified continuation receipts. When both rank lifecycles are present, the provider atomically publishes the paired `RESIDENT_LIFECYCLE.json`; rank processes never race on one rank-specific receipt.

## Ported production implementation

The formerly mission-local resident implementation now lives in the package:

- `resident_balanced64.py`: fixed 64-window/1,024-position/8,192-support `KL(teacher || candidate)` scorer and resident in-memory reduction;
- `resident_proven_api.py`: scientific identity validation, candidate builder glue, continuation checkpoint persistence, resume comparison, and two-Spark continuation orchestration;
- `resident_continuation.py`: official grouped-K2 resident continuation with sealed `PIPELINE_MICROBATCH=4` geometry;
- `resident_core.py`: shared basis, claim, memory, and checkpoint preflight;
- `production_rails.py`: the concrete `PipelineRails` provider, exact artifact admission, one-construction lifecycle, in-memory swaps, mandatory canary, and publication instrumentation.

## Acceptance

Before publishing a result:

1. Verify the exact config and artifact identities.
2. Run `smash resident arm` in one process.
3. Require `RESIDENT_LIFECYCLE.json` to prove one construction and all phase events.
4. Require both scores to finish within the facade's 300-second phase budget.
5. Require each score receipt to report `execution_mode: resident_model_in_memory`, 64 windows, zero checkpoint loads during score, and no candidate-file reads during score.
6. Compare the full64 score to the sealed fixture oracle; do not substitute precomputed or synthetic benchmark output.
