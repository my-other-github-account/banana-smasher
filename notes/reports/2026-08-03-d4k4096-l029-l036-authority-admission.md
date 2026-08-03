# D4K4096 L029-L036 authority admission

Status: fail closed. Allocation remains forbidden.

The exact-basis L029-L036 partition contains 4,096 candidate cells. Its previously sealed packed-wire-byte sidecar remains immutable at SHA-256 `b44d179dc6d81f915c2b26aa1f8ceb55d83785dc4bf9472c6facc05adc91dfda`, with 4,096 rows and 20,937,965,568 packed physical bytes. This pass did not replay, relabel, or rewrite those rows.

A complete public admission cannot be emitted because authoritative regular-file sources are unavailable for the required dimension graph:

- packed physical bytes: prior sealed authority exists, but no current regular-file path was opened or read back during this pass;
- projection corrections: 4,096 per-candidate values absent;
- six-class predictions: 4,096 per-candidate vectors absent; aggregate class evidence is not substituted;
- six-class ceilings: 4,096 bindings absent because no accepted basis-bound class-cap policy exists.

The machine-readable blocker is `notes/manifests/2026-08-03-d4k4096-l029-l036-missing-authority.json`. It binds the exact basis, layer range, partition hash, expected row counts, sealed predecessor hashes, and explicit null path/SHA fields for every unavailable authority. It also records the no-inference, no-synthetic-class, no-fixed-quota, no-ring-policy, and no-QTIP-substitution policy.

The public `smash backpack-dimensions` admission now requires absolute regular-file `{path, sha256}` references for packed physical bytes, projection corrections, six-class predictions, and six-class ceilings. Files are opened without following a leaf symlink and hashed by streaming reads before any allocation-eligible output is published. The ceilings authority must be the exact class-ceilings input. Duplicate JSON keys and non-standard JSON constants are rejected.

The public authority contract is `banana-smasher/schema/bs-dynamic-backpack-authority-v1.schema.json`.
