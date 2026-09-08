# Opt-in parallel capture integrity verification

The public `solver_qtip_profile.main` accepts `capture_hash_workers` set to 1
(default), 2, or 4. Only host-file MD5 computations run concurrently; capture
loading, routing, source weights, seeds, numerical solve, packing, and the
capture-bank release boundary remain unchanged. No digest is skipped or reused.
At most four 8 MiB hashing buffers are live. The executor joins before tensor
loading or returning an error.

Use qualified public singleton calls in one process. `main_many(batch_size=1)`
needs a genuine aggregate root manifest; per-cell manifests alone do not admit
that API. Do not manufacture a root manifest merely to bypass this contract.
No cross-unit numerical batching was qualified by this experiment.

Measured on a dedicated accelerator with authentic production inputs for
L004/E242_down and E243_down, fixed L16/K1/V2, 8 warps/full branch unroll:

- Capture-only ABBA repeated twice: 0.277859 -> 0.076720 s (3.621735x), identical
  16-window materialized tensor digest for every arm.
- Same-work resident pair ABBA, fresh processes, shared compiler-cache namespace:
  whole process mean 16.069004 -> 15.834733 s (1.014795x).
- Staging sum for that pair: 1.679742 -> 1.367837 s (1.228028x).
- Two separate processes on the same pair: 22.283152 s vs 16.069004 s resident
  baseline (1.386716x). This scheduling comparison has only one separate pair;
  unlike the hashing comparison, it is not a repeated ABBA result.
- All ten output comparisons exactly match the original owner decoded artifacts;
  authentic clean-fit output SSE ratios are 1.0, within predeclared 1.0001 bound.

The whole-work hashing gain is small and should not be generalized into a
504-cells/hour production claim. The production owner's existing resident loop
already amortizes most cold staging; warm LDLQ remains the main bottleneck.
No held-out/frozen evaluation or full-model qualification was performed.

Verification: 10 new CPU tests pass, including real bounded concurrency,
materialized tensor equality, capture-bank release, corrupt-byte refusal,
missing-receipt refusal, and invalid worker counts. 102 focused public API,
resume, memory-lifetime and timing tests pass. Full CPU suite reached 1110 passed,
29 skipped, five failed and one deselected. The deselected failure was separately
exercised; all six failures reproduce on untouched baseline be54b7d. They concern
existing layer-condition checks, capability/config documentation, resident
materialization, and an isolated AST fixture missing `_require_torch`.

Production adoption is separate from these dedicated-host measurements and must
be recorded by the production owner at a clean missing-cell boundary.
