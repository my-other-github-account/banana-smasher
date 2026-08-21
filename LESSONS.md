# Campaign seam catalog

- **Truncation:** bind the scored row count and support before reduction. A
  shorter candidate, padded page, or partial window is not an equivalent score.
- **Clamps:** numerical safety clamps change the objective. Keep the accepted
  clamp-free expert path and fail on non-finite inputs instead of silently
  changing the measured function.
- **Cache semantics:** caches are scoped to exact checkpoint, corpus, window,
  dtype, and code revision identities. Reuse only a complete matching entry;
  never treat a filename or warm process as proof of identity.
- **Microbatch geometry:** pipeline and score microbatches are part of the
  runtime contract. Preserve full-row shapes and derive packing from the
  manifest; do not encode layer- or fixture-specific geometry.
- **Checkpoint identity split:** published PRE is `f9bffe04…` (KLD `0.229392`,
  Top-1 `56,533/65,536`); raw U0 `7978d100…` is a distinct state near `0.2356`.
  See the [authoritative Evals table](Evals/README.md#results).
- **Page cache and Shmem:** process RSS is not the machine budget. Preflight
  `MemAvailable`, account for page cache and shared memory, and drop only this
  workload's proven disposable cache before retrying an allocation.
- **MEMLOCK:** pinned-memory and CUDA registration can fail despite free RAM.
  Check the effective memlock limit and requested pinned bytes before launch;
  surface the exact refusal rather than switching to an unmeasured path.