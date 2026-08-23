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
## The Five Root-Cause Laws (2026-08-23 post-mortem of the scoring/training campaign)
~80% of wasted GPU-hours traced to five repeating behaviors. Binding on all future work:
1. **Instrument before experiment** — prove the measurement reads a known value on a known input before any run (loss on PRE ≈ 0.137-class; scorer on U0 ≈ 0.2292). We trained 48 updates on a loss reading 100× wrong.
2. **Reference before implementation** — never rewrite what proven code does (sealed builder, tailfix_repair_e2e, kld_score); cite file:line of what you reuse. Reimplementations cost 5 days total.
3. **Mechanism before fix** — no change without the cause named from source/receipt/A-B differential. Try-and-see made things worse every time.
4. **Gates cite sealed truth only** — no circular self-references, no synthetic extrapolations, no train-set leakage (we validated on a training window for a day).
5. **One-variable experiments** — declare variable, control, kill-gate before launch. Structural variables (data, trainable surface, objective) before scalar knobs (LR).
