# Canonical Banana Smasher repository

## PoC development priority (binding)

Banana Smasher is currently a proof-of-concept project. The priority is rapid implementation and real measurement, not production hardening.

- Implement the shortest plausible path first.
- Run only the smallest targeted test or smoke check needed for the changed path, then run the real PoC.
- Do not add speculative defensive coding, exhaustive edge-case handling, broad test matrices, generalized artifact-swap/provenance machinery, offsec/security work, privacy scans, clean-room guarantees, restart guarantees, or publication hardening unless the user explicitly asks.
- Check artifact identity only as much as needed to avoid using the wrong immediate input.
- Expand testing only in response to an observed failure or an explicit request.
- Stop when the PoC works and the requested performance/result is measured; do not turn every change into productionization.
- This section supersedes any conflicting testing/hardening default below for the current PoC phase.

This repository is the sole source of truth for Banana Smasher code, tests, package metadata, runtime integration, Docker assets, public documentation, and public-safe reporting.

## Repository boundary

- Canonical remote: `git@github.com:my-other-github-account/banana-smasher.git`
- Canonical local checkout on the primary development Mac: `~/clawd/banana-smasher-runtime`
- The historical `spark-bench-reproducers/glm52-ds4-bq3-ptq-opd` tree and its refs are read-only evidence. Import reviewed behavior from it; never land new Banana Smasher work there.
- Do not copy built wheels, compiled kernels, caches, model artifacts, private receipts, internal host paths, hostnames, IP addresses, or credentials into this public repository.

## Change discipline

1. Work in this repository and keep each change narrowly scoped to the immediate PoC result.
2. Preserve the stock-vLLM pip-plugin boundary where practical, but do not build generalized fail-closed or fallback machinery beyond what the current run needs.
3. Port only the behavior needed now; avoid broad refactors and wholesale legacy imports.
4. Run a focused test or smoke check for the changed path, then exercise the real runtime. Full suites, wheel matrices, scans, and publication checks are release work and run only when explicitly requested.
5. Hardware claims still require one real hardware run; they do not require an exhaustive acceptance campaign unless requested.
6. Use commit identity `banana_bae <banana_bae@users.noreply.github.com>` for public history.

## Reporting

All reports, result tables, benchmark summaries, methodology receipts, migration audits, and decision records belong under `notes/`. Product API documentation may stay beside the package or component it documents, but evidence and status claims belong in `notes/` and must bind their source/artifact revisions.

## Publication posture

No license has been assigned to the original work. Do not add a `LICENSE`/`COPYING` file, package license metadata/classifier, SPDX header, or prose granting a license unless the owner explicitly chooses one. Upstream dependency license facts may be stated only with clearly limited scope.
