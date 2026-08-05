# Contributor Rules

Banana Smasher has one canonical source tree. Keep reusable code, tests, build metadata, and release tooling in this repository.

## Binding PoC development priority

Banana Smasher is a proof-of-concept project. Optimize for the fastest direct path to working code and a real measured result. This section supersedes conflicting card bodies, plans, reviews, and generic agent habits.

- Make the smallest plausible code change, run at most one focused test or smoke for the changed path, then execute the real PoC.
- Add more tests only after an observed failure requires them or the user explicitly asks.
- Do not add speculative defensive coding, exhaustive edge cases, broad matrices, review fan-outs, artifact-swap/tamper/TOCTOU defenses, generalized provenance/receipt/hash/CAS machinery, offsec/security work, clean-room/restart guarantees, privacy/license scans, or publication hardening unless explicitly requested.
- Treat ordinary local artifacts as trusted. Check identity only enough to avoid using the wrong immediate input.
- Preserve useful product behavior during integration, not every old test, defensive invariant, schema, receipt, or agent-authored process layer.
- Stop when the requested path works and its requested result is measured. Do not invent additional audits, gates, reports, or successor cards.
- Basic credential hygiene and physical-host non-preemption remain binding because they prevent destructive interference; they do not authorize unrelated defensive engineering.

## Repository boundaries

- `banana-smasher/` owns the portable artifact and command API.
- `banana-smasher-plugin/` owns stock-vLLM runtime integration.
- `notes/` owns curated human-readable reports and compact, scrubbed evidence summaries.
- Keep runtime integration plugin-based. Do not fork or vendor vLLM.

Do not copy deployment controllers, host-claim logic, task orchestration, machine-specific wrappers, or private source trees into this repository. Do not publish credentials, usernames, hostnames, private network addresses, absolute private paths, task identifiers, or raw private receipts.

## Public command surface

Treat these commands as the primary user API:

- `smash solve`
- `smash update`
- `smash evaluate`
- `smash bank`
- `smash knapsack`
- `smash backpack-dimensions`
- `smash qtip-configs`
- `smash kernels build`
- `smash export`
- `smash verify`

Reference and debug implementations must require explicit opt-in and must never be silent fallbacks.

## Evidence and release gates

Keep claims honest: do not present an unrun hardware result as measured. During ordinary development, one focused smoke or test is enough. Broader suites, package matrices, clean-clone checks, scans, and hardware gates run only when the user explicitly requests publication/release work or when they are directly necessary for the claim being made.
