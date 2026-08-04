# Contributor Rules

Banana Smasher has one canonical source tree. Keep reusable code, tests, build metadata, and release tooling in this repository.

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

Every performance or quality claim must name same-work basis and receipt hashes. Mark hardware-dependent results as pending until the real gate runs. Before publication, run the full test suite, build both wheels, inspect wheel contents, scan tracked files for private identities and large binaries, and verify the documented CLI from a fresh clone.
