# Driver workstreams and delivery gates

**Status date:** 2026-08-03 09:45 PDT
**Canonical baseline:** `origin/main` at `21cd7964e9eb6ead7fb354af00ca9c44a909636a`
**Purpose:** concise priority source for fleet arbitration. Operational status is not a hardware or release claim.

## Priority and status

### P0 — 0731 production pipeline

`anchors ✅ → authorities 🟠 → pre-repair pack 🔴 → pre-repair KLD 🔴 → Update12 repair 🔴 → post-repair KLD 🔴 → production deployment 🟠`

- **Anchors — DONE.** QTIP2, QTIP3, and D4K2048 are sealed. Existing valid D4K4096 rows are immutable. Never replay sealed work.
- **Predictions and authorities — IN PROGRESS.** Finish authentic six-class predictions and ceilings, projection corrections, routing/expert importance, and exact packed-wire-byte admission. Missing inputs fail closed; they are never inferred from ring geometry or substituted from another tier.
- **Exact pre-repair Backpack pack — BLOCKED on authorities.** Deterministically choose a tier per layer/expert/projection under exactly `101,346,700,411` physical bytes, then export through the public CLI.
- **Pre-repair quality — BLOCKED on the pack.** Run genuine own-teacher BALANCED64 KLD and top-1 before repair. Record GB, packed bpw, basis, teacher, and instrument identity.
- **Update12 repair — BLOCKED on pre-repair quality and integrated update code.** Require real-token geometry, memory admission, relocation-safe checkpoints, persistent resume, and no warm start.
- **Post-repair quality — BLOCKED downstream.** Repeat the same KLD/top-1 instrument and basis so the repair delta is attributable.
- **Production deployment — ENABLER IN PROGRESS.** Ship through the public stock-vLLM pip-plugin path and the from-source container gate below.

### P0 — serving and product APIs

- **Public V5 source image — IN PROGRESS, NOT ACCEPTED.** Replace the disqualified frozen/private Docker recipe. Build repo wheels in-image from official `vllm/vllm-openai:v0.24.0`; zero runtime environment glue. Acceptance is clean clone → no-cache build → zero-env model mount → health/models/three prompts → warmup, ladder, and prefill, with digest, SBOM, and source receipt.
- **V4/V5 correctness and performance — IN PROGRESS.** Preserve stock-plugin behavior and every accepted acceleration. Diagnostic kernel GREEN does not substitute for clean-room server boot or C1/C2/C4/C8/C16 and prefill receipts.
- **Dynamic Backpack API — IN PROGRESS.** Deterministic candidate enumeration, authority schemas, exact physical-byte knapsack, fail-loud admission, public export, and lightweight import without Torch/vLLM/CUDA.
- **Update API — READY FOR CENTRAL INTEGRATION.** Admit the reviewed portable update/checkpoint/token-sizing handoff, then verify source/wheel behavior and hardware-only gates.
- **Bank/evaluate API — IN PROGRESS.** Canonical schemas, provenance, resumability, symlink-safe writes, and same-instrument KLD/top-1. Bounded top-1 tooling remains under review; sealed evaluation rows are not replayed.
- **Central fan-in and PR — CRITICAL BOTTLENECK.** Consume only clean committed handoffs in an isolated integration worktree. Run full static tests, Ruff, both wheel/ZIP checks, privacy/credential and license-surface scans, fresh-clone validation, then publish one reviewable topic branch/PR. Do not merge automatically.

### P1 — acceleration, future serving, and comparisons

- **QTIP matrix lifetime/runtime — HARDWARE CANARY LAUNCHING.** Clean source/wheel gates first; smallest SM121 packed-wire/dispatch/lifetime canary next. No slow fallback.
- **Native plane/CUDA graphs — DIAGNOSTIC GREEN, INTEGRATION PENDING.** Port the minimal proven closure with tests and acceleration sentinels; final acceptance is the public source-image server gate.
- **Activation-cache persistence — HANDOFF READY, FAN-IN PENDING.** Preserve exact-equal serial behavior; exclude campaign orchestration.
- **DeepGEMM SM120/SM121 — IN PROGRESS.** Keep the required upstream revision and fail closed if unavailable; full server acceptance remains pending.
- **V6 MTP-on serving — BLOCKED on accepted V5 and repaired 0731.** Do not steal the V5 floor early.
- **Competitor IQ2/IQ3/IQ4 — PARTIAL.** Only distinct own-base BALANCED64 KLD + top-1 rows count. Historical-candidate replay does not.
- **Release evidence — PENDING FINAL FAN-IN.** Re-run clean image, SBOM, provenance, privacy, license-surface, wheel, ZIP, and hardware gates against the exact release candidate.

## Driver selection rule

1. Newest direct operator instruction wins.
2. Preserve any goal-linked process with a real advancing physical counter.
3. Choose the highest-priority **runnable missing gate** above, not merely the largest board number.
4. If a gate is blocked, assign a non-overlapping prerequisite, test, integration, or acceleration lane; do not manufacture utilization by replaying sealed artifacts.
5. One owner per host, worktree, branch, and file family. Coordinate foreground agents through Kanban comments and exact committed SHA handoffs.
6. Product code, tests, reusable scripts, docs, and curated reports live only in isolated worktrees of this canonical repository. Historical campaign trees are read-only provenance.
7. Missing accelerated dependencies, manifests, authority files, or identity inputs fail closed. Never add silent fallback.
8. The board carries live ownership and dependencies; this note carries outcome order. Update both whenever operator priorities change.
