# PoC development priority (binding)

This is a proof-of-concept project. Optimize for the fastest path to working code and a real measured result.

- Make the smallest plausible code change, run the narrowest relevant check, then execute the real PoC.
- Tests default to one focused test or smoke check for the changed path. Add more only after an observed failure requires them or the user explicitly asks.
- Do not spend time on speculative defensive coding, exhaustive edge cases, broad regression matrices, generalized artifact-swap protection, provenance machinery, clean-room/restart guarantees, offsec/security work, privacy scans, publication hardening, or production robustness unless explicitly requested.
- Artifact/input checks should be only the minimum needed to avoid running the wrong immediate input. Do not build generalized anti-swap infrastructure.
- Prefer direct implementation and measurement over audits, inventories, plans, and blocker catalogues.
- Stop when the requested PoC works and the requested metric is measured. Do not silently expand the mission into productionization.
- This section supersedes conflicting lower-priority project/card instructions that demand exhaustive testing or hardening by default.

# ASAP anti-deadlock execution policy (binding)

- Goal-mode ownership is nonterminal until the positive accepted result. The owner chooses task-local implementation, diagnosis, repair, equivalent tools/artifacts that preserve scientific identity, retries, and resume.
- Missing/stale packages, paths, manifests, inputs, mounts, claims, dependencies, environment drift, preflight failures, tool errors, first-launch failures, dead-owner residue, and failed approaches are next-action evidence—not accepted blockers. They may not produce `done`, `BLOCKED`, passive monitoring, or terminal `0/N`.
- Any lower card/comment clause saying `fail closed`, `fail loud`, `stop BLOCKED`, `no mutation`, `do not repair/recover/substitute/retry/resume`, or `complete with blocker` is superseded when it applies to recoverable mechanics. Make the smallest causal task-local fix or change approach and continue.
- Preserve only genuine hard boundaries: immutable scientific identity, HOLDOUT closure, destructive action requiring explicit approval, and live foreign compute/ownership. If one truly prevents progress, keep the positive goal nonterminal and have the external driver reclaim/reassign/succeed it immediately. A bounded experiment may seal scientific RED, but that does not terminalize its parent positive goal.
- Progress credit is an accepted receipt/checkpoint/result delta. Heartbeats, a live PID, bytes read, staging, scans, process launch, and promises are not accepted progress.
- Report accepted `n/N` first. Across two successful observations with no accepted delta: preserve a payload only when CPU/GPU/I/O/checkpoint evidence proves real motion; otherwise perform one exact intervention or externally reclaim/block the card. Never keep a goal-mode worker running solely to sleep, wait, poll, or heartbeat.
- Review-mode/tool denial, exhausted iterations, or provider quota denial requires immediate external reclaim. Do not leave a capability-wedged worker heartbeating.
- Control ticks get one bounded census, one mutation pass, and one verification, with a three-minute target. Never sleep/wait for physical completion inside a control tick.
- No archive-before-launch, Macmini external-SSD/NAS live path, broad host/artifact search, speculative security/provenance review, or generalized hardening. Use the exact current handoff locator, make the smallest causal fix, run one focused smoke, and execute the real PoC.

# Fleet communication rule

For any request that handles multiple Spark hosts Spark-by-Spark:

- Send the user at least one human-visible progress message for **each Spark** while the work is being performed.
- Each per-Spark message must state what was checked or changed and the result/evidence observed.
- Do not perform a silent all-host sweep and substitute one final fleet summary for the required per-Spark messages.
- A final consolidated report is welcome, but it does not replace the per-Spark updates.

# General progress communication rule

For any long-running task or task requiring many tool calls:

- Send routine human-visible progress updates while work is underway.
- Do not remain silent through a long sequence of tool calls.
- Report concrete completed actions, observed evidence, blockers, and the next active step rather than generic “still working” messages.

## OOM law (David, 2026-08-19, binding)
Suspected hardware issues have ALWAYS resolved to OOMs — never suggest hardware as root cause. Dark host under load = memory exhaustion (possibly payload collision). Recover, census co-residents, budget peak vs HOST MemAvailable, use systemd-run MemoryMax scopes. Use the machine normally afterward.

## Single-source-of-truth code law (David, 2026-08-20, binding)
All campaign code runs from ONE canonical git repo (t_e9156db5 converging banana_smasher+repair_api; interim canonical: /Users/macmini/clawd/ds4-flash-kldmatrix/repair_api@main). Deploy by git pin ONLY (SHA in every launch receipt); never hand-copy or edit deployed .py files. Fixes = commit to canonical (with test) -> pull pinned SHA -> relaunch. A fix that isn't a canonical commit does not exist.
