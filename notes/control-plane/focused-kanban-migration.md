# Focused Kanban control plane

Status date: 2026-08-03

Banana Smasher development now uses a focused Kanban control plane rather than the historical mixed campaign backlog. The focused board tracks only the current product path:

1. Four exact Anchor64 candidate and measurement lanes.
2. Numerical aggregation into the exact-budget Backpack solve.
3. Pre-repair measurement, persistent Update12 repair, and post-repair measurement.
4. Exact U004/V4 and U012/V5 serving through stock vLLM plus the pip plugin.
5. Full mixed-plane acceleration and canonical API/runtime integration.

## Repository boundary

Reusable implementation, tests, package metadata, runtime/plugin work, and result notes belong in this repository. Fleet allocation, machine claims, private mission paths, and private receipts remain outside the public tree. Historical campaign source is evidence only and is not a destination for new implementation.

The current integration surface is collected in [PR #4](https://github.com/my-other-github-account/banana-smasher/pull/4), based on head `c6997025912ec33838826d91f0e312914ce11e53` at this migration boundary. The focused control plane directs reusable fixes and interfaces back through isolated worktrees and the public package boundary.

## Transition rules

- One reasoning driver owns board mutation on a 15-minute cadence.
- Legacy schedulers are stopped and the historical board is dispatch-frozen.
- Already-productive work is adopted at an ownership boundary; it is not restarted merely to obtain a new task identity.
- A board state, claim, heartbeat, or supervisor is not reported as physical progress without a changing result-linked counter.
- The proof-of-concept speed law applies: one narrow smoke check, then the real run and measurement.
- V4 names only the exact F521/U004 mixed QTIP2 + QTIP3 + D4 + MXFP4 Backpack. V5 is U012 on that same basis.

## Cutover verification

A controlled first driver tick completed successfully and delivered its report on the configured cadence destination. It adopted current work into the focused roots, preserved productive payloads, and left dependency-gated work parked rather than creating duplicate execution. Scheduler readback showed one enabled driver and no running legacy manager process.

This note describes the control-plane boundary, not a scientific or performance result. Hardware, quality, and throughput claims still require their own revision-bound receipts under `notes/`.
