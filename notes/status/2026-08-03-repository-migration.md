# Canonical repository migration status

**Date:** 2026-08-03  
**Status:** IN PROGRESS

## Decision

This standalone repository is now the canonical source for new Banana Smasher code, documentation, reports, and tables. The legacy campaign repository is read-only input for targeted semantic porting.

## Already present

- Fail-closed pack export and verification.
- Stock-vLLM general plugin integration.
- Native mixed-tier runtime and repair application.
- Preserved SM120/SM121 acceleration assets with exact admission manifests.
- Public source-built Docker path and pinned runtime defaults.
- Static extraction, package, plugin, and Docker contract tests.

## Minimal modern API gaps being audited

| Surface | Current canonical state | Required action |
|---|---|---|
| Explicit candidate authority publication | Missing | Port the sealed fail-closed producer and tests |
| Dynamic Backpack dimension binding | Present | Integrate the explicit six-class/importance/byte binding into the standalone run-root API |
| Dynamic exact-byte knapsack | Present | Integrate the manifest-bound solver into the standalone run-root API |
| Source-vs-wheel command parity | Partial | Extend release tests to cover every newly admitted public verb |
| Reporting and result tables | Present | Keep revision-bound reports, status, decisions, and tables under `notes/` |

## Excluded by design

Training controllers, host allocation, task ledgers, mission trees, historical receipts, benchmark scratch data, and cluster-specific wrappers are not product API and will not be copied.

## Acceptance gates

1. Focused API tests pass from source.
2. Built-wheel command surface matches source exactly.
3. Existing exporter, plugin, acceleration-manifest, and Docker tests remain green.
4. Fresh-clone build/install/test passes without legacy-repository imports.
5. Public privacy and credential scan is clean.
6. GPU serve/performance parity is reported separately when exercised on supported hardware.
