# Canonical repository migration — 2026-08-03

## Decision

`https://github.com/my-other-github-account/banana-smasher` is the sole canonical Banana Smasher repository. The former subtree and refs in `spark-bench-reproducers` are read-only provenance sources.

## Semantic source authorities audited

| Surface | Legacy authority audited | Role in canonical integration |
|---|---|---|
| Standalone export/runtime baseline | `c00714c6803f7e2de7a95d103dbe172236b22adf` plus standalone extraction corrections | Preserve self-contained pack, stock-vLLM plugin boundary, manifests, and public build context. |
| Full fail-closed workflow API | `2d6bb2ec1fe2db09f9b3c48d1da52628711b6311` | Import public producer/workflow surfaces and exact completeness gates without legacy deployment baggage. |
| Serving/plugin acceleration | `50468029e846c926e8f0aaeb6c9efc1c1a1ac0de` | Import graph-safe native planes, padded-route safety, o-proj scaling fixes, and latest focused tests. |
| Grouped update acceleration | `19f08389c8e736b453c029f1f066ddb958961899` | Preserve grouped-VJP and BMM layer-graph update behavior with lazy optional dependencies. |

## Explicitly excluded

- built wheels and compiled `.so`/`.o`/cache outputs;
- model/tokenizer payloads and private deployment artifacts;
- internal host commands, paths, IPs, raw receipts, and unsanitized logs;
- duplicated golden-container trees and historical fork patches when the standalone stock-vLLM build already owns the behavior;
- stale or incomplete uncommitted work unless it is independently reviewed, minimized, and tested in this repository.

## Acceptance

The migration is complete only when the integrated canonical revision passes focused source tests, the complete static suite, both wheel builds and ZIP checks, Ruff, privacy/credential/history scans, license-surface checks, local/remote SHA readback, and a fresh anonymous network-clone gate. Linux ARM64 SM120/SM121 image build, stock-vLLM boot, API correctness, and acceleration/performance checks remain separate physical evidence gates and must be recorded under `notes/reports/`.
