# D4K4096 L040–L042 dynamic Backpack dimension sidecars

Date: 2026-08-03

## Scope and immutable inputs

This report covers the fail-closed dimension-sidecar pass for D4K4096 layers 40–42. It consumed the already-sealed solve handoff without replaying solver rows, allocating tiers, packing a model, or booting a model runtime.

| Binding | SHA-256 |
|---|---|
| Source model/index basis | `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` |
| Immutable L040–L042 producer handoff | `6be1684b135a2f287cec2c2e77c425b58092a2a71c348197aba478b6a08126cc` |
| Final sidecar manifest v2 | `820e0087e2253775308ceac4e7e2d710e556ea01212f09aacfe6b9d97c4d15e0` |
| Exact authority blocker receipt v2 | `c2dbbaa5f89c70c9f93be1e4d9930b9de2b11802b87291293dd944a51600151c` |
| Final terminal receipt v2 | `7c9d91aa26b2e5ca4bacb26bf225176ecd7ed672f0ed015bc37962877b32ff48` |

## Published non-inferred bindings

The producer verifies the source handoff hash, basis, requested layer list, and every referenced objective/profile member before publishing. Candidate identity is parsed only from authenticated objective assignment keys. Packed-wire bytes are the exact integer product of recorded assignment count and the recorded `int16-le` element width. Routing importance is the authenticated per-expert `routed_rows` value; it is not normalized or converted into a quota.

| Layer | Candidate identities | Routing-importance bindings | Packed-wire-byte bindings | Exact packed-wire bytes |
|---:|---:|---:|---:|---:|
| 40 | 512 | 512 | 512 | 3,221,225,472 |
| 41 | 512 | 512 | 512 | 3,221,225,472 |
| 42 | 512 | 512 | 512 | 3,221,225,472 |
| **Total** | **1,536** | **1,536** | **1,536** | **9,663,676,416** |

The three published groups therefore contain 4,608 binding rows in total.

## Fail-closed authority result

No explicit authenticated per-candidate authority was present for the following required groups:

| Required group | Disposition |
|---|---|
| Six-class predictions (`agentic`, `chat`, `code`, `multilingual`, `prose`, `reasoning`) | Blocked; no per-candidate authority consumed |
| Accepted six-class ceilings | Blocked; no accepted six-class cap authority consumed |
| Projection weights | Blocked; no explicit per-candidate projection-weight authority consumed |
| Projection corrections | Blocked; no explicit per-candidate projection-correction authority consumed |

The private blocker receipt records the exact required keys, expected authority locations, and hashes of every searched source member. A separately known predictor candidate was not local to the assigned execution boundary and was not consumed. No aggregate-to-cell inference, qtip2 substitution, fixed quota, default cap, or projection heuristic was used. Allocation remains forbidden.

## Public implementation and verification

Reusable implementation and tests are in the canonical package:

| File | SHA-256 |
|---|---|
| `banana-smasher/src/banana_smasher/backpack_dimensions.py` | `71a752bc3bb218929eeb28195da1b3e69bde2dfda6e0d818057f148dc3fc9173` |
| `banana-smasher/src/banana_smasher/cli.py` | `7c74800db4e6b2760aa6aa214bcc4b4110498b01debfb81941c8ff2732ab3895` |
| `banana-smasher/tests/test_dynamic_backpack_dimensions.py` | `47c25a8715ccc8a8949083d25964cd2a618a3ad6eae0ffbc023d5b6da78921e0` |
| `banana-smasher/tests/test_cli.py` | `36a950e35052877fda7d9e2ec8a55ccd09eacf09a414cc2ed3e6b58a7216072e` |

The public command is `smash backpack-solved-sidecars`. Focused verification passed 9 tests; the complete package suite passed 49 tests with 5 platform-specific skips. Ruff reported no issues. The wheel build and ZIP/license-surface inspection passed for wheel SHA-256 `fdfbaf8e45ea70bc99e8d9d6f9c97549f1bb28c07bba56466f47ccfbb52466e4`.

The first runtime pass used module SHA-256 `192fe4e755311fd2ab06816037145cca219f2eebdcfcfc7a363892593e878ff4`. The final reusable source adds explicit authority-expectation input validation and the public CLI. The v2 blocker/manifest correction was additive and referenced the original sealed sidecars; it did not replay the 1,536 source rows.
