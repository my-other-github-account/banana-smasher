# Canonical Backpack provider fan-in

This branch replaces the conflicting accelerated-QTIP-only PR #7 with one
Backpack construction surface.

## Public surface

- composable provider operations: generate, price, predict, materialize, verify
- plan stages: inspect, generate, candidate anchor, predict, exact solve,
  materialize, repair, score, and final pack
- one orchestration path: `build_backpack(...)` and `smash backpack build`
- built-ins: native MXFP4, QTIP2/QTIP2.5/QTIP3, D4 K2048/K4096
- extension proof: authentic QTIP1 and declaration-only QTIP1.5 with distinct
  K1/V1 and K2/V2 semantics
- standalone `physical-repair` update backend; no legacy runtime factory

Shared activation bytes are priced from candidate receipts and charged once by
the exact solver. The orchestration path resolves and calls the same public
provider bindings; there is no second private family-dispatch implementation.

## Migration from PR #7

Use `solve_qtip_profiles(...)` or the existing batched QTIP CLI options for
accelerated QTIP production. Use provider declarations in a Backpack plan when
QTIP is one tier in an exact whole-model solve. Existing QTIP2/QTIP2.5/QTIP3
runtime family names and wire semantics are preserved; native MXFP4 is not
relabeled.

The installed-wheel lifecycle smoke is recorded in
`notes/backpack-provider-api-smoke.json`. QTIP1/QTIP1.5 direct stock-vLLM K1/V1
execution remains a known limit, not a serving claim.
