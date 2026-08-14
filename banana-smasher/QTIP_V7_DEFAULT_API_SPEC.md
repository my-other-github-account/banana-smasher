# Native QTIP2-V7 Default API Specification

**Status:** Normative target contract

**Scope:** `banana-smasher` Python API, `smash` CLI, Backpack plan validation, provider dispatch, lifecycle receipts, package documentation, and installed wheel

**Primary lifecycle:** fresh DeepSeek-V4-Flash-0731 model → native QTIP2-V7 candidates → selected materialized Backpack → pre-repair anchor

**Core rule:** The ordinary public spelling `family: "qtip"` means native QTIP2-V7. V7 is not an opt-in mode.

## 1. Purpose

This document prevents the public API from drifting back to a legacy packaged-QTIP importer while separately exposing V7-named utilities. A V7 symbol, command, or module is insufficient by itself: the ordinary end-to-end Backpack lifecycle MUST physically invoke the V7 producer, V7 materializer/wire, and V7 runtime semantics.

The API is not compliant until an installed package proves this call edge from a fresh Flash declaration through a sealed `pre_repair_anchor` receipt.

## 2. Normative language

`MUST`, `MUST NOT`, `SHOULD`, and `MAY` are requirements for the public product surface.

- **Fresh Flash** means the caller supplies the standard model root/revision and ordinary declared training/anchor inputs. The caller does not supply campaign task roots or prebuilt legacy QTIP solve receipts.
- **Native QTIP2-V7** means the shipped V7 generation and physical wire semantics, including learned layer LUTs, V7 packed controls/indices/scales, hash-bound materialization, and fallback-zero execution.
- **Pre-repair anchor** means the measured selected mixed Backpack after exact solve/materialization and before any repair update.
- **Legacy packaged QTIP** means the path identified by `qtip-packaged-v1`, `banana-smasher-qtip-solve-v1`, `banana-smasher-qtip-unit-v1`, or `ds4-qtip-hyb-bounded36-unit-v1`.

## 3. Non-negotiable default

The following ordinary declaration MUST dispatch native QTIP2-V7:

```json
{
  "id": "qtip2",
  "family": "qtip",
  "bpw": 2.0
}
```

The caller MUST NOT need any of the following:

- `--v7` or an equivalent version selector;
- `provider: "qtip-v7"` to escape a legacy default;
- `backend: "v7"`;
- a campaign mission/task path;
- `source_root` pointing at generic packaged solve units;
- a separate V7 preprocessing workflow whose outputs are manually spliced into Backpack.

If a legacy importer remains for compatibility, it MUST have an explicit legacy identity that cannot be selected by `family: "qtip"`, cannot be advertised as the ordinary QTIP workflow, and cannot emit V7 method claims.

## 4. Public lifecycle API

### 4.1 Python one-call API

The intended stable surface is:

```python
from banana_smasher import BackpackPlan, build_backpack

plan = BackpackPlan.from_mapping(plan_document, base_dir=".")
receipt = build_backpack(
    plan,
    run_root="./backpack-run",
    through="pre_repair_anchor",
)
```

`build_backpack(..., through=...)` MUST accept public stage identifiers. At minimum:

```text
inspect
candidates
candidate_anchor
pred
solve_materialize
pre_repair_anchor
repair
final_score
```

Python uses the underscore spelling `pre_repair_anchor`. The function MUST execute every required predecessor in order, resume verified completed stages, stop after the requested stage, and not create later-stage receipts.

### 4.2 CLI one-call API

The equivalent CLI is:

```bash
smash backpack build \
  --plan ./fresh-flash-qtip2-v7.json \
  --run-root ./backpack-run \
  --through pre-repair-anchor

smash backpack status --run-root ./backpack-run
```

CLI uses the human-facing hyphen spelling `pre-repair-anchor` and maps it to the same internal stage as Python. CLI MUST call the same public implementation; it MUST NOT duplicate the lifecycle in command-specific code.

### 4.3 Composable stage API

The same lifecycle MUST remain callable as public stages:

```python
from banana_smasher import (
    inspect_backpack,
    generate_backpack_candidates,
    anchor_backpack_candidates,
    predict_backpack,
    solve_backpack,
    anchor_backpack,
)

inspect_backpack(plan, run_root=run_root)
generate_backpack_candidates(plan, run_root=run_root)
anchor_backpack_candidates(plan, run_root=run_root)
predict_backpack(plan, run_root=run_root)
solve_backpack(plan, run_root=run_root)
pre_repair = anchor_backpack(plan, run_root=run_root)
```

Calling these functions individually or through `build_backpack` MUST produce the same provider identity, stage outputs, hashes, and final pre-repair receipt.

## 5. Plan contract

The complete plan continues to use `banana-smasher-backpack-plan-v1` or an explicitly versioned successor if new fresh-input fields require a schema bump. The existing model, target, anchor, prediction, repair, and output declarations remain the lifecycle-level contract.

A representative plan shape is:

```json
{
  "schema": "banana-smasher-backpack-plan-v1",
  "model": {
    "root": "/models/DeepSeek-V4-Flash-0731",
    "revision": "<immutable-model-revision>"
  },
  "target": {
    "whole_model_bpw": 2.7
  },
  "tiers": [
    {
      "id": "qtip2",
      "family": "qtip",
      "bpw": 2.0
    }
  ],
  "anchor": {
    "bank": "/banks/anchor64.npz",
    "teacher": "model"
  },
  "prediction": {
    "class_caps": {
      "agentic": 1,
      "chat": 1,
      "code": 1,
      "multilingual": 1,
      "prose": 1,
      "reasoning": 1
    }
  },
  "repair": {
    "method": "residual",
    "strength": 0.5
  },
  "output": {
    "pack": "/packs/flash-qtip2-v7",
    "model_id": "DeepSeek-V4-Flash-0731",
    "instance_id": "flash-qtip2-v7"
  }
}
```

The exact checked-in example MUST validate against the shipped schema. If V7 generation requires additional inputs that the current plan cannot express, the implementation MAY introduce one generic, portable input declaration. It MUST NOT expose campaign-specific host names, task IDs, `/dev/shm` paths, or prebuilt mission receipt paths as public API requirements.

For `through="pre_repair_anchor"`, repair/output declarations MAY remain present for one reusable full-lifecycle plan, but no repair code may execute and no repair-stage receipt may be written.

## 6. Required effective call graph

For `family: "qtip", bpw: 2.0`, executable control flow MUST be equivalent to:

```text
BackpackPlan validation
  → ordinary QTIP provider resolution
  → native QTIP2-V7 fresh-input adapter
  → native V7 batch/unit production
  → V7 candidate receipt and exact byte pricing
  → V7 materialization / fixed physical wire
  → V7 wire verification and accounting
  → same-instrument candidate anchor
  → prediction
  → exact-byte solve and selected materialization
  → selected pre-repair anchor
```

The production edge MUST reach `produce_qtip2_v7_batch10()` or its declared generic V7 successor. The physical edge MUST reach the V7 wire implementation represented by `pack_qtip_v7_layer()`, `verify_qtip_v7_layer()`, and model-level V7 accounting, or a direct successor with identical stronger semantics.

Merely importing a V7 module, reporting a V7 provider name, or testing a V7 helper separately does not satisfy this contract.

## 7. Algorithm and geometry requirements

The ordinary QTIP2 route MUST preserve native V7 semantics:

- V7 QTIP2 geometry and current production batching;
- learned LUT identity at the required layer granularity;
- native packed controls/indices/scales and embedded LUT payload;
- complete expert roster and projection identity;
- exact physical stored-wire byte accounting;
- source model/basis/calibration bindings;
- deterministic resume from verified stage receipts;
- no silent CPU/reference/legacy substitution;
- positive native V7 execution counters;
- zero fallback calls.

A legacy artifact may not be relabeled as V7 based on rate, geometry similarity, or a compatible decoder.

## 8. Stage order and stopping semantics

The required pre-repair transaction is:

1. `inspect` — bind the immutable Flash model and derive the cell inventory.
2. `candidates` — physically generate native V7 QTIP candidates from fresh declared inputs.
3. `candidate_anchor` — measure every admitted candidate with the declared anchor instrument.
4. `pred` — emit the six-class prediction rows used by the solver.
5. `solve_materialize` — solve the exact byte envelope and materialize the selected V7-backed assignment.
6. `pre_repair_anchor` — measure and seal the selected assignment before repair.
7. Stop.

When `through=pre_repair_anchor`:

- `repair` MUST NOT execute;
- `final_score` MUST NOT execute;
- no repair checkpoint/update may be loaded or created;
- status MUST report `pre_repair_anchor` complete and `repair` as the first incomplete boundary;
- rerunning the same command MUST resume verified predecessor receipts rather than regenerate them.

## 9. Required receipts and proof fields

Every QTIP candidate and selected pre-repair receipt MUST contain a readable, machine-verifiable method identity. Exact field placement may follow the existing receipt schema, but the information MUST include:

```json
{
  "family": "qtip",
  "method": "qtip2-v7",
  "runtime_family": "qtip2-v7",
  "generation": {
    "qfn_calls": 1,
    "extension_calls": 1,
    "cuda_tiles": 1,
    "fallback_calls": 0
  },
  "wire": {
    "format": "qtip-v7",
    "verified": true,
    "physical_bytes_authenticated": true
  }
}
```

The positive integers above are illustrative; acceptance requires `qfn_calls > 0`, `extension_calls > 0` or the current native-kernel equivalent, `cuda_tiles > 0` for a CUDA physical run, and `fallback_calls == 0`.

The terminal pre-repair result MUST additionally bind:

- plan SHA-256;
- immutable model revision/identity;
- anchor bank and teacher identities;
- selected assignment SHA-256;
- selected materialized pack/wire SHA-256 values;
- prior-stage receipt SHA-256 chain;
- finite pre-repair KLD and Top-1 counts;
- exact stored-wire bytes and denominator;
- `completed_stage: "pre_repair_anchor"` or schema-equivalent state;
- `repair_executed: false` or equivalent receipt-level proof by absent later stages.

## 10. Forbidden evidence and fail-closed behavior

The native V7 route MUST reject or quarantine as legacy any ordinary-QTIP candidate whose effective path contains:

```text
qtip-packaged-v1
banana-smasher-qtip-solve-v1
banana-smasher-qtip-unit-v1
ds4-qtip-hyb-bounded36-unit-v1
fixture_reference
```

The public lifecycle MUST fail closed on:

- missing fresh model/calibration inputs;
- model or plan hash drift;
- foreign or campaign-only source paths;
- missing V7 producer/runtime dependencies;
- V7 counter absence;
- any fallback count above zero;
- partial layer/expert/projection rosters;
- wire byte or SHA drift;
- candidate/selection identity mismatch;
- pre-repair anchor measured against a different selected materialization;
- a requested stage name not in the public lifecycle.

There is no legacy fallback when V7 prerequisites are unavailable.

## 11. Status and export semantics

`smash backpack status --run-root ROOT` MUST report completed stages, their receipt identities, and the first incomplete boundary.

A future `smash backpack export --lifecycle pre-repair` MUST require and bind the actual `pre_repair_anchor` receipt, not merely `solve_materialize`. Exporting a selected pre-repair model without the measured pre-repair anchor is noncompliant.

## 12. Documentation and packaging requirements

This specification MUST be:

1. checked into the canonical `banana-smasher` package repository;
2. linked from the public package README next to the Backpack quick start;
3. identified as normative, with conflicting packaged-ring/source-root text removed or explicitly marked legacy;
4. included in the built wheel through package data or the existing force-include mechanism;
5. readable from an extracted installed wheel;
6. accompanied by one schema-valid example fresh-Flash QTIP2-V7 plan;
7. exercised by the exact Python and CLI commands printed in the documentation.

Chat messages, Kanban descriptions, and campaign notes are corroborating evidence only. They do not replace the checked-in and shipped contract.

## 13. Conformance tests

The release is blocked unless focused tests prove:

1. **Default dispatch:** ordinary `family: "qtip", bpw: 2.0` resolves to V7.
2. **No version flag:** neither Python nor CLI requires a V7 selector.
3. **Fresh inputs:** the documented fixture starts from the public model/input declaration, not a legacy packaged solve root.
4. **Call-edge sentinel:** the V7 producer/wire sentinel is called; the legacy packaged loader sentinel is not.
5. **Through boundary:** build stops after `pre_repair_anchor`; no repair/final receipt exists.
6. **Resume:** a second run reuses hash-verified completed stages.
7. **Receipt proof:** V7 method identity and positive native counters are present; fallback is zero.
8. **Python/CLI parity:** both surfaces produce equivalent provider/stage identity.
9. **Schema parity:** the exact example plan validates with the shipped schema and executable parser.
10. **Installed artifact:** a clean environment imports the built wheel, prints module provenance, runs the documented command, and reads the expected receipt.
11. **Wheel contents:** this specification, example plan, schemas, and links are present and resolve.
12. **Legacy rejection:** an ordinary QTIP declaration cannot silently load a legacy packaged-QTIP receipt.

A physical hardware canary is required before claiming CUDA performance or production deployment. Source and installed-wheel API closure may be reported separately from hardware proof, but may not imply hardware execution that did not occur.

## 14. Versioning and compatibility

The semantic default `family: "qtip" → native QTIP2-V7` is part of the public API contract once released. Reverting it requires an explicit breaking API decision and documentation update; it may not occur through provider-table drift.

Existing callers that intentionally need legacy packaged artifacts MUST migrate to an explicit legacy import surface, if retained. Compatibility code MUST NOT weaken the ordinary V7 path or its receipts.

A later V8 may replace V7 only through a deliberate default-version policy and migration specification. The current contract is V7.

## 15. Release acceptance table

| Surface | Required result |
|---|---|
| Plan | Ordinary `family: qtip`, `bpw: 2.0`; no V7 flag or legacy source root |
| Python | `build_backpack(..., through="pre_repair_anchor")` |
| CLI | `smash backpack build ... --through pre-repair-anchor` |
| Provider | Native QTIP2-V7 selected by default |
| Producer | V7 batch/unit producer physically invoked |
| Wire | V7 physical wire packed, verified, and accounted |
| Anchor | Selected V7-backed materialization measured before repair |
| Counters | Native calls positive; fallback zero |
| Stop boundary | Repair and final score absent |
| Docs | Normative spec + example linked and shipped in wheel |
| Tests | Source, CLI, schema, wheel, and call-edge conformance PASS |

## 16. Current source-truth gap at specification freeze

At canonical `origin/main` commit `e6d0fb1f2b2f7d1f414bc5d1387aee1bf40d578b`, the contract is not implemented:

```text
family: qtip
  → qtip_ring_backpack_provider()
  → generate_qtip_backpack_candidate()
  → _packaged_qtip_record()
```

That implementation consumes legacy packaged solve schemas and does not call the separate V7 producer/wire modules. `smash backpack build` also lacks a `--through` boundary and runs through repair/final score. This appendix is evidence of the gap, not permission to preserve it.

The API is GREEN only after the checked-in implementation, installed artifact, and conformance receipts satisfy this specification.
