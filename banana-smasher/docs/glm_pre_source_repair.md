# GLM PRE-only source repair: t_29906bc1

This is a source-only change, not deployment, a producer restart, clean-lineage
admission, a cell rebuild, or a model-quality result. The physical owner remains
`t_024b05d4`. The authoritative scope is `REPAIR_SCOPE.txt` from `t_3e8c53bc`.
The implementation baseline is canonical main
`6bf59df320a76f06bf6f58adddc0dcf9b759a31a` (fetched after path adjudication).

## Exact changed-path / evidence-scope map

Paths below are relative to the repository root.

1. `banana-smasher/src/banana_smasher/glm_qtip_source_adapter.py`
   — scope 3, lines 31–33; scope 4, lines 35–38. Canonical FP8 source loading,
   gate/up row concatenation, down orientation, all configured routed IDs,
   separate scale-shard identity, and actual imported closure. The loader uses
   the existing HF dtype/block-scale decoder; no quantizer, TLUT, fit data,
   calibration policy or metric is changed.
2. `banana-smasher/src/banana_smasher/solver_qtip_profile.py`
   — scope 3, lines 31–33. Dispatch GLM weights through that canonical adapter
   rather than requiring an external monkeypatch. Bind closure before reference
   loading and include it in solve and profile receipts. Update the trusted runner
   digest to the actual canonical runner introduced by `56424c1`,
   `f9da4f5cf97ffab622da3444556e281f710b8b73c5eaf8fc3c48cf857dcdf9df`.
   The old digest rejected the unchanged current runner; the existing physical
   pack regression reproduced that rejection and now passes. No runner bytes
   were copied from deployment or changed by this card.
3. `banana-smasher/src/banana_smasher/qtip_batch_controller.py`
   — scope 3, lines 31–33. Bind the same launch closure before reference loading
   and carry it into every batch-member receipt.
4. `banana-smasher/src/banana_smasher/qtip_batch.py`
   — scope 3, line 33; provenance audit line 59. Derive receipt K/L/V and the
   implementation name from the actual codebook. Previously K1/K3/K4 builds
   emitted K2 metadata. Historical receipts are not rewritten.
5. `banana-smasher/src/banana_smasher/glm_qtip_producers.py`
   — scope 2, lines 24–29; scope 3, line 33; scope 4, lines 35–38. A single
   read-only registry for the three existing protected roots, correct host/layer
   ranges, all288 fused13/down cells, exact fan-in membership and honest native
   exclusions. Adoption validation refuses stale/copied claims, wrong host/range,
   dead/reused PID observations, expired claims and 256-expert shard maps. It does
   not create a second producer or write claims. Owner-controlled CAS and a fresh
   process census remain prerequisites, not inferred authorization.
6. `banana-smasher/tests/test_glm_qtip_repair.py`
   — regressions for paths 1–4: CPU synthetic source/receipt fixtures, split FP8
   scale shards, fused/down suffix loading, K1..K4 metadata, actual import-role
   hashes, pin drift, external-loader rejection, and both receipt/gate call sites.
   GPU kernels are test doubles in the receipt test, not quality evidence.
7. `banana-smasher/tests/test_glm_qtip_producers.py`
   — regressions for path 5: complete/disjoint 24,192-cell Cartesian roster,
   source projection omissions, basis mismatch, duplicate/missing/misassigned
   fan-in rows and unsafe adoption observations.
8. `banana-smasher/docs/glm_pre_source_repair.md`
   — this change map, integration contract and explicit verification boundary.

## Existing producer integration (owner clean boundary only)

Use `producer_plan(host, source_root, intended_basis=...)` from
`banana_smasher.glm_qtip_producers`. Host names are exactly `spark-3`,
`spark-5-work`, and `spark-7`; roots retain the existing owner and spelling.
Ranges are respectively L003–L016, L017–L030, L031–L044. Each plan contains
8,064 cells, all E000–E287 and both projections. `source_names` preserves the
`[gate; up]` order. `historical_receipt` is a preservation/census pointer, never
an instruction to overwrite or a claim of existence/admission.

The existing producer owner consumes the plan's cells instead of its old
`range(256)` / hardcoded fan-in count. Use the existing public
`solver_qtip_profile.main_many` / `qtip_batch_controller.main_batch` path with
fresh, owner-admitted configs and output generation. Do not introduce another
solver loop, hand-copy deployed modules, or replay `FULL_SOLVE.sh`, the old
non-CAS wrapper, or the old `NEXT_EVAL` fan-in. This card changes no remote
launcher or control file. Migrate those external call sites only under the
physical owner's clean-boundary authorization; a plan is not a deployment.

`validate_adoption(plan, claim, shards, process, now=...)` checks fresh
owner-provided observations, including PID/start ticks and actual argv-derived
host/ranges/root. It is deliberately not a remote census or a CAS writer.
Any rejection requires owner reconciliation; it never grants a launch or
preempts a live claim. Even `ADOPT_EXISTING` has `launch_authorized=False` and
`heldout_admitted=False`. The historical claim snapshots in the parent audit
must not be submitted as current process observations.

`validate_fanin_roster(source_root, basis, rows)` requires exactly one
`{"host": ..., "id": "L003_E000_fused13"}` row for every protected host/cell.
Its result is only `ROSTER_COMPLETE_ONLY`: physical hashes, clean fitting,
consumer parity and held-out admission are explicitly false/unverified.
The existing HF public routed-only admission gate already rejects a native
E256–E287 suffix; no competing runtime or metric-admission path was added.

## Launch closure

In the real authorized interpreter, load the hash-bound canonical runner and
its `load_official_qtip()` modules. Call `capture_source_closure(runner,
{"bitshift": bitshift, "ldlq": ldlq, "math_utils": math_utils,
"kernel_decompress": kernel_decode})`. Preserve this raw local attestation and
bind its `sha256` as `glm_source_closure_sha256` in each new config. Both single
and batch builders recompute it before loading references and reject missing
or mismatched pins. External `_load_weight` monkeypatches are rejected.

The closure records resolved module files, all four actually loaded runtime
roles (the decoder is now canonical package code), controller/batch/solver,
runner/viterbi/cache/rings, HF adapter and producer integration, Python binary
hash/version and installed torch/triton/numpy/safetensors versions. A missing
distribution is recorded as null, not fabricated. It asserts no upstream Git
commit for an unversioned runtime and does not pretend to hash historical
Python memory. The digest covers the explicit path-independent `identity`
object so it remains reproducible after the standard public receipt path
redaction; retain the raw local closure for absolute import paths.

## Deliberately retained prerequisites

The current canonical baseline already includes K1..K4 controller/decode
geometry, wide-call chunking, two-pass K1 memory accounting and the canonical
Inductor-safe packed decoder. Those implementations were not replaced with
K1-only deployment diffs or graph-break workarounds.

All original cells, source trees, Hessians, captures, seeds, receipts, native
rest tensors and historical scores remain untouched. Q1 historical output is
`partial K1/256-expert-plan diagnostic`; Q2 is the native-suffix hybrid described
in the plan. Neither is uniform held-out PRE. Main-text MTP-excluded/native-rest
behavior is retained; the main-layer suffix is not an exemption.

Before any solve: physical-owner clean boundary plus fresh claim/shard CAS and
census reconciliation; immutable canonical deployment and raw import closure;
authenticated disjoint fit ledger and clean dependent lineage; the inherited
bounded same-cell and real-output gates with owner numerical adjudication.
Nothing here authorizes those operations or changes their thresholds.

## Verification boundary

The new tests run on CPU with synthetic tensors/metadata. They exercise real
source descaling and real batch receipt generation, but do not perform the
real-model/GPU eight-unit parity suite or the paired four-row output gate.
Review/deployment/real-output verification remain with the existing downstream
cards. No new worker card or producer was created.

Final local verification: 123 focused tests passed (14 existing Torch deprecation
warnings). A broader 152-test run had 149 passes and three failures: missing
`hf-sharded` runtime entry-point discovery, plus two native-output-fit failures.
All three failures were reproduced with immutable baseline `6bf59df` source in
the same interpreter; no new failures were introduced. The formerly failing
canonical-runner physical-pack test now passes after the reviewed digest update.
Ruff checks on the four new Python files and `git diff --check` passed. This is
not an all-repository or GPU-suite pass; no environment installation, storage
workaround or unrelated source repair was substituted for the baseline failures.
