# banana-smasher

`banana-smasher` is the reusable, fail-closed `bs-pack v1` build and validation toolchain. `PACK_FORMAT.md` is the versioned pack contract: plane layout, per-layer metadata, `config.json` auto-detection keys, complete byte-count/SHA-256 manifest, and rejection rules.

## Improve a resident checkpoint with one verb

Use the same command on both ranks of the reserved pair; the distributed
launcher supplies `RANK=0` and `RANK=1`. The admitted artifact contains its two
rank configs, so callers do not choose a rails config or training recipe:

```console
smash improve /local/admitted-pre \
  --run-root /local/improve-run \
  --checkpoint-sha CHECKPOINT_SHA256 \
  --updates 45
```

The command runs the zero-update Balanced64 score, 45 guarded updates, and the
post-update score in fresh phase processes, restoring only authenticated phase
receipts. The validated U45 recipe is package owned: broad rotation, FP32-safe
loss reduction, FP64 optimizer moments, per-class learning rates, per-update
loss instrumentation, and held-out kill gates. The command exits nonzero unless
`post_kld < pre_kld` and writes `score_pre.json`, `repair_train.json`,
`score_post.json`, and `IMPROVE_RESULT.json` under the run root.

For the exact `ResidentRepairAPI.build_uniform(...) -> score_pre() ->
repair_train(updates=45) -> score_post()` sequence and the two-rank launch
contract, see [WORKED_EXAMPLE.md](WORKED_EXAMPLE.md) and
[REPAIR_TRAINING_GUIDE.md](REPAIR_TRAINING_GUIDE.md).

## End-to-end Backpack plans

### Five-minute quickstart

The public surface has two layers. Family providers expose independently
callable generation, materialization, receipt pricing, prediction, and
verification bindings; `build_backpack` and `smash backpack build` compose the
same public stage APIs into one resumable run.

```python
from banana_smasher import (
    BackpackPlan,
    build_backpack,
    builtin_backpack_family_providers,
    qtip1_5_provider_declaration,
)

providers = builtin_backpack_family_providers()
assert set(providers) == {
    "native-mxfp4", "qtip@2.00", "qtip@2.50", "qtip@3.00",
    "d4-k2048", "d4-k4096",
}
qtip15 = qtip1_5_provider_declaration()
assert qtip15.tier == "qtip@1.50"
assert [(row.geometry.K, row.geometry.V) for row in qtip15.components] == [(1, 1), (2, 2)]
plan = BackpackPlan.from_mapping(plan_mapping, base_dir=".")
result = build_backpack(plan, run_root="./backpack-run")
```

The provider menu is declaration-driven: QTIP rates use the packaged ring
table, D4K2048/K4096 bind the production fixed-D4 prepare/materialize/logit
APIs, and `vector_vq_backpack_provider(...)` covers independently callable D4
or D8 vector-VQ fixtures. Prices are read from candidate receipts as per-cell
payload bytes plus shared activation artifacts; the exact solver charges each
activation identity once.

The equivalent CLI path is:

```console
smash backpack build --plan plan.json --run-root ./backpack-run
smash backpack status --run-root ./backpack-run
smash verify ./backpack-run/pre-repair-pack
```

### Sparse mixed Q2/Q3 inventories under a model-size budget

`smash backpack solve-mixed` is the config-driven allocation seam for a
partially available QTIP3 inventory. It consumes the sealed per-candidate rows
emitted by `backpack-dimensions`, verifies their basis and ledger SHA, and calls
the same `solve_class_balanced_options` MILP used by normal Backpack plans.
Missing QTIP3 rows are disabled options, never inferred data; the configured
QTIP2 fallback must exist for every cell. Decimal GB is explicit: 102 GB is
`102000000000` bytes.

```json
{
  "schema": "banana-smasher-mixed-backpack-config-v1",
  "basis_sha256": "MODEL_INDEX_SHA256",
  "target": {
    "whole_model_bytes": 102000000000,
    "fixed_nonexpert_bytes": 9032112614,
    "exact": true
  },
  "allowed_tiers": ["qtip2", "qtip3"],
  "fallback_tier": "qtip2",
  "topology": {
    "layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
    "experts_per_layer": 256,
    "projections": ["down", "fused13"]
  },
  "dimensions": {
    "sources": [
      {"path": "./PARTIAL_DIMENSIONS.jsonl", "sha256": "PARTIAL_SHA256"},
      {"locator_path": "./FINAL_DIMENSIONS_LOCATOR.json"},
      {"locator_path": "./QTIP3_PHYSICAL_LOCATOR.json"}
    ]
  },
  "class_caps": {
    "agentic": 1000000,
    "chat": 1000000,
    "code": 1000000,
    "multilingual": 1000000,
    "prose": 1000000,
    "reasoning": 1000000
  }
}
```

```console
smash backpack preflight-mixed --config mixed-102gb.json
smash backpack solve-mixed --config mixed-102gb.json --output ./mixed-102gb
```

When an exact solve proves that the admitted physical inventory cannot reach the
configured byte target, select a recovery tranche in a
`banana-smasher-mixed-v7-recovery-plan-v1` document and bind its future
measurements with a
`banana-smasher-mixed-v7-sensitivity-extraction-contract-v1` document. Admit
both, without acquiring a host claim, through the same Backpack API:

```console
smash backpack preflight-mixed-v7-recovery \
  --plan recovery18.json \
  --sensitivity-contract recovery18-sensitivity.json
```

The preflight reopens every local source-stage terminal, verifies its bytes,
SHA-256, model-index basis, layer, and payload identity, proves the tranche
contains enough expert options, and checks that the sensitivity contract closes
over the exact ordered layer set and canonical Git commit. A
`PASS_READY_TO_CAS` receipt is launch input, not physical-product acceptance;
callers must still acquire host and shard claims, generate the sealed products,
measure the declared rows, then feed them to `bind-mixed-v7-physical` and the
normal exact solve.

`preflight-mixed` admits already sealed shards, reports exact missing fallback
projection cells, and returns `WAITING_FOR_DIMENSION_LOCATORS` while a declared
locator is absent. The config does not change when a producer publishes that
locator. A locator uses schema
`banana-smasher-mixed-backpack-dimensions-locator-v1`, binds the model basis,
and contains a normal `{path, sha256}` dimensions descriptor. A dimensions
source may contain either allocation-ready projection rows or sealed
`banana-smasher-sensitivity-row-v1` expert rows. Expert sensitivity rows bind
one combined down+fused13 byte total and measured scalar damage; the API
losslessly carries that total on the canonical down row and a zero companion
row so the existing projection aggregator reconstructs the exact expert
option. The class-neutral scalar is used identically by all six balancing
lanes and this policy is recorded in the option authority; no projection or
class-specific value is inferred.

A physical locator uses schema
`banana-smasher-mixed-backpack-physical-locator-v1` and binds a complete
`physical_manifest` descriptor. When that manifest uses schema
`banana-smasher-mixed-backpack-physical-members-v1`, each member binds one
`Lxxx.Exxx.{down,fused13}` plus tier to a host, path, byte count, and SHA-256.
The solve filters this reusable full inventory by the chosen assignment and
seals the hash-bound subset as `SELECTED_PHYSICAL_MEMBERS.json`; no canary path
or unselected member is promoted.

`solve-mixed`
then auto-consumes all locators, refuses if any remains pending or if topology
lacks QTIP2 fallback, and records every source and locator hash in its receipt.

The output seals `ASSIGNMENT.json`, `identity.json`, `RECEIPT.json`, and (when
physical member bindings are supplied) `SELECTED_PHYSICAL_MEMBERS.json`.
`identity.json` records the complete cell assignment, QTIP3 coverage and
missing layers, per-layer tier counts, and a deterministic relative-path/hash
descriptor for the selected physical roster. `ASSIGNMENT.json` expands each
layer/expert choice into the projection-level materialization assignments used
by the existing loader/contract assembly path. A 96/108/115 GB variant changes only
`target.whole_model_bytes`; tier policy remains in `allowed_tiers`. The public
schema is `schema/banana-smasher-mixed-backpack-config-v1.schema.json`.


Migration: callers using `generate_vector_vq_backpack_candidate`,
`generate_qtip_backpack_candidate`, or `materialize_backpack_source` may keep
those specialized functions. New integrations should use
`generate_backpack_candidate`, `materialize_backpack_assignment`,
`price_backpack_candidate`, `predict_backpack_candidate`, and
`verify_backpack_candidate`; the specialized functions remain implementation
bindings rather than competing workflows.

`BackpackPlan` is the versioned public input for the complete resumable path:
model inspection, D4/D8/QTIP candidates, same-instrument Anchor64, six-class
prediction rows, exact-byte assignment/materialization, pre-repair anchor,
repair, and final score/pack. The JSON schema is shipped in the source tree at
`schema/banana-smasher-backpack-plan-v1.schema.json`.

```json
{
  "schema": "banana-smasher-backpack-plan-v1",
  "model": {"root": "/models/M", "revision": "MODEL_REVISION"},
  "target": {"whole_model_bpw": 2.7},
  "tiers": [
    {"id": "d4-k2048", "family": "vector_vq", "dimension": 4, "codebook_size": 2048},
    {"id": "d8-2bpw", "family": "vector_vq", "dimension": 8, "bpw": 2.0},
    {"id": "qtip-2.0", "family": "qtip", "bpw": 2.0, "source_root": "/qtip/configs"}
  ],
  "anchor": {"bank": "/banks/anchor64.npz", "teacher": "model"},
  "prediction": {"class_caps": {"agentic": 1, "chat": 1, "code": 1, "multilingual": 1, "prose": 1, "reasoning": 1}},
  "repair": {"method": "residual", "strength": 0.5},
  "output": {"pack": "/packs/M-backpack", "model_id": "M", "instance_id": "M-backpack-v1"}
}
```

The v1 direct adapter infers cells and fixed dense/metadata/repair bytes from
`BACKPACK_MODEL.json` under the model root. Its Anchor64 bank is an NPZ with
`features: float32[64, weight_count]` and six-class `classes: str[64]`.
Impossible grouping, geometry, QTIP increments, class caps, or byte envelopes
fail explicitly; no family is substituted. D4 and D8 use true 4- and 8-weight
vector grouping with packed code indices. Production QTIP tiers require a
canonical ring-bound `source_root` and fail closed rather than falling back to
the CPU fixture backend. Synthetic tests must opt in explicitly with
`"backend": "fixture_reference"`.

Run every stage or inspect the first incomplete boundary:

```console
smash backpack build --plan plan.json --run-root ./backpack-run
smash backpack status --run-root ./backpack-run
smash backpack export --run-root ./backpack-run --lifecycle uniform-anchor --tier d4-k2048 --serving-model-root /models/M --output ./uniform-model
smash backpack export --run-root ./backpack-run --lifecycle pre-repair --serving-model-root /models/M --output ./pre-repair-model
smash backpack export --run-root ./backpack-run --lifecycle post-repair --serving-model-root /models/M --output ./post-repair-model
```

All three lifecycle exports use the same `bs-pack` exporter, model directory,
`quant_method="banana_smasher"`, and `PackLoader` ABI. The export receipt binds
the selected assignment, expert wire layout, whole-model tensor shapes, and
reports expert planes, base weights, repair state, and metadata bytes separately.
Uniform export assigns the named tier to every cell. Pre- and post-repair export
the selected assignment from the same run root; post-repair reuses the verified
repair inputs from the plan.

The orchestrator is only a composition of the same independently callable
public stage functions:

```python
from banana_smasher import (
    BackpackPlan, inspect_backpack, generate_backpack_candidates,
    anchor_backpack_candidates, predict_backpack, solve_backpack,
    anchor_backpack, repair_backpack, score_backpack,
)

plan = BackpackPlan.from_mapping(plan_mapping, base_dir=".")
for stage in (
    inspect_backpack, generate_backpack_candidates, anchor_backpack_candidates,
    predict_backpack, solve_backpack, anchor_backpack, repair_backpack,
    score_backpack,
):
    stage(plan, run_root="./backpack-run")
```

The lower-level public adapters
`generate_vector_vq_backpack_candidate(...)`,
`generate_qtip_backpack_candidate(...)`, and
`materialize_backpack_source(...)` are also importable from `banana_smasher`
for producer/materializer integration without calling the private orchestrator.

Each successful stage writes a plan-bound receipt and is skipped on resume.
Plan entries that bind admitted `candidates` and `candidate_anchor` stage
receipts import those stages without candidate-generation or anchor replay;
`reuse_backpack_receipts(...)` also hash-binds other completed campaign
receipts as evidence. Use
`admission="evidence_only"` for quarantined diagnostics or historical rails;
they are retained as evidence but cannot be promoted as current solve inputs.

## QTIP 2.5 native V4 cell API

The homogeneous `L16/B10/V4` winner is available through
`build_qtip25_native_v4_cell(...)` and
`anchor_qtip25_native_v4_cell(...)`, with matching
`smash qtip-native-v4 build-cell` and `anchor-cell` commands. The CUDA path
binds physical source weights, a same-basis compact QTIP transform, and the
shared Q9/V2 TLUT; it emits exact 2.5-BPW codes, a physical decoded cell, and
hash-bound mechanics/anchor receipts. See
[QTIP25_NATIVE_V4_API.md](../archive/notes/QTIP25_NATIVE_V4_API.md) for the measured
configuration and a pasteable agent brief.

## Three-command release path

```bash
smash export --source-root /path/to/materialized-quant-source --runtime-floor-bytes "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}" --serving-model-root /path/to/base-model --output /model --model-id MODEL --instance-id PACK_INSTANCE --link-mode copy
smash verify /model
vllm serve /model
```

The P1016 export requires `RUNTIME_FLOOR_BYTES` from the caller's measured runtime receipt. The example intentionally has no default and never guesses this residency value. The first command builds `/model`, merges the full base-model `config.json` with the pack-owned `quantization_config`, copies `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json`, and writes `BANANA_PACK_MANIFEST.json` last after self-verification. `--serving-model-root` must point at a serveable model directory whose config has a non-empty `architectures` list; the exporter rejects missing tokenizer metadata rather than creating a quant-only pack that vLLM cannot boot. The second command fails closed on missing or extra files, byte-count or SHA-256 drift, schema/version mismatch, invalid metadata, and incompatible config auto-detection keys. The third is the stock vLLM command; no banana-smasher launcher or environment-only format selection is required.

To repair serving metadata in an already validated pack without touching tensor files, rerun the same verb in metadata-only mode by adding `--refresh-metadata` to that export command (and keep the same `--serving-model-root`).

This preserves the existing pack `quantization_config`, rewrites only the four serving metadata files plus their manifest rows/provenance, revalidates the pack, and reports `tensor_payloads_rewritten: false`.

## Standard whole-model BPW accounting

Use the public `build_bpw_accounting(...)` API or `smash bpw` command for model
sizes, comparison tables, and repository names. The versioned
`banana-smasher.bpw-accounting.v1` record keeps the two valid denominators
separate:

- `bpw.comparison` is complete shipped model-weight bytes × 8 divided by the
  canonical base-model logical parameter inventory. This apples-to-apples value
  supplies `publication.label` and is the only BPW used in public model names.
- `bpw.including_auxiliary` divides the same bytes by the base plus separately
  shipped auxiliary-model parameters. It is useful operational accounting, but
  never changes the public quant label.

Packed-container element counts such as Hugging Face `safetensors.total` are
storage metadata and are not a substitute for the canonical logical parameter
inventory. Comparisons should call `require_comparable_bpw(...)`, which rejects
a different inventory SHA or parameter count.

```console
smash bpw \
  --weight-bytes 106623252108 \
  --base-model-parameters 284334567511 \
  --base-parameter-inventory-sha256 98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b
```

For the deterministic FF0731 QTIP2.5 artifact this emits exact comparison BPW
`2.9999377997928467...` and publication label `3.0bpw`.

## Bound repair-checkpoint export

`smash export` can materialize a sealed `banana-smasher-basic-repair-v1` checkpoint directly into a canonical plane source. Every repair input requires its expected SHA-256; the active overlay must bind the exact assignment. The exporter replaces codebook planes by their source-wire hashes (including indexed multi-codebook planes), writes the 235 RMSNorm tensors and 43 attention output gains to `repair/repair_state.safetensors`, binds both repair files in the pack manifest, and fails if any of the 196 checkpoint codebooks is not consumed. When a serving-model root is supplied, the copied RMSNorm tensors replace their deployment tensors and each `output_log_gain` is folded once into the matching deployment weight or scale. The exported config records `repair_application="export-folded-v1"` and `runtime_output_gain=false`; no per-token gain wrapper is required at serving time.

The bound inputs are supplied with `--repair-checkpoint`, `--repair-checkpoint-sha256`, `--active-overlay`, `--active-overlay-sha256`, `--assignment`, `--assignment-sha256`, and `--repair-update` alongside the ordinary export arguments. Run the unchanged `smash validate-pack PACK_ROOT` public verifier after export. The export receipt records the resolved command and all three bound SHA-256 identities.

Repair checkpoint loading is weights-only and requires PyTorch in the export environment. Pack loading and validation retain the lightweight NumPy + safetensors runtime.

## Fixed-D4 exact solve and real bank producer

`smash fixed-d4 solve` exhaustively selects each normalized D4 objective vector's
nearest K2048 or K4096 codeword and calls `persist_fixed_d4_solve` before the
winner arrays leave memory. Its bound JSON config uses schema
`banana-smasher-fixed-d4-exact-solve-v1`, names `tier`, `layer`, `basis_index`,
`basis_sha256`, and `chunk_vectors`, and binds `normalized_vectors`, `scales`,
and `codebook` NPY files by relative `path`, byte count, and SHA-256 under both
`down` and `fused13`.

`smash fixed-d4 prepare-solve` is the real source-model adapter for native
DeepSeek packed-MXFP4 checkpoints. It verifies the exact model-index SHA,
memory-maps the index-bound `I8` expert `w1`/`w2`/`w3` tensors and their
`F8_E8M0` scales, decodes E2M1 values one expert at a time, and writes the six
bound arrays plus `solve.json`. With no `--codebook`, it deterministically uses
the K most frequent source D4 vectors (frequency descending, packed-vector key
ascending for ties); `--codebook PATH.npy` instead binds a supplied finite
`[K,4]` floating codebook. The command reserves 4 GiB by default and refuses
before allocation when its streamed one-layer payload does not fit.

```console
smash fixed-d4 prepare-solve --model /path/to/source-model --tier d4_k2048 --layer 0 --output /path/to/prepared-layer-000 --basis-sha256 "$BASIS_SHA256" --chunk-vectors 256
smash fixed-d4 solve --config /path/to/prepared-layer-000/solve.json --output /path/to/solve-layer-000 --basis-sha256 "$BASIS_SHA256"
smash fixed-d4 materialize --manifest /path/to/solve-layer-000/materialize.json --output /path/to/wire --basis-sha256 "$BASIS_SHA256"
smash export --source-root /path/to/wire --serving-model-root /path/to/source-model --output /path/to/model --model-id d4-k2048 --instance-id d4-k2048-exact --link-mode hardlink
```

Repeat the prepare/solve/materialize transaction for each layer into the same
wire root. The exported model hardlinks both wire planes and same-filesystem
base shards, so construction does not duplicate either large payload; a
cross-device hardlink fails loudly rather than silently consuming copy-sized
capacity. Prepared vectors and solve winners are per-layer intermediates and
can be released only after that layer's materialization receipt is sealed.

The wheel ships `banana_smasher/producer_configs/fixed_d4_vllm.json`. This
built-in producer loads the verified materialized model with public `vllm.LLM`,
runs each bank token window, requests all-vocabulary next-token log
probabilities (`logprobs=-1`), and imports the resulting real 64 rows without a
caller-supplied producer command:

```console
smash anchor materialize-candidate --run-root "$RUN_ROOT" --bank "$BANK_ID" --candidate-id "$CANDIDATE_ID" --model /path/to/model --config /path/to/fixed_d4_vllm.json --basis-sha256 "$BASIS_SHA256"
```

The first sealed model instance has no special framework name. Reusable package, schema, CLI, and documentation names remain `banana-smasher`, `bs-pack`, and `smash`.

## Portable teacher bank and paired evaluation

`smash bank` builds or resumes a content-hashed teacher bank from a declared model runtime, corpus, windows manifest, and optional instrument profile. `smash evaluate` requires both `--candidate` and `--reference` packs and persists the explicit `paired_real_axis` mode. Use `smash bank --help` and `smash evaluate --help` for the complete public arguments.

Members and checkpoints use relative paths, byte counts, SHA-256 identities, and chained completion markers. Verification rejects missing, extra, tampered, unsafe, or unpaired artifacts. The optional `real_axis` object in `bs-pack-v1` binds a pack to its numerical runtime descriptor without changing the required export or repair-pack contract.

These metrics compare declared numerical artifacts. They do not assert causal-context equivalence or same-work language-model equivalence. See `archive/notes/reports/paired-real-axis-api.md` for the portable schemas, durability rules, and interpretation boundary.

## Anchor evaluation

`ANCHOR_EVALUATION.md` documents the generic four-bank API and the complete
`smash anchor` workflow from manifest registration through solver-ready rows.
The wheel includes the bank, raw-score, and aggregate schemas plus a public-safe
four-bank provenance bundle. Training banks may drive fitting and solver rows;
holdout banks fail closed on those uses unless an explicit diagnostic-only
override is supplied.
