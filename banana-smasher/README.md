# banana-smasher

`banana-smasher` is the reusable, fail-closed `bs-pack v1` build and validation toolchain. `PACK_FORMAT.md` is the versioned pack contract: plane layout, per-layer metadata, `config.json` auto-detection keys, complete byte-count/SHA-256 manifest, and rejection rules.

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
    resolve_backpack_family_provider,
)

providers = builtin_backpack_family_providers()
assert set(providers) == {
    "native-mxfp4", "qtip@2.00", "qtip@2.50", "qtip@3.00",
    "d4-k2048", "d4-k4096",
}
qtip15 = resolve_backpack_family_provider(
    {"kind": "qtip_ring", "bpw": 1.5}
)
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

## Three-command release path

```bash
smash export --source-root /path/to/materialized-quant-source --runtime-floor-bytes "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}" --serving-model-root /path/to/base-model --output /model --model-id MODEL --instance-id PACK_INSTANCE --link-mode copy
smash verify /model
vllm serve /model
```

The P1016 export requires `RUNTIME_FLOOR_BYTES` from the caller's measured runtime receipt. The example intentionally has no default and never guesses this residency value. The first command builds `/model`, merges the full base-model `config.json` with the pack-owned `quantization_config`, copies `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json`, and writes `BANANA_PACK_MANIFEST.json` last after self-verification. `--serving-model-root` must point at a serveable model directory whose config has a non-empty `architectures` list; the exporter rejects missing tokenizer metadata rather than creating a quant-only pack that vLLM cannot boot. The second command fails closed on missing or extra files, byte-count or SHA-256 drift, schema/version mismatch, invalid metadata, and incompatible config auto-detection keys. The third is the stock vLLM command; no banana-smasher launcher or environment-only format selection is required.

To repair serving metadata in an already validated pack without touching tensor files, rerun the same verb in metadata-only mode by adding `--refresh-metadata` to that export command (and keep the same `--serving-model-root`).

This preserves the existing pack `quantization_config`, rewrites only the four serving metadata files plus their manifest rows/provenance, revalidates the pack, and reports `tensor_payloads_rewritten: false`.

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

These metrics compare declared numerical artifacts. They do not assert causal-context equivalence or same-work language-model equivalence. See `notes/reports/paired-real-axis-api.md` for the portable schemas, durability rules, and interpretation boundary.

## Anchor evaluation

`ANCHOR_EVALUATION.md` documents the generic four-bank API and the complete
`smash anchor` workflow from manifest registration through solver-ready rows.
The wheel includes the bank, raw-score, and aggregate schemas plus a public-safe
four-bank provenance bundle. Training banks may drive fitting and solver rows;
holdout banks fail closed on those uses unless an explicit diagnostic-only
override is supplied.
