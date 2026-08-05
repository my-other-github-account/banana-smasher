# banana-smasher

`banana-smasher` is the reusable, fail-closed `bs-pack v1` build and validation toolchain. `PACK_FORMAT.md` is the versioned pack contract: plane layout, per-layer metadata, `config.json` auto-detection keys, complete byte-count/SHA-256 manifest, and rejection rules.

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

`smash export` can materialize a sealed `banana-smasher-basic-repair-v1` checkpoint directly into a canonical plane source. Every repair input requires its expected SHA-256; the active overlay must bind the exact assignment. The exporter replaces codebook planes by their source-wire hashes (including indexed multi-codebook planes), writes the 235 RMSNorm tensors and 43 attention output gains to `repair/repair_state.safetensors`, binds both repair files in the pack manifest, and fails if any of the 196 checkpoint codebooks is not consumed.

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

```bash
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

```bash
smash anchor materialize-candidate --run-root "$RUN_ROOT" --bank "$BANK_ID" --candidate-id "$CANDIDATE_ID" --model /path/to/model --config /path/to/fixed_d4_vllm.json --basis-sha256 "$BASIS_SHA256"
```

For authentic Anchor64, use the shipped
`producer_configs/fixed_d4_offline_layerwise.json` with a top-8192 teacher
manifest. `schema/anchor-teacher-sidecars-v1.schema.json` binds ordered windows,
`t8192`-compatible `idx` int32 and `logprob` fp16 `[T,8192]` tensors, hashes,
and bank/teacher identities. The layerwise producer emits
`schema/anchor-candidate-sidecars-v1.schema.json` with `q_lp_at_ref` fp16
`[T,8192]` full-softmax logprob and full-vocabulary `q_argmax` int32 `[T]` in
`q8192`-compatible PyTorch sidecars. Each completed window is hash-bound before
the next window runs, so reruns skip it. Teacher rows require unique token IDs
ordered by descending teacher logprob. Candidate position count may be shorter
than the teacher position count; scoring uses the historical minimum of teacher
positions, candidate positions, and the fixed 1024-position cutoff.

```python
from banana_smasher import produce_fixed_d4_layerwise_logits, score_anchor_sidecars

receipt = produce_fixed_d4_layerwise_logits(
    "/path/to/verified-pack",
    "/path/to/fixed_d4_offline_layerwise.json",
    "/path/to/balanced64.jsonl",
    "/path/to/q8192.json",
    basis_sha256=BASIS_SHA256,
)
metrics = score_anchor_sidecars(
    "/path/to/teacher_support.json", "/path/to/q8192.json"
)
```

`materialize_candidate_producer(...)` accepts the same binary sidecar manifest
through the ordinary public materialization route, scores the validated
sidecars, and returns bound teacher/candidate/score descriptors under
`quality_rail`. To rescore a
completed physical layerwise bank against a replacement teacher without
replaying any transformer layer, call the public terminal-only entrypoint:

```python
from banana_smasher import rescore_fixed_d4_layerwise_terminal

receipt = rescore_fixed_d4_layerwise_terminal(
    "/path/to/verified-pack",
    "/path/to/original-producer-config.json",
    "/path/to/balanced64.jsonl",
    "/path/to/original-output.layerwise/STATE.json",
    "/path/to/new-teacher-support.json",
    "/path/to/new-q8192.json",
    basis_sha256=BASIS_SHA256,
    terminal_runtime_adapter={
        "module": "banana_smasher.hf_deepseek_v4_d4_adapter",
        "sha256": INSTALLED_ADAPTER_SHA256,
        "class": "DeepseekV4D4Runtime",
        "api_version": 1,
    },
)
assert receipt["window_layer_forwards"] == 0
```

The original config authenticates model, bank, basis, runtime adapter, layers,
positions, and the completed state. Only the replacement teacher and output
identities change. `terminal_runtime_adapter` separately binds the currently
installed adapter used for terminal scoring, so a completed state remains bound
to its original adapter without requiring source-state or config mutation. The
receipt binds both adapter hashes, reports exact global/per-window `kld_sum` and
integer `top1_matches`, and writes a digest-bound score JSON. The built-in
DeepSeek adapter discovers the bound D4 subtier per layer and streams K2048 or
K4096 experts directly from `bs-pack` members when compatibility
`vq3u_layer_*.pt` files are not present; it does not require generating those
large compatibility files.

The scorer renormalizes teacher and candidate logprob on the declared support,
matching the historical Anchor64 KLD convention, while top-1 compares the
candidate full-vocabulary argmax with `idx[:, 0]`. Width-2 JSONL remains a
backward-compatible backend smoke path only. A width-2 hardware run is not an
authentic top-8192 Anchor64 result and must not be published as one.

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
