# Worked example: routed-only Q2 repair

## General Hugging Face MoE source plan

Before any large build, use the public metadata-only planner. It pins the local
HF source revision and config/index hashes, selects one registered MoE adapter,
reads only safetensors headers, and writes complete routed/native inventories.
The adapter is selected from config capabilities and tensor-name structure; the
caller does not provide model-family code or a routed layer roster.

```python
from banana_smasher import (
    ResidentRepairAPI,
    build_balanced64_token_ledger,
    capture_balanced64_teacher,
    estimate_hf_moe_uniform,
    open_hf_moe_uniform,
    plan_hf_moe_uniform,
    preflight_hf_moe_output_fit,
    recover_balanced64_source_text,
    score_balanced64_pre,
)

plan = plan_hf_moe_uniform(
    "/local/hf-model",
    revision="<immutable-hf-revision>",
    tier="q2",
    scope="routed_only",
    native_rest=True,
    receipt_path="./uniform-plan/UNIFORM_PLAN.json",
)

fit = preflight_hf_moe_output_fit(
    plan,
    output_root="/local/output-filesystem",
    receipt_path="./uniform-plan/OUTPUT_FIT.json",
)
if fit["status"] != "PASS":
    raise RuntimeError("routed-Q2/native-rest output does not fit with reserve")

# On Linux, when the primary filesystem cannot hold the complete serialized
# artifact with its reserve, the public preflight and builder automatically
# consider a local /dev/shm native-byte root. OUTPUT_FIT.json and ARTIFACT.json
# record both roots, both reserves, and exact bytes; callers still provide only
# the model, immutable revision, high-level routed-Q2/native-rest intent, and
# primary output. No remote relay or timed streaming path is introduced.

estimate = estimate_hf_moe_uniform(
    "/local/hf-model",
    revision="<immutable-hf-revision>",
    tier="q2",
    scope="routed_only",
    native_rest=True,
    receipt_path="./uniform-plan/BUILD_ESTIMATE.json",
)
if estimate["projection"]["complete_wall_seconds"] > 6 * 60 * 60:
    raise RuntimeError("complete build projection requires encoder optimization")

built = ResidentRepairAPI.build_uniform(
    "/local/hf-model",
    revision="<immutable-hf-revision>",
    tier="q2",
    scope="routed_only",
    native_rest=True,
    output="/local/output-filesystem/uniform-q2",
)
reopened = open_hf_moe_uniform("/local/output-filesystem/uniform-q2")
assert reopened == built
assert reopened["artifact_root"] == "/local/output-filesystem/uniform-q2"

source_text = recover_balanced64_source_text(
    historical_token_ledger="/local/historical-balanced64-token-ledger.json",
    suite_lock="Evals/configs/<model>-balanced64-v1.json",
    source_tokenizer_model="/local/historical-teacher-model",
    output="/local/eval/recovered-balanced64-source-text.json",
    receipt_path="/local/eval/SOURCE_TEXT_RECOVERY.json",
)
assert source_text["roundtrip_verified_rows"] == 64

ledger = build_balanced64_token_ledger(
    "/local/hf-model",
    revision="<immutable-hf-revision>",
    suite_lock="Evals/configs/<model>-balanced64-v1.json",
    source_manifest="/local/eval/recovered-balanced64-source-text.json",
    output="/local/eval/model-balanced64-token-ledger.json",
    bound_suite_lock="/local/eval/model-balanced64-suite-lock.json",
    receipt_path="/local/eval/TOKEN_LEDGER.json",
)
assert ledger["row_count"] == 64
assert ledger["positions"] == 65536

teacher_canary = capture_balanced64_teacher(
    "/local/hf-model",
    revision="<immutable-hf-revision>",
    suite_lock="/local/eval/model-balanced64-suite-lock.json",
    corpus="/local/eval/model-balanced64-token-ledger.json",
    output="/local/eval/teacher-canary",
    receipt_path="/local/eval/TEACHER_CANARY.json",
    windows=[28],
)
assert teacher_canary["status"] == "PASS_DIAGNOSTIC"
assert teacher_canary["artifact_admissible"] is False

teacher = capture_balanced64_teacher(
    "/local/hf-model",
    revision="<immutable-hf-revision>",
    suite_lock="/local/eval/model-balanced64-suite-lock.json",
    corpus="/local/eval/model-balanced64-token-ledger.json",
    output="/local/eval/teacher",
    receipt_path="/local/eval/TEACHER_CAPTURE.json",
)
pre = score_balanced64_pre(
    "/local/output-filesystem/uniform-q2",
    teacher_capture=teacher,
    suite_lock="/local/eval/model-balanced64-suite-lock.json",
    corpus="/local/eval/model-balanced64-token-ledger.json",
    receipt_path="/local/eval/PRE.json",
)
```

The caller never injects a runtime object or model-family script. A historical
model's integer token IDs are not portable to a different tokenizer. When the
authenticated historical corpus has no raw-text fields,
`recover_balanced64_source_text` first requires its exact file SHA to match the
suite lock, decodes each selected window with the historical source tokenizer,
and re-encodes it. Any token-ID or ordering mismatch is RED. The resulting
source-text manifest and recovery receipt bind the original ledger SHA, source
tokenizer SHA identity, frozen item/window/class roster, and per-item text SHA.
This is deterministic exact-round-trip recovery, not detokenization inference.
The token-ledger builder then accepts that authenticated text, binds the selected
model index and tokenizer identity, writes the model-specific token ledger, and
derives a suite lock whose `source_windows_sha256` is that ledger's exact SHA-256.
Teacher capture and PRE must use that derived lock and ledger together.

The package selects exactly one registered
`banana_smasher.balanced64_runtimes` capability from source config/index
semantics for teacher capture and from the admitted
artifact contract for PRE. Zero or multiple matches fail closed. Teacher capture
must precede candidate scoring and is bound to the model-specific suite lock; a
DeepSeek teacher bank or numeric baseline cannot satisfy a GLM lock.

The receipt has `source.model_index_sha256`, `adapter.id`, sorted
`routed_tensors` and `native_tensors`, exact source bytes and parameters for
both classes, `coverage.gaps=[]`, `coverage.duplicates=[]`, and
`mechanisms.fallback=0`. Planning does not mutate or quantize the source tree.
Only a PASS plan and PASS output-fit receipt may precede the serialized build.
The bounded estimate encodes exactly one representative routed tensor and is
diagnostic only (`artifact_admissible=false`); it reports measured wall/peak
memory and projected complete wall/bytes and never creates an artifact.
`ResidentRepairAPI.build_uniform` dispatches the generic HF builder, Q2-encodes
every adapter-selected routed matrix, copies
each non-routed tensor's exact safetensors data bytes, and atomically seals
`ARTIFACT.json`. `open_hf_moe_uniform` reloads that receipt and re-hashes every
routed wire member and native byte member; opening is mandatory before scoring.

This is the public end-to-end resident path. It uses the published PRE checkpoint
identity and the package-owned U45 recipe; callers do not select teacher paths,
corpus paths, layer splits, microbatch geometry, learning rates, optimizer
dtypes, or held-out gates.

## Inputs

- An **admitted resident artifact directory** produced by `smash resident admit`.
  It contains `identity.json`, `ARTIFACT.json`, the authenticated checkpoint,
  and `production-rails.rank0.json` / `production-rails.rank1.json`.
- Published PRE checkpoint SHA-256:
  `f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70`.
- Two reserved CUDA ranks launched with the artifact-owned distributed settings.
  Both ranks must be admitted from the same scientific contract: the public
  admission binding covers the exact Balanced64 window roster plus the shared
  resident/trainer/roster/corpus digests, and the runtime exchanges that binding
  across the initialized process group before any score. A mixed historical
  rank pair fails nonzero before the expensive Full64 scorer instead of emitting
  a meaningless canary value. The launcher sets `RANK=0` or `RANK=1`. When
  admitted rank-local copies carry different historical rendezvous addresses,
  set the conventional
  `MASTER_ADDR` and `MASTER_PORT` environment variables to the same reachable
  rank-0 endpoint on both ranks; these override only deployment rendezvous, not
  scientific paths or geometry.
- Python 3.11 or newer. Install the runtime dependencies into the interpreter
  that launches the API with `python -m pip install './banana-smasher[solve]'`.
  The admitted API verifies the solve extra and restores its packaged `ninja`
  executable to `PATH` before loading any PyTorch CUDA extension, including from
  a constrained service environment.

Admission is the staging boundary: it copies and re-hashes the explicit
checkpoint, verifies every `authenticated_inputs` row, writes rank configs, and
refuses basis, corpus, teacher, or checkpoint identity drift. On the two Spark
ranks, run under a service scope with `MemoryMax=105G` and
`LimitMEMLOCK=infinity`; these are host safety limits, not recipe knobs. The
service launcher supplies `RANK=0` or `RANK=1`, a shared `MASTER_ADDR` /
`MASTER_PORT` when the admitted copies do not already agree, optionally points
`BANANA_SMASHER_RUN_ROOT` at durable local storage, and invokes the same API
program below; it does not supply corpus, teacher, source-model, layer-split, or
recipe paths.

## Exact API calls

Run the same program on both reserved ranks. `model` is the local admitted
artifact directory on that rank.

```python
from banana_smasher import ResidentRepairAPI

CHECKPOINT_SHA = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
model = "/local/admitted-q2-pre"

api = ResidentRepairAPI.build_uniform(
    model,
    tier="q2",
    checkpoint_sha=CHECKPOINT_SHA,
)
pre = api.score_pre()
training = api.repair_train(updates=45)
post = api.score_post()

print({
    "pre_kld": pre["mean_kld"],
    "updates": training["updates"],
    "post_kld": post["mean_kld"],
})
```

`build_uniform` accepts the short public tier name `q2`, validates that the
admitted composition is routed-only `qtip2_v7` plus native rest, selects the
rank config, and binds the checkpoint SHA for the later calls. The default run
root is `./banana-smasher-resident-run`; set `BANANA_SMASHER_RUN_ROOT` only when
the service needs a different durable filesystem.

`repair_train` owns the validated recipe:

- 16-window broad rotation in four-window pipeline microbatches;
- scorer-exact support-renormalized KL with FP32 pre-backward reduction;
- per-class base learning rates (`luts=1e-2`, `norms=1e-4`, `outputs=1e-2`)
  under the package-owned `0.1` scale;
- Adam with FP64 first/second moments, including after checkpoint reload;
- immutable checkpoint/loss receipts for every update tranche;
- held-out Full64 scoring every four updates, with a hard kill after two
  consecutive flat/rising boundaries.

## Expected result and receipts

The sealed PRE reference is KLD `0.2292069946743951`, Top-1
`56,534/65,536` (the published table row is `0.229392`, `56,533/65,536`,
within the documented rail noise class). The validated U45 lineage is KLD
`0.211277616743619`, Top-1 `56,508/65,536`. A fresh physical run may differ
within the artifact-declared canary tolerance, but it is accepted only when
`post_kld < pre_kld`.

The separated call sequence is fail-closed: `score_post()` writes
`facade/rankN/RESIDENT_ARM_RESULT.json` and raises `ValueError` when KLD does
not improve. Each rank also writes lifecycle, continuation, loss-guard,
held-out-gate, timing, and score-attempt receipts under the run root. A result
is complete only when both rank lifecycles and the post<pre result are PASS.
