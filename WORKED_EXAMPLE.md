# Worked example: routed-only Q2 repair

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
  The launcher sets only `RANK=0` or `RANK=1`; scientific paths and geometry come
  from the corresponding rank config.
- Python 3.11 or newer. Install with `python -m pip install ./banana-smasher`.

Admission is the staging boundary: it copies and re-hashes the explicit
checkpoint, verifies every `authenticated_inputs` row, writes rank configs, and
refuses basis, corpus, teacher, or checkpoint identity drift. On the two Spark
ranks, run under a service scope with `MemoryMax=105G` and
`LimitMEMLOCK=infinity`; these are host safety limits, not recipe knobs.

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
