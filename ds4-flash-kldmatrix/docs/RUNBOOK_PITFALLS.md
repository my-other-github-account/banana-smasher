# Operational runbook and failure modes

This runbook captures the failures that changed campaign procedure. It is
organized as symptom → cause → required fix → proof. The rules apply to any
single-host GB10 run unless a receipt explicitly says otherwise.

## Preflight: refuse unsafe work

Before a model build, rail, or server launch:

1. Identify the exact base artifact and manifest hash.
2. Verify the corpus, tokenizer, teacher, and scorer hashes.
3. Check that the host has one owner/controller.
4. Reject a second GPU-heavy tenant.
5. Verify free disk and unified-memory headroom.
6. Verify that no stale server core or respawner remains.
7. Write into a new output directory; never mutate a sealed artifact.
8. Use atomic completion sentinels only after hash readback succeeds.

A process being alive is not proof of progress. Use fresh output timestamps,
row-count growth, GPU activity, and a completion ledger.

## 1. Unified memory and hidden EngineCore processes

**Symptom:** a previously fast server reports roughly 0.1–0.2 decode token/s,
while the new server appears healthy and the GPU remains busy.

**Cause:** killing the visible vLLM process can leave an `EngineCore` child
holding roughly a model's worth of unified memory. Loading a second server then
thrashes managed memory.

**Fix:** stop the full process tree, including compute PIDs visible to NVIDIA
tools. Do not assume `pkill -f vllm` is sufficient.

**Proof before relaunch:**

- the compute-app list is empty;
- used unified memory has returned near the host baseline;
- no listener remains on the API port;
- no launcher or supervisor is about to respawn the service.

The campaign used a conservative “well below one model load” memory gate before
relaunch. Record the actual value rather than copying a fixed threshold to a
different machine.

## 2. One host, one controller

**Symptom:** a server reaches readiness, answers one request, then shuts down or
is immediately replaced.

**Cause:** multiple controllers independently manage the same host: a worker,
a shell auto-chain, and a launcher can each decide to restart the endpoint.

**Fix:** assign the entire lifecycle to one controller: cleanup, launch,
readiness, throughput gate, semantic canary, benchmark, and shutdown. Other
agents may observe but must not mutate.

**Proof:** the ownership receipt names one controller and only its process tree
changes the server state.

## 3. Kill the process group, not one child

**Symptom:** a cancelled rail or server returns seconds after its child is
killed.

**Cause:** a parent shell loop, service unit, or timeout wrapper respawns it.

**Fix:** identify and terminate the process group or supervising unit. Preserve
logs and write a tombstone explaining why the old launcher must not resume.

**Proof:** no matching parent, child, unit, or listener reappears over the next
watch interval.

## 4. Never co-tenant two heavy jobs on one GB10

**Symptom:** NVIDIA Xid errors, GSP timeouts, `nvidia-smi` failure, or a required
GPU reset.

**Cause:** a training arm and a large teacher/model build were launched on the
same unified-memory GPU.

**Fix:** enforce one heavy GPU tenant per host. CPU-side preparation is allowed
only when its memory and I/O footprint is bounded and recorded.

**Proof:** the ownership ledger and compute-app list show one heavy job for the
entire measurement interval.

## 5. Context and KV settings are part of the artifact receipt

**Symptom:** a relaunch is slower or fails after increasing maximum context,
maximum sequences, or memory utilization.

**Cause:** a wider context/concurrency configuration reserves a different KV
pool and changes unified-memory pressure. On unified memory, lowering
`gpu_memory_utilization` merely shrinks the KV allocation; it is not a general
out-of-memory cure.

**Fix:** preserve the last known-good context/concurrency values while isolating
performance. Change one variable per run and record KV capacity after startup.
Use `max_num_seqs`, `max_num_batched_tokens`, and `max_model_len` as the primary
memory levers.

**Proof:** startup logs show the expected KV pool and the benchmark receipt
records the exact values.

## 6. A cubin directory is not proof that the candidate uses cubins

**Symptom:** scalar W2/W3 artifacts run quickly, but a mixed learned-VQ artifact
is dramatically slower even though W2/W3 cubins are present.

**Cause:** dispatch depends on the artifact's on-disk format and tier. The
learned-VQ tiers followed a Triton `vq_gemm` path with no matching precompiled
cubin. Inspecting only the scalar cubin directory produced a false lead.

**Fix:** instrument dispatch at the call site. Record the function/kernel name
for every tier family used by the manifest.

**Proof:** a request-level trace ties each candidate tier to the expected kernel
and reports launch count and synchronized time.

## 7. Queue and warmup artifacts are not throughput

**Symptom:** early probes report about 0.1 token/s, but a timed streaming request
later reports materially higher throughput.

**Cause:** readiness, compilation, queued work, and timing boundaries were mixed
into the first estimate.

**Fix:** separate:

1. server readiness;
2. JIT/autotune warmup;
3. a fixed streaming decode request;
4. synchronized token timing;
5. repeated trials under no competing requests.

**Proof:** preserve per-trial timestamps, completion-token counts, TTFT, and the
formula used for decode throughput.

## 8. Learned codebooks travel with their own codes

**Symptom:** replacing codebooks in an existing pack turns a predicted gain into
a several-fold KLD regression.

**Cause:** the repair arm trained codebooks together with its own code
assignments. Only a small fraction of those assignments matched the target
pack. A codebook swap onto foreign codes was not semantics-preserving.

**Fix:** transplant by **re-encoding the weights and refitting scales** under the
new codebook. Pin the exact best checkpoint step in the export receipt.

**Proof:**

- code assignments are regenerated from the intended checkpoint;
- scale refit is recorded;
- exported bytes read back through the pack loader;
- a small paired gate reproduces the repair direction before a 512-window rail.

## 9. Patch the corrected base, not an older base-selected pack

**Symptom:** a repaired export scores much worse than both the prediction and
its unchanged-tier baseline.

**Cause:** the exporter patched an older base pack instead of the corrected
base-plus-delta artifact.

**Fix:** content-address every base and delta. The export receipt must identify
the materialized source manifest, not a directory nickname.

**Proof:** unchanged tensors are byte-identical to the corrected source and the
manifest hash matches the registered input.

## 10. Warm-start ancestry changes provenance labels

**Symptom:** a run trains on a disjoint corpus and is described as
“zero-leakage,” but its initial checkpoint was trained on evaluation windows.

**Cause:** only the latest fine-tuning split was audited; checkpoint ancestry
was omitted.

**Fix:** recursively record every warm start and the data used to produce it.
Use labels such as `disjoint_finetune_corpus=true` and
`inherited_eval_warmstart=true` rather than “contamination-proof.”

**Proof:** the result row names the complete checkpoint ancestry. A true
zero-leakage claim requires clean initialization or a fully disjoint ancestry.

## 11. Gate results do not become full rows

**Symptom:** a favorable 64-window result is copied into the public ladder as a
canonical model result.

**Cause:** the gate and the full rail share a metric name, but not sample size
or artifact coverage.

**Fix:** put the sample size and coverage in the row name. A five-layer overlay
measured on 64 windows remains a five-layer/64-window pilot.

**Proof:** canonical rows have 512 windows, 524,288 positions, complete artifact
coverage, and immutable candidate/score ledgers.

## 12. Predictions are ranking aids, not sealed quality

**Symptom:** a solver predicts a target pass, but the measured candidate misses.

**Cause:** linear damage addition omits interaction terms. In this campaign the
relative optimism increased as predicted KLD decreased; separate export bugs
also created invalid apparent misses.

**Fix:**

- label predictions `PREDICTED`;
- use them to choose what to build;
- measure the selected artifact;
- fit bias only from same-regime, same-instrument pairs;
- never “correct” an invalid export by adjusting the prediction.

**Proof:** result tables retain both predicted and measured values and clearly
identify invalid/quarantined exports.

## 13. Cross-instrument numbers must not be subtracted

Examples of non-identical instruments in this repository include:

- the paired DS4 top-8,192 teacher rail;
- llama.cpp KLD with a Q8 GGUF teacher and re-chunked text;
- serve NLL;
- MMLU choice-loglik;
- generative ToolBench or GPQA;
- kernel microbenchmarks;
- request-level served throughput.

They may provide adjacent context, but a numerical delta is valid only when
teacher, corpus, tokenization, positions, scorer, and artifact are matched.

## 14. Disk pressure can corrupt the experiment before it crashes

**Symptom:** builds stop mid-layer, checkpoints disappear, or a host becomes
unresponsive while large candidates are assembled.

**Cause:** teacher rails, candidate payloads, hard-linked staging copies, and
logs can fill local storage. A directory may look duplicated but still be the
only durable copy.

**Fix:** reserve space before launch, use chunked output, hash mirrors before
deleting, check open handles, and preserve a provenance note for every GC
action.

**Proof:** final receipts include row counts, byte totals, payload hashes, and a
complete `DONE` ledger.

## 15. Fastbuild deltas are not generic overlays

**Symptom:** a complete pack loads, but KLD is several times worse than both its
base and prediction.

**Cause:** changed-row payloads were applied to the wrong base, were applied
again to a pack that already included them, or were omitted while the exporter
still labeled the base as the corrected target.

**Fix:** bind every delta to `(base_manifest, target_manifest)`, materialize it
exactly once, and reject export unless target hash/readback passes. Never rsync
a changed-row delta over a complete pack.

**Proof:** changed and unchanged row inventories reproduce the target manifest;
base and target hashes are both present in the completion receipt.

## 16. A claim file needs an atomic lock and a live process receipt

A JSON ownership note is useful to humans but is not mutual exclusion. Two
watchers can both read “free” and launch concurrently.

Use this sequence for heavy work:

1. acquire a host-local `flock` before checking or writing ownership;
2. read `HOST_CLAIM.json` and verify any named owner is still active;
3. verify compute processes and available memory;
4. atomically write host, owner, purpose, UTC claim time, PID/process-group ID,
   and expected output directory;
5. hold the lock for the launched workload's lifetime, or use a separate
   lifetime lock whose file descriptor remains open in the supervisor;
6. on resume, attach to a matching live PID rather than launching a duplicate;
7. release only after the final receipt or an explicit failure tombstone.

A stale JSON file may be reclaimed only after board state, PID liveness, and
output freshness all agree that the owner is gone.

## 17. Detach with a host-side file launcher, not nested SSH/nohup

Nested `ssh ... "nohup ... &"` chains make it hard to know which shell owns the
file descriptors, process group, exit status, and retry. A disconnected outer
SSH can leave a live child with no durable launch receipt or a dead child that a
watcher relaunches twice.

Instead:

1. copy a versioned launcher script to the target host;
2. have the script acquire the lifetime lock and write PID/start receipts;
3. redirect stdout and stderr to named files on the target filesystem;
4. start it with one remote invocation;
5. verify within 30 seconds that PID, log growth, output freshness, and GPU use
   agree;
6. monitor files/PID from later sessions; never keep a nested SSH session as the
   source of truth.

The launcher must terminate the whole process group and write a tombstone on
failure. A child-only kill is not sufficient.

## 18. Board assignment is not host ownership

**Symptom:** a worker assumes a host is free because a copied task description
or workspace path names a different assignee.

**Cause:** assignee and status in the board database can change while exported
card text, local workspace metadata, or old comments remain on disk.

**Fix:** read current board state immediately before mutation, then reconcile it
with the host claim and live PID. Treat the board database as assignment truth,
the host-local claim as resource intent, and the live process/output receipts
as execution truth. All three must agree.

Before creating or reassigning a card, check that the assignee profile exists on
disk. The dispatcher can silently leave a card idle when its assignee name has
no installed profile; a `ready` status alone is not proof that a worker can
start.

**Proof:** the launch receipt records the current board owner, confirms the
assignee profile was present, and records the claim/PID pair observed at launch
time. Never infer ownership from a directory nickname.

## 19. Predictions rank; paired rails claim

The first 64 evaluation windows were historically about 3.2% easier than the
full 512-window corpus. The campaign used `×1.0316` as a rough gate-to-full KLD
projection. It is a scheduling heuristic, not a sealed result.

Solver additivity bias also depended on quality regime:

- around KLD 0.09–0.10, measured rows were roughly 3.5% worse than prediction;
- around KLD 0.06, optimism reached roughly 10–15%;
- below 0.05, the correction was unknown.

Therefore:

- apply `×1.0316` only to clearly labeled 64-window projections;
- retain the raw gate number and window IDs;
- use predictions to rank build candidates;
- let a paired 512-window rail make the public claim;
- never use a bias factor to “repair” an invalid export.

## 20. Clock discipline

Use UTC ISO-8601 timestamps in claims and receipts, plus monotonic elapsed time
inside launchers. Do not compare wall-clock strings from hosts with unknown
clock skew. Record start, last-progress, completion, and measurement-window
times separately.

Operational decisions should use both:

- process liveness; and
- output freshness measured against the observing host's current time.

An ETA is not evidence of progress. When wall-clock budget matters, spend it on
the shortest gate that can change the decision: integrity check before a rail,
64-window gate before 512 windows, and semantic canary before a full behavioral
benchmark. Skip expensive startup features that are irrelevant to the current
gate, but preserve the production configuration for any number presented as a
production result.

## 21. An environment flag is not dispatch proof

**Symptom:** a fast row is attributed to CUDA graphs or another optimization
because its environment variable was set, but the installed source no longer
contains the wrapper and the flag is inert.

**Cause:** launch configuration was copied across runtime overlays without
checking the effective code and sentinel. This occurred post-cutover: the
installed glue used for a 14-class learned-VQ row did not contain the per-layer
decode-graph wrapper even though `VLLM_MOE_W2_DECODE_GRAPH=1` was present. The
CUDA warp-GEMV path, not that inert graph flag, carried the throughput.

**Fix:** verify the loaded file/build hash, inspect the effective branch, and
require an on-path sentinel or counter for every claimed optimization.

**Proof:** the receipt embeds source/build hashes and a nonzero dispatch
sentinel attributable to the timed requests. If dispatch is not proven, report
the configuration fact but do not assign causal speedup.

## 22. Attribute served quality against a matched served control

**Symptom:** a new kernel fails an offline-referenced NLL threshold and is
declared quality-regressing, even though the unchanged served path also fails
the same threshold.

**Cause:** the comparison combines kernel numerics with KV-cache dtype,
activation quantization, scheduling, prompt history, and sampler/rendering
effects.

**Fix:** run three arms on the same windows and instrument: offline exact,
served bitwise/reference kernel, and served candidate kernel. Use
candidate-versus-served-control to attribute kernel cost; retain
served-control-versus-offline as a separate serve-path delta.

**Proof:** the campaign's 64-window control measured NLL `1.3167553125`
(offline), `1.3344500772447927` (served bitwise), and `1.3344540367` (served
warp). The isolated kernel delta was only `+0.0002967106%`; the larger
`+1.3438157%` delta already existed in the served control. This authorizes the
relative-kernel conclusion, not a bit-identical/offline-equivalent claim.

## 23. Cold and warm throughput are separate rows

**Symptom:** one 4K stream scores below the product gate while a later stream on
the same boot is well above it.

**Cause:** first-request JIT, graph capture, page faults, or payload warming are
inside the cold row. Hiding either row makes the comparison ambiguous.

**Fix:** adopt an effective-environment verifier, generate at least two warmups,
then measure. Preserve any cold diagnostic separately. The post-cutover golden
serve rule requires the verifier output and environment dump inside every TPS
receipt.

**Proof:** record cold and warm request identities, output counts, windowed
rates, loaded hashes, and warmup count. One campaign product receipt disclosed
9.86 token/s on the cold first 4K and 13.913 token/s on the comparable warm 4K;
only the latter was used against the warm precedent.

## 24. Public-release hygiene

Before committing public documentation:

- remove personal names and usernames;
- remove private IP addresses and endpoints;
- remove absolute home directories;
- remove internal scheduler/task identifiers;
- redact credentials rather than replacing them with examples that look real;
- keep repository-relative receipt paths and cryptographic hashes;
- state when a required artifact is not included in the repository.

A sanitized explanation must not silently upgrade the evidence. If only an
internal receipt exists, say so and avoid pretending the public checkout can
recompute it unaided.
