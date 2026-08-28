# Serving learned-VQ DS4 artifacts

This document reconstructs the 2026-07-17 serving/performance sequence and the
major post-cutover seals through 2026-07-18 10:00 PDT from measured throughput
receipts and ledgered operations. It distinguishes
end-to-end served throughput from kernel microbenchmarks and model-quality
measurements; operational controls without raw receipts are labeled as such.

## Foundation and acknowledgement

The serving work builds on
[`kacper-daftcode/vllm-Moet`](https://github.com/kacper-daftcode/vllm-Moet).
That project supplied the baseline W2 recipe and the practical prepacked expert
plane/kernel path used by this campaign. The quantization-quality program in
this repository extends the artifact menu on top of that serving foundation;
it should not be presented as an independent serving stack.

The stock W2 path and the learned-VQ path are materially different formats.
A fast scalar W2/W3 serve is not evidence that a learned-VQ tier dispatched to
the same kernel.

## Metric classes used here

- **Served decode throughput:** completion tokens after the first token divided
  by the corresponding timed decode interval from a live request. This is the
  class of the `2.14`, `2.99`, `6.59`, `14.13`, and `13.91` token/s
  observations below.
- **Kernel microbenchmark:** synchronized milliseconds for one named kernel and
  shape. No public standalone old/new kernel timing table was available for
  this sequence at cutover. A synchronized real-shape table did land later for
  the CUDA warp candidate and is kept separate from the served rows.
- **KLD quality:** teacher-forced paired quality. The serving speed experiments
  did not replace or re-seal the KLD rows.
- **Model footprint:** bytes/bpw. It does not imply throughput.

## Starting point

The learned-VQ IQ3 package had already passed a quality bar at approximately
101.95 GB, but its learned codebook tiers used a separate `vq_gemm` execution
path. The serving goal was to make that exact format usable without recoding it
into a different quantization artifact.

An exact learned-VQ-to-scalar conversion was rejected for two reasons:

1. a same-size scalar grid cannot represent arbitrary learned codebook values
   exactly, so recoding changes the artifact;
2. exact expansion to wider values would multiply the footprint and memory
   bandwidth, defeating the package's size/speed rationale.

The correct path was therefore to optimize the learned-VQ decode kernel while
preserving packed indices, codebooks, scales, and output semantics.

## Timeline and measured ladder

### 1. The first 0.1–0.2 token/s readings were not benchmark rows

Early probes mixed readiness, JIT/warmup, server lifecycle, and incompatible
memory states. The captured 0.1 token/s log itself reported `Waiting=0`, so
queueing is **not** evidenced as the cause of that particular sample. It also
showed an automatically sized 97.59 GiB KV cache, which left little margin for
the learned-VQ blobs.

One separate, concrete failure was an orphan `EngineCore` process retaining
roughly one model load of unified memory after the visible server was killed.
Loading a second server caused severe managed-memory thrash. Killing the orphan
returned memory near baseline and removed that confound.

The low rate persisted after cleanup. The orphan was a real incident, but not
the final learned-VQ performance root cause.

### 2. Context/KV changes removed a confound, not the hot-path cost

The later performance launches used 8,192 maximum context, two maximum
sequences, memory utilization 0.90, a manual 3 GiB KV reservation, and an
explicit warmup/read of 95.76 GB of learned-VQ blobs. That made the memory state
more controlled and prevented the automatic 97.59 GiB KV reservation from
dominating the run.

The narrower context/concurrency configuration did not recover scalar-W2 speed.
Context, sequence count, batched tokens, and KV capacity belong in every
receipt, but the persistent bottleneck was the learned-VQ execution path.

### 3. The “wrong W3 cubin directory” hypothesis was a false lead

The mixed artifact contained W3-family tier support, and an initially
configured cubin directory did not contain the expected W3 cubins. Correcting
the path was necessary for scalar W3 dispatch, but this candidate selected zero
scalar W2/W3 rows.

Instrumentation showed its selected tiers dispatched through
`moe_w2_cubit._moe_w2_forward_timed` into Triton
`moe_vq_triton.vq_gemm`, with no matching precompiled scalar cubin. The scalar
W2/W3/W4 cubin families loaded successfully; they were simply off-path for this
artifact.

### 4. First honest served result: 2.1439 token/s

A timed streaming request produced:

- decode throughput: **2.1439 token/s**;
- TTFT: **4.7 s**;
- sample count: one 16-token completion.

This was the first request-level number considered diagnostically honest. The
small sample count keeps it a diagnostic baseline, not a promotion-quality
benchmark.

### 5. Fresh d4-fast baseline: 2.9894 token/s

A fresh run with the d4-fast path produced **2.9894 token/s**. This was the
correct baseline for the next no-sync A/B, rather than the earlier generic
2.1439 result.

### 6. No-sync iteration: 2.9691 token/s (neutral)

Removing per-operation synchronization measured **2.9691 token/s**, effectively
neutral and slightly below the fresh 2.9894 baseline. The large apparent jump
from 2.1439 to about 2.99 came from changing the baseline/path and run state;
it must not be attributed to no-sync.

These remain end-to-end served results, not standalone kernel microbenchmarks.

### 7. Group expert work and cap pairs: 6.5891 token/s

The dominant issue was launch granularity: the direct heterogeneous VQ path
issued many small operations across layers, experts, and projections. Grouping
d4/d8 expert work and capping pairs reduced launch overhead and produced
**6.5891 token/s** served decode.

This was the best measured learned-VQ serve result at the historical
documentation cutover.

### 8. CUDA graphs regressed to 5.3044 token/s

A subsequent CUDA-graph experiment interacted poorly with the grouped path and
measured **5.3044 token/s**, a regression from 6.5891. It was rejected rather
than promoted because “more optimization machinery” is not evidence of
improvement.

## Outcome at cutover

| Runtime state | Served decode | Classification | Verdict |
| --- | ---: | --- | --- |
| Generic direct path | 2.1439 token/s | request-level served diagnostic, one 16-token sample | Honest but low-N; too slow |
| Fresh d4-fast baseline | 2.9894 token/s | request-level served measurement | Baseline for no-sync |
| No-sync iteration | 2.9691 token/s | request-level served measurement | Neutral/slight regression |
| Grouped d4/d8 + pair cap | **6.5891 token/s** | request-level served measurement | Best measured state; below gate |
| Grouped + CUDA graphs | 5.3044 token/s | request-level served measurement | Regression; rejected |

The product/behavioral-evaluation gate was **≥ 10 token/s decode**. At the
cutover, the best measured result, 6.5891 token/s, did not pass and the full
local 69×3 ToolBench row was held back. That sentence is a historical status,
not the final campaign result; the gate passed later as documented below.

### Immutable cutover throughput receipts

The raw JSON receipts contain a private endpoint and are not mirrored in the
public checkout. The sanitized identity map is:

| Runtime state | Receipt filename | SHA-256 | Trials × completion cap |
| --- | --- | --- | --- |
| Generic direct path | `live_tps_generic_kv3.json` | `d8701fb8edd679d13170709df5ab61e5f160f95a55371a9f5364c2ee3970fa1a` | 1 × 16; 1 warmup × 8 |
| Fresh d4-fast | `live_tps_fast_kv3_fresh.json` | `948e2364082c310b6375b1ff91c46fc0165ba8f5bfcc5ba63ca34e9ddba18311` | 3 × 64; 1 warmup × 8 |
| No-sync | `live_tps_nosync_iter1.json` | `d0552330d3e4d87ea2551222ee9a225101ca709376e831150b439f816dca8541` | 3 × 64; 1 warmup × 8 |
| Grouped + pair cap | `live_tps_grouped_iter2.json` | `7f09ad975106c67d7d3ed81c8573c6449a479f104800c0ccc3cadb57aa5cf100` | 3 × 64; 1 warmup × 8 |
| Grouped + graph experiment | `live_tps_graph_iter3.json` | `0b03dc4db3dda83ea5de0037d0178dc9bfa37aac5b39a5e21482a242df44e5e2` | 3 × 64; 1 warmup × 8 |

These receipts time `(completion_tokens - 1) / decode_seconds_after_first`.
They are not all promotion-grade: the first row is N=1/16 tokens, and the
three-run rows used only one short warmup. They are sufficient to reconstruct
the iteration ladder, not to imply production confidence intervals.

## Post-cutover warp-GEMV result

An all-lanes CUDA warp-GEMV candidate replaced the small learned-VQ decode
operation while leaving packed indices, codebooks, scales, and artifact files
unchanged. Independent real-shape validation measured the combined VQ work at
`2.512816 → 0.358832 ms/layer` (`7.003×`), passed 1,000/1,000 packed-decoder
properties, and passed all 10 d4/d8 × bits-8–12 synthetic configurations. The
real full-K comparison was not bit-identical because FP32 scalar reduction
order differed from the tensor-core reference; the measured envelope was
maximum absolute error `0.015625`, cosine `1.0`, and equal fraction `0.998`.

The package identities were:

- source SHA-256 `7e13796973689f87e354494a5fb5fe6434bddce352aaaf9eb60cb20b59032f9f`;
- wheel SHA-256 `33fa891a78728932f0342273982dd97966e9e85b6d94db775e3e33d61a3fafdf`;
- installed extension SHA-256 `ba170dec6cff8d92327713dded855853184a63352a18be4108c491e95ff36843`.

On one uncontended server, with no MTP/speculative decoding, the candidate
measured:

| Request shape | Candidate | Same-session bitwise control | Detail |
| --- | ---: | ---: | --- |
| one 4,096-token stream | **14.134 token/s** | 6.99 token/s | 2.02×; windows 0–256 `10.73`, 1024–1280 `14.55`, 3840–4096 `15.01`; no depth sag |
| 5 × 64-token requests | **14.93 token/s median** | not reported in the same receipt | candidate range 14.87–14.94 |

The full-4K claim receipt is retained in the internal campaign archive as
`CLAIM_sustained4k.json`. Its private mission path is intentionally omitted
from this publication. A behavior canary passed, but the candidate's token
stream diverged from the bitwise control, so a quality attribution control was
required before product use.

### Served-quality attribution

The matched 64-window / 65,536-position control measured:

| Arm | NLL |
| --- | ---: |
| Offline exact | `1.3167553125` |
| Served bitwise VQ | `1.3344500772447927` |
| Served CUDA warp | `1.3344540367` |

The isolated warp-versus-bitwise delta was
`+0.00029671062820376114%`, while the bitwise serve path itself differed from
offline by `+1.3438157094803975%`. Receipt `ATTRIBUTION_CONTROL.json` has
SHA-256 `05d46ac204f32df230357ddfe1d3a1f67b1a6dd7b6080bb69c99621201fa41b0`.
The honest conclusion is “warp is effectively quality-free relative to the
matched served control,” not “served output is bit-identical to offline.”

## Post-cutover product-overlay result

The V4-step32 product overlay, on the same raw-AR/no-MTP operating class,
measured:

| Request shape | Result | Receipt identity |
| --- | ---: | --- |
| 5 × 64, concurrency 1 | **13.964 token/s median**; runs 13.992, 13.999, 13.964, 13.940, 13.962 | `P1_5X64_RECEIPT.json`, preserved SHA-256 prefix `79b03aba…` |
| warm 4,096-token stream | **13.913 token/s**; windows 13.817, 13.802, 13.915 | `P2_SUSTAINED4K_RECEIPT.json`, preserved SHA-256 prefix `44c126bd…` |

The receipt also discloses a cold first-4K aggregate of 9.86 token/s, including
first-request JIT/paging; the warm row is the like-for-like comparison to the
14.13 precedent. The earlier `9.03` overlay result used a different runtime
overlay and a heavier full checkpoint, so it was a launch/path mismatch rather
than a measured property of the product artifact.

One attribution correction matters: on the installed glue used for the
13.96/13.91 rows, `VLLM_MOE_W2_DECODE_GRAPH=1` was inert because that file no
longer contained the per-layer graph wrapper. The CUDA warp-GEMV path carried
the 14-class throughput. Do not describe those rows as a graph speedup merely
because the environment variable was present.

Passing the 10 token/s gate authorized behavioral evaluation work; it did not
make any ToolBench score comparable across evaluator versions. See
`docs/EVAL_PARITY.md`.

## Clean launch checklist

### 1. Enforce one controller

One owner must perform cleanup, launch, readiness, timing, semantic canary,
benchmark, and shutdown. Do not run an auto-chain and a separate worker against
the same host.

### 2. Remove the complete old process tree

Before launch, verify:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

The list should be empty. Also verify that no API listener, EngineCore child,
service unit, shell loop, or timeout wrapper remains.

### 3. Verify unified-memory baseline

Record host memory before loading the model. A clean launch should start from a
small baseline relative to the model, not from a stale model-sized allocation.
In this GB10 campaign the relaunch gate was **residual used memory below 12
GiB** after cleanup; the exact threshold is hardware-specific, but the check is
mandatory.

The ledger records this gate and the orphan cleanup, but the publication set
does not contain the raw `nvidia-smi`, `/proc/meminfo`, or `drop_caches` output.
Treat the steps below as the required reproduction protocol, not as a claim
that those missing raw receipts were independently revalidated here.

`pkill -f vllm` can leave `EngineCore` alive. Kill the compute PID reported by
NVIDIA tooling, then prove the compute-app list is empty:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
kill -TERM "$ENGINECORE_PID"
kill -KILL "$ENGINECORE_PID"   # only if TERM did not remove it
```

### 4. Control page-cache state

For cold-load timing, first stop every model process, flush filesystem writes,
and drop the page cache only under the host owner's control:

```bash
sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
awk '/MemTotal:/{t=$2} /MemAvailable:/{a=$2} \
     END{printf "residual_used_gib=%.2f\n", (t-a)/1048576}' /proc/meminfo
```

Record the command, time, and before/after memory. For warm serving trials, read
the exact candidate payload set once and verify its byte count before the timed
request. The campaign's later VQ trial explicitly warmed 95.76 GB of VQ blobs.

Do not drop caches while another workload owns the host, and do not compare a
cold artifact load with a warm one. Cache state is part of the receipt.

### 5. Pin memory/scheduler settings

The repaired-IQ3 diagnostic launch first returned to the bounded shape:

```text
max_model_len=8192
max_num_seqs=2
gpu_memory_utilization=0.90
```

The scalar W3 runtime directory was
`VLLM_MOE_W3_CUBIT_DIR=$HOME/ds4w3/cubins_e43` (some wrappers expose the shorter
`W3_CUBIT_DIR` alias). Record the effective environment, not only the launch
snippet. This directory fixes scalar W3 dispatch only; learned IQ3-VQ tiers
still call `moe_vq_triton.vq_gemm` and have no cubin implementation.

Record at least:

```text
max_model_len
max_num_seqs
max_num_batched_tokens
gpu_memory_utilization
KV-cache dtype and capacity
enforce-eager / graph mode
prefix caching
speculative decoding
```

On unified memory, lowering `gpu_memory_utilization` only reduces the KV pool;
it is not a universal OOM fix.

### 6. Pin semantic serving settings

For ToolBench comparison, include:

```bash
--tokenizer-mode deepseek_v4 \
--generation-config vllm \
--reasoning-parser deepseek_v4 \
--default-chat-template-kwargs '{"enable_thinking":true}' \
--enable-auto-tool-choice \
--tool-call-parser deepseek_v4
```

See `docs/EVAL_PARITY.md` before running a behavioral row.

### 7. Prove dispatch

For every tier family in the manifest, capture the actual kernel/function path.
A directory containing cubins is not dispatch proof.

### 8. Verify the effective environment, warm twice, then measure

Separate server readiness and compilation/autotune from the timed trials. The
post-cutover “golden serve” rule requires a verifier pass against the effective
environment and loaded hashes, at least two generated warmups, and a receipt
that embeds the verifier output plus environment dump. Use a fixed
prompt/output shape, one client, no competing requests, and repeated streaming
trials. A launch flag without a dispatch sentinel or code-path check is not
evidence that the feature executed.

## Throughput receipt schema

A public served-throughput row should include:

```json
{
  "status": "MEASURED_SERVED",
  "artifact_manifest_sha256": "...",
  "server_commit": "...",
  "kernel_commit": "...",
  "launch_flags_sha256": "...",
  "context": 0,
  "max_num_seqs": 0,
  "max_num_batched_tokens": 0,
  "kv_cache_dtype": "...",
  "prompt_tokens": 0,
  "completion_tokens": 0,
  "ttft_seconds": 0.0,
  "decode_seconds": 0.0,
  "decode_tokens_per_second": 0.0,
  "trials": [],
  "kernel_dispatch": ["..."],
  "no_competing_clients": true
}
```

For a kernel microbenchmark, use a separate receipt containing the exact tensor
shape, dtype, codebook geometry, synchronization points, warmup/iteration
counts, and old/new kernel milliseconds. Do not insert token/s into that table
unless it came from a server request.

## Packaging and upstreamability

At cutover, the optimization work was an experiment against a campaign serving
checkout. Post-cutover, the CUDA warp candidate acquired a buildable wheel,
source hash, independent validation, opt-in dispatch, and real served A/B. That
makes it a credible packaging candidate, but the audited evidence contains no
merged upstream change or published package. Upstream completion still
requires:

- a stable public ABI for packed indices, codebooks, scales, pair limits, and
  stream semantics;
- adversarial correctness against the reference implementation;
- synchronized real-shape microbenchmarks;
- integration tests in the target serving/runtime package;
- build metadata and licensing suitable for upstream distribution;
- a real served-throughput A/B using the exact packaged kernel.

Quality receipts and local runtime patches are not substitutes for a reviewed,
published upstream artifact with CI on the target ABI.

The exact target-path audit found no learned weight-VQ dispatch in the target
Atlas model/runtime path. Porting even the later CUDA warp candidate requires a
fresh target-shape correctness microbenchmark and served A/B on spark-6; none
of the DS4 token/s rows can be transplanted as an Atlas throughput claim. Atlas
success remains measured spark-6 throughput plus packageable upstream kernels,
not documentation completeness or a fast result on this separate stack.
