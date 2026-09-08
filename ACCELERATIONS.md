# Runtime accelerations

## Fused K3 eager conformance: matched cold-process panel (2026-09-07)

The public `packed_decode_execution='eager'` opt-in was extended to four
real L005 E001--E004 fused13 K3 cells on dedicated Spark-6, pinned to
`b2d30432e5de596261cb6d58ae80d33ee496ce4c`. Both arms retain unitwise
factorization and eight Viterbi warps; only conformance execution differs.
Four reversed fresh-process arms each use private canonical/Triton/Inductor/
CUDA/XDG caches and cold/warm companions. Source files and OS caches are shared,
not globally cold. All 32 new builds passed canonical conformance.

Paired four-cell build wall was 48.407624 / 44.520410 seconds compiled versus
29.445928 / 26.748347 seconds eager: 1.643950x / 1.664417x cold-process gain.
Warm wall was 8.183567 / 8.230978 versus 8.298362 / 8.114429 seconds:
0.986167x / 1.014363x, with no demonstrated warm throughput improvement.
Import/config costs were 2.540661 / 1.766471 seconds compiled versus
1.741986 / 1.702689 seconds eager, outside the build denominator.
Peak GPU allocation was identical in both arms: 2,114,731,008 bytes cold and
2,116,832,768 bytes warm. Reserved peaks were 3,321,888,768 / 3,323,985,920 bytes.
The unchanged 32-GiB estimate plus 4-GiB MemAvailable reserve passed.

A separate canonical source/decode validator took 120.697850 seconds and passed
16/16 candidate reconstruction checks. Limits were frozen from two baseline
repeats before candidate adjudication: worse baseline NMSE plus the maximum of
five times repeat variation and 1e-6 times worse baseline NMSE. Baseline repeats
were identical; all candidate decoded max-absolute differences to the matched
compiled baseline were zero. This is not a new held-out/full-model equivalence
claim. The older sparse held-out receipts survive locally, but their volatile
incumbent payloads disappeared after a host reboot; consumer-qualified durable
reference linkage and production owner adoption remain pending.

No sealed build or calibration was replayed during recovery. Four missing K3
configs were restored using the existing authenticated binder from durable
source configs, codebook references and Hessian manifests (zero solves/captures).
New durable host roots are
`/home/dnola/missions/t_ebcba52e_glm_fused_eager_run8341` and
`/home/dnola/missions/t_ebcba52e_quality_fused_eager_run8341`.
Task-local `GLM_FUSED_EAGER_{TERMINAL,SUMMARY,QUALITY}_run8341.json` contains
full timings and the independent checks. No production promotion is implied.


## Exact sparse consumer staging (2026-09-07, unpromoted)

`ArtifactTensorStore` accepts an explicit `native_payload_reads: true` on the
admitted subject. This reads exact exported native bytes with path, source/hash,
size and dtype checks instead of staging whole original source shards. The
default remains source-backed; selected corrupt native payloads never fall back.
The focused suite reports 46 passed, one skipped on the Mac (transformers absent).
This is consumer infrastructure, not a measured producer speed improvement.

The dedicated GLM diagnostic resumes three immutable L004 windows (ordinals
0/1/3) and substitutes four sealed L005 fused experts. Baseline singleton repeats
and grouped candidate share the fixed K2 prefix and suffix. It is explicitly a
paired sparse diagnostic, not native-prefix or uniform-K3 equivalence evidence.
The first physical L005 materialization took 269.6501306 seconds, with 1759 payload
reads and zero source-model reads. Downstream output-KLD remains pending; no
candidate or production default has been promoted. Cold consumer setup and
validation wall are not hidden inside the producer speedup denominator.

The first mixed-K2/K3 replacement consumer then stalled inside Inductor/SymPy
symbolic stride simplification and was stopped with no replacement/forward
sealed. Canonical `73e04f9fd8d02a7e70bc4edd59595ad626a08d3c` keeps
`torch.compile(dynamic=False)` for the packed decoder, without changing its
arithmetic. The physical mixed-tier canary subsequently completed in
1.693785015 seconds and produced finite 4096-by-4096 K3 weights. This is a
mechanical canary, not an independent numerical or held-out quality gate.
The recovered L005 materialization took 155.398136120 seconds; all twelve
replacement bindings and nine L005 arm/window forwards sealed with zero
source-model reads. These sequential cold/warm consumer costs are not a
matched producer speed comparison.

The diagnostic now checkpoints every arm/window/layer. Its unchanged
40-GiB estimated-peak plus 4-GiB reserve gate stopped before L010 allocation
at MemAvailable 47,160,709,120 bytes. All L005--L009 forwards were preserved.
After the source owner reclaimed only three hash-verified duplicate input
shards (16,089,895,160 bytes), the next attempt resumed all nine slots from
L009 and allocated L010 with MemAvailable 66,324,426,752 bytes. No sealed
forward, baseline build, or source scorer was replayed; no resource gate was
weakened.

The suffix has now sealed through L044 and readout. An independent CPU verifier
recomputed binary64 KLD from all nine hash-authenticated readouts and the exact
teacher-selected 8192-token supports (1024 positions/window). This is conditional
teacher-support KLD, not full-vocabulary KL. Singleton repeats were identical;
the predeclared repeat-derived per-window and pooled limits were each 1e-7.
Grouped-minus-singleton deltas for windows 28, 56, and 71 were respectively
-0.014696289583460309, -0.001827300157524414, and +0.0020095746115649515.
The pooled delta improved by -0.004838005043139924, but window 71 failed the
frozen per-window gate. Therefore this grouped candidate is NOT promoted.
This is a held-out sparse diagnostic failure, not a byte-equality rejection or
full-model/uniform-K3 conclusion; do not weaken tolerances after observing it.
The retained independent receipt is
`/dev/shm/t_ebcba52e_glm_suffix_a5/INDEPENDENT_VERIFY_run8290.json` on Spark-6,
with verifier SHA256
`4707a831d4f0aea9d49520a3e189e1967e1173a6fd56af156cbbd27013f83240`.
Producer timing remains separate from this full suffix validation cost.
The next causal direction is to isolate batched linear-algebra numerical changes
from execution-scheduling gains, preserving this failed candidate and all sealed
frontiers rather than replaying them or fitting to these frozen windows.

## Unitwise factorization with resident LDLQ batching (experimental)

The same-input four-cell GLM causal profile isolated a numerical batch-axis
change in block-LDL: max_abs 0.0012226838152855635, 33,338,796 changed entries
among 67,108,864. With exactly the same lower factors and transformed weights,
batched versus singleton LDLQ returned identical quantized values and states for
all four cells. Profile repetitions are instrumentation, never speed evidence.

The opt-in public batch config `block_ldl_unitwise: true` retains singleton
block-LDL factorization while keeping cross-unit LDLQ and resident staging.
Default remains false; mixed/nonboolean settings fail admission. Build receipts
record the selected factorization axis. CPU tests cover actual factorization,
input preservation, configuration admission and controller/builder wiring.
This isolates the measured mechanism rather than changing frozen output gates;
physical speed, paired reconstruction and held-out acceptance are still required.

The initial factor-only physical experiment (`dd478bc0`) produced warm four-cell
singleton 10.6650465/10.7929246 s versus grouped unitwise10.7913408/9.26281808 s
(0.988297x/1.165188x), not a repeatably substantial improvement. All24 builds
passed packed decode conformance. E001--003 reconstructed identically in both
repeats, but E004 differed (max_abs0.019915267825126648). A subsequent same-input
canonical regularization profile found only E004 differed across the batch axis:
13,146 Hessian entries, max_abs2.9802322387695312e-8; E001--003 were identical.
The opt-in now also preserves singleton regularization before singleton LDL;
this causal correction remains unpromoted until a fresh physical gate.

The corrected physical comparison at `06224c6f2d8bd9e4d2fc848124a02d736377018e`
sealed all six arms under `/dev/shm/t_ebcba52e_unitwise_a2` on Spark-6.
Warm four-cell singleton wall was 10.5641761/10.7341428 s versus
9.29377619/9.23560884 s grouped (1.136694x/1.162256x). All eight paired
canonical FP32 reconstructions were identical across four experts and two
repeats; this is diagnostic evidence, not a replacement for output acceptance.
Setup 29.9007654/10.0703240 s is order/JIT-confounded, not a cold speedup claim.
Final grouped warm peak allocated memory was 2,144,155,648 bytes; batch core
8.48452742 s included LDLQ 7.32521271 s and conformance 0.43555430 s.
The modest warm win is not yet a substantial representative build improvement.
The candidate-only sparse held-out suffix (`t_ebcba52e_glm_suffix_a6`) sealed
through L044/readout, reusing the independently verified baseline by digest
without repeating baseline forwards. Independent binary64 recomputation of the
three authenticated candidate readouts and exact teacher-selected 8192-token
supports (1024 positions/window) gives candidate-minus-baseline KLD deltas
0.0/0.0/0.0 for windows28/56/71 and pooled0.0. All frozen1e-7 gates pass.
The verifier authenticates source index, capture, prefix, baseline-result and
freeze-time ordering. Receipt on Spark-6:
`/dev/shm/t_ebcba52e_glm_suffix_a6/INDEPENDENT_VERIFY_run8299.json`;
verifier SHA256 `556acfe7f0570b9278c06ee0b23b456bde93da7ed88bf1e0b44e44edaf730259`.
This is conditional teacher-support KL for four sparse fusedK3 replacements
under a fixed K2 prefix/suffix, NOT full-vocabulary or uniform-K3 equivalence.
Down/fused and DS4 representative acceptance, a substantial repeatable build
win, and production-owner adoption remain outstanding. Defaults unchanged.

## DS4 composed scheduling and unitwise preprocessing (2026-09-07)

At `7abb90b4c512ed1257283f087e099fe713c902e3`, the same four authentic
L013/E084,E085 down+fused K1 cells were measured with unchanged source,
calibration, geometry, seeds and public API. Unitwise preprocessing/factorization
alone gave warm37.720959/38.793669 s singleton versus36.359096/36.500272 s
batched (1.037456x/1.062832x);24/24 build conformance checks passed.

Composing `block_ldl_unitwise:true`, `viterbi_structured_gather:true` and
`viterbi_num_warps:8` then measured28.835305/28.773447 s for the same four
cells (1.308152x/1.348245x versus that sealed singleton baseline). This is a
measured composition, not a product of independently measured speedup ratios.
The candidate-only experiment reused the just-sealed baseline by digest;
setup31.533846 s is not a matched cold-JIT comparison. Candidate12/12 build
conformance and8/8 paired canonical reconstructions passed. The separate
source-weight NMSE validator passed8/8 unchanged objectives in4.505666 s.
Final warm grouped peak allocated/reserved bytes were5,196,188,160/9,223,274,496;
LDLQ26.156191 s and conformance0.351373 s were included in28.773447 s wall.

Spark-6 receipts are `/dev/shm/t_ebcba52e_ds4_unitwise_a1`,
`/dev/shm/t_ebcba52e_ds4_combined_a1`, and
`/dev/shm/t_ebcba52e_quality_ds4_combined_a1/RESULT.json`.
DS4 held-out output KLD remains unmeasured; reconstruction equality is diagnostic,
not full-model acceptance. No production defaults or adoption changed.
The candidate-only4-warp composition has now sealed12/12 cells. Setup wall was
825.786606 s and warm walls822.130711/822.020531 s, respectively28.511254x/
28.568720x slower than the sealed8-warp composition. It is rejected on measured
runtime, not assignment equality;8-warps remains the unpromoted incumbent.
Receipt root: `/dev/shm/t_ebcba52e_ds4_combined4_a1` on Spark-6.

## GLM authentic down projection and selected source ranges (2026-09-07)

Canonical `216dd1dbd4a43e9112f57949ef400106a6eaceed` supports explicitly pinned
selected safetensors payloads with unchanged original index, headers, offsets,
FP8 weights and scale identities. The24 authentic components for L030 E000--003
(gate/up/down plus scales) occupy100,687,872 bytes instead of requiring two full
source shards. This is source coverage for named tensors only, never a claim to
possess/hash complete parent shards. All16 existing fit windows were reused.

The first down harness failed before any solve sealed because it accidentally
overrode the owner's K2 recipe with a persistent K3 backend. The recipe guard
correctly refused. The retry preserved the original canonical K2 v46 backend;
no production guard changed and no sealed baseline was replayed.

Six four-cell arms under `/dev/shm/t_ebcba52e_glm_down_unitwise_a2` completed.
Warm singleton10.273801/10.557645 s versus unitwise8.335971/8.063009 s gives
1.232466x/1.309393x same-input build speedup. Setup16.099151/8.326594 s is
order/JIT-confounded, not a matched cold-start speedup. Independent canonical
decode/source-weight NMSE validation passed8/8 pairs, with unchanged objectives,
in3.894444 s separately charged validation wall. Reconstruction alone does not
establish full-model equivalence. The new sparse down output gate resumes the
existing fixed L029 prefix and replaces these four down units at L030.
All12 replacement bindings and9 L030 arm/window checkpoints sealed. The gate
then stopped before L033 allocation: MemAvailable46,721,032,192 bytes was below
the unchanged40-GiB peak plus4-GiB reserve requirement47,244,640,256 bytes.
The process is verified dead; L031/L032 forwards were in-memory and unsealed.
The recovery preserved every L030 checkpoint and reclaimed only hash-identical
immutable task-owned duplicate frontier storage by hardlinking (3,724,732,596
bytes); it did not weaken the memory reserve. The resumed suffix reached L044
and all nine readouts without replaying sealed forwards or producer builds.
Independent binary64 verification passes all three windows28/56/71 (1024
positions each) and pooled teacher-support8192 KL: candidate-minus-baseline
0.0/0.0/0.0, pooled0.0, against frozen repeat-derived1e-7 limits. The validator
checks raw teacher/readout hashes, identical support IDs, source/prefix identity,
all12 consumed down-unit bindings, and freeze-before-candidate ordering.
This is a four-down-K2 sparse diagnostic under the fixed earlier L005-K3 prefix,
NOT full-vocabulary KL, uniform-K3/full-model proof, or production adoption.
The retained independent receipt is
`/dev/shm/t_ebcba52e_glm_down_suffix_a2/INDEPENDENT_VERIFY_run8313.json`
(SHA256 `1ddb3052485bcda309761e144db339ad07e486d1fd4a94128e2c41acade8736b`);
producer result SHA256
`e9755d1bf459ff1f85553bd6fc77462878f3913016197d47055f92852c5d4d32`.
Validation is separately charged, never part of the producer speed denominator.
Production owner confirms no cutover: live K3 builds remain on03c531049cef93bfc020de1b313ad1270a78260e.
A matched authentic K3-down representative is the next gate; no live production
replay, claim transfer, or automatic promotion is authorized.

## GLM authentic K3 down schedule composition (2026-09-07, unpromoted)

At pinned public API `e6ba5ebd8c90ae3bebaabe9fd132824314084438`, the owner-approved
K3 recipe bound the existing L030 E000--003 down source inputs with controls-only
references, canonical `resolve_qtip_ring(3)`, original per-cell RHT identity,
Hessians and16 clean-fit windows. No K2 packed artifact was relabeled and no
source/calibration/scorer was recaptured. Nine four-cell arms sealed36 builds.

Warm singleton walls7.840529/6.743711s versus unitwise16 5.918271/5.322746s give
1.324801x/1.266961x. Composing unitwise preprocessing with8-warps measured
4.445550/4.432097s, or1.763680x/1.521562x against those same singleton repeats.
Setup27.238075/5.954162/5.880815s is order/JIT-confounded, not a cold speedup.
Final warp8 batch core3.250987s includes LDLQ2.722792s and packed conformance
0.237834s inside4.432097s total build wall. Peak allocated/reserved bytes are
1,280,087,040/1,619,001,344; this is measured CUDA allocation, not system peak.

Independent canonical packed decode/source-weight NMSE passes16/16 candidate
comparisons, all objective deltas0. Validation wall97.561887s is separately
charged and includes first-use K3-down consumer specialization. Raw roots on
Spark-6: `/dev/shm/t_ebcba52e_glm_down_k3_a1` and
`/dev/shm/t_ebcba52e_quality_glm_down_k3_a1`. A new singleton/repeat/warp8 sparse
K3-down output comparison uses the exact fixed L029 frontier, not the prior K2
quality result. All nine L044 frontiers sealed before READOUT memory admission
failed at MemAvailable46,999,478,272 bytes. A fresh READOUT-ONLY process passed
at49,930,211,328 bytes with the unchanged40-GiB estimate plus4-GiB reserve;
no forward, build or prefix was replayed. Independent binary64 verification of
all nine raw readouts passes: candidate-minus-singleton conditional teacher-support
8192 KLD deltas are0 for windows28/56/71 (1024 positions each) and pooled.
Singleton repeats agree; all frozen reproducibility-derived limits are1e-7.
The receipt is `/dev/shm/t_ebcba52e_glm_down_k3_readout_a1/INDEPENDENT_VERIFY_run8325.json`
on Spark-6; verifier SHA256
`d6587730b3b49e0c1eb99266caf1f6a6375dc26a4b071c68a5e50d98202ce20a`.
This is a sparse four-down-K3 replacement at the existing fixed prefix and K2
suffix, not uniform-K3 or full-vocabulary/full-model equivalence. Production
owner readback and matched clean-boundary rollout remain required; defaults
and live production are unchanged.

A subsequent matched fresh-process/cache experiment at `32a13a73` isolated
canonical/Triton/Inductor/CUDA/XDG caches per arm, in baseline/candidate/candidate/
baseline order. Source assets and OS filesystem cache remained shared. Cold
four-cell baseline29.034211/28.991158s versus candidate27.709112/28.005384s gives
only1.047822x/1.035199x. Warm companion6.999757/6.759336s versus4.933405/4.882107s
is1.418849x/1.384512x. These replace no earlier measurement and do not justify
calling the warm gain a cold-start gain. The first baseline cell alone spent
19.631835s in packed-decode conformance versus2.301603s solver core, identifying
cold consumer compilation as the next causal target. No conformance was skipped.
Candidate warm peak allocated/reserved bytes1,256,967,680/1,595,932,672;
baseline415,758,336/700,448,768. Process RSS is recorded separately in raw receipts,
not mislabeled as total unified-memory peak. All32 builds sealed; independent
source-NMSE checks pass16/16 and each decoded candidate is identical to the already
sparse-heldout-passed incumbent (diagnostic, not a byte-equality acceptance rule).
Independent validation6.306971s is outside build wall. Spark-6 roots:
`/dev/shm/t_ebcba52e_glm_k3_cold_a1` and
`/dev/shm/t_ebcba52e_quality_glm_k3_cold_a1`. No production adoption yet.

## GLM K3 down opt-in eager conformance (2026-09-07, unpromoted)

Public API pin `a32e63428a6461570fe6c30befb29172c1f37fc9` exposes
`packed_decode_execution="eager"` for the existing canonical decoder; compiled
remains the default. No decoder fork, skipped conformance, changed quantization,
or fallback is involved. Focused decoder-execution tests: 2 passed.

Four authentic L030 E000--003 K3 down cells, unitwise preprocessing and warp8
fixed in both arms, were built in fresh processes with private canonical,
Triton, Inductor, CUDA and XDG caches, baseline/candidate/candidate/baseline order.
Source assets and OS filesystem cache remained shared. Cold build walls
27.759437/28.058089 s versus8.252979/8.518711 s yield3.363566x/3.293701x.
Warm walls5.262888/5.238376 s versus5.179128/4.974959 s yield only
1.016173x/1.052949x; do not present the cold compilation saving as a warm gain.
Cold-plus-warm child totals including import/config are34.931091/35.194530 s
versus15.291786/15.370143 s. CUDA peak allocated/reserved bytes agree between
arms: cold1,254,865,920/1,593,835,520 and warm1,256,967,680/1,595,932,672.
Process RSS is separately recorded, not total unified-memory peak.

All32 cell builds sealed. Independent canonical consumer/source NMSE checks
pass16/16 under baseline-repeat-derived limits; decoded max_abs versus the
already sparse-K3-down-heldout-passed incumbent is0 in every check. Independent
validation6.494484 s is outside producer timing. No new output KLD was computed
for this decoder-only rung; equality of decoded weights connects these four
cells to the existing sparse heldout result, not to full-model equivalence.
Raw Spark-6 roots: `/dev/shm/t_ebcba52e_glm_k3_eager_a1` and
`/dev/shm/t_ebcba52e_quality_glm_k3_eager_a1`. Task-local receipts:
`GLM_K3_EAGER_A1_{TERMINAL,SUMMARY,QUALITY}_run8328.json`.
Next representative gate extends this exact public opt-in to authentic fused
geometry. Production adoption requires owner clean-boundary readback;
no default or live producer was changed by this experiment.

## Unpromoted single-Spark QTIP build experiments (2026-09-07)

These are bounded same-input diagnostics, not production quality acceptance.
The DS4 panel is four distinct authentic K1 cells: L013 E084/E085, each down
and fused13, with unchanged source/calibration/geometry/seeds. Decode conformance
is included in build wall; first-use consumer compilation is not producer speed.

- Existing cross-unit batching at `dca8699d2539917c7b1259952eefd7d3649de007`:
  warm four-cell singleton 38.194566 s versus grouped 36.810715 s (1.037594x).
  Grouped numerical reconstruction differs: three cells exceed the strict
  baseline-repeat diagnostic limit, one improves. This is neither a byte-based
  rejection nor proof of meaningful held-out regression.
- Opt-in `viterbi_backpointer_dtype="uint16"` at
  `b5f59663e009d663d2618c460dc46fb6cd95f1e6`: lossless internal K1 backpointer
  compression, int32 returned states and full recurrence unchanged. Warm default
  int32 four-cell repeats 37.961138/38.300984 s versus uint16
  48.923851/48.165708 s (0.775923x/0.795192x): **negative speed result**.
  Final warm arm peak allocated/reserved bytes were 2677950464/4624220160
  (int32) and 1595801600/2468347904 (uint16). The storage reduction did not
  accelerate this panel. Packed assignments match on all four cells as a
  diagnostic only. Default remains int32; no production activation.

Provenance: task `t_ebcba52e`, dedicated Spark-6; immutable experiment roots
`/run/user/1000/t_ebcba52e_ds4_a1`, `t_ebcba52e_quality_ds4_a1`, and
`t_ebcba52e_bp16_a1` on that host. Compact terminal receipts are retained in the
task workspace as `DS4_A1_TERMINAL.json` and `BP16_A1_TERMINAL.json`.
Held-out output KLD remains unmeasured; neither variant is promoted.

Further same-panel results (all opt-in, production defaults unchanged):
- Device profile: generic Viterbi dominated two real cells (768 launches,
  16.006374 s device). CPU synchronization is waiting, not additive opportunity.
  Instrumented wall is never a speed denominator.
- Warp8 vs warp16: warm four-cell 35.272042/35.248076 s vs
  37.844731/38.561673 s (1.072938x/1.094008x). Canonical paired reconstruction
  passes 8/8 comparisons, decoded max_abs 0, validation 8.310000 s.
  That is a bounded numerical result, not measured output KLD or full-model proof.
- Full branch-loop unroll4 vs rolled1, both warp8: warm four-cell
  34.629889/34.678049 s vs 34.893328/35.640234 s (1.007607x/1.027746x),
  24/24 builds pass canonical decode conformance. Near-neutral, not substantial.
  Setup arms 40.015426/37.233308 s include different first-use compilation;
  they do not establish a matched cold-build speedup.
- Compiler metadata still reports 65,536 shared bytes for the generic gather
  in all six compiled warp/unroll variants. Next isolated causal experiment is
  `viterbi_structured_gather=true` for K1 only: select a contiguous predecessor
  row, then fourfold broadcast, rather than a general full-prefix gather.
  Matched physical result: generic8 warm four-cell 34.769903/35.412527 s
  versus structured8 31.155750/31.166651 s (1.116003x/1.136231x), all24 builds
  pass canonical decode conformance. Subsequent canonical paired reconstruction
  passes8/8, decoded max_abs0, separately charged validation7.828695 s.
  Compiler entries for the new movement report4096 shared bytes; this is not a
  measured occupancy claim. Output KLD is still unmeasured, so no promotion.
  Structured16 was then measured under the changed shared-resource envelope:
  warm 32.018977/31.943393 s versus structured8 31.001402/31.182052 s
  (0.968220x/0.976166x). All24 builds pass; retain structured8, not16.
- Structured8 grouped2 vs singleton on the same four cells at `624c210f`:
  warm 28.204494/28.214014 s vs31.105050/31.288451 s
  (1.102840x/1.108968x). All24 builds pass. Final warm producer-core totals
  are26.130852 vs27.864315 s, conformance0.364080 vs0.351104 s,
  staging1.682423 vs3.025813 s; these phases are included in wall.
  Peak allocated/reserved bytes are5213002240/9240051712 grouped,
  2665339904/4611637248 singleton. Setup walls28.948990/33.555770 s
  have first-use/order confounds, not matched cold-JIT evidence.
  Separate canonical reconstruction validator took8.543186 s: only2/8
  numerical comparisons meet baseline-repeat limits. E084 down NMSE
  0.499220566 exceeds0.499113813; fused E0840.597373550 exceeds0.597026913,
  fused E0850.562766462 exceeds0.562458819. E085 down improves.
  This is a numerical diagnostic, not byte-based rejection or measured output
  KLD harm. Preserve candidate for decisive held-out adjudication; no promotion.
- `e782a5fe` adds opt-in K1 `viterbi_backpointer_dtype="uint8"` branch-index
  storage, reconstructing the full state using the implicit prefix column.
  Full branch recurrence and returned int32 wire remain unchanged; default
  int32 and conservative memory admission remain. CPU tests exercise actual
  store/restore expressions over all65536 states plus public installer/launch
  wiring. Physical result: warm byte-q34.983784/35.026989 s vs
  int32 31.160164/31.181123 s (0.890703x/0.890203x), all24 builds pass.
  Final warm peak allocated bytes1058930688 vs2677950464. Canonical
  reconstruction passes8/8 with decoded max_abs0, validation7.999939 s.
  The byte table saves memory but is slower; retain structured8/int32.
  This still does not measure held-out output KLD.

Exact receipts: `t_ebcba52e_k1warps_a1`, `t_ebcba52e_quality_k1warps_a1`,
`t_ebcba52e_unroll_a1`; local terminal copies retain individual phases and peaks.

`runtime/ACCELERATION_MANIFEST.json` is the exact machine-readable inventory. This document is its concise operator view.

| ID | Development source | Image build input | Runtime activation | Principal test |
|---|---|---|---|---|
| `bs-pack-export-verify` | exporter, schemas, repair/repack/materialized-wire code | `banana-smasher` wheel | `smash export`, `smash verify`, `smash serve-check` | exporter CLI/contract/materialized-wire tests |
| `stock-vllm-general-plugin` | plugin entry point and `register()` | plugin wheel built from checkout | stock vLLM discovers `vllm.general_plugins` | plugin contract tests |
| `native-plane-p1016` | native-plane loader and quantization config | NumPy, safetensors, plugin wheel | `quant_method=banana_smasher` selects native routed experts behind a `cudagraph_unsafe` opaque custom op; breakable `PIECEWISE` capture is mandatory and other capture modes fail closed | native-plane runtime, compile-boundary, and image-default tests |
| `p1016-cutedsl-tlut` | P1016 kernels plus packaged QTIP TLUT | quack-kernels and `qtip_tlut.npy` | fail-closed `mixed_exact_gemv` dispatch | CuteDSL and native-plane tests |
| `sm121-deepgemm-dense-e8m0` | SM12x O-projection and dense preflight hooks | official DeepGEMM 2.6.1 `nv_dev_f8e8fb5` source | stock SM100+ packed-scale O-projection with `VLLM_USE_DEEP_GEMM=1` and `VLLM_USE_DEEP_GEMM_E8M0=1` | SM121 dense/V4 tests |
| `deepgemm-ue8m0-warmup` | dense/grouped warmup scale initializer | same pinned DeepGEMM source wheel | initializes every dummy UE8M0 activation scale to one before DeepGEMM warmup | dense-capability tests |
| `sm121-deepgemm-sparse-indexer` | external DeepGEMM registration hook | same pinned DeepGEMM source wheel | boot-time SM12x lazy symbol registration | sparse-indexer and dense-capability tests |
| `sm121-persistent-topk` | TopK correction hook | stock vLLM persistent TopK op | replaces unsupported cooperative TopK on SM12x only | sparse-indexer TopK tests |
| `sm121-v4-attention-flashinfer` | V4 attention selector hook | source-built FlashInfer with pinned fixes | SM12x FlashMLA request routes to FlashInfer sparse MLA | attention tests |
| `stock-deepgemm-mhc` | plugin preserves stock MHC dispatch | pinned DeepGEMM source wheel | no plugin override; stock public backend remains active | SM121 MHC tests |
| `flashinfer-sparse-decode-compat` | sparse-decode signature adapter | source-built FlashInfer | one-time API variant adapter during plugin registration | FlashInfer compatibility tests |
| `sm120-aot-cubins` | 26 SM120 cubins plus immutable producer map | exact names, bytes, and SHA-256 admitted from `runtime/ASSET_MANIFEST.json` | AOT root and MoE W2 environment paths | exact source/image admission and native runtime tests |
| `e43-aot-cubins` | 6 E43 cubins plus immutable producer map | exact names, bytes, and SHA-256 admitted from `runtime/ASSET_MANIFEST.json` | MoE W3 cubin environment path | exact source/image admission tests |
| `flashinfer-autotune-cache` | version-aware validator and excluded 0.6.14 provenance only | no cache is baked; 0.6.17/121a must be generated on a GPU | persistent named volume at the versioned vLLM cache root | mismatch rejection and capture tests |
| `real-libcudart-link` | checked-in FlashInfer patch | real CUDA 13 runtime link replaces TileLang stub | image-build verification imports FlashInfer against real runtime | Docker static/image verification tests |

## Stage coverage

Development includes both package sources, all package/plugin tests, JSON schemas, repair/repack/materialized-wire handling, and every AOT asset consumed by the image. Image build compiles both local wheels, source-builds pinned FlashInfer and DeepGEMM revisions, verifies package imports, links real `libcudart`, writes package provenance, and preserves the exact stock-vLLM `CMD`. Serving mounts only a verified pack at `/model`; plugin registration activates fail-closed runtime hooks before model load.

`runtime/KERNEL_PRODUCERS.json` and `archive/KERNEL_DEVELOPMENT.md` distinguish shipped/hash-gated cubins from exact-source-rebuild seals. The SM120 set remains unsealed because cubit short identity `5912400` is unresolved. The E43 recipe has a sealed independent 6/6 byte-identical rebuild receipt.
