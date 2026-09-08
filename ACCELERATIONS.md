# Runtime accelerations

## Bounded batch partition and unsolved-prefix experiment (run8445)

Two resident batches of two authentic L005 E001--004 K3 fused13 cells were
compared with the sealed run8441 batch-of-four baselines (no baseline replay).
All sixteen new builds and sixteen independent reconstruction checks passed,
with zero decoded maximum difference. Cold walls were 12.830212/12.785283 s
versus 12.661063/11.952383 s (0.960881x aggregate); warm walls were
8.508531/7.979239 s versus 9.234433/8.124412 s (1.052832x aggregate).
Private process/compiler caches were fresh; shared source/OS caches were not.
Peak allocated memory was 1,102,538,240 bytes. Independent validation cost
6.307325 s is separate. This is a small warm tradeoff, not a substantial
repeatable build win; retain the batch-four incumbent. No heldout or adoption
claim. Receipts: t_ebcba52e_group2_run8445 and t_ebcba52e_quality_group2_run8445
under /dev/shm on the dedicated seat; compact summaries in the task workspace.

The next structural research opt-in is `ldlq_update_unsolved_only=true` in the
public batch config. It omits outer-buffer product updates to already-solved
columns, including the unused final update, while keeping all predecessor
terms for the remaining prefix. Default false preserves historical behavior.
Changing BMM output geometry can change device kernel rounding; CPU nonzero
residual regression is not accelerator or heldout equivalence. Decode,
reconstruction/objective and independent output gates remain mandatory before
production adoption. No encoder, native fallback, or relaxed validation added.

At canonical pin `34bad2607dd9e88c8ee810d7345758e92ca589f5`, one real smoke
and sixteen four-cell cold/warm candidate builds passed. Unsolved-prefix cold
walls were 12.249842/12.791585 s (0.982909x aggregate versus the same sealed
batch-four baselines); warm walls were 7.912666/8.030731 s (1.088780x).
The two paired cold ratios disagree in sign. All sixteen independent
reconstruction checks passed with zero decoded delta; validation cost
6.049619 s and peak allocated memory 2,116,832,768 bytes. This does not establish
a substantial repeatable end-to-end win, and the opt-in remains default-off.
No new heldout gate was run. Dedicated-seat roots:
`/dev/shm/t_ebcba52e_unsolved_run8445` and
`/dev/shm/t_ebcba52e_quality_unsolved_run8445` (volatile, sealed).

The production owner separately read back physical selected-source integration:
original L005/E000 and L014/E000 fused13 fitting sources produced two fresh K3
products using three selected-source modules from `1e35a502`, retaining its
previous producer runtime. Owner receipt `JS1_ACCEPTED_run8446.json` binds
destination ACK `b0364fc1022e3c1f054e8df56cc7ede44556289fc6a1005f6ad5097b3d007f61`.
This is selected-source loader integration/admission, not adoption of the
unsolved-prefix candidate or full-model heldout equivalence. Adjacent original
`qtip_rings` runtime assets were necessary for source closure; the failed
zero-solve predecessor is preserved rather than silently replacing its receipt.

## Fresh consumer cohort closure: ordinal3 (run8421)

The predeclared ordinal3/window71 reasoning gate completed without replaying
ordinal0/1/2. Frozen API `cb460bb16b63aa567c5e44f3c5e81c1cb12d7f04`
resumed the owner's authenticated L004 frontier with identical K2 wire, 1024
positions and fixed teacher-selected 8192-token support. After producer death,
the prepared independent verifier authenticated 18/18 frontier/readout bindings
and all 120 unique arm/layer timing records. CPU control, repeat and rounded
CUDA each measured conditional-support mean KL 0.08681243092070783. Candidate
delta was zero against the frozen repeat-derived 1e-7 limit. Paired support
logits had zero maximum difference; full-vocabulary top1 agreed 1024/1024,
with teacher top1 matches 930/1024 in every arm.

Forty-layer materialization: CPU 5199.927947 s versus CUDA 1481.353961 s
(3.510254x); forwards 16.578090 s versus 15.565842 s. Peak CUDA allocation was
21,612,405,760 bytes in both. Repeat shared CPU materialization and is charged
zero, not another speed replicate. Full three-arm diagnostic wall was
6976.389502 s. Transfer, hashing, readout and startup are excluded from the
materialization ratio. This closes the predefined fresh ordinal2/3 consumer
cohort, NOT producer-build throughput, process-cold speed, full-model equivalence
or production adoption. Primary Gutenberg-fit versus evaluation document ancestry
remains qualification-open; ordinal0 RED is preserved, with no tolerance tuning,
automatic full64 expansion or production/default changes.

Spark-6 receipt:
`/home/dnola/missions/t_ebcba52e_ordinal3_run8411_a2/INDEPENDENT.json`
SHA256 `97b29a25d33aa4678c6bddf0d748ac8edddb0b63d7a5a0ab5dc9e367c292f34a`;
RESULT SHA256 `04ab946a5517bf2b61397ffa3a698bc3c5e65fa43f0d09a54b92ba0d070195cc`.
Next work is production-owner adoption coordination and a separate bounded
producer acceleration experiment, not another replay of this consumer cohort.

## Fresh ordered consumer expansion: ordinal2 (run8411)

The next driver-authorized bounded cohort was selected before readout: fresh
ordinal2/window68 prose, then ordinal3/window71 reasoning. Ordinal0/1 were not
replayed. Frozen API `cb460bb16b63aa567c5e44f3c5e81c1cb12d7f04` resumed the
owner-authenticated ordinal2 L004 frontier with identical K2 wire and fixed
1024 positions / 8192 teacher-selected support. All three arms (CPU control,
CPU repeat, rounded CUDA) had conditional-support mean KL 0.5307037282072324.
Candidate delta was zero versus the predeclared repeat-derived 1e-7 limit;
paired support-logit max difference was zero and full-vocabulary top1 agreed
1024/1024. Teacher top1 matches were 724 in every arm. After producer death,
an independent verifier authenticated 18/18 frontier/readout bindings and
120 unique arm/layer timing records.

Forty-layer materialization was CPU 5163.612175 s versus CUDA 1485.084203 s
(3.476983x). Forward cost was 16.906256 s versus 15.855320 s, with peak CUDA
allocation 21,612,405,760 bytes in both. The repeat reused CPU materialization
and is charged zero, not another speed sample. Full three-arm experiment wall
was 6944.809709 s; transfer, hashing, readout and startup are outside the
materialization ratio. This is consumer acceleration, NOT producer-build speed,
process-cold speed, full-model equivalence or production adoption. Document
ancestry remains qualification-open. Ordinal0 RED is preserved; no defaults
or production settings changed. Ordinal3 remains the predefined next gate.

Spark-6 receipt:
`/home/dnola/missions/t_ebcba52e_ordinal2_run8411_a2/INDEPENDENT.json`
SHA256 `26b17209cd6facd62011a8b0d41efbaa55eb8cd7f3be8630618cf3bfa5a55f65`;
RESULT SHA256 `8958837215e5efe9b2ba9d6df0c69f7620f836d29f38f3f751c75c3668f339b5`.

## Corrected consumer: one independently verified output row (run8407)

Frozen API `cb460bb16b63aa567c5e44f3c5e81c1cb12d7f04` completed the explicitly
authorized ordinal1/window56 same-wire K2 control/repeat/corrected-CUDA diagnostic.
It resumed the owner's authenticated L004 frontier, not the original prefix.
All three arms' conditional teacher-support mean KL was 0.30121949575380447
on 1024 positions and fixed 8192 support. Candidate delta was zero against the
pre-readout frozen 1e-7 limit. Paired support logits had zero maximum difference;
full-vocabulary top1 agreed on 1024/1024 positions. Teacher top1 matches were
841/1024 in each arm. The independent raw-logit reducer verified all 18 persisted
frontier/readout bindings and all 120 arm/layer timing rows after producer death.

Forty-layer materialization: CPU 5223.142911 s; corrected CUDA 1486.547205 s
(3.513607x). Forward costs: 16.471606 s and 15.455129 s, respectively; both
peaked at 21,612,405,760 CUDA bytes. Control-repeat shared CPU materialization
and was charged zero materialization, not presented as another speed replicate.
Transfer, hashes, readout, and startup are excluded from this materialization
ratio. This is consumer materialization evidence, NOT producer build speed,
process-cold speed, full-model equivalence, or production adoption. The failed
ordinal0 diagnostic remains RED and was not replayed. Primary Gutenberg-fit
versus frozen-evaluation document ancestry remains qualification-open.

Spark-6 receipt: `/home/dnola/missions/t_ebcba52e_ordinal1_run8387_a2/INDEPENDENT.json`
SHA256 `f59e7872c0f6c88977a1d5f54eb9ab0cbd2a6178dd77ab03e76e62ba85fd34d0`;
producer RESULT SHA256 `13a0f044b862ae8a2fce0d1c3480761eedd75c5b561ee194b3fbca7ce73fb8ac`.
No defaults or production settings changed. Next output expansion requires an
explicit owner/driver-bounded gate, with frozen code and no heldout tuning.

## Device canary closure and normalization isolation (2026-09-07, run8379)

At `373f7d2677551a6cb1eca62d7259847ff3c161c7`, eight authentic same-wire
GLM rows (K2 L044 E000/001 gate/down and K3 L005 E001--004 fused13) passed
predeclared decoded-relative-squared-error limits. Summed warm CPU eager decode
was 1.266493 s versus CUDA eager 0.378261 s (3.3482x); CUDA peak was
516,428,288 bytes. These are decode costs with context warmed by physical tests,
not process-cold costs or build-speed evidence. Eight physical tests passed.

The paired frozen ordinal0/window28 diagnostic then reused the sealed K2
control L034 frontier and independently verified CPU readout, forwarding only
L035--044 with unchanged packed product on CUDA eager. Materialization summed
1336.661200 s CPU versus 369.985808 s CUDA (3.6127x); candidate consumer wall
was 439.196128 s. No full native-model reads occurred. Output gate **RED**:
conditional teacher-support KL 0.2728915764209244 versus
0.2726494667965614, delta +0.0002421096243629961 exceeds frozen 1e-7.
Teacher Top1 was 802 versus 804; paired full-vocabulary Top1 was 1009/1024.
This is not acceptable consumer promotion, despite tiny panel errors.
The earlier changed-tier K2 versus sparse K3 diagnostic also failed (+0.00694885);
it does not adjudicate a fixed-bit build optimization.

Non-heldout phase instrumentation localized K2 down differences to the final
FWHT division by sqrt(2048), not trellis decode or butterfly inputs. CPU
compiled/eager agreed; CUDA default changed 45/63 BF16 weights in the two real
down rows. A correctly rounded FP64-intermediate division probe restored both
without changing any assignments, but its instrumentation wall is not speed
evidence. The research-only `normalization="rounded"` public decoder option
(`packed_decode_normalization` on artifacts) uses FP32 Triton `div_rn` instead,
leaving every existing default unchanged. CPU retains native division; explicit
CUDA rounded mode requires contiguous FP32 and fails rather than falling back.
Physical kernel, representative same-wire, and independent output gates remain
required before promotion; no full-model equivalence, ancestry qualification,
or producer adoption follows from these diagnostics.

Physical follow-up at `cb460bb16b63aa567c5e44f3c5e81c1cb12d7f04` passed
19 focused tests without skips on the claimed accelerator. The original eight
rows and sixteen disjoint K2 L044 E002--009 gate/down rows all had zero measured
FP32/BF16 decoded delta. Paired summed warm CPU/CUDA costs were
1.343047/0.378358 s (3.5497x, eight rows) and 2.520546/0.614720 s
(4.1003x, sixteen rows), with peaks 516,428,288 and 409,473,024 bytes.
A subsequent twelve-process order-alternated gate (three authentic cells, two
processes/device/cell, private compiler/CUDA caches) preserved all per-cell
decoded hashes across first and warm calls. CUDA process-cold initialization
was slower than CPU: the warm gain must not be advertised as a single-call
cold-process win. Parent wall includes import, digest copies and teardown;
source/OS caches were not cold.

This closes a 24-distinct-cell numerical/decode gate, not downstream-output
acceptance or build acceleration. No failed frozen output diagnostic was replayed.

Remote scientific roots on the dedicated seat:
`t_ebcba52e_decode_device_run8362`, `t_ebcba52e_cuda_suffix_run8379`, and
`t_ebcba52e_decode_normalization_run8379`. Independently verified CUDA suffix
RESULT SHA256: `9646fa4871ae91d24fe066be8b805dcb0f493d8c5d34b12ee3fea81af4b36103`.

## Explicit packed-consumer device selection (2026-09-07, run8362)

`decode_sealed_unit(..., device="cpu" | "cuda:0")` and the artifact's
`packed_decode_device` field select where the existing sealed-wire decoder and
inverse transforms run. CPU remains the default; decoder execution selection,
wire validation, and source-fallback refusal are unchanged. This is a research
opt-in, not a promoted numerical or performance result. A missing CUDA device
fails rather than falling back to CPU. Native and legacy non-unit payloads retain
their existing placement semantics; the option concerns sealed packed units.

Motivation: the bounded ordinal0 sparse diagnostic at `6fe1f5fa` measured L005
and L006 materialization at 140.129809 and 133.562836 seconds, versus sub-second
warm forwards. A physical stack sample identified CPU FWHT work, not a persistent
compiler failure. Those are consumer costs, not producer build timings. The
running diagnostic remains immutable on CPU; CUDA K1--K4 tests and authentic
same-wire CPU/CUDA numerical, memory, cold/warm and output gates are required
before any consumer cutover. Local tests alone do not establish device parity.


## K3 inner-axis scheduling closure (2026-09-07, run8359)

Two additional candidate-only public-API panels at
`54be49f02ea95ec1eaf5ee4df459b66e16e11660` reused the same sealed four-cell
L005 E001--004 fused13 K3 incumbent and unchanged numerical limits. No baseline
or calibration was replayed. Eager conformance, selected source, unitwise
preprocessing and the new inner-axis four-branch reduction were fixed; only
warp count changed. Private process/compiler caches do not imply cold OS/source
caches. The incumbent remains cold 12.915914/12.177085 s and warm
7.960290/8.082386 s.

| Inner-axis schedule | Cold walls (s) | Warm walls (s) | Decision |
| --- | --- | --- | --- |
| 16 warps | 37.096614 / 37.786794 | 33.243002 / 33.320512 | RED, 0.3223--0.3482x cold and 0.2395--0.2426x warm |
| 4 warps | 14.374929 / 14.107082 | 10.217348 / 10.225253 | RED, 0.8632--0.8985x cold and 0.7791--0.7904x warm |

Both panels passed 17/17 actual builds and 16/16 independent reconstruction/decode
checks; maximum decoded delta against the incumbent was zero. Validation cost
was separately 23.156630 and 23.051268 s. Each four-cell panel peaked at
2,114,731,008 allocated bytes cold / 2,116,832,768 warm and
3,321,888,768 reserved bytes cold / 3,323,985,920 warm. Including the previous
8-warp result, all tested schedules of this changed layout lose to the true
incumbent. Close this grouping/layout family rather than compare against its
slower rejected members or infer occupancy from shared-memory metadata.

Receipts: task-local `GLM_K3_{INNER16,INNER4}_{TERMINAL,QUALITY,SUMMARY}_run8359.json`.
The two durable Spark-6 archives are
`/home/dnola/missions/t_ebcba52e_preserved_inner16_run8354` and
`/home/dnola/missions/t_ebcba52e_preserved_inner4_run8359`; each preserves 99 new
scientific paths, with byte-identical units hardlinked and no scientific deletion.
No new speed winner or production promotion is claimed.

The production owner subsequently supplied a predetermined ordinal0/window28
teacher row, token ledger, suite and CAPTURE. All four files (51,389,480 bytes)
were localized directly over the Spark fabric and authenticated, including
ordered int32[1024,8192] teacher support and float32 logits. This is input closure,
not model-output quality evidence. The separately authorized next step restores
only the K2 routed/native-rest EMBED--L004 prefix with per-layer durable
activation/topk checkpoints; it is not a native prefix, uniform-K3 evaluation,
64-window scorer restart or adoption authorization. Document-ancestry
qualification and a paired downstream output gate remain outstanding.

## K3 branch scheduling and reduction layouts (2026-09-07, run8354)

Three new candidate-only rungs reused the sealed selected-source four-cell
L005 E001--004 fused13 K3 incumbent; no baseline build or calibration replay.
All keep eager conformance, unitwise preprocessing, eight warps, identical
source/calibration/geometry and private per-process compilation caches.
Shared OS/source caches remain disclosed. The paired incumbent walls are
cold 12.915914/12.177085 s and warm 7.960290/8.082386 s.

| Opt-in / deployed pin | Cold walls (s) | Warm walls (s) | Decision versus incumbent |
| --- | --- | --- | --- |
| Four-way branch unroll, `4417a0a0d9f54f5de60db0bd4ac70d11497b7962` | 13.731646 / 13.897338 | 7.596536 / 7.657551 | 0.8762--0.9406x cold; only 1.0479--1.0555x warm, not substantial end-to-end |
| Four-branch outer-axis reduction, `5ed85c10c7f7e765d6bbb69fcadda4e8ce4af4b5` | 26.514541 / 27.257612 | 22.348673 / 22.852550 | Rejected for speed |
| Four-branch inner-axis reduction, `a196fe2cca9f0b4040309603d1a0fb4fcbf4ccd7` | 13.788614 / 14.109511 | 9.496375 / 9.424644 | Better than failed outer-axis, still slower than incumbent; rejected |

Each rung passed 17/17 actual public-API builds (one smoke then two cold/warm
four-cell arms) and 16/16 independent source-NMSE/decode checks. Maximum decoded
change versus incumbent was zero. Separate validation walls were 26.754476,
24.846130 and 23.684414 s respectively. Peak allocated bytes were 2,114,731,008
cold and 2,116,832,768 warm; peak reserved 3,321,888,768 / 3,323,985,920 bytes.
No new held-out output KLD or full-model equivalence is claimed.

`viterbi_branch_unroll=true` now admits K3 as an opt-in. Independent
`viterbi_branch_grouped=true` changes the existing specialized kernel's
branch-reduction layout, not its encoder, wire, source or bit tier; it rejects
mixed batch schedules, incompatible tiers, structured gather and simultaneous
unroll. Defaults remain OFF. The reduction explicitly ignores NaNs and retains
the lowest state on finite ties; all 64 branches are evaluated. A first smoke
at `3e7d44bd` failed before any unit sealed because Triton treated a scalar
`q` from overlap initialization as loop-carried into a rank-2 grouped tensor.
The recovered namespace used distinct grouped-variable names; no fallback,
failed output overwrite or scientific timing credit. Focused suite: 97 passed.

The outer-axis eight-warp compiled entries allocate 16,384 shared bytes;
inner-axis entries allocate 0/64. Default AOT sixteen-warp entries are distinct
and must not be counted as executed candidate variants. This metadata does not
measure occupancy. The inner-axis improvement against a failed experiment is
not an improvement against the incumbent.

Receipts: task-local `GLM_K3_{UNROLL,GROUPED_A2,INNER}_{TERMINAL,QUALITY,SUMMARY}_run8354.json`
and `GLM_K3_GROUP_LAYOUT_METADATA_run8354.json`. Durable Spark-6 archives retain
all new scientific bytes under `/home/dnola/missions/t_ebcba52e_preserved_{unroll,grouped,inner}_run8354`;
byte-identical unit files are hardlinked without losing any original path.
The producer owner explicitly has no surviving hash-closed independent held-out
incumbent/frontier/support seam to authorize adoption. Producer fitting support
is NOT held-out support. No scorer restart, down-canary adoption or production
promotion is authorized. Best incumbent remains selected+eager+unitwise+8warps,
rolled global predecessor costs. Next work should target another measured
mechanism rather than replay these rejected branch layouts.


## K3 operator attribution and device conformance (2026-09-07, unpromoted)

A new instrumented four-cell selected-source eager panel attributed 5.186006s
of GPU time to 512 persistent Viterbi kernels and 0.903068s to pageable DtoH.
These are profiler attribution, not build throughput; nested operator totals
must not be added to GPU kernel totals. The existing generic register-row
broadcast was extended to K3 behind `viterbi_structured_gather=true` at
`17fbf435a03bd76d10f4ecf3084ed9e5862a297c`. One real smoke and two four-cell
cold/warm arms passed 17/17 builds, but the experiment was REJECTED FOR SPEED:
cold 30.881981/30.961459s versus saved 12.915914/12.177085s; warm
26.866802/26.792446s versus 7.960290/8.082386s. Independent source-NMSE and
canonical decode passed 16/16 with decoded max-abs delta zero in 23.948502s.
The default remains unchanged. Fewer global scratch transfers were not a win;
compiled metadata reports 512 shared bytes, not measured occupancy.

A separate public batch opt-in, `packed_conformance_on_device=true`, shipped at
`918b59f02084b875489bde880829c4dbbe385105`, default false. It compares complete
FP16 bit patterns on the device and retains the full FP32-versus-FP16 maximum
error check, avoiding the redundant decoded-matrix DtoH copy. It does not skip
validation, change packed bytes, or change stored reconstruction residency.
Structured gathering was OFF. With the same saved selected-source baseline,
one real smoke plus two cold/warm arms passed 17/17 builds:

| Arm | Baseline wall | Device comparison wall | Baseline/candidate |
|---|---:|---:|---:|
| cold 1 | 12.915914s | 11.493025s | 1.123805x |
| cold 2 | 12.177085s | 11.795556s | 1.032345x |
| warm 1 | 7.960290s | 8.627524s | 0.922662x |
| warm 2 | 8.082386s | 7.561282s | 1.068917x |

Warm conformance alone fell from 1.000918/1.081824s to 0.407988/0.409145s,
but whole-build warm results are inconsistent: no warm throughput win is
claimed. Peak CUDA allocation was unchanged at 2,114,731,008 cold and
2,116,832,768 warm bytes. Independent validation passed 16/16 with decoded
delta zero in 23.446800s; its cost is separate from producer wall. These are
sequential same-input continuations with shared OS/source caches, not a fresh
interleaved performance gate or held-out/full-model acceptance. Production
adoption and durable held-out incumbent linkage remain missing.

Receipts: `GLM_OPERATORS_run8343.json`, `GLM_K3_STRUCTURED_*_run8343.json`,
`GLM_DEVICE_CONFORMANCE_*_run8343.json`. The selected source and all sealed
selected/structured scientific outputs are now durable at Spark-6
`/home/dnola/missions/t_ebcba52e_preserved_run8343`; the device-comparison
rung is at `/home/dnola/missions/t_ebcba52e_preserved_device_run8343`.
All 32 original scientific paths were retained using four byte-identical
unit inodes. Only 280 inventoried task-owned compiler-cache files were
removed. Scientific bytes deleted: zero. Final free space 4,361,072,640
bytes remained above the unchanged 4GiB reserve. Volatile originals remain;
archived configs retain their original absolute lineage paths and need
explicit source-root rebinding before post-reboot reuse.

## Authenticated selected-source continuation (2026-09-07, unpromoted)

The follow-on canonical staging profile isolated full-shard hashing as the
remaining first-process cost: E001 and E004 source reads took 7.28875 and
7.40763 seconds, of which SHA paths took 7.1701 and 7.2601 seconds.
The existing `SelectedTensorSource` path was reused, not a new encoder or a
cross-process trust cache. Sixteen original FP8-weight/F32-scale ranges
(67,125,248 bytes) were locally materialized and authenticated in 0.167743
seconds. All four canonical dequantized source matrices matched exactly.
Original model index/config, header/range descriptors, payload hashes,
calibration, RHT seeds and bit geometry remain bound.

Two new private-cache selected-source eager arms, using code pin
`96510e966680afe7cc5ac186fa1099e0bac6999e`, sealed 16/16 builds without
replaying prior baselines. Their cold four-cell wall was 12.915914 / 12.177085
seconds, versus the preceding eager-only 29.445928 / 26.748347 seconds:
2.279818x / 2.196613x incremental improvement. Against the original compiled
full-source arms this is 3.747905x / 3.656081x; charging the entire selected
materialization to each cold arm gives 3.699854x / 3.606402x. This is a
sequential continuation against saved same-input baselines, not a newly
interleaved randomized experiment. Source/OS caches remain shared.
Warm walls were 7.960290 / 8.082386 seconds; no dramatic warm gain is claimed.

Independent validation took 28.100709 seconds and passed 16/16 source-NMSE
checks with decoded maximum absolute difference zero versus the preceding
eager-only artifacts. Held-out incumbent linkage and production adoption
remain unestablished, as confirmed by the production owner. No full-model
quality or automatic promotion is implied. Receipts are task-local
`GLM_SELECTED_EAGER_{TERMINAL,SUMMARY,QUALITY}_run8341.json`,
`GLM_SELECTED_SOURCE_PARITY_run8341.json` and `GLM_STAGING_PROFILE_run8341.json`.
The new selected payloads are under `/dev/shm/t_ebcba52e_selected_fused_run8341`
and `/dev/shm/t_ebcba52e_glm_selected_eager_run8341`: volatile storage is explicit.
Disk admission refused the four-GiB reserve; scratch relocation preserved the
same reserve rather than deleting source or sealed output. Durable payload
archival needs space before another reboot; local receipts are not payload bytes.

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

## Producer admission recovery and host-thread experiment (2026-09-08)

Refreshing the consumer-tested main pin exposed a real producer admission error:
`qtip_runner.py` changed in cb460bb1 but the independent trusted runner anchor
remained stale. Canonical 1e35a502 repairs the digest without weakening admission.
The shipped-anchor regression first failed; anchor plus solve-resume tests then
passed69/69. The failed attempt sealed zero cells and was not used for timing.

At 1e35a5022d3cb43ce2baa34de0c586695f6260a4, four authentic GLM L005
E001--004 fused-K3 cells compared8 CPU threads against1, keeping selected source,
eager conformance, unitwise preprocessing, warp8 and the existing full recurrence
fixed. Fresh-private-cache order was baseline/candidate/candidate/baseline, each
with a cold and warm call; OS/source assets remained shared. All32 builds passed.
Cold four-cell walls were12.661063/11.952383s baseline versus12.522379/13.108021s
candidate (mean ratio0.960322x). Warm walls9.234433/8.124412s versus
8.409387/8.501477s give mean1.026491x, but paired1.098110x/0.955647x.
This is NOT a repeatable substantial scheduling win;8 threads remains incumbent.
Maximum CUDA allocated/reserved bytes across arms:2,116,832,768/3,323,985,920.

The independent canonical decode/source-NMSE gate passed16/16 comparisons under
the predeclared baseline-repeat formula. Maximum decoded tensor difference was0,
a diagnostic rather than a byte-equality requirement. Validation9.308261s is
separate from build wall. No new held-out output KLD, full-model equivalence or
production adoption is claimed. Spark-6 roots:
`/dev/shm/t_ebcba52e_glm_threads_run8441_a2` and
`/dev/shm/t_ebcba52e_quality_threads_run8441`. Compact receipts live in
`receipts/acceleration/threads-run8441/`.


## Selected-source DOWN transitive fitting closure (run8449)

The original down projection source is insufficient for an authentic fitting replay: public `_prepare_fit_windows` derives down activations using the original gate/up projections. A DOWN selected-source package requires six tensors (three weights and scales); FUSED13 requires four. The acknowledged two-tensor predecessor is immutable and retained. Eight L031 DOWN cells (E000, E001, E010, E100–E104) now have fresh six-tensor namespaces and independently rehashed destination ACKs plus path-only config/Hessian relocations. The original source index, full parent hashes, headers, fit16 captures and ledger identities remain bound; no recapture or solve was performed by this lane.

Evidence: `accel/receipts/L031_SIX_SOURCE_CLOSURE_run8449.json`. These are source availability and relocation gates, NOT producer admission, numerical equivalence, a speed improvement, or production acceleration adoption. Prior default-off unsolved-prefix GLM results remain unchanged. The next DS4 structural experiment requires restoration of its original-byte input binding: the previous dedicated staged source and baseline roots are absent, and their owner was asked for surviving locations without recapture or full scorer restart.

## DS4 restored-input unsolved-prefix and 16-warp gates (run8451)

The original four L013/E084,E085 down/fused13 K1 configs and all22 original
clean-fit capture bindings survived on the source owner. They were restored
by bounded QSFP transfer and destination hashes, not recapture. The resident
0731 source index physically matched98efab45. Both experiments imported
canonical mainffa38284d57664206ac1f0540d51d7199ebc66dc. Historical K1 remains
distinct from current K3 production.

Unsolved-prefix BMM versus the current grouped/unitwise/structured8-warp
incumbent completed24 builds. Warm four-cell baseline28.355848/28.521460s
versus candidate27.927281/28.157272s gives aggregate1.014135x, not a substantial
build win. LDLQ remains dominant: baseline25.930913/26.105558s versus
candidate25.490609/25.703773s, counting each batch once. Setup43.797747s versus
27.909833s is order/shared-JIT-confounded; baseline included11.726618s packed
conformance, so these setup walls are NOT cold-JIT speedup evidence. Maximum
allocated/reserved bytes were5,206,696,960/9,233,760,256. Independent canonical
decode/source-NMSE validation passed8/8 under baseline-repeat-derived limits;
maximum decoded and source-NMSE deltas were0. Validation81.191607s is separate.

A candidate-only16-warp scheduling rung reused that sealed8-warp baseline by
digest, with unsolved-prefix OFF in both arms. All12 builds passed. Warm walls
28.884976/29.011167s give0.982402x; reject on performance, not assignment bytes.
Independent8/8 decode/NMSE checks passed with zero decoded delta in4.132856s.
Eight warps remains incumbent. Neither gate changes defaults, establishes
full-model output equivalence, or authorizes production promotion.

Compact measured evidence: `accel/receipts/DS4_UNSOLVED_SUMMARY_run8451.json`
and `accel/receipts/DS4_WARP16_SUMMARY_run8451.json`. Producer/validator roots
and raw receipt hashes are bound there. A frozen DS4 teacher/support/corpus
seam was relayed, but no hidden-state frontier was located in the named prior
owner forward root; no full-forward/scorer replay was authorized or performed.

Separately, the GLM owner reports physical K3 admission of all8 six-tensor
L031 packages from run8449. This is authentic source/producer integration,
not acceleration adoption. A next L032/E000 six-tensor package plus path-only
relocation was sealed for the owner's missing-only request; no duplicate
product solve or calibration capture was performed.

The current-incumbent operator profile (four authentic cells, instrumentation
only) recorded768 `_persistent_prefix_viterbi_generic` calls with25.062639s
self-device time. CPU operator/device-child totals overlap and are not summed
as independent costs. Root: `/dev/shm/t_ebcba52e_profile_ds4_incumbent_run8451`.
It motivates targeting the persistent recurrence rather than another BMM sweep.

Receipt correction: `qtip_batch` had hardcoded K2 prefix4096/branch16 metadata
for every ring. This run's top-level K1 solver receipt correctly reports
prefix16384/branch4; its nested batch metadata is historical and preserved,
not a changed execution geometry. The public batch receipt now derives both
counts from L/K/V. A direct execution test of the receipt expression failed
for K1/K3/K4 before the two-field correction and all10 focused batch tests
pass afterward. This changes metadata only, not arithmetic or assignments.
