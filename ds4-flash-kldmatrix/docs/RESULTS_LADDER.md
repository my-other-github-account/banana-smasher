# Results ladder

Historical cutover: **2026-07-17**. Evidence reconciliation through
**2026-07-18 10:00 PDT**. Rows sealed after cutover are labeled explicitly.

This is a reproducibility ladder, not a single blended leaderboard. It keeps
full-corpus measured quality, small gates, predictions, footprint, task
metrics, kernel timings, and served throughput in separate columns.

## Status vocabulary

- **MEASURED-512** — complete artifact, 512 paired windows, 524,288 scored
  positions, immutable candidate and score ledgers.
- **MEASURED-GATE** — real paired measurement on fewer windows and/or partial
  artifact coverage. Useful for decisions; not a canonical row.
- **PREDICTED** — solver or projection result; not measured candidate quality.
- **PENDING-AT-CUTOVER** — work had not produced the required receipt by the
  2026-07-17 snapshot; a later row may supersede it.
- **POST-CUTOVER-MEASURED** — sealed after the historical cutover and included
  so this document is not mistaken for current state.

## Metric and target definitions

The campaign quality metric is teacher-forced `KL(ref || candidate)` on the
reference top-8,192 support, renormalized, over the first
`min(1024, real_len - 1)` positions of each common evaluation window.

The size/quality targets used by the campaign were:

| Target | Requirement | Interpretation |
| --- | --- | --- |
| T1 | KLD `< 0.0927` at `≤ 101.95 GB` | IQ3-size quality target |
| T2 | KLD `< 0.0927` at `≤ 95.75 GB` | smaller Q2-size target |
| NVFP4 reference bar | KLD `≤ 0.05936` and top-1 `≥ 93.01%` | cross-model reference from an official NVIDIA NVFP4 row; not a same-model formal delta |

The NVFP4 bar is a demanding external reference, not a DS4 teacher-self row.
See `sealed_rows/NVFP4_LOSSLESS_BAR_TABLE.md` for its exact model/protocol
context.

## R1–R5 campaign ladder

| Rung | Artifact class | Quality evidence | Footprint | Status and honest verdict |
| --- | --- | --- | ---: | --- |
| **R1 — basic PTQ** | Corrected mixed-tier IQ3 backpack | **0.096640** (receipt `0.0966397578125`), paired 512-window KLD | **101.95 GB**, 2.927 bpw | **MEASURED-512.** Misses T1 by 4.25%. |
| **R1 — larger PTQ** | T3EDGE mixed-tier artifact | **0.066274125**, top-1 **0.92466925**, paired 512-window KLD | campaign label **111.5 GB**; build receipt **119.16540608 GB** | **MEASURED-512 quality; footprint disputed.** Misses the `0.05936` external NVFP4 KLD bar by 11.65%. Resolve the size-receipt conflict before a fit claim. |
| **R2 — GPTQ/code reassignment** | Five-layer-origin calibrated-code pilot; later overlay receipt counted 2,026 units | **0.092471**, top-1 **0.9092**, windows `0..63`; corrected gate baseline **0.093239**, improvement **0.824%** | Overlay only; no complete new package size | **MEASURED-GATE.** Below the 2% rebuild trigger. It did not become a complete 43-layer/full-bin canonical backpack with a 512-window rail. |
| **R3 — repair after solve, Arm A** | Codebooks + all RMSNorms + attention-output gains | **0.077061** (receipt `0.07706103515625`), top-1 **0.916632**, JS **0.016803** | **101.95 GB**, 94.4 GiB expert payload, 2.927 bpw | **MEASURED-512.** Paired reduction **22.1210%** from 0.0989496484375; 501/512 windows improved. Passes T1 by 16.87%. Campaign headline at cutover. |
| **R3 — repair after solve, V2** | Same mechanism, fine-tuned on a disjoint corpus but warm-started from an eval-trained checkpoint | **0.076285939453125**, top-1 about **0.916** | **101.95 GB** | **MEASURED-512 confirmation.** Paired reduction **22.904%** from 0.09894965; 504/512 improved. It is only about 1.006% below Arm A, within the campaign effect floor. **Not zero-leakage** because of checkpoint ancestry. |
| **R4 — repair tiers before solve, cutover gate** | Three-tier prerepair mechanism gate | **0.088607**, top-1 **0.9091**, 64-window gate; a later current-checkpoint gate receipt was **0.090104** | Gate artifact | **MEASURED-GATE at cutover.** Validated that re-encode + scale-refit translated repair direction; not a canonical backpack row. |
| **R4 — early two-tier backpack** | Two repaired tiers inserted into a complete backpack | **0.095608** pooled over 512 windows (halves: 0.095080 and 0.096137) | Backpack-class artifact | **MEASURED-512, non-final R4.** Only 1.07% below the corrected 0.096640 baseline, using early checkpoints. |
| **R4 — three-tier prerepaired pack, post-cutover** | Exact runtime composition of the three repaired tiers | **0.091723068359375**, top-1 **0.91039658984375**, JS **0.01851071875**; 512 windows / 524,288 positions | Footprint not sealed in the audited result row | **POST-CUTOVER-MEASURED.** Improves 5.0879% over R1 and 4.0634% over the early two-tier row. It passes the KLD side of T1, but the missing size seal prevents a T1 fit claim. |
| **R5 — repair after prerepair, post-cutover** | Step-35 repair composed over the R4 three-tier pack | **0.07506409375**, top-1 **0.916774736328125**, JS **0.016424763671875**; 512 windows / 524,288 positions | receipt reports **193.063787137 GB** (179.804663301 GiB) runtime-composed pack | **POST-CUTOVER-MEASURED.** Lowest R3–R5 repair-lineage KLD in this evidence cut and 1.6017% below COMBO V2, but not the campaign-wide minimum (T3EDGE measured 0.066274) and far outside the 101.95 GB T1 envelope. It is not the IQ3-size winner. |

### Row provenance

Every R1–R5 row above is qualified by the matrix below. Host aliases and home
directories are intentionally sanitized; the durable campaign ledger retains
the originals. `DS4-t8192` means the DS4 source-teacher top-8,192 rail, corpus
MD5 `1701920b4ba96dea0b18fe9df0151876`, scorer MD5 prefix `8011368c`, first
1,024 scored positions per window.

The audited evidence did **not** expose an immutable full teacher-payload MD5.
That missing digest is a publication gap, not a value to infer from the corpus
or scorer hash. Every row below therefore names the teacher convention and the
available row/ledger hashes without pretending that a teacher MD5 was sealed.

| Rung/row | Window set and coverage | Host and UTC seal time | Receipt path | Hash pins and evidence status |
| --- | --- | --- | --- | --- |
| R1 corrected IQ3 | eval `0..511`; 512/512; DS4-t8192 | `<host-A> + <host-B>`; `2026-07-17T05:20:01Z` | `$MISSION_ROOT/IQ3_CORRECTED_INCR/out/IQ3_CORRECTED_FULLMENU_MEASURED_ROW.json` | row MD5 `bbbc3ad7dcec8022ea07301d9449e2c9`; row SHA-256 `35cdb79b…`; KLD-ledger MD5 `e3fd7891190db251da6e6d1ff9f3c476`; manifest MD5 `3681e74aa4369ed4cd8d36844686a876` |
| R1 T3EDGE | eval `0..511`; 512/512; DS4-t8192 | `<host-C>`; `2026-07-17T04:49:53Z` | `$MISSION_ROOT/T3EDGE/T3EDGE_256K_FINAL_REPORT.md` and measured-row receipt | row MD5 `68a71a9f…`; row SHA-256 `520f9cba…`; KLD-ledger SHA-256 `7636abf5…`; manifest SHA-256 `0af9f591…`; build-receipt SHA-256 `5e90bef7…`. **Footprint conflict: campaign label 111.5 GB vs receipt 119.16540608 GB.** |
| R2 calibrated-code gate | eval `0..63`; 64 windows; five-layer-origin pilot; DS4-t8192 | `<host-B>`; `2026-07-17T17:29:48Z` | gate row and overlay seal in `$MISSION_ROOT/GPTQ_OVERLAY/` | row MD5 `fc343cf4f3ac038484aca31ae0977700`; KLD-ledger MD5 `8249965fa3b362399b9af750321c1e2d`; target/base manifests `3681e74a…` / `427dd779…`; overlay seal `d07d5d1b…`. No complete 43-layer/512-window package. |
| R3 COMBO Arm A | eval `0..511`; 512/512; DS4-t8192 | `<host-D>`; `2026-07-17T11:28:14Z` | `$MISSION_ROOT/COMBO_REPAIR/rail512_A/COMBO_ARM_A_IQ3BIN_K4096MENU_REPAIRED_512W_MEASURED_ROW.json` | row MD5 `5084e3ad1a48dcf7ed2732ea5e21bbff`; row SHA-256 `26f75b39…`; candidate-ledger SHA-256 `87a9933b…`; checkpoint SHA-256 `98ec4da4…`; provenance SHA-256 `1cb04f84…` |
| R3 COMBO V2 | eval `0..511`; 512/512; DS4-t8192 | `<host-E>`; `2026-07-17T19:50:21Z` | `$MISSION_ROOT/COMBO_V2_RAIL/rail512/` | row MD5 `d472a4bf…`; row SHA-256 `aaec5ab6…`; candidate-ledger SHA-256 `2081050b…`; provenance SHA-256 `ade18eec…`; fine-tune corpus MD5 `6ed646bba44ce214693c0b7dc610e282`. The 248-window extension has zero document/content overlap, but the warm start inherits evaluation exposure. |
| R4 three-tier gate | eval `0..63`; 64 windows; three repaired tiers; DS4-t8192 | `<host-C>`; `2026-07-17T20:58:25Z` | `$CAMPAIGN_RECEIPTS/R4_3TIER/SCORE_3TIER.json` | score MD5 `cc2e2911…`; score SHA-256 `47cb1cd1…`; KLD-ledger MD5 `2be1b468…`; manifest `3681e74a…`. **Provenance hole:** score text has a progress prefix, `corpus_md5` is null, and no pack receipt exists; retain as a gate, not a hash-clean package row. |
| R4 early two-tier backpack | eval `0..511`; H1 `0..255`, H2 `256..511`; DS4-t8192 | `<host-C> / <host-B>`; completed `2026-07-17T21:10:00Z` | `$CAMPAIGN_RECEIPTS/R4_2TIER/PACK_2TIER_RECEIPT.json` plus half-rail ledgers | no unified measured-row JSON; pack receipt MD5 `c69ddbe5…`; H2 KLD-ledger MD5 `2882461fe7794cd4fe579ef74c2d00db`; H2 score-ledger MD5 `30bf71aa…`; H1 ledger MD5 prefixes `bfd9f44e`, `7b917c03`, `48e5a3e5`, `6a0238c2` |
| R4 three-tier full-512 | eval `0..511`; 512/512; 524,288 positions; DS4-t8192; corpus MD5 `1701920b…` | `<host-E>`; `2026-07-18 01:34 PDT` | `$CAMPAIGN_RECEIPTS/R4_3TIER_FULL512_RESULT.json` | result SHA-256 `76f1efeb8432a5dbed6ac056bd0c445222ed5c7c7c7f5a4ee054fdd44205bb19`; validation PASS 32/32, SHA-256 `9edc7cf245c714a5988d3b1d7e4e3d5cd25f837b8f65af90435b5935a1ddf7d5`; exact gate64 reproduction `0.08860671875`. Receipt is not mirrored publicly. |
| R5 full-512 | eval `0..511`; 512/512; 524,288 positions; DS4-t8192; same R4 baseline ledger | `<host-F>`; `2026-07-18 09:59 PDT` | `$CAMPAIGN_RECEIPTS/R5_FULL512_MEASURED_ROW.json` | row SHA-256 `9baf4ad4c0d9030c20cccad3768c977ee56a62ef18f15974fbcf5b0a3b0620fb`; 512 unique completion rows and 512 unique KLD rows; checkpoint/corpus/instrument/scorer bindings passed. Receipt is not mirrored publicly. |

Ellipses identify hash prefixes preserved by the audited handoff rather than
invented full digests. The repository does not mirror every multi-gigabyte rail
payload. A reader can audit the documented chain above, but a turnkey rerun
still requires the omitted source assets, teacher payloads, and campaign
receipts.

## R3 paired evidence in more detail

### Arm A

- baseline: `0.0989496484375`;
- repaired: `0.07706103515625` (canonical rounded display `0.077061`);
- relative reduction: `22.1210%`;
- windows improved: `501/512`;
- `clean_496` partition: `0.099431 → 0.078414` (`−21.14%`), top-1
  `0.9156`.

The `clean_496` label means those 496 windows exclude the arm's 16 inherited
training windows. It does not prove that every ancestor artifact was trained
without evaluation exposure.

### V2

- baseline: `0.09894965`;
- repaired: `0.076285939453125`;
- relative reduction: `22.904%`;
- windows improved: `504/512`;
- `clean_496`: `0.099431 → 0.077649` (`−21.91%`), top-1 `0.9149`.

V2 fine-tuned on a disjoint 248-window corpus, but it warm-started from a
checkpoint trained on evaluation-pool data. The correct provenance is
“disjoint fine-tuning with inherited evaluation warm start,” not
“contamination-proof” or “zero evaluation leakage.”

## R4/R5 interpretation

The R4 work established a critical export rule: learned codebooks must be
transplanted with their own re-encoded codes and refit scales. Earlier
codebook-only swaps were invalid because only a small fraction of code
assignments overlapped.

The 64-window `0.088607` result answered a mechanism question: the corrected
transplant carried a real quality gain. The early two-tier and three-tier gates
translated roughly 47–49% of the local tier-repair prediction; the campaign
used `0.48` as an empirical pre-reanchor pricing coefficient. The early
512-window `0.095608` row then showed only a modest full-corpus gain.

After cutover, the exact three-tier composition sealed at `0.091723068359375`.
The first 64 windows reproduced the gate within `2.8125e-7`, so the
better-than-preregistered full result was investigated and retained rather than
spun away. R5 repair over that pack then sealed at `0.07506409375`. That is a
real measured quality improvement, but its reported 193.06 GB runtime-composed
footprint makes it a different operating point from the 101.95 GB COMBO rows.

## Reference comparators

These are context, not interchangeable rows:

| Comparator | Reported quality | Size | Provenance | Why it is not a formal same-row delta |
| --- | ---: | ---: | --- | --- |
| UD-IQ4_XS | **0.092683015625** KLD (campaign display 0.0927) | **137.9 GB** | 502 re-chunked llama.cpp chunks; Q8_K_XL GGUF teacher; corpus MD5 prefix `b559b14a`; `<host-F>`, `2026-07-13T00:10:16Z`; row MD5 `f8c3c0f737c3f43ff61aa83ed37f2ea3`; KLD-log MD5 `d5120276…`; GGUF MD5 `70972129…`; harness commit `e3546c79` | llama.cpp uses a Q8 GGUF teacher and re-chunked text; the scorer/window convention differs. |
| Official NVIDIA NVFP4 Qwen row | **0.05936** KLD, **93.01%** top-1 | model-specific | External model/protocol; receipt summarized in `sealed_rows/NVFP4_LOSSLESS_BAR_TABLE.md`; no DS4 host/date | Different model pair; used as a lossless-grade reference bar, not as a DS4 paired candidate. |
| DS4 source-teacher MMLU anchor | **0.844** on MMLU-500 | not a package-size row | 500 questions; question-set SHA-256 prefix `24d60b46`; `<host-C>`, `2026-07-11T15:49:27Z`; `out/mmlu_ladder_s8/M_ref_MMLU500_ROW.json` | Task choice-loglik, not KLD. |

Do not subtract UD-IQ4_XS KLD from a DS4 teacher-rail row. Compare only the
shape/operating point and retain the instrument caveat.

## Task-quality evidence

The repository-local MMLU-500 ladder predates the newest R1–R4 backpack artifacts.
It establishes evaluator behavior for the original W2/W3 matrix:

These task rows share the 500-question set with SHA-256 prefix `24d60b46`
and choice-loglik ground-truth scoring rather than a teacher rail. Per-question
and rollup receipts are repository-local under `out/mmlu_ladder_s8/`; a
standalone publication must include that small receipt bundle.

| Variant | MMLU-500 | Window/evaluator identity | Host/date | Receipt | Note |
| --- | ---: | --- | --- | --- | --- |
| Source-teacher anchor | 0.844 (422/500) | 500 questions; question set `24d60b46`; no KLD teacher/scorer | `<host>`; 2026-07-11 UTC | `out/mmlu_ladder_s8/M_ref_MMLU500_ROW.json` | Same offline choice-loglik harness |
| Q2 RTN | 0.810 (405/500) | same 500 questions and harness | `<host>`; 2026-07-11 UTC | `out/mmlu_ladder_s8/M_Q2_MMLU500_ROW.json` | Offline |
| Q3 RTN | 0.788 (394/500) | same 500 questions and harness | `<host>`; 2026-07-11 UTC | `out/mmlu_ladder_s8/M_Q3_MMLU500_ROW.json` | Offline |
| Calibrated Q2 | 0.810 (405/500) | same 500 questions and harness | `<host>`; 2026-07-12 UTC | `out/mmlu_ladder_s8/M_Q4_MMLU500_ROW.json` | No detectable gain over Q2 at `n=500` |
| Calibrated Q3 | 0.832 (416/500) | same 500 questions and harness | `<host>`; 2026-07-12 UTC | `out/mmlu_ladder_s8/M_Q5_MMLU500_ROW.json` | Statistically indistinguishable from the anchor in the recorded paired test |

These task rows must not be attached to the newer COMBO/R4 artifacts unless
those exact artifacts are evaluated.

## Throughput status

No quality row above should be read as a speed result. At the historical
cutover, the learned-VQ serve measured `2.1439` on the generic path, `2.9894`
on a fresh d4-fast baseline, `2.9691` after no-sync (neutral), and `6.5891`
after grouped d4/d8 batching and pair capping; the `≥10` gate had not passed.

Post-cutover, the opt-in warp-GEMV arm measured `14.13` token/s over a complete
4,096-token stream and `14.93` median on 5×64. The V4-step32 product overlay
measured `13.913` sustained-4K and `13.964` median 5×64, so the operational gate
did pass later. These are serving-stack-specific rows, not speed measurements
of the R5 quality pack. See `docs/SERVING.md` for receipts and caveats.

## Public evidence map

| Evidence | Repository path |
| --- | --- |
| Common corpus identities | `out/DS4_CORPUS_META.json`, `out/DS4_CALIB_META.json` |
| Original six-row matrix and evaluator convention | `SCOREBOARD.md` |
| Public summary at cutover | `RESULTS.md` and this file |
| Mixed-VQ measured receipt example | `sealed_rows/VQ3_MIXED_100G_MEASURED_ROW.json` |
| 96 GB full-menu measured receipt example | `sealed_rows/R7_FULLMENU_96G_NLL.json` |
| NVFP4 reference table | `sealed_rows/NVFP4_LOSSLESS_BAR_TABLE.md` |
| Operational chronology for later rows | `LIVE_STATE.md` |
| R4/R5 post-cutover machine receipts | internal `$CAMPAIGN_RECEIPTS`; filenames and hashes above; not mirrored in this checkout |

Some later campaign row payloads remain referenced by durable ledgers but are
not mirrored into this checkout. The numbers above are documented results, not
a claim that every multi-gigabyte candidate is bundled here.
