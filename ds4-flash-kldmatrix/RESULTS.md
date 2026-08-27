# DS4-Flash backpack results

This file follows the MASTER DELIVERABLE LADDER in `LIVE_STATE.md`. All reported rows below are measured unless explicitly marked pending. The common evaluation corpus is 512 windows with corpus MD5 prefix `1701920b`; the paired effect-size floor is ±2.6%.

## Ladder summary

| Rung | Deliverable class | Canonical measured result | Verdict | Provenance |
|---|---|---|---|---|
| R1-BASIC | Pure-PTQ backpack | Corrected IQ3: KL_vs_teacher 0.096640 @ 101.95GB; T3EDGE: KL_vs_teacher 0.066274125 @ 111.5GB | IQ3 MISSES T1 by +4.25%; T3EDGE MISSES the NVFP4 0.0594 bar by +11.5726% | ptq-deterministic |
| R2-GPTQ | Function-space error-feedback code assignment | No sealed canonical backpack row yet | Pending | ptq-calibrated |
| R3-REPAIR-SECOND | Basic backpack followed by bin-level e2e-KL repair | **COMBO repaired IQ3: KL_vs_teacher 0.077061044921875 @ 101.95GB** | **PASSES T1 by 16.87%; deployable campaign headline** | bin-repaired |
| R4-PREREPAIR | Repair precision tiers before solving the backpack | No sealed canonical backpack row yet | In progress | tier-repaired/ptq-assigned |
| R5-REREPAIR | Bin-level repair on top of an R4 prerepaired backpack | No measured row yet | Future; requires an R4 artifact | tier-repaired + bin-repaired |

## Flash-Full 0731 crush table — table of record

Teacher for every KLD cell in this section is the **DeepSeek-V4-Flash-0731 FP8 teacher**, under basis gate `98efab45`. These cells must not be mixed with the older preview-basis or llama.cpp/Q8-GGUF instruments elsewhere in this document.

| Artifact | Declared model bytes | BALANCED64 KLD | HOLDOUT512 KLD | C1 plain decode | C1 MTP/DSpark | Current state / provenance |
|---|---:|---:|---:|---:|---:|---|
| Unsloth `UD-IQ3_XXS` 0731 | 104,207,848,032 | **0.17770788160865483** (64/64) | **Paused: 477/512** durable through W476; operator preempted the walk for pre-repair BALANCED64 | Pending production serve reproduction | N/A | BALANCED64 row-set SHA `c2a223203ea5b3d35bb19d6903fba627238c413e67a51923b199afe5d6ff6821`; FINAL SHA `59e9d38dcfeb17ae7e78d826eed9c33ffc5ef06067e8ee7bd4f1fc19554814a8`. HOLDOUT scorer is stopped; rows remain immutable and resume only after the pre-repair BALANCED64 row seals. |
| Unsloth `UD-IQ4_XS` 0731 | 136,662,446,656 | **0.0683488486737012** (64/64) | **Paused: 113/512** durable through W112; operator preempted the walk for pre-repair BALANCED64 | Pending production serve reproduction | N/A | BALANCED64 final SHA `f0ea63a331713993f52a33dfa70fa8cab1b03276c848db0104ff5e3f8ddf4aa0`. HOLDOUT scorer/endpoint are stopped; rows remain immutable and resume only after the pre-repair BALANCED64 row seals. |
| Our BQ3-0731 backpack, pre-repair | Pending STEP5 pack seal | Pending | Pending | Pending | Pending V6 | QTIP2 anchors: `t_aedb517e` is physically solving L000-L008 on s3 (FIRST100 1.7213 s/unit; L000 sealed at 1.6461 s/unit and L001 running), and L015-L042 are sealed. L009-L014 is input-blocked: `t_e4624830` correctly failed rc=2 at 0/3,072 because the staged manifest contains no layer-9 row (blocker SHA `099fb4cd…09e6`); the driver must authorize the exact L009-L014 manifest/Hessian producer handoff before public `smash qtip-configs`. Then STEP5 damage model/knapsack → public `smash export` backpack-pred → BALANCED64 through this same 0731 harness. |
| DwarfStar asymmetric Q2 mix | Pending correct 0731 base+drafter seal | Pending. The prior generic-llama/legacy-GGUF 8-row root is **INVALID** and never aggregates. | Pending; BALANCED64 first | Prior generic-route receipt only: **19.89 ± 0.24 tok/s** plain C1, not promoted for the corrected row | Prior assisted receipt: **13.27 ± 2.67 tok/s** C1 (peak **26.33 ± 4.03**), pending corrected native-route reproduction | Canonical continuation `t_93687ad4` must install `Entrpi/ds4-on-spark` v0.5.0, the exact 0731 base plus matching 0731 drafter, and native `ds4-server`. Paris plus three coherent verbatim samples gate any KLD; no llama.cpp fallback and no reuse of the invalid 8 rows. |

## Behavioral quality — EvalPlus HumanEval(+)

| Row | HumanEval pass@1 | HumanEval+ pass@1 | Protocol | Receipt |
|---|---:|---:|---|---|
| OpenRouter `deepseek/deepseek-v4-flash` reference — 159.63 GB native source / 4.49 bpw | 161/164 (98.17%) | 150/164 (91.46%) | N=1 greedy; EvalPlus 0.4.0.dev44 @ `26d6d00`; HumanEvalPlus v0.1.10; true 4,096 completion cap bound by audited OpenAI-decoder shim; exact-commit Docker sandbox | `7f51b02190c069c69e3daead6890b0952e601463375eb3c11fb3041f54b3c0d5` |
| Served IQ4 `deepseek-v4-flash-ud-iq4-xs` | 161/164 (98.17%) | 152/164 (92.68%) | Same frozen N=1 greedy true-4096 instrument; four disjoint shards; HumanEval/116 empty retained as failure; Spark Docker scoring | `82237887fcbdffe60b02c0c12840e0a1126cccebe39c641e4c0a42633ac4aac6` |
| Served repaired IQ3 16K `deepseek-v4-flash-iq3-combo-v4-step32` | 157/164 (95.73%) | 149/164 (90.85%) | Same frozen N=1 greedy true-4096 instrument; HumanEval/116 and /132 null completions retained once as empty failures; Spark-6 Docker scoring | `ca2e9f8bf44eccf3f7ecacaf38e69655d0c6411d14fa72f56726ea6ba46b0e3f` |
| Served Unsloth UD-IQ3_XXS `deepseek-v4-flash-ud-iq3-xxs` | 158/164 (96.34%) | 151/164 (92.07%) | Same frozen N=1 greedy true-4096 instrument; HumanEval/132 null completion retained once as an empty failure; Spark-1 Docker scoring | `9026f674ea816fba12447cc3f1fd7bd0f859207622ada53965e1ef9dc448e9cd` |

The earlier OpenRouter 140/164 and 133/164 row is superseded: EvalPlus `26d6d00` silently dropped the advertised cap in its OpenAI constructor, so that run used the 768-token default rather than 8,192. The corrected shimmed row above binds and prints the effective 4,096-token cap. OpenRouter provider routing was not pinned or recorded, so its row remains a model-slug aggregate. The IQ4 auxiliary overlap-risk refill was quarantined before merge and contributed zero samples. The IQ3 merge is exactly one sample per task; its two model-level null completions were retained as failures without response retries.

## Master cross-evaluation table

Every row carries explicit ToolEvalBench, HumanEval(+), MMLU-500, KLD, total decimal GB, and total byte-derived bpw cells. `Pending` is retained rather than borrowing a result from a different artifact. KLD and MMLU protocol tags are binding: the UD llama.cpp KLD instrument uses the UD-Q8_K_XL GGUF teacher and cannot be subtracted from fp8-rail KLD; the offline MMLU plane harness, vLLM serve choice-loglik, and llama.cpp MC are separate serving/inference instruments even where they share the frozen 500-qid prompts and choice-token math. Total bpw below is `bytes × 8 / 284.6B parameters`; parenthetical tensor-table bpw is retained where sealed.

| row | ToolEvalBench | HumanEval / HumanEval+ | MMLU-500 | KLD vs teacher | total GB | total bpw | status / receipt |
|---|---:|---:|---:|---:|---:|---:|---|
| OpenRouter `deepseek/deepseek-v4-flash` reference | 85.4 ± 2.2, N=5 (displayed final 86) | 98.17% / 91.46% | N/A — provider-routed API slug, no artifact-bound MMLU row | N/A — provider-routed API slug, no local teacher rail | N/A | N/A | HumanEval receipt `7f51b021…`; provider was not pinned |
| Our repaired IQ3 flagship | **86.60 ± 1.20, N=5 (88/86/88/85/86)** | **95.73% / 90.85%** | **85.0%** (425/500; vLLM serve choice-loglik, frozen qids `24d60b46…`) | **0.0770610** (fp8-source rail) | **101.95** | **2.8658** | ToolEval t_38340b68; HumanEval `ca2e9f8b…`; MMLU seal qrows `7c642544…`; R3 COMBO row |
| Our Q2 | Pending sealed artifact/serve | Pending sealed artifact/serve | Pending sealed artifact | Pending final sealed rail | **95.75** | **2.6915** | Package cells fixed; quality cells wait for the sealed 95.75GB artifact |
| Unsloth UD-IQ4_XS | **86.33 ± 2.87**, N=3 (86/90/83) | **98.17% / 92.68%** | **84.8%** (424/500; offline plane harness `24d60b46…`) | **0.092683** (llama.cpp/Q8-GGUF instrument) | **137.904** | **3.8764** (tensor table 3.880) | ToolEval t_5b2f947b; EvalPlus `82237887…`; MMLU t_1821475e; KLD t_91e811e8 |
| Unsloth UD-IQ3_XXS | **86.0 ± 0.0**, N=3 (86/86/86) | **96.34% / 92.07%** | **82.4%** (412/500; offline plane harness `24d60b46…`) | **0.151021** (llama.cpp/Q8-GGUF instrument) | **102.999887616** | **2.8953** (tensor table 2.898) | ToolEval t_842cdcb8; EvalPlus `9026f674…`; MMLU t_add3bdaf; KLD t_91e811e8; exact 4-shard byte seal |
| Unsloth UD-Q2_K_XL | Pending serve | Pending serve | Pending same 500-qid instrument | Pending llama.cpp/Q8-GGUF instrument | **96.832507552** | **2.7219** (prior tensor-table inventory 2.729) | HF 3-shard manifest sealed; on-disk/KLD/MMLU/serve rows pending |

The UD-Q2_K_XL size is the authoritative three-shard sum `5,256,864 + 49,437,013,568 + 47,390,237,120 = 96,832,507,552` bytes from the Unsloth repository API. It is not yet promoted to an on-disk receipt on an allowed idle host. UD-IQ3_XXS is independently byte-sealed at `102,999,887,616` bytes with all four shard MD5s verified by t_add3bdaf.

## R1-BASIC

| Row | KL_vs_teacher | Size | Top-1 | Target verdict | Receipt |
|---|---:|---:|---:|---|---|
| Lane 1 / corrected-IQ3 | 0.096640 | 101.95GB total / 2.927 bpw | ~0.909 | MISSES T1 (`<0.0927`) by +4.25% | `s4+s7 ~/missions/IQ3_CORRECTED_INCR/out/IQ3_CORRECTED_FULLMENU_MEASURED_ROW.json` |
| Lane 2 / T3EDGE | 0.066274125 | 111.5GB total | 0.92466925 | MISSES the NVFP4 KLD bar (`0.0594`) by +11.5726%; 256K envelope PASS | `T3EDGE_256K_FINAL_REPORT.md` |

## R2-GPTQ

| Row | Result | Status | Provenance |
|---|---|---|---|
| Canonical backpack | — | No sealed row yet | ptq-calibrated |

## R3-REPAIR-SECOND

| Rank | Row | Measured quality | Paired repair delta | Target verdict | Provenance / receipt |
|---:|---|---|---|---|---|
| **1** | **COMBO_ARM_A_IQ3BIN_K4096MENU_REPAIRED_512W_MEASURED** | **KL_vs_teacher 0.077061044921875; top1 0.916632; JS 0.016803; 512 windows @ 101.95GB total / 94.4GiB expert / 2.927 bpw** | **Unrepaired k4096-menu IQ3 baseline 0.0989496484375 → −22.1210%; 501/512 windows improved; FAR ABOVE the ±2.6% floor** | **PASSES T1 (`<0.0927 @ 101.95GB`) by 16.87%** | **bin-repaired; `s5w ~/missions/COMBO_REPAIR_t_fe7ff68e/rail512_A/`; ROW.md5 `5084e3ad1a48dcf7ed2732ea5e21bbff`** |
| 2 | ARM4_IQ3BIN_K4096MENU_REPAIRED_512W_MEASURED_ROW | KL_vs_teacher 0.09224036; top1 0.909956; top1_in_top64 0.998823; 512 windows @ 101.95GB total / 94.4GiB expert / 2.927 bpw | Unrepaired k4096-menu IQ3 baseline 0.09894965 → −6.7805%; 470/512 windows improved; ABOVE the ±2.6% floor | PASSES T1 by 0.4958% below the threshold | bin-repaired; `s1 ~/missions/RAIL512/MERGE_ARM4_IQ3_512/` |

COMBO mechanism: codebooks + all-rmsnorms + attention-output-gains; `n_trainable=1,855,147`; checkpoint SHA-256 prefix `98ec4da4`. Its `clean_496` held-out partition excludes the ARM4 16-window train set and moves from baseline 0.099431 to 0.078414 (−21.14%, top1 0.9156), confirming that the repair generalizes. Corpus MD5 prefix: `1701920b`.

COMBO supersedes ARM4 as the deployable T1-passing headline. ARM4 remains row #2 as an independent measured T1 pass with a different mechanism stack.

## R4-PREREPAIR

| Row | Result | Status | Provenance |
|---|---|---|---|
| Canonical backpack | — | No sealed canonical backpack row yet | tier-repaired/ptq-assigned |

## R5-REREPAIR

| Row | Result | Status | Provenance |
|---|---|---|---|
| Canonical backpack | — | Future; requires an R4 prerepaired backpack | tier-repaired + bin-repaired |

## Per-class KLD (identical held-out windows)

Teacher: **fp8-source top-8192 rows**. Direction is `KL(teacher||candidate)`; both distributions are renormalized on the teacher top-8192 support. This is a **cross-quant comparison on identical window IDs and labels**. Per the calibration seal, class KLD is a **damage-location map, not a quality ranking**.

| Class | Variant | Windows | Positions | Mean | p90 | p95 | p99 | Max | % > 0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agentic | UD-IQ4_XS | 154 | 157696 | 0.102613 | 0.216447 | 0.505788 | 1.767866 | 10.820953 | 5.0604% |
| agentic | repaired IQ3 | 154 | 157696 | 0.084410 | 0.178075 | 0.375875 | 1.353336 | 11.027664 | 3.6596% |
| chat | UD-IQ4_XS | 52 | 53248 | 0.030418 | 0.059922 | 0.103011 | 0.394732 | 4.835150 | 0.7493% |
| chat | repaired IQ3 | 52 | 53248 | 0.033820 | 0.066671 | 0.111783 | 0.345354 | 11.103209 | 0.5972% |
| code | UD-IQ4_XS | 76 | 77824 | 0.054216 | 0.119952 | 0.253806 | 0.847767 | 10.874256 | 2.2397% |
| code | repaired IQ3 | 76 | 77824 | 0.067247 | 0.147834 | 0.289001 | 0.987761 | 12.588837 | 2.5969% |
| prose | UD-IQ4_XS | 78 | 79872 | 0.085025 | 0.190572 | 0.341761 | 0.999021 | 9.384199 | 2.9835% |
| prose | repaired IQ3 | 78 | 79872 | 0.096667 | 0.216301 | 0.369949 | 1.030024 | 10.087885 | 3.2502% |
| reasoning | UD-IQ4_XS | 76 | 77824 | 0.016024 | 0.039980 | 0.063102 | 0.154617 | 3.638194 | 0.1773% |
| reasoning | repaired IQ3 | 76 | 77824 | 0.021450 | 0.047212 | 0.075269 | 0.190526 | 8.318620 | 0.2120% |
| multilingual | UD-IQ4_XS | 76 | 77824 | 0.099108 | 0.247901 | 0.446824 | 1.236114 | 8.004041 | 4.3226% |
| multilingual | repaired IQ3 | 76 | 77824 | 0.137059 | 0.335259 | 0.606910 | 1.706343 | 10.031125 | 6.3361% |

Instrument provenance: frozen 512-window corpus, 1,024 causal-prefill positions per window (524,288 positions total), same `CLASS_BY_WIN.json` and fp8 teacher bank as the repaired-IQ3 all-class audit. IQ4 candidate was the four-shard local Unsloth UD-IQ4_XS GGUF, replayed as raw frozen token IDs with the DSV4 llama.cpp branch (`e3546c7-dirty`) on spark-1 + spark-6 RPC; model shard MD5s and raw-output SHA-256 are sealed in the JSON receipt.

Receipt: `/home/dnola/missions/IQ4_PERCLASS_t_ef275349/out/IQ4_PER_CLASS_AUDIT.json`
