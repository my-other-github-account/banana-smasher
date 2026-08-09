# EXL3 K2 scope ledger decision

## Decision

- EXL3 K2 current whole-L034 result: **Top-1 1,968/2,048 (96.09375%)**, then mean support-renormalized natural-log KLD **0.018689766940723482**; top1=`1968/2048`; kld=`0.018689766940723482`; wire_bpw=`2.0117225646972656`; measurement_key=`ff0731/exl3-k2/l034-experts000-255-fused13-down/rows10-12/positions1024/support8192/eagerteacher/scorer0fa1c72b`; bank_positions=`rows10-12:2048`; intervention_scope=`l034-experts000-255-fused13-down`; support=`8192`; scorer_sha256=`0fa1c72b095479f7dc99264eae9ae25ad3ccd9dac22b85dbb62e15a081305e25`; terminal_sha256=`9b671c0926a17c2449b71cf602df767cf5f69a147807bc865235f4b9d876200c`; artifact_sha256=`12dfb4cdb961fbe53a8c741a009825e4a575ff954a7e347092f558babe54559b`.
- EXL3 K2 historical two-cell Full Train8 result: **Top-1 7,983/8,192 (97.44873046875%)**, then mean support-renormalized natural-log KLD **0.007272738103343451**; top1=`7983/8192`; kld=`0.007272738103343451`; wire_bpw=`2.0117225646972656`; measurement_key=`ff0731/exl3-k2/l000-e000-e001-down/rows10-12-19-24-37-45-57-60/positions1024/support8192/teacher31c5ec76/scorer704f34b2`; bank_positions=`rows10-12-19-24-37-45-57-60:8192`; intervention_scope=`l000-e000-e001-down`; support=`8192`; scorer_sha256=`704f34b24f86f0c3cd00be65b0bfb2a36bbbfb9a367294ad55dee0733c255101`; terminal_sha256=`54c4c751728689f377b8772e48da7bbb7096ee449ed1e3d9624330df56b0b49c`; artifact_sha256=`ec067e5c859ac2a0699c562b64ba0a9dd679d18783c19f933a34035f12784975`.

The current ordinary Q2 RMS one-encode control ties the current endpoint comparator at Top-1 1,968/2,048, then loses on KLD: 0.024969158662642835 versus 0.018689766940723482. Gate `cd30688a38e863921525e233548ce0ecd3326a33a11ca7d741f11672ae3ee395` is **RED** and does not authorize Full Train8 expansion.

The two measurements above are not interchangeable. They change different tensors, use different row banks, and use different scorer/reducer programs. A bare family label is not a comparison key.

## Canonical rows

| Use | Scope key | Intervention | Row bank | Top-1 first | KLD | Selected wire | Terminal SHA-256 | Artifact SHA-256 |
|---|---|---|---|---:|---:|---:|---|---|
| **Current endpoint comparator** | `ff0731/exl3-k2/l034-experts000-255-fused13-down/rows10-12/positions1024/support8192/eagerteacher/scorer0fa1c72b` | L034, all 256 experts, fused13 + down; 512 changed tensors | 10, 12; 1,024 positions each | **1,968/2,048 (96.09375%)** | 0.018689766940723482 | 2.0117225646972656 bpw | `9b671c0926a17c2449b71cf602df767cf5f69a147807bc865235f4b9d876200c` | `12dfb4cdb961fbe53a8c741a009825e4a575ff954a7e347092f558babe54559b`; top1=`1968/2048`; kld=`0.018689766940723482`; wire_bpw=`2.0117225646972656`; measurement_key=`ff0731/exl3-k2/l034-experts000-255-fused13-down/rows10-12/positions1024/support8192/eagerteacher/scorer0fa1c72b`; bank_positions=`rows10-12:2048`; intervention_scope=`l034-experts000-255-fused13-down`; support=`8192`; scorer_sha256=`0fa1c72b095479f7dc99264eae9ae25ad3ccd9dac22b85dbb62e15a081305e25`; terminal_sha256=`9b671c0926a17c2449b71cf602df767cf5f69a147807bc865235f4b9d876200c`; artifact_sha256=`12dfb4cdb961fbe53a8c741a009825e4a575ff954a7e347092f558babe54559b` |
| Historical two-cell Full Train8 | `ff0731/exl3-k2/l000-e000-e001-down/rows10-12-19-24-37-45-57-60/positions1024/support8192/teacher31c5ec76/scorer704f34b2` | L000 E000/E001 down only; 2 changed tensors | 10, 12, 19, 24, 37, 45, 57, 60; 1,024 each | **7,983/8,192 (97.44873046875%)** | 0.007272738103343451 | 2.0117225646972656 bpw | `54c4c751728689f377b8772e48da7bbb7096ee449ed1e3d9624330df56b0b49c` | `ec067e5c859ac2a0699c562b64ba0a9dd679d18783c19f933a34035f12784975`; top1=`7983/8192`; kld=`0.007272738103343451`; wire_bpw=`2.0117225646972656`; measurement_key=`ff0731/exl3-k2/l000-e000-e001-down/rows10-12-19-24-37-45-57-60/positions1024/support8192/teacher31c5ec76/scorer704f34b2`; bank_positions=`rows10-12-19-24-37-45-57-60:8192`; intervention_scope=`l000-e000-e001-down`; support=`8192`; scorer_sha256=`704f34b24f86f0c3cd00be65b0bfb2a36bbbfb9a367294ad55dee0733c255101`; terminal_sha256=`54c4c751728689f377b8772e48da7bbb7096ee449ed1e3d9624330df56b0b49c`; artifact_sha256=`ec067e5c859ac2a0699c562b64ba0a9dd679d18783c19f933a34035f12784975` |

Both rows bind basis `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` and support width 8,192. That shared basis/support does not erase their intervention, bank, or scorer differences.

## Current rows2 comparisons

Top-1 determines the first gate; KLD determines the second.

| Gate | Candidate | Candidate Top-1 | K2 Top-1 | Candidate KLD | K2 KLD | Decision |
|---|---|---:|---:|---:|---:|---|
| `cd30688a38e863921525e233548ce0ecd3326a33a11ca7d741f11672ae3ee395` | ordinary Q2 RMS, one encode | 1,968/2,048 | 1,968/2,048 | 0.024969158662642835 | 0.018689766940723482 | **RED**: Top-1 tie, KLD loss; top1=`1968/2048`; kld=`0.018689766940723482`; wire_bpw=`2.0117225646972656`; measurement_key=`ff0731/exl3-k2/l034-experts000-255-fused13-down/rows10-12/positions1024/support8192/eagerteacher/scorer0fa1c72b`; bank_positions=`rows10-12:2048`; intervention_scope=`l034-experts000-255-fused13-down`; support=`8192`; scorer_sha256=`0fa1c72b095479f7dc99264eae9ae25ad3ccd9dac22b85dbb62e15a081305e25`; terminal_sha256=`9b671c0926a17c2449b71cf602df767cf5f69a147807bc865235f4b9d876200c`; artifact_sha256=`12dfb4cdb961fbe53a8c741a009825e4a575ff954a7e347092f558babe54559b` |
| `476ee64e7e919bfaa851ccc8e0e1e3e760831dd547e7c9d1dfdc837b2126a0da` | earlier shared-method Q2 | 1,959/2,048 | 1,968/2,048 | 0.02153485798949228 | 0.018689766940723482 | **RED**: Top-1 and KLD loss; top1=`1968/2048`; kld=`0.018689766940723482`; wire_bpw=`2.0117225646972656`; measurement_key=`ff0731/exl3-k2/l034-experts000-255-fused13-down/rows10-12/positions1024/support8192/eagerteacher/scorer0fa1c72b`; bank_positions=`rows10-12:2048`; intervention_scope=`l034-experts000-255-fused13-down`; support=`8192`; scorer_sha256=`0fa1c72b095479f7dc99264eae9ae25ad3ccd9dac22b85dbb62e15a081305e25`; terminal_sha256=`9b671c0926a17c2449b71cf602df767cf5f69a147807bc865235f4b9d876200c`; artifact_sha256=`12dfb4cdb961fbe53a8c741a009825e4a575ff954a7e347092f558babe54559b` |

Both gates consume terminal `9b671c0926a17c2449b71cf602df767cf5f69a147807bc865235f4b9d876200c`; neither consumes the historical two-cell terminal.

## Full Train8 row closure

The historical aggregate was independently rebuilt from the eight immutable candidate and teacher payload pairs. Each payload hash matched its manifest, and float64 scoring plus `math.fsum` reproduced Top-1 7,983/8,192 and KLD 0.007272738103343451 exactly.

| Row | Top-1 | Rate | Mean KLD | Candidate payload SHA-256 | Teacher payload SHA-256 |
|---|---:|---:|---:|---|---|
| 10 | 1,012/1,024 | 98.828125% | 0.002391004574826025 | `7d8d6b0e3942a8aff996dbda1bf64be05ff6079c6d2f3e2156c93af34a58ec29` | `b2f0fcd9ba40a22c3bd395a5c38fce727b929722a5a5f30bbfbac1dc8ed5a656` |
| 12 | 973/1,024 | 95.01953125% | 0.02462181041041953 | `bda843bbc24b29651fe4e633b0f2b549da38266ab370dc92f9796b85e30baa93` | `a3cc101890449488729160b5c1d52d4070c572c8a08ae38df8d208c889e096a3` |
| 19 | 991/1,024 | 96.77734375% | 0.006427243053653169 | `10e4955f5ad0e7c2e6076f9cf10396bb57c6d80b91aeec7b9b08d714c6f9f642` | `3f4ec10fb4656ab9c5e85f32cbfe8757a67575977fa72582358bcb28818c0c8e` |
| 24 | 1,006/1,024 | 98.2421875% | 0.006262797505077219 | `c6daf2cf24f99a4a501f35852c81ba83a42981bae504962637bdd3448e4b822c` | `7e292743acf28289a0c83ea6828ac593d68e5b8eabb6a67e07b36464c9b80f86` |
| 37 | 997/1,024 | 97.36328125% | 0.007455668874977674 | `e426f9184ef3c71b19f4279621f4a7c38be9e72090b38ab61362bba49b221d09` | `bbfb8a46960b4cb3990131a43395f77f5a118403cda55db8bc08b58cc2224c59` |
| 45 | 996/1,024 | 97.265625% | 0.004797843676874159 | `d5d9f3b14c750f31c4a2bfc57a8c4ece674ed53e156406cae8cdb68c0cf8ee84` | `7732daaa8d4b616900bfcda5106ae154d0f18219b966c6c9d6cdfb877c0fb574` |
| 57 | 1,004/1,024 | 98.046875% | 0.0035891696829442924 | `1df164f7cd3dacc481b2f2bb99bba44a3e5839addafc4c76a3a0d001230f6b34` | `e2eadcdb46c2533a0425b86a9b65e3b1d45e499257b1793038749bc058bac281` |
| 60 | 1,004/1,024 | 98.046875% | 0.0026363670479755408 | `f8bfda695726d2e02a75b357f7d7730fb0216d56ebeeb1e903f694f06d0672bc` | `40404acd80c768bab084b1bd3a21f0a51c67476153f308adbf2347043cd2f8be` |

The whole-L034 result does not expose equivalent candidate sidecars or per-row counters. Its terminal persists only the two-row aggregate, so a per-row reconstruction would require a prohibited evaluation rerun. The ledger records this as `NOT_PERSISTED`, not as zero or inferred data.

## Evidence identities

### Current whole-L034

- Base/index basis: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Terminal: `results/attempt14-exl3-k2-eagerteacher/TERMINAL.json` at `9b671c0926a17c2449b71cf602df767cf5f69a147807bc865235f4b9d876200c`
- Whole-layer artifact: `12dfb4cdb961fbe53a8c741a009825e4a575ff954a7e347092f558babe54559b`
- Candidate payload: `17427008ac6bfd28371fe5c11d549918abb038bfb90023d32ccf8e924fdd7758`
- Roster: 512/512 target tensors, digest `499f9e139f05376ec976ba2524db4868637c6dda11c27284ee211b4a7866983e`; zero nontarget changes
- Teacher support: `5144020ccea986c85f2a6c21c2f6f409926681af2cd2f6634aa5d173ef4e1917`
- Prefix identity: bound payload `1031f9a134501f88345337efeb3b26ede5b81fbae680ad9c50b3f3e5a58c57b8`
- Suffix identity: bound L034-L042 shard `349f5259193638c30bd1de69d31c7f98a0bdde12d65f554f79a79470f2452efd`
- Scorer: `0fa1c72b095479f7dc99264eae9ae25ad3ccd9dac22b85dbb62e15a081305e25`; teacher-support argmax match plus natural-log support-renormalized KLD
- Reducer: `71b74c0e0a289143db3e3cd939cb60902d0ecbd69d81232f23521d145d3d2d88`; sealed aggregate over the exact 2,048-position bank (per-row counters not persisted)
- Physical selected wire: 12,960,423,936 numerator bits / 6,442,450,944 denominator weights = 2.0117225646972656 bpw

### Historical two-cell Full Train8

- Base/index basis: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Terminal: `receipts/NATURAL_ANCHORS_MATCHED_EXL3_ACCEPTANCE.json` at `54c4c751728689f377b8772e48da7bbb7096ee449ed1e3d9624330df56b0b49c`
- Score artifact manifest: `ec067e5c859ac2a0699c562b64ba0a9dd679d18783c19f933a34035f12784975`
- Candidate payload: `3e1c359f6cebc704cfb798b9eced9af66eec246532763642906392686877a9d8`
- Prefix identity: not applicable; full-model candidate and teacher row sidecars were scored directly
- Suffix identity: not applicable for the same full-model-sidecar execution
- Scorer: `704f34b24f86f0c3cd00be65b0bfb2a36bbbfb9a367294ad55dee0733c255101`; teacher-support argmax match plus natural-log support-renormalized KLD
- Reducer: `dacc4ea7460c8d59e7d4e61386d0203185617c979977e317a61d09954d816a5e`; position-weighted float64 mean over 8,192 positions using `math.fsum`
- Wrapper: `815a4eadaaf83a37e414d1dfed0c7053998c4107dd7acf66fefbc60c35f97e86`

- Teacher artifact: `31c5ec7676916825633017373235822316629b95ca222683265688182e4e9efc`
- Physical selected wire: 33,751,104 numerator bits / 16,777,216 denominator weights = 2.0117225646972656 bpw

## Explicitly missing scope

The routed-only/native-rest Exact64 rows for `EXL3 K2` and `EXL3 K3` remain `MISSING_NOT_A_MEASUREMENT`.
No frozen 64-window, 65,536-position terminal exists yet. Full all-linear terminals and rows8 diagnostics are not substitutes and are recorded in quarantine Q-006.

## Quarantine ledger

| ID | Quarantined material | Why it cannot support a decision |
|---|---|---|
| Q-001 | Claimed Full Train8 terminal digest `54c4c751728689f377b877c7e48da7bbb7096ee449ed1e3d9624330df56b0b49c` | It does not match the located terminal. Direct rehash is `54c4c751728689f377b8772e48da7bbb7096ee449ed1e3d9624330df56b0b49c`, which also closes against the rows. |
| Q-002 | A GREEN narrative attached to gate `476ee64e7e919bfaa851ccc8e0e1e3e760831dd547e7c9d1dfdc837b2126a0da` | Direct rehash matches that digest, but the file itself says RED with 1,959 versus 1,968 Top-1 and 0.02153485798949228 versus 0.018689766940723482 KLD. |
| Q-003 | Reuse of 0.007272738103343451 / 7,983 as the whole-L034 comparator | Those values belong to the L000 two-cell Full Train8 scope, not the L034 all-expert rows2 scope. |
| Q-004 | Old-teacher whole-L034 846/2,048 and 0.31855187054408535 | The old teacher failed zero-control and was superseded by the corrected eager-teacher terminal. |
| Q-005 | Input document `e1b94586aa40a97248f9abb17740286b385e290575e439a955b944b2a6c12d83` | It labels a K2 metric and malformed digest as ordinary Q2. Copied labels are not terminal authority. |
| Q-006 | Routed-only/native-rest comparator treated as already existing | Keep the scope explicitly missing until its frozen window terminal is handed into the ledger; do not substitute full all-linear or rows8 evidence. |

## Reporting rule

Any quantitative sentence or table row naming K2/K3 must carry a ledger-bound measurement key, bank/positions, intervention scope, support width, scorer SHA-256, terminal SHA-256, and artifact SHA-256 on that same line. `tools/validate_exl3_k2_report.py` enforces these bindings and validates the machine ledger's metric closure.
