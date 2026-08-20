# QTIP2 V7 attempt9 batch-10 adoption

Date: 2026-08-11

Status: **Accepted for Banana API adoption.**

## Scope

This adopts the exact source-only QTIP2 V7 attempt9 producer into the Banana Smasher Python API as `produce_qtip2_v7_batch10`. The producer preserves the measured invocation:

1. `prepare_v7_unit` for every member with an exact shared-Hessian factor cache;
2. ten experts grouped independently for each of `w1`, `w2`, and `w3`;
3. `buffered_ldlq_cross_unit` once per projection group;
4. `finalize_batch_unit` for deterministic physical outputs and packing.

The adopted CUDA path combines packed 2-bit traceback, FP16 `half2` recurrence arithmetic, and exact DP4A parent-index accumulation. It does not use one-member-per-call execution, comparator-derived states, fallback, authentic fill, LUT or scale refitting, hidden FP32 deployment controls, or repair.

## Basis

| Item | Identity |
| --- | --- |
| Candidate source revision | `bb807b9f6ffcae211f9dc779b5b576198c3ac6da` |
| Banana integration base | `7e69927503727c29ec714df35643b6b6a85cec41` |
| Source model index | `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b` |
| Complete-30 terminal | `a134c89e0cbf2e618ef6ec58ea9789dce27cbd607449cfa2ffd8c1645370579a` |
| Adoption authority | `c80d6a74ee53e75731a66796b7f33402f171561233808176dccd4b7395fd0c77` |
| Candidate bundle manifest | `ea6264d323f2b4d74cf3ad6883cd2407c5bf7c1e56c478b4719f776a06ce4238` |
| QTIP K2 Python source | `9934246b97ea20ba100e684d86dfa1716d9f52b2b1e8dff3f69980642f472b8e` |
| Q2 codec source | `274830efb9c3595408a294f5d939a22197d29087fa131d499b0d8a581865e5f9` |
| DP4A + half2 kernel | `17f1241c5fdf866da3cc7f4029ce0a73760b19e4a4fdb1cbdcc6682e0547d59f` |
| Attempt9 benchmark helper | `6f70dc26cd6f2893522b5b8d92cc9f41002c1baf69c2b9715cccd00467670946` |
| Cross-unit helper | `b3697db080c978873864928d259789dd91d33f346b25ffb69a3a86f7b6d9e279` |

## Measured result

The immutable complete-30 mixed-geometry comparison used the same source members, Hessians, frozen V7 recurrence, and physical-output checks in both arms.

| Method | Members | Wall time | Relative speed | Exactness | CUDA tiles | Fallback |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Frozen serial V7 | 30 | 181.739197535906 s | 1.000000000x | reference | n/a | 0 |
| Attempt9 batch-10 DP4A + half2 | 30 | 87.33239316300023 s | 2.0810055805604866x | 30/30 exact | 983,040 | 0 |

Exactness covers assignment states, objectives, packed codes, FP16 SUH/SVH controls, billed scale, FP32 and BF16 physical outputs, and compact-wire reconstruction. The accelerated arm recorded no fallback.

## Banana API contract

`produce_qtip2_v7_batch10` accepts exactly 30 members: ten distinct experts with one `w1`, `w2`, and `w3` member each. It returns the 30 finalized physical rows plus a receipt that records projection group sizes, factor-cache usage, CUDA and extension counters, deterministic packing, source-only execution, and fallback zero.

Focused CPU verification covers:

- exact ten-expert projection grouping and deterministic order;
- cross-unit LDLQ parity with independent-unit fixtures;
- shared-Hessian factor reuse;
- deterministic packing and fallback-zero output contracts;
- measured prepare/group/solve/finalize invocation shape;
- presence of the attempt9 DP4A, `half2`, packed-branch kernel path;
- public package export.

The focused test file passed `7/7` under Python 3.13 with `PYTHONPATH=src`; Ruff and `git diff --check` also passed on the adoption scope.

## Hardware canary

After the Banana candidate bundle was staged, the central producer sealed the same-work hardware canary as `PRODUCT_CANARY30_TERMINAL.json`, SHA-256 `82a4fbaa0bbcdf0680d1b8d7acbb70a8f3c572325ab66e2676235e2d1fbd5eea`. The canary used the candidate manifest's exact QTIP K2, Q2 codec, DP4A + half2 kernel, and attempt9 helper identities. It admitted all 30 members across E000-E009 and `w1`/`w2`/`w3`: 30/30 exact, 983,040 CUDA tiles, fallback zero, no gaps or duplicates, and zero hidden FP32 control bytes.

The required E000 triplet is independently receipt-bound:

| Member | Receipt SHA-256 | CUDA | Fallback | Compact wire |
| --- | --- | --- | ---: | --- |
| `L039/E000/w1` | `a3542aa3a8f23b80552803823ef41552bd7c718bcd12419702e4b21e407b5c3f` | positive | 0 | exact readback |
| `L039/E000/w2` | `77bd097b77dc567d2f57929d595b449e09f83f765b2bf6ea7fb085e5d0de6df0` | positive | 0 | exact readback |
| `L039/E000/w3` | `2184fa57517031840039c638a684282f58bb351cff66fb9fd15170aa8084e913` | positive | 0 | exact readback |

Each triplet member records exact states, packed bytes, FP16 SUH/SVH controls, billed FP32 scalar, assignment-physical BF16, and an independently reconstructed compact-wire BF16 output. Assignment-physical and compact-wire BF16 hashes are retained as distinct evidence axes because FP32 assignment controls and billed FP16 wire controls intentionally round differently; compact readback does not use hidden FP32 controls.

## Evidence boundary

Raw hardware receipts remain outside Git. This report records only immutable receipt names and SHA-256 identities. It makes no broader serving, whole-model quality, or all-layer performance claim.
