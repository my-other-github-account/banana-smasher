# BALANCED64 PRE Reproduction — Canonical Protocol & Lineage (SEALED 2026-08-22)

## The number
**QTIP2 V7 routed-only pre-repair (published PRE): Balanced64 KLD mean `0.2292069946743951`, Top-1 `56,534/65,536`.**
Matches the published Evals row (0.229392 / 56,533) within 0.08% (kernel/shard-split noise class).

## Sealed receipts (immutable)
- Layerwise single-host: `B1_RAW_U0.json` — SHA256 `30b7dcdae28755b7441345677323635f3f6ebe71d3441a6283c1c0dde7fd957c`
  (spark-2 `/home/dnola/missions/PRE_SCORE_t_9e5a36e1/track-b-s2/receipts/`), scored raw U0, KLD 0.2292069946743951.
- Parallel 4-shard fan-out: `PRE_FANIN_TERMINAL.json` — SHA256 `3d0bbad9fc73c4eaae...` (kanban workspace t_896a422b),
  scored published PRE, 64/64 windows, aggregate 0.22920699467439512 / Top-1 56,534, fallback_calls=0 on all shards.

## Artifact identities (NEVER confuse these)
- **Published PRE checkpoint**: SHA256 `f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70` (serialized pre-repair artifact; the Evals-row subject).
- **Raw U0**: SHA256 `7978d1002d7e4ecfa280f646f70cc76638c0e7bd833cc3cc13a2de999050133f` (same weights different serialization; scores identically).
- Basis: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`. Teacher inventory: `017c7e9261b3...aad04a92`
  (t8192_win*.pt files; W28 file SHA `561753481a1e08aee88e28f5fa0c6e727f4af679494c39679e87ed5189e2653d`).
- Score corpus: `windows_ds4_TRAIN.json`; builder corpus: `windows_ds4_eval.json` (DUAL-corpus convention — score != builder).

## The method that works (window-split fan-out, <15 min/checkpoint)
Run the UNCHANGED sealed builder as N parallel single-host subprocesses with `--windows` subsets
(16 windows/host on 4 hosts ≈ 6-8 min each), fan-in per-window files, aggregate per kld_score renorm convention
(pos-cutoff 1024, support 8192, FP8 e4m3 own-base teacher). Reference implementation: kanban workspace t_896a422b.

## Fixture answer keys (for ANY new scorer/optimization)
- W28 grouped [28,56] mb2: KLD mean `0.13678686618849925` / Top-1 882 (trusted RUN1698).
- W28 singleton [28]: `0.14062129470098408` / 884 (grouping omitted — the historical 2.8% divergence signature).
- W28 via attempt-17 resident pin `12f53abc`: `0.13712959240533734` (0.25%). Per-window rows for all 64: PRE_FANIN_TERMINAL.

## Hard-won laws (violate = repeat our failures)
1. MICROBATCH WINDOW-GROUPING IS SEMANTIC on this architecture: score windows in the sealed mb-groups. Singleton scoring inflates KLD ~2.8% uniformly.
2. `swiglu_limit: 10.0` (model config.json) is CANONICAL — never strip activation clamps.
3. ONE forward implementation: import the sealed builder; never reimplement (weeks lost proving this).
4. Explicit checkpoint path+SHA per invocation; loader re-hashes bytes into the receipt; no indirect resolution.
5. Trainable-form conversion must preserve score identity: converted model MUST read fixture-class BEFORE first update
   (zero-LR control proved conversion, not training, caused all "detonations" — see t_9651651d lineage).
6. Batching windows is mathematically harmless (kernel noise, unbiased); geometry/masking bugs riding along are not — diff per-window vs the fan-out answer key.

## Next steps (live)
- Sub-5-min in-memory scoring: t_4b771037 (batch windows through resident validate(); answer key = fan-out rows).
- Conversion-boundary fix: t_9651651d (trainable-form init must preserve 0.137-class W28).
- Zero-reload train->validate chassis: PROVEN (attempt-26 receipt, t_6031426c) — 83s/window, one process, no reloads.
