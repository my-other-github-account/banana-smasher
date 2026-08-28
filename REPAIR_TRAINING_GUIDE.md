# Repair Training — Reproduction Guide & Field Manual

**Status: validated 2026-08-26** — this recipe produced the first accepted full-64 Balanced64 improvement:
**PRE 0.2292069946743951 → U45 0.211277616743619 (−7.8%)**, Top-1 56,508/65,536, scored on the
sealed layerwise rail (4-shard fan-out, 4/4 PASS, checkpoint `ffb796ce…`, basis `98efab45…`).

This document is the complete how-to: the harness, the exact recipe, why every element is there,
and every trap we hit on the way (each one cost real GPU-days — do not rediscover them).

---

## 1. What repair training is (scope)

Post-quant **function-space repair** of a routed-only K2 (2-bit QTIP) DeepSeek-V4-Flash artifact.
The routed expert weights stay frozen quantized codes. Training updates ONLY the sanctioned
trainable surfaces that are part of the artifact spec:

- **LUTs** (per-layer QTIP codebooks, FP32, artifact-shaped)
- **norms** (RMSNorm scales at artifact dtype)
- **outputs** (lm_head / output projections at artifact dtype)

Because updates rewrite values in-place, the serving artifact size is byte-identical pre/post —
this is what keeps the result apples-to-apples with the Evals table rows at matched footprint.

## 2. The harness

- Repo: `repair_api/` (canonical: github.com/my-other-github-account/banana-smasher; interim canonical
  `ds4-flash-kldmatrix/repair_api@main`). Deploy by git pin ONLY; every launch receipt echoes pin+SHA.
- Entry: `ResidentRepairAPI` (public API — admission, training, validation verbs). One explicit
  checkpoint path+SHA per verb; mismatch refuses.
- Topology: two-rank resident (2 Sparks, layer split rank0 L0-20 / rank1 L21-42), NCCL over QSFP
  (192.168.200.x), `LimitMEMLOCK=infinity` on the systemd units, `MemoryMax<=105G` per rank.
- Reused machinery: `banana-smasher/src/banana_smasher/` (loader, contract, backpack dims).

## 3. The exact recipe that produced U45 (all elements load-bearing)

1. **Start**: published PRE checkpoint `f9bffe04…` + fresh optimizer state.
   - **Zero-update gate (mandatory)**: score the loaded start BEFORE any update. It must read the
     sealed PRE numbers (W28 singleton 0.1364830…). A poisoned start fails here, not at U40.
2. **Loss**: the tailfix `_token_kld` — teacher-forced KL on the teacher's top-8192 support,
   **support-renormalized** (gather → logsumexp over the support). This is scorer-exact.
   - **Instrument-before-experiment**: the loss evaluated at PRE/U0 must read ~0.137-class (the
     known true W28 KL). If it reads anything else (we once had 0.22, and 13.x), the loss path is
     broken — do not train.
3. **Teacher**: FP8 e4m3 own-base t8192 rows, **bound by SHA per window** (accepted W28 teacher
   `56175348…`). Basename/path is NOT identity — we lost days to a foreign teacher file
   (`4d00819d…`) with the right name.
4. **Data**: broad rotation over the full TRAIN corpus (SHA `16575…`), many windows per update,
   rotating — NEVER a fixed small window set (see Gotcha #1).
5. **Gradient scale**: the BF16 pre-backward underflow is the root cause of every early
   detonation (Adam moments exploding to e46, the 2^-96 scaling hack era). The fix: keep the
   pre-backward path out of BF16 underflow territory (FP32 loss reduction), no magic scale hacks.
   - Per-param-class LR calibration from the measured gradient A/B (old-vs-port ratios:
     LUTs 4.6×, norms 4.3×, outputs 5.2× — divide if porting from tailfix-era LRs).
6. **Optimizer**: Adam with FP64 moments. **Gotcha**: `Optimizer.load_state_dict` casts FP64
   moments to FP32 masters on resume — rescale/reconstruct AFTER load (fix committed `9671d7c`-era),
   or moments overflow to inf on the first resumed step.
7. **Validation cadence**: every boundary (~4 updates) score held-out windows the model never
   trains on. GREEN only while the held-out series decreases. Kill after two flat/rising boundaries.
8. **Checkpointing**: sealed markers per update; resume never replays accepted updates.
9. **Final verdict**: full-64 Balanced64 on the **sealed layerwise rail** (4-shard fan-out,
   `Evals/protocols/BALANCED64_PRE_REPRO.md`), ~1h wall. W28 alone is NEVER the verdict (Gotcha #2).

## 4. Gotchas (each cost us days — in order of blood spilled)

1. **Fixed-window diets memorize, not repair.** The early campaign trained on the SAME two
   windows ([28,56]) for 48 updates and validated on one of them. Loss went down (memorization);
   the aggregate was untouched/damaged. Single-window gradients are **mutually destructive**:
   one update on [56,56] from a healthy U40 detonated W28 from 0.1318 to 13.7 (100×), symmetric
   in both windows, gradient directions near-orthogonal (dot≈0.44). Broad rotation is not a
   tuning choice — it is the difference between training and vandalism.
2. **A fixture window is not the aggregate.** Stride-class scorers produced EXACT W28 while the
   full-64 was +2.5% wrong. Training W28 improvements (−3.9% at old-U40) coexisted with zero
   aggregate improvement. Only full-64 counts.
3. **The loss must be numerically verified against a known value before ANY arm.** We ran ~6 LR
   arms on a loss that read 100× wrong (support-renormalization inputs mismatched) and later a
   port that read 0.22-vs-0.137. One 20-minute A/B (old code vs new, identical inputs) ends all
   ambiguity — run it FIRST.
4. **Gradient underflow masquerades as recipe failure.** BF16 pre-backward underflow produced:
   glacial convergence, moment blow-ups (5.6e6 → 7e23 → e46), an amplitude cliff exactly at 1.0×
   (1.008× nonfinite!), and "LR doesn't help" symptoms. If tiny LRs and huge LRs both fail
   strangely, check the gradient dtype path before touching hyperparameters.
5. **Optimizer resume cast bug**: FP64 moments silently cast to FP32 on `load_state_dict` —
   post-load rescale then explodes. Reconstruct moments at FP64 after load.
6. **Teacher/corpus identity by SHA, always.** Two corpora exist (TRAIN `16575…`,
   eval `5aada…`) and multiple teacher copies with identical basenames float around /dev/shm.
   Every receipt must echo (checkpoint SHA, corpus SHA, per-window teacher SHAs). Volatile
   /dev/shm staging is rebuilt-on-reboot — never authoritative.
7. **Zero-update score gate**: two "promising" arm families turned out to start from a poisoned
   U16 (zero-update KLD 15.5 instead of 0.226). Every start must be scored before update 1.
8. **Patch the code the runtime executes.** Deployed bundles freeze provider assets
   (`repair_api/assets/…`); commits to the canonical tree do NOT reach the runtime unless the
   bundle is rebuilt. Five scorer "fixes" changed nothing because they patched non-executing
   files. Verify with in-process introspection (`module.__file__`, `co_filename`) that your fix
   is IN the running process before believing any result.
9. **Host safety**: 121G Sparks die (SSH-dark, OOM) at ~112G+ resident. `MemoryMax=105G` per
   payload, `LimitMEMLOCK=infinity` for NCCL. Host-death is ALWAYS memory, never hardware.
10. **Two-rank geometry is part of the science.** Keep the scorer/trainer geometry fixed
    (mb, window pairing, layer split). Changing co-batching changes results (~2% class) on this
    architecture — treat geometry as identity, not as a tuning knob.

## 5. Reproduction, step by step

```bash
# 0. Pin: canonical repair_api @ <pin>; verify basis 98efab45… and checkpoint f9bffe04… by SHA.
# 1. Stage: QSFP-copy checkpoint + TRAIN corpus + teacher bank to both ranks; verify SHAs local.
# 2. Boot: run `smash improve <artifact> --checkpoint-sha <sha> --run-root <durable-root> --updates 45`.
#    The CLI launches PRE, training, and POST in three fresh processes with sealed receipts.
# 3. PRE/loss gates: restore the authenticated zero-update score; loss must be ~0.137-class.
# 4. Train: broad rotation, FP32-safe backward, FP64 Adam, per-class LRs, sealed marker/update.
# 5. Every 4 updates: held-out multi-window validation; kill if not decreasing twice.
# 6. POST: fresh process restores the trained checkpoint; command accepts only POST < PRE.
```

The U45 acceptance receipt chain lives in the kanban board (t_e12abea8 / t_f76a1035 lineage)
with per-shard terminals and the fan-in pooled receipt.

## 6. What is still open

- Continuation past U45 (dose-response toward <0.19).
- Resume-equivalence inside a healthy recipe (continuous-N vs N/2+reload+N/2).
- The resident fast scorer's per-window parity vs the sealed rail (window-specific,
  sign-mixed deltas; sealed rail remains the verdict instrument until closed).
