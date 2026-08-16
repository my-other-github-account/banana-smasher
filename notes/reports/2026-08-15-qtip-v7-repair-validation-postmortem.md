# QTIP V7 repair validation postmortem

Date: 2026-08-15
Status: Evidence-backed campaign retrospective; raw worker receipts remain outside this repository.

## Executive answer

**Validation does improve, but not for every repair arm.** The earlier report was wrong to collapse the campaign into “training loss improves while validation worsens.” The exact official-k2 BALANCED64 validation improved on the successful arms and ultimately crossed the product bar:

- Historical published PRE: **KLD 0.22939197531977115 / Top-1 56,533**.
- `UPDATE_012`: **0.23308353655472713 / 56,491** — validation worsened.
- A2 `UPDATE_001`: **0.21633723306929561 / 56,849** — validation improved.
- batch16 `UPDATE_001`: **0.22044718522477830 / 56,798** — validation improved.
- A2 `UPDATE_003`: **0.19897596057719977 / 57,014** — validation improved and is product-green.

The product bar was KLD `<= 0.20645277778779403` and Top-1 `> 56,700`. The A2 `UPDATE_003` receipt records `product_green: true`, 64 scored windows, 65,536 positions, and a passing serialized-reload check. Its checkpoint SHA-256 is:

`25c90ccda27c90d3732807c77e09dd335f44ec465dba8b3978382b3b090501ba`

The exact official scorer SHA-256 for that terminal is:

`5d86ce215426b2de5124a68d8160a5b14bbe83717ce2e1fa9c3826c342cca6dd`

## The actual issue

The primary issue was **instrument and objective mismatch**, amplified by lineage confusion.

The trainer's in-graph loss was not guaranteed to be the same function of the same student that the official scorer evaluates after routed-k2 serialization and reload. The training path used an approximate/quadratic student representation, while the decisive validation path used the official routed-k2 physical decode and the published-teacher BALANCED64 rail. A falling loss on the former therefore did not imply a falling KLD on the latter. The optimizer could improve the proxy while moving the serialized physical student in a direction that hurt the official scorer.

That explains the contradictory-looking pair:

- The `UPDATE_012` repair improved its local training signal but scored **0.2330835366**, worse than PRE.
- The correctly bound A2 and batch16 arms improved the exact same BALANCED64 scorer.
- A2 `UPDATE_003` reached **0.1989759606 / 57,014**, proving that repair training can improve the real validation rail when the start state, scorer, and representation are correctly bound.

### What the evidence rules out

- **“Repair training always worsens validation.”** False. A2 U1, batch16 U1, and A2 U3 all improved the official validation.
- **“The first-step LR blast is always the cause.”** False. The A2 and batch16 arms improved with different low-step recipes. A large first step may damage a particular arm, but it is not the universal explanation.
- **“The untrained-start score is the canonical U17 start.”** False. The untrained run was a diagnostic branch. The canonical continuation was later corrected to start from the authenticated repaired U16 checkpoint plus its sparse Adam state, without replaying U1-U16.
- **“A PASS receipt means product success.”** False. `PASS` means the scorer completed. The acceptance fields are the explicit quality comparison, `product_green`, complete scope, and serialized-reload evidence.

## Scope-separated measurement ledger

These rows must not be mixed. The BALANCED64 rows use the exact 64-window published-teacher validation rail. The TRAIN20-23 rows use a 4,096-position training bank and are useful for training diagnostics only; their KLD and Top-1 counts are not comparable to BALANCED64.

### Exact BALANCED64 validation rail

| Arm | KLD | Top-1 | Positions | Windows | Interpretation |
|---|---:|---:|---:|---:|---|
| Published PRE | 0.22939197531977115 | 56,533 | 65,536 | 64 | Authoritative baseline |
| `UPDATE_012` | 0.23308353655472713 | 56,491 | 65,536 | 64 | RED: worse than PRE |
| A2 `UPDATE_001` | 0.21633723306929561 | 56,849 | 65,536 | 64 | Improved; not yet product-green |
| batch16 `UPDATE_001` | 0.22044718522477830 | 56,798 | 65,536 | 64 | Improved; not yet product-green |
| A2 `UPDATE_003` | 0.19897596057719977 | 57,014 | 65,536 | 64 | **Product-green** |

The A2 `UPDATE_003` terminal also recorded `validation_kld_improved_vs_pre: true`, `serialize_reload: true`, and `serialized_reloaded: true`. The old U5 liar-student yardstick, KLD `0.226162314683653`, is explicitly marked in the receipt as the wrong yardstick and must not be used in this table.

A second untrained diagnostic family also existed:

- Untrained published-teacher diagnostic: **0.31633586839760885 / 56,566**.
- Untrained V1 diagnostic: **0.32362727364135740 / 54,449**.

Those two untrained rows are not interchangeable with each other or with the canonical repaired-U16 continuation. Their existence is itself a warning that “untrained start” was not a single stable identity.

### TRAIN20-23 diagnostic rail

| Arm | KLD | Top-1 | Positions | Interpretation |
|---|---:|---:|---:|---|
| `UPDATE_001` | 0.36647892312918270 | 3,388 | 4,096 | Training-bank diagnostic |
| A3 `UPDATE_002` | 0.27764024646740490 | 3,436 | 4,096 | Training-bank diagnostic |
| A3 `UPDATE_003` | 0.26203924007511940 | 3,443 | 4,096 | Training-bank diagnostic |
| `UPDATE_016` | 0.36923815519442693 | 3,373 | 4,096 | Training-bank diagnostic |
| A3 | 0.32006317469201730 | 3,401 | 4,096 | Training-bank diagnostic |

These values show movement on the training rail, but they do not answer whether the published-teacher BALANCED64 validation improved. The answer to that question comes only from the first table.

## Lineage and start-state mistakes

### 1. The untrained diagnostic was promoted into the canonical story

A worker scored an untrained-start official-k2 artifact at **0.3236272736 / 54,449**. That result was useful: it demonstrated that an untrained or wrong-lineage start is materially worse than the published PRE. The mistake was treating that diagnostic as if it were the canonical U17 continuation.

The correction was to bind the real continuation to authenticated repaired U16 plus sparse Adam state and to state explicitly that U1-U16 must not be replayed. The lesson is simple: every run needs a declared `start_checkpoint_sha256`, optimizer-state identity, lineage label, and “replay or continue” field before it is compared with another run.

### 2. Two untrained baselines were allowed to coexist without an immediate identity warning

The source ledger contained both `0.3163358684 / 56,566` and `0.3236272736 / 54,449` untrained diagnostics. They were different receipt families, not one canonical baseline. A result table that says only “untrained PRE” hides a dangerous ambiguity. Baseline labels must include the exact student identity and scorer family, not just a phase name.

### 3. A stale narrative was carried forward after the start-state correction

The early “wrong start state confirmed” wording was too broad. The correct statement is:

- wrong/untrained start is a confirmed failure mode for the diagnostic branch;
- it is not the start state of the canonical repaired-U16 continuation;
- the canonical continuation must be judged independently from that diagnostic.

The stale wording distorted prioritization and made later U17 status reports appear contradictory.

## Scorer and objective mistakes

### 4. Training loss was treated as if it were official validation

This was the central analytical mistake. The training loss, TRAIN20-23 KLD, and BALANCED64 published-teacher KLD were allowed to appear in the same mental scoreboard. They have different data, geometry, student representations, and sometimes different serialization boundaries.

Prevention:

- Every receipt must carry a machine-readable rail label: `train_proxy`, `train_official`, `validation_balanced64`, or another explicit scope.
- Never call a train-bank improvement a validation improvement.
- Never compute a before/after delta across different scorer, teacher bank, support, position count, or window set.
- A validation claim requires a same-scorer PRE row in the same report.

### 5. The wrong U5 yardstick survived too long

The U5 value **0.226162314683653** came from a liar-student B64 measurement and was later explicitly marked `wrong_yardstick_u5_kld`. It was close enough to the real PRE to look plausible, which made it especially dangerous. It should have been quarantined immediately and removed from any “best so far” comparison.

Prevention: every candidate score must carry the teacher label, student representation, scorer SHA, support, position count, and serialization status. A near-looking number is not evidence of compatibility.

### 6. “PASS” was confused with “accepted quality”

Several mechanical scorer receipts were correctly marked `PASS` because they ran to completion. That status was not a quality verdict. The product decision requires:

1. same-instrument comparison against PRE;
2. explicit KLD and Top-1 deltas;
3. complete window and position coverage;
4. serialized-reload evidence;
5. explicit quality `PASS` or `PASS_TIE` where the schema provides it;
6. product-bar evaluation.

The A2 `UPDATE_003` terminal satisfies the stronger gate. A generic scorer `PASS` does not.

### 7. The validation result was not surfaced immediately

The most embarrassing reporting failure was that A2 `UPDATE_003` had already sealed at **0.1989759606 / 57,014**, but a later loop report said there was no new KLD. The direct terminal receipt was not queried and indexed promptly. The board/status summary was treated as fresher than the source JSON.

Prevention: the post-seal action is deterministic: locate the terminal receipt, verify checkpoint and scorer hashes, append one row to the campaign ledger, and report the before/after pair before doing anything else.

## Artifact and infrastructure pitfalls

### 8. A live PID was mistaken for accepted progress

A worker can be alive while blocked on page cache, waiting on I/O, or repeating a failed initialization path. GPU utilization and a PID are not a scientific receipt. Accepted progress is a new checkpoint, a new sealed window/chunk, or another declared artifact delta.

Prevention:

- inspect PID, GPU, CPU, and I/O together;
- compare artifact and receipt timestamps across two observations;
- require a positive accepted delta;
- do not call a worker healthy merely because it has not exited.

### 9. Missing shared-memory recovery files were misread as scientific failure

One scorer exit was caused by expected recovery artifacts no longer being present at their shared-memory paths. The provider had remapped the members correctly, but the expected files had been evicted, causing a fallback to a stale receipt path. This was an infrastructure/path failure, not a checkpoint-quality result.

Prevention: diagnose missing inputs by exact path, size, and hash; re-stage the intended immutable artifact or use an equivalent task-local path; only classify the science after the scorer actually evaluates the candidate.

### 10. A wrong package was nearly assigned to an available worker

An idle worker had Preview/P832 handoffs available, but those artifacts were not the official-k2 codec needed for this comparison. Filling an idle GPU with an incompatible package would have produced activity without useful evidence.

Prevention: availability is not compatibility. Before redeployment, bind the required codec family, teacher bank, scorer, and artifact manifest. Do not revive a lottery or mix Preview with the official 0731/official-k2 rail.

### 11. Large two-node work was allowed to remain a slow “front” without a measured gate

The all-in-memory two-node build was a plausible optimization route, but it did not have an accepted step-time result quickly enough. A running multi-node build is not progress unless the measured step-time target is reached or the artifact frontier advances.

Prevention: define the first measurable gate before launch; if it misses the bounded gate, change the implementation or redeploy the workers rather than preserving the process for its own sake.

### 12. Shared worktrees made publication and investigation state easy to confuse

The operational workspace and the canonical public repository both contain many unrelated worktrees and dirty files. A broad `git add` or a commit from the wrong checkout could publish private receipts, internal paths, or another worker's changes.

Prevention:

- use a fresh worktree from the intended remote base for each publication;
- inspect `git status`, worktree ownership, and the exact staged file list;
- stage only the named report;
- never use `git add -A` in a shared campaign checkout;
- verify the pushed branch from GitHub after publication.

## Things I forgot to update or should have retired

- I did not update the result ledger immediately when A2 `UPDATE_003` sealed.
- I reported “no new KLD” while a product-green terminal already existed.
- I failed to mark the U5 liar-student value as quarantined in every summary that mentioned it.
- I allowed the untrained diagnostic and canonical repaired-U16 continuation to share an ambiguous “PRE” narrative.
- I did not put the scorer identity, teacher identity, serialized-reload state, window count, and position count beside every headline number.
- I let a training-bank KLD and a BALANCED64 KLD occupy the same informal scoreboard.
- I treated board summaries, process liveness, and worker prose as current truth instead of reading the source receipt.
- I allowed stale status language to survive after the start-state correction.
- I spent attention on keeping fronts busy rather than requiring every front to produce the next accepted artifact or a changed hypothesis.
- I failed to make the first post-seal action “score and report on the exact same instrument.”

## What should happen on every future repair run

1. **Bind identity before launch:** start checkpoint, optimizer state, model index, plane/source manifest, teacher bank, scorer, code revision, and intended rail.
2. **Calibrate first:** score the declared PRE through the exact candidate scorer and require the known PRE number before scoring a repair.
3. **Separate rails in storage and prose:** training proxy, training official, validation BALANCED64, and holdout must be different schemas or namespaces.
4. **Use one same-scorer pair:** PRE and candidate must share scorer hash, teacher bank, support, positions, window IDs, and serialization/reload protocol.
5. **Seal every update with a receipt:** no manual inference from logs or training loss.
6. **Report immediately:** append `update -> KLD -> Top-1 -> delta -> checkpoint SHA -> scorer SHA` as soon as the terminal receipt appears.
7. **Quarantine bad yardsticks:** a receipt marked liar, preview, wrong codec, incomplete windows, or non-canonical start cannot enter the main comparison table.
8. **Treat mechanical failures as repair work:** missing files, stale paths, evicted shared-memory artifacts, and dead owners require diagnosis and recovery, not a scientific RED verdict.
9. **Use accepted-delta supervision:** PID/GPU/I/O activity is evidence of motion, not acceptance.
10. **Publish only scrubbed curated evidence:** raw worker paths, hostnames, private addresses, credentials, and in-flight control-plane dumps stay out of the public repository.

## Final scientific conclusion

The repaired official-k2 candidate is not merely overfitting a training proxy. The exact published-teacher BALANCED64 validation improved on multiple repair arms and reached a product-green terminal at **0.19897596057719977 KLD / 57,014 Top-1**.

The earlier negative result was real for that arm, but it was not a universal property of repair. It arose from a combination of a non-identical trainer/scorer objective and, in one diagnostic branch, the wrong start lineage. Learning-rate choice can affect the size and direction of an arm's movement, but the evidence does not support it as the sole or universal root cause.

This report does not claim that every U17-U64 update is sealed, nor does it claim holdout or whole-model quality. It records the narrower conclusion that the exact official-k2 BALANCED64 validation rail can improve, and that the product bar was crossed by the A2 `UPDATE_003` checkpoint.

## Related public reports

- [`qtip2-root-cause-2026-08-03.md`](qtip2-root-cause-2026-08-03.md) — earlier QTIP2 recurrence and accounting root cause.
- [`2026-08-11-qtip2-v7-attempt9-adoption.md`](2026-08-11-qtip2-v7-attempt9-adoption.md) — public API adoption and exactness boundary.
