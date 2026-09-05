# Single-writer control and metric admission

This directory is explicitly authorized operational tooling, not part of the public smash API. No GPU scheduler is created here. Install only bytes extracted from a tested commit present on canonical main; record deployed commit and per-file SHA. Keep real hosts, paths, owners and artifacts in private state, never in this directory.

## Ownership

Retain the existing default dispatcher as the only decision writer. Retain the existing autonomy sensor job but set no_agent=true; it may write only its local sensor_observation.json. Disable any independent scheduled claim sweeper that writes HOST_CLAIM without owner CAS. Other read-only reporters may remain. Never stop a running foreign process during migration. Archive original scripts, registry files and cron jobs before edits. Modify cron with the supported default-profile CLI, after listing jobs. Do not change profile configs.

Sensor CLI: `python3 controller.py --state STATE_DIR --receipt STATE_DIR/sensor_observation.json`. Fixed bounded read-only SSH programs; registry strings are data, never arbitrary shell commands. Missing roots and SSH failures remain unknown. Frontier identity is (host, root, pattern), not display label. Same-owner duplicate labels collapse; conflicting owners fail loudly. Three unchanged observations request owner inspection, not termination. No deletion function exists.

## State v2

GATES.tsv, QUEUE.tsv and FRONTIERS.tsv share three columns: id, owner, JSON payload. First line `# id<TAB>owner<TAB>json`; payload must repeat id and owner. Unknown schema fails visibly. Keep one pending gate for every outstanding measurement, even while scientific identity is being obtained. Never silently drop failed rows. Keep queue rows with status pending/blocked/ready, priority, next_action, reason and owner. Resource allocations additionally require resource, half-open start/end, eligible hosts and authoritative claims + SHARDS. A range-only selection is a proposal, not a launch. `select_work` scans all ready rows by priority, excludes occupied hosts and overlapping resource ranges. The dispatcher must acquire authoritative CAS before action; absent claim evidence never authorizes launch.

Gate fields: id, owner, host, marker, result, expected, score_argv. Missing fields retain pending with exact error. expected contains model_index_sha256, artifact_sha256, teacher_sha256, scorer_sha256, exact ordered window_ids, support_sha256, position_policy and reference. The owner must derive expected from the sealed reference, never from the candidate result. Exact window count is necessary but never sufficient: teacher, support and position policy must match the requested reference. Missing registration is actionable work, not a reason to manufacture a value.

Result JSON: identity exactly equals expected; metric is forward_kl; 64 finite nonnegative window_values; value equals their arithmetic mean (equal-weight frozen windows). The upstream scorer must compute the defined KL from distributions and preserve per-window evidence. For a differently weighted protocol, extend explicitly with a test rather than relabel it. `verify_metric` parses and reaggregates; echo, touch, exit=0 and existence alone cannot admit. A metric admission is NOT scientific acceptance. No built-in <=0.75 GLM threshold, and no arbitrary 64-window Q4 GREEN.

`evaluate_gate(gate, reader)` observes only. `evaluate_gate(..., execute=True, owner=...)` is an owner-local bounded scorer helper, not a remote actuator: matching owner and host=local required. It executes registered argv once (no shell), checks return code, reads actual result bytes and validates metric. It does not launch detached GPU work. Long scorers remain with the existing worker/claim and supply readback; sensor must never call execute=True. `retain_gates` writes every row atomically; sole writer must archive prior state and preserve failures. Do not use concurrent registry writers.

## Corrections and priorities

PRE only. Authentic Q4 Balanced64 matched to sealed Q3 precedes unrelated Q2 build work. Windows 0..63 with fresh native teacher do not establish Balanced64. NLL subtraction is not KL and small weight MSE does not establish end-to-end quality. GLM ~0.3-class is an expectation, not an automatic pass at <=0.75. DS4 Q1 and GLM Q1 require bounded native-rest output-quality gates; GLM fused13/expert completeness must be reconciled, not inferred from DS4 counts. No repair, trained lineage or TLUT calibration to hide defects.

## Verification

`python3 tools/autonomy/test_controller.py` is the focused stdlib suite. Fixtures exercise actual subprocess invocation/readback, failure/echo/touch rejection, native/window identity mismatch, failed-row retention, overlap prevention, blocked-head bypass and read-only deduplication. Fixture scores are tests, not real model results. After pinned deployment run exactly one bounded sensor tick and retain its JSON and elapsed time; no sleep-loop monitoring. Report unmeasured scientific acceptance explicitly.
