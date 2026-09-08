# GLM K1 LDLQ scheduling: representative acceptance

Opt-in configuration only. No solver source or default is changed. Existing canonical selectors reduce K1 CTA warps from 16 to 8 and unroll its four branches. Full branch search, int32 state/backpointer wire, original TLUT, source, fit measure and seed are preserved.

Apply CONFIG_PATCH.json only to L16/K1/V2 singleton configs. Tested source: eb3b97314416440bd835fb8327d60b52b0826bdb. Public entry point physically exercised: banana_smasher.solver_qtip_profile.main(config_path, run_root, layer, profile_mode=False). Owner resident main_many must retain batch_size=1; cross-unit main_batch is not covered. Rebind the actual installed source closure, runner digest and copied run manifest before solving, preserving the source-config identity/seed and fit ledger. public_singleton_run8503.py records the tested rebinding procedure; paths are diagnostic locators, not a command to reuse accepted production output directories.

Measured on independently claimed spark-6 with original L003/E145_down source bytes and 16 clean-fit captures:

- Counterbalanced old1/new1/new2/old2 processes each perform cold+warm full builds. Both arms use the same canonical bounded builder; the baseline uses the transferred original owner Viterbi. This isolates the dominant operator, not arbitrary differences between hosts or full runtime versions.
- Whole process including setup, both builds, output writes and teardown: mean 20.191052 -> 16.679773 seconds, 1.210511x.
- Warm LDLQ: mean 7.158610 -> 5.246590 seconds, 1.364431x.
- Warm complete build: 7.323965 -> 5.406487 seconds, 1.354662x.
- Compilation caches are shared and already micro-warmed; initial compile costs that still occurred remain included. Cold rows, all process walls and rejected micro variants are retained. These are not pristine-cache timings.
- Eight diagnostic outputs match the original accepted owner artifact's decoded matrix exactly. Source weight NMSE remains 0.3226495367066126. Authentic clean-fit output SSE ratio is 1.0 over 65 routed rows across 16 windows. The predeclared allowed bound was 1.0001 plus 1e-12; exactness is an observed result, not an assumed acceptance requirement.
- The final public singleton API also produced the exact original serialized artifact SHA a202fc9d67cd75e7db8d1f5db95d2de502ef0e67f9bf0a69c6ddd1a6e753ee0a; main wall 6.683354 seconds, enclosing setup/validation worker 11.390909 seconds. This is a diagnostic admission, not a new production cell.

No held-out/frozen evaluation was touched. No full-model equivalence or 504 production cells/hour claim follows from this single representative. The original 96-cell s7 baseline remains untouched and is not compared directly against s6 timing. Owner-executed missing-cell rollout is a separate required receipt, pending at publication.

One earlier public main_batch smoke succeeded but exercised cross-unit numerical work; it is excluded from quality acceptance and rollout instructions. The initial micro attempt failed before kernel execution because the historical owner module depended on its transferred qtip_memory module; failure and successor CAS receipts are retained.

Verification: python -m unittest -v test_evidence_run8503 (4 passing tests, including bad artifact hash, duplicate phase and quality regression negatives). The reducer expects the separately retained evidence_run8503 directory with .pt payloads. Raw receipt hashes are sealed by MANIFEST.json; large artifacts stay in the task workspace rather than Git. Dedicated s6 remains retained, all payloads terminal; no foreign host actions, children, crons, cleanup or registry writes.
