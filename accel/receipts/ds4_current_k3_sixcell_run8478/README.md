# Current-K3 six-cell paired output gate

The existing unitwise grouped public `main_batch` incumbent passes the bounded paired output gate. Independent NumPy/float64 normalization and `math.fsum` reproduce all three readouts exactly: baseline1, baseline2 and candidate each have teacher-support KLD 1.7578433325743197 and 548/1024 full-vocabulary top-1 matches. Candidate delta is 0 against the frozen 1e-8 tolerance. All 102 suffix frontier hashes and all six consumed replacement bindings per arm pass verification.

Scope: frozen ordinal 0/window 28, support 8192, 1024 positions. Current-K3 replacements are L009 down experts 90/91/92 and L026 fused13 experts 78/79/80. The retained L008 prefix and other routed weights have historical-K1 ancestry with native rest. This is NOT uniform-current-K3/full-model equivalence, a full64 evaluation, an absolute model-quality endorsement, or production adoption. Exact byte equality is diagnostic, not the acceptance criterion. No recipe was fitted against this frozen window.

## Producer performance already sealed

- Down distinct three-cell matched private-cache cold speedup 1.071806x; warm 1.378596x; 33 independently verified products.
- Fused distinct three-cell matched private-cache cold 34.192152 -> 32.909438 seconds (1.038977x); warm 8.868182 -> 7.914626 seconds (1.120480x); 42 independently verified products.
- Fused warm peak allocation increases 835515392 -> 1607583744 bytes. Cache conditions keep source/prebuilt/OS caches shared; private canonical/Triton/Inductor/CUDA/XDG caches define cold.
- See adjacent `ds4_current_k3_fused_run8474` and prior down receipts for per-arm timing and peaks. These are representative build measurements, not full-model build speedups.

## Consumer cost and identity

The suffix/readout process took 1545.451635 seconds, excluding the reused prefix; CUDA peak allocated 22932329984 bytes, reserved 23607640064 bytes. This is validation cost, never producer-build throughput. The three arms share layer materialization; baseline2 is a reproducibility control, not an independent cold throughput replicate. Detailed materialization/readout phase separation is not available in this terminal receipt; per-layer cumulative clocks must not be summed across arms as independent work.

Producer/consumer runtime pin: 7927793657c7a48e1e02f5cb9e437e1c5705002d. Source index: 98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b. Raw logits/frontiers remain on Spark-6; only bounded metadata was transferred to this publication. MANIFEST.json binds original remote receipt bytes. The independent verifier is stored alongside these receipts for reproducibility; it performs no model forward.

Continuation consumed the live prior-run suffix rather than replaying it. Output PID 1068698/start5395040 and verifier PID1107075/start5550581 are terminal/dead. Spark-6 stays reserved for this persistent acceleration card. Production owner t_68e5892b was asked for an explicit adoption decision in board comment44537; no production configuration was changed. The full task remains open until its wider quality and adoption requirements are met.
