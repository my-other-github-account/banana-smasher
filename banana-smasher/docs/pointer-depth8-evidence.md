# K1 temporal pointer depth8

Source change: internal K1 (16384 prefixes) pointer addressing uses eight temporal rows of width64 when STEPS is divisible by8. Other prefix counts and nonaligned lengths retain the original flat addresses. Recurrence, wire, and solver recipes are unchanged.

The source file is byte-identical to physically tested private source efd7ad7038d6ea8cb35891b59f94b43a46d7e4ff (qtip_viterbi.py SHA256 aa6f8fa0235e4464529593b068c54be30458fd0e76aff9e68782c5c6289f284b).

Task t_5ade4a57 depth8 pilot: physical200 passed,12/12 frozen clean-PRE reconstruction bounds passed. Follow-up primed ABBA compared canonical9193c991d2039a37b225520138b78fce29d98586 with private depth8 on authentic L004 E242/E243 down and E242 fused13, including public setup/warm API calls. Capacity-only failures split this ABBA across attempts; retained B1 and later C1/C2/B2 were each executed once. B2 additionally trimmed unused startup heap after CUDA initialization (no immediate CUDA-free change observed). Shared source, OS and compiler caches; setup is process-cold, not cold JIT. The interruption and startup treatment prevent treating the small gains as strong isolated causal evidence.

Original fixed arithmetic qualification: all setup/warm/group/process ratios >1,24/24 PRE sse_ratio_vs_true_vq <=1.0001. Aggregate ratios: warm1.0098076285, setup1.0125903634, process1.0093080954. Down warm1.0106860840, fused warm1.0088754858. Quality verifier consumed original packed artifacts through the canonical decoder and clean-fit metric; byte equality was not the acceptance rule. PRE is not heldout output quality.

Raw results and four per-file preservation manifests are bound in task-local DEPTH8_ABBA_VERDICT8626.json. All original output paths and payloads are retained. No campaign, production-rate or new owner-adoption claim follows from this local result. Previously authenticated204-product owner rollout remains evidence for the preceding canonical implementation only.

CPU regression updated depth8 bijection plus nonaligned fallback: RED2 failures then GREEN2. Expanded actual-helper exhaustive test covers batch1/3, tiny/K2/K1, lengths2/4/7/8/16/128. Related focused CPU suite:34 passed,5 skipped (CUDA/Triton unavailable locally); skips are not device evidence.
