# Explicit empty-route clean-fit policy

`solver_qtip_profile` accepts `empty_fit_policy: clean-capture-unit-weight-v1`
in the hashed per-cell configuration. The default remains `refuse`.

The policy applies only if the selected expert has zero routed rows and zero
routed mass across the existing manifest-bound fit capture bank. It uses all
of that same bank's activation rows with unit weights. For the down projection,
activations are computed through the selected expert's original source fused13
weights, as on the established nonempty path. It does not substitute native
weights for a quantized output, alter the TLUT, or introduce training.

The caller must establish that the immutable fit bank is clean and disjoint
from the scoring population before opting in. This option does not make an
unknown or evaluation capture bank clean. The existing Hessian manifest and
capture hash gates remain mandatory. No new capture population is inferred or
selected by the solver.

The solve receipt's fit-source metadata records `counterfactual_fit: true`,
policy name, original zero routed rows/mass, and fallback row count. The source
fit distribution has changed from conditional routing to unconditional clean
calibration tokens; do not claim routed-Hessian equivalence, unchanged fit
semantics, or improved quality. Uniformity refers to actual quantized tier
coverage, not identical fitting populations. Report this qualification with
any resulting uniform score.

Nonempty routed fits and configurations without this option retain their
established behavior. An empty clean bank still fails. No layer/expert is
hardcoded and no evaluation score controls the policy.
