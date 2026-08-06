# Backpack provider API migration

The canonical Backpack surface now has two layers:

1. Composable provider functions: `generate_backpack_candidate`, `materialize_backpack_assignment`, `price_backpack_candidate`, `predict_backpack_candidate`, and `verify_backpack_candidate`.
2. The unchanged one-plan workflow: `build_backpack(...)` or `smash backpack build`, which calls the public stage APIs in order.

The built-in provider menu contains native MXFP4, QTIP2/2.5/3, and production fixed-D4 K2048/K4096 bindings. D4/D8 fixture-scale vector VQ remains available through `vector_vq_backpack_provider(...)`. Additional packaged quarter-grid QTIP tiers resolve from a serializable declaration; no tier-specific solver or materializer branch is required.

Existing callers of `generate_vector_vq_backpack_candidate`, `generate_qtip_backpack_candidate`, and `materialize_backpack_source` remain valid. New code should use the generic names so one provider declaration controls generation, exact receipt pricing, prediction, materialization, and verification.

Wire prices no longer assume that every byte is additive per cell. Candidate receipts declare `cell_payload_bytes` plus shared `activation_artifacts`; the class-balanced exact solver includes one binary per activation identity and charges it once even when several selected cells use it.
