# Package-builder breakable-graph prerequisite

Date: 2026-08-03

The first no-cache `linux/arm64` image build from commit
`8a3cd1816491f73b75962ad278f52df9caf1f72e` reached the package-builder
wheel test gate and failed closed before any model boot or pack allocation.
The gate reported 6 failures, 92 passes, and 8 skips because native-plane
construction requires `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, while that variable
was only declared in the later runtime stage.

The corrected package-builder test command supplies the same required value
already baked into the runtime stage. A static contract test now requires the
prerequisite to appear before the package-builder pytest invocation. This does
not add an eager or stock-kernel fallback and does not weaken the native-plane
fail-closed behavior.
