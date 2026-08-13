# QTIP V7 resident-weight API

`smash qtip-v7-residency --accounting QTIP_V7_MODEL_ACCOUNTING.json`
reports a conservative `PROJECTED` receipt. Without a physical runtime readback it
reports the direct binding's zero-copy LUT design and 264,192 bytes of projected
source-pointer metadata, but marks parity `UNPROVEN`. It never promotes stored-wire
or source inspection evidence to serving proof.

The direct plugin binding is
`banana_smasher_plugin.qtip_v7_runtime.QtipV7DirectLayer`. It maps each fixed
envelope once, passes the R=2 trellis views to `_v4_moe.qtip2_v7_direct`, aliases
the embedded FP16[1024] LUT, and expands SU/SV/Wscale controls only into bounded
per-call workspace. It has no generic/dequant fallback, decoded-state cache,
second packed copy, or dense weight cache.

Hardware acceptance supplies an explicit JSON readback:

`smash qtip-v7-residency --accounting QTIP_V7_MODEL_ACCOUNTING.json --capture-hardware --hardware-readback QTIP_V7_HARDWARE_READBACK.json --output QTIP_V7_RESIDENT_WEIGHT.json`

`PROVEN` requires direct-kernel dispatch, 43 unique mapped envelopes totaling
69,662,278,656 bytes, zero duplicate/decoded/dense/fallback counters, process and
CUDA telemetry, and physical LUT storage identity when the separate LUT tensor is
zero. Only that zero-copy readback can close full parity at 89,371,076,344 bytes.