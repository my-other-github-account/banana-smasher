# Authentic QTIP1 and QTIP1.5 provider verdict

Date: 2026-08-05

Verdict: **PARTIAL — canonical QTIP1 geometry and fresh exact-FF0731 installed-public-API generation/materialization/runtime decode are validated. Tiny-window quality is weak and direct stock-vLLM K1/V1 execution remains open.**

## Source binding

The authority for the trellis transition and packing contract is Cornell RelaxML QTIP:

- repository: `https://github.com/Cornell-RelaxML/qtip`
- revision: `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`
- source: `lib/codebook/bitshift.py`
- source SHA-256: `a299ae97d2ccc80a142095c3c16ed619b435b68736fd52702ab396bc37218531`

The implementation was developed against Banana Smasher baseline `a2c45b8fa7aea15f355f40ff119044914c9d78d4`, integrated with the generic Backpack provider at `64e7cd0f96daaacb54f4d6cc1475930cc1c762d2`, and exercised from installed-wheel source commit `4b99b173929f8c4ea38f31939c5e0a72286902c8`.

No private campaign code was used to define the K1/V1 primitive.

## Canonical parity result

Authentic QTIP permits `K=1, V=1`. The production declaration is:

- state bits `L=16`
- branch bits per trellis step `K*V=1`
- one reconstructed scalar per state, so the exact code plane is `1.0` bit/weight
- 9-bit shared table index
- canonical `quantlut` state-to-table mapping

An independent source-parity run executed the upstream implementation and the Banana Smasher primitive over the same recognizable `1x32` matrix and shared table. Both produced:

- packed words: `[50608, 48565]`
- packed uint16-byte SHA-256: `8900941f18dd7c64618669e63e765f93be195235d9f55b23a673f4d6817eb408`
- state-index int32-byte SHA-256: `2c6b6f7def6a0a2b1f013955bfb157f9d930947c9f95bace99ce636e0551ffaf`
- reconstruction float32-byte SHA-256: `fdf4b61c76eed55d8cd45de050021c93b38267849d940e8aff0480855d37a668`
- exact pack/unpack state parity: PASS

The fixture is frozen in `banana-smasher/tests/test_qtip1.py`.

## Provider and runtime surface

`banana_smasher.qtip1` adds declaration-driven components rather than a `qtip1_5` algorithm branch:

- `qtip1`: four quarters of authentic `L16/K1/V1`
- `qtip1_5`: two quarters `L16/K1/V1` plus two quarters existing `L16/K2/V2`
- deterministic ring assignment over `(layer, expert, projection)` identities
- generic declaration parser and count reporter
- one physical shared-TLUT artifact with explicit per-component column selection and decode mode
- exact cyclic trellis pack/unpack, CPU Viterbi encode, and materialization
- `QtipWireConsumer`, which reads and decodes both constituent formats from the same receipt

For 256 identities in one projection, `qtip1_5` deterministically reports `128` K1/V1 and `128` K2/V2. For equal-sized members its code plane is exactly `1.5` bits/weight.

## Reference-only physical PoC evidence

The following run used fresh sampled weights on one Spark, but its receipt did not bind the exact current 0731 model config/tokenizer/source identity and did not invoke the installed modern Backpack provider API. It is therefore **REFERENCE_ONLY** and does not satisfy the current-model acceptance gate. No result from the July QTIP1.5 campaign is used here.

After source parity passed, one physical GB10 host was claimed exclusively and released after the run. No holdout data was touched.

Tiny L000/down canary:

- two experts, one `16x32` source-weight window each
- one disjoint-train routing window
- one shared learned TLUT
- both constituent formats written and read by `QtipWireConsumer`
- exact pack-to-state and direct-to-wire-consumer decode parity: PASS
- same-work encode wall: K1/V1 `0.479238 s`; K2/V2 `0.151434 s`
- code planes: `1.0`, `2.0`, composed `1.5` bits/weight

One-layer identity expansion:

- L000, both expert projections, all 256 experts
- 512 members total, each using the same `16x32` weight-window scope
- 256 K1/V1 and 256 K2/V2 members
- all 512 wire consumers returned finite, shape-correct reconstructions
- exact aggregate code plane: `1.5` bits/weight
- encode and write wall: `168.988 s`
- read, decode, and train-only quality wall: `0.324851 s`
- train-only positions: `20,062`
- K1/V1: Top-1 `0.404194`, mean KLD `1.53785e-4`
- K2/V2: Top-1 `0.726314`, mean KLD `3.02311e-5`
- aggregate: Top-1 `0.564002`, mean KLD `9.24885e-5`

These are scoped mechanics/quality measurements on sampled weight windows, not a full-weight model-quality claim. The one-layer receipt SHA-256 is `bf5415b47971780f7fb555afc3286bda2751c96ba577abe9c58adddacab133b5` and its exact wire receipt SHA-256 is `4bccf6482f5ca389ee662a9f92df33d2f5fe85283613d34cf987358502d20c49`.

## Fresh exact-FF0731 installed-public-API canary

The acceptance canary was regenerated from current FF0731 source tensors; it does not reuse any July K1/QTIP1.5 artifact or the reference-only run above.

Identity and invocation:

- model index SHA-256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- model config SHA-256: `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`
- tokenizer SHA-256: `8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf`
- installed wheel SHA-256: `fddec4fcfb20988d3daa41939babb7ce1a36f087af17ab70a233e33aa205dc88`
- public sequence: `qtip1_5_provider_declaration` → `encode_qtip` → `write_qtip_wire` → `verify_qtip_wire` → `QtipWireConsumer.decode`
- source scope: exact L000/down expert 0 and 1 FP8+scale slices, `16x32` weights each
- execution: one exclusively claimed GB10; claim released after clean postflight

Result:

- both constituents generated and consumed: `1` K1/V1 + `1` K2/V2
- pack→state and direct→runtime decode parity: PASS for both formats
- exact code plane: `1.5` bit/weight
- tiny unamortized wire-data rate: `34.5` bit/weight, including row scales and one `4096`-byte shared TLUT over only `1024` sampled weights
- generate wall: K1/V1 `0.453677 s`; K2/V2 `0.187513 s`
- materialize `0.632700 s`; verify `0.001901 s`; runtime decode plus quality `0.002635 s`
- deterministic train-only source-window logit rail, 256 positions: aggregate Top-1 `109/256` (`0.42578125`), mean KLD `2.148092462799682`
- K1/V1: Top-1 `91/256`, mean KLD `2.720487383868603`
- K2/V2: Top-1 `160/256`, mean KLD `0.8001663153743845`

The quality rail is a tiny deterministic source-window probe, not full-model KLD. Its weak K1/V1 result is not green enough to justify a one-layer expansion. The terminal canary receipt SHA-256 is `243059f9da9c90e07d348bbb838afd26e181075a2c62e604c0c0bc267074de34`; wire receipt SHA-256 `208db65bd9d70b903045f408d66b140550780c4e07a83950f1eac5896b8a9252`; independently rehashed six-file wire tree SHA-256 `14fd4ede719697bc1a134c14c40054a57fb00cf0758bd071419d6b2ba6cd929a`.

Before this canary, focused regressions corrected the canonical fixed 16-bit `quantlut`/`quantlut_sym` hash shifts, made wire publication transactional, required PASS/hash/count/finite-positive-scale verification, and exported the full runtime wire steps from the top-level API. A two-step `[[0,1]]` L2/K1 path is explicitly rejected because it does not close the cyclic trellis and cannot be represented reversibly at the exact K-bit code rate; a valid minimal cycle roundtrips exactly.

## Remaining direct-serving work

The provider and materialization architecture can represent and consume authentic QTIP1 and QTIP1.5 now. The remaining production gap is direct stock-vLLM execution of the K1/V1 constituent. The current QTIP2 kernel path assumes two reconstructed lanes and canonical `quantlut_sym` decoding.

The minimum extension is:

1. Dispatch by constituent geometry instead of relabeling K1/V1 as QTIP2.
2. Add a scalar K1/V1 decode lane using canonical `quantlut`: `index = (((state + 1) * state) >> (L - qbits)) & ((1 << qbits) - 1)`.
3. Consume one branch bit and emit one scalar per trellis step; do not apply the V2 first-lane sign rule.
4. Bind the K1/V1 member's scale and selected shared-TLUT column, then reuse the existing accumulation path.

Until quality improves and that kernel/dispatch work is exercised against a full-weight serving artifact, the correct verdict is PARTIAL rather than a serving-level VALIDATED claim.
