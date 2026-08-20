# QTIP1 and QTIP1.5 wire acceptance

Date: 2026-08-06

Verdict: **ACCEPTED for the authentic QTIP1 primitive and generic QTIP1.5 wire materialization/runtime consumer. Not accepted as a stock-vLLM serving claim.**

## Accepted scope

This phase preserves QTIP1 and QTIP1.5 as distinct geometry-driven contracts:

- QTIP1 is canonical `L16/K1/V1` with `quantlut` decode.
- QTIP1.5 is an exact deterministic 50/50 composition of QTIP1 `K1/V1` and QTIP2 `K2/V2` members.
- QTIP2.5 remains a separate provider and wire contract; nothing in this phase relabels it.
- Public wire publication is transactional and verifies receipt status, receipt/artifact hashes, counts, geometry, finite positive scales, non-empty manifests, and runtime decode.
- Automatic QTIP1 row scales use a fixed per-row error grid.
- Quality-fitted pre-encoded paths can be materialized through `write_encoded_qtip_wire`, then verified and decoded through the same generic public API.

The direct stock-vLLM K1/V1 dispatch/kernel remains absent. Product-level paired exact64 evaluation was cancelled as superseded before a terminal result, so this report makes no serving-quality claim.

## Source binding

Canonical algorithm reference:

- repository: `https://github.com/Cornell-RelaxML/qtip`
- revision: `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`
- source: `lib/codebook/bitshift.py`
- source SHA-256: `a299ae97d2ccc80a142095c3c16ed619b435b68736fd52702ab396bc37218531`

Integration base: `27dfff50004a0c4a2465b428913ac5e414137130` (`origin/main` after phase 5).

Retained source semantics:

- `642134d`: canonical 16-bit hash shifts, transactional publication, fail-closed wire verification.
- `4b99b17`: top-level pack/unpack/verify runtime exports.
- `20554b2`: automatic QTIP1 row-scale calibration.
- `0fda992`: validated quality-fitted pre-encoded wire materialization.
- `0eb4e917`: geometry/TLUT validation before publication and empty-manifest rejection.

The integrated `qtip1.py` and focused test blobs exactly match accepted source tip `0eb4e917`:

- `qtip1.py`: `0387a0b3f2983e9944521e838d10cd2a5f66336e88e69f264b48f89b8d0217d9`
- `test_qtip1.py`: `b33c5245cb5b46f700112cb0a361e7b0a24e4d636070200993c56b541f410d51`

Rejected from integration:

- stale report commit `483b908`, whose PARTIAL narrative predates later accepted wire materialization;
- unrelated ancestry from the source worktree;
- any direct-serving claim or stock-vLLM K1/V1 implementation;
- any runtime, build, install, test, development, import, subprocess, path, wheel, or generated-code dependency on historical/private trees.

## Physical acceptance boundary

Fresh FF0731 public-API wire mechanics were sealed against model-index SHA-256 `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`.

The accepted all-layer fan-in verified 43/43 layers exactly once with:

- exact code plane: `1.5` bit/weight;
- family counts: `11,008` QTIP1 `K1/V1` and `11,008` QTIP2 `K2/V2` members;
- logical/unique wire bytes: `31,069,776`;
- common-pack SHA-256: `ce47f0c430bc95b7d31d6e75b4aa2b3bbff7540ffb4df8ef37aead8e3e283cfa`;
- terminal SHA-256: `16b99cc1b2abc11dee064909143cfc3f64bd7a9eacf0f5bcea8961fcfdc67ff9`;
- final-handoff SHA-256: `4b275d1eb3a6022438df5d9b2cfb435a0cc3723a7b0947408fa8dada3ea9f00e`.

This proves generic materialization, accounting, verification, and constituent runtime decode. It does not prove full-model stock-vLLM serving.

## Focused verification

Commands run from `banana-smasher/` with Python 3.13:

- focused QTIP1/QTIP1.5 wire, provider-menu, and CLI tests: `18 passed`;
- Ruff on `qtip1.py`, package exports, and focused tests: PASS;
- `git diff --check`: PASS;
- public runtime smoke: quality-fitted QTIP1.5 pre-encoded materialization -> verification -> both constituent decodes: PASS, 2 members, exact `1.5` code bpw;
- runtime decode SHA-256: QTIP1 `ff54ce43ce50975291164d8e50af1ee7f6abeda7b33dfef847a397067a77ec07`, QTIP2 `49d90c7a46c962e0af89167ff7c93644d708b71a2c6cd9b3ad91916edd5b4559`.
