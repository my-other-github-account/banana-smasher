# EXL3 code-provenance audit

- **Status:** product gate — remediation required
- **Audit date:** 2026-08-10
- **Upstream authority:** `turboderp-org/exllamav3` at `791c83073f7f90c44f765a0ceeab7a05fa15b96b`
- **Banana default branch audited:** `origin/main` at `50ddd88654c3dad549caf6670a620c1cd81c27d3`
- **Active Q2 fan-in audited:** `29c5a126f40ade02c48a647fcc6cc4d1df7c64be`

## Answer

Banana does **not** currently import, link, dynamically load, or declare a runtime dependency on EXL3. The audit found no complete-file copy and no exact 40-token implementation block in the active Q2 fan-in.

However, it would be inaccurate to call the active Q2 implementation independent or clean-room:

- one non-shipping test helper explicitly identifies itself as a **literal NumPy transcription** of EXL3 LDLQ;
- several production functions are close source-guided mechanical translations of EXL3 functions;
- the active production code contains two exact EXL lines and two short exact reshape/reduction idioms;
- the default branch and historical QTIP tooling contain the exact standard NVFP4 E2M1 value table also present in EXL3.

The active fan-in is therefore **diagnostic-only** until the RED/AMBER findings below are rewritten from a sealed behavior specification or explicitly retained by owner decision with appropriate provenance treatment.

## Scope and method

The audit compared:

1. all tracked source at the three revisions above;
2. every source blob reachable from every local Git ref;
3. clean wheels built from `origin/main` and the active Q2 fan-in;
4. Python/C/C++/CUDA and related source using:
   - exact SHA-256 file matches;
   - normalized meaningful-line matches;
   - exact 20-token and 40-token shingles with comments removed;
   - embedded CUDA source-string tokenization;
   - identifier-normalized Python AST/control-flow similarity;
   - manual import, dependency, and data-flow review.

The scanner is `tools/audit_exl3_exact_overlap.py`. Machine-readable classifications are in `notes/reports/2026-08-10-exl3-code-provenance-audit.json`.

### Quantitative results

| Scope | Source files/blobs | Exact file copies | Exact 40-token groups | Interpretation |
|---|---:|---:|---:|---|
| `origin/main` | 185 files | 0 | 2 | Both are the standard NVFP4 E2M1 table |
| Active Q2 fan-in `29c5a12` | 64 files | 0 | 0 | No long verbatim implementation block detected |
| All reachable history | 2,647 unique source blobs | 0 | 52 | All 52 are versions/copies of the same NVFP4 E2M1 table |

Wheel receipts:

| Wheel source | Members | SHA-256 | EXL dependency | Exact 40-token implementation block |
|---|---:|---|---|---|
| `origin/main` | 104 | `ae9b82327ef9298fba785ab909ee5a55b2f43d57c977fb5973e71f56f38210d7` | none | none beyond the two NVFP4 table occurrences |
| Q2 fan-in `29c5a12` | 25 | `a0f1490e805c21248a6099db957d7d3c687c57284c8b045562a5e7dfc1225ef9` | none | none |

## Finding ledger

### EXL-001 — no EXL runtime import or package dependency

- **Classification:** independent package boundary
- **Status:** GREEN
- **Evidence:** neither wheel metadata nor product source imports `exllamav3`; no EXL library, source tree, subprocess invocation, or EXL file is packaged.
- **Action:** keep the product path one-way. EXL may produce external oracle traces, but Banana artifacts must not require EXL at runtime.

### EXL-002 — exact NVFP4 E2M1 standard table

- **Banana:**
  - `banana-smasher/src/banana_smasher/qtip_runner.py:48`
  - `banana-smasher/src/banana_smasher/solver_qtip_profile.py:906`
- **EXL3:** `eval/qbench/engines.py:433`
- **Introducing Banana commit:** `1ae202c`
- **Classification:** constants/data only; exact shared standard-format table
- **Status:** GREEN WITH PROVENANCE COMMENT RECOMMENDED
- **Evidence:** `[0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6]` is the NVFP4 E2M1 representable-value table. It is not an EXL solver body.
- **Action:** retain if needed, but label it as the NVFP4 E2M1 format table rather than implying Banana invented the values.

### EXL-003 — tensor-core permutation is source-guided

- **Banana:** active `q2_assignment.py:46-65`
- **EXL3:** `quantize.py:22-45` (`tensor_core_perm`)
- **Introducing Banana commit:** `c4015c9`
- **Classification:** mechanical/structural reimplementation
- **Status:** AMBER — rewrite before acceptance
- **Evidence:** identifier-normalized AST similarity `0.758`; same thread-to-row/column construction and output ordering. No exact 40-token block.
- **Action:** replace from a documented MMA/tensor-core layout specification and fixed input/output vectors, without consulting EXL source during implementation.

### EXL-004 — block RMS is a close translation

- **Banana:** active `q2_assignment.py:185-191`
- **EXL3:** `quantize.py:1074-1087` (`block_rms`)
- **Introducing Banana commit:** `c4015c9`
- **Classification:** mechanical/structural reimplementation
- **Status:** RED — rewrite before acceptance
- **Evidence:** identifier-normalized AST similarity `0.825`; same split, square-sum accumulation, division, and square-root sequence.
- **Action:** replace from the mathematical definition plus dtype/order acceptance vectors. Do not merely rename variables.

### EXL-005 — scale-tile sampling is a very close translation

- **Banana:** active `q2_assignment.py:194-213`
- **EXL3:** `quantize.py:949-977` (`sample_scale_tiles`)
- **Introducing Banana commit:** `c4015c9`
- **Classification:** mechanical translation with exact snippets
- **Status:** RED — rewrite before acceptance
- **Evidence:** identifier-normalized AST similarity `0.905`; exact line `tiles_k = weight.shape[0] // 16`; exact 25-token reshape/permute idioms; same diagonal sampling and extreme-tile selection sequence.
- **Action:** replace with a Banana-owned sampler specified by selected tile indices and output hashes. If authentic EXL scale behavior is mandatory for the comparator, keep the current function only in a non-shipping diagnostic harness.

### EXL-006 — global scale search is source-guided and contains an exact line

- **Banana:** active `q2_assignment.py:216-250`
- **EXL3:** `quantize.py:979-1042` (`g_scale_search_batch`)
- **Introducing Banana commit:** `c4015c9`
- **Classification:** source-guided algorithmic reimplementation with an exact line/constants
- **Status:** RED — rewrite before acceptance
- **Evidence:** same coarse grid (`0.1 + 0.2*i`), one-third subsample, five-point fine grid, `0.075` step, parabolic interpolation, and exact clamp `offset = max(-0.5, min(0.5, offset))`.
- **Action:** specify the search mathematically and reimplement from that specification, or replace it with a Banana-owned deterministic search whose behavior satisfies the product contract. Preserve comparison behavior only in diagnostic code if required.

### EXL-007 — block LDL and buffered LDLQ follow EXL execution structure

- **Banana:**
  - active `q2_assignment.py:392-423`
  - active `q2_ldlq.py:279-362`
- **EXL3:** `quantize.py:411-485` (`block_ldl`) and `488-604` (`ldlq`)
- **Relevant Banana commits:** `d056d18`, `03e2f7f`, `cd10bc3`, `bd97e36`, `b4f8fed`, `22d6be0`, `03efa8f`, `965b5af`, `29c5a12`
- **Classification:** source-guided behavioral/mechanical reimplementation; no long verbatim block detected
- **Status:** RED — diagnostic-only pending clean replacement
- **Evidence:** commit intent and code preserve EXL-specific in-place Cholesky copy-back, explicit block-identity insertion, reverse buffer order, aliasing of product-cache slices, serial in-place `addmm_`, and output collection order. The newest commit is explicitly named “alias upstream product cache slices.”
- **Action:** freeze immutable boundary vectors now. Rewrite from an implementation-neutral recurrence and storage-semantics specification by an author/process that does not inspect EXL source. Validate against hashes only after implementation.

### EXL-008 — literal EXL LDLQ transcription in a test

- **Banana:** active `banana-smasher/tests/test_q2_ldlq.py:89-119`
- **EXL3:** `quantize.py:488-604`
- **Introducing Banana commit:** `cd10bc3`
- **Classification:** literal translated test oracle
- **Status:** RED, non-shipping
- **Evidence:** the helper docstring states: “Literal NumPy transcription of EXL3 ldlq's row-axis recurrence.”
- **Action:** delete the translated helper. Replace it with immutable EXL-generated input/output fixtures and hashes, or an equation-derived reference written without EXL source access. Tests should consume outputs, not carry a translated upstream implementation.

### EXL-009 — Q2 CUDA solver is source-guided but not textually copied in bulk

- **Banana:** active `q2_k2_exact.cu:30-227`, wrapper `q2_k2_cuda.py`
- **EXL3:** `exllamav3_ext/quant/quantize_tiles_kernel.cuh` and associated quantization path
- **Relevant Banana commits:** `aa01c9e`, `f84dede`, `4d8289d`
- **Classification:** algorithmic/source-guided reimplementation
- **Status:** AMBER — manual clean-reimplementation review required
- **Evidence:** no exact 40-token block. Exact short overlaps are framework includes and the common CUDA warp reduction loop `for (int offset = 16; offset > 0; offset >>= 1)`. Comments explicitly pin revision-specific terminal-buffer behavior.
- **Action:** preserve the sealed CUDA input/output oracle, then review/rewrite recurrence, reduction, and backtrack from the trellis specification. Do not copy upstream kernels. Generic PyTorch extension includes and standard warp-shuffle reduction idioms need no special treatment, but revision-specific behavior must remain documented as comparator behavior.

### EXL-010 — EXL oracle traces are external evidence, not shipped code

- **Classification:** test fixture / behavioral oracle
- **Status:** GREEN WITH ONE-WAY DATA-FLOW RULE
- **Evidence:** external EXL executions create hashes and trace bundles; the Banana package does not package or invoke EXL.
- **Action:** keep only immutable test vectors, hashes, and methodology receipts. Do not feed EXL source, implementation objects, or hidden state into the production artifact.

## License posture

Pinned EXL3 revision `791c830` carries the MIT License, copyright © 2025 Turboderp. Its notice requires preservation when copies or substantial portions are distributed.

This audit is not a legal determination of whether each structural translation is a “substantial portion.” The engineering default is stricter: rewrite the RED/AMBER source-guided code rather than rely on a licensing argument. If the owner elects to retain any copied or mechanically translated component, record the exact retained lines and add the necessary third-party notice before distribution. Such a notice would cover only the upstream component and would not assign a license to Banana Smasher’s original work.

No `LICENSE`, SPDX header, or blanket license grant should be added to Banana Smasher without an explicit owner decision.

## Binding remediation gate

Before native Q2 can be merged or accepted as a product:

1. keep `29c5a12` and descendants diagnostic-only;
2. stop further implementation work derived by reading EXL source;
3. seal implementation-neutral specifications and immutable boundary vectors for permutation, scale selection, block LDL, LDLQ, trellis, packing, and decode;
4. replace EXL-003 through EXL-009 where marked RED/AMBER;
5. remove the literal test transcription in EXL-008;
6. rerun exact-file, 20/40-token, wheel, dependency, AST, and manual data-flow audits;
7. require no unexplained exact 40-token block, no EXL runtime dependency, and no high structural-similarity exception without an owner-approved ledger entry;
8. rerun E000 physical parity and the whole-product acceptance contract on the remediated candidate.

Do not “fix” similarity by cosmetic renaming or reformatting. The replacement must be independently structured from the behavior/equation contract.

## Current decision status

- **Large verbatim EXL import:** not found
- **Direct EXL runtime dependency:** not found
- **Exact small snippets/constants:** found and tracked
- **Literal translated non-shipping test code:** found
- **Close production translations:** found
- **Safe to claim clean-room independence:** no
- **Safe to merge current Q2 as independently owned product code:** no
