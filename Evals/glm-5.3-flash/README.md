# GLM-5.3-Flash BALANCED64 results

This page is model-family-local. It uses the frozen BALANCED64 population and
scorer semantics, but a new GLM-5.3-Flash suite lock and a teacher bank captured
from GLM's own immutable native FP8 source. DeepSeek teacher rows, fixtures, and
numeric baselines are inadmissible.

Suite lock: [`../configs/glm-5.3-flash-balanced64-v1.json`](../configs/glm-5.3-flash-balanced64-v1.json)

## Results

| Quant | Top-1 ↑ | KLD ↓ | Exact accounting GB | Payload scope | Comparison BPW | FP basis |
|---|---:|---:|---:|---|---:|---|
| QTIP2 routed-only + exact native rest (PRE) | — pending 0/64 | — pending | — pending serialized reload | routed experts Q2; every non-routed tensor exact native bytes | — pending | GLM-5.3-Flash native FP8 e4m3 own-base teacher |

Null cells are deliberate. No metric, artifact size, or BPW is inferred before a
serialized artifact reload and canonical 64/64 PRE terminal. The first measured
row will replace this same pending row; no repair/Post result may precede it.

## Public API path

The package-owned path is `capture_balanced64_teacher(...)` followed by
`score_balanced64_pre(...)`, as shown in [`WORKED_EXAMPLE.md`](../../WORKED_EXAMPLE.md).
Callers supply the pinned model, suite lock, frozen corpus, and output locations;
they do not construct a runtime plugin or name ranks, hosts, teacher paths, or a
model-family script. Capability selection must resolve exactly one registered
package runtime and otherwise fails closed.

## Machine and protocol files

- [Pending machine-readable result](../results/glm-5.3-flash-balanced64-v1.json)
- [Model-specific protocol](../protocols/glm-5.3-flash-balanced64-v1.md)
- [Model-specific suite lock](../configs/glm-5.3-flash-balanced64-v1.json)
