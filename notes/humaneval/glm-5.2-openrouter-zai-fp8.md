# GLM-5.2 OpenRouter HumanEval / HumanEval+ results

**Owner:** TBD · **Status:** **NEW**, independently verified measurement · **Recorded:** 2026-08-13

## Primary result

| OpenRouter model rail | HumanEval pass@1 | HumanEval+ pass@1 |
|---|---:|---:|
| **GLM-5.2, Z.AI FP8, reasoning ON / medium / 16,384 tokens** | **99.39% (163/164)** | **95.73% (157/164)** |
| DeepSeek-V4-Flash-0731, DeepSeek FP8, reasoning ON / medium / 16,384 tokens | 97.56% (160/164) | 93.29% (153/164) |
| **GLM-5.2 delta** | **+3 tasks / +1.83 percentage points** | **+4 tasks / +2.44 percentage points** |

This is a same frozen benchmark, request, and scorer comparison. Provider and wall-clock matching were intentionally not required: GLM-5.2 used Z.AI's native OpenRouter FP8 endpoint, while the retained FF0731 comparator used the historical pinned DeepSeek provider rail. The table therefore measures the two served OpenRouter model rails under the same evaluation contract; it does not isolate model weights from provider/runtime effects. The historical comparator receipt remains preserved at commit [`ad02132c`](https://github.com/my-other-github-account/banana-smasher/commit/ad02132c77df9b68a2869e88b5ccdd6e7239dfa7) on `notes/t802-humaneval-0731-two-row`.

## Frozen contract

- Requested GLM model: `z-ai/glm-5.2`; returned-model closure: 164/164.
- Canonical endpoint release: `z-ai/glm-5.2-20260616`.
- Provider: `Z.AI` only; returned-provider closure: 164/164; fallbacks disabled.
- Endpoint tag and quantization: `z-ai/fp8`, `fp8`.
- Tasks: ordered `HumanEval/0` through `HumanEval/163`; one completion per task; pass@1.
- Reasoning: enabled, `medium`; reasoning retained for audit but never scored as the answer.
- Generation: `max_tokens=16384`, `temperature=0`, `top_p=0.95`, `n=1`, client concurrency 1.
- Scored answer: visible `message.content` only; semantic empty/null and valid length-capped responses remain failures.
- Retries: transient transport/API failures only, never semantic failures or capped model outcomes.
- Prompt messages SHA-256: `6b4f5f30169054a7505f6704c246984f53671c6f4f168c9ee05fdfbd8444b598`.
- EvalPlus commit: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`; package version `0.4.0.dev44`.
- HumanEvalPlus release: `v0.1.10`; dataset hash: `fe585eb4df8c88d844eeb463ea4d0302`.
- Sanitizer: `evalplus.sanitize(content, entrypoint=...)`.

## Outcome details

- Accepted ordered unique rows: **164/164**.
- HumanEval failures: `HumanEval/145`.
- HumanEval+ failures: `HumanEval/55`, `HumanEval/76`, `HumanEval/91`, `HumanEval/99`, `HumanEval/132`, `HumanEval/145`, `HumanEval/151`.
- Finish reasons: `stop=163`, `length=1`.
- Semantic empty/null rows: `HumanEval/145` only.
- Completion tokens: **177,433** total.
- Reasoning tokens: **137,553** total.
- Response-reported primary-arm cost: **$0.810299358**.

`HumanEval/145` consumed the complete 16,384-token primary budget in reasoning and returned no visible answer. It was retained as the primary failure. A separately labeled 32,768-token diagnostic also ended with `finish_reason=length`, exactly 32,768 completion tokens, and no visible answer; it did not replace the primary row.

## Verification receipt

Independent verification returned `PASS`: all 164 self-hashed checkpoints were reloaded; canonical and request-audit bytes were independently reconstructed and matched byte-for-byte; ordered task closure, request/raw-response bindings, model/provider/token closure, sanitizer output, endpoint identity, totals, and evaluator task closure were checked.

- Canonical JSONL SHA-256: `257e1b9d5f8ccef4c91f812deb4c33c4700c97f5c88b73d94742a173e635343b`
- Request audit JSONL SHA-256: `1b48dc902f2ab37c4002b9d2397e012121d339d1ffa20416c34c22b33d5cc179`
- Run identity SHA-256: `d19784b2d10acfc01e10eda11710a027ae73d7de2281805bedd5abd66c39d8d2`
- Endpoint snapshot SHA-256: `72e2f05021913d79bce9e8c23bd1115c8fe2908bc865909e1f1398965edbcb45`
- EvalPlus result SHA-256: `c5331fb20afc99ddd4ee3d847d535fc902063ee5a3bf50a2e3ccc39b87509d2d`
- Per-task score vector SHA-256: `558752663f3031e4c1399ffff8c71edd3e305a8b536800d03dd833cdb5d56bf8`
- Generation handoff SHA-256: `b1d507af52c2c8d6c5c5ac28c5acbfd8e8f5c0291fa7e1aa8cb840877cecc0a4`
- Independent verification receipt SHA-256: `17f69e265e5bc58cd3a60c9a584bf93c67d3ebb670acc1a11d9fd0db9f1b1dec`

The adjacent machine-readable receipt, [`glm-5.2-openrouter-zai-fp8.json`](glm-5.2-openrouter-zai-fp8.json), records the exact counts, task IDs, protocol identity, diagnostic outcome, and hash bindings.
