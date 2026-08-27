# Evaluation parity

Parity is established per instrument. Matching one hash or one temperature is
not enough; preserve the complete request, render, artifact, and scorer
contract.

## 1. Paired KLD protocol

A DS4 paired-KLD row is comparable only when all of the following match:

| Field | Required value |
| --- | --- |
| Evaluation corpus | `out/windows_ds4_eval.json` |
| Corpus MD5 | `1701920b4ba96dea0b18fe9df0151876` |
| Windows | 512 |
| Position cutoff | first `min(1024, real_len - 1)` per window |
| Scored positions | 524,288 |
| Direction | `KL(reference ∥ candidate)` |
| Support | reference top-8,192 token IDs |
| Normalization | renormalize reference and candidate on that support |
| Aggregation | position-weighted over all scored positions |
| Tokenizer MD5 | `3f75dbea81fe67dd8c07843bdf9ce36e` |

Required row provenance:

- teacher/model identity;
- candidate manifest and payload hashes;
- candidate and score ledgers;
- scorer identity;
- candidate byte count;
- full artifact coverage;
- sample size and status (`MEASURED-512`, gate, or predicted).

Serve NLL, llama.cpp KLD, task accuracy, and throughput are separate
instruments even if they use related text.

## 2. MMLU-500 protocol

The repository-local question-set receipt is
`out/MMLU_QUESTION_SET_DS4.json`.

| Field | Required value |
| --- | --- |
| Source | Hendrycks MMLU archive |
| Source archive MD5 | `20bb207676c1f58dc70afc9267cd206c` |
| Selection | sorted subjects, concatenated test rows, `[::28][:500]` |
| Question-set SHA-256 | `24d60b46aa7d0268b5f230760f3caa1391211fdd2893c9073c9e037135b4443a` |
| Questions | 500 |
| Shots | 0 |
| Prompt | header + question + `A.`…`D.` + `Answer:` |
| Continuations | single-token `<space>A`, `<space>B`, `<space>C`, `<space>D` |
| DS4 choice token IDs | `[334, 406, 345, 420]` |
| Score | choice log-likelihood at the answer position |

MMLU-500 is deterministic choice-loglik; generation temperature and random
seed are not part of this instrument. Preserve per-question predictions and
choice log-probabilities so paired McNemar and continuous gold-logprob tests can
be recomputed.

The 500-question sample resolves only multi-point differences. A non-significant
result is not proof of exact equality.

## 3. ToolBench baseline audit

The ToolBench baseline to match is the immutable `v4flashv3` run audited in the
adjacent tool-evaluation repository. Its verified receipt is:

| Field | Verified baseline value |
| --- | --- |
| Toolbench version | `v2.0.6` at commit prefix `c3868bf` |
| Sealed file | `runs/v4flashv3.json`; 331,852 bytes; SHA-256 `99cabdea96ab7c222e3f26c41dce9328a721b213fb8d2dac4bfd7bd8eed58f06`; 2026-06-12 |
| Model | `deepseek/deepseek-v4-flash` |
| Backend label | `litellm` (direct OpenAI-compatible adapter) |
| Maximum completion tokens | `4096` |
| Temperature | `1.0` |
| Top-p | `1.0` |
| Seed | `42` |
| Max turns | `8` |
| Timeout | `240 s` |
| Concurrency | `5` |
| Provider policy | FP8 filter only; no provider slug pinned |
| Scenarios | 69 |
| Trials in the sealed baseline file | 5 (69×5; final score 86/100, mean 85.4) |
| Intended local comparison budget | 3 trials per scenario (69×3), held behind the throughput gate |
| Tools | `tool_choice=auto`, parallel tool calls enabled |
| Reasoning parameter | omitted |

The sealed file contained 191 visible reasoning blocks. Exact DS4-tokenizer
counts were median 24, mean 28.6021, p90 50, maximum 103. Earlier shorthand
estimates around 32/47/93 do not reproduce from the immutable file and should
not be used as exact parity targets.

The baseline finished with 55 pass, 8 partial, and 6 fail scenarios. The
campaign's 69×3 local budget is therefore a comparison design, not a statement
that the immutable hosted baseline itself used three trials.

The baseline did not record which eligible OpenRouter provider served each
request and did not expose rendered prompt token IDs. Therefore a public
receipt can prove local official-encoder ↔ local vLLM render parity, but cannot
claim inaccessible OpenRouter token-ID identity.

### Multiple baseline warning

A separate OpenRouter ToolBench receipt used temperature 0. Do not mix its
configuration or score with `v4flashv3`. Every comparison directory must pin
one immutable baseline file and hash.

### Evaluator-version split discovered after cutover

The immutable hosted baseline above was scored with tool-eval-bench `v2.0.6`
at commit prefix `c3868bf`. Later local rows used `v2.1.0` at commit prefix
`8d5c48a` after evaluator hardening. The versions differ in answer validation
for multiple scenarios; their final scores are therefore **not a
same-instrument delta**.

Publication law after this discovery:

1. pin the exact tool-eval-bench version and commit in every row;
2. compare local and hosted models only when that evaluator identity matches;
3. label the `v2.0.6` rows legacy when shown beside `v2.1.0` rows;
4. never explain a cross-version score gap as provider/model drift without a
   same-version rerun;
5. preserve all scenario-level status and points, not only the final score.

One useful but artifact-specific post-cutover row is the UD-IQ3_XXS GGUF:
three complete 69-scenario trials on tool-eval-bench `v2.1.0` scored
`[86, 86, 86]` (`86.0 ± 0.0`), with 207/207 attempts and 53 pass / 13 partial /
3 fail per trial. It used a llama.cpp `--jinja` endpoint and is **not** a row
for the learned-VQ COMBO artifact or its vLLM server.

## 4. Required local DS4 serving semantics

The DS4 tokenizer package has no Jinja `chat_template`; rendering is implemented
by the model's DS4 encoder. The audited vLLM wrapper defaults thinking off, so
`--tokenizer-mode deepseek_v4` alone does not match the empirical baseline.

A parity launch requires the artifact/kernel flags plus:

```bash
--tokenizer-mode deepseek_v4 \
--generation-config vllm \
--reasoning-parser deepseek_v4 \
--default-chat-template-kwargs '{"enable_thinking":true}' \
--enable-auto-tool-choice \
--tool-call-parser deepseek_v4
```

The request must explicitly carry:

```text
temperature            1.0
top_p                   1.0
top_k                   0
min_p                   0.0
seed                    42
max_turns               8
chat_template_kwargs    {"enable_thinking": true}
reasoning_effort        omitted
```

`--generation-config vllm` prevents hidden model defaults from overriding the
explicit benchmark settings.

## 5. Render-parity gate

The fixed weather-tool request produced the following local receipt:

| Check | Result |
| --- | --- |
| Official DS4 encoder vs installed vLLM prompt text | exact equality |
| Rendered UTF-8 SHA-256 | `54e2525814c5ea32b3f6a7962dc939fa132f1454654033d9ce167a787fe93392` |
| Token count | 295 |
| Token-ID CSV SHA-256 | `7d8497f4c0e6d5e18045565ad06732cebd5b0861714d91f75d0481571e10f924` |
| Offline IDs vs live local render endpoint | exact equality |

Before a full ToolBench row, the TC-01 canary must prove:

1. a non-empty reasoning block;
2. a parsed `get_weather` tool request;
3. a tool result followed by a final answer;
4. reasoning is returned in `reasoning`/`reasoning_content`, not ordinary
   content;
5. tool-call continuation preserves prior reasoning;
6. launch and effective request settings match the pinned baseline.

A failed canary is a protocol/configuration result, not model-quality evidence.

## 6. Throughput gate before behavioral benchmarking

The campaign imposed a **minimum measured decode throughput of 10 token/s** on
the learned-VQ server before spending the full 69×3 ToolBench budget.

This is an operational acceptance gate, not a quality metric. It must be
measured from a timed served request after effective-environment verification,
at least two generated warmups, and under no competing clients. At the
2026-07-17 cutover, the best recorded learned-VQ served decode result was 6.59
token/s, so the full local IQ3 ToolBench row was correctly held back.

Post-cutover, the raw-AR V4-step32 product measured `13.964` token/s median on
5×64 and `13.913` token/s over a warm 4,096-token stream, so the throughput gate
did pass. That permits the behavioral run; it does not create one. At this
evidence cut, the public package still lacks a complete same-instrument,
artifact-bound 69×3 learned-VQ receipt comparable to a same-version hosted
reference. Do not promote partial, cross-version, or different-artifact rows to
fill that gap.

## 7. Invalid parity shortcuts

Do not claim parity by doing only one of the following:

- reusing the same question count with a different question-set hash;
- matching temperature while thinking mode differs;
- matching the template while tool parsing differs;
- using a provider filter as if it pinned one provider;
- comparing a 64-window KLD gate to a 512-window canonical row;
- attaching old MMLU results to a new artifact;
- comparing generative accuracy to choice-loglik accuracy;
- comparing model footprint or kernel timing to served throughput;
- reporting a local render hash as proof of an inaccessible hosted render.
- comparing tool-eval-bench `v2.0.6` and `v2.1.0` final scores as if the
  evaluator were unchanged;
- attaching a llama.cpp/GGUF ToolBench row to the learned-VQ vLLM artifact.

## 8. Publication checklist

For each public evaluation row, include:

- immutable artifact/manifest hash;
- corpus or scenario-set hash;
- evaluator version/commit;
- all request sampling parameters;
- prompt/template/parser configuration;
- exact sample count and completion count;
- raw per-window/per-question/per-scenario rows;
- status label and failure exclusions;
- separate quality, footprint, kernel, and served-throughput fields;
- a note for any source artifact not present in the public checkout.
