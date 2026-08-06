# HumanEval 0731 standardized protocol

`HUMANEVAL_0731_V1` is the frozen code-correctness instrument for DeepSeek-V4-Flash-0731 artifacts exposed through OpenAI-compatible endpoints.

## Frozen contract

- EvalPlus `0.4.0.dev44` at commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`
- HumanEvalPlus `v0.1.10`, dataset hash `fe585eb4df8c88d844eeb463ea4d0302`
- 164 tasks, one sample per task
- N=1 greedy, temperature `0`, top-p `0.95`
- `max_completion_tokens=4096`; prompt tokens are excluded
- Reasoning and final answer share the 4,096 generated-token budget
- Score `message.content` only; never rescue code from a reasoning field
- A semantic null/empty response is written once as an empty solution and fails
- Four disjoint resumable ranges: `0–41`, `41–82`, `82–123`, `123–164`
- HumanEval and HumanEval+ pass@1 are both reported

The suite lock is [`../configs/humaneval-0731-v1.json`](../configs/humaneval-0731-v1.json).

## Why the wrapper is required

The pinned EvalPlus OpenAI provider accepts `max_new_tokens` in `run_codegen()` but drops it while constructing the decoder. Without the wrapper, the effective cap silently becomes 768 tokens. `Evals.tools.humaneval` binds the returned decoder to 4,096 and prints:

```text
EFFECTIVE_DECODER_MAX_NEW_TOKENS=4096
```

The wrapper also converts a successful `content=null` response to one empty canonical solution. It does not retry or mine reasoning content.

## Build the frozen environment

From the repository root:

```bash
docker build -f Evals/docker/humaneval/Dockerfile -t banana-humaneval:0731-v1 .
```

The image installs EvalPlus from the exact Git commit and caches the frozen HumanEvalPlus ground truth during the networked build.

## Generate one shard

Generation requires network access to the model endpoint. For a keyless local endpoint, any nonempty API-key value works.

```bash
export OPENAI_API_KEY=local
python3 -m Evals.tools.humaneval generate \
  --model deepseek-v4-flash-0731-candidate \
  --base-url http://127.0.0.1:8000/v1 \
  --root work/humaneval/candidate/shard-000-041 \
  --id-range 0 41
```

Run the same command for the other three frozen ranges. Use a separate root for every range and retain the raw and sanitized JSONL emitted by EvalPlus. Each shard root also contains `request-audit.jsonl`, a prompt-content-free log of the actual OpenAI SDK payload controls and message hashes. Generation fails if that log does not prove one unique prompt per task with `max_completion_tokens=4096`, temperature `0`, top-p `0.95`, N=1, and a single user message.

## Merge and audit

Pass the four sanitized JSONL paths printed by the shard commands:

```bash
python3 -m Evals.tools.humaneval merge \
  "$SHARD_000_041_SANITIZED" \
  "$SHARD_041_082_SANITIZED" \
  "$SHARD_082_123_SANITIZED" \
  "$SHARD_123_164_SANITIZED" \
  --output work/humaneval/candidate/canonical.jsonl
```

Do not pass the adjacent `.raw.jsonl` files to this merge. The merger rejects duplicates, missing tasks, extra tasks, malformed rows, and non-string solutions. It writes tasks in numeric order and emits `canonical.jsonl.audit.json` with counts and SHA-256.

## Score safely

Generated code is untrusted. Score only inside the frozen container with network disabled:

```bash
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 512 \
  --memory 8g \
  --tmpfs /tmp:rw,nosuid,noexec,size=2g \
  --tmpfs /run:rw,nosuid,noexec,size=64m \
  -v "$PWD/work:/work:rw" \
  banana-humaneval:0731-v1 \
  score /work/humaneval/candidate/canonical.jsonl
```

The score command re-audits the complete 164-task population before invoking EvalPlus with eight workers and the frozen execution limits. Score every model row on the same scorer host class: timeout-sensitive solutions can change status across materially different CPUs even when the image and sample JSONL are identical. Record the scorer host, architecture, image ID, and image digest in the run receipt.

## Primary comparison policy

For quant comparisons, use target-only decoding with speculative decoding disabled. Match KV precision where the serving stacks allow it. Record exact model files, model revisions, engine revisions, tokenizer/chat-template identity, live serving command, KV dtype, and topology alongside each result.

Do not merge shards generated under different model identities, serving commands, prompt templates, completion caps, or EvalPlus commits. If the instrument changes, restart all 164 tasks for that row.
