# GLM-5.3-Flash BALANCED64 V1 protocol

## Identity

- Model: `zai-org/GLM-5.3-Flash`
- Revision: `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`
- Model index SHA-256: `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05`
- Suite lock SHA-256: `dc5e1a78d0b1ae0975d52b89ee6cfbdc7f8d3207784fe0d7fd5afd3abe844846`
- Frozen population SHA-256: `24089eea1b3e5650265b971930571dbf249aba0b2f62e954a9628dcbfd182f09`
- Teacher bank: `TEACHER_GLM_5_3_FLASH_BALANCED64_V1`

The ordered 64-window population and corrected class map are frozen identically
to BALANCED64 V1. Teacher rows are not shared across model families. This lock
binds GLM's own native FP8 source and rejects the DeepSeek bank and baseline.

## Public producer

Use only the documented public calls. The checked-in lock is the immutable
population/model template. Its historical `source_windows_sha256` is not a GLM
token ledger and therefore must not be passed directly to teacher capture. First
recover authenticated source text, tokenize it with GLM, and use the public
builder's derived suite lock for both capture and PRE:

```python
from banana_smasher import (
    build_balanced64_token_ledger,
    capture_balanced64_teacher,
    recover_balanced64_source_text,
    score_balanced64_pre,
)

model = "/local/hf/GLM-5.3-Flash"
revision = "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"
template_lock = "Evals/configs/glm-5.3-flash-balanced64-v1.json"
source_text_manifest = "/local/eval/recovered-balanced64-source-text.json"
glm_token_ledger = "/local/eval/glm-balanced64-token-ledger.json"
derived_suite_lock = "/local/eval/glm-balanced64-derived-suite-lock.json"

source_text = recover_balanced64_source_text(
    historical_token_ledger="/local/eval/historical-balanced64-token-ledger.json",
    suite_lock=template_lock,
    source_tokenizer_model="/local/hf/historical-source-model",
    output=source_text_manifest,
    receipt_path="/local/eval/SOURCE_TEXT_RECOVERY.json",
)
assert source_text["roundtrip_verified_rows"] == 64

ledger = build_balanced64_token_ledger(
    model,
    revision=revision,
    suite_lock=template_lock,
    source_manifest=source_text_manifest,
    output=glm_token_ledger,
    bound_suite_lock=derived_suite_lock,
    receipt_path="/local/eval/GLM_TOKEN_LEDGER.json",
)
assert ledger["row_count"] == 64
assert ledger["positions"] == 65536

canary = capture_balanced64_teacher(
    model,
    revision=revision,
    suite_lock=derived_suite_lock,
    corpus=glm_token_ledger,
    output="/local/eval/teacher-canary",
    receipt_path="/local/eval/TEACHER_CANARY.json",
    windows=[28],
)
assert canary["status"] == "PASS_DIAGNOSTIC"
assert canary["artifact_admissible"] is False

teacher = capture_balanced64_teacher(
    model,
    revision=revision,
    suite_lock=derived_suite_lock,
    corpus=glm_token_ledger,
    output="/local/eval/teacher-full64",
    receipt_path="/local/eval/TEACHER_CAPTURE.json",
)
pre = score_balanced64_pre(
    "/local/artifacts/glm-routed-q2-native-rest",
    teacher_capture=teacher,
    suite_lock=derived_suite_lock,
    corpus=glm_token_ledger,
    receipt_path="/local/eval/PRE.json",
)
```

The one-window canary is diagnostic only and never enters the results table.
The Full64 capture must follow it. No runtime object is supplied. The package
resolves exactly one registered capability from config/index and the reopened
admitted-artifact semantics.

## PRE acceptance

The terminal must seal exactly 64 ordered rows, 65,536 positions, support 8,192,
`KL(teacher||candidate)`, shortest-round-trip binary64 per-position values and
one ordered `math.fsum` reduction, integer Top-1, resident readiness, and zero
timed payload/model reads, fallback, relay, reconstruction, and streaming. It
must bind the reloaded artifact, suite lock, model index, corpus, teacher bank,
runtime identity, exact bytes, and comparison denominator. Null metrics remain
null until this terminal exists.
