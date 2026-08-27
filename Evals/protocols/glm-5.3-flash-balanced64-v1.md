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

Use only the documented public calls:

```python
teacher = capture_balanced64_teacher(
    model,
    revision=revision,
    suite_lock="Evals/configs/glm-5.3-flash-balanced64-v1.json",
    corpus=balanced64_corpus,
    output=teacher_output,
    receipt_path=teacher_receipt,
)
pre = score_balanced64_pre(
    admitted_artifact,
    teacher_capture=teacher,
    suite_lock="Evals/configs/glm-5.3-flash-balanced64-v1.json",
    corpus=balanced64_corpus,
    receipt_path=pre_receipt,
)
```

No runtime object is supplied. The package resolves exactly one registered
capability from config/index and admitted-artifact semantics.

## PRE acceptance

The terminal must seal exactly 64 ordered rows, 65,536 positions, support 8,192,
`KL(teacher||candidate)`, shortest-round-trip binary64 per-position values and
one ordered `math.fsum` reduction, integer Top-1, resident readiness, and zero
timed payload/model reads, fallback, relay, reconstruction, and streaming. It
must bind the reloaded artifact, suite lock, model index, corpus, teacher bank,
runtime identity, exact bytes, and comparison denominator. Null metrics remain
null until this terminal exists.
