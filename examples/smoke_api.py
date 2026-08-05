#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def get_json(url: str, *, timeout: int = 30) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict[str, Any], *, timeout: int = 120) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def require_semantic_ok(message: Any, *, response_kind: str) -> str:
    if not isinstance(message, str) or message.strip() not in {"OK", "OK."}:
        raise SystemExit(f"{response_kind} did not return the semantic OK answer")
    return message.strip()


def post_chat_stream(
    url: str, payload: dict[str, Any], *, timeout: int = 120
) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks: list[str] = []
    done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            choices = event.get("choices", [])
            if choices:
                content = choices[0].get("delta", {}).get("content")
                if isinstance(content, str):
                    chunks.append(content)
    if not done:
        raise SystemExit("streaming chat did not terminate with [DONE]")
    return "".join(chunks)


def main() -> None:
    api_base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    service_base = api_base[:-3] if api_base.endswith("/v1") else api_base
    model = os.environ.get("MODEL", "banana-smasher-v5")

    with urllib.request.urlopen(f"{service_base}/health", timeout=30) as response:
        response.read()

    models = get_json(f"{api_base}/models")
    model_ids: set[str] = set()
    for item in models.get("data", []):
        if isinstance(item, dict):
            identifier = item.get("id")
            if isinstance(identifier, str):
                model_ids.add(identifier)
    if model not in model_ids:
        raise SystemExit(
            f"expected served model {model!r} is absent from /v1/models: {sorted(model_ids)}"
        )

    chat_url = f"{api_base}/chat/completions"
    chat_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 8,
    }
    body = post_json(chat_url, chat_payload)
    message = require_semantic_ok(
        body["choices"][0]["message"]["content"], response_kind="nonstreaming chat"
    )
    print(message)

    stream_message = require_semantic_ok(
        post_chat_stream(chat_url, chat_payload), response_kind="streaming chat"
    )
    print(stream_message)


if __name__ == "__main__":
    main()
