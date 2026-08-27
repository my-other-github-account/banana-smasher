from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _model(root: Path) -> tuple[Path, str]:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"model_type": "fixture_moe"}) + "\n")
    shard = root / "model.safetensors"
    shard.write_bytes(b"fixture")
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": shard.name}}) + "\n")
    (root / "tokenizer.json").write_text("{}\n")
    return root, _sha(index)


class _Tokenizer:
    tokenizer_id = "fixture-tokenizer-v1"
    tokenizer_sha256 = "c" * 64

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token) for token in token_ids)


def _inputs(tmp_path: Path) -> tuple[Path, dict[str, object], str]:
    items = [
        {
            "item_id": f"item-{index}",
            "window_id": 1000 + index,
            "source_class": "fixture",
            "text": chr(0x1000 + index) * 1025,
        }
        for index in range(64)
    ]
    for item in items:
        item["text_sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()
    provenance = _canonical_sha(
        [
            {
                "item_id": item["item_id"],
                "window_id": item["window_id"],
                "source_class": item["source_class"],
                "text_sha256": item["text_sha256"],
            }
            for item in items
        ]
    )
    manifest = {
        "schema": "banana-smasher.balanced64-source-text.v1",
        "source_provenance_sha256": provenance,
        "item_roster_sha256": provenance,
        "items": items,
    }
    source_path = tmp_path / "source-text.json"
    source_path.write_text(json.dumps(manifest, sort_keys=True, ensure_ascii=False) + "\n")
    lock = {
        "schema": "banana-smasher.balanced64-suite-lock.v1",
        "name": "BALANCED64_FIXTURE_V1",
        "positions": 65536,
        "positions_per_window": 1024,
        "support": 8192,
        "window_count": 64,
        "window_population_sha256": "a" * 64,
        "source_windows_sha256": "b" * 64,
        "source_provenance_sha256": provenance,
        "teacher_bank": "TEACHER_FIXTURE_BALANCED64_V1",
        "teacher_source_model_index_sha256": "MODEL_INDEX_PLACEHOLDER",
        "metrics": {"kld": {}, "top1": {}},
        "windows": [
            {"ordinal": index, "window_id": 1000 + index, "source_class": "fixture"}
            for index in range(64)
        ],
    }
    return source_path, lock, provenance


def test_build_model_specific_token_ledger_from_authenticated_source(tmp_path: Path) -> None:
    from banana_smasher import build_balanced64_token_ledger

    model, model_index_sha = _model(tmp_path / "model")
    source_path, lock, provenance = _inputs(tmp_path)
    lock["teacher_source_model_index_sha256"] = model_index_sha
    lock["suite_lock_sha256"] = _canonical_sha(lock)
    lock_path = tmp_path / "suite-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n")

    receipt = build_balanced64_token_ledger(
        model,
        revision="1" * 40,
        suite_lock=lock_path,
        source_manifest=source_path,
        output=tmp_path / "glm-token-ledger.json",
        bound_suite_lock=tmp_path / "glm-suite-lock.json",
        receipt_path=tmp_path / "TOKEN_LEDGER.json",
        tokenizer=_Tokenizer(),
    )

    ledger_path = tmp_path / "glm-token-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    bound_lock = json.loads((tmp_path / "glm-suite-lock.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["api"] == {"method": "build_balanced64_token_ledger", "version": 1}
    assert receipt["source_provenance_sha256"] == provenance
    assert receipt["model_index_sha256"] == model_index_sha
    assert receipt["tokenizer"] == {"id": "fixture-tokenizer-v1"}
    assert receipt["row_count"] == 64
    assert receipt["positions"] == 65536
    assert receipt["ledger_sha256"] == _sha(ledger_path)
    assert receipt["bound_suite_lock_sha256"] == bound_lock["suite_lock_sha256"]
    assert bound_lock["source_windows_sha256"] == receipt["ledger_sha256"]
    assert bound_lock["historical_source_windows_sha256"] == "b" * 64
    assert bound_lock["token_ledger"] == {
        "model_index_sha256": model_index_sha,
        "row_count": 64,
        "schema": "banana-smasher.balanced64-token-ledger.v1",
        "sha256": receipt["ledger_sha256"],
        "tokenizer_id": "fixture-tokenizer-v1",
    }
    unhashed_bound_lock = dict(bound_lock)
    unhashed_bound_lock.pop("suite_lock_sha256")
    assert bound_lock["suite_lock_sha256"] == _canonical_sha(unhashed_bound_lock)
    assert ledger["schema"] == "banana-smasher.balanced64-token-ledger.v1"
    assert [row["window_id"] for row in ledger["rows"]] == list(range(1000, 1064))
    assert ledger["rows"][0]["token_ids"] == [0x1000] * 1025
    assert json.loads((tmp_path / "TOKEN_LEDGER.json").read_text()) == receipt


def test_token_ledger_refuses_pretokenized_or_tampered_source(tmp_path: Path) -> None:
    from banana_smasher import build_balanced64_token_ledger

    model, model_index_sha = _model(tmp_path / "model")
    source_path, lock, _ = _inputs(tmp_path)
    source = json.loads(source_path.read_text())
    source["items"][0]["text"] += "tamper"
    source["items"][0]["token_ids"] = [1, 2, 3]
    source_path.write_text(json.dumps(source, sort_keys=True) + "\n")
    lock["teacher_source_model_index_sha256"] = model_index_sha
    lock["suite_lock_sha256"] = _canonical_sha(lock)

    with pytest.raises(ValueError, match="must not supply historical token_ids"):
        build_balanced64_token_ledger(
            model,
            revision="1" * 40,
            suite_lock=lock,
            source_manifest=source_path,
            output=tmp_path / "ledger.json",
            bound_suite_lock=tmp_path / "bound-suite-lock.json",
            receipt_path=tmp_path / "receipt.json",
            tokenizer=_Tokenizer(),
        )


def test_recover_source_text_requires_exact_tokenizer_round_trip(tmp_path: Path) -> None:
    from banana_smasher import recover_balanced64_source_text

    _, lock, provenance = _inputs(tmp_path)
    historical_path = tmp_path / "historical-token-ledger.json"
    historical_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "id_ds4": f"source-{index}",
                        "id_gold": 1000 + index,
                        "name": f"fixture-{index}",
                        "real_len": 1025,
                        "token_ids": [0x1000 + index] * 1025,
                    }
                    for index in range(64)
                ]
            },
            sort_keys=True,
        )
        + "\n"
    )
    lock["source_windows_sha256"] = _sha(historical_path)
    lock["suite_lock_sha256"] = _canonical_sha(lock)

    receipt = recover_balanced64_source_text(
        historical_path,
        suite_lock=lock,
        output=tmp_path / "recovered-source-text.json",
        receipt_path=tmp_path / "SOURCE_TEXT_RECOVERY.json",
        tokenizer=_Tokenizer(),
    )

    manifest_path = tmp_path / "recovered-source-text.json"
    manifest = json.loads(manifest_path.read_text())
    assert receipt["status"] == "PASS"
    assert receipt["api"] == {
        "method": "recover_balanced64_source_text",
        "version": 1,
    }
    assert receipt["historical_token_ledger_sha256"] == _sha(historical_path)
    assert receipt["source_tokenizer"] == {
        "id": "fixture-tokenizer-v1",
        "sha256": "c" * 64,
    }
    assert receipt["suite_lock_sha256"] == lock["suite_lock_sha256"]
    assert receipt["source_provenance_sha256"] == provenance
    assert receipt["row_count"] == 64
    assert receipt["roundtrip_verified_rows"] == 64
    assert receipt["manifest_sha256"] == _sha(manifest_path)
    assert manifest["schema"] == "banana-smasher.balanced64-source-text.v1"
    assert manifest["historical_token_ledger"] == {
        "sha256": _sha(historical_path),
        "source_tokenizer_id": "fixture-tokenizer-v1",
        "source_tokenizer_sha256": "c" * 64,
    }
    assert [item["window_id"] for item in manifest["items"]] == list(range(1000, 1064))
    assert manifest["items"][0]["item_id"] == "source-0"
    assert manifest["items"][0]["source_class"] == "fixture"
    assert manifest["items"][0]["text"] == chr(0x1000) * 1025
    assert manifest["items"][0]["text_sha256"] == hashlib.sha256(
        (chr(0x1000) * 1025).encode()
    ).hexdigest()
    assert manifest["item_roster_sha256"] == _canonical_sha(
        [
            {
                "item_id": item["item_id"],
                "window_id": item["window_id"],
                "source_class": item["source_class"],
                "text_sha256": item["text_sha256"],
            }
            for item in manifest["items"]
        ]
    )
    assert json.loads((tmp_path / "SOURCE_TEXT_RECOVERY.json").read_text()) == receipt


def test_recover_source_text_refuses_non_roundtripping_or_ambiguous_rows(
    tmp_path: Path,
) -> None:
    from banana_smasher import recover_balanced64_source_text

    _, lock, _ = _inputs(tmp_path)
    historical_path = tmp_path / "historical-token-ledger.json"
    rows = [
        {
            "window_id": 1000 + index,
            "item_id": f"source-{index}",
            "id_gold": 1000 + index,
            "token_ids": [0x1000 + index] * 1025,
        }
        for index in range(64)
    ]
    historical_path.write_text(json.dumps({"rows": rows}, sort_keys=True) + "\n")
    lock["source_windows_sha256"] = _sha(historical_path)
    lock["suite_lock_sha256"] = _canonical_sha(lock)

    with pytest.raises(ValueError, match="ambiguous historical window identity"):
        recover_balanced64_source_text(
            historical_path,
            suite_lock=lock,
            output=tmp_path / "ambiguous.json",
            receipt_path=tmp_path / "ambiguous-receipt.json",
            tokenizer=_Tokenizer(),
        )

    del rows[0]["window_id"]
    del rows[0]["id_gold"]
    historical_path.write_text(json.dumps({"rows": rows}, sort_keys=True) + "\n")
    lock.pop("suite_lock_sha256")
    lock["source_windows_sha256"] = _sha(historical_path)
    lock["suite_lock_sha256"] = _canonical_sha(lock)
    with pytest.raises(ValueError, match="missing historical window identity"):
        recover_balanced64_source_text(
            historical_path,
            suite_lock=lock,
            output=tmp_path / "missing.json",
            receipt_path=tmp_path / "missing-receipt.json",
            tokenizer=_Tokenizer(),
        )

    for row in rows:
        row.pop("window_id", None)
    rows[0]["id_gold"] = 1000

    class _LossyTokenizer(_Tokenizer):
        tokenizer_id = "lossy-tokenizer-v1"

        def decode(self, token_ids: list[int]) -> str:
            return "lossy"

    historical_path.write_text(json.dumps({"rows": rows}, sort_keys=True) + "\n")
    lock.pop("suite_lock_sha256")
    lock["source_windows_sha256"] = _sha(historical_path)
    lock["suite_lock_sha256"] = _canonical_sha(lock)
    with pytest.raises(ValueError, match="tokenizer round-trip mismatch: window_id=1000"):
        recover_balanced64_source_text(
            historical_path,
            suite_lock=lock,
            output=tmp_path / "lossy.json",
            receipt_path=tmp_path / "lossy-receipt.json",
            tokenizer=_LossyTokenizer(),
        )
