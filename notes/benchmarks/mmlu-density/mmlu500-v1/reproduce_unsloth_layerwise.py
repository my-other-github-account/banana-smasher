#!/usr/bin/env python3
"""Reproduce the published 412/500 Kimi-K3 Unsloth UD-IQ1_S MMLU-500 row.

Requires Python 3.10+, tiktoken==0.12.0, the pinned 14 GGUF shards, one CUDA
DGX Spark, and these executable SparkInfer K3 layer kernels in --binary-dir:
  kimi_k3_prefix_dump, kimi_k3_boundary_advance, kimi_k3_boundary_score

Example:
  uv run --with tiktoken==0.12.0 reproduce_unsloth_layerwise.py \
    --model-dir /models/Kimi-K3-GGUF/UD-IQ1_S \
    --binary-dir /path/to/sparkinfer/build \
    --output-dir ./mmlu500-unsloth

Use --prepare-only for a CPU/network smoke without weights or CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

LABELS = "ABCD"
ROWS = 500
N_LAYERS = 93
ITEMS_SHA256 = "df6704c4d02550b9155e106bc9a9e1bfe1164a663d509e41a76736bb60d01ded"
ITEMS_URL = (
    "https://raw.githubusercontent.com/my-other-github-account/banana-smasher/"
    "a7827bec747973ced70cd5b589bfc8b4c956862e/notes/benchmarks/"
    "mmlu-density/mmlu500-v1/items.jsonl"
)
K3_REPO = "moonshotai/Kimi-K3"
K3_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
TOKENIZER_FILES = {
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
}
CANDIDATE_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
TOKEN_AGGREGATE_SHA256 = "184b2144ec602c300018911874918d83a5057c6d1e44a173225f87cac024fccc"
TOKEN_FRAMED_SHA256 = "e95dc16aefdf1a5bc3b64e8f2f9f13706f48e183dba38142c4c5b9e62814fa50"
TOKEN_COUNTS = {"total": 57168, "min": 40, "max": 558}

MODEL_REPO = "unsloth/Kimi-K3-GGUF"
MODEL_REVISION = "a0836360ce58dfec088d966a97f2ddc8a606279b"
VARIANT = "UD-IQ1_S"
EXPECTED_CORRECT = 412
EXPECTED_GOLD_CE_BITS = 0.7522193724299207
EXPECTED_QROWS_SHA256 = "8d2514d7ee5f71a6c551d280b39e95a1bd4ae99afc42093b69cfcfde99391124"
ID_FORMAT = "{ordinal:04d}"
MODEL_FILES = (
    ("Kimi-K3-UD-IQ1_S-00001-of-00014.gguf", 6934144, "5022014e7c49d8844e9f1bc7d9fb824c0d640214540aa845690518d800286083"),
    ("Kimi-K3-UD-IQ1_S-00002-of-00014.gguf", 47094999008, "dbcc677f04a20f2be5a060c428be61230939e877749004ac6d9a528e905c7bc6"),
    ("Kimi-K3-UD-IQ1_S-00003-of-00014.gguf", 49220694048, "f3af81f43e7da2f0d5b842415135e72d6ab1542a0f49aa4a54950488f1d30445"),
    ("Kimi-K3-UD-IQ1_S-00004-of-00014.gguf", 48208733792, "c12769e969cd75f4f71008ba06801ebe20ddabd774e28ca9cb237a2c93a7ca1d"),
    ("Kimi-K3-UD-IQ1_S-00005-of-00014.gguf", 48276643808, "993ae1e7cfe60aa1bf7e52f345fae8ca0d1547d04e1daf3b276b6b44ff206d64"),
    ("Kimi-K3-UD-IQ1_S-00006-of-00014.gguf", 48006508320, "0f14f006956e1b937e2633a65e817572c35aa2a8a5b0c9cc77f3d0eaf1fe6e86"),
    ("Kimi-K3-UD-IQ1_S-00007-of-00014.gguf", 48208733792, "acd73f30b113c1b93ac343ce2f985baeb84e10e1e525b3fc3cf467fdb5605140"),
    ("Kimi-K3-UD-IQ1_S-00008-of-00014.gguf", 50742894560, "02045bdfc1c57103834d148b745631589c72d433de1a35c64b8fc3cef646a648"),
    ("Kimi-K3-UD-IQ1_S-00009-of-00014.gguf", 51608874592, "98ab3d6a28bcd5b6bacdaa4ab43f318a5d98c1eb2ef47b505c55cccd74c57703"),
    ("Kimi-K3-UD-IQ1_S-00010-of-00014.gguf", 49441859168, "03e45316276a83ae3270f6787d752a6d2e8be312d86271063e72bae07f58f6f5"),
    ("Kimi-K3-UD-IQ1_S-00011-of-00014.gguf", 51976019936, "ca93c7cfdaf48cfb4fc6910d3a6339b32a11a8e54c814ac0dfc4863ac10d7451"),
    ("Kimi-K3-UD-IQ1_S-00012-of-00014.gguf", 48526061152, "5dcc9ccdccca908527e66a6fbb6b69355f791555fccf0772d34accf98cbb849a"),
    ("Kimi-K3-UD-IQ1_S-00013-of-00014.gguf", 48728286656, "a1d376d9d85b1d7cd7a069d9a1891f0e4409e14fd9f844034f03fb9be22fd4aa"),
    ("Kimi-K3-UD-IQ1_S-00014-of-00014.gguf", 3993680640, "308eac91daf86af3f699f883fe47a88c36118329b9b78cd88d5bb5d9a2a7451f"),
)
COMPLETE_ARTIFACT_BYTES = 594040923616
PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def download_exact(url: str, path: Path, expected_sha256: str) -> None:
    if path.is_file() and sha256_file(path) == expected_sha256:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "banana-mmlu500-layerwise/1"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"download SHA mismatch for {url}: {actual}")
    atomic_write(path, data)


def frozen_encoding(asset_dir: Path):
    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError as error:
        raise RuntimeError("install the exact tokenizer: uv run --with tiktoken==0.12.0 ...") from error
    if importlib.metadata.version("tiktoken") != "0.12.0":
        raise RuntimeError("this benchmark requires tiktoken==0.12.0")
    config = json.loads((asset_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    ranks = load_tiktoken_bpe(str(asset_dir / "tiktoken.model"))
    base_count = len(ranks)
    mapped = {int(key): value["content"] for key, value in config["added_tokens_decoder"].items()}
    special = {
        mapped.get(token_id, f"<|reserved_token_{token_id}|>"): token_id
        for token_id in range(base_count, base_count + 256)
    }
    encoding = tiktoken.Encoding(
        name="kimi-k3-frozen", pat_str=PATTERN, mergeable_ranks=ranks, special_tokens=special
    )
    for letter, token_id in CANDIDATE_IDS.items():
        if encoding.encode(letter, disallowed_special=()) != [token_id]:
            raise RuntimeError(f"candidate token mismatch for {letter}")
        if encoding.decode_single_token_bytes(token_id) != letter.encode("ascii"):
            raise RuntimeError(f"candidate decode mismatch for {letter}")
    return encoding


def prepare_basis(output_dir: Path, limit: int) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    if limit < 0 or limit > ROWS:
        raise ValueError("--limit must be between 0 and 500; 0 means all 500")
    basis = output_dir / "basis"
    items_path = basis / "items.jsonl"
    download_exact(ITEMS_URL, items_path, ITEMS_SHA256)
    assets = basis / "tokenizer"
    for name, expected_sha in TOKENIZER_FILES.items():
        download_exact(
            f"https://huggingface.co/{K3_REPO}/resolve/{K3_REVISION}/{name}",
            assets / name,
            expected_sha,
        )
    encoding = frozen_encoding(assets)
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line]
    if len(items) != ROWS:
        raise RuntimeError(f"expected 500 benchmark rows, got {len(items)}")

    aggregate = hashlib.sha256()
    framed = hashlib.sha256()
    counts: list[int] = []
    token_rows: list[tuple[str, list[int]]] = []
    for ordinal, item in enumerate(items):
        if item.get("sample_ordinal") != ordinal:
            raise RuntimeError(f"sample ordinal drift at row {ordinal}")
        prompt = item.get("prompt")
        answer = item.get("answer_letter")
        if not isinstance(prompt, str) or answer not in LABELS:
            raise RuntimeError(f"invalid prompt/answer at row {ordinal}")
        if item.get("answer_index") != LABELS.index(answer):
            raise RuntimeError(f"answer index drift at row {ordinal}")
        ids = encoding.encode(prompt, disallowed_special=())
        packed = b"".join(struct.pack("<I", token_id) for token_id in ids)
        aggregate.update(packed)
        framed.update(struct.pack("<I", len(ids)))
        framed.update(packed)
        counts.append(len(ids))
        token_rows.append((ID_FORMAT.format(ordinal=ordinal), ids))

    observed_counts = {"total": sum(counts), "min": min(counts), "max": max(counts)}
    if aggregate.hexdigest() != TOKEN_AGGREGATE_SHA256:
        raise RuntimeError("frozen literal token aggregate SHA mismatch")
    if framed.hexdigest() != TOKEN_FRAMED_SHA256 or observed_counts != TOKEN_COUNTS:
        raise RuntimeError("frozen literal token framing/count mismatch")

    selected_n = ROWS if limit == 0 else limit
    selected = items[:selected_n]
    token_rows = token_rows[:selected_n]
    tokens_path = basis / "tokens.tsv"
    token_text = "".join(
        item_id + "\t" + ",".join(map(str, token_ids)) + "\n"
        for item_id, token_ids in token_rows
    )
    atomic_write(tokens_path, token_text.encode("utf-8"))
    receipt = {
        "status": "PASS",
        "rows": selected_n,
        "items_sha256": ITEMS_SHA256,
        "token_ids_500_sha256": TOKEN_AGGREGATE_SHA256,
        "token_counts_500": TOKEN_COUNTS,
        "selected_tokens": sum(counts[:selected_n]),
        "tokens_tsv": str(tokens_path),
        "tokens_tsv_sha256": sha256_file(tokens_path),
        "candidate_ids": CANDIDATE_IDS,
        "prompt": "literal item['prompt']; no BOS/EOS, chat template, thinking, or generation wrapper",
    }
    atomic_write(basis / "receipt.json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode())
    return selected, tokens_path, receipt


def verify_model(model_dir: Path, output_dir: Path) -> Path:
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if sum(size for _, size, _ in MODEL_FILES) != COMPLETE_ARTIFACT_BYTES:
        raise RuntimeError("embedded model manifest byte total is internally inconsistent")
    verified = []
    for index, (name, expected_size, expected_sha) in enumerate(MODEL_FILES, 1):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned model member: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(f"model size mismatch: {name}: {actual_size} != {expected_size}")
        print(f"model {index}/{len(MODEL_FILES)} hashing {name}", flush=True)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"model SHA mismatch: {name}: {actual_sha}")
        verified.append({"name": name, "bytes": actual_size, "sha256": actual_sha})
    receipt = {
        "status": "PASS",
        "repository": MODEL_REPO,
        "revision": MODEL_REVISION,
        "variant": VARIANT,
        "complete_artifact_bytes": COMPLETE_ARTIFACT_BYTES,
        "files": verified,
    }
    atomic_write(
        output_dir / "model-verification.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    return model_dir / MODEL_FILES[0][0]


def find_binaries(binary_dir: Path) -> dict[str, Path]:
    names = ("kimi_k3_prefix_dump", "kimi_k3_boundary_advance", "kimi_k3_boundary_score")
    binaries = {name: (binary_dir / name).resolve() for name in names}
    for name, path in binaries.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"missing executable {name}: {path}")
    return binaries


def boundary_header(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        raw = handle.read(36)
    if len(raw) != 36:
        raise RuntimeError(f"short boundary header: {path}")
    magic, version, hidden, first, last, items, tokens = struct.unpack("<8sIIIIIQ", raw)
    if magic != b"K3PFX01\0" or version != 1:
        raise RuntimeError(f"invalid K3 boundary: {path}")
    return {"hidden": hidden, "first": first, "last": last, "items": items, "tokens": tokens}


def run_stage(
    name: str,
    command: list[str],
    output: Path,
    output_dir: Path,
    env: dict[str, str],
    expected_header: dict[str, int] | None = None,
) -> None:
    if output.exists():
        print(f"resume: {name} already exists", flush=True)
    else:
        partial = Path(str(output) + ".partial")
        if partial.exists():
            raise RuntimeError(f"stale partial must be removed or recovered: {partial}")
        print(f"run: {name}", flush=True)
        with (output_dir / f"{name}.stdout").open("w") as stdout, (
            output_dir / f"{name}.stderr"
        ).open("w") as stderr:
            subprocess.run(command, check=True, env=env, stdout=stdout, stderr=stderr)
        if not output.is_file():
            raise RuntimeError(f"stage returned success without output: {name}")
    if expected_header is not None:
        actual = boundary_header(output)
        for key, value in expected_header.items():
            if actual[key] != value:
                raise RuntimeError(f"boundary mismatch for {name}: {key}={actual[key]} != {value}")


def resume_boundary(output_dir: Path, rows: int, tokens: int) -> tuple[Path | None, int]:
    """Return the sole retained boundary and its last layer, or no boundary."""
    boundaries = sorted(output_dir.glob("boundary-l*.k3pfx"))
    if not boundaries:
        return None, -1
    if len(boundaries) != 1:
        raise RuntimeError(f"expected at most one retained boundary, found {boundaries}")
    header = boundary_header(boundaries[0])
    if header["items"] != rows or header["tokens"] != tokens:
        raise RuntimeError(f"retained boundary has the wrong row/token basis: {boundaries[0]}")
    return boundaries[0], header["last"]


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [value / total for value in weights]


def aggregate(qrows_path: Path, items: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in qrows_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != len(items):
        raise RuntimeError(f"expected {len(items)} scored rows, got {len(rows)}")
    correct = 0
    gold_bits: list[float] = []
    for ordinal, (row, item) in enumerate(zip(rows, items)):
        expected_id = ID_FORMAT.format(ordinal=ordinal)
        if row.get("id") != expected_id:
            raise RuntimeError(f"qrow ID/order mismatch at {ordinal}: {row.get('id')!r}")
        logits_map = row.get("candidate_logits")
        if not isinstance(logits_map, dict) or set(logits_map) != set(LABELS):
            raise RuntimeError(f"missing exact A/B/C/D logits at row {ordinal}")
        logits = [float(logits_map[label]) for label in LABELS]
        if not all(math.isfinite(value) for value in logits):
            raise RuntimeError(f"non-finite logits at row {ordinal}")
        probabilities = softmax(logits)
        prediction = LABELS[max(range(4), key=logits.__getitem__)]
        if row.get("prediction") != prediction:
            raise RuntimeError(f"prediction/argmax mismatch at row {ordinal}")
        native_probabilities = row.get("candidate_probabilities")
        if isinstance(native_probabilities, dict):
            error = max(
                abs(float(native_probabilities[label]) - probabilities[index])
                for index, label in enumerate(LABELS)
            )
            if error > 3e-7:
                raise RuntimeError(f"candidate probability mismatch at row {ordinal}: {error}")
        correct += prediction == item["answer_letter"]
        gold_bits.append(-math.log2(probabilities[item["answer_index"]]))

    qrows_sha = sha256_file(qrows_path)
    full = len(items) == ROWS
    summary = {
        "status": "PASS" if not full or correct == EXPECTED_CORRECT else "FAIL",
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "variant": VARIANT,
        "n": len(items),
        "correct": correct,
        "mmlu_percent": 100.0 * correct / len(items),
        "gold_cross_entropy_bits": sum(gold_bits) / len(gold_bits),
        "candidate_ids": CANDIDATE_IDS,
        "qrows_sha256": qrows_sha,
        "historical": {
            "correct": EXPECTED_CORRECT,
            "n": ROWS,
            "gold_cross_entropy_bits": EXPECTED_GOLD_CE_BITS,
            "qrows_sha256": EXPECTED_QROWS_SHA256,
        },
        "full_score_matches_historical": full and correct == EXPECTED_CORRECT,
        "qrows_bit_identical_to_historical": full and qrows_sha == EXPECTED_QROWS_SHA256,
    }
    atomic_write(output_dir / "result.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
    return summary


def execute(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items, tokens_path, basis = prepare_basis(output_dir, args.limit)
    print(json.dumps({"basis": basis}, sort_keys=True), flush=True)
    if args.prepare_only:
        return 0
    qrows = output_dir / "candidate-logits.jsonl"
    if args.aggregate_only:
        summary = aggregate(qrows, items, output_dir)
        print(json.dumps(summary, sort_keys=True))
        return 0 if summary["status"] == "PASS" else 3
    if args.model_dir is None or args.binary_dir is None:
        raise RuntimeError("normal execution requires --model-dir and --binary-dir")

    model = verify_model(args.model_dir.resolve(), output_dir)
    binaries = find_binaries(args.binary_dir.resolve())
    selected_tokens = int(basis["selected_tokens"])
    config = {
        "model": {"repository": MODEL_REPO, "revision": MODEL_REVISION, "variant": VARIANT},
        "items_sha256": ITEMS_SHA256,
        "tokens_tsv_sha256": basis["tokens_tsv_sha256"],
        "rows": len(items),
        "n_layers": N_LAYERS,
        "initial_layers": args.initial_layers,
        "chunk_layers": args.chunk_layers,
        "batch": args.batch,
    }
    config_path = output_dir / "run-config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError(f"existing run config differs: {config_path}")
    atomic_write(config_path, (json.dumps(config, indent=2, sort_keys=True) + "\n").encode())

    if not 0 < args.initial_layers <= N_LAYERS or args.chunk_layers < 1 or args.batch < 1:
        raise ValueError("invalid layer or batch configuration")
    env = os.environ.copy()
    env.setdefault("SPARKINFER_K3_PREFILL_CHUNK", "64")
    env.setdefault("SPARKINFER_K3_KDA_QKVG_BATCH", "1")
    env.setdefault("SPARKINFER_K3_PREFILL_ROUTED_DOWN", "1")

    boundary, last = resume_boundary(output_dir, len(items), selected_tokens)
    if boundary is None:
        last = args.initial_layers - 1
        boundary = output_dir / f"boundary-l0-{last}.k3pfx"
        run_stage(
            f"layers-0-{last}",
            [str(binaries["kimi_k3_prefix_dump"]), str(model), str(tokens_path), str(boundary), str(args.initial_layers), "0"],
            boundary,
            output_dir,
            env,
            {"first": 0, "last": last, "items": len(items), "tokens": selected_tokens},
        )
    first = last + 1
    while first < N_LAYERS:
        last = min(N_LAYERS - 1, first + args.chunk_layers - 1)
        next_boundary = output_dir / f"boundary-l{first}-{last}.k3pfx"
        run_stage(
            f"layers-{first}-{last}",
            [str(binaries["kimi_k3_boundary_advance"]), str(model), str(boundary), str(next_boundary), str(first), str(last), str(args.batch)],
            next_boundary,
            output_dir,
            env,
            {"first": first, "last": last, "items": len(items), "tokens": selected_tokens},
        )
        boundary.unlink()
        boundary = next_boundary
        first = last + 1
    run_stage(
        "score",
        [str(binaries["kimi_k3_boundary_score"]), str(model), str(boundary), str(qrows)],
        qrows,
        output_dir,
        env,
    )
    summary = aggregate(qrows, items, output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 3


def self_test() -> None:
    assert sum(size for _, size, _ in MODEL_FILES) == COMPLETE_ARTIFACT_BYTES
    assert softmax([4.0, 3.0, 2.0, 1.0])[0] > 0.64
    assert ID_FORMAT.format(ordinal=7) == "0007"
    print("self-test: PASS")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    value.add_argument("--model-dir", type=Path, help="directory containing the exact pinned GGUF members")
    value.add_argument("--binary-dir", type=Path, help="directory containing the three SparkInfer layer executables")
    value.add_argument("--output-dir", type=Path, default=Path("mmlu500-unsloth-layerwise"))
    value.add_argument("--limit", type=int, default=0, help="smoke prefix; 0 means all 500")
    value.add_argument("--initial-layers", type=int, default=1)
    value.add_argument("--chunk-layers", type=int, default=4)
    value.add_argument("--batch", type=int, default=64)
    value.add_argument("--prepare-only", action="store_true", help="authenticate and tokenize the frozen 500 rows, then stop")
    value.add_argument("--aggregate-only", action="store_true", help="aggregate an existing candidate-logits.jsonl")
    value.add_argument("--self-test", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    return execute(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
