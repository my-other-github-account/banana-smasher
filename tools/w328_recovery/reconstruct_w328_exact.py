#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from typing import Any

TASK = "t_03c6894c"
RUN = int(os.environ.get("W328_CLAIM_RUN_ID", "4725"))
BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
EXPECTED_CANDIDATE_SHA = "9ff57dceb525fdb695596145fd527e24e84852069c977d6ef4237d57cd3dcc78"
EXPECTED_TEACHER_SHA = "e494b7fd83bcce7ee0bbf14371bee2d87005ea846cdd178dbd69379c2c336a82"
ROOT = Path("/home/dnola/missions/W328_SEALED_RECON_t_03c6894c_s5w")
CODE_ROOT = Path(__file__).resolve().parent
MODEL = Path("/home/dnola/models/hf/DeepSeek-V4-Flash-0731")
CLAIM = Path("/home/dnola/HOST_CLAIM.json")
BUILDER = CODE_ROOT / "t8192_ds4_build_v3.py"
PLANE = CODE_ROOT / "all43_planesource_run1698_retry14_routekinds.py"
TEACHER_ROOT = ROOT / "reference/inputs/BALANCED64_TEACHER"
TEACHER = TEACHER_ROOT / "t8192_win328.pt"
CORPUS = ROOT / "reference/inputs/windows_ds4_eval.json"
FREEZE = ROOT / "receipts/FINAL_43_ROUTE_CENSUS_RUN1698.json"
L024_MANIFEST = ROOT / "receipts/L024_PRODUCT_MANIFEST.json"
OUT = Path("/dev/shm/W328_SEALED_RECON_t_03c6894c/output")
TERMINAL = ROOT / "receipts/W328_EXACT_RECONSTRUCTION_TERMINAL.json"

EXPECTED_FILES = {
    BUILDER: "686f4d1fbe367811d203556891a16c7adc9ba9fbb71d5a078666f169c0bdc054",
    PLANE: "454793702ede1305cb19a11ec0967c95e3ae77b20de0ebb3f7d2bc87c1b0bb81",
    ROOT / "code/complete_provider_recovery4_local30.py": "37addd4d86479194d15eb727a17a7920aa2bcc063f1645e74a5bcf7b24c60780",
    ROOT / "code/exact_k2_provider.py": "04b70b06450bf94320543e0f34b806f8ac705382fad04dc6b4e6cc401fd9bb7c",
    FREEZE: "2dcc28497deb834164be26e267fdf4c30cc951342c73f47ce78b207354275fc9",
    ROOT / "receipts/BALANCED64_V1_SCORER_INPUTS.json": "a3814092c1a2dab253b348a444e5a9c5bdc426c0b85a05965c404e5bae954091",
    ROOT / "receipts/PARTIAL_ANCHOR_STAGE6_L024_DESTINATION_READBACK_RUN1489.json": "150e0b869ec1879c0a79c44a48619ea861993b02bcabb8034ca4fb0e347ce595",
    ROOT / "receipts/L034_E161_E170_RECOVERY3_TERMINAL.json": "3c62229f5a17eaf9ee4b0923f1048de65b591bea8aaeeb28342bcaa7492baedb",
    ROOT / "l034/COMPLETE_768_ROSTER.json": "13aaa61931aa362a355854aad7bfdb78db328833dfcb83f2444435d058ad2140",
    ROOT / "receipts/L039_PRODUCT_LAYER_TERMINAL.json": "1ee173b3a5bdb7966684d0027054a68eb35433ee5d26e184198f71675bc2c56a",
    L024_MANIFEST: "5089b25cba123a5564df7dd819221968ace8b71c0c82654fbf2d5f28b264ffa1",
    ROOT / "receipts/L024_PRODUCT_TERMINAL.json": "3bbf52081d4a4a70617c44565c4cf1b76d6ef84978d158586ee310a3f65cd394",
    TEACHER: EXPECTED_TEACHER_SHA,
    MODEL / "model.safetensors.index.json": BASIS,
}


def sha_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(value: Any) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(data).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_l024() -> dict[str, Any]:
    manifest = json.loads(L024_MANIFEST.read_text())
    if manifest.get("status") != "PASS" or manifest.get("basis") != BASIS or manifest.get("layer") != 24 or manifest.get("complete_members") != 768:
        raise RuntimeError("L024 manifest contract mismatch")
    root = ROOT / "l024/product_artifacts/L024"
    bad = []
    total = 0
    for row in manifest["members"]:
        artifact = row["artifact"]
        path = root / f"E{int(row['expert']):03d}" / f"{row['projection']}.k2wire"
        if not path.is_file() or path.stat().st_size != int(artifact["bytes"]) or sha_file(path) != artifact["sha256"]:
            bad.append(str(path))
        else:
            total += path.stat().st_size
    if bad:
        raise RuntimeError(f"L024 exact readback failed: {bad[:4]}")
    lut = ROOT / "l024/artifacts/config/L024_PARENT_LUT.fp16.bin"
    if sha_file(lut) != "1fcb3546038bc65ab7847ef4473a2d1a8c66631315655c1b3d9f989325572a3c":
        raise RuntimeError("L024 LUT mismatch")
    return {"members": 768, "bytes": total, "manifest_sha256": EXPECTED_FILES[L024_MANIFEST], "lut_sha256": sha_file(lut)}


def main() -> int:
    started_unix = time.time()
    if socket.gethostname() != "spark-5-work":
        raise RuntimeError(f"wrong host {socket.gethostname()}")
    claim_raw = CLAIM.read_bytes()
    claim = json.loads(claim_raw)
    required_claim = {"status": "CLAIMED", "state": "CLAIMED", "task_id": TASK, "owner_task_id": TASK, "owner_run_id": RUN, "intended_basis": BASIS}
    drift = {key: (claim.get(key), expected) for key, expected in required_claim.items() if claim.get(key) != expected}
    if drift or float(claim.get("lease_expires_unix", 0)) <= time.time():
        raise RuntimeError(f"claim authority refused {drift}")
    observed = {str(path): sha_file(path) if path.is_file() else None for path in EXPECTED_FILES}
    bad_inputs = {path: (EXPECTED_FILES[Path(path)], value) for path, value in observed.items() if value != EXPECTED_FILES[Path(path)]}
    if bad_inputs:
        raise RuntimeError(f"immutable input drift {bad_inputs}")
    l024 = verify_l024()
    import torch
    cuda_free, cuda_total = torch.cuda.mem_get_info()
    if cuda_free < 72 * (1 << 30):
        raise RuntimeError(f"CUDA free refusal {cuda_free}")
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("nonempty reconstruction output; retries are forbidden")
    OUT.mkdir(parents=True, exist_ok=True)
    builder = load_module("w328_exact_builder", BUILDER)
    plane_source = load_module("w328_exact_plane_source", PLANE)
    plane_source.BUILDER = builder
    builder.PlaneSource = plane_source.PlaneSource
    original_argv = sys.argv
    try:
        sys.argv = [
            str(BUILDER), "--mode", "planes", "--planes-dir", str(FREEZE),
            "--ref-dir", str(TEACHER_ROOT), "--corpus", str(CORPUS),
            "--meta-dir", str(MODEL),
            "--remote", "dnola@192.168.200.9:/home/dnola/models/hf/DeepSeek-V4-Flash-0731",
            "--shard-buf", "/dev/shm/W328_SEALED_RECON_t_03c6894c/shard_buf", "--keep-shards", "3",
            "--out", str(OUT),
            "--cand-pos-limit", "1024", "--count", "1", "--chunk", "1", "--mb", "2",
            "--windows", "328", "--tag", "QTIP2_V7_ALL43_UNIFORM_BALANCED64_RUN1698",
            "--hidden-checkpoint-dir", str(ROOT / "hidden_checkpoints"),
        ]
        return_code = int(builder.main() or 0)
    finally:
        sys.argv = original_argv
    candidate = OUT / "q8192_win328.pt"
    candidate_sha = sha_file(candidate) if candidate.is_file() else None
    status = "PASS_EXACT_RECONSTRUCTION" if return_code == 0 and candidate_sha == EXPECTED_CANDIDATE_SHA else "FAILED_CLOSED_CANDIDATE_SHA_MISMATCH"
    payload = torch.load(candidate, map_location="cpu", weights_only=True) if candidate.is_file() else {}
    terminal = {
        "schema": "banana-smasher-w328-exact-sealed-candidate-reconstruction-v1",
        "status": status,
        "task_id": TASK,
        "run_id": RUN,
        "host": socket.gethostname(),
        "basis_sha256": BASIS,
        "claim_sha256_at_start": hashlib.sha256(claim_raw).hexdigest(),
        "producer_fixture": {"task_id": "t_bb990c93", "board_run": 1698, "freeze_sha256": EXPECTED_FILES[FREEZE], "candidate_expected_sha256": EXPECTED_CANDIDATE_SHA},
        "reconstruction": {"builder_original_sha256": EXPECTED_FILES[BUILDER], "transport_only_plane_source_sha256": EXPECTED_FILES[PLANE], "count": 1, "window_order": [328], "microbatch": 2, "tag": "QTIP2_V7_ALL43_UNIFORM_BALANCED64_RUN1698"},
        "l024_readback": l024,
        "teacher": {"path": str(TEACHER), "sha256": EXPECTED_TEACHER_SHA},
        "candidate": {"path": str(candidate), "sha256": candidate_sha, "expected_sha256": EXPECTED_CANDIDATE_SHA, "bytes": candidate.stat().st_size if candidate.is_file() else None},
        "q_lp_at_ref": {"sha256": tensor_sha(payload["q_lp_at_ref"]) if payload else None, "dtype": str(payload["q_lp_at_ref"].dtype) if payload else None, "shape": list(payload["q_lp_at_ref"].shape) if payload else None},
        "q_argmax": {"sha256": tensor_sha(payload["q_argmax"]) if payload else None, "dtype": str(payload["q_argmax"].dtype) if payload else None, "shape": list(payload["q_argmax"].shape) if payload else None},
        "started_unix": started_unix,
        "completed_unix": time.time(),
        "elapsed_seconds": time.time() - started_unix,
        "no_public_parity_run": True,
        "no_private_scorer": True,
    }
    terminal_sha = atomic_json(TERMINAL, terminal)
    print(json.dumps({"status": status, "terminal": str(TERMINAL), "terminal_sha256": terminal_sha, "candidate_sha256": candidate_sha, "q_lp_at_ref_sha256": terminal["q_lp_at_ref"]["sha256"]}, sort_keys=True), flush=True)
    return 0 if status == "PASS_EXACT_RECONSTRUCTION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
