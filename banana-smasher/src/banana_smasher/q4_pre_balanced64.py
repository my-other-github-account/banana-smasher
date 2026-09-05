"""Original-Q4 PRE against the externally frozen DS4 Balanced64 bank.

The native control is a candidate on frozen support, never a new teacher.
Historical executable availability is not asserted. This is a source-derived
canonical successor with its effective runtime recorded for both arms.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
MANIFEST = "c8f9e8a38120c2916863812f9266eb12994a076341f7a9a14951e4b7e3225010"
INVENTORY = "213381d479f45a0ccef36bd77c778b04e7f06a5c377eae4f486c4c1ad442a624"
WINDOWS = [28, 56, 68, 71, 76, 99, 107, 122, 124, 130, 141, 156, 160,
           171, 180, 183, 185, 186, 196, 210, 212, 213, 218, 228, 232,
           235, 249, 270, 272, 273, 283, 288, 290, 295, 297, 306, 307,
           309, 311, 328, 331, 357, 362, 365, 368, 374, 376, 380, 384,
           385, 391, 396, 413, 429, 430, 437, 442, 447, 454, 462, 464,
           475, 489, 499]
CONTRACT = dict(window_ids=WINDOWS, q4_inventory_sha256=INVENTORY,
                teacher_bank="TEACHER_0731_BALANCED64_V2",
                source_index_sha256=BASIS, forward_tokens=2048,
                scored_positions=1024, support=8192, microbatch=2, chunk=64,
                padding_token=1, attention="eager", attention_mask=None,
                cache="fresh-per-microbatch", bank_dtype="float16",
                score_dtype="float64", reduction="math.fsum",
                negative_policy="reject", top1="common-support-first-index")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, detail):
    if not condition:
        raise ValueError("Balanced64 PRE contract: " + detail)


def runtime_identity():
    import torch
    import transformers
    import transformers.models.deepseek_v4.modeling_deepseek_v4 as modeling
    import transformers.models.deepseek_v4.configuration_deepseek_v4 as configuration
    return dict(torch=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, modeling_sha256=sha(modeling.__file__),
                configuration_sha256=sha(configuration.__file__))


def admit(config):
    require(config["contract"] == CONTRACT, "population/source/teacher/position/runtime metadata drift")
    require(sha(config["manifest"]) == MANIFEST, "frozen manifest bytes drift")
    manifest = json.loads(Path(config["manifest"]).read_text())
    frozen = manifest["q3_frozen_contract"]
    require(frozen["ordered_window_ids"] == WINDOWS, "frozen population drift")
    require(sha(Path(config["model"]) / "model.safetensors.index.json") == BASIS, "source basis drift")
    require(sha(config["corpus"]) == frozen["corpus"]["sha256"], "token corpus bytes drift")
    require(sha(config["inventory"]) == INVENTORY, "original Q4 inventory bytes drift")
    # Exact frozen file bytes bind every position's ordered support and logprob.
    for row in frozen["row_bindings"]:
        require(sha(Path(config["teacher"]) / f't8192_win{row["window_id"]}.pt')
                == row["teacher_input_sha256"], f'teacher/support bytes window {row["window_id"]}')
    require(runtime_identity() == config["runtime"], "effective imported runtime drift")
    return frozen


class OriginalQ4:
    """Read-only K4 units; each member is checked against the frozen inventory."""
    def __init__(self, config, output):
        self.root = Path(config["q4"])
        rows = [json.loads(line) for line in Path(config["inventory"]).read_text().splitlines()]
        self.inventory = {row["relative_path"]: row for row in rows}
        require(len(self.inventory) == len(rows) == 66091, "original inventory cardinality")
        self.decoded_units = 0
        self.output = Path(output)
        self.lookup = self.sign = None

    def decode(self, layer, expert, projection):
        import torch
        from . import qtip_kernel_decompress
        from .fwht import bounded_fwht as fwht
        rel = f"L{layer:03d}/E{expert:03d}_{projection}/QTIP_UNIT.pt"
        path = self.root / rel
        expected = self.inventory[rel]
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                "original Q4 source bytes " + rel)
        unit = torch.load(path, map_location="cpu", weights_only=True)
        geom = unit["geometry"]
        require((geom["L"], geom["K"], geom["V"], geom["tlut_bits"], geom["decode_mode"])
                == (16, 4, 2, 9, "quantlut_sym"), "original K4 geometry " + rel)
        m, k = unit["shape"]
        require((m, k) == (4096, 4096 if projection == "fused13" else 2048), "K4 shape " + rel)
        if self.lookup is None:
            index = torch.arange(1 << 16, device="cuda")
            quadratic = (index + 1) * index
            self.sign = 1 - ((quadratic >> 15) & 1) * 2
            self.lookup = (quadratic >> 6) & 511
        expanded = unit["tlut"].float().to("cuda")[self.lookup]
        expanded[:, 0] *= self.sign
        raw = qtip_kernel_decompress.decode_compressed(
            16, 9, 4, 1, m, k, unit["trellis"].to("cuda").reshape(-1), expanded)
        value = raw * unit["Wscale"].to("cuda")
        value = fwht(value.T).T * unit["SV"].float().to("cuda")[:, None]
        value = fwht(value) * unit["SU"].float().to("cuda")
        require(bool(torch.isfinite(value).all()), "nonfinite K4 decode " + rel)
        self.decoded_units += 1
        return value.to(torch.bfloat16)

    def fill_layer(self, layer, gate_up, down):
        from .q4_pre_forward import atomic_json, log
        for expert in range(256):
            gate_up[expert] = self.decode(layer, expert, "fused13")
            down[expert] = self.decode(layer, expert, "down")
        atomic_json(str(self.output / "DECODE_PROGRESS.json"),
                    dict(layer=layer, decoded_units=self.decoded_units, basis=BASIS))
        log(f"Balanced64 original Q4 layer={layer} decoded_units={self.decoded_units}")


def reduce_bank(teacher_root, candidate_root, window_ids):
    """f08e1d2c FP64 normalization and ordered fsum, without a negative clamp."""
    import torch
    from .q4_pre_forward import atomic_json
    all_values, rows = [], []
    for window_id in window_ids:
        tpath = Path(teacher_root) / f"t8192_win{window_id}.pt"
        qpath = Path(candidate_root) / f"q8192_win{window_id}.pt"
        teacher = torch.load(tpath, map_location="cpu", weights_only=True)
        candidate = torch.load(qpath, map_location="cpu", weights_only=True)
        t, q = teacher["logprob"][:1024], candidate["q_lp_at_ref"]
        require(t.dtype == q.dtype == torch.float16, "FP16 bank boundary")
        require(tuple(t.shape) == tuple(q.shape) == (1024, 8192), "bank position/support shape")
        require(bool(torch.isfinite(t).all() and torch.isfinite(q).all()), "nonfinite bank")
        log_p, log_q = t.double(), q.double()
        log_p -= torch.logsumexp(log_p, dim=1, keepdim=True)
        log_q -= torch.logsumexp(log_q, dim=1, keepdim=True)
        values = (torch.exp(log_p) * (log_p - log_q)).sum(dim=1)
        require(bool(torch.isfinite(values).all() and (values >= 0).all()), "negative/nonfinite KL; no clamp")
        require(bool((log_p.argmax(dim=1) == 0).all()), "frozen teacher support order")
        matches = int((log_p.argmax(dim=1) == log_q.argmax(dim=1)).sum())
        row = dict(window_id=window_id, ordinal=WINDOWS.index(window_id), positions=1024,
                   kld_values=[repr(v) for v in values.tolist()], top1_matches=matches,
                   teacher_sha256=sha(tpath), candidate_sha256=sha(qpath))
        atomic_json(str(Path(candidate_root) / f"window-{WINDOWS.index(window_id):02d}.json"), row)
        rows.append(row)
        all_values.extend(values.tolist())
    return dict(positions=len(all_values), forward_kl=math.fsum(all_values) / len(all_values),
                top1_matches=sum(row["top1_matches"] for row in rows),
                row_files=[f'window-{row["ordinal"]:02d}.json' for row in rows])


def run(config, output, *, window_ids=None):
    # Cheap independent drift rejection must precede device/model allocation.
    require(config["contract"] == CONTRACT, "population/source/teacher/position/runtime metadata drift")
    window_ids = WINDOWS if window_ids is None else list(window_ids)
    require(window_ids == WINDOWS or window_ids == WINDOWS[:2], "only full64 or matched mb2 W28/56 gate")
    claim_path = Path(config["claim"])
    claim = json.loads(claim_path.read_text())
    require(claim.get("state") == "CLAIMED" and claim.get("intended_basis") == BASIS,
            "exclusive basis-bound compute claim absent")
    require(claim.get("executor_task_id") == config["executor_task_id"], "claim executor mismatch")
    pid = claim["workload_pid"]
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    require(stat[2] != "Z" and str(stat[21]) == str(claim["workload_startticks"]), "claim payload identity")
    admit(config)
    from .q4_pre_forward import forward, atomic_json, mem_available_bytes
    import torch
    # Worst simultaneous layer payload, hidden/cache, readout, source prefetch,
    # static weights, and allocator headroom. Require four GiB remaining.
    peak_estimate = (44 if len(window_ids) == 2 else 100) * (1 << 30)
    require(peak_estimate <= mem_available_bytes() - (4 << 30), "memory preflight")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(contract=CONTRACT, executed_window_ids=window_ids,
                  historical_runtime_recovered=False, peak_estimate_bytes=peak_estimate)
    atomic_json(str(output / "ADMISSION.json"), {**result, "runtime": config["runtime"]})
    for mode in ("native", "q4"):
        root = output / mode
        root.mkdir()
        source = OriginalQ4(config, root) if mode == "q4" else None
        args = SimpleNamespace(mode=mode, source=source, out=str(root), windows=window_ids,
                               corpus=config["corpus"], meta_dir=config["model"],
                               local_dir=config["model"], ref_dir=config["teacher"],
                               attn_implementation="eager", mb=2, chunk=64,
                               readout_mode="sort", mem_floor_gib=8, limit_layers=0,
                               cand_pos_limit=1024, timing_json=None, tag="Balanced64-PRE")
        forward(args)
        arm = reduce_bank(config["teacher"], root, window_ids)
        arm.update(runtime=runtime_identity(), decoded_units=source.decoded_units if source else 0)
        require(arm["runtime"] == config["runtime"], "runtime changed during forward")
        result[mode] = arm
        atomic_json(str(root / "ARM_TERMINAL.json"), arm)
        del source
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(str(output / "RESULT.json"), result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.config).read_text()), args.output), sort_keys=True))
