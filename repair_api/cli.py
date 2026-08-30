"""The only command-line entry point for repair_api."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

from .api import ResidentRepairAPI
from .balanced64 import ARTIFACT_SCHEMA, BALANCED64_SPEC
from .core import CANONICAL_BASIS_SHA256


def distributed_identity(environment: Mapping[str, str] | None = None) -> dict[str, int]:
    """Read rank identity only from a genuine torch.distributed.run environment."""
    env = os.environ if environment is None else environment
    world_size = int(env.get("WORLD_SIZE", "1"))
    rank = int(env.get("RANK", "0"))
    local_rank = int(env.get("LOCAL_RANK", str(rank)))
    if world_size > 1 and not env.get("TORCHELASTIC_RUN_ID"):
        raise RuntimeError(
            "multi-rank repair_api must be launched with torch.distributed.run -m repair_api"
        )
    if world_size < 1 or rank not in range(world_size) or local_rank < 0:
        raise RuntimeError("invalid torch.distributed.run rank identity")
    return {"world_size": world_size, "rank": rank, "local_rank": local_rank}


def _windows(value: str | None):
    if value is None:
        return None
    return [int(item) for item in value.split(",") if item.strip()]


def _add_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--claim-path", type=Path)
    parser.add_argument("--task-id")
    parser.add_argument("--peak-gib", type=float, default=0.0)


def _preflight_arguments(args: argparse.Namespace) -> dict[str, object]:
    return {
        "claim_path": args.claim_path,
        "task_id": args.task_id,
        "peak_gib": args.peak_gib,
    }


def _write_smoke_artifact(root: Path) -> None:
    windows = list(range(64))
    checkpoint = root / "checkpoints" / "UPDATE_000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"repair-api-smoke\n")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics = root / "score" / "rows.json"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        json.dumps(
            {
                "rows": [
                    {"window": window, "positions": 1024, "kld_sum": 0.0, "top1": 1024}
                    for window in windows
                ]
            }
        )
        + "\n"
    )
    manifest = {
        "schema": ARTIFACT_SCHEMA,
        "identity": {
            "basis_sha256": CANONICAL_BASIS_SHA256,
            "builder_eval_corpus_sha256": "2" * 64,
            "train_score_corpus_sha256": "3" * 64,
            "teacher_inventory": {"schema": "smoke", "sha256": "4" * 64},
        },
        "score": {
            "spec": BALANCED64_SPEC,
            "positions_per_window": 1024,
            "support": 8192,
            "window_ids": windows,
            "teacher_dir": "score/teacher",
            "candidate_dir_template": "score/candidates/{checkpoint}",
            "row_metrics": {"UPDATE_000": "score/rows.json"},
        },
        "checkpoints": {
            "UPDATE_000": {
                "path": "checkpoints/UPDATE_000.pt",
                "sha256": checkpoint_sha,
                "identity_sha256": "1" * 64,
                "parent_sha256": "0" * 64,
                "next_update": 0,
            }
        },
    }
    (root / "ARTIFACT.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")


def _smoke(window_count: int) -> dict[str, object]:
    if window_count < 1 or window_count > 64:
        raise RuntimeError("smoke --windows must be between 1 and 64")
    with tempfile.TemporaryDirectory(prefix="repair-api-smoke-") as temporary:
        root = Path(temporary)
        _write_smoke_artifact(root)
        api = ResidentRepairAPI.open(root)
        result = api.score("UPDATE_000", windows=range(window_count))
        return {
            "status": "PASS",
            "operation": "score",
            "entrypoint": "ResidentRepairAPI.score",
            "windows": window_count,
            "positions": result.positions,
            "support": result.support,
            "basis_sha256": result.identity["basis_sha256"],
            "preflight": api.last_preflight,
            "kld_mean": result.kld,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m repair_api")
    verbs = parser.add_subparsers(dest="verb", required=True)

    generate = verbs.add_parser("generate-candidates")
    generate.add_argument("--artifact-root", type=Path, required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--builder-template", type=Path, required=True)
    generate.add_argument("--ref-dir", type=Path, required=True)
    generate.add_argument("--corpus", type=Path, required=True)
    generate.add_argument("--meta-dir", type=Path, required=True)
    generate.add_argument("--python-executable", default=sys.executable)
    generate.add_argument("--mode", choices=("w2", "planes"), default="w2")
    generate.add_argument("--remote")
    generate.add_argument("--local-dir", type=Path)
    generate.add_argument("--windows")
    generate.add_argument("--chunk", type=int, default=8)
    generate.add_argument("--mb", type=int, default=1)
    _add_preflight_arguments(generate)

    score = verbs.add_parser("score")
    score.add_argument("--artifact-root", type=Path, required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--then-checkpoint")
    score.add_argument("--then-receipt", type=Path)
    score.add_argument("--routed-post")
    score.add_argument("--route", type=Path)
    score.add_argument("--receipt", type=Path)
    score.add_argument("--windows")
    _add_preflight_arguments(score)

    parity = verbs.add_parser("parity-tap")
    parity.add_argument("--artifact-root", type=Path, required=True)
    parity.add_argument("--checkpoint", required=True)
    parity.add_argument("--window", type=int, required=True)
    parity.add_argument("--mode", choices=("current", "sealed_reference"), default="current")
    parity.add_argument("--route", type=Path)
    parity.add_argument("--receipt", type=Path, required=True)
    _add_preflight_arguments(parity)

    continuation = verbs.add_parser("continue-training")
    continuation.add_argument("--artifact-root", type=Path, required=True)
    continuation.add_argument("--start-checkpoint", default="UPDATE_016")
    continuation.add_argument("--milestones", default="20,32,48,64")
    continuation.add_argument("--config", type=Path, required=True)
    continuation.add_argument("--receipt", type=Path, required=True)
    _add_preflight_arguments(continuation)

    lut_only = verbs.add_parser("v7-lut-only-update")
    lut_only.add_argument("--artifact-root", type=Path, required=True)
    lut_only.add_argument("--start-checkpoint", default="PRE")
    lut_only.add_argument("--config", type=Path, required=True)
    lut_only.add_argument("--trainable-luts", required=True)
    lut_only.add_argument("--lut-lr", type=float, required=True)
    lut_only.add_argument("--receipt", type=Path, required=True)

    diagnostic = verbs.add_parser("diagnostic-perturb-validate")
    diagnostic.add_argument("--artifact-root", type=Path, required=True)
    diagnostic.add_argument("--start-checkpoint", required=True)
    diagnostic.add_argument("--config", type=Path, required=True)
    diagnostic.add_argument("--direction", type=int, choices=(-1, 1), default=-1)
    diagnostic.add_argument("--train-windows")
    diagnostic.add_argument(
        "--objective-composition",
        choices=(
            "equal_norm",
            "pcgrad_equal_norm",
            "symmetric_always_project_equal_norm",
            "symmetric_always_project_residual_equal_norm",
            "symmetric_always_project_residual_reciprocal_original_mean_norm",
            "symmetric_always_project_residual_reciprocal_second_original_mean_norm",
            "symmetric_always_project_residual_original_mean_norm",
            "symmetric_always_project_residual_second_only_original_mean_norm",
            "symmetric_always_project_residual_first_only_original_mean_norm",
            "symmetric_always_project_residual_first_only_original_mean_projection",
            "symmetric_always_project_residual_second_only_original_mean_projection",
            "symmetric_always_project_residual_common_original_mean_projection",
            "symmetric_always_project_residual_reciprocal_original_mean_projection",
            "symmetric_always_project_residual_reciprocal_second_original_mean_projection",
            "symmetric_always_project_residual_reciprocal_second_first_constituent_projection",
            "symmetric_always_project_residual_reciprocal_second_second_constituent_projection",
            "symmetric_always_project_residual_reciprocal_first_second_constituent_projection",
            "symmetric_always_project_residual_reciprocal_first_first_constituent_projection",
            "symmetric_always_project_residual_reciprocal_first_first_constituent_projected_mean_target",
            "symmetric_always_project_residual_reciprocal_first_second_constituent_projected_mean_target",
            "symmetric_always_project_residual_reciprocal_second_first_constituent_projected_mean_target",
            "symmetric_always_project_residual_reciprocal_second_second_constituent_projected_mean_target",
            "symmetric_always_project_residual_reciprocal_second_projected_mean_axis_projected_mean_target",
            "symmetric_always_project_residual_reciprocal_second_projected_mean_axis_renormalized_projected_mean_target",
            "ordered_second_project_residual_equal_norm",
            "ordered_first_project_residual_equal_norm",
            "ordered_first_project_equal_norm",
            "ordered_second_project_equal_norm",
            "ordered_second_project_original_mean_norm",
            "ordered_second_project_residual_equal_norm_original_mean_norm",
            "ordered_second_project_residual_equal_norm_residual_only_original_mean_norm",
            "ordered_second_project_residual_equal_norm_first_only_original_mean_norm",
            "ordered_second_project_residual_equal_norm_reciprocal_original_mean_norm",
            "ordered_second_project_residual_equal_norm_reciprocal_residual_original_mean_norm",
            "ordered_second_project_residual_reciprocal_original_mean_norm",
            "ordered_second_project_residual_reciprocal_first_original_mean_norm",
            "ordered_first_project_residual_reciprocal_second_original_mean_norm",
            "ordered_first_project_residual_reciprocal_first_original_mean_norm",
        ),
    )
    diagnostic.add_argument("--windows", default="28")
    diagnostic.add_argument("--receipt", type=Path, required=True)
    _add_preflight_arguments(diagnostic)

    stage = verbs.add_parser("resident-stage")
    stage.add_argument("--artifact-root", type=Path, required=True)
    stage.add_argument("--checkpoint", required=True)
    stage.add_argument("--config", type=Path, required=True)
    stage.add_argument("--ready", type=Path, required=True)
    stage.add_argument("--control", type=Path, required=True)

    compare = verbs.add_parser("resume-compare")
    compare.add_argument("--artifact-root", type=Path, required=True)
    compare.add_argument("--resume-checkpoint", required=True)
    compare.add_argument("--scratch-checkpoint", required=True)
    compare.add_argument("--windows")
    compare.add_argument("--receipt", type=Path)
    _add_preflight_arguments(compare)

    equivalence = verbs.add_parser("resume-equivalence")
    equivalence.add_argument("--artifact-root", type=Path, required=True)
    equivalence.add_argument("--config", type=Path, required=True)
    equivalence.add_argument("--checkpoint-dir", type=Path, required=True)
    equivalence.add_argument("--receipt", type=Path, required=True)

    continuous = verbs.add_parser("continuous-four-updates")
    continuous.add_argument("--artifact-root", type=Path, required=True)
    continuous.add_argument("--config", type=Path, required=True)
    continuous.add_argument("--receipt", type=Path, required=True)

    smoke = verbs.add_parser("smoke")
    smoke.add_argument("--windows", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verb == "smoke":
        result = _smoke(args.windows)
    elif args.verb == "parity-tap":
        result = ResidentRepairAPI.open(args.artifact_root).parity_tap(
            args.checkpoint,
            window=args.window,
            mode=args.mode,
            route=json.loads(args.route.read_text()) if args.route is not None else None,
            receipt_path=args.receipt,
            preflight=_preflight_arguments(args),
        )
    elif args.verb == "score":
        api = ResidentRepairAPI.open(args.artifact_root)
        if args.routed_post is not None:
            if args.then_checkpoint is not None:
                raise RuntimeError("score --routed-post cannot be combined with --then-checkpoint")
            if args.route is None:
                raise RuntimeError("score --routed-post requires --route")
            result = api.score_routed_k2(
                args.checkpoint,
                args.routed_post,
                route=json.loads(args.route.read_text()),
                windows=_windows(args.windows),
                receipt_path=args.receipt,
            )
        else:
            first = api.score(
                args.checkpoint,
                windows=_windows(args.windows),
                receipt_path=args.receipt,
                preflight=_preflight_arguments(args),
            ).as_dict()
            if args.then_checkpoint is None:
                result = first
            else:
                if args.then_receipt is None:
                    raise RuntimeError("score --then-checkpoint requires --then-receipt")
                second = api.score(
                    args.then_checkpoint,
                    windows=_windows(args.windows),
                    receipt_path=args.then_receipt,
                    preflight=_preflight_arguments(args),
                ).as_dict()
                result = {
                    "schema": "repair-api-resident-score-sequence-v1",
                    "status": "PASS",
                    "public_api_method": "ResidentRepairAPI.score",
                    "resident_backend_reused": True,
                    "checkpoints": [args.checkpoint, args.then_checkpoint],
                    "results": [first, second],
                }
    elif args.verb == "continuous-four-updates":
        from .continuous_four_updates_official import run_official_continuous_four_updates
        api = ResidentRepairAPI.open(args.artifact_root)
        config = json.loads(args.config.read_text())
        config.update(distributed_identity())
        core_receipt = args.receipt.with_name(args.receipt.stem + ".api-core.json")
        result = run_official_continuous_four_updates(
            api, config=config, receipt_path=core_receipt
        )
        api._write_immutable_receipt(args.receipt, result)
    elif args.verb == "resume-equivalence":
        from .resume_equivalence_official import run_official_resume_equivalence
        api = ResidentRepairAPI.open(args.artifact_root)
        config = json.loads(args.config.read_text())
        config.update(distributed_identity())
        core_receipt = args.receipt.with_name(args.receipt.stem + ".api-core.json")
        result = run_official_resume_equivalence(
            api, config=config, receipt_path=core_receipt,
            checkpoint_dir=args.checkpoint_dir,
        )
        api._write_immutable_receipt(args.receipt, result)
    elif args.verb == "resume-compare":
        result = ResidentRepairAPI.open(args.artifact_root).resume_compare(
            args.resume_checkpoint,
            args.scratch_checkpoint,
            windows=_windows(args.windows),
            receipt_path=args.receipt,
            preflight=_preflight_arguments(args),
        )
    elif args.verb == "resident-stage":
        config = json.loads(args.config.read_text())
        result = ResidentRepairAPI.open(args.artifact_root).stage_two_spark_real(
            args.checkpoint,
            config=config,
            ready_path=args.ready,
            control_path=args.control,
        )
    elif args.verb == "v7-lut-only-update":
        identity = distributed_identity()
        config = json.loads(args.config.read_text())
        config.update(identity)
        result = ResidentRepairAPI.open(args.artifact_root).continue_v7_lut_only_update(
            args.start_checkpoint,
            trainable_luts=[name for name in args.trainable_luts.split(",") if name],
            lut_lr=args.lut_lr,
            config=config,
            receipt_path=args.receipt,
        )
    elif args.verb == "continue-training":
        identity = distributed_identity()
        config = json.loads(args.config.read_text())
        config.update(identity)
        result = ResidentRepairAPI.continue_training(
            args.artifact_root,
            args.start_checkpoint,
            [int(value) for value in args.milestones.split(",")],
            config=config,
            receipt_path=args.receipt,
        )
    elif args.verb == "diagnostic-perturb-validate":
        identity = distributed_identity()
        config = json.loads(args.config.read_text())
        config.update(identity)
        result = ResidentRepairAPI.open(args.artifact_root).diagnostic_perturb_and_validate(
            args.start_checkpoint,
            config=config,
            train_windows=_windows(args.train_windows),
            windows=_windows(args.windows),
            direction=args.direction,
            objective_composition=args.objective_composition,
            receipt_path=args.receipt,
        )
    else:
        result = ResidentRepairAPI.open(args.artifact_root).generate_candidates(
            args.checkpoint,
            builder_template=args.builder_template,
            ref_dir=args.ref_dir,
            corpus=args.corpus,
            meta_dir=args.meta_dir,
            python_executable=args.python_executable,
            mode=args.mode,
            remote=args.remote,
            local_dir=args.local_dir,
            windows=_windows(args.windows),
            chunk=args.chunk,
            mb=args.mb,
            preflight=_preflight_arguments(args),
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
