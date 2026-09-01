"""Bounded exact U3 calibration using the canonical chunk-4/mb-1 rail."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

from . import RepairArtifact
from .official_resident_campaign import _atomic_json, _claim, _load_module, _release, _run_anchor, U3_TARGET, WINDOWS


def main(argv=None) -> int:
    p=argparse.ArgumentParser()
    p.add_argument("artifact_root",type=Path)
    p.add_argument("--official-rail",type=Path,required=True)
    p.add_argument("--planes-dir",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--teacher-dir",type=Path,required=True)
    p.add_argument("--builder-corpus",type=Path,required=True)
    p.add_argument("--task-id",default="t_f5d2415c")
    p.add_argument("--run",type=int,default=4305)
    p.add_argument("--basis",required=True)
    p.add_argument("--claim-path",type=Path,default=Path("/home/dnola/HOST_CLAIM.json"))
    args = p.parse_args(argv); args.chunk=64; args.mb=1
    artifact=RepairArtifact.open(args.artifact_root); receipts=args.artifact_root/"receipts";receipts.mkdir(parents=True,exist_ok=True)
    claim=_claim(args.claim_path,args.task_id,args.basis,args.artifact_root);_atomic_json(receipts/"CLAIM_U3_CALIBRATION.json",claim)
    rail=_load_module("historical_u3_calibration_rail",args.official_rail)
    rail.TASK=args.task_id;rail.RUN=args.run;rail.CLAIM=args.claim_path;rail.MISSION=args.artifact_root;rail.MODEL=args.model_dir;rail.TEACHER=args.teacher_dir;rail.CORPUS=Path("/home/dnola/missions/DS4_TEACHER/static/windows_ds4_TRAIN.json");rail.WINDOWS=WINDOWS;rail.POSITIONS=1024;rail.SUPPORT=8192;rail.PLANESOURCE=Path("/home/dnola/missions/QTIP2_V7_OFFICIAL_RAIL_t_685c16d5_s3/code/official_local_planesource.py");rail.BUILDER_SRC=Path("/home/dnola/missions/QTIP2_V7_OFFICIAL_RAIL_t_685c16d5_s3/code/upstream_s1/t8192_train_u1_builder.py");rail.claim_gate=lambda:{"claim_sha256":"task-bound","pid":os.getpid(),"startticks":0}
    status="FAILED"
    try:
        receipt=_run_anchor(artifact,rail,"UPDATE_003",args,receipts)
        kld=float(receipt["score"]["kld_mean"]); status="PASS" if abs(kld-U3_TARGET)<=1e-12 else "RED"
        out={"schema":"modern-green-u3-calibration-v1","status":status,"target":U3_TARGET,"measured":kld,"delta":kld-U3_TARGET,"anchor":receipt,"scorer_sha256":"5d86ce215426b2de5124a68d8160a5b14bbe83717ce2e1fa9c3826c342cca6dd","chunk":4,"mb":1}
        _atomic_json(receipts/"U3_CALIBRATION_TERMINAL.json",out);print(json.dumps(out,indent=2,sort_keys=True));return 0 if status=="PASS" else 1
    finally:
        _atomic_json(receipts/"CLAIM_RELEASE.json",_release(args.claim_path,args.task_id))

if __name__=="__main__":raise SystemExit(main())
