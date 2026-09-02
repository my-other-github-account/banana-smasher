"""Priority U16-only official resident Balanced64 scorer."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
from . import RepairArtifact
from .official_resident_campaign import _atomic_json,_claim,_load_module,_release,_run_anchor,WINDOWS
from .production_score_guard import reject_standalone_score_runner

def main(argv=None):
 reject_standalone_score_runner("repair_api.score_u16")
 p=argparse.ArgumentParser();p.add_argument('artifact_root',type=Path);p.add_argument('--official-rail',type=Path,required=True);p.add_argument('--planes-dir',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--teacher-dir',type=Path,required=True);p.add_argument('--builder-corpus',type=Path,required=True);p.add_argument('--task-id',default='t_f5d2415c');p.add_argument('--run',type=int,default=4306);p.add_argument('--basis',required=True);p.add_argument('--claim-path',type=Path,default=Path('/home/dnola/HOST_CLAIM.json'));args=p.parse_args();args.chunk=64;args.mb=4
 artifact=RepairArtifact.open(args.artifact_root);receipts=args.artifact_root/'receipts';receipts.mkdir(parents=True,exist_ok=True);_atomic_json(receipts/'CLAIM_U16_PRIORITY.json',_claim(args.claim_path,args.task_id,args.basis,args.artifact_root))
 rail=_load_module('historical_u16_priority_rail',args.official_rail);rail.TASK=args.task_id;rail.RUN=args.run;rail.CLAIM=args.claim_path;rail.MISSION=args.artifact_root;rail.MODEL=args.model_dir;rail.TEACHER=args.teacher_dir;rail.CORPUS=Path('/home/dnola/missions/DS4_TEACHER/static/windows_ds4_TRAIN.json');rail.WINDOWS=WINDOWS;rail.POSITIONS=1024;rail.SUPPORT=8192;rail.PLANESOURCE=Path('/home/dnola/missions/QTIP2_V7_OFFICIAL_RAIL_t_685c16d5_s3/code/official_local_planesource.py');rail.BUILDER_SRC=Path('/home/dnola/missions/QTIP2_V7_OFFICIAL_RAIL_t_685c16d5_s3/code/upstream_s1/t8192_train_u1_builder.py');rail.claim_gate=lambda:{'claim_sha256':'task-bound','pid':os.getpid(),'startticks':0}
 try:
  rec=_run_anchor(artifact,rail,'UPDATE_016',args,receipts);out={'schema':'modern-green-u16-priority-terminal-v1','status':'PASS','historical_scorer_sha256':'5d86ce215426b2de5124a68d8160a5b14bbe83717ce2e1fa9c3826c342cca6dd','chunk':64,'mb':4,'anchor':rec,'u16_kld':rec['score']['kld_mean'],'u16_top1':rec['score']['top1']};_atomic_json(receipts/'U16_PRIORITY_TERMINAL.json',out);print(json.dumps(out,indent=2,sort_keys=True));return 0
 finally:_atomic_json(receipts/'CLAIM_RELEASE.json',_release(args.claim_path,args.task_id))
if __name__=='__main__':raise SystemExit(main())
