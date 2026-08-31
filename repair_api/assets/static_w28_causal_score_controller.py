#!/usr/bin/env python3
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

TASK=""; RUN_ID=0
PIN=""
BASIS="98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT_SHA=""; PARENT_SHA=""; CHECKPOINT=""; COMPARATOR=""; ATTEMPT=0; SLUG=""
CALIBRATION_SHA="75b3f4d0bceb072fe42c53377f0a758d8d459f58d73c2005e6592cf5cb2b01d1"
ROOT=Path("/home/dnola/missions/CANDD_RELBOUND_t_e88f2c1a/run7028")
CLAIM=Path("/home/dnola/HOST_CLAIM.json"); DRIVER=Path("/home/dnola/missions/DRIVER_GOALS.md")
LIMIT=1200.0
ACTIVE=None

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def ticks(pid:int)->int: return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
def atomic(path:Path,value:dict)->str:
 data=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode(); tmp=path.with_name(path.name+f".tmp.{os.getpid()}")
 with tmp.open("wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path); d=os.open(path.parent,os.O_RDONLY)
 try: os.fsync(d)
 finally: os.close(d)
 return hashlib.sha256(data).hexdigest()
def payloads()->list[dict]:
 out=[]; needle=str(ROOT)
 for p in Path("/proc").iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid(): continue
  try: vals=[x.decode(errors="replace") for x in (p/"cmdline").read_bytes().split(b"\0") if x]
  except OSError: continue
  if needle in vals and ("repair_api" in vals or "score" in vals): out.append({"pid":int(p.name),"argv":vals})
 return out
def score_argv(rank:int,receipt:Path)->list[str]:
 return ["/home/dnola/humming_env/bin/python","-u","-m","torch.distributed.run","--nnodes","2","--nproc-per-node","1",
  "--node-rank",str(rank),"--master-addr","192.168.200.7","--master-port","30638","-m","repair_api","score",
  "--artifact-root",str(ROOT),"--checkpoint",CHECKPOINT,"--windows","28","--receipt",str(receipt)]
def target_identity()->dict:
 return {"task_id":TASK,"board_run_id":RUN_ID,"canonical_git_pin":PIN,"basis_sha256":BASIS,
  "checkpoint":CHECKPOINT,"checkpoint_sha256":CHECKPOINT_SHA,"parent_sha256":PARENT_SHA,"attempt":ATTEMPT,"slug":SLUG}
def accepted(rank:int)->bool:
 p=ROOT/"receipts"/f"SCORE.candidate_d.{SLUG}.rank{rank}.json"
 if not p.exists(): return False
 try: x=json.loads(p.read_text()); c=x["runtime_counters"]; ident=x["identity"]
 except (KeyError,TypeError,ValueError,json.JSONDecodeError): return False
 return (x.get("status")=="PASS" and x.get("checkpoint")==CHECKPOINT and x.get("windows")==[28]
  and int(x.get("positions",-1))==1024 and float(x.get("kld_mean",99.0)) < 0.19
  and ident.get("checkpoint_sha256")==CHECKPOINT_SHA
  and ident.get("production_admission",{}).get("scientific_question",{}).get("matched_parent_sha256")==PARENT_SHA
  and int(c.get("fallback_calls",-1))==0 and int(c.get("file_reads_during_timed_score",-1))==0)
def release(rank:int,host:str,status:str,child:int|None,rc:int|None,started:float)->None:
 now=time.time(); rp=ROOT/"receipts"/f"U{int(CHECKPOINT[7:])}_W28_SCORE_CONTROLLER_TERMINAL.attempt{ATTEMPT}.rank{rank}.json"
 row={"schema":"banana-smasher-static-w28-causal-score-controller-terminal-v1","status":status,"task_id":TASK,"board_run_id":RUN_ID,
  "host":host,"rank":rank,"canonical_git_pin":PIN,"basis_sha256":BASIS,"checkpoint_sha256":CHECKPOINT_SHA,
  "started_unix":started,"created_unix":now,"elapsed_seconds":now-started,"child_pid":child,"child_returncode":rc,
  "scientific_acceptance":accepted(rank),"threshold":0.19,"windows":[28]}
 rsha=atomic(rp,row)
 for path in (ROOT/"SHARDS.json",CLAIM):
  lock=path.with_name(path.name+".lock")
  with lock.open("a+") as lf:
   fcntl.flock(lf,fcntl.LOCK_EX); cur=json.loads(path.read_text())
   if cur.get("task_id")!=TASK: raise RuntimeError(f"release owner drift {path}")
   cur.update({"status":"RELEASED","state":"RELEASED","heartbeat_unix":now,"expires_unix":now,"lease_until_unix":now,
    "holder_pid":None,"holder_startticks":None,"controller_pid":None,"controller_startticks":None,"workload_pid":None,
    "workload_startticks":None,"release_receipt":str(rp),"release_receipt_sha256":rsha,"released_unix":now})
   atomic(path,cur)
def main()->int:
 global ACTIVE,TASK,RUN_ID,PIN,CHECKPOINT_SHA,PARENT_SHA,CHECKPOINT,COMPARATOR,ATTEMPT,SLUG
 ap=argparse.ArgumentParser()
 ap.add_argument("--rank",type=int,choices=(0,1),required=True)
 ap.add_argument("--task",required=True); ap.add_argument("--run-id",type=int,required=True)
 ap.add_argument("--git-pin",required=True); ap.add_argument("--checkpoint-update",type=int,required=True)
 ap.add_argument("--checkpoint-sha",required=True); ap.add_argument("--parent-sha",required=True)
 ap.add_argument("--attempt",type=int,required=True); ap.add_argument("--comparator-update",type=int,required=True)
 ap.add_argument("--comparator-terminal",type=Path,required=True); ap.add_argument("--comparator-terminal-sha",required=True)
 ap.add_argument("--comparator-kld",type=float,required=True); ap.add_argument("--comparator-top1",type=int,required=True)
 args=ap.parse_args(); rank=args.rank
 TASK=args.task; RUN_ID=args.run_id; PIN=args.git_pin; CHECKPOINT_SHA=args.checkpoint_sha; PARENT_SHA=args.parent_sha
 CHECKPOINT=f"UPDATE_{args.checkpoint_update:03d}"; COMPARATOR=f"UPDATE_{args.comparator_update:03d}"; ATTEMPT=args.attempt
 SLUG=f"u{args.checkpoint_update}w28.attempt{ATTEMPT}"
 host=os.uname().nodename; started=time.time(); pid=os.getpid(); st=ticks(pid)
 expected="spark-5-work" if rank==0 else "spark-7"
 if expected not in host and not(rank==0 and host=="spark-work"): raise RuntimeError("host/rank mismatch")
 if f"HOST_ALLOCATION {TASK} seat_order [spark-5-work rank0, spark-7 rank1] NOW" not in DRIVER.read_text(): raise RuntimeError("allocation missing")
 artp=ROOT/"ARTIFACT.json"; art=json.loads(artp.read_text()); cfg=art["score"]["official_k2_resident"]
 index=Path(cfg["model_root"])/"model.safetensors.index.json"
 if sha(index)!=BASIS or cfg.get("basis_sha256")!=BASIS: raise RuntimeError("basis mismatch before claim")
 if 105*(1<<30) > int(next(x.split()[1] for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")))*1024-4*(1<<30): raise RuntimeError("memory preflight refused")
 if os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize < 1*(1<<30): raise RuntimeError("storage preflight refused")
 code=ROOT/"code"
 if subprocess.check_output(["git","rev-parse","HEAD"],cwd=code,text=True).strip()!=PIN or subprocess.check_output(["git","status","--porcelain"],cwd=code): raise RuntimeError("code pin/clean mismatch")
 if payloads(): raise RuntimeError(f"duplicate payload refused: {payloads()}")
 now=time.time()
 for path in (CLAIM,ROOT/"SHARDS.json"):
  lock=path.with_name(path.name+".lock")
  with lock.open("a+") as lf:
   fcntl.flock(lf,fcntl.LOCK_EX); pre=path.read_bytes(); cur=json.loads(pre)
   if cur.get("status")!="RELEASED" or cur.get("state")!="RELEASED" or any(cur.get(k) is not None for k in ("holder_pid","controller_pid","workload_pid")): raise RuntimeError(f"not released {path}")
   cur.update({"task_id":TASK,"owner_task_id":TASK,"owner_profile":"bs11","board_run_id":RUN_ID,"canonical_code_commit":PIN,
    "canonical_git_pin":PIN,"basis_sha256":BASIS,"intended_basis":BASIS,"status":"CLAIMED","state":"CLAIMED",
    "phase":f"CANDIDATE_D_U{args.checkpoint_update}_W28_STATIC_SCORE","claimed_unix":now,"heartbeat_unix":now,"expires_unix":now+1500,
    "lease_until_unix":now+1200,"holder_pid":pid,"holder_startticks":st,"controller_pid":pid,"controller_startticks":st,
    "workload_pid":None,"workload_startticks":None,"rank":rank,"host":host,"previous_sha256":hashlib.sha256(pre).hexdigest()})
   atomic(path,cur)
 ACTIVE={"rank":rank,"host":host,"started":started,"child":None}
 question=ROOT/"receipts"/f"CANDIDATE_D_U{args.checkpoint_update}_W28_SCORE_BINDING.attempt{ATTEMPT}.json"; calibration=ROOT/"receipts"/"CANONICAL_RESIDENT_CALIBRATION.json"
 expected_question={"changed_variable":f"candidate {CHECKPOINT} versus sealed catastrophic {COMPARATOR}; source teacher window basis and checkpoint frozen",
  "checkpoint_update":args.checkpoint_update,"dose":args.checkpoint_update,"matched_parent_sha256":PARENT_SHA,
  "ordered_windows_sha256":"66e7ff50b3d423ac1608a1c537613e5b87b1f516283b7bb49c91cd6af1bb412f",
  "schema":"official-k2-resident-scientific-question-v1","status":"PRE_REGISTERED","task_id":TASK,"threshold":0.19}
 question_sha=hashlib.sha256((json.dumps(expected_question,sort_keys=True,indent=2)+"\n").encode()).hexdigest()
 if question.exists():
  if sha(question)!=question_sha: raise RuntimeError(f"immutable {CHECKPOINT} question receipt drift")
 else:
  if atomic(question,expected_question)!=question_sha: raise RuntimeError(f"{CHECKPOINT} question receipt construction drift")
 instrument=json.loads((ROOT/"receipts"/"INSTRUMENT.U64.W28.json").read_text())
 comparator_key=f"u{args.comparator_update}"
 if sha(args.comparator_terminal)!=args.comparator_terminal_sha: raise RuntimeError(f"sealed {COMPARATOR} terminal SHA drift")
 comparator=json.loads(args.comparator_terminal.read_text())
 if float(comparator.get(comparator_key,{}).get("kld_mean",-1))!=args.comparator_kld or int(comparator.get(comparator_key,{}).get("top1",-1))!=args.comparator_top1: raise RuntimeError(f"sealed {COMPARATOR} numeric drift")
 if instrument.get("status")!="PASS" or instrument.get("known_value_fixture")!={"window":28,"kld_mean":0.1364830042977786,"top1":880}: raise RuntimeError("sealed W28 known-value instrument drift")
 if sha(question)!=question_sha or sha(calibration)!=CALIBRATION_SHA: raise RuntimeError("score admission receipt SHA drift")
 meta=art["checkpoints"][CHECKPOINT]
 if meta.get("sha256")!=CHECKPOINT_SHA or meta.get("parent_sha256")!=PARENT_SHA or sha(ROOT/meta["path"])!=CHECKPOINT_SHA: raise RuntimeError(f"{CHECKPOINT} envelope drift")
 cfg.update({"rank":rank,"scientific_question_receipt":str(question),
  "pre_calibration_receipt":str(calibration),"master_port":30638,"master_addr":"192.168.200.7"})
 cfg["recipe_id"]="published_pre_lower_lr_warmup16_cosine64_v1"
 cfg["resident_validation_proof"]=True
 cfg["provider_resolution_mode"]="STATIC_W28_GROUPED"
 cfg["resident_validation_expert_implementation"]="accepted_static_w28"
 cfg["score_window_batch_size"]=2
 cfg["sealed_builder_window_microbatch"]=2
 cfg["trainer_source"]=str(ROOT/"code/repair_api/assets/static_w28_modern_green_clean_u0.py")
 cfg["trainer_source_sha256"]="cc0520e00a6cc5b979c638e3f1fd98ae92c882f3cf9f48cbcdf3fa55fad343cc"
 cfg["qsfp_host_ip_by_rank"]={"0":"192.168.200.7","1":"192.168.200.8"}
 cfg["fast_k2_wrapper_source"]=str(ROOT/"code/repair_api/assets/static_w28_fast_k2_grouped.py")
 cfg["fast_k2_wrapper_source_sha256"]="ec681dd1ac35d5c4368071db12c8bb0801cbf78c3677c51ef9a56d0cacdf3454"
 cfg["resident_expert_source"]=str(ROOT/"code/repair_api/assets/static_w28_fast_v7_expert_base.py")
 cfg["resident_expert_source_sha256"]="4ba1411601b186dd0d6a3a89c829320f1b50e3112a40db40034e9fbadfb5d552"
 if rank==0: cfg["model_root"]="/home/dnola/missions/RANK0_WHOLE_t_b26eb32b_swork"
 if rank==0: cfg["parent_root"]="/home/dnola/missions/V7_CODEBOOK_FULLPARENT_t_0c44dcc6_s5w"
 if rank==0:
  cfg["asset_root"]="/home/dnola/missions/STAGE_U20_t_3a6f22a5_spark-5-work/inputs/attempt4b/asset_view"
  cfg["binrepair_delta_dir"]="/home/dnola/missions/STAGE_U20_t_3a6f22a5_spark-5-work/inputs/attempt4b/delta"
  cfg["binrepair_manifest"]=cfg["asset_root"]+"/code/DUALVQ_K4096MENU_IQ3_BIN_MANIFEST.json"
  cfg["binrepair_vq3b_dir"]="/home/dnola/missions/BINREPAIR_t_2956f863/planes"
  cfg["fast_k2_extension"]="/home/dnola/missions/STAGE_U20_t_3a6f22a5_spark-5-work/inputs/banana_fast_k2_grouped_0c3cc723fe66.so"
  cfg["official_expert_source"]="/home/dnola/missions/STAGE_U20_t_3a6f22a5_spark-5-work/repo-r19/ds4-flash-kldmatrix/repair_api/assets/fast_v7_expert_base.py"

  cfg["lp4_pack_source"]=str(ROOT/"code/runtime/v7/vendor/src_lp4/lp4_pack.py")
  cfg["lp4_pack_source_sha256"]="7a8e48547824a87a48db4c7142ec53f73303a91ce6a0c95cf1a88b1b87d22350"
  cfg["lp4_train_source"]=str(ROOT/"code/runtime/v7/vendor/src_lp4/lp4_train.py")
  cfg["lp4_train_source_sha256"]="10abc4b04a9bc88bf348cd121d3d072456a54de1cd801a1425edc15b104e4523"
 shards=json.loads((ROOT/"SHARDS.json").read_text())
 if shards.get("intended_basis")!=BASIS: raise RuntimeError("claimed shard basis mismatch")
 atomic(artp,art); now=time.time()
 receipt=ROOT/"receipts"/f"SCORE.candidate_d.{SLUG}.rank{rank}.json"
 if receipt.exists(): raise RuntimeError(f"immutable score receipt exists: {receipt}")
 argv=score_argv(rank,receipt)
 env=os.environ.copy(); env["PYTHONPATH"]=f"{code}/banana-smasher/src:{code}"; env["NCCL_SOCKET_IFNAME"]="enp1s0f1np1"; env["GLOO_SOCKET_IFNAME"]="enp1s0f1np1"
 logp=ROOT/"logs"/f"SCORE.candidate_d.{SLUG}.rank{rank}.log"
 with logp.open("ab",buffering=0) as log: child=subprocess.Popen(argv,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ct=ticks(child.pid)
 ACTIVE["child"]=child.pid
 for path in (CLAIM,ROOT/"SHARDS.json"):
  lock=path.with_name(path.name+".lock")
  with lock.open("a+") as lf: fcntl.flock(lf,fcntl.LOCK_EX); cur=json.loads(path.read_text()); cur.update({"workload_pid":child.pid,"workload_startticks":ct}); atomic(path,cur)
 atomic(ROOT/"receipts"/f"LAUNCH.score.{SLUG}.rank{rank}.json",{"schema":"banana-smasher-static-w28-causal-score-launch-v1","task_id":TASK,"board_run_id":RUN_ID,
  "rank":rank,"host":host,"controller_pid":pid,"controller_startticks":st,"launcher_pid":child.pid,"launcher_startticks":ct,
  "canonical_git_pin":PIN,"basis_sha256":BASIS,"checkpoint_sha256":CHECKPOINT_SHA,"launcher_sha256":sha(Path(__file__)),
  "argv":argv,"log":str(logp),"absolute_limit_seconds":LIMIT,"launched_unix":time.time()})
 reason=None
 while child.poll() is None:
  now=time.time()
  if now-started>LIMIT: reason="MECHANICAL_RED_W28_900S_LIMIT"; os.killpg(child.pid,signal.SIGTERM); break
  if int(now)%30==0:
   for path in (CLAIM,ROOT/"SHARDS.json"):
    lock=path.with_name(path.name+".lock")
    with lock.open("a+") as lf: fcntl.flock(lf,fcntl.LOCK_EX); cur=json.loads(path.read_text()); cur.update({"heartbeat_unix":now,"expires_unix":now+1200,"lease_until_unix":now+1200}); atomic(path,cur)
  time.sleep(1)
 if reason:
  try: child.wait(timeout=30)
  except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait(timeout=30)
 rc=child.wait(); reason=reason or ("GREEN_W28_ACCEPTANCE" if rc==0 and accepted(rank) else "SCIENTIFIC_RED_OR_PROCESS_EXIT")
 release(rank,host,reason,child.pid,rc,started); ACTIVE=None; return 0 if reason=="GREEN_W28_ACCEPTANCE" else 2
if __name__=="__main__":
 try: raise SystemExit(main())
 except Exception as e:
  print(f"FATAL {type(e).__name__}: {e}",file=sys.stderr,flush=True)
  if ACTIVE is not None:
   child=ACTIVE.get("child")
   if child is not None and Path(f"/proc/{child}").exists():
    try: os.killpg(child,signal.SIGTERM)
    except ProcessLookupError: pass
    for _ in range(300):
     if not Path(f"/proc/{child}").exists(): break
     time.sleep(0.1)
    if Path(f"/proc/{child}").exists():
     try: os.killpg(child,signal.SIGKILL)
     except ProcessLookupError: pass
     for _ in range(300):
      if not Path(f"/proc/{child}").exists(): break
      time.sleep(0.1)
    if Path(f"/proc/{child}").exists(): raise RuntimeError(f"fatal cleanup failed; child still alive {child}") from e
   release(ACTIVE["rank"],ACTIVE["host"],"CONTROLLER_FATAL_EXACT_RELEASE",child,None,ACTIVE["started"])
   ACTIVE=None
  raise
