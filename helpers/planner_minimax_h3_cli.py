from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent if HERE.name.lower()=="helpers" else HERE
PYTHON=ROOT/"environments"/".minimax_h3_int4"/"python.exe"
MODEL_ROOT=ROOT/"models"/"minimax_h3"

def find_hybrid()->str:
    if not MODEL_ROOT.is_dir(): return ""
    hits=[p for p in MODEL_ROOT.rglob("*.safetensors") if "hybrid" in p.name.lower()]
    if not hits: return ""
    diffusion_root=MODEL_ROOT/"diffusion_models"
    def sort_key(path: Path):
        try:
            rel=path.resolve().relative_to(diffusion_root.resolve()); return (0,len(rel.parts),path.name.lower(),str(path).lower())
        except Exception:
            try:
                rel=path.resolve().relative_to(MODEL_ROOT.resolve()); return (1,len(rel.parts),path.name.lower(),str(path).lower())
            except Exception:
                return (2,len(path.parts),path.name.lower(),str(path).lower())
    hits.sort(key=sort_key)
    return str(hits[0].resolve())

def find_turbo_lora()->str:
    """Find the MiniMax H3 Turbo LoRA, preferring filenames that explicitly say 4-step."""
    lora_root=MODEL_ROOT/"loras"
    if not lora_root.is_dir(): return ""
    hits=[p for p in lora_root.rglob("*.safetensors") if "turbo" in p.name.lower()]
    if not hits: return ""
    def sort_key(path: Path):
        name=path.name.lower().replace("_","-")
        explicit_4step=("4step" in name or "4-step" in name or "4-steps" in name or "4steps" in name)
        try:
            rel=path.resolve().relative_to(lora_root.resolve())
            depth=len(rel.parts)
        except Exception:
            depth=len(path.parts)
        return (0 if explicit_4step else 1, depth, path.name.lower(), str(path).lower())
    hits.sort(key=sort_key)
    return str(hits[0].resolve())

def native_frames(n:int)->int:
    # H3 native temporal grid used by the GUI: 17k+5 (124, 141, ...).
    n=max(124,min(1433,int(n)))
    vals=list(range(124,1434,17))+[480]
    return min(vals,key=lambda x:abs(x-n))

def main()->int:
    ap=argparse.ArgumentParser(description="FrameVision Planner MiniMax H3 launcher")
    ap.add_argument("--prompt",required=True); ap.add_argument("--output",required=True)
    ap.add_argument("--width",type=int,required=True); ap.add_argument("--height",type=int,required=True)
    ap.add_argument("--frames",type=int,required=True); ap.add_argument("--seed",type=int,default=-1)
    ap.add_argument("--ref-image",action="append",default=[])
    ap.add_argument("--continue-video",default="")
    ap.add_argument("--lora",action="append",default=[])
    ap.add_argument("--lora-strength",action="append",type=float,default=[])
    ns=ap.parse_args()
    if len(ns.lora) != len(ns.lora_strength):
        raise SystemExit("Each --lora needs one matching --lora-strength")
    if len(ns.lora) > 2:
        raise SystemExit("Planner supports maximum 2 user LoRAs; the automatic Turbo LoRA occupies the third backend slot")
    _user_loras=[]
    _seen_loras=set()
    for _lp, _ls in zip(ns.lora, ns.lora_strength):
        _p=Path(_lp).resolve()
        if not _p.is_file():
            raise SystemExit(f"MiniMax H3 extra LoRA not found: {_p}")
        if float(_ls) == 0.0:
            continue
        _k=os.path.normcase(str(_p))
        if _k in _seen_loras:
            continue
        _seen_loras.add(_k)
        _user_loras.append((str(_p), max(-10.0,min(10.0,float(_ls)))))
    if not PYTHON.is_file(): raise SystemExit(f"MiniMax H3 environment not found: {PYTHON}")
    refs=[str(Path(p).resolve()) for p in ns.ref_image if p and Path(p).is_file()][:9]
    continue_video=str(Path(ns.continue_video).resolve()) if ns.continue_video and Path(ns.continue_video).is_file() else ""
    use_refs=bool(refs)
    # Native continuation is an FL2VA operation. The hybrid checkpoint is passed
    # to both model slots, so the same checkpoint can switch modes between shots.
    script=ROOT/"helpers"/("generate.py" if continue_video else ("generate_ref.py" if use_refs else "generate.py"))
    if not script.is_file(): raise SystemExit(f"MiniMax H3 generator not found: {script}")
    out=Path(ns.output).resolve(); out.parent.mkdir(parents=True,exist_ok=True)
    turbo_lora=find_turbo_lora()
    steps=4 if turbo_lora else 15
    _native_frames=native_frames(ns.frames)
    cmd=[str(PYTHON),str(script),"--width",str(ns.width),"--height",str(ns.height),"--frames",str(_native_frames),
         "--steps",str(steps),"--cfg","1.0","--shift","12","--audio-shift","3","--seed",str(ns.seed),
         "--sampler","euler","--scheduler","simple","--prompt",ns.prompt,"--output",str(out),
         "--vram-manager-auto","--video-vae-tile-size","256","--video-vae-tile-overlap","128"]
    if _native_frames > 719:
        cmd += ["--experimental-long-duration"]
    _active_lora_paths=set()
    if turbo_lora:
        cmd += ["--lora",turbo_lora,"--lora-strength","1.0"]
        _active_lora_paths.add(os.path.normcase(str(Path(turbo_lora).resolve())))
        print(f"[minimax-planner] 4-step Turbo LoRA default: {turbo_lora}",flush=True)
        print("[minimax-planner] sampling steps: 4",flush=True)
    else:
        print(f"[minimax-planner] no Turbo LoRA found under {MODEL_ROOT / 'loras'}; sampling steps: 15",flush=True)
    _added_extra=0
    for _lp, _ls in _user_loras:
        _key=os.path.normcase(str(Path(_lp).resolve()))
        if _key in _active_lora_paths:
            print(f"[minimax-planner] extra LoRA skipped because it is already active: {_lp}",flush=True)
            continue
        if len(_active_lora_paths) >= 3:
            break
        cmd += ["--lora",_lp,"--lora-strength",str(_ls)]
        _active_lora_paths.add(_key)
        _added_extra += 1
        print(f"[minimax-planner] extra LoRA {_added_extra}: {_lp} | strength={_ls:g}",flush=True)
    if continue_video:
        cmd += ["--continue-video",continue_video,"--continue-context-frames","39"]
        print(f"[minimax-planner] native continuation source: {continue_video}",flush=True)
    elif use_refs:
        cmd += ["--ref-image-size","match"]
        for p in refs: cmd += ["--ref-image",p]
    hybrid=find_hybrid()
    if hybrid:
        # Same preference as the GUI: one hybrid checkpoint supplies either generation mode.
        cmd += ["--fl2va-checkpoint",hybrid,"--ref2va-checkpoint",hybrid]
        print(f"[minimax-planner] hybrid default: {hybrid}",flush=True)
    elif use_refs:
        print("[minimax-planner] no hybrid found; Ref2VA default",flush=True)
    else:
        print("[minimax-planner] no hybrid and no refs; FL2VA text-to-video default",flush=True)
    env=os.environ.copy(); env["PYTHONUTF8"]="1"; env["PYTHONIOENCODING"]="utf-8"
    rc=subprocess.call(cmd,cwd=str(ROOT),env=env)
    if rc not in (0,3): return int(rc)
    return 0 if out.is_file() and out.stat().st_size>=1024 else 3
if __name__=="__main__": raise SystemExit(main())
