from __future__ import annotations
import argparse, json, os, random, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "helpers" else HERE
ENV_PY = ROOT / "environments" / "ltx25" / "Scripts" / "python.exe"
HELPER = ROOT / "helpers" / "ltx25_helper.py"
MODELS = ROOT / "models" / "ltx-2.5"

def _frames(n:int)->int:
    n=max(9,int(n))
    return max(9, n - ((n-1) % 8))

def main()->int:
    ap=argparse.ArgumentParser(description="FrameVision Planner LTX 2.5 launcher")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=-1)
    ns=ap.parse_args()
    if not ENV_PY.is_file(): raise SystemExit(f"LTX 2.5 environment not found: {ENV_PY}")
    if not HELPER.is_file(): raise SystemExit(f"LTX 2.5 helper not found: {HELPER}")
    seed=ns.seed if ns.seed >= 0 else random.randint(0,2147483647)
    image=str(Path(ns.image).resolve()) if ns.image else ""
    if image and not Path(image).is_file(): raise SystemExit(f"Start image not found: {image}")
    out=Path(ns.output).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    job={
      "cmd":"generate", "prompt":ns.prompt, "workflow":"two_phase", "audio_path":"", "seed":seed,
      "width":int(ns.width), "height":int(ns.height), "frames":_frames(ns.frames), "fps":float(ns.fps),
      "output":str(out), "images":([{"path":image,"frame_idx":0,"strength":1.0}] if image else []),
      "paths":{
        "transformer":str(MODELS/"diffusion_models"/"ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        "text_encoder":str(MODELS/"text_encoders"/"gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        "video_vae":str(MODELS/"vae"/"ltx-2.5-video-vae-bf16.safetensors"),
        "audio_vae":str(MODELS/"vae"/"ltx-2.5-audio-vae-bf16.safetensors"),
        "upsampler":str(MODELS/"latent_upscale_models"/"ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
      },
      "offload":"cpu", "quantization":"fp8-cast", "max_batch_size":1,
      "use_sage_attention":False, "use_int8_transformer":False, "int8_transformer_bundle":"",
      "use_int8_text_encoder":False, "int8_text_encoder_bundle":"", "enhance_prompt":False,
      "defer_trim":True, "cache_prompt_embeddings":True,
    }
    fd,tmp=tempfile.mkstemp(prefix="planner_ltx25_",suffix=".json",dir=str(out.parent)); os.close(fd)
    Path(tmp).write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding="utf-8")
    env=os.environ.copy(); env["PYTHONUTF8"]="1"; env["PYTHONIOENCODING"]="utf-8"
    try:
        rc=subprocess.call([str(ENV_PY),str(HELPER),"--queue-job",tmp],cwd=str(ROOT),env=env)
    finally:
        try: Path(tmp).unlink(missing_ok=True)
        except Exception: pass
    if rc!=0: return int(rc)
    return 0 if out.is_file() and out.stat().st_size>=1024 else 3
if __name__=="__main__": raise SystemExit(main())
