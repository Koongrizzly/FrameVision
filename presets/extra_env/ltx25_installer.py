from __future__ import annotations
import argparse, concurrent.futures, json, os, shutil, subprocess, sys, time, urllib.error, urllib.request, zipfile
from pathlib import Path

class Tee:
    def __init__(self, *streams): self.streams=streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()


W_REPO = "Winnougan/ltx-2.5-w4a8-convrot-int4-convrot-Winnougan-Blessing"
O_REPO = "Lightricks/LTX-2.5"
LTX_GITHUB_ZIP = "https://github.com/Lightricks/LTX-2/archive/refs/heads/main.zip"
CONNECTIONS = max(1, min(16, int(os.environ.get("LTX_DOWNLOAD_CONNECTIONS", "8"))))

COMMON = [
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
]
FP16 = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
] + COMMON
W4A8 = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-w4a8_convrot.safetensors",
]
INT4 = [
    "diffusion_models/ltx-2.5-22b-distilled-transformer-int4_convrot.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-int4_convrot.safetensors",
]


def token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(key): return os.environ[key].strip()
    p = Path.home()/".cache"/"huggingface"/"token"
    try:
        t=p.read_text(encoding="utf-8").strip()
        return t or None
    except Exception: return None


def headers(auth=True):
    h={"User-Agent":"FrameVision-LTX25-Installer/1.0"}
    t=token() if auth else None
    if t: h["Authorization"] = f"Bearer {t}"
    return h


def hf_url(repo, rel):
    return f"https://huggingface.co/{repo}/resolve/main/{rel}?download=true"


def remote_size(url, auth=True):
    req=urllib.request.Request(url, headers=headers(auth), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n=r.headers.get("Content-Length")
            return int(n) if n else None, r.geturl()
    except Exception:
        req=urllib.request.Request(url, headers={**headers(auth),"Range":"bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr=r.headers.get("Content-Range","")
            if "/" in cr: return int(cr.rsplit("/",1)[1]), r.geturl()
            n=r.headers.get("Content-Length")
            return int(n) if n else None, r.geturl()


def _part(url, dest, start, end, auth):
    dest.parent.mkdir(parents=True, exist_ok=True)
    have=dest.stat().st_size if dest.exists() else 0
    pos=start+have
    if pos>end: return
    req=urllib.request.Request(url, headers={**headers(auth),"Range":f"bytes={pos}-{end}"})
    with urllib.request.urlopen(req, timeout=180) as r, open(dest,"ab") as f:
        while True:
            b=r.read(8*1024*1024)
            if not b: break
            f.write(b)


def download(url, dest:Path, auth=True, connections=CONNECTIONS):
    dest.parent.mkdir(parents=True, exist_ok=True)
    size, final_url = remote_size(url, auth)
    if dest.exists() and size and dest.stat().st_size==size:
        print(f"[SKIP] {dest.name} already complete")
        return
    if size and size > 32*1024*1024 and connections>1:
        chunk=(size+connections-1)//connections
        parts=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=connections) as ex:
            futs=[]
            for i in range(connections):
                s=i*chunk; e=min(size-1,(i+1)*chunk-1)
                if s>e: break
                p=dest.with_name(dest.name+f".part{i:02d}"); parts.append(p)
                futs.append(ex.submit(_part, final_url, p, s, e, auth))
            for f in concurrent.futures.as_completed(futs): f.result()
        with open(dest,"wb") as out:
            for p in parts:
                with open(p,"rb") as src: shutil.copyfileobj(src,out,16*1024*1024)
                p.unlink(missing_ok=True)
        if dest.stat().st_size!=size:
            raise RuntimeError(f"Size mismatch for {dest}: {dest.stat().st_size} != {size}")
    else:
        part=dest.with_name(dest.name+".part")
        have=part.stat().st_size if part.exists() else 0
        h=headers(auth)
        if have: h["Range"]=f"bytes={have}-"
        req=urllib.request.Request(final_url,headers=h)
        with urllib.request.urlopen(req,timeout=180) as r, open(part,"ab") as f:
            while True:
                b=r.read(8*1024*1024)
                if not b: break
                f.write(b)
        part.replace(dest)
    print(f"[OK] {dest}")


def safe_rel(rel):
    low=rel.lower()
    if "dev-transformer" in low or "-dev-" in low:
        raise RuntimeError(f"DEV model blocked by installer: {rel}")
    return rel


def ensure_repo(model_root:Path):
    repo_dir=model_root/"LTX-2"
    if (repo_dir/"pyproject.toml").exists():
        print(f"[SKIP] Repo already present: {repo_dir}")
        return repo_dir
    tmp=model_root/"_ltx2_repo.zip"
    stage=model_root/"_repo_extract"
    shutil.rmtree(stage,ignore_errors=True); stage.mkdir(parents=True,exist_ok=True)
    print("[REPO] Downloading LTX-2 repo into model folder...")
    download(LTX_GITHUB_ZIP,tmp,auth=False,connections=4)
    with zipfile.ZipFile(tmp) as z: z.extractall(stage)
    candidates=[p for p in stage.iterdir() if p.is_dir()]
    if not candidates: raise RuntimeError("LTX-2 repo archive was empty")
    if repo_dir.exists(): shutil.rmtree(repo_dir)
    shutil.move(str(candidates[0]),str(repo_dir))
    shutil.rmtree(stage,ignore_errors=True); tmp.unlink(missing_ok=True)
    print(f"[OK] Repo: {repo_dir}")
    return repo_dir


def run(cmd, env=None):
    print("[RUN]", " ".join(map(str,cmd)))
    subprocess.run([str(x) for x in cmd],check=True,env=env)


def install_fp16(root:Path, uv:Path):
    model=root/"models"/"ltx_2_5"; envdir=root/"environments"/"ltx25_fp16"
    model.mkdir(parents=True,exist_ok=True)
    repo=ensure_repo(model)
    print("[MODEL] Installing FULL distilled BF16/FP16 path. DEV is blocked.")
    for rel in FP16:
        rel=safe_rel(rel); download(hf_url(O_REPO,rel),model/rel,auth=True)
    e=os.environ.copy(); e["UV_PROJECT_ENVIRONMENT"]=str(envdir)
    run([uv,"sync","--project",repo],env=e)


def install_convrot(root:Path, uv:Path, mode:str):
    model=root/"models"/"ltx_2_5_convrot"; envdir=root/"environments"/"ltx25_convrot"
    model.mkdir(parents=True,exist_ok=True)
    ensure_repo(model)
    files=W4A8 if mode=="w4a8" else INT4
    print(f"[MODEL] Installing {mode.upper()} distilled ConvRot. DEV is blocked.")

    # Verify the quant files before any large downloads. The upstream README
    # may list files that are not actually present in the repository yet.
    checked=[]
    missing=[]
    for rel in files:
        rel=safe_rel(rel)
        url=hf_url(W_REPO,rel)
        try:
            remote_size(url,auth=False)
            checked.append((rel,url))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.append(rel)
            else:
                raise

    if missing:
        print("")
        print(f"[UNAVAILABLE] {mode.upper()} ConvRot is not currently fully published upstream.")
        print(f"Repository: https://huggingface.co/{W_REPO}")
        print("Missing required file(s):")
        for rel in missing:
            print(f"  - {rel}")
        print("")
        print("Nothing was downloaded. Try again later after the missing files are uploaded.")
        return False

    for rel,url in checked:
        download(url,model/rel,auth=False)

    for rel in COMMON:
        rel=safe_rel(rel); download(hf_url(O_REPO,rel),model/rel,auth=True)
    if not (envdir/"Scripts"/"python.exe").exists():
        run([uv,"venv","--python","3.12",envdir])
    py=envdir/"Scripts"/"python.exe"
    # Separate ConvRot environment. Shared /vendor code is never copied or modified.
    run([uv,"pip","install","--python",py,"torch","torchvision","torchaudio","--index-url","https://download.pytorch.org/whl/cu130"])
    run([uv,"pip","install","--python",py,"comfy-kitchen>=0.2.24","safetensors","transformers","sentencepiece","protobuf","numpy","pillow","einops","psutil","pyyaml","av","soundfile","scipy","tqdm"])


def remove_install(root:Path):
    print("\nRemove which installation?")
    print("  1. Full FP16/BF16")
    print("  2. ConvRot (W4A8/INT4)")
    print("  3. Cancel")
    c=input("Choose [1-3]: ").strip()
    if c=="3": return
    if c=="1": targets=[root/"models"/"ltx_2_5", root/"environments"/"ltx25_fp16"]
    elif c=="2": targets=[root/"models"/"ltx_2_5_convrot", root/"environments"/"ltx25_convrot"]
    else: raise RuntimeError("Invalid removal choice")
    print("\nWill remove ONLY:")
    for p in targets: print(" ",p)
    print("Protected: /vendor, /helpers, /presets, /logs and unrelated models/environments")
    if input("Type REMOVE to continue: ").strip()!="REMOVE":
        print("Cancelled."); return
    for p in targets:
        if p.exists(): shutil.rmtree(p); print(f"[REMOVED] {p}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--action",required=True); ap.add_argument("--root",required=True); ap.add_argument("--uv",required=True)
    a=ap.parse_args(); root=Path(a.root).resolve(); uv=Path(a.uv).resolve()
    logdir=root/"logs"; logdir.mkdir(parents=True,exist_ok=True)
    logf=open(logdir/"ltx25_install.log","a",encoding="utf-8",buffering=1)
    sys.stdout=Tee(sys.__stdout__,logf); sys.stderr=Tee(sys.__stderr__,logf)
    print("\n"+"="*64)
    print(time.strftime("LTX 2.5 installer run - %Y-%m-%d %H:%M:%S"))
    print(f"FrameVision root: {root}")
    print(f"Parallel download connections: {CONNECTIONS}")
    if a.action=="install-w4a8": install_convrot(root,uv,"w4a8")
    elif a.action=="install-int4": install_convrot(root,uv,"int4")
    elif a.action=="install-fp16": install_fp16(root,uv)
    elif a.action=="remove": remove_install(root)
    else: raise RuntimeError(f"Unknown action: {a.action}")

if __name__=="__main__":
    try: main()
    except urllib.error.HTTPError as e:
        if e.code in (401,403):
            print("\n[ERROR] Hugging Face denied access. For the official LTX-2.5 files, accept the model license and make sure HF_TOKEN or your Hugging Face cached token is available.")
        raise
