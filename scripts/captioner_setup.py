
import os, sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT/".venv"/"Scripts"/"python.exe"
STATUS = ROOT/"models"/"captioner_status.json"
HF_CACHE = ROOT/".hf_cache"
HF_CACHE.mkdir(exist_ok=True)

def pip(*args):
    cmd = [str(VENV_PY), "-m", "pip"] + list(args)
    return subprocess.call(cmd)

def write_status(**kw):
    data = {"ready": False, "backend": None, "device": None, "notes": []}
    data.update(kw)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(data, indent=2), encoding="utf-8")

def main():
    backend = "BLIP"
    device = "cpu"
    for a in sys.argv[1:]:
        if a.startswith("--backend="): backend = a.split("=",1)[1]
        if a.startswith("--device="): device = a.split("=",1)[1]

    notes = []

    pip("install","-U","pip")
    pip("install","-U","transformers>=4.43","huggingface-hub>=0.24","accelerate>=0.33","pillow","einops")

    if device == "cuda":
        rc = pip("install","--no-cache-dir","--force-reinstall",
                 "torch==2.3.1+cu121","--index-url","https://download.pytorch.org/whl/cu121")
        if rc != 0:
            write_status(ready=False, backend=backend, device="cpu",
                         notes=["CUDA torch install failed; staying on CPU"])
            return

    if backend.lower().startswith("florence"):
        rc = pip("install","-U","flash-attn")
        if rc != 0:
            notes.append("flash-attn missing (common on Windows); Florence unavailable.")

    write_status(ready=True, backend=backend, device=device, notes=notes)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        write_status(ready=False, backend=None, device=None, notes=[f"setup error: {e!s}"])
        sys.exit(1)
