
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]

MODEL_ROOT = ROOT / "models" / "ltx_2_5_convrot"
COMFY_DIR = MODEL_ROOT / "ComfyUI"
ENV_DIR = ROOT / "environments" / "ltx25_convrot"
ENV_PY = ENV_DIR / "Scripts" / "python.exe"
UV_EXE = ROOT / "presets" / "bin" / "uv" / "uv.exe"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "ltx25_comfy_backend_install.log"

COMFY_ZIP_URL = "https://github.com/Comfy-Org/ComfyUI/archive/refs/heads/master.zip"

EXCLUDE_EXACT = {
    "torch",
    "torchvision",
    "torchaudio",
    "comfyui_frontend_package",
    "comfyui_workflow_templates",
    "comfyui_embedded_docs",
}

def log(message: str = "") -> None:
    print(message, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(message + "\n")

def fail(message: str, code: int = 1) -> None:
    log(f"[ERROR] {message}")
    raise SystemExit(code)

def requirement_name(line: str) -> str:
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("-"):
        return ""
    left = s.split(";", 1)[0].strip()
    for sep in (" @ ", "==", ">=", "<=", "~=", "!=", ">", "<"):
        if sep in left:
            left = left.split(sep, 1)[0].strip()
            break
    if "[" in left:
        left = left.split("[", 1)[0].strip()
    return left.lower().replace("-", "_")

def download(url: str) -> bytes:
    log(f"[DOWNLOAD] {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FrameVision-LTX25-Comfy-Backend-Installer/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        total = response.headers.get("Content-Length")
        total_n = int(total) if total and total.isdigit() else None
        chunks = []
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            done += len(chunk)
            if total_n:
                pct = done * 100.0 / total_n
                print(
                    f"\r[DOWNLOAD] {done/1048576:.1f}/{total_n/1048576:.1f} MiB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
        if total_n:
            print("", flush=True)
        return b"".join(chunks)

def install_comfy_source() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    payload = download(COMFY_ZIP_URL)

    with tempfile.TemporaryDirectory(prefix="ltx25_comfy_", dir=str(MODEL_ROOT)) as td:
        temp = Path(td)
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(temp)

        roots = [p for p in temp.iterdir() if p.is_dir()]
        if len(roots) != 1:
            fail("Unexpected ComfyUI archive layout.")
        extracted = roots[0]

        staging = MODEL_ROOT / "ComfyUI.new"
        old = MODEL_ROOT / "ComfyUI.old"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(extracted, staging)

        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
        if COMFY_DIR.exists():
            COMFY_DIR.replace(old)
        staging.replace(COMFY_DIR)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)

    log(f"[OK] Current ComfyUI source: {COMFY_DIR}")

def install_backend_requirements() -> None:
    if not ENV_PY.is_file():
        fail(f"ConvRot environment not found: {ENV_PY}")
    if not UV_EXE.is_file():
        fail(f"uv not found: {UV_EXE}")

    req = COMFY_DIR / "requirements.txt"
    if not req.is_file():
        fail(f"ComfyUI requirements.txt not found: {req}")

    selected = []
    skipped = []
    for raw in req.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = requirement_name(line)
        if name in EXCLUDE_EXACT:
            skipped.append(line)
            continue
        selected.append(line)

    required_names = {requirement_name(x) for x in selected}
    for needed in ("comfy-kitchen>=0.2.24", "comfy-aimdo>=0.4.13", "torchsde"):
        if requirement_name(needed) not in required_names:
            selected.append(needed)

    filtered = MODEL_ROOT / "_comfy_backend_requirements.txt"
    filtered.write_text("\n".join(selected) + "\n", encoding="utf-8")

    log("[DEPS] Installing current Comfy backend requirements.")
    if skipped:
        log("[DEPS] Preserving existing Torch/UI packages; skipped:")
        for item in skipped:
            log(f"       {item}")

    cmd = [
        str(UV_EXE), "pip", "install",
        "--python", str(ENV_PY),
        "-r", str(filtered),
    ]
    log("[RUN] " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        fail(f"Dependency installation failed with exit code {rc}.", rc)

    try:
        filtered.unlink()
    except OSError:
        pass

    log("[OK] Backend dependencies installed without replacing Torch.")

def show_versions() -> None:
    code = (
        "import torch; print('[CHECK] torch=' + torch.__version__); "
        "import comfy_kitchen; print('[CHECK] comfy-kitchen import OK'); "
        "import comfy_aimdo; print('[CHECK] comfy-aimdo import OK'); "
        "import torchsde; print('[CHECK] torchsde import OK')"
    )
    rc = subprocess.call([str(ENV_PY), "-c", code], cwd=str(ROOT))
    if rc != 0:
        fail("Backend dependency verification failed.", rc)

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 72 + "\n")
        fh.write(f"LTX 2.5 isolated Comfy backend install - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        fh.write("=" * 72 + "\n")

    log(f"[ROOT] {ROOT}")
    log(f"[TARGET] {COMFY_DIR}")
    log(f"[ENV] {ENV_DIR}")
    log("[SAFE] /vendor is not modified.")
    log("[SAFE] Native FP16 LTX files are not modified.")
    log("[SAFE] Existing CUDA Torch installation is preserved.")

    install_comfy_source()
    install_backend_requirements()
    show_versions()

    log("")
    log("[OK] LTX 2.5 isolated current Comfy backend is ready.")
    log(f"[OK] Backend path: {COMFY_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Cancelled by user.", 130)
