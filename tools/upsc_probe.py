
import sys, os, hashlib, subprocess, shutil
from pathlib import Path

TITLE = "FrameVision Upscaler Probe (root-aware)"
sep = "="*56

def println(s=""):
    try:
        print(s, flush=True)
    except Exception:
        pass

def detect_root(script_dir: Path) -> Path:
    # project root is parent of tools/
    if script_dir.name.lower() == "tools":
        return script_dir.parent
    return script_dir

def sha12(path: Path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024*256), b""):
                h.update(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return "?"*12

def main():
    script_dir = Path(__file__).resolve().parent
    root = detect_root(script_dir)
    sys.path.insert(0, str(root))

    println(f"{TITLE}\n{sep}")
    println(f"start cwd: {Path.cwd()}")
    println(f"script dir: {script_dir}")
    println(f"detected root: {root}")
    println(f"sys.path[0]: {sys.path[0]}")

    # List important helper files
    suspects = [
        "helpers/upsc.py",
        "helpers/framevision_app.py",
        "helpers/queue_tab.py",
    ]
    for s in suspects:
        p = root / s
        if p.exists():
            println(f"{s}: OK  {sha12(p)}")
        else:
            println(f"{s}: MISSING")

    # FFmpeg
    ff = root / "presets/bin/ffmpeg.exe"
    ff2 = shutil.which("ffmpeg")
    println("\n[FFmpeg]")
    if ff.exists():
        println(f"bundled: {ff}")
    else:
        println("bundled: NOT FOUND (presets/bin/ffmpeg.exe)")
    println(f"PATH ffmpeg: {ff2 or 'NOT FOUND'}")
    try:
        cp = subprocess.run([str(ff if ff.exists() else "ffmpeg"), "-version"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        println(f"launch rc: {cp.returncode}")
        println((cp.stdout.splitlines() or [''])[0])
    except Exception as e:
        println(f"launch failed: {e}")

    # Real-ESRGAN
    rexe = root / "models/realesrgan/realesrgan-ncnn-vulkan.exe"
    println("\n[Real-ESRGAN]")
    println(f"exe: {rexe if rexe.exists() else 'NOT FOUND (expect models/realesrgan/realesrgan-ncnn-vulkan.exe)'}")

    # List models
    mdir = root / "models/realesrgan"
    if mdir.exists():
        params = sorted(mdir.glob("*.param"))
        println(f"models dir: {mdir}  ({len(params)} .param files)")
        for p in params[:12]:
            b = p.with_suffix(".bin")
            println(f"- {p.name:<36} | bin={'OK' if b.exists() else 'MISS'} | sha12={sha12(p)}")
        if len(params) > 12:
            println(f"... ({len(params)-12} more)")
    else:
        println("models dir: NOT FOUND")

    # Try --help on realesrgan
    if rexe.exists():
        try:
            cp = subprocess.run([str(rexe), "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            println(f"realesrgan launch rc: {cp.returncode}")
            line = (cp.stdout or cp.stderr).splitlines()[:1]
            if line:
                println(line[0])
        except Exception as e:
            println(f"realesrgan launch failed: {e}")

    println("\nTip: run this from each install to compare hashes quickly.")
    println(sep)

if __name__ == "__main__":
    main()
