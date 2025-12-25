
import os, sys, json, subprocess, shutil, time
from pathlib import Path

def p(msg): print(msg, flush=True)

def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:
        p(f"[ERR] Could not read JSON: {path} -> {e}")
        return None

def rel_to(base, child):
    try:
        return str(Path(child).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return str(Path(child).resolve())

def find_candidates(models_root: Path, names):
    found = []
    try:
        for name in names:
            for p in models_root.rglob(name):
                if p.is_file():
                    found.append(p)
    except Exception:
        pass
    return found

def run_exe_help(exe: Path, timeout=7):
    try:
        start = time.time()
        proc = subprocess.Popen([str(exe), "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(exe.parent))
        out, _ = proc.communicate(timeout=timeout)
        elapsed = time.time() - start
        text = out.decode(errors="ignore")
        return True, elapsed, text[:800]
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        return False, timeout, "[timeout]"
    except Exception as e:
        return False, 0, f"[exception] {e}"

def main():
    # Assume script is placed under tools/diagnostics
    app_root = Path(__file__).resolve().parents[2]
    p(f"[diag] App root: {app_root}")
    cfg_path = app_root / "config.json"
    mani_path = app_root / "models_manifest.json"
    cfg = read_json(cfg_path) if cfg_path.exists() else None
    if not cfg:
        p(f"[warn] config.json not found or invalid at {cfg_path}")

    models_root = None
    if cfg and isinstance(cfg, dict) and "models_folder" in cfg:
        models_root = Path(cfg["models_folder"])
        p(f"[diag] config.models_folder = {cfg['models_folder']}")
    else:
        models_root = app_root / "models"
        p(f"[diag] using default models folder = {models_root}")

    p(f"[diag] models_root exists? {models_root.exists()}")

    mani = read_json(mani_path) if mani_path.exists() else None
    if mani is None:
        p(f"[warn] models_manifest.json not found or invalid at {mani_path}")
    else:
        keys = list(mani.keys())
        p(f"[diag] manifest keys ({len(keys)}): {', '.join(keys[:8])}{' ...' if len(keys)>8 else ''}")
        # Try resolving 1-2 common keys
        for key in ["RealESR-general-x4v3","RealESRGAN-x4plus","RealESR-animevideov3-x4"]:
            if key in mani:
                exe_rel = (mani[key] or {}).get("exe", "")
                p(f"[diag] manifest[{key!r}].exe = {exe_rel} -> {models_root / exe_rel} (exists? {(models_root / exe_rel).exists()})")

    # Look for typical NCNN exe locations
    typical = [
        models_root / "realesrgan" / ("realesrgan-ncnn-vulkan.exe" if os.name=="nt" else "realesrgan-ncnn-vulkan"),
        models_root / "realesrgan-ncnn-vulkan.exe",
    ]
    for t in typical:
        p(f"[diag] check typical: {t} exists? {t.exists()}")
    # Find any candidates in tree
    names = ["realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan"]
    cands = find_candidates(models_root, names)
    if cands:
        p("[diag] found candidates:")
        for c in cands[:12]:
            p(f"  - {rel_to(models_root, c)}")

    # If we have a strong candidate, try '-h'
    exe = None
    for t in typical:
        if t.exists():
            exe = t; break
    if not exe and cands:
        exe = cands[0]

    if exe:
        p(f"[diag] trying to run: {exe.name} -h (cwd={exe.parent}) ...")
        ok, elapsed, head = run_exe_help(exe)
        p(f"[diag] run ok? {ok}, elapsed ~{elapsed:.1f}s")
        p("----- output head -----")
        print(head)
        p("------------------------")
        if not ok:
            p("[hint] If this doesn't print help quickly, Vulkan/GPU may be unavailable or DLLs missing.")
    else:
        p("[ERR] No realesrgan-ncnn-vulkan executable found under models_root.")

    # Recommend manifest entry based on exe location
    if exe:
        try:
            rel = exe.relative_to(models_root)
            p(f"[suggest] models_manifest.json entry:\n{{\n  \"RealESR-general-x4v3\": {{ \"exe\": \"{rel.as_posix()}\" }}\n}}")
        except Exception:
            pass

if __name__ == "__main__":
    main()
