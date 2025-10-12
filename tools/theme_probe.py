
import sys, os, importlib, inspect
from pathlib import Path

print("FrameVision Theme Probe (fixed)")
print("="*60)
print("cwd:", os.getcwd())
print("sys.path[0]:", sys.path[0])

def safe_import(name):
    try:
        m = importlib.import_module(name)
        print(f"import {name}: OK ->", getattr(m, "__file__", "?"))
        return m
    except Exception as e:
        print(f"import {name}: FAIL -> {e}")
        return None

themes = safe_import("helpers.themes")
fvapp = safe_import("helpers.framevision_app")

if themes:
    ap = getattr(themes, "apply_theme", None)
    qfun = getattr(themes, "qss_for_theme", None)
    print("helpers.themes.apply_theme:", "OK" if callable(ap) else "MISSING")
    if qfun:
        try:
            q_even = qfun("Evening")
            q_night = qfun("Night")
            print("QSS length Evening:", len(q_even or ""))
            print("QSS length Night:", len(q_night or ""))
        except Exception as e:
            print("qss_for_theme error:", e)

if fvapp:
    ap2 = getattr(fvapp, "apply_theme", None)
    print("helpers.framevision_app.apply_theme:", "OK" if callable(ap2) else "MISSING")
    try:
        src = inspect.getsource(ap2) if ap2 else ""
        print("--- framevision_app.apply_theme (first 6 lines) ---")
        print("\n".join(src.splitlines()[:6]))
    except Exception as e:
        print("could not inspect apply_theme:", e)

try:
    from helpers.framevision_app import config
    print("config['theme']:", config.get("theme"))
except Exception as e:
    print("config read failed:", e)

print("="*60)
