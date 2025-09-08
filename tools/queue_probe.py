
import sys, os, re
from pathlib import Path

TITLE = "FrameVision Queue Probe (static scan, root-aware)"
sep = "="*56

def println(s=""):
    try:
        print(s, flush=True)
    except Exception:
        pass

def detect_root(script_dir: Path) -> Path:
    if script_dir.name.lower() == "tools":
        return script_dir.parent
    return script_dir

def scan_enqueue_hooks(root: Path):
    helpers = root / "helpers"
    if not helpers.exists():
        return [], []
    py_files = list(helpers.glob("*.py"))
    hits = []
    names = set()
    pat = re.compile(r"\b(enqueue_job|enqueue_external|enqueue_single_action|enqueue|queue_add|add_job|put)\b")
    for f in py_files:
        try:
            txt = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in pat.finditer(txt):
            hits.append((f.name, m.group(1)))
            names.add(m.group(1))
    return hits, sorted(names)

def main():
    script_dir = Path(__file__).resolve().parent
    root = detect_root(script_dir)
    sys.path.insert(0, str(root))

    println(f"{TITLE}\n{sep}")
    println(f"start cwd: {Path.cwd()}")
    println(f"script dir: {script_dir}")
    println(f"detected root: {root}")
    println(f"sys.path[0]: {sys.path[0]}")

    # 1) Report likely enqueue function names by scanning helpers/*.py
    hits, names = scan_enqueue_hooks(root)
    if hits:
        println("\n[Likely enqueue hooks found in helpers/]")
        for fn, nm in hits:
            println(f"- {fn}: {nm}")
        println(f"Unique names: {', '.join(names)}")
    else:
        println("\n[Likely enqueue hooks] none found in helpers/*.py")

    # 2) Try importing a dedicated queue module if it exists
    imported = False
    for modname in ("helpers.queue_tab", "helpers.queue", "helpers.tasks", "helpers.jobs"):
        try:
            __import__(modname)
            println(f"import {modname}: OK")
            imported = True
        except Exception as e:
            pass
    if not imported:
        println("No queue module imported (none of queue_tab/queue/tasks/jobs present).")

    # 3) Explain next steps (because creating MainWindow requires the app runtime)
    println("\nTip: This probe avoids creating the GUI (which needs a Qt event loop).")
    println("To test the live queue, open the app and use the in-app 'Probe…' button on the Upscale tab.")
    println("Or tell me which enqueue name you use (e.g. main.enqueue_job), and I’ll wire the tab to it explicitly.")
    println(sep)

if __name__ == '__main__':
    main()
