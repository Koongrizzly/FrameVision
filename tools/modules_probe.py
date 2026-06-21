import inspect, sys, os, pathlib
print("FrameVision Modules Probe (root-aware)")
print("="*60)
start_dir = os.getcwd()
here = pathlib.Path(__file__).resolve().parent
root = None; up = here
for _ in range(6):
    if (up / "helpers").exists(): root = up; break
    up = up.parent
if not root and pathlib.Path("helpers").exists(): root = pathlib.Path(".").resolve()
print("start cwd:", start_dir)
print("script dir:", str(here))
print("detected root:", str(root) if root else "NOT FOUND")
if root:
    sys.path.insert(0, str(root)); os.chdir(str(root))
    print("sys.path[0]:", sys.path[0])
else:
    print("ERROR: couldn't locate project root with a 'helpers' folder."); sys.exit(1)

def src(m):
    try: return inspect.getsourcefile(m) or inspect.getfile(m)
    except Exception as e: return f"ERR: {e}"

try:
    import helpers.settings_tab as s; print("helpers.settings_tab:", src(s))
except Exception as e:
    print("helpers.settings_tab: import error ->", e)
try:
    import helpers.theme_creator as tc; print("helpers.theme_creator:", src(tc))
except Exception as e:
    print("helpers.theme_creator: import error ->", e)
