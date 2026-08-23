from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
UV=ROOT/"presets"/"bin"/"uv"/"uv.exe"
PY=ROOT/"environments"/"ltx25"/"python.exe"
INDEX="https://download.pytorch.org/whl/cu128"
def run(args):
    print(">>>", " ".join(map(str,args)), flush=True)
    subprocess.check_call([str(x) for x in args], cwd=str(ROOT))
if not PY.is_file(): raise SystemExit(f"Missing env Python: {PY}")
if not UV.is_file(): raise SystemExit(f"Missing uv.exe: {UV}")
subprocess.call([str(UV),"pip","uninstall","--python",str(PY),"torchvision"], cwd=str(ROOT))
run([UV,"pip","install","--python",PY,"--reinstall","torch==2.7.0","torchaudio==2.7.0","--index-url",INDEX])
run([PY,"-c","import torch, torchaudio; print(torch.__version__); print(torchaudio.__version__)"])
print("[OK] FP16 Torch ABI repaired. torchvision intentionally not installed.")
