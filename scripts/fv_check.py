import os, sys, shutil
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from path_resolver import MODELS_DIR, BIN_DIR

def human(n):
    for u in ["B","KB","MB","GB","TB"]:
        if n < 1024 or u=="TB": return f"{n:.1f} {u}"
        n/=1024

def disk_free(path):
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except Exception:
        return 0

def have_cuda_gpu():
    try:
        import torch
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"], text=True, timeout=3)
        name, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return True, f"{name} ({mem})"
    except Exception:
        return False, None

def python_info(): return sys.version.split()[0]

def torch_info():
    try:
        import torch
        return torch.__version__, getattr(torch.version, "cuda", None), torch.cuda.is_available()
    except Exception:
        return None, None, False

def estimate_space():
    base_app = 200 * 1024**2
    blip_models = 2100 * 1024**2
    florence = 1200 * 1024**2
    torch_cpu = 1800 * 1024**2
    torch_cuda = 3200 * 1024**2
    return base_app + torch_cpu + blip_models, base_app + torch_cuda + blip_models + florence

def main():
    print("=== FrameVision Environment Check ===")
    print(f"Python: {python_info()}")
    tv, tc, has = torch_info()
    print(f"PyTorch: {tv or 'not installed'}  CUDA: {tc or '-'}  cuda_available={has}")
    has_gpu, gpu_name = have_cuda_gpu()
    print(f"Detected CUDA GPU: {has_gpu}  {gpu_name or ''}".strip())
    print(f"Models dir: {MODELS_DIR}")
    print(f"Bin dir   : {BIN_DIR}")
    free = disk_free(ROOT)
    print(f"Free space on install drive: {human(free)}")
    cpu_need, gpu_need = estimate_space()
    print("\nApprox space needed:")
    print(f"  Core only: ~{human(200*1024**2)}")
    print(f"  Full CPU : ~{human(cpu_need)}")
    print(f"  Full CUDA: ~{human(gpu_need)}")
    print("\nNotes:")
    print(" - All model/binary downloads are extracted into the resolved folders above.")
    print(" - Zip files are deleted after extraction.")
    print(" - Hugging Face cache is redirected to .hf_cache under the app folder.")
if __name__ == '__main__':
    raise SystemExit(main())
