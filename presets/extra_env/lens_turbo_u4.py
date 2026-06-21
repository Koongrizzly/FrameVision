from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

MODEL_ID = "WaveCut/Lens-Turbo-SDNQ-uint4-static"
LENS_GIT_URL = "https://github.com/microsoft/Lens.git"
LENS_ZIP_URL = "https://github.com/microsoft/Lens/archive/refs/heads/main.zip"
ENV_NAME = ".lens"
MODEL_CACHE_SUBDIR = Path("models") / "lens" / "hf_cache"

# Lens is sensitive to the exact SDNQ / Triton / kernels / Transformers stack.
# This stack mirrors the working standalone Lens install reported by the user:
# torch 2.11.0+cu128, triton-windows 3.6.0.post26, diffusers 0.38.0,
# transformers 5.9.0, kernels/kernels-data 0.14.1, sdnq 0.1.9.
# Keep these pinned; loose installs can silently replace CUDA Torch or mix broken
# kernels/kernels-data versions.
PINNED_LENS_DEPS = [
    "diffusers==0.38.0",
    "transformers==5.9.0",
    "huggingface_hub[hf_xet]==1.18.0",
    "accelerate==1.13.0",
    "safetensors==0.8.0rc0",
    "sentencepiece",
    "protobuf",
    "pillow",
    "numpy==2.4.6",
    "scipy",
    "einops",
    "peft",
    "tqdm",
]
PINNED_TRITON_STACK = [
    "triton-windows==3.6.0.post26",
    "kernels==0.14.1",
    "kernels-data==0.14.1",
    "bitsandbytes==0.49.2",
]
PINNED_SDNQ = "sdnq==0.1.9"


def model_cache_dir(root: Path) -> Path:
    return root / MODEL_CACHE_SUBDIR


def runtime_cache_dirs(root: Path) -> dict[str, Path]:
    base = root / "models" / "lens"
    return {
        "hf_home": base / "hf_home",
        "hf_hub_cache": base / "hf_cache",
        "transformers_cache": base / "transformers_cache",
        "hf_modules_cache": base / "hf_modules",
        "triton_cache": base / "triton_cache",
        "kernels_cache": base / "kernels_cache",
        "torch_home": base / "torch_home",
        "xdg_cache_home": base / "xdg_cache",
    }


def runtime_env(root: Path, offline: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    dirs = runtime_cache_dirs(root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(dirs["hf_home"])
    env["HF_HUB_CACHE"] = str(dirs["hf_hub_cache"])
    env["HUGGINGFACE_HUB_CACHE"] = str(dirs["hf_hub_cache"])
    # Do not set TRANSFORMERS_CACHE; Transformers 4.57 warns that it is deprecated.
    env.pop("TRANSFORMERS_CACHE", None)
    env["HF_MODULES_CACHE"] = str(dirs["hf_modules_cache"])
    env["TRITON_CACHE_DIR"] = str(dirs["triton_cache"])
    env["KERNELS_CACHE"] = str(dirs["kernels_cache"])
    env["TORCH_HOME"] = str(dirs["torch_home"])
    env["XDG_CACHE_HOME"] = str(dirs["xdg_cache_home"])
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    return env


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd, cwd: Path | None = None, env: dict | None = None, check: bool = True) -> int:
    printable = " ".join(str(x) for x in cmd)
    log(f"\n>> {printable}")
    p = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}: {printable}")
    return p.returncode


def which(name: str) -> str | None:
    return shutil.which(name)


def find_conda() -> str | None:
    candidates = []
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(conda_exe)
    found = which("conda")
    if found:
        candidates.append(found)
    user = Path.home()
    candidates.extend([user / "miniconda3" / "Scripts" / "conda.exe", user / "anaconda3" / "Scripts" / "conda.exe", Path("C:/ProgramData/miniconda3/Scripts/conda.exe"), Path("C:/ProgramData/anaconda3/Scripts/conda.exe"), Path("C:/Miniconda3/Scripts/conda.exe"), Path("C:/Anaconda3/Scripts/conda.exe")])
    for c in candidates:
        p = Path(c)
        if p.exists(): return str(p)
    return None


def env_python(env_dir: Path) -> Path:
    # Conda envs on Windows use python.exe directly in the env root.
    # Keep Scripts/python.exe as a fallback for older venv-based installs.
    p1 = env_dir / "python.exe"
    p2 = env_dir / "Scripts" / "python.exe"
    return p1 if p1.exists() else p2


def create_env(root: Path, force: bool = False) -> Path:
    env_dir = root / "environments" / ENV_NAME
    py = env_python(env_dir)
    if force and env_dir.exists():
        log(f"Removing existing env: {env_dir}")
        shutil.rmtree(env_dir)
    if py.exists():
        log(f"Using existing Lens env: {env_dir}")
        return py
    conda = find_conda()
    if not conda:
        raise RuntimeError("Conda was not found. Install Miniconda/Anaconda or run from a shell where conda is available.")
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    run([conda, "create", "-y", "-p", env_dir, "python=3.11", "pip"])
    py = env_python(env_dir)
    if not py.exists(): raise RuntimeError(f"Python was not created at expected env path: {py}")
    return py


def pip_install(py: Path, args: list[str], check: bool = True) -> int:
    args = list(args)
    if "--no-warn-script-location" not in args:
        args.insert(0, "--no-warn-script-location")
    return run([py, "-m", "pip", "install", *args], check=check)


def py_check(py: Path, code: str) -> bool:
    return run([py, "-c", code], check=False) == 0


def get_py_output(py: Path, code: str) -> str:
    p = subprocess.run([str(py), "-c", code], text=True, capture_output=True)
    return ((p.stdout or "") + (p.stderr or "")).strip()


def torch_cuda128_ok(py: Path) -> bool:
    code = '''
try:
    import torch
    ver = str(torch.__version__)
    cuda = str(torch.version.cuda)
    major_minor = tuple(int(x) for x in ver.split('+')[0].split('.')[:2])
    ok = ver.startswith('2.11.0+cu128') and cuda == '12.8' and torch.cuda.is_available()
    print(f'torch={ver} cuda={cuda} cuda_available={torch.cuda.is_available()} ok={ok}')
    raise SystemExit(0 if ok else 1)
except Exception as e:
    print(repr(e))
    raise SystemExit(1)
'''
    return py_check(py, code)


def imports_ok(py: Path, modules: list[str]) -> bool:
    lines = ["import importlib"]
    lines += [f"importlib.import_module({m!r})" for m in modules]
    lines.append("print('imports ok')")
    return py_check(py, "\n".join(lines))


def lens_stack_report(py: Path) -> str:
    code = """
import importlib
names = ['torch', 'triton', 'kernels', 'sdnq', 'diffusers', 'transformers', 'huggingface_hub', 'bitsandbytes']
for name in names:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, '__version__', 'ok')
        if name == 'torch':
            import torch
            print(f'torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}')
        else:
            print(f'{name}={ver}')
    except Exception as exc:
        print(f'{name}_error={repr(exc)}')
"""
    return get_py_output(py, code)


def sdnq_triton_ok(py: Path) -> bool:
    code = """
try:
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        import sdnq
    text = buf.getvalue()
    print(text.strip())
    if 'Triton is not available' in text:
        raise RuntimeError('SDNQ imported but rejected Triton.')
    print('sdnq triton preflight ok')
    raise SystemExit(0)
except Exception as e:
    print(repr(e))
    raise SystemExit(1)
"""
    return py_check(py, code)


def install_cuda_torch_stack(py: Path, force: bool = False) -> None:
    args = []
    if force:
        args.append("--force-reinstall")
    args.extend([
        "torch==2.11.0+cu128",
        "torchvision==0.26.0+cu128",
        "torchaudio==2.11.0+cu128",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
    ])
    pip_install(py, args)


def install_python_packages(py: Path, allow_sdnq_fail: bool = False, skip_triton: bool = False) -> None:
    run([py, "-m", "pip", "install", "--no-warn-script-location", "--upgrade", "pip", "wheel"])
    if torch_cuda128_ok(py):
        log("\nPyTorch 2.11.0 CUDA 12.8 is already installed; skipping PyTorch reinstall.")
    else:
        log("\nInstalling PyTorch 2.11.0 CUDA 12.8 build from the official cu128 index...")
        install_cuda_torch_stack(py)

    log("\nInstalling/pinning Lens Python dependency stack...")
    log("This mirrors the working standalone Lens stack instead of guessing newer/older package versions.")
    pip_install(py, ["--upgrade", *PINNED_LENS_DEPS])

    if skip_triton:
        log("\nSkipping Triton stack install because --skip-triton was used.")
    else:
        log("\nInstalling/pinning Lens Triton/SDNQ support stack without dependency changes...")
        log("Using --no-deps here is intentional: otherwise pip may replace CUDA torch 2.11.0+cu128 or pull a mismatched kernels-data package.")
        # Remove mismatched leftovers first. kernels 0.14.1 with kernels-data 0.15.2 can import-fail
        # with StrictDataclassFieldValidationError before Lens even loads.
        pip_install(py, ["uninstall", "-y", "kernels", "kernels-data"], check=False)
        pip_install(py, ["--upgrade", "--force-reinstall", "--no-deps", *PINNED_TRITON_STACK])

    log("\nInstalling/pinning SDNQ package for UINT4 model support without dependency changes...")
    rc = pip_install(py, ["--upgrade", "--force-reinstall", "--no-deps", PINNED_SDNQ], check=False)
    if rc != 0:
        msg = "sdnq failed to install. This model will probably not load without SDNQ support."
        if allow_sdnq_fail:
            log("WARNING: " + msg)
        else:
            raise RuntimeError(msg)

    # Repair CUDA Torch if a polluted existing .lens env still managed to disturb it.
    # Do not force Hugging Face Hub below 1.0; the working standalone stack uses the
    # newer Lens/Transformers line and allows huggingface_hub 1.x.
    log("\nRe-checking CUDA Torch 2.11.0+cu128 after Lens support packages...")
    if not torch_cuda128_ok(py):
        install_cuda_torch_stack(py)

    if not torch_cuda128_ok(py):
        raise RuntimeError("Torch 2.11.0 CUDA 12.8 check still failed after repair. Run this installer with --force-env to recreate environments/.lens.")

    if imports_ok(py, ["PySide6"]):
        out = get_py_output(py, "import PySide6; print(getattr(PySide6, '__version__', 'unknown'))")
        log(f"\nPySide6 already imports; skipping PySide6 reinstall. Version: {out}")
    else:
        log("\nInstalling PySide6 for the standalone Lens UI...")
        pip_install(py, ["--upgrade", "PySide6"])

    log("\nLens stack after install/pin:")
    log(lens_stack_report(py))
    if not skip_triton and not sdnq_triton_ok(py):
        raise RuntimeError(
            "SDNQ still reports that Triton is not available after installing the pinned Lens stack. "
            "Delete environments/.lens and rerun this installer, or run with --force-env."
        )


def download_lens_repo(root: Path, force_repo: bool = False) -> Path:
    repo_dir = root / "models" / "lens" / "repos" / "Lens"
    if force_repo and repo_dir.exists():
        log(f"Removing existing Lens repo: {repo_dir}")
        shutil.rmtree(repo_dir)
    if (repo_dir / "lens").exists():
        log(f"Using existing Lens repo: {repo_dir}")
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    git = which("git")
    if git:
        run([git, "clone", "--depth", "1", LENS_GIT_URL, repo_dir])
        return repo_dir
    log("Git was not found; downloading Lens repo ZIP instead...")
    tmp = root / "temp" / "lens_repo_main.zip"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LENS_ZIP_URL, tmp)
    extract_root = root / "temp" / "lens_repo_extract"
    if extract_root.exists(): shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tmp, "r") as zf: zf.extractall(extract_root)
    src = extract_root / "Lens-main"
    if not src.exists(): raise RuntimeError("Downloaded Lens ZIP did not contain expected Lens-main folder.")
    if repo_dir.exists(): shutil.rmtree(repo_dir)
    shutil.move(str(src), str(repo_dir))
    shutil.rmtree(extract_root, ignore_errors=True)
    return repo_dir


def verify(py: Path, root: Path, repo_dir: Path) -> None:
    code = rf'''
import os, sys, json
sys.path.insert(0, r"{repo_dir}")
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
if torch.cuda.is_available(): print("gpu", torch.cuda.get_device_name(0))
try:
    import triton
    print("triton import", "ok", getattr(triton, "__version__", "unknown"))
except Exception as e: print("triton import failed", repr(e))
try:
    import bitsandbytes as bnb
    print("bitsandbytes import", "ok", getattr(bnb, "__version__", "unknown"))
except Exception as e: print("bitsandbytes import failed", repr(e))
try:
    import kernels
    print("kernels import", "ok", getattr(kernels, "__version__", "ok"))
except Exception as e:
    print("kernels import failed", repr(e))
    raise
try:
    import sdnq
    print("sdnq import", "ok")
except Exception as e:
    print("sdnq import failed", repr(e))
    raise
try:
    import PySide6
    print("PySide6 import", "ok", getattr(PySide6, "__version__", "unknown"))
except Exception as e:
    print("PySide6 import failed", repr(e))
    raise
from lens import LensPipeline
print("LensPipeline import", "ok")
'''
    run([py, "-c", code], env=runtime_env(root, offline=False))


def prepare_kernel_cache(py: Path, root: Path) -> None:
    """Download/trust the runtime kernels into FrameVision's portable Lens cache folders."""
    env = runtime_env(root, offline=False)
    kernel_cache = runtime_cache_dirs(root)["kernels_cache"]
    code = rf'''
from pathlib import Path
kernel_cache = Path(r"{kernel_cache}")
kernel_cache.mkdir(parents=True, exist_ok=True)
try:
    from kernels import get_kernel
    print("Preparing trusted local kernel cache: kernels-community/gpt-oss-triton-kernels")
    get_kernel("kernels-community/gpt-oss-triton-kernels", trust_remote_code=True)
    hits = list(kernel_cache.rglob("metadata.json"))
    print("Kernel cache metadata files:", len(hits))
    if not hits:
        raise RuntimeError(f"Kernel cache did not contain metadata.json under {kernel_cache}")
    print("Kernel cache prepared:", kernel_cache)
except Exception as e:
    print("Kernel cache prime failed:", repr(e))
    raise
'''
    run([py, "-c", code], env=env)


def maybe_snapshot_model(py: Path, root: Path) -> None:
    """Download the Lens Turbo U4 model snapshot into the portable FrameVision model cache."""
    target = model_cache_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    env = runtime_env(root, offline=False)
    code = rf'''
from huggingface_hub import snapshot_download
from pathlib import Path
cache_dir = Path(r"{target}")
print("Downloading portable model cache for:", "{MODEL_ID}")
print("cache_dir=", cache_dir)
path = snapshot_download(repo_id="{MODEL_ID}", cache_dir=str(cache_dir))
print("snapshot_download path=", path)
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("{MODEL_ID}", cache_dir=str(cache_dir), trust_remote_code=True)
    print("AutoConfig primed:", type(cfg).__name__)
except Exception as e:
    print("AutoConfig prime skipped:", repr(e))
print("Portable model cache prepared.")
'''
    run([py, "-c", code], env=env)

def print_model_cache_hint(root: Path) -> None:
    target = model_cache_dir(root)
    target.mkdir(parents=True, exist_ok=True)
    expected = target / "models--WaveCut--Lens-Turbo-SDNQ-uint4-static"
    log("\nPortable Hugging Face model cache:")
    log(f"  {target}")
    if expected.exists():
        log(f"Found local Lens Turbo U4 cache folder: {expected}")
    else:
        log("Model cache folder was not found yet.")
        log("If you already downloaded it in the default Hugging Face cache, move/copy:")
        log(r"  %USERPROFILE%\.cache\huggingface\hub\models--WaveCut--Lens-Turbo-SDNQ-uint4-static")
        log("to:")
        log(f"  {expected}")
        log("Then the test runner will reuse it instead of downloading again.")

def write_env_info(root: Path, py: Path, repo_dir: Path) -> None:
    logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    info = logs / "lens_turbo_u4_install_info.txt"
    extra = get_py_output(py, '''
import sys
print('python=' + sys.version.split()[0])
try:
    import torch
    print('torch=' + str(torch.__version__))
    print('torch_cuda=' + str(torch.version.cuda))
except Exception as e: print('torch_error=' + repr(e))
for name in ['triton', 'bitsandbytes', 'kernels', 'sdnq', 'PySide6', 'diffusers', 'transformers', 'accelerate']:
    try:
        mod = __import__(name)
        print(name + '=' + str(getattr(mod, '__version__', 'ok')))
    except Exception as e:
        print(name + '_error=' + repr(e))
''')
    caches = runtime_cache_dirs(root)
    info.write_text("Lens Turbo SDNQ UINT4 installer\n" + f"installed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\nroot={root}\nenv_python={py}\nlens_repo={repo_dir}\nmodel_id={MODEL_ID}\nmodel_cache={model_cache_dir(root)}\nportable_hf_home={caches['hf_home']}\nportable_transformers_cache={caches['transformers_cache']}\nportable_hf_modules_cache={caches['hf_modules_cache']}\nportable_triton_cache={caches['triton_cache']}\nportable_kernels_cache={caches['kernels_cache']}\nportable_torch_home={caches['torch_home']}\n\nPackage info:\n{extra}\n", encoding="utf-8")
    log(f"Wrote install info: {info}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Lens Turbo SDNQ UINT4 dedicated environment, repo, portable kernels, and optional model cache.")
    parser.add_argument("--force-env", action="store_true", help="Delete and recreate the dedicated Lens conda env at environments/.lens.")
    parser.add_argument("--force-repo", action="store_true", help="Delete and redownload the Microsoft Lens repo.")
    parser.add_argument("--download-model", action="store_true", help="Also download the Lens Turbo U4 model files into the portable FrameVision cache now. Default: model download happens on first use.")
    parser.add_argument("--skip-kernel-cache", action="store_true", help="Skip preparing/trusting the portable kernels cache. Default: kernels are cached during optional install.")
    parser.add_argument("--allow-sdnq-fail", action="store_true", help="Continue if pip install sdnq fails.")
    parser.add_argument("--skip-triton", action="store_true", help="Do not install/check Triton Windows.")
    args = parser.parse_args()
    root = root_dir(); log(f"Root: {root}")
    (root / "models" / "lens").mkdir(parents=True, exist_ok=True)
    model_cache_dir(root).mkdir(parents=True, exist_ok=True)
    for path in runtime_cache_dirs(root).values():
        path.mkdir(parents=True, exist_ok=True)
    (root / "output" / "lens_turbo_u4").mkdir(parents=True, exist_ok=True)
    (root / "temp").mkdir(parents=True, exist_ok=True)
    py = create_env(root, force=args.force_env)
    install_python_packages(py, allow_sdnq_fail=args.allow_sdnq_fail, skip_triton=args.skip_triton)
    repo_dir = download_lens_repo(root, force_repo=args.force_repo)
    verify(py, root, repo_dir)
    if args.skip_kernel_cache:
        log("\nSkipping Lens runtime kernel cache because --skip-kernel-cache was used.")
    else:
        prepare_kernel_cache(py, root)
    if args.download_model:
        maybe_snapshot_model(py, root)
    else:
        log("\nSkipping Lens model download during optional install.")
        log("The model files will be downloaded automatically on first use inside the Lens tool.")
    print_model_cache_hint(root)
    write_env_info(root, py, repo_dir)
    log("\nDONE"); log("Lens Turbo U4 environment, repo, and portable kernel cache are ready."); log("Model files are downloaded on first use unless --download-model was used.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
