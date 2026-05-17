#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

OFFICIAL_REPO_ZIP = "https://github.com/HiDream-ai/HiDream-O1-Image/archive/refs/heads/main.zip"

MODEL_INFO = {
    "base": {
        "label": "Base / Full BF16",
        "repo_id": "drbaph/HiDream-O1-Image-BF16",
        "folder": "HiDream-O1-Image-BF16",
    },
    "base_fp8": {
        "label": "Base / Full FP8",
        "repo_id": "drbaph/HiDream-O1-Image-FP8",
        "folder": "HiDream-O1-Image-FP8",
    },
    "dev": {
        "label": "Dev BF16",
        "repo_id": "drbaph/HiDream-O1-Image-Dev-BF16",
        "folder": "HiDream-O1-Image-Dev-BF16",
    },
    "dev_2604_bf16": {
        "label": "Dev 2604 BF16",
        "repo_id": "drbaph/HiDream-O1-Image-Dev-2604-BF16",
        "folder": "HiDream-O1-Image-Dev-2604-BF16",
    },
    "dev_fp8": {
        "label": "Dev FP8",
        "repo_id": "drbaph/HiDream-O1-Image-Dev-FP8",
        "folder": "HiDream-O1-Image-Dev-FP8",
    },
}

PYTHON_VERSION = "3.11"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TORCH_VERSION = "2.8.0+cu128"
TORCHVISION_VERSION = "0.23.0+cu128"
FLASH_ATTN_WHEEL_URL = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.8-cp311-cp311-win_amd64.whl"
SAGE_ATTENTION_WHEEL_URL = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"
TRITON_WINDOWS_SPEC = "triton-windows<3.5"

SMALL_HF_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.json", "chat_template.jinja",
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json", "model.safetensors.index.json",
]


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = root_dir()
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRA_ENV_DIR = ROOT / "presets" / "extra_env"
ENVIRONMENTS_DIR = ROOT / "environments"
MINICONDA_DIR = ENVIRONMENTS_DIR / "_miniconda"
ENV_DIR = ENVIRONMENTS_DIR / ".hidream_dev"
MODELS_ROOT = ROOT / "models" / "hidream_bf16"
OFFICIAL_REPO_DIR = MODELS_ROOT / "HiDream-O1-Image"
RESULTS_DIR = MODELS_ROOT / "results"
DOWNLOADS_DIR = EXTRA_ENV_DIR / "_downloads_hidream"
LOCAL_CACHE_DIR = MODELS_ROOT / ".hf_cache"
LOCAL_HF_HOME = MODELS_ROOT / ".hf_home"
ARIA2C_CANDIDATES = [
    ROOT / "aria2c.exe",
    ROOT.parent / "aria2c.exe",
]


def log(message: str) -> None:
    print(f"[HiDream installer] {message}", flush=True)


def fail(message: str, code: int = 1) -> None:
    print("")
    print("[HiDream installer] ERROR")
    print(message)
    print("")
    raise SystemExit(code)


def run(cmd, cwd: Path | None = None, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    if isinstance(cmd, (list, tuple)):
        shown = " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)
    else:
        shown = str(cmd)
    log(f"Running: {shown}")
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def ensure_dirs() -> None:
    for p in [EXTRA_ENV_DIR, ENVIRONMENTS_DIR, MODELS_ROOT, RESULTS_DIR, DOWNLOADS_DIR, LOCAL_CACHE_DIR, LOCAL_HF_HOME]:
        p.mkdir(parents=True, exist_ok=True)


def parse_model_selection(selection: str | None) -> list[str]:
    """Normalize interactive/CLI model choices to internal MODEL_INFO keys."""
    choice = (selection or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "1": ["base"],
        "base": ["base"],
        "b": ["base"],
        "bf16_base": ["base"],
        "base_bf16": ["base"],
        "full": ["base"],
        "full_bf16": ["base"],
        "2": ["dev"],
        "dev": ["dev"],
        "d": ["dev"],
        "bf16_dev": ["dev"],
        "dev_bf16": ["dev"],
        "3": ["dev_2604_bf16"],
        "dev_2604": ["dev_2604_bf16"],
        "dev2604": ["dev_2604_bf16"],
        "2604": ["dev_2604_bf16"],
        "dev_2604_bf16": ["dev_2604_bf16"],
        "bf16_dev_2604": ["dev_2604_bf16"],
        "dev2604_bf16": ["dev_2604_bf16"],
        "4": ["base_fp8"],
        "base_fp8": ["base_fp8"],
        "bfp8": ["base_fp8"],
        "full_fp8": ["base_fp8"],
        "fp8_base": ["base_fp8"],
        "5": ["dev_fp8"],
        "dev_fp8": ["dev_fp8"],
        "dfp8": ["dev_fp8"],
        "fp8_dev": ["dev_fp8"],
        "6": ["base", "dev"],
        "both": ["base", "dev"],
        "bf16": ["base", "dev"],
        "both_bf16": ["base", "dev"],
        "7": ["base_fp8", "dev_fp8"],
        "fp8": ["base_fp8", "dev_fp8"],
        "both_fp8": ["base_fp8", "dev_fp8"],
        "8": ["base", "base_fp8", "dev", "dev_2604_bf16", "dev_fp8"],
        "all": ["base", "base_fp8", "dev", "dev_2604_bf16", "dev_fp8"],
        "all_four": ["base", "base_fp8", "dev", "dev_2604_bf16", "dev_fp8"],
    }
    if choice in mapping:
        return mapping[choice]
    if choice in {"9", "q", "quit", "exit", "none", ""}:
        raise SystemExit(0)
    valid = ", ".join(sorted(k for k in mapping if not k.isdigit()))
    fail(f"Unknown --models selection: {selection!r}\nValid values include: {valid}")


def choose_models() -> list[str]:
    print("")
    print("============================================================")
    print(" HiDream installer")
    print("============================================================")
    print("Choose what to install/download:")
    print("  1) Base / Full BF16 only")
    print("  2) Dev BF16 only")
    print("  3) Dev 2604 BF16 only")
    print("  4) Base / Full FP8 only")
    print("  5) Dev FP8 only")
    print("  6) Both original BF16 models")
    print("  7) Both FP8 models")
    print("  8) All models, including Dev 2604 BF16")
    print("  9) Nothing / Quit")
    print("")
    return parse_model_selection(input("Selection [1-9]: "))


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading: {url}")
    log(f"To: {dest}")
    urllib.request.urlretrieve(url, dest)
    log("Download complete.")


def find_conda() -> Path | None:
    candidates = [
        MINICONDA_DIR / "Scripts" / "conda.exe",
        MINICONDA_DIR / "condabin" / "conda.bat",
    ]
    for c in candidates:
        if c.exists():
            return c
    from_path = shutil.which("conda")
    return Path(from_path) if from_path else None


def ensure_miniconda() -> Path:
    conda = find_conda()
    if conda and conda.exists():
        log(f"Using conda: {conda}")
        return conda
    installer = DOWNLOADS_DIR / "Miniconda3-latest-Windows-x86_64.exe"
    if not installer.exists():
        download_file("https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe", installer)
    log(f"Installing local Miniconda to: {MINICONDA_DIR}")
    MINICONDA_DIR.mkdir(parents=True, exist_ok=True)
    run([str(installer), "/InstallationType=JustMe", "/AddToPath=0", "/RegisterPython=0", "/S", f"/D={MINICONDA_DIR}"])
    conda = find_conda()
    if not conda or not conda.exists():
        fail("Miniconda install finished, but conda was not found.")
    return conda


def ensure_env(conda: Path) -> tuple[Path, bool]:
    """Return (env_python, env_already_existed)."""
    env_python = ENV_DIR / "python.exe"
    if env_python.exists():
        log(f"Using existing environment, skipping env creation: {ENV_DIR}")
        return env_python, True
    log(f"Creating Miniconda environment at: {ENV_DIR}")
    run([str(conda), "create", "-y", "-p", str(ENV_DIR), f"python={PYTHON_VERSION}"])
    if not env_python.exists():
        fail(f"Environment was created, but python.exe was not found: {env_python}")
    return env_python, False


def pip_install(env_python: Path, packages: list[str], extra_args: list[str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [str(env_python), "-m", "pip", "install"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(packages)
    return run(cmd, check=check)


def module_import_works(env_python: Path, module_name: str) -> bool:
    result = run([str(env_python), "-c", f"import {module_name}; print('{module_name}: OK')"], check=False)
    return result.returncode == 0


def existing_env_stack_ok(env_python: Path) -> bool:
    """Fast verification for an existing env.

    If this returns True, do not reinstall PyTorch/CUDA/attention packages.
    This avoids repeatedly downloading the multi-GB CUDA wheels.
    """
    code = r"""
import importlib
import torch

print("Python env check")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit(10)
if not str(torch.__version__).startswith("2.8.0"):
    raise SystemExit(11)
if str(torch.version.cuda) != "12.8":
    raise SystemExit(12)

for name in ["transformers", "diffusers", "accelerate", "einops", "PIL", "safetensors", "tqdm", "huggingface_hub", "PySide6", "triton", "flash_attn", "sageattention"]:
    importlib.import_module(name)
    print(name + ": OK")
"""
    result = run([str(env_python), "-c", code], check=False)
    if result.returncode == 0:
        log("Existing environment stack verified. Skipping heavy dependency reinstall.")
        return True
    log("Existing environment stack is incomplete or mismatched. Dependencies will be installed/updated.")
    return False


def flash_attention_import_works(env_python: Path) -> bool:
    return module_import_works(env_python, "flash_attn") or module_import_works(env_python, "flash_attn_interface")


def install_local_wheels(env_python: Path, patterns: list[str], label: str) -> bool:
    wheels: list[Path] = []
    for folder in [SCRIPT_DIR, EXTRA_ENV_DIR, ROOT]:
        for pat in patterns:
            wheels.extend(sorted(folder.glob(pat)))
    installed = False
    for wheel in wheels:
        log(f"Installing local {label} wheel: {wheel}")
        result = pip_install(env_python, [str(wheel)], extra_args=["--no-deps", "--force-reinstall"], check=False)
        if result.returncode == 0:
            installed = True
    return installed


def install_torch_stack(env_python: Path) -> None:
    log(f"Installing pinned CUDA PyTorch stack: torch {TORCH_VERSION}, torchvision {TORCHVISION_VERSION}.")
    result = pip_install(env_python, [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"], extra_args=["--index-url", TORCH_INDEX_URL, "--no-cache-dir"], check=False)
    if result.returncode != 0:
        log("Exact +cu128 pin failed. Trying torch==2.8.0 / torchvision==0.23.0 from the cu128 index.")
        pip_install(env_python, ["torch==2.8.0", "torchvision==0.23.0"], extra_args=["--index-url", TORCH_INDEX_URL, "--no-cache-dir"])
    code = r'''
import sys, torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Installed torch does not see CUDA.")
if not str(torch.__version__).startswith("2.8.0"):
    raise SystemExit("Wrong torch version; expected 2.8.0.x")
if str(torch.version.cuda) != "12.8":
    raise SystemExit("Wrong torch CUDA build; expected 12.8")
'''
    run([str(env_python), "-c", code])


def install_attention_backends(env_python: Path) -> None:
    log(f"Installing Triton for Windows: {TRITON_WINDOWS_SPEC}")
    run([str(env_python), "-m", "pip", "uninstall", "-y", "triton"], check=False)
    run([str(env_python), "-m", "pip", "uninstall", "-y", "triton-windows"], check=False)
    pip_install(env_python, [TRITON_WINDOWS_SPEC], extra_args=["--no-cache-dir"], check=False)
    module_import_works(env_python, "triton")
    log("Installing FlashAttention from pinned Windows wheel URL.")
    if not install_local_wheels(env_python, ["flash_attn*.whl", "flash-attn*.whl"], "FlashAttention"):
        pip_install(env_python, [FLASH_ATTN_WHEEL_URL], extra_args=["--no-deps", "--force-reinstall", "--no-cache-dir"], check=False)
    flash_attention_import_works(env_python)
    log("Installing SageAttention from pinned Windows wheel URL.")
    if not install_local_wheels(env_python, ["sageattention*.whl", "SageAttention*.whl"], "SageAttention"):
        pip_install(env_python, [SAGE_ATTENTION_WHEEL_URL], extra_args=["--no-deps", "--force-reinstall", "--no-cache-dir"], check=False)
    module_import_works(env_python, "sageattention")


def ensure_python_packages(env_python: Path) -> None:
    log("Upgrading pip/setuptools/wheel.")
    run([str(env_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    install_torch_stack(env_python)
    log("Installing HiDream dependencies.")
    pip_install(env_python, ["transformers==4.57.1", "diffusers", "accelerate", "einops", "pillow", "scipy", "numpy", "safetensors", "tqdm", "huggingface_hub", "hf_xet", "packaging", "ninja", "PySide6"], extra_args=["--no-cache-dir"])
    install_attention_backends(env_python)


def download_official_repo() -> None:
    if (OFFICIAL_REPO_DIR / "models" / "pipeline.py").exists():
        log(f"Official HiDream repo already exists: {OFFICIAL_REPO_DIR}")
        return
    for old_repo in [ROOT / "models" / "hidream_bf16" / "HiDream-O1-Image", ROOT / "models" / "hidream" / "HiDream-O1-Image", ROOT / "models" / "hidream_dev" / "HiDream-O1-Image", ROOT / "models" / "hidream_fp8" / "HiDream-O1-Image"]:
        try:
            if old_repo.resolve() == OFFICIAL_REPO_DIR.resolve():
                continue
        except Exception:
            pass
        if (old_repo / "models" / "pipeline.py").exists():
            log(f"Copying official HiDream repo from previous install: {old_repo}")
            if OFFICIAL_REPO_DIR.exists():
                shutil.rmtree(OFFICIAL_REPO_DIR, ignore_errors=True)
            OFFICIAL_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(old_repo, OFFICIAL_REPO_DIR)
            return
    zip_path = DOWNLOADS_DIR / "HiDream-O1-Image-main.zip"
    if not zip_path.exists():
        download_file(OFFICIAL_REPO_ZIP, zip_path)
    tmp_extract = DOWNLOADS_DIR / "_hidream_repo_extract"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract, ignore_errors=True)
    tmp_extract.mkdir(parents=True, exist_ok=True)
    log("Extracting official HiDream repo.")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_extract)
    extracted = tmp_extract / "HiDream-O1-Image-main"
    if not extracted.exists():
        fail("Official repo ZIP extracted, but expected folder was not found.")
    if OFFICIAL_REPO_DIR.exists():
        shutil.rmtree(OFFICIAL_REPO_DIR, ignore_errors=True)
    OFFICIAL_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted), str(OFFICIAL_REPO_DIR))
    shutil.rmtree(tmp_extract, ignore_errors=True)
    log(f"Official HiDream repo installed at: {OFFICIAL_REPO_DIR}")


def patch_pipeline_for_attention(env_python: Path) -> None:
    pipeline = OFFICIAL_REPO_DIR / "models" / "pipeline.py"
    if not pipeline.exists():
        fail(f"Cannot patch pipeline because it does not exist: {pipeline}")
    if flash_attention_import_works(env_python):
        log("FlashAttention import works. Keeping use_flash_attn=True.")
        return
    text = pipeline.read_text(encoding="utf-8")
    if '"use_flash_attn": False' in text:
        log("Pipeline already patched to use_flash_attn=False.")
        return
    if '"use_flash_attn": True' not in text:
        log("Could not find exact use_flash_attn=True text. Skipping no-flash patch.")
        return
    backup = pipeline.with_suffix(".py.bak_before_no_flash_patch")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    pipeline.write_text(text.replace('"use_flash_attn": True', '"use_flash_attn": False'), encoding="utf-8")
    log("Patched official pipeline to use_flash_attn=False because FlashAttention import failed.")


def env_python_code(env_python: Path, code: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return run([str(env_python), "-c", code], env=env)


def has_aria2c() -> bool:
    return find_aria2c() is not None


def find_aria2c() -> Path | None:
    for candidate in ARIA2C_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def should_use_aria2_for_file(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(".safetensors") or lower.endswith(".bin")


def huggingface_resolve_url(repo_id: str, filename: str) -> str:
    safe_name = filename.replace("\\", "/")
    return f"https://huggingface.co/{repo_id}/resolve/main/{safe_name}?download=true"


def download_with_aria2(repo_id: str, filename: str, target_dir: Path) -> None:
    aria2c = find_aria2c()
    if aria2c is None:
        raise RuntimeError("aria2c was requested but no aria2c.exe was found.")
    target_dir.mkdir(parents=True, exist_ok=True)
    url = huggingface_resolve_url(repo_id, filename)
    target_path = target_dir / Path(filename).name
    log(f"Using aria2 for large file download: {filename}")
    run([
        str(aria2c),
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--continue=true",
        "--file-allocation=none",
        "--max-connection-per-server=8",
        "--split=8",
        "--min-split-size=16M",
        "--summary-interval=1",
        "--console-log-level=warn",
        "--dir", str(target_dir),
        "--out", target_path.name,
        url,
    ])


def download_hf_file(env_python: Path, repo_id: str, filename: str, target_dir: Path) -> None:
    if has_aria2c() and should_use_aria2_for_file(filename):
        return download_with_aria2(repo_id, filename, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    code = f'''
import os
from huggingface_hub import hf_hub_download
os.environ["HF_HOME"] = r"{LOCAL_HF_HOME}"
os.environ["HF_HUB_CACHE"] = r"{LOCAL_CACHE_DIR}"
os.environ["TRANSFORMERS_CACHE"] = r"{LOCAL_CACHE_DIR}"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
path = hf_hub_download(repo_id={repo_id!r}, filename={filename!r}, local_dir=r"{target_dir}", cache_dir=r"{LOCAL_CACHE_DIR}")
print(path)
'''
    env = os.environ.copy()
    env["HF_HOME"] = str(LOCAL_HF_HOME)
    env["HF_HUB_CACHE"] = str(LOCAL_CACHE_DIR)
    env["TRANSFORMERS_CACHE"] = str(LOCAL_CACHE_DIR)
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["HF_HUB_DISABLE_XET"] = "1"
    env_python_code(env_python, code, env=env)


def get_hf_file_plan(env_python: Path, repo_id: str) -> list[str]:
    code = f'''
import json
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info(repo_id={repo_id!r}, files_metadata=True)
files = []
for s in info.siblings:
    files.append({{"name": s.rfilename, "size": getattr(s, "size", 0) or 0}})
print(json.dumps(files))
'''
    result = subprocess.run([str(env_python), "-c", code], capture_output=True, text=True, check=True)
    files = json.loads(result.stdout.strip())
    names = {f["name"] for f in files}
    size_by_name = {f["name"]: int(f.get("size") or 0) for f in files}
    selected: list[str] = [name for name in SMALL_HF_FILES if name in names]
    if "model.safetensors" in names:
        selected.append("model.safetensors")
    elif "model.safetensors.index.json" in names:
        selected.append("model.safetensors.index.json")
        tmp = DOWNLOADS_DIR / f"{repo_id.replace('/', '_')}_model.safetensors.index.json"
        download_hf_file(env_python, repo_id, "model.safetensors.index.json", tmp.parent)
        data = json.loads(tmp.read_text(encoding="utf-8"))
        selected.extend(sorted(set(data.get("weight_map", {}).values())))
    else:
        candidates = sorted(n for n in names if n.endswith(".safetensors"))
        if len(candidates) == 1:
            selected.append(candidates[0])
        else:
            fail("No unambiguous safetensors file found in the Hugging Face repo.")
    deduped = []
    seen = set()
    for item in selected:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    total = sum(size_by_name.get(n, 0) for n in deduped)
    log("Selective model download plan:")
    for n in deduped:
        size = size_by_name.get(n, 0)
        log(f"  {n}" + (f" ({size / (1024**3):.2f} GiB)" if size else ""))
    if total:
        log(f"Total selected download size: {total / (1024**3):.2f} GiB")
    return deduped


def cleanup_wrong_model_downloads(model_dir: Path, allowed_files: list[str]) -> None:
    allowed = {Path(f).as_posix() for f in allowed_files}
    if not model_dir.exists():
        return
    for p in model_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(model_dir).as_posix()
            if p.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth", ".ckpt"} and rel not in allowed:
                p.unlink(missing_ok=True)
                log(f"Removed old/unwanted large file: {rel}")


def download_model(env_python: Path, model_key: str) -> None:
    info = MODEL_INFO[model_key]
    repo_id = info["repo_id"]
    model_dir = MODELS_ROOT / info["folder"]
    log(f"Preparing selective model download: {info['label']} ({repo_id})")
    model_dir.mkdir(parents=True, exist_ok=True)
    plan = get_hf_file_plan(env_python, repo_id)
    cleanup_wrong_model_downloads(model_dir, plan)
    for filename in plan:
        target = model_dir / filename
        if target.exists() and target.stat().st_size > 0:
            log(f"Already exists, skipping: {filename}")
            continue
        log(f"Downloading selected file: {filename}")
        download_hf_file(env_python, repo_id, filename, model_dir)
    cleanup_wrong_model_downloads(model_dir, plan)
    log(f"Model folder ready: {model_dir}")


RUNNER_CODE = '#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\nimport json\nimport os\nimport random\nimport sys\nfrom pathlib import Path\n\nimport torch\nfrom transformers import AutoProcessor\n\nTHIS_DIR = Path(__file__).resolve().parent\nREPO_DIR = THIS_DIR / "HiDream-O1-Image"\n\nMODEL_MAP = {\n    "base": {\n        "label": "Base / Full BF16",\n        "folder": "HiDream-O1-Image-BF16",\n        "default_steps": 50,\n        "default_guidance": 5.0,\n        "default_shift": 3.0,\n        "default_scheduler": "flash",\n        "default_timesteps": "none",\n    },\n    "dev": {\n        "label": "Dev BF16",\n        "folder": "HiDream-O1-Image-Dev-BF16",\n        "default_steps": 28,\n        "default_guidance": 0.0,\n        "default_shift": 1.0,\n        "default_scheduler": "flash",\n        "default_timesteps": "dev",\n    },\n}\n\nsys.path.insert(0, str(REPO_DIR))\n\nfrom models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration\nfrom models.pipeline import generate_image, DEFAULT_TIMESTEPS\nfrom inference import add_special_tokens, get_tokenizer\n\n\nFALLBACK_QWEN_CHAT_TEMPLATE = """{% for message in messages %}{% if message[\'role\'] == \'user\' %}<|im_start|>user\n{{ message[\'content\'] }}<|im_end|>\n{% elif message[\'role\'] == \'assistant\' %}<|im_start|>assistant\n{{ message[\'content\'] }}<|im_end|>\n{% elif message[\'role\'] == \'system\' %}<|im_start|>system\n{{ message[\'content\'] }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"""\n\n\ndef read_chat_template_from_folder(folder: Path) -> str | None:\n    for candidate in [folder / "chat_template.jinja", folder / "chat_template.json", folder / "tokenizer_config.json", folder / "processor_config.json"]:\n        if not candidate.exists():\n            continue\n        try:\n            raw = candidate.read_text(encoding="utf-8")\n            if candidate.suffix.lower() == ".json":\n                data = json.loads(raw)\n                template = data.get("chat_template") if isinstance(data, dict) else data\n            else:\n                template = raw\n            if isinstance(template, str) and template.strip():\n                print(f"[HiDream runner] Loaded chat template from: {candidate}")\n                return template\n        except Exception as exc:\n            print(f"[HiDream runner] Could not read chat template {candidate}: {exc}")\n    return None\n\n\ndef ensure_processor_chat_template(processor, model_dir: Path):\n    tokenizer = get_tokenizer(processor)\n    template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)\n    if template:\n        print("[HiDream runner] Processor already has a chat template.")\n    if not template:\n        template = read_chat_template_from_folder(model_dir)\n    if not template:\n        template = FALLBACK_QWEN_CHAT_TEMPLATE\n        print("[HiDream runner] WARNING: no model chat template found; using built-in fallback template.")\n    processor.chat_template = template\n    try:\n        tokenizer.chat_template = template\n    except Exception:\n        pass\n    return tokenizer\n\n\ndef parse_timesteps(mode: str):\n    if mode == "dev":\n        return DEFAULT_TIMESTEPS\n    return None\n\n\ndef main() -> None:\n    parser = argparse.ArgumentParser("HiDream BF16 model runner")\n    parser.add_argument("--model_key", choices=["base", "dev"], default="base")\n    parser.add_argument("--prompt", type=str, default="A realistic photo of an elegant woman sitting alone in a warm restaurant, natural skin texture, correct hands, realistic lighting, shallow depth of field.")\n    parser.add_argument("--ref_images", nargs="*", default=[], help="Optional reference image(s) for edit/reference workflow.")\n    parser.add_argument("--output_image", type=str, default=str(THIS_DIR / "results" / "hidream.png"))\n    parser.add_argument("--height", type=int, default=720)\n    parser.add_argument("--width", type=int, default=1280)\n    parser.add_argument("--seed", type=int, default=-1, help="Use -1 for random seed.")\n    parser.add_argument("--steps", type=int, default=None)\n    parser.add_argument("--guidance_scale", type=float, default=None)\n    parser.add_argument("--shift", type=float, default=None)\n    parser.add_argument("--scheduler_name", choices=["default", "flash"], default=None)\n    parser.add_argument("--timesteps", choices=["none", "dev"], default=None)\n    parser.add_argument("--keep_original_aspect", action="store_true")\n    parser.add_argument("--noise_scale_start", type=float, default=7.5)\n    parser.add_argument("--noise_scale_end", type=float, default=7.5)\n    parser.add_argument("--noise_clip_std", type=float, default=2.5)\n    args = parser.parse_args()\n\n    info = MODEL_MAP[args.model_key]\n    model_dir = THIS_DIR / info["folder"]\n\n    if args.steps is None:\n        args.steps = info["default_steps"]\n    if args.guidance_scale is None:\n        args.guidance_scale = info["default_guidance"]\n    if args.shift is None:\n        args.shift = info["default_shift"]\n    if args.scheduler_name is None:\n        args.scheduler_name = info["default_scheduler"]\n    if args.timesteps is None:\n        args.timesteps = info["default_timesteps"]\n\n    if not model_dir.exists():\n        raise RuntimeError(f"Selected model is not installed: {model_dir}. Run install.bat again and select {info[\'label\']}.")\n\n    if not torch.cuda.is_available():\n        raise RuntimeError("CUDA is required. CPU mode is not useful for this model.")\n\n    if args.seed < 0:\n        args.seed = random.randint(0, 2**31 - 1)\n\n    os.makedirs(os.path.dirname(os.path.abspath(args.output_image)), exist_ok=True)\n    timesteps_list = parse_timesteps(args.timesteps)\n\n    print("[HiDream runner] Torch:", torch.__version__, "CUDA:", torch.version.cuda)\n    print("[HiDream runner] Repo:", REPO_DIR)\n    print("[HiDream runner] Model:", model_dir)\n    print("[HiDream runner] Model key:", args.model_key, "-", info["label"])\n    print("[HiDream runner] Processor source:", model_dir)\n    print("[HiDream runner] Output:", args.output_image)\n    print("[HiDream runner] Active generation settings:")\n    print(f"  size: {args.width}x{args.height}")\n    print(f"  seed: {args.seed}")\n    print(f"  steps: {args.steps}")\n    print(f"  guidance_scale: {args.guidance_scale}")\n    print(f"  shift: {args.shift}")\n    if args.scheduler_name == "flash":\n        print("  scheduler_name: flash (FlashFlowMatchEulerDiscreteScheduler / Euler path)")\n    else:\n        print("  scheduler_name: default (FlowUniPCMultistepScheduler / UniPC path)")\n    print(f"  timesteps: {args.timesteps}" + (f" ({len(timesteps_list)})" if timesteps_list else ""))\n    print(f"  noise_scale_start: {args.noise_scale_start}")\n    print(f"  noise_scale_end: {args.noise_scale_end}")\n    print(f"  noise_clip_std: {args.noise_clip_std}")\n\n    print("[HiDream runner] Loading processor and BF16 model...")\n    processor = AutoProcessor.from_pretrained(str(model_dir))\n    tokenizer = ensure_processor_chat_template(processor, model_dir)\n    add_special_tokens(tokenizer)\n\n    model = Qwen3VLForConditionalGeneration.from_pretrained(\n        str(model_dir),\n        dtype=torch.bfloat16,\n        device_map="cuda",\n    ).eval()\n\n    print("[HiDream runner] Generating...")\n    image = generate_image(\n        model=model,\n        processor=processor,\n        prompt=args.prompt,\n        ref_image_paths=args.ref_images,\n        height=args.height,\n        width=args.width,\n        num_inference_steps=args.steps,\n        guidance_scale=args.guidance_scale,\n        shift=args.shift,\n        timesteps_list=timesteps_list,\n        scheduler_name=args.scheduler_name,\n        seed=args.seed,\n        noise_scale_start=args.noise_scale_start,\n        noise_scale_end=args.noise_scale_end,\n        noise_clip_std=args.noise_clip_std,\n        keep_original_aspect=args.keep_original_aspect,\n    )\n\n    image.save(args.output_image)\n    print("[HiDream runner] Saved:", args.output_image)\n\n\nif __name__ == "__main__":\n    main()\n'


def write_runner_files() -> None:
    runner_py = MODELS_ROOT / "run_hidream.py"
    runner_bat = MODELS_ROOT / "run_hidream.bat"
    runner_source = '''#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
HELPER_CLI = THIS_DIR.parents[1] / "helpers" / "hidream_cli.py"


def main() -> None:
    if not HELPER_CLI.exists():
        raise RuntimeError(f"HiDream helper CLI was not found: {HELPER_CLI}")
    sys.path.insert(0, str(HELPER_CLI.parent))
    runpy.run_path(str(HELPER_CLI), run_name="__main__")


if __name__ == "__main__":
    main()
'''
    runner_py.write_text(runner_source, encoding="utf-8")
    runner_bat.write_text(r'''@echo off
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "HERE=%~dp0"
set "ROOT=%HERE%..\.."
set "ENV_PY=%ROOT%\environments\.hidream_dev\python.exe"
if not exist "%ENV_PY%" (
    echo HiDream environment was not found:
    echo %ENV_PY%
    echo Run install.bat first.
    pause
    exit /b 1
)
"%ENV_PY%" "%HERE%run_hidream.py" %*
pause
''', encoding="utf-8")
    log(f"Wrote runner: {runner_py}")
    log(f"Wrote runner BAT: {runner_bat}")


def verify_install(env_python: Path, selected_models: list[str]) -> None:
    code = r'''
import importlib
import torch
import transformers
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
for name in ["triton", "flash_attn", "sageattention", "PySide6"]:
    try:
        importlib.import_module(name)
        print(f"{name}: OK")
    except Exception as exc:
        print(f"{name}: not available ({type(exc).__name__}: {exc})")
        if name != "PySide6":
            raise
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in torch.")
'''
    env_python_code(env_python, code)
    expected = [OFFICIAL_REPO_DIR / "models" / "pipeline.py", MODELS_ROOT / "run_hidream.py"]
    for key in selected_models:
        expected.append(MODELS_ROOT / MODEL_INFO[key]["folder"] / "config.json")
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        fail("Install verification found missing files:\n" + "\n".join(missing))
    log("Install verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser("HiDream installer")
    parser.add_argument(
        "--models",
        default=None,
        help="Model selection for non-interactive use: base, dev, dev_2604, base_fp8, dev_fp8, both, both_fp8, all.",
    )
    parser.add_argument("--no-prompt", action="store_true", help="Do not show the interactive model selection menu.")
    args = parser.parse_args()

    if os.name != "nt":
        log("Warning: this installer was written for Windows paths, but it will still try to run.")
    if args.models is not None:
        selected_models = parse_model_selection(args.models)
    elif args.no_prompt:
        fail("--no-prompt requires --models so the installer knows what to download.")
    else:
        selected_models = choose_models()
    log(f"Root folder: {ROOT}")
    ensure_dirs()
    conda = ensure_miniconda()
    env_python, env_already_existed = ensure_env(conda)
    if env_already_existed and existing_env_stack_ok(env_python):
        log("Skipping pip dependency installation because the existing env is already ready.")
    else:
        ensure_python_packages(env_python)
    download_official_repo()
    patch_pipeline_for_attention(env_python)
    for model_key in selected_models:
        download_model(env_python, model_key)
    write_runner_files()
    verify_install(env_python, selected_models)
    print("")
    print("============================================================")
    print(" HiDream install finished")
    print("============================================================")
    print(f"Environment: {ENV_DIR}")
    print(f"Repo:        {OFFICIAL_REPO_DIR}")
    print(f"Models root: {MODELS_ROOT}")
    for model_key in selected_models:
        print(f"Installed:   {MODEL_INFO[model_key]['label']} -> {MODELS_ROOT / MODEL_INFO[model_key]['folder']}")
    print("")
    print("Run UI:")
    print(f'  "{ROOT / "launch_hidream_ui.bat"}"')
    print("")
    print("If you later select a missing model in the UI, run install.bat again and choose that model.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Cancelled by user.", code=130)
