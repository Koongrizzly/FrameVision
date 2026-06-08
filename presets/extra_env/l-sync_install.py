#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
import zipfile
from pathlib import Path
from venv import EnvBuilder

# Portable LatentSync 1.5 installer for FrameVision
# Target layout:
#   <framevision>/presets/extra_env/l-sync_install.py
#   <framevision>/environments/.l-sync/
#   <framevision>/models/l-sync/
#
# What it does:
#   - creates a local venv under environments/.l-sync
#   - downloads the official ByteDance LatentSync repo under models/l-sync/repo
#   - installs official dependencies, keeping caches/temp inside FrameVision
#   - prefers a prebuilt insightface wheel to avoid fragile source builds
#   - downloads LatentSync 1.5 weights into models/l-sync/checkpoints
#   - creates small Windows launcher .bat files under models/l-sync

PYTHON_VERSION_HINT = "3.10"
REPO_ZIP_URL = "https://github.com/bytedance/LatentSync/archive/refs/heads/main.zip"
HF_REPO_ID = "ByteDance/LatentSync-1.5"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"

# Official repo pins as of the current public requirements.txt.
TORCH_PACKAGES = [
    "torch==2.5.1",
    "torchvision==0.20.1",
]

BASE_PACKAGES = [
    "diffusers==0.32.2",
    "transformers==4.48.0",
    "decord==0.6.0",
    "accelerate==0.26.1",
    "einops==0.7.0",
    "omegaconf==2.3.0",
    "opencv-python==4.9.0.80",
    "mediapipe==0.10.11",
    "python_speech_features==0.6",
    "librosa==0.10.1",
    "scenedetect==0.6.1",
    "ffmpeg-python==0.2.0",
    "imageio==2.31.1",
    "imageio-ffmpeg==0.5.1",
    "lpips==0.1.4",
    "face-alignment==1.4.1",
    "gradio==5.24.0",
    "huggingface-hub==0.30.2",
    "numpy==1.26.4",
    "scipy==1.11.4",
    "scikit-image==0.22.0",
    "kornia==0.8.0",
    "onnx==1.17.0",
    "onnxruntime-gpu==1.21.0",
    "DeepCache==0.1.1",
    "soundfile",
    "packaging",
    "safetensors",
]

# Keep these pinned and force-reinstall them after insightface on Windows.
# This avoids ABI mismatches such as:
#   ValueError: numpy.dtype size changed
# when pip pulls a bad numpy/scikit-image combination into an existing env.
BINARY_COMPAT_PACKAGES = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "scikit-image==0.22.0",
]

# On Windows, avoid source builds entirely. Community-maintained prebuilt wheels are
# commonly used for InsightFace when the official pip path falls back to MSVC/Cython.
INSIGHTFACE_PACKAGE = "insightface==0.7.3"
INSIGHTFACE_WHEEL_URLS = {
    "3.10": "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl",
    "3.11": "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl",
    "3.12": "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl",
    "3.13": "https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp313-cp313-win_amd64.whl",
}
OPTIONAL_STABLE_SYNCNET = False


def log(msg: str) -> None:
    print(f"[l-sync] {msg}")


def fail(msg: str, code: int = 1) -> None:
    print(f"[l-sync][ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def script_path() -> Path:
    return Path(__file__).resolve()


def detect_root() -> Path:
    here = script_path()
    # Expected: <root>/presets/extra_env/l-sync_install.py
    if here.parent.name == "extra_env" and here.parent.parent.name == "presets":
        return here.parent.parent.parent

    # Fallback: walk upwards and look for presets + models + environments folders.
    for parent in [here.parent, *here.parents]:
        if (parent / "presets").exists() and (parent / "models").exists() and (parent / "environments").exists():
            return parent

    # Last resort: assume 2 levels up from script.
    guessed = here.parent.parent.parent
    log(f"Could not prove FrameVision root from layout; falling back to: {guessed}")
    return guessed


def env_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def env_pip(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    shown = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    log(shown)
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def merged_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra)
    return env


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "script_dir": root / "presets" / "extra_env",
        "venv": root / "environments" / ".l-sync",
        "models_root": root / "models" / "l-sync",
        "repo_root": root / "models" / "l-sync" / "repo",
        "repo_nested": root / "models" / "l-sync" / "repo" / "LatentSync-main",
        "repo_dir": root / "models" / "l-sync" / "repo",
        "checkpoints": root / "models" / "l-sync" / "checkpoints",
        "cache_root": root / "models" / "l-sync" / ".cache",
        "hf_home": root / "models" / "l-sync" / ".cache" / "huggingface",
        "pip_cache": root / "models" / "l-sync" / ".cache" / "pip",
        "tmp": root / "models" / "l-sync" / ".tmp",
        "logs": root / "models" / "l-sync" / "logs",
    }
    for p in paths.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return paths


def repo_runtime_dir(paths: dict[str, Path]) -> Path:
    repo_root = paths["repo_root"]
    nested = paths.get("repo_nested", repo_root / "LatentSync-main")

    if (repo_root / "scripts" / "inference.py").exists():
        return repo_root
    if nested.exists() and (nested / "scripts" / "inference.py").exists():
        return nested
    return repo_root


def portable_env(paths: dict[str, Path], venv_dir: Path | None = None) -> dict[str, str]:
    env = {
        "PIP_CACHE_DIR": str(paths["pip_cache"]),
        "HF_HOME": str(paths["hf_home"]),
        "HUGGINGFACE_HUB_CACHE": str(paths["hf_home"] / "hub"),
        "TRANSFORMERS_CACHE": str(paths["hf_home"] / "transformers"),
        "XDG_CACHE_HOME": str(paths["cache_root"]),
        "TEMP": str(paths["tmp"]),
        "TMP": str(paths["tmp"]),
        "TMPDIR": str(paths["tmp"]),
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if venv_dir is not None:
        scripts = venv_dir / ("Scripts" if os.name == "nt" else "bin")
        env["VIRTUAL_ENV"] = str(venv_dir)
        env["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
    return merged_env(env)


def create_or_reuse_venv(venv_dir: Path) -> None:
    py = env_python(venv_dir)
    if py.exists():
        log(f"Reusing environment: {venv_dir}")
        return
    log(f"Creating venv at: {venv_dir}")
    EnvBuilder(with_pip=True, clear=False, symlinks=False, upgrade=False).create(str(venv_dir))


def upgrade_bootstrap(venv_dir: Path, env: dict[str, str]) -> None:
    py = env_python(venv_dir)
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], env=env)


def download_and_extract_repo(paths: dict[str, Path]) -> None:
    repo_root = paths["repo_root"]
    nested_dir = paths["repo_nested"]
    runtime_repo = repo_runtime_dir(paths)
    marker = repo_root / ".framevision_repo_ready"
    if marker.exists() and (runtime_repo / "scripts" / "inference.py").exists():
        log(f"Repo already present: {runtime_repo}")
        return

    tmp_zip = paths["tmp"] / "LatentSync-main.zip"
    if tmp_zip.exists():
        tmp_zip.unlink()

    log("Downloading official ByteDance LatentSync GitHub repo archive...")
    urllib.request.urlretrieve(REPO_ZIP_URL, tmp_zip)

    if repo_root.exists():
        shutil.rmtree(repo_root, ignore_errors=True)
    repo_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(tmp_zip, "r") as zf:
        zf.extractall(repo_root)

    if nested_dir.exists():
        for item in list(nested_dir.iterdir()):
            dest = repo_root / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        shutil.rmtree(nested_dir, ignore_errors=True)

    runtime_repo = repo_runtime_dir(paths)
    if not (runtime_repo / "scripts" / "inference.py").exists():
        fail(f"Repo extract failed; expected scripts/inference.py under: {runtime_repo}")

    marker.write_text("ok\n", encoding="utf-8")
    log(f"Repo ready: {runtime_repo}")


def install_torch(venv_dir: Path, env: dict[str, str]) -> None:
    pip = env_pip(venv_dir)
    cmd = [str(pip), "install", "--index-url", TORCH_INDEX_URL, *TORCH_PACKAGES]
    run(cmd, env=env)


def install_base_packages(venv_dir: Path, env: dict[str, str]) -> None:
    pip = env_pip(venv_dir)
    run([str(pip), "install", *BASE_PACKAGES], env=env)


def repair_binary_compat_packages(venv_dir: Path, env: dict[str, str]) -> None:
    py = env_python(venv_dir)
    run(
        [
            str(py),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            *BINARY_COMPAT_PACKAGES,
        ],
        env=env,
    )


def install_insightface(venv_dir: Path, paths: dict[str, Path], env: dict[str, str]) -> None:
    py = env_python(venv_dir)
    pip = env_pip(venv_dir)

    code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    proc = subprocess.run([str(py), "-c", code], check=True, capture_output=True, text=True, env=env)
    py_mm = proc.stdout.strip()

    if os.name == "nt":
        wheel_url = INSIGHTFACE_WHEEL_URLS.get(py_mm)
        if not wheel_url:
            fail(f"No prebuilt Windows insightface wheel URL is configured for Python {py_mm}. Use Python 3.10-3.13.")

        wheel_name = wheel_url.rsplit("/", 1)[-1]
        wheel_path = paths["tmp"] / wheel_name
        if not wheel_path.exists() or wheel_path.stat().st_size == 0:
            log(f"Downloading prebuilt insightface wheel for Python {py_mm}...")
            urllib.request.urlretrieve(wheel_url, wheel_path)

        # Local wheel install only. Do not fall back to a source build on Windows.
        run([str(py), "-m", "pip", "install", str(wheel_path)], env=env)
        log("insightface installed from a downloaded Windows wheel.")
        return

    # Non-Windows fallback: wheel-only from pip, no source build.
    run([str(pip), "install", "--only-binary=insightface", INSIGHTFACE_PACKAGE], env=env)


def verify_python_version(venv_dir: Path, env: dict[str, str]) -> None:
    py = env_python(venv_dir)
    code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    proc = subprocess.run([str(py), "-c", code], check=True, capture_output=True, text=True, env=env)
    version = proc.stdout.strip()
    log(f"Environment Python: {version}")
    if not version.startswith("3.10."):
        log("Warning: official setup uses Python 3.10.13. Other versions may work, but 3.10 is the safe target.")


def download_hf_assets(venv_dir: Path, paths: dict[str, Path], env: dict[str, str]) -> None:
    py = env_python(venv_dir)
    code = textwrap.dedent(
        f"""
        import shutil
        from huggingface_hub import hf_hub_download
        from pathlib import Path

        target = Path(r"{paths['checkpoints']}")
        repo_ckpt = Path(r"{repo_runtime_dir(paths)}") / "checkpoints"
        repo_whisper = repo_ckpt / "whisper"
        target.mkdir(parents=True, exist_ok=True)
        (target / "whisper").mkdir(parents=True, exist_ok=True)
        repo_whisper.mkdir(parents=True, exist_ok=True)

        files = [
            ("latentsync_unet.pt", target / "latentsync_unet.pt"),
            ("whisper/tiny.pt", target / "whisper" / "tiny.pt"),
        ]
        if {str(OPTIONAL_STABLE_SYNCNET)}:
            files.append(("stable_syncnet.pt", target / "stable_syncnet.pt"))

        for filename, final_path in files:
            final_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {{filename}} -> {{final_path}}")
            downloaded = Path(
                hf_hub_download(
                    repo_id="{HF_REPO_ID}",
                    filename=filename,
                    local_dir=str(target),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            )
            if downloaded.resolve() != final_path.resolve():
                shutil.copy2(downloaded, final_path)

        whisper_src = target / "whisper" / "tiny.pt"
        whisper_dst = repo_whisper / "tiny.pt"
        if not whisper_src.exists():
            raise RuntimeError(f"Whisper checkpoint missing after download: {{whisper_src}}")
        shutil.copy2(whisper_src, whisper_dst)
        print(f"Mirrored whisper checkpoint into repo runtime path: {{whisper_dst}}")
        print("HF download complete")
        """
    )
    run([str(py), "-c", code], env=env)


def ensure_repo_runtime_links(paths: dict[str, Path]) -> None:
    repo_ckpt = repo_runtime_dir(paths) / "checkpoints"
    repo_whisper = repo_ckpt / "whisper"
    repo_whisper.mkdir(parents=True, exist_ok=True)

    src_tiny = paths["checkpoints"] / "whisper" / "tiny.pt"
    dst_tiny = repo_whisper / "tiny.pt"
    if not src_tiny.exists():
        fail(f"Whisper checkpoint missing: {src_tiny}")
    shutil.copy2(src_tiny, dst_tiny)
    log(f"Whisper tiny checkpoint ready in repo runtime path: {dst_tiny}")


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def create_launchers(paths: dict[str, Path], venv_dir: Path) -> None:
    if os.name != "nt":
        return

    repo_dir = repo_runtime_dir(paths)
    ckpt_dir = paths["checkpoints"]
    venv_py = env_python(venv_dir)
    cache_env = textwrap.dedent(
        f"""
        set "HF_HOME={paths['hf_home']}"
        set "HUGGINGFACE_HUB_CACHE={paths['hf_home'] / 'hub'}"
        set "TRANSFORMERS_CACHE={paths['hf_home'] / 'transformers'}"
        set "XDG_CACHE_HOME={paths['cache_root']}"
        set "PIP_CACHE_DIR={paths['pip_cache']}"
        set "TEMP={paths['tmp']}"
        set "TMP={paths['tmp']}"
        set "PYTHONUTF8=1"
        """
    ).strip()

    run_gradio = textwrap.dedent(
        f"""
        @echo off
        setlocal
        {cache_env}
        cd /d "{repo_dir}"
        set "PYTHONPATH={repo_dir}"
        "{venv_py}" gradio_app.py
        endlocal
        """
    ).strip() + "\n"

    run_inference = textwrap.dedent(
        f"""
        @echo off
        setlocal
        {cache_env}
        if "%~3"=="" (
            echo Usage: %~nx0 input_video.mp4 input_audio.wav output_video.mp4
            exit /b 1
        )
        cd /d "{repo_dir}"
        set "PYTHONPATH={repo_dir}"
        "{venv_py}" -m scripts.inference ^
          --unet_config_path "configs/unet/stage2.yaml" ^
          --inference_ckpt_path "{ckpt_dir / 'latentsync_unet.pt'}" ^
          --inference_steps 20 ^
          --guidance_scale 1.5 ^
          --enable_deepcache ^
          --video_path "%~1" ^
          --audio_path "%~2" ^
          --video_out_path "%~3"
        endlocal
        """
    ).strip() + "\n"

    info_txt = textwrap.dedent(
        f"""
        LatentSync 1.5 portable install for FrameVision

        Main folders
        - Env: {venv_dir}
        - Repo: {repo_dir}
        - Checkpoints: {ckpt_dir}
        - HF cache: {paths['hf_home']}

        Windows launchers
        - run_gradio.bat
        - run_inference_1_5.bat

        Notes
        - This installer uses the official repo code from GitHub main.
        - It downloads the safer LatentSync 1.5 weights, not 1.6.
        - Torch is pinned to the official project requirement line (2.5.1 / torchvision 0.20.1 with CUDA 12.1 wheels).
        - On Windows the installer uses a downloaded prebuilt insightface wheel and does not fall back to source builds.
        - NumPy/SciPy/scikit-image are force-reinstalled to a known-good pinned set after insightface.
        """
    ).strip() + "\n"

    write_file(paths["models_root"] / "run_gradio.bat", run_gradio)
    write_file(paths["models_root"] / "run_inference_1_5.bat", run_inference)
    write_file(paths["models_root"] / "INSTALL_INFO.txt", info_txt)


def write_manifest(paths: dict[str, Path], venv_dir: Path) -> None:
    manifest = {
        "name": "LatentSync 1.5 portable install",
        "root": str(paths["root"]),
        "venv": str(venv_dir),
        "repo": str(repo_runtime_dir(paths)),
        "checkpoints": str(paths["checkpoints"]),
        "hf_repo": HF_REPO_ID,
        "torch_index_url": TORCH_INDEX_URL,
        "torch_packages": TORCH_PACKAGES,
        "base_packages": BASE_PACKAGES,
        "binary_compat_packages": BINARY_COMPAT_PACKAGES,
        "insightface": INSIGHTFACE_PACKAGE,
        "optional_stable_syncnet": OPTIONAL_STABLE_SYNCNET,
    }
    write_file(paths["models_root"] / "install_manifest.json", json.dumps(manifest, indent=2))


def check_basic_tools() -> None:
    if sys.version_info < (3, 10):
        fail("Run this installer with Python 3.10+ so the created environment stays close to the official setup.")


def main() -> None:
    check_basic_tools()
    root = detect_root()
    paths = ensure_dirs(root)

    log(f"FrameVision root: {root}")
    log(f"Target env: {paths['venv']}")
    log(f"Target repo/models: {paths['models_root']}")

    create_or_reuse_venv(paths["venv"])
    env = portable_env(paths, paths["venv"])

    verify_python_version(paths["venv"], env)
    upgrade_bootstrap(paths["venv"], env)
    download_and_extract_repo(paths)

    log("Installing official CUDA-enabled PyTorch pair...")
    install_torch(paths["venv"], env)

    log("Installing remaining Python packages...")
    install_base_packages(paths["venv"], env)

    log("Installing insightface with wheel-first strategy...")
    install_insightface(paths["venv"], paths, env)

    log("Repairing pinned NumPy/SciPy/scikit-image compatibility...")
    repair_binary_compat_packages(paths["venv"], env)

    log("Downloading LatentSync 1.5 checkpoints from Hugging Face...")
    download_hf_assets(paths["venv"], paths, env)
    ensure_repo_runtime_links(paths)

    create_launchers(paths, paths["venv"])
    write_manifest(paths, paths["venv"])

    log("Install complete.")
    log(f"Repo: {repo_runtime_dir(paths)}")
    log(f"Checkpoints: {paths['checkpoints']}")
    if os.name == "nt":
        log(f"Try: {paths['models_root'] / 'run_gradio.bat'}")
        log(f"Or:  {paths['models_root'] / 'run_inference_1_5.bat'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        fail(f"Command failed with exit code {e.returncode}: {e.cmd}")
    except KeyboardInterrupt:
        fail("Interrupted by user.", code=130)
