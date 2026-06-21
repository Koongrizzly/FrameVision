#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

APP_NAME = "FrameVision dots.tts installer"
REPO_URL = "https://github.com/rednote-hilab/dots.tts.git"
REPO_ZIP_URL = "https://github.com/rednote-hilab/dots.tts/archive/refs/heads/main.zip"
HF_MODEL_ID = "rednote-hilab/dots.tts-mf"

# dots.tts currently supports Python >=3.10,<3.13.
# Keep 3.10 first because the tested Windows pynini wheel is cp310.
PYTHON_VERSION = "3.10"
PYNINI_VERSION = "2.1.6.post1"
PYNINI_WHEEL_BASE = "https://github.com/billwuhao/pynini-windows-wheels/releases/download/v2.1.6.post1"
PYNINI_WHEEL_TAGS = {"cp310", "cp311", "cp312", "cp313"}

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[2]
ENV_DIR = ROOT / "environments" / ".dots_tts"
MODELS_DIR = ROOT / "models" / "dots_tts"
REPO_DIR = MODELS_DIR / "repo"
MODEL_DIR = MODELS_DIR / "dots.tts-mf"
TEMP_DIR = ROOT / "temp" / "dots_tts_installer"
PIP_CACHE_DIR = TEMP_DIR / "pip_cache"
PIP_TMP_DIR = TEMP_DIR / "pip_tmp"
WHEEL_DIR = TEMP_DIR / "wheels"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "dottstts_install.log"
MANIFEST_FILE = MODELS_DIR / "install_manifest.json"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dirs() -> None:
    for folder in (MODELS_DIR, TEMP_DIR, PIP_CACHE_DIR, PIP_TMP_DIR, WHEEL_DIR, LOG_DIR, ENV_DIR.parent):
        folder.mkdir(parents=True, exist_ok=True)


def log(message: str = "") -> None:
    line = f"[{now()}] {message}" if message else ""
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def quoted(parts: Iterable[object]) -> str:
    return subprocess.list2cmdline([str(p) for p in parts])


def _base_env(extra: Optional[dict] = None) -> dict:
    merged_env = os.environ.copy()
    merged_env.update({
        "PYTHONUTF8": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CACHE_DIR": str(PIP_CACHE_DIR),
        "TMP": str(PIP_TMP_DIR),
        "TEMP": str(PIP_TMP_DIR),
        "HF_HOME": str(MODELS_DIR / "_hf_home"),
        "HF_HUB_CACHE": str(MODELS_DIR / "_hf_cache"),
        "HF_XET_CACHE": str(MODELS_DIR / "_hf_xet_cache"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    })
    if extra:
        merged_env.update(extra)
    return merged_env


def run(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[dict] = None, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + quoted(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=_base_env(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        text = line.rstrip("\n")
        print(text, flush=True)
        try:
            with LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except Exception:
            pass
    rc = proc.wait()
    result = subprocess.CompletedProcess(cmd, rc)
    if check and rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}: {quoted(cmd)}")
    return result


def run_conda(conda_cmd: Path, args: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    if os.name == "nt" and conda_cmd.suffix.lower() in {".bat", ".cmd"}:
        cmd_line = "call " + quoted([conda_cmd, *args])
        return run(["cmd.exe", "/d", "/s", "/c", cmd_line], check=check)
    return run([str(conda_cmd), *args], check=check)


def find_conda() -> Optional[Path]:
    candidates: List[Path] = []
    for key in ("CONDA_EXE", "MAMBA_EXE"):
        value = os.environ.get(key)
        if value:
            p = Path(value)
            candidates.append(p)
            if p.name.lower() == "conda.exe":
                candidates.append(p.parent.parent / "condabin" / "conda.bat")

    for name in ("conda", "conda.bat", "mamba", "micromamba"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))

    home = Path.home()
    program_data = Path(os.environ.get("ProgramData", r"C:\ProgramData"))
    local_app_data = Path(os.environ.get("LocalAppData", home / "AppData" / "Local"))
    candidates.extend([
        program_data / "miniconda3" / "condabin" / "conda.bat",
        program_data / "anaconda3" / "condabin" / "conda.bat",
        home / "miniconda3" / "condabin" / "conda.bat",
        home / "anaconda3" / "condabin" / "conda.bat",
        local_app_data / "miniconda3" / "condabin" / "conda.bat",
        local_app_data / "anaconda3" / "condabin" / "conda.bat",
        ROOT / "miniconda3" / "condabin" / "conda.bat",
        ROOT / "conda" / "condabin" / "conda.bat",
    ])

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate.expanduser()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    return None


def env_python() -> Path:
    if os.name == "nt":
        candidates = [ENV_DIR / "python.exe", ENV_DIR / "Scripts" / "python.exe"]
    else:
        candidates = [ENV_DIR / "bin" / "python", ENV_DIR / "python"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def ensure_conda_env(conda_cmd: Path) -> None:
    py = env_python()
    if py.exists():
        log(f"Conda env already exists: {ENV_DIR}")
        return
    log(f"Creating conda env: {ENV_DIR}")
    run_conda(conda_cmd, ["create", "-y", "-p", str(ENV_DIR), f"python={PYTHON_VERSION}", "pip"])


def pip_install(args: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    py = env_python()
    return run([str(py), "-m", "pip", *args], check=check)


def pip_install_binary(args: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    base = ["install", "--upgrade", "--no-warn-script-location", "--only-binary=:all:"]
    return pip_install(base + args, check=check)


def install_helper_ui_dependencies() -> None:
    """Optional helper UI packages for running helpers/dotstts.py from this conda env.

    pyqtgraph is only the waveform/plot widget. It still needs a Qt binding.
    FrameVision's main Python normally already has PySide6, but this separate
    dots.tts conda env does not, so install PySide6 here for standalone tests.
    """
    py = env_python()
    already = run(
        [
            str(py),
            "-c",
            "import PySide6; import pyqtgraph; print('PySide6 import: ok'); print('pyqtgraph import: ok')",
        ],
        check=False,
    )
    if already.returncode == 0:
        log("PySide6 + pyqtgraph already installed for standalone helper UI.")
        return

    log("Installing optional helper UI dependencies: PySide6 + pyqtgraph")
    pip_install_binary(["PySide6", "pyqtgraph"])
    run(
        [
            str(py),
            "-c",
            "import PySide6; import pyqtgraph; print('PySide6 import: ok'); print('pyqtgraph import: ok')",
        ]
    )


def py_eval(code: str) -> str:
    py = env_python()
    proc = subprocess.run(
        [str(py), "-c", code],
        env=_base_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout.strip())
    return proc.stdout.strip()


def python_cp_tag() -> str:
    return py_eval("import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")


def site_packages_dir() -> Path:
    text = py_eval("import sysconfig; print(sysconfig.get_paths()['purelib'])")
    return Path(text)


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    log(f"Saved: {target}")


def ensure_git_repo() -> None:
    if (REPO_DIR / "pyproject.toml").exists():
        log(f"Repo already exists: {REPO_DIR}")
        git = shutil.which("git")
        if git and (REPO_DIR / ".git").exists():
            log("Updating existing repo with git pull --ff-only")
            run([git, "pull", "--ff-only"], cwd=REPO_DIR, check=False)
        return

    if REPO_DIR.exists() and any(REPO_DIR.iterdir()):
        backup = MODELS_DIR / f"repo_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log(f"Existing repo folder is incomplete. Moving it to: {backup}")
        shutil.move(str(REPO_DIR), str(backup))

    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if git:
        log("Cloning dots.tts repo with git")
        run([git, "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
        return

    log("Git not found. Downloading repo ZIP fallback.")
    zip_path = TEMP_DIR / "dots_tts_main.zip"
    if zip_path.exists():
        zip_path.unlink()
    download_file(REPO_ZIP_URL, zip_path)
    extract_dir = TEMP_DIR / "repo_zip"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    roots = [p for p in extract_dir.iterdir() if p.is_dir()]
    if not roots:
        raise RuntimeError("Downloaded repo ZIP did not contain a source folder.")
    shutil.move(str(roots[0]), str(REPO_DIR))


def install_pynini(conda_cmd: Path) -> None:
    py = env_python()
    already = run([str(py), "-c", "import pynini; print('pynini import: ok')"], check=False)
    if already.returncode == 0:
        log("pynini already installed.")
        return

    tag = python_cp_tag()
    if tag not in PYNINI_WHEEL_TAGS:
        raise RuntimeError(
            f"No known prebuilt Windows pynini wheel mapping for {tag}. "
            "Use Python 3.10, 3.11, or 3.12 for this installer."
        )

    filename = f"pynini-{PYNINI_VERSION}-{tag}-{tag}-win_amd64.whl"
    url = f"{PYNINI_WHEEL_BASE}/{filename}"
    wheel_path = WHEEL_DIR / filename

    log(f"Installing prebuilt pynini wheel for {tag}: {filename}")
    if not wheel_path.exists() or wheel_path.stat().st_size < 1024:
        try:
            download_file(url, wheel_path)
        except Exception as exc:
            log(f"Direct pynini wheel download failed: {exc}")
            try:
                wheel_path.unlink()
            except FileNotFoundError:
                pass

    if wheel_path.exists():
        result = pip_install(["install", "--upgrade", "--no-warn-script-location", str(wheel_path)], check=False)
        if result.returncode == 0:
            run([str(py), "-c", "import pynini; print('pynini import: ok')"])
            return
        log("Prebuilt pynini wheel install failed. Trying official conda-forge fallback.")

    result = run_conda(conda_cmd, ["install", "-y", "-p", str(ENV_DIR), "-c", "conda-forge", "pynini"], check=False)
    if result.returncode == 0:
        run([str(py), "-c", "import pynini; print('pynini import: ok')"])
        return

    raise RuntimeError(
        "Could not install pynini from a prebuilt Windows wheel or conda-forge. "
        "Installer stopped on purpose because source wheel builds are disabled."
    )


def add_repo_pth() -> None:
    sp = site_packages_dir()
    sp.mkdir(parents=True, exist_ok=True)
    src_dir = REPO_DIR / "src"
    pth = sp / "framevision_dots_tts_repo.pth"
    pth.write_text(str(REPO_DIR) + "\n" + str(src_dir) + "\n", encoding="utf-8")
    log(f"Added repo + src to env via .pth file: {pth}")


def create_cli_shim() -> Path:
    scripts_dir = ENV_DIR / ("Scripts" if os.name == "nt" else "bin")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shim_py = scripts_dir / "framevision_dots_tts_cli.py"
    shim_py.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"repo = Path({str(REPO_DIR)!r})\n"
        "src = repo / 'src'\n"
        "for path in (repo, src):\n"
        "    path_str = str(path)\n"
        "    if path.exists() and path_str not in sys.path:\n"
        "        sys.path.insert(0, path_str)\n"
        "from dots_tts.cli import main\n"
        "raise SystemExit(main())\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        cli = scripts_dir / "dots.tts.bat"
        cli.write_text(
            "@echo off\r\n"
            "setlocal\r\n"
            "\"%~dp0..\\python.exe\" \"%~dp0framevision_dots_tts_cli.py\" %*\r\n",
            encoding="utf-8",
        )
    else:
        cli = scripts_dir / "dots.tts"
        cli.write_text(
            "#!/usr/bin/env bash\n"
            "DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"\n"
            "\"$DIR/../bin/python\" \"$DIR/framevision_dots_tts_cli.py\" \"$@\"\n",
            encoding="utf-8",
        )
        try:
            cli.chmod(0o755)
        except Exception:
            pass
    log(f"Created dots.tts CLI shim: {cli}")
    return cli


def install_python_packages(conda_cmd: Path) -> None:
    py = env_python()
    if not py.exists():
        raise RuntimeError(f"Env Python not found: {py}")

    log("Upgrading pip/setuptools/wheel without source builds")
    pip_install_binary(["pip", "setuptools", "wheel"])

    log("Installing PyTorch CUDA 12.8 stack")
    cuda_result = pip_install([
        "install", "--upgrade", "--no-warn-script-location", "--only-binary=:all:",
        "torch==2.8.0+cu128",
        "torchaudio==2.8.0+cu128",
        "--index-url", "https://download.pytorch.org/whl/cu128",
        "--extra-index-url", "https://pypi.org/simple",
    ], check=False)
    if cuda_result.returncode != 0:
        log("CUDA wheel install failed. Falling back to standard torch/torchaudio 2.8.0 wheels.")
        pip_install_binary(["torch==2.8.0", "torchaudio==2.8.0"])

    constraints = REPO_DIR / "constraints" / "recommended.txt"
    common_args: List[str] = []
    if constraints.exists():
        common_args += ["-c", str(constraints)]

    runtime_packages = [
        "transformers>=4.57.0",
        "huggingface-hub",
        "hf_xet",
        "loguru",
        "langcodes[data]",
        "einops",
        "librosa>=0.11.0",
        "soundfile>=0.13.1",
        "numpy>=2.2.6",
        "pydantic>=2.12.5,<3",
        "PyYAML>=6.0.3",
        "safetensors>=0.8.0rc0",
        "torchdiffeq",
        "tqdm",
        "lingua-language-detector",
        # WeTextProcessing is installed with --no-deps to avoid pynini source builds.
        # This is its other runtime dependency and is safe/pure Python.
        "importlib_resources",
    ]
    log("Installing dots.tts runtime dependencies without Gradio/server extras and without source builds")
    pip_install_binary(common_args + runtime_packages)

    install_pynini(conda_cmd)

    log("Installing WeTextProcessing without pulling pynini/source-build dependencies")
    pip_install_binary(common_args + ["--no-deps", "WeTextProcessing"])

    install_helper_ui_dependencies()

    log("Adding dots.tts local repo to Python path without editable wheel build")
    add_repo_pth()
    create_cli_shim()


def download_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    py = env_python()
    downloader = TEMP_DIR / "download_dots_tts_model.py"
    downloader.write_text(
        "from pathlib import Path\n"
        "from huggingface_hub import snapshot_download\n"
        f"target = Path(r'''{MODEL_DIR}''')\n"
        "target.mkdir(parents=True, exist_ok=True)\n"
        "print('Downloading/verifying Hugging Face snapshot into:', target)\n"
        "try:\n"
        f"    snapshot_download(repo_id={HF_MODEL_ID!r}, local_dir=str(target), local_dir_use_symlinks=False)\n"
        "except TypeError:\n"
        f"    snapshot_download(repo_id={HF_MODEL_ID!r}, local_dir=str(target))\n"
        "print('Model snapshot ready:', target)\n",
        encoding="utf-8",
    )

    for attempt in range(1, 4):
        try:
            log(f"Downloading/verifying model snapshot ({HF_MODEL_ID}), attempt {attempt}/3")
            run([str(py), str(downloader)])
            return
        except Exception as exc:
            log(f"Model download attempt {attempt} failed: {exc}")
            if attempt >= 3:
                raise
            time.sleep(5)


def verify_install() -> None:
    py = env_python()
    log("Running import/GPU verification")
    verify = (
        "import sys\n"
        "from pathlib import Path\n"
        f"repo = Path({str(REPO_DIR)!r})\n"
        "src = repo / 'src'\n"
        "for path in (repo, src):\n"
        "    path_str = str(path)\n"
        "    if path.exists() and path_str not in sys.path:\n"
        "        sys.path.insert(0, path_str)\n"
        "import torch\n"
        "import pynini\n"
        "import importlib_resources\n"
        "import PySide6\n"
        "import pyqtgraph\n"
        "import tn\n"
        "import itn\n"
        "import dots_tts\n"
        "from dots_tts.cli import main as dots_tts_cli_main\n"
        "print('dots_tts import: ok')\n"
        "print('dots_tts.cli import: ok')\n"
        "print('pynini import: ok')\n"
        "print('importlib_resources import: ok')\n"
        "print('PySide6 import: ok')\n"
        "print('pyqtgraph import: ok')\n"
        "print('WeTextProcessing tn/itn import: ok')\n"
        "print('torch:', torch.__version__)\n"
        "print('cuda available:', torch.cuda.is_available())\n"
        "print('cuda device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\n"
    )
    run([str(py), "-c", verify])


def write_manifest() -> None:
    cli_path = ENV_DIR / ("Scripts/dots.tts.bat" if os.name == "nt" else "bin/dots.tts")
    data = {
        "installer": APP_NAME,
        "installed_at": now(),
        "root": str(ROOT),
        "environment": str(ENV_DIR),
        "repo": str(REPO_DIR),
        "model_id": HF_MODEL_ID,
        "model_dir": str(MODEL_DIR),
        "cli": str(cli_path),
        "log": str(LOG_FILE),
        "pip_cache": str(PIP_CACHE_DIR),
        "pip_tmp": str(PIP_TMP_DIR),
        "notes": [
            "Installed dots.tts runtime dependencies only; Gradio/server extras are intentionally skipped.",
            "Source wheel builds are intentionally disabled with --only-binary=:all: where possible.",
            "pynini is installed from a prebuilt Windows wheel first, with conda-forge as prebuilt fallback.",
            "WeTextProcessing is installed with --no-deps so it cannot trigger a pynini source build.",
            "importlib_resources is installed explicitly because it is the other WeTextProcessing runtime dependency.",
            "dots.tts repo and repo/src are added with a .pth file instead of pip install -e, avoiding editable wheel creation.",
            "A small dots.tts CLI shim is created because skipping pip install -e also skips normal entry point creation.",
            "PySide6 and pyqtgraph are installed as optional helper UI dependencies for standalone helpers/dotstts.py waveform previews from this env.",
            "If helpers/dotstts.py is imported into the main FrameVision app, pyqtgraph must also exist in the main FrameVision Python environment; FrameVision usually already provides PySide6.",
            "Model and Hugging Face caches are local to models/dots_tts.",
            "Installer logs are stored in root logs/.",
        ],
    }
    MANIFEST_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {MANIFEST_FILE}")


def main() -> int:
    ensure_dirs()
    if LOG_FILE.exists():
        try:
            LOG_FILE.unlink()
        except Exception:
            pass

    log("============================================================")
    log(APP_NAME)
    log("============================================================")
    log(f"Root:      {ROOT}")
    log(f"Env:       {ENV_DIR}")
    log(f"Repo:      {REPO_DIR}")
    log(f"Model:     {MODEL_DIR}")
    log(f"Log:       {LOG_FILE}")
    log(f"Pip cache: {PIP_CACHE_DIR}")
    log(f"Pip temp:  {PIP_TMP_DIR}")
    log("")

    conda_cmd = find_conda()
    if not conda_cmd:
        raise RuntimeError("Conda was not found. Install Miniconda/Anaconda first, or make sure conda.bat is available.")
    log(f"Conda:    {conda_cmd}")

    ensure_conda_env(conda_cmd)
    ensure_git_repo()
    install_python_packages(conda_cmd)
    download_model()
    verify_install()
    write_manifest()

    log("")
    log("DONE: dots.tts installer finished successfully.")
    log("Use the env Python/CLI from environments/.dots_tts and the local model folder under models/dots_tts.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Cancelled by user.")
        raise SystemExit(130)
    except Exception as exc:
        log("")
        log("ERROR: " + str(exc))
        log(f"Check log: {LOG_FILE}")
        raise SystemExit(1)
