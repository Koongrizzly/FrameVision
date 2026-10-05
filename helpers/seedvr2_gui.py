from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PySide6.QtCore import QEvent, QProcess, QProcessEnvironment, QSettings, Qt, QThread, Signal
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QAbstractSpinBox, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QSpinBox, QDoubleSpinBox, QTextEdit, QVBoxLayout, QWidget, QTabWidget, QSlider
)
from PySide6.QtCore import QUrl

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    HAVE_QT_MULTIMEDIA = True
except Exception:
    QAudioOutput = QMediaPlayer = QVideoWidget = None
    HAVE_QT_MULTIMEDIA = False


APP_NAME = "SeedVR2 PySide6"


# Community/safetensor models used by the working SeedVR2 backend.
# DiT source was supplied earlier for the BF16 build; the VAE is the community
# FP16 VAE used by the same backend.
SAFETENSOR_FILES = (
    ("szwagros/SeedVR2-3B-bf16", "seedvr2_ema_3b_bf16.safetensors"),
    ("numz/SeedVR2_comfyUI", "ema_vae_fp16.safetensors"),
)

# Runtime packages taken from the official SeedVR requirements.txt, excluding torch/torchvision.
# We deliberately do not auto-reinstall the CUDA torch stack because that can break an existing
# working NVIDIA environment. Torch and torchvision are validated separately.
OFFICIAL_RUNTIME_PACKAGES = (
    # requirements.txt
    "einops==0.7.0",
    "omegaconf==2.3.0",
    "opencv-python==4.9.0.80",
    "diffusers==0.29.1",
    "rotary-embedding-torch==0.5.3",
    "transformers==4.38.2",
    "mediapy==1.2.0",
    # Runtime packages present in the official environment.yml and needed by
    # torchvision video I/O / model loading on a clean Windows environment.
    "av==12.0.0",
    "numpy==1.24.4",
    "pillow==10.3.0",
    "tqdm==4.66.4",
    "safetensors==0.4.3",
    "imageio==2.34.0",
    "imageio-ffmpeg==0.5.1",
    "psutil==6.0.0",
    "packaging==22.0",
)

# Apex and flash_attn are intentionally NOT required here. The official Linux
# environment uses them, but this Windows frontend provides portable fallbacks:
# Apex fused RMS/LayerNorm -> regular RMSNorm/LayerNorm, flash_attn -> torch SDPA.
RUNTIME_IMPORTS = (
    "torch", "torchvision", "einops", "omegaconf", "cv2",
    "diffusers", "rotary_embedding_torch", "transformers", "mediapy", "tqdm",
    "av", "numpy", "PIL", "safetensors", "imageio", "psutil", "packaging",
)

# Standalone community backend install target.  Everything lives relative to this
# GUI folder so the complete SeedVR2 folder can be moved without breaking paths.
COMMUNITY_REPO_GIT = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
COMMUNITY_REPO_ZIP = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/archive/refs/heads/main.zip"
COMMUNITY_TORCH = "2.8.0"
COMMUNITY_TORCHVISION = "0.23.0"
COMMUNITY_TORCHAUDIO = "2.8.0"
COMMUNITY_TRANSFORMERS = "4.55.4"
COMMUNITY_DIFFUSERS = "0.35.1"
COMMUNITY_ACCELERATE = "1.10.1"
COMMUNITY_TRITON = "triton-windows>=3.4,<3.5"
COMMUNITY_SAGE_WHEEL = "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0%2Bcu128torch2.8.0.post3-cp39-abi3-win_amd64.whl"
COMMUNITY_FLASH_WHEEL_CP311 = "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2%2Bcu128torch2.8-cp311-cp311-win_amd64.whl"


class SafetensorModelDownloadWorker(QThread):
    progress = Signal(int, str)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, destination: Path, parent=None):
        super().__init__(parent)
        self.destination = Path(destination)
        self.file_specs = {name: repo for repo, name in SAFETENSOR_FILES}
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._done_bytes = 0
        self._total_bytes = 0
        self._last_emit = 0.0

    def cancel(self):
        self._cancel.set()

    def _check_cancel(self):
        if self._cancel.is_set():
            raise RuntimeError("Download cancelled by user.")

    def _url(self, name: str) -> str:
        repo = self.file_specs[name]
        return f"https://huggingface.co/{repo}/resolve/main/{name}?download=true"

    def _remote_size(self, name: str) -> int:
        req = urllib.request.Request(self._url(name), method="HEAD", headers={"User-Agent": "FrameVision-SeedVR2/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            for key in ("X-Linked-Size", "x-linked-size", "Content-Length", "content-length"):
                v = r.headers.get(key)
                if v:
                    try:
                        return int(v)
                    except Exception:
                        pass
        return 0

    def _supports_ranges(self, name: str) -> bool:
        req = urllib.request.Request(self._url(name), headers={"Range": "bytes=0-0", "User-Agent": "FrameVision-SeedVR2/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(getattr(r, "status", 200)) == 206 or bool(r.headers.get("Content-Range"))

    def _add_progress(self, n: int, label: str):
        with self._lock:
            self._done_bytes += int(n)
            now = time.monotonic()
            if now - self._last_emit < 0.08 and self._done_bytes < self._total_bytes:
                return
            self._last_emit = now
            pct = int((self._done_bytes * 100) / self._total_bytes) if self._total_bytes else 0
        self.progress.emit(max(0, min(100, pct)), label)

    def _download_range(self, name: str, start: int, end: int, part_path: Path, label: str):
        self._check_cancel()
        expected = end - start + 1
        have = part_path.stat().st_size if part_path.exists() else 0
        if have > expected:
            part_path.unlink(missing_ok=True); have = 0
        if have == expected:
            return
        req_start = start + have
        headers = {"Range": f"bytes={req_start}-{end}", "User-Agent": "FrameVision-SeedVR2/1.0"}
        req = urllib.request.Request(self._url(name), headers=headers)
        mode = "ab" if have else "wb"
        with urllib.request.urlopen(req, timeout=120) as r, part_path.open(mode) as f:
            if int(getattr(r, "status", 200)) != 206 and not r.headers.get("Content-Range"):
                raise RuntimeError(f"Server did not honor range request for {name}.")
            while True:
                self._check_cancel()
                data = r.read(4 * 1024 * 1024)
                if not data:
                    break
                f.write(data)
                self._add_progress(len(data), label)
        if part_path.stat().st_size != expected:
            raise RuntimeError(f"Incomplete part for {name}: got {part_path.stat().st_size}, expected {expected} bytes")

    def _download_single(self, name: str, size: int, target: Path):
        temp = target.with_suffix(target.suffix + ".download")
        have = temp.stat().st_size if temp.exists() else 0
        if size and have > size:
            temp.unlink(missing_ok=True); have = 0
        if size and have == size:
            temp.replace(target); return
        headers = {"User-Agent": "FrameVision-SeedVR2/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(self._url(name), headers=headers)
        mode = "ab" if have else "wb"
        with urllib.request.urlopen(req, timeout=120) as r:
            if have and int(getattr(r, "status", 200)) != 206 and not r.headers.get("Content-Range"):
                have = 0; mode = "wb"
            with temp.open(mode) as f:
                while True:
                    self._check_cancel()
                    data = r.read(4 * 1024 * 1024)
                    if not data:
                        break
                    f.write(data)
                    self._add_progress(len(data), f"Downloading {name}")
        if size and temp.stat().st_size != size:
            raise RuntimeError(f"Incomplete download for {name}: got {temp.stat().st_size}, expected {size} bytes")
        temp.replace(target)

    def _download_segmented(self, name: str, size: int, target: Path, parts: int = 6):
        if size <= 0 or not self._supports_ranges(name):
            return self._download_single(name, size, target)
        parts = max(2, min(parts, 8))
        span = (size + parts - 1) // parts
        jobs = []
        for idx in range(parts):
            start = idx * span
            if start >= size:
                break
            end = min(size - 1, start + span - 1)
            pp = target.with_name(target.name + f".part{idx:02d}")
            jobs.append((idx, start, end, pp))
        with ThreadPoolExecutor(max_workers=min(6, len(jobs))) as ex:
            futs = [ex.submit(self._download_range, name, st, en, pp, f"Downloading {name} ({i+1}/{len(jobs)})") for i, st, en, pp in jobs]
            for fut in as_completed(futs):
                fut.result()
        self._check_cancel()
        merged = target.with_suffix(target.suffix + ".download")
        with merged.open("wb") as out:
            for _, _, _, pp in jobs:
                with pp.open("rb") as src:
                    while True:
                        b = src.read(8 * 1024 * 1024)
                        if not b: break
                        out.write(b)
        if merged.stat().st_size != size:
            raise RuntimeError(f"Merged size mismatch for {name}.")
        merged.replace(target)
        for _, _, _, pp in jobs:
            pp.unlink(missing_ok=True)

    def run(self):
        try:
            self.destination.mkdir(parents=True, exist_ok=True)
            sizes = {}
            self.progress.emit(0, "Checking SeedVR2 safetensor files…")
            for _, name in SAFETENSOR_FILES:
                self._check_cancel()
                sizes[name] = self._remote_size(name)
            self._total_bytes = sum(sizes.values())
            # Count complete existing files toward the global progress.
            for name, size in sizes.items():
                target = self.destination / name
                if target.exists() and size and target.stat().st_size == size:
                    self._done_bytes += size
            if self._total_bytes:
                self.progress.emit(int(self._done_bytes * 100 / self._total_bytes), "Starting parallel downloads…")

            # Download the two large checkpoints with multiple simultaneous HTTP ranges.
            # Small embedding files are downloaded normally.
            for _, name in SAFETENSOR_FILES:
                self._check_cancel()
                target = self.destination / name
                size = sizes[name]
                if target.exists() and size and target.stat().st_size == size:
                    continue
                if size >= 256 * 1024 * 1024:
                    self._download_segmented(name, size, target, parts=6)
                else:
                    self._download_single(name, size, target)
            self.progress.emit(100, "SeedVR2 safetensor files downloaded.")
            self.completed.emit(str(self.destination))
        except Exception as e:
            self.failed.emit(str(e))


class DependencyInstallWorker(QThread):
    progress = Signal(str)
    completed = Signal()
    failed = Signal(str)

    def __init__(self, python_exe: str, parent=None):
        super().__init__(parent)
        self.python_exe = str(python_exe)

    def run(self):
        try:
            cmd = [self.python_exe, "-m", "pip", "install", "--upgrade", *OFFICIAL_RUNTIME_PACKAGES]
            self.progress.emit("RUN: " + " ".join(cmd))
            p = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1, universal_newlines=True
            )
            assert p.stdout is not None
            for line in p.stdout:
                line = line.rstrip("\r\n")
                if line:
                    self.progress.emit(line)
            rc = p.wait()
            if rc != 0:
                raise RuntimeError(f"pip exited with code {rc}")
            self.completed.emit()
        except Exception as e:
            self.failed.emit(str(e))


class StandaloneInstallWorker(QThread):
    """Create/repair a relocatable SeedVR2 runtime beside this GUI.

    Layout:
      <app>/environments/.seedvr2
      <app>/models/seedvr2/ComfyUI-SeedVR2_VideoUpscaler

    The repository deliberately lives under models/seedvr2, not under an installer
    or presets folder.  Re-running this worker repairs the existing installation.
    """
    progress = Signal(str)
    completed = Signal(str, str)  # python.exe, community repo
    failed = Signal(str)

    def __init__(self, app_root: Path, parent=None):
        super().__init__(parent)
        self.app_root = Path(app_root)

    def _log(self, msg: str):
        self.progress.emit(str(msg))

    def _run(self, cmd, cwd: Path | None = None, check: bool = True):
        cmd = [str(x) for x in cmd]
        self._log("RUN: " + " ".join(shlex.quote(x) for x in cmd))
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        p = subprocess.Popen(
            cmd, cwd=str(cwd) if cwd else None, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1, universal_newlines=True,
        )
        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\r\n")
            if line:
                self._log(line)
        rc = p.wait()
        if check and rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")
        return rc

    @staticmethod
    def _env_python(env_dir: Path) -> Path:
        if os.name == "nt":
            return env_dir / "Scripts" / "python.exe"
        return env_dir / "bin" / "python"

    def _find_python311(self):
        # Prefer Python 3.11 because the known Windows FlashAttention wheel is cp311.
        if sys.version_info[:2] == (3, 11):
            return [sys.executable]
        if os.name == "nt" and shutil.which("py"):
            try:
                cp = subprocess.run(["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                                    capture_output=True, text=True, timeout=20)
                if cp.returncode == 0 and cp.stdout.strip():
                    return ["py", "-3.11"]
            except Exception:
                pass
        # Fall back to the interpreter running the GUI. Core SeedVR2 can still work;
        # optional acceleration wheels are installed only when compatible.
        return [sys.executable]

    def _download_repo_zip(self, repo_root: Path):
        import tempfile, zipfile
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="seedvr2_repo_") as td:
            td = Path(td)
            zp = td / "seedvr2.zip"
            self._log("Downloading community SeedVR2 backend ZIP...")
            req = urllib.request.Request(COMMUNITY_REPO_ZIP, headers={"User-Agent": "SeedVR2-Standalone/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, zp.open("wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
            with zipfile.ZipFile(zp, "r") as z:
                z.extractall(td)
            extracted = td / "ComfyUI-SeedVR2_VideoUpscaler-main"
            if not extracted.exists():
                candidates = [x for x in td.iterdir() if x.is_dir() and x.name.startswith("ComfyUI-SeedVR2_VideoUpscaler")]
                if not candidates:
                    raise RuntimeError("Downloaded SeedVR2 repository ZIP did not contain the expected folder.")
                extracted = candidates[0]
            if repo_root.exists():
                shutil.rmtree(repo_root, ignore_errors=True)
            shutil.copytree(extracted, repo_root)

    def _ensure_repo(self, repo_root: Path):
        cli = repo_root / "inference_cli.py"
        if cli.exists():
            self._log(f"Community backend already present: {repo_root}")
            # Repair/update when it is a git checkout; never fail installation merely
            # because the user is offline or the repository has local changes.
            if (repo_root / ".git").exists() and shutil.which("git"):
                self._run(["git", "-C", str(repo_root), "pull", "--ff-only"], check=False)
            return
        repo_root.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("git"):
            rc = self._run(["git", "clone", "--depth", "1", COMMUNITY_REPO_GIT, str(repo_root)], check=False)
            if rc == 0 and cli.exists():
                return
            if repo_root.exists():
                shutil.rmtree(repo_root, ignore_errors=True)
        self._download_repo_zip(repo_root)
        if not cli.exists():
            raise RuntimeError("Community backend installation finished without inference_cli.py")

    def _pip(self, py: Path, *args, check=True):
        return self._run([str(py), "-m", "pip", *args], check=check)

    def _runtime_status(self, py: Path) -> dict:
        """Inspect the existing venv without modifying it."""
        code = r"""
import json, importlib.util
try:
    import importlib.metadata as md
except Exception:
    md = None

def ver(pkg):
    if md is None:
        return None
    try:
        return md.version(pkg)
    except Exception:
        return None

out = {
    "torch": None, "torchvision": None, "torchaudio": None,
    "cuda": None, "cuda_available": False,
    "transformers": ver("transformers"), "diffusers": ver("diffusers"),
    "accelerate": ver("accelerate"), "huggingface_hub": ver("huggingface-hub"),
    "tokenizers": ver("tokenizers"), "triton_windows": ver("triton-windows"),
    "sageattention": ver("sageattention"), "flash_attn": ver("flash-attn"),
}
try:
    import torch
    out["torch"] = getattr(torch, "__version__", None)
    out["cuda"] = getattr(torch.version, "cuda", None)
    out["cuda_available"] = bool(torch.cuda.is_available())
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
try:
    import torchvision; out["torchvision"] = getattr(torchvision, "__version__", None)
except Exception as e: out["torchvision_error"] = f"{type(e).__name__}: {e}"
try:
    import torchaudio; out["torchaudio"] = getattr(torchaudio, "__version__", None)
except Exception as e: out["torchaudio_error"] = f"{type(e).__name__}: {e}"
mods = ["safetensors", "cv2", "einops", "omegaconf", "peft", "rotary_embedding_torch", "gguf", "psutil"]
out["missing_modules"] = [m for m in mods if importlib.util.find_spec(m) is None]
print(json.dumps(out))
"""
        cp = subprocess.run([str(py), "-c", code], capture_output=True, text=True, errors="replace")
        if cp.returncode != 0:
            return {"probe_error": (cp.stderr or cp.stdout or "runtime probe failed").strip()}
        try:
            return json.loads((cp.stdout or "").strip().splitlines()[-1])
        except Exception:
            return {"probe_error": (cp.stdout or cp.stderr or "invalid runtime probe output").strip()}

    @staticmethod
    def _base_version(v) -> str:
        return str(v or "").split("+", 1)[0]

    def _install_stack(self, py: Path, repo_root: Path):
        # Repair should be cheap when the environment is already healthy. The older
        # installer force-reinstalled the 3.4GB torch wheel and acceleration wheels
        # on every click; only touch components whose probe says they are missing or wrong.
        self._pip(py, "install", "--upgrade", "pip", "setuptools", "wheel")

        has_nvidia = bool(shutil.which("nvidia-smi"))
        torch_tag = "cu128" if has_nvidia else "cpu"
        torch_index = f"https://download.pytorch.org/whl/{torch_tag}"
        status = self._runtime_status(py)
        self._log("Runtime probe: " + json.dumps(status, ensure_ascii=False))

        torch_ok = (
            self._base_version(status.get("torch")) == COMMUNITY_TORCH and
            self._base_version(status.get("torchvision")) == COMMUNITY_TORCHVISION and
            self._base_version(status.get("torchaudio")) == COMMUNITY_TORCHAUDIO and
            ((not has_nvidia) or (bool(status.get("cuda_available")) and str(status.get("cuda") or "").startswith("12.8")))
        )
        if not torch_ok:
            if has_nvidia:
                self._log("PyTorch/CUDA runtime is missing or wrong; repairing Torch 2.8.0 + CUDA 12.8 once...")
            else:
                self._log("PyTorch runtime is missing or wrong; repairing CPU runtime...")
            self._pip(
                py, "install", "--upgrade",
                f"torch=={COMMUNITY_TORCH}", f"torchvision=={COMMUNITY_TORCHVISION}", f"torchaudio=={COMMUNITY_TORCHAUDIO}",
                "--index-url", torch_index,
            )
            status = self._runtime_status(py)
        else:
            self._log("PyTorch runtime already correct; skipping multi-GB Torch reinstall.")

        constraint_path = repo_root.parent / "_seedvr2_constraints.txt"
        torch_suffix = "+cu128" if has_nvidia else ""
        constraint_path.write_text(
            "\n".join([
                f"torch=={COMMUNITY_TORCH}{torch_suffix}",
                f"torchvision=={COMMUNITY_TORCHVISION}{torch_suffix}",
                f"torchaudio=={COMMUNITY_TORCHAUDIO}{torch_suffix}",
                f"transformers=={COMMUNITY_TRANSFORMERS}",
                f"diffusers=={COMMUNITY_DIFFUSERS}",
                f"accelerate=={COMMUNITY_ACCELERATE}",
                "huggingface-hub<1.0",
                "tokenizers<0.22",
                "",
            ]), encoding="utf-8"
        )

        req = repo_root / "requirements.txt"
        req_hash = ""
        if req.exists():
            import hashlib
            req_hash = hashlib.sha256(req.read_bytes()).hexdigest()
        marker = repo_root.parent / "_seedvr2_runtime_state.json"
        marker_data = {}
        try:
            marker_data = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
        except Exception:
            marker_data = {}

        pinned_ok = (
            self._base_version(status.get("transformers")) == COMMUNITY_TRANSFORMERS and
            self._base_version(status.get("diffusers")) == COMMUNITY_DIFFUSERS and
            self._base_version(status.get("accelerate")) == COMMUNITY_ACCELERATE and
            not status.get("missing_modules")
        )
        deps_unchanged = marker_data.get("requirements_sha256") == req_hash and bool(req_hash)

        if pinned_ok and deps_unchanged:
            self._log("Python dependencies already healthy and repository requirements unchanged; skipping dependency reinstall.")
        else:
            core = [
                "numpy", "tqdm", "pillow", "opencv-python", "safetensors", "einops",
                "huggingface_hub", "requests", "packaging", "gguf", "mediapy", "psutil", "PySide6",
                f"transformers=={COMMUNITY_TRANSFORMERS}", f"diffusers=={COMMUNITY_DIFFUSERS}",
                f"accelerate=={COMMUNITY_ACCELERATE}",
            ]
            self._log("Installing only missing/outdated SeedVR2 dependencies...")
            self._pip(
                py, "install", "--upgrade-strategy", "only-if-needed",
                "--extra-index-url", torch_index, "-c", str(constraint_path), *core
            )
            if req.exists():
                self._log("Checking community requirements under pinned CUDA/runtime constraints...")
                self._pip(
                    py, "install", "--upgrade-strategy", "only-if-needed",
                    "--extra-index-url", torch_index, "-c", str(constraint_path), "-r", str(req)
                )

        # Optional acceleration wheels: install only when absent. Do not uninstall
        # and reinstall the same working wheel during every repair.
        status = self._runtime_status(py)
        if os.name == "nt" and has_nvidia:
            triton_ver = str(status.get("triton_windows") or "")
            if not triton_ver.startswith("3.4"):
                self._log("Triton missing/outdated; installing Windows Triton...")
                self._pip(py, "install", "--upgrade", COMMUNITY_TRITON, check=False)
            else:
                self._log(f"Triton already installed ({triton_ver}); skipping.")

            if not status.get("sageattention"):
                self._log("SageAttention missing; installing verified Windows wheel...")
                self._pip(py, "install", "--no-deps", COMMUNITY_SAGE_WHEEL, check=False)
            else:
                self._log(f"SageAttention already installed ({status.get('sageattention')}); skipping.")

            code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            cp = subprocess.run([str(py), "-c", code], capture_output=True, text=True)
            if cp.returncode == 0 and cp.stdout.strip() == "3.11":
                if not status.get("flash_attn"):
                    self._log("FlashAttention missing; installing verified Python 3.11 wheel...")
                    self._pip(py, "install", "--no-deps", COMMUNITY_FLASH_WHEEL_CP311, check=False)
                else:
                    self._log(f"FlashAttention already installed ({status.get('flash_attn')}); skipping.")
            else:
                self._log("FlashAttention wheel skipped: verified prebuilt wheel is for Python 3.11.")

        marker.write_text(json.dumps({
            "requirements_sha256": req_hash,
            "torch": COMMUNITY_TORCH,
            "torchvision": COMMUNITY_TORCHVISION,
            "torchaudio": COMMUNITY_TORCHAUDIO,
            "transformers": COMMUNITY_TRANSFORMERS,
            "diffusers": COMMUNITY_DIFFUSERS,
            "accelerate": COMMUNITY_ACCELERATE,
        }, indent=2), encoding="utf-8")

    def _verify(self, py: Path, repo_root: Path):
        # Fail the installer instead of reporting success when pip has silently
        # replaced the CUDA runtime with a CPU wheel or drifted the pinned stack.
        code = f"""
import json, torch, torchvision, torchaudio, safetensors, cv2, diffusers, transformers, accelerate
info = {{
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'torchaudio': torchaudio.__version__,
    'cuda': torch.version.cuda,
    'cuda_available': bool(torch.cuda.is_available()),
    'transformers': transformers.__version__,
    'diffusers': diffusers.__version__,
    'accelerate': accelerate.__version__,
}}
print(json.dumps(info))
if torch.__version__.split('+', 1)[0] != '{COMMUNITY_TORCH}':
    raise SystemExit('Wrong torch version: ' + torch.__version__)
if torchvision.__version__.split('+', 1)[0] != '{COMMUNITY_TORCHVISION}':
    raise SystemExit('Wrong torchvision version: ' + torchvision.__version__)
if torchaudio.__version__.split('+', 1)[0] != '{COMMUNITY_TORCHAUDIO}':
    raise SystemExit('Wrong torchaudio version: ' + torchaudio.__version__)
if transformers.__version__.split('+', 1)[0] != '{COMMUNITY_TRANSFORMERS}':
    raise SystemExit('Wrong transformers version: ' + transformers.__version__)
if diffusers.__version__.split('+', 1)[0] != '{COMMUNITY_DIFFUSERS}':
    raise SystemExit('Wrong diffusers version: ' + diffusers.__version__)
if accelerate.__version__.split('+', 1)[0] != '{COMMUNITY_ACCELERATE}':
    raise SystemExit('Wrong accelerate version: ' + accelerate.__version__)
if {str(bool(shutil.which('nvidia-smi')))} and (not torch.cuda.is_available() or not torch.version.cuda):
    raise SystemExit('NVIDIA GPU is present but the installed PyTorch runtime has no CUDA support.')
"""
        self._run([str(py), "-c", code])
        self._pip(py, "check")
        if not (repo_root / "inference_cli.py").exists():
            raise RuntimeError("Community repository verification failed: inference_cli.py missing.")
        # Verify the community CLI can at least import and expose help without loading models.
        self._run([str(py), str(repo_root / "inference_cli.py"), "--help"], cwd=repo_root)

    def run(self):
        try:
            app_root = self.app_root
            env_dir = app_root / "environments" / ".seedvr2"
            model_root = app_root / "models" / "seedvr2"
            repo_root = model_root / "ComfyUI-SeedVR2_VideoUpscaler"
            model_root.mkdir(parents=True, exist_ok=True)

            py = self._env_python(env_dir)
            if not py.exists():
                self._log(f"Creating dedicated SeedVR2 environment: {env_dir}")
                launcher = self._find_python311()
                self._run([*launcher, "-m", "venv", str(env_dir)])
            else:
                self._log(f"Reusing existing SeedVR2 environment: {env_dir}")

            if not py.exists():
                raise RuntimeError(f"Virtual environment was not created correctly: {py}")

            self._ensure_repo(repo_root)
            self._install_stack(py, repo_root)
            self._verify(py, repo_root)
            self.completed.emit(str(py), str(repo_root))
        except Exception as e:
            self.failed.emit(str(e))


class SeedVR2Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 900)
        self.proc: QProcess | None = None
        self.download_worker: SafetensorModelDownloadWorker | None = None
        self.dependency_worker: DependencyInstallWorker | None = None
        self.standalone_install_worker: StandaloneInstallWorker | None = None
        # Embedded SeedVR2 uses its own settings namespace.  The previous
        # standalone/legacy integration shared a settings store that could contain
        # values from older widgets with incompatible ranges/meanings.  Do not let
        # those stale values leak into the new tab.
        self.settings = QSettings("FrameVision", "SeedVR2EmbeddedV2")
        self.legacy_settings = QSettings("SeedVR2Standalone", "SeedVR2GUI")
        self._build_ui()
        self._load_settings()
        self._autofill_ckpts()

    def _build_ui(self):
        # Three-tab layout: everyday controls, rarely changed quality/model controls,
        # and machine/log settings. Every tab has its own always-visible vertical
        # scrollbar so controls never compress into each other on smaller displays.
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)
        self._tab_scrolls = []

        def make_tab(title: str):
            scroll = QScrollArea(self.tabs)
            scroll.setWidgetResizable(True)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            body = QWidget()
            body.setMinimumWidth(760)
            layout = QVBoxLayout(body)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(10)
            scroll.setWidget(body)
            self.tabs.addTab(scroll, title)
            self._tab_scrolls.append(scroll)
            return body, layout, scroll

        simple_body, simple, self.simple_scroll = make_tab("Simple")
        advanced_body, advanced, self.advanced_scroll = make_tab("Advanced")
        system_body, system, self.system_scroll = make_tab("System / Log")

        # ---------------- Simple tab ----------------
        intro = QLabel(
            "Everyday SeedVR2 controls. Choose the input/output, model backend, resolution and seed here. "
            "Less frequently changed quality/memory controls are kept out of the way on the other tabs."
        )
        intro.setWordWrap(True)
        simple.addWidget(intro)

        io_group = QGroupBox("Input / output")
        io = QGridLayout(io_group)
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        for r, (name, edit, handler) in enumerate([
            ("Input video / image", self.input_edit, self._pick_input),
            ("Output", self.output_edit, self._pick_output),
        ]):
            io.addWidget(QLabel(name), r, 0)
            io.addWidget(edit, r, 1)
            b = QPushButton("Browse…"); b.clicked.connect(handler); io.addWidget(b, r, 2)
        simple.addWidget(io_group)

        basic = QGroupBox("Basic settings")
        bg = QGridLayout(basic)
        self.backend = QComboBox(); self.backend.addItems(["Community / ComfyUI backend (safetensors/GGUF)", "Original ByteDance backend (.pth)"])
        self.backend.setToolTip("Community mode uses the memory-efficient numz/AInVFX backend. Original mode keeps the ByteDance .pth release available.")
        self.community_resolution = QComboBox()
        for val, label in [(720, "720p"), (768, "768p"), (1080, "1080p"), (1440, "1440p"), (2560, "2560p")]:
            self.community_resolution.addItem(label, val)
        self.community_max_resolution = QSpinBox(); self.community_max_resolution.setRange(0, 8192); self.community_max_resolution.setSingleStep(16); self.community_max_resolution.setValue(0); self.community_max_resolution.setSpecialValueText("none")
        self.width = QSpinBox(); self.width.setRange(256, 8192); self.width.setSingleStep(16); self.width.setValue(1920)
        self.height = QSpinBox(); self.height.setRange(256, 8192); self.height.setSingleStep(16); self.height.setValue(1080)
        self.seed = QSpinBox(); self.seed.setRange(-1, 2_147_483_647); self.seed.setValue(-1); self.seed.setSpecialValueText("random")
        self.fps = QDoubleSpinBox(); self.fps.setRange(0.0, 240.0); self.fps.setDecimals(3); self.fps.setValue(0.0); self.fps.setSpecialValueText("source")
        basic_items = [
            ("Backend", self.backend), ("Target short-side resolution", self.community_resolution),
            ("Maximum edge", self.community_max_resolution), ("Seed", self.seed),
            ("Official width", self.width), ("Official height", self.height), ("Output FPS", self.fps),
        ]
        for i, (label, widget) in enumerate(basic_items):
            row, col = divmod(i, 3)
            cell = QVBoxLayout(); cell.addWidget(QLabel(label)); cell.addWidget(widget); bg.addLayout(cell, row, col)
        simple.addWidget(basic)

        run_group = QGroupBox("Run")
        run_layout = QHBoxLayout(run_group)
        self.validate_btn = QPushButton("Validate")
        self.run_btn = QPushButton("Run SeedVR2")
        self.queue_btn = QPushButton("Add to queue")
        self.queue_btn.setToolTip(
            "Queue this SeedVR2 render with exactly the settings shown in this tab. "
            "The queue worker runs the SeedVR2 command directly and does not apply the normal Upscale codec/FPS pipeline."
        )
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.setEnabled(False)
        self.open_btn = QPushButton("Open output folder")
        self.validate_btn.clicked.connect(lambda: self._validate(show=True))
        self.run_btn.clicked.connect(self.run)
        self.queue_btn.clicked.connect(self.queue_run)
        self.cancel_btn.clicked.connect(self.cancel)
        self.open_btn.clicked.connect(self.open_output_folder)
        for b in (self.validate_btn, self.run_btn, self.queue_btn, self.cancel_btn, self.open_btn): run_layout.addWidget(b)
        simple.addWidget(run_group)

        preview_group = QGroupBox("Finished result preview")
        pv = QVBoxLayout(preview_group)
        self.preview_status = QLabel("The finished output will load here automatically after a successful run.")
        self.preview_status.setWordWrap(True)
        pv.addWidget(self.preview_status)
        if HAVE_QT_MULTIMEDIA:
            self.preview_video = QVideoWidget()
            self.preview_video.setMinimumHeight(360)
            self.preview_player = QMediaPlayer(self)
            self.preview_audio = QAudioOutput(self)
            self.preview_player.setAudioOutput(self.preview_audio)
            self.preview_player.setVideoOutput(self.preview_video)
            pv.addWidget(self.preview_video)
            controls = QHBoxLayout()
            self.preview_play_btn = QPushButton("Play / Pause")
            self.preview_restart_btn = QPushButton("Restart")
            self.preview_open_btn = QPushButton("Open in default player")
            self.preview_play_btn.clicked.connect(self._toggle_preview_play)
            self.preview_restart_btn.clicked.connect(self._restart_preview)
            self.preview_open_btn.clicked.connect(self._open_preview_external)
            controls.addWidget(self.preview_play_btn); controls.addWidget(self.preview_restart_btn); controls.addWidget(self.preview_open_btn)
            pv.addLayout(controls)
        else:
            self.preview_video = None; self.preview_player = None; self.preview_audio = None
            self.preview_play_btn = None; self.preview_restart_btn = None; self.preview_open_btn = QPushButton("Open result in default player")
            self.preview_open_btn.clicked.connect(self._open_preview_external)
            pv.addWidget(self.preview_open_btn)
            self.preview_status.setText("Qt Multimedia is unavailable in this Python environment. The result can still be opened in the default media player.")
        simple.addWidget(preview_group, 1)

        # ---------------- Advanced tab ----------------
        model_group = QGroupBox("Models and backend paths")
        pf = QGridLayout(model_group)
        self.repo_edit = QLineEdit()
        self.community_repo_edit = QLineEdit(str(self._default_community_repo()))
        self.dit_edit = QLineEdit()
        self.vae_edit = QLineEdit()
        rows = [
            ("Official SeedVR repo", self.repo_edit, self._pick_repo),
            ("Community SeedVR2 repo", self.community_repo_edit, self._pick_community_repo),
            ("DiT checkpoint", self.dit_edit, self._pick_dit),
            ("VAE checkpoint", self.vae_edit, self._pick_vae),
        ]
        for r, (name, edit, handler) in enumerate(rows):
            pf.addWidget(QLabel(name), r, 0); pf.addWidget(edit, r, 1)
            b = QPushButton("Browse…"); b.clicked.connect(handler); pf.addWidget(b, r, 2)
        advanced.addWidget(model_group)

        dl_group = QGroupBox("SeedVR2 safetensor model files")
        dl = QGridLayout(dl_group)
        self.model_dir_label = QLabel(str(self._official_model_dir())); self.model_dir_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.download_btn = QPushButton("Download SeedVR2 safetensor models")
        self.download_cancel_btn = QPushButton("Cancel download"); self.download_cancel_btn.setEnabled(False)
        self.download_progress = QProgressBar(); self.download_progress.setRange(0, 100); self.download_progress.setValue(0)
        self.download_status = QLabel("Downloads the BF16 SeedVR2 3B DiT and FP16 VAE safetensors used by the community backend. Existing complete files are skipped."); self.download_status.setWordWrap(True)
        self.download_btn.clicked.connect(self.download_official_models); self.download_cancel_btn.clicked.connect(self.cancel_model_download)
        dl.addWidget(QLabel("Destination"), 0, 0); dl.addWidget(self.model_dir_label, 0, 1, 1, 3)
        dl.addWidget(self.download_btn, 1, 0, 1, 2); dl.addWidget(self.download_cancel_btn, 1, 2)
        dl.addWidget(self.download_progress, 2, 0, 1, 4); dl.addWidget(self.download_status, 3, 0, 1, 4)
        advanced.addWidget(dl_group)

        community = QGroupBox("Community quality / temporal settings")
        cg = QGridLayout(community)
        self.batch_size = QComboBox()
        for val in range(1, 82, 4):
            self.batch_size.addItem(str(val), val)
        self.chunk_size = QSpinBox(); self.chunk_size.setRange(0, 5000); self.chunk_size.setValue(330)
        self.uniform_batch = QCheckBox("Uniform batch size"); self.uniform_batch.setChecked(True)
        self.temporal_overlap = QSpinBox(); self.temporal_overlap.setRange(0, 16); self.temporal_overlap.setValue(3)
        self.prepend_frames = QSpinBox(); self.prepend_frames.setRange(0, 32); self.prepend_frames.setValue(0)
        self.color_correction = QComboBox(); self.color_correction.addItems(["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"])
        self.vae_encode_tiled = QCheckBox("VAE encode tiled"); self.vae_encode_tiled.setChecked(True)
        self.vae_encode_tile_size = QSpinBox(); self.vae_encode_tile_size.setRange(256, 4096); self.vae_encode_tile_size.setSingleStep(64); self.vae_encode_tile_size.setValue(1024)
        self.vae_encode_overlap = QSpinBox(); self.vae_encode_overlap.setRange(0, 1024); self.vae_encode_overlap.setSingleStep(16); self.vae_encode_overlap.setValue(128)
        self.vae_decode_tiled = QCheckBox("VAE decode tiled"); self.vae_decode_tiled.setChecked(True)
        self.vae_decode_tile_size = QSpinBox(); self.vae_decode_tile_size.setRange(256, 4096); self.vae_decode_tile_size.setSingleStep(64); self.vae_decode_tile_size.setValue(768)
        self.vae_decode_overlap = QSpinBox(); self.vae_decode_overlap.setRange(0, 1024); self.vae_decode_overlap.setSingleStep(16); self.vae_decode_overlap.setValue(128)
        citems = [
            ("Batch size (4n+1)", self.batch_size), ("Chunk size", self.chunk_size), ("Temporal overlap", self.temporal_overlap),
            ("Prepend frames", self.prepend_frames), ("Color correction", self.color_correction),
            ("Encode tile size", self.vae_encode_tile_size), ("Encode overlap", self.vae_encode_overlap),
            ("Decode tile size", self.vae_decode_tile_size), ("Decode overlap", self.vae_decode_overlap),
        ]
        for i, (label, widget) in enumerate(citems):
            row, col = divmod(i, 3); cell = QVBoxLayout(); cell.addWidget(QLabel(label)); cell.addWidget(widget); cg.addLayout(cell, row, col)
        cg.addWidget(self.uniform_batch, 3, 0); cg.addWidget(self.vae_encode_tiled, 3, 1); cg.addWidget(self.vae_decode_tiled, 3, 2)
        self.community_group = community
        advanced.addWidget(community)

        official_adv = QGroupBox("Original ByteDance advanced settings")
        og = QGridLayout(official_adv)
        self.model_size = QComboBox(); self.model_size.addItems(["3B", "7B"])
        self.steps = QSpinBox(); self.steps.setRange(1, 50); self.steps.setValue(1)
        self.cfg = QDoubleSpinBox(); self.cfg.setRange(0.0, 20.0); self.cfg.setDecimals(2); self.cfg.setValue(1.0)
        self.cfg_rescale = QDoubleSpinBox(); self.cfg_rescale.setRange(0.0, 1.0); self.cfg_rescale.setDecimals(3); self.cfg_rescale.setValue(0.0)
        self.sp_size = QSpinBox(); self.sp_size.setRange(1, 1); self.sp_size.setValue(1)
        self.color_fix = QCheckBox("Use optional wavelet color fix if color_fix.py exists")
        for i, (label, widget) in enumerate([("Model architecture", self.model_size), ("Sampling steps", self.steps), ("CFG scale", self.cfg), ("CFG rescale", self.cfg_rescale), ("Sequence parallel size", self.sp_size)]):
            row, col = divmod(i, 3); cell = QVBoxLayout(); cell.addWidget(QLabel(label)); cell.addWidget(widget); og.addLayout(cell, row, col)
        og.addWidget(self.color_fix, 2, 0, 1, 3)
        advanced.addWidget(official_adv)
        advanced.addStretch(1)

        # ---------------- System / Log tab ----------------
        env_group = QGroupBox("Standalone installation / Python environment")
        eg = QGridLayout(env_group)
        self.python_edit = QLineEdit(str(self._default_env_python()) if self._default_env_python().exists() else sys.executable)
        py_browse = QPushButton("Browse…"); py_browse.clicked.connect(self._pick_python)
        self.install_deps_btn = QPushButton("Install / repair standalone SeedVR2"); self.install_deps_btn.clicked.connect(self.install_standalone_runtime)
        self.env_status = QLabel(
            "Creates/reuses environments/.seedvr2 and installs the community backend into "
            "models/seedvr2/ComfyUI-SeedVR2_VideoUpscaler. Re-running repairs the installation."
        ); self.env_status.setWordWrap(True)
        eg.addWidget(QLabel("Python environment"), 0, 0); eg.addWidget(self.python_edit, 0, 1); eg.addWidget(py_browse, 0, 2)
        eg.addWidget(self.install_deps_btn, 1, 0, 1, 3); eg.addWidget(self.env_status, 2, 0, 1, 3)
        system.addWidget(env_group)

        machine = QGroupBox("Machine / VRAM settings")
        mg = QGridLayout(machine)
        self.blocks_to_swap = QSpinBox(); self.blocks_to_swap.setRange(0, 36); self.blocks_to_swap.setValue(32)
        self.attention_mode = QComboBox(); self.attention_mode.addItems(["sageattn_2", "flash_attn_2", "sdpa", "auto"])
        self.cpu_offload = QCheckBox("DiT + VAE offload to CPU"); self.cpu_offload.setChecked(True)
        self.dit_offload = QCheckBox("Strict staged VRAM offload (original backend)"); self.dit_offload.setChecked(True); self.dit_offload.setEnabled(False)
        for i, (label, widget) in enumerate([("Blocks to swap", self.blocks_to_swap), ("Attention", self.attention_mode)]):
            cell = QVBoxLayout(); cell.addWidget(QLabel(label)); cell.addWidget(widget); mg.addLayout(cell, 0, i)
        mg.addWidget(self.cpu_offload, 1, 0, 1, 2); mg.addWidget(self.dit_offload, 2, 0, 1, 2)
        system.addWidget(machine)

        log_group = QGroupBox("Log")
        lg = QVBoxLayout(log_group)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(420)
        lg.addWidget(self.log)
        system.addWidget(log_group, 1)

        self.repo_edit.textChanged.connect(self._autofill_ckpts)
        self.backend.currentIndexChanged.connect(self._backend_changed)
        self._backend_changed()

        self.community_resolution.setToolTip("Default: 1080p. Quick presets for the community backend short-side target: 720p, 768p, 1080p, 1440p or 2560p.")
        self.community_max_resolution.setToolTip("Default: none. Set a maximum long-edge clamp for the community backend, or leave it at none.")
        self.batch_size.setToolTip("Default: 25. Only valid 4n+1 values are available here: 1, 5, 9, 13, 17, 21, 25, 29, 33, and so on.")
        self.chunk_size.setToolTip("Default: 330. Streaming chunk size for long videos. Lower values can reduce memory pressure.")
        self.temporal_overlap.setToolTip("Default: 3. Frames shared between adjacent temporal chunks to reduce seams.")
        self.prepend_frames.setToolTip("Default: 0. Reuse a number of previous frames at the start of each chunk when needed.")
        self.color_correction.setToolTip("Default: lab. Community backend color correction mode.")
        self.uniform_batch.setToolTip("Default: ON. Pads the final short batch to the full batch size to reduce end-of-video temporal artifacts.")
        self.vae_encode_tiled.setToolTip("Default: ON. Encode the VAE in tiles to reduce VRAM use.")
        self.vae_encode_tile_size.setToolTip("Default: 1024. Community VAE encode tile size.")
        self.vae_encode_overlap.setToolTip("Default: 128. Community VAE encode tile overlap.")
        self.vae_decode_tiled.setToolTip("Default: ON. Decode the VAE in tiles to reduce VRAM use.")
        self.vae_decode_tile_size.setToolTip("Default: 768. Community VAE decode tile size.")
        self.vae_decode_overlap.setToolTip("Default: 128. Community VAE decode tile overlap.")
        self.blocks_to_swap.setToolTip("Default: 32. Number of DiT transformer blocks swapped to CPU to reduce VRAM usage.")
        self.attention_mode.setToolTip("Default: sageattn_2. Attention implementation used by the community backend.")
        self.cpu_offload.setToolTip("Default: ON. Offload DiT and VAE work to CPU when needed to lower VRAM pressure.")
        self.model_size.setToolTip("Default: 3B. Original ByteDance backend model architecture.")
        self.steps.setToolTip("Default: 1. Official SeedVR2 is a one-step restoration model.")
        self.cfg.setToolTip("Default: 1.0. Classifier-free guidance scale for the original backend.")
        self.cfg_rescale.setToolTip("Default: 0.0. CFG rescale for the original backend.")
        self.sp_size.setToolTip("Default: 1. Single-GPU Windows mode uses sequence-parallel size 1.")
        self.color_fix.setToolTip("Default: OFF. Uses the optional wavelet color fix only when color_fix.py exists.")
        self.dit_offload.setToolTip("Original backend staged offload. Community mode uses its own CPU offload / BlockSwap controls.")

        # Mouse wheel must scroll the active tab and never silently change a setting.
        self._install_wheel_protection()

    def _install_wheel_protection(self):
        for widget in self.findChildren(QAbstractSpinBox):
            widget.installEventFilter(self)
        for widget in self.findChildren(QComboBox):
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel and isinstance(watched, (QAbstractSpinBox, QComboBox)):
            try:
                parent = watched.parentWidget()
                scroll = None
                while parent is not None:
                    if isinstance(parent, QScrollArea):
                        scroll = parent
                        break
                    parent = parent.parentWidget()
                if scroll is None and getattr(self, "_tab_scrolls", None):
                    scroll = self._tab_scrolls[self.tabs.currentIndex()]
                if scroll is None:
                    event.accept()
                    return True
                bar = scroll.verticalScrollBar()
                delta = event.angleDelta().y()
                if delta:
                    steps = delta / 120.0
                    move = int(steps * max(40, bar.singleStep() * 3))
                    bar.setValue(bar.value() - move)
                event.accept()
                return True
            except Exception:
                event.accept()
                return True
        return super().eventFilter(watched, event)


    def _load_preview(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            self.preview_status.setText("Finished output was not found: " + str(p))
            return
        self.preview_status.setText(f"Preview: {p.name}")
        if self.preview_player is not None:
            self.preview_player.stop()
            self.preview_player.setSource(QUrl.fromLocalFile(str(p.resolve())))

    def _toggle_preview_play(self):
        if self.preview_player is None:
            return
        try:
            if self.preview_player.playbackState() == QMediaPlayer.PlayingState:
                self.preview_player.pause()
            else:
                self.preview_player.play()
        except Exception:
            pass

    def _restart_preview(self):
        if self.preview_player is not None:
            self.preview_player.setPosition(0)
            self.preview_player.play()

    def _open_preview_external(self):
        p = Path(self.output_edit.text().strip())
        if p.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.resolve())))



    def _app_root(self) -> Path:
        # Standalone layout: normally the GUI file sits in the application root.
        # During FrameVision development it lives in <FrameVision>/helpers; in that
        # specific case helpers is NOT the app root, otherwise models/environments
        # would incorrectly be created inside helpers.
        here = Path(__file__).resolve().parent
        if here.name.lower() == "helpers":
            return here.parent
        return here

    def _framevision_root(self) -> Path:
        # Kept as an alias for older helper code; no FrameVision installation is required.
        return self._app_root()

    def _default_community_repo(self) -> Path:
        return self._app_root() / "models" / "seedvr2" / "ComfyUI-SeedVR2_VideoUpscaler"

    def _default_env_python(self) -> Path:
        env = self._app_root() / "environments" / ".seedvr2"
        return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def _pick_community_repo(self):
        p = QFileDialog.getExistingDirectory(self, "Select community ComfyUI-SeedVR2_VideoUpscaler repository", self.community_repo_edit.text())
        if p: self.community_repo_edit.setText(p)

    def _backend_changed(self):
        community = self.backend.currentIndex() == 0
        self.community_group.setEnabled(community)
        self.model_size.setEnabled(not community)
        self.steps.setEnabled(not community)
        self.cfg.setEnabled(not community)
        self.cfg_rescale.setEnabled(not community)
        self.sp_size.setEnabled(not community)
        self.dit_offload.setEnabled(False)
        self.color_fix.setEnabled(not community)
        if community:
            self.dit_offload.setText("Original-backend staged VRAM offload (not used in community mode)")
        else:
            self.dit_offload.setText("Strict staged VRAM offload (recommended for RTX 3090 / 24 GB)")
        self._autofill_ckpts()

    def _community_runner_path(self):
        return Path(__file__).resolve().with_name("seedvr2_community_runner.py")

    def _official_model_dir(self) -> Path:
        # This downloader now targets the safetensor models used by the community backend.
        # Use a separate settings key so an older saved /official path cannot silently
        # redirect the new safetensor download.
        try:
            saved = str(self.settings.value("safetensor_model_dir", "") or "").strip()
            if saved:
                return Path(saved)
        except Exception:
            pass
        return self._app_root() / "models" / "seedvr2"

    def download_official_models(self):
        if self.download_worker is not None:
            return

        # Never begin a multi-GB model download without first showing exactly what
        # will be downloaded and where it will be written.  The user can change the
        # destination before any network transfer starts.
        dest = self._official_model_dir()
        while True:
            box = QMessageBox(self)
            box.setWindowTitle("Download SeedVR2 safetensor models")
            box.setIcon(QMessageBox.Information)
            box.setText("The SeedVR2 safetensor models used by this community backend will be downloaded.")
            box.setInformativeText(
                "Files:\n"
                "  • seedvr2_ema_3b_bf16.safetensors\n"
                "    from szwagros/SeedVR2-3B-bf16\n"
                "  • ema_vae_fp16.safetensors\n"
                "    from numz/SeedVR2_comfyUI\n\n"
                f"Destination folder:\n{dest}\n\n"
                "Existing complete files are skipped. Incomplete downloads use temporary .download files."
            )
            download_here = box.addButton("Download here", QMessageBox.AcceptRole)
            choose_folder = box.addButton("Choose folder…", QMessageBox.ActionRole)
            cancel_btn = box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(download_here)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_btn or clicked is None:
                self.download_status.setText("Model download cancelled before starting.")
                return
            if clicked is choose_folder:
                chosen = QFileDialog.getExistingDirectory(
                    self, "Choose folder for SeedVR2 safetensor model files", str(dest)
                )
                if chosen:
                    dest = Path(chosen)
                    self.settings.setValue("safetensor_model_dir", str(dest))
                    self.model_dir_label.setText(str(dest))
                # Show the confirmation again so the selected path is visible before download.
                continue
            if clicked is download_here:
                break

        self.settings.setValue("safetensor_model_dir", str(dest))
        self.model_dir_label.setText(str(dest))
        self.download_progress.setValue(0)
        self.download_status.setText(f"Preparing SeedVR2 safetensor download to {dest} …")
        self.download_btn.setEnabled(False)
        self.download_cancel_btn.setEnabled(True)
        self.download_worker = SafetensorModelDownloadWorker(dest, self)
        self.download_worker.progress.connect(self._download_progress_changed)
        self.download_worker.completed.connect(self._download_completed)
        self.download_worker.failed.connect(self._download_failed)
        self.download_worker.start()

    def cancel_model_download(self):
        if self.download_worker is not None:
            self.download_worker.cancel()
            self.download_status.setText("Cancelling after active network reads finish…")

    def _download_progress_changed(self, value: int, message: str):
        self.download_progress.setValue(value)
        self.download_status.setText(message)

    def _download_completed(self, folder: str):
        self.download_worker = None
        self.download_btn.setEnabled(True)
        self.download_cancel_btn.setEnabled(False)
        self.download_progress.setValue(100)
        self.download_status.setText(f"Complete: {folder}")
        base = Path(folder)
        self.dit_edit.setText(str(base / "seedvr2_ema_3b_bf16.safetensors"))
        self.vae_edit.setText(str(base / "ema_vae_fp16.safetensors"))
        QMessageBox.information(self, "SeedVR2 models", "SeedVR2 safetensor model files are ready.")

    def _download_failed(self, message: str):
        self.download_worker = None
        self.download_btn.setEnabled(True)
        self.download_cancel_btn.setEnabled(False)
        self.download_status.setText("Download stopped: " + message)
        if "cancelled" not in message.lower():
            QMessageBox.critical(self, "SeedVR2 model download failed", message)

    def _python_missing_imports(self):
        py = self.python_edit.text().strip()
        if not py or not Path(py).exists():
            return ["python executable"]
        code = (
            "import importlib.util, json; "
            f"mods={list(RUNTIME_IMPORTS)!r}; "
            "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
        )
        try:
            p = subprocess.run([py, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, errors="replace", timeout=60)
            if p.returncode != 0:
                return ["environment check failed: " + (p.stderr.strip() or f"exit {p.returncode}")]
            line = (p.stdout or "").strip().splitlines()[-1]
            return json.loads(line)
        except Exception as e:
            return [f"environment check failed: {e}"]

    def install_standalone_runtime(self):
        if self.standalone_install_worker is not None:
            return
        self.install_deps_btn.setEnabled(False)
        self.env_status.setText("Preparing standalone SeedVR2 environment and community backend…")
        self.standalone_install_worker = StandaloneInstallWorker(self._app_root(), self)
        self.standalone_install_worker.progress.connect(self._dependency_log)
        self.standalone_install_worker.completed.connect(self._standalone_install_done)
        self.standalone_install_worker.failed.connect(self._standalone_install_failed)
        self.standalone_install_worker.start()

    def _standalone_install_done(self, python_exe: str, repo_folder: str):
        self.standalone_install_worker = None
        self.install_deps_btn.setEnabled(True)
        self.python_edit.setText(python_exe)
        self.community_repo_edit.setText(repo_folder)
        self.env_status.setText("Standalone SeedVR2 runtime ready.")
        self._autofill_ckpts()
        QMessageBox.information(
            self, "SeedVR2 installation",
            "Standalone SeedVR2 runtime is ready.\n\n"
            f"Environment:\n{python_exe}\n\n"
            f"Community backend:\n{repo_folder}"
        )

    def _standalone_install_failed(self, message: str):
        self.standalone_install_worker = None
        self.install_deps_btn.setEnabled(True)
        self.env_status.setText("Standalone install failed: " + message)
        QMessageBox.critical(self, "SeedVR2 installation failed", message)

    def install_runtime_dependencies(self):
        if self.dependency_worker is not None:
            return
        py = self.python_edit.text().strip()
        if not py or not Path(py).exists():
            QMessageBox.critical(self, "SeedVR2 environment", "Select a valid SeedVR2 python.exe first.")
            return
        self.install_deps_btn.setEnabled(False)
        self.env_status.setText("Installing official SeedVR runtime dependencies…")
        self.dependency_worker = DependencyInstallWorker(py, self)
        self.dependency_worker.progress.connect(self._dependency_log)
        self.dependency_worker.completed.connect(self._dependency_done)
        self.dependency_worker.failed.connect(self._dependency_failed)
        self.dependency_worker.start()

    def _dependency_log(self, line: str):
        self.env_status.setText(line)
        self.log.append("[ENV] " + line)

    def _dependency_done(self):
        self.dependency_worker = None
        self.install_deps_btn.setEnabled(True)
        missing = self._python_missing_imports()
        if missing:
            msg = "Install finished, but these imports are still missing: " + ", ".join(missing)
            self.env_status.setText(msg)
            QMessageBox.warning(self, "SeedVR2 environment", msg)
        else:
            self.env_status.setText("Runtime dependencies OK.")
            QMessageBox.information(self, "SeedVR2 environment", "Official SeedVR runtime dependencies are installed.")

    def _dependency_failed(self, message: str):
        self.dependency_worker = None
        self.install_deps_btn.setEnabled(True)
        self.env_status.setText("Dependency install failed: " + message)
        QMessageBox.critical(self, "SeedVR2 environment", message)

    def _pick_repo(self):
        p = QFileDialog.getExistingDirectory(self, "Select extracted official SeedVR repository")
        if p: self.repo_edit.setText(p)
    def _pick_python(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Python", filter="Python (python.exe python)")
        if p: self.python_edit.setText(p)
    def _pick_dit(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select SeedVR2 DiT checkpoint", filter="Checkpoint (*.pth *.pt *.safetensors);;All files (*)")
        if p: self.dit_edit.setText(p)
    def _pick_vae(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select SeedVR VAE checkpoint", filter="Checkpoint (*.pth *.pt *.safetensors);;All files (*)")
        if p: self.vae_edit.setText(p)
    @staticmethod
    def _next_available_output(path: Path) -> Path:
        """Return path unchanged when free, otherwise add _002, _003, ... .

        SeedVR2 renders are never allowed to silently replace an earlier upscale.
        """
        path = Path(path)
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix or ".mp4"
        for n in range(2, 10000):
            candidate = path.with_name(f"{stem}_{n:03d}{suffix}")
            if not candidate.exists():
                return candidate
        # Practically unreachable, but a timestamp guarantees a unique fallback.
        return path.with_name(f"{stem}_{int(time.time())}{suffix}")

    def _suggest_output_for_input(self, input_path: str) -> Path:
        src = Path(input_path)
        base = src.with_name(src.stem + "_seedvr2.mp4")
        return self._next_available_output(base)

    def _ensure_unique_output_path(self):
        """Populate/advance the output name before every run so nothing is overwritten."""
        inp = self.input_edit.text().strip()
        out = self.output_edit.text().strip()
        if not out and inp:
            self.output_edit.setText(str(self._suggest_output_for_input(inp)))
            return
        if out:
            p = Path(out)
            unique = self._next_available_output(p)
            if unique != p:
                self.output_edit.setText(str(unique))
                try:
                    self.log.append(f"Output already exists; using new file: {unique.name}")
                except Exception:
                    pass

    def _pick_input(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select input", filter="Media (*.mp4 *.mov *.mkv *.avi *.webm *.png *.jpg *.jpeg *.webp);;All files (*)")
        if p:
            self.input_edit.setText(p)
            # A newly selected source always gets a name based on that source, not a
            # stale output filename remembered from the previous job.
            self.output_edit.setText(str(self._suggest_output_for_input(p)))
    def _pick_output(self):
        p, _ = QFileDialog.getSaveFileName(self, "Output", self.output_edit.text() or "seedvr2_output.mp4", "MP4 (*.mp4);;All files (*)")
        if p:
            # Respect the chosen base name, but do not allow a silent overwrite.
            self.output_edit.setText(str(self._next_available_output(Path(p))))

    def _autofill_ckpts(self):
        community = hasattr(self, "backend") and self.backend.currentIndex() == 0
        dit_text = self.dit_edit.text().strip()
        vae_text = self.vae_edit.text().strip()

        if community:
            model_dir = self._app_root() / "models" / "seedvr2"
            # Replace an old official .pth auto-selection when community mode is active,
            # but never overwrite a valid user-selected community model.
            if (not dit_text) or Path(dit_text).suffix.lower() == ".pth":
                candidates = [
                    "seedvr2_ema_3b_fp16.safetensors",
                    "seedvr2_ema_3b_bf16.safetensors",
                    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                    "seedvr2_ema_3b-Q8_0.gguf",
                    "seedvr2_ema_3b-Q4_K_M.gguf",
                ]
                for name in candidates:
                    q = model_dir / name
                    if q.exists():
                        self.dit_edit.setText(str(q)); break
            if (not vae_text) or Path(vae_text).suffix.lower() == ".pth":
                for name in ("ema_vae_fp16.safetensors", "ema_vae.safetensors"):
                    q = model_dir / name
                    if q.exists():
                        self.vae_edit.setText(str(q)); break
            return

        official = self._official_model_dir()
        if not dit_text or Path(dit_text).suffix.lower() in (".safetensors", ".gguf"):
            q = official / "seedvr2_ema_3b.pth"
            if q.exists(): self.dit_edit.setText(str(q))
        if not vae_text or Path(vae_text).suffix.lower() == ".safetensors":
            q = official / "ema_vae.pth"
            if q.exists(): self.vae_edit.setText(str(q))

        repo = Path(self.repo_edit.text().strip())
        if not repo.exists(): return
        if not self.dit_edit.text().strip():
            for name in ("seedvr2_ema_3b.pth", "seedvr2_ema_3b.safetensors"):
                q = repo / "ckpts" / name
                if q.exists(): self.dit_edit.setText(str(q)); break
        if not self.vae_edit.text().strip():
            for name in ("ema_vae.pth", "ema_vae_fp16.safetensors", "ema_vae.safetensors"):
                q = repo / "ckpts" / name
                if q.exists(): self.vae_edit.setText(str(q)); break

    def _validate(self, show=True):
        errors = []
        py = Path(self.python_edit.text().strip())
        dit = Path(self.dit_edit.text().strip())
        vae = Path(self.vae_edit.text().strip())
        inp = Path(self.input_edit.text().strip())
        community = self.backend.currentIndex() == 0

        if not py.exists(): errors.append("Python executable does not exist.")
        if not dit.exists(): errors.append("DiT checkpoint does not exist.")
        if not vae.exists(): errors.append("VAE checkpoint does not exist.")
        if not self.input_edit.text().strip(): errors.append("Select an input video or image.")
        elif not inp.exists(): errors.append("Input file does not exist.")
        if not self.output_edit.text().strip(): errors.append("Select an output path.")

        if community:
            crepo = Path(self.community_repo_edit.text().strip())
            cli = crepo / "inference_cli.py"
            if not cli.exists(): errors.append("Community repository is missing inference_cli.py.")
            if dit.suffix.lower() not in (".safetensors", ".gguf"):
                errors.append("Community backend expects a .safetensors or .gguf DiT model.")
            if vae.suffix.lower() != ".safetensors":
                errors.append("Community backend expects a .safetensors VAE.")
            if dit.exists() and vae.exists() and dit.parent.resolve() != vae.parent.resolve():
                errors.append("For the community backend, place the selected DiT and VAE in the same model folder.")
            b = self._combo_int_value(self.batch_size, 25)
            if b != 1 and (b - 1) % 4 != 0:
                errors.append("Community batch size must follow 4n+1 (1, 5, 9, 13, 17, ...).")
            if self.vae_encode_overlap.value() >= self.vae_encode_tile_size.value():
                errors.append("VAE encode overlap must be smaller than encode tile size.")
            if self.vae_decode_overlap.value() >= self.vae_decode_tile_size.value():
                errors.append("VAE decode overlap must be smaller than decode tile size.")
        else:
            repo = Path(self.repo_edit.text().strip())
            required_repo = [repo / "projects" / "inference_seedvr2_3b.py", repo / "configs_3b" / "main.yaml"]
            if not repo.exists() or not all(p.exists() for p in required_repo): errors.append("Selected folder is not a complete official SeedVR repository.")
            emb_dir = dit.parent if dit.exists() else self._official_model_dir()
            have_embeds = (emb_dir / "pos_emb.pt").exists() and (emb_dir / "neg_emb.pt").exists()
            if not have_embeds:
                have_embeds = (repo / "pos_emb.pt").exists() and (repo / "neg_emb.pt").exists()
            if not have_embeds: errors.append("pos_emb.pt and neg_emb.pt are missing beside the DiT checkpoint and from the SeedVR repo root.")
            missing = self._python_missing_imports() if py.exists() else []
            if missing: errors.append("SeedVR2 Python environment is missing: " + ", ".join(missing) + ". Click 'Install / repair official runtime dependencies'.")
            if self.model_size.currentText() == "7B" and not (repo / "configs_7b" / "main.yaml").exists(): errors.append("7B config is missing.")

        if errors:
            if show: QMessageBox.critical(self, "Validation failed", "\n".join(errors))
            return False
        if show:
            mode = "Community/ComfyUI safetensors backend" if community else "Original ByteDance backend"
            QMessageBox.information(self, "Validation", f"{mode}: paths and settings look valid.")
        return True

    def _runner_path(self):
        return Path(__file__).resolve().with_name("seedvr2_official_runner.py")

    def _command(self):
        seed = self.seed.value()
        if seed < 0:
            import secrets
            seed = secrets.randbelow(2_147_483_647)
        if self.backend.currentIndex() == 0:
            args = [
                str(self._community_runner_path()),
                "--repo", self.community_repo_edit.text().strip(),
                "--input", self.input_edit.text().strip(), "--output", self.output_edit.text().strip(),
                "--dit", self.dit_edit.text().strip(), "--vae", self.vae_edit.text().strip(),
                "--resolution", str(self._combo_int_value(self.community_resolution, 1080)),
                "--max-resolution", str(self.community_max_resolution.value()),
                "--batch-size", str(self._combo_int_value(self.batch_size, 25)), "--chunk-size", str(self.chunk_size.value()),
                "--temporal-overlap", str(self.temporal_overlap.value()), "--prepend-frames", str(self.prepend_frames.value()),
                "--blocks-to-swap", str(self.blocks_to_swap.value()), "--attention-mode", self.attention_mode.currentText(),
                "--color-correction", self.color_correction.currentText(), "--seed", str(seed),
                "--vae-encode-tile-size", str(self.vae_encode_tile_size.value()), "--vae-encode-overlap", str(self.vae_encode_overlap.value()),
                "--vae-decode-tile-size", str(self.vae_decode_tile_size.value()), "--vae-decode-overlap", str(self.vae_decode_overlap.value()),
            ]
            if self.uniform_batch.isChecked(): args.append("--uniform-batch-size")
            if self.vae_encode_tiled.isChecked(): args.append("--vae-encode-tiled")
            if self.vae_decode_tiled.isChecked(): args.append("--vae-decode-tiled")
            if self.cpu_offload.isChecked(): args.append("--cpu-offload")
            return self.python_edit.text().strip(), args

        args = [
            str(self._runner_path()), "--repo", self.repo_edit.text().strip(), "--input", self.input_edit.text().strip(),
            "--output", self.output_edit.text().strip(), "--dit", self.dit_edit.text().strip(), "--vae", self.vae_edit.text().strip(),
            "--model-size", self.model_size.currentText(), "--width", str(self.width.value()), "--height", str(self.height.value()),
            "--seed", str(seed), "--steps", str(self.steps.value()), "--cfg-scale", str(self.cfg.value()),
            "--cfg-rescale", str(self.cfg_rescale.value()), "--sp-size", str(self.sp_size.value())
        ]
        if self.fps.value() > 0: args += ["--out-fps", str(self.fps.value())]
        if self.dit_offload.isChecked(): args.append("--dit-offload")
        if self.color_fix.isChecked(): args.append("--color-fix")
        return self.python_edit.text().strip(), args

    def queue_run(self):
        """Queue the exact SeedVR2 command produced by this GUI.

        This intentionally does not use FrameVision's generic upscale_video/upscale_photo
        queue functions. Those functions have their own decode/encode/FPS pipeline. A
        SeedVR2 queue item is a first-class external SeedVR2 job so all resolution, model,
        memory, output FPS and backend choices remain owned by the SeedVR2 runners.
        """
        if self.proc is not None:
            QMessageBox.warning(self, "SeedVR2 queue", "A local SeedVR2 run is already active. Cancel or finish it before queueing another run from this tab.")
            return

        self._ensure_unique_output_path()
        if not self._validate(show=True):
            return
        self._save_settings()

        program, args = self._command()
        cmd = [str(program)] + [str(x) for x in args]
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        out_dir = str(Path(output_path).resolve().parent) if output_path else ""
        backend_name = "community" if self.backend.currentIndex() == 0 else "official"

        job = {
            "name": "SeedVR2",
            "label": "SeedVR2",
            "category": "upscale",
            "engine": "seedvr2",
            "seedvr2_backend": backend_name,
            "input": input_path,
            "output": output_path,
            "outfile": output_path,
            "out_dir": out_dir,
            "cmd": cmd,
            "cwd": str(Path(__file__).resolve().parent),
            # Keep the child process deterministic on Windows. No codec/FPS values
            # are injected here; they stay entirely in the command built above.
            "env": {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        }

        try:
            try:
                from helpers import queue_adapter as qa
            except Exception:
                import queue_adapter as qa

            enqueue_fn = getattr(qa, "enqueue_seedvr2", None) or getattr(qa, "enqueue", None)
            if not callable(enqueue_fn):
                raise RuntimeError("helpers/queue_adapter.py does not expose a SeedVR2 queue function.")
            enqueue_fn(job)
        except Exception as exc:
            QMessageBox.critical(self, "SeedVR2 queue", f"Could not add SeedVR2 to the queue:\n{exc}")
            return

        self.log.append("QUEUED: " + shlex.join(cmd))
        self.log.append("Queue mode preserves the SeedVR2 command exactly; normal Upscale FPS/codec processing is bypassed.")
        QMessageBox.information(self, "SeedVR2 queue", "SeedVR2 was added to the FrameVision queue with the current settings.")

    def run(self):
        if self.proc is not None: return
        # Resolve a fresh output name before validation/launch. This protects prior
        # renders even when the previous output path is still stored in settings.
        self._ensure_unique_output_path()
        if not self._validate(show=True): return
        self._save_settings()
        program, args = self._command()
        self.log.clear()
        self.log.append("RUN: " + shlex.join([program] + args))
        self.tabs.setCurrentIndex(2)
        self.proc = QProcess(self)
        # The community SeedVR2 backend prints Unicode/emoji during imports.
        # Force UTF-8 in the child process so Windows cp1252 cannot crash it.
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._finished)
        self.run_btn.setEnabled(False); self.cancel_btn.setEnabled(True)
        self.proc.start(program, args)

    def _read_output(self):
        if not self.proc: return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log.moveCursor(QTextCursor.End)
            self.log.insertPlainText(data)
            self.log.ensureCursorVisible()

    def _finished(self, code, status):
        self._read_output()
        self.log.append(f"\nProcess finished with exit code {code}.")
        self.proc = None
        self.run_btn.setEnabled(True); self.cancel_btn.setEnabled(False)
        if code == 0:
            self._load_preview(self.output_edit.text().strip())
            self.tabs.setCurrentIndex(0)
            QMessageBox.information(self, "SeedVR2", "Inference finished. The result is loaded in the Simple tab preview.")
        else:
            QMessageBox.warning(self, "SeedVR2", "Inference failed. See the System / Log tab for the exact runtime error.")

    def cancel(self):
        if self.proc:
            self.proc.kill()
            self.log.append("\nCancelled by user.")

    def open_output_folder(self):
        p = Path(self.output_edit.text().strip()).parent
        if p.exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _apply_embedded_defaults(self):
        """Known-good defaults for the new embedded SeedVR2 integration."""
        # Everyday/community controls.
        self.backend.setCurrentIndex(0)
        self._set_combo_int_value(self.community_resolution, 1080)
        self.community_max_resolution.setValue(0)
        self.width.setValue(1920)
        self.height.setValue(1080)
        self.seed.setValue(-1)
        self.fps.setValue(0.0)  # source FPS

        self._set_combo_int_value(self.batch_size, 25)
        self.chunk_size.setValue(330)
        self.temporal_overlap.setValue(3)
        self.prepend_frames.setValue(0)
        self.color_correction.setCurrentText("lab")
        self.uniform_batch.setChecked(True)
        self.vae_encode_tiled.setChecked(True)
        self.vae_encode_tile_size.setValue(1024)
        self.vae_encode_overlap.setValue(128)
        self.vae_decode_tiled.setChecked(True)
        self.vae_decode_tile_size.setValue(768)
        self.vae_decode_overlap.setValue(128)

        # Machine / VRAM controls used by the community backend.
        self.blocks_to_swap.setValue(32)
        self.attention_mode.setCurrentText("sageattn_2")
        self.cpu_offload.setChecked(True)

        # Original ByteDance backend defaults. These do not affect community mode,
        # but keeping them sane avoids the misleading 50 / 2.0 / 1.0 values that
        # were appearing after the old settings collision.
        self.model_size.setCurrentText("3B")
        self.steps.setValue(1)
        self.cfg.setValue(1.0)
        self.cfg_rescale.setValue(0.0)
        self.sp_size.setValue(1)
        self.color_fix.setChecked(False)

    @staticmethod
    def _combo_int_value(combo: QComboBox, default: int) -> int:
        data = combo.currentData()
        try:
            return int(data)
        except Exception:
            try:
                return int(combo.currentText().strip())
            except Exception:
                return int(default)

    @staticmethod
    def _set_combo_int_value(combo: QComboBox, value: int, default_value: int | None = None):
        try:
            value = int(value)
        except Exception:
            value = default_value if default_value is not None else SeedVR2Window._combo_int_value(combo, 0)
        idx = combo.findData(value)
        if idx < 0 and default_value is not None:
            idx = combo.findData(int(default_value))
        if idx < 0:
            idx = 0
        combo.setCurrentIndex(idx)

    @staticmethod
    def _setting_bool(store, key, default=False):
        value = store.value(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in ("false", "0", "no", "off", "")

    def _load_settings(self):
        # First launch of the embedded-v2 tab: start from known-good values and
        # migrate only harmless path fields from the old standalone settings.
        schema = int(self.settings.value("settings_schema_version", 0) or 0)
        if schema < 2:
            self._apply_embedded_defaults()
            for edit, key in [
                (self.repo_edit, "repo"),
                (self.community_repo_edit, "community_repo"),
                (self.python_edit, "python"),
                (self.dit_edit, "dit"),
                (self.vae_edit, "vae"),
            ]:
                try:
                    v = self.legacy_settings.value(key, "")
                    if v:
                        edit.setText(str(v))
                except Exception:
                    pass
            self.settings.setValue("settings_schema_version", 2)
            self._save_settings()
            return

        # Paths / I/O.
        for edit, key in [
            (self.repo_edit,"repo"), (self.community_repo_edit,"community_repo"),
            (self.python_edit,"python"), (self.dit_edit,"dit"), (self.vae_edit,"vae"),
            (self.input_edit,"input"), (self.output_edit,"output")
        ]:
            v = self.settings.value(key, "")
            if v:
                edit.setText(str(v))

        # Simple tab.
        self.backend.setCurrentIndex(int(self.settings.value("backend", 0)))
        self._set_combo_int_value(self.community_resolution, int(self.settings.value("community_resolution", 1080)), default_value=1080)
        self.community_max_resolution.setValue(int(self.settings.value("community_max_resolution", 0)))
        self.width.setValue(int(self.settings.value("width", 1920)))
        self.height.setValue(int(self.settings.value("height", 1080)))
        self.seed.setValue(int(self.settings.value("seed", -1)))
        self.fps.setValue(float(self.settings.value("fps", 0.0)))

        # Community quality / temporal.
        self._set_combo_int_value(self.batch_size, int(self.settings.value("batch_size", 25)), default_value=25)
        self.chunk_size.setValue(int(self.settings.value("chunk_size", 330)))
        self.temporal_overlap.setValue(int(self.settings.value("temporal_overlap", 3)))
        self.prepend_frames.setValue(int(self.settings.value("prepend_frames", 0)))
        self.color_correction.setCurrentText(str(self.settings.value("color_correction", "lab")))
        self.uniform_batch.setChecked(self._setting_bool(self.settings, "uniform_batch", True))
        self.vae_encode_tiled.setChecked(self._setting_bool(self.settings, "vae_encode_tiled", True))
        self.vae_encode_tile_size.setValue(int(self.settings.value("vae_encode_tile_size", 1024)))
        self.vae_encode_overlap.setValue(int(self.settings.value("vae_encode_overlap", 128)))
        self.vae_decode_tiled.setChecked(self._setting_bool(self.settings, "vae_decode_tiled", True))
        self.vae_decode_tile_size.setValue(int(self.settings.value("vae_decode_tile_size", 768)))
        self.vae_decode_overlap.setValue(int(self.settings.value("vae_decode_overlap", 128)))

        # Machine / VRAM.
        self.blocks_to_swap.setValue(int(self.settings.value("blocks_to_swap", 32)))
        self.attention_mode.setCurrentText(str(self.settings.value("attention_mode", "sageattn_2")))
        self.cpu_offload.setChecked(self._setting_bool(self.settings, "cpu_offload", True))

        # Original backend advanced settings.
        self.model_size.setCurrentText(str(self.settings.value("model_size", "3B")))
        self.steps.setValue(int(self.settings.value("steps", 1)))
        self.cfg.setValue(float(self.settings.value("cfg", 1.0)))
        self.cfg_rescale.setValue(float(self.settings.value("cfg_rescale", 0.0)))
        self.sp_size.setValue(int(self.settings.value("sp_size", 1)))
        self.color_fix.setChecked(self._setting_bool(self.settings, "color_fix", False))

    def _save_settings(self):
        self.settings.setValue("settings_schema_version", 2)
        for edit, key in [
            (self.repo_edit,"repo"), (self.community_repo_edit,"community_repo"),
            (self.python_edit,"python"), (self.dit_edit,"dit"), (self.vae_edit,"vae"),
            (self.input_edit,"input"), (self.output_edit,"output")
        ]:
            self.settings.setValue(key, edit.text())

        # Simple tab.
        self.settings.setValue("backend", self.backend.currentIndex())
        self.settings.setValue("community_resolution", self._combo_int_value(self.community_resolution, 1080))
        self.settings.setValue("community_max_resolution", self.community_max_resolution.value())
        self.settings.setValue("width", self.width.value())
        self.settings.setValue("height", self.height.value())
        self.settings.setValue("seed", self.seed.value())
        self.settings.setValue("fps", self.fps.value())

        # Community quality / temporal.
        self.settings.setValue("batch_size", self._combo_int_value(self.batch_size, 25))
        self.settings.setValue("chunk_size", self.chunk_size.value())
        self.settings.setValue("temporal_overlap", self.temporal_overlap.value())
        self.settings.setValue("prepend_frames", self.prepend_frames.value())
        self.settings.setValue("color_correction", self.color_correction.currentText())
        self.settings.setValue("uniform_batch", self.uniform_batch.isChecked())
        self.settings.setValue("vae_encode_tiled", self.vae_encode_tiled.isChecked())
        self.settings.setValue("vae_encode_tile_size", self.vae_encode_tile_size.value())
        self.settings.setValue("vae_encode_overlap", self.vae_encode_overlap.value())
        self.settings.setValue("vae_decode_tiled", self.vae_decode_tiled.isChecked())
        self.settings.setValue("vae_decode_tile_size", self.vae_decode_tile_size.value())
        self.settings.setValue("vae_decode_overlap", self.vae_decode_overlap.value())

        # Machine / VRAM.
        self.settings.setValue("blocks_to_swap", self.blocks_to_swap.value())
        self.settings.setValue("attention_mode", self.attention_mode.currentText())
        self.settings.setValue("cpu_offload", self.cpu_offload.isChecked())

        # Original backend advanced settings.
        self.settings.setValue("model_size", self.model_size.currentText())
        self.settings.setValue("steps", self.steps.value())
        self.settings.setValue("cfg", self.cfg.value())
        self.settings.setValue("cfg_rescale", self.cfg_rescale.value())
        self.settings.setValue("sp_size", self.sp_size.value())
        self.settings.setValue("color_fix", self.color_fix.isChecked())

    def closeEvent(self, event):
        self._save_settings()
        if self.proc: self.proc.kill()
        if self.download_worker is not None:
            self.download_worker.cancel()
            self.download_worker.wait(2000)
        if self.dependency_worker is not None:
            self.dependency_worker.wait(2000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    w = SeedVR2Window(); w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
