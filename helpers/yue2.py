from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QDoubleSpinBox,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

HELPER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HELPER_DIR.parent
AUDIOCPP_DIR = PROJECT_ROOT / "presets" / "bin" / "audio"
AUDIOCPP_EXE = AUDIOCPP_DIR / "audiocpp_cli.exe"
MODEL_DIR = PROJECT_ROOT / "models" / "yue2"
SETTINGS_PATH = PROJECT_ROOT / "presets" / "setsave" / "yue2.json"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "output" / "yue2"
TEMP_DIR = PROJECT_ROOT / "temp"

GITHUB_REPO = "0xShug0/audio.cpp"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}"
HF_MODEL_PAGE = "https://huggingface.co/audio-cpp/Yue2-3B-GGUF"

DEFAULT_LYRICS = """[Verse]\nSoft morning light is touching the window.\nI hear the city waking below.\n\n[Chorus]\nStay with the rhythm let it carry us home.\nSing with the sunrise we are never alone."""
DEFAULT_STYLE = "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix"

DEFAULTS = {
    "model_gguf": "",
    "vae_gguf": "",
    "threads": 8,
    "steps": 32,
    "seed": -1,
    "duration_seconds": 180,
    "instrumental_generate": False,
    "instrumental_cover": False,
    "instrumental_score": False,
    "backend": "cuda",
    "output_dir": str(OUTPUT_DIR),
    "last_abc": "",
    "last_style": DEFAULT_STYLE,
    "last_lyrics": DEFAULT_LYRICS,
    "log_cli": True,
    "cfg_auto": True,
    "cfg_scale": 1.01,
    "abc_temperature": 0.7,
    "abc_top_p": 0.9,
    "abc_top_k": 30,
    "abc_repetition_penalty": 1.005,
    "abc_penalty_window": 100,
    "abc_min_tokens": 32,
    "abc_max_tokens": 4096,
    "semantic_temperature": 1.0,
    "semantic_top_p": 0.95,
    "semantic_top_k": 100,
    "semantic_repetition_penalty": 1.2,
    "semantic_penalty_window": 50,
    "semantic_min_tokens": 200,
    "semantic_max_tokens": 9000,
    "weight_type": "native",
    "model_weight_type": "native",
    "vae_weight_type": "native",
    "model_weight_context_mb": 6144,
    "vae_weight_context_mb": 1536,
    "ar_prefill_graph_arena_mb": 4096,
    "ar_decode_graph_arena_mb": 1536,
    "nar_graph_arena_mb": 6144,
    "vae_graph_arena_mb": 1536,
}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

# YuE2 semantic audio runs at roughly 50 semantic tokens per second.  The
# duration-enabled Gradio implementations use the semantic token budget as the
# duration controller, then trim the decoded audio to the requested sample
# count.  Keep a small overshoot margin so EOS / boundary effects do not leave
# the rendered WAV short of the requested duration.
YUE2_SEMANTIC_TOKENS_PER_SECOND = 50.0
YUE2_DURATION_OVERSHOOT = 1.08
YUE2_MAX_DURATION_SECONDS = 360


def _duration_token_budget(seconds: int) -> int:
    seconds = max(1, min(YUE2_MAX_DURATION_SECONDS, int(seconds)))
    tokens = int(round(seconds * YUE2_SEMANTIC_TOKENS_PER_SECOND * YUE2_DURATION_OVERSHOOT))
    return max(1, min(24576, tokens))


def _trim_wav_with_wave(path: Path, seconds: int) -> tuple[bool, str]:
    """Trim a PCM WAV to an exact frame count using only the stdlib."""
    temp = path.with_suffix(path.suffix + ".duration_tmp.wav")
    try:
        with wave.open(str(path), "rb") as src:
            channels = src.getnchannels()
            sampwidth = src.getsampwidth()
            framerate = src.getframerate()
            comptype = src.getcomptype()
            compname = src.getcompname()
            available = src.getnframes()
            wanted = int(round(float(seconds) * framerate))
            if wanted <= 0:
                return False, "invalid target duration"
            if available < wanted:
                return False, f"render is short ({available / framerate:.3f}s < {seconds}s)"
            frames = src.readframes(wanted)
        with wave.open(str(temp), "wb") as dst:
            dst.setnchannels(channels)
            dst.setsampwidth(sampwidth)
            dst.setframerate(framerate)
            dst.setcomptype(comptype, compname)
            dst.writeframes(frames)
        os.replace(temp, path)
        return True, f"{seconds}.000s at {framerate} Hz"
    except Exception as exc:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, str(exc)


def _find_ffmpeg() -> Optional[Path]:
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    candidates = (
        PROJECT_ROOT / "presets" / "bin" / "ffmpeg.exe",
        PROJECT_ROOT / "presets" / "bin" / "ffmpeg" / "ffmpeg.exe",
        PROJECT_ROOT / "tools" / "ffmpeg" / "ffmpeg.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _trim_wav_exact(path: Path, seconds: int) -> tuple[bool, str]:
    """Trim the generated WAV to exactly the requested duration."""
    ok, detail = _trim_wav_with_wave(path, seconds)
    if ok:
        return True, detail

    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return False, f"stdlib WAV trim failed ({detail}); ffmpeg was not found"

    temp = path.with_suffix(path.suffix + ".duration_tmp.wav")
    try:
        result = subprocess.run(
            [
                str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                "-t", f"{float(seconds):.6f}",
                "-c:a", "pcm_s16le",
                str(temp),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0 or not temp.is_file():
            temp.unlink(missing_ok=True)
            return False, (result.stderr or "ffmpeg trim failed").strip()
        os.replace(temp, path)
        return True, f"{seconds}.000s (ffmpeg)"
    except Exception as exc:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, str(exc)


def _ensure_dirs() -> None:
    for path in (AUDIOCPP_DIR, MODEL_DIR, SETTINGS_PATH.parent, LOG_DIR, OUTPUT_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            merged = dict(fallback)
            merged.update(data)
            return merged
    except Exception:
        pass
    return dict(fallback)


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_filename(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in text.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:64] or "yue2"


def _find_first(patterns: Iterable[str], root: Path) -> Optional[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    files = [p for p in candidates if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: (len(str(p)), str(p).lower()))
    return files[0]


def _auto_model_defaults() -> tuple[str, str]:
    model = _find_first(("*q4_0*.gguf", "*q8_0*.gguf", "*bf16*.gguf"), MODEL_DIR)
    vae = _find_first(("*vae*f16*.gguf", "*vae*f32*.gguf"), MODEL_DIR)
    return (str(model) if model else "", str(vae) if vae else "")


def _headers(*, artifact_download: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": "FrameVision-YuE2-Helper/1.1",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    # GitHub Actions artifact archive endpoints require authentication even for
    # public repositories. Honour an existing token without ever requiring one.
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and artifact_download:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def _url_json(url: str, timeout: int = 30) -> Any:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _download(
    url: str,
    destination: Path,
    progress_cb=None,
    timeout: int = 120,
    *,
    artifact_download: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=_headers(artifact_download=artifact_download))
    with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress_cb and total:
                progress_cb(done, total)


def _nightly_link_url(run_id: int, artifact_name: str) -> str:
    # nightly.link mirrors public GitHub Actions artifacts without requiring the
    # user to sign in to GitHub. Artifact names in the Actions API do not carry
    # the .zip suffix, while nightly.link expects it.
    safe_name = urllib.parse.quote(artifact_name.rstrip('/'), safe='._-')
    if not safe_name.lower().endswith('.zip'):
        safe_name += '.zip'
    return f"https://nightly.link/{GITHUB_REPO}/actions/runs/{run_id}/{safe_name}"


def _flatten_copy(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    # GitHub action/release zips may contain files directly or one/more wrapper dirs.
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        parts = list(rel.parts)
        # Strip wrapper folders until a runtime-looking level is reached.
        while len(parts) > 1 and parts[0].lower() in {"bin", "release", "audio", "audiocpp", "artifact"}:
            parts.pop(0)
        target = destination.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _runtime_has_yue2(exe: Path) -> tuple[bool, str]:
    if not exe.exists():
        return False, "audiocpp_cli.exe is missing"
    commands = [
        [str(exe), "--list-loaders", "--json"],
        [str(exe), "--help"],
    ]
    combined = ""
    for command in commands:
        try:
            proc = subprocess.run(
                command,
                cwd=str(exe.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=25,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            combined += "\n" + (proc.stdout or "")
            if "yue2" in combined.lower():
                return True, "YuE2 loader detected"
        except Exception as exc:
            combined += f"\n{exc}"
    return False, "Downloaded audio.cpp build does not advertise the YuE2 loader"


class SafeComboBox(QComboBox):
    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


class SafeSpinBox(QSpinBox):
    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


class SafeDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):  # noqa: N802
        event.ignore()


class RuntimeInstallerThread(QThread):
    message = Signal(str)
    finished_ok = Signal(bool, str)

    def run(self) -> None:
        try:
            _ensure_dirs()
            ok, why = _runtime_has_yue2(AUDIOCPP_EXE)
            if ok:
                self.finished_ok.emit(True, why)
                return

            self.message.emit("[INSTALL] YuE2-capable audio.cpp runtime not found. Searching downloads...")

            # YuE2 is currently a dev-branch-only feature. Published releases up to
            # v0.7.4 do not contain it, so do NOT waste bandwidth downloading them.
            # Dev artifacts are attempted newest-first. GitHub authenticated artifact
            # download is preferred when a token already exists; otherwise a public
            # nightly.link mirror is used automatically.
            if self._try_dev_actions():
                self.finished_ok.emit(True, "Installed YuE2-capable audio.cpp CUDA runtime from dev artifacts.")
                return

            raise RuntimeError(
                "No working YuE2 dev Windows CUDA artifact could be downloaded. "
                "YuE2 is not in the published audio.cpp releases yet. Open the Logs tab "
                "for the exact GitHub/nightly.link download failure."
            )
        except Exception as exc:
            self.finished_ok.emit(False, str(exc))

    def _try_dev_actions(self) -> bool:
        # The YuE2 model card points at the dev Release workflow. Querying workflow
        # metadata is public; only the archive endpoint itself may require auth.
        urls = [
            f"{GITHUB_API}/actions/workflows/release.yml/runs?branch=dev&status=success&per_page=10",
            f"{GITHUB_API}/actions/runs?branch=dev&status=success&per_page=20",
        ]
        runs: list[dict[str, Any]] = []
        for url in urls:
            try:
                payload = _url_json(url)
                runs = payload.get("workflow_runs", []) if isinstance(payload, dict) else []
                if runs:
                    break
            except Exception as exc:
                self.message.emit(f"[INSTALL] Could not query dev workflow runs: {exc}")

        if not runs:
            self.message.emit("[INSTALL] No successful dev workflow runs were returned by GitHub.")
            return False

        seen_asset_ids: set[tuple[Any, Any]] = set()
        for run in runs:
            run_id = run.get("id")
            if not run_id:
                continue
            try:
                artifacts_payload = _url_json(f"{GITHUB_API}/actions/runs/{run_id}/artifacts?per_page=100")
                artifacts = artifacts_payload.get("artifacts", [])
            except Exception as exc:
                self.message.emit(f"[INSTALL] Could not inspect dev run {run_id}: {exc}")
                continue

            pairs = self._artifact_pairs(artifacts, url_key="archive_download_url")
            if not pairs:
                self.message.emit(f"[INSTALL] Dev run {run_id} has no Windows CUDA artifact pair.")
                continue

            for label, runtime_asset, cuda_asset in pairs:
                key = (runtime_asset.get("id") or runtime_asset.get("name"),
                       cuda_asset.get("id") if cuda_asset else None)
                if key in seen_asset_ids:
                    continue
                seen_asset_ids.add(key)
                self.message.emit(f"[INSTALL] Trying dev run {run_id} CUDA {label}...")
                if self._install_pair(runtime_asset, cuda_asset, f"dev_{run_id}_{label}", run_id=int(run_id)):
                    return True
        return False

    @staticmethod
    def _artifact_pairs(assets: list[dict[str, Any]], url_key: str) -> list[tuple[str, dict[str, Any], Optional[dict[str, Any]]]]:
        # Match each concrete CUDA version exactly. The old implementation treated
        # cuda12.4 as both "12.4" and "12", causing identical multi-GB downloads.
        def cuda_version(name: str) -> str:
            m = re.search(r"cuda(?:rt)?[-_]?([0-9]+(?:\.[0-9]+)?)", name.lower())
            return m.group(1) if m else ""

        runtimes: dict[str, list[dict[str, Any]]] = {}
        cudarts: dict[str, list[dict[str, Any]]] = {}
        broad_runtime: list[dict[str, Any]] = []

        for asset in assets:
            if not asset.get(url_key):
                continue
            name = str(asset.get("name", ""))
            low = name.lower()
            if "windows" not in low or "cuda" not in low:
                continue
            version = cuda_version(low)
            is_cudart = "cudart" in low or "cuda-runtime" in low
            if is_cudart:
                cudarts.setdefault(version, []).append(asset)
            else:
                runtimes.setdefault(version, []).append(asset)
                broad_runtime.append(asset)

        # RTX 3090 / SM86: prefer the CUDA 12.x artifact, then other variants.
        def version_rank(v: str) -> tuple[int, tuple[int, ...]]:
            parts = tuple(int(x) for x in v.split('.') if x.isdigit()) if v else ()
            if v.startswith("12.4"):
                return (0, parts)
            if v.startswith("12"):
                return (1, parts)
            if v.startswith("13"):
                return (2, parts)
            return (3, parts)

        pairs: list[tuple[str, dict[str, Any], Optional[dict[str, Any]]]] = []
        seen: set[Any] = set()
        for version in sorted(runtimes, key=version_rank):
            for runtime in runtimes[version]:
                rid = runtime.get("id") or runtime.get("name")
                if rid in seen:
                    continue
                seen.add(rid)
                cuda = (cudarts.get(version) or [None])[0]
                pairs.append((version or "auto", runtime, cuda))

        if not pairs:
            for runtime in broad_runtime:
                rid = runtime.get("id") or runtime.get("name")
                if rid not in seen:
                    seen.add(rid)
                    pairs.append((cuda_version(str(runtime.get("name", ""))) or "auto", runtime, None))
        return pairs

    def _install_pair(
        self,
        runtime_asset: dict[str, Any],
        cuda_asset: Optional[dict[str, Any]],
        stamp: str,
        *,
        run_id: Optional[int] = None,
    ) -> bool:
        work = TEMP_DIR / f"yue2_audio_cpp_{_safe_filename(stamp)}_{int(time.time())}"
        stage = work / "stage"
        work.mkdir(parents=True, exist_ok=True)
        stage.mkdir(parents=True, exist_ok=True)
        try:
            assets = [runtime_asset] + ([cuda_asset] if cuda_asset else [])
            for index, asset in enumerate(assets):
                if not asset:
                    continue
                name = str(asset.get("name") or f"bundle_{index}.zip")
                url = str(asset.get("archive_download_url") or asset.get("browser_download_url") or "")
                if not url:
                    return False
                archive = work / (name if name.lower().endswith(".zip") else name + ".zip")
                self.message.emit(f"[INSTALL] Downloading {name}")
                downloaded = False
                github_error = None
                try:
                    _download(url, archive, artifact_download=bool(run_id))
                    downloaded = True
                except urllib.error.HTTPError as exc:
                    github_error = f"HTTP {exc.code}"
                except Exception as exc:
                    github_error = str(exc)

                # Public GitHub Actions artifacts commonly return 401 from the API
                # archive endpoint without a token. Fall back to nightly.link, which
                # exposes public-repository artifacts without a GitHub login.
                if not downloaded and run_id:
                    mirror = _nightly_link_url(run_id, name)
                    self.message.emit(
                        f"[INSTALL] GitHub artifact download unavailable ({github_error}); "
                        f"trying public mirror..."
                    )
                    try:
                        _download(mirror, archive)
                        downloaded = True
                    except urllib.error.HTTPError as exc:
                        self.message.emit(f"[INSTALL] Public mirror rejected ({exc.code}) for {name}")
                    except Exception as exc:
                        self.message.emit(f"[INSTALL] Public mirror failed for {name}: {exc}")

                if not downloaded:
                    self.message.emit(f"[INSTALL] Could not download {name}; skipping this dev artifact.")
                    return False
                try:
                    with zipfile.ZipFile(archive, "r") as zf:
                        zf.extractall(stage)
                except zipfile.BadZipFile:
                    self.message.emit(f"[INSTALL] {name} was not a valid ZIP archive")
                    return False
                finally:
                    try:
                        archive.unlink(missing_ok=True)
                    except Exception:
                        pass

            found_exe = _find_first(("audiocpp_cli.exe",), stage)
            if not found_exe:
                self.message.emit("[INSTALL] Bundle did not contain audiocpp_cli.exe")
                return False

            # Replace only after a complete pair has unpacked. Keep no download zips.
            backup = None
            if AUDIOCPP_DIR.exists() and any(AUDIOCPP_DIR.iterdir()):
                backup = TEMP_DIR / f"audio_cpp_backup_{int(time.time())}"
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
                shutil.copytree(AUDIOCPP_DIR, backup)
            shutil.rmtree(AUDIOCPP_DIR, ignore_errors=True)
            AUDIOCPP_DIR.mkdir(parents=True, exist_ok=True)
            _flatten_copy(stage, AUDIOCPP_DIR)

            # If the executable ended up below a wrapper directory, move that runtime
            # tree to the requested canonical /presets/bin/audio location.
            actual_exe = _find_first(("audiocpp_cli.exe",), AUDIOCPP_DIR)
            if actual_exe and actual_exe != AUDIOCPP_EXE:
                runtime_root = actual_exe.parent
                temp_flat = work / "flat"
                temp_flat.mkdir(parents=True, exist_ok=True)
                for item in runtime_root.iterdir():
                    target = temp_flat / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)
                # Also preserve root-level CUDA DLLs that may have unpacked separately.
                for dll in AUDIOCPP_DIR.rglob("*.dll"):
                    target = temp_flat / dll.name
                    if not target.exists():
                        shutil.copy2(dll, target)
                shutil.rmtree(AUDIOCPP_DIR, ignore_errors=True)
                shutil.copytree(temp_flat, AUDIOCPP_DIR)

            ok, reason = _runtime_has_yue2(AUDIOCPP_EXE)
            self.message.emit(f"[INSTALL] Validation: {reason}")
            if ok:
                if backup:
                    shutil.rmtree(backup, ignore_errors=True)
                return True

            # Restore previous runtime if candidate did not support YuE2.
            shutil.rmtree(AUDIOCPP_DIR, ignore_errors=True)
            AUDIOCPP_DIR.mkdir(parents=True, exist_ok=True)
            if backup and backup.exists():
                shutil.copytree(backup, AUDIOCPP_DIR, dirs_exist_ok=True)
                shutil.rmtree(backup, ignore_errors=True)
            return False
        finally:
            shutil.rmtree(work, ignore_errors=True)


class GenerationProcess(QProcess):
    line = Signal(str)
    completed = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProcessChannelMode(QProcess.MergedChannels)
        self.readyReadStandardOutput.connect(self._read_output)
        self.finished.connect(self._finished)
        self.errorOccurred.connect(self._error)
        self._target = ""

    def start_command(self, program: str, args: list[str], target: str) -> None:
        self._target = target
        self.setWorkingDirectory(str(AUDIOCPP_DIR))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PATH", str(AUDIOCPP_DIR) + os.pathsep + env.value("PATH"))
        self.setProcessEnvironment(env)
        self.start(program, args)

    def _read_output(self) -> None:
        data = bytes(self.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in data.replace("\r", "\n").splitlines():
            if line.strip():
                self.line.emit(line.rstrip())

    def _finished(self, exit_code: int, _status) -> None:
        self._read_output()
        self.completed.emit(exit_code == 0, self._target)

    def _error(self, error) -> None:
        self.line.emit(f"[PROCESS] QProcess error: {error}")


class YuE2Window(QMainWindow):
    """Standalone-friendly YuE2 helper. FrameVision can instantiate YuE2Window directly."""

    def __init__(self, parent=None):
        super().__init__(parent)
        _ensure_dirs()
        self.settings = _read_json(SETTINGS_PATH, DEFAULTS)
        auto_model, auto_vae = _auto_model_defaults()
        if not self.settings.get("model_gguf") and auto_model:
            self.settings["model_gguf"] = auto_model
        if not self.settings.get("vae_gguf") and auto_vae:
            self.settings["vae_gguf"] = auto_vae

        self.setWindowTitle("YuE2 Music Generator — audio.cpp")
        self.resize(1040, 820)
        self._log_file = LOG_DIR / f"yue2_{_timestamp()}.log"
        self._installer: Optional[RuntimeInstallerThread] = None
        self._pending_duration_target: Optional[tuple[str, int]] = None
        self._process = GenerationProcess(self)
        self._process.line.connect(self.log)
        self._process.completed.connect(self._generation_done)

        self._build_ui()
        self._load_settings_to_ui()
        self._refresh_runtime_status()
        self.log(f"[YuE2] Helper started | project={PROJECT_ROOT}")
        self.log(f"[YuE2] Log file: {self._log_file}")

        # Requirement: if /presets/bin/audio/audiocpp_cli.exe is missing, bootstrap it.
        if not AUDIOCPP_EXE.exists():
            QTimer.singleShot(0, self.install_runtime)

    # ------------------------------ UI ---------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)

        header = QFrame()
        header_layout = QHBoxLayout(header)
        self.runtime_status = QLabel("audio.cpp: checking...")
        self.runtime_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.install_button = QPushButton("Install / Repair audio.cpp")
        self.install_button.clicked.connect(self.install_runtime)
        header_layout.addWidget(self.runtime_status, 1)
        header_layout.addWidget(self.install_button)
        outer.addWidget(header)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

        self.tabs.addTab(self._build_generate_tab(), "Create Song")
        self.tabs.addTab(self._build_cover_tab(), "Cover / Melody")
        self.tabs.addTab(self._build_score_tab(), "Score / ABC")
        self.tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs.addTab(self._build_log_tab(), "Logs")

        self.setCentralWidget(central)

    @staticmethod
    def _scroll_tab(content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _common_song_widgets(self, prefix: str) -> tuple[QPlainTextEdit, QPlainTextEdit, SafeSpinBox, SafeSpinBox, SafeSpinBox, QLineEdit]:
        lyrics = QPlainTextEdit()
        lyrics.setPlaceholderText("Use section tags such as [Verse] [Chorus] [Bridge]...")
        lyrics.setMinimumHeight(260)

        style = QPlainTextEdit()
        style.setPlaceholderText("Language, genre, instruments, vocal character, production style...")
        style.setMaximumHeight(110)

        steps = SafeSpinBox()
        steps.setRange(1, 100)
        steps.setValue(int(self.settings.get("steps", 32)))

        seed = SafeSpinBox()
        seed.setRange(-1, 2_147_483_647)
        seed.setSpecialValueText("Random (-1)")
        seed.setValue(int(self.settings.get("seed", -1)))

        duration = SafeSpinBox()
        duration.setRange(1, YUE2_MAX_DURATION_SECONDS)
        duration.setSuffix(" s")
        duration.setSingleStep(5)
        duration.setValue(int(self.settings.get("duration_seconds", 180)))
        duration.setToolTip(
            "Target output duration. The helper converts seconds to a forced YuE2 semantic-token "
            "budget with overshoot and trims the rendered WAV to the exact requested sample length."
        )

        name = QLineEdit()
        name.setPlaceholderText(f"{prefix}_YYYYMMDD_HHMMSS.wav")
        return lyrics, style, steps, seed, duration, name

    def _make_instrumental_checkbox(self, lyrics_widget: QPlainTextEdit, settings_key: str) -> QCheckBox:
        check = QCheckBox("Instrumental — no vocals / no singing")
        check.setChecked(bool(self.settings.get(settings_key, False)))
        check.setToolTip(
            "Allow a completely empty Lyrics field. YuE2 receives no lyric text and the helper "
            "adds instrumental / no vocals / no singing to the style request for this run."
        )

        def _apply(enabled: bool) -> None:
            lyrics_widget.setEnabled(not enabled)
            if enabled:
                lyrics_widget.clear()
                lyrics_widget.setPlaceholderText("Instrumental mode: lyrics intentionally blank")
            else:
                lyrics_widget.setPlaceholderText("Use section tags such as [Verse] [Chorus] [Bridge]...")

        check.toggled.connect(_apply)
        _apply(check.isChecked())
        return check

    def _build_generate_tab(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)

        box = QGroupBox("Direct song generation / full planning")
        form = QFormLayout(box)
        self.gen_lyrics, self.gen_style, self.gen_steps, self.gen_seed, self.gen_duration, self.gen_name = self._common_song_widgets("yue2")
        self.gen_instrumental = self._make_instrumental_checkbox(self.gen_lyrics, "instrumental_generate")
        self.gen_mode = SafeComboBox()
        self.gen_mode.addItem("Direct generation — fastest / no planner", "off")
        self.gen_mode.addItem("Full planning — YuE2 plans before generation", "full")
        form.addRow("Mode", self.gen_mode)
        form.addRow("Instrumental", self.gen_instrumental)
        form.addRow("Lyrics", self.gen_lyrics)
        form.addRow("Style", self.gen_style)
        form.addRow("NAR ODE steps", self.gen_steps)
        form.addRow("Seed", self.gen_seed)
        form.addRow("Duration", self.gen_duration)
        form.addRow("Output name", self.gen_name)
        layout.addWidget(box)

        row = QHBoxLayout()
        self.gen_button = QPushButton("Generate Song")
        self.gen_button.clicked.connect(lambda: self._run_mode("generate"))
        row.addStretch(1)
        row.addWidget(self.gen_button)
        layout.addLayout(row)
        layout.addStretch(1)
        return self._scroll_tab(page)

    def _build_cover_tab(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        info = QLabel(
            "Melody-conditioned cover generation uses an ABC melody file (cot=melody). "
            "YuE2 keeps the supplied melody as conditioning while creating a new arrangement/style."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        box = QGroupBox("Melody-conditioned cover")
        form = QFormLayout(box)
        self.cover_lyrics, self.cover_style, self.cover_steps, self.cover_seed, self.cover_duration, self.cover_name = self._common_song_widgets("yue2_cover")
        self.cover_instrumental = self._make_instrumental_checkbox(self.cover_lyrics, "instrumental_cover")
        self.cover_abc = QLineEdit()
        abc_row = QWidget()
        abc_layout = QHBoxLayout(abc_row)
        abc_layout.setContentsMargins(0, 0, 0, 0)
        abc_layout.addWidget(self.cover_abc, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse_abc(self.cover_abc))
        abc_layout.addWidget(browse)
        form.addRow("ABC melody file", abc_row)
        form.addRow("Instrumental", self.cover_instrumental)
        form.addRow("Lyrics", self.cover_lyrics)
        form.addRow("Style", self.cover_style)
        form.addRow("NAR ODE steps", self.cover_steps)
        form.addRow("Seed", self.cover_seed)
        form.addRow("Duration", self.cover_duration)
        form.addRow("Output name", self.cover_name)
        layout.addWidget(box)

        row = QHBoxLayout()
        self.cover_button = QPushButton("Create Cover")
        self.cover_button.clicked.connect(lambda: self._run_mode("cover"))
        row.addStretch(1)
        row.addWidget(self.cover_button)
        layout.addLayout(row)
        layout.addStretch(1)
        return self._scroll_tab(page)

    def _build_score_tab(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        info = QLabel(
            "Full score conditioning uses an ABC score together with full planning (cot=full). "
            "Use this when the musical structure/notes should be conditioned by a complete score rather than only a melody line."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        box = QGroupBox("Full score-conditioned generation")
        form = QFormLayout(box)
        self.score_lyrics, self.score_style, self.score_steps, self.score_seed, self.score_duration, self.score_name = self._common_song_widgets("yue2_score")
        self.score_instrumental = self._make_instrumental_checkbox(self.score_lyrics, "instrumental_score")
        self.score_abc = QLineEdit()
        abc_row = QWidget()
        abc_layout = QHBoxLayout(abc_row)
        abc_layout.setContentsMargins(0, 0, 0, 0)
        abc_layout.addWidget(self.score_abc, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse_abc(self.score_abc))
        abc_layout.addWidget(browse)
        form.addRow("ABC score file", abc_row)
        form.addRow("Instrumental", self.score_instrumental)
        form.addRow("Lyrics", self.score_lyrics)
        form.addRow("Style", self.score_style)
        form.addRow("NAR ODE steps", self.score_steps)
        form.addRow("Seed", self.score_seed)
        form.addRow("Duration", self.score_duration)
        form.addRow("Output name", self.score_name)
        layout.addWidget(box)

        row = QHBoxLayout()
        self.score_button = QPushButton("Generate From Score")
        self.score_button.clicked.connect(lambda: self._run_mode("score"))
        row.addStretch(1)
        row.addWidget(self.score_button)
        layout.addLayout(row)
        layout.addStretch(1)
        return self._scroll_tab(page)

    def _build_settings_tab(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)

        model_box = QGroupBox("YuE2 GGUF files")
        form = QFormLayout(model_box)
        self.model_edit = QLineEdit()
        self.vae_edit = QLineEdit()
        form.addRow("Main GGUF", self._file_picker_row(self.model_edit, "GGUF model", "GGUF files (*.gguf)"))
        form.addRow("VAE GGUF", self._file_picker_row(self.vae_edit, "YuE2 VAE", "GGUF files (*.gguf)"))
        hint = QLabel(f"Default model folder: {MODEL_DIR}")
        hint.setWordWrap(True)
        form.addRow("", hint)
        layout.addWidget(model_box)

        runtime_box = QGroupBox("Runtime")
        form = QFormLayout(runtime_box)
        self.backend_combo = SafeComboBox()
        self.backend_combo.addItems(["cuda", "cpu", "vulkan"])
        self.threads_spin = SafeSpinBox()
        self.threads_spin.setRange(1, 128)
        self.output_edit = QLineEdit()
        self.cli_log_check = QCheckBox("Pass --log to audio.cpp")
        form.addRow("Backend", self.backend_combo)
        form.addRow("CPU threads", self.threads_spin)
        form.addRow("Output folder", self._folder_picker_row(self.output_edit))
        form.addRow("CLI logging", self.cli_log_check)
        layout.addWidget(runtime_box)

        gen_box = QGroupBox("YuE2 generation")
        gen_form = QFormLayout(gen_box)
        self.cfg_auto_check = QCheckBox("Use YuE2 mode default (1.01 direct / 1.0 planned)")
        self.cfg_scale_spin = SafeDoubleSpinBox()
        self.cfg_scale_spin.setRange(0.0, 20.0)
        self.cfg_scale_spin.setDecimals(3)
        self.cfg_scale_spin.setSingleStep(0.01)
        cfg_row = QWidget()
        cfg_layout = QHBoxLayout(cfg_row)
        cfg_layout.setContentsMargins(0, 0, 0, 0)
        cfg_layout.addWidget(self.cfg_auto_check)
        cfg_layout.addWidget(self.cfg_scale_spin)
        gen_form.addRow("Semantic CFG scale", cfg_row)
        ode_note = QLabel("NAR ODE steps are set per generation tab. audio.cpp/YuE2 default is 32; lower values are faster.")
        ode_note.setWordWrap(True)
        gen_form.addRow("", ode_note)
        layout.addWidget(gen_box)

        semantic_box = QGroupBox("Semantic music-token sampling")
        semantic_form = QFormLayout(semantic_box)
        self.semantic_temperature_spin = SafeDoubleSpinBox(); self.semantic_temperature_spin.setRange(0.0, 5.0); self.semantic_temperature_spin.setDecimals(3); self.semantic_temperature_spin.setSingleStep(0.05)
        self.semantic_top_p_spin = SafeDoubleSpinBox(); self.semantic_top_p_spin.setRange(0.0, 1.0); self.semantic_top_p_spin.setDecimals(3); self.semantic_top_p_spin.setSingleStep(0.01)
        self.semantic_top_k_spin = SafeSpinBox(); self.semantic_top_k_spin.setRange(1, 100000)
        self.semantic_rep_spin = SafeDoubleSpinBox(); self.semantic_rep_spin.setRange(0.001, 10.0); self.semantic_rep_spin.setDecimals(4); self.semantic_rep_spin.setSingleStep(0.01)
        self.semantic_window_spin = SafeSpinBox(); self.semantic_window_spin.setRange(1, 100000)
        self.semantic_min_tokens_spin = SafeSpinBox(); self.semantic_min_tokens_spin.setRange(0, 24576)
        self.semantic_max_tokens_spin = SafeSpinBox(); self.semantic_max_tokens_spin.setRange(1, 24576)
        semantic_form.addRow("Temperature", self.semantic_temperature_spin)
        semantic_form.addRow("Top P", self.semantic_top_p_spin)
        semantic_form.addRow("Top K", self.semantic_top_k_spin)
        semantic_form.addRow("Repetition penalty", self.semantic_rep_spin)
        semantic_form.addRow("Penalty window", self.semantic_window_spin)
        semantic_form.addRow("Minimum tokens", self.semantic_min_tokens_spin)
        semantic_form.addRow("Maximum tokens", self.semantic_max_tokens_spin)
        semantic_hint = QLabel("When a song is generated, the Duration control overrides semantic min/max tokens for that run. The helper forces enough semantic tokens (with overshoot) and trims the decoded WAV to the exact requested duration. Manual token values remain saved here for advanced/reference use.")
        semantic_hint.setWordWrap(True)
        semantic_form.addRow("", semantic_hint)
        layout.addWidget(semantic_box)

        abc_box = QGroupBox("ABC planner sampling")
        abc_form = QFormLayout(abc_box)
        self.abc_temperature_spin = SafeDoubleSpinBox(); self.abc_temperature_spin.setRange(0.0, 5.0); self.abc_temperature_spin.setDecimals(3); self.abc_temperature_spin.setSingleStep(0.05)
        self.abc_top_p_spin = SafeDoubleSpinBox(); self.abc_top_p_spin.setRange(0.0, 1.0); self.abc_top_p_spin.setDecimals(3); self.abc_top_p_spin.setSingleStep(0.01)
        self.abc_top_k_spin = SafeSpinBox(); self.abc_top_k_spin.setRange(1, 100000)
        self.abc_rep_spin = SafeDoubleSpinBox(); self.abc_rep_spin.setRange(0.001, 10.0); self.abc_rep_spin.setDecimals(4); self.abc_rep_spin.setSingleStep(0.001)
        self.abc_window_spin = SafeSpinBox(); self.abc_window_spin.setRange(1, 100000)
        self.abc_min_tokens_spin = SafeSpinBox(); self.abc_min_tokens_spin.setRange(0, 24576)
        self.abc_max_tokens_spin = SafeSpinBox(); self.abc_max_tokens_spin.setRange(1, 24576)
        abc_form.addRow("Temperature", self.abc_temperature_spin)
        abc_form.addRow("Top P", self.abc_top_p_spin)
        abc_form.addRow("Top K", self.abc_top_k_spin)
        abc_form.addRow("Repetition penalty", self.abc_rep_spin)
        abc_form.addRow("Penalty window", self.abc_window_spin)
        abc_form.addRow("Minimum tokens", self.abc_min_tokens_spin)
        abc_form.addRow("Maximum tokens", self.abc_max_tokens_spin)
        abc_hint = QLabel("These controls affect YuE2 symbolic ABC planning and are mainly relevant to Full Planning / Melody / Score modes.")
        abc_hint.setWordWrap(True)
        abc_form.addRow("", abc_hint)
        layout.addWidget(abc_box)

        advanced_runtime_box = QGroupBox("Advanced YuE2 runtime / memory")
        advanced_runtime_form = QFormLayout(advanced_runtime_box)
        self.weight_type_combo = SafeComboBox(); self.weight_type_combo.addItems(["native", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k"])
        self.model_weight_type_combo = SafeComboBox(); self.model_weight_type_combo.addItems(["native", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k"])
        self.vae_weight_type_combo = SafeComboBox(); self.vae_weight_type_combo.addItems(["native", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k"])
        self.model_weight_context_spin = SafeSpinBox(); self.model_weight_context_spin.setRange(1, 65536)
        self.vae_weight_context_spin = SafeSpinBox(); self.vae_weight_context_spin.setRange(1, 65536)
        self.ar_prefill_arena_spin = SafeSpinBox(); self.ar_prefill_arena_spin.setRange(1, 65536)
        self.ar_decode_arena_spin = SafeSpinBox(); self.ar_decode_arena_spin.setRange(1, 65536)
        self.nar_arena_spin = SafeSpinBox(); self.nar_arena_spin.setRange(1, 65536)
        self.vae_arena_spin = SafeSpinBox(); self.vae_arena_spin.setRange(1, 65536)
        advanced_runtime_form.addRow("Shared weight type", self.weight_type_combo)
        advanced_runtime_form.addRow("Main model weight type", self.model_weight_type_combo)
        advanced_runtime_form.addRow("VAE weight type", self.vae_weight_type_combo)
        advanced_runtime_form.addRow("Main weight context (MiB)", self.model_weight_context_spin)
        advanced_runtime_form.addRow("VAE weight context (MiB)", self.vae_weight_context_spin)
        advanced_runtime_form.addRow("AR prefill graph arena (MiB)", self.ar_prefill_arena_spin)
        advanced_runtime_form.addRow("AR decode graph arena (MiB)", self.ar_decode_arena_spin)
        advanced_runtime_form.addRow("NAR graph arena (MiB)", self.nar_arena_spin)
        advanced_runtime_form.addRow("VAE graph arena (MiB)", self.vae_arena_spin)
        runtime_hint = QLabel("These are official audio.cpp YuE2 session options. Leave weight types on native to use the precision stored in the selected GGUF files. Change memory arenas only when tuning runtime memory behaviour or troubleshooting allocation failures.")
        runtime_hint.setWordWrap(True)
        advanced_runtime_form.addRow("", runtime_hint)
        layout.addWidget(advanced_runtime_box)

        install_box = QGroupBox("audio.cpp runtime")
        install_layout = QVBoxLayout(install_box)
        self.runtime_path_label = QLabel(str(AUDIOCPP_EXE))
        self.runtime_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.runtime_path_label.setWordWrap(True)
        install_layout.addWidget(self.runtime_path_label)
        install_note = QLabel(
            "The helper prefers a CUDA 12.4 Windows build because current audio.cpp CI explicitly includes SM86. "
            "Downloaded ZIP files are deleted after extraction. Candidates are validated for the YuE2 loader before they are accepted."
        )
        install_note.setWordWrap(True)
        install_layout.addWidget(install_note)
        buttons = QHBoxLayout()
        repair = QPushButton("Install / Repair audio.cpp")
        repair.clicked.connect(self.install_runtime)
        open_runtime = QPushButton("Open Runtime Folder")
        open_runtime.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(AUDIOCPP_DIR))))
        buttons.addWidget(repair)
        buttons.addWidget(open_runtime)
        buttons.addStretch(1)
        install_layout.addLayout(buttons)
        layout.addWidget(install_box)

        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save = QPushButton("Save Settings")
        save.clicked.connect(self.save_settings)
        save_row.addWidget(save)
        layout.addLayout(save_row)
        layout.addStretch(1)
        return self._scroll_tab(page)

    def _build_log_tab(self) -> QScrollArea:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_view.setMinimumHeight(520)
        layout.addWidget(self.log_view, 1)
        row = QHBoxLayout()
        open_logs = QPushButton("Open Logs Folder")
        open_logs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(LOG_DIR))))
        clear = QPushButton("Clear View")
        clear.clicked.connect(self.log_view.clear)
        row.addWidget(open_logs)
        row.addWidget(clear)
        row.addStretch(1)
        layout.addLayout(row)
        return self._scroll_tab(page)

    def _file_picker_row(self, edit: QLineEdit, caption: str, filter_text: str) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse_file(edit, caption, filter_text))
        row.addWidget(button)
        return holder

    def _folder_picker_row(self, edit: QLineEdit) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: self._browse_folder(edit))
        row.addWidget(button)
        return holder

    # --------------------------- settings -------------------------------

    def _load_settings_to_ui(self) -> None:
        self.model_edit.setText(str(self.settings.get("model_gguf", "")))
        self.vae_edit.setText(str(self.settings.get("vae_gguf", "")))
        backend = str(self.settings.get("backend", "cuda"))
        idx = self.backend_combo.findText(backend)
        self.backend_combo.setCurrentIndex(max(0, idx))
        self.threads_spin.setValue(int(self.settings.get("threads", 8)))
        self.output_edit.setText(str(self.settings.get("output_dir", OUTPUT_DIR)))
        self.cli_log_check.setChecked(bool(self.settings.get("log_cli", True)))
        self.cfg_auto_check.setChecked(bool(self.settings.get("cfg_auto", True)))
        self.cfg_scale_spin.setValue(float(self.settings.get("cfg_scale", 1.01)))
        self.abc_temperature_spin.setValue(float(self.settings.get("abc_temperature", 0.7)))
        self.abc_top_p_spin.setValue(float(self.settings.get("abc_top_p", 0.9)))
        self.abc_top_k_spin.setValue(int(self.settings.get("abc_top_k", 30)))
        self.abc_rep_spin.setValue(float(self.settings.get("abc_repetition_penalty", 1.005)))
        self.abc_window_spin.setValue(int(self.settings.get("abc_penalty_window", 100)))
        self.abc_min_tokens_spin.setValue(int(self.settings.get("abc_min_tokens", 32)))
        self.abc_max_tokens_spin.setValue(int(self.settings.get("abc_max_tokens", 4096)))
        self.semantic_temperature_spin.setValue(float(self.settings.get("semantic_temperature", 1.0)))
        self.semantic_top_p_spin.setValue(float(self.settings.get("semantic_top_p", 0.95)))
        self.semantic_top_k_spin.setValue(int(self.settings.get("semantic_top_k", 100)))
        self.semantic_rep_spin.setValue(float(self.settings.get("semantic_repetition_penalty", 1.2)))
        self.semantic_window_spin.setValue(int(self.settings.get("semantic_penalty_window", 50)))
        self.semantic_min_tokens_spin.setValue(int(self.settings.get("semantic_min_tokens", 200)))
        self.semantic_max_tokens_spin.setValue(int(self.settings.get("semantic_max_tokens", 9000)))
        for combo, key in ((self.weight_type_combo, "weight_type"), (self.model_weight_type_combo, "model_weight_type"), (self.vae_weight_type_combo, "vae_weight_type")):
            value = str(self.settings.get(key, "native"))
            idx = combo.findText(value)
            combo.setCurrentIndex(max(0, idx))
        self.model_weight_context_spin.setValue(int(self.settings.get("model_weight_context_mb", 6144)))
        self.vae_weight_context_spin.setValue(int(self.settings.get("vae_weight_context_mb", 1536)))
        self.ar_prefill_arena_spin.setValue(int(self.settings.get("ar_prefill_graph_arena_mb", 4096)))
        self.ar_decode_arena_spin.setValue(int(self.settings.get("ar_decode_graph_arena_mb", 1536)))
        self.nar_arena_spin.setValue(int(self.settings.get("nar_graph_arena_mb", 6144)))
        self.vae_arena_spin.setValue(int(self.settings.get("vae_graph_arena_mb", 1536)))

        lyrics = str(self.settings.get("last_lyrics", DEFAULT_LYRICS))
        style = str(self.settings.get("last_style", DEFAULT_STYLE))
        abc = str(self.settings.get("last_abc", ""))
        for widget in (self.gen_lyrics, self.cover_lyrics, self.score_lyrics):
            widget.setPlainText(lyrics)
        for widget in (self.gen_style, self.cover_style, self.score_style):
            widget.setPlainText(style)
        self.cover_abc.setText(abc)
        self.score_abc.setText(abc)

    def save_settings(self, quiet: bool = False) -> None:
        source_lyrics = self.gen_lyrics.toPlainText().strip()
        source_style = self.gen_style.toPlainText().strip()
        self.settings.update({
            "model_gguf": self.model_edit.text().strip(),
            "vae_gguf": self.vae_edit.text().strip(),
            "threads": self.threads_spin.value(),
            "steps": self.gen_steps.value(),
            "seed": self.gen_seed.value(),
            "duration_seconds": self.gen_duration.value(),
            "instrumental_generate": self.gen_instrumental.isChecked(),
            "instrumental_cover": self.cover_instrumental.isChecked(),
            "instrumental_score": self.score_instrumental.isChecked(),
            "backend": self.backend_combo.currentText(),
            "output_dir": self.output_edit.text().strip() or str(OUTPUT_DIR),
            "last_abc": self.cover_abc.text().strip() or self.score_abc.text().strip(),
            "last_style": source_style,
            "last_lyrics": source_lyrics,
            "log_cli": self.cli_log_check.isChecked(),
            "cfg_auto": self.cfg_auto_check.isChecked(),
            "cfg_scale": self.cfg_scale_spin.value(),
            "abc_temperature": self.abc_temperature_spin.value(),
            "abc_top_p": self.abc_top_p_spin.value(),
            "abc_top_k": self.abc_top_k_spin.value(),
            "abc_repetition_penalty": self.abc_rep_spin.value(),
            "abc_penalty_window": self.abc_window_spin.value(),
            "abc_min_tokens": self.abc_min_tokens_spin.value(),
            "abc_max_tokens": self.abc_max_tokens_spin.value(),
            "semantic_temperature": self.semantic_temperature_spin.value(),
            "semantic_top_p": self.semantic_top_p_spin.value(),
            "semantic_top_k": self.semantic_top_k_spin.value(),
            "semantic_repetition_penalty": self.semantic_rep_spin.value(),
            "semantic_penalty_window": self.semantic_window_spin.value(),
            "semantic_min_tokens": duration_tokens,
            "semantic_max_tokens": duration_tokens,
            "weight_type": self.weight_type_combo.currentText(),
            "model_weight_type": self.model_weight_type_combo.currentText(),
            "vae_weight_type": self.vae_weight_type_combo.currentText(),
            "model_weight_context_mb": self.model_weight_context_spin.value(),
            "vae_weight_context_mb": self.vae_weight_context_spin.value(),
            "ar_prefill_graph_arena_mb": self.ar_prefill_arena_spin.value(),
            "ar_decode_graph_arena_mb": self.ar_decode_arena_spin.value(),
            "nar_graph_arena_mb": self.nar_arena_spin.value(),
            "vae_graph_arena_mb": self.vae_arena_spin.value(),
        })
        try:
            _write_json_atomic(SETTINGS_PATH, self.settings)
            self.log(f"[SETTINGS] Saved: {SETTINGS_PATH}")
            if not quiet:
                QMessageBox.information(self, "YuE2", "Settings saved.")
        except Exception as exc:
            self.log(f"[SETTINGS] Save failed: {exc}")
            if not quiet:
                QMessageBox.critical(self, "YuE2", f"Could not save settings:\n{exc}")

    # --------------------------- logging --------------------------------

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        if hasattr(self, "log_view"):
            self.log_view.appendPlainText(line)
            bar = self.log_view.verticalScrollBar()
            bar.setValue(bar.maximum())
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with self._log_file.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

    # --------------------------- runtime --------------------------------

    def _refresh_runtime_status(self) -> None:
        ok, reason = _runtime_has_yue2(AUDIOCPP_EXE)
        if ok:
            self.runtime_status.setText(f"audio.cpp: READY — {AUDIOCPP_EXE}")
            self.runtime_status.setStyleSheet("color: #58d68d;")
        elif AUDIOCPP_EXE.exists():
            self.runtime_status.setText(f"audio.cpp: found but YuE2 unavailable — {reason}")
            self.runtime_status.setStyleSheet("color: #f5b041;")
        else:
            self.runtime_status.setText(f"audio.cpp: NOT INSTALLED — expected {AUDIOCPP_EXE}")
            self.runtime_status.setStyleSheet("color: #ec7063;")

    def install_runtime(self) -> None:
        if self._installer and self._installer.isRunning():
            return
        self.install_button.setEnabled(False)
        self.install_button.setText("Installing...")
        self._installer = RuntimeInstallerThread(self)
        self._installer.message.connect(self.log)
        self._installer.finished_ok.connect(self._install_done)
        self._installer.start()

    def _install_done(self, ok: bool, message: str) -> None:
        self.log(f"[INSTALL] {message}")
        self.install_button.setEnabled(True)
        self.install_button.setText("Install / Repair audio.cpp")
        self._refresh_runtime_status()
        if not ok:
            QMessageBox.warning(
                self,
                "YuE2 audio.cpp installation",
                message
                + "\n\nThe helper did not accept a build without verified YuE2 support. "
                  "You can install a YuE2-capable dev CUDA build manually into:\n"
                + str(AUDIOCPP_DIR),
            )

    # --------------------------- browsing -------------------------------

    def _browse_file(self, edit: QLineEdit, caption: str, filter_text: str) -> None:
        start = edit.text().strip() or str(MODEL_DIR)
        filename, _ = QFileDialog.getOpenFileName(self, caption, start, filter_text)
        if filename:
            edit.setText(filename)

    def _browse_folder(self, edit: QLineEdit) -> None:
        start = edit.text().strip() or str(OUTPUT_DIR)
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if folder:
            edit.setText(folder)

    def _browse_abc(self, edit: QLineEdit) -> None:
        start = edit.text().strip() or str(MODEL_DIR)
        filename, _ = QFileDialog.getOpenFileName(self, "Select ABC notation file", start, "ABC notation (*.abc);;All files (*.*)")
        if filename:
            edit.setText(filename)

    # -------------------------- generation ------------------------------

    def _run_mode(self, mode: str) -> None:
        if self._process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "YuE2", "A YuE2 generation is already running.")
            return

        ok, reason = _runtime_has_yue2(AUDIOCPP_EXE)
        if not ok:
            self.log(f"[RUN] Runtime unavailable: {reason}")
            self.install_runtime()
            QMessageBox.warning(self, "YuE2", "A YuE2-capable audio.cpp runtime is required. Installation/repair has started.")
            return

        model = Path(self.model_edit.text().strip())
        vae = Path(self.vae_edit.text().strip())
        if not model.is_file():
            QMessageBox.warning(self, "YuE2", f"Select a valid YuE2 main GGUF file.\n\nDefault folder:\n{MODEL_DIR}")
            self.tabs.setCurrentIndex(3)
            return
        if not vae.is_file():
            QMessageBox.warning(self, "YuE2", f"Select a valid YuE2 VAE GGUF file.\n\nDefault folder:\n{MODEL_DIR}")
            self.tabs.setCurrentIndex(3)
            return
        if model.parent != vae.parent:
            self.log("[RUN] Warning: main GGUF and VAE are in different directories. YuE2 sidecars are resolved from the main model directory.")

        if mode == "generate":
            lyrics = self.gen_lyrics.toPlainText().strip()
            style = self.gen_style.toPlainText().strip()
            steps = self.gen_steps.value()
            seed = self.gen_seed.value()
            duration_seconds = self.gen_duration.value()
            cot = str(self.gen_mode.currentData())
            abc_file = ""
            requested_name = self.gen_name.text().strip()
            instrumental = self.gen_instrumental.isChecked()
            label = "yue2"
        elif mode == "cover":
            lyrics = self.cover_lyrics.toPlainText().strip()
            style = self.cover_style.toPlainText().strip()
            steps = self.cover_steps.value()
            seed = self.cover_seed.value()
            duration_seconds = self.cover_duration.value()
            cot = "melody"
            abc_file = self.cover_abc.text().strip()
            requested_name = self.cover_name.text().strip()
            instrumental = self.cover_instrumental.isChecked()
            label = "yue2_cover"
        else:
            lyrics = self.score_lyrics.toPlainText().strip()
            style = self.score_style.toPlainText().strip()
            steps = self.score_steps.value()
            seed = self.score_seed.value()
            duration_seconds = self.score_duration.value()
            cot = "full"
            abc_file = self.score_abc.text().strip()
            requested_name = self.score_name.text().strip()
            instrumental = self.score_instrumental.isChecked()
            label = "yue2_score"

        if not lyrics and not instrumental:
            QMessageBox.warning(self, "YuE2", "Lyrics cannot be empty unless Instrumental mode is enabled.")
            return
        if instrumental:
            lyrics = ""
            style_lower = style.lower()
            instrumental_tags = []
            if "instrumental" not in style_lower:
                instrumental_tags.append("instrumental")
            if "no vocals" not in style_lower:
                instrumental_tags.append("no vocals")
            if "no singing" not in style_lower:
                instrumental_tags.append("no singing")
            if instrumental_tags:
                style = (style.rstrip(" ,") + ", " + ", ".join(instrumental_tags)).strip(", ")
        if not style:
            QMessageBox.warning(self, "YuE2", "Style cannot be empty.")
            return
        if instrumental:
            self.log("[RUN] Instrumental mode enabled: sending empty lyrics and no-vocal style guidance.")
        if abc_file and not Path(abc_file).is_file():
            QMessageBox.warning(self, "YuE2", "The selected ABC file does not exist.")
            return
        if mode in {"cover", "score"} and not abc_file:
            QMessageBox.warning(self, "YuE2", "This mode requires an ABC file.")
            return

        output_dir = Path(self.output_edit.text().strip() or OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        if requested_name:
            output_name = requested_name if requested_name.lower().endswith(".wav") else requested_name + ".wav"
        else:
            output_name = f"{label}_{_timestamp()}.wav"
        output_path = output_dir / output_name

        if self.semantic_max_tokens_spin.value() < self.semantic_min_tokens_spin.value():
            QMessageBox.warning(self, "YuE2", "Semantic maximum tokens must be greater than or equal to minimum tokens.")
            return
        if self.abc_max_tokens_spin.value() < self.abc_min_tokens_spin.value():
            QMessageBox.warning(self, "YuE2", "ABC maximum tokens must be greater than or equal to minimum tokens.")
            return

        actual_seed = seed
        if seed < 0:
            actual_seed = int.from_bytes(os.urandom(4), "little") & 0x7FFFFFFF

        # audio.cpp Yue2 takes the package root as --model and the chosen GGUF
        # files through yue2.model_gguf / yue2.vae_gguf session options.
        args = [
            "--task", "gen",
            "--family", "yue2",
            "--model", str(model.parent),
            "--backend", self.backend_combo.currentText(),
            "--threads", str(self.threads_spin.value()),
            "--lyrics", lyrics,
            "--request-option", f"style={style}",
            "--request-option", f"cot={cot}",
            "--request-option", f"num_inference_steps={steps}",
            "--seed", str(actual_seed),
            "--session-option", f"yue2.model_gguf={model.name}",
            "--session-option", f"yue2.vae_gguf={vae.name}",
            "--session-option", f"yue2.weight_type={self.weight_type_combo.currentText()}",
            "--session-option", f"yue2.model_weight_type={self.model_weight_type_combo.currentText()}",
            "--session-option", f"yue2.vae_weight_type={self.vae_weight_type_combo.currentText()}",
            "--session-option", f"yue2.model_weight_context_mb={self.model_weight_context_spin.value()}",
            "--session-option", f"yue2.vae_weight_context_mb={self.vae_weight_context_spin.value()}",
            "--session-option", f"yue2.ar_prefill_graph_arena_mb={self.ar_prefill_arena_spin.value()}",
            "--session-option", f"yue2.ar_decode_graph_arena_mb={self.ar_decode_arena_spin.value()}",
            "--session-option", f"yue2.nar_graph_arena_mb={self.nar_arena_spin.value()}",
            "--session-option", f"yue2.vae_graph_arena_mb={self.vae_arena_spin.value()}",
            "--out", str(output_path),
        ]
        if abc_file:
            args.extend(["--request-option", f"abc_file={abc_file}"])

        # Official YuE2 request options from audio.cpp dev docs. Keep these as
        # explicit request options so the GUI exactly reflects the runtime API.
        if not self.cfg_auto_check.isChecked():
            args.extend(["--request-option", f"cfg_scale={self.cfg_scale_spin.value():.6g}"])
        # Duration controller: YuE2 is ~50 semantic tokens/second.  Force the
        # semantic stage to run slightly beyond the requested duration, then
        # sample-trim the final WAV in _generation_done().
        duration_tokens = _duration_token_budget(duration_seconds)
        self.log(
            f"[RUN] target duration={duration_seconds}s -> forced semantic token budget={duration_tokens} "
            f"(50 tok/s + {int((YUE2_DURATION_OVERSHOOT - 1.0) * 100)}% overshoot)"
        )

        advanced = {
            "abc_temperature": self.abc_temperature_spin.value(),
            "abc_top_p": self.abc_top_p_spin.value(),
            "abc_top_k": self.abc_top_k_spin.value(),
            "abc_repetition_penalty": self.abc_rep_spin.value(),
            "abc_penalty_window": self.abc_window_spin.value(),
            "abc_min_tokens": self.abc_min_tokens_spin.value(),
            "abc_max_tokens": self.abc_max_tokens_spin.value(),
            "semantic_temperature": self.semantic_temperature_spin.value(),
            "semantic_top_p": self.semantic_top_p_spin.value(),
            "semantic_top_k": self.semantic_top_k_spin.value(),
            "semantic_repetition_penalty": self.semantic_rep_spin.value(),
            "semantic_penalty_window": self.semantic_window_spin.value(),
            "semantic_min_tokens": self.semantic_min_tokens_spin.value(),
            "semantic_max_tokens": self.semantic_max_tokens_spin.value(),
        }
        for key, value in advanced.items():
            if isinstance(value, float):
                value = f"{value:.6g}"
            args.extend(["--request-option", f"{key}={value}"])

        if self.cli_log_check.isChecked():
            args.append("--log")

        self.settings.update({
            "model_gguf": str(model),
            "vae_gguf": str(vae),
            "threads": self.threads_spin.value(),
            "steps": steps,
            "seed": seed,
            "duration_seconds": duration_seconds,
            "backend": self.backend_combo.currentText(),
            "output_dir": str(output_dir),
            "last_abc": abc_file,
            "last_style": style,
            "last_lyrics": lyrics,
            "log_cli": self.cli_log_check.isChecked(),
            "cfg_auto": self.cfg_auto_check.isChecked(),
            "cfg_scale": self.cfg_scale_spin.value(),
            "abc_temperature": self.abc_temperature_spin.value(),
            "abc_top_p": self.abc_top_p_spin.value(),
            "abc_top_k": self.abc_top_k_spin.value(),
            "abc_repetition_penalty": self.abc_rep_spin.value(),
            "abc_penalty_window": self.abc_window_spin.value(),
            "abc_min_tokens": self.abc_min_tokens_spin.value(),
            "abc_max_tokens": self.abc_max_tokens_spin.value(),
            "semantic_temperature": self.semantic_temperature_spin.value(),
            "semantic_top_p": self.semantic_top_p_spin.value(),
            "semantic_top_k": self.semantic_top_k_spin.value(),
            "semantic_repetition_penalty": self.semantic_rep_spin.value(),
            "semantic_penalty_window": self.semantic_window_spin.value(),
            "semantic_min_tokens": self.semantic_min_tokens_spin.value(),
            "semantic_max_tokens": self.semantic_max_tokens_spin.value(),
            "weight_type": self.weight_type_combo.currentText(),
            "model_weight_type": self.model_weight_type_combo.currentText(),
            "vae_weight_type": self.vae_weight_type_combo.currentText(),
            "model_weight_context_mb": self.model_weight_context_spin.value(),
            "vae_weight_context_mb": self.vae_weight_context_spin.value(),
            "ar_prefill_graph_arena_mb": self.ar_prefill_arena_spin.value(),
            "ar_decode_graph_arena_mb": self.ar_decode_arena_spin.value(),
            "nar_graph_arena_mb": self.nar_arena_spin.value(),
            "vae_graph_arena_mb": self.vae_arena_spin.value(),
        })
        _write_json_atomic(SETTINGS_PATH, self.settings)

        self.log(f"[RUN] mode={mode} cot={cot} backend={self.backend_combo.currentText()} ode_steps={steps} seed={actual_seed}")
        self.log(
            f"[RUN] semantic temp={self.semantic_temperature_spin.value()} top_p={self.semantic_top_p_spin.value()} "
            f"top_k={self.semantic_top_k_spin.value()} rep={self.semantic_rep_spin.value()} "
            f"tokens={duration_tokens}..{duration_tokens} (duration managed)"
        )
        if cot != "off":
            self.log(
                f"[RUN] abc temp={self.abc_temperature_spin.value()} top_p={self.abc_top_p_spin.value()} "
                f"top_k={self.abc_top_k_spin.value()} rep={self.abc_rep_spin.value()} "
                f"tokens={self.abc_min_tokens_spin.value()}..{self.abc_max_tokens_spin.value()}"
            )
        self.log(f"[RUN] model={model}")
        self.log(f"[RUN] vae={vae}")
        if abc_file:
            self.log(f"[RUN] abc={abc_file}")
        self.log(f"[RUN] output={output_path}")
        self._pending_duration_target = (str(output_path), int(duration_seconds))
        self._set_generation_enabled(False)
        self._process.start_command(str(AUDIOCPP_EXE), args, str(output_path))

    def _set_generation_enabled(self, enabled: bool) -> None:
        self.gen_button.setEnabled(enabled)
        self.cover_button.setEnabled(enabled)
        self.score_button.setEnabled(enabled)

    def _generation_done(self, ok: bool, target: str) -> None:
        self._set_generation_enabled(True)
        if ok and Path(target).is_file():
            requested_duration = None
            if self._pending_duration_target and self._pending_duration_target[0] == target:
                requested_duration = self._pending_duration_target[1]
            self._pending_duration_target = None

            if requested_duration is not None:
                trim_ok, trim_detail = _trim_wav_exact(Path(target), requested_duration)
                if trim_ok:
                    self.log(f"[DURATION] Exact output length applied: {trim_detail}")
                else:
                    self.log(f"[DURATION] WARNING: could not enforce exact {requested_duration}s: {trim_detail}")
                    QMessageBox.warning(
                        self,
                        "YuE2 duration",
                        f"Generation finished, but exact-duration trimming failed.\n\n"
                        f"Requested: {requested_duration} seconds\n"
                        f"Reason: {trim_detail}\n\n"
                        f"The generated WAV was kept.",
                    )

            self.log(f"[DONE] YuE2 generation finished: {target}")
            QMessageBox.information(self, "YuE2", f"Generation complete:\n{target}")
        elif ok:
            self.log(f"[DONE] Process returned success but output was not found: {target}")
            QMessageBox.warning(self, "YuE2", "audio.cpp returned success but the expected WAV file was not found. Check the Logs tab.")
        else:
            self._pending_duration_target = None
            self.log("[ERROR] YuE2 generation failed. See log output above.")
            QMessageBox.critical(self, "YuE2", "Generation failed. Check the Logs tab and the saved log file.")

    def closeEvent(self, event):  # noqa: N802
        try:
            self.save_settings(quiet=True)
        except Exception:
            pass
        if self._process.state() != QProcess.NotRunning:
            self._process.kill()
            self._process.waitForFinished(3000)
        super().closeEvent(event)


# Friendly aliases for FrameVision import patterns.
Yue2Window = YuE2Window
Yue2Helper = YuE2Window


def create_window(parent=None) -> YuE2Window:
    return YuE2Window(parent)


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    window = YuE2Window()
    window.show()
    sys.exit(app.exec())
