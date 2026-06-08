#!/usr/bin/env python3
"""
Whisper Lab (Faster-Whisper) with logging
-----------------------------------------

PySide6 widget that wraps Faster-Whisper for:

- Lyrics / transcript extraction from audio or video
- Subtitle generation (SRT / VTT)
- Optional TXT "lyrics" file
- Subtitle burn-in / embedding into a video using ffmpeg

Assumptions / integration points:

- Runs inside a project where this file lives in:  <project_root>/helpers/whisper.py
- Faster-Whisper model is stored locally at:
      <project_root>/models/faster_whisper/medium/
- ffmpeg / ffprobe live in:
      <project_root>/presets/bin/ffmpeg(.exe)
      <project_root>/presets/bin/ffprobe(.exe)

The widget is self-contained and can also be run standalone:

    python helpers/whisper.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# OpenMP safety switch
# ---------------------------------------------------------------------------
# FrameVision already pulls in other libraries that use Intel OpenMP (libiomp5md.dll).
# When Faster-Whisper (via CTranslate2 / onnxruntime) is imported in the same process,
# Windows sometimes loads a second copy of that DLL, which triggers:
#
#   OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#
# The official workaround from Intel is to set KMP_DUPLICATE_LIB_OK=TRUE. This allows
# multiple OpenMP runtimes in one process. It is not "perfectly safe" in theory, but it
# is commonly used in Python apps that mix PyTorch + other OpenMP libs.
#
# We only set this env var here if it is NOT already set by the parent process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QFileDialog,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Paths, logging & settings helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent  # helpers/ -> project root


ROOT = _project_root()
PRESETS_DIR = ROOT / "presets"
MODELS_DIR = ROOT / "models"
FWHISPER_DIR = MODELS_DIR / "faster_whisper"
FWHISPER_MEDIUM_DIR = FWHISPER_DIR / "medium"
SETTINGS_PATH = PRESETS_DIR / "whisper_settings.json"

# Logging setup
LOG_DIR = ROOT / "output" / "logs"
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    # Fallback to project root if something goes wrong
    LOG_DIR = ROOT

logger = logging.getLogger("whisper_lab")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    log_file = LOG_DIR / "whisper_lab.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

logger.debug("Whisper Lab module imported. ROOT=%s", ROOT)


# ---------------------------------------------------------------------------
# Optional external venv loader (FrameVision Optional Installs style)
# ---------------------------------------------------------------------------
# FrameVision can keep Whisper in its own isolated venv at:
#     <project_root>/environments/.whisper/
#
# To avoid dependency conflicts inside the main app environment, we try to
# dynamically add that venv's site-packages to sys.path before importing
# faster-whisper. This keeps installation "one click" while letting the
# main app remain lean.
# ---------------------------------------------------------------------------
# Whisper environment helpers
# ---------------------------------------------------------------------------
# FrameVision can keep Whisper in its own isolated venv at:
#     <project_root>/environments/.whisper/
#
# For stability (and to avoid DLL conflicts), we prefer running transcription
# in a separate process using that venv's python.exe.
ENV_DIR = ROOT / "environments" / ".whisper"

def _whisper_env_python() -> Path:
    return ENV_DIR / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")

def _has_isolated_whisper_env() -> bool:
    try:
        return _whisper_env_python().is_file()
    except Exception:
        return False

# Lazy import for in-process mode (standalone usage / no venv available)
_IMPORT_ERROR: Optional[str] = None
WhisperModel = None  # type: ignore

def _inject_whisper_venv_into_process() -> None:
    """Best-effort add the Whisper venv (if present) to sys.path + PATH for in-process imports."""
    try:
        if not ENV_DIR.exists():
            return

        scripts_dir = ENV_DIR / ("Scripts" if os.name == "nt" else "bin")
        if scripts_dir.is_dir():
            os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")

        candidates = []
        if os.name == "nt":
            candidates.append(ENV_DIR / "Lib" / "site-packages")
        else:
            lib_dir = ENV_DIR / "lib"
            if lib_dir.is_dir():
                for p in lib_dir.glob("python*/site-packages"):
                    candidates.append(p)

        for sp in candidates:
            if sp.is_dir() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))
                logger.info("Injected Whisper venv site-packages (in-process): %s", sp)
                break
    except Exception as e:
        logger.warning("Whisper venv injection failed: %s", e)

def _lazy_import_faster_whisper() -> bool:
    """Import faster_whisper only when needed (avoid loading native libs in the main app unless required)."""
    global WhisperModel, _IMPORT_ERROR
    if WhisperModel is not None:
        return True
    try:
        _inject_whisper_venv_into_process()
        from faster_whisper import WhisperModel as _WM  # type: ignore
        WhisperModel = _WM  # type: ignore
        logger.info("Successfully imported faster_whisper (lazy)")
        return True
    except Exception as e:
        WhisperModel = None  # type: ignore
        _IMPORT_ERROR = f"{type(e).__name__}: {e}"
        logger.exception("Failed to import faster_whisper (lazy)")
        return False


def _load_settings() -> dict:
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load whisper settings: %s", e)
        return {}


def _save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save whisper settings: %s", e)


def _default_output_dir() -> Path:
    return ROOT / "output" / "whisper"


def _find_binary(name: str) -> Optional[Path]:
    """Look for ffmpeg / ffprobe in presets/bin/"""
    bin_dir = PRESETS_DIR / "bin"
    candidates: List[Path] = []
    if os.name == "nt":
        candidates.append(bin_dir / f"{name}.exe")
        candidates.append(bin_dir / name)
    else:
        candidates.append(bin_dir / name)
        candidates.append(bin_dir / f"{name}.exe")
    for c in candidates:
        if c.is_file():
            logger.debug("Found binary %s at %s", name, c)
            return c
    logger.warning("Binary %s not found in %s", name, bin_dir)
    return None


def _probe_duration(ffprobe_path: Path, media_path: Path) -> float:
    cmd = [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(media_path),
    ]
    try:
        logger.debug("Probing duration with ffprobe: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("ffprobe failed (%s): %s", result.returncode, result.stderr)
            return 0.0
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning("Exception during ffprobe: %s", e)
        return 0.0


def _guess_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            logger.info("CUDA is available, using 'cuda' device")
            return "cuda"
        logger.info("CUDA not available, using CPU device")
    except Exception as e:
        logger.info("Torch not available or failed (%s), falling back to CPU", e)
    return "cpu"


# ---------------------------------------------------------------------------
# Data structures & formatting
# ---------------------------------------------------------------------------


@dataclass
class WhisperSegment:
    index: int
    start: float
    end: float
    text: str


def _format_timestamp_srt(seconds: float) -> str:
    """HH:MM:SS,mmm for SRT."""
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000.0))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """HH:MM:SS.mmm for VTT."""
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000.0))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _write_srt(segments: List[WhisperSegment], out_path: Path) -> None:
    lines: List[str] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        lines.append(str(seg.index))
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")  # blank line
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote SRT file with %d segments to %s", len(segments), out_path)


def _write_vtt(segments: List[WhisperSegment], out_path: Path) -> None:
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        if not seg.text.strip():
            continue
        start = _format_timestamp_vtt(seg.start)
        end = _format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        lines.append(seg.text.strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote VTT file with %d segments to %s", len(segments), out_path)


def _write_txt(text: str, out_path: Path) -> None:
    out_path.write_text(text.strip() + "\n", encoding="utf-8")
    logger.info("Wrote TXT lyrics to %s", out_path)


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

def _whisper_runner_code() -> str:
    # JSON-lines protocol:
    #   {"type":"progress","value": <int>}
    #   {"type":"result","data": {...}}
    #   {"type":"error","message": "..."}
    return r"""import os, sys, json, time, traceback
from pathlib import Path

def emit(o):
    sys.stdout.write(json.dumps(o, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def main():
    payload_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not payload_path or not payload_path.exists():
        raise RuntimeError("Missing payload JSON argument.")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))

    root = Path(payload.get("root", ".")).resolve()
    media_path = Path(payload["media_path"]).resolve()
    model_dir = Path(payload["model_dir"]).resolve()
    ffprobe_path = payload.get("ffprobe_path") or ""
    ffprobe_path = Path(ffprobe_path).resolve() if ffprobe_path else None
    language = (payload.get("language") or "").strip().lower()
    task = payload.get("task") or "transcribe"
    device = payload.get("device") or "cpu"
    compute_type = payload.get("compute_type") or ("float16" if device == "cuda" else "int8")

    out_temp = root / "output" / "_temp"
    out_temp.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    seg_json = out_temp / f"_whisper_segments_{ts}.json"
    txt_path = out_temp / f"_whisper_text_{ts}.txt"

    # Optional OpenMP safety switch
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Import in the isolated env
    from faster_whisper import WhisperModel

    model = WhisperModel(str(model_dir), device=device, compute_type=compute_type)

    options = {"beam_size": 5, "vad_filter": True, "task": "translate" if task == "translate" else "transcribe"}
    if language and language != "auto":
        options["language"] = language

    segments_gen, info = model.transcribe(str(media_path), **options)

    duration = getattr(info, "duration", None)
    if (duration is None or not duration) and ffprobe_path and ffprobe_path.exists():
        # ffprobe duration
        import subprocess
        cmd = [str(ffprobe_path), "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(media_path)]
        try:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
            if r.returncode == 0:
                duration = float((r.stdout or "").strip() or "0")
        except Exception:
            duration = None
    if not duration or duration <= 0:
        duration = None

    segs = []
    texts = []
    last_pct = -1

    for idx, seg in enumerate(segments_gen, start=1):
        t = (getattr(seg, "text", "") or "").strip()
        s = float(getattr(seg, "start", 0.0))
        e = float(getattr(seg, "end", 0.0))
        segs.append({"index": idx, "start": s, "end": e, "text": t})
        if t:
            texts.append(t)

        if duration:
            pct = int(min(99.0, (e / float(duration)) * 100.0))
            if pct != last_pct:
                last_pct = pct
                emit({"type": "progress", "value": pct})

    full_text = ("\n".join(texts)).strip()
    txt_path.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")
    seg_json.write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")

    info_dict = {"language": getattr(info, "language", None), "duration": getattr(info, "duration", None)}
    emit({"type": "progress", "value": 100})
    emit({"type": "result", "data": {"segments_json": str(seg_json), "text_path": str(txt_path), "info": info_dict}})

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        emit({"type": "error", "message": traceback.format_exc()})
        sys.exit(1)
"""

def _ensure_whisper_runner_file() -> Path:
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_path = out_dir / "_whisper_runner.py"
    code = _whisper_runner_code()
    try:
        existing = runner_path.read_text(encoding="utf-8") if runner_path.exists() else ""
    except Exception:
        existing = ""
    if existing != code:
        try:
            runner_path.write_text(code, encoding="utf-8")
        except Exception:
            pass
    return runner_path


class IsolatedTranscriptionWorker(QThread):
    progress_changed = Signal(int)
    finished_ok = Signal(list, str, dict)  # segments, full_text, info_dict
    failed = Signal(str)

    def __init__(
        self,
        media_path: Path,
        language: Optional[str],
        task: str,
        model_dir: Path,
        device: str,
        compute_type: str,
        ffprobe_path: Optional[Path] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.media_path = media_path
        self.language = language
        self.task = task
        self.model_dir = model_dir
        self.device = device
        self.compute_type = compute_type
        self.ffprobe_path = ffprobe_path

    def run(self) -> None:  # type: ignore[override]
        try:
            self.progress_changed.emit(0)

            env_py = _whisper_env_python()
            if not env_py.exists():
                raise RuntimeError(f"Whisper environment not found: {env_py}")

            runner = _ensure_whisper_runner_file()

            out_temp = ROOT / "output" / "_temp"
            out_temp.mkdir(parents=True, exist_ok=True)
            payload = {
                "root": str(ROOT),
                "media_path": str(self.media_path),
                "model_dir": str(self.model_dir),
                "device": self.device,
                "compute_type": self.compute_type,
                "language": self.language,
                "task": self.task,
                "ffprobe_path": str(self.ffprobe_path) if self.ffprobe_path else "",
            }

            import time as _time
            payload_file = out_temp / f"_whisper_payload_{int(_time.time()*1000)}.json"
            payload_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            cmd = [str(env_py), str(runner), str(payload_file)]
            logger.info("Running isolated Whisper transcription: %s", " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                text=True,
                bufsize=1,
            )

            last_result = None
            # Stream json lines
            if proc.stdout:
                for line in proc.stdout:
                    line = (line or "").strip()
                    if not line:
                        continue
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            msg = json.loads(line)
                            t = msg.get("type")
                            if t == "progress":
                                try:
                                    self.progress_changed.emit(int(msg.get("value", 0)))
                                except Exception:
                                    pass
                            elif t == "result":
                                last_result = msg.get("data") or {}
                            elif t == "error":
                                raise RuntimeError(str(msg.get("message", "")))
                        except Exception:
                            # Not valid JSON or unexpected format; ignore but log
                            logger.debug("Runner line: %s", line)
                    else:
                        logger.debug("Runner line: %s", line)

            rc = proc.wait()

            # We prefer to treat a non-zero exit code as a failure, but on Windows
            # the isolated runner can occasionally hard-crash on exit (e.g. 0xC0000005)
            # after having already written valid output files. In that case we accept
            # the run as successful as long as the expected outputs exist.
            if not last_result:
                raise RuntimeError("Whisper runner returned no result.")

            seg_json = Path(last_result.get("segments_json", ""))
            txt_path = Path(last_result.get("text_path", ""))
            info = last_result.get("info") or {}

            if rc != 0:
                if seg_json.is_file() and txt_path.is_file():
                    logger.warning(
                        "Whisper runner exited with code %s but output files exist; accepting result.",
                        rc,
                    )
                else:
                    raise RuntimeError(f"Whisper runner failed (exit code {rc}).")

            if not seg_json.is_file():
                raise RuntimeError(f"Missing segments JSON: {seg_json}")
            if not txt_path.is_file():
                raise RuntimeError(f"Missing transcript text file: {txt_path}")

            segs_raw = json.loads(seg_json.read_text(encoding="utf-8"))
            segments: List[WhisperSegment] = []
            for s in segs_raw:
                try:
                    segments.append(
                        WhisperSegment(
                            index=int(s.get("index", 0)),
                            start=float(s.get("start", 0.0)),
                            end=float(s.get("end", 0.0)),
                            text=str(s.get("text", "") or "").strip(),
                        )
                    )
                except Exception:
                    continue

            full_text = txt_path.read_text(encoding="utf-8").strip()
            self.progress_changed.emit(100)
            self.finished_ok.emit(segments, full_text, info)
        except Exception as exc:
            logger.exception("IsolatedTranscriptionWorker failed")
            self.failed.emit(str(exc))


class TranscriptionWorker(QThread):
    """
    In-process transcription worker (used only when the isolated venv is NOT available).
    """
    progress_changed = Signal(int)
    finished_ok = Signal(list, str, dict)  # segments, full_text, info_dict
    failed = Signal(str)

    def __init__(
        self,
        model,
        media_path: Path,
        language: Optional[str],
        task: str,
        ffprobe_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.media_path = media_path
        self.language = language
        self.task = task
        self.ffprobe_path = ffprobe_path

    def run(self) -> None:  # type: ignore[override]
        try:
            logger.info(
                "TranscriptionWorker started (in-process): media=%s, language=%s, task=%s",
                self.media_path,
                self.language,
                self.task,
            )
            self.progress_changed.emit(0)

            options = {"beam_size": 5, "vad_filter": True}
            options["task"] = "translate" if self.task == "translate" else "transcribe"

            lang = (self.language or "").strip().lower()
            if lang and lang != "auto":
                options["language"] = lang

            logger.debug("Calling model.transcribe with options: %s", options)
            segments_gen, info = self.model.transcribe(str(self.media_path), **options)

            duration = getattr(info, "duration", None)
            if (duration is None or not duration) and self.ffprobe_path:
                duration = _probe_duration(self.ffprobe_path, self.media_path)
            if not duration or duration <= 0:
                duration = None

            segments: List[WhisperSegment] = []
            texts: List[str] = []
            last_pct = 0

            for idx, seg in enumerate(segments_gen, start=1):
                text = (seg.text or "").strip()
                segment = WhisperSegment(
                    index=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                )
                segments.append(segment)
                if text:
                    texts.append(text)

                if duration is not None and duration > 0:
                    pct = int(min(99.0, (float(seg.end) / float(duration)) * 100.0))
                    if pct != last_pct:
                        last_pct = pct
                        self.progress_changed.emit(pct)

            full_text = "\n".join(texts).strip() if texts else ""
            info_dict = {}
            try:
                info_dict = {
                    "language": getattr(info, "language", None),
                    "duration": getattr(info, "duration", None),
                }
            except Exception:
                pass

            self.progress_changed.emit(100)
            self.finished_ok.emit(segments, full_text, info_dict)
        except Exception as exc:
            logger.exception("TranscriptionWorker failed (in-process)")
            self.failed.emit(str(exc))


class SubtitleEmbedWorker(QThread):
    progress_changed = Signal(int)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        ffmpeg_path: Path,
        video_path: Path,
        subtitle_path: Path,
        output_path: Path,
    ) -> None:
        super().__init__()
        self.ffmpeg_path = ffmpeg_path
        self.video_path = video_path
        self.subtitle_path = subtitle_path
        self.output_path = output_path

    def run(self) -> None:  # type: ignore[override]
        try:
            self.progress_changed.emit(0)
            sub_arg_raw = str(self.subtitle_path)
            sub_arg = sub_arg_raw.replace('\\', '/').replace(':', r'\:')
            sub_arg = sub_arg.replace("'", r"\'").replace(',', r'\,').replace('[', r'\[').replace(']', r'\]')
            cmd = [
                str(self.ffmpeg_path),
                "-y",
                "-i",
                str(self.video_path),
                "-vf",
                f"subtitles=filename='{sub_arg}'",
                "-c:v",
                "libx264",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ]
            logger.info("Running ffmpeg for subtitle burn-in: %s", " ".join(cmd))
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                logger.error("ffmpeg failed with code %s: %s", proc.returncode, proc.stderr)
                raise RuntimeError(f"ffmpeg failed:\n{proc.stderr}")
            self.progress_changed.emit(100)
            logger.info("Subtitle burn-in finished: %s", self.output_path)
            self.finished_ok.emit(str(self.output_path))
        except Exception as exc:
            logger.exception("SubtitleEmbedWorker failed")
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# Main Widget
# ---------------------------------------------------------------------------


class WhisperWidget(QWidget):
    """
    Main PySide6 widget for Whisper Lab.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("WhisperWidget")

        self.ffmpeg_path: Optional[Path] = _find_binary("ffmpeg")
        self.ffprobe_path: Optional[Path] = _find_binary("ffprobe")
        self.model_dir: Path = FWHISPER_MEDIUM_DIR
        self.device: str = _guess_device()
        self.compute_type: str = "float16" if self.device == "cuda" else "int8"
        self.model = None

        self.current_segments: List[WhisperSegment] = []
        self.current_text: str = ""
        self.last_srt_path: Optional[Path] = None

        self._settings: dict = _load_settings()

        logger.info(
            "WhisperWidget initialized. model_dir=%s, device=%s, compute_type=%s",
            self.model_dir,
            self.device,
            self.compute_type,
        )

        self._setup_ui()
        self._apply_settings_to_ui()
        self._update_backend_labels()

    # ----------------------------- UI setup -----------------------------

    def _setup_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        # Backend
        backend_group = QGroupBox("Whisper Backend")
        backend_form = QFormLayout(backend_group)

        self.label_model_path = QLabel(str(self.model_dir))
        self.label_model_status = QLabel("Model: not loaded")
        self.label_device = QLabel("")

        btn_reload_model = QPushButton("Reload model")
        btn_reload_model.clicked.connect(self._on_reload_model)

        backend_form.addRow("Model dir:", self.label_model_path)
        backend_form.addRow("Device:", self.label_device)
        backend_form.addRow("Status:", self.label_model_status)
        backend_form.addRow("", btn_reload_model)
        main.addWidget(backend_group)

        # Input
        input_group = QGroupBox("Input Media")
        input_form = QFormLayout(input_group)

        self.edit_media = QLineEdit()
        btn_browse_media = QPushButton("Browse…")
        btn_browse_media.clicked.connect(self._browse_media)
        row_media = QHBoxLayout()
        row_media.addWidget(self.edit_media)
        row_media.addWidget(btn_browse_media)
        input_form.addRow("Audio / Video:", row_media)

        self.combo_language = QComboBox()
        self.combo_language.addItem("Auto")
        for code in ["en", "de", "fr", "es", "it", "nl", "pt", "ru", "ja", "zh"]:
            self.combo_language.addItem(code)
        input_form.addRow("Language:", self.combo_language)

        self.check_translate = QCheckBox("Translate to English")
        input_form.addRow("", self.check_translate)

        main.addWidget(input_group)

        # Output settings
        out_group = QGroupBox("Outputs")
        out_form = QFormLayout(out_group)

        self.edit_output_dir = QLineEdit()
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._browse_output_dir)
        row_out = QHBoxLayout()
        row_out.addWidget(self.edit_output_dir)
        row_out.addWidget(btn_browse_out)
        out_form.addRow("Output folder:", row_out)

        self.check_make_srt = QCheckBox("Generate .srt subtitles")
        self.check_make_srt.setChecked(True)
        self.check_make_vtt = QCheckBox("Generate .vtt subtitles")
        self.check_make_txt = QCheckBox("Generate .txt lyrics")
        self.check_make_txt.setChecked(True)

        out_form.addRow(self.check_make_srt)
        out_form.addRow(self.check_make_vtt)
        out_form.addRow(self.check_make_txt)

        main.addWidget(out_group)

        # Transcription controls
        trans_group = QGroupBox("Transcription")
        trans_form = QFormLayout(trans_group)

        self.btn_transcribe = QPushButton("Transcribe / Extract Lyrics")
        self.btn_transcribe.clicked.connect(self._on_transcribe_clicked)
        trans_form.addRow("", self.btn_transcribe)

        self.progress_transcribe = QProgressBar()
        self.progress_transcribe.setRange(0, 100)
        self.progress_transcribe.setValue(0)
        trans_form.addRow("Progress:", self.progress_transcribe)

        self.text_preview = QPlainTextEdit()
        self.text_preview.setPlaceholderText("Transcript / lyrics will appear here…")
        trans_form.addRow("Preview:", self.text_preview)

        main.addWidget(trans_group, stretch=1)

        # Segments table
        self.table_segments = QTableWidget(0, 4)
        self.table_segments.setHorizontalHeaderLabels(["#", "Start", "End", "Text"])
        header = self.table_segments.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table_segments.verticalHeader().setVisible(False)
        main.addWidget(self.table_segments, stretch=2)

        # Subtitle embedder
        embed_group = QGroupBox("Subtitle Embedder (burn-in)")
        embed_form = QFormLayout(embed_group)

        self.edit_embed_video = QLineEdit()
        btn_embed_video = QPushButton("Browse video…")
        btn_embed_video.clicked.connect(self._browse_embed_video)
        row_ev = QHBoxLayout()
        row_ev.addWidget(self.edit_embed_video)
        row_ev.addWidget(btn_embed_video)
        embed_form.addRow("Video:", row_ev)

        self.edit_embed_subs = QLineEdit()
        btn_embed_subs = QPushButton("Browse subs…")
        btn_embed_subs.clicked.connect(self._browse_embed_subs)
        row_es = QHBoxLayout()
        row_es.addWidget(self.edit_embed_subs)
        row_es.addWidget(btn_embed_subs)
        embed_form.addRow("Subtitle file (.srt):", row_es)

        self.btn_burn_subs = QPushButton("Burn subtitles into video")
        self.btn_burn_subs.clicked.connect(self._on_burn_subtitles)
        embed_form.addRow("", self.btn_burn_subs)

        self.progress_embed = QProgressBar()
        self.progress_embed.setRange(0, 100)
        self.progress_embed.setValue(0)
        embed_form.addRow("Progress:", self.progress_embed)

        main.addWidget(embed_group)

        main.addStretch(1)

    # ----------------------------- Settings -----------------------------

    def _apply_settings_to_ui(self) -> None:
        s = self._settings or {}

        media_path = s.get("last_media", "")
        out_dir = s.get("last_output_dir", "")
        embed_video = s.get("embed_video", "")
        embed_subs = s.get("embed_subs", "")
        language = s.get("language", "Auto")
        translate = bool(s.get("translate_to_en", False))
        make_srt = s.get("make_srt", True)
        make_vtt = s.get("make_vtt", False)
        make_txt = s.get("make_txt", True)

        if media_path:
            self.edit_media.setText(media_path)
        if out_dir:
            self.edit_output_dir.setText(out_dir)
        else:
            default_out = _default_output_dir()
            default_out.mkdir(parents=True, exist_ok=True)
            self.edit_output_dir.setText(str(default_out))

        if embed_video:
            self.edit_embed_video.setText(embed_video)
        if embed_subs:
            self.edit_embed_subs.setText(embed_subs)

        # language combo
        idx = self.combo_language.findText(language, Qt.MatchFixedString)
        if idx < 0:
            idx = 0
        self.combo_language.setCurrentIndex(idx)

        self.check_translate.setChecked(translate)
        self.check_make_srt.setChecked(bool(make_srt))
        self.check_make_vtt.setChecked(bool(make_vtt))
        self.check_make_txt.setChecked(bool(make_txt))

    def _update_settings_from_ui(self) -> None:
        data = {
            "last_media": self.edit_media.text().strip(),
            "last_output_dir": self.edit_output_dir.text().strip(),
            "embed_video": self.edit_embed_video.text().strip(),
            "embed_subs": self.edit_embed_subs.text().strip(),
            "language": self.combo_language.currentText(),
            "translate_to_en": bool(self.check_translate.isChecked()),
            "make_srt": bool(self.check_make_srt.isChecked()),
            "make_vtt": bool(self.check_make_vtt.isChecked()),
            "make_txt": bool(self.check_make_txt.isChecked()),
        }
        self._settings = data
        _save_settings(data)

    # ----------------------------- Backend helpers -----------------------------

    def _update_backend_labels(self) -> None:
        self.label_model_path.setText(str(self.model_dir))

        if _has_isolated_whisper_env():
            env_py = _whisper_env_python()
            self.label_device.setText(f"{self.device} (compute={self.compute_type})  |  isolated env: {env_py}")
            self.label_model_status.setText("Model: isolated (separate process)")
            return

        dev_text = f"{self.device} (compute={self.compute_type})"
        if (WhisperModel is None) and _IMPORT_ERROR:
            dev_text += "  |  faster-whisper import failed"
        self.label_device.setText(dev_text)

        if self.model is None:
            self.label_model_status.setText("Model: not loaded")
        else:
            self.label_model_status.setText("Model: loaded")

    def _ensure_model_loaded(self) -> bool:
        # Prefer isolated venv subprocess mode when available (keeps FrameVision responsive and avoids DLL conflicts)
        if _has_isolated_whisper_env():
            self.label_model_status.setText("Model: isolated (separate process)")
            return True

        # In-process fallback (standalone usage)
        if not _lazy_import_faster_whisper():
            logger.error("Cannot load Whisper model because import failed: %s", _IMPORT_ERROR)
            QMessageBox.critical(
                self,
                "Whisper import failed",
                f"Failed to import faster-whisper in-process.\n\n{_IMPORT_ERROR or ''}\n\n"                "Tip: install Whisper via Optional Installs (environments/.whisper) to run isolated.",
            )
            return False

        if not self.model_dir.is_dir():
            logger.error("Model directory does not exist: %s", self.model_dir)
            QMessageBox.warning(
                self,
                "Model not found",
                f"Expected Faster-Whisper model in:\n{self.model_dir}\n\n"
                "Run your model downloader first.",
            )
            return False

        if self.model is not None:
            return True

        try:
            self.label_model_status.setText("Model: loading…")
            QApplication.processEvents()
            logger.info(
                "Loading Whisper model (in-process) from %s (device=%s, compute_type=%s)",
                self.model_dir,
                self.device,
                self.compute_type,
            )
            self.model = WhisperModel(
                str(self.model_dir),
                device=self.device,
                compute_type=self.compute_type,
            )
            self.label_model_status.setText("Model: loaded")
            logger.info("Whisper model loaded successfully (in-process)")
            return True
        except Exception as exc:
            self.model = None
            self.label_model_status.setText("Model: failed to load")
            logger.exception("Failed to load Whisper model (in-process)")
            QMessageBox.critical(self, "Model load failed", str(exc))
            return False

        if not self.model_dir.is_dir():
            logger.error("Model directory does not exist: %s", self.model_dir)
            QMessageBox.warning(
                self,
                "Model not found",
                f"Expected Faster-Whisper model in:\n{self.model_dir}\n\n"
                "Run your model downloader first.",
            )
            return False

        if self.model is not None:
            return True

        # Load model
        try:
            self.label_model_status.setText("Model: loading…")
            QApplication.processEvents()
            logger.info(
                "Loading Whisper model from %s (device=%s, compute_type=%s)",
                self.model_dir,
                self.device,
                self.compute_type,
            )
            self.model = WhisperModel(
                str(self.model_dir),
                device=self.device,
                compute_type=self.compute_type,
            )
            self.label_model_status.setText("Model: loaded")
            logger.info("Whisper model loaded successfully")
            return True
        except Exception as exc:
            self.model = None
            self.label_model_status.setText("Model: failed to load")
            logger.exception("Failed to load Whisper model")
            QMessageBox.critical(self, "Model load failed", str(exc))
            return False

    # ----------------------------- File pickers -----------------------------

    def _browse_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio or video to transcribe",
            "",
            "Audio/Video Files (*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.mp4 *.mov *.mkv *.avi *.webm);;All Files (*.*)",
        )
        if path:
            self.edit_media.setText(path)
            # also suggest as embed video if it's a video
            ext = Path(path).suffix.lower()
            if ext in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                self.edit_embed_video.setText(path)
            self._update_settings_from_ui()

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        if path:
            self.edit_output_dir.setText(path)
            self._update_settings_from_ui()

    def _browse_embed_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video for subtitle burn-in",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi *.webm);;All Files (*.*)",
        )
        if path:
            self.edit_embed_video.setText(path)
            self._update_settings_from_ui()

    def _browse_embed_subs(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select .srt subtitle file",
            "",
            "Subtitle Files (*.srt);;All Files (*.*)",
        )
        if path:
            self.edit_embed_subs.setText(path)
            self._update_settings_from_ui()

    # ----------------------------- Transcription -----------------------------

    @Slot()
    def _on_reload_model(self) -> None:
        self.model = None
        self._update_backend_labels()
        logger.info("Reload model requested by user")
        if self._ensure_model_loaded():
            QMessageBox.information(self, "Whisper", "Model reloaded successfully.")

    @Slot()
    def _on_transcribe_clicked(self) -> None:
        logger.info("Transcribe button clicked")
        if not self._ensure_model_loaded():
            return

        media_path = Path(self.edit_media.text().strip())
        if not media_path.is_file():
            QMessageBox.warning(self, "Invalid input", "Please select a valid audio or video file.")
            return

        out_dir_str = self.edit_output_dir.text().strip()
        if not out_dir_str:
            out_dir = _default_output_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            self.edit_output_dir.setText(str(out_dir))
        else:
            out_dir = Path(out_dir_str)
            out_dir.mkdir(parents=True, exist_ok=True)

        self._update_settings_from_ui()

        language = self.combo_language.currentText()
        task = "translate" if self.check_translate.isChecked() else "transcribe"

        logger.info(
            "Starting transcription: media=%s, language=%s, task=%s, out_dir=%s",
            media_path,
            language,
            task,
            out_dir,
        )

        self.progress_transcribe.setValue(0)
        self.btn_transcribe.setEnabled(False)
        self.text_preview.clear()
        self.table_segments.setRowCount(0)
        self.current_segments = []
        self.current_text = ""
        self.last_srt_path = None

        if _has_isolated_whisper_env():
            # Run in separate process (Whisper venv) so long transcripts can't freeze or crash the main app
            self.worker = IsolatedTranscriptionWorker(
                media_path,
                language,
                task,
                self.model_dir,
                self.device,
                self.compute_type,
                self.ffprobe_path,
                parent=self,
            )
        else:
            # In-process fallback
            self.worker = TranscriptionWorker(
                self.model,
                media_path,
                language,
                task,
                self.ffprobe_path,
            )
        self.worker.progress_changed.connect(self.progress_transcribe.setValue)
        self.worker.finished_ok.connect(self._on_transcription_finished)
        self.worker.failed.connect(self._on_transcription_failed)
        self.worker.start()

    @Slot(list, str, dict)
    def _on_transcription_finished(self, segments, full_text: str, info: dict) -> None:
        logger.info(
            "Transcription finished callback: segments=%d, has_text=%s",
            len(segments),
            bool(full_text),
        )
        self.btn_transcribe.setEnabled(True)
        self.current_segments = segments
        self.current_text = full_text

        # Update preview
        if full_text:
            self.text_preview.setPlainText(full_text)
        else:
            self.text_preview.setPlainText("(no text detected)")

        # Fill table (cap rows to keep UI snappy on very long transcripts)
        MAX_ROWS = 2000
        self.table_segments.setUpdatesEnabled(False)
        self.table_segments.setRowCount(0)
        shown = 0
        for seg in (segments or []):
            if shown >= MAX_ROWS:
                break
            r = self.table_segments.rowCount()
            self.table_segments.insertRow(r)
            self.table_segments.setItem(r, 0, QTableWidgetItem(str(seg.index)))
            self.table_segments.setItem(r, 1, QTableWidgetItem(f"{seg.start:.3f}"))
            self.table_segments.setItem(r, 2, QTableWidgetItem(f"{seg.end:.3f}"))
            self.table_segments.setItem(r, 3, QTableWidgetItem(seg.text))
            shown += 1
        self.table_segments.setUpdatesEnabled(True)
        if segments and len(segments) > MAX_ROWS:
            QMessageBox.information(
                self,
                "Whisper",
                f"Transcript is very long ({len(segments)} segments).\n\n"                f"To keep the app responsive, only the first {MAX_ROWS} segments are shown in the table.\n"                "All output files (SRT/VTT/TXT) will still include the full transcript.",
            )
        # Write files
        out_dir = Path(self.edit_output_dir.text().strip() or str(_default_output_dir()))
        out_dir.mkdir(parents=True, exist_ok=True)

        media_path = Path(self.edit_media.text().strip())
        stem = media_path.stem or "whisper_output"

        made_files = []

        try:
            if self.check_make_srt.isChecked():
                srt_path = out_dir / f"{stem}.srt"
                _write_srt(segments, srt_path)
                made_files.append(srt_path)
                self.last_srt_path = srt_path
                # also update embed subs edit if empty
                if not self.edit_embed_subs.text().strip():
                    self.edit_embed_subs.setText(str(srt_path))

            if self.check_make_vtt.isChecked():
                vtt_path = out_dir / f"{stem}.vtt"
                _write_vtt(segments, vtt_path)
                made_files.append(vtt_path)

            if self.check_make_txt.isChecked() and full_text:
                txt_path = out_dir / f"{stem}_lyrics.txt"
                _write_txt(full_text, txt_path)
                made_files.append(txt_path)
        except Exception as exc:
            logger.exception("Failed to write output files")
            QMessageBox.critical(self, "Write failed", f"Failed to write output files:\n{exc}")
            return

        self._update_settings_from_ui()

        msg_lines = ["Transcription finished."]
        if made_files:
            msg_lines.append("Created:")
            for p in made_files:
                msg_lines.append(f"  - {p}")
        if info.get("language"):
            msg_lines.append(f"Detected language: {info.get('language')}")
        if info.get("duration"):
            try:
                dur = float(info.get("duration"))
                msg_lines.append(f"Duration: {dur:.1f} s")
            except Exception:
                pass
        QMessageBox.information(self, "Whisper", "\n".join(msg_lines))

    @Slot(str)
    def _on_transcription_failed(self, message: str) -> None:
        logger.error("Transcription failed: %s", message)
        self.btn_transcribe.setEnabled(True)
        QMessageBox.critical(self, "Transcription failed", message)

    # ----------------------------- Subtitle burn-in -----------------------------

    @Slot()
    def _on_burn_subtitles(self) -> None:
        logger.info("Burn subtitles button clicked")
        if not self.ffmpeg_path:
            QMessageBox.warning(
                self,
                "ffmpeg not found",
                "Could not locate ffmpeg in presets/bin.\n"
                "Subtitle burn-in requires ffmpeg.",
            )
            return

        video_path = Path(self.edit_embed_video.text().strip())
        if not video_path.is_file():
            QMessageBox.warning(self, "Invalid video", "Please select a valid video file.")
            return

        subs_path = Path(self.edit_embed_subs.text().strip())
        if not subs_path.is_file():
            QMessageBox.warning(self, "Invalid subtitles", "Please select a valid .srt subtitles file.")
            return

        out_dir = Path(self.edit_output_dir.text().strip() or str(_default_output_dir()))
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{video_path.stem}_subtitled.mp4"

        self._update_settings_from_ui()

        self.progress_embed.setValue(0)
        self.btn_burn_subs.setEnabled(False)

        logger.info(
            "Starting subtitle burn-in: video=%s, subs=%s, out=%s",
            video_path,
            subs_path,
            out_path,
        )

        self.embed_worker = SubtitleEmbedWorker(
            self.ffmpeg_path,
            video_path,
            subs_path,
            out_path,
        )
        self.embed_worker.progress_changed.connect(self.progress_embed.setValue)
        self.embed_worker.finished_ok.connect(self._on_burn_finished)
        self.embed_worker.failed.connect(self._on_burn_failed)
        self.embed_worker.start()

    @Slot(str)
    def _on_burn_finished(self, out_path: str) -> None:
        logger.info("Subtitle burn-in finished successfully: %s", out_path)
        self.btn_burn_subs.setEnabled(True)
        self.progress_embed.setValue(100)
        QMessageBox.information(
            self,
            "Subtitle burn-in",
            f"Subtitled video saved to:\n{out_path}",
        )

    @Slot(str)
    def _on_burn_failed(self, message: str) -> None:
        logger.error("Subtitle burn-in failed: %s", message)
        self.btn_burn_subs.setEnabled(True)
        QMessageBox.critical(self, "Subtitle burn-in failed", message)


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------


def main() -> int:
    app = QApplication(sys.argv)
    w = WhisperWidget()
    w.resize(1000, 800)
    w.setWindowTitle("Whisper Lab (Faster-Whisper)")
    w.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
