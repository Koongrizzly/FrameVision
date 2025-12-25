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

# Track import error details (so we can show real reason in GUI)
_IMPORT_ERROR: Optional[str] = None

try:
    from faster_whisper import WhisperModel  # type: ignore
    logger.info("Successfully imported faster_whisper")
except Exception as e:  # handled gracefully later
    WhisperModel = None  # type: ignore
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"
    logger.exception("Failed to import faster_whisper")


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


class TranscriptionWorker(QThread):
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
                "TranscriptionWorker started: media=%s, language=%s, task=%s",
                self.media_path,
                self.language,
                self.task,
            )
            self.progress_changed.emit(0)

            # Call Faster-Whisper
            options = {
                "beam_size": 5,
                "vad_filter": True,
            }
            if self.task == "translate":
                options["task"] = "translate"
            else:
                options["task"] = "transcribe"

            lang = (self.language or "").strip().lower()
            if lang and lang != "auto":
                options["language"] = lang

            logger.debug("Calling model.transcribe with options: %s", options)
            segments_gen, info = self.model.transcribe(str(self.media_path), **options)

            # Determine duration for progress
            duration = getattr(info, "duration", None)
            if (duration is None or not duration) and self.ffprobe_path:
                duration = _probe_duration(self.ffprobe_path, self.media_path)
            if not duration or duration <= 0:
                duration = None  # disable progress based on time
            logger.debug("Detected media duration for progress: %s", duration)

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

            full_text = "\n".join(texts).strip()
            if not full_text:
                full_text = ""

            info_dict = {}
            try:
                info_dict = {
                    "language": getattr(info, "language", None),
                    "duration": getattr(info, "duration", None),
                }
            except Exception as e:
                logger.debug("Failed to extract info fields: %s", e)

            logger.info(
                "TranscriptionWorker finished: segments=%d, language=%s, duration=%s",
                len(segments),
                info_dict.get("language"),
                info_dict.get("duration"),
            )
            self.progress_changed.emit(100)
            self.finished_ok.emit(segments, full_text, info_dict)
        except Exception as exc:
            logger.exception("TranscriptionWorker failed")
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
            sub_arg = self.subtitle_path.as_posix()
            cmd = [
                str(self.ffmpeg_path),
                "-y",
                "-i",
                str(self.video_path),
                "-vf",
                f"subtitles={sub_arg}",
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

        dev_text = f"{self.device} (compute={self.compute_type})"
        if 'WhisperModel' in globals() and WhisperModel is None:
            dev_text += "  |  faster-whisper import failed"
        self.label_device.setText(dev_text)

        if self.model is None:
            self.label_model_status.setText("Model: not loaded")
        else:
            self.label_model_status.setText("Model: loaded")

    def _ensure_model_loaded(self) -> bool:
        if 'WhisperModel' not in globals() or WhisperModel is None:
            logger.error("Cannot load Whisper model because import failed: %s", _IMPORT_ERROR)
            msg = (
                "The 'faster-whisper' package is not installed or failed to import.\n"
                "Install it in your environment to use Whisper Lab."
            )
            if _IMPORT_ERROR:
                msg += f"\n\nImport error:\n{_IMPORT_ERROR}"
            QMessageBox.warning(self, "Missing dependency", msg)
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

        # Fill table
        self.table_segments.setRowCount(0)
        for seg in segments:
            r = self.table_segments.rowCount()
            self.table_segments.insertRow(r)
            self.table_segments.setItem(r, 0, QTableWidgetItem(str(seg.index)))
            self.table_segments.setItem(r, 1, QTableWidgetItem(f"{seg.start:.3f}"))
            self.table_segments.setItem(r, 2, QTableWidgetItem(f"{seg.end:.3f}"))
            self.table_segments.setItem(r, 3, QTableWidgetItem(seg.text))

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
