# -*- coding: utf-8 -*-
"""
FrameVision dots.tts helper tab.

Drop this file into: helpers/dotstts.py

Importer examples:
    from helpers.dotstts import create_dots_tts_tab
    tab = create_dots_tts_tab(parent=self)

or:
    from helpers.dotstts import DotsTtsTab
    tab = DotsTtsTab(parent=self)

This helper intentionally runs dots.tts through the installed conda env Python:
    environments/.dots_tts/python.exe -m dots_tts.cli ...

It does not import torch/dots.tts inside the main FrameVision process, so the big model
is isolated in the subprocess and can release VRAM when the job exits.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import wave
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, Signal, QUrl
from PySide6.QtGui import QAction, QTextCursor

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except Exception:  # pragma: no cover - keeps helper import-safe on stripped Qt installs.
    QAudioOutput = None
    QMediaPlayer = None

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - helper still works without waveform dependency.
    pg = None
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
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
TEMPLATE_CHOICES = [
    ("TTS / voice clone", "tts"),
    ("Instruction TTS", "instruction_tts"),
    ("Text to audio", "text_to_audio"),
    ("Interleave", "tts_interleave"),
]
PRECISION_CHOICES = ["bfloat16", "float16", "float32"]
ODE_CHOICES = ["euler"]
LANGUAGE_CHOICES = [
    ("None", ""),
    ("Auto detect", "auto_detect"),
    ("English", "EN"),
    ("Dutch", "NL"),
    ("Chinese", "ZH"),
    ("French", "FR"),
    ("German", "DE"),
    ("Spanish", "ES"),
    ("Italian", "IT"),
    ("Japanese", "JA"),
    ("Korean", "KO"),
    ("Portuguese", "PT"),
    ("Russian", "RU"),
]


@dataclass
class DotsPaths:
    root: Path
    env_dir: Path
    env_python: Path
    repo_dir: Path
    model_dir: Path
    output_dir: Path
    settings_file: Path
    voice_presets_file: Path
    log_dir: Path
    temp_dir: Path


def _find_framevision_root(start: Optional[Path] = None) -> Path:
    """Best-effort FrameVision root discovery."""
    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    try:
        candidates.append(Path(__file__).resolve())
    except Exception:
        pass
    candidates.append(Path.cwd().resolve())

    for candidate in candidates:
        for folder in [candidate, *candidate.parents]:
            if (folder / "helpers").exists() and (folder / "presets").exists():
                return folder
            if (folder / "environments").exists() and (folder / "models").exists():
                return folder
    return Path.cwd().resolve()


def build_default_paths(root: Optional[Path] = None) -> DotsPaths:
    root = (root or _find_framevision_root()).resolve()
    env_dir = root / "environments" / ".dots_tts"
    return DotsPaths(
        root=root,
        env_dir=env_dir,
        env_python=env_dir / "python.exe",
        repo_dir=root / "models" / "dots_tts" / "repo",
        model_dir=root / "models" / "dots_tts" / "dots.tts-mf",
        output_dir=root / "output" / "dots_tts",
        settings_file=root / "presets" / "setsave" / "dots_tts.json",
        voice_presets_file=root / "presets" / "setsave" / "dots_tts_voices.json",
        log_dir=root / "logs",
        temp_dir=root / "temp" / "dots_tts_runtime",
    )


def _safe_read_json(path: Path, default: Any) -> Any:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _safe_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _quote_for_display(parts: list[str]) -> str:
    def quote(part: str) -> str:
        if not part:
            return '""'
        if any(ch.isspace() for ch in part) or any(ch in part for ch in '()[]{}&^%=;!\''):
            return '"' + part.replace('"', '\\"') + '"'
        return part
    return " ".join(quote(str(p)) for p in parts)


def _slug(text: str, fallback: str = "dots_tts") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())[:48].strip("._-")
    return value or fallback



def _format_ms(ms: int) -> str:
    ms = max(0, int(ms))
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _read_wav_preview(path: Path, max_points: int = 12000) -> tuple[list[float], float]:
    """Read a PCM WAV into mono preview points without requiring numpy/soundfile."""
    with wave.open(str(path), "rb") as wav:
        channels = max(1, wav.getnchannels())
        sample_width = wav.getsampwidth()
        frame_rate = max(1, wav.getframerate())
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if not raw or frame_count <= 0:
        return [], 0.0

    values: list[float] = []

    if sample_width == 1:
        # unsigned 8-bit PCM
        for i in range(0, len(raw), channels):
            values.append((raw[i] - 128) / 128.0)
    elif sample_width == 2:
        count = len(raw) // 2
        samples = struct.unpack("<" + "h" * count, raw[: count * 2])
        step = channels
        for i in range(0, len(samples), step):
            acc = 0.0
            used = 0
            for ch in range(channels):
                idx = i + ch
                if idx < len(samples):
                    acc += samples[idx] / 32768.0
                    used += 1
            values.append(acc / max(1, used))
    elif sample_width == 3:
        frame_size = channels * 3
        for frame_start in range(0, len(raw), frame_size):
            acc = 0.0
            used = 0
            for ch in range(channels):
                idx = frame_start + ch * 3
                if idx + 3 <= len(raw):
                    chunk = raw[idx:idx + 3]
                    value = int.from_bytes(chunk, byteorder="little", signed=True)
                    acc += value / 8388608.0
                    used += 1
            values.append(acc / max(1, used))
    elif sample_width == 4:
        count = len(raw) // 4
        samples = struct.unpack("<" + "i" * count, raw[: count * 4])
        step = channels
        for i in range(0, len(samples), step):
            acc = 0.0
            used = 0
            for ch in range(channels):
                idx = i + ch
                if idx < len(samples):
                    acc += samples[idx] / 2147483648.0
                    used += 1
            values.append(acc / max(1, used))
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if len(values) > max_points:
        stride = max(1, len(values) // max_points)
        values = values[::stride]

    duration = frame_count / float(frame_rate)
    return values, duration


class AudioWaveformPreview(QGroupBox):
    """Small pyqtgraph waveform/player widget for reference and generated audio."""

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self.setToolTip("Waveform preview. Click on the waveform to seek and play from that position.")
        self.path: Optional[Path] = None
        self.duration_ms = 0
        self._playing_path: Optional[Path] = None

        self.audio_output = QAudioOutput(self) if QAudioOutput is not None else None
        self.player = QMediaPlayer(self) if QMediaPlayer is not None else None
        if self.player is not None and self.audio_output is not None:
            self.player.setAudioOutput(self.audio_output)
            self.audio_output.setVolume(0.85)
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.durationChanged.connect(self._on_duration_changed)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("No audio loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setToolTip("Shows the loaded audio path, duration, or why a waveform could not be displayed.")
        layout.addWidget(self.status_label)

        if pg is not None:
            self.plot = pg.PlotWidget()
            self.plot.setMinimumHeight(90)
            self.plot.setBackground(None)
            self.plot.showGrid(x=True, y=True, alpha=0.25)
            self.plot.setMouseEnabled(x=True, y=False)
            self.plot.setMenuEnabled(False)
            self.plot.hideAxis("left")
            self.plot.setLabel("bottom", "Time", units="s")
            self.curve = self.plot.plot([], [], pen=pg.mkPen(width=1))
            self.playhead = pg.InfiniteLine(pos=0, angle=90, movable=False, pen=pg.mkPen(width=2))
            self.plot.addItem(self.playhead)
            self.plot.scene().sigMouseClicked.connect(self._on_plot_clicked)
            layout.addWidget(self.plot)
        else:
            self.plot = None
            self.curve = None
            self.playhead = None
            missing = QLabel("Waveform preview needs pyqtgraph in the Python environment running this UI.")
            missing.setWordWrap(True)
            missing.setToolTip("Install pyqtgraph next to the PySide6 environment that starts this helper.")
            layout.addWidget(missing)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setToolTip("Play from the current playhead position.")
        self.play_btn.clicked.connect(self.play)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setToolTip("Pause playback and keep the current position.")
        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop playback and return to the start.")
        self.stop_btn.clicked.connect(self.stop)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setToolTip("Played time / total duration.")
        controls.addWidget(self.play_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        controls.addWidget(self.time_label)
        layout.addLayout(controls)

        self._set_controls_enabled(False)

    def load_audio(self, path: str | Path | None) -> None:
        self.stop()
        self.path = None
        self.duration_ms = 0
        self._playing_path = None
        self._set_controls_enabled(False)
        self._set_time(0, 0)

        if self.curve is not None:
            self.curve.setData([], [])
        if self.playhead is not None:
            self.playhead.setValue(0)

        if not path:
            self.status_label.setText("No audio loaded.")
            return

        audio_path = Path(str(path))
        if not audio_path.is_file():
            self.status_label.setText(f"Audio not found: {audio_path}")
            return

        self.path = audio_path

        if audio_path.suffix.lower() != ".wav":
            self.status_label.setText(f"Loaded: {audio_path.name} — playback available, waveform currently supports WAV only.")
            self.duration_ms = 0
            self._set_controls_enabled(self.player is not None)
            self._prepare_player()
            return

        try:
            values, duration = _read_wav_preview(audio_path)
            self.duration_ms = int(duration * 1000)
            if self.curve is not None and values:
                # X values in seconds over the real duration.
                if len(values) == 1:
                    xs = [0.0]
                else:
                    step = duration / max(1, len(values) - 1)
                    xs = [i * step for i in range(len(values))]
                self.curve.setData(xs, values)
                self.plot.setXRange(0, max(0.1, duration), padding=0.02)
                self.plot.setYRange(-1.0, 1.0, padding=0.05)
            self.status_label.setText(f"Loaded: {audio_path.name} — {_format_ms(self.duration_ms)}")
            self._set_time(0, self.duration_ms)
            self._set_controls_enabled(self.player is not None)
            self._prepare_player()
        except Exception as exc:
            self.status_label.setText(f"Could not draw waveform for {audio_path.name}: {exc}")
            self._set_controls_enabled(self.player is not None)
            self._prepare_player()

    def _prepare_player(self) -> None:
        if self.player is None or self.path is None:
            return
        self.player.setSource(QUrl.fromLocalFile(str(self.path)))
        self._playing_path = self.path

    def play(self) -> None:
        if self.player is None or self.path is None:
            return
        if self._playing_path != self.path:
            self._prepare_player()
        self.player.play()

    def pause(self) -> None:
        if self.player is not None:
            self.player.pause()

    def stop(self) -> None:
        if self.player is not None:
            self.player.stop()
            self.player.setPosition(0)
        if self.playhead is not None:
            self.playhead.setValue(0)
        self._set_time(0, self.duration_ms)

    def _on_plot_clicked(self, event: Any) -> None:
        if self.plot is None or self.path is None or self.player is None:
            return
        try:
            mouse_point = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
            seconds = max(0.0, float(mouse_point.x()))
            if self.duration_ms > 0:
                seconds = min(seconds, self.duration_ms / 1000.0)
            self.player.setPosition(int(seconds * 1000))
            if self.playhead is not None:
                self.playhead.setValue(seconds)
            self.play()
        except Exception:
            pass

    def _on_position_changed(self, position: int) -> None:
        if self.playhead is not None:
            self.playhead.setValue(max(0, position) / 1000.0)
        total = self.duration_ms or (self.player.duration() if self.player is not None else 0)
        self._set_time(position, total)

    def _on_duration_changed(self, duration: int) -> None:
        if duration > 0:
            self.duration_ms = duration
        current = self.player.position() if self.player is not None else 0
        self._set_time(current, self.duration_ms)

    def _on_playback_state_changed(self, _state: Any) -> None:
        # Kept for later styling if needed.
        pass

    def _set_time(self, current_ms: int, total_ms: int) -> None:
        self.time_label.setText(f"{_format_ms(current_ms)} / {_format_ms(total_ms)}")

    def _set_controls_enabled(self, enabled: bool) -> None:
        for btn in (self.play_btn, self.pause_btn, self.stop_btn):
            btn.setEnabled(enabled)


class DotsTtsTab(QWidget):
    """Full featured FrameVision UI helper for dots.tts."""

    output_ready = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, root: Optional[str | Path] = None):
        super().__init__(parent)
        self.paths = build_default_paths(Path(root) if root else None)
        self.process: Optional[QProcess] = None
        self._loading_settings = False
        self._last_output: Optional[Path] = None
        self._voice_presets: list[dict[str, str]] = []
        self.setObjectName("DotsTtsTab")
        self._build_ui()
        self._load_voice_presets()
        self._load_settings()
        self._on_prompt_audio_changed()
        self._refresh_status()
        QTimer.singleShot(250, self._refresh_status)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        title = QLabel("dots.tts")
        title.setObjectName("DotsTtsTitle")
        title.setToolTip("FrameVision helper for local dots.tts text-to-speech / voice cloning. Runs in the separate environments/.dots_tts conda env.")
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        subtitle = QLabel("Voice clone / text-to-speech using the FrameVision dots.tts conda environment.")
        subtitle.setToolTip("The model is run as a subprocess so Torch and the 2B TTS model do not load inside the main FrameVision process.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("opacity: 0.85;")
        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        self.status_label = QLabel("Checking install...")
        self.status_label.setToolTip("Shows whether the helper can find the conda Python, dots.tts repo, and local model folder.")
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        self.tabs.setToolTip("Generate is the normal workflow. Advanced keeps paths and low-level sampler settings out of the main page. Log shows the live console output.")
        root_layout.addWidget(self.tabs, 1)

        self.main_page = self._make_scrolled_page()
        self.advanced_page = self._make_scrolled_page()
        self.logs_page = QWidget()
        self.tabs.addTab(self.main_page, "Generate")
        self.tabs.addTab(self.advanced_page, "Advanced")
        self.tabs.addTab(self.logs_page, "Log")
        self.tabs.currentChanged.connect(self._schedule_save_settings)

        self._build_main_page(self.main_page.widget().layout())
        self._build_advanced_page(self.advanced_page.widget().layout())
        self._build_logs_page()

    def _make_scrolled_page(self) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)
        area.setWidget(content)
        return area

    def _build_main_page(self, layout: QVBoxLayout) -> None:
        text_group = QGroupBox("Text to speak")
        text_group.setToolTip("The new text dots.tts should generate. Longer text usually produces longer audio and takes more time.")
        text_layout = QVBoxLayout(text_group)
        self.text_edit = QTextEdit()
        self.text_edit.setToolTip("Type the sentence or narration to generate. For best tests, use clear punctuation and avoid extremely long blocks until the voice is tuned.")
        self.text_edit.setPlaceholderText("Type the text dots.tts should speak...")
        self.text_edit.setMinimumHeight(140)
        self.text_edit.textChanged.connect(self._schedule_save_settings)
        text_layout.addWidget(self.text_edit)
        layout.addWidget(text_group)

        transcript_group = QGroupBox("Reference transcript")
        transcript_group.setToolTip("Optional transcript for the reference audio. Keep this near the text area because it is text input, while the audio picker is placed near the waveform preview.")
        transcript_layout = QVBoxLayout(transcript_group)
        self.prompt_text_edit = QTextEdit()
        self.prompt_text_edit.setToolTip("Optional transcript for the reference audio. For voice cloning, make it match the spoken words in the reference clip as closely as possible.")
        self.prompt_text_edit.setPlaceholderText("Transcript of the reference audio. For voice cloning, this should match the reference audio exactly.")
        self.prompt_text_edit.setMinimumHeight(90)
        self.prompt_text_edit.textChanged.connect(self._schedule_save_settings)
        transcript_layout.addWidget(self.prompt_text_edit)
        layout.addWidget(transcript_group)

        output_group = QGroupBox("Output")
        output_group.setToolTip("Where generated WAV files are saved. Use 'auto' for a timestamped filename.")
        out_layout = QGridLayout(output_group)
        self.output_dir_edit = QLineEdit(str(self.paths.output_dir))
        self.output_dir_edit.setToolTip("Output folder for generated WAV files. Default: output/dots_tts.")
        self.output_dir_edit.textChanged.connect(self._schedule_save_settings)
        browse_output = QPushButton("Browse")
        browse_output.setToolTip("Choose the output folder for generated audio.")
        browse_output.clicked.connect(self._browse_output_dir)
        self.output_name_edit = QLineEdit("auto")
        self.output_name_edit.setToolTip("Use 'auto' for a timestamped filename, or type a custom .wav filename.")
        self.output_name_edit.setPlaceholderText("auto or filename.wav")
        self.output_name_edit.textChanged.connect(self._schedule_save_settings)
        out_layout.addWidget(QLabel("Folder:"), 0, 0)
        out_layout.addWidget(self.output_dir_edit, 0, 1)
        out_layout.addWidget(browse_output, 0, 2)
        out_layout.addWidget(QLabel("Name:"), 1, 0)
        out_layout.addWidget(self.output_name_edit, 1, 1, 1, 2)
        layout.addWidget(output_group)

        voice_group = QGroupBox("Reference voice")
        voice_group.setToolTip("Optional zero-shot voice clone reference. This is placed near the waveform preview so the selected audio and its waveform stay together.")
        voice_layout = QVBoxLayout(voice_group)

        preset_row = QHBoxLayout()
        self.voice_preset_combo = QComboBox()
        self.voice_preset_combo.setToolTip("Saved local reference voices. Presets store only the audio path and transcript text in presets/setsave/dots_tts_voices.json.")
        self.voice_preset_combo.currentIndexChanged.connect(self._on_voice_preset_selected)
        self.save_voice_btn = QPushButton("Save voice preset")
        self.save_voice_btn.setToolTip("Save the current reference audio path and transcript as a reusable voice preset.")
        self.save_voice_btn.clicked.connect(self._save_current_voice_preset)
        self.delete_voice_btn = QPushButton("Delete preset")
        self.delete_voice_btn.setToolTip("Remove the selected voice preset from the local presets JSON. This does not delete the audio file.")
        self.delete_voice_btn.clicked.connect(self._delete_current_voice_preset)
        preset_row.addWidget(QLabel("Preset:"))
        preset_row.addWidget(self.voice_preset_combo, 1)
        preset_row.addWidget(self.save_voice_btn)
        preset_row.addWidget(self.delete_voice_btn)
        voice_layout.addLayout(preset_row)

        audio_row = QHBoxLayout()
        self.prompt_audio_edit = QLineEdit()
        self.prompt_audio_edit.setToolTip("Optional reference voice audio. A short clean clip with little silence usually works best. The waveform below updates from this file.")
        self.prompt_audio_edit.setPlaceholderText("Optional: reference voice audio, e.g. D:\\test.wav")
        self.prompt_audio_edit.textChanged.connect(self._schedule_save_settings)
        self.prompt_audio_edit.textChanged.connect(self._on_prompt_audio_changed)
        browse_audio = QPushButton("Browse audio")
        browse_audio.setToolTip("Select a WAV, MP3, FLAC, M4A, or OGG reference voice file.")
        browse_audio.clicked.connect(self._browse_prompt_audio)
        clear_audio = QPushButton("Clear")
        clear_audio.setToolTip("Clear the reference audio path. This switches back toward normal TTS without a voice reference.")
        clear_audio.clicked.connect(lambda: self.prompt_audio_edit.clear())
        audio_row.addWidget(QLabel("Audio:"))
        audio_row.addWidget(self.prompt_audio_edit, 1)
        audio_row.addWidget(browse_audio)
        audio_row.addWidget(clear_audio)
        voice_layout.addLayout(audio_row)
        layout.addWidget(voice_group)

        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)
        lower_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Waveform previews")
        preview_group.setToolTip(
            "Compact waveform preview: reference audio on the left, generated output on the right. "
            "Click a waveform position to seek and play from there. Drag the splitter handle to resize."
        )
        preview_layout = QVBoxLayout(preview_group)

        self.waveform_splitter = QSplitter(Qt.Horizontal)
        self.waveform_splitter.setObjectName("dotsTtsWaveformHorizontalSplitter")
        self.waveform_splitter.setChildrenCollapsible(False)
        self.waveform_splitter.setToolTip("Drag the middle handle to change how much width each waveform gets. This position is saved to presets/setsave/dots_tts.json.")

        self.reference_waveform = AudioWaveformPreview("Reference audio")
        self.reference_waveform.setToolTip(
            "Shows the reference voice waveform. Useful for spotting leading silence, clipped audio, or very quiet reference files. "
            "Click the waveform to play from that point."
        )
        self.generated_waveform = AudioWaveformPreview("Generated output")
        self.generated_waveform.setToolTip(
            "Shows the generated dots.tts WAV after generation. Click the waveform to play from that time."
        )
        self.waveform_splitter.addWidget(self.reference_waveform)
        self.waveform_splitter.addWidget(self.generated_waveform)
        self.waveform_splitter.setSizes([1, 1])
        self.waveform_splitter.splitterMoved.connect(self._schedule_save_settings)
        preview_layout.addWidget(self.waveform_splitter)

        quick_group = QGroupBox("Quick settings")
        quick_group.setToolTip("Common settings kept on the main page. Advanced sampler/path settings are in the Advanced tab.")
        quick_layout = QGridLayout(quick_group)
        self.template_combo = QComboBox()
        self.template_combo.setToolTip("Generation template. Helper default: TTS / voice clone. CLI default is no template. Use other modes only when testing dots.tts special workflows.")
        for label, value in TEMPLATE_CHOICES:
            self.template_combo.addItem(label, value)
        self.template_combo.currentIndexChanged.connect(self._schedule_save_settings)
        self.language_combo = QComboBox()
        self.language_combo.setToolTip("Language hint. CLI default is none. Use Auto detect or a specific language code when pronunciation/language detection needs help.")
        for label, value in LANGUAGE_CHOICES:
            self.language_combo.addItem(label, value)
        self.language_combo.currentIndexChanged.connect(self._schedule_save_settings)
        self.steps_spin = QSpinBox()
        self.steps_spin.setToolTip("Sampling steps / NFE. Official CLI default is 10. For dots.tts-mf, the model card recommends 4 steps for the quality/latency tradeoff.")
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(4)
        self.steps_spin.valueChanged.connect(self._schedule_save_settings)
        self.seed_spin = QSpinBox()
        self.seed_spin.setToolTip("Random seed. CLI default is 42. Use the same seed for repeatable tests, or enable random seed for variety.")
        self.seed_spin.setRange(0, 2147483647)
        self.seed_spin.setValue(42)
        self.seed_spin.valueChanged.connect(self._schedule_save_settings)
        self.random_seed_check = QCheckBox("Random seed each run")
        self.random_seed_check.setToolTip("When enabled, a new seed is chosen each run and shown in the seed box. Disable for repeatable comparisons.")
        self.random_seed_check.stateChanged.connect(self._schedule_save_settings)
        quick_layout.addWidget(QLabel("Mode:"), 0, 0)
        quick_layout.addWidget(self.template_combo, 0, 1)
        quick_layout.addWidget(QLabel("Language:"), 0, 2)
        quick_layout.addWidget(self.language_combo, 0, 3)
        quick_layout.addWidget(QLabel("Steps:"), 1, 0)
        quick_layout.addWidget(self.steps_spin, 1, 1)
        quick_layout.addWidget(QLabel("Seed:"), 1, 2)
        quick_layout.addWidget(self.seed_spin, 1, 3)
        quick_layout.addWidget(self.random_seed_check, 2, 0, 1, 4)
        lower_layout.addWidget(quick_group)

        button_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setToolTip("Start dots.tts generation in the separate conda env. The main FrameVision process stays clean.")
        self.generate_btn.clicked.connect(self.generate)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the currently running dots.tts subprocess.")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        self.open_output_btn = QPushButton("Open output")
        self.open_output_btn.setToolTip("Open the last generated WAV file.")
        self.open_output_btn.clicked.connect(self.open_last_output)
        self.open_folder_btn = QPushButton("Open folder")
        self.open_folder_btn.setToolTip("Open the output folder in Windows Explorer.")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.copy_cmd_btn = QPushButton("Copy command")
        self.copy_cmd_btn.setToolTip("Copy the exact console command used by the helper. Useful for testing outside FrameVision.")
        self.copy_cmd_btn.clicked.connect(self.copy_command)
        button_row.addWidget(self.generate_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.copy_cmd_btn)
        button_row.addWidget(self.open_output_btn)
        button_row.addWidget(self.open_folder_btn)
        lower_layout.addLayout(button_row)
        lower_layout.addStretch(1)

        self.preview_vertical_splitter = QSplitter(Qt.Vertical)
        self.preview_vertical_splitter.setObjectName("dotsTtsWaveformVerticalSplitter")
        self.preview_vertical_splitter.setChildrenCollapsible(False)
        self.preview_vertical_splitter.setToolTip(
            "Drag this splitter to make the waveform area taller or smaller without making the whole page crowded. "
            "This position is saved to presets/setsave/dots_tts.json."
        )
        self.preview_vertical_splitter.addWidget(preview_group)
        self.preview_vertical_splitter.addWidget(lower_widget)
        self.preview_vertical_splitter.setSizes([260, 170])
        self.preview_vertical_splitter.setMinimumHeight(360)
        self.preview_vertical_splitter.splitterMoved.connect(self._schedule_save_settings)
        layout.addWidget(self.preview_vertical_splitter)
        layout.addStretch(1)


    def _build_advanced_page(self, layout: QVBoxLayout) -> None:
        install_group = QGroupBox("Install paths")
        install_group.setToolTip("Portable FrameVision paths for the dots.tts conda env, repo, model, and Hugging Face cache.")
        install_form = QFormLayout(install_group)
        self.env_python_edit = QLineEdit(str(self.paths.env_python))
        self.env_python_edit.setToolTip("Python executable inside the dots.tts conda env. Default: environments/.dots_tts/python.exe.")
        self.repo_dir_edit = QLineEdit(str(self.paths.repo_dir))
        self.repo_dir_edit.setToolTip("Local dots.tts repository folder. Default: models/dots_tts/repo. The helper adds repo/src to PYTHONPATH.")
        self.model_dir_edit = QLineEdit(str(self.paths.model_dir))
        self.model_dir_edit.setToolTip("Local model snapshot folder. Default installer target: models/dots_tts/dots.tts-mf.")
        self.cache_dir_edit = QLineEdit(str(self.paths.root / "models" / "dots_tts" / "_hf_cache"))
        self.cache_dir_edit.setToolTip("Hugging Face cache folder kept inside FrameVision for portability/offline cleanup.")
        self.revision_edit = QLineEdit("")
        self.revision_edit.setToolTip("Optional Hugging Face revision/branch/commit. Leave empty for the local/default snapshot.")
        for edit in (self.env_python_edit, self.repo_dir_edit, self.model_dir_edit, self.cache_dir_edit, self.revision_edit):
            edit.textChanged.connect(self._schedule_save_settings)
        install_form.addRow("Conda python:", self._with_browse_file(self.env_python_edit, "Select dots.tts Python", "Python (*.exe);;All files (*.*)"))
        install_form.addRow("Repo folder:", self._with_browse_dir(self.repo_dir_edit))
        install_form.addRow("Model folder:", self._with_browse_dir(self.model_dir_edit))
        install_form.addRow("HF cache dir:", self._with_browse_dir(self.cache_dir_edit))
        install_form.addRow("Revision:", self.revision_edit)
        layout.addWidget(install_group)

        gen_group = QGroupBox("Generation settings")
        gen_group.setToolTip("Advanced dots.tts CLI settings. Defaults are based on the official CLI and dots.tts-mf model card where available.")
        gen_form = QFormLayout(gen_group)
        self.precision_combo = QComboBox()
        self.precision_combo.setToolTip("Inference precision. Official CLI default: bfloat16. On GPUs without good bfloat16 support, float16 may be worth testing.")
        self.precision_combo.addItems(PRECISION_CHOICES)
        self.precision_combo.currentIndexChanged.connect(self._schedule_save_settings)
        self.ode_combo = QComboBox()
        self.ode_combo.setToolTip("ODE solver method. Official CLI default: euler. Keep euler unless the repo adds and documents more solvers.")
        self.ode_combo.addItems(ODE_CHOICES)
        self.ode_combo.setEditable(True)
        self.ode_combo.currentTextChanged.connect(self._schedule_save_settings)
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setToolTip("Classifier-free guidance scale. Official CLI default: 1.2. For dots.tts-mf this has no inference effect because guidance is fused into the distilled student.")
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setSingleStep(0.05)
        self.guidance_spin.setValue(1.2)
        self.guidance_spin.valueChanged.connect(self._schedule_save_settings)
        self.speaker_spin = QDoubleSpinBox()
        self.speaker_spin.setToolTip("Reference speaker embedding scale. Official CLI default: 1.5. Higher may push speaker similarity harder but can also affect stability/naturalness.")
        self.speaker_spin.setRange(0.0, 20.0)
        self.speaker_spin.setDecimals(2)
        self.speaker_spin.setSingleStep(0.05)
        self.speaker_spin.setValue(1.5)
        self.speaker_spin.valueChanged.connect(self._schedule_save_settings)
        self.max_len_spin = QSpinBox()
        self.max_len_spin.setToolTip("Maximum total audio patch count, including prompt and generated audio. Official CLI default: 500. This is not seconds; it is a safety/length cap.")
        self.max_len_spin.setRange(32, 5000)
        self.max_len_spin.setValue(500)
        self.max_len_spin.valueChanged.connect(self._schedule_save_settings)
        self.normalize_check = QCheckBox("Normalize text before inference")
        self.normalize_check.setToolTip("CLI default: off. Enables dots.tts text normalization before inference. Useful for numbers/symbols, but test because it can change wording behavior.")
        self.normalize_check.stateChanged.connect(self._schedule_save_settings)
        self.profile_check = QCheckBox("Profile inference timing")
        self.profile_check.setToolTip("CLI default: off. Adds per-module timing/profiling output to the log; useful for debugging speed, not needed for normal generation.")
        self.profile_check.stateChanged.connect(self._schedule_save_settings)
        gen_form.addRow("Precision:", self.precision_combo)
        gen_form.addRow("ODE method:", self.ode_combo)
        gen_form.addRow("Guidance scale:", self.guidance_spin)
        gen_form.addRow("Speaker scale:", self.speaker_spin)
        gen_form.addRow("Max generate length:", self.max_len_spin)
        gen_form.addRow("Normalize:", self.normalize_check)
        gen_form.addRow("Profiling:", self.profile_check)
        layout.addWidget(gen_group)

        extra_group = QGroupBox("Extra command args")
        extra_group.setToolTip("Raw extra CLI arguments appended at the end. Advanced testing only; invalid args will make dots.tts fail.")
        extra_layout = QVBoxLayout(extra_group)
        self.extra_args_edit = QLineEdit()
        self.extra_args_edit.setToolTip("Optional raw args appended to the command. Leave empty unless you are testing a newly added dots.tts CLI flag.")
        self.extra_args_edit.setPlaceholderText("Optional raw args, advanced use only")
        self.extra_args_edit.textChanged.connect(self._schedule_save_settings)
        extra_layout.addWidget(self.extra_args_edit)
        layout.addWidget(extra_group)

        tools_group = QGroupBox("Tools")
        tools_group.setToolTip("Maintenance actions for checking the install, opening logs, or resetting this helper's saved JSON settings.")
        tools_layout = QHBoxLayout(tools_group)
        self.verify_btn = QPushButton("Verify install")
        self.verify_btn.setToolTip("Run a quick import/GPU/path verification without generating audio.")
        self.verify_btn.clicked.connect(self.verify_install)
        self.open_log_btn = QPushButton("Open log file")
        self.open_log_btn.setToolTip("Open logs/dots_tts_helper.log.")
        self.open_log_btn.clicked.connect(self.open_log_file)
        self.reset_btn = QPushButton("Reset UI settings")
        self.reset_btn.setToolTip("Delete/reset presets/setsave/dots_tts.json. Voice presets are kept.")
        self.reset_btn.clicked.connect(self.reset_settings)
        tools_layout.addWidget(self.verify_btn)
        tools_layout.addWidget(self.open_log_btn)
        tools_layout.addStretch(1)
        tools_layout.addWidget(self.reset_btn)
        layout.addWidget(tools_group)
        layout.addStretch(1)

    def _build_logs_page(self) -> None:
        layout = QVBoxLayout(self.logs_page)
        self.log_edit = QTextEdit()
        self.log_edit.setToolTip("Live console output from the dots.tts subprocess and helper messages.")
        self.log_edit.setReadOnly(True)
        self.log_edit.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.log_edit, 1)
        row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Clear the visible log window only. The saved log file is not deleted.")
        clear_btn.clicked.connect(self.log_edit.clear)
        copy_btn = QPushButton("Copy log")
        copy_btn.setToolTip("Copy the visible log text to the clipboard.")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.log_edit.toPlainText()))
        row.addWidget(clear_btn)
        row.addWidget(copy_btn)
        row.addStretch(1)
        layout.addLayout(row)

    def _with_browse_dir(self, edit: QLineEdit) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        btn = QPushButton("Browse")
        btn.setToolTip("Browse for a folder.")
        btn.clicked.connect(lambda: self._browse_dir_into(edit))
        row.addWidget(btn)
        return widget

    def _with_browse_file(self, edit: QLineEdit, title: str, flt: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        btn = QPushButton("Browse")
        btn.setToolTip("Browse for a file.")
        btn.clicked.connect(lambda: self._browse_file_into(edit, title, flt))
        row.addWidget(btn)
        return widget

    # ------------------------------------------------------------------ Settings
    def _settings_dict(self) -> dict[str, Any]:
        return {
            "text": self.text_edit.toPlainText(),
            "prompt_audio": self.prompt_audio_edit.text(),
            "prompt_text": self.prompt_text_edit.toPlainText(),
            "output_dir": self.output_dir_edit.text(),
            "output_name": self.output_name_edit.text(),
            "template": self.template_combo.currentData(),
            "language": self.language_combo.currentData(),
            "steps": self.steps_spin.value(),
            "seed": self.seed_spin.value(),
            "random_seed": self.random_seed_check.isChecked(),
            "env_python": self.env_python_edit.text(),
            "repo_dir": self.repo_dir_edit.text(),
            "model_dir": self.model_dir_edit.text(),
            "cache_dir": self.cache_dir_edit.text(),
            "revision": self.revision_edit.text(),
            "precision": self.precision_combo.currentText(),
            "ode_method": self.ode_combo.currentText(),
            "guidance_scale": self.guidance_spin.value(),
            "speaker_scale": self.speaker_spin.value(),
            "max_generate_length": self.max_len_spin.value(),
            "normalize_text": self.normalize_check.isChecked(),
            "profile_inference": self.profile_check.isChecked(),
            "extra_args": self.extra_args_edit.text(),
            "ui": {
                "active_tab": self.tabs.currentIndex() if hasattr(self, "tabs") else 0,
                "waveform_splitter_sizes": self.waveform_splitter.sizes() if hasattr(self, "waveform_splitter") else [1, 1],
                "preview_vertical_splitter_sizes": self.preview_vertical_splitter.sizes() if hasattr(self, "preview_vertical_splitter") else [260, 170],
                "last_output": str(self._last_output) if self._last_output else "",
            },
        }

    def _load_settings(self) -> None:
        self._loading_settings = True
        try:
            data = _safe_read_json(self.paths.settings_file, {})
            if not isinstance(data, dict):
                data = {}
            self.text_edit.setPlainText(str(data.get("text", "")))
            self.prompt_audio_edit.setText(str(data.get("prompt_audio", "")))
            self.prompt_text_edit.setPlainText(str(data.get("prompt_text", "")))
            self.output_dir_edit.setText(str(data.get("output_dir", self.paths.output_dir)))
            self.output_name_edit.setText(str(data.get("output_name", "auto")))
            self._set_combo_by_data(self.template_combo, data.get("template", "tts"))
            self._set_combo_by_data(self.language_combo, data.get("language", ""))
            self.steps_spin.setValue(int(data.get("steps", 4)))
            self.seed_spin.setValue(int(data.get("seed", 42)))
            self.random_seed_check.setChecked(bool(data.get("random_seed", False)))
            self.env_python_edit.setText(str(data.get("env_python", self.paths.env_python)))
            self.repo_dir_edit.setText(str(data.get("repo_dir", self.paths.repo_dir)))
            self.model_dir_edit.setText(str(data.get("model_dir", self.paths.model_dir)))
            self.cache_dir_edit.setText(str(data.get("cache_dir", self.paths.root / "models" / "dots_tts" / "_hf_cache")))
            self.revision_edit.setText(str(data.get("revision", "")))
            self._set_combo_by_text(self.precision_combo, data.get("precision", "bfloat16"))
            self._set_combo_by_text(self.ode_combo, data.get("ode_method", "euler"))
            self.guidance_spin.setValue(float(data.get("guidance_scale", 1.2)))
            self.speaker_spin.setValue(float(data.get("speaker_scale", 1.5)))
            self.max_len_spin.setValue(int(data.get("max_generate_length", 500)))
            self.normalize_check.setChecked(bool(data.get("normalize_text", False)))
            self.profile_check.setChecked(bool(data.get("profile_inference", False)))
            self.extra_args_edit.setText(str(data.get("extra_args", "")))

            ui = data.get("ui", {})
            if isinstance(ui, dict):
                self._restore_splitter_sizes(self.waveform_splitter, ui.get("waveform_splitter_sizes"), [1, 1])
                self._restore_splitter_sizes(self.preview_vertical_splitter, ui.get("preview_vertical_splitter_sizes"), [260, 170])
                try:
                    self.tabs.setCurrentIndex(int(ui.get("active_tab", 0)))
                except Exception:
                    self.tabs.setCurrentIndex(0)
                last_output = str(ui.get("last_output", "")).strip()
                if last_output:
                    self._last_output = Path(last_output)
                    self._load_generated_waveform()
        finally:
            self._loading_settings = False

    def _schedule_save_settings(self, *args: Any) -> None:
        if self._loading_settings:
            return
        QTimer.singleShot(150, self._save_settings)

    def _save_settings(self) -> None:
        if self._loading_settings:
            return
        try:
            _safe_write_json(self.paths.settings_file, self._settings_dict())
        except Exception as exc:
            self._append_log(f"Settings save failed: {exc}")

    def reset_settings(self) -> None:
        reply = QMessageBox.question(self, "Reset dots.tts settings", "Reset dots.tts UI settings to defaults?")
        if reply != QMessageBox.Yes:
            return
        try:
            self.paths.settings_file.unlink(missing_ok=True)
        except Exception:
            pass
        self._load_settings()
        self._refresh_status()

    @staticmethod
    def _restore_splitter_sizes(splitter: QSplitter, sizes: Any, fallback: list[int]) -> None:
        """Restore splitter sizes from JSON without using QSettings/registry persistence."""
        try:
            if not isinstance(sizes, (list, tuple)):
                sizes = fallback
            clean = [max(1, int(value)) for value in sizes]
            if len(clean) < splitter.count():
                clean = fallback
            splitter.setSizes(clean[: splitter.count()])
        except Exception:
            splitter.setSizes(fallback[: splitter.count()])

    @staticmethod
    def _set_combo_by_data(combo: QComboBox, value: Any) -> None:
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                combo.setCurrentIndex(idx)
                return

    @staticmethod
    def _set_combo_by_text(combo: QComboBox, value: Any) -> None:
        text = str(value)
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.isEditable():
            combo.setCurrentText(text)

    # ------------------------------------------------------------------ Voice presets
    def _load_voice_presets(self) -> None:
        data = _safe_read_json(self.paths.voice_presets_file, [])
        self._voice_presets = data if isinstance(data, list) else []
        self.voice_preset_combo.blockSignals(True)
        self.voice_preset_combo.clear()
        self.voice_preset_combo.addItem("No preset", "")
        for preset in self._voice_presets:
            name = str(preset.get("name", "Preset"))
            self.voice_preset_combo.addItem(name, name)
        self.voice_preset_combo.blockSignals(False)

    def _save_voice_presets(self) -> None:
        _safe_write_json(self.paths.voice_presets_file, self._voice_presets)

    def _on_voice_preset_selected(self) -> None:
        name = self.voice_preset_combo.currentData()
        if not name:
            return
        for preset in self._voice_presets:
            if preset.get("name") == name:
                self.prompt_audio_edit.setText(str(preset.get("audio", "")))
                self.prompt_text_edit.setPlainText(str(preset.get("text", "")))
                return

    def _save_current_voice_preset(self) -> None:
        audio = self.prompt_audio_edit.text().strip()
        text = self.prompt_text_edit.toPlainText().strip()
        if not audio or not text:
            QMessageBox.warning(self, "Voice preset", "Add both reference audio and reference transcript first.")
            return
        default_name = _slug(Path(audio).stem, "voice")
        name, ok = self._simple_text_dialog("Save voice preset", "Preset name:", default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        updated = False
        for preset in self._voice_presets:
            if preset.get("name") == name:
                preset["audio"] = audio
                preset["text"] = text
                updated = True
                break
        if not updated:
            self._voice_presets.append({"name": name, "audio": audio, "text": text})
        self._save_voice_presets()
        self._load_voice_presets()
        self._set_combo_by_data(self.voice_preset_combo, name)

    def _delete_current_voice_preset(self) -> None:
        name = self.voice_preset_combo.currentData()
        if not name:
            return
        self._voice_presets = [preset for preset in self._voice_presets if preset.get("name") != name]
        self._save_voice_presets()
        self._load_voice_presets()

    def _simple_text_dialog(self, title: str, label: str, value: str) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog
        return QInputDialog.getText(self, title, label, text=value)

    # ------------------------------------------------------------------ Command
    def _output_path(self) -> Path:
        out_dir = Path(self.output_dir_edit.text().strip() or str(self.paths.output_dir))
        raw_name = self.output_name_edit.text().strip()
        if not raw_name or raw_name.lower() == "auto":
            base = _slug(self.text_edit.toPlainText().strip().split("\n", 1)[0][:36], "dots_tts")
            raw_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base}.wav"
        if not raw_name.lower().endswith(".wav"):
            raw_name += ".wav"
        return out_dir / raw_name

    def _build_command(self) -> tuple[str, list[str], dict[str, str], Path]:
        python_exe = Path(self.env_python_edit.text().strip() or str(self.paths.env_python))
        repo_dir = Path(self.repo_dir_edit.text().strip() or str(self.paths.repo_dir))
        model_dir = Path(self.model_dir_edit.text().strip() or str(self.paths.model_dir))
        cache_dir = Path(self.cache_dir_edit.text().strip() or str(self.paths.root / "models" / "dots_tts" / "_hf_cache"))
        output_path = self._output_path()
        text = self.text_edit.toPlainText().strip()
        prompt_audio = self.prompt_audio_edit.text().strip()
        prompt_text = self.prompt_text_edit.toPlainText().strip()

        if not python_exe.is_file():
            raise FileNotFoundError(f"dots.tts Python not found: {python_exe}")
        if not repo_dir.is_dir():
            raise FileNotFoundError(f"dots.tts repo folder not found: {repo_dir}")
        if not (repo_dir / "src" / "dots_tts" / "cli.py").is_file():
            raise FileNotFoundError(f"dots.tts cli.py not found under repo/src: {repo_dir}")
        if not model_dir.exists():
            raise FileNotFoundError(f"dots.tts model folder not found: {model_dir}")
        if not text:
            raise ValueError("Text to speak is empty.")
        if prompt_text and not prompt_audio:
            raise ValueError("Reference transcript requires reference audio.")
        if prompt_audio and not Path(prompt_audio).is_file():
            raise FileNotFoundError(f"Reference audio not found: {prompt_audio}")

        args = [
            "-m",
            "dots_tts.cli",
            "--model-name-or-path",
            str(model_dir),
            "--text",
            text,
            "--output",
            str(output_path),
            "--precision",
            self.precision_combo.currentText().strip() or "bfloat16",
            "--seed",
            str(self._resolved_seed()),
            "--template-name",
            str(self.template_combo.currentData() or "tts"),
            "--ode-method",
            self.ode_combo.currentText().strip() or "euler",
            "--num-steps",
            str(self.steps_spin.value()),
            "--guidance-scale",
            str(self.guidance_spin.value()),
            "--speaker-scale",
            str(self.speaker_spin.value()),
            "--max-generate-length",
            str(self.max_len_spin.value()),
        ]
        language = self.language_combo.currentData()
        if language:
            args.extend(["--language", str(language)])
        revision = self.revision_edit.text().strip()
        if revision:
            args.extend(["--revision", revision])
        if cache_dir:
            args.extend(["--cache-dir", str(cache_dir)])
        if prompt_audio:
            args.extend(["--prompt-audio", prompt_audio])
        if prompt_text:
            args.extend(["--prompt-text", prompt_text])
        if self.normalize_check.isChecked():
            args.append("--normalize-text")
        if self.profile_check.isChecked():
            args.append("--profile-inference")
        extra = self.extra_args_edit.text().strip()
        if extra:
            import shlex
            args.extend(shlex.split(extra, posix=False))

        env = os.environ.copy()
        src_dir = repo_dir / "src"
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(src_dir), str(repo_dir)] + ([existing_pythonpath] if existing_pythonpath else [])
        )
        env["HF_HOME"] = str(self.paths.root / "models" / "dots_tts" / "_hf_home")
        env["HF_HUB_CACHE"] = str(cache_dir)
        env["HF_XET_CACHE"] = str(self.paths.root / "models" / "dots_tts" / "_hf_xet_cache")
        env["PIP_CACHE_DIR"] = str(self.paths.temp_dir / "pip_cache")
        env["TMP"] = str(self.paths.temp_dir / "tmp")
        env["TEMP"] = str(self.paths.temp_dir / "tmp")
        env["TOKENIZERS_PARALLELISM"] = "false"
        return str(python_exe), args, env, repo_dir

    def _resolved_seed(self) -> int:
        if self.random_seed_check.isChecked():
            import random
            seed = random.randint(0, 2147483647)
            self.seed_spin.blockSignals(True)
            self.seed_spin.setValue(seed)
            self.seed_spin.blockSignals(False)
            return seed
        return self.seed_spin.value()

    def generate(self) -> None:
        """Queue through FrameVision when available; fall back to direct standalone run."""
        try:
            if self._enqueue_framevision_queue():
                return
        except Exception as exc:
            self._append_log(f"FrameVision queue enqueue failed, using direct run: {exc}")
        self._generate_direct()

    def _enqueue_framevision_queue(self) -> bool:
        """Send this dots.tts command to FrameVision's normal jobs/pending queue."""
        try:
            from helpers.queue_adapter import enqueue_dotstts_from_widget
        except Exception:
            try:
                from queue_adapter import enqueue_dotstts_from_widget
            except Exception:
                return False
        jid = enqueue_dotstts_from_widget(self)
        if not jid:
            return False
        try:
            self._save_settings()
        except Exception:
            pass
        try:
            self._last_output = self._output_path()
            self._load_generated_waveform(None)
        except Exception:
            pass
        self.status_label.setText(f"Queued dots.tts job: {jid}")
        self._append_log("\n" + "=" * 72)
        self._append_log(f"Queued dots.tts through FrameVision queue: {jid}")
        try:
            win = self.window()
            tabs = win.findChild(QTabWidget)
            if tabs:
                for i in range(tabs.count()):
                    if str(tabs.tabText(i)).strip().lower() == "queue":
                        tabs.setCurrentIndex(i)
                        break
        except Exception:
            pass
        return True

    def _generate_direct(self) -> None:
        if self.process is not None:
            QMessageBox.information(self, "dots.tts", "A dots.tts job is already running.")
            return
        try:
            exe, args, env, workdir = self._build_command()
            output_path = self._output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            self.paths.temp_dir.mkdir(parents=True, exist_ok=True)
            (self.paths.temp_dir / "tmp").mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "dots.tts", str(exc))
            return

        self._save_settings()
        self._last_output = output_path
        self._load_generated_waveform(None)
        self._append_log("\n" + "=" * 72)
        self._append_log(f"Starting dots.tts: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._append_log(_quote_for_display([exe, *args]))

        proc = QProcess(self)
        proc.setProgram(exe)
        proc.setArguments(args)
        proc.setWorkingDirectory(str(workdir))
        qenv = QProcessEnvironment.systemEnvironment()
        for key, value in env.items():
            qenv.insert(str(key), str(value))
        proc.setProcessEnvironment(qenv)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read_process_output)
        proc.finished.connect(self._process_finished)
        proc.errorOccurred.connect(self._process_error)
        self.process = proc
        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("dots.tts is generating...")
        proc.start()
        if not proc.waitForStarted(3000):
            self._process_error(proc.error())

    def stop_generation(self) -> None:
        if self.process is None:
            return
        self._append_log("Stopping dots.tts process...")
        self.process.terminate()
        QTimer.singleShot(2500, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._append_log(data.rstrip())

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        output = self._last_output
        self._append_log(f"Process finished: exit_code={exit_code} status={exit_status.name}")
        self.process = None
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if exit_code == 0 and output and output.is_file():
            self.status_label.setText(f"Done: {output}")
            self._load_generated_waveform(output)
            self.output_ready.emit(str(output))
        else:
            self.status_label.setText("dots.tts failed. Check the Log tab.")
        self._refresh_status()

    def _process_error(self, error: QProcess.ProcessError) -> None:
        self._append_log(f"Process error: {error}")
        self.process = None
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("dots.tts process error. Check the Log tab.")

    def verify_install(self) -> None:
        try:
            exe = Path(self.env_python_edit.text().strip())
            repo = Path(self.repo_dir_edit.text().strip())
            if not exe.is_file():
                raise FileNotFoundError(f"Python not found: {exe}")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), str(repo)])
            cmd = [
                str(exe),
                "-c",
                "import sys, torch, pynini, importlib_resources; import tn, itn; import dots_tts; "
                "print('python:', sys.executable); print('torch:', torch.__version__); "
                "print('cuda:', torch.cuda.is_available()); print('dots_tts import: ok')",
            ]
            self._append_log("\nVerify install:")
            result = subprocess.run(cmd, cwd=str(repo), env=env, text=True, capture_output=True, timeout=60)
            if result.stdout:
                self._append_log(result.stdout.rstrip())
            if result.stderr:
                self._append_log(result.stderr.rstrip())
            if result.returncode != 0:
                raise RuntimeError(f"Verify failed with exit code {result.returncode}")
            QMessageBox.information(self, "dots.tts", "Install verification passed.")
        except Exception as exc:
            QMessageBox.warning(self, "dots.tts", str(exc))

    # ------------------------------------------------------------------ Waveform previews
    def _on_prompt_audio_changed(self, *args: Any) -> None:
        if hasattr(self, "reference_waveform"):
            self.reference_waveform.load_audio(self.prompt_audio_edit.text().strip())

    def _load_generated_waveform(self, path: Optional[Path] = None) -> None:
        target = path if path is not None else self._last_output
        if hasattr(self, "generated_waveform"):
            self.generated_waveform.load_audio(target if target and Path(target).is_file() else None)

    # ------------------------------------------------------------------ Actions
    def copy_command(self) -> None:
        try:
            exe, args, _env, _workdir = self._build_command()
            QApplication.clipboard().setText(_quote_for_display([exe, *args]))
            self._append_log("Command copied to clipboard.")
        except Exception as exc:
            QMessageBox.warning(self, "dots.tts", str(exc))

    def open_last_output(self) -> None:
        target = self._last_output or self._output_path()
        if not target.is_file():
            QMessageBox.information(self, "dots.tts", "No generated output found yet.")
            return
        self._open_path(target)

    def open_output_folder(self) -> None:
        folder = Path(self.output_dir_edit.text().strip() or str(self.paths.output_dir))
        folder.mkdir(parents=True, exist_ok=True)
        self._open_path(folder)

    def open_log_file(self) -> None:
        log_file = self.paths.log_dir / "dots_tts_helper.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not log_file.exists():
            log_file.write_text(self.log_edit.toPlainText(), encoding="utf-8")
        self._open_path(log_file)

    def _open_path(self, path: Path) -> None:
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            QMessageBox.warning(self, "Open", str(exc))

    def _browse_prompt_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference audio",
            str(Path(self.prompt_audio_edit.text()).parent if self.prompt_audio_edit.text() else self.paths.root),
            "Audio files (*.wav *.mp3 *.flac *.m4a *.ogg);;All files (*.*)",
        )
        if path:
            self.prompt_audio_edit.setText(path)

    def _browse_output_dir(self) -> None:
        self._browse_dir_into(self.output_dir_edit)

    def _browse_dir_into(self, edit: QLineEdit) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder", edit.text() or str(self.paths.root))
        if folder:
            edit.setText(folder)

    def _browse_file_into(self, edit: QLineEdit, title: str, flt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, edit.text() or str(self.paths.root), flt)
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------ Logs/status
    def _append_log(self, text: str) -> None:
        self.log_edit.moveCursor(QTextCursor.End)
        self.log_edit.insertPlainText(text + "\n")
        self.log_edit.moveCursor(QTextCursor.End)
        try:
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.paths.log_dir / "dots_tts_helper.log").open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except Exception:
            pass

    def _refresh_status(self) -> None:
        checks = []
        env_ok = Path(self.env_python_edit.text().strip() or str(self.paths.env_python)).is_file()
        repo_ok = (Path(self.repo_dir_edit.text().strip() or str(self.paths.repo_dir)) / "src" / "dots_tts" / "cli.py").is_file()
        model_ok = Path(self.model_dir_edit.text().strip() or str(self.paths.model_dir)).exists()
        checks.append("env ok" if env_ok else "env missing")
        checks.append("repo ok" if repo_ok else "repo missing")
        checks.append("model ok" if model_ok else "model missing")
        if env_ok and repo_ok and model_ok and self.process is None:
            self.status_label.setText("Ready — " + ", ".join(checks))
        elif self.process is None:
            self.status_label.setText("Not ready — " + ", ".join(checks))


def create_dots_tts_tab(parent: Optional[QWidget] = None, root: Optional[str | Path] = None) -> DotsTtsTab:
    return DotsTtsTab(parent=parent, root=root)


# Friendly aliases for future imports.
DotsTTSWidget = DotsTtsTab
DotsTtsWidget = DotsTtsTab
create_tab = create_dots_tts_tab


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DotsTtsTab()
    window.resize(980, 760)
    window.show()
    sys.exit(app.exec())
