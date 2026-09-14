from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
import wave
from array import array
from pathlib import Path


def _detect_root(script_path: Path) -> Path:
    p = script_path.resolve()
    if p.parent.name.lower() == "helpers":
        return p.parent.parent
    return Path.cwd().resolve()


SCRIPT_PATH = Path(__file__).resolve()
FRAMEVISION_ROOT = _detect_root(SCRIPT_PATH)
ENV_DIR = FRAMEVISION_ROOT / "environments" / ".auk"
MODELS_ROOT = FRAMEVISION_ROOT / "models" / "AuK"
REPO_DIR = MODELS_ROOT / "repo"
FLASH_DIR = MODELS_ROOT / "ckpts" / "AuK-Flash"
QWEN_DIR = MODELS_ROOT / "ckpts" / "Qwen2.5-Omni-3B"
FLASH_CKPT = FLASH_DIR / "auk_flash.safetensors"
FLASH_CONFIG = FLASH_DIR / "config.yaml"
FLASH_VAE = FLASH_DIR / "vae.safetensors"
SETTINGS_FILE = FRAMEVISION_ROOT / "presets" / "setsave" / "auk.json"
INSTALLER_FILE = FRAMEVISION_ROOT / "presets" / "extra_env" / "auk_install.py"
LOG_DIR = FRAMEVISION_ROOT / "logs"
LOG_FILE = LOG_DIR / "auk.log"
TEMP_DIR = FRAMEVISION_ROOT / "temp" / "auk"
DEFAULT_OUTPUT_DIR = FRAMEVISION_ROOT / "output" / "audio" / "AuK"


def _env_python() -> Path:
    if os.name == "nt":
        return ENV_DIR / "Scripts" / "python.exe"
    return ENV_DIR / "bin" / "python"


def _install_state() -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not _env_python().is_file():
        missing.append("AuK environment")
    for label, path in (
        ("AuK-Flash checkpoint", FLASH_CKPT),
        ("AuK-Flash config", FLASH_CONFIG),
        ("AuK VAE", FLASH_VAE),
        ("Qwen2.5-Omni-3B", QWEN_DIR / "config.json"),
    ):
        if not path.exists():
            missing.append(label)
    return not missing, missing


def _append_worker_log(message: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{stamp} [worker] {message}\n")
    except Exception:
        pass


def _worker_main(job_path: Path) -> int:
    """Runs inside the dedicated AuK environment. No PySide dependency required."""
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        _append_worker_log(f"job start task={job.get('task', 'unknown')}")

        # The editable AuK install should make this import available in .auk.
        from auk.infer.infer_auk import AukInfer, save_audio  # type: ignore

        config_path = str(Path(job.get("config_path") or FLASH_CONFIG).resolve())
        ckpt_path = str(Path(job.get("ckpt_path") or FLASH_CKPT).resolve())
        qwen_path = str(Path(job.get("qwen_path") or QWEN_DIR).resolve())
        output_path = Path(job["output_path"]).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        device = job.get("device") or None
        if device == "auto":
            device = None
        dtype = str(job.get("dtype") or "bf16")
        cpu_offload = bool(job.get("cpu_offload", True))

        engine = AukInfer(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=device,
            dtype=dtype,
            qwen_path=qwen_path,
            cpu_offload=cpu_offload,
        )

        instruction = str(job["instruction"]).strip()
        audio_path = job.get("audio_path") or None
        if audio_path:
            audio_path = str(Path(audio_path).resolve())

        content = [{"type": "text", "text": instruction}]
        if audio_path:
            content.append({"type": "audio", "audio": audio_path})
        messages = [{"role": "user", "content": content}]

        gen_seconds = job.get("gen_seconds")
        if gen_seconds is not None:
            gen_seconds = float(gen_seconds)
            if gen_seconds <= 0:
                gen_seconds = None

        # Speed editing needs an explicit scaled target duration per upstream cookbook.
        speed_factor = job.get("speed_factor")
        if speed_factor and audio_path:
            import torchaudio  # type: ignore
            info = torchaudio.info(audio_path)
            source_seconds = float(info.num_frames) / float(info.sample_rate)
            gen_seconds = source_seconds / float(speed_factor)

        seed = job.get("seed")
        if seed in (None, "", -1, "-1"):
            seed = None
        else:
            seed = int(seed)

        audio, sr = engine.generate(
            messages,
            audio=audio_path,
            gen_seconds=gen_seconds,
            seed=seed,
        )
        save_audio(audio, sr, str(output_path))

        seconds = float(audio.shape[-1]) / float(sr)
        result = {
            "ok": True,
            "output_path": str(output_path),
            "seconds": seconds,
            "sample_rate": int(sr),
        }
        print("AUK_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)
        _append_worker_log(f"job complete output={output_path} duration={seconds:.2f}s")
        return 0
    except Exception as exc:
        detail = traceback.format_exc()
        print("AUK_ERROR=" + json.dumps({"error": str(exc), "traceback": detail}, ensure_ascii=False), flush=True)
        _append_worker_log("job failed\n" + detail)
        return 1


# Worker mode must be handled before importing PySide6 because the dedicated
# AuK environment deliberately contains inference dependencies only.
if "--auk-worker" in sys.argv:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--auk-worker", required=True)
    ns, _ = parser.parse_known_args()
    raise SystemExit(_worker_main(Path(ns.auk_worker)))


from PySide6.QtCore import QEvent, QObject, QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QDesktopServices
import pyqtgraph as pg
from PySide6.QtCore import QUrl
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
except Exception:
    QAudioOutput = None
    QMediaPlayer = None
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class _NoWheelFilter(QObject):
    """Prevent accidental setting changes while the user scrolls a page."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel and isinstance(obj, (QComboBox, QAbstractSpinBox, QSlider)):
            # Do not change the hovered control. Route the wheel movement to the
            # containing scroll page instead so normal mouse-wheel scrolling keeps working.
            parent = obj.parentWidget()
            while parent is not None and not isinstance(parent, QScrollArea):
                parent = parent.parentWidget()
            if isinstance(parent, QScrollArea):
                bar = parent.verticalScrollBar()
                delta = event.angleDelta().y()
                if delta:
                    bar.setValue(bar.value() - delta)
            event.accept()
            return True
        return super().eventFilter(obj, event)


class _PathPicker(QWidget):
    changed = Signal(str)

    def __init__(self, title: str, audio: bool = True, parent=None):
        super().__init__(parent)
        self.title = title
        self.audio = audio
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("No file selected")
        self.button = QPushButton("Browse…")
        self.clear_button = QPushButton("Clear")
        self.clear_button.setMaximumWidth(64)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button)
        row.addWidget(self.clear_button)
        self.button.clicked.connect(self._browse)
        self.clear_button.clicked.connect(lambda: self.set_path(""))
        self.edit.textChanged.connect(self.changed)

    def _browse(self):
        if self.audio:
            filt = "Audio files (*.wav *.flac *.mp3 *.m4a *.ogg *.opus *.aac);;All files (*.*)"
        else:
            filt = "All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, self.title, self.edit.text(), filt)
        if path:
            self.set_path(path)

    def set_path(self, path: str):
        self.edit.setText(path)

    def path(self) -> str:
        return self.edit.text().strip()


class _TaskPage(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.body = QWidget()
        self.layout_box = QVBoxLayout(self.body)
        self.layout_box.setContentsMargins(12, 12, 12, 12)
        self.layout_box.setSpacing(12)
        self.layout_box.addStretch(1)
        self.setWidget(self.body)

    def add(self, widget: QWidget):
        self.layout_box.insertWidget(self.layout_box.count() - 1, widget)


class _WaveformWidget(QWidget):
    """Interactive pyqtgraph waveform: seek, pan, zoom and selection."""

    playFromRequested = Signal(float)
    playSelectionRequested = Signal(float, float)
    selectionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path: Path | None = None
        self._duration = 0.0
        self._selection_anchor: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(145)
        self.plot.setMaximumHeight(220)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.hideAxis("left")
        self.plot.showAxis("bottom")
        self.plot.getAxis("bottom").setLabel("Time", units="s")
        self.plot.setMenuEnabled(False)

        bg = self.palette().base().color()
        line = self.palette().highlight().color()
        playhead_color = self.palette().text().color()
        self.plot.setBackground(bg)
        self._wave_pen = pg.mkPen(line, width=1.25)

        self._playhead = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            movable=False,
            pen=pg.mkPen(playhead_color, width=1.1),
        )
        self._playhead.setZValue(20)
        self.plot.addItem(self._playhead)

        self._region = pg.LinearRegionItem(
            values=(0.0, 0.0),
            orientation=pg.LinearRegionItem.Vertical,
            movable=True,
        )
        self._region.setZValue(10)
        self._region.hide()
        self.plot.addItem(self._region)
        self._region.sigRegionChanged.connect(self._region_changed)

        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        layout.addWidget(self.plot)

        hint = QLabel("Click: play from position   •   Wheel: zoom   •   Drag: pan   •   Shift+click: select   •   Right-click: options")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    @property
    def duration(self) -> float:
        return self._duration

    def _clamp_time(self, seconds: float) -> float:
        return max(0.0, min(float(seconds), max(0.0, self._duration)))

    def _scene_seconds(self, scene_pos) -> float:
        view_pos = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        return self._clamp_time(view_pos.x())

    def _mouse_clicked(self, event):
        if self._duration <= 0:
            return

        seconds = self._scene_seconds(event.scenePos())

        if event.button() == Qt.MouseButton.LeftButton:
            modifiers = QApplication.keyboardModifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                if self._selection_anchor is None:
                    self._selection_anchor = seconds
                    self.set_selection(seconds, seconds)
                else:
                    start, end = sorted((self._selection_anchor, seconds))
                    self._selection_anchor = None
                    self.set_selection(start, end)
            else:
                self.set_playhead(seconds)
                self.playFromRequested.emit(seconds)
            event.accept()
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event, seconds)
            event.accept()

    def _show_context_menu(self, event, seconds: float):
        menu = QMenu(self)

        play_here = menu.addAction(f"Play from {seconds:.2f} s")
        menu.addSeparator()
        set_start = menu.addAction("Set selection start here")
        set_end = menu.addAction("Set selection end here")
        select_visible = menu.addAction("Select visible range")

        selection = self.selection()
        play_selection = menu.addAction("Play selection")
        zoom_selection = menu.addAction("Zoom to selection")
        clear_selection = menu.addAction("Clear selection")
        have_selection = selection is not None and selection[1] > selection[0]
        play_selection.setEnabled(have_selection)
        zoom_selection.setEnabled(have_selection)
        clear_selection.setEnabled(selection is not None)

        menu.addSeparator()
        reset_zoom = menu.addAction("Reset zoom")

        chosen = menu.exec(event.screenPos().toPoint())

        if chosen == play_here:
            self.set_playhead(seconds)
            self.playFromRequested.emit(seconds)
        elif chosen == set_start:
            current = self.selection()
            end = current[1] if current else seconds
            self.set_selection(seconds, max(seconds, end))
        elif chosen == set_end:
            current = self.selection()
            start = current[0] if current else seconds
            self.set_selection(min(start, seconds), seconds)
        elif chosen == select_visible:
            x_range = self.plot.getPlotItem().vb.viewRange()[0]
            self.set_selection(self._clamp_time(x_range[0]), self._clamp_time(x_range[1]))
        elif chosen == play_selection and have_selection:
            a, b = self.selection()
            self.set_playhead(a)
            self.playSelectionRequested.emit(a, b)
        elif chosen == zoom_selection and have_selection:
            a, b = self.selection()
            pad = max(0.05, (b - a) * 0.06)
            self.plot.setXRange(max(0.0, a - pad), min(self._duration, b + pad), padding=0)
        elif chosen == clear_selection:
            self.clear_selection()
        elif chosen == reset_zoom:
            self.reset_zoom()

    def _region_changed(self):
        if not self._region.isVisible():
            return
        a, b = sorted(self._region.getRegion())
        a = self._clamp_time(a)
        b = self._clamp_time(b)
        self.selectionChanged.emit(a, b)

    def selection(self) -> tuple[float, float] | None:
        if not self._region.isVisible():
            return None
        a, b = sorted(self._region.getRegion())
        return self._clamp_time(a), self._clamp_time(b)

    def set_selection(self, start: float, end: float):
        start, end = sorted((self._clamp_time(start), self._clamp_time(end)))
        self._region.setRegion((start, end))
        self._region.show()
        self.selectionChanged.emit(start, end)

    def clear_selection(self):
        self._selection_anchor = None
        self._region.hide()
        self.selectionChanged.emit(0.0, 0.0)

    def reset_zoom(self):
        if self._duration > 0:
            self.plot.setXRange(0.0, self._duration, padding=0)

    def set_playhead(self, seconds: float):
        self._playhead.setValue(self._clamp_time(seconds))

    def _load_with_torchaudio(self, path: Path):
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        if waveform is None or waveform.numel() == 0:
            raise RuntimeError("Audio file contains no samples")

        sample_rate = max(1, int(sample_rate))
        self._duration = waveform.shape[-1] / float(sample_rate)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        waveform = waveform.detach().to("cpu").float()

        total = int(waveform.numel())
        max_points = 10000
        if total > max_points:
            step = max(1, total // max_points)
            waveform = waveform[::step]

        return waveform.tolist()

    def _load_with_wave(self, path: Path):
        with wave.open(str(path), "rb") as wf:
            channels = max(1, wf.getnchannels())
            sample_width = wf.getsampwidth()
            rate = max(1, wf.getframerate())
            frames = wf.getnframes()
            self._duration = frames / float(rate)
            raw = wf.readframes(frames)

        samples: list[float] = []
        if sample_width == 1:
            vals = raw
            for i in range(0, len(vals), channels):
                frame = vals[i:i + channels]
                if frame:
                    samples.append(sum((v - 128) / 128.0 for v in frame) / len(frame))
        elif sample_width == 2:
            vals = array("h")
            vals.frombytes(raw)
            if sys.byteorder != "little":
                vals.byteswap()
            for i in range(0, len(vals), channels):
                frame = vals[i:i + channels]
                if frame:
                    samples.append(sum(v / 32768.0 for v in frame) / len(frame))
        elif sample_width == 4:
            vals = array("i")
            vals.frombytes(raw)
            if sys.byteorder != "little":
                vals.byteswap()
            for i in range(0, len(vals), channels):
                frame = vals[i:i + channels]
                if frame:
                    samples.append(sum(v / 2147483648.0 for v in frame) / len(frame))

        if not samples:
            raise RuntimeError("Unsupported WAV encoding")

        max_points = 10000
        if len(samples) > max_points:
            step = max(1, len(samples) // max_points)
            samples = samples[::step]
        return samples

    def load_wav(self, path: Path) -> bool:
        self._path = Path(path)
        self._duration = 0.0
        self._selection_anchor = None
        self.plot.clear()

        self.plot.addItem(self._region)
        self.plot.addItem(self._playhead)
        self._region.hide()
        self._playhead.setValue(0.0)

        samples = None
        try:
            samples = self._load_with_torchaudio(self._path)
        except Exception:
            try:
                samples = self._load_with_wave(self._path)
            except Exception:
                samples = None

        if not samples:
            return False

        peak = max(abs(float(v)) for v in samples) if samples else 0.0
        display = [float(v) / peak for v in samples] if peak > 0 else [0.0 for _ in samples]
        count = max(1, len(display))
        if count == 1:
            x_values = [0.0]
        else:
            step_s = self._duration / float(count - 1)
            x_values = [i * step_s for i in range(count)]

        self.plot.plot(x_values, display, pen=self._wave_pen)
        self.plot.setYRange(-1.05, 1.05, padding=0)
        self.reset_zoom()
        return True


class _ResultCard(QFrame):
    playRequested = Signal(str)
    playFromRequested = Signal(str, float)
    playSelectionRequested = Signal(str, float, float)
    stopRequested = Signal()
    openRequested = Signal(str)
    useRequested = Signal(str)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        top = QHBoxLayout()
        self.name_label = QLabel(self.path.name)
        f = self.name_label.font()
        f.setBold(True)
        self.name_label.setFont(f)
        self.duration_label = QLabel("")
        self.duration_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.name_label, 1)
        top.addWidget(self.duration_label)
        root.addLayout(top)

        self.waveform = _WaveformWidget()
        ok = self.waveform.load_wav(self.path)
        if ok:
            seconds = self.waveform.duration
            mins = int(seconds // 60)
            secs = seconds - mins * 60
            self.duration_label.setText(f"{mins}:{secs:04.1f}")
        else:
            self.duration_label.setText("waveform unavailable")
        root.addWidget(self.waveform)

        self.selection_label = QLabel("Selection: none")
        root.addWidget(self.selection_label)
        self.waveform.selectionChanged.connect(self._selection_changed)
        self.waveform.playFromRequested.connect(
            lambda seconds: self.playFromRequested.emit(str(self.path), float(seconds))
        )
        self.waveform.playSelectionRequested.connect(
            lambda start, end: self.playSelectionRequested.emit(
                str(self.path), float(start), float(end)
            )
        )

        buttons = QHBoxLayout()
        self.play_button = QPushButton("Play / Pause")
        self.stop_button = QPushButton("Stop")
        self.use_button = QPushButton("Use in Studio")
        self.open_button = QPushButton("Open file")
        self.path_label = QLabel(str(self.path))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setToolTip(str(self.path))
        buttons.addWidget(self.play_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.use_button)
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.path_label, 1)
        root.addLayout(buttons)

        self.play_button.clicked.connect(lambda: self.playRequested.emit(str(self.path)))
        self.stop_button.clicked.connect(self.stopRequested.emit)
        self.use_button.clicked.connect(lambda: self.useRequested.emit(str(self.path)))
        self.open_button.clicked.connect(lambda: self.openRequested.emit(str(self.path)))

    def _selection_changed(self, start: float, end: float):
        if end > start:
            self.selection_label.setText(
                f"Selection: {start:.2f} – {end:.2f} s  ({end - start:.2f} s)"
            )
        else:
            self.selection_label.setText("Selection: none")

    def set_playhead_seconds(self, seconds: float):
        self.waveform.set_playhead(seconds)


class AuKHelperWidget(QWidget):
    generationStarted = Signal()
    generationFinished = Signal(str)
    generationFailed = Signal(str)
    installStateChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wheel_filter = _NoWheelFilter(self)
        self._process: QProcess | None = None
        self._install_process: QProcess | None = None
        self._job_file: Path | None = None
        self._last_output: Path | None = None
        self._result_cards: dict[str, _ResultCard] = {}
        self._player_path: Path | None = None
        self._selection_stop_ms: int | None = None
        self._audio_output = None
        self._player = None
        if QMediaPlayer is not None and QAudioOutput is not None:
            try:
                self._audio_output = QAudioOutput(self)
                self._audio_output.setVolume(1.0)
                self._player = QMediaPlayer(self)
                self._player.setAudioOutput(self._audio_output)
                self._player.positionChanged.connect(self._player_position_changed)
            except Exception:
                self._audio_output = None
                self._player = None
        self._loading_settings = True
        self._settings = self._read_settings()

        self._build_ui()
        self._apply_settings()
        self._loading_settings = False
        self._connect_save_signals()
        self._load_recent_results()
        self.refresh_install_state()

    # --------------------------- UI helpers ---------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("AuK-Flash Audio Studio")
        font = title.font()
        font.setPointSize(max(font.pointSize() + 3, 12))
        font.setBold(True)
        title.setFont(font)
        self.status_label = QLabel("Checking installation…")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_label)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self._build_generate_tab()
        self._build_results_tab()
        self._build_settings_tab()

        footer = QHBoxLayout()
        self.activity_label = QLabel("Ready")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_job)
        self.open_output_button = QPushButton("Open last output")
        self.open_output_button.setEnabled(False)
        self.open_output_button.clicked.connect(self._open_last_output)
        footer.addWidget(self.activity_label, 1)
        footer.addWidget(self.cancel_button)
        footer.addWidget(self.open_output_button)
        root.addLayout(footer)

        # Install event filter on all controls that normally react to wheel input.
        wheel_widgets = []
        wheel_widgets.extend(self.findChildren(QComboBox))
        wheel_widgets.extend(self.findChildren(QAbstractSpinBox))
        wheel_widgets.extend(self.findChildren(QSlider))
        for widget in wheel_widgets:
            widget.installEventFilter(self._wheel_filter)
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        return box, form

    def _run_button(self, text: str, task_key: str) -> QPushButton:
        button = QPushButton(text)
        button.setMinimumHeight(34)
        button.clicked.connect(lambda: self._start_task(task_key))
        button.setProperty("auk_run_button", True)
        return button

    def _duration_spin(self, default=6.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.5, 120.0)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" s")
        spin.setValue(default)
        return spin

    def _seed_spin(self) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(-1, 2147483647)
        spin.setValue(-1)
        spin.setSpecialValueText("Random (-1)")
        return spin

    @staticmethod
    def _estimate_natural_tts_seconds(text: str) -> float:
        """Approximate normal speech duration for AuK's required gen_seconds."""
        text = (text or "").strip()
        if not text:
            return 0.0

        words = re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE)
        word_count = len(words)

        if word_count >= 2:
            # Neutral speech/narration baseline: about 150 words/minute.
            seconds = word_count / 2.5
        else:
            # Fallback for languages or text without whitespace word boundaries.
            visible_chars = len(re.sub(r"\s+", "", text))
            seconds = visible_chars / 13.0

        seconds += len(re.findall(r"[,;:]", text)) * 0.16
        seconds += len(re.findall(r"[.!?]+", text)) * 0.34
        seconds += len(re.findall(r"\n+", text)) * 0.22
        seconds += 0.35
        return max(0.8, min(120.0, seconds))

    def _toggle_auto_duration(self, prefix: str, enabled: bool):
        spin = getattr(self, f"{prefix}_duration")
        spin.setEnabled(not enabled)
        self._update_tts_duration_estimates()
        if not self._loading_settings:
            self._save_settings()

    def _update_tts_duration_estimates(self):
        for prefix in ("zs", "it"):
            text_widget = getattr(self, f"{prefix}_text", None)
            label = getattr(self, f"{prefix}_estimate", None)
            checkbox = getattr(self, f"{prefix}_auto_duration", None)
            if text_widget is None or label is None or checkbox is None:
                continue

            raw_text = text_widget.toPlainText()
            seconds = self._estimate_natural_tts_seconds(raw_text)
            if seconds <= 0:
                label.setText("Enter text to see an estimate.")
            else:
                words = len(re.findall(r"\b[\w’'-]+\b", raw_text, flags=re.UNICODE))
                mode = "will be used" if checkbox.isChecked() else "estimate only"
                label.setText(
                    f"≈ {seconds:.1f} s at normal pace"
                    + (f" ({words} words)" if words else "")
                    + f" — {mode}"
                )

    # --------------------------- tabs ---------------------------

    def _build_generate_tab(self):
        page = _TaskPage()

        box, form = self._group("Zero-shot TTS — clone a reference voice")
        self.zs_ref = _PathPicker("Select reference voice audio")
        self.zs_text = QTextEdit()
        self.zs_text.setPlaceholderText("Text that should be spoken in the reference voice")
        self.zs_text.setMaximumHeight(110)
        self.zs_duration = self._duration_spin(6.0)
        self.zs_auto_duration = QCheckBox("Auto natural duration")
        self.zs_auto_duration.setChecked(True)
        self.zs_estimate = QLabel("")
        self.zs_estimate.setWordWrap(True)
        self.zs_auto_duration.toggled.connect(
            lambda on: self._toggle_auto_duration("zs", on)
        )
        self.zs_text.textChanged.connect(self._update_tts_duration_estimates)
        self.zs_seed = self._seed_spin()
        form.addRow("Reference audio", self.zs_ref)
        form.addRow("Text", self.zs_text)
        form.addRow("Duration mode", self.zs_auto_duration)
        form.addRow("Estimated natural duration", self.zs_estimate)
        form.addRow("Manual target duration", self.zs_duration)
        form.addRow("Seed", self.zs_seed)
        form.addRow(self._run_button("Generate cloned speech", "zero_shot_tts"))
        page.add(box)

        box, form = self._group("Instruct TTS — create a voice from a description")
        self.it_voice = QTextEdit()
        self.it_voice.setPlaceholderText("Example: Warm calm female narrator in her thirties with a clear natural voice")
        self.it_voice.setMaximumHeight(90)
        self.it_text = QTextEdit()
        self.it_text.setPlaceholderText("Text to speak")
        self.it_text.setMaximumHeight(110)
        self.it_duration = self._duration_spin(6.0)
        self.it_auto_duration = QCheckBox("Auto natural duration")
        self.it_auto_duration.setChecked(True)
        self.it_estimate = QLabel("")
        self.it_estimate.setWordWrap(True)
        self.it_auto_duration.toggled.connect(
            lambda on: self._toggle_auto_duration("it", on)
        )
        self.it_text.textChanged.connect(self._update_tts_duration_estimates)
        self.it_seed = self._seed_spin()
        form.addRow("Voice description", self.it_voice)
        form.addRow("Text", self.it_text)
        form.addRow("Duration mode", self.it_auto_duration)
        form.addRow("Estimated natural duration", self.it_estimate)
        form.addRow("Manual target duration", self.it_duration)
        form.addRow("Seed", self.it_seed)
        form.addRow(self._run_button("Generate described voice", "instruct_tts"))
        page.add(box)

        self.tabs.addTab(page, "Generate Speech")

    def _build_content_tab(self, target_tabs=None):
        page = _TaskPage()

        box, form = self._group("Speech content editing")
        self.ce_audio = _PathPicker("Select speech audio")
        self.ce_mode = QComboBox()
        self.ce_mode.addItems(["Replace", "Insert before", "Insert after", "Remove"])
        self.ce_original = QLineEdit()
        self.ce_original.setPlaceholderText("Original text / anchor / text to remove")
        self.ce_new = QLineEdit()
        self.ce_new.setPlaceholderText("Replacement or inserted text")
        self.ce_duration = self._duration_spin(6.0)
        self.ce_match_source = QCheckBox("Match source duration")
        self.ce_match_source.setChecked(True)
        self.ce_match_source.toggled.connect(lambda on: self.ce_duration.setEnabled(not on))
        self.ce_seed = self._seed_spin()
        form.addRow("Source audio", self.ce_audio)
        form.addRow("Operation", self.ce_mode)
        form.addRow("Original / anchor", self.ce_original)
        form.addRow("New text", self.ce_new)
        dur = QWidget(); drow = QHBoxLayout(dur); drow.setContentsMargins(0,0,0,0); drow.addWidget(self.ce_match_source); drow.addWidget(self.ce_duration); drow.addStretch(1)
        form.addRow("Duration", dur)
        form.addRow("Seed", self.ce_seed)
        form.addRow(self._run_button("Edit speech content", "content_edit"))
        page.add(box)

        box, form = self._group("Lyric editing — preserve melody and voice")
        self.ly_audio = _PathPicker("Select vocal recording")
        self.ly_original = QLineEdit()
        self.ly_new = QLineEdit()
        self.ly_seed = self._seed_spin()
        form.addRow("Vocal recording", self.ly_audio)
        form.addRow("Original lyric", self.ly_original)
        form.addRow("New lyric", self.ly_new)
        form.addRow("Seed", self.ly_seed)
        form.addRow(self._run_button("Change lyric", "lyric_edit"))
        page.add(box)

        (target_tabs or self.tabs).addTab(page, "Speech & Lyrics")

    def _build_voice_tab(self, target_tabs=None):
        page = _TaskPage()
        box, form = self._group("Voice and acoustic editing")
        self.ve_audio = _PathPicker("Select source audio")
        self.ve_task = QComboBox()
        self.ve_task.addItems([
            "Pitch", "Speed", "Volume", "Emotion", "Timbre", "Remove accent",
            "Remove nonverbal sounds", "Add nonverbal sound", "To whisper", "From whisper",
        ])

        self.ve_direction = QComboBox(); self.ve_direction.addItems(["Increase / raise", "Decrease / lower"])
        self.ve_pitch = QSpinBox(); self.ve_pitch.setRange(1, 3); self.ve_pitch.setValue(1); self.ve_pitch.setSuffix(" semitone(s)")
        self.ve_speed = QComboBox(); self.ve_speed.addItems(["0.5", "0.75", "1.25", "1.5", "2.0"]); self.ve_speed.setCurrentText("1.25")
        self.ve_volume = QComboBox(); self.ve_volume.addItems(["5", "10", "15"]); self.ve_volume.setCurrentText("5")
        self.ve_emotion = QComboBox(); self.ve_emotion.addItems(["happy", "angry", "sad", "fearful", "surprised", "disgusted", "calm", "excited"])
        self.ve_timbre = QLineEdit(); self.ve_timbre.setPlaceholderText("Describe the desired voice timbre")
        self.ve_sound = QComboBox(); self.ve_sound.setEditable(True); self.ve_sound.addItems(["breaths", "laughs", "coughs"])
        self.ve_position = QComboBox(); self.ve_position.addItems(["beginning", "end"])
        self.ve_seed = self._seed_spin()

        form.addRow("Source audio", self.ve_audio)
        form.addRow("Task", self.ve_task)
        form.addRow("Direction", self.ve_direction)
        form.addRow("Pitch amount", self.ve_pitch)
        form.addRow("Speed factor", self.ve_speed)
        form.addRow("Volume amount", self.ve_volume)
        form.addRow("Emotion", self.ve_emotion)
        form.addRow("Timbre description", self.ve_timbre)
        form.addRow("Nonverbal sound", self.ve_sound)
        form.addRow("Add position", self.ve_position)
        form.addRow("Seed", self.ve_seed)
        form.addRow(self._run_button("Apply voice edit", "voice_edit"))
        page.add(box)

        hint = QLabel("AuK-Flash uses its fixed distilled 4-step / CFG-off recipe automatically. Speed editing calculates the output duration from the source length and selected speed factor.")
        hint.setWordWrap(True)
        page.add(hint)
        (target_tabs or self.tabs).addTab(page, "Voice & Style")

    def _build_enhance_tab(self, target_tabs=None):
        page = _TaskPage()
        box, form = self._group("Enhancement and separation")
        self.es_audio = _PathPicker("Select source audio")
        self.es_task = QComboBox()
        self.es_task.addItems([
            "Denoise", "Dereverberate", "Enhance speech", "Quality restoration",
            "Speech separation", "Singing voice only", "All human voices", "Target speaker extraction",
        ])
        self.es_restore = QComboBox(); self.es_restore.setEditable(True); self.es_restore.addItems(["telephone effect", "muffling", "clipping", "dropouts"])
        self.es_speaker = QComboBox(); self.es_speaker.setEditable(True); self.es_speaker.addItems(["first", "second", "third", "fourth"])
        self.es_target = QLineEdit(); self.es_target.setPlaceholderText("Words spoken by the speaker you want to keep")
        self.es_seed = self._seed_spin()
        form.addRow("Source audio", self.es_audio)
        form.addRow("Task", self.es_task)
        form.addRow("Restore problem", self.es_restore)
        form.addRow("Speaker order", self.es_speaker)
        form.addRow("Target speaker says", self.es_target)
        form.addRow("Seed", self.es_seed)
        form.addRow(self._run_button("Process audio", "enhance_separate"))
        page.add(box)
        (target_tabs or self.tabs).addTab(page, "Enhance & Separate")

    def _build_custom_tab(self, target_tabs=None):
        page = _TaskPage()
        box, form = self._group("Custom AuK instruction")
        self.cu_audio = _PathPicker("Select optional source/reference audio")
        self.cu_instruction = QTextEdit()
        self.cu_instruction.setPlaceholderText("Natural-language AuK instruction")
        self.cu_instruction.setMinimumHeight(130)
        self.cu_duration_enabled = QCheckBox("Set output duration")
        self.cu_duration = self._duration_spin(6.0)
        self.cu_duration.setEnabled(False)
        self.cu_duration_enabled.toggled.connect(self.cu_duration.setEnabled)
        self.cu_seed = self._seed_spin()
        form.addRow("Audio (optional)", self.cu_audio)
        form.addRow("Instruction", self.cu_instruction)
        dur = QWidget(); row = QHBoxLayout(dur); row.setContentsMargins(0,0,0,0); row.addWidget(self.cu_duration_enabled); row.addWidget(self.cu_duration); row.addStretch(1)
        form.addRow("Duration", dur)
        form.addRow("Seed", self.cu_seed)
        form.addRow(self._run_button("Run custom instruction", "custom"))
        page.add(box)
        (target_tabs or self.tabs).addTab(page, "Custom")

    def _build_results_tab(self):
        page = QWidget()
        self.results_page = page
        outer = QVBoxLayout(page)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        source_box = QGroupBox("Audio Studio source")
        source_layout = QVBoxLayout(source_box)
        source_layout.setContentsMargins(10, 8, 10, 8)

        source_help = QLabel(
            "Choose any audio file or press 'Use in Studio' on a generated result. "
            "The selected file is automatically supplied to all editing tools below."
        )
        source_help.setWordWrap(True)
        source_layout.addWidget(source_help)

        self.studio_source = _PathPicker("Select audio for AuK editing")
        self.studio_source.changed.connect(self._studio_source_changed)
        source_layout.addWidget(self.studio_source)
        outer.addWidget(source_box)

        self.studio_tabs = QTabWidget()
        outer.addWidget(self.studio_tabs, 1)

        # Waveforms/history stays available as the first workspace.
        history = _TaskPage()
        header = QFrame()
        h = QHBoxLayout(header)
        h.setContentsMargins(0, 0, 0, 0)
        info = QLabel(
            "Generated AuK results. Play them here or press 'Use in Studio' "
            "to continue editing that version."
        )
        info.setWordWrap(True)
        clear = QPushButton("Clear result list")
        clear.clicked.connect(self._clear_result_cards)
        h.addWidget(info, 1)
        h.addWidget(clear)
        history.add(header)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 0, 0)
        self.results_layout.setSpacing(10)
        self.results_empty = QLabel("No AuK results found yet.")
        self.results_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_empty.setMinimumHeight(80)
        self.results_layout.addWidget(self.results_empty)
        history.add(self.results_container)

        self.studio_tabs.addTab(history, "Waveforms & Results")

        # Editing features now live beside the results rather than as distant
        # top-level pages.
        self._build_content_tab(self.studio_tabs)
        self._build_voice_tab(self.studio_tabs)
        self._build_enhance_tab(self.studio_tabs)
        self._build_custom_tab(self.studio_tabs)

        self.tabs.addTab(page, "Audio Studio")

    def _build_settings_tab(self):
        page = _TaskPage()

        self.install_box, install_form = self._group("Installation")
        self.install_detail = QLabel()
        self.install_detail.setWordWrap(True)
        self.install_button = QPushButton("Install AuK")
        self.install_button.setMinimumHeight(36)
        self.install_button.clicked.connect(self._start_install)
        install_form.addRow(self.install_detail)
        install_form.addRow(self.install_button)
        page.add(self.install_box)

        box, form = self._group("Runtime")
        self.set_cpu_offload = QCheckBox("CPU offload (recommended for 24 GB GPUs)")
        self.set_cpu_offload.setChecked(True)
        self.set_dtype = QComboBox(); self.set_dtype.addItems(["bf16", "fp16", "fp32"])
        self.set_device = QComboBox(); self.set_device.addItems(["auto", "cuda", "cuda:0", "cpu"])
        self.set_output_dir = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        output_pick = QPushButton("Browse…")
        output_pick.clicked.connect(self._choose_output_dir)
        outrow = QWidget(); h = QHBoxLayout(outrow); h.setContentsMargins(0,0,0,0); h.addWidget(self.set_output_dir, 1); h.addWidget(output_pick)
        form.addRow("Memory", self.set_cpu_offload)
        form.addRow("Data type", self.set_dtype)
        form.addRow("Device", self.set_device)
        form.addRow("Output folder", outrow)
        page.add(box)

        box, form = self._group("Detected paths")
        self.path_env = QLineEdit(str(ENV_DIR)); self.path_env.setReadOnly(True)
        self.path_model = QLineEdit(str(FLASH_DIR)); self.path_model.setReadOnly(True)
        self.path_qwen = QLineEdit(str(QWEN_DIR)); self.path_qwen.setReadOnly(True)
        self.path_settings = QLineEdit(str(SETTINGS_FILE)); self.path_settings.setReadOnly(True)
        form.addRow("Environment", self.path_env)
        form.addRow("AuK-Flash", self.path_model)
        form.addRow("Qwen encoder", self.path_qwen)
        form.addRow("Saved settings", self.path_settings)
        page.add(box)

        box, form = self._group("Diagnostics")
        self.log_view = QTextEdit(); self.log_view.setReadOnly(True); self.log_view.setMinimumHeight(160)
        clear_log = QPushButton("Clear view")
        clear_log.clicked.connect(self.log_view.clear)
        form.addRow(self.log_view)
        form.addRow(clear_log)
        page.add(box)

        self.tabs.addTab(page, "Settings")

    # --------------------------- settings ---------------------------

    def _read_settings(self) -> dict:
        try:
            if SETTINGS_FILE.is_file():
                data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _apply_settings(self):
        s = self._settings
        self.set_cpu_offload.setChecked(bool(s.get("cpu_offload", True)))
        self.set_dtype.setCurrentText(str(s.get("dtype", "bf16")))
        self.set_device.setCurrentText(str(s.get("device", "auto")))
        self.set_output_dir.setText(str(s.get("output_dir", DEFAULT_OUTPUT_DIR)))
        self.zs_duration.setValue(float(s.get("zero_shot_duration", 6.0)))
        self.it_duration.setValue(float(s.get("instruct_duration", 6.0)))
        self.zs_auto_duration.setChecked(bool(s.get("zero_shot_auto_duration", True)))
        self.it_auto_duration.setChecked(bool(s.get("instruct_auto_duration", True)))
        self.zs_duration.setEnabled(not self.zs_auto_duration.isChecked())
        self.it_duration.setEnabled(not self.it_auto_duration.isChecked())
        self._update_tts_duration_estimates()
        self.ce_match_source.setChecked(bool(s.get("content_match_source", True)))
        self.ce_duration.setEnabled(not self.ce_match_source.isChecked())
        self.ve_speed.setCurrentText(str(s.get("speed_factor", "1.25")))
        self.ve_emotion.setCurrentText(str(s.get("emotion", "happy")))

    def _settings_payload(self) -> dict:
        return {
            "cpu_offload": self.set_cpu_offload.isChecked(),
            "dtype": self.set_dtype.currentText(),
            "device": self.set_device.currentText(),
            "output_dir": self.set_output_dir.text().strip() or str(DEFAULT_OUTPUT_DIR),
            "zero_shot_duration": self.zs_duration.value(),
            "instruct_duration": self.it_duration.value(),
            "zero_shot_auto_duration": self.zs_auto_duration.isChecked(),
            "instruct_auto_duration": self.it_auto_duration.isChecked(),
            "content_match_source": self.ce_match_source.isChecked(),
            "speed_factor": self.ve_speed.currentText(),
            "emotion": self.ve_emotion.currentText(),
        }

    def _connect_save_signals(self):
        widgets = [
            self.set_cpu_offload, self.set_dtype, self.set_device, self.set_output_dir,
            self.zs_duration, self.it_duration, self.zs_auto_duration, self.it_auto_duration,
            self.ce_match_source, self.ve_speed, self.ve_emotion,
        ]
        for w in widgets:
            if isinstance(w, QCheckBox):
                w.toggled.connect(self._save_settings)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._save_settings)
            elif isinstance(w, QLineEdit):
                w.editingFinished.connect(self._save_settings)
            elif isinstance(w, QAbstractSpinBox):
                w.editingFinished.connect(self._save_settings)

    def _save_settings(self, *args):
        if self._loading_settings:
            return
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = SETTINGS_FILE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._settings_payload(), indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, SETTINGS_FILE)
        except Exception as exc:
            self._log(f"Could not save settings: {exc}")

    # --------------------------- install ---------------------------

    def refresh_install_state(self):
        ready, missing = _install_state()
        self.install_box.setVisible(not ready)
        self.install_button.setVisible(not ready)
        self.install_detail.setText("Missing: " + ", ".join(missing) if missing else "AuK is installed.")
        self.status_label.setText("AuK ready" if ready else "AuK not installed")
        for b in self.findChildren(QPushButton):
            if b.property("auk_run_button"):
                b.setEnabled(ready and self._process is None)
        self.installStateChanged.emit(ready)

    def _start_install(self):
        if self._install_process is not None:
            return
        if not INSTALLER_FILE.is_file():
            QMessageBox.critical(self, "AuK installer", f"Installer not found:\n{INSTALLER_FILE}")
            return
        self.install_button.setEnabled(False)
        self.install_button.setText("Installing AuK…")
        self.activity_label.setText("Installing AuK…")
        self._log(f"Starting installer: {INSTALLER_FILE}")
        proc = QProcess(self)
        self._install_process = proc
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read_install_output)
        proc.finished.connect(self._install_finished)
        proc.errorOccurred.connect(lambda e: self._log(f"Installer process error: {e}"))
        proc.start(sys.executable, [str(INSTALLER_FILE)])

    def _read_install_output(self):
        if self._install_process:
            text = bytes(self._install_process.readAllStandardOutput()).decode(errors="replace")
            if text:
                self._log(text.rstrip())

    def _install_finished(self, code: int, _status):
        self._read_install_output()
        self._install_process = None
        self.install_button.setEnabled(True)
        self.install_button.setText("Install AuK")
        if code == 0:
            self.activity_label.setText("AuK installation finished")
        else:
            self.activity_label.setText(f"AuK installation failed (code {code})")
        self.refresh_install_state()

    # --------------------------- task building ---------------------------

    def _require_audio(self, picker: _PathPicker, label="source audio") -> str | None:
        path = picker.path()
        if not path:
            QMessageBox.warning(self, "AuK", f"Select {label} first.")
            return None
        if not Path(path).is_file():
            QMessageBox.warning(self, "AuK", f"Audio file not found:\n{path}")
            return None
        return path

    @staticmethod
    def _required_text(value: str, name: str, parent: QWidget) -> str | None:
        value = value.strip()
        if not value:
            QMessageBox.warning(parent, "AuK", f"Enter {name} first.")
            return None
        return value

    def _build_task(self, key: str) -> dict | None:
        job: dict = {"task": key}

        if key == "zero_shot_tts":
            audio = self._require_audio(self.zs_ref, "reference voice audio")
            text = self._required_text(self.zs_text.toPlainText(), "the text to speak", self)
            if not audio or not text: return None
            gen_seconds = (
                self._estimate_natural_tts_seconds(text)
                if self.zs_auto_duration.isChecked()
                else self.zs_duration.value()
            )
            job.update(audio_path=audio, instruction=f'Say the following with the same voice: "{text}"', gen_seconds=gen_seconds, seed=self.zs_seed.value())

        elif key == "instruct_tts":
            desc = self._required_text(self.it_voice.toPlainText(), "a voice description", self)
            text = self._required_text(self.it_text.toPlainText(), "the text to speak", self)
            if not desc or not text: return None
            gen_seconds = (
                self._estimate_natural_tts_seconds(text)
                if self.it_auto_duration.isChecked()
                else self.it_duration.value()
            )
            job.update(audio_path=None, instruction=f'Generate speech based on the following description: "{desc}". The content to speak is: "{text}".', gen_seconds=gen_seconds, seed=self.it_seed.value())

        elif key == "content_edit":
            audio = self._require_audio(self.ce_audio)
            original = self._required_text(self.ce_original.text(), "the original text or anchor", self)
            if not audio or not original: return None
            mode = self.ce_mode.currentText()
            new = self.ce_new.text().strip()
            if mode == "Replace":
                if not new: QMessageBox.warning(self, "AuK", "Enter replacement text first."); return None
                instruction = f"Replace '{original}' with '{new}'."
            elif mode == "Insert before":
                if not new: QMessageBox.warning(self, "AuK", "Enter text to insert first."); return None
                instruction = f"Add '{new}' before '{original}'."
            elif mode == "Insert after":
                if not new: QMessageBox.warning(self, "AuK", "Enter text to insert first."); return None
                instruction = f"Add '{new}' after '{original}'."
            else:
                instruction = f"Remove '{original}'."
            gen_seconds = None if self.ce_match_source.isChecked() else self.ce_duration.value()
            job.update(audio_path=audio, instruction=instruction, gen_seconds=gen_seconds, seed=self.ce_seed.value())

        elif key == "lyric_edit":
            audio = self._require_audio(self.ly_audio, "a vocal recording")
            old = self._required_text(self.ly_original.text(), "the original lyric", self)
            new = self._required_text(self.ly_new.text(), "the new lyric", self)
            if not audio or not old or not new: return None
            job.update(audio_path=audio, instruction=f'Change "{old}" to "{new}" in the vocal recording.', gen_seconds=None, seed=self.ly_seed.value())

        elif key == "voice_edit":
            audio = self._require_audio(self.ve_audio)
            if not audio: return None
            task = self.ve_task.currentText()
            direction_up = self.ve_direction.currentIndex() == 0
            speed_factor = None
            if task == "Pitch":
                verb = "Raise" if direction_up else "Lower"
                instruction = f"{verb} the pitch by {self.ve_pitch.value()} semitones."
            elif task == "Speed":
                speed_factor = float(self.ve_speed.currentText())
                instruction = f"Adjust the speech speed to {speed_factor}x."
            elif task == "Volume":
                verb = "Increase" if direction_up else "Decrease"
                instruction = f"{verb} the volume by {self.ve_volume.currentText()} dB."
            elif task == "Emotion":
                instruction = f"Change the emotion to {self.ve_emotion.currentText()}."
            elif task == "Timbre":
                desc = self._required_text(self.ve_timbre.text(), "a timbre description", self)
                if not desc: return None
                instruction = f'Keep the spoken content unchanged and change the timbre to: "{desc}".'
            elif task == "Remove accent":
                instruction = "Remove the regional accent while preserving the speaker's voice and content."
            elif task == "Remove nonverbal sounds":
                sound = self.ve_sound.currentText().strip() or "breaths"
                instruction = f"Remove all {sound} from the audio."
            elif task == "Add nonverbal sound":
                sound = self.ve_sound.currentText().strip() or "laugh"
                instruction = f"Add a {sound} at the {self.ve_position.currentText()} of the speech."
            elif task == "To whisper":
                instruction = "Convert this speech into a soft whisper while preserving the speaker and content."
            else:
                instruction = "Convert this whispered speech into a normal speaking voice while preserving the speaker and content."
            job.update(audio_path=audio, instruction=instruction, gen_seconds=None, seed=self.ve_seed.value())
            if speed_factor is not None:
                job["speed_factor"] = speed_factor

        elif key == "enhance_separate":
            audio = self._require_audio(self.es_audio)
            if not audio: return None
            task = self.es_task.currentText()
            if task == "Denoise":
                instruction = "Remove only the background noise, preserve everything else, and output audio of the same length."
            elif task == "Dereverberate":
                instruction = "Remove only the room reverberation, preserve everything else, and output audio of the same length."
            elif task == "Enhance speech":
                instruction = "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length."
            elif task == "Quality restoration":
                problem = self.es_restore.currentText().strip() or "muffling"
                instruction = f"Repair the {problem} and restore natural, clear speech."
            elif task == "Speech separation":
                who = self.es_speaker.currentText().strip() or "first"
                instruction = f"Keep only the {who} speaker to start talking and remove all other speakers."
            elif task == "Singing voice only":
                instruction = "Keep only the singing voice and remove everything else."
            elif task == "All human voices":
                instruction = "Keep all human voices, including speech and singing, and remove everything else."
            else:
                target = self._required_text(self.es_target.text(), "words spoken by the target speaker", self)
                if not target: return None
                instruction = f'Keep only the speaker who says "{target}" and remove all other speakers.'
            job.update(audio_path=audio, instruction=instruction, gen_seconds=None, seed=self.es_seed.value())

        elif key == "custom":
            instruction = self._required_text(self.cu_instruction.toPlainText(), "an instruction", self)
            if not instruction: return None
            audio = self.cu_audio.path() or None
            if audio and not Path(audio).is_file():
                QMessageBox.warning(self, "AuK", f"Audio file not found:\n{audio}"); return None
            if not audio and not self.cu_duration_enabled.isChecked():
                QMessageBox.warning(self, "AuK", "Text-only generation needs an output duration.")
                return None
            job.update(audio_path=audio, instruction=instruction, gen_seconds=self.cu_duration.value() if self.cu_duration_enabled.isChecked() else None, seed=self.cu_seed.value())
        else:
            return None

        return job

    # --------------------------- generation ---------------------------

    def _start_task(self, key: str):
        ready, _ = _install_state()
        if not ready:
            self.refresh_install_state()
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            QMessageBox.warning(self, "AuK", "AuK is not installed yet. Use Install AuK in Settings.")
            return
        if self._process is not None:
            QMessageBox.information(self, "AuK", "An AuK job is already running.")
            return

        job = self._build_task(key)
        if not job:
            return

        out_dir = Path(self.set_output_dir.text().strip() or DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = out_dir / f"auk_{key}_{stamp}.wav"

        job.update({
            "output_path": str(output),
            "config_path": str(FLASH_CONFIG),
            "ckpt_path": str(FLASH_CKPT),
            "qwen_path": str(QWEN_DIR),
            "cpu_offload": self.set_cpu_offload.isChecked(),
            "dtype": self.set_dtype.currentText(),
            "device": self.set_device.currentText(),
        })

        if job["device"] == "cpu" and job["cpu_offload"]:
            job["cpu_offload"] = False
            self._log("CPU device selected: CPU offload disabled for this job.")

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self._job_file = TEMP_DIR / f"job_{stamp}_{os.getpid()}.json"
        self._job_file.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")

        proc = QProcess(self)
        self._process = proc
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._read_job_output)
        proc.finished.connect(self._job_finished)
        proc.errorOccurred.connect(lambda e: self._log(f"Worker process error: {e}"))

        self._set_running(True)
        self.activity_label.setText(f"Running {key.replace('_', ' ')}…")
        self._log(f"Starting task: {key}")
        self._log(f"Instruction: {job['instruction']}")
        self._log(f"Output: {output}")
        self.generationStarted.emit()
        proc.start(str(_env_python()), [str(SCRIPT_PATH), "--auk-worker", str(self._job_file)])

    def _set_running(self, running: bool):
        for b in self.findChildren(QPushButton):
            if b.property("auk_run_button"):
                b.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _cancel_job(self):
        if self._process is None:
            return
        self._log("Cancelling AuK job…")
        self.activity_label.setText("Cancelling AuK job…")
        self._process.kill()

    def _read_job_output(self):
        if not self._process:
            return
        text = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.startswith("AUK_RESULT="):
                try:
                    data = json.loads(line[len("AUK_RESULT="):])
                    self._last_output = Path(data["output_path"])
                except Exception:
                    pass
            elif line.startswith("AUK_ERROR="):
                try:
                    data = json.loads(line[len("AUK_ERROR="):])
                    self._log("ERROR: " + data.get("error", "Unknown AuK error"))
                    if data.get("traceback"):
                        self._log(data["traceback"])
                except Exception:
                    self._log(line)
            else:
                self._log(line)

    def _job_finished(self, code: int, _status):
        self._read_job_output()
        output = self._last_output
        self._process = None
        self._set_running(False)
        self.refresh_install_state()

        if self._job_file:
            try:
                self._job_file.unlink(missing_ok=True)
                if TEMP_DIR.is_dir() and not any(TEMP_DIR.iterdir()):
                    TEMP_DIR.rmdir()
            except Exception:
                pass
            self._job_file = None

        if code == 0 and output and output.is_file():
            self.activity_label.setText(f"Finished: {output.name}")
            self.open_output_button.setEnabled(True)
            self._log(f"Finished successfully: {output}")
            self._add_result_card(output, bring_to_front=True)
            self.studio_source.set_path(str(output))
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Audio Studio":
                    self.tabs.setCurrentIndex(i)
                    break
            if hasattr(self, "studio_tabs"):
                self.studio_tabs.setCurrentIndex(0)
            self.generationFinished.emit(str(output))
        else:
            msg = f"AuK job failed (exit code {code}). See Settings > Diagnostics or {LOG_FILE.name}."
            self.activity_label.setText(msg)
            self.generationFailed.emit(msg)
            QMessageBox.warning(self, "AuK", msg)

    # --------------------------- audio studio ---------------------------

    def _studio_source_changed(self, path_text: str):
        """Feed the selected studio source into every source-audio editor."""
        path_text = (path_text or "").strip()
        for picker_name in ("ce_audio", "ly_audio", "ve_audio", "es_audio", "cu_audio"):
            picker = getattr(self, picker_name, None)
            if picker is not None and picker.path() != path_text:
                picker.set_path(path_text)

    def _use_result_in_studio(self, path_text: str):
        path = Path(path_text)
        if not path.is_file():
            QMessageBox.warning(self, "AuK Audio Studio", f"Audio file not found:\n{path}")
            return

        self.studio_source.set_path(str(path.resolve()))

        # Open Audio Studio and start on Speech & Lyrics, the most common
        # follow-up for generated/edited audio. The user can switch to the
        # other processing tabs without selecting the source again.
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Audio Studio":
                self.tabs.setCurrentIndex(i)
                break

        if hasattr(self, "studio_tabs") and self.studio_tabs.count() > 1:
            self.studio_tabs.setCurrentIndex(1)

        self._log(f"Studio source: {path}")

    # --------------------------- results / playback ---------------------------

    def _result_output_dir(self) -> Path:
        return Path(self.set_output_dir.text().strip() or DEFAULT_OUTPUT_DIR)

    def _load_recent_results(self):
        try:
            out_dir = self._result_output_dir()
            if not out_dir.is_dir():
                return
            files = sorted(
                (p for p in out_dir.glob("auk_*.wav") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:12]
            for path in reversed(files):
                self._add_result_card(path, bring_to_front=True)
            if files and not self.studio_source.path():
                self.studio_source.set_path(str(files[0].resolve()))
        except Exception as exc:
            self._log(f"Could not load recent AuK results: {exc}")

    def _add_result_card(self, path: Path, *, bring_to_front: bool = True):
        path = Path(path).resolve()
        key = str(path).lower()
        existing = self._result_cards.get(key)
        if existing is not None:
            if bring_to_front:
                self.results_layout.removeWidget(existing)
                self.results_layout.insertWidget(0, existing)
            return

        self.results_empty.setVisible(False)
        card = _ResultCard(path, self.results_container)
        card.playRequested.connect(self._play_pause_result)
        card.playFromRequested.connect(self._play_from_result)
        card.playSelectionRequested.connect(self._play_selection_result)
        card.stopRequested.connect(self._stop_result)
        card.useRequested.connect(self._use_result_in_studio)
        card.openRequested.connect(self._open_result_file)
        self._result_cards[key] = card
        if bring_to_front:
            self.results_layout.insertWidget(0, card)
        else:
            self.results_layout.addWidget(card)

    def _clear_result_cards(self):
        self._stop_result()
        for card in list(self._result_cards.values()):
            self.results_layout.removeWidget(card)
            card.deleteLater()
        self._result_cards.clear()
        self.results_empty.setVisible(True)

    def _play_pause_result(self, path_text: str):
        path = Path(path_text)
        if not path.is_file():
            QMessageBox.warning(self, "AuK player", f"Audio file not found:\n{path}")
            return
        if self._player is None:
            QMessageBox.warning(
                self,
                "AuK player",
                "Qt Multimedia is not available in this FrameVision PySide6 installation."
            )
            return

        try:
            self._selection_stop_ms = None
            same_file = self._player_path is not None and self._player_path.resolve() == path.resolve()
            playing = self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            if same_file and playing:
                self._player.pause()
                return

            if not same_file:
                self._player.stop()
                self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
                self._player_path = path.resolve()

            self._player.play()
        except Exception as exc:
            QMessageBox.warning(self, "AuK player", f"Could not play audio:\n{exc}")

    def _prepare_player_file(self, path: Path) -> bool:
        if self._player is None:
            return False
        same_file = (
            self._player_path is not None
            and self._player_path.resolve() == path.resolve()
        )
        if not same_file:
            self._player.stop()
            self._player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            self._player_path = path.resolve()
        return True

    def _play_from_result(self, path_text: str, seconds: float):
        path = Path(path_text)
        if not path.is_file():
            return
        if self._player is None:
            QMessageBox.warning(self, "AuK player", "Qt Multimedia is not available.")
            return
        try:
            self._selection_stop_ms = None
            self._prepare_player_file(path)
            self._player.setPosition(max(0, int(float(seconds) * 1000)))
            self._player.play()
        except Exception as exc:
            QMessageBox.warning(self, "AuK player", f"Could not seek/play audio:\n{exc}")

    def _play_selection_result(self, path_text: str, start: float, end: float):
        path = Path(path_text)
        if not path.is_file() or end <= start:
            return
        if self._player is None:
            QMessageBox.warning(self, "AuK player", "Qt Multimedia is not available.")
            return
        try:
            self._prepare_player_file(path)
            self._selection_stop_ms = max(0, int(float(end) * 1000))
            self._player.setPosition(max(0, int(float(start) * 1000)))
            self._player.play()
        except Exception as exc:
            QMessageBox.warning(self, "AuK player", f"Could not play selection:\n{exc}")

    def _player_position_changed(self, position_ms: int):
        if self._player_path is not None:
            key = str(self._player_path.resolve()).lower()
            card = self._result_cards.get(key)
            if card is not None:
                card.set_playhead_seconds(position_ms / 1000.0)

        if self._selection_stop_ms is not None and position_ms >= self._selection_stop_ms:
            self._player.pause()
            self._selection_stop_ms = None

    def _stop_result(self):
        self._selection_stop_ms = None
        if self._player is not None:
            try:
                self._player.stop()
            except Exception:
                pass

    def _open_result_file(self, path_text: str):
        path = Path(path_text)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # --------------------------- misc ---------------------------

    def _choose_output_dir(self):
        start = self.set_output_dir.text().strip() or str(DEFAULT_OUTPUT_DIR)
        path = QFileDialog.getExistingDirectory(self, "Select AuK output folder", start)
        if path:
            self.set_output_dir.setText(path)
            self._save_settings()
            self._clear_result_cards()
            self._load_recent_results()

    def _open_last_output(self):
        if self._last_output and self._last_output.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output)))

    def _log(self, text: str):
        if not text:
            return
        self.log_view.append(text.replace("\r", ""))
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with LOG_FILE.open("a", encoding="utf-8") as f:
                for line in text.replace("\r", "").splitlines() or [""]:
                    f.write(f"{stamp} [gui] {line}\n")
        except Exception:
            pass

    def closeEvent(self, event):
        self._stop_result()
        if self._process is not None:
            self._process.kill()
            self._process.waitForFinished(3000)
        if self._install_process is not None:
            self._install_process.kill()
            self._install_process.waitForFinished(3000)
        self._save_settings()
        super().closeEvent(event)


def create_widget(parent=None) -> AuKHelperWidget:
    """FrameVision-friendly factory."""
    return AuKHelperWidget(parent)


def main() -> int:
    # Standalone mode is self-contained. If launched with another Python,
    # re-launch through FrameVision's dedicated AuK environment where PySide6,
    # pyqtgraph and AuK are installed by auk_install.py.
    target_python = _env_python()
    try:
        current_python = Path(sys.executable).resolve()
        target_resolved = target_python.resolve()
    except Exception:
        current_python = Path(sys.executable)
        target_resolved = target_python

    if target_python.exists() and current_python != target_resolved:
        os.execv(
            str(target_python),
            [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )

    app = QApplication.instance() or QApplication(sys.argv)
    w = AuKHelperWidget()
    w.resize(980, 760)
    w.setWindowTitle("FrameVision — AuK-Flash Audio Studio")
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
