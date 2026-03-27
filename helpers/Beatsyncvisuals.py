from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import sys
import importlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,,
    QProgressBar,
    QPlainTextEdit)

try:
    from .visual_thumbs import VisualThumbManager
except Exception:
    try:
        from visual_thumbs import VisualThumbManager  # type: ignore
    except Exception:
        VisualThumbManager = None  # type: ignore


# ----------------------------- paths / io ---------------------------------


def _framevision_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    probes = [
        here,
        os.path.dirname(here),
        os.path.dirname(os.path.dirname(here)),
    ]
    seen = set()
    for base in probes:
        base = os.path.abspath(base)
        if base in seen:
            continue
        seen.add(base)
        if os.path.isdir(os.path.join(base, "presets", "viz")) or os.path.isdir(os.path.join(base, "presets", "bin")):
            return base
    return os.path.abspath(os.path.join(here, ".."))


def _ensure_framevision_import_paths() -> None:
    root = _framevision_root()
    helpers_dir = os.path.join(root, "helpers")
    for path in (root, helpers_dir):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _find_ffmpeg_from_env() -> str:
    env_ffmpeg = os.environ.get("FV_FFMPEG")
    if env_ffmpeg and os.path.exists(env_ffmpeg):
        return env_ffmpeg

    root = _framevision_root()
    candidates = [
        os.path.join(root, "presets", "bin", "ffmpeg.exe"),
        os.path.join(root, "presets", "bin", "ffmpeg"),
        "ffmpeg.exe",
        "ffmpeg",
    ]
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            return c
    return "ffmpeg"


def _find_ffprobe_from_env() -> str:
    env_ffprobe = os.environ.get("FV_FFPROBE")
    if env_ffprobe and os.path.exists(env_ffprobe):
        return env_ffprobe

    root = _framevision_root()
    candidates = [
        os.path.join(root, "presets", "bin", "ffprobe.exe"),
        os.path.join(root, "presets", "bin", "ffprobe"),
        "ffprobe.exe",
        "ffprobe",
    ]
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            return c
    return "ffprobe"


def _settings_json_path() -> str:
    return os.path.abspath(
        os.path.join(_framevision_root(), "presets", "setsave", "beatsyncvisuals.json")
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, data: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        out, _ = proc.communicate()
        return int(proc.returncode or 0), out or ""
    except Exception as e:
        return 1, f"Failed to run command: {e!r}"


# ----------------------------- probe helpers ------------------------------


def _ffprobe_duration(ffprobe: str, path: str) -> Optional[float]:
    code, out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ]
    )
    if code != 0:
        return None
    try:
        return float(str(out).strip().splitlines()[-1].strip())
    except Exception:
        return None


def _ffprobe_resolution(ffprobe: str, path: str) -> Optional[Tuple[int, int]]:
    code, out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            path,
        ]
    )
    if code != 0:
        return None
    txt = str(out).strip().splitlines()[-1].strip()
    if "x" not in txt:
        return None
    try:
        w_s, h_s = txt.split("x", 1)
        w = int(float(w_s))
        h = int(float(h_s))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        return None
    return None


def _ffprobe_fps(ffprobe: str, path: str) -> float:
    code, out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=0",
            path,
        ]
    )
    if code != 0:
        return 30.0
    avg = None
    raw = None
    for line in str(out).splitlines():
        line = line.strip()
        if line.startswith("avg_frame_rate="):
            avg = line.split("=", 1)[1].strip()
        elif line.startswith("r_frame_rate="):
            raw = line.split("=", 1)[1].strip()
    expr = avg or raw or "30/1"
    try:
        if "/" in expr:
            a, b = expr.split("/", 1)
            a_f = float(a)
            b_f = float(b)
            if b_f > 0:
                return max(1.0, a_f / b_f)
        return max(1.0, float(expr))
    except Exception:
        return 30.0


def _video_has_audio(ffprobe: str, path: str) -> bool:
    code, out = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            path,
        ]
    )
    return code == 0 and bool(str(out).strip())


# ----------------------------- beat analysis ------------------------------


@dataclass
class Beat:
    time: float
    strength: float



def _analyze_beats(video_path: str, ffmpeg: str) -> List[Beat]:
    tmpdir = tempfile.mkdtemp(prefix="fv_bsv_beats_")
    wav_path = os.path.join(tmpdir, "mono.wav")
    try:
        code, out = _run(
            [
                ffmpeg,
                "-y",
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-f",
                "wav",
                wav_path,
            ]
        )
        if code != 0 or not os.path.exists(wav_path):
            raise RuntimeError("Could not extract audio for beat analysis.\n" + out)

        import struct
        import wave

        with wave.open(wav_path, "rb") as wf:
            fr = wf.getframerate()
            total_frames = wf.getnframes()
            win_size = int(fr * 0.05)
            values: List[float] = []
            times: List[float] = []
            read = 0
            while read < total_frames:
                n = min(win_size, total_frames - read)
                raw = wf.readframes(n)
                read += n
                if not raw:
                    break
                count = len(raw) // 2
                if count <= 0:
                    values.append(0.0)
                    times.append(read / float(fr))
                    continue
                fmt = "<" + "h" * count
                samples = struct.unpack(fmt, raw)
                acc = 0.0
                for s in samples:
                    acc += (s / 32768.0) ** 2
                rms = math.sqrt(acc / count)
                values.append(rms)
                times.append(read / float(fr))

        if not values:
            return []

        max_v = max(values) or 1.0
        norm = [v / max_v for v in values]
        mean = sum(norm) / len(norm)
        var = sum((v - mean) ** 2 for v in norm) / len(norm)
        std = math.sqrt(var)
        beat_thr = mean + std * 0.7

        beats: List[Beat] = []
        last_peak = -9999
        min_dist = int(0.15 / 0.05)
        for i in range(1, len(norm) - 1):
            v = norm[i]
            if v < beat_thr:
                continue
            if v >= norm[i - 1] and v >= norm[i + 1]:
                if i - last_peak < min_dist:
                    if beats and v > beats[-1].strength:
                        beats[-1] = Beat(time=times[i], strength=v)
                        last_peak = i
                    continue
                beats.append(Beat(time=times[i], strength=v))
                last_peak = i
        return beats
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------- visuals helpers ----------------------------



def _normalize_visual_mode(mode: str) -> str:
    s = str(mode or "").strip()
    if not s:
        return ""
    # Keep backend mode names exactly as the music visual backend expects.
    # Only normalize fallback file-scan results.
    if s.endswith(".py"):
        s = s[:-3]
    if s == "spectrum":
        return "spectrum"
    if s.startswith("viz:"):
        return s
    return f"viz:{s}"


def _list_visual_modes() -> List[str]:
    found: List[str] = []

    # First try the existing backend so we keep the native mode names when possible.
    try:
        _ensure_framevision_import_paths()
        _configure_visual_backend()
        viz_mod = importlib.import_module("helpers.viz_offline")
        music_mod = importlib.import_module("helpers.music")
        real_list_visual_modes = getattr(viz_mod, "_list_visual_modes", None)
        VisualEngine = getattr(music_mod, "VisualEngine", None)
        if real_list_visual_modes is not None and VisualEngine is not None:
            engine = VisualEngine(parent=None)
            modes = real_list_visual_modes(engine)
            for m in modes or []:
                s = _normalize_visual_mode(m)
                if s:
                    found.append(s)
    except Exception:
        pass

    # Fallback: scan FrameVision root/presets/viz directly.
    viz_dir = os.path.join(_framevision_root(), "presets", "viz")
    if os.path.isdir(viz_dir):
        try:
            for name in sorted(os.listdir(viz_dir), key=lambda x: x.lower()):
                full = os.path.join(viz_dir, name)
                low = name.lower()
                if not os.path.isfile(full):
                    continue
                if low in {"__init__.py"} or low.startswith("_"):
                    continue
                if not low.endswith(".py"):
                    continue
                stem = os.path.splitext(name)[0].strip()
                if not stem or stem.lower().endswith("_thumb"):
                    continue
                found.append(_normalize_visual_mode(stem))
        except Exception:
            pass

    out: List[str] = []
    seen = set()
    for m in found:
        s = str(m).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out



def _configure_visual_backend(ffmpeg_bin: str = "", ffprobe_bin: str = "") -> None:
    """
    Force helpers.music / helpers.viz_offline to use the real FrameVision root
    when this helper is launched directly from the helpers folder.
    """
    try:
        _ensure_framevision_import_paths()
        root = Path(_framevision_root())

        music_mod = importlib.import_module("helpers.music")

        # Fix the backend's idea of the app root.
        try:
            setattr(music_mod, "ROOT", root)
        except Exception:
            pass

        # Force ffmpeg/ffprobe helpers to use FrameVision's normal binaries.
        try:
            if ffmpeg_bin:
                setattr(music_mod, "ffmpeg_path", lambda: str(ffmpeg_bin))
        except Exception:
            pass
        try:
            if ffprobe_bin:
                setattr(music_mod, "ffprobe_path", lambda: str(ffprobe_bin))
        except Exception:
            pass

        # Rebuild visual registry from the correct presets/viz folder.
        try:
            reg = getattr(music_mod, "_VISUAL_REGISTRY", None)
            if isinstance(reg, list):
                reg.clear()
        except Exception:
            pass

        try:
            load_plugins = getattr(music_mod, "_load_visual_plugins", None)
            if callable(load_plugins):
                load_plugins()
        except Exception:
            pass

        # Import viz_offline after music has been corrected.
        try:
            importlib.import_module("helpers.viz_offline")
        except Exception:
            pass
    except Exception:
        pass



class SelectedVisualsDialog(QDialog):
    def __init__(self, parent: QWidget, modes: List[str], selected: List[str], ffmpeg: str):
        super().__init__(parent)
        self.setWindowTitle("Select beat-synced visuals")
        self.resize(940, 620)
        self._modes = list(modes or [])
        self._selected = list(selected or [])
        self._thumbs = None
        if VisualThumbManager is not None:
            try:
                self._thumbs = VisualThumbManager(self, ffmpeg=ffmpeg)
            except Exception:
                self._thumbs = None

        root = QVBoxLayout(self)
        info = QLabel(
            "Choose which installed music-player visual presets may be used for the shuffle.\n"
            "These are the same visual preset names used by the auto music sync overlay.",
            self,
        )
        info.setWordWrap(True)
        root.addWidget(info)

        main = QHBoxLayout()
        root.addLayout(main, 1)

        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.SingleSelection)
        main.addWidget(self.list, 1)

        right = QVBoxLayout()
        main.addLayout(right, 1)

        self.preview = QLabel("Select a visual to preview it here.", self)
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumSize(360, 220)
        self.preview.setStyleSheet(
            "background-color: rgba(0,0,0,80); border: 1px solid rgba(255,255,255,60);"
        )
        right.addWidget(self.preview)

        row_btn = QHBoxLayout()
        self.btn_all_on = QPushButton("All on", self)
        self.btn_all_off = QPushButton("All off", self)
        row_btn.addWidget(self.btn_all_on)
        row_btn.addWidget(self.btn_all_off)
        row_btn.addStretch(1)
        root.addLayout(row_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        root.addWidget(buttons)

        from PySide6.QtWidgets import QListWidgetItem

        for m in self._modes:
            pretty = str(m)
            if pretty.startswith("viz:"):
                pretty = pretty[4:]
            item = QListWidgetItem(pretty, self.list)
            item.setData(Qt.UserRole, m)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if m in self._selected else Qt.Unchecked)
            if self._thumbs is not None:
                try:
                    icon = self._thumbs.icon_for_mode(str(m))
                    if isinstance(icon, QIcon) and not icon.isNull():
                        item.setIcon(icon)
                except Exception:
                    pass

        self.list.currentItemChanged.connect(self._update_preview)
        self.btn_all_on.clicked.connect(self._set_all_on)
        self.btn_all_off.clicked.connect(self._set_all_off)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _set_all_on(self) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item is not None:
                item.setCheckState(Qt.Checked)

    def _set_all_off(self) -> None:
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item is not None:
                item.setCheckState(Qt.Unchecked)

    def _update_preview(self) -> None:
        item = self.list.currentItem()
        if item is None:
            self.preview.setText("Select a visual to preview it here.")
            self.preview.setPixmap(None)
            return
        mode = str(item.data(Qt.UserRole) or "")
        label = str(item.text() or mode)
        pm = None
        if self._thumbs is not None:
            try:
                pm = self._thumbs.preview_pixmap_for_mode(mode)
            except Exception:
                pm = None
        if pm is None or getattr(pm, "isNull", lambda: True)():
            self.preview.setPixmap(None)
            self.preview.setText(label)
            return
        self.preview.setText("")
        self.preview.setPixmap(pm)

    def selected_modes(self) -> List[str]:
        out: List[str] = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item is not None and item.checkState() == Qt.Checked:
                mode = str(item.data(Qt.UserRole) or "")
                if mode:
                    out.append(mode)
        return out


# ----------------------------- main widget --------------------------------




class _RenderWorker(QObject):
    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, tool: "BeatSyncVisualsTool", video: str, out_dir: str) -> None:
        super().__init__()
        self.tool = tool
        self.video = video
        self.out_dir = out_dir

    def run(self) -> None:
        try:
            out_path = self.tool._render_sync(self.video, self.out_dir, self.progress.emit)
            self.finished.emit(True, str(out_path or ""))
        except Exception as e:
            self.finished.emit(False, str(e))


class BeatSyncVisualsTool(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ffmpeg = _find_ffmpeg_from_env()
        self._ffprobe = _find_ffprobe_from_env()
        self._settings_path = _settings_json_path()
        self._loading_settings = False
        self._available_modes: List[str] = []
        self._selected_modes: List[str] = []
        self._render_thread = None
        self._render_worker = None
        self._build_ui()
        self._load_settings()
        self._refresh_modes()
        self._update_ui_state()

    # ----------------------------- ui -------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)

        body = QWidget(scroll)
        scroll.setWidget(body)

        root = QVBoxLayout(body)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        title = QLabel("Beat-synced visuals", body)
        title.setStyleSheet("font-size: 16px; font-weight: 600;")
        root.addWidget(title)

        desc = QLabel(
            "Overlay FrameVision music-player visuals on top of an existing music video.\n"
            "You can keep one visual for the whole clip, shuffle from all visuals, or shuffle from only your chosen visuals.",
            body,
        )
        desc.setWordWrap(True)
        root.addWidget(desc)

        form = QFormLayout()
        root.addLayout(form)

        row_video = QHBoxLayout()
        self.edit_video = QLineEdit(body)
        self.btn_browse_video = QPushButton("Browse", body)
        row_video.addWidget(self.edit_video, 1)
        row_video.addWidget(self.btn_browse_video)
        form.addRow("Music video:", row_video)

        row_out = QHBoxLayout()
        self.edit_output = QLineEdit(body)
        self.btn_browse_output = QPushButton("Browse", body)
        row_out.addWidget(self.edit_output, 1)
        row_out.addWidget(self.btn_browse_output)
        form.addRow("Output folder:", row_out)

        row_mode = QHBoxLayout()
        self.combo_strategy = QComboBox(body)
        self.combo_strategy.addItem("1 visual for the whole video", "single")
        self.combo_strategy.addItem("Shuffle (all visuals)", "shuffle_all")
        self.combo_strategy.addItem("Shuffle (selected visuals)", "shuffle_selected")
        row_mode.addWidget(self.combo_strategy, 1)
        form.addRow("Visuals:", row_mode)

        row_single = QHBoxLayout()
        self.combo_single_visual = QComboBox(body)
        row_single.addWidget(self.combo_single_visual, 1)
        form.addRow("Whole-video visual:", row_single)

        row_select = QHBoxLayout()
        self.btn_select_visuals = QPushButton("Select my own…", body)
        self.label_selected_summary = QLabel("", body)
        self.label_selected_summary.setWordWrap(True)
        row_select.addWidget(self.btn_select_visuals)
        row_select.addWidget(self.label_selected_summary, 1)
        form.addRow("Selected visuals:", row_select)

        row_switch = QHBoxLayout()
        self.combo_switch_mode = QComboBox(body)
        self.combo_switch_mode.addItem("By time", "time")
        self.combo_switch_mode.addItem("By beats", "beats")
        row_switch.addWidget(self.combo_switch_mode)
        form.addRow("Change visual:", row_switch)

        # time controls
        row_time = QHBoxLayout()
        self.slider_seconds = QSlider(Qt.Horizontal, body)
        self.slider_seconds.setRange(1, 99)
        self.spin_seconds = QSpinBox(body)
        self.spin_seconds.setRange(1, 99)
        row_time.addWidget(self.slider_seconds, 1)
        row_time.addWidget(self.spin_seconds)
        form.addRow("Seconds:", row_time)

        # beat controls
        row_beats = QHBoxLayout()
        self.slider_beats = QSlider(Qt.Horizontal, body)
        self.slider_beats.setRange(4, 128)
        self.slider_beats.setSingleStep(4)
        self.slider_beats.setPageStep(4)
        self.spin_beats = QSpinBox(body)
        self.spin_beats.setRange(4, 128)
        self.spin_beats.setSingleStep(4)
        row_beats.addWidget(self.slider_beats, 1)
        row_beats.addWidget(self.spin_beats)
        form.addRow("Beats:", row_beats)

        row_alpha_toggle = QHBoxLayout()
        self.check_transparency = QCheckBox("Use transparency", body)
        row_alpha_toggle.addWidget(self.check_transparency)
        row_alpha_toggle.addStretch(1)
        form.addRow("Transparency:", row_alpha_toggle)

        row_alpha = QHBoxLayout()
        self.slider_alpha = QSlider(Qt.Horizontal, body)
        self.slider_alpha.setRange(0, 100)
        self.spin_alpha = QSpinBox(body)
        self.spin_alpha.setRange(0, 100)
        row_alpha.addWidget(self.slider_alpha, 1)
        row_alpha.addWidget(self.spin_alpha)
        form.addRow("Opacity %:", row_alpha)

        self.label_info = QLabel("", body)
        self.label_info.setWordWrap(True)
        root.addWidget(self.label_info)

        row_actions = QHBoxLayout()
        self.btn_render = QPushButton("Create visualized video", body)
        row_actions.addWidget(self.btn_render)
        row_actions.addStretch(1)
        root.addLayout(row_actions)

        self.label_status = QLabel("Ready.", body)
        self.label_status.setWordWrap(True)
        root.addWidget(self.label_status)

        # signal wiring
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.btn_browse_output.clicked.connect(self._browse_output)
        self.btn_select_visuals.clicked.connect(self._choose_selected_visuals)
        self.btn_render.clicked.connect(self._render)
        self.combo_strategy.currentIndexChanged.connect(self._update_ui_state)
        self.combo_switch_mode.currentIndexChanged.connect(self._update_ui_state)
        self.check_transparency.toggled.connect(self._update_ui_state)
        self.edit_video.editingFinished.connect(self._save_settings)
        self.edit_output.editingFinished.connect(self._save_settings)

        self.slider_seconds.valueChanged.connect(self.spin_seconds.setValue)
        self.spin_seconds.valueChanged.connect(self.slider_seconds.setValue)
        self.slider_seconds.valueChanged.connect(self._save_settings)
        self.spin_seconds.valueChanged.connect(self._save_settings)

        self.slider_beats.valueChanged.connect(self._sync_beats_from_slider)
        self.spin_beats.valueChanged.connect(self._sync_beats_from_spin)

        self.slider_alpha.valueChanged.connect(self.spin_alpha.setValue)
        self.spin_alpha.valueChanged.connect(self.slider_alpha.setValue)
        self.slider_alpha.valueChanged.connect(self._save_settings)
        self.spin_alpha.valueChanged.connect(self._save_settings)

        self.combo_single_visual.currentIndexChanged.connect(self._save_settings)
        self.combo_strategy.currentIndexChanged.connect(self._save_settings)
        self.combo_switch_mode.currentIndexChanged.connect(self._save_settings)
        self.check_transparency.toggled.connect(self._save_settings)

    def _sync_beats_from_slider(self, value: int) -> None:
        snap = max(4, min(128, int(round(value / 4.0) * 4)))
        if self.slider_beats.value() != snap:
            self.slider_beats.blockSignals(True)
            self.slider_beats.setValue(snap)
            self.slider_beats.blockSignals(False)
        if self.spin_beats.value() != snap:
            self.spin_beats.blockSignals(True)
            self.spin_beats.setValue(snap)
            self.spin_beats.blockSignals(False)
        self._save_settings()

    def _sync_beats_from_spin(self, value: int) -> None:
        snap = max(4, min(128, int(round(value / 4.0) * 4)))
        if self.spin_beats.value() != snap:
            self.spin_beats.blockSignals(True)
            self.spin_beats.setValue(snap)
            self.spin_beats.blockSignals(False)
        if self.slider_beats.value() != snap:
            self.slider_beats.blockSignals(True)
            self.slider_beats.setValue(snap)
            self.slider_beats.blockSignals(False)
        self._save_settings()

    # ----------------------------- settings --------------------------------
    def _default_output_dir(self) -> str:
        return os.path.abspath(os.path.join(_framevision_root(), "output", "videovisuals"))

    def _load_settings(self) -> None:
        self._loading_settings = True
        try:
            data = _read_json(self._settings_path)
            self.edit_video.setText(str(data.get("video_path") or ""))
            self.edit_output.setText(str(data.get("output_dir") or self._default_output_dir()))
            self.slider_seconds.setValue(int(data.get("seconds") or 4))
            self.slider_beats.setValue(int(data.get("beats") or 8))
            self.slider_alpha.setValue(int(data.get("opacity_percent") or 25))
            self.check_transparency.setChecked(bool(data.get("use_transparency", True)))

            strategy = str(data.get("strategy") or "single")
            for i in range(self.combo_strategy.count()):
                if self.combo_strategy.itemData(i) == strategy:
                    self.combo_strategy.setCurrentIndex(i)
                    break

            switch_mode = str(data.get("switch_mode") or "time")
            for i in range(self.combo_switch_mode.count()):
                if self.combo_switch_mode.itemData(i) == switch_mode:
                    self.combo_switch_mode.setCurrentIndex(i)
                    break

            self._selected_modes = [_normalize_visual_mode(x) for x in (data.get("selected_modes") or []) if str(x)]
            self._pending_single_mode = _normalize_visual_mode(data.get("single_mode") or "")
        finally:
            self._loading_settings = False

    def _save_settings(self, *args) -> None:
        if self._loading_settings:
            return
        data = {
            "video_path": self.edit_video.text().strip(),
            "output_dir": self.edit_output.text().strip() or self._default_output_dir(),
            "strategy": str(self.combo_strategy.currentData() or "single"),
            "single_mode": _normalize_visual_mode(self.combo_single_visual.currentData() or ""),
            "selected_modes": [_normalize_visual_mode(m) for m in self._selected_modes],
            "switch_mode": str(self.combo_switch_mode.currentData() or "time"),
            "seconds": int(self.spin_seconds.value()),
            "beats": int(self.spin_beats.value()),
            "use_transparency": bool(self.check_transparency.isChecked()),
            "opacity_percent": int(self.spin_alpha.value()),
        }
        _write_json(self._settings_path, data)
        self._update_selected_summary()

    # ----------------------------- visuals list ----------------------------
    def _refresh_modes(self) -> None:
        self._available_modes = _list_visual_modes()
        self.combo_single_visual.clear()
        for m in self._available_modes:
            pretty = str(m)
            if pretty.startswith("viz:"):
                pretty = pretty[4:]
            icon = QIcon()
            if VisualThumbManager is not None:
                try:
                    manager = VisualThumbManager(self, ffmpeg=self._ffmpeg)
                    icon = manager.icon_for_mode(str(m))
                except Exception:
                    icon = QIcon()
            if icon.isNull():
                self.combo_single_visual.addItem(pretty, m)
            else:
                self.combo_single_visual.addItem(icon, pretty, m)

        pending = getattr(self, "_pending_single_mode", "")
        if pending:
            for i in range(self.combo_single_visual.count()):
                if str(self.combo_single_visual.itemData(i) or "") == pending:
                    self.combo_single_visual.setCurrentIndex(i)
                    break
        elif self.combo_single_visual.count() > 0:
            self.combo_single_visual.setCurrentIndex(0)

        self._selected_modes = [m for m in self._selected_modes if m in self._available_modes]
        self._update_selected_summary()

    def _update_selected_summary(self) -> None:
        if not self._selected_modes:
            self.label_selected_summary.setText("No visuals selected yet.")
            return
        pretty = []
        for m in self._selected_modes[:6]:
            txt = str(m)
            if txt.startswith("viz:"):
                txt = txt[4:]
            pretty.append(txt)
        extra = ""
        if len(self._selected_modes) > 6:
            extra = f" (+{len(self._selected_modes) - 6} more)"
        self.label_selected_summary.setText(", ".join(pretty) + extra)

    # ----------------------------- state -----------------------------------
    def _set_status(self, text: str) -> None:
        try:
            self.label_status.setText(str(text))
        except Exception:
            pass

    def _append_log(self, text: str) -> None:
        try:
            self.log_box.appendPlainText(str(text))
        except Exception:
            pass

    def _set_busy_ui(self, busy: bool) -> None:
        try:
            if busy:
                self.progress.setRange(0, 0)
                self.progress.show()
            else:
                self.progress.setRange(0, 1)
                self.progress.setValue(1)
                self.progress.hide()
        except Exception:
            pass
        self.label_status.setText(str(text))

    def _update_ui_state(self) -> None:
        strategy = str(self.combo_strategy.currentData() or "single")
        switch_mode = str(self.combo_switch_mode.currentData() or "time")
        is_single = strategy == "single"
        is_selected_shuffle = strategy == "shuffle_selected"

        self.combo_single_visual.setEnabled(is_single)
        self.btn_select_visuals.setEnabled(is_selected_shuffle)
        self.combo_switch_mode.setEnabled(not is_single)
        self.slider_seconds.setEnabled((not is_single) and switch_mode == "time")
        self.spin_seconds.setEnabled((not is_single) and switch_mode == "time")
        self.slider_beats.setEnabled((not is_single) and switch_mode == "beats")
        self.spin_beats.setEnabled((not is_single) and switch_mode == "beats")
        self.slider_alpha.setEnabled(self.check_transparency.isChecked())
        self.spin_alpha.setEnabled(self.check_transparency.isChecked())

        if is_single:
            self.label_info.setText(
                "One visual preset will stay active for the full video."
            )
        elif switch_mode == "time":
            self.label_info.setText(
                "The helper will change visuals every N seconds."
            )
        else:
            self.label_info.setText(
                "The helper will analyze the video's audio and change visuals every N detected beats."
            )

    # ----------------------------- browse ----------------------------------
    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select music video",
            self.edit_video.text().strip() or "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpeg *.mpg);;All files (*.*)",
        )
        if path:
            self.edit_video.setText(path)
            self._save_settings()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.edit_output.text().strip() or self._default_output_dir(),
        )
        if path:
            self.edit_output.setText(path)
            self._save_settings()

    def _choose_selected_visuals(self) -> None:
        if not self._available_modes:
            QMessageBox.warning(
                self,
                "No visuals found",
                "No music-player visual presets were found.\n\nLooked in: " + os.path.join(_framevision_root(), "presets", "viz"),
                QMessageBox.Ok,
            )
            return
        dlg = SelectedVisualsDialog(self, self._available_modes, self._selected_modes, self._ffmpeg)
        if dlg.exec() == QDialog.Accepted:
            self._selected_modes = [_normalize_visual_mode(m) for m in dlg.selected_modes()]
            self._update_selected_summary()
            self._save_settings()

    # ----------------------------- render helpers --------------------------
    def _validate(self) -> Optional[str]:
        video = self.edit_video.text().strip()
        if not video:
            return "Select a music video first."
        if not os.path.isfile(video):
            return "The selected music video does not exist."
        if not self._available_modes:
            return "No beat-synced visual presets were found."
        strategy = str(self.combo_strategy.currentData() or "single")
        if strategy == "single" and not str(self.combo_single_visual.currentData() or ""):
            return "Pick a visual for the whole video."
        if strategy == "shuffle_selected" and not self._selected_modes:
            return "Select at least one custom visual for the shuffle."
        if str(self.combo_switch_mode.currentData() or "time") == "beats" and not _video_has_audio(self._ffprobe, video):
            return "Beat-based switching needs a video that contains audio."
        return None

    def _build_segment_plan(self, duration: float, video_path: str) -> Tuple[List[Tuple[float, float, str]], Dict[str, str]]:
        strategy = str(self.combo_strategy.currentData() or "single")
        if strategy == "single":
            mode = _normalize_visual_mode(self.combo_single_visual.currentData() or "")
            return [(0.0, max(0.0, duration), "whole")], {"whole": mode}

        switch_mode = str(self.combo_switch_mode.currentData() or "time")
        boundaries: List[float] = [0.0]
        if switch_mode == "time":
            step = max(1, int(self.spin_seconds.value()))
            t = float(step)
            while t < duration:
                boundaries.append(float(t))
                t += float(step)
        else:
            beats = _analyze_beats(video_path, self._ffmpeg)
            if not beats:
                raise RuntimeError("No beats were detected in the video's audio.")
            beat_step = max(4, int(self.spin_beats.value()))
            idx = beat_step - 1
            while idx < len(beats):
                t = float(beats[idx].time)
                if 0.0 < t < duration:
                    boundaries.append(t)
                idx += beat_step

        boundaries.append(duration)
        clean: List[float] = []
        for t in boundaries:
            if clean and abs(clean[-1] - t) < 0.02:
                continue
            clean.append(max(0.0, min(duration, float(t))))
        if clean[-1] < duration:
            clean.append(duration)

        plan: List[Tuple[float, float, str]] = []
        overrides: Dict[str, str] = {}

        if strategy == "shuffle_all":
            pool = [_normalize_visual_mode(m) for m in self._available_modes]
        else:
            pool = [_normalize_visual_mode(m) for m in self._selected_modes]
        if not pool:
            raise RuntimeError("No visuals are available for the chosen shuffle mode.")

        prev_mode = None
        for i in range(len(clean) - 1):
            start_t = float(clean[i])
            end_t = float(clean[i + 1])
            if end_t <= start_t + 0.02:
                continue
            label = f"seg_{i:04d}"
            choices = [m for m in pool if m != prev_mode] or list(pool)
            mode = str(__import__("random").choice(choices))
            overrides[label] = mode
            plan.append((start_t, end_t, label))
            prev_mode = mode

        if not plan:
            label = "whole"
            mode = str(__import__("random").choice(pool))
            plan = [(0.0, duration, label)]
            overrides[label] = mode
        return plan, overrides

    def _render_visual_track(self, audio_path: str, out_video: str, resolution: Tuple[int, int], fps: int,
                             section_map: List[Tuple[float, float, str]], overrides: Dict[str, str]) -> None:
        try:
            _ensure_framevision_import_paths()
            _configure_visual_backend(self._ffmpeg, self._ffprobe)
            viz_mod = importlib.import_module("helpers.viz_offline")
            render_visual_track = getattr(viz_mod, "render_visual_track", None)
            if render_visual_track is None:
                raise RuntimeError("helpers.viz_offline does not expose render_visual_track")
        except Exception as e:
            raise RuntimeError(
                "Could not import helpers.viz_offline.render_visual_track.\n"
                f"{e}"
            )

        ok = render_visual_track(
            audio_path=audio_path,
            out_video=out_video,
            ffmpeg_bin=self._ffmpeg,
            resolution=(int(resolution[0]), int(resolution[1])),
            fps=int(fps),
            strategy=2,
            segment_boundaries=None,
            section_map=section_map,
            section_visual_overrides=overrides,
        )
        if not ok or not os.path.exists(out_video):
            sample = ", ".join(sorted({str(v) for v in (overrides or {}).values()})[:6])
            raise RuntimeError(
                "The visual track could not be rendered.\n"
                f"Visual override sample: {sample or '(none)'}\n"
                "If this still fails, the next thing to inspect is helpers.viz_offline runtime behavior after backend root/path correction."
            )

    def _overlay_visuals(self, src_video: str, visuals_video: str, out_video: str, alpha: float) -> None:
        filter_complex = (
            f"[1:v]format=rgba,colorchannelmixer=aa={alpha:.4f}[viz];"
            f"[0:v][viz]overlay=(W-w)/2:(H-h)/2:shortest=1[vout]"
        )
        cmd = [
            self._ffmpeg,
            "-y",
            "-i",
            src_video,
            "-i",
            visuals_video,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            "-pix_fmt",
            "yuv420p",
            out_video,
        ]
        code, out = _run(cmd)
        if code != 0 or not os.path.exists(out_video):
            raise RuntimeError("Overlay step failed.\n" + out)

    # ----------------------------- render ----------------------------------
    def _render(self) -> None:
        err = self._validate()
        if err:
            QMessageBox.warning(self, "Beat-synced visuals", err, QMessageBox.Ok)
            return

        video = self.edit_video.text().strip()
        out_dir = self.edit_output.text().strip() or self._default_output_dir()
        _ensure_dir(out_dir)

        self.setEnabled(False)
        self._set_status("Starting render...")
        self._append_log("Starting render...")
        self._append_log(f"Source video: {video}")
        self._append_log(f"Output folder: {out_dir}")
        self._set_busy_ui(True)

        self._render_thread = QThread(self)
        self._render_worker = _RenderWorker(self, video, out_dir)
        self._render_worker.moveToThread(self._render_thread)
        self._render_thread.started.connect(self._render_worker.run)
        self._render_worker.progress.connect(self._on_worker_progress)
        self._render_worker.finished.connect(self._on_render_finished)
        self._render_worker.finished.connect(self._render_thread.quit)
        self._render_worker.finished.connect(self._render_worker.deleteLater)
        self._render_thread.finished.connect(self._render_thread.deleteLater)
        self._render_thread.start()

    def _render_sync(self, video: str, out_dir: str, progress_cb=None) -> str:
                if progress_cb:
            progress_cb("Reading source video...")
        duration = _ffprobe_duration(self._ffprobe, video)
                if progress_cb:
            progress_cb("Reading source resolution...")
        res = _ffprobe_resolution(self._ffprobe, video)
                if progress_cb:
            progress_cb("Reading source FPS...")
        fps = _ffprobe_fps(self._ffprobe, video)
        if duration is None or duration <= 0:
            raise RuntimeError("Could not read the source video duration.")
        if res is None:
            raise RuntimeError("Could not read the source video resolution.")

                if progress_cb:
            progress_cb("Analyzing audio and building visual plan...")
        plan, overrides = self._build_segment_plan(duration, video)

        tmpdir = tempfile.mkdtemp(prefix="fv_bsv_")
        try:
            visuals_path = os.path.join(tmpdir, "visuals_track.mp4")
            section_map = [(float(st), float(en), str(key)) for st, en, key in plan]
                        if progress_cb:
                progress_cb("Rendering visual track...")
            self._render_visual_track(
                audio_path=video,
                out_video=visuals_path,
                resolution=res,
                fps=int(round(fps)),
                section_map=section_map,
                overrides=overrides,
            )

            alpha = 1.0
            if self.check_transparency.isChecked():
                alpha = max(0.0, min(1.0, float(self.spin_alpha.value()) / 100.0))

            base_name = os.path.splitext(os.path.basename(video))[0]
            out_path = os.path.join(out_dir, f"{base_name}_beatsync_visuals.mp4")
                        if progress_cb:
                progress_cb("Overlaying visuals on the source video...")
            self._overlay_visuals(video, visuals_path, out_path, alpha)
            return out_path
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _on_worker_progress(self, message: str) -> None:
        self._set_status(str(message))
        self._append_log(str(message))

    def _on_render_finished(self, ok: bool, message: str) -> None:
        self.setEnabled(True)
        self._set_busy_ui(False)
        self._save_settings()
        try:
            self._render_thread = None
            self._render_worker = None
        except Exception:
            pass

        if ok:
            self._set_status(f"Done. Saved to:\n{message}")
            self._append_log("Done.")
            self._append_log(f"Saved to: {message}")
            QMessageBox.information(self, "Beat-synced visuals", f"Finished.\n\nSaved to:\n{message}", QMessageBox.Ok)
        else:
            self._set_status(message)
            self._append_log("Failed.")
            self._append_log(str(message))
            QMessageBox.warning(self, "Beat-synced visuals", message, QMessageBox.Ok)


# ----------------------------- install hook -------------------------------


def install_beatsyncvisuals_tool(parent: QWidget, container: QWidget) -> BeatSyncVisualsTool:
    tool = BeatSyncVisualsTool(parent)
    lay = container.layout()
    if lay is None:
        lay = QVBoxLayout(container)
    lay.addWidget(tool)
    return tool



def _run_standalone() -> int:
    _ensure_framevision_import_paths()
    app = QApplication.instance() or QApplication(sys.argv)
    w = BeatSyncVisualsTool()
    w.resize(980, 760)
    w.setWindowTitle('Beat-synced visuals')
    w.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(_run_standalone())
