"""
MusicEdit tool - simple audio crop & convert panel for PySide6.

Fixes in this build:
- Robust player switching: safely tear down and recreate QMediaPlayer/QAudioOutput when loading a new track (prevents crashes).
- Fully reset BPM UI on new load; BPM is no longer saved to settings.
- Waveform preview with pyqtgraph (zoom, horizontal pan only), region selection.
- "Save As" with conversion WAV<->MP3, metadata tagging, settings memory (excluding BPM).
"""

import json
import os
import shutil
import subprocess
import tempfile
import wave
import struct
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QSlider,
    QFormLayout,
    QLineEdit,
    QGroupBox,
    QComboBox,
    QCheckBox,
    QMessageBox,
    QMenu,
)

from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

# Optional waveform visualization (pyqtgraph)
_pg = None
_pg_import_err = None
try:
    import pyqtgraph as _pg  # type: ignore
except Exception as e:  # pragma: no cover - best-effort import
    _pg = None
    _pg_import_err = e


def _ms_to_time_str(ms: int) -> str:
    if ms <= 0:
        return "00:00.000"
    seconds_total = ms // 1000
    msec = ms % 1000
    minutes = seconds_total // 60
    seconds = seconds_total % 60
    return f"{minutes:02d}:{seconds:02d}.{msec:03d}"


def _find_ffmpeg() -> str:
    """
    Try to locate ffmpeg binary.

    Preference order:
    - <root>/presets/bin/ffmpeg(.exe)
    - PATH
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    candidates = [
        os.path.join(root, "presets", "bin", "ffmpeg.exe"),
        os.path.join(root, "presets", "bin", "ffmpeg"),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return c

    # Fallback to PATH
    ffmpeg_in_path = shutil.which("ffmpeg")
    if ffmpeg_in_path:
        return ffmpeg_in_path

    raise RuntimeError("ffmpeg executable not found. Expected in presets/bin or in PATH.")



def _get_musicedit_temp_dir() -> str:
    """Return the temp directory for MusicEdit intermediate files.

    This always resolves to <root>/temp/musicedit and ensures
    the directory exists so that cut/copy/paste never write next
    to the source audio file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, os.pardir))
    temp_dir = os.path.join(root, "temp", "musicedit")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

@dataclass
class MusicEditSettings:
    last_input_dir: str = ""
    last_output_dir: str = ""
    output_format: str = "mp3"  # "mp3" or "wav"
    mp3_bitrate_kbps: int = 320
    wav_samplerate_hz: int = 48000
    meta_artist: str = ""
    meta_album: str = ""
    meta_title: str = ""
    meta_year: str = ""
    # kept for backwards compat with old JSON; UI won't use or update it
    meta_bpm: str = ""
    meta_extra: str = ""


class WaveformWidget(QWidget):
    """
    Waveform preview using pyqtgraph, with zoom, horizontal pan and region selection.

    X-axis: milliseconds from start.
    Y-axis: normalized amplitude.
    """

    selectionChanged = Signal(int, int)  # start_ms, end_ms
    playSelectionRequested = Signal()
    cutRequested = Signal()
    copyRequested = Signal()
    pasteRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._plot = None
        self._curve = None
        self._region = None
        self._duration_ms = 0
        self._has_waveform = False
        self._selection_clipboard: Optional[Tuple[float, float]] = None

        if _pg is None:
            # Fallback: text message instead of crashing if pyqtgraph is missing
            msg = "Waveform preview requires pyqtgraph.\n"
            if _pg_import_err is not None:
                msg += f"(Import failed: {_pg_import_err})"
            label = QLabel(msg, self)
            label.setAlignment(Qt.AlignCenter)
            try:
                label.setWordWrap(True)
            except Exception:
                pass
            self._layout.addWidget(label)
        else:
            self._plot = _pg.PlotWidget(self)
            try:
                self._plot.setBackground(None)  # inherit theme (light/dark)
                self._plot.hideButtons()
                self._plot.showGrid(x=False, y=False)
                self._plot.getPlotItem().hideAxis("left")
                self._plot.getPlotItem().hideAxis("bottom")
                vb = self._plot.getViewBox()
                # Horizontal pan/zoom only
                vb.setMouseEnabled(x=True, y=False)
                vb.setMenuEnabled(False)
            except Exception:
                pass

            # Slightly brighter pen so it's visible on both light/dark themes
            pen = None
            if _pg is not None:
                try:
                    pen = _pg.mkPen(240, 240, 230)
                except Exception:
                    pen = None

            self._curve = self._plot.plot([], [], pen=pen)
            self._layout.addWidget(self._plot)
            try:
                self._plot.setContextMenuPolicy(Qt.CustomContextMenu)
                self._plot.customContextMenuRequested.connect(self._on_context_menu)
            except Exception:
                pass


            # Region item for crop selection
            self._region = _pg.LinearRegionItem(values=(0, 0))
            try:
                self._region.setZValue(10)
                self._region.setBrush(_pg.mkBrush(180, 200, 80, 80))
            except Exception:
                pass
            self._region.hide()
            self._region.sigRegionChanged.connect(self._on_region_changed)
            self._plot.addItem(self._region)

    def clear_waveform(self):
        self._duration_ms = 0
        self._has_waveform = False
        if self._curve is not None:
            self._curve.setData([], [])
        if self._region is not None:
            self._region.hide()
            try:
                self._region.setRegion((0, 0))
            except Exception:
                pass

    def set_duration_ms(self, duration_ms: int):
        """Optional: override duration if needed."""
        self._duration_ms = max(int(duration_ms), 0)
        # Update region bounds if we already have data
        if self._has_waveform and self._region is not None:
            try:
                self._region.setBounds((0, float(self._duration_ms)))
            except Exception:
                pass

    def set_audio(self, path: str, duration_ms: int, ffmpeg_path: Optional[str] = None):
        """
        Generate waveform data from an audio file and show it.

        The X-axis is in milliseconds, to match player position/crop logic.
        """
        self.clear_waveform()
        self._duration_ms = max(int(duration_ms), 0)

        if _pg is None:
            return
        if not path or not os.path.isfile(path):
            return

        # Resolve ffmpeg path if not provided
        ff = ffmpeg_path
        if not ff:
            try:
                ff = _find_ffmpeg()
            except Exception:
                ff = None
        if not ff:
            return

        tmp_wav = None
        try:
            fd, tmp_wav = tempfile.mkstemp(suffix="_musicedit_wave.wav")
            os.close(fd)

            # Down-mix to mono, low sample rate to keep waveform light-weight
            cmd = [
                ff,
                "-y",
                "-i",
                path,
                "-ac",
                "1",
                "-ar",
                "1000",
                "-vn",
                "-f",
                "wav",
                tmp_wav,
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                # Best-effort only; don't break the rest of the tool
                print("MusicEdit: failed to generate waveform WAV:", proc.stderr)
                return

            with wave.open(tmp_wav, "rb") as wf:
                nframes = wf.getnframes()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                raw = wf.readframes(nframes)

            if sampwidth != 2 or nframes <= 0 or framerate <= 0:
                return

            fmt = "<%dh" % (len(raw) // 2)
            data = struct.unpack(fmt, raw)

            # Downsample to at most ~4000 points for performance
            total_samples = len(data)
            if total_samples <= 0:
                return
            step = max(1, total_samples // 4000)

            xs = []
            ys = []
            for i in range(0, total_samples, step):
                # X in milliseconds
                t_ms = (i / float(framerate)) * 1000.0
                xs.append(t_ms)
                ys.append(data[i] / 32768.0)

            if not xs or not ys:
                return

            # Duration in ms based on the resampled audio, if we don't have one yet
            dur_ms_audio = int((nframes / float(framerate)) * 1000.0)
            if self._duration_ms <= 0:
                self._duration_ms = dur_ms_audio

            self._has_waveform = True

            if self._curve is not None:
                self._curve.setData(xs, ys)

            # Fit view: no vertical panning, but enough vertical range to make peaks clear
            x_max = max(xs[-1], float(self._duration_ms))
            if self._plot is not None:
                try:
                    vb = self._plot.getViewBox()
                    ymin = min(ys)
                    ymax = max(ys)
                    if ymin == ymax:
                        ymin -= 0.1
                        ymax += 0.1
                    # Keep vertical range centered and a bit expanded so it fills more height
                    pad = max(abs(ymin), abs(ymax)) * 1.1
                    if pad <= 0:
                        pad = 1.0
                    vb.setLimits(xMin=0, xMax=x_max)
                    vb.setRange(
                        xRange=(0, x_max),
                        yRange=(-pad, pad),
                        padding=0.02,
                    )
                except Exception:
                    pass

            # Configure selection region to cover full track by default
            if self._region is not None:
                try:
                    self._region.setBounds((0, x_max))
                    # Use the timeline duration for region, so cropping matches UI time
                    self._region.setRegion((0, float(self._duration_ms or x_max)))
                    self._region.show()
                except Exception:
                    pass

            # Emit "full track" selection
            self.selectionChanged.emit(0, self._duration_ms)
        except Exception as e:  # pragma: no cover - defensive
            print("MusicEdit: exception while building waveform:", e)
        finally:
            if tmp_wav and os.path.isfile(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def _on_region_changed(self):
        if not self._has_waveform or self._region is None:
            return
        try:
            start, end = self._region.getRegion()
        except Exception:
            return

        if self._duration_ms > 0:
            start = max(0.0, min(start, float(self._duration_ms)))
            end = max(0.0, min(end, float(self._duration_ms)))
        if end < start:
            start, end = end, start

        self.selectionChanged.emit(int(start), int(end))

    def get_selection_ms(self) -> Tuple[int, int]:
        """Return (start_ms, end_ms). If no region or duration, use full track."""
        if not self._has_waveform or self._region is None or self._duration_ms <= 0:
            return 0, self._duration_ms
        try:
            start, end = self._region.getRegion()
        except Exception:
            return 0, self._duration_ms
        if end < start:
            start, end = end, start
        start_ms = max(0, min(int(start), self._duration_ms))
        end_ms = max(0, min(int(end), self._duration_ms))
        return start_ms, end_ms

    def _has_valid_selection(self) -> bool:
        if not self._has_waveform or self._region is None or self._duration_ms <= 0:
            return False
        try:
            start, end = self._region.getRegion()
        except Exception:
            return False
        return end > start

    def _on_context_menu(self, pos):
        """Show a context menu with selection tools on right-click."""
        if _pg is None or not self._has_waveform or self._plot is None:
            return
        try:
            global_pos = self._plot.mapToGlobal(pos)
        except Exception:
            return
    
        has_sel = self._has_valid_selection()
        menu = QMenu(self)
    
        act_play = menu.addAction("Play selection")
        act_play.setEnabled(has_sel)
        menu.addSeparator()
    
        act_cut = menu.addAction("Cut selection")
        act_cut.setEnabled(has_sel)
        act_copy = menu.addAction("Copy selection")
        act_copy.setEnabled(has_sel)
        act_paste = menu.addAction("Paste selection")
        menu.addSeparator()
    
        act_clear = menu.addAction("Clear selection")
        act_zoom_sel = menu.addAction("Zoom to selection")
        act_zoom_sel.setEnabled(has_sel)
        act_zoom_full = menu.addAction("Show full waveform")
    
        action = menu.exec(global_pos)
        if action is None:
            return
        if action is act_play:
            if has_sel:
                self.playSelectionRequested.emit()
        elif action is act_cut:
            if has_sel:
                self.cutRequested.emit()
        elif action is act_copy:
            if has_sel:
                self.copyRequested.emit()
        elif action is act_paste:
            self.pasteRequested.emit()
        elif action is act_clear:
            self.clear_selection()
        elif action is act_zoom_sel:
            self.zoom_to_selection()
        elif action is act_zoom_full:
            self.zoom_full()


    def _do_cut_selection(self):
        """Cut the current selection into an internal clipboard and clear it."""
        if not self._has_valid_selection():
            return
        if self._region is None:
            return
        try:
            start, end = self._region.getRegion()
        except Exception:
            return
        if end < start:
            start, end = end, start
        self._selection_clipboard = (start, end)
        # Clearing behaves like "full track" selection.
        self.select_full()

    def _do_copy_selection(self):
        """Copy the current selection range into an internal clipboard."""
        if not self._has_valid_selection():
            return
        if self._region is None:
            return
        try:
            start, end = self._region.getRegion()
        except Exception:
            return
        if end < start:
            start, end = end, start
        self._selection_clipboard = (start, end)

    def _do_paste_selection(self):
        """Paste the previously copied selection range as the active selection."""
        if not self._selection_clipboard or self._region is None:
            return
        start, end = self._selection_clipboard
        if self._duration_ms > 0:
            max_end = float(self._duration_ms)
            start = max(0.0, min(start, max_end))
            end = max(0.0, min(end, max_end))
        if end < start:
            start, end = end, start
        if start == end:
            return
        try:
            self._region.setRegion((start, end))
            self._region.show()
        except Exception:
            return
        self.selectionChanged.emit(int(start), int(end))

    def select_full(self):
        """Select the full track duration."""
        if not self._has_waveform or self._region is None or self._duration_ms <= 0:
            return
        try:
            self._region.setRegion((0, float(self._duration_ms)))
            self._region.show()
        except Exception:
            return
        self.selectionChanged.emit(0, self._duration_ms)

    def clear_selection(self):
        """Clear selection back to full track coverage."""
        self.select_full()

    def zoom_to_selection(self):
        """Zoom the view to the current selection range."""
        if not self._has_valid_selection() or self._plot is None:
            return
        if self._region is None:
            return
        try:
            start, end = self._region.getRegion()
        except Exception:
            return
        if end < start:
            start, end = end, start
        if start == end:
            return
        try:
            vb = self._plot.getViewBox()
        except Exception:
            return
        x_min = max(0.0, start)
        x_max = end
        if self._duration_ms > 0:
            x_max = min(x_max, float(self._duration_ms))
        vb.setRange(xRange=(x_min, x_max), padding=0.02)

    def zoom_full(self):
        """Zoom out to show the full waveform in view."""
        if not self._has_waveform or self._plot is None:
            return
        try:
            vb = self._plot.getViewBox()
        except Exception:
            return
        x_max = float(self._duration_ms)
        if x_max <= 0:
            if self._curve is not None:
                xs, _ = self._curve.getData()
                if xs is not None and len(xs) > 0:
                    x_max = float(xs[-1])
        if x_max <= 0:
            return
        vb.setRange(xRange=(0.0, x_max), padding=0.02)



class MusicEditWidget(QWidget):
    """
    Simple audio crop & convert widget.

    Integrate into your app by instantiating MusicEditWidget(parent)
    and adding it into a layout or tab.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._ffmpeg_path = None
        try:
            self._ffmpeg_path = _find_ffmpeg()
        except Exception:
            # We'll warn later when needed
            self._ffmpeg_path = None

        # paths & settings
        self._settings_path = self._compute_settings_path()
        self._settings = self._load_settings()

        self._current_file: Optional[str] = None
        self._duration_ms: int = 0
        self._waveform_needs_update: bool = False
        self._is_loading: bool = False  # guard against re-entrancy
        self._play_selection_end_ms: int = 0
        self._edit_clipboard_path: Optional[str] = None
        self._open_file_dir: str = ""

        # Media player for preview
        self._audio_output: Optional[QAudioOutput] = None
        self._player: Optional[QMediaPlayer] = None
        self._setup_player()

        self._build_ui()
        self._apply_settings_to_ui()

    # ----- Player lifecycle -----
    def _setup_player(self):
        """Create a fresh player + audio output and wire signals."""
        self._audio_output = QAudioOutput(self)
        self._player = QMediaPlayer(self)
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(1.0)

        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)

    def _teardown_player(self):
        """Stop, disconnect, and delete the player stack safely."""
        p = self._player
        a = self._audio_output
        if p is not None:
            try:
                p.stop()
                p.setSource(QUrl())
            except Exception:
                pass
            try:
                p.durationChanged.disconnect(self._on_duration_changed)
                p.positionChanged.disconnect(self._on_position_changed)
                p.playbackStateChanged.disconnect(self._on_state_changed)
            except Exception:
                pass
            try:
                p.deleteLater()
            except Exception:
                pass
        if a is not None:
            try:
                a.deleteLater()
            except Exception:
                pass
        self._player = None
        self._play_selection_end_ms = 0
        self._audio_output = None

    # ----- Settings paths -----
    def _compute_settings_path(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(here, os.pardir))
        setsave_dir = os.path.join(root, "presets", "setsave")
        os.makedirs(setsave_dir, exist_ok=True)
        # The user requested this exact filename (note the extension).
        return os.path.join(setsave_dir, "musicedit.json")

    def _load_settings(self) -> MusicEditSettings:
        try:
            if os.path.isfile(self._settings_path):
                with open(self._settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return MusicEditSettings(**data)
        except Exception:
            pass
        return MusicEditSettings()

    def _save_settings(self):
        try:
            data = asdict(self._settings)
            # Do NOT persist meta_bpm (BPM is per-track only)
            data.pop("meta_bpm", None)
            with open(self._settings_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            # Non-fatal
            pass

    # ----- UI -----
    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # File row
        file_row = QHBoxLayout()
        self.btn_open = QPushButton("Open audio…", self)
        self.btn_open.clicked.connect(self._on_open_clicked)
        self.lbl_file = QLabel("No file loaded", self)
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)

        file_row.addWidget(self.btn_open)
        file_row.addWidget(self.lbl_file, 1)
        main_layout.addLayout(file_row)

        # Waveform + selection info
        self.waveform_view = WaveformWidget(self)
        # Give waveform extra vertical space so it's easier to read
        try:
            self.waveform_view.setMinimumHeight(160)
        except Exception:
            pass
        self.waveform_view.selectionChanged.connect(self._on_selection_changed)
        self.waveform_view.playSelectionRequested.connect(self._on_play_selection_clicked)
        self.waveform_view.cutRequested.connect(self._on_cut_selection)
        self.waveform_view.copyRequested.connect(self._on_copy_selection)
        self.waveform_view.pasteRequested.connect(self._on_paste_selection)
        main_layout.addWidget(self.waveform_view, 1)

        self.lbl_selection = QLabel("Selection: full track", self)
        main_layout.addWidget(self.lbl_selection)

        # Playback controls
        playback_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("Play", self)
        self.btn_stop = QPushButton("Stop", self)
        self.btn_play_selection = QPushButton("Play selection", self)

        self.btn_play_pause.clicked.connect(self._on_play_pause_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.btn_play_selection.clicked.connect(self._on_play_selection_clicked)

        self.position_slider = QSlider(Qt.Horizontal, self)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)

        self.lbl_time = QLabel("00:00.000 / 00:00.000", self)

        playback_row.addWidget(self.btn_play_pause)
        playback_row.addWidget(self.btn_stop)
        playback_row.addWidget(self.btn_play_selection)
        playback_row.addWidget(self.position_slider, 1)
        playback_row.addWidget(self.lbl_time)
        main_layout.addLayout(playback_row)

        # Export options
        export_group = QGroupBox("Export / Save As", self)
        export_layout = QFormLayout(export_group)

        self.combo_format = QComboBox(self)
        self.combo_format.addItem("MP3", userData="mp3")
        self.combo_format.addItem("WAV", userData="wav")
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)

        self.combo_mp3_bitrate = QComboBox(self)
        for br in [24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]:
            self.combo_mp3_bitrate.addItem(f"{br} kbps", userData=br)

        self.combo_wav_samplerate = QComboBox(self)
        for sr in [8000, 16000, 22050, 32000, 44100, 48000]:
            self.combo_wav_samplerate.addItem(f"{sr} Hz", userData=sr)

        self.chk_use_selection = QCheckBox("Export selection only", self)
        self.chk_use_selection.setChecked(True)

        export_layout.addRow("Format:", self.combo_format)
        export_layout.addRow("MP3 bitrate:", self.combo_mp3_bitrate)
        export_layout.addRow("WAV sample rate:", self.combo_wav_samplerate)
        export_layout.addRow("", self.chk_use_selection)

        self.btn_save_as = QPushButton("Save As…", self)
        self.btn_save_as.clicked.connect(self._on_save_as_clicked)
        export_layout.addRow("", self.btn_save_as)

        main_layout.addWidget(export_group)

        # Metadata group
        meta_group = QGroupBox("MP3 Metadata (optional)", self)
        meta_layout = QFormLayout(meta_group)

        self.edit_artist = QLineEdit(self)
        self.edit_album = QLineEdit(self)
        self.edit_title = QLineEdit(self)
        self.edit_year = QLineEdit(self)
        self.edit_bpm = QLineEdit(self)
        self.edit_extra = QLineEdit(self)

        # BPM detection UI
        self.lbl_bpm_detect = QLabel("Detected BPM: –", self)
        self.btn_detect_bpm = QPushButton("Detect BPM", self)
        self.btn_detect_bpm.clicked.connect(self._on_detect_bpm_clicked)

        bpm_row_layout = QHBoxLayout()
        bpm_row_layout.setContentsMargins(0, 0, 0, 0)
        bpm_row_layout.addWidget(self.edit_bpm)
        bpm_row_layout.addWidget(self.btn_detect_bpm)
        bpm_row_widget = QWidget(self)
        bpm_row_widget.setLayout(bpm_row_layout)

        meta_layout.addRow("Artist:", self.edit_artist)
        meta_layout.addRow("Album:", self.edit_album)
        meta_layout.addRow("Track title:", self.edit_title)
        meta_layout.addRow("Year:", self.edit_year)
        meta_layout.addRow("BPM:", bpm_row_widget)
        meta_layout.addRow("", self.lbl_bpm_detect)
        meta_layout.addRow("Extra info:", self.edit_extra)

        main_layout.addWidget(meta_group)

        # Spacer at bottom
        main_layout.addStretch(0)

        self._on_format_changed()  # show/hide bitrate/sample rate appropriately

    def _apply_settings_to_ui(self):
        # Format
        fmt = self._settings.output_format.lower()
        idx = 0
        for i in range(self.combo_format.count()):
            if self.combo_format.itemData(i) == fmt:
                idx = i
                break
        self.combo_format.setCurrentIndex(idx)

        # MP3 bitrate
        bitrate = self._settings.mp3_bitrate_kbps
        idx = 0
        for i in range(self.combo_mp3_bitrate.count()):
            if self.combo_mp3_bitrate.itemData(i) == bitrate:
                idx = i
                break
        self.combo_mp3_bitrate.setCurrentIndex(idx)

        # WAV samplerate
        sr = self._settings.wav_samplerate_hz
        idx = 0
        for i in range(self.combo_wav_samplerate.count()):
            if self.combo_wav_samplerate.itemData(i) == sr:
                idx = i
                break
        self.combo_wav_samplerate.setCurrentIndex(idx)

        # Metadata (BPM is not restored from settings)
        self.edit_artist.setText(self._settings.meta_artist)
        self.edit_album.setText(self._settings.meta_album)
        self.edit_title.setText(self._settings.meta_title)
        self.edit_year.setText(self._settings.meta_year)
        self.edit_bpm.setText("")  # start empty each session
        self.edit_extra.setText(self._settings.meta_extra)

        # Detected BPM label always resets
        self.lbl_bpm_detect.setText("Detected BPM: –")

    # ----- Slots / handlers -----
    @Slot()
    def _on_open_clicked(self):
        if self._is_loading:
            return

        # Stop playback before opening a new file to avoid crashes while timers are running
        try:
            self._on_stop_clicked()
        except Exception:
            # Fallback safety: stop the player directly if available
            if getattr(self, "_player", None) is not None:
                self._player.stop()
                self._player.setPosition(0)
            self._play_selection_end_ms = 0

        start_dir = self._settings.last_input_dir or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open audio file",
            start_dir,
            "Audio files (*.wav *.mp3);;All files (*)",
        )
        if not path:
            return
        self._settings.last_input_dir = os.path.dirname(path)
        self._save_settings()
        self.load_file(path)

    def load_file(self, path: str):
        """Load an audio file into the preview/player."""
        if self._is_loading:
            return
        self._is_loading = True
        try:
            if not os.path.isfile(path):
                QMessageBox.warning(self, "File not found", f"File not found:\n{path}")
                return

            # Hard reset player stack to avoid backend crashes
            self._teardown_player()
            self._setup_player()

            self._current_file = path
            self.lbl_file.setText(path)
            self._duration_ms = 0
            self._waveform_needs_update = True
            self._play_selection_end_ms = 0

            # Reset UI
            self.waveform_view.clear_waveform()
            self.position_slider.setRange(0, 0)
            self.lbl_time.setText("00:00.000 / 00:00.000")
            self.lbl_selection.setText("Selection: full track")
            self.lbl_bpm_detect.setText("Detected BPM: –")
            self.edit_bpm.clear()

            # Load into media player
            if self._player is not None:
                self._player.setSource(QUrl.fromLocalFile(path))
        finally:
            self._is_loading = False

    @Slot()
    def _on_play_pause_clicked(self):
        if not self._current_file or self._player is None:
            return
        state = self._player.playbackState()
        if state == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    @Slot()
    def _on_play_selection_clicked(self):
        """Start playback from the beginning of the current selection."""
        if not self._current_file or self._player is None:
            return
        start_ms, end_ms = self.waveform_view.get_selection_ms()
        if start_ms < 0:
            start_ms = 0
        if end_ms <= start_ms:
            # No or invalid selection: fall back to start of track.
            start_ms = 0
            end_ms = 0
        try:
            self._player.setPosition(start_ms)
        except Exception:
            pass
        # Remember where to stop if we have a real selection.
        self._play_selection_end_ms = end_ms if end_ms > start_ms else 0
        self._player.play()


    @Slot()
    def _on_stop_clicked(self):
        if self._player is None:
            return
        self._play_selection_end_ms = 0
        self._player.stop()
        self._player.setPosition(0)

    @Slot(int)
    def _on_slider_moved(self, value: int):
        if self._duration_ms > 0 and self._player is not None:
            self._player.setPosition(value)

    @Slot(int)
    def _on_duration_changed(self, duration: int):
        self._duration_ms = max(int(duration), 0)
        self.position_slider.setRange(0, self._duration_ms)
        if self._player is not None:
            self._update_time_label(self._player.position())

        # Update waveform once duration is known
        if self._waveform_needs_update and self._current_file:
            self._generate_waveform_preview()
            self._waveform_needs_update = False

    @Slot(int)
    def _on_position_changed(self, position: int):
        if self._duration_ms <= 0:
            return
        if self._play_selection_end_ms > 0 and position >= self._play_selection_end_ms:
            if self._player is not None and self._player.playbackState() == QMediaPlayer.PlayingState:
                self._player.pause()
            self._play_selection_end_ms = 0
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        self._update_time_label(position)

    @Slot()
    def _on_state_changed(self, *args):
        if self._player is None:
            return
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self.btn_play_pause.setText("Pause")
        else:
            self.btn_play_pause.setText("Play")

    def _update_time_label(self, pos_ms: int):
        cur = _ms_to_time_str(pos_ms)
        total = _ms_to_time_str(self._duration_ms)
        self.lbl_time.setText(f"{cur} / {total}")

    @Slot(int, int)
    def _on_selection_changed(self, start_ms: int, end_ms: int):
        if start_ms == 0 and (end_ms == 0 or end_ms == self._duration_ms):
            self.lbl_selection.setText("Selection: full track")
        else:
            self.lbl_selection.setText(
                f"Selection: {_ms_to_time_str(start_ms)} → {_ms_to_time_str(end_ms)}"
            )

    def _get_ffmpeg_for_edit(self) -> Optional[str]:
        """Helper to get ffmpeg and show an error message if missing."""
        try:
            return _find_ffmpeg()
        except Exception:
            QMessageBox.critical(
                self,
                "ffmpeg not found",
                "ffmpeg executable not found. Expected in presets/bin or in PATH.",
            )
            return None
    

    def _make_edit_temp_path(self) -> str:
        """Create a temp output path for intermediate edited audio.

        All MusicEdit intermediate files (cut/copy/paste) are stored in
        the app-level temp directory: <root>/temp/musicedit/.
        """
        base_name = "edit"
        if self._current_file:
            base_name = os.path.splitext(os.path.basename(self._current_file))[0] or "edit"

        ext = ".wav"
        if self._current_file:
            ext_cur = os.path.splitext(self._current_file)[1]
            if ext_cur:
                ext = ext_cur

        base_dir = _get_musicedit_temp_dir()
        try:
            fd, tmp = tempfile.mkstemp(prefix=f"{base_name}_edit_", suffix=ext, dir=base_dir)
        except Exception:
            # Fallback to default temp dir if something goes wrong
            fd, tmp = tempfile.mkstemp(prefix=f"{base_name}_edit_", suffix=ext)
        os.close(fd)
        return tmp

    def _on_copy_selection(self):
        """Copy the current selection into an internal audio clipboard file."""
        if not self._current_file or self._duration_ms <= 0:
            return
        ff = self._get_ffmpeg_for_edit()
        if not ff:
            return
        start_ms, end_ms = self.waveform_view.get_selection_ms()
        if end_ms <= start_ms:
            return
        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0
        out_path = self._make_edit_temp_path()
        ext = os.path.splitext(self._current_file)[1].lower()
        cmd = [ff, "-y", "-i", self._current_file, "-ss", f"{start_sec:.3f}", "-t", f"{duration_sec:.3f}"]
        if ext == ".mp3":
            cmd += ["-vn", "-c:a", "libmp3lame", "-b:a", "320k"]
        else:
            cmd += ["-vn", "-c:a", "pcm_s16le"]
        cmd.append(out_path)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "Copy selection failed", f"Failed to run ffmpeg:\n{e}")
            return
        if proc.returncode != 0:
            QMessageBox.critical(
                self,
                "Copy selection failed",
                "ffmpeg reported an error while copying the selection.\n\n"
                f"Command:\n{' '.join(cmd)}\n\nError:\n{proc.stderr}",
            )
            return
        # Replace previous clipboard file if any
        if self._edit_clipboard_path and os.path.isfile(self._edit_clipboard_path):
            try:
                os.remove(self._edit_clipboard_path)
            except Exception:
                pass
        self._edit_clipboard_path = out_path
    
    def _on_cut_selection(self):
        """Cut selection from the current track and copy it to clipboard."""
        # First copy selection into clipboard
        self._on_copy_selection()
        if not self._current_file or self._duration_ms <= 0:
            return
        ff = self._get_ffmpeg_for_edit()
        if not ff:
            return
        start_ms, end_ms = self.waveform_view.get_selection_ms()
        if end_ms <= start_ms:
            return
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        out_path = self._make_edit_temp_path()
        ext = os.path.splitext(self._current_file)[1].lower()
        filter_complex = (
            f"[0:a]atrim=start=0:end={start_sec:.6f},asetpts=N/SR/TB[a0];"
            f"[0:a]atrim=start={end_sec:.6f},asetpts=N/SR/TB[a1];"
            f"[a0][a1]concat=n=2:v=0:a=1[aout]"
        )
        cmd = [
            ff,
            "-y",
            "-i",
            self._current_file,
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
        ]
        if ext == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "320k"]
        else:
            cmd += ["-c:a", "pcm_s16le"]
        cmd.append(out_path)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "Cut selection failed", f"Failed to run ffmpeg:\n{e}")
            return
        if proc.returncode != 0:
            QMessageBox.critical(
                self,
                "Cut selection failed",
                "ffmpeg reported an error while cutting the selection.\n\n"
                f"Command:\n{' '.join(cmd)}\n\nError:\n{proc.stderr}",
            )
            return
        # Load the edited file as the new current file
        self.load_file(out_path)
    
    def _on_paste_selection(self):
        """Paste the clipboard audio into the current track by inserting it (no audio is removed)."""
        if not self._current_file or not self._edit_clipboard_path or self._duration_ms <= 0:
            return
        ff = self._get_ffmpeg_for_edit()
        if not ff:
            return
        start_ms, end_ms = self.waveform_view.get_selection_ms()
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
        # If selection is empty, insert at end of track
        if end_ms <= start_ms:
            start_ms = self._duration_ms
            end_ms = self._duration_ms

        # We insert at the end of the selection
        insertion_ms = end_ms
        insertion_sec = insertion_ms / 1000.0

        out_path = self._make_edit_temp_path()
        ext = os.path.splitext(self._current_file)[1].lower()

        # a0 = original from 0 -> insertion point
        # a2 = original from insertion point -> end
        # a1 = clipboard audio
        filter_complex = (
            f"[0:a]atrim=start=0:end={insertion_sec:.6f},asetpts=N/SR/TB[a0];"
            f"[0:a]atrim=start={insertion_sec:.6f},asetpts=N/SR/TB[a2];"
            f"[1:a]asetpts=N/SR/TB[a1];"
            f"[a0][a1][a2]concat=n=3:v=0:a=1[aout]"
        )

        cmd = [
            ff,
            "-y",
            "-i",
            self._current_file,
            "-i",
            self._edit_clipboard_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[aout]",
        ]
        if ext == ".mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", "320k"]
        else:
            cmd += ["-c:a", "pcm_s16le"]
        cmd.append(out_path)
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "Paste selection failed", "Failed to run ffmpeg:\n{0}".format(e))
            return
        if proc.returncode != 0:
            QMessageBox.critical(
                self,
                "Paste selection failed",
                "ffmpeg reported an error while pasting the selection.\n\nCommand:\n{0}\n\nError:\n{1}".format(' '.join(cmd), proc.stderr),
            )
            return
        self.load_file(out_path)


    @Slot()
    def _on_format_changed(self):
        fmt = self.combo_format.currentData()
        if fmt == "mp3":
            self.combo_mp3_bitrate.setEnabled(True)
            self.combo_wav_samplerate.setEnabled(False)
        else:
            self.combo_mp3_bitrate.setEnabled(False)
            self.combo_wav_samplerate.setEnabled(True)

    @Slot()
    def _on_save_as_clicked(self):
        if not self._current_file:
            QMessageBox.information(self, "No file", "Please open an audio file first.")
            return

        ffmpeg_ok = True
        try:
            _ = _find_ffmpeg()
        except Exception:
            ffmpeg_ok = False

        if not ffmpeg_ok:
            QMessageBox.critical(
                self,
                "ffmpeg not found",
                "ffmpeg executable not found. Expected in presets/bin or in PATH.",
            )
            return

        fmt = self.combo_format.currentData()
        base_dir = (
            self._settings.last_output_dir
            or self._settings.last_input_dir
            or os.path.dirname(self._current_file)
        )
        base_name = os.path.splitext(os.path.basename(self._current_file))[0]
        if fmt == "mp3":
            default_name = base_name + "_edit.mp3"
            filter_str = "MP3 files (*.mp3);;All files (*)"
        else:
            default_name = base_name + "_edit.wav"
            filter_str = "WAV files (*.wav);;All files (*)"

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save As",
            os.path.join(base_dir, default_name),
            filter_str,
        )
        if not out_path:
            return

        ok = self._export_audio(out_path, fmt)
        if ok:
            self._settings.last_output_dir = os.path.dirname(out_path)
            self._save_settings()

    @Slot()
    def _on_detect_bpm_clicked(self):
        """Detect BPM using ffmpeg + pure Python envelope autocorrelation."""
        if not self._current_file:
            QMessageBox.information(self, "No file", "Please open an audio file first.")
            return
        try:
            ff = _find_ffmpeg()
        except Exception:
            QMessageBox.critical(
                self,
                "ffmpeg not found",
                "ffmpeg executable not found. Expected in presets/bin or in PATH.",
            )
            return

        bpm = self._estimate_bpm_from_file(self._current_file, ff)
        if bpm is None or bpm <= 0:
            self.lbl_bpm_detect.setText("Detected BPM: n/a")
            QMessageBox.information(
                self,
                "BPM detection",
                "Could not detect BPM reliably for this track.",
            )
            return

        bpm_str = f"{bpm:.2f}".rstrip("0").rstrip(".")
        self.lbl_bpm_detect.setText(f"Detected BPM: {bpm_str}")
        self.edit_bpm.setText(bpm_str)
        # Do NOT save BPM into settings; it is per-track only.

    # ----- Core waveform + ffmpeg actions -----
    def _generate_waveform_preview(self):
        if not self._current_file:
            return
        # Let the waveform widget handle ffmpeg + resampling
        self.waveform_view.set_audio(self._current_file, self._duration_ms, _find_ffmpeg())

    def _estimate_bpm_from_file(self, path: str, ffmpeg_path: str) -> Optional[float]:
        """Estimate BPM using a simple envelope autocorrelation.

        This uses ffmpeg to downmix to mono 8 kHz WAV (max 60 seconds),
        then computes an amplitude envelope and finds the dominant period
        in a 60–200 BPM range.
        """
        ff = ffmpeg_path
        if not ff or not os.path.isfile(path):
            return None

        tmp_wav = None
        try:
            fd, tmp_wav = tempfile.mkstemp(suffix="_musicedit_bpm.wav")
            os.close(fd)

            # Use only the first 60 seconds for speed
            cmd = [
                ff,
                "-y",
                "-i",
                path,
                "-ac",
                "1",
                "-ar",
                "8000",
                "-t",
                "60",
                "-vn",
                "-f",
                "wav",
                tmp_wav,
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                print("MusicEdit BPM: ffmpeg failed:", proc.stderr)
                return None

            with wave.open(tmp_wav, "rb") as wf:
                nframes = wf.getnframes()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                raw = wf.readframes(nframes)

            if sampwidth != 2 or nframes <= 0 or framerate <= 0:
                return None

            fmt = "<%dh" % (len(raw) // 2)
            data = struct.unpack(fmt, raw)

            # Build a simple amplitude envelope (rectified signal),
            # decimated to ~250 Hz to keep things light.
            target_env_sr = 250.0
            step = max(1, int(round(framerate / target_env_sr)))
            if step <= 0:
                step = 1

            env = []
            # Skip first 2 seconds to avoid silence / non-rhythmic intro
            start_sample = int(2.0 * framerate)
            if start_sample >= len(data):
                start_sample = 0

            for i in range(start_sample, len(data), step):
                env.append(abs(data[i]) / 32768.0)

            if len(env) < 16:
                return None

            # Remove DC offset
            mean_env = sum(env) / len(env)
            env = [e - mean_env for e in env]

            # Limit length for safety
            max_points = 4000
            if len(env) > max_points:
                env = env[:max_points]

            N = len(env)
            env_sr = framerate / step  # actual envelope sample rate

            # Search tempo between 60 and 200 BPM
            min_bpm = 60.0
            max_bpm = 200.0
            min_lag = int(env_sr * 60.0 / max_bpm)
            max_lag = int(env_sr * 60.0 / min_bpm)
            if min_lag < 1:
                min_lag = 1
            if max_lag >= N - 2:
                max_lag = N - 2
            if max_lag <= min_lag:
                return None

            best_lag = None
            best_corr = 0.0

            # Simple autocorrelation scan
            for lag in range(min_lag, max_lag + 1):
                s = 0.0
                # To avoid heavy CPU, cap inner loop to a fixed window
                limit = min(N - lag, 3000)
                for i in range(limit):
                    s += env[i] * env[i + lag]
                if s > best_corr:
                    best_corr = s
                    best_lag = lag

            if best_lag is None or best_corr <= 0.0:
                return None

            bpm = 60.0 * env_sr / float(best_lag)

            # Normalize into a more typical dance range if obvious half/double
            while bpm < 80.0:
                bpm *= 2.0
            while bpm > 180.0:
                bpm /= 2.0

            return bpm
        except Exception as e:
            print("MusicEdit BPM: exception while estimating BPM:", e)
            return None
        finally:
            if tmp_wav and os.path.isfile(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def _export_audio(self, out_path: str, fmt: str) -> bool:
        # ensure ffmpeg exists at export time
        try:
            ff = _find_ffmpeg()
        except Exception:
            QMessageBox.critical(
                self,
                "ffmpeg not found",
                "ffmpeg executable not found. Expected in presets/bin or in PATH.",
            )
            return False

        if not self._current_file:
            return False

        start_ms, end_ms = self.waveform_view.get_selection_ms()
        # If selection is invalid or disabled, fall back to full track
        if not self.chk_use_selection.isChecked() or start_ms >= end_ms:
            start_ms = 0
            end_ms = self._duration_ms

        start_sec = start_ms / 1000.0
        duration_sec = (end_ms - start_ms) / 1000.0 if end_ms > start_ms else 0.0

        cmd = [ff, "-y"]

        # Start offset & input
        if start_sec > 0:
            cmd += ["-ss", f"{start_sec:.3f}"]
        cmd += ["-i", self._current_file]

        if duration_sec > 0 and end_ms < self._duration_ms:
            cmd += ["-t", f"{duration_sec:.3f}"]

        # Audio encoding options
        if fmt == "mp3":
            bitrate = self.combo_mp3_bitrate.currentData()
            if bitrate is None:
                bitrate = 320
            cmd += ["-vn", "-c:a", "libmp3lame", "-b:a", f"{bitrate}k"]

            # Metadata
            artist = self.edit_artist.text().strip()
            album = self.edit_album.text().strip()
            title = self.edit_title.text().strip()
            year = self.edit_year.text().strip()
            bpm = self.edit_bpm.text().strip()
            extra = self.edit_extra.text().strip()

            if artist:
                cmd += ["-metadata", f"artist={artist}"]
            if album:
                cmd += ["-metadata", f"album={album}"]
            if title:
                cmd += ["-metadata", f"title={title}"]
            if year:
                cmd += ["-metadata", f"date={year}"]
            if bpm:
                cmd += ["-metadata", f"TBPM={bpm}", "-metadata", f"bpm={bpm}"]
            if extra:
                cmd += ["-metadata", f"comment={extra}"]

            # Save settings
            self._settings.output_format = "mp3"
            self._settings.mp3_bitrate_kbps = int(bitrate)
        else:
            sr = self.combo_wav_samplerate.currentData()
            if sr is None:
                sr = 48000
            cmd += ["-vn", "-c:a", "pcm_s16le", "-ar", str(sr)]

            # Save settings
            self._settings.output_format = "wav"
            self._settings.wav_samplerate_hz = int(sr)

        # Persist metadata text fields into settings (but not BPM)
        self._settings.meta_artist = self.edit_artist.text()
        self._settings.meta_album = self.edit_album.text()
        self._settings.meta_title = self.edit_title.text()
        self._settings.meta_year = self.edit_year.text()
        self._settings.meta_extra = self.edit_extra.text()
        self._save_settings()

        cmd.append(out_path)

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Failed to run ffmpeg:\n{e}")
            return False

        if proc.returncode != 0:
            QMessageBox.critical(
                self,
                "Export failed",
                "ffmpeg reported an error while exporting the audio.\n\n"
                f"Command:\n{' '.join(cmd)}\n\n"
                f"Error:\n{proc.stderr}",
            )
            return False

        QMessageBox.information(self, "Export complete", f"Saved:\n{out_path}")
        return True