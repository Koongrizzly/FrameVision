# helpers/audiotool.py — Audio tool with waveform preview.
# Now with a smart fallback that auto-adds the base Python's site-packages if pyqtgraph import fails in a venv,
# and beginner-friendly tooltips for new users.

import os, re, sys, subprocess, tempfile, wave, json
from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import (
    QWidget, QLineEdit, QToolButton, QCheckBox, QPushButton, QFormLayout,
    QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox, QLabel, QComboBox,
    QSlider, QListWidget, QGroupBox, QRadioButton, QSpinBox, QAbstractItemView
)

# --- Optional waveform visualization (pyqtgraph) ---
_pg = None
_pg_import_err = None
_added_paths = []

def _try_add_base_sitepackages():
    """If we're inside a venv, add the *base* Python's site-packages to sys.path.
    This lets us use a single global install (your Install Menu) without touching the venv.
    """
    try:
        base = Path(getattr(sys, "base_prefix", sys.prefix) or sys.prefix)
        pre = Path(sys.prefix)
        if base and base != pre:
            # Windows
            cand1 = base / "Lib" / "site-packages"
            # Fallbacks (Unix/WSL etc.)
            ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
            cand2 = base / "lib" / ver / "site-packages"
            for p in (cand1, cand2):
                if p.exists() and str(p) not in sys.path:
                    sys.path.append(str(p))
                    _added_paths.append(str(p))
    except Exception:
        pass

try:
    import pyqtgraph as _pg  # noqa: F401
except Exception as e:
    _pg = None
    _pg_import_err = e
    # Try again after adding base site-packages
    _try_add_base_sitepackages()
    if _pg is None:
        try:
            import pyqtgraph as _pg  # noqa: F401
            _pg_import_err = None
        except Exception as e2:
            _pg = None
            _pg_import_err = e2

# --- Safe imports for shared paths/constants ---
try:
    from helpers.framevision_app import ROOT, OUT_VIDEOS
except Exception:
    from pathlib import Path as _Path
    ROOT = _Path('.').resolve()
    BASE = ROOT
    OUT_VIDEOS = BASE/'output'/'video'

def _tip(key:str, default:str) -> str:
    """
    Fetch a human-friendly tooltip string.

    We first try to read a shared assets/tooltips.json so the app can
    centralize wording. If that fails (for example older builds without
    that file), we gracefully fall back to the local default text.

    key: small string like "audiotool.btn_add"
    default: fallback text shown if key isn't found.
    """
    global _TOOLTIP_MAP
    try:
        _TOOLTIP_MAP
    except NameError:
        _TOOLTIP_MAP = None
    if _TOOLTIP_MAP is None:
        _TOOLTIP_MAP = {}
        try:
            assets_dir = ROOT / "assets"
            with open(assets_dir / "tooltips.json", "r", encoding="utf-8") as f:
                import json as _json
                maybe = _json.load(f)
                if isinstance(maybe, dict):
                    _TOOLTIP_MAP.update(maybe)
        except Exception:
            pass
    return _TOOLTIP_MAP.get(key, default)

def _candidate_ff_bins(settings:QSettings):
    override = settings.value("ffmpeg_path", "")
    if override:
        yield override
    env_ff = os.environ.get("FRAMEVISION_FFMPEG")
    if env_ff:
        yield env_ff

    exe = "ffmpeg.exe" if os.name=="nt" else "ffmpeg"
    app_dirs = [
        ROOT/'bin',
        ROOT/'presets'/'bin',
        Path(sys.argv[0]).resolve().parent/'bin',
        Path(sys.argv[0]).resolve().parent/'presets'/'bin',
    ]
    for d in app_dirs:
        yield str(d/exe)
    yield "ffmpeg"

def _candidate_probe_bins(settings:QSettings):
    override = settings.value("ffprobe_path", "")
    if override:
        yield override
    env_fp = os.environ.get("FRAMEVISION_FFPROBE")
    if env_fp:
        yield env_fp
    pro = "ffprobe.exe" if os.name=="nt" else "ffprobe"
    app_dirs = [
        ROOT/'bin',
        ROOT/'presets'/'bin',
        Path(sys.argv[0]).resolve().parent/'bin',
        Path(sys.argv[0]).resolve().parent/'presets'/'bin',
    ]
    for d in app_dirs:
        yield str(d/pro)
    yield "ffprobe"

def _first_ok(cmdname_iter):
    for c in cmdname_iter:
        try:
            subprocess.check_output([str(c), "-version"], stderr=subprocess.STDOUT)
            return str(c)
        except Exception:
            continue
    return None

def ffmpeg_path(settings:QSettings):
    p = _first_ok(_candidate_ff_bins(settings))
    return p or "ffmpeg"

def ffprobe_path(settings:QSettings):
    p = _first_ok(_candidate_probe_bins(settings))
    return p or "ffprobe"

class _Waveform(QWidget):
    """Waveform preview via pyqtgraph. Falls back gracefully if pyqtgraph missing."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        if _pg is None:
            # Show a detailed reason so users don't think it's "not installed" when it's an env/import issue.
            reason = "pyqtgraph not available"
            if _pg_import_err is not None:
                reason = f"import failed: {type(_pg_import_err).__name__}: {_pg_import_err}"
            details = ""
            if _added_paths:
                details = "\\nSearched additionally: " + "; ".join(_added_paths)
            msg = (
                "Waveform preview disabled — " + reason +
                "\\nPython: " + sys.executable + details +
                "\\nTip: use your Install Menu (global site-packages). This panel already attempts to import from base."
            )
            self._label = QLabel(msg)
            try:
                self._label.setWordWrap(True)
            except Exception:
                pass
            self._label.setAlignment(Qt.AlignCenter)
            self._layout.addWidget(self._label)
            self.plot = None
            # Tooltip for when waveform isn't available
            try:
                self.setToolTip(_tip(
                    "audiotool.waveform_disabled",
                    "Waveform preview\\n\\nShows a visual of the selected audio file so you can spot silence and beats. "
                    "It's disabled right now because the graph library (pyqtgraph) is missing, "
                    "but you can still add audio and export normally."
                ))
            except Exception:
                pass
        else:
            self.plot = _pg.PlotWidget()
            try:
                self.plot.setBackground(None)
                self.plot.hideButtons()
                self.plot.setMouseEnabled(x=False, y=False)
                self.plot.showGrid(x=False, y=False)
                self.plot.getPlotItem().hideAxis('left')
                self.plot.getPlotItem().hideAxis('bottom')
            except Exception:
                pass
            self._curve = self.plot.plot([], [])
            self._layout.addWidget(self.plot)
            try:
                self.setToolTip(_tip(
                    "audiotool.waveform",
                    "Waveform preview\\n\\nThis mini-graph lets you SEE the selected audio (loud parts, quiet parts, "
                    "silences). It's only a preview: changing it here will NOT edit the sound."
                ))
            except Exception:
                pass

    def clear(self):
        if getattr(self, "_curve", None):
            self._curve.setData([], [])

    def set_audio(self, path:str):
        if not getattr(self, "plot", None) or _pg is None:
            return
        try:
            tmpdir = tempfile.gettempdir()
            outwav = Path(tmpdir)/("waveprev_%d.wav" % os.getpid())
            if outwav.exists():
                try: outwav.unlink()
                except Exception: pass
            settings = QSettings("FrameVision", "AudioTool")
            cmd = [ffmpeg_path(settings), "-y", "-i", path, "-ac", "1", "-ar", "8000", "-vn", "-f", "wav", str(outwav)]
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            import wave, struct
            with wave.open(str(outwav), 'rb') as wf:
                n = wf.getnframes(); sampwidth = wf.getsampwidth(); framerate = wf.getframerate()
                raw = wf.readframes(n)
                if sampwidth == 2:
                    fmt = "<%dh" % (len(raw)//2)
                    data = struct.unpack(fmt, raw)
                    step = max(1, len(data)//2000)
                    xs = [i/float(framerate) for i in range(0, len(data), step)]
                    ys = [data[i]/32768.0 for i in range(0, len(data), step)]
                else:
                    xs, ys = [], []
            self._curve.setData(xs, ys)
        except Exception:
            self.clear()

def _hms_to_seconds(text:str)->float:
    if not text: return 0.0
    parts = text.split(':')
    try:
        parts = [float(p) for p in parts]
    except Exception:
        return 0.0
    if len(parts)==1:   return parts[0]
    if len(parts)==2:   return parts[0]*60 + parts[1]
    if len(parts)==3:   return parts[0]*3600 + parts[1]*60 + parts[2]
    return 0.0

def install_audio_tool(pane, sec_audio):
    """Install the Audio UI/logic into an existing CollapsibleSection."""
    # Files
    audio_list = QListWidget()
    audio_list.setSelectionMode(QAbstractItemView.SingleSelection)
    # Ensure at least ~4 visible rows by default
    try:
        _row_h = max(22, audio_list.fontMetrics().height() + 8)
        audio_list.setMinimumHeight(_row_h * 4 + 12)
    except Exception:
        pass
    try:
        audio_list.setToolTip(_tip(
            "audiotool.audio_list",
            "Step 1: your audio files (music, voiceover, etc.) that will go into the video.\\n\\n"
            "Click an item to preview its waveform and tweak settings. Use ↑ / ↓ to reorder. "
            "The first item is chosen by default in 'Replace' mode."
        ))
    except Exception:
        pass

    btn_add = QToolButton(); btn_add.setText("Add…")
    try:
        btn_add.setToolTip(_tip(
            "audiotool.btn_add",
            "Add a music / voiceover file (MP3, WAV, etc.) to the list.\\n\\n"
            "This does NOT copy or move the file on disk; it just tells FrameVision to use it."
        ))
    except Exception:
        pass

    btn_remove = QToolButton(); btn_remove.setText("Remove")
    try:
        btn_remove.setToolTip(_tip(
            "audiotool.btn_remove",
            "Remove the selected file from the list.\\n\\n"
            "Your original file on disk is untouched."
        ))
    except Exception:
        pass

    btn_up = QToolButton(); btn_up.setText("↑")
    try:
        btn_up.setToolTip(_tip(
            "audiotool.btn_up",
            "Move the selected audio UP in the list.\\n\\n"
            "Tip: The top item is used automatically if you choose 'Replace with selected track'."
        ))
    except Exception:
        pass

    btn_down = QToolButton(); btn_down.setText("↓")
    try:
        btn_down.setToolTip(_tip(
            "audiotool.btn_down",
            "Move the selected audio DOWN in the list."
        ))
    except Exception:
        pass

    waveform = _Waveform()

    # Mode
    rb_replace = QRadioButton("Replace with selected track")
    rb_mix = QRadioButton("Mix added tracks")
    cb_include_original = QCheckBox("Include original video audio in mix")
    try:
        rb_mix.setToolTip(_tip(
            "audiotool.mode_mix",
            "Blend all added audio files together.\\n\\n"
            "Great for background music under dialogue or voiceover."
        ))
        cb_include_original.setToolTip(_tip(
            "audiotool.include_original",
            "Keep the video's original audio (camera voice / game sounds) in the final result.\\n\\n"
            "Turn this OFF if you want only the new music / voiceover."
        ))
        rb_replace.setToolTip(_tip(
            "audiotool.mode_replace",
            "Throw away the video's original audio and use ONLY one selected track from the list "
            "(full soundtrack swap)."
        ))
    except Exception:
        pass
    rb_mix.setChecked(True)

    # Controls
    vol_slider = QSlider(Qt.Horizontal); vol_slider.setRange(0, 300); vol_slider.setValue(100)
    vol_label = QLabel("Volume: 1.00x")
    try:
        vol_slider.setToolTip(_tip(
            "audiotool.volume",
            "Overall loudness for the ADDED audio tracks.\\n\\n"
            "1.00x = original volume. Lower to duck music under speech, raise to boost it."
        ))
        vol_label.setToolTip(_tip(
            "audiotool.volume",
            "Overall loudness for the ADDED audio tracks.\\n\\n"
            "1.00x = original volume. Lower to duck music under speech, raise to boost it."
        ))
    except Exception:
        pass
    vol_slider.valueChanged.connect(lambda v: vol_label.setText(f"Volume: {v/100.0:.2f}x"))

    spin_fadein = QSpinBox(); spin_fadein.setRange(0, 3600); spin_fadein.setSuffix(" s")
    spin_fadeout_st = QSpinBox(); spin_fadeout_st.setRange(0, 24*3600); spin_fadeout_st.setSuffix(" s")
    spin_fadeout_d = QSpinBox(); spin_fadeout_d.setRange(0, 3600); spin_fadeout_d.setSuffix(" s")
    try:
        spin_fadein.setToolTip(_tip(
            "audiotool.fadein",
            "Fade in (seconds): how long the added audio takes to go from silent to full volume at the start."
        ))
        spin_fadeout_st.setToolTip(_tip(
            "audiotool.fadeout_start",
            "Fade out start (seconds): when the fade-out should BEGIN in the added audio timeline."
        ))
        spin_fadeout_d.setToolTip(_tip(
            "audiotool.fadeout_dur",
            "Fade out dur (seconds): how long the fade-out lasts once it starts."
        ))
    except Exception:
        pass

    edit_ss = QLineEdit(); edit_ss.setPlaceholderText("start (e.g. 5 or 0:05)")
    edit_to = QLineEdit(); edit_to.setPlaceholderText("end (e.g. 1:00)")
    spin_delay = QSpinBox(); spin_delay.setRange(0, 24*3600*1000); spin_delay.setSuffix(" ms")
    cb_loudnorm = QCheckBox("Normalize (EBU R128 loudnorm)")
    fmt = QComboBox(); fmt.addItems(["mp4", "mkv", "mov", "webm"])
    try:
        edit_ss.setToolTip(_tip(
            "audiotool.trim_start",
            "Trim start: begin using the audio from this time.\\nExamples: 5 or 0:05.\\n"
            "Use this to skip any silence at the beginning of the song."
        ))
        edit_to.setToolTip(_tip(
            "audiotool.trim_end",
            "Trim end: stop using the audio after this time.\\nLeave blank to play until the file ends."
        ))
        spin_delay.setToolTip(_tip(
            "audiotool.delay_ms",
            "Delay start (ms): wait this long before the added audio begins.\\n"
            "Example: 1000 = start music 1 second after the video begins."
        ))
        cb_loudnorm.setToolTip(_tip(
            "audiotool.loudnorm",
            "Normalize loudness: automatically level volumes so the result isn't too loud or too quiet.\\n"
            "Recommended if you mix multiple clips with different volumes."
        ))
        fmt.setToolTip(_tip(
            "audiotool.format",
            "Output file type for the new video.\\nMP4 works almost everywhere.\\n"
            "WEBM is smaller but not ideal for some editors."
        ))
    except Exception:
        pass

    # ffmpeg/ffprobe override
    edit_ffpath = QLineEdit(); edit_ffpath.setPlaceholderText("ffmpeg path (optional)")
    btn_pick_ff = QToolButton(); btn_pick_ff.setText("…")
    edit_fbpath = QLineEdit(); edit_fbpath.setPlaceholderText("ffprobe path (optional)")
    btn_pick_fb = QToolButton(); btn_pick_fb.setText("…")
    try:
        edit_ffpath.setToolTip(_tip(
            "audiotool.ffmpeg_override",
            "Advanced: manually choose your ffmpeg.exe (the video+audio engine).\\n"
            "Leave this blank if everything works."
        ))
        btn_pick_ff.setToolTip(_tip(
            "audiotool.ffmpeg_browse",
            "Browse for ffmpeg.exe manually.\\nMost people don't need this."
        ))
        edit_fbpath.setToolTip(_tip(
            "audiotool.ffprobe_override",
            "Advanced: manually choose your ffprobe.exe (media inspector).\\n"
            "Leave this blank if everything works."
        ))
        btn_pick_fb.setToolTip(_tip(
            "audiotool.ffprobe_browse",
            "Browse for ffprobe.exe manually.\\nMost people don't need this."
        ))
    except Exception:
        pass

    btn_audio = QPushButton("Add Audio to Video")
    try:
        btn_audio.setToolTip(_tip(
            "audiotool.render",
            "Final step: create a NEW video file with the audio settings above.\\n\\n"
            "Your original video is never overwritten."
        ))
    except Exception:
        pass

    # Grouping
    grp_files = QGroupBox("Audio files")
    try:
        grp_files.setToolTip(_tip(
            "audiotool.grp_files",
            "Step 1: Add one or more audio files that you want in your final video "
            "(music bed, voiceover, etc.)."
        ))
    except Exception:
        pass
    lf = QVBoxLayout(grp_files)
    row_files = QHBoxLayout()
    row_files.addWidget(btn_add); row_files.addWidget(btn_remove); row_files.addWidget(btn_up); row_files.addWidget(btn_down); row_files.addStretch(1)
    lf.addWidget(audio_list); lf.addLayout(row_files)

    grp_wave = QGroupBox("Waveform preview")
    try:
        grp_wave.setToolTip(_tip(
            "audiotool.grp_wave",
            "Step 2: Preview the currently selected audio as a waveform so you can spot "
            "silence, drops and beats."
        ))
    except Exception:
        pass
    lw = QVBoxLayout(grp_wave); lw.addWidget(waveform)
    # Keep waveform from eating too much vertical space
    try:
        grp_wave.setMaximumHeight(220)
    except Exception:
        pass

    grp_mode = QGroupBox("Mode")
    try:
        grp_mode.setToolTip(_tip(
            "audiotool.grp_mode",
            "Step 3: Tell FrameVision how to use the audio:\\n"
            "• Mix added tracks (background music under dialogue)\\n"
            "• Replace with selected track (full soundtrack swap)"
        ))
    except Exception:
        pass
    lm = QVBoxLayout(grp_mode); lm.addWidget(rb_mix); lm.addWidget(cb_include_original); lm.addWidget(rb_replace)

    form = QFormLayout()
    form.addRow(vol_label, vol_slider)
    form.addRow("Fade in", spin_fadein)
    form.addRow("Fade out start", spin_fadeout_st)
    form.addRow("Fade out dur", spin_fadeout_d)
    form.addRow("Trim start", edit_ss)
    form.addRow("Trim end", edit_to)
    form.addRow("Delay start", spin_delay)
    form.addRow(cb_loudnorm)
    form.addRow("Output", fmt)

    row_ff = QHBoxLayout(); row_ff.addWidget(edit_ffpath); row_ff.addWidget(btn_pick_ff)
    row_fb = QHBoxLayout(); row_fb.addWidget(edit_fbpath); row_fb.addWidget(btn_pick_fb)
    ffgrp = QGroupBox("Advanced: ffmpeg/ffprobe location (leave blank to auto-detect)")
    try:
        ffgrp.setToolTip(_tip(
            "audiotool.ffgrp",
            "Advanced only: If FrameVision can't auto-detect ffmpeg / ffprobe, point to them here.\\n"
            "Most people can ignore this section."
        ))
    except Exception:
        pass
    lf2 = QVBoxLayout(ffgrp); lf2.addLayout(row_ff); lf2.addLayout(row_fb)
    try:
        ffgrp.setVisible(False)
        ffgrp.setMaximumHeight(1)
    except Exception:
        pass


    left = QVBoxLayout(); left.addWidget(grp_files); left.addWidget(grp_wave)
    # Make files area ~3x the height of waveform (when space allows)
    try:
        left.setStretch(0, 3)
        left.setStretch(1, 1)
    except Exception:
        pass
    right = QVBoxLayout(); right.addWidget(grp_mode); right.addLayout(form); right.addWidget(ffgrp); right.addStretch(1); right.addWidget(btn_audio)

    root = QHBoxLayout()
    root.addLayout(left, 1); root.addLayout(right, 1)
    try:
        root.setStretch(0,1); root.setStretch(1,1)
    except Exception:
        pass
    container = QWidget(); container.setLayout(root)
    try:
        sp = container.sizePolicy()
        sp.setHorizontalStretch(0); sp.setVerticalStretch(0)
        sp.setRetainSizeWhenHidden(False)
        container.setSizePolicy(sp)
    except Exception:
        pass

    lay_audio = QVBoxLayout(); lay_audio.addWidget(container)
    sec_audio.setContentLayout(lay_audio)
    try:
        sec_audio.setToolTip(_tip(
            "audiotool.section",
            "Audio Tool workflow:\\n"
            "1. Add your music / voiceover.\\n"
            "2. Pick Mix or Replace.\\n"
            "3. Adjust volume, fades, etc.\\n"
            "4. Click 'Add Audio to Video' to render a NEW file."
        ))
    except Exception:
        pass

    # Settings
    settings = QSettings("FrameVision", "AudioTool")
    def _load_settings():
        vol_slider.setValue(int(settings.value("volume_pct", 100)))
        spin_fadein.setValue(int(settings.value("fadein", 0)))
        spin_fadeout_st.setValue(int(settings.value("fadeout_st", 0)))
        spin_fadeout_d.setValue(int(settings.value("fadeout_d", 0)))
        edit_ss.setText(settings.value("trim_ss", ""))
        edit_to.setText(settings.value("trim_to", ""))
        spin_delay.setValue(int(settings.value("delay_ms", 0)))
        cb_loudnorm.setChecked(bool(int(settings.value("loudnorm", 0))))
        fmt.setCurrentText(settings.value("fmt", "mp4"))
        rb_mix.setChecked(bool(int(settings.value("rb_mix", 1))))
        rb_replace.setChecked(bool(int(settings.value("rb_replace", 0))))
        cb_include_original.setChecked(bool(int(settings.value("include_orig", 1))))
        edit_ffpath.setText(settings.value("ffmpeg_path", ""))
        edit_fbpath.setText(settings.value("ffprobe_path", ""))
        recent = settings.value("recent_files", "")
        if recent:
            for p in recent.split("|"):
                p = p.strip()
                if p and os.path.isfile(p):
                    audio_list.addItem(p)
    def _save_settings():
        settings.setValue("volume_pct", vol_slider.value())
        settings.setValue("fadein", spin_fadein.value())
        settings.setValue("fadeout_st", spin_fadeout_st.value())
        settings.setValue("fadeout_d", spin_fadeout_d.value())
        settings.setValue("trim_ss", edit_ss.text())
        settings.setValue("trim_to", edit_to.text())
        settings.setValue("delay_ms", spin_delay.value())
        settings.setValue("loudnorm", int(cb_loudnorm.isChecked()))
        settings.setValue("fmt", fmt.currentText())
        settings.setValue("rb_mix", int(rb_mix.isChecked()))
        settings.setValue("rb_replace", int(rb_replace.isChecked()))
        settings.setValue("include_orig", int(cb_include_original.isChecked()))
        settings.setValue("ffmpeg_path", edit_ffpath.text().strip())
        settings.setValue("ffprobe_path", edit_fbpath.text().strip())
        files = [audio_list.item(i).text() for i in range(audio_list.count())]
        settings.setValue("recent_files", "|".join(files))

    _load_settings()

    # File actions
    def _add_file():
        start_dir = settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(pane, "Choose audio file.", start_dir,
                    "Audio files (*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus);All files (*)")
        if path:
            settings.setValue("last_dir", str(Path(path).parent))
            audio_list.addItem(path)
    def _remove_selected():
        row = audio_list.currentRow()
        if row >= 0:
            audio_list.takeItem(row)
    def _move_selected(offset:int):
        row = audio_list.currentRow()
        if row < 0: return
        new_row = max(0, min(audio_list.count()-1, row+offset))
        if new_row == row: return
        item = audio_list.takeItem(row)
        audio_list.insertItem(new_row, item.text())
        audio_list.setCurrentRow(new_row)
    def _refresh_waveform():
        it = audio_list.currentItem()
        if it: waveform.set_audio(it.text())
        else: waveform.clear()

    btn_add.clicked.connect(_add_file)
    btn_remove.clicked.connect(_remove_selected)
    btn_up.clicked.connect(lambda: _move_selected(-1))
    btn_down.clicked.connect(lambda: _move_selected(1))
    audio_list.currentItemChanged.connect(lambda *_: _refresh_waveform())
    btn_pick_ff.clicked.connect(lambda: edit_ffpath.setText(QFileDialog.getOpenFileName(pane, "Locate ffmpeg", "", "ffmpeg (ffmpeg*.*);;All files (*)")[0]))
    btn_pick_fb.clicked.connect(lambda: edit_fbpath.setText(QFileDialog.getOpenFileName(pane, "Locate ffprobe", "", "ffprobe (ffprobe*.*);;All files (*)")[0]))

    # Build and run
    def _build_and_run():
        inp = pane._ensure_input()
        if not inp:
            return

        # preflight ffmpeg presence
        ff = ffmpeg_path(settings)
        try:
            subprocess.check_output([ff, "-version"], stderr=subprocess.STDOUT)
        except Exception:
            try:
                QMessageBox.critical(pane, "FFmpeg not found",
                    "Couldn't find FFmpeg. Set its path under 'Advanced' or place ffmpeg in:\\n"
                    f"{ROOT/'bin'}\\n{ROOT/'presets'/'bin'}")
            except Exception:
                pass
            return

        files = [audio_list.item(i).text() for i in range(audio_list.count())]
        if not files:
            try: QMessageBox.warning(pane, "Add Audio", "Please add at least one audio file.")
            except Exception: pass
            return

        container_ext = fmt.currentText()
        if container_ext == "webm":
            vcodec = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30"]
            acodec = ["-c:a", "libopus"]
        else:
            vcodec = ["-c:v", "copy"]
            acodec = ["-c:a", "aac"]

        try:
            out_dir = OUT_VIDEOS
        except Exception:
            out_dir = Path('.')
        out = out_dir / f"{inp.stem}_withaudio.{container_ext}"

        vol = vol_slider.value()/100.0
        fadein = int(spin_fadein.value())
        fadeout_st = int(spin_fadeout_st.value())
        fadeout_d = int(spin_fadeout_d.value())
        trim_ss = _hms_to_seconds(edit_ss.text().strip())
        trim_to = _hms_to_seconds(edit_to.text().strip())
        delay_ms = int(spin_delay.value())
        use_loudnorm = cb_loudnorm.isChecked()
        mode_mix = rb_mix.isChecked()
        include_orig = cb_include_original.isChecked()

        cmd = [ff, "-y", "-i", str(inp)]
        for f in files:
            cmd += ["-i", f]

        filter_parts = []
        audio_labels = []

        def process_added(idx:int, label_out:str):
            chain = [f"[{idx}:a]"]
            if trim_ss>0 or trim_to>0:
                if trim_to>0 and trim_to>trim_ss:
                    chain.append(f"atrim=start={trim_ss}:end={trim_to}")
                elif trim_ss>0:
                    chain.append(f"atrim=start={trim_ss}")
                chain.append("asetpts=PTS-STARTPTS")
            if delay_ms>0:
                chain.append(f"adelay={delay_ms}|{delay_ms}")
            if vol!=1.0:
                chain.append(f"volume={vol}")
            if fadein>0:
                chain.append(f"afade=t=in:ss=0:d={fadein}")
            if fadeout_d>0 and fadeout_st>0:
                chain.append(f"afade=t=out:st={fadeout_st}:d={fadeout_d}")
            chain.append(f"anull[{label_out}]")
            filter_parts.append(chain[0] + ",".join(chain[1:]))

        if mode_mix:
            if include_orig:
                filter_parts.append("[0:a]anull[orig]")
                audio_labels.append("orig")
            for i in range(len(files)):
                lab = f"a{i}"
                process_added(i+1, lab)
                audio_labels.append(lab)
            if len(audio_labels) < 1:
                try: QMessageBox.warning(pane, "Add Audio", "Nothing to mix. Add audio files or include the original audio.")
                except Exception: pass
                return
            mix_label = "mixout"
            amix = "[" + "][".join(audio_labels) + "]"
            amix += f"amix=inputs={len(audio_labels)}:duration=shortest:dropout_transition=0[{mix_label}]"
            filter_parts.append(amix)
            final_audio = mix_label
            if use_loudnorm:
                filter_parts.append(f"[{final_audio}]loudnorm=I=-16:TP=-1.5:LRA=11:linear=true[normout]")
                final_audio = "normout"
            cmd += ["-filter_complex", ";".join(filter_parts),
                    "-map", "0:v:0", "-map", f"[{final_audio}]"]
            cmd += vcodec + acodec + ["-shortest", str(out)]
        else:
            idx = audio_list.currentRow()
            if idx < 0: idx = 0
            if idx >= len(files): idx = 0
            sel = idx + 1
            process_added(sel, "selout")
            final_audio = "selout"
            if use_loudnorm:
                filter_parts.append(f"[{final_audio}]loudnorm=I=-16:TP=-1.5:LRA=11:linear=true[normout]")
                final_audio = "normout"
            cmd += ["-filter_complex", ";".join(filter_parts),
                    "-map", "0:v:0", "-map", f"[{final_audio}]"]
            cmd += vcodec + acodec + ["-shortest", str(out)]

        _save_settings()
        pane._run(cmd, out)

    btn_audio.clicked.connect(_build_and_run)

    # Expose a couple of widgets (optional)
    try:
        pane.audio_list = audio_list
        pane.waveform_widget = waveform
    except Exception:
        pass
