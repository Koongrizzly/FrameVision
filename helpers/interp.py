
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, datetime, shutil, zipfile, time
from pathlib import Path
from pathlib import Path as _P_INT

from PySide6.QtCore import Qt, QSettings, QTimer, QUrl, QProcess, QCoreApplication, QThread, QSize
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider,
    QCheckBox, QMessageBox, QFileDialog, QComboBox, QSizePolicy, QScrollArea, QFrame
)
from PySide6.QtWidgets import QProgressBar

from helpers.queue_adapter import enqueue_tool_job, default_outdir, jobs_dirs
from helpers.mediainfo import refresh_info_now
try:
    from helpers.queue_system import QueueSystem
except Exception:
    QueueSystem = None  # type: ignore

QWIDGETSIZE_MAX = 16777215

# --------- utility paths ---------
def _outputs_dir(root: Path) -> Path:
    d = Path(default_outdir(True, "rife")); d.mkdir(parents=True, exist_ok=True); return d

def _assets_zip(root: Path) -> Path:
    return root / "assets" / "rife.zip"

def _ncnn_bin_dir(root: Path) -> Path:
    # Prefer new layout: models/rife-ncnn-vulkan; fallback to legacy rife/bin
    preferred = root / "models" / "rife-ncnn-vulkan"
    legacy = root / "rife" / "bin"
    return preferred if preferred.exists() else legacy

def _model_dir(root: Path, key: str) -> Path:
    # Models live under base/<model-key>. Provide a compatibility fallback for renamed keys.
    base = _ncnn_bin_dir(root)
    preferred = base / key
    if preferred.exists():
        return preferred
    # Fallbacks for older installers
    if key == 'rife-v4.6':
        alt = base / 'rife-v4'
        if alt.exists():
            return alt
    return preferred  # default (may not exist)

# --------- probes ---------
def _guess_current_video(main) -> Path | None:
    for attr in ("current_path","current_video"):
        p = getattr(main, attr, None)
        if isinstance(p,(str,os.PathLike)):
            p = Path(p)
            if p.exists(): return p
    pl = getattr(main, "player", None)
    if pl is not None:
        for attr in ("current_path","path"):
            p = getattr(pl, attr, None)
            if isinstance(p,(str,os.PathLike)):
                p = Path(p)
                if p.exists(): return p
    return None

def _probe_fps(path: Path) -> float:
    fps = 30.0
    try:
        proc = QProcess()
        proc.start("ffprobe", ["-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","default=nokey=1:noprint_wrappers=1", str(path)])
        proc.waitForFinished(1500)
        txt = proc.readAllStandardOutput().data().decode("utf-8","ignore").strip()
        if "/" in txt:
            num, den = txt.split("/",1); fps = float(num) / (float(den) or 1.0)
        elif txt: fps = float(txt)
    except Exception: pass
    return max(1.0, fps)

def _count_frames_fast(path: Path) -> int:
    """Try to count frames quickly via ffprobe. Fallback 0 on failure."""
    try:
        proc = QProcess()
        proc.start("ffprobe", ["-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","default=nokey=1:noprint_wrappers=1", str(path)])
        proc.waitForFinished(3000)
        txt = proc.readAllStandardOutput().data().decode("utf-8","ignore").strip()
        return int(txt) if txt.isdigit() else 0
    except Exception:
        return 0

def _extract_first_frame(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = QProcess()
        p.start("ffmpeg", ["-hide_banner","-y","-ss","0","-i", str(src), "-frames:v","1", "-q:v","3", str(dst)])
        p.waitForFinished(4000)
        return dst.exists()
    except Exception:
        return False

def _atempo_chain(mult: float) -> str:
    f=float(mult); chain=[]
    while f < 0.5: chain.append(0.5); f*=2.0
    chain.append(max(0.5, min(2.0, f)))
    return ",".join(f"atempo={x:.3g}" for x in chain)

# --------- UI helpers ---------

class Collapsible(QWidget):
    def __init__(self, title: str, parent=None, start_open=True):
        super().__init__(parent)
        self._open = start_open
        self.header = QPushButton(("▼ " if start_open else "▶ ") + title)
        self.header.setCheckable(True)
        self.header.setChecked(start_open)
        self.header.clicked.connect(self._toggle)

        self.body = QWidget()
        self.body_l = QVBoxLayout(self.body)
        self.body_l.setContentsMargins(0, 2, 0, 0)
        self.body_l.setSpacing(4)
        self.body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.header)
        lay.addWidget(self.body)
        self._sync()

    def body_layout(self):
        return self.body_l

    def _toggle(self):
        self._open = not self._open
        self._sync()
        # Notify parent to persist, if provided
        try:
            p = self.parent()
            if p is not None and hasattr(p, "_persist_state"):
                p._persist_state()
        except Exception:
            pass

    def _sync(self):
        if self._open:
            self.body.setMaximumHeight(QWIDGETSIZE_MAX)
            self.body.show()
        else:
            self.body.setMaximumHeight(0)
            self.body.hide()
        t = self.header.text()
        self.header.setText(("▼ " if self._open else "▶ ") + t[2:])
# --------- main pane ---------
class InterpPane(QWidget):
    def __init__(self, main, paths: dict, parent=None):
        super().__init__(parent)
        self.main = main; self.ROOT = Path(paths.get('ROOT') or '.').resolve()
        self.settings = QSettings("FrameVision","FrameVision")
        self._last_job_id = None; self._last_expected_output = None
        self._watch_timer = QTimer(self); self._watch_timer.setInterval(500); self._watch_timer.timeout.connect(self._check_job_done)
        self._ffmpeg_timer = QTimer(self); self._ffmpeg_timer.setInterval(400); self._ffmpeg_timer.timeout.connect(self._poll_ffmpeg_progress)
        self._ffmpeg_progress_file = None

        # live progress sampling
        self._progress_timer = QTimer(self); self._progress_timer.setInterval(300); self._progress_timer.timeout.connect(self._update_time_left)
        self._progress_started_at = None
        self._progress_total_frames = 0
        self._progress_done_frames = 0

        # known models for NCNN path
        self._models = [
            ("rife",      "RIFE default — legacy base"),
            ("rife-v4",   "RIFE v4 — older general use"),
            ("rife-v4.6", "RIFE v4.6 — fast, general use"),
            ("rife-HD",   "RIFE HD — 1080p–1440p detail"),
            ("rife-UHD",  "RIFE UHD — 4K masters"),
            ("rife-anime","RIFE Anime — line-art/Animation"),
        ]
        self._model_blurbs = {
            "rife-v4.6": ("Balanced quality/speed.", "Best for 720p–1080p; default choice."),
            "rife-HD":   ("More temporal stability for texture.", "Good for 1080p→1440p; slower, more VRAM."),
            "rife-UHD":  ("Maximum stability for 4K/fine patterns.", "Slowest; highest VRAM usage."),
            "rife-anime":("Edge-stable for cel/line-art.", "Can oversoften live-action."),
        }

        self._build_ui(); self._restore_state(); self._on_slider_changed(self.slider.value())
        self._refresh_recent()

    def _build_ui(self):
        # Wrap everything in a scroll area so controls never go out of sight
        outer = QVBoxLayout(self); outer.setContentsMargins(10,10,10,10); outer.setSpacing(4)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)
        content = QWidget(); scroll.setWidget(content)
        root = QVBoxLayout(content); root.setContentsMargins(0,0,0,0); root.setSpacing(4)

        # ---- Options ----
        opt = Collapsible("Options", start_open=True)
        root.addWidget(opt); lay = opt.body_layout()

        # Row 1: multiplier + slider + quick-set buttons
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(QLabel("Multiplier:"))
        self.slider = QSlider(Qt.Horizontal); self.slider.setMinimum(15); self.slider.setMaximum(400); self.slider.setTickInterval(5); self.slider.setSingleStep(5)
        self.slider.setFixedHeight(28)
        self.slider.setStyleSheet("QSlider::groove:horizontal{height:10px;border-radius:5px;} QSlider::handle:horizontal{width:18px;height:18px;margin:-6px 0;border-radius:9px;}")
        row.addWidget(self.slider)
        self.lbl_mult = QLabel("2.00×"); row.addWidget(self.lbl_mult)
        self.btn_2x = QPushButton("2×"); self.btn_4x = QPushButton("4×")
        row.addWidget(self.btn_2x); row.addWidget(self.btn_4x)
        row.addStretch(1); lay.addLayout(row)

        lay.addSpacing(8)

        # Row 2: toggles on their own line
        row2 = QHBoxLayout(); row2.setSpacing(8)
        self.cb_stream = QCheckBox("Streaming (low memory)"); row2.addWidget(self.cb_stream)
        self.cb_autoplay = QCheckBox(\"Auto-play when finished\"); row2.addWidget(self.cb_autoplay)
        self.cb_autoplay.toggled.connect(lambda _on: self._persist_state())
        row2.addStretch(1); lay.addLayout(row2)

        # Result info (icon + text)
        res_row = QHBoxLayout(); res_row.setSpacing(8)
        self.lbl_result_icon = QLabel("·")  # dot placeholder
        self.lbl_result = QLabel("Result: —")
        self.lbl_result_icon.setFixedWidth(16)
        self.lbl_result_icon.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.lbl_result.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.lbl_result.setWordWrap(False)
        res_row.addWidget(self.lbl_result_icon)
        res_row.addWidget(self.lbl_result)
        res_row.addStretch(1)
        lay.addLayout(res_row)

        # Inline progress widgets (always available; hidden until running)
        self.pb_wrap = QHBoxLayout(); self.pb_wrap.setSpacing(8)
        self.pb_label = QLabel("Interpolating…"); self.pb_label.setVisible(False)
        self.pb = QProgressBar(); self.pb.setRange(0,100); self.pb.setValue(0); self.pb.setTextVisible(True); self.pb.setVisible(False)
        self.pb_eta = QLabel(""); self.pb_eta.setVisible(False)
        self.pb_cancel = QPushButton("Cancel"); self.pb_cancel.setVisible(False)
        self.pb_wrap.addWidget(self.pb_label); self.pb_wrap.addWidget(self.pb, 1); self.pb_wrap.addWidget(self.pb_eta); self.pb_wrap.addWidget(self.pb_cancel)
        lay.addLayout(self.pb_wrap)

        # Row 3: Play buttons
        lay.addSpacing(6)
        play_row = QHBoxLayout(); play_row.setSpacing(8)
        self.btn_play_last = QPushButton("Play last result (internal)")
        self.btn_play_last_ext = QPushButton("Result in external player")
        play_row.addWidget(self.btn_play_last); play_row.addWidget(self.btn_play_last_ext); play_row.addStretch(1)
        lay.addLayout(play_row)

        # ---- Recent results gallery (NEW) ----
        self.recent = Collapsible("Recent results", start_open=False)
        root.addWidget(self.recent)
        rlay = self.recent.body_layout()
        self.recent_area = QScrollArea(); self.recent_area.setWidgetResizable(True); self.recent_area.setFrameShape(QFrame.NoFrame); self.recent_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); self.recent_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.recent_container = QWidget(); self.recent_h = QHBoxLayout(self.recent_container); self.recent_h.setContentsMargins(0,0,0,0); self.recent_h.setSpacing(8)
        self.recent_area.setWidget(self.recent_container)
        rlay.addWidget(self.recent_area)

        # ---- Advanced ----
        adv = Collapsible("Advanced", start_open=True)
        root.addWidget(adv); advl = adv.body_layout()

        # Load profile
        sp = QHBoxLayout(); sp.setSpacing(8)
        self.combo_speed = QComboBox()
        self.combo_speed.addItems([
            "1 — Minimal load (responsive)",
            "2 — Fast",
            "3 — Balanced",
            "4 — High quality (heavier)",
            "5 — Superfast (default)",
        ])
        self.combo_speed.setCurrentIndex(4)
        sp.addWidget(QLabel("Load profile:")); sp.addWidget(self.combo_speed)
        self.cb_speed_default = QCheckBox("Default"); self.cb_speed_default.setChecked(False)
        sp.addStretch(1); advl.addLayout(sp)
        self.warn_lbl = QLabel(""); advl.addWidget(self.warn_lbl)

        # Model selector (applies to internal NCNN path when Superfast selected)
        mp = QHBoxLayout(); mp.setSpacing(8)
        mp.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        for key, title in self._models:
            self.combo_model.addItem(title, key)
        mp.addWidget(self.combo_model)
        self.btn_model_install = QPushButton("Install/Verify")
        mp.addWidget(self.btn_model_install)
        mp.addStretch(1); advl.addLayout(mp)

        # Two info lines under the selector
        self.model_info_1 = QLabel(""); self.model_info_1.setWordWrap(True)
        self.model_info_2 = QLabel(""); self.model_info_2.setWordWrap(True)
        advl.addWidget(self.model_info_1); advl.addWidget(self.model_info_2)
        self.model_info_1.setVisible(False); self.model_info_2.setVisible(False)

        # ---- Buttons (2 rows) ----
        btns = QVBoxLayout(); btns.setSpacing(6)
        r2 = QHBoxLayout(); r2.setSpacing(8)
        self.btn_start = QPushButton("Add to Queue") 
        self.btn_batch = QPushButton("Add Batch"); self.btn_open_folder = QPushButton("Open RIFE folder")
        r2.addWidget(self.btn_start); r2.addWidget(self.btn_batch); r2.addWidget(self.btn_open_folder); r2.addStretch(1); btns.addLayout(r2)
        root.addLayout(btns)

        # tooltips
        self.slider.setToolTip("FPS multiplier (0.15×–4.00×). ≥1.0× raises FPS. <1.0× makes slow motion.")
        self.lbl_mult.setToolTip("Current multiplier applied to input FPS.")
        self.btn_2x.setToolTip("Set multiplier to 2×."); self.btn_4x.setToolTip("Set multiplier to 4×.")
        self.cb_stream.setToolTip("Low‑memory streaming path (for the ONNX backend).")
        self.cb_autoplay.setToolTip("Play result in the internal player when the job finishes.")
        self.combo_speed.setToolTip("Processing profile for FFmpeg path. Select 'Superfast' to use the NCNN engine in the foreground with real progress.")
        self.cb_speed_default.setToolTip("Reset to a safe default (Balanced).")
        self.combo_model.setToolTip("RIFE model (used by the Superfast NCNN path).")
        self.btn_model_install.setToolTip("Extract engine & models from assets/rife.zip if needed.")
        self.btn_play_last.setToolTip("Open the most recent result in the internal player.")
        self.btn_play_last_ext.setToolTip("Open the most recent result in your system player.")
        self.btn_start.setToolTip("Add to Queue (or run Superfast immediately if profile 5 is selected).")
        self.btn_batch.setToolTip("Select multiple files and enqueue with current settings.")
        self.btn_open_folder.setToolTip("Open the folder where RIFE outputs are saved.")

        # wiring
        self.btn_2x.clicked.connect(lambda: self.slider.setValue(200))
        self.btn_4x.clicked.connect(lambda: self.slider.setValue(400))
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.btn_start.clicked.connect(self._on_start_clicked)
        self.btn_batch.clicked.connect(self._on_batch_clicked)
        self.btn_play_last.clicked.connect(self._on_play_last_internal)
        self.btn_play_last_ext.clicked.connect(self._on_play_last_external)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        self.combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        self.cb_speed_default.toggled.connect(self._on_speed_default)
        self.combo_model.currentIndexChanged.connect(self._on_model_changed)
        self.btn_model_install.clicked.connect(self._on_model_install)

        # Set initial result state
        self._set_result_state('idle')

    # ---- state / progress helpers ----
    def _progress_start(self, label: str = "Interpolating…"):
        # determinate (0-100) progress
        self.pb.setRange(0, 100)
        self.pb_label.setText(label)
        self.pb_label.setVisible(True); self.pb.setVisible(True); self.pb_eta.setVisible(True); self.pb_cancel.setVisible(True)
        self.pb.setValue(0); self._cancel_flag = False
        try:
            self.pb_cancel.clicked.disconnect()
        except Exception:
            pass
        self.pb_cancel.clicked.connect(lambda: setattr(self, '_cancel_flag', True))
        self._progress_started_at = time.time()
        self._progress_done_frames = 0
        self._progress_timer.start()
        QCoreApplication.processEvents()

    def _progress_start_indeterminate(self, label: str = "Processing (queued)…"):
        # indeterminate bar used while FFmpeg runs in the queue
        self.pb_label.setText(label)
        self.pb_label.setVisible(True); self.pb.setVisible(True); self.pb_eta.setVisible(False); self.pb_cancel.setVisible(False)
        self.pb.setRange(0, 0)  # 0,0 = Qt marquee
        QCoreApplication.processEvents()
        self.pb_label.setText(label)
        self.pb_label.setVisible(True); self.pb.setVisible(True); self.pb_eta.setVisible(True); self.pb_cancel.setVisible(True)
        self.pb.setValue(0); self._cancel_flag = False
        try:
            self.pb_cancel.clicked.disconnect()
        except Exception:
            pass
        self.pb_cancel.clicked.connect(lambda: setattr(self, '_cancel_flag', True))
        self._progress_started_at = time.time()
        self._progress_done_frames = 0
        self._progress_timer.start()
        QCoreApplication.processEvents()

    def _progress_set(self, val: int, done_frames: int | None = None):
        if done_frames is not None:
            self._progress_done_frames = max(self._progress_done_frames, done_frames)
        self.pb.setValue(max(0, min(100, int(val))))
        QCoreApplication.processEvents()

    def _progress_end(self):
        self._progress_timer.stop()
        self.pb_label.setVisible(False); self.pb.setVisible(False); self.pb_cancel.setVisible(False); self.pb_eta.setVisible(False)
        QCoreApplication.processEvents()

    def _update_time_left(self):
        if not self._progress_started_at or self._progress_total_frames <= 0:
            self.pb_eta.setText("")
            return
        elapsed = max(0.001, time.time() - self._progress_started_at)
        fps = self._progress_done_frames / elapsed if self._progress_done_frames else 0.0
        remaining = max(0, self._progress_total_frames - self._progress_done_frames)
        eta = int(remaining / fps) if fps > 0 else 0
        if fps > 0:
            self.pb_eta.setText(f"ETA ~{eta}s @ {fps:.1f} fps")
        else:
            self.pb_eta.setText("")

    
    # ---- simple JSON store (robust persistence) ----
    def _store_path(self) -> Path:
        sp = self.ROOT / "settings"
        sp.mkdir(parents=True, exist_ok=True)
        return sp / "interp.json"

    def _kv_read(self, key: str, default=None):
        try:
            data = json.loads(self._store_path().read_text(encoding="utf-8"))
        except Exception:
            data = {}
        return data.get(key, default)

    def _kv_write(self, key: str, value):
        sp = self._store_path()
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data[key] = value
        sp.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _kv_sync_all_from_runtime(self):
        self._kv_write("rife/multiplier", int(self.slider.value()))
        self._kv_write("rife/streaming", int(self.cb_stream.isChecked()))
        self._kv_write("rife/autoplay", int(self.cb_autoplay.isChecked()))
        self._kv_write("rife/speed_idx", int(self.combo_speed.currentIndex()))
        self._kv_write("rife/speed_default", int(self.cb_speed_default.isChecked()))
        self._kv_write("rife/model_key", self.combo_model.currentData())
def _set_result_state(self, state: str, text: str | None = None):
        # States: idle, working, ok, err
        colors = {
            'idle': '#9aa0a6',
            'working': '#c7a800',
            'ok': '#3ddc84',
            'err': '#ff6b6b',
        }
        icons = {
            'idle': '·',
            'working': '⏳',
            'ok': '✔',
            'err': '✖',
        }
        if text is not None:
            self.lbl_result.setText(text)
        self.lbl_result_icon.setText(icons.get(state, '·'))
        if state == 'idle':
            self.lbl_result_icon.setStyleSheet(''); self.lbl_result.setStyleSheet('')
        else:
            col = colors.get(state, '#9aa0a6')
            self.lbl_result_icon.setStyleSheet(f'color:{col}')
            self.lbl_result.setStyleSheet(f'color:{col}')

    def _set_working(self):
        self._set_result_state('working', 'Working…')

    def _set_ok(self, msg: str = 'Result: ready'):
        self._set_result_state('ok', msg)

    def _set_err(self, msg: str = 'Result: error'):
        self._set_result_state('err', msg)

    def _on_speed_changed(self, idx: int):
        self.warn_lbl.setText("⚠️ Superfast may peg your machine and freeze UI until done." if idx==4 else "")
        if self.cb_speed_default.isChecked(): self.cb_speed_default.setChecked(False)
        self._persist_state()
        self._persist_state()

    def _on_speed_default(self, on: bool):
        if on: self.combo_speed.setCurrentIndex(3-1); self.warn_lbl.setText("")  # set to "Balanced"
        self._persist_state()

    def _restore_state(self):
        # Block signals during restore to avoid handlers overwriting values
        self.combo_speed.blockSignals(True)
        self.cb_speed_default.blockSignals(True)
        self.combo_model.blockSignals(True)
        s=self.settings
        self.slider.setValue(int(s.value("rife/multiplier", 200)))
        self.cb_stream.setChecked(bool(int(s.value("rife/streaming", 0))))
        self.cb_autoplay.setChecked(bool(int(s.value("rife/autoplay", 1))))
        self.combo_speed.setCurrentIndex(int(s.value("rife/speed_idx", 4)))
        self.cb_speed_default.setChecked(bool(int(s.value("rife/speed_default", 0))))
        model_key = s.value("rife/model_key", "rife-UHD")
        self._set_model_ui(model_key)
        # Migration block removed: no longer force overwrite of saved speed/model        # One-time corrective migration v3: force Superfast + UHD if previous bug left bad values
        migrated_v3 = int(self.settings.value("rife/migrated_superfast_default_v3", 0))
        try:
            cur_idx = int(s.value("rife/speed_idx", -1))
        except Exception:
            cur_idx = -1
        model_key_cur = s.value("rife/model_key", "")
        if not migrated_v3 and (cur_idx in (-1, 0) or model_key_cur in ("", "rife", "rife-v4")):
            self.combo_speed.setCurrentIndex(4)
            s.setValue("rife/speed_idx", 4)
            s.setValue("rife/model_key", "rife-UHD")
            self._set_model_ui("rife-UHD")
            s.setValue("rife/migrated_superfast_default_v3", 1)
        # Unblock signals
        self.combo_speed.blockSignals(False)
        self.cb_speed_default.blockSignals(False)
        self.combo_model.blockSignals(False)



    def _persist_state(self):
        s=self.settings
        s.setValue('rife/multiplier', int(self.slider.value()))
        s.setValue('rife/streaming', int(self.cb_stream.isChecked()))
        s.setValue('rife/autoplay', int(self.cb_autoplay.isChecked()))
        s.setValue('rife/speed_idx', int(self.combo_speed.currentIndex()))
        s.setValue('rife/speed_default', int(self.cb_speed_default.isChecked()))
        s.setValue('rife/model_key', self.combo_model.currentData())
        try:
            s.sync()
        except Exception:
            pass
        # JSON store as truth
        self._kv_sync_all_from_runtime()

# ---- model handling ----
    def _set_model_ui(self, key: str):
        # select in combo and update blurbs
        idx = max(0, self.combo_model.findData(key))
        self.combo_model.setCurrentIndex(idx)
        bl = self._model_blurbs.get(key) or ("","")
        t1 = ("• " + bl[0]) if bl[0].strip() else ""
        t2 = ("• " + bl[1]) if bl[1].strip() else ""
        self.model_info_1.setText(t1); self.model_info_2.setText(t2)
        self.model_info_1.setVisible(bool(t1)); self.model_info_2.setVisible(bool(t2))

    def _on_model_changed(self, idx:int):
        key = self.combo_model.currentData()
        self._set_model_ui(key)
        self._persist_state()
        self.settings.sync()

    def _extract_assets_if_needed(self, parent=None) -> bool:
        exe = _ncnn_bin_dir(self.ROOT) / "rife-ncnn-vulkan.exe"
        if exe.exists(): return True
        z = _assets_zip(self.ROOT)
        if not z.exists():
            QMessageBox.warning(self, "RIFE engine", "Missing engine.\nExpected at 'models/rife-ncnn-vulkan/'.\nOptionally place 'assets/rife.zip' (contains 'models/rife-ncnn-vulkan/rife-ncnn-vulkan.exe' and models) for auto-extract.")
            return False
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                zf.extractall(self.ROOT)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Install error", f"Could not extract rife.zip:\n{e}")
            return False

    def _on_model_install(self):
        if self._extract_assets_if_needed(self):
            key = self.combo_model.currentData()
            mdir = _model_dir(self.ROOT, key)
            if mdir.exists():
                actual = mdir.name
                note = f" (using '{actual}' folder)" if actual != key else ""
                QMessageBox.information(self, "Model", f"Model '{key}' is ready{note}.")
            else:
                QMessageBox.information(self, "Model", f"Engine installed, but '{key}' folder not found.\nExpected: {mdir}")

    # ---- helpers (math) ----
    @staticmethod
    def _expected_out_frames(n: int, in_count: int) -> int:
        # RIFE inserts (n-1) frames between each pair; total = (in-1)*n + 1
        return max(1, (max(1, in_count)-1)*n + 1)

    # ---- queue path (FFmpeg) ----
    def _filters_for_speed(self)->tuple[str,list[str]]:
        i=int(self.combo_speed.currentIndex())
        if i==0:   return ("minterpolate=fps={fps}:mi_mode=blend", ["-preset","veryfast","-threads","1"])
        if i==1:   return ("minterpolate=fps={fps}:mi_mode=blend", ["-preset","veryfast","-threads","0"])
        if i==2:   return ("minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1", ["-preset","fast","-threads","0"])
        if i==3:   return ("minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:me=umh", ["-preset","medium","-threads","0"])
        return ("minterpolate=fps={fps}:mi_mode=dup", ["-preset","ultrafast","-threads","0"])

    def _build_ffmpeg_cmd(self, src: Path, mult: float, in_fps: float, out_path: Path, progress_file: Path | None = None)->list[str]:
        vf_tmpl, preset = self._filters_for_speed()
        if mult >= 1.0:
            target = in_fps * mult
            vf = vf_tmpl.format(fps=f"{target:.3f}")
            base = ["ffmpeg","-hide_banner","-y"] + (["-progress", str(progress_file)] if progress_file else []) + ["-i", str(src)]
            return base + ["-vf", vf, "-c:a","copy","-c:v","libx264", *preset, "-crf","18", str(out_path)]
        else:
            vf = vf_tmpl.format(fps=f"{in_fps:.3f}") + f",setpts=PTS/({max(0.01,mult):.4f})"
            atempo = _atempo_chain(mult)
            base = ["ffmpeg","-hide_banner","-y"] + (["-progress", str(progress_file)] if progress_file else []) + ["-i", str(src)]
            return base + ["-vf", vf, "-af", atempo, "-c:v","libx264", *preset, "-crf","18", str(out_path)]

    def _enqueue_job(self, src: Path, mult: float, in_fps: float)->str|None:
        try:
            out_dir = _outputs_dir(self.ROOT)
            suffix = f"{mult:.2f}x".rstrip("0").rstrip(".")
            out_path = out_dir / f"{src.stem}_interp_{suffix}.mp4"
            progress_file = out_path.with_suffix(out_path.suffix + ".progress")
            ff = self._build_ffmpeg_cmd(src, mult, in_fps, out_path, progress_file)
            self._ffmpeg_progress_file = str(progress_file)
            # establish total frames for ETA
            in_frames = _count_frames_fast(src)
            if in_frames <= 0:
                in_frames = int(round(_probe_fps(src) * 60))  # fallback: 1 minute estimate
            # multiply when increasing FPS (approx)
            self._progress_total_frames = int(in_frames * (mult if mult >= 1.0 else 1.0))
            self._progress_start("Processing in Queue…")
            self._set_working()
            self._ffmpeg_timer.start()
            self._progress_started_at = time.time()
            args = {"ffmpeg_cmd": ff, "streaming": bool(self.cb_stream.isChecked()), "label": "rife", "title": "Rife"}
            fn = enqueue_tool_job("tools_ffmpeg", str(src), str(out_dir), args, priority=560)
            self._last_expected_output = str(out_path)
            return str(fn)
        except Exception as e:
            QMessageBox.critical(self, "Queue error", f"Could not enqueue job:\n{e}"); return None

    # ---- superfast NCNN (foreground) ----
    def _run_superfast(self, src: Path, mult: float, in_fps: float) -> None:
        if not self._extract_assets_if_needed(self): return
        exe = _ncnn_bin_dir(self.ROOT) / "rife-ncnn-vulkan.exe"
        if not exe.exists():
            QMessageBox.critical(self, "RIFE", "Engine not found after install."); return
        key = self.combo_model.currentData()
        mdir = _model_dir(self.ROOT, key)
        if not mdir.exists():
            QMessageBox.warning(self, "RIFE", f"Model '{key}' not found. Trying to install…")
            if not self._extract_assets_if_needed(self): return
            if not mdir.exists():
                QMessageBox.critical(self, "RIFE", f"Model '{key}' still missing at:\n{mdir}"); return

        out_dir = _outputs_dir(self.ROOT)
        suffix = f"{mult:.2f}x".rstrip("0").rstrip(".")
        out_path = out_dir / f"{src.stem}_interp_{suffix}.mp4"

        tmp = self.ROOT / "logs" / "rife" / f"ncnn_tmp_{int(time.time())}"
        frames_in = tmp / "in"; frames_mid = tmp / "mid"; frames_out = tmp / "out"
        tmp.mkdir(parents=True, exist_ok=True); frames_in.mkdir(parents=True, exist_ok=True); frames_out.mkdir(parents=True, exist_ok=True)

        # inline progress bar (no popup)
        self._progress_total_frames = 0
        self._progress_start("Interpolating (Superfast)…")

        # extract audio
        audio = tmp / "audio.m4a"
        p1 = QProcess(self); p1.start("ffmpeg", ["-hide_banner","-y","-i", str(src), "-vn","-acodec","copy", str(audio)]); p1.waitForFinished(-1)
        if getattr(self, '_cancel_flag', False): self._progress_end(); self._set_result_state('idle', 'Result: —'); shutil.rmtree(tmp, ignore_errors=True); return
        self._progress_set(3)

        # decode frames
        p2 = QProcess(self); p2.start("ffmpeg", ["-hide_banner","-y","-i", str(src), str(frames_in / "frame_%08d.png")]); p2.waitForFinished(-1)
        if getattr(self, '_cancel_flag', False): self._progress_end(); self._set_result_state('idle', 'Result: —'); shutil.rmtree(tmp, ignore_errors=True); return
        in_count = len(list(frames_in.glob('*.png')))
        self._progress_total_frames = in_count if mult < 1.0 else in_count * (2 if mult >= 1.0 else 1)
        self._progress_set(8, 0)

        def run_ncnn(input_dir: Path, output_dir: Path, n: int, start_pct: int, end_pct: int, expected_in: int) -> bool:
            args = [str(exe), "-i", str(input_dir), "-o", str(output_dir), "-n", str(n), "-m", str(mdir)]
            expected = self._expected_out_frames(n, expected_in)
            p = QProcess(self)
            p.setProcessChannelMode(QProcess.MergedChannels)
            p.start(args[0], args[1:])
            while p.state() != QProcess.NotRunning:
                if getattr(self, '_cancel_flag', False): p.kill(); p.waitForFinished(2000); return False
                produced = len(list(output_dir.glob("*.png")))
                frac = min(1.0, produced / max(1, expected))
                val = int(start_pct + frac * (end_pct - start_pct))
                self._progress_set(val, produced)
                QCoreApplication.processEvents()
                QThread.msleep(150)
            p.waitForFinished(-1)
            ok = any(output_dir.glob("*.png"))
            if not ok:
                alt = _model_dir(self.ROOT, "rife-v4.6")
                if mdir != alt and alt.exists():
                    alt_args = [str(exe), "-i", str(input_dir), "-o", str(output_dir), "-n", str(n), "-m", str(alt)]
                    p2 = QProcess(self); p2.setProcessChannelMode(QProcess.MergedChannels)
                    p2.start(alt_args[0], alt_args[1:])
                    while p2.state() != QProcess.NotRunning:
                        if getattr(self, '_cancel_flag', False): p2.kill(); p2.waitForFinished(2000); return False
                        produced = len(list(output_dir.glob("*.png")))
                        frac = min(1.0, produced / max(1, expected))
                        val = int(start_pct + frac * (end_pct - start_pct))
                        self._progress_set(val, produced); QCoreApplication.processEvents(); QThread.msleep(150)
                    p2.waitForFinished(-1)
                    ok = any(output_dir.glob("*.png"))
                    if ok: self._set_ok("Result: ready")
            return ok

        if mult >= 1.75:
            frames_mid.mkdir(parents=True, exist_ok=True)
            ok = run_ncnn(frames_in, frames_mid, 2, 10, 55, in_count) and run_ncnn(frames_mid, frames_out, 2, 55, 90, len(list(frames_mid.glob('*.png'))))
        elif mult >= 1.0:
            ok = run_ncnn(frames_in, frames_out, 2, 10, 90, in_count)
        else:
            factor = 4 if (1.0/mult) >= 3.0 else 2
            ok = run_ncnn(frames_in, frames_out, factor, 10, 90, in_count)

        if not ok:
            frames_out.mkdir(parents=True, exist_ok=True)
            ok = run_ncnn(frames_in, frames_out, 2, 10, 90, in_count)
        if not ok:
            shutil.rmtree(tmp, ignore_errors=True)
            self._progress_end()
            self._set_err("Result: error")
            QMessageBox.critical(self, "RIFE error", f"rife-ncnn-vulkan produced no frames.\nChecked: {mdir}")
            return
        if getattr(self, '_cancel_flag', False): self._progress_end(); self._set_result_state('idle', 'Result: —'); shutil.rmtree(tmp, ignore_errors=True); return
        self._progress_set(95, self._progress_total_frames)

        # encode
        if mult >= 1.0:
            out_fps = int(round(in_fps * mult))
            enc = ["-framerate", str(out_fps), "-i", str(frames_out / "%08d.png")]
            cmd = ["ffmpeg","-hide_banner","-y", *enc, "-i", str(audio), "-c:a","copy","-crf","18","-c:v","libx264","-pix_fmt","yuv420p", str(out_path)]
        else:
            enc = ["-framerate", str(int(round(in_fps))), "-i", str(frames_out / "%08d.png")]
            atempo = _atempo_chain(mult)
            cmd = ["ffmpeg","-hide_banner","-y", *enc, "-i", str(audio), "-af", atempo, "-c:v","libx264","-crf","18","-pix_fmt","yuv420p", str(out_path)]
        p4 = QProcess(self); p4.start(cmd[0], cmd[1:]); p4.waitForFinished(-1)
        if getattr(self, '_cancel_flag', False): self._progress_end(); self._set_result_state('idle', 'Result: —'); shutil.rmtree(tmp, ignore_errors=True); return
        self._progress_set(100, self._progress_total_frames)
        shutil.rmtree(tmp, ignore_errors=True)

        self._last_expected_output = str(out_path)
        self._progress_end()
        self._set_ok("Result: ready")
        self.settings.setValue("rife/last_output", str(out_path))
        self._refresh_recent()
        if self.cb_autoplay.isChecked():
            player = getattr(self.main,"video",None)
            if player and hasattr(player,"open"):
                player.open(str(out_path))

    # ---- actions ----

    # ---- lifecycle ----
    def closeEvent(self, event):
        # Always persist current settings on close so selections stick across restarts.
        try:
            self._persist_state()
        except Exception:
            pass
        try:
            super().closeEvent(event)
        except Exception:
            pass
    def _on_open_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(_outputs_dir(self.ROOT))))

    def _select_src(self)->Path|None:
        p=_guess_current_video(self.main)
        if p: return p
        dlg = QFileDialog(self, "Select input video"); dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Videos (*.mp4 *.mkv *.mov *.webm *.avi)")
        return Path(dlg.selectedFiles()[0]) if (dlg.exec() and dlg.selectedFiles()) else None

    def _on_start_clicked(self):
        self._set_working()
        self._persist_state()

        src=self._select_src()
        if not src: QMessageBox.warning(self, "No input", "Please pick a video (or open one in the player)."); return
        in_fps=_probe_fps(src)
        mult = max(0.15, min(4.0, self.slider.value()/100.0))

        # If user selected 'Superfast', run the foreground NCNN path with inline progress.
        if int(self.combo_speed.currentIndex()) == 4:
            self._run_superfast(src, mult, in_fps)
            return

        job = self._enqueue_job(src, mult, in_fps)
        if not job: return
        try: self._last_job_id = Path(job).stem.split("_",1)[0]
        except Exception: self._last_job_id=None
        try:
            if QueueSystem: QueueSystem(self.ROOT).nudge_pending()
        except Exception: pass
        QMessageBox.information(self, "Queued", "Interpolation added to Queue.")
        self._watch_timer.start()

    def _on_batch_clicked(self):
        self._persist_state()
        dlg = QFileDialog(self, "Select videos for batch"); dlg.setFileMode(QFileDialog.ExistingFiles)
        dlg.setNameFilter("Videos (*.mp4 *.mkv *.mov *.webm *.avi)")
        if not dlg.exec(): return
        files=[Path(p) for p in dlg.selectedFiles() if p]
        ok=0; cache={}
        mult=max(0.15, min(4.0, self.slider.value()/100.0))
        for src in files:
            if not src.exists(): continue
            in_fps = cache.get(str(src)) or _probe_fps(src); cache[str(src)]=in_fps
            if self._enqueue_job(src, mult, in_fps): ok+=1
        QMessageBox.information(self, "Add Batch", f"Queued {ok} item(s).")
        try:
            if QueueSystem: QueueSystem(self.ROOT).nudge_pending()
        except Exception: pass

    def _on_play_last_internal(self):
        self._set_ok('Result: opened')
        p = self._last_expected_output or str(self.settings.value("rife/last_output",""))
        if not p: QMessageBox.information(self, "Play last", "No recent result yet."); return
        player = getattr(self.main, "video", None)
        if player and hasattr(player, "open"): player.open(p); self.main.current_path = _P_INT(str(p)); refresh_info_now(p); return
        QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _on_play_last_external(self):
        self._set_ok('Result: opened (external)')
        p = self._last_expected_output or str(self.settings.value("rife/last_output",""))
        if not p: QMessageBox.information(self, "External player", "No recent result yet."); return
        QDesktopServices.openUrl(QUrl.fromLocalFile(p))

    def _on_slider_changed(self, v: int):
        mult = max(0.15, min(4.0, v/100.0))
        self.lbl_mult.setText(f"{mult:.2f}×")
        self._update_result_info()

    def _poll_ffmpeg_progress(self):
        pf = getattr(self, '_ffmpeg_progress_file', None)
        if not pf: return
        try:
            txt = Path(pf).read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return
        last = txt.strip().splitlines()
        if not last: return
        data = {}
        for line in last[-40:]:
            if '=' in line:
                k,v = line.strip().split('=',1); data[k]=v
        # Prefer 'frame' from ffmpeg progress
        done = 0
        if 'frame' in data and data['frame'].isdigit():
            done = int(data['frame'])
        elif 'out_time_ms' in data and data['out_time_ms'].isdigit():
            # fallback: estimate from time with input fps
            t_ms = int(data['out_time_ms']);
            fps = max(1.0, float(self._progress_total_frames) / max(1, self._in_video_frames_duration_ms)) if getattr(self, '_in_video_frames_duration_ms', 0) else 30.0
            done = int((t_ms/1000.0) * fps)
        self._progress_done_frames = max(self._progress_done_frames, done)
        total = max(1, int(self._progress_total_frames))
        pct = int(min(100, (self._progress_done_frames/total)*100))
        self._progress_set(pct, self._progress_done_frames)

    def _update_result_info(self):
        src=_guess_current_video(self.main)
        if not src: self.lbl_result.setText("Result: —"); return
        in_fps=_probe_fps(src); mult=max(0.15, min(4.0, self.slider.value()/100.0))
        if mult>=1.0:
            self.lbl_result.setText(f"Result: (same resolution) @ {int(round(in_fps*mult))} fps")
        else:
            slow=1.0/mult; self.lbl_result.setText(f"Result: (same resolution) @ {int(round(in_fps))} fps  (slow‑mo ×{slow:.2f})")

    def _check_job_done(self):
        jid = self._last_job_id
        if not jid:
            self._watch_timer.stop(); return
        try:
            d = jobs_dirs(); done = Path(d["done"])
            for j in sorted(done.glob(f"{jid}_*.json")):
                self._watch_timer.stop()
                outp = self._last_expected_output or ""
                # Mark finished properly
                self._ffmpeg_timer.stop()
                self._progress_end()
                self._set_ok("Result: ready")
                try:
                    pf = getattr(self, '_ffmpeg_progress_file', None)
                    if pf:
                        Path(pf).unlink(missing_ok=True)
                        self._ffmpeg_progress_file = None
                except Exception:
                    pass
                if outp:
                    player = getattr(self.main, "video", None)
                    if player and hasattr(player, "open"):
                        player.open(outp); self.settings.setValue("rife/last_output", outp); self._refresh_recent(); return
                try:
                    data = json.loads(Path(j).read_text(encoding="utf-8")); prod = data.get("produced","")
                    if prod:
                        player = getattr(self.main, "video", None)
                        if player and hasattr(player, "open"):
                            player.open(prod); self.settings.setValue("rife/last_output", prod); self._refresh_recent(); return
                except Exception:
                    pass
                break
        except Exception:
            self._watch_timer.stop(); return

    # ---- recent gallery ----
    def _refresh_recent(self):
        # Clear
        while self.recent_h.count():
            item = self.recent_h.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()

        out_dir = _outputs_dir(self.ROOT)
        vids = sorted(out_dir.glob("*_interp_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:7]
        thumbs_dir = self.ROOT / "logs" / "rife" / "thumbs"
        for v in vids:
            thumb = thumbs_dir / (v.stem + ".jpg")
            if not thumb.exists():
                _extract_first_frame(v, thumb)
            pm = QPixmap(str(thumb)) if thumb.exists() else QPixmap()
            lbl = QLabel()
            if not pm.isNull():
                pm2 = pm.scaled(QSize(180, 100), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(pm2)
            else:
                lbl.setText(v.stem)
            lbl.setToolTip(v.name)
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.mousePressEvent = (lambda p=v: (lambda evt: QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))) )()
            self.recent_h.addWidget(lbl)
        self.recent_h.addStretch(1)
