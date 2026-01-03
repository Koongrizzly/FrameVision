
# helpers/trim_tool.py — adaptive grid + draggable range bar + dim outside + no auto-regenerate
from __future__ import annotations
import os, subprocess
import sys
from pathlib import Path
from PySide6.QtCore import Qt, QEvent, QObject, QRect, Signal, QThread, Slot
from PySide6.QtGui import QPainter, QPixmap
from helpers.batch import BatchSelectDialog
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QSlider,
    QScrollArea, QWidget, QMessageBox, QSpinBox, QGridLayout, QGraphicsOpacityEffect,
    QDialog, QVBoxLayout, QListWidget, QFileDialog, QDialogButtonBox, QGroupBox, QRadioButton, QListWidgetItem,
)
# ---- paths ----
def ffmpeg_path():
    try:
        from helpers.framevision_app import ROOT  # type: ignore
    except Exception:
        ROOT = Path('.').resolve()
    candidates = [ROOT/'presets'/'bin'/('ffmpeg.exe' if os.name=='nt' else 'ffmpeg'), 'ffmpeg']
    for c in candidates:
        try:
            subprocess.check_output([str(c), '-version'], stderr=subprocess.STDOUT); return str(c)
        except Exception:
            continue
    return 'ffmpeg'

def ffprobe_path():
    try:
        from helpers.framevision_app import ROOT  # type: ignore
    except Exception:
        ROOT = Path('.').resolve()
    candidates = [ROOT/'presets'/'bin'/('ffprobe.exe' if os.name=='nt' else 'ffprobe'), 'ffprobe']
    for c in candidates:
        try:
            subprocess.check_output([str(c), '-version'], stderr=subprocess.STDOUT); return str(c)
        except Exception:
            continue
    return 'ffprobe'

# ---- time utils ----
def _fmt_ms(ms:int)->str:
    if ms < 0: ms = 0
    s = int(ms//1000); msec = int(ms%1000)
    h = s//3600; s = s%3600; m = s//60; s = s%60
    return f"{h:02d}:{m:02d}:{s:02d}.{msec:03d}"

def _parse_time(s:str)->int:
    s = (s or '').strip()
    if not s: return 0
    try:
        if s.isdigit(): return int(s)
        if '.' in s:
            main, frac = s.split('.', 1); frac = (frac + '000')[:3]
        else:
            main, frac = s, '000'
        parts = [int(p or 0) for p in main.split(':')]
        while len(parts) < 3: parts = [0] + parts
        h, m, sec = parts[-3], parts[-2], parts[-1]
        return ((h*60 + m)*60 + sec)*1000 + int(frac)
    except Exception:
        return 0

# ---- custom widgets ----
class RangeBar(QWidget):
    changed = Signal(int, int)  # start_ms, end_ms
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(28)
        self.setMouseTracking(True)
        self.duration_ms = 1000
        self.start_ms = 0
        self.end_ms = 1000
        self._dragging = None  # 'start'|'end'|None

    def set_range(self, duration_ms:int, start_ms:int, end_ms:int):
        self.duration_ms = max(1, int(duration_ms))
        self.start_ms = max(0, min(int(start_ms), self.duration_ms))
        self.end_ms = max(0, min(int(end_ms), self.duration_ms))
        if self.start_ms > self.end_ms:
            self.start_ms, self.end_ms = self.end_ms, self.start_ms
        self.update()

    def set_start(self, ms:int):
        self.start_ms = max(0, min(int(ms), self.duration_ms))
        if self.start_ms > self.end_ms: self.end_ms = self.start_ms
        self.update()
        self.changed.emit(self.start_ms, self.end_ms)

    def set_end(self, ms:int):
        self.end_ms = max(0, min(int(ms), self.duration_ms))
        if self.end_ms < self.start_ms: self.start_ms = self.end_ms
        self.update()
        self.changed.emit(self.start_ms, self.end_ms)

    def _x_to_ms(self, x:int)->int:
        w = max(1, self.width()-1)
        frac = max(0.0, min(1.0, x / w))
        return int(round(frac * self.duration_ms))

    def _ms_to_x(self, ms:int)->int:
        w = max(1, self.width()-1)
        return int(round(w * max(0.0, min(1.0, ms / max(1, self.duration_ms)))))

    def paintEvent(self, ev):
        p = QPainter(self)
        rect = self.rect().adjusted(6, 8, -6, -8)  # margins
        p.fillRect(rect, self.palette().mid())                # base
        sel_left = rect.left() + self._ms_to_x(self.start_ms) # selection
        sel_right = rect.left() + self._ms_to_x(self.end_ms)
        if sel_right < sel_left: sel_left, sel_right = sel_right, sel_left
        sel = rect.adjusted(sel_left - rect.left(), 0, -(rect.right() - sel_right), 0)
        p.fillRect(sel, self.palette().highlight())
        for ms in (self.start_ms, self.end_ms):               # handles
            x = rect.left() + self._ms_to_x(ms)
            p.fillRect(x-3, rect.top(), 6, rect.height(), self.palette().brightText())

    def mousePressEvent(self, e):
        rect = self.rect().adjusted(6, 8, -6, -8)
        x = e.position().toPoint().x()
        sx = rect.left() + self._ms_to_x(self.start_ms)
        ex = rect.left() + self._ms_to_x(self.end_ms)
        self._dragging = 'start' if abs(x - sx) <= abs(x - ex) else 'end'
        ms = self._x_to_ms(x - rect.left())
        if self._dragging == 'start': self.set_start(ms)
        else: self.set_end(ms)

    def mouseMoveEvent(self, e):
        if not self._dragging: return
        rect = self.rect().adjusted(6, 8, -6, -8)
        x = e.position().toPoint().x()
        ms = self._x_to_ms(x - rect.left())
        if self._dragging == 'start': self.set_start(ms)
        else: self.set_end(ms)

    def mouseReleaseEvent(self, e):
        self._dragging = None

class _ClickableLabel(QLabel):
    def __init__(self, t_ms:int, parent=None):
        super().__init__(parent); self.t_ms=int(t_ms); self._orig_pm=None
    def mousePressEvent(self, e):
        pane = self.parent()
        while pane and not hasattr(pane, '_on_thumb_click'):
            pane = pane.parent()
        if pane and hasattr(pane, '_on_thumb_click'):
            pane._on_thumb_click(self.t_ms, e.button())

class _ResizeWatcher(QObject):
    def __init__(self, pane): super().__init__(pane); self.pane=pane
    def eventFilter(self, obj, ev):
        if ev.type()==QEvent.Resize:
            self.pane._reflow_thumbs()
        return False


class _ThumbGenWorker(QObject):
    """Generate preview thumbnails in a background thread.

    Emits file paths; UI thread is responsible for loading QPixmap.
    """
    thumbReady = Signal(int, int, str)   # index, t_ms, filepath
    progress = Signal(int, int)         # done, total
    finished = Signal(bool)             # cancelled?
    error = Signal(str)

    def __init__(self, video_path:str, times:list[float], out_dir:str, ffmpeg_bin:str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.times = times
        self.out_dir = out_dir
        self.ffmpeg_bin = ffmpeg_bin
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @Slot()
    def run(self):
        total = len(self.times)
        done = 0
        out_base = Path(self.out_dir)
        out_base.mkdir(parents=True, exist_ok=True)

        for i, t in enumerate(self.times):
            if self._cancel:
                self.finished.emit(True)
                return

            t_ms = int(round(float(t) * 1000.0))
            outp = out_base / f"th_{t_ms:08d}.jpg"

            try:
                if not outp.exists():
                    subprocess.check_call(
                        [self.ffmpeg_bin, "-y", "-ss", f"{t:.3f}", "-i", str(self.video_path),
                         "-frames:v", "1", "-q:v", "4", str(outp)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                if outp.exists():
                    self.thumbReady.emit(i, t_ms, str(outp))
            except Exception as e:
                # Keep going; missing thumbs are not fatal.
                try:
                    self.error.emit(str(e))
                except Exception:
                    pass

            done += 1
            try:
                self.progress.emit(done, total)
            except Exception:
                pass

        self.finished.emit(False)

# ---- installer ----
def install_trim_tool(pane, section_widget):
    # Base controls
    pane.trim_mode = QComboBox(); pane.trim_mode.addItems(["Fast copy (keyframe)", "Precise re-encode"])  # type: ignore
    pane.trim_start = QLineEdit("00:00:00.000"); pane.trim_end = QLineEdit("")
    pane.btn_trim = QPushButton("Trim"); pane.btn_trim_batch = QPushButton("Batch…")
    pane.btn_trim_open_folder = QPushButton("View results")
    pane.btn_trim_open_folder.setToolTip("Open Trim results in Media Explorer.")
    lay = QFormLayout(); lay.addRow("Trim mode", pane.trim_mode); lay.addRow("Start", pane.trim_start); lay.addRow("End", pane.trim_end)
    row = QHBoxLayout(); row.addWidget(pane.btn_trim); row.addWidget(pane.btn_trim_batch); row.addWidget(pane.btn_trim_open_folder); lay.addRow(row)

    # Preview controls
    pane.btn_trim_preview = QPushButton("Generate preview")
    pane.btn_trim_preview.setToolTip("Generate / update the preview thumbnails and timeline for the loaded video. Use this after changing the thumbnail count or loading a new video.")
    pane.trim_thumbs_spin = QSpinBox(); pane.trim_thumbs_spin.setRange(6, 999); pane.trim_thumbs_spin.setValue(12); pane.trim_thumbs_spin.setSuffix(" thumbs")
    pane.trim_thumbs_spin.setToolTip("How many thumbnails to generate. Press \"Generate preview\" to rebuild. (don't use more thumbnails then there are frames)")
    pane.thumb_size = QSlider(Qt.Horizontal); pane.thumb_size.setRange(24, 140); pane.thumb_size.setValue(86)
    pane.thumb_size.setToolTip("Thumbnail display size only. This just scales the preview thumbnails visually; it does not regenerate them.")
    r2 = QHBoxLayout(); r2.addWidget(pane.btn_trim_preview); r2.addWidget(pane.trim_thumbs_spin); r2.addWidget(QLabel("Size")); r2.addWidget(pane.thumb_size)
    lay.addRow(r2)

    # Thumbnails: adaptive grid
    scroll = pane.trim_preview_area = QScrollArea(); scroll.setWidgetResizable(True)
    strip_host = QWidget(); grid = QGridLayout(strip_host); grid.setContentsMargins(6,6,6,6); grid.setHorizontalSpacing(6); grid.setVerticalSpacing(6)
    pane._thumb_grid = grid; pane._thumb_host = strip_host; pane._thumb_labels = []
    scroll.setWidget(strip_host); scroll.setFixedHeight( (pane.thumb_size.value()+24)*2 )
    lay.addRow(scroll)

    # Range bar + sliders
    pane.range_bar = RangeBar(); lay.addRow(QLabel("Trim range")); lay.addRow(pane.range_bar)
    pane.trim_start_slider = QSlider(Qt.Horizontal); pane.trim_end_slider = QSlider(Qt.Horizontal)
    pane.trim_start_slider.setRange(0, 1000); pane.trim_end_slider.setRange(0, 1000)
    rs = QHBoxLayout(); rs.addWidget(QLabel("Start")); rs.addWidget(pane.trim_start_slider); rs.addWidget(QLabel("End")); rs.addWidget(pane.trim_end_slider)
    lay.addRow(rs)

    section_widget.setContentLayout(lay)

    # ---- helpers ----
    def _duration_seconds(path)->float:
        try:
            out = subprocess.check_output([ffprobe_path(), "-v","error","-show_entries","format=duration","-of","default=nokey=1:noprint_wrappers=1", str(path)], stderr=subprocess.STDOUT)
            return float(out.decode().strip())
        except Exception:
            return 0.0

    def _apply_dimming():
        # dim thumbs with timestamps outside [start,end]
        try:
            s = _parse_time(pane.trim_start.text())
            e = _parse_time(pane.trim_end.text())
            for lab in pane._thumb_labels:
                t = getattr(lab, "t_ms", 0)
                dim = (t < s) or (t > e)
                eff = getattr(lab, "_dim_eff", None)
                if dim:
                    if eff is None:
                        eff = QGraphicsOpacityEffect(lab); eff.setOpacity(0.35); lab.setGraphicsEffect(eff); lab._dim_eff = eff
                    else:
                        eff.setOpacity(0.35)
                else:
                    if eff is not None:
                        lab.setGraphicsEffect(None); lab._dim_eff = None
        except Exception:
            pass

    
    def _build_thumbs():
        """Build/rebuild the preview thumbnail strip without blocking the UI."""
        # Ensure bookkeeping attrs exist
        if not hasattr(pane, "_thumb_gen_id"):
            pane._thumb_gen_id = 0
        if not hasattr(pane, "_thumb_thread"):
            pane._thumb_thread = None
        if not hasattr(pane, "_thumb_worker"):
            pane._thumb_worker = None

        # Cancel any previous run (won't interrupt a currently-running ffmpeg call,
        # but prevents further UI updates from the old run).
        try:
            w = getattr(pane, "_thumb_worker", None)
            if w is not None:
                try:
                    w.cancel()
                except Exception:
                    pass
            th = getattr(pane, "_thumb_thread", None)
            if th is not None:
                try:
                    th.quit()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            p = getattr(pane.main, "current_path", None)
            if not p:
                QMessageBox.information(pane, "Trim preview", "Open a video first (File ▶ Open).")
                return

            dur = _duration_seconds(p)
            N = int(pane.trim_thumbs_spin.value())

            # Clear existing labels from grid & list
            grid = pane._thumb_grid
            while grid.count():
                it = grid.takeAt(0)
                wdg = it.widget()
                if wdg:
                    wdg.setParent(None)
            pane._thumb_labels.clear()

            # Placeholder labels first (instant UI), then fill them asynchronously.
            times = [(dur * i) / (N - 1) if dur > 0 and N > 1 else 0 for i in range(N)]
            h = int(pane.thumb_size.value())
            for t in times:
                t_ms = int(round(t * 1000))
                lab = _ClickableLabel(t_ms)
                lab.t_ms = t_ms
                lab.setMinimumHeight(h)
                lab.setMaximumHeight(h)
                lab._orig_pm = None
                lab.setText(_fmt_ms(t_ms))
                pane._thumb_labels.append(lab)

            _reflow_thumbs()
            _apply_dimming()

            # Busy state
            btn = pane.btn_trim_preview
            btn.setEnabled(False)
            btn.setText(f"Generating… 0/{N}")
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
            except Exception:
                pass

            # New generation id (guards stale signal delivery)
            pane._thumb_gen_id = int(getattr(pane, "_thumb_gen_id", 0)) + 1
            gid = pane._thumb_gen_id

            # Cache thumbs per (video mtime + count)
            base = Path("./output/_temp/trim_preview")
            try:
                mtime = int(Path(p).stat().st_mtime)
            except Exception:
                mtime = 0
            tag = f"{Path(p).stem}_{mtime}_{N}"
            out_dir = base / tag

            worker = _ThumbGenWorker(str(p), times, str(out_dir), ffmpeg_path())
            thread = QThread(pane)
            worker.moveToThread(thread)

            pane._thumb_worker = worker
            pane._thumb_thread = thread

            def _ui_thumb_ready(i:int, t_ms:int, fp:str, gid=gid):
                if int(getattr(pane, "_thumb_gen_id", 0)) != gid:
                    return
                if i < 0 or i >= len(pane._thumb_labels):
                    return
                lab = pane._thumb_labels[i]
                lab.t_ms = int(t_ms)
                pm = QPixmap(fp)
                if pm and not pm.isNull():
                    lab._orig_pm = pm
                    hh = int(pane.thumb_size.value())
                    lab.setPixmap(pm.scaledToHeight(hh, Qt.SmoothTransformation))
                else:
                    lab.setText(_fmt_ms(int(t_ms)))

            def _ui_progress(done:int, total:int, gid=gid):
                if int(getattr(pane, "_thumb_gen_id", 0)) != gid:
                    return
                try:
                    pane.btn_trim_preview.setText(f"Generating… {done}/{total}")
                except Exception:
                    pass

            def _ui_finished(cancelled:bool, gid=gid):
                if int(getattr(pane, "_thumb_gen_id", 0)) != gid:
                    return
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass
                try:
                    pane.btn_trim_preview.setEnabled(True)
                    pane.btn_trim_preview.setText("Generate preview")
                except Exception:
                    pass
                _reflow_thumbs()
                _apply_dimming()
                try:
                    pane._thumb_worker = None
                    pane._thumb_thread = None
                except Exception:
                    pass

            worker.thumbReady.connect(_ui_thumb_ready)
            worker.progress.connect(_ui_progress)
            worker.finished.connect(_ui_finished)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.started.connect(worker.run)
            thread.start()

        except Exception:
            # Restore UI if something went wrong.
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
            try:
                pane.btn_trim_preview.setEnabled(True)
                pane.btn_trim_preview.setText("Generate preview")
            except Exception:
                pass

    def _rescale_thumbs():
        # rescale existing labels only (no ffmpeg)
        try:
            h = int(pane.thumb_size.value())
            for lab in pane._thumb_labels:
                lab.setMinimumHeight(h); lab.setMaximumHeight(h)
                pm = getattr(lab, "_orig_pm", None)
                if pm and not pm.isNull():
                    lab.setPixmap(pm.scaledToHeight(h, Qt.SmoothTransformation))
            _reflow_thumbs()
        except Exception:
            pass

    def _reflow_thumbs():
        # place labels in grid based on viewport width and current thumb size
        try:
            grid = pane._thumb_grid; labels = pane._thumb_labels
            if not labels: return
            h = int(pane.thumb_size.value())
            first = labels[0]
            # if we don't have a pixmap yet, assume 16:9
            w = first.pixmap().width() if (first.pixmap() and not first.pixmap().isNull()) else int(h*16/9)
            cell = w + grid.horizontalSpacing()
            vp_w = pane.trim_preview_area.viewport().width() - (grid.contentsMargins().left()+grid.contentsMargins().right())
            cols = max(1, vp_w // max(1, cell))
            while grid.count():
                it = grid.takeAt(0); wdg = it.widget()
                if wdg: wdg.setParent(None)
            for i,lab in enumerate(labels):
                r = i // cols; c = i % cols
                grid.addWidget(lab, r, c)
            pane._thumb_host.adjustSize()
        except Exception:
            pass

    def _load_preview():
        # set slider ranges and build thumbs
        try:
            p = getattr(pane.main, "current_path", None)
            if not p: QMessageBox.information(pane, "Trim preview", "Open a video first (File ▶ Open)."); return
            dur = _duration_seconds(p); rng = int(dur*1000) if dur>0 else 1000
            pane.trim_start_slider.setRange(0, rng); pane.trim_end_slider.setRange(0, rng)
            if not pane.trim_end.text().strip(): pane.trim_end.setText(_fmt_ms(rng))
            s_ms = _parse_time(pane.trim_start.text()); e_ms = _parse_time(pane.trim_end.text())
            pane.range_bar.set_range(rng, s_ms, e_ms)
            _build_thumbs()
        except Exception:
            pass

    # guards to avoid feedback loops
    pane._trim_syncing = False

    def _set_start(ms:int):
        if pane._trim_syncing: return
        pane._trim_syncing = True
        try:
            pane.trim_start.setText(_fmt_ms(int(ms)))
            pane.trim_start_slider.blockSignals(True); pane.trim_start_slider.setValue(int(ms)); pane.trim_start_slider.blockSignals(False)
            pane.range_bar.set_start(int(ms))
            _apply_dimming()
        finally:
            pane._trim_syncing = False

    def _set_end(ms:int):
        if pane._trim_syncing: return
        pane._trim_syncing = True
        try:
            pane.trim_end.setText(_fmt_ms(int(ms)))
            pane.trim_end_slider.blockSignals(True); pane.trim_end_slider.setValue(int(ms)); pane.trim_end_slider.blockSignals(False)
            pane.range_bar.set_end(int(ms))
            _apply_dimming()
        finally:
            pane._trim_syncing = False

    def _update_from_edits():
        _set_start(_parse_time(pane.trim_start.text()))
        _set_end(_parse_time(pane.trim_end.text()))

    def _cursor_ms()->int:
        return int(pane.trim_start_slider.value())

    def _on_thumb_click(t_ms:int, button):
        if button == Qt.RightButton: _set_end(t_ms)
        else: _set_start(t_ms)

    def _clear_end():
        pane.trim_end.setText("")
        pane.trim_end_slider.setValue(pane.trim_end_slider.maximum())

    def _on_bar_changed(s,e):
        if pane._trim_syncing: return
        pane._trim_syncing = True
        try:
            pane.trim_start.setText(_fmt_ms(int(s)))
            pane.trim_end.setText(_fmt_ms(int(e)))
            pane.trim_start_slider.blockSignals(True); pane.trim_start_slider.setValue(int(s)); pane.trim_start_slider.blockSignals(False)
            pane.trim_end_slider.blockSignals(True); pane.trim_end_slider.setValue(int(e)); pane.trim_end_slider.blockSignals(False)
            _apply_dimming()
        finally:
            pane._trim_syncing = False

    # expose for host
    pane._load_trim_preview = _load_preview
    pane._set_start = _set_start
    pane._set_end = _set_end
    pane._cursor_ms = _cursor_ms
    pane._on_thumb_click = _on_thumb_click
    pane._clear_end = _clear_end
    pane._reflow_thumbs = _reflow_thumbs

    def _open_trim_results_folder():
        """Open Trim tool results in Media Explorer (fallback: OS folder)."""
        # Default output folder
        try:
            fp = Path("./output/video/trims")
            try:
                fp.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        except Exception:
            fp = None

        # Prefer Media Explorer if available (FrameVision main window)
        main = None
        try:
            main = getattr(pane, "main", None)
        except Exception:
            main = None
        if main is None:
            try:
                main = pane.window() if hasattr(pane, "window") else None
            except Exception:
                main = None

        if main is not None and hasattr(main, "open_media_explorer_folder") and fp is not None:
            try:
                main.open_media_explorer_folder(str(fp), preset="videos", include_subfolders=False)
                return
            except TypeError:
                try:
                    main.open_media_explorer_folder(str(fp))
                    return
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: open OS file browser
        if fp is not None:
            try:
                if os.name == "nt":
                    os.startfile(str(fp))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(fp)])
                else:
                    subprocess.Popen(["xdg-open", str(fp)])
            except Exception:
                pass

    # connect
    def _on_generate_preview():
        # Generate preview thumbnails & timeline without blocking the UI.
        _load_preview()

    pane.btn_trim_preview.clicked.connect(_on_generate_preview)   # manual generate only
    # NO auto-regenerate on spin/size; size only rescales & reflows
    pane.thumb_size.valueChanged.connect(lambda _: _rescale_thumbs())
    pane.trim_preview_area.viewport().installEventFilter(_ResizeWatcher(pane))
    pane.trim_start_slider.valueChanged.connect(lambda v: _set_start(v))
    pane.trim_end_slider.valueChanged.connect(lambda v: _set_end(v))
    pane.trim_start.editingFinished.connect(_update_from_edits)
    pane.trim_end.editingFinished.connect(_update_from_edits)
    pane.range_bar.changed.connect(_on_bar_changed)


    # ---- batch dialog ----
    def _open_batch_dialog():
        try:
            files, conflict = BatchSelectDialog.pick(pane, title="Batch trim")
            if files is None:
                return
            # Convert None->[] if user accepted with empty list; treat as no-op
            selected = list(files or [])
            if not selected:
                QMessageBox.information(pane, "Batch trim", "No files selected.")
                return
            s_ms = _parse_time(pane.trim_start.text())
            e_ms = _parse_time(pane.trim_end.text())
            mode = "copy" if pane.trim_mode.currentIndex() == 0 else "reencode"

            handled = False
            try:
                if hasattr(pane, "enqueue_trim_batch") and callable(getattr(pane, "enqueue_trim_batch")):
                    pane.enqueue_trim_batch(selected, s_ms, e_ms, mode, conflict or "version"); handled = True
                elif hasattr(pane, "on_trim_batch") and callable(getattr(pane, "on_trim_batch")):
                    pane.on_trim_batch(selected, s_ms, e_ms, mode, conflict or "version"); handled = True
                elif hasattr(pane, "main") and hasattr(pane.main, "enqueue_job"):
                    for fp in selected:
                        job = {"tool":"trim","input":fp,"start_ms":int(s_ms),"end_ms":int(e_ms),"mode":mode,"conflict":(conflict or "version")}
                        try:
                            pane.main.enqueue_job(job)
                            handled = True
                        except Exception:
                            pass
            except Exception:
                pass

            if not handled:
                msg = f"Selected {len(selected)} file(s).\nMode: {mode}\nConflict: {conflict or 'version'}\nStart: {s_ms} ms\nEnd: {e_ms} ms"
                QMessageBox.information(pane, "Batch trim", msg)
        except Exception as e:
            try:
                QMessageBox.critical(pane, "Batch trim", str(e))
            except Exception:
                pass
    # wire up the Batch button
    try:
        pane.btn_trim_batch.clicked.connect(_open_batch_dialog)
    except Exception:
        pass

    # wire up the View results button
    try:
        pane.btn_trim_open_folder.clicked.connect(_open_trim_results_folder)
    except Exception:
        pass

    return section_widget
