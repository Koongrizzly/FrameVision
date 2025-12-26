
# helpers/trim_tool.py — adaptive grid + draggable range bar + dim outside + no auto-regenerate
from __future__ import annotations
import os, subprocess
from pathlib import Path
from PySide6.QtCore import Qt, QEvent, QObject, QRect, Signal, QThread, Slot
from PySide6.QtGui import QPainter, QPixmap
from helpers.batch import BatchSelectDialog
from PySide6.QtWidgets import (
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QSlider,
    QScrollArea, QWidget, QMessageBox, QSpinBox, QGridLayout, QGraphicsOpacityEffect
, QDialog, QVBoxLayout, QListWidget, QFileDialog, QDialogButtonBox, QGroupBox, QRadioButton, QListWidgetItem)

# ---- paths ----
# Finished trim jobs output folder (relative to app root)
DEFAULT_TRIM_OUTPUT_DIR = Path('./output/video/trims')

# Preview thumbnails are generated off the UI thread and scaled down at extraction time
PREVIEW_THUMB_MAX_H = 220

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


# ---- threaded preview generation ----
class TrimPreviewWorker(QObject):
    init_ready = Signal(int, int, object)   # run_id, duration_ms, times_ms(list[int])
    thumb_ready = Signal(int, int, str)     # run_id, t_ms, filepath
    done = Signal(int, bool, str)           # run_id, canceled, error_message

    def __init__(self, run_id:int, input_path:str, thumb_count:int, out_dir:Path, thumb_max_h:int):
        super().__init__()
        self.run_id = int(run_id)
        self.input_path = str(input_path)
        self.thumb_count = max(1, int(thumb_count))
        self.out_dir = Path(out_dir)
        self.thumb_max_h = max(64, int(thumb_max_h))
        self._cancel = False
        self._proc = None

    def cancel(self):
        self._cancel = True
        try:
            p = self._proc
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
        except Exception:
            pass

    def _duration_seconds(self, path:str) -> float:
        try:
            out = subprocess.check_output(
                [ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
                stderr=subprocess.STDOUT
            )
            return float(out.decode(errors="ignore").strip() or "0")
        except Exception:
            return 0.0

    @Slot()
    def run(self):
        err = ""
        canceled = False
        try:
            dur = self._duration_seconds(self.input_path)
            N = self.thumb_count
            # Build timeline sample points
            times = [(dur*i)/(N-1) if dur > 0 and N > 1 else 0.0 for i in range(N)]
            times_ms = [int(round(t*1000.0)) for t in times]
            duration_ms = int(round(dur*1000.0)) if dur > 0 else 1000
            self.out_dir.mkdir(parents=True, exist_ok=True)

            # Tell UI to create placeholders + configure sliders
            self.init_ready.emit(self.run_id, duration_ms, times_ms)

            ffm = ffmpeg_path()
            for t, t_ms in zip(times, times_ms):
                if self._cancel:
                    canceled = True
                    break
                outp = self.out_dir / f"th_{t_ms:08d}.jpg"
                cmd = [
                    str(ffm), "-nostdin", "-y",
                    "-ss", f"{t:.3f}",
                    "-i", str(self.input_path),
                    "-frames:v", "1",
                    "-vf", f"scale=-2:{self.thumb_max_h}",
                    "-q:v", "4",
                    str(outp)
                ]
                try:
                    self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    rc = self._proc.wait()
                finally:
                    self._proc = None
                if self._cancel:
                    canceled = True
                    break
                if rc == 0 and outp.exists():
                    self.thumb_ready.emit(self.run_id, int(t_ms), str(outp))
        except Exception as e:
            err = str(e)
        self.done.emit(self.run_id, bool(canceled), err)

# ---- installer ----
def install_trim_tool(pane, section_widget):
    # Default output folder for finished trims
    pane.trim_output_dir = str(DEFAULT_TRIM_OUTPUT_DIR)
    # Base controls
    pane.trim_mode = QComboBox(); pane.trim_mode.addItems(["Fast copy (keyframe)", "Precise re-encode"])  # type: ignore
    pane.trim_start = QLineEdit("00:00:00.000"); pane.trim_end = QLineEdit("")
    pane.btn_trim = QPushButton("Trim"); pane.btn_trim_batch = QPushButton("Batch…")
    lay = QFormLayout(); lay.addRow("Trim mode", pane.trim_mode); lay.addRow("Start", pane.trim_start); lay.addRow("End", pane.trim_end)
    row = QHBoxLayout(); row.addWidget(pane.btn_trim); row.addWidget(pane.btn_trim_batch); lay.addRow(row)

    # Preview controls
    pane.btn_trim_preview = QPushButton("Generate preview")
    pane.btn_trim_preview.setToolTip("Generate / update the preview thumbnails and timeline for the loaded video. Use this after changing the thumbnail count or loading a new video.")
    pane.trim_thumbs_spin = QSpinBox(); pane.trim_thumbs_spin.setRange(6, 60); pane.trim_thumbs_spin.setValue(12); pane.trim_thumbs_spin.setSuffix(" thumbs")
    pane.trim_thumbs_spin.setToolTip("How many thumbnails to generate for the preview strip. After changing this number, press \"Generate preview\" to rebuild.")
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

    # --- threaded preview generation (keeps UI responsive) ---
    pane._trim_preview_run_id = 0
    pane._trim_preview_thread = None
    pane._trim_preview_worker = None
    pane._thumb_by_t = {}

    def _set_preview_busy(busy:bool):
        try:
            pane.btn_trim_preview.setEnabled(not busy)
            pane.trim_thumbs_spin.setEnabled(not busy)
            pane.btn_trim_preview.setText("Generating preview…" if busy else "Generate preview")
        except Exception:
            pass

    def _cancel_preview_worker():
        try:
            w = getattr(pane, "_trim_preview_worker", None)
            if w is not None:
                try:
                    w.cancel()
                except Exception:
                    pass
        except Exception:
            pass

    def _start_preview_worker():
        try:
            p = getattr(pane.main, "current_path", None)
            if not p:
                QMessageBox.information(pane, "Trim preview", "Open a video first (File ▶ Open).")
                return

            # Bump run id so late signals from old runs are ignored
            pane._trim_preview_run_id = int(getattr(pane, "_trim_preview_run_id", 0)) + 1
            run_id = int(pane._trim_preview_run_id)

            # Cancel previous run (best effort)
            _cancel_preview_worker()

            out_dir = Path("./output/_temp/trim_preview") / f"run_{run_id:04d}"
            worker = TrimPreviewWorker(run_id, str(p), int(pane.trim_thumbs_spin.value()), out_dir, PREVIEW_THUMB_MAX_H)
            thread = QThread(pane)
            worker.moveToThread(thread)

            def _on_init(rid:int, duration_ms:int, times_ms_obj):
                if rid != pane._trim_preview_run_id:
                    return
                try:
                    times_ms = list(times_ms_obj or [])
                except Exception:
                    times_ms = []

                duration_ms = int(duration_ms) if int(duration_ms) > 0 else 1000

                # configure sliders and range
                pane.trim_start_slider.setRange(0, duration_ms)
                pane.trim_end_slider.setRange(0, duration_ms)
                if not pane.trim_end.text().strip():
                    pane.trim_end.setText(_fmt_ms(duration_ms))

                s_ms = _parse_time(pane.trim_start.text())
                e_ms = _parse_time(pane.trim_end.text())
                s_ms = max(0, min(int(s_ms), duration_ms))
                e_ms = max(0, min(int(e_ms), duration_ms))

                pane.range_bar.set_range(duration_ms, s_ms, e_ms)

                # clear existing labels from grid & list
                grid = pane._thumb_grid
                while grid.count():
                    it = grid.takeAt(0)
                    w = it.widget()
                    if w:
                        w.setParent(None)

                pane._thumb_labels.clear()
                pane._thumb_by_t = {}
                pane._preview_first_pix = False

                h = int(pane.thumb_size.value())
                for t_ms in times_ms:
                    t_ms = int(t_ms)
                    lab = _ClickableLabel(t_ms)
                    lab.t_ms = t_ms
                    lab._orig_pm = None
                    lab.setMinimumHeight(h)
                    lab.setMaximumHeight(h)
                    lab.setText(_fmt_ms(t_ms))
                    pane._thumb_labels.append(lab)
                    pane._thumb_by_t[t_ms] = lab

                _reflow_thumbs()
                _apply_dimming()

            def _on_thumb(rid:int, t_ms:int, fp:str):
                if rid != pane._trim_preview_run_id:
                    return
                try:
                    lab = getattr(pane, "_thumb_by_t", {}).get(int(t_ms))
                    if not lab:
                        return
                    pm = QPixmap(str(fp))
                    lab._orig_pm = pm if not pm.isNull() else None
                    h = int(pane.thumb_size.value())
                    if lab._orig_pm:
                        lab.setPixmap(lab._orig_pm.scaledToHeight(h, Qt.SmoothTransformation))
                        # reflow once when the first pixmap arrives to improve column calc
                        if not getattr(pane, "_preview_first_pix", False):
                            pane._preview_first_pix = True
                            _reflow_thumbs()
                    else:
                        lab.setText(_fmt_ms(int(t_ms)))
                except Exception:
                    pass

            def _on_done(rid:int, canceled:bool, err_msg:str):
                # Always stop the thread for this worker
                try:
                    thread.quit()
                except Exception:
                    pass

                if rid != pane._trim_preview_run_id:
                    return

                _set_preview_busy(False)
                pane._trim_preview_thread = None
                pane._trim_preview_worker = None

                try:
                    if err_msg:
                        QMessageBox.warning(pane, "Trim preview", err_msg)
                except Exception:
                    pass

                _reflow_thumbs()
                _apply_dimming()

            worker.init_ready.connect(_on_init)
            worker.thumb_ready.connect(_on_thumb)
            worker.done.connect(_on_done)

            thread.started.connect(worker.run)
            thread.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)

            pane._trim_preview_thread = thread
            pane._trim_preview_worker = worker

            _set_preview_busy(True)
            thread.start()
        except Exception:
            _set_preview_busy(False)

    def _load_preview():
        # Start preview generation (runs in background thread)
        _start_preview_worker()

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

    # connect
    pane.btn_trim_preview.clicked.connect(_load_preview)   # manual generate only
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
                        job = {"tool":"trim","input":fp,"start_ms":int(s_ms),"end_ms":int(e_ms),"mode":mode,"conflict":(conflict or "version"),"output_dir": str(DEFAULT_TRIM_OUTPUT_DIR), "out_dir": str(DEFAULT_TRIM_OUTPUT_DIR)}
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

    return section_widget
