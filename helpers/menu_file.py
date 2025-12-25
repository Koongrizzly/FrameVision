# helpers/menu_file.py
import os, shutil, json
from pathlib import Path

try:
    from helpers.framevision_app import ROOT
except Exception:
    ROOT = Path(__file__).resolve().parents[1]


from PySide6.QtCore import QEvent, QObject, QProcess, QSettings, QUrl, Qt, QPoint
from PySide6.QtGui import QAction, QIcon, QCursor
from PySide6.QtWidgets import QMenuBar, QMenu, QFileDialog, QMessageBox, QWidget, QPushButton, QApplication, QInputDialog, QDialog, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QSlider, QCheckBox, QLineEdit, QDialogButtonBox
from PySide6.QtGui import QDesktopServices

# --- Right-clickable menu subclass -----------------------------------------
class _RemovableMenu(QMenu):
    def __init__(self, title, parent=None, remove_fn=None, repopulate_cb=None):
        super().__init__(title, parent)
        self._remove_fn = remove_fn
        self._repopulate_cb = repopulate_cb
        self._last_hovered = None
        try:
            self.hovered.connect(self._on_hovered)
        except Exception:
            pass

    def _on_hovered(self, action):
        self._last_hovered = action

    def _action_at_pos(self, pos):
        act = None
        try:
            act = self.actionAt(pos)
        except Exception:
            act = None
        if not act:
            act = self._last_hovered
        return act

    def _show_remove_menu(self, global_pos, path:str):
        ctx = QMenu(self)
        rm = QAction("Remove from list", ctx)
        def _do_remove():
            try:
                if self._remove_fn: self._remove_fn(path)
            finally:
                try:
                    if self._repopulate_cb: self._repopulate_cb()
                except Exception:
                    pass
        rm.triggered.connect(_do_remove)
        ctx.addAction(rm)
        if hasattr(ctx, "exec"):
            ctx.exec(global_pos)
        else:
            ctx.popup(global_pos)

    def contextMenuEvent(self, event):
        try:
            act = self._action_at_pos(event.pos())
            if act and hasattr(act, "data") and act.data():
                self._show_remove_menu(event.globalPos(), str(act.data()))
                event.accept()
                return
        except Exception:
            pass
        try:
            super().contextMenuEvent(event)
        except Exception:
            pass

    def mousePressEvent(self, event):
        try:
            if hasattr(event, "button") and event.button() == Qt.RightButton:
                act = self._action_at_pos(event.pos())
                if act and hasattr(act, "data") and act.data():
                    self._show_remove_menu(self.mapToGlobal(event.pos()), str(act.data()))
                    event.accept()
                    return
        except Exception:
            pass
        try:
            super().mousePressEvent(event)
        except Exception:
            pass

ORG="FrameVision"; APP="FrameVision"

# ------------------------------- Utility helpers -------------------------------

def _find_button(root: QWidget, prefixes: tuple[str,...]) -> QPushButton | None:
    if not root: return None
    for b in root.findChildren(QPushButton) or []:
        t=(b.text() or "").lower()
        if any(t.startswith(p) for p in prefixes):
            return b
    return None

def _current_input_path(main_window) -> str | None:
    # Prefer the new attribute name used by the app
    for attr in ("current_path","current_media_path","current_input_path","input_path","current_file"):
        if hasattr(main_window, attr):
            p = getattr(main_window, attr)
            try:
                # Accept either str or pathlib.Path
                if isinstance(p, (str, Path)) and p:
                    p2 = str(p)
                    if os.path.isfile(p2):
                        return p2
            except Exception:
                pass
    try:
        s = QSettings(ORG, APP)
        p = s.value("last_open_file","") or s.value("last_file","")
        if p:
            return p
    except Exception:
        pass
    return None

def _set_current_input_path(main_window, path: str):
    try:
        setattr(main_window, "current_path", Path(path))
    except Exception:
        pass
    try:
        s=QSettings(ORG,APP); s.setValue("last_open_file", path)
    except Exception:
        pass

def _grab_visual_frame(widget: QWidget):
    try:
        pm=widget.grab()
        if not pm.isNull(): return pm.toImage()
    except Exception:
        pass
    return None

def _find_video_widget(root: QWidget) -> QWidget | None:
    if not root: return None
    for w in root.findChildren(QWidget) or []:
        name=(w.objectName() or "").lower()
        if any(k in name for k in ("video","player","view")):
            return w
    # fallback: biggest widget on left half
    biggest=None; area=0
    try:
        for w in root.findChildren(QWidget) or []:
            g=w.geometry(); a=g.width()*g.height()
            if a>area and g.left() < root.width()*0.5:
                area=a; biggest=w
    except Exception:
        pass
    return biggest

def _presets_bin_dir() -> Path:
    """Return the bundled tools directory: (root)/presets/bin/"""
    try:
        return Path(ROOT) / "presets" / "bin"
    except Exception:
        return Path(__file__).resolve().parents[1] / "presets" / "bin"

def _tool_in_presets_bin(candidates: tuple[str, ...]) -> str | None:
    try:
        b = _presets_bin_dir()
        for cand in candidates:
            p = b / cand
            if p.is_file():
                return str(p)
    except Exception:
        pass
    return None

def _ffmpeg_exe() -> str | None:
    # 0) bundled ffmpeg in (root)/presets/bin/
    p = _tool_in_presets_bin(("ffmpeg.exe", "ffmpeg"))
    if p:
        return p

    # 1) user override
    try:
        s = QSettings(ORG, APP)
        v = s.value("ffmpeg_path", "") or ""
        if v:
            v = str(v)
            if os.path.isdir(v):
                for cand in ("ffmpeg.exe", "ffmpeg"):
                    pp = os.path.join(v, cand)
                    if os.path.isfile(pp):
                        return pp
            if os.path.isfile(v):
                return v
    except Exception:
        pass

    # 2) PATH
    for cand in ("ffmpeg.exe", "ffmpeg"):
        ff = shutil.which(cand)
        if ff:
            return ff
    return None

def _ffprobe_exe() -> str | None:
    # 0) bundled ffprobe in (root)/presets/bin/
    p = _tool_in_presets_bin(("ffprobe.exe", "ffprobe"))
    if p:
        return p

    # 1) user override (optional)
    try:
        s = QSettings(ORG, APP)
        v = s.value("ffprobe_path", "") or ""
        if v:
            v = str(v)
            if os.path.isdir(v):
                for cand in ("ffprobe.exe", "ffprobe"):
                    pp = os.path.join(v, cand)
                    if os.path.isfile(pp):
                        return pp
            if os.path.isfile(v):
                return v
    except Exception:
        pass

    # 2) PATH
    for cand in ("ffprobe.exe", "ffprobe"):
        fp = shutil.which(cand)
        if fp:
            return fp
    return None

def _icon_for_path(p: str) -> QIcon:
    try:
        return QIcon(p)
    except Exception:
        return QIcon()

# ------------------------------- Recent & Favorites -------------------------------

RECENT_KEY="recent_files"
FAV_KEY="favorite_files"
LAST_CLOSED_KEY="last_closed_file"
RECENT_LIMIT=10

def _load_list(key: str) -> list[str]:
    try:
        s=QSettings(ORG,APP); v=s.value(key, [])
        if isinstance(v, list): return [str(x) for x in v if x]
        if isinstance(v, str):
            try: return [x for x in json.loads(v) if x]
            except Exception: return [x for x in v.split("|") if x]
    except Exception:
        pass
    return []

def _save_list(key: str, items: list[str]):
    try:
        s=QSettings(ORG,APP); s.setValue(key, items)
    except Exception:
        pass

def _remember_recent(path: str):
    if not path: return
    rec=_load_list(RECENT_KEY)
    path=str(Path(path))
    if path in rec: rec.remove(path)
    rec.insert(0, path)
    if len(rec)>RECENT_LIMIT: rec=rec[:RECENT_LIMIT]
    _save_list(RECENT_KEY, rec)

def _remember_favorite(path: str):
    fav=_load_list(FAV_KEY); path=str(Path(path))
    if path not in fav:
        fav.append(path)
        _save_list(FAV_KEY, fav)

def _remove_favorite(path: str):
    fav=_load_list(FAV_KEY); path=str(Path(path))
    if path in fav:
        fav.remove(path); _save_list(FAV_KEY, fav)


def _remove_recent(path: str):
    rec = _load_list(RECENT_KEY); path = str(Path(path))
    if path in rec:
        rec.remove(path)
        _save_list(RECENT_KEY, rec)

def _set_last_closed(path: str):
    try:
        s=QSettings(ORG,APP); s.setValue(LAST_CLOSED_KEY, path)
    except Exception:
        pass

def _get_last_closed() -> str | None:
    try:
        s=QSettings(ORG,APP); v=s.value(LAST_CLOSED_KEY, "")
        return str(v) if v else None
    except Exception:
        return None

# ------------------------------- Save/Convert helpers -------------------------------

def _save_frame_dialog(main_window):
    vw=_find_video_widget(main_window or QApplication.activeWindow())
    if not vw:
        QMessageBox.information(main_window,"Save Frame","No video/photo widget found.")
        return
    img=_grab_visual_frame(vw)
    if img is None:
        btn=_find_button(main_window, ("screenshot","screen"))
        if btn:
            btn.click()
            QMessageBox.information(main_window,"Save Frame","Used the existing Screenshot feature.")
        else:
            QMessageBox.information(main_window,"Save Frame","No frame available.")
        return
    path, sel = QFileDialog.getSaveFileName(main_window,"Save Frame As…","frame.png","PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
    if not path: return
    fmt = "PNG" if path.lower().endswith("png") else "JPG"
    quality = 95 if fmt=="JPG" else -1
    ok = img.save(path, fmt, quality)
    if not ok:
        QMessageBox.warning(main_window,"Save Frame","Failed to save the image.")

def _save_media_dialog(main_window):
    """Unified 'Save As…' for both images and videos."""
    src = _current_input_path(main_window)
    if not src or not os.path.isfile(src):
        QMessageBox.information(main_window, "Save As", "No input file loaded.")
        return

    src_ext = Path(src).suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

    if src_ext in image_exts:
        # Save/convert still images using Qt
        default_name = os.path.splitext(os.path.basename(src))[0] + src_ext
        filters = "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;BMP Image (*.bmp);;WEBP Image (*.webp);;TIFF Image (*.tif *.tiff)"
        path, _ = QFileDialog.getSaveFileName(main_window, "Save As…", default_name, filters)
        if not path:
            return
        try:
            from PySide6.QtGui import QImage
            img = QImage(src)
            if img.isNull():
                shutil.copy2(src, path)
                return
            suffix = Path(path).suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                ok = img.save(path, "JPG", 95)
            elif suffix in (".tif", ".tiff"):
                ok = img.save(path, "TIFF")
            elif suffix == ".webp":
                ok = img.save(path, "WEBP")
            elif suffix == ".bmp":
                ok = img.save(path, "BMP")
            else:
                ok = img.save(path, "PNG")
            if not ok:
                QMessageBox.warning(main_window, "Save As", "Failed to save the image.")
            return
        except Exception as e:
            try:
                shutil.copy2(src, path)
                return
            except Exception:
                QMessageBox.warning(main_window, "Save As", f"Failed to save the image: {e}")
                return

    # Otherwise treat as video – use ffmpeg
    default_name = os.path.splitext(os.path.basename(src))[0] + ".mp4"
    filters = "MP4 Video (*.mp4);;Matroska Video (*.mkv);;MOV (*.mov);;WebM (*.webm)"
    path, _ = QFileDialog.getSaveFileName(main_window, "Save As…", default_name, filters)
    if not path:
        return
    ff = _ffmpeg_exe()
    if not ff:
        QMessageBox.information(main_window, "Save As", "FFmpeg not found. Put ffmpeg.exe in presets/bin, add it to PATH, or set 'ffmpeg_path' in settings.")
        return
    ext = Path(path).suffix.lower()
    # Try stream copy when saving to MP4/MKV/MOV/WEBM; otherwise re-encode to H.264 + AAC for compatibility
    if ext in (".mp4", ".mkv", ".mov", ".webm"):
        args = [ff, "-y", "-i", src, "-map", "0", "-c", "copy", path]
    else:
        args = [ff, "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", path]
    proc = QProcess(main_window)
    proc.startDetached(args[0], args[1:])


def _probe_audio_bitrate_kbps(src: str) -> int | None:
    """Best-effort bitrate probe for the first audio stream."""
    try:
        fp = _ffprobe_exe()
        if fp:
            import subprocess
            # Try stream bitrate first
            r = subprocess.run(
                [fp, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=bit_rate", "-of", "default=nw=1:nk=1", src],
                capture_output=True, text=True
            )
            if r.returncode == 0:
                v = (r.stdout or "").strip()
                if v.isdigit():
                    bps = int(v)
                    if bps > 0:
                        return max(1, int(round(bps / 1000.0)))
            # Fallback: container/format bitrate
            r = subprocess.run(
                [fp, "-v", "error",
                 "-show_entries", "format=bit_rate", "-of", "default=nw=1:nk=1", src],
                capture_output=True, text=True
            )
            if r.returncode == 0:
                v = (r.stdout or "").strip()
                if v.isdigit():
                    bps = int(v)
                    if bps > 0:
                        return max(1, int(round(bps / 1000.0)))
    except Exception:
        pass
    return None

def _save_audio_mp3_dialog(main_window):
    src = _current_input_path(main_window)
    if not src or not os.path.isfile(src):
        QMessageBox.information(main_window, "Save As MP3", "No input file loaded.")
        return

    ff = _ffmpeg_exe()
    if not ff:
        QMessageBox.information(
            main_window,
            "Save As MP3",
            "FFmpeg not found. Put ffmpeg.exe in presets/bin, add it to PATH, or set 'ffmpeg_path' in settings."
        )
        return

    s = QSettings(ORG, APP)
    last_dir = str(s.value("mp3_last_save_dir", "")) or str(Path(src).parent)
    default_out = str(Path(last_dir) / (Path(src).stem + ".mp3"))

    # ---- Dialog UI ---------------------------------------------------------
    dlg = QDialog(main_window)
    dlg.setWindowTitle("Save As MP3")
    dlg.setModal(True)

    root = QVBoxLayout(dlg)

    # Quality
    row_q = QHBoxLayout()
    row_q.addWidget(QLabel("Quality:"))
    cmb = QComboBox(dlg)
    bitrates = [64, 96, 128, 160, 192, 256, 320]
    for b in bitrates:
        cmb.addItem(f"{b} kbps", b)
    # Default to 192 kbps
    try:
        cmb.setCurrentIndex(bitrates.index(192))
    except Exception:
        pass
    row_q.addWidget(cmb, 1)
    root.addLayout(row_q)

    # Volume
    row_v = QHBoxLayout()
    lab_v = QLabel("Volume:")
    row_v.addWidget(lab_v)
    slider = QSlider(Qt.Horizontal, dlg)
    slider.setMinimum(-12)
    slider.setMaximum(12)
    slider.setSingleStep(1)
    slider.setValue(0)
    row_v.addWidget(slider, 1)
    lab_vv = QLabel("0 dB")
    lab_vv.setMinimumWidth(60)
    row_v.addWidget(lab_vv)
    root.addLayout(row_v)

    def _update_vol_label(v: int):
        if v == 0:
            lab_vv.setText("0 dB")
        elif v > 0:
            lab_vv.setText(f"+{v} dB")
        else:
            lab_vv.setText(f"{v} dB")

    slider.valueChanged.connect(_update_vol_label)

    # Use source
    chk = QCheckBox("Use source (auto-pick closest bitrate)", dlg)
    root.addWidget(chk)

    def _apply_source_bitrate():
        kbps = _probe_audio_bitrate_kbps(src)
        if not kbps:
            return
        nearest = min(bitrates, key=lambda x: abs(x - kbps))
        try:
            cmb.setCurrentIndex(bitrates.index(nearest))
        except Exception:
            pass

    chk.toggled.connect(lambda on: _apply_source_bitrate() if on else None)

    # Output path + Save As...
    row_o = QHBoxLayout()
    row_o.addWidget(QLabel("Output:"))
    out_edit = QLineEdit(default_out, dlg)
    out_edit.setReadOnly(True)
    row_o.addWidget(out_edit, 1)
    btn_saveas = QPushButton("Save As…", dlg)
    row_o.addWidget(btn_saveas)
    root.addLayout(row_o)

    def _pick_output():
        nonlocal last_dir
        start = str(Path(last_dir) / (Path(out_edit.text()).name or (Path(src).stem + ".mp3")))
        p, _ = QFileDialog.getSaveFileName(dlg, "Save As MP3…", start, "MP3 Audio (*.mp3)")
        if not p:
            return
        if not p.lower().endswith(".mp3"):
            p += ".mp3"
        out_edit.setText(p)
        try:
            last_dir = str(Path(p).parent)
            s.setValue("mp3_last_save_dir", last_dir)
        except Exception:
            pass

    btn_saveas.clicked.connect(_pick_output)

    # Buttons
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
    buttons.button(QDialogButtonBox.Ok).setText("Convert")
    buttons.rejected.connect(dlg.reject)
    root.addWidget(buttons)

    def _do_convert():
        outp = out_edit.text().strip()
        if not outp:
            QMessageBox.information(dlg, "Save As MP3", "Choose an output file first.")
            return
        bitrate = cmb.currentData()
        try:
            bitrate = int(bitrate)
        except Exception:
            bitrate = 192

        vol_db = int(slider.value())
        args = [ff, "-y", "-i", src, "-vn", "-c:a", "libmp3lame", "-b:a", f"{bitrate}k"]
        if vol_db != 0:
            # Use dB so 0 is a true midpoint
            args += ["-filter:a", f"volume={vol_db}dB"]
        args += [outp]

        try:
            # Remember folder even if user didn't press Save As...
            try:
                s.setValue("mp3_last_save_dir", str(Path(outp).parent))
            except Exception:
                pass
            QProcess(main_window).startDetached(args[0], args[1:])
            dlg.accept()
        except Exception:
            QMessageBox.warning(dlg, "Save As MP3", "Failed to start FFmpeg.")

    buttons.accepted.connect(_do_convert)

    dlg.exec()

# ------------------------------- Batch operations -------------------------------

VIDEO_EXTS=(".mp4",".mkv",".mov",".avi",".webm",".m4v",".mpg",".mpeg",".mts",".m2ts",".flv")
AUDIO_EXTS=(".mp3",".aac",".wav",".flac",".m4a",".ogg",".opus")
IMAGE_EXTS=(".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".gif")

def _choose_dir(title: str, parent) -> str | None:
    d=QFileDialog.getExistingDirectory(parent, title)
    return d if d else None

def _batch_convert_to_mp3(main_window):
    inf=_choose_dir("Select input folder (videos/audio)", main_window)
    if not inf: return
    outf=_choose_dir("Select output folder for MP3s", main_window)
    if not outf: return
    ff=_ffmpeg_exe()
    if not ff:
        QMessageBox.information(main_window,"Batch MP3","FFmpeg not found. Put ffmpeg.exe in presets/bin, add it to PATH, or set 'ffmpeg_path' in settings."); return
    count=0
    for root,_,files in os.walk(inf):
        for f in files:
            if not f.lower().endswith(VIDEO_EXTS + AUDIO_EXTS): continue
            src=os.path.join(root,f)
            base=os.path.splitext(os.path.basename(src))[0]+".mp3"
            dst=os.path.join(outf, base)
            args=[ff,"-y","-i",src,"-vn","-acodec","libmp3lame","-b:a","192k",dst]
            QProcess(main_window).startDetached(args[0], args[1:]); count+=1
    QMessageBox.information(main_window,"Batch MP3", f"Started {count} MP3 conversions.")

def _batch_extract_audio(main_window):
    # Supports WAV/AAC/FLAC selection
    fmt, ok = QInputDialog.getItem(main_window, "Audio format", "Choose output format:", ["wav","aac","flac"], 0, False)
    if not ok: return
    inf=_choose_dir("Select input folder (videos)", main_window); 
    if not inf: return
    outf=_choose_dir("Select output folder for audio", main_window); 
    if not outf: return
    ff=_ffmpeg_exe()
    if not ff:
        QMessageBox.information(main_window,"Batch Extract","FFmpeg not found. Put ffmpeg.exe in presets/bin, add it to PATH, or set 'ffmpeg_path' in settings."); return
    count=0
    for root,_,files in os.walk(inf):
        for f in files:
            if not f.lower().endswith(VIDEO_EXTS): continue
            src=os.path.join(root,f)
            base=os.path.splitext(os.path.basename(src))[0]+"."+fmt
            dst=os.path.join(outf, base)
            if fmt=="wav":
                args=[ff,"-y","-i",src,"-vn","-acodec","pcm_s16le","-ar","44100","-ac","2",dst]
            elif fmt=="aac":
                args=[ff,"-y","-i",src,"-vn","-c:a","aac","-b:a","192k",dst]
            else: # flac
                args=[ff,"-y","-i",src,"-vn","-c:a","flac",dst]
            QProcess(main_window).startDetached(args[0], args[1:]); count+=1
    QMessageBox.information(main_window,"Batch Extract", f"Started {count} extractions to .{fmt}.")

def _batch_video_convert(main_window):
    fmt, ok = QInputDialog.getItem(main_window, "Video format", "Output format:", ["mp4","mkv","mov","webm"], 0, False)
    if not ok: return
    mode, ok2 = QInputDialog.getItem(main_window, "Mode", "Conversion mode:", ["Copy (no re-encode)","Re-encode (quality preset)"], 0, False)
    if not ok2: return
    inf=_choose_dir("Select input folder (videos)", main_window); 
    if not inf: return
    outf=_choose_dir("Select output folder", main_window); 
    if not outf: return
    ff=_ffmpeg_exe()
    if not ff:
        QMessageBox.information(main_window,"Batch Convert","FFmpeg not found. Put ffmpeg.exe in presets/bin, add it to PATH, or set 'ffmpeg_path' in settings."); return
    count=0
    for root,_,files in os.walk(inf):
        for f in files:
            if not f.lower().endswith(VIDEO_EXTS): continue
            src=os.path.join(root,f)
            base=os.path.splitext(os.path.basename(src))[0]+"."+fmt
            dst=os.path.join(outf, base)
            if mode.startswith("Copy"):
                args=[ff,"-y","-i",src,"-map","0","-c","copy",dst]
            else:
                if fmt=="webm":
                    args=[ff,"-y","-i",src,"-c:v","libvpx-vp9","-b:v","0","-crf","32","-c:a","libopus",dst]
                else:
                    args=[ff,"-y","-i",src,"-c:v","libx264","-preset","veryfast","-crf","20","-c:a","aac","-b:a","160k",dst]
            QProcess(main_window).startDetached(args[0], args[1:]); count+=1
    QMessageBox.information(main_window,"Batch Convert", f"Started {count} video conversions.")

def _batch_image_convert(main_window):
    fmt, ok = QInputDialog.getItem(main_window, "Image format", "Output format:", ["png","webp"], 0, False)
    if not ok: return
    inf=_choose_dir("Select input folder (images)", main_window); 
    if not inf: return
    outf=_choose_dir("Select output folder", main_window); 
    if not outf: return
    from PySide6.QtGui import QImage
    count=0
    for root,_,files in os.walk(inf):
        for f in files:
            if not f.lower().endswith(IMAGE_EXTS): continue
            src=os.path.join(root,f)
            img=QImage(src)
            if img.isNull(): continue
            base=os.path.splitext(os.path.basename(src))[0]+"."+fmt
            dst=os.path.join(outf, base)
            if fmt=="webp":
                ok = img.save(dst, "WEBP")
            else:
                ok = img.save(dst, "PNG")
            if ok: count+=1
    QMessageBox.information(main_window,"Batch Image Convert", f"Exported {count} images to .{fmt}.")

def _export_all_audio_tracks(main_window):
    ff = _ffmpeg_exe()
    if not ff:
        QMessageBox.information(main_window, "Export Audio Tracks",
            "FFmpeg not found. Put ffmpeg.exe in presets/bin, add it to PATH, or set 'ffmpeg_path' in settings.")
        return

    fp = _ffprobe_exe()

    src, _ = QFileDialog.getOpenFileName(
        main_window,
        "Select video to extract audio tracks",
        "",
        "Videos (*.mp4 *.mkv *.mov *.webm *.avi);;All files (*.*)"
    )
    if not src:
        return
    outdir = _choose_dir("Select output folder", main_window)
    if not outdir:
        return

    # Probe to count audio streams
    import subprocess, json
    n = 1

    # Prefer ffprobe when available (more reliable than parsing ffmpeg output)
    if fp:
        try:
            r = subprocess.run(
                [fp, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", src],
                capture_output=True,
                text=True
            )
            if r.returncode == 0:
                j = json.loads(r.stdout or "{}")
                streams = j.get("streams") or []
                if isinstance(streams, list) and streams:
                    n = len(streams)
        except Exception:
            pass

    # Fallback: parse ffmpeg -i stderr for "Audio:" lines
    if n == 1:
        try:
            p = subprocess.Popen([ff, "-i", src], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            err = p.communicate()[1]
            lines = [ln for ln in (err or "").splitlines() if "Audio:" in ln]
            if lines:
                n = max(1, len(lines))
        except Exception:
            pass

    for i in range(n):
        dst = os.path.join(outdir, f"{Path(src).stem}_a{i+1}.mka")
        args = [ff, "-y", "-i", src, "-map", f"0:a:{i}", "-c", "copy", dst]
        QProcess(main_window).startDetached(args[0], args[1:])
    QMessageBox.information(main_window, "Export Audio Tracks", f"Started export for {n} track(s).")



from PySide6.QtCore import Qt, QRect, QEventLoop, QPoint
from PySide6.QtWidgets import QRubberBand

def _screenshot_region_dialog(main_window):
    """Select a region INSIDE the app window and save the capture as PNG/JPG."""
    try:
        parent = main_window if isinstance(main_window, QWidget) else QApplication.activeWindow()
        if parent is None:
            QMessageBox.information(main_window, "Screenshot", "No active window to capture.")
            return

        class _RegionOverlay(QWidget):
            def __init__(self, parent):
                super().__init__(parent)
                self.setAttribute(Qt.WA_NoSystemBackground, True)
                self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
                self.setWindowFlags(Qt.SubWindow | Qt.FramelessWindowHint)
                self.setMouseTracking(True)
                self.rb = QRubberBand(QRubberBand.Rectangle, self)
                self.origin = QPoint()
                self.result = None
                self.setGeometry(parent.rect())
                self.show()

            def mousePressEvent(self, e):
                self.origin = e.position().toPoint()
                self.rb.setGeometry(QRect(self.origin, self.origin))
                self.rb.show()

            def mouseMoveEvent(self, e):
                if not self.rb.isVisible(): return
                rect = QRect(self.origin, e.position().toPoint()).normalized()
                self.rb.setGeometry(rect)

            def mouseReleaseEvent(self, e):
                rect = self.rb.geometry().intersected(self.rect())
                self.result = rect if rect.isValid() else None
                self.rb.hide()
                self.hide()
                loop.quit()

        loop = QEventLoop(parent)
        tool = _RegionOverlay(parent)
        loop.exec()

        rect = tool.result
        tool.setParent(None); tool.deleteLater()

        if not rect or rect.isNull():
            QMessageBox.information(main_window, "Screenshot", "Selection cancelled.")
            return

        # Grab selected portion
        pm = parent.grab(rect)
        if pm.isNull():
            QMessageBox.information(main_window, "Screenshot", "Nothing to capture in selected region.")
            return

        path, _ = QFileDialog.getSaveFileName(main_window, "Save Screenshot…", "screenshot.png",
                                              "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)")
        if not path:
            return
        fmt = "PNG" if path.lower().endswith(("png",)) else "JPG"
        quality = 95 if fmt == "JPG" else -1
        ok = pm.toImage().save(path, fmt, quality)
        if not ok:
            QMessageBox.warning(main_window, "Screenshot", "Failed to save screenshot.")
    except Exception:
        # Silent for end users
        pass

# ------------------------------- Open helpers -------------------------------

def _open_in_player(main_window, path: str) -> bool:
    """Try to open 'path' in the app's player without spawning a file dialog.
    Prefer the video widget's .open(path), then fall back to main_window.open_file(path)."""
    if not path:
        return False
    p = Path(path)
    if not p.exists():
        return False
    # 1) Prefer the player widget's .open(path) (never opens a dialog)
    try:
        vw = getattr(main_window, "video", None)
        if vw and hasattr(vw, "open") and callable(getattr(vw, "open")):
            vw.open(str(p))
            _set_current_input_path(main_window, str(p))
            return True
    except Exception:
        pass
    # 2) Fall back to an app-level open_file(path) if provided (some impls pop a dialog)
    try:
        fn = getattr(main_window, 'open_file', None)
        if callable(fn):
            fn(str(p))
            _set_current_input_path(main_window, str(p))
            return True
    except Exception:
        pass
    return False
    p = Path(path)
    if not p.exists(): return False
    try:
        fn = getattr(main_window, 'open_file', None)
        if callable(fn):
            fn(str(p))
            _set_current_input_path(main_window, str(p))
            return True
    except Exception:
        pass
    try:
        vw = getattr(main_window, "video", None)
        if vw and hasattr(vw, "open"):
            vw.open(p); _set_current_input_path(main_window, str(p)); return True
    except Exception:
        pass
    return False

def _trigger_open(main_window):
    # Remember currently open path as "last closed", then open a new file via Qt dialog only.
    prev = _current_input_path(main_window)
    if prev:
        _set_last_closed(prev)

    # Always use a QFileDialog here to avoid side-effects (like opening Windows Explorer).
    filters = "Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)"
    try:
        d = QSettings(ORG, APP).value("last_open_dir", str(Path.home()))
    except Exception:
        d = str(Path.home())
    fn, _ = QFileDialog.getOpenFileName(main_window, "Open media", d, filters)
    if not fn:
        return

    p = Path(fn)
    try:
        s = QSettings(ORG, APP); s.setValue("last_open_dir", str(p.parent))
    except Exception:
        pass

    if _open_in_player(main_window, str(p)):
        _remember_recent(str(p))
        return

    # If we get here, we don't know how to load into the app. Tell the user.
    QMessageBox.information(main_window, "Open", "Loaded path selected, but app video widget integration wasn't found.")


def _add(menu: QMenu, text: str, slot, shortcut: str | None = None, checkable: bool=False):
    act = QAction(text, menu)
    if shortcut: act.setShortcut(shortcut)
    if checkable: act.setCheckable(True)
    act.triggered.connect(slot)
    menu.addAction(act)
    return act



# --- Dynamic (re)population for Recent & Favorites ----------------------------
def _populate_recent_menu(menu: QMenu, main_window):
    try:
        menu.clear()
        recent = _load_list(RECENT_KEY)
        if not recent:
            dummy = QAction("(Empty)", menu); dummy.setEnabled(False); menu.addAction(dummy)
        else:
            for path in recent:
                act = QAction(_icon_for_path(path), path, menu)
                try:
                    act.setData(path)
                except Exception:
                    pass
                act.triggered.connect(lambda checked=False, p=path: (_open_in_player(main_window, p) and _remember_recent(p)))
                menu.addAction(act)
        menu.addSeparator()
        clear_act = QAction("Clear Recent List", menu)
        def _clr():
            _save_list(RECENT_KEY, [])
            QMessageBox.information(main_window, "Recent Files", "Recent list cleared.")
        clear_act.triggered.connect(_clr)
        menu.addAction(clear_act)
        try:
            _attach_right_click_remove(menu, _remove_recent, lambda: _populate_recent_menu(menu, main_window), main_window)
        except Exception:
            pass
    except Exception:
        pass

def _populate_favorites_menu(menu: QMenu, main_window):
    try:
        menu.clear()
        # Pin/Unpin
        pin_act = QAction("Pin Current File", menu)
        def _pin():
            p = _current_input_path(main_window)
            if not p:
                QMessageBox.information(main_window, "Favorites", "No file to pin."); return
            _remember_favorite(p); QMessageBox.information(main_window, "Favorites", "Pinned.")
        pin_act.triggered.connect(_pin); menu.addAction(pin_act)

        # Unpin when applicable
        try:
            cur = _current_input_path(main_window) or ""
            fav_list = _load_list(FAV_KEY)
            if cur and cur in fav_list:
                unpin_act = QAction("Unpin Current File", menu)
                unpin_act.triggered.connect(lambda: (_remove_favorite(cur), QMessageBox.information(main_window, "Favorites", "Unpinned.")))
                menu.addAction(unpin_act)
        except Exception:
            pass

        # List as submenu to match existing UI
        fav = _RemovableMenu("Open Favorite", menu, remove_fn=_remove_favorite, repopulate_cb=lambda: _populate_favorites_menu(menu, main_window))
        menu.addMenu(fav)
        try:
            _attach_right_click_remove(fav, _remove_favorite, lambda: _populate_favorites_menu(menu, main_window), main_window)
        except Exception:
            pass
        fav_list = _load_list(FAV_KEY)
        if not fav_list:
            dummy = QAction("(Empty)", fav); dummy.setEnabled(False); fav.addAction(dummy)
        else:
            for path in fav_list:
                act = QAction(_icon_for_path(path), path, fav)
                try:
                    act.setData(path)
                except Exception:
                    pass
                act.triggered.connect(lambda checked=False, p=path: (_open_in_player(main_window, p) and _remember_recent(p)))
                fav.addAction(act)

        menu.addSeparator()
        clear_act = QAction("Clear Favorites", menu)
        def _clr():
            _save_list(FAV_KEY, [])
            QMessageBox.information(main_window, "Favorites", "Favorites cleared.")
        clear_act.triggered.connect(_clr); menu.addAction(clear_act)
        try:
            _attach_right_click_remove(menu, _remove_recent, lambda: _populate_recent_menu(menu, main_window), main_window)
        except Exception:
            pass
    except Exception:
        pass

def _build_recent_menu(file_menu: QMenu, main_window):
    menu = _RemovableMenu("Recent Files", file_menu, remove_fn=_remove_recent, repopulate_cb=lambda: _populate_recent_menu(menu, main_window))
    file_menu.addMenu(menu)
    try:
        menu.aboutToShow.connect(lambda: _populate_recent_menu(menu, main_window))
    except Exception:
        pass
    _populate_recent_menu(menu, main_window)  # initial fill so it isn't blank before opening
    return menu

def _build_favorites_menu(file_menu: QMenu, main_window):
    menu = file_menu.addMenu("Favorites")
    try:
        menu.aboutToShow.connect(lambda: _populate_favorites_menu(menu, main_window))
    except Exception:
        pass
    _populate_favorites_menu(menu, main_window)  # initial fill so it isn't blank before opening
    return menu

def _open_containing_folder(main_window):
    p=_current_input_path(main_window)
    if not p:
        QMessageBox.information(main_window,"Open Containing Folder","No file loaded.")
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(p).parent)))

def _reopen_last_closed(main_window):
    p=_get_last_closed()
    if not p or not os.path.exists(p):
        QMessageBox.information(main_window,"Reopen Last Closed","No last closed file found.")
        return
    _open_in_player(main_window, p)
    _remember_recent(p)

# ------------------------------- Public entry -------------------------------



# --- Recents Watcher (menu-only, safe) ----------------------------------------
def _start_recents_watcher(main_window):
    """Start a lightweight watcher that mirrors the currently loaded media into
    Recents using _current_input_path(main_window). No changes to the player.
    """
    try:
        from PySide6.QtCore import QTimer
    except Exception:
        return
    try:
        # Only start once per window
        if getattr(main_window, "__recents_watcher", None):
            return
        t = QTimer(main_window)
        t.setInterval(1000)  # 1s
        last_key = "__recents_last_seen"
        def _tick():
            try:
                cur = _current_input_path(main_window)
                if not cur:
                    return
                prev = getattr(main_window, last_key, None)
                if cur != prev:
                    setattr(main_window, last_key, cur)
                    try:
                        _remember_recent(cur)
                    except Exception:
                        pass
                    try:
                        s = QSettings(ORG, APP)
                        s.setValue("last_open_file", cur)
                    except Exception:
                        pass
            except Exception:
                pass
        t.timeout.connect(_tick)
        t.start()
        setattr(main_window, "__recents_watcher", t)
    except Exception:
        pass

def install_file_menu(main_window):
    # Start Recents watcher (menu-only, safe)
    try:
        _start_recents_watcher(main_window)
    except Exception:
        pass
    if main_window is None: return
    try:
        mb=getattr(main_window,"menuBar",None)
        mb=mb() if callable(mb) else mb
        if mb is None:
            mb=QMenuBar(main_window); main_window.setMenuBar(mb)

        # Find or create the single "File" menu
        file_menu=None
        for m in mb.findChildren(QMenu) or []:
            t=(m.title() or "").replace("&","").strip().lower()
            if t == "file":
                file_menu=m; break
        if file_menu is None:
            file_menu=mb.addMenu("File")

        # Always rebuild the contents to avoid duplicates from older code
        file_menu.clear()

        # Core open/describe/screenshot
        _add(file_menu, "Open", lambda: _trigger_open(main_window), "Ctrl+O")
        _add(file_menu, "Screenshot (select region)…", lambda: _screenshot_region_dialog(main_window), "Ctrl+S")

        # Recent + Favorites
        file_menu.addSeparator()
        _build_recent_menu(file_menu, main_window)
        _build_favorites_menu(file_menu, main_window)

        # Save/Export
        file_menu.addSeparator()
        _add(file_menu,"Save Frame As…", lambda: _save_frame_dialog(main_window))
        _add(file_menu,"Save As…", lambda: _save_media_dialog(main_window))
        _add(file_menu,"Save As MP3…", lambda: _save_audio_mp3_dialog(main_window))

        # Batch ops
        batch_menu=file_menu.addMenu("Batch Tools")
        _add(batch_menu,"Convert to MP3 (folder)…", lambda: _batch_convert_to_mp3(main_window))
        _add(batch_menu,"Video → Audio Extract (WAV/AAC/FLAC)…", lambda: _batch_extract_audio(main_window))
        _add(batch_menu,"Video Convert (folder)…", lambda: _batch_video_convert(main_window))
        _add(batch_menu,"Image Convert (folder)…", lambda: _batch_image_convert(main_window))
        _add(batch_menu,"Export ALL Audio Tracks (1 file)…", lambda: _export_all_audio_tracks(main_window))

        file_menu.addSeparator()
        # Utilities
        file_menu.addSeparator()
        _add(file_menu,"Open Containing Folder", lambda: _open_containing_folder(main_window))
        _add(file_menu,"Reopen Last Closed File", lambda: _reopen_last_closed(main_window))

        # Exit
        file_menu.addSeparator()
        _add(file_menu,"Exit (to console)", lambda: QApplication.instance().quit())
        _add(file_menu,"Exit (close process)", lambda: os._exit(0))

        # Maybe auto reopen last session once at startup
    except Exception as e:
        # fail silently for end-users
        # print(e)  # Uncomment for debugging
        pass

# Backward compatible alias (if referenced elsewhere)
_save_as_mp3_dialog = _save_audio_mp3_dialog