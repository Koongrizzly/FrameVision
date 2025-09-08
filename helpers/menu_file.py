# helpers/menu_file.py
from __future__ import annotations
import os, shutil, json
from pathlib import Path

from PySide6.QtCore import QProcess, QSettings, QUrl
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QMenuBar, QMenu, QFileDialog, QMessageBox, QWidget, QPushButton, QApplication, QInputDialog
)
from PySide6.QtGui import QDesktopServices

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

def _ffmpeg_exe() -> str | None:
    try:
        s=QSettings(ORG,APP); p=s.value("ffmpeg_path","")
        if p and os.path.isfile(p): return p
    except Exception:
        pass
    for cand in ("ffmpeg.exe","ffmpeg"):
        ff=shutil.which(cand)
        if ff: return ff
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
        QMessageBox.information(main_window, "Save As", "FFmpeg not found. Add it to PATH or set 'ffmpeg_path' in settings.")
        return
    ext = Path(path).suffix.lower()
    # Try stream copy when saving to MP4/MKV/MOV/WEBM; otherwise re-encode to H.264 + AAC for compatibility
    if ext in (".mp4", ".mkv", ".mov", ".webm"):
        args = [ff, "-y", "-i", src, "-map", "0", "-c", "copy", path]
    else:
        args = [ff, "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "160k", path]
    proc = QProcess(main_window)
    proc.startDetached(args[0], args[1:])

def _save_audio_mp3_dialog(main_window):
    src = _current_input_path(main_window)
    if not src or not os.path.isfile(src):
        QMessageBox.information(main_window,"Save As MP3","No input file loaded."); return
    path, _ = QFileDialog.getSaveFileName(main_window,"Save As MP3…",
        os.path.splitext(os.path.basename(src))[0] + ".mp3",
        "MP3 Audio (*.mp3)")
    if not path: return
    ff = _ffmpeg_exe()
    if not ff:
        QMessageBox.information(main_window,"Save As MP3","FFmpeg not found. Add it to PATH or set 'ffmpeg_path' in settings."); return
    args = [ff, "-y", "-i", src, "-vn", "-acodec", "libmp3lame", "-b:a", "192k", path]
    proc = QProcess(main_window)
    proc.startDetached(args[0], args[1:])

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
        QMessageBox.information(main_window,"Batch MP3","FFmpeg not found. Add it to PATH or set 'ffmpeg_path' in settings."); return
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
        QMessageBox.information(main_window,"Batch Extract","FFmpeg not found."); return
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
        QMessageBox.information(main_window,"Batch Convert","FFmpeg not found."); return
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
    ff=_ffmpeg_exe()
    if not ff:
        QMessageBox.information(main_window,"Export Audio Tracks","FFmpeg not found."); return
    src, _ = QFileDialog.getOpenFileName(main_window, "Select video to extract audio tracks", "", "Videos (*.mp4 *.mkv *.mov *.webm *.avi);;All files (*.*)")
    if not src: return
    outdir=_choose_dir("Select output folder", main_window)
    if not outdir: return
    # Probe to count audio streams
    # Best effort: try a small ffprobe via ffmpeg -hide_banner -i and parse 'Audio:' lines
    import subprocess
    try:
        p=subprocess.Popen([ff, "-i", src], stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        out=p.communicate()[1]
        lines=[ln for ln in out.splitlines() if "Audio:" in ln]
        n=max(1, len(lines))
    except Exception:
        n=1
    for i in range(n):
        dst=os.path.join(outdir, f"{Path(src).stem}_a{i+1}.mka")
        args=[ff,"-y","-i",src,"-map",f"0:a:{i}","-c","copy",dst]
        QProcess(main_window).startDetached(args[0], args[1:])
    QMessageBox.information(main_window,"Export Audio Tracks", f"Started export for {n} track(s).")


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
    # Remember currently open path as "last closed", then open a new file.
    prev = _current_input_path(main_window)
    if prev: _set_last_closed(prev)

    # Preferred: call the app's own open_file() method
    try:
        fn = getattr(main_window, 'open_file', None)
        if callable(fn):
            fn()
            # Best effort: read back current path and remember
            newp=_current_input_path(main_window)
            if newp: _remember_recent(newp)
            return
    except Exception:
        pass
    # Fallback 1: click any button labeled "Open"
    btn = _find_button(main_window, ("open",))
    if btn:
        btn.click()
        newp=_current_input_path(main_window)
        if newp: _remember_recent(newp)
        return
    # Fallback 2: generic file dialog -> open in player if possible
    filters = "Videos (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)"
    d = QSettings(ORG, APP).value("last_open_dir", str(Path.home()))
    fn, _ = QFileDialog.getOpenFileName(main_window, "Open media", d, filters)
    if not fn:
        return
    p = Path(fn)
    try:
        s = QSettings(ORG, APP); s.setValue("last_open_dir", str(p.parent))
    except Exception:
        pass
    if _open_in_player(main_window, str(p)):
        _remember_recent(str(p)); return
    # If we get here, we don't know how to load into the app. Tell the user.
    QMessageBox.information(main_window, "Open", "Loaded path selected, but app video widget integration wasn't found.")

# ------------------------------- Sessions -------------------------------

SESSION_KEY="last_session"
AUTO_REOPEN_KEY="reopen_last_session_on_startup"
SESSION_LOADED_FLAG="session_loaded_this_run"

def _save_session(main_window):
    path=_current_input_path(main_window)
    pos=None
    # Best-effort: pull a millisecond position if the player exposes it
    for attr in ("position","pos_ms","current_ms","tell"):
        try:
            obj=getattr(main_window, "video", None) or main_window
            fn=getattr(obj, attr, None)
            if callable(fn):
                v=fn()
            else:
                v=fn
            if isinstance(v, int):
                pos=v; break
        except Exception:
            pass
    data={"path": path or "", "position_ms": pos}
    try:
        s=QSettings(ORG,APP); s.setValue(SESSION_KEY, json.dumps(data))
        QMessageBox.information(main_window,"Save Session","Session saved.")
    except Exception:
        pass

def _load_session(main_window, silent: bool=False):
    try:
        s=QSettings(ORG,APP); raw=s.value(SESSION_KEY,"")
        if not raw: 
            if not silent: QMessageBox.information(main_window,"Load Session","No saved session."); 
            return
        data=json.loads(raw) if isinstance(raw,str) else raw
        path=data.get("path") or ""
        if path and os.path.exists(path):
            _open_in_player(main_window, path)
            _remember_recent(path)
            if not silent: QMessageBox.information(main_window,"Load Session","Session restored.")
        elif not silent:
            QMessageBox.information(main_window,"Load Session","Saved file not found.")
    except Exception:
        if not silent:
            QMessageBox.information(main_window,"Load Session","Failed to load session.")

def _toggle_auto_reopen(action: QAction):
    s=QSettings(ORG,APP); s.setValue(AUTO_REOPEN_KEY, action.isChecked())

def _maybe_auto_reopen(main_window):
    s=QSettings(ORG,APP)
    if s.value(SESSION_LOADED_FLAG, False): 
        return
    if str(s.value(AUTO_REOPEN_KEY, "false")).lower() in ("1","true","yes"):
        _load_session(main_window, silent=True)
        s.setValue(SESSION_LOADED_FLAG, True)

# ------------------------------- Menu builders -------------------------------

def _add(menu: QMenu, text: str, slot, shortcut: str | None = None, checkable: bool=False):
    act = QAction(text, menu)
    if shortcut: act.setShortcut(shortcut)
    if checkable: act.setCheckable(True)
    act.triggered.connect(slot)
    menu.addAction(act)
    return act

def _build_recent_menu(file_menu: QMenu, main_window):
    recent_menu=file_menu.addMenu("Recent Files")
    recent = _load_list(RECENT_KEY)
    if not recent:
        dummy=QAction("(Empty)", recent_menu); dummy.setEnabled(False); recent_menu.addAction(dummy)
    else:
        for path in recent:
            act=QAction(_icon_for_path(path), path, recent_menu)
            act.triggered.connect(lambda checked=False, p=path: (_open_in_player(main_window, p) and _remember_recent(p)))
            recent_menu.addAction(act)
    recent_menu.addSeparator()
    clear_act=QAction("Clear Recent List", recent_menu)
    def _clr():
        _save_list(RECENT_KEY, [])
        QMessageBox.information(main_window,"Recent Files","Recent list cleared.")
    clear_act.triggered.connect(_clr)
    recent_menu.addAction(clear_act)
    return recent_menu

def _build_favorites_menu(file_menu: QMenu, main_window):
    fav_menu=file_menu.addMenu("Favorites")
    cur=_current_input_path(main_window)
    pin_act=QAction("Pin Current File", fav_menu)
    def _pin():
        p=_current_input_path(main_window)
        if not p:
            QMessageBox.information(main_window,"Favorites","No file to pin."); return
        _remember_favorite(p); QMessageBox.information(main_window,"Favorites","Pinned.")
    pin_act.triggered.connect(_pin); fav_menu.addAction(pin_act)

    fav=fav_menu.addMenu("Open Favorite")
    fav_list=_load_list(FAV_KEY)
    if not fav_list:
        dummy=QAction("(Empty)", fav); dummy.setEnabled(False); fav.addAction(dummy)
    else:
        for path in fav_list:
            act=QAction(_icon_for_path(path), path, fav)
            act.triggered.connect(lambda checked=False, p=path: (_open_in_player(main_window, p) and _remember_recent(p)))
            fav.addAction(act)

    fav_menu.addSeparator()
    clear_act=QAction("Clear Favorites", fav_menu)
    def _clr():
        _save_list(FAV_KEY, [])
        QMessageBox.information(main_window,"Favorites","Favorites cleared.")
    clear_act.triggered.connect(_clr); fav_menu.addAction(clear_act)
    return fav_menu

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

def install_file_menu(main_window):
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

        # Session / Project
        file_menu.addSeparator()
        _add(file_menu,"Save Session", lambda: _save_session(main_window))
        _add(file_menu,"Load Session", lambda: _load_session(main_window))
        auto=_add(file_menu,"Reopen Last Session on Startup", lambda checked=False: _toggle_auto_reopen(auto), checkable=True)
        try:
            s=QSettings(ORG,APP); auto.setChecked(bool(s.value(AUTO_REOPEN_KEY, False)))
        except Exception:
            pass

        # Utilities
        file_menu.addSeparator()
        _add(file_menu,"Open Containing Folder", lambda: _open_containing_folder(main_window))
        _add(file_menu,"Reopen Last Closed File", lambda: _reopen_last_closed(main_window))

        # Exit
        file_menu.addSeparator()
        _add(file_menu,"Exit (to console)", lambda: QApplication.instance().quit())
        _add(file_menu,"Exit (close process)", lambda: os._exit(0))

        # Maybe auto reopen last session once at startup
        _maybe_auto_reopen(main_window)

    except Exception as e:
        # fail silently for end-users
        # print(e)  # Uncomment for debugging
        pass

# Backward compatible alias (if referenced elsewhere)
_save_as_mp3_dialog = _save_audio_mp3_dialog
