from __future__ import annotations
import os

import json
from pathlib import Path
from typing import Optional, Any, Union
import subprocess
from functools import lru_cache

from PySide6.QtCore import QUrl, Qt, QTimer, Signal, QEvent
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QToolButton, QProgressBar,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QMenu, QStyle, QFrame
)

ETA_FUDGE_SEC = 3  # small cushion for cleanup (temp deletion, final writes)





# ---- Thumbnail helpers (video preview, rounded pixmaps, cache) --------------
VIDEO_EXTS = {'.mp4','.mov','.avi','.mkv','.webm','.m4v','.mpg','.mpeg'}

def _bin_dir() -> Path:
    try:
        return Path.cwd() / "presets" / "bin"
    except Exception:
        return Path("presets") / "bin"

def _ffmpeg_candidates() -> list[Path]:
    names = ["ffmpeg.exe","ffmpeg","mmpeg.exe","mmpeg"]
    out = []
    b = _bin_dir()
    try:
        for n in names:
            p = b / n
            if p.exists() and p.is_file():
                out.append(p)
    except Exception:
        pass
    return out

def _rounded_with_border(pm: QPixmap, radius: int = 8) -> QPixmap:
    try:
        if pm.isNull():
            return pm
        w, h = pm.width(), pm.height()
        out = QPixmap(w, h)
        out.fill(Qt.transparent)
        from PySide6.QtGui import QPainterPath
        from PySide6.QtCore import QRectF
        p = QPainter(out)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        path = QPainterPath()
        rect = QRectF(0.5, 0.5, w-1.0, h-1.0)
        path.addRoundedRect(rect, radius, radius)
        p.setClipPath(path)
        p.drawPixmap(0, 0, pm)
        pen = QPen(QColor(0,0,0,40))
        pen.setWidthF(1.0)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect, radius, radius)
        p.end()
        return out
    except Exception:
        return pm

@lru_cache(maxsize=128)
def _cached_scaled_rounded_pixmap(path_str: str, w: int, h: int, mtime_key: float, radius: int = 8) -> QPixmap:
    try:
        pm = QPixmap(path_str)
        if pm.isNull():
            return QPixmap()
        pm = pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return _rounded_with_border(pm, radius)
    except Exception:
        return QPixmap()

def _ensure_video_preview(video_path: Path, thumbs_dir: Path, scale_width: int = 256, suffix: str = "_preview") -> Path | None:
    try:
        thumbs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        out_png = thumbs_dir / f"{video_path.stem}{suffix}.png"
        # Up-to-date?
        try:
            if out_png.exists() and out_png.stat().st_mtime >= video_path.stat().st_mtime:
                return out_png
        except Exception:
            pass
        # Find ffmpeg-like binary in presets/bin
        ffmpegs = _ffmpeg_candidates()
        if not ffmpegs:
            return out_png if out_png.exists() else None
        exe = str(ffmpegs[0])
        cmd = [
            exe, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"thumbnail,scale={scale_width}:-1",
            str(out_png)
        ]
        try:
            import subprocess as _sp
            _sp.run(cmd, check=False, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, timeout=15)
        except Exception:
            pass
        return out_png if out_png.exists() else None
    except Exception:
        return None

# ---- Queue Autoplay (Play last result) --------------------------------------
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QCheckBox

class _AutoPlayLastController(QObject):
    def __init__(self, host_widget, checkbox: QCheckBox, done_dir: Path):
        super().__init__(host_widget)
        self.host = host_widget
        self.chk = checkbox
        self.done_dir = Path(done_dir) if done_dir else None
        self.seen: set[str] = set()
        try:
            if self.done_dir and self.done_dir.exists():
                self.seen = {str(p) for p in self.done_dir.glob("*.json")}
        except Exception:
            self.seen = set()
        self.poll = QTimer(self)
        self.poll.setInterval(900)
        self.poll.timeout.connect(self._tick)
        self.poll.start()

    def _tick(self):
        # Only act when toggle is on
        try:
            if not (self.chk and self.chk.isChecked() and self.done_dir and self.done_dir.exists()):
                return
            latest = None; latest_ts = -1.0
            for p in self.done_dir.glob("*.json"):
                try:
                    ts = p.stat().st_mtime
                    if ts > latest_ts:
                        latest = p; latest_ts = ts
                except Exception:
                    continue
            if not latest:
                return
            key = str(latest)
            if key in self.seen:
                return
            # mark seen and schedule open with 2s delay
            self.seen.add(key)
            QTimer.singleShot(2000, lambda: self._open_job(latest))
        except Exception:
            pass

    
    def _open_job(self, job_json: Path):
        try:
            # Resolve media using JobRowWidget logic
            w = JobRowWidget(job_json, "done")
            media = w._resolve_output_file()
            if not (media and media.exists()):
                return
            path_obj = Path(media)

            # Prefer internal player on the real app host
            hosts = [self.host, getattr(self.host, "main", None), self.host.window() if hasattr(self.host, "window") else None]
            for h in hosts:
                if not h:
                    continue
                # Direct attributes used throughout the app
                for attr in ("video", "player"):
                    try:
                        v = getattr(h, attr, None)
                        if v and hasattr(v, "open"):
                            v.open(path_obj)
                            # Optional HUD/info if present; best-effort
                            try:
                                if hasattr(h, "hud"):
                                    h.hud.set_info(path_obj)
                                if hasattr(h, "video") and hasattr(h.video, "set_info_text"):
                                    # compose_video_info_text is defined in the app; best-effort import
                                    try:
                                        from helpers.framevision_app import compose_video_info_text  # type: ignore
                                        h.video.set_info_text(compose_video_info_text(path_obj))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            return
                    except Exception:
                        pass
            # If we couldn't find an internal player, do nothing to avoid launching external apps.
        except Exception:
            pass



def install_queue_toggle_play_last(pane_widget, grid_layout, config: dict, save_config_callable, done_dir: Path):
    """Add 'Play last result' toggle to the Queue header and run an autoplay controller.
    This function is made to be called from QueuePane in framevision_app with ONE line.
    """
    try:
        cb = QCheckBox("Play last result")
        cb.setToolTip("Auto-open the newest Finished item in the internal player (waits 2s).")
        cb.setChecked(bool(config.get("queue_play_last", False)))
        def _on_toggle(v):
            try:
                config["queue_play_last"] = bool(v)
                if callable(save_config_callable):
                    save_config_callable()
            except Exception:
                pass
        cb.toggled.connect(_on_toggle)
        # place right below counts row (row 4, col 0) like other controls
        try:
            from PySide6.QtCore import Qt
            grid_layout.addWidget(cb, 4, 0, 1, 1, Qt.AlignLeft)
        except Exception:
            try:
                grid_layout.addWidget(cb, 4, 0, 1, 1)
            except Exception:
                pass
        # Start controller that polls done_dir and plays via internal player
        ctl = _AutoPlayLastController(pane_widget, cb, done_dir)
        # Keep a reference on the pane to avoid GC
        try:
            setattr(pane_widget, "_autoplay_last_ctl", ctl)
        except Exception:
            pass
    except Exception:
        pass

class JobRowWidget(QWidget):
    def _themed_icon(self, theme_names: list[str], fallback_std) -> QIcon:
        try:
            for name in theme_names:
                ic = QIcon.fromTheme(name)
                if ic and not ic.isNull():
                    return ic
        except Exception:
            pass
        try:
            return self.style().standardIcon(fallback_std)
        except Exception:
            return QIcon()

    def _refresh_icons(self) -> None:
        try:
            self.btn_play.setIcon(self._themed_icon([
                'media-playback-start','playback-start','media-playback-start-symbolic','media-start'
            ], QStyle.SP_MediaPlay))
            self.btn_open.setIcon(self._themed_icon([
                'folder-open','document-open-folder','folder-open-symbolic'
            ], QStyle.SP_DirOpenIcon))
            # Trash fallback if theme doesn't provide delete
            self.btn_del.setIcon(self._themed_icon([
                'edit-delete','user-trash','user-trash-symbolic','edit-delete-symbolic'
            ], QStyle.SP_TrashIcon))
        except Exception:
            pass

    """
    Professional job row widget with:
      - Icon buttons (Open/Play/Delete)
      - Left status color strip (running/done/failed/queued)
      - Bold/monospace title + smaller subtitle (input, model, timing)
      - Context menu (Open, Play, Delete, View JSON)

    Signals:
      - playRequested(str path): ask the app to play media in its internal player
      - openRequested(str path): ask the app to open a folder/file internally
    """

    playRequested = Signal(str)
    openRequested = Signal(str)

    
    def _notify_recent(self, p) -> None:
        # Best-effort: find any widget exposing _add_recent and call it.
        try:
            from pathlib import Path as _P
            from PySide6.QtWidgets import QApplication
            path = _P(p) if p else None
            if not path or not path.exists():
                return
            app = QApplication.instance()
            if not app:
                return
            for w in app.allWidgets():
                try:
                    fn = getattr(w, "_add_recent", None)
                    if callable(fn):
                        fn(path)
                        return
                except Exception:
                    continue
        except Exception:
            pass

    def _maybe_notify_recent_initial(self) -> None:
        # If this row is already 'done' when created, push it to Recents once.
        try:
            if getattr(self, "_recent_notified", False):
                return
            status = (self._status_from_fs() or self.status).lower()
            if status == "done":
                p = self._resolve_output_file()
                if p and p.exists():
                    self._notify_recent(p)
                    self._recent_notified = True
        except Exception:
            pass
    def __init__(self, job_path: Path, status: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("JobRowWidget")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.job_path = Path(job_path)
        self.status = status
        self.data: dict[str, Any] = {}
        self._load_json_safely()

        # --- Left status strip ---
        self.status_strip = QFrame(self)
        self.status_strip.setObjectName("StatusStrip")
        self.status_strip.setFixedWidth(6)
        self.status_strip.setFrameShape(QFrame.NoFrame)

        # --- Thumbnail (left) ---
        self.thumb = QLabel()
        self.thumb.setFixedSize(48, 48)
        self.thumb.setScaledContents(True)
        self._set_thumbnail()


        # --- Title (bold/monospace) ---
        self.title = QLabel(self._title_text())
        self.title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f = QFont()
        f.setBold(True)
        self.title.setFont(f)
        self.title.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        self.title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # --- Subtitle (smaller/secondary color) ---
        self.subtitle = QLabel(self._subtitle_text())
        self.subtitle.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.subtitle.setStyleSheet("color: palette(mid); font-size: 11px;")
        self.subtitle.setWordWrap(False)
        self.subtitle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # --- Progress bar ---
        self.bar = QProgressBar()
        self.bar.setMinimumHeight(8)
        self.bar.setTextVisible(False)
        self.bar.setObjectName("JobProgress")

        # --- Buttons (icons, no text) ---
        self.btn_play = QToolButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip("Play output media")
        self.btn_play.setAutoRaise(True)
        self.btn_play.clicked.connect(self._play_output)

        self.btn_open = QToolButton()
        self.btn_open.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_open.setToolTip("Open output folder")
        self.btn_open.setAutoRaise(True)
        self.btn_open.clicked.connect(self._open_folder)

        self.btn_del = QToolButton()
        trash_icon = self.style().standardIcon(QStyle.SP_TrashIcon) or self.style().standardIcon(QStyle.SP_DialogCloseButton)
        self.btn_del.setIcon(trash_icon)
        self.btn_del.setToolTip("Delete output and remove from queue")
        self.btn_del.setAutoRaise(True)
        self.btn_del.clicked.connect(self._delete_job)

        # Apply theme-aware icons now
        self._refresh_icons()

        for b in (self.btn_play, self.btn_open, self.btn_del):
            b.setFixedSize(28, 24)

        # --- Layouts ---
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(2)
        text_col.addWidget(self.title)
        text_col.addWidget(self.subtitle)
        text_col.addWidget(self.bar)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(2)
        actions.addWidget(self.btn_play)
        actions.addWidget(self.btn_open)
        actions.addWidget(self.btn_del)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)
        top.addWidget(self.status_strip, 0)
        top.addWidget(self.thumb, 0)
        top.addLayout(text_col, 1)
        top.addLayout(actions, 0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        lay.addLayout(top)

        # --- Styling (hover/card feel) ---
        self.setStyleSheet(
            """
            #JobRowWidget {
                border: 1px solid rgba(0,0,0,20%);
                border-radius: 8px;
                background: palette(base);
            }
            #JobRowWidget:hover {
                background: rgba(0,0,0,4%);
            }
            #StatusStrip {
                border-top-left-radius: 8px;
                border-bottom-left-radius: 8px;
            }
            #JobProgress {
                border: 1px solid rgba(0,0,0,15%);
                border-radius: 4px;
            }
            """
        )

        # Initial state
        self._update_progressbar()
        self._update_button_visibility()
        self._apply_status_style()

        # If running, poll for refresh
        if self._status_from_fs() == "running" or self.status == "running":
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.refresh)
            self.timer.start(1000)

                # Enable custom context menu
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    def changeEvent(self, event) -> None:
        try:
            if event.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange, QEvent.StyleChange):
                self._refresh_icons()
        except Exception:
            pass
        try:
            super().changeEvent(event)
        except Exception:
            pass

    # ---------- Public ----------
    def refresh(self) -> None:
        """Re-read JSON and update UI state. Called externally by QueuePane."""
        self._load_json_safely()
        self.title.setText(self._title_text())
        self.subtitle.setText(self._subtitle_text())
        self._update_progressbar()
        self._update_button_visibility()
        self._apply_status_style()
        self._set_thumbnail()

    # ---------- Context menu ----------
    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        act_open = QAction(self.style().standardIcon(QStyle.SP_DirOpenIcon), "Open output folder", self)
        act_play = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Play output media", self)
        act_del = QAction(self.style().standardIcon(QStyle.SP_TrashIcon), "Delete output + remove from queue", self)
        act_view = QAction(self.style().standardIcon(QStyle.SP_FileIcon), "View job JSON", self)

        act_open.triggered.connect(self._open_folder)
        act_play.triggered.connect(self._play_output)
        act_del.triggered.connect(self._delete_job)
        act_view.triggered.connect(self._view_json)

        status = (self._status_from_fs() or self.status).lower()
        term = status in ("done", "failed")

        menu.addAction(act_open).setEnabled(term)
        menu.addAction(act_play).setEnabled(term)
        menu.addSeparator()
        menu.addAction(act_del).setEnabled(term)
        menu.addSeparator()
        menu.addAction(act_view).setEnabled(True)

        menu.exec(event.globalPos())

    # ---------- Internals ----------
    def _load_json_safely(self) -> None:
        try:
            txt = Path(self.job_path).read_text(encoding="utf-8")
            self.data = json.loads(txt)
        except Exception:
            self.data = {}

    def _status_from_fs(self) -> str:
        try:
            return self.job_path.parent.name.lower()
        except Exception:
            return self.status

    def _shorten_basename(self, name: str, max_chars: int = 42) -> str:
        """Return a shortened file name keeping the extension, using middle ellipsis."""
        try:
            if not name:
                return ""
            if len(name) <= max_chars:
                return name
            p = Path(name)
            stem = p.stem
            ext = p.suffix or ""
            avail = max(3, max_chars - len(ext) - 3)
            left = max(1, avail // 2)
            right = max(1, avail - left)
            if len(stem) > avail:
                stem = f"{stem[:left]}...{stem[-right:]}"
            return f"{stem}{ext}"
        except Exception:
            if len(name) <= max_chars:
                return name
            half = max_chars // 2
            return name[:half] + "..." + name[-(max_chars - half - 3):]

    def _shorten_finished_display(self, name: str, limit: int = 16, space_before_ext: bool = True) -> str:
        """
        Finished-row name style: first `limit` chars of the stem, then '...' and the extension.
        Example: 'a_very_long_photo_name.jpg' -> 'a_very_long_ph... .jpg'
        """
        try:
            if not name:
                return ""
            p = Path(name)
            stem, ext = p.stem, p.suffix or ""
            if len(stem) <= limit:
                return stem + ext
            sep = " " if space_before_ext else ""
            return f"{stem[:limit]}...{sep}{ext}"
        except Exception:
            # naive fallback
            dot = name.rfind(".")
            if dot > 0:
                stem = name[:dot]; ext = name[dot:]
            else:
                stem, ext = name, ""
            sep = " " if space_before_ext else ""
            if len(stem) <= limit:
                return stem + ext
            return stem[:limit] + "..." + sep + ext


    def _thumbs_dir(self) -> Path:
        # Root: ./output/last results/queue
        try:
            return Path.cwd() / "output" / "last results" / "queue"
        except Exception:
            return Path("output") / "last results" / "queue"

    def _find_thumbnail_for_job(self) -> Optional[Path]:
        # Prefer explicit thumbnail path in job data if present
        d = self.data or {}
        args = d.get("args") or {}
        thumb = d.get("thumbnail") or args.get("thumbnail")
        if thumb:
            p = Path(str(thumb)).expanduser()
            if p.exists():
                return p

        # Derive from output stem
        stem = None
        try:
            outp = self._resolve_output_file()
            stem = outp.stem if outp else None
        except Exception:
            stem = None

        td = self._thumbs_dir()
        if not stem or not td.exists():
            return None

        try:
            candidates = []
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidates.extend(td.glob(stem + "*" + ext))
            if not candidates:
                return None
            # pick most recent
            best = max(candidates, key=lambda p: p.stat().st_mtime)
            return best
        except Exception:
            return None

    
    
    
    def _find_source_thumbnail_for_job(self) -> Optional[Path]:
        # Source (input-based) thumbnail for this job.
        # Priority:
        #   1) explicit fields in job/args: source_thumbnail/src_thumbnail/input_thumbnail/source_thumb/thumb_input
        #   2) derive from INPUT file stem and search thumbs dir
        d = self.data or {}
        args = d.get("args") or {}

        # 1) explicit keys
        for key in ("source_thumbnail","src_thumbnail","input_thumbnail","source_thumb","thumb_input"):
            try:
                val = d.get(key) or (args.get(key) if isinstance(args, dict) else None)
                if val:
                    pth = Path(str(val)).expanduser()
                    if pth.exists():
                        return pth
            except Exception:
                pass

        # 2) derive from INPUT stem
        stem = None
        try:
            inp = self._resolve_input_file()
            stem = inp.stem if inp else None
        except Exception:
            stem = None
        if not stem:
            return None
        td = self._thumbs_dir()
        try:
            candidates = []
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidates.extend(td.glob(stem + "*" + ext))
            if not candidates:
                return None
            best = max(candidates, key=lambda p: p.stat().st_mtime)
            return best
        except Exception:
            return None


    def _set_thumbnail(self) -> None:
        """Render a thumbnail for this row.")"""
        try:
            w, h = self.thumb.width(), self.thumb.height()
            status = (self._status_from_fs() or self.status).lower()

            def _show_from_path(path: Path, is_input: bool = False) -> bool:
                try:
                    if not path or not path.exists():
                        return False
                    suf = path.suffix.lower()
                    if suf in VIDEO_EXTS:
                        thumbs = self._thumbs_dir()
                        suffix = "_preview_input" if is_input else "_preview"
                        prev = _ensure_video_preview(path, thumbs, 256, suffix)
                        if prev and prev.exists():
                            mt = prev.stat().st_mtime
                            pm = _cached_scaled_rounded_pixmap(str(prev), w, h, mt, 8)
                            if not pm.isNull():
                                self.thumb.setPixmap(pm)
                                self.thumb.setToolTip(str(prev))
                                return True
                        return False
                    if suf in (".png",".jpg",".jpeg",".webp",".bmp",".gif",".tif",".tiff"):
                        mt = path.stat().st_mtime
                        pm = _cached_scaled_rounded_pixmap(str(path), w, h, mt, 8)
                        if not pm.isNull():
                            self.thumb.setPixmap(pm)
                            self.thumb.setToolTip(str(path))
                            return True
                except Exception:
                    return False
                return False

            if status in ("running","queued","pending"):
                # 1) Prefer source (input-based) job thumbnail
                try:
                    sp = self._find_source_thumbnail_for_job()
                    if sp and sp.exists():
                        mt = sp.stat().st_mtime
                        pm = _cached_scaled_rounded_pixmap(str(sp), w, h, mt, 8)
                        if not pm.isNull():
                            self.thumb.setPixmap(pm)
                            self.thumb.setToolTip(str(sp))
                            return
                except Exception:
                    pass

                # 2) Next, any explicit/derived generic thumbnail
                try:
                    pth = self._find_thumbnail_for_job()
                    if pth and pth.exists():
                        mt = pth.stat().st_mtime
                        pm = _cached_scaled_rounded_pixmap(str(pth), w, h, mt, 8)
                        if not pm.isNull():
                            self.thumb.setPixmap(pm)
                            self.thumb.setToolTip(str(pth))
                            return
                except Exception:
                    pass

                # 3) Then, show preview directly from INPUT
                inp = self._resolve_input_file()
                if _show_from_path(inp, is_input=True):
                    return

                # 4) Fallback to OUTPUT if preview already exists
                outp = self._resolve_output_file()
                if _show_from_path(outp, is_input=False):
                    return

            else:
                # Finished/failed
                outp = self._resolve_output_file()
                if _show_from_path(outp, is_input=False):
                    return
                # Fallbacks
                try:
                    pth = self._find_thumbnail_for_job()
                    if pth and pth.exists():
                        mt = pth.stat().st_mtime
                        pm = _cached_scaled_rounded_pixmap(str(pth), w, h, mt, 8)
                        if not pm.isNull():
                            self.thumb.setPixmap(pm)
                            self.thumb.setToolTip(str(pth))
                            return
                except Exception:
                    pass
                inp = self._resolve_input_file()
                if _show_from_path(inp, is_input=True):
                    return

            # Final fallback: clear
            self.thumb.clear()
        except Exception:
            try:
                self.thumb.clear()
            except Exception:
                pass

    def _title_text(self) -> str:
        d = self.data or {}
        status = (self._status_from_fs() or self.status).lower()

        # When finished: first line shows OUTPUT filename + model
        if status in ("done",):
            try:
                outp = self._resolve_output_file()
            except Exception:
                outp = None
            out_name = self._shorten_finished_display(outp.name, 16, True) if outp else None

            args = d.get("args") or {}
            model = d.get("model") or d.get("model_name") or args.get("model") or args.get("ai_model")
            if out_name and model:
                return f"{out_name}  |  {model}"
            if out_name:
                return str(out_name)

        # Fallback for running/pending/failed
        title = d.get("title")
        if not title:
            title = (d.get("args") or {}).get("label") or self.job_path.stem
        return str(title)
    def _subtitle_text(self) -> str:
        d = self.data or {}
        args = d.get("args") or {}

        status = (self._status_from_fs() or self.status).lower()
        if status in ("done",):
            jobname = self.job_path.stem
            started = d.get("started_at")
            finished = d.get("finished_at")
            dur = d.get("duration_sec") or self._compute_elapsed(d)

            s_txt = self._fmt_clock(started) if started else ""
            e_txt = self._fmt_clock(finished) if finished else ""
            dur_txt = self._fmt_dur(dur) if dur is not None else ""

            timings = " ".join([p for p in [s_txt, f"({dur_txt})" if dur_txt else "", e_txt] if p])
            if timings:
                return f"{jobname}  |  {timings}"
            return str(jobname)

        # Informative subtitle for other states
        parts = []
        # sampler / steps / model (basename)
        try:
            smp = (args.get("sampler") or "").strip()
            stp = args.get("steps")
            mdl = (d.get("model") or d.get("model_name") or args.get("model") or args.get("ai_model") or args.get("model_path"))
            mdl = os.path.basename(str(mdl)) if mdl else None
            if smp:
                parts.append(str(smp))
            try:
                if stp: parts.append(f"{int(stp)} steps")
            except Exception:
                pass
            if mdl:
                parts.append(str(mdl))
        except Exception:
            pass


        src = d.get("input") or args.get("infile") or args.get("input")
        if src:
            base = Path(str(src)).name
            parts.append(self._shorten_basename(base))

        res = self._extract_resolution(d, args)
        if res:
            if parts:
                parts[-1] = f"{parts[-1]} \u2192 {res}"
            else:
                parts.append(res)

        model = d.get("model") or d.get("model_name") or args.get("model") or args.get("ai_model")
        if model:
            parts.append(str(model))

        started = d.get("started_at")
        finished = d.get("finished_at")
        eta_sec_raw = d.get("eta_sec")
        elapsed_sec = d.get("duration_sec") or self._compute_elapsed(d)

        time_frag = []
        if status == "running":
            if started:
                time_frag.append(f"Started {self._fmt_clock(started)}")
            if elapsed_sec is not None:
                time_frag.append(f"Elapsed {self._fmt_dur(elapsed_sec)}")
            if eta_sec_raw is not None:
                try:
                    eta_display = int(float(eta_sec_raw)) + int(ETA_FUDGE_SEC)
                except Exception:
                    eta_display = None
                if eta_display and eta_display > 0:
                    time_frag.append(f"ETA {self._fmt_dur(eta_display)}")
        elif status in ("failed",):
            if finished:
                time_frag.append(f"Finished {self._fmt_clock(finished)}")
            if elapsed_sec is not None:
                time_frag.append(f"Duration {self._fmt_dur(elapsed_sec)}")
        elif status in ("queued", "pending"):
            if started:
                time_frag.append(f"Queued since {self._fmt_clock(started)}")

        if time_frag:
            parts.append(" \u2022 ".join(time_frag))

        return "  |  ".join(parts)

    def _compute_elapsed(self, d: dict) -> Optional[int]:
        try:
            started = self._as_epoch(d.get("started_at"))
            finished = self._as_epoch(d.get("finished_at"))
            if started and finished:
                return max(0, int(finished - started))
            if started and (self._status_from_fs() or self.status).lower() == "running":
                return max(0, int(self._now_epoch() - started))
        except Exception:
            pass
        return None

    def _as_epoch(self, v: Union[str, float, int, None]) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            pass
        try:
            import datetime as _dt
            try:
                return _dt.datetime.fromisoformat(str(v)).timestamp()
            except Exception:
                return _dt.datetime.strptime(str(v), "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return None

    def _now_epoch(self) -> float:
        import time as _t
        return _t.time()

    def _fmt_clock(self, v: Union[str, float, int]) -> str:
        ts = self._as_epoch(v)
        if ts is None:
            return str(v)
        from datetime import datetime
        dt = datetime.fromtimestamp(ts)
        return dt.strftime("%H:%M:%S")

    def _fmt_dur(self, seconds: Union[int, float]) -> str:
        try:
            s = int(seconds)
            h, rem = divmod(s, 3600)
            m, s = divmod(rem, 60)
            if h:
                return f"{h:d}:{m:02d}:{s:02d}"
            return f"{m:d}:{s:02d}"
        except Exception:
            return str(seconds)

    def _extract_resolution(self, d: dict, args: dict) -> Optional[str]:
        w = d.get("in_w") or args.get("in_w") or args.get("width") or args.get("w")
        h = d.get("in_h") or args.get("in_h") or args.get("height") or args.get("h")
        scale = args.get("scale") or d.get("scale")
        if w and h:
            try:
                return f"{int(w)}x{int(h)}"
            except Exception:
                return f"{w}x{h}"
        if scale:
            return f"scale x{scale}"
        return None

    def _status_color(self, status: str) -> str:
        s = (status or "").lower()
        if s == "running":
            return "#2f80ed"
        if s == "done":
            return "#27ae60"
        if s == "failed":
            return "#e74c3c"
        if s in ("queued", "pending"):
            return "#f2c94c"
        return "rgba(0,0,0,25%)"

    def _apply_status_style(self) -> None:
        status = self._status_from_fs() or self.status
        color = self._status_color(status)
        self.status_strip.setStyleSheet(f"background: {color};")

    def _update_progressbar(self) -> None:
        status = self._status_from_fs() or self.status
        d = self.data or {}
        # Accept various keys for progress
        pct = d.get("pct") or d.get("progress") or d.get("percent") or (d.get("metrics") or {}).get("progress")
        try:
            if pct is not None:
                pct = float(pct)
        except Exception:
            pct = None

        if status == "running":
            if isinstance(pct, (int, float)) and 0.0 <= pct <= 100.0:
                self.bar.setRange(0, 100)
                self.bar.setValue(int(pct))
                self.bar.setTextVisible(True)
                self.bar.setFormat("%p%")
            else:
                self.bar.setRange(0, 0)
                self.bar.setTextVisible(False)
        elif status == "done":
            self.bar.setRange(0, 100)
            self.bar.setValue(100)
            self.bar.setTextVisible(True)
            self.bar.setFormat("100%")
        elif status == "failed":
            self.bar.setRange(0, 100)
            self.bar.setValue(100)
            self.bar.setTextVisible(True)
            self.bar.setFormat("Failed")
        else:
            self.bar.setRange(0, 100)
            self.bar.setValue(0)
            self.bar.setTextVisible(False)

    def _update_button_visibility(self) -> None:
        status = (self._status_from_fs() or self.status).lower()
        term = status in ("done", "failed")
        self.btn_open.setVisible(term)
        self.btn_play.setVisible(term)
        self.btn_del.setVisible(term)

    # ---------- Internal player helpers ----------
    def _set_current_path_if_any(self, scope, path_str: str) -> None:
        try:
            for attr in ("current_path", "current_video"):
                if hasattr(scope, attr):
                    try:
                        setattr(scope, attr, Path(path_str))
                    except Exception:
                        pass
        except Exception:
            pass

    def _try_internal_player(self, path_str: str) -> bool:
        """
        Try to play via app's internal player, similar to the RIFE tab:
        prefer <main>.video.open(path) if available; otherwise look for video/player on parent/window.
        Returns True if handled.
        """
        try:
            hosts = [self, self.parent(), self.window()]
            for h in hosts:
                if not h:
                    continue
                # direct attributes on host
                for attr in ("video", "player"):
                    v = getattr(h, attr, None)
                    if v is not None and hasattr(v, "open"):
                        try:
                            v.open(path_str)
                            self._set_current_path_if_any(h, path_str)
                            return True
                        except Exception:
                            pass
                # look for a 'main' object on the host
                m = getattr(h, "main", None)
                if m is not None:
                    for attr in ("video", "player"):
                        v = getattr(m, attr, None)
                        if v is not None and hasattr(v, "open"):
                            try:
                                v.open(path_str)
                                self._set_current_path_if_any(m, path_str)
                                return True
                            except Exception:
                                pass
        except Exception:
            pass
        return False

    # ---------- Actions ----------
    def _open_folder(self) -> None:
        d = self.data or {}
        args = d.get("args") or {}
        path = args.get("outfile") or d.get("out_dir")
        if path:
            p = Path(path).expanduser()
            folder = p if p.is_dir() else p.parent
            if folder.exists():
                # Prefer internal handler if connected
                try:
                    if self.receivers(self.openRequested) > 0:
                        self.openRequested.emit(str(folder))
                        return
                except Exception:
                    pass
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    
    
    
    def _resolve_input_file(self) -> Optional[Path]:
        d = self.data or {}
        args = d.get("args") or {}
        try:
            src = d.get("input") or args.get("infile") or args.get("input") or args.get("source") or d.get("source")
            if src:
                p = Path(str(src)).expanduser()
                if p.exists() and p.is_file():
                    return p
        except Exception:
            pass
        # Try a few common alt keys
        for k in ("file","path","src","media","input_path","in_path","source_path","filepath","filename"):
            try:
                v = (args.get(k) if isinstance(args, dict) else None) or d.get(k)
                if v:
                    p = Path(str(v)).expanduser()
                    if p.exists() and p.is_file():
                        return p
            except Exception:
                pass
        return None

    def _resolve_output_file(self) -> Optional[Path]:
        d = self.data or {}
        args = d.get("args") or {}
    
        # Identify source (never return this)
        src_path = None
        try:
            src = d.get("input") or args.get("infile") or args.get("input")
            src_path = Path(src).expanduser() if src else None
        except Exception:
            src_path = None
    
        # Helper to normalize a value into a Path (respect out_dir when relative)
        def _as_path(val) -> Optional[Path]:
            if not val:
                return None
            try:
                p = Path(str(val)).expanduser()
                if not p.is_absolute():
                    out_dir = d.get("out_dir") or args.get("out_dir")
                    if out_dir:
                        p = Path(out_dir).expanduser() / p
                return p
            except Exception:
                return None
    
        # Media extensions we care about
        media_exts = {
            '.mp4','.mov','.mkv','.avi','.webm',
            '.mp3','.wav','.flac','.m4a','.aac','.ogg',
            '.gif','.png','.jpg','.jpeg','.bmp','.tif','.tiff'
        }
    
        def _valid_file(p: Optional[Path]) -> Optional[Path]:
            try:
                if not p:
                    return None
                p = p.expanduser()
                if not p.exists() or not p.is_file():
                    return None
                if p.suffix.lower() not in media_exts:
                    return None
                if src_path and p.resolve() == src_path.resolve():
                    return None
                return p
            except Exception:
                return None
    
        # 0) Prefer explicit single-file fields in common schemas
        single_keys = ("produced","outfile","output","result","file","path")
        for k in single_keys:
            p = _valid_file(_as_path((d.get(k) if k in d else args.get(k))))
            if p:
                return p
    
        # 0b) Prefer list fields if present
        list_keys = ("outputs","produced_files","results","files","artifacts","saved")
        for k in list_keys:
            seq = d.get(k) or args.get(k)
            if isinstance(seq, (list, tuple)):
                for item in seq:
                    p = _valid_file(_as_path(item))
                    if p:
                        return p
    
        # 1) If explicit outfile exists but is a directory, look inside that
        out = args.get("outfile") or d.get("outfile")
        out_path = _as_path(out)
        if out_path and out_path.is_dir():
            out_dir = out_path
        else:
            out_dir = _as_path(d.get("out_dir") or args.get("out_dir"))
    
        # 2) Scoring-based selection inside out_dir instead of always picking the newest
        if out_dir and out_dir.is_dir():
            # Reference stems to match against
            ref_stems = set()
            try:
                if src_path:
                    ref_stems.add(src_path.stem)
            except Exception:
                pass
            try:
                if out_path and out_path.is_file():
                    ref_stems.add(out_path.stem)
            except Exception:
                pass
            for ref_key in ("label","title","outname","name","basename"):
                v = (args.get(ref_key) if isinstance(args, dict) else None) or d.get(ref_key)
                if v:
                    try:
                        ref_stems.add(Path(str(v)).stem)
                    except Exception:
                        pass
            for id_key in ("job_id","id","jobid"):
                v = (args.get(id_key) if isinstance(args, dict) else None) or d.get(id_key)
                if v:
                    try:
                        ref_stems.add(str(v))
                    except Exception:
                        pass
    
            # Reference times
            def _as_epoch(val):
                if val is None:
                    return None
                try:
                    return float(val)
                except Exception:
                    pass
                import datetime as _dt
                for fmt in ("%Y-%m-%d %H:%M:%S",):
                    try:
                        return _dt.datetime.strptime(str(val), fmt).timestamp()
                    except Exception:
                        continue
                try:
                    return _dt.datetime.fromisoformat(str(val)).timestamp()
                except Exception:
                    return None
    
            started = _as_epoch(d.get("started_at"))
            finished = _as_epoch(d.get("finished_at"))
            # Fall back to JSON mtime if finished is unknown
            if finished is None:
                try:
                    finished = Path(self.job_path).stat().st_mtime
                except Exception:
                    finished = None
    
            candidates = []
            for p in out_dir.iterdir():
                try:
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in media_exts:
                        continue
                    if src_path and p.resolve() == src_path.resolve():
                        continue
                    st = p.stat()
                    mtime = st.st_mtime
    
                    # Scoring
                    score = 0.0
    
                    # Strong match: exactly matches declared outfile
                    if out_path and out_path.is_file():
                        if p.resolve() == out_path.resolve():
                            score += 100.0
    
                    # Name similarity
                    try:
                        for rs in ref_stems:
                            if not rs:
                                continue
                            if p.stem == rs:
                                score += 50.0
                            elif rs in p.stem:
                                score += 30.0
                    except Exception:
                        pass
    
                    # Time proximity (prefer files finished near the job's finish time)
                    ref_time = finished or started
                    if ref_time is not None:
                        try:
                            minutes = abs(mtime - ref_time) / 60.0
                            # up to +10 when within 0–1 min, tapering to 0 by 10 min
                            time_score = max(0.0, 10.0 - min(10.0, minutes))
                            score += time_score
                        except Exception:
                            pass
    
                    # As a very weak tie-breaker, prefer larger (likely final) files
                    try:
                        size_mb = st.st_size / (1024 * 1024.0)
                        score += min(2.0, size_mb / 512.0)  # +2 at 1GB, tiny otherwise
                    except Exception:
                        pass
    
                    candidates.append((score, mtime, p))
                except Exception:
                    continue
    
            if candidates:
                # sort by score desc, then by mtime desc
                candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
                best = candidates[0][2]
                return best
    
        return None
    def _play_output(self) -> None:
            try:
                p = self._resolve_output_file()
                if p and p.exists():
                    path_str = str(p)
                    # 1) Try internal player like RIFE tab (main.video.open / *.player.open)
                    if self._try_internal_player(path_str):
                        return
                    # 2) If signal is connected, let app handle it
                    try:
                        if self.receivers(self.playRequested) > 0:
                            self.playRequested.emit(path_str)
                            return
                    except Exception:
                        pass
                    # 3) Fallback to OS default player
                    QDesktopServices.openUrl(QUrl.fromLocalFile(path_str))
            except Exception:
                pass
    
    def _view_json(self) -> None:
        try:
            if self.job_path and Path(self.job_path).exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.job_path)))
        except Exception:
            pass

    def _delete_job(self) -> None:
        d = self.data or {}
        args = d.get("args") or {}

        # Identify source for safety
        src = d.get("input") or args.get("infile") or args.get("input")
        src_path = Path(src).expanduser() if src else None

        # Try to delete resolved output file
        try:
            out_file = self._resolve_output_file()
            if out_file and out_file.exists():
                if not src_path or out_file.resolve() != src_path.resolve():
                    try:
                        out_file.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        # Remove the job JSON from the queue
        try:
            if self.job_path.exists():
                self.job_path.unlink()
        except Exception:
            pass

        # Hide/remove this row from UI
        try:
            self.setDisabled(True)
            self.setVisible(False)
            self.deleteLater()
        except Exception:
            pass
