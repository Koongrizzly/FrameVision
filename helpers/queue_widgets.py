from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any, Union

from PySide6.QtCore import QUrl, Qt, QTimer, Signal, QEvent
from PySide6.QtGui import QDesktopServices, QAction, QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QLabel, QToolButton, QProgressBar,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QMenu, QStyle, QFrame
)

ETA_FUDGE_SEC = 3  # small cushion for cleanup (temp deletion, final writes)


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

        # --- Title (bold/monospace) ---
        self.title = QLabel(self._title_text())
        self.title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f = QFont()
        f.setBold(True)
        self.title.setFont(f)
        self.title.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")

        # --- Subtitle (smaller/secondary color) ---
        self.subtitle = QLabel(self._subtitle_text())
        self.subtitle.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.subtitle.setStyleSheet("color: palette(mid); font-size: 11px;")

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

    def _title_text(self) -> str:
        d = self.data or {}
        title = d.get("title")
        if not title:
            title = (d.get("args") or {}).get("label") or self.job_path.stem
        return str(title)

    # Build a compact, informative subtitle
    def _subtitle_text(self) -> str:
        d = self.data or {}
        args = d.get("args") or {}

        parts = []

        # Input filename
        src = d.get("input") or args.get("infile") or args.get("input")
        if src:
            parts.append(Path(str(src)).name)

        # Resolution or transform
        res = self._extract_resolution(d, args)
        if res:
            if parts:
                parts[-1] = f"{parts[-1]} \u2192 {res}"
            else:
                parts.append(res)

        # Model used
        model = d.get("model") or d.get("model_name") or args.get("model") or args.get("ai_model")
        if model:
            parts.append(str(model))

        # Timing info
        status = (self._status_from_fs() or self.status).lower()
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
            # Show ETA immediately after Elapsed when available
            if eta_sec_raw is not None:
                try:
                    eta_display = int(float(eta_sec_raw)) + int(ETA_FUDGE_SEC)
                except Exception:
                    eta_display = None
                if eta_display and eta_display > 0:
                    time_frag.append(f"ETA {self._fmt_dur(eta_display)}")
        elif status in ("done", "failed"):
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

    
    def _resolve_output_file(self) -> Optional[Path]:
        d = self.data or {}
        args = d.get("args") or {}

        # Source (avoid deleting/playing it)
        src_path = None
        try:
            src = d.get("input") or args.get("infile") or args.get("input")
            src_path = Path(src).expanduser() if src else None
        except Exception:
            src_path = None

        # 0) Prefer explicit 'produced' if present
        try:
            produced = d.get("produced")
            if produced:
                p = Path(produced).expanduser()
                if p.exists() and p.is_file() and (not src_path or p.resolve() != src_path.resolve()):
                    return p
        except Exception:
            pass

        # 1) Prefer explicit outfile
        try:
            out = args.get("outfile") or d.get("outfile")
            if out:
                p = Path(out).expanduser()
                if p.exists() and p.is_file() and (not src_path or p.resolve() != src_path.resolve()):
                    return p
        except Exception:
            pass

        # 2) Fallback: search in out_dir for the most relevant file
        try:
            out_dir = d.get("out_dir") or args.get("out_dir")
            if out_dir:
                folder = Path(out_dir).expanduser()
                if folder.is_dir():
                    media_exts = {
                        '.mp4','.mov','.mkv','.avi','.webm',
                        '.mp3','.wav','.flac','.m4a','.aac','.ogg',
                        '.gif','.png','.jpg','.jpeg','.bmp','.tif','.tiff'
                    }
                    cand = None
                    newest = -1.0

                    # Try to bias selection by stem similarity to input/outfile if available
                    ref_stems = set()
                    try:
                        if src_path: ref_stems.add(src_path.stem)
                    except Exception:
                        pass
                    try:
                        if out: ref_stems.add(Path(out).stem)
                    except Exception:
                        pass

                    for p in folder.iterdir():
                        try:
                            if not p.is_file():
                                continue
                            if p.suffix.lower() not in media_exts:
                                continue
                            if src_path and p.resolve() == src_path.resolve():
                                continue
                            score = p.stat().st_mtime
                            # Slight boost if stem matches reference stems
                            try:
                                if p.stem in ref_stems:
                                    score += 0.5
                            except Exception:
                                pass
                            if score > newest:
                                newest = score
                                cand = p
                        except Exception:
                            continue
                    if cand:
                        return cand
        except Exception:
            pass
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
