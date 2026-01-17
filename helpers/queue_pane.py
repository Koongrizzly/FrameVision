from __future__ import annotations

# Extracted from framevision_app.py to reduce coupling between the Queue and other app systems.
# QueuePane is designed to be passive: MainWindow controls playback hooks and calls set_playback_active().

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl, QFileSystemWatcher, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QScrollArea, QSizePolicy, QMessageBox
)

# Collapsible sections used by the Queue tab (same widget as the Tools tab).
try:
    from helpers.tools_tab import CollapsibleSection as ToolsCollapsibleSection
except Exception:
    try:
        from tools_tab import CollapsibleSection as ToolsCollapsibleSection  # type: ignore
    except Exception:
        ToolsCollapsibleSection = None  # type: ignore

try:
    from helpers import state_persist
except Exception:
    state_persist = None

try:
    from helpers.worker_led import WorkerStatusWidget
except Exception:
    from worker_led import WorkerStatusWidget  # type: ignore

class QueuePane(QWidget):
    """
    Queue tab with vertical-only scrolling, responsive header (3 rows), worker LED, and live counters.
    """

    # --- Queue list limits ---

    MAX_PENDING_SHOW = 199

    MAX_DONE_KEEP    = 50

    MAX_DONE_SHOW    = 50

    MAX_FAILED_KEEP  = 50

    MAX_FAILED_SHOW  = 50


    def __init__(self, main, ctx=None, parent=None):
        super().__init__(parent); self.main = main
        self.ctx = ctx or {}
        # Resolve dependencies from ctx or MainWindow to keep QueuePane passive
        try:
            _ctx = self.ctx if isinstance(self.ctx, dict) else {}
        except Exception:
            _ctx = {}
        self.config = _ctx.get("config", getattr(main, "config", {}))
        self.save_config = _ctx.get("save_config", getattr(main, "save_config", None))
        self.BASE = _ctx.get("BASE", getattr(main, "BASE", None))
        self.JOBS_DIRS = _ctx.get("JOBS_DIRS", getattr(main, "JOBS_DIRS", None))
        if self.BASE is None:
            self.BASE = Path.cwd()
        if not isinstance(self.JOBS_DIRS, dict) or not self.JOBS_DIRS:
            self.JOBS_DIRS = {
                "pending": self.BASE / "jobs" / "pending",
                "running": self.BASE / "jobs" / "running",
                "done": self.BASE / "jobs" / "done",
                "failed": self.BASE / "jobs" / "failed",
            }
        if not isinstance(self.config, dict):
            self.config = {}
        if not callable(self.save_config):
            try:
                self.save_config = lambda *a, **k: None
            except Exception:
                pass

        from PySide6.QtCore import QUrl, QFileSystemWatcher

        # Timers (keep internal refresh cadence intact)
        self.auto_timer = QTimer(self); self.auto_timer.setInterval(7500); self.auto_timer.timeout.connect(self.request_refresh)
        self.watch_timer = QTimer(self); self.watch_timer.setInterval(1300); self.watch_timer.timeout.connect(self.request_refresh)
        self.worker_timer = QTimer(self); self.worker_timer.setInterval(4500); self.worker_timer.timeout.connect(self._update_worker_led)

        # Optional safeguard: pause queue refreshing while video is playing.
        # MainWindow owns the QMediaPlayer signal and calls set_playback_active().
        self._q_refresh_paused_for_playback = False
        self._playback_active = False
        # Only refresh when the Queue tab is active/visible (prevents background refresh stutter).
        self._ui_active = False


        # Queue system and paths
        try:
            from helpers.queue_system import QueueSystem
        except Exception:
            from queue_system import QueueSystem
        self.qs = QueueSystem(self.BASE)

        # Root layout and scroll container
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # Fancy banner at the top of the Queue tab
        self.queue_banner = QLabel('Advanced Queue System')
        self.queue_banner.setObjectName('queueBanner')
        self.queue_banner.setAlignment(Qt.AlignCenter)
        self.queue_banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.queue_banner.setFixedHeight(45)
        self.queue_banner.setStyleSheet(
            "#queueBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: #1a2500;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #d4ff66,"
            "   stop:0.5 #a9ff28,"
            "   stop:1 #7acb1f"
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )
        root.addWidget(self.queue_banner)
        root.addSpacing(4)
        topw = QWidget()
        grid = QGridLayout(topw)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        # Header layout: compact + balanced after hiding some legacy buttons.
        # Columns: [left controls] [left controls] [stretch spacer] [right status]
        try:
            grid.setColumnStretch(0, 0)
            grid.setColumnStretch(1, 0)
            grid.setColumnStretch(2, 1)
            grid.setColumnStretch(3, 0)
        except Exception:
            pass

        # Row 1: Refresh · Clear finished/failed (left) · Worker LED (right)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_remove_done = QPushButton("Clear Finished")
        self.btn_remove_failed = QPushButton("Clear Failed")
        clearw = QWidget()
        cl = QHBoxLayout(clearw)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        cl.addWidget(self.btn_remove_done)
        cl.addWidget(self.btn_remove_failed)

        self.worker_status = WorkerStatusWidget()
        self.lbl_worker = self.worker_status.label

        grid.addWidget(self.btn_refresh, 0, 0)
        grid.addWidget(clearw, 0, 1)
        grid.addWidget(self.worker_status, 0, 3, 1, 1, Qt.AlignRight)

        # Row 2: Repair tools (left) · Delete Selected (right)
        self.btn_mark_running_failed = QPushButton("Cancel running job(s)")
        self.btn_reset_running = QPushButton("Move to Pending")
        self.btn_delete_sel = QPushButton("Delete Selected")
        grid.addWidget(self.btn_mark_running_failed, 1, 0)
        grid.addWidget(self.btn_reset_running, 1, 1)
        grid.addWidget(self.btn_delete_sel, 1, 3, 1, 1, Qt.AlignRight)

        # Legacy re-order buttons (kept for compatibility but hidden)
        self.btn_move_up = QPushButton("Move Upwards")
        self.btn_move_down = QPushButton("Move Down")
        self.btn_move_up.setVisible(False); self.btn_move_up.setEnabled(False)
        self.btn_move_down.setVisible(False); self.btn_move_down.setEnabled(False)

        # Row 3: Counters (left) · last refresh timestamp (right)
        self.counts = QLabel("Running 0 | Pending 0 | Done 0 | Failed 0")
        self.last_updated = QLabel("--:--:--")
        grid.addWidget(self.counts, 2, 0, 1, 3, Qt.AlignLeft)
        grid.addWidget(self.last_updated, 2, 3, 1, 1, Qt.AlignRight)

        __import__("helpers.queue_widgets", fromlist=["install_queue_toggle_play_last"]).install_queue_toggle_play_last(
            self, grid, self.config, self.save_config, self.JOBS_DIRS["done"]
        )
        root.addWidget(topw)

        # Scroll area with sections
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff); sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget(); v = QVBoxLayout(content); v.setContentsMargins(0,0,0,0); v.setSpacing(8)

        # Lists (5-row viewport, vertical only)
        self.lst_running = QListWidget(); self._apply_policies(self.lst_running)
        try:
            _h3 = 56*3 + 8
            self.lst_running.setMinimumHeight(_h3)
            self.lst_running.setMaximumHeight(_h3)
        except Exception:
            pass
        self.lst_pending = QListWidget(); self._apply_policies(self.lst_pending)
        self.lst_done = QListWidget(); self._apply_policies(self.lst_done)
        self.lst_failed = QListWidget(); self._apply_policies(self.lst_failed)

        sec_running = ToolsCollapsibleSection("Running", expanded=True)
        lay_sec_running = QVBoxLayout(); lay_sec_running.setContentsMargins(0,0,0,0); lay_sec_running.setSpacing(6)
        lay_sec_running.addWidget(self.lst_running)
        try:
            sec_running.setContentLayout(lay_sec_running)
        except Exception:
            # Fallback if API differs
            sec_running.content = QWidget(); sec_running.content.setLayout(lay_sec_running)
        v.addWidget(sec_running)
        sec_pending = ToolsCollapsibleSection("Pending", expanded=False)
        lay_sec_pending = QVBoxLayout(); lay_sec_pending.setContentsMargins(0,0,0,0); lay_sec_pending.setSpacing(6)
        lay_sec_pending.addWidget(self.lst_pending)
        try:
            sec_pending.setContentLayout(lay_sec_pending)
        except Exception:
            # Fallback if API differs
            sec_pending.content = QWidget(); sec_pending.content.setLayout(lay_sec_pending)
        v.addWidget(sec_pending)
        sec_done = ToolsCollapsibleSection("Finished", expanded=False)
        lay_sec_done = QVBoxLayout(); lay_sec_done.setContentsMargins(0,0,0,0); lay_sec_done.setSpacing(6)
        lay_sec_done.addWidget(self.lst_done)
        try:
            sec_done.setContentLayout(lay_sec_done)
        except Exception:
            # Fallback if API differs
            sec_done.content = QWidget(); sec_done.content.setLayout(lay_sec_done)
        v.addWidget(sec_done)
        sec_failed = ToolsCollapsibleSection("Failed", expanded=False)
        lay_sec_failed = QVBoxLayout(); lay_sec_failed.setContentsMargins(0,0,0,0); lay_sec_failed.setSpacing(6)
        lay_sec_failed.addWidget(self.lst_failed)
        try:
            sec_failed.setContentLayout(lay_sec_failed)
        except Exception:
            # Fallback if API differs
            sec_failed.content = QWidget(); sec_failed.content.setLayout(lay_sec_failed)
        v.addWidget(sec_failed)
        sc.setWidget(content); root.addWidget(sc)

        # File-system watcher for live counters + inserts (debounced <=5 Hz)
        self._fsw = QFileSystemWatcher(self)
        for k in ("pending","running","done","failed"):
            self._fsw.addPath(str(self.JOBS_DIRS[k]))
        self._debounce = QTimer(self); self._debounce.setInterval(180); self._debounce.setSingleShot(True)
        self._fsw.directoryChanged.connect(lambda _:
            (self._debounce.start() if (bool(getattr(self,'_ui_active',False)) and (not self._debounce.isActive())) else None)
        )
        self._debounce.timeout.connect(self._on_queue_changed)

        # Wire actions
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_remove_done.clicked.connect(self.clear_done)
        self.btn_remove_failed.clicked.connect(self.clear_failed)
        self.btn_move_up.clicked.connect(self.move_up)
        self.btn_move_down.clicked.connect(self.move_down)
        self.btn_delete_sel.clicked.connect(self.delete_selected)
        self.btn_reset_running.clicked.connect(self.recover_running_to_pending)
        self.btn_mark_running_failed.clicked.connect(self.cancel_running_jobs)

        # First refresh (no background timers until the tab is actually visible)
        try:
            self._ui_active = False
        except Exception:
            pass
        try:
            self._context_menu_open = False
        except Exception:
            pass
        self.refresh()
        # timers start in showEvent / start_auto() when the Queue tab becomes visible

    def _apply_policies(self, w: QListWidget):
        try:
            w.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            w.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            min_h = 56 * 5 + 8  # ~5 rows
            w.setMinimumHeight(min_h)
            w.setMaximumHeight(16777215)
        except Exception:
            pass


    def _is_main_job_json(self, p: Path) -> bool:
        name = p.name
        if not name.endswith(".json"):
            return False
        if name.endswith(".progress.json") or name.endswith(".json.progress") or name.endswith(".meta.json") or name.startswith("_"):
            return False
        return True


    def _populate(self, folder: Path, widget: QListWidget, status: str):
        from helpers.queue_widgets import JobRowWidget
        # In-place diff to minimize flicker
        try:
            from PySide6.QtCore import QUrl, Qt
        except Exception:
            pass
        existing_keys = {}
        try:
            for i in range(widget.count()):
                it = widget.item(i)
                key = it.data(Qt.UserRole) if 'Qt' in globals() else None
                w = widget.itemWidget(it) if key is None else None
                if not key and w is not None:
                    key = getattr(w, 'path', None) or getattr(w, 'job_path', None) or getattr(w, 'json_path', None)
                if key:
                    existing_keys[str(key)] = i
        except Exception:
            existing_keys = {}
        
        # Build desired file list first (path + sort key)
        files = []
        try:
            # IMPORTANT: reading/parsing hundreds of JSONs on every refresh will hitch video playback.
            # For Done/Failed buckets we preselect the newest candidates by mtime and only parse that subset.
            paths = []
            try:
                for p in folder.glob('*.json'):
                    name = p.name
                    if (name.endswith('.progress.json') or name.endswith('.json.progress') or name.endswith('.progress')
                        or name.endswith('.meta.json') or name.startswith('_')):
                        continue
                    paths.append(p)
            except Exception:
                paths = []

            if status in ("done", "failed"):
                try:
                    show_n = int(getattr(self, "MAX_DONE_SHOW", 20))
                except Exception:
                    show_n = 20
                pre_n = max(50, show_n * 3)
                try:
                    paths.sort(key=lambda _p: _p.stat().st_mtime, reverse=True)
                except Exception:
                    pass
                paths = paths[:pre_n]

            for p in paths:
                name = p.name
                try:
                    d = json.loads(p.read_text(encoding='utf-8') or '{}')
                    if not isinstance(d, dict) or (not d.get('type')) or ((not d.get('input') and not d.get('frames')) and d.get('type')!='txt2img'):
                        continue
                except Exception:
                    continue
                sort_ts = p.stat().st_mtime
                try:
                    from datetime import datetime
                    def _parse(s):
                        if not s: return None
                        for fmt in ("%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M:%S.%f"):
                            try:
                                return datetime.strptime(s, fmt).timestamp()
                            except Exception:
                                pass
                        return None
                    if status == "pending":
                        # For pending jobs, always sort by file mtime so manual reordering (mtime swaps) is reflected.
                        pass
                    elif status == "running":
                        sort_ts = _parse(d.get("started_at")) or sort_ts
                    elif status in ("done","failed"):
                        sort_ts = _parse(d.get("finished_at")) or sort_ts
                except Exception:
                    pass
                files.append((sort_ts, p))
        except Exception:
            files = []

        
        # Apply per-status limits (keep vs show) and stable ordering (newest first)
        max_keep = None
        max_show = None
        if status == "pending":
            max_show = getattr(self, "MAX_PENDING_SHOW", 199)
        elif status == "done":
            max_keep = getattr(self, "MAX_DONE_KEEP", 50)
            max_show = getattr(self, "MAX_DONE_SHOW", 50)
        elif status == "failed":
            max_keep = getattr(self, "MAX_FAILED_KEEP", 50)
            max_show = getattr(self, "MAX_FAILED_SHOW", 50)

        if files:
            try:
                files_sorted = sorted(files, key=lambda t_p: t_p[0], reverse=(status != "pending"))
            except Exception:
                files_sorted = list(files)
        else:
            files_sorted = []

        # For finished/failed queues, prune/move extra JSON files on disk.
        # When "Auto clean-up queue" is enabled, we *move* old job JSONs to:
        #   (root)\jobs\done\old_jobs
        # and we start doing so slightly before hitting the max, to keep the UI fast.
        if max_keep is not None and status in ("done", "failed"):
            try:
                auto_cleanup = bool(self.config.get("queue_auto_cleanup", False))
            except Exception:
                auto_cleanup = False

            if auto_cleanup:
                try:
                    # Start cleanup slightly before hitting the max.
                    # Done: allow up to (max_keep - 2) then remove a batch of 10 so we drop to ~38-39.
                    # Failed: keep gentler cleanup so errors remain visible.
                    if status == "done":
                        trigger_at = max(1, int(max_keep) - 25)  # e.g. 50 -> start at 48
                        max_per_refresh = 5
                    else:
                        trigger_at = max(1, int(max_keep) - 5)  # e.g. 50 -> start at 49
                        max_per_refresh = 3
                except Exception:
                    trigger_at = max_keep
                    max_per_refresh = 3

                moved = 0
                # Only kick in when we reach the trigger threshold.
                if files_sorted and (len(files_sorted) >= int(trigger_at)):
                    try:
                        import shutil as _shutil
                        import time as _time
                        old_dir = (self.BASE / "jobs" / "done" / "old_jobs")
                        old_dir.mkdir(parents=True, exist_ok=True)

                        # Move the oldest jobs first (files_sorted is newest-first).
                        while files_sorted and moved < int(max_per_refresh):
                            victim = files_sorted[-1][1]
                            if victim is None:
                                break
                            try:
                                dest = old_dir / victim.name
                                if dest.exists():
                                    dest = old_dir / f"{victim.stem}_{int(_time.time())}{victim.suffix}"
                                _shutil.move(str(victim), str(dest))
                                files_sorted.pop(-1)
                                moved += 1
                            except Exception:
                                break
                    except Exception:
                        pass

            else:
                # Legacy behavior: hard-delete beyond max_keep
                if len(files_sorted) > max_keep:
                    try:
                        for _ts, p in files_sorted[max_keep:]:
                            try:
                                p.unlink()
                            except Exception:
                                pass
                    except Exception:
                        pass

        if max_show is not None and len(files_sorted) > max_show:
            files = files_sorted[:max_show]
        else:
            files = files_sorted


        # --- Pending safety rebuild ---
        # Qt/PySide can hard-crash (access violation) if we reorder QListWidgetItems that have
        # setItemWidget() row widgets attached. Pending order changes (move up/down) require
        # re-sorting, so for pending we rebuild the list from scratch in a widget-safe way.
        if str(status).lower() == "pending":
            try:
                from PySide6.QtCore import Qt
            except Exception:
                Qt = None  # type: ignore

            # 1) Clear existing items safely: detach row widgets before removing items.
            try:
                i = widget.count() - 1
                while i >= 0:
                    it_old = widget.item(i)
                    try:
                        w_old = widget.itemWidget(it_old)
                    except Exception:
                        w_old = None
                    if w_old is not None:
                        try:
                            widget.removeItemWidget(it_old)
                        except Exception:
                            pass
                        try:
                            w_old.setParent(None)
                        except Exception:
                            pass
                        try:
                            w_old.deleteLater()
                        except Exception:
                            pass
                    try:
                        widget.takeItem(i)
                    except Exception:
                        # Worst-case fallback
                        try:
                            widget.clear()
                        except Exception:
                            pass
                        break
                    i -= 1
            except Exception:
                try:
                    widget.clear()
                except Exception:
                    pass

            # 2) Rebuild in the desired order.
            for _idx, (_ts, p) in enumerate(files):
                it = QListWidgetItem("")
                w = JobRowWidget(str(p), status)
                try:
                    hint = w.sizeHint()
                    if hint.height() < 56:
                        hint.setHeight(56)
                    it.setSizeHint(hint)
                except Exception:
                    try:
                        it.setSizeHint(w.sizeHint())
                    except Exception:
                        pass

                try:
                    widget.addItem(it)
                except Exception:
                    # fallback
                    try:
                        widget.insertItem(_idx, it)
                    except Exception:
                        pass

                try:
                    widget.setItemWidget(it, w)
                except Exception:
                    pass

                try:
                    if Qt is not None:
                        it.setData(Qt.UserRole, str(p))
                except Exception:
                    pass

                # Wire pending order change → safe refresh (debounced, queued).
                try:
                    if hasattr(w, "queueOrderChanged"):
                        from PySide6.QtCore import QTimer
                        w.queueOrderChanged.connect(lambda *a: QTimer.singleShot(0, self.request_refresh))
                except Exception:
                    pass

            return

        # Remove stale items not in target set (in-place, from bottom)
        try:
            from PySide6.QtCore import QUrl, Qt
            target = {str(p) for _, p in files}
            i = widget.count() - 1
            while i >= 0:
                it = widget.item(i)
                key = None
                try:
                    key = it.data(Qt.UserRole)
                except Exception:
                    key = None
                if key is None:
                    w = widget.itemWidget(it)
                    key = getattr(w, 'path', None) if w is not None else None
                if (key is None) or (str(key) not in target):
                    widget.takeItem(i)
                i -= 1
        except Exception:
            pass

        
        # Add or refresh rows (preserve order by newest first) — and **reposition** to keep newest on top live.
        for idx, (_ts, p) in enumerate(files):
            try:
                from PySide6.QtCore import QUrl, Qt
            except Exception:
                pass

            found = False
            found_item = None
            found_widget = None
            found_row = None

            # Locate existing row for this path (if any)
            try:
                for _i in range(widget.count()):
                    _it = widget.item(_i)
                    _key = _it.data(Qt.UserRole)
                    if _key is None:
                        _w = widget.itemWidget(_it)
                        _key = getattr(_w, 'path', None) if _w is not None else None
                    if str(_key) == str(p):
                        found = True
                        found_item = _it
                        found_widget = widget.itemWidget(_it)
                        found_row = _i
                        break
            except Exception:
                found = False

            if found:
                # Refresh existing row's contents
                try:
                    if hasattr(found_widget, 'refresh'):
                        found_widget.refresh()
                except Exception:
                    pass
                # Reposition item if order changed
                try:
                    if found_row is not None and found_row != idx:
                        it_take = widget.takeItem(found_row)
                        # Reinsert at the correct index and reattach the widget
                        widget.insertItem(idx, it_take)
                        if found_widget is not None:
                            widget.setItemWidget(it_take, found_widget)
                        try:
                            it_take.setData(Qt.UserRole, str(p))
                        except Exception:
                            pass
                except Exception:
                    pass
                continue

            # If not found, create a new row and insert at the correct position
            it = QListWidgetItem("")
            w = JobRowWidget(str(p), status)
            try:
                hint = w.sizeHint()
                if hint.height() < 56:
                    from PySide6.QtCore import QUrl, QSize
                    hint.setHeight(56)
                it.setSizeHint(hint)
            except Exception:
                it.setSizeHint(w.sizeHint())
            try:
                widget.insertItem(idx, it)
            except Exception:
                widget.addItem(it)  # fallback
            widget.setItemWidget(it, w)
            try:
                from PySide6.QtCore import QUrl, Qt
                it.setData(Qt.UserRole, str(p))
            except Exception:
                pass


    def set_playback_active(self, playing: bool):
        """Called by MainWindow when the internal video player starts/stops.

        QueuePane never reaches into the media player directly; it only pauses/resumes
        its own timers based on this flag (if enabled in config).
        """
        try:
            self._playback_active = bool(playing)
        except Exception:
            self._playback_active = False

        try:
            try:
                _pause_on_play = bool(self.config.get("queue_pause_refresh_while_playing", True))
            except Exception:
                _pause_on_play = True

            if not _pause_on_play:
                # Safeguard disabled: ensure we are not stuck paused from a previous session.
                if bool(getattr(self, "_q_refresh_paused_for_playback", False)):
                    self._q_refresh_paused_for_playback = False
                    self.start_auto()
                return

            if self._playback_active:
                self._q_refresh_paused_for_playback = True
                if bool(getattr(self, '_ui_active', False)):
                    self.stop_auto()
            else:
                if bool(getattr(self, "_q_refresh_paused_for_playback", False)):
                    self._q_refresh_paused_for_playback = False
                    if bool(getattr(self, '_ui_active', False)):
                        self.start_auto()
        except Exception:
            pass

    def _is_video_playing(self) -> bool:
        """Best-effort check of the internal player state.

        This is a safety net: if signal wiring breaks after a VideoPane rebuild,
        we still avoid queue refresh while the user is watching a video.
        """
        try:
            from PySide6.QtMultimedia import QMediaPlayer as _QMP
            v = getattr(self.main, "video", None)
            p = getattr(v, "player", None) if v is not None else None
            if p is None:
                return False
            return (p.playbackState() == _QMP.PlayingState)
        except Exception:
            return False

    def _enter_context_menu(self):
        """Called by row widgets before opening a right-click menu."""
        try:
            self._context_menu_open = True
        except Exception:
            pass
        # Stop timers immediately; rebuilds during menu.exec() can close the menu.
        try:
            self.auto_timer.stop()
        except Exception:
            pass
        try:
            self.watch_timer.stop()
        except Exception:
            pass
        try:
            self.worker_timer.stop()
        except Exception:
            pass

    def _exit_context_menu(self):
        """Called by row widgets after a right-click menu closes."""
        try:
            self._context_menu_open = False
        except Exception:
            pass
        # Resume only if we are visible and not paused for playback.
        try:
            if bool(getattr(self, "_ui_active", False)) and (not bool(getattr(self, "_q_refresh_paused_for_playback", False))):
                self.start_auto()
        except Exception:
            pass

    def showEvent(self, e):
        try:
            super().showEvent(e)
        except Exception:
            pass
        # Queue tab became visible: enable auto refresh.
        try:
            self._ui_active = True
        except Exception:
            pass
        try:
            self.start_auto()
        except Exception:
            try:
                self.request_refresh()
            except Exception:
                pass

    def hideEvent(self, e):
        # Queue tab hidden: stop background activity.
        try:
            self._ui_active = False
        except Exception:
            pass
        try:
            self.stop_auto()
        except Exception:
            pass
        try:
            super().hideEvent(e)
        except Exception:
            pass


    def request_refresh(self):
        try:
            # If the Queue tab isn't active, ignore background filesystem/timer signals.
            if not bool(getattr(self, '_ui_active', False)):
                return
            # Pause refresh while a right-click context menu is open (prevents menu closing / selection jumps).
            if bool(getattr(self, '_context_menu_open', False)):
                return
            # Optional safeguard: pause queue refresh while video is playing (prevents playback stutter).
            try:
                _pause_on_play = bool(self.config.get("queue_pause_refresh_while_playing", True))
            except Exception:
                _pause_on_play = True

            if _pause_on_play:
                # If video is currently playing, keep the queue refresh paused.
                # (This is robust against VideoPane rebuilding the QMediaPlayer instance.)
                playing_now = bool(getattr(self, '_playback_active', False))
                try:
                    if hasattr(self, '_is_video_playing'):
                        playing_now = playing_now or bool(self._is_video_playing())
                except Exception:
                    pass

                if playing_now:
                    try:
                        if not bool(getattr(self, "_q_refresh_paused_for_playback", False)):
                            self._q_refresh_paused_for_playback = True
                            self.stop_auto()
                    except Exception:
                        pass
                    return

                # If we were paused due to playback but the player isn't playing now, unpause.
                if bool(getattr(self, "_q_refresh_paused_for_playback", False)):
                    try:
                        self._q_refresh_paused_for_playback = False
                        self.start_auto()
                    except Exception:
                        pass
            else:
                # Safeguard disabled: ensure we are not stuck in a paused state.
                if bool(getattr(self, "_q_refresh_paused_for_playback", False)):
                    try:
                        self._q_refresh_paused_for_playback = False
                        self.start_auto()
                    except Exception:
                        pass


            # Debounce/coalesce refresh calls

            if not hasattr(self, '_refresh_coalesce'):

                from PySide6.QtCore import QUrl, QTimer

                self._refresh_coalesce = QTimer(self)

                self._refresh_coalesce.setSingleShot(True)

                self._refresh_coalesce.setInterval(350)

                self._refresh_coalesce.timeout.connect(self._do_refresh)

            try:
                if not self._refresh_coalesce.isActive():
                    self._refresh_coalesce.start()
            except Exception:
                self._refresh_coalesce.start()

        except Exception:

            # Fallback: direct refresh

            self.refresh()


    def _do_refresh(self):

        try:

            self.refresh()

        except Exception:

            pass
    def refresh(self):
        # Avoid UI hitches: block repaints while we rebuild lists.
        try:
            if bool(getattr(self, "_context_menu_open", False)):
                return
        except Exception:
            pass

        try:
            _pause_on_play = bool(self.config.get("queue_pause_refresh_while_playing", True))
        except Exception:
            _pause_on_play = True

        if _pause_on_play:
            try:
                playing_now = bool(getattr(self, "_playback_active", False))
                try:
                    if hasattr(self, "_is_video_playing"):
                        playing_now = playing_now or bool(self._is_video_playing())
                except Exception:
                    pass
                if playing_now:
                    return
            except Exception:
                pass

        try:
            self.setUpdatesEnabled(False)
            for _lst in (getattr(self, "lst_running", None), getattr(self, "lst_pending", None),
                         getattr(self, "lst_done", None), getattr(self, "lst_failed", None)):
                try:
                    if _lst is not None:
                        _lst.setUpdatesEnabled(False)
                except Exception:
                    pass

            self._populate(self.JOBS_DIRS['running'], self.lst_running, 'running')
            self._populate(self.JOBS_DIRS['pending'], self.lst_pending, 'pending')
            self._populate(self.JOBS_DIRS['done'], self.lst_done, 'done')
            self._populate(self.JOBS_DIRS['failed'], self.lst_failed, 'failed')
            self._update_counts_label()
            self._update_worker_led()
            try:
                from datetime import datetime as _dt
                if hasattr(self, 'last_updated'):
                    self.last_updated.setText(_dt.now().strftime("%H:%M:%S"))
            except Exception:
                pass
        finally:
            try:
                for _lst in (getattr(self, "lst_running", None), getattr(self, "lst_pending", None),
                             getattr(self, "lst_done", None), getattr(self, "lst_failed", None)):
                    try:
                        if _lst is not None:
                            _lst.setUpdatesEnabled(True)
                    except Exception:
                        pass
                self.setUpdatesEnabled(True)
            except Exception:
                pass

    def _update_counts_label(self):
        try:
            r = sum(1 for pth in self.JOBS_DIRS["running"].glob("*.json") if self._is_main_job_json(pth))
            p = sum(1 for pth in self.JOBS_DIRS["pending"].glob("*.json") if self._is_main_job_json(pth))
            d = sum(1 for pth in self.JOBS_DIRS["done"].glob("*.json") if self._is_main_job_json(pth))
            f = sum(1 for pth in self.JOBS_DIRS["failed"].glob("*.json") if self._is_main_job_json(pth))
            done_show = getattr(self, "MAX_DONE_SHOW", 20)
            done_txt = f"Done {d}" if d <= done_show else f"Done {d} (showing {done_show})"
            self.counts.setText(f"Running {r} | Pending {p} | {done_txt} | Failed {f}")
        except Exception:
            self.counts.setText("Running ? | Pending ? | Done ? | Failed ?")

    def _watch_tick(self):
        try:
            self.request_refresh()
        except Exception:
            pass

    def _on_queue_changed(self):
        # Use debounced refresh to avoid thrash
        self.request_refresh()

    # --- Queue actions (filesystem-based; minimal and safe) ---
    def _selected_job_path(self):
        # Return (bucket, Path) for the currently selected row; None if nothing selected
        try_lists = [
            ("running", self.lst_running),
            ("pending", self.lst_pending),
            ("done", self.lst_done),
            ("failed", self.lst_failed),
        ]
        for bucket, lst in try_lists:
            it = lst.currentItem()
            if it is None:
                continue
            try:
                w = lst.itemWidget(it)
                p = Path(getattr(w, "path", "") or getattr(w, "job_path", ""))
                if p and p.exists():
                    return bucket, p
            except Exception:
                pass
        return None, None

    def clear_done(self):
        try:
            for p in list(self.JOBS_DIRS["done"].glob("*.json")):
                try: p.unlink()
                except Exception: pass
        except Exception:
            pass
        self.refresh()

    def clear_failed(self):
        try:
            try:
                self.qs.remove_failed()
            except Exception:
                for p in list(self.JOBS_DIRS["failed"].glob("*.json")):
                    try: p.unlink()
                    except Exception: pass
        except Exception:
            pass
        self.refresh()

    def clear_done_failed(self):
        try:
            try:
                self.qs.clear_finished_failed()
            except Exception:
                for p in list(self.JOBS_DIRS["done"].glob("*.json")):
                    try: p.unlink()
                    except Exception: pass
                for p in list(self.JOBS_DIRS["failed"].glob("*.json")):
                    try: p.unlink()
                    except Exception: pass
        except Exception:
            pass
        self.refresh()

    def delete_selected(self):
        bucket, p = self._selected_job_path()
        if not p:
            return
        try:
            p.unlink()
        except Exception:
            pass
        self.refresh()

    def move_up(self):
        bucket, p = self._selected_job_path()
        if bucket != "pending" or not p:
            return
        try:
            now = time.time()
            os.utime(p, (now, now + 60))
        except Exception:
            pass
        self.refresh()

    def move_down(self):
        bucket, p = self._selected_job_path()
        if bucket != "pending" or not p:
            return
        try:
            now = time.time()
            os.utime(p, (now, now - 60))
        except Exception:
            pass
        self.refresh()

    def recover_running_to_pending(self):
        bucket, p = self._selected_job_path()
        if bucket != "running" or not p:
            return
        try:
            dest = self.JOBS_DIRS["pending"] / p.name
            try:
                d = json.loads(p.read_text(encoding="utf-8") or "{}")
                for k in ("started_at","finished_at","ended_at","duration_sec","error"):
                    if k in d: d.pop(k, None)
                p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            except Exception:
                pass
            import shutil as _shutil
            _shutil.move(str(p), str(dest))
            try:
                self.qs.nudge_pending()
            except Exception:
                pass
        except Exception:
            pass
        self.refresh()


    def cancel_running_jobs(self):
        """Cancel selected running job(s) (or all running jobs if none selected).

        This mirrors the Running-row right-click 'Cancel job' action:
          - Write <job>.json.cancel marker
          - Set job['cancel_requested'] = True

        The worker is responsible for killing the underlying process and moving the job to failed.
        """
        try:
            from pathlib import Path
            from PySide6.QtCore import Qt
        except Exception:
            Path = None
            Qt = None

        paths = []
        # Prefer selected items in the Running list
        try:
            if Qt is not None:
                for it in self.lst_running.selectedItems():
                    k = it.data(Qt.UserRole)
                    if k:
                        paths.append(Path(str(k)))
        except Exception:
            paths = []

        # Fallback: cancel all running jobs
        if not paths:
            try:
                paths = [p for p in self.JOBS_DIRS["running"].glob("*.json")
                         if getattr(self, "_is_main_job_json", lambda x: True)(p)]
            except Exception:
                paths = []

        for p in paths:
            try:
                if not p or (not p.exists()):
                    continue
            except Exception:
                continue

            # Marker file: <job>.json.cancel
            try:
                marker = p.with_suffix(p.suffix + ".cancel")
                marker.write_text("cancel", encoding="utf-8")
            except Exception:
                pass

            # JSON flag: cancel_requested = True
            try:
                d = {}
                try:
                    d = json.loads(p.read_text(encoding="utf-8") or "{}")
                except Exception:
                    d = {}
                d["cancel_requested"] = True
                try:
                    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            except Exception:
                pass

        self.refresh()


    def mark_running_failed(self):
        bucket, p = self._selected_job_path()
        if bucket != "running" or not p:
            return
        try:
            d = {}
            try:
                d = json.loads(p.read_text(encoding="utf-8") or "{}")
            except Exception:
                d = {}
            try:
                from datetime import datetime
                d["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            d["error"] = d.get("error") or "Manually marked as failed"
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
            import shutil as _shutil
            dest = self.JOBS_DIRS["failed"] / p.name
            _shutil.move(str(p), str(dest))
        except Exception:
            pass
        self.refresh()

    def _update_worker_led(self):
        try:
            try:
                running_count = sum(1 for pth in self.JOBS_DIRS["running"].glob("*.json")
                                    if getattr(self, "_is_main_job_json", lambda p: True)(pth))
            except Exception:
                running_count = 0

            import time
            hb = globals().get('HEARTBEAT_PATH', self.BASE / 'logs' / 'worker_heartbeat.txt')
            age = None
            if hb.exists():
                try:
                    age = time.time() - hb.stat().st_mtime
                except Exception:
                    age = None

            if running_count > 0:
                self.worker_status.set_state("running", f"{running_count} active job(s)")
            elif age is not None and age < 12.0:
                self.worker_status.set_state("idle", f"Heartbeat {int(age)}s ago")
            elif age is not None:
                self.worker_status.set_state("stopped", f"No heartbeat for {int(age)}s")
            else:
                self.worker_status.set_state("stopped", "No heartbeat file")
        except Exception as e:
            try:
                self.worker_status.set_state("error", str(e))
            except Exception:
                pass
    def stop_auto(self):
        try:
            self.auto_timer.stop()
        except Exception:
            pass
        try:
            self.watch_timer.stop()
        except Exception:
            pass
        try:
            self.worker_timer.stop()
        except Exception:
            pass

    def start_auto(self):
        # Start/Restart timers safely and schedule a debounced refresh
        try:
            self._ui_active = True
        except Exception:
            pass

        # Respect context-menu + playback pause flags
        try:
            if bool(getattr(self, '_context_menu_open', False)):
                return
        except Exception:
            pass
        try:
            if bool(getattr(self, '_q_refresh_paused_for_playback', False)):
                return
        except Exception:
            pass

        try:
            self.auto_timer.start()
        except Exception:
            pass
        try:
            self.watch_timer.start()
        except Exception:
            pass
        try:
            self.worker_timer.start()
        except Exception:
            pass
        try:
            self.request_refresh()
        except Exception:
            try:
                self.refresh()
            except Exception:
                pass


    def closeEvent(self, e):
        try:
            state_persist.save_all(self)
        except Exception:
            pass
        try:
            self.stop_auto()
        except Exception:
            pass
        try:
            super().closeEvent(e)
        except Exception:
            pass

