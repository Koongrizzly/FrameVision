# FrameVision — state_persist.py (v3 stable-keys + live-splitter-save)
# Fixes:
# - Use **stable widget keys** (derived from the widget's path in the parent/child tree)
#   instead of object IDs, so settings persist across restarts even if objectName is unset.
# - Add live saving for QSplitter via splitterMoved so the side pane width sticks immediately.
# - Tabs + collapsibles still save on change; restore only applies when a value exists.
#
from __future__ import annotations

from PySide6.QtCore import QSettings, QByteArray, Qt
from PySide6.QtWidgets import (
    QWidget, QMainWindow, QTabWidget, QSplitter, QGroupBox,
    QTreeView, QTableView, QToolButton
)
import json
import typing as _t

ORG = "FrameVision"
APP = "FrameVision"

KEY_GEOM = "ui/geometry"
KEY_STATE = "ui/window_state"
KEY_TABS  = "tabs/{name}/index"
KEY_SPLIT = "splitter/{name}/state"
KEY_GBOX  = "collapsible/{name}/checked"
KEY_HDR   = "header/{name}/state"
KEY_KEEP  = "keep_settings_after_restart"   # bool, default True
KEY_START_FORCE = "startup/force_mode"      # "maximized" or "fullscreen"

def _qs() -> QSettings:
    return QSettings(ORG, APP)

def _keep_enabled() -> bool:
    try:
        val = _qs().value(KEY_KEEP, True, type=bool)
        return True if val is None else bool(val)
    except Exception:
        return True

def _sanitize(s: str) -> str:
    return (s or "").replace(" ", "_").replace("/", "_").replace("\\", "_").lower()

def _class_name(w: QWidget) -> str:
    try:
        return w.metaObject().className() or w.__class__.__name__
    except Exception:
        return w.__class__.__name__

def _stable_path_name(w: QWidget, prefix: str) -> str:
    """Build a deterministic name based on the widget's position in the tree.
    Format: prefix + "__MainWindow:0__QSplitter:1__QTabWidget:0" (sanitized lowercase).
    """
    parts = []
    cur: QWidget | None = w
    while cur is not None and isinstance(cur, QWidget):
        cls = _class_name(cur)
        parent = cur.parentWidget()
        idx = 0
        if parent is not None:
            # index among parent's QWidget children with same class
            siblings = [c for c in parent.children() if isinstance(c, QWidget) and _class_name(c) == cls]
            try:
                idx = siblings.index(cur)
            except ValueError:
                idx = 0
        parts.append(f"{cls}:{idx}")
        cur = parent
    parts.reverse()
    name = prefix + "__" + "__".join(parts)
    return _sanitize(name)

def _name(w: QWidget, fallback_prefix: str) -> str:
    n = (w.objectName() or "").strip()
    if n:
        return _sanitize(n)
    # No objectName -> use stable path
    return _stable_path_name(w, fallback_prefix)

def restore_all(root: QWidget) -> None:
    if not _keep_enabled():
        return
    s = _qs()

    # Main window geometry/state (with optional forced mode)
    if isinstance(root, QMainWindow):
        mode = s.value(KEY_START_FORCE, "maximized")
        mode = (str(mode).strip().lower() if mode is not None else "maximized")
        forced = False
        try:
            if mode in ("fullscreen", "full", "fs"):
                root.setWindowState(root.windowState() | Qt.WindowFullScreen)
                forced = True
            elif mode in ("maximized", "max"):
                root.setWindowState(root.windowState() | Qt.WindowMaximized)
                forced = True
        except Exception:
            pass
        if not forced:
            geom = s.value(KEY_GEOM, None)
            if isinstance(geom, QByteArray):
                try: root.restoreGeometry(geom)
                except Exception: pass
            st = s.value(KEY_STATE, None)
            if isinstance(st, QByteArray):
                try: root.restoreState(st)
                except Exception: pass

    # Tabs — restore saved index only; then save on change
    for tabw in root.findChildren(QTabWidget):
        try:
            key = KEY_TABS.format(name=_name(tabw, "tab"))
            idx = s.value(key, None, type=int)
            if idx is not None and 0 <= idx < tabw.count():
                tabw.setCurrentIndex(idx)
            def _on_tab_changed(i, _key=key):
                qs = _qs()
                qs.setValue(_key, int(i))
                qs.sync()
            tabw.currentChanged.connect(_on_tab_changed)
        except Exception:
            pass

    # Splitters — restore + live-save when moved
    for spl in root.findChildren(QSplitter):
        try:
            key = KEY_SPLIT.format(name=_name(spl, "split"))
            ba = s.value(key, None)
            if isinstance(ba, QByteArray):
                spl.restoreState(ba)
            def _on_splitter_moved(*_args, _spl=spl, _key=key):
                qs = _qs()
                try:
                    qs.setValue(_key, _spl.saveState())
                    qs.sync()
                except Exception:
                    pass
            spl.splitterMoved.connect(_on_splitter_moved)
        except Exception:
            pass

    # Collapsible GroupBoxes (checkable)
    for gb in root.findChildren(QGroupBox):
        try:
            if gb.isCheckable():
                key = KEY_GBOX.format(name=_name(gb, "gbox"))
                val = s.value(key, None, type=bool)
                if val is not None:
                    gb.setChecked(bool(val))
                def _on_gb_toggled(v, _key=key):
                    qs = _qs()
                    qs.setValue(_key, bool(v))
                    qs.sync()
                gb.toggled.connect(_on_gb_toggled)
        except Exception:
            pass

    # Custom collapsible sections (any widget containing a checkable QToolButton)
    for sec in root.findChildren(QWidget):
        try:
            tb = sec.findChild(QToolButton)
            if tb and tb.isCheckable():
                key = KEY_GBOX.format(name=_name(sec, "gbox"))
                val = s.value(key, None, type=bool)
                if val is not None:
                    tb.setChecked(bool(val))
                def _on_tb_toggled(v, _key=key):
                    qs = _qs()
                    qs.setValue(_key, bool(v))
                    qs.sync()
                tb.toggled.connect(_on_tb_toggled)
        except Exception:
            pass

    # Header state
    for view in list(root.findChildren(QTreeView)) + list(root.findChildren(QTableView)):
        try:
            hdr = view.header()
            key = KEY_HDR.format(name=_name(view, "view"))
            ba = s.value(key, None)
            if isinstance(ba, QByteArray):
                hdr.restoreState(ba)
        except Exception:
            pass

def save_all(root: QWidget) -> None:
    if not _keep_enabled():
        return
    s = _qs()

    if isinstance(root, QMainWindow):
        try: s.setValue(KEY_GEOM, root.saveGeometry())
        except Exception: pass
        try: s.setValue(KEY_STATE, root.saveState())
        except Exception: pass

    # Tabs
    for tabw in root.findChildren(QTabWidget):
        try:
            key = KEY_TABS.format(name=_name(tabw, "tab"))
            s.setValue(key, int(tabw.currentIndex()))
        except Exception:
            pass

    # Splitters
    for spl in root.findChildren(QSplitter):
        try:
            key = KEY_SPLIT.format(name=_name(spl, "split"))
            s.setValue(key, spl.saveState())
        except Exception:
            pass

    # Collapsible GroupBoxes
    for gb in root.findChildren(QGroupBox):
        try:
            if gb.isCheckable():
                key = KEY_GBOX.format(name=_name(gb, "gbox"))
                s.setValue(key, bool(gb.isChecked()))
        except Exception:
            pass

    # Custom collapsible sections
    for sec in root.findChildren(QWidget):
        try:
            tb = sec.findChild(QToolButton)
            if tb and tb.isCheckable():
                key = KEY_GBOX.format(name=_name(sec, "gbox"))
                s.setValue(key, bool(tb.isChecked()))
        except Exception:
            pass

    # Headers
    for view in list(root.findChildren(QTreeView)) + list(root.findChildren(QTableView)):
        try:
            hdr = view.header()
            key = KEY_HDR.format(name=_name(view, "view"))
            s.setValue(key, hdr.saveState())
        except Exception:
            pass
    try: s.sync()
    except Exception: pass

class SettingsHelper:
    def __init__(self, settings: QSettings | None = None) -> None:
        self._qs = settings or _qs()
    def set_json(self, key: str, value: _t.Any) -> None:
        try:
            self._qs.setValue(key, json.dumps(value)); self._qs.sync()
        except Exception: pass
    def get_json(self, key: str, default: _t.Any = None) -> _t.Any:
        existing = self._qs.value(key, None)
        if existing is None: return default
        try:
            s = str(existing)
            if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                return json.loads(s)
        except Exception: pass
        return default
    def sync(self) -> None:
        try: self._qs.sync()
        except Exception: pass
