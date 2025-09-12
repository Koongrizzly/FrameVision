
from PySide6.QtCore import QSettings, QByteArray
from PySide6.QtWidgets import QMainWindow, QTabWidget, QSplitter, QGroupBox, QWidget, QTreeView, QTableView
import json

KEY_GEOM = "ui/geometry"
KEY_STATE = "ui/window_state"
KEY_TABS  = "tabs/{name}/index"
KEY_SPLIT = "splitter/{name}/state"
KEY_GBOX  = "collapsible/{name}/checked"
KEY_HDR   = "header/{name}/state"


def _name(w: QWidget, fallback_prefix: str) -> str:
    n = w.objectName()
    if not n:
        # Try user-facing label text first for stability
        try:
            t = getattr(w, "text", None)
            if callable(t):
                txt = t().strip()
                if txt:
                    n = f"{fallback_prefix}_{txt}"
        except Exception:
            pass
    if not n:
        try:
            an = w.accessibleName().strip()
            if an:
                n = f"{fallback_prefix}_{an}"
        except Exception:
            pass
    if not n:
        try:
            wt = w.windowTitle().strip()
            if wt:
                n = f"{fallback_prefix}_{wt}"
        except Exception:
            pass
    if not n:
        # Give anonymous widgets stable-ish names based on class name
        n = f"{fallback_prefix}_{w.__class__.__name__}"
    # Stabilize for future runs
    try:
        if not w.objectName():
            w.setObjectName(n)
    except Exception:
        pass
    return n.replace(' ', '_').replace('/', '_').lower()
def restore_all(root: QWidget) -> None:
    s = QSettings("FrameVision","FrameVision")
    if not s.value("keep_settings_after_restart", True, type=bool):
        return

    # Window geometry/state
    if isinstance(root, QMainWindow):
        geom = s.value(KEY_GEOM, None)
        if isinstance(geom, QByteArray):
            try:
                root.restoreGeometry(geom)
            except Exception:
                pass
        st = s.value(KEY_STATE, None)
        if isinstance(st, QByteArray):
            try:
                root.restoreState(st)
            except Exception:
                pass

    # Tabs
    for tabw in root.findChildren(QTabWidget):
        try:
            key = KEY_TABS.format(name=_name(tabw, "tab"))
            idx = s.value(key, None, type=int)
            if idx is not None and 0 <= idx < tabw.count():
                tabw.setCurrentIndex(idx)
        except Exception:
            pass



    # Common input widgets
    from PySide6.QtWidgets import QCheckBox, QRadioButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit
    for w in root.findChildren(QCheckBox):
        try:
            key = f"check/{{name}}".format(name=_name(w, "check"))
            val = s.value(key, None, type=bool)
            if val is not None:
                w.setChecked(bool(val))
        except Exception:
            pass
    for w in root.findChildren(QRadioButton):
        try:
            key = f"radio/{{name}}".format(name=_name(w, "radio"))
            val = s.value(key, None, type=bool)
            if val is not None:
                w.setChecked(bool(val))
        except Exception:
            pass
    for w in root.findChildren(QComboBox):
        try:
            key = f"combo/{{name}}".format(name=_name(w, "combo"))
            idx = s.value(key, None, type=int)
            if idx is not None and 0 <= idx < w.count():
                w.setCurrentIndex(idx)
        except Exception:
            pass
    for w in root.findChildren(QSpinBox):
        try:
            key = f"spin/{{name}}".format(name=_name(w, "spin"))
            val = s.value(key, None, type=int)
            if val is not None:
                w.setValue(int(val))
        except Exception:
            pass
    for w in root.findChildren(QDoubleSpinBox):
        try:
            key = f"dspin/{{name}}".format(name=_name(w, "dspin"))
            val = s.value(key, None, type=float)
            if val is not None:
                w.setValue(float(val))
        except Exception:
            pass
    for w in root.findChildren(QLineEdit):
        try:
            key = f"line/{{name}}".format(name=_name(w, "line"))
            val = s.value(key, None, type=str)
            if val is not None:
                w.setText(val)
        except Exception:
            pass



    # Common input widgets
    from PySide6.QtWidgets import QCheckBox, QRadioButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit
    for w in root.findChildren(QCheckBox):
        try:
            key = f"check/{{name}}".format(name=_name(w, "check"))
            s.setValue(key, bool(w.isChecked()))
        except Exception:
            pass
    for w in root.findChildren(QRadioButton):
        try:
            key = f"radio/{{name}}".format(name=_name(w, "radio"))
            s.setValue(key, bool(w.isChecked()))
        except Exception:
            pass
    for w in root.findChildren(QComboBox):
        try:
            key = f"combo/{{name}}".format(name=_name(w, "combo"))
            s.setValue(key, w.currentIndex())
        except Exception:
            pass
    for w in root.findChildren(QSpinBox):
        try:
            key = f"spin/{{name}}".format(name=_name(w, "spin"))
            s.setValue(key, int(w.value()))
        except Exception:
            pass
    for w in root.findChildren(QDoubleSpinBox):
        try:
            key = f"dspin/{{name}}".format(name=_name(w, "dspin"))
            s.setValue(key, float(w.value()))
        except Exception:
            pass
    for w in root.findChildren(QLineEdit):
        try:
            key = f"line/{{name}}".format(name=_name(w, "line"))
            s.setValue(key, w.text())
        except Exception:
            pass

    # Splitters
    for spl in root.findChildren(QSplitter):
        try:
            key = KEY_SPLIT.format(name=_name(spl, "split"))
            ba = s.value(key, None)
            if isinstance(ba, QByteArray):
                spl.restoreState(ba)
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
        except Exception:
            pass

    # Custom collapsible sections (QWidget with a checkable QToolButton child)
    try:
        from PySide6.QtWidgets import QToolButton
        for sec in root.findChildren(QWidget):
            try:
                tb = sec.findChild(QToolButton)
                if tb and tb.isCheckable():
                    key = KEY_GBOX.format(name=_name(sec, "gbox"))
                    val = s.value(key, None, type=bool)
                    if val is not None:
                        tb.setChecked(bool(val))
            except Exception:
                pass
    except Exception:
        pass

    # Headers for trees/tables
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
    s = QSettings("FrameVision","FrameVision")
    if not s.value("keep_settings_after_restart", True, type=bool):
        # If disabled, do nothing (we intentionally don't clear keys automatically)
        return

    # Window geometry/state
    if isinstance(root, QMainWindow):
        try:
            s.setValue(KEY_GEOM, root.saveGeometry())
            s.setValue(KEY_STATE, root.saveState())
        except Exception:
            pass

    # Tabs
    for tabw in root.findChildren(QTabWidget):
        try:
            key = KEY_TABS.format(name=_name(tabw, "tab"))
            s.setValue(key, tabw.currentIndex())
        except Exception:
            pass



    # Common input widgets
    from PySide6.QtWidgets import QCheckBox, QRadioButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit
    for w in root.findChildren(QCheckBox):
        try:
            key = f"check/{{name}}".format(name=_name(w, "check"))
            s.setValue(key, bool(w.isChecked()))
        except Exception:
            pass
    for w in root.findChildren(QRadioButton):
        try:
            key = f"radio/{{name}}".format(name=_name(w, "radio"))
            s.setValue(key, bool(w.isChecked()))
        except Exception:
            pass
    for w in root.findChildren(QComboBox):
        try:
            key = f"combo/{{name}}".format(name=_name(w, "combo"))
            s.setValue(key, w.currentIndex())
        except Exception:
            pass
    for w in root.findChildren(QSpinBox):
        try:
            key = f"spin/{{name}}".format(name=_name(w, "spin"))
            s.setValue(key, int(w.value()))
        except Exception:
            pass
    for w in root.findChildren(QDoubleSpinBox):
        try:
            key = f"dspin/{{name}}".format(name=_name(w, "dspin"))
            s.setValue(key, float(w.value()))
        except Exception:
            pass
    for w in root.findChildren(QLineEdit):
        try:
            key = f"line/{{name}}".format(name=_name(w, "line"))
            s.setValue(key, w.text())
        except Exception:
            pass

    # Splitters
    for spl in root.findChildren(QSplitter):
        try:
            key = KEY_SPLIT.format(name=_name(spl, "split"))
            s.setValue(key, spl.saveState())
        except Exception:
            pass

    # Collapsible GroupBoxes (checkable)
    for gb in root.findChildren(QGroupBox):
        try:
            if gb.isCheckable():
                key = KEY_GBOX.format(name=_name(gb, "gbox"))
                s.setValue(key, gb.isChecked())
        except Exception:
            pass

    # Custom collapsible sections (QWidget with a checkable QToolButton child)
    try:
        from PySide6.QtWidgets import QToolButton
        for sec in root.findChildren(QWidget):
            try:
                tb = sec.findChild(QToolButton)
                if tb and tb.isCheckable():
                    key = KEY_GBOX.format(name=_name(sec, "gbox"))
                    s.setValue(key, bool(tb.isChecked()))
            except Exception:
                pass
    except Exception:
        pass

    # Headers for trees/tables
    for view in list(root.findChildren(QTreeView)) + list(root.findChildren(QTableView)):
        try:
            hdr = view.header()
            key = KEY_HDR.format(name=_name(view, "view"))
            s.setValue(key, hdr.saveState())
        except Exception:
            pass


# --- Lightweight QSettings-backed config shim (no global hardcoding) ---------

class ConfigShim(dict):
    """Dict-like wrapper around QSettings that:
    - reads from QSettings on demand
    - uses provided defaults only when a key is missing (first-use)
    - never forces defaults on subsequent runs
    """
    def __init__(self, defaults: dict | None = None, org: str = "FrameVision", app: str = "FrameVision"):
        super().__init__()
        from PySide6.QtCore import QSettings
        self._qs = QSettings(org, app)
        self._defaults = defaults or {}

    # helpers
    def _read(self, key, fallback=None):
        from PySide6.QtCore import QSettings
        v = self._qs.value(key, None)
        if v is None:
            return self._defaults.get(key, fallback)
        # try json decode if it looks like JSON
        if isinstance(v, str):
            s = v.strip()
            if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                try:
                    return json.loads(s)
                except Exception:
                    pass
        return v

    def _write(self, key, value):
        # store complex types as JSON, simple as-is
        if isinstance(value, (dict, list, tuple)):
            self._qs.setValue(key, json.dumps(value))
        else:
            self._qs.setValue(key, value)

    # dict API
    def get(self, key, default=None):
        return self._read(key, default)

    def __getitem__(self, key):
        v = self._read(key)
        if v is None:
            raise KeyError(key)
        return v

    def __setitem__(self, key, value):
        self._write(key, value)

    def update(self, other=None, **kw):
        if other:
            for k, v in dict(other).items():
                self._write(k, v)
        for k, v in kw.items():
            self._write(k, v)

    def setdefault(self, key, default=None):
        existing = self._qs.value(key, None)
        if existing is None:
            self._write(key, default)
            return default
        # decode if json
        if isinstance(existing, str):
            s = existing.strip()
            if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                try:
                    return json.loads(s)
                except Exception:
                    pass
        return existing

    def sync(self):
        # Ensure values are flushed to disk
        try:
            self._qs.sync()
        except Exception:
            pass

