
# helpers/settings_tab.py — canonical Settings builder for FrameVision (v0.7.6)
from __future__ import annotations
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtWidgets import (
    QWidget, QTabWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QComboBox, QPushButton, QCheckBox, QSizePolicy
)

try:
    from helpers.framevision_app import apply_theme, config, save_config
except Exception:
    apply_theme=None; config={}

# ---- helpers to locate the Settings page -------------------------------------------------
def _locate_settings_container(root: QWidget) -> QWidget | None:
    if root is None:
        return None
    tabs = getattr(root, "tabs", None)
    if isinstance(tabs, QTabWidget):
        for i in range(tabs.count()):
            if "setting" in (tabs.tabText(i) or "").lower():
                w = tabs.widget(i)
                return w.widget() if isinstance(w, QScrollArea) else w
    # fallback: direct child named 'settings'
    for w in root.findChildren(QWidget) or []:
        if (w.objectName() or "").lower() == "settings":
            return w
    return None

def _ensure_vbox(page: QWidget) -> QVBoxLayout:
    lay = page.layout()
    if not isinstance(lay, QVBoxLayout):
        lay = QVBoxLayout(page)
        lay.setContentsMargins(12,12,12,12); lay.setSpacing(12)
        page.setLayout(lay)
    return lay

def _wipe_layout(lay: QVBoxLayout) -> None:
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w:
            w.setParent(None)

# ---- component builders ------------------------------------------------------------------
def _theme_row(page: QWidget) -> QWidget:
    row = QWidget(page)
    h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
    lab = QLabel("Theme:")
    box = QComboBox(); box.addItems(["Day","Solarized Light","Sunburst","Evening","Night","Slate","High Contrast","Cyberpunk","Neon","Ocean","CRT","Aurora","Mardi Gras","Tropical Fiesta","Color Mix","Random","Auto"])
    try:
        cur = (config.get("theme") or "Auto"); idx = box.findText(cur); box.setCurrentIndex(max(0, idx))
    except Exception: pass
    btn = QPushButton("Apply theme")
    def do_apply():
        try:
            t = box.currentText(); config["theme"] = t
            app = QtWidgets.QApplication.instance()
            if apply_theme and app:
                apply_theme(app, t)
            try:
                save_config()
            except Exception:
                pass
        except Exception:
            pass
    btn.clicked.connect(do_apply); box.currentIndexChanged.connect(lambda _i: do_apply())
    h.addWidget(lab); h.addWidget(box); h.addWidget(btn); h.addStretch(1)
    return row

def _options_group(page: QWidget) -> QGroupBox:
    g = QGroupBox("Options", page)
    v = QVBoxLayout(g); v.setContentsMargins(0,0,0,0); v.setSpacing(6)
    s = QSettings("FrameVision","FrameVision")

    cb1 = QCheckBox("Show random intro image on startup")
    cb1.setChecked(s.value("intro_enabled", True, type=bool))
    cb1.toggled.connect(lambda b: s.setValue("intro_enabled", bool(b)))

    cb2 = QCheckBox("Intro follows theme routine (Day/Evening/Night)")
    cb2.setChecked(s.value("intro_follow_theme", False, type=bool))
    cb2.toggled.connect(lambda b: s.setValue("intro_follow_theme", bool(b)))

    v.addWidget(cb1); v.addWidget(cb2);
    cb2.setEnabled(cb1.isChecked())
    cb1.toggled.connect(cb2.setEnabled)
    # --- Startup toggles ----------------------------------------------------
    cb_clear_pyc = QCheckBox(r"Clear app Python cache files in /__pycache__ at (re)start")
    cb_clear_pyc.setChecked(s.value("clear_pyc_on_start", False, type=bool))
    cb_clear_pyc.toggled.connect(lambda b: s.setValue("clear_pyc_on_start", bool(b)))

    cb_keep_settings = QCheckBox("Keep all used settings after restart")
    cb_keep_settings.setChecked(s.value("keep_settings_after_restart", True, type=bool))
    cb_keep_settings.toggled.connect(lambda b: s.setValue("keep_settings_after_restart", bool(b)))

    v.addWidget(cb_clear_pyc)
    v.addWidget(cb_keep_settings)

            # Temperature units (C/F)
    row = QtWidgets.QWidget(g)
    h2 = QtWidgets.QHBoxLayout(row); h2.setContentsMargins(0,4,0,0); h2.setSpacing(8)
    h2.addWidget(QtWidgets.QLabel("Temperature units:"))
    combo = QtWidgets.QComboBox(row); combo.addItem("Celsius (°C)", "C"); combo.addItem("Fahrenheit (°F)", "F")
    try:
        from helpers.framevision_app import config as _cfg
        cur = (_cfg.get("temp_units","C") or "C").upper()
    except Exception:
        cur = "C"
    combo.setCurrentIndex(0 if cur == "C" else 1)
    def _apply_units():
        try:
            from helpers.framevision_app import config as _cfg, save_config as _save
            val = combo.currentData() or "C"
            _cfg["temp_units"] = val
            _save()
        except Exception:
            pass
    combo.currentIndexChanged.connect(lambda _i: _apply_units())
    h2.addWidget(combo); h2.addStretch(1)
    v.addWidget(row)
    # (folder picker omitted in this build)
    v.addStretch(0)
    return g

def _buttons_row(page: QWidget) -> QWidget:
    row = QWidget(page)
    h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)

    # Clear cache
    btn_cache = QPushButton("Clear program cache…")
    def do_cache():
        try:
            from helpers.settings_boost import _make_cache_dialog
            _make_cache_dialog(page).exec()
        except Exception:
            try:
                from helpers.cleanup_cache import run_cleanup
                run_cleanup(project_root=".", clean_pyc=True, clean_logs=True, clean_thumbs=True, clean_qt_cache=True, clean_hf_cache=False)
            except Exception: pass
    btn_cache.clicked.connect(do_cache)

    # Restart
    btn_restart = QPushButton("Restart App")
    def do_restart():
        try:
            ss = config.setdefault('session_restore', {})
            # Remember that restart was triggered from Settings; restore last non-settings tab
            ss['restart_from_settings'] = True
            try: save_config()
            except Exception: pass
            import sys
            app = QtWidgets.QApplication.instance()
            QtCore.QProcess.startDetached(sys.executable, sys.argv)
            app.quit()
        except Exception:
            pass
    btn_restart.clicked.connect(do_restart)

    # Dump QSettings
    btn_dump = QPushButton("Dump QSettings now")
    def do_dump():
        try:
            from helpers.diagnostics import dump_qsettings
            dump_qsettings("manual")
        except Exception: pass
    btn_dump.clicked.connect(do_dump)

    for b in (btn_cache, btn_restart, btn_dump):
        h.addWidget(b)
    h.addStretch(1)
    return row

def _logo_group(page: QWidget) -> QWidget:
    g = QWidget(page)
    v = QVBoxLayout(g); v.setContentsMargins(8,8,8,8); v.setSpacing(8)
    lab = QLabel(g); lab.setAlignment(Qt.AlignCenter)
    lab.setMinimumHeight(240); lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    lab.setStyleSheet("QLabel { background:#1a1a1a; color:#aaaaaa; border-radius:8px; }")
    lab.setAutoFillBackground(True)
    from helpers.kv_index import attach_click_hint
    attach_click_hint(lab)
    v.addWidget(lab)

    def refresh():
        try:
            from helpers.intro_data import get_logo_sources
            from helpers.intro_screen import _load_pixmap
            
            urls = get_logo_sources(theme=None)
            pm = QtGui.QPixmap()
            if urls:
                import random
                pm = _load_pixmap(random.choice(urls))
            if not pm.isNull():
                pm = pm.scaled(lab.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lab.setPixmap(pm)
        except Exception:
            pass

    lab.resizeEvent = lambda _e: refresh()
    QTimer.singleShot(0, refresh)
    tm = QTimer(g); tm.setInterval(3500); tm.timeout.connect(refresh); tm.start()  # slideshow in settings preview
    return g

# ---- public installer --------------------------------------------------------------------
def install_settings_tab(main_window: QWidget) -> None:
    try:
        # Locate Settings container
        page = _locate_settings_container(main_window)
        if not page:
            return

        # Apply light QSS so titles don't overlap buttons
        page.setStyleSheet(
            "QGroupBox { margin-top: 10px; padding-top: 16px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )

        lay = _ensure_vbox(page)
        _wipe_layout(lay)  # authoritative rebuild, but only for this tab

        # Desired order
        lay.addWidget(_theme_row(page))
        lay.addWidget(_options_group(page))
        lay.addWidget(_buttons_row(page))
        # Separator to keep the logo block from crowding previous controls
        try:
            from PySide6.QtWidgets import QFrame
            hr = QFrame(page); hr.setFrameShape(QFrame.HLine); hr.setFrameShadow(QFrame.Sunken)
            lay.addWidget(hr)
        except Exception:
            pass
        # Optional developer block could be placed here if needed
        lay.addWidget(_logo_group(page))
        lay.addStretch(1)

    except Exception:
        pass


# === Social buttons at the very bottom of Settings ===
def _install_social_bottom_runtime():
    try:
        from PySide6 import QtCore, QtWidgets, QtGui
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        root = None
        for w in app.allWidgets():
            try:
                if (w.objectName() or "") == "FvSettingsContent":
                    root = w; break
            except Exception:
                pass
        if not root or getattr(root, "_fv_social_bottom_installed", False):
            return
        v = root.layout()
        if not isinstance(v, QtWidgets.QVBoxLayout):
            return
        # Add a thin separator
        sep = QtWidgets.QFrame(root)
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep.setObjectName("FvSocialSeparator")
        v.addWidget(sep)
        # Row with buttons aligned right
        row = QtWidgets.QWidget(root)
        h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0,6,0,0); h.setSpacing(8)
        h.addStretch(1)
        btn_gh = QtWidgets.QPushButton("GitHub", row)
        btn_gh.setMinimumWidth(120); btn_gh.setMinimumHeight(24)
        btn_yt = QtWidgets.QPushButton("YouTube", row)
        btn_yt.setMinimumWidth(120); btn_yt.setMinimumHeight(24)
        def open_url(url: str):
            try:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
            except Exception:
                pass
        btn_gh.clicked.connect(lambda: open_url("https://github.com/"))
        btn_yt.clicked.connect(lambda: open_url("https://www.youtube.com/"))
        h.addWidget(btn_gh); h.addWidget(btn_yt)
        v.addWidget(row)
        root._fv_social_bottom_installed = True
    except Exception:
        pass

# schedule attempt after page builds
try:
    from PySide6.QtCore import QTimer
    QTimer.singleShot(800, _install_social_bottom_runtime)
except Exception:
    pass
