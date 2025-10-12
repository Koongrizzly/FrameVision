
# helpers/settings_tab.py — Settings page builder (overlay row + cache defaults + extra temp dirs)
from __future__ import annotations
from typing import Optional, Iterable
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
import os, shutil

# ---- small filesystem helpers ------------------------------------------------------------
def _clean_directory_contents(path: str) -> None:
    """Delete files/folders inside *path* without deleting *path* itself."""
    try:
        if not path:
            return
        p = QtCore.QDir.toNativeSeparators(path)
        if not os.path.isdir(path):
            return
        for name in os.listdir(path):
            fp = os.path.join(path, name)
            try:
                if os.path.isdir(fp) and not os.path.islink(fp):
                    shutil.rmtree(fp, ignore_errors=True)
                else:
                    try:
                        os.remove(fp)
                    except PermissionError:
                        # read-only on Windows; try chmod then remove
                        try:
                            os.chmod(fp, 0o666)
                            os.remove(fp)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        pass

def _project_root() -> str:
    try:
        # heuristic: helpers/ is under project root
        here = os.path.abspath(os.path.dirname(__file__))
        return os.path.abspath(os.path.join(here, ".."))
    except Exception:
        return os.getcwd()

def _extra_temp_cleanup() -> None:
    """Clean temp directories requested by user: output/_temp and work."""
    root = _project_root()
    targets = [
        os.path.join(root, "output", "_temp"),
        os.path.join(root, "work"),
    ]
    for t in targets:
        _clean_directory_contents(t)

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
    """Top section with (1) Theme controls and (2) Intro overlay controls on a NEW LINE."""
    container = QWidget(page)
    v = QVBoxLayout(container)
    v.setContentsMargins(0,0,0,0)
    v.setSpacing(6)

    # --- Row 1: Theme selection + apply ---------------------------------------------------
    top = QWidget(container)
    h = QHBoxLayout(top); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
    lab = QLabel("Theme:")
    box = QComboBox(); box.addItems([
        "Day","Solarized Light","Sunburst","Evening","Night","Slate","High Contrast",
        "Cyberpunk","Neon","Ocean","CRT","Aurora","Mardi Gras","Tropical Fiesta",
        "Color Mix","Random","Auto"
    ])
    try:
        cur = (config.get("theme") or "Auto"); idx = box.findText(cur); box.setCurrentIndex(max(0, idx))
    except Exception:
        pass
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
    v.addWidget(top)

    # --- Row 2: Intro overlay controls (moved to NEW LINE) -------------------------------
    bottom = QWidget(container)
    h2 = QHBoxLayout(bottom); h2.setContentsMargins(0,0,0,0); h2.setSpacing(8)

    ov_toggle = QCheckBox("Intro overlay")
    ov_toggle.setToolTip("Enable a visual overlay during the startup intro image (e.g., Matrix rain).")

    ov_combo = QComboBox()
    ov_combo.addItems(["Random","Matrix (Green)","Matrix (Blue)","Bokeh","Rain","FirefliesParallax","StarfieldHyperjump","CometTrails","AuroraFlow"])

    ov_preview = QCheckBox("Preview in Settings")
    ov_preview.setToolTip("If enabled, shows the intro overlay briefly in the Settings preview.")

    # Persist/restore via QSettings
    _s = QSettings("FrameVision", "FrameVision")
    ov_toggle.setChecked(_s.value("intro_overlay_enabled", False, type=bool))
    mode = _s.value("intro_overlay_mode", "Random", type=str) or "Random"
    idx = ov_combo.findText(mode); ov_combo.setCurrentIndex(max(0, idx))
    ov_preview.setChecked(_s.value("intro_overlay_preview_enabled", True, type=bool))

    ov_toggle.toggled.connect(lambda b: _s.setValue("intro_overlay_enabled", bool(b)))
    ov_combo.currentTextChanged.connect(lambda t: _s.setValue("intro_overlay_mode", t))
    ov_preview.toggled.connect(lambda b: _s.setValue("intro_overlay_preview_enabled", bool(b)))

    for w in (ov_combo, ov_toggle, ov_preview):
        w.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

    h2.addWidget(ov_toggle)
    h2.addWidget(ov_combo)
    h2.addWidget(ov_preview)
    h2.addStretch(1)
    v.addWidget(bottom)

    return container

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
    v.addStretch(0)
    return g

def _buttons_row(page: QWidget) -> QWidget:
    row = QWidget(page)
    h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)

    # Clear cache
    btn_cache = QPushButton("Clear program cache…")
    def do_cache():
        # Prefer the app dialog if present so users can fine-tune;
        # but pre-check "temp" and "__pycache__" options by default.
        try:
            from helpers.settings_boost import _make_cache_dialog
            
            dlg = _make_cache_dialog(page)
            # --- Runtime UI tweaks without touching settings_boost.py ---
            try:
                # Hide HuggingFace cache option entirely
                for cb in dlg.findChildren(QtWidgets.QCheckBox):
                    t = (cb.text() or "").lower()
                    if "huggingface" in t or "hugging face" in t:
                        cb.setChecked(False)
                        cb.setVisible(False)
                    # Update thumbnails label and keep the checkbox available
                    if "thumbnails" in t:
                        try:
                            cb.setText("Thumbnails (last results) — remove items older than 7 days")
                        except Exception:
                            pass
                # Update the tip label
                for lab in dlg.findChildren(QtWidgets.QLabel):
                    txt = (lab.text() or "").strip().lower()
                    if txt.startswith("tip:"):
                        try:
                            lab.setText("Tip: Do this every now and then to keep a clean running app")
                        except Exception:
                            pass
                        break
            except Exception:
                pass

            try:
                # Heuristically find relevant checkboxes by text and enable them.
                for cb in dlg.findChildren(QtWidgets.QCheckBox):
                    txt = (cb.text() or "").lower()
                    if "temp" in txt or "temporary" in txt:
                        cb.setChecked(True)
                    if "__pycache__" in txt or "pycache" in txt or "cache py" in txt:
                        cb.setChecked(True)
            except Exception:
                pass
            # Run dialog; if accepted, do our extra folders cleanup as well.
            res = dlg.exec()
            if res == QtWidgets.QDialog.Accepted:
                _extra_temp_cleanup()
            return
        except Exception:
            pass

        # Fallback: do a safe default cleanup with our own routine.
        try:
            from helpers.cleanup_cache import run_cleanup
            # Ensure temp + pycache are ON by default
            run_cleanup(
                project_root=_project_root(),
                clean_pyc=True,
                clean_logs=True,
                clean_thumbs=True,
                clean_qt_cache=True,
                clean_hf_cache=False,
                clean_temp=True
            )
        except Exception:
            pass
        # Our additional folders, regardless of run_cleanup outcome
        _extra_temp_cleanup()

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

    # --- Overlay preview on the Settings logo block ---
    try:
        from helpers.overlay_animations import apply_intro_overlay_from_settings, stop_overlay
    except Exception:
        apply_intro_overlay_from_settings = None; stop_overlay = None

    def _sync_overlay_preview():
        try:
            s = QSettings('FrameVision','FrameVision')
            en = s.value('intro_overlay_enabled', False, type=bool)
            prev_ok = s.value('intro_overlay_preview_enabled', True, type=bool)
            theme = s.value('theme','Auto')
            st = getattr(lab, "_fv_overlay_state", None)
            if st is None:
                st = {"applied": False, "last_en": None, "last_theme": None}
                setattr(lab, "_fv_overlay_state", st)
            if not en or not prev_ok:
                if stop_overlay:
                    try:
                        stop_overlay(lab, 0)
                    except TypeError:
                        try:
                            stop_overlay(lab)
                        except Exception:
                            pass
                st["applied"] = False
                st["last_en"] = en; st["last_theme"] = theme
                return
            changed = (st["last_en"] != en) or (st["last_theme"] != theme) or (not st["applied"])
            if changed and apply_intro_overlay_from_settings:
                if stop_overlay:
                    try:
                        stop_overlay(lab, 0)
                    except TypeError:
                        try:
                            stop_overlay(lab)
                        except Exception:
                            pass
                apply_intro_overlay_from_settings(lab, theme, force_topmost=False)
                st["applied"] = True; st["last_en"] = en; st["last_theme"] = theme
                try:
                    t_prev = getattr(lab, "_fv_overlay_kill", None)
                    if t_prev is not None:
                        t_prev.stop(); t_prev.deleteLater()
                    kill = QtCore.QTimer(lab); kill.setSingleShot(True); kill.setInterval(4000)
                    def _kill_now():
                        try:
                            try:
                                stop_overlay(lab, 600)
                            except TypeError:
                                stop_overlay(lab)
                        except Exception:
                            pass
                        st["applied"] = False
                    kill.timeout.connect(_kill_now); kill.start()
                    setattr(lab, "_fv_overlay_kill", kill)
                except Exception:
                    pass
        except Exception:
            pass

    # Start a lightweight poll timer
    try:
        _prev_tm = QTimer(g); _prev_tm.setInterval(1500); _prev_tm.timeout.connect(_sync_overlay_preview); _prev_tm.start()
    except Exception:
        pass

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
