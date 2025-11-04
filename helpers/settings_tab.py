from __future__ import annotations
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtWidgets import (
    QWidget, QTabWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QComboBox, QPushButton, QCheckBox, QSizePolicy, QAbstractButton,
    QToolButton
)

try:
    from helpers.framevision_app import apply_theme, config, save_config
except Exception:
    apply_theme=None; config={}

# TooltipManager import (with safe fallback so Settings never crashes)
try:
    from helpers.tooltip_manager import TooltipManager
except Exception:
    class TooltipManager:  # fallback no-op if file missing
        @staticmethod
        def set_enabled(_b: bool): pass
        @staticmethod
        def is_enabled() -> bool: return True

import os, shutil, sys, time

# ------------------------------------------------------------------------------------------
# NOTE: Easter Egg system has been moved to helpers/settings_more.py
# This file now contains only the Settings UI and minimal wiring.
# ------------------------------------------------------------------------------------------

# ---- small filesystem helpers ------------------------------------------------------------
def _clean_directory_contents(path: str) -> None:
    """Delete files/folders inside *path* without deleting *path* itself."""
    try:
        if not path:
            return
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
    container = QWidget(page)
    v = QVBoxLayout(container)
    v.setContentsMargins(0,0,0,0)
    v.setSpacing(6)

    top = QWidget(container)
    h = QHBoxLayout(top); h.setContentsMargins(0,0,0,0); h.setSpacing(8)
    lab = QLabel("Theme:")
    box = QComboBox(); box.addItems([
        "Day","Pastel Light","Solarized Light","Sunburst","Cloud Grey","Signal Grey",
        "Evening","Night","Graphite Dusk","Slate","High Contrast","Cyberpunk","Neon","Ocean","CRT","Aurora",
        "Mardi Gras","Tropical Fiesta","Color Mix","Candy Pop","Rainbow Riot","Random","Auto"
    ])
    try:
        cur = (config.get("theme") or "Auto"); idx = box.findText(cur); box.setCurrentIndex(max(0, idx))
    except Exception:
        pass
    btn = QPushButton("Apply")
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

    bottom = QWidget(container)
    h2 = QHBoxLayout(bottom); h2.setContentsMargins(0,0,0,0); h2.setSpacing(8)

    ov_toggle = QCheckBox("Intro overlay")
    ov_toggle.setToolTip("Enable a visual overlay during the startup intro image (e.g., Matrix rain).")

    ov_combo = QComboBox()
    ov_combo.addItems(["Random","Matrix (Green)","Matrix (Blue)","Bokeh","Rain","Fireworks","FirefliesParallax","Glitch Shards","LightningStrike","StarfieldHyperjump","Warp in","CometTrails","AuroraFlow"])

    ov_preview = QCheckBox("Preview")
    ov_preview.setToolTip("If enabled, shows the intro overlay briefly in the Settings preview.")

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

    # Force intro_follow_theme OFF since that toggle is hidden now
    try:
        s.setValue("intro_follow_theme", False)
    except Exception:
        pass

    cb1 = QCheckBox("Show random intro image on startup")
    cb1.setVisible(False)
    cb1.setChecked(s.value("intro_enabled", True, type=bool))
    cb1.toggled.connect(lambda b: s.setValue("intro_enabled", bool(b)))
    v.addWidget(cb1)

    # -- Clear pycache checkbox ------------------------------------------------------
    cb_clear_pyc = QCheckBox(r"Clear app Python cache at (re)start")
    cb_clear_pyc.setToolTip(
        "If enabled, FrameVision will try to delete compiled .pyc cache folders "
        "on startup to avoid stale bytecode."
    )
    clear_default = s.value("clear_pyc_on_start", False, type=bool)
    cb_clear_pyc.setChecked(clear_default)

    # -- Tooltip visibility checkbox -------------------------------------------------
    cb_tooltips = QCheckBox("Show hover tips / help bubbles")
    cb_tooltips.setToolTip(
        "Turn off to hide ALL mouse-hover tooltips and help popups "
        "everywhere in FrameVision right away."
    )
    tooltips_default = s.value("tooltips_enabled", True, type=bool)
    cb_tooltips.setChecked(tooltips_default)

    # Sync TooltipManager immediately to saved state and make sure
    # the eventFilter is active (lazy install will grab QApplication).
    try:
        TooltipManager.set_enabled(bool(tooltips_default))
    except Exception:
        pass

    # -- Keep settings after restart checkbox ---------------------------------------
    cb_keep_settings = QCheckBox("Keep user preferences")
    cb_keep_settings.setToolTip(
        "If enabled, FrameVision will remember the last values you used for tools, "
        "models, and UI options and restore them next time."
    )
    keep_default = s.value("keep_settings_after_restart", True, type=bool)
    cb_keep_settings.setChecked(keep_default)

    # -- Diagnostic logging checkbox -------------------------------------------------
    cb_diag = QCheckBox("Enable diagnostic logging (restart to apply)")
    cb_diag.setToolTip(
        "Turn this off to disable FrameVision's background diagnostics logs."
    )
    diag_default = s.value("diag_probe_enabled", True, type=bool)
    cb_diag.setChecked(diag_default)

    # mirror value into config so headless/worker processes (no Qt) can read it
    try:
        from helpers.framevision_app import config as _cfg, save_config as _save
        _cfg["diag_probe_enabled"] = bool(diag_default)
        _save()
    except Exception:
        pass

    # ---- shared visual helper: grey out when OFF ---------------------------------
    def _sync_grey_state(chk: QCheckBox, is_on: bool):
        # we tint text color when off (unchecked). when on, reset to theme default
        if is_on:
            chk.setStyleSheet("")
        else:
            chk.setStyleSheet("color: #666666;")

    # initial grey states
    _sync_grey_state(cb_clear_pyc, bool(clear_default))
    _sync_grey_state(cb_tooltips, bool(tooltips_default))
    _sync_grey_state(cb_keep_settings, bool(keep_default))
    _sync_grey_state(cb_diag, bool(diag_default))

    # ---- handlers + persistence ---------------------------------------------------
    def _on_clear_pyc_toggle(b: bool):
        s.setValue("clear_pyc_on_start", bool(b))
        _sync_grey_state(cb_clear_pyc, bool(b))

    def _on_tooltips_toggle(b: bool):
        s.setValue("tooltips_enabled", bool(b))
        try:
            TooltipManager.set_enabled(bool(b))
        except Exception:
            pass
        _sync_grey_state(cb_tooltips, bool(b))

    def _on_keep_settings_toggle(b: bool):
        s.setValue("keep_settings_after_restart", bool(b))
        _sync_grey_state(cb_keep_settings, bool(b))

    def _on_diag_toggle(b: bool):
        s.setValue("diag_probe_enabled", bool(b))
        try:
            from helpers.framevision_app import config as _cfg, save_config as _save
            _cfg["diag_probe_enabled"] = bool(b)
            _save()
        except Exception:
            pass
        _sync_grey_state(cb_diag, bool(b))

    # connect signals
    cb_clear_pyc.toggled.connect(lambda b: _on_clear_pyc_toggle(b))
    cb_tooltips.toggled.connect(lambda b: _on_tooltips_toggle(b))
    cb_keep_settings.toggled.connect(lambda b: _on_keep_settings_toggle(b))
    cb_diag.toggled.connect(lambda b: _on_diag_toggle(b))

    # Add widgets to layout in the requested visual order:
    # 1. Clear pycache
    # 2. Tooltip visibility (NEW)
    # 3. Keep settings after restart
    # 4. Diagnostic logging
    v.addWidget(cb_clear_pyc)
    v.addWidget(cb_tooltips)
    v.addWidget(cb_keep_settings)
    v.addWidget(cb_diag)

    
    # -- Emoji labels toggle --------------------------------------------------------
    cb_emoji = QCheckBox("Emoji labels")
    cb_emoji.setToolTip("Replace feature labels with emoji like 🔍 ⏱️ 📝 📸 🖌️ 🤖 ⏳ ⚙️/⚒️. "
                        "Turn off if your system font renders emoji poorly.")
    emoji_default = s.value("emoji_labels_enabled", False, type=bool)
    cb_emoji.setChecked(emoji_default)
    _sync_grey_state(cb_emoji, bool(emoji_default))

    def _on_emoji_toggle(b: bool):
        s.setValue("emoji_labels_enabled", bool(b))
        _sync_grey_state(cb_emoji, bool(b))
        try:
            cb_emoji_tabs.setVisible(bool(b))

            win = page.window()
            if b:
                apply_emoji_labels_globally(win)
            else:
                restore_emoji_labels_globally(win)
        except Exception:
            pass

    cb_emoji.toggled.connect(lambda b: _on_emoji_toggle(b))
    v.addWidget(cb_emoji)
    # -- Emoji tabs: show labels + emojis option (dependent) ---------------------
    cb_emoji_tabs = QCheckBox("Show labels with emojis on the tabs")
    cb_emoji_tabs.setToolTip(
        "When Emoji labels are enabled, show both the emoji and the text on tab titles."
    )
    emoji_tabs_default = s.value("emoji_tabs_show_labels", False, type=bool)
    cb_emoji_tabs.setChecked(emoji_tabs_default)
    _sync_grey_state(cb_emoji_tabs, bool(emoji_tabs_default))
    cb_emoji_tabs.setVisible(bool(emoji_default))

    def _on_emoji_tabs_toggle(b: bool):
        s.setValue("emoji_tabs_show_labels", bool(b))
        _sync_grey_state(cb_emoji_tabs, bool(b))
        try:
            win = page.window()
            if cb_emoji.isChecked():
                # Re-apply so tabs update between emoji-only and emoji+label
                restore_emoji_labels_globally(win)
                apply_emoji_labels_globally(win)
        except Exception:
            pass

    cb_emoji_tabs.toggled.connect(lambda b: _on_emoji_tabs_toggle(b))
    v.addWidget(cb_emoji_tabs)

# -- Temperature units row -------------------------------------------------------
    row = QWidget(g)
    h2 = QHBoxLayout(row); h2.setContentsMargins(0,4,0,0); h2.setSpacing(8)
    h2.addWidget(QLabel("Temperature units:"))

    combo = QComboBox(row)
    combo.addItem("Celsius (°C)", "C")
    combo.addItem("Fahrenheit (°F)", "F")

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

    h2.addWidget(combo)
    h2.addStretch(1)
    v.addWidget(row)

    v.addStretch(0)
    return g

def _buttons_row(page: QWidget) -> QWidget:
    row = QWidget(page)
    h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)

    btn_cache = QPushButton("Clear program cache…")
    def do_cache():
        try:
            from helpers.settings_boost import _make_cache_dialog
            dlg = _make_cache_dialog(page)
            try:
                for cb in dlg.findChildren(QtWidgets.QCheckBox):
                    t = (cb.text() or "").lower()
                    if "huggingface" in t or "hugging face" in t:
                        cb.setChecked(False)
                        cb.setVisible(False)
                    if "thumbnails" in t:
                        try:
                            cb.setText("Thumbnails (last results) — remove items older than 7 days")
                        except Exception:
                            pass
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
                for cb in dlg.findChildren(QtWidgets.QCheckBox):
                    txt = (cb.text() or "").lower()
                    if "temp" in txt or "temporary" in txt:
                        cb.setChecked(True)
                    if "__pycache__" in txt or "pycache" in txt or "cache py" in txt:
                        cb.setChecked(True)
            except Exception:
                pass
            res = dlg.exec()
            if res == QtWidgets.QDialog.Accepted:
                _extra_temp_cleanup()
            return
        except Exception:
            pass

        try:
            from helpers.cleanup_cache import run_cleanup
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
        _extra_temp_cleanup()
    btn_cache.clicked.connect(do_cache)

    btn_restart = QPushButton("Restart App")
    def do_restart():
        try:
            ss = config.setdefault('session_restore', {})
            ss['restart_from_settings'] = True
            try: save_config()
            except Exception: pass
            app = QtWidgets.QApplication.instance()
            QtCore.QProcess.startDetached(sys.executable, sys.argv)
            app.quit()
        except Exception:
            pass
    btn_restart.clicked.connect(do_restart)

    btn_dump = QPushButton("Show QSettings in CMD")
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
    tm = QTimer(g); tm.setInterval(3500); tm.timeout.connect(refresh); tm.start()

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

    try:
        _prev_tm = QTimer(g); _prev_tm.setInterval(2000); _prev_tm.timeout.connect(_sync_overlay_preview); _prev_tm.start()
    except Exception:
        pass

    return g


# === Emoji Labels (single-file implementation) ===========================================
# Borderless emoji set & mappers (no boxed symbols). Replaces tab titles with emoji-only
# and prefixes matching buttons/labels with an emoji. Restores originals when toggled off.
_EMOJI_MAP = {
    "down": "👇",
    "selected": "👉",
    "wards": "☝",
    "eastereggs": "🕹️",
    "framevision": "🍀️",
    "upscale": "🧪",
    "engine": "🚀",
    "interpolator": "⏱️",
    "model": "🧩️",
    "cancel": "❌️",
    "describer": "📝",
    "describe": "📝",
    "folder": "🗂️",
    "batch": "📦",
    "info": "💡",
    "txt2img": "📸",
    "txt to img": "📸",
    "running": "🏃️",
    "pending": "💤️",
    "background": "🖌️",
    "preset": "💾️",
    "inpaint": "💫",
    "ptofile": "🧍",
    "queue": "⏳",
    "settings": "️⚙️",
    "tools": "🧱",
    "txttoimg": "📸",
    "copy": "🧬",
    "preview": "👁️",
    "texttoimg": "📸",
    "file": "📂",
    "rifefps": "⏱️",
    "rife fps": "⏱️",
    "cpu": "⚡️",
    "memory": "🧮",
    "askframie": "👽",
    "failed": "❌️",
    "finished": "🎉",
    "refresh": "🌀",
    "current": "🧲️",
    "reset": "♻️",
    "Browse": "🗂",
    "undo": "↶",
    "save": "🗃️",
    "vibevoice": "🎙️",
    "profile": "🎭️",
    "units": "🌤️",
    "theme": "🌈️",
    "overlay": "💠️",
    "update": "🌟",
    "East": "🌟",
}
def _canon_text_for_emoji(s: str) -> str:
    import re as _re
    s = (s or "").lower()
    s = _re.sub(r"[\s\-_]+", "", s)
    s = _re.sub(r"[^a-z0-9]+", "", s)
    return s
def _emoji_for_text(s: str) -> str | None:
    c = _canon_text_for_emoji(s)
    if c in _EMOJI_MAP:
        return _EMOJI_MAP[c]
    # partial contains — check longer keys first to avoid "chat" vs "chatbot" mismatches
    for k in sorted(_EMOJI_MAP.keys(), key=len, reverse=True):
        if k in c:
            return _EMOJI_MAP[k]
    return None

def _emoji_store_tab_orig(tabs: QTabWidget, idx: int, text: str) -> None:
    try:
        store = tabs.property("emoji_orig_tab_texts") or {}
        if int(idx) not in store:
            store[int(idx)] = text
            tabs.setProperty("emoji_orig_tab_texts", store)
    except Exception:
        pass
def _emoji_restore_tabs(tabs: QTabWidget) -> None:
    try:
        store = tabs.property("emoji_orig_tab_texts") or {}
        for i, txt in list(store.items()):
            try:
                tabs.setTabText(int(i), str(txt))
            except Exception:
                pass
        tabs.setProperty("emoji_orig_tab_texts", {})
    except Exception:
        pass
def _emoji_store_widget_orig(w: QWidget, text: str) -> None:
    try:
        if not w.property("emoji_orig_text"):
            w.setProperty("emoji_orig_text", text)
    except Exception:
        pass
def _emoji_restore_widgets(root: QWidget) -> None:
    for cls in (QAbstractButton, QPushButton, QToolButton, QLabel):
        for w in root.findChildren(cls):
            try:
                orig = w.property("emoji_orig_text")
                if orig:
                    w.setText(str(orig))
                    w.setProperty("emoji_orig_text", None)
            except Exception:
                pass

def apply_emoji_labels_globally(root: QWidget) -> None:
    """Idempotently apply emoji label replacements across the app."""
    if root is None:
        return
    # Tabs: emoji or emoji+label depending on setting
    show_labels = _emoji_tabs_show_labels_qsettings()
    for tabs in root.findChildren(QTabWidget):
        try:
            count = tabs.count()
        except Exception:
            count = 0
        for i in range(count):
            try:
                old = tabs.tabText(i) or ""
            except Exception:
                continue
            e = _emoji_for_text(old)
            if not e:
                continue
            _emoji_store_tab_orig(tabs, i, old)
            try:
                new_txt = f"{e} {old}" if show_labels else e
                tabs.setTabText(i, new_txt)
            except Exception:
                pass
    # Buttons / toolbuttons / labels: prefix emoji + space
    for cls in (QAbstractButton, QPushButton, QToolButton, QLabel):
        for w in root.findChildren(cls):
            try:
                old = w.text() or ""
            except Exception:
                continue
            e = _emoji_for_text(old)
            if not e:
                continue
            _emoji_store_widget_orig(w, old)
            if not old.startswith(e):
                try:
                    w.setText(f"{e} {old}")
                except Exception:
                    pass

def restore_emoji_labels_globally(root: QWidget) -> None:
    """Restore original labels for tabs/buttons/labels that were modified."""
    if root is None:
        return
    for tabs in root.findChildren(QTabWidget):
        _emoji_restore_tabs(tabs)
    _emoji_restore_widgets(root)

def _emoji_enabled_qsettings() -> bool:
    try:
        s = QSettings("FrameVision","FrameVision")
        return bool(s.value("emoji_labels_enabled", False, type=bool))
    except Exception:
        return False
def _emoji_set_enabled_qsettings(val: bool) -> None:
    try:
        s = QSettings("FrameVision","FrameVision")
        s.setValue("emoji_labels_enabled", bool(val))
    except Exception:
        pass



def _emoji_tabs_show_labels_qsettings() -> bool:
    """Return whether tabs should show both emoji and label text when emoji mode is on."""
    try:
        s = QSettings("FrameVision","FrameVision")
        return bool(s.value("emoji_tabs_show_labels", False, type=bool))
    except Exception:
        return False
def _apply_emoji_on_start(root: QWidget) -> None:
    """Auto-apply if enabled; called from install_settings_tab once UI exists."""
    try:
        if _emoji_enabled_qsettings():
            # slight delay: ensure tabs/buttons are live
            def _go():
                apply_emoji_labels_globally(root)
            QTimer.singleShot(50, _go)
    except Exception:
        pass
# ==========================================================================================

# ---- public installer --------------------------------------------------------------------
def install_settings_tab(main_window: QWidget) -> None:
    try:
        page = _locate_settings_container(main_window)
        if not page:
            return

        # Mark the settings page so settings_more can locate it reliably
        try:
            page.setObjectName("FvSettingsContent")
        except Exception:
            pass

        page.setStyleSheet(
            "QGroupBox { margin-top: 10px; padding-top: 16px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )

        lay = _ensure_vbox(page)
        _wipe_layout(lay)

        lay.addWidget(_theme_row(page))
        lay.addWidget(_options_group(page))
        lay.addWidget(_buttons_row(page))
        try:
            from PySide6.QtWidgets import QFrame
            hr = QFrame(page); hr.setFrameShape(QFrame.HLine); hr.setFrameShadow(QFrame.Sunken)
            lay.addWidget(hr)
        except Exception:
            pass
        lay.addWidget(_logo_group(page))
        lay.addStretch(1)

        
        # Auto-apply emoji labels if previously enabled
        try:
            _apply_emoji_on_start(main_window)
        except Exception:
            pass
# Minimal wiring: delegate Easter Eggs UI injection + tracker to settings_more
        try:
            from helpers import settings_more as _sm
            QtCore.QTimer.singleShot(100, _sm.ensure_usage_tracker)
            QtCore.QTimer.singleShot(200, _sm.install_social_bottom_runtime)
        except Exception:
            pass

    except Exception:
        pass
