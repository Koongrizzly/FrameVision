from __future__ import annotations
from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtWidgets import (
    QWidget, QTabWidget, QScrollArea, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QComboBox, QPushButton, QCheckBox, QSizePolicy, QAbstractButton,
    QToolButton, QSlider
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


# AnimatedButtonsManager import (safe fallback so Settings never crashes)
try:
    from helpers.animated_buttons import AnimatedButtonsManager
except Exception:
    class AnimatedButtonsManager:  # fallback no-op if file missing
        @staticmethod
        def install(_app=None): 
            return None
        @staticmethod
        def apply_from_settings(_app=None): 
            return None

import os, shutil, sys, time, json


# ---- Global font size (px delta) helpers -----------------------------------------------
def _font_size__orig_font(app: QtWidgets.QApplication) -> QtGui.QFont:
    """Get or store the original application font (used to restore later)."""
    try:
        orig = app.property("fv_orig_app_font")
        if isinstance(orig, QtGui.QFont):
            return QtGui.QFont(orig)
        f = QtGui.QFont(app.font())
        app.setProperty("fv_orig_app_font", QtGui.QFont(f))
        return QtGui.QFont(f)
    except Exception:
        return QtGui.QFont()

def _font_size_apply(app: Optional[QtWidgets.QApplication], enabled: bool, delta_px: int) -> None:
    """Apply a global font size delta in *pixels* using QApplication.setFont()."""
    try:
        if app is None:
            app = QtWidgets.QApplication.instance()
        if app is None:
            return

        orig = _font_size__orig_font(app)

        if not enabled:
            app.setFont(orig)
            return

        try:
            delta_px = int(delta_px)
        except Exception:
            delta_px = 0

        # Determine actual rendered pixel size (works even if font uses pointSize internally).
        try:
            base_px = int(QtGui.QFontInfo(orig).pixelSize())
        except Exception:
            base_px = -1

        if base_px <= 0:
            # Fallback: use font metrics height (approx) if pixelSize is unavailable.
            try:
                base_px = int(QtGui.QFontMetrics(orig).height())
            except Exception:
                base_px = 14

        new_px = max(8, base_px + delta_px)

        f = QtGui.QFont(orig)
        f.setPixelSize(int(new_px))
        app.setFont(f)
    except Exception:
        pass

def _font_size_apply_from_qsettings(app: Optional[QtWidgets.QApplication] = None) -> None:
    """Apply the saved font size settings on startup/UI build."""
    try:
        s = QSettings("FrameVision", "FrameVision")
        en = bool(s.value("ui_font_size_enabled", False, type=bool))
        delta = s.value("ui_font_size_delta_px", 0)
        try:
            delta = int(delta)
        except Exception:
            delta = 0
        _font_size_apply(app or QtWidgets.QApplication.instance(), en, delta)
    except Exception:
        pass


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
    """Return the *top-level* widget used as the Settings tab page.

    Important: we must target the actual tab page widget (not an inner scroll widget),
    otherwise we can end up leaving legacy header/placeholder UI above our injected UI.
    """
    if root is None:
        return None

    # Prefer the real main tabs by objectName (your app uses "main_tabs").
    tabs = root.findChild(QTabWidget, "main_tabs") or getattr(root, "tabs", None)
    if isinstance(tabs, QTabWidget):
        for i in range(tabs.count()):
            if "setting" in (tabs.tabText(i) or "").lower():
                return tabs.widget(i)

    # Fallback: try to find a page explicitly named "settings"
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

def _wipe_layout(lay: QtWidgets.QLayout) -> None:
    """Remove *all* items from a layout (widgets, spacers, nested layouts)."""
    try:
        while lay.count():
            it = lay.takeAt(0)
            if it is None:
                break

            w = it.widget()
            if w is not None:
                try:
                    w.setParent(None)
                    w.deleteLater()
                except Exception:
                    pass
                continue

            child_lay = it.layout()
            if child_lay is not None:
                try:
                    _wipe_layout(child_lay)
                    child_lay.setParent(None)
                except Exception:
                    pass
                continue

            # Spacer item: nothing else to do.
    except Exception:
        pass

def _force_theme_refresh(app: Optional[QtWidgets.QApplication]) -> None:
    """Force a full style repolish across existing widgets.

    This helps when switching between light/dark themes where some widgets keep old palette roles
    or cached stylesheet colors (e.g., text becomes low-contrast until a repolish).
    """
    try:
        if app is None:
            app = QtWidgets.QApplication.instance()
        if app is None:
            return

        widgets = list(app.allWidgets() or [])
        # Pause updates briefly to reduce flicker while repolishing.
        for w in widgets:
            try:
                w.setUpdatesEnabled(False)
            except Exception:
                pass

        # Repolish + push the current app palette everywhere.
        pal = app.palette()
        for w in widgets:
            try:
                st = w.style()
                st.unpolish(w)
                st.polish(w)
            except Exception:
                pass
            try:
                w.setPalette(pal)
            except Exception:
                pass
            try:
                w.update()
                vp = getattr(w, "viewport", None)
                if callable(vp):
                    v = vp()
                    if v is not None:
                        v.update()
            except Exception:
                pass

        for w in widgets:
            try:
                w.setUpdatesEnabled(True)
            except Exception:
                pass

        # Nudge top-level windows.
        for w in app.topLevelWidgets() or []:
            try:
                w.update()
                w.repaint()
            except Exception:
                pass
    except Exception:
        pass

def _theme_row(page: QWidget) -> QWidget:
    container = QWidget(page)
    v = QVBoxLayout(container)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(6)

    # --- Theme row ------------------------------------------------------------
    top = QWidget(container)
    h = QHBoxLayout(top)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)

    lab = QLabel("Theme:")
    box = QComboBox()
    box.addItems([
        "Day", "Pastel Light", "Solarized Light", "Sunburst", "Cloud Grey", "Signal Grey",
        "Blue in the Dark", "Green in the Dark", "Orange in the Dark", "Red in the Dark",
        "Evening", "Night", "Graphite Dusk", "Slate", "High Contrast", "NeonGreen", "Ocean", "CRT", "Aurora",
        "Purple Life", "Tropical Fiesta", "Cyberpunk", "Color Mix", "Candy Pop", "Rainbow Riot", "Random"
    ])

    # "Just enough" width: size to the longest theme name + padding (avoid huge popup).
    try:
        fm = box.fontMetrics()
        longest_px = 0
        for i in range(box.count()):
            t = box.itemText(i)
            try:
                longest_px = max(longest_px, int(fm.horizontalAdvance(t)))
            except Exception:
                pass

        extra_px = 60  # arrow/margins/scrollbar safety padding
        try:
            st = box.style()
            if st is not None:
                extra_px = int(
                    st.pixelMetric(QtWidgets.QStyle.PM_ScrollBarExtent)
                    + st.pixelMetric(QtWidgets.QStyle.PM_DefaultFrameWidth) * 2
                    + 44
                )
        except Exception:
            pass

        w = max(220, min(420, longest_px + extra_px))
        box.setMinimumWidth(int(w))
        box.setMaximumWidth(int(w))
        box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        if box.view() is not None:
            box.view().setMinimumWidth(int(w))
    except Exception:
        pass

    try:
        cur = (config.get("theme") or "Auto")

        # Backwards compatibility: older builds used "Mardi Gras".
        try:
            if isinstance(cur, str):
                _n = cur.strip().lower()
                if _n in ("mardi gras", "mardi grass", "mardigras", "mardigrass"):
                    cur = "Purple Life"
                    try:
                        config["theme"] = cur
                    except Exception:
                        pass
        except Exception:
            pass

        idx = box.findText(cur)
        if idx < 0:
            # Extra fallback for odd stored variants.
            try:
                if isinstance(cur, str) and cur.strip().lower() in ("purplelife",):
                    idx = box.findText("Purple Life")
            except Exception:
                pass

        box.setCurrentIndex(max(0, idx))
    except Exception:
        pass

    btn = QPushButton("Apply")
    safe_btn = QPushButton("Safe apply")

    # Tooltips (rich text supported)
    btn.setToolTip("<b>Apply</b><br>Applies the theme directly, but may freeze the app until it finishes.")
    safe_btn.setToolTip("<b>Safe apply</b><br>Applies the theme by restarting the UI, so it won’t freeze while applying.")

    _busy = {"on": False}

    def _set_busy(on: bool) -> None:
        on = bool(on)
        if _busy.get("on") == on:
            return
        _busy["on"] = on
        try:
            if on:
                btn.setProperty("_fv_prev_text", btn.text() or "Apply")
                safe_btn.setProperty("_fv_prev_text", safe_btn.text() or "Safe apply")
                btn.setText("Applying…")
                safe_btn.setText("Applying…")
                btn.setEnabled(False)
                safe_btn.setEnabled(False)
                box.setEnabled(False)
            else:
                prev = btn.property("_fv_prev_text") or "Apply"
                btn.setText(str(prev))
                prev2 = safe_btn.property("_fv_prev_text") or "Safe apply"
                safe_btn.setText(str(prev2))
                btn.setEnabled(True)
                safe_btn.setEnabled(True)
                box.setEnabled(True)
        except Exception:
            pass

    def do_apply() -> None:
        if _busy.get("on"):
            return
        _set_busy(True)
        try:
            t = box.currentText()
            config["theme"] = t

            app = QtWidgets.QApplication.instance()

            # Let the UI paint the button state before the heavier work.
            try:
                if app:
                    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            except Exception:
                pass

            # Clear any previous global QSS first to avoid "layered" styles / mixed palettes.
            try:
                if app:
                    app.setStyleSheet("")
            except Exception:
                pass

            # Apply theme (in-process). This may freeze briefly on some systems.
            try:
                if apply_theme and app:
                    apply_theme(app, t)
            except Exception:
                pass

            # Force a repolish so existing widgets (like System monitor) pick up the new palette/QSS.
            try:
                _force_theme_refresh(app)
            except Exception:
                pass

            try:
                save_config()
            except Exception:
                pass
        except Exception:
            pass
        finally:
            _set_busy(False)

    def do_safe_apply() -> None:
        if _busy.get("on"):
            return
        _set_busy(True)
        try:
            t = box.currentText()
            config["theme"] = t

            # Mark restart intent (optional; some builds restore focus/state based on this)
            try:
                ss = config.setdefault("session_restore", {})
                ss["restart_from_settings"] = True
            except Exception:
                pass

            try:
                save_config()
            except Exception:
                pass

            # Small event pump so UI shows the state before we exit.
            try:
                app = QtWidgets.QApplication.instance()
                if app:
                    app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            except Exception:
                pass

            # Restart the app shell/UI.
            try:
                QtCore.QProcess.startDetached(sys.executable, sys.argv)
            except Exception:
                try:
                    QtCore.QProcess.startDetached(sys.executable, sys.argv, os.getcwd())
                except Exception:
                    pass

            try:
                app = QtWidgets.QApplication.instance()
                if app:
                    app.quit()
            except Exception:
                pass
        except Exception:
            # If restart fails, restore UI state.
            pass
        finally:
            # If we didn't quit, re-enable UI.
            _set_busy(False)

    btn.clicked.connect(do_apply)
    safe_btn.clicked.connect(do_safe_apply)

    h.addWidget(lab)
    h.addWidget(box)
    h.addWidget(btn)
    h.addWidget(safe_btn)
    h.addStretch(1)
    v.addWidget(top)

    # --- Intro overlay row ----------------------------------------------------
    bottom = QWidget(container)
    h2 = QHBoxLayout(bottom)
    h2.setContentsMargins(0, 0, 0, 0)
    h2.setSpacing(8)

    ov_toggle = QCheckBox("Intro overlay")
    ov_toggle.setToolTip("Enable a visual overlay during the startup intro image (e.g., Matrix rain).")

    ov_combo = QComboBox()
    ov_combo.addItems([
        "Random", "Matrix (Green)", "Matrix (Blue)", "Bokeh", "Rain", "Fireworks",
        "FirefliesParallax", "Glitch Shards", "LightningStrike", "StarfieldHyperjump",
        "Warp in", "CometTrails", "AuroraFlow"
    ])

    ov_preview = QCheckBox("Preview")
    ov_preview.setToolTip("If enabled, shows the intro overlay briefly in the Settings preview.")

    _s = QSettings("FrameVision", "FrameVision")
    ov_toggle.setChecked(_s.value("intro_overlay_enabled", True, type=bool))
    mode = _s.value("intro_overlay_mode", "Fireworks", type=str) or "Random"
    idx = ov_combo.findText(mode)
    ov_combo.setCurrentIndex(max(0, idx))
    ov_preview.setChecked(_s.value("intro_overlay_preview_enabled", False, type=bool))

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
    """Options (collapsible, lightweight).

    Why this implementation:
    - Avoids the random clipping bug seen with *checkable* QGroupBox inside scroll areas.
    - Avoids stylesheet-heavy custom painting that can cause UI-thread hitches.
    - Forces a relayout after expand/collapse so the scroll area recalculates height reliably.

    State is stored in QSettings key: settings_options_expanded (bool).
    """
    s = QSettings("FrameVision", "FrameVision")
    expanded_default = s.value("settings_options_expanded", True, type=bool)

    # Outer group purely for visual grouping (NOT checkable).
    g = QGroupBox("", page)
    outer = QVBoxLayout(g)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(6)

    # Header row (toolbutton toggle).
    header = QWidget(g)
    h = QHBoxLayout(header)
    h.setContentsMargins(8, 4, 8, 0)
    h.setSpacing(6)

    btn = QToolButton(header)
    btn.setText("Options")
    btn.setCheckable(True)
    btn.setChecked(bool(expanded_default))
    btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    btn.setAutoRaise(True)
    btn.setArrowType(Qt.DownArrow if btn.isChecked() else Qt.RightArrow)

    h.addWidget(btn)
    h.addStretch(1)
    outer.addWidget(header)

    # Content container (built lazily).
    content = QWidget(g)
    v = QVBoxLayout(content)
    v.setContentsMargins(8, 0, 8, 8)
    v.setSpacing(6)
    outer.addWidget(content)

    _built = {"done": False}

    def _find_scroll_area(w: QWidget) -> Optional[QScrollArea]:
        try:
            p = w.parentWidget()
            while p is not None:
                if isinstance(p, QScrollArea):
                    return p
                p = p.parentWidget()
        except Exception:
            pass
        return None

    def _force_relayout():
        """Kick Qt's layout/scroll recalculation after show/hide changes."""
        try:
            if page.layout():
                page.layout().activate()
        except Exception:
            pass
        try:
            content.updateGeometry()
            content.adjustSize()
        except Exception:
            pass

        sa = _find_scroll_area(g)
        if sa is not None:
            try:
                w = sa.widget()
                if w and w.layout():
                    w.layout().activate()
                if w:
                    w.updateGeometry()
                    w.adjustSize()
            except Exception:
                pass
            try:
                sa.viewport().update()
            except Exception:
                pass

        # One more pass next tick (this is what fixes the random "shrinking" in practice).
        try:
            QTimer.singleShot(0, lambda: (
                (page.layout().activate() if page.layout() else None),
                content.updateGeometry(),
                content.adjustSize()
            ))
        except Exception:
            pass

    # ---- shared visual helper: grey out when OFF (no stylesheets) -----------------
    def _sync_grey_state(chk: QCheckBox, is_on: bool):
        try:
            orig = chk.property("fv_orig_palette")
            if orig is None:
                chk.setProperty("fv_orig_palette", QtGui.QPalette(chk.palette()))
                orig = chk.property("fv_orig_palette")
            if is_on:
                if isinstance(orig, QtGui.QPalette):
                    chk.setPalette(orig)
                return
            pal = QtGui.QPalette(chk.palette())
            pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(120, 120, 120))
            chk.setPalette(pal)
        except Exception:
            pass

    def _ensure_built():
        if _built["done"]:
            return
        _built["done"] = True

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
        cb_tooltips = QCheckBox("Show hover tips")
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
        cb_diag.setToolTip("Turn this off to disable FrameVision's background diagnostics logs.")
        diag_default = s.value("diag_probe_enabled", False, type=bool)
        cb_diag.setChecked(diag_default)

        # mirror value into config so headless/worker processes (no Qt) can read it
        try:
            from helpers.framevision_app import config as _cfg, save_config as _save
            _cfg["diag_probe_enabled"] = bool(diag_default)
            _save()
        except Exception:
            pass

        # -- Fancy banner toggles ------------------------------------------------------
        cb_banner = QCheckBox("Banners")
        cb_banner.setToolTip("Show or hide the fancy banner headers at the top of each tab.")
        banner_default = s.value("banner_enabled", True, type=bool)
        cb_banner.setChecked(banner_default)

        cb_banner_color = QCheckBox("Colored")
        cb_banner_color.setToolTip("If off, all banners use a neutral grey gradient instead of colorful themes.")
        banner_color_default = s.value("banner_colored", True, type=bool)
        cb_banner_color.setChecked(banner_color_default)
        cb_banner_color.setVisible(bool(banner_default))

        # Gradient animated banner mode (disables 'Colored' while active)
        cb_banner_gradient = QCheckBox("Gradient")
        cb_banner_gradient.setToolTip("Animated rainbow gradient for tab banners. Disables 'Colored' while active.")
        banner_grad_default = s.value("banner_gradient_enabled", False, type=bool)
        cb_banner_gradient.setChecked(bool(banner_grad_default))
        cb_banner_gradient.setVisible(bool(banner_default))

        lab_banner_speed = QLabel("Speed")
        lab_banner_speed.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        lab_banner_speed.setVisible(bool(banner_default) and bool(banner_grad_default))

        slider_banner_speed = QSlider(Qt.Horizontal)
        slider_banner_speed.setRange(1, 50)
        speed_default = int(s.value("banner_gradient_speed", 20, type=int) or 20)
        if speed_default < 1: speed_default = 1
        if speed_default > 50: speed_default = 50
        slider_banner_speed.setValue(speed_default)
        slider_banner_speed.setToolTip("Gradient animation speed (higher = faster).")
        slider_banner_speed.setVisible(bool(banner_default) and bool(banner_grad_default))
        slider_banner_speed.setFixedWidth(220)
        try:
            cb_banner_color.setEnabled(not bool(banner_grad_default))
        except Exception:
            pass

        # initial grey states
        _sync_grey_state(cb_clear_pyc, bool(clear_default))
        _sync_grey_state(cb_tooltips, bool(tooltips_default))
        _sync_grey_state(cb_keep_settings, bool(keep_default))
        _sync_grey_state(cb_diag, bool(diag_default))
        _sync_grey_state(cb_banner, bool(banner_default))
        _sync_grey_state(cb_banner_color, bool(banner_color_default))
        _sync_grey_state(cb_banner_gradient, bool(banner_grad_default))

        # Apply banner settings to the whole main window (not just the Settings tab widget).
        root_for_banners = page.window() if hasattr(page, 'window') else page

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

        def _on_banner_toggle(b: bool):
            s.setValue("banner_enabled", bool(b))
            _sync_grey_state(cb_banner, bool(b))
            try:
                cb_banner_color.setVisible(bool(b))
                cb_banner_gradient.setVisible(bool(b))
            except Exception:
                pass
            try:
                lab_banner_speed.setVisible(bool(b) and bool(cb_banner_gradient.isChecked()))
                slider_banner_speed.setVisible(bool(b) and bool(cb_banner_gradient.isChecked()))
            except Exception:
                pass
            try:
                _banner_apply_visibility(root_for_banners, bool(b))
            except Exception:
                pass

            # Start/stop the animated gradient as needed
            try:
                if not bool(b):
                    _banner_apply_gradient(root_for_banners, False, int(slider_banner_speed.value()))
                else:
                    if bool(cb_banner_gradient.isChecked()):
                        _banner_apply_gradient(root_for_banners, True, int(slider_banner_speed.value()))
                    else:
                        _banner_apply_gradient(root_for_banners, False, int(slider_banner_speed.value()))
                        _banner_apply_colored(root_for_banners, bool(cb_banner_color.isChecked()))
            except Exception:
                pass

        def _on_banner_color_toggle(b: bool):
            s.setValue("banner_colored", bool(b))
            _sync_grey_state(cb_banner_color, bool(b))
            # If gradient mode is active, remember this setting but don't force a recolor now.
            try:
                if cb_banner_gradient.isChecked():
                    return
            except Exception:
                pass
            try:
                _banner_apply_colored(root_for_banners, bool(b))
            except Exception:
                pass

        def _on_banner_gradient_toggle(b: bool):
            s.setValue("banner_gradient_enabled", bool(b))
            _sync_grey_state(cb_banner_gradient, bool(b))
            try:
                cb_banner_color.setEnabled(not bool(b))
            except Exception:
                pass
            try:
                lab_banner_speed.setVisible(bool(cb_banner.isChecked()) and bool(b))
                slider_banner_speed.setVisible(bool(cb_banner.isChecked()) and bool(b))
            except Exception:
                pass
            try:
                if bool(cb_banner.isChecked()) and bool(b):
                    _banner_apply_gradient(root_for_banners, True, int(slider_banner_speed.value()))
                else:
                    _banner_apply_gradient(root_for_banners, False, int(slider_banner_speed.value()))
                    if bool(cb_banner.isChecked()):
                        _banner_apply_colored(root_for_banners, bool(cb_banner_color.isChecked()))
            except Exception:
                pass

        def _on_banner_gradient_speed(v: int):
            try:
                v = int(v)
            except Exception:
                v = 20
            if v < 1: v = 1
            if v > 50: v = 50
            s.setValue("banner_gradient_speed", int(v))
            try:
                if bool(cb_banner.isChecked()) and bool(cb_banner_gradient.isChecked()):
                    _banner_apply_gradient(root_for_banners, True, int(v))
            except Exception:
                pass

        cb_clear_pyc.toggled.connect(lambda b: _on_clear_pyc_toggle(b))
        cb_tooltips.toggled.connect(lambda b: _on_tooltips_toggle(b))
        cb_keep_settings.toggled.connect(lambda b: _on_keep_settings_toggle(b))
        cb_diag.toggled.connect(lambda b: _on_diag_toggle(b))
        cb_banner.toggled.connect(lambda b: _on_banner_toggle(b))
        cb_banner_color.toggled.connect(lambda b: _on_banner_color_toggle(b))
        cb_banner_gradient.toggled.connect(lambda b: _on_banner_gradient_toggle(b))
        slider_banner_speed.valueChanged.connect(lambda v: _on_banner_gradient_speed(v))

        v.addWidget(cb_clear_pyc)
        v.addWidget(cb_tooltips)
        v.addWidget(cb_keep_settings)
        v.addWidget(cb_diag)

        row_banner = QWidget(content)
        h_banner = QHBoxLayout(row_banner)
        h_banner.setContentsMargins(0,0,0,0)
        h_banner.setSpacing(12)
        h_banner.addWidget(cb_banner)
        h_banner.addWidget(cb_banner_color)
        h_banner.addWidget(cb_banner_gradient)
        h_banner.addWidget(lab_banner_speed)
        h_banner.addWidget(slider_banner_speed)
        h_banner.addStretch(1)
        v.addWidget(row_banner)
        # -- Animated buttons ----------------------------------------------------------
        cb_animbtn = QCheckBox("Animated buttons")
        cb_animbtn.setToolTip("If enabled, 'Generate' and 'View results' buttons will animate on hover.")
        animbtn_default = s.value("animated_buttons_enabled", False, type=bool)
        cb_animbtn.setChecked(animbtn_default)
        _sync_grey_state(cb_animbtn, bool(animbtn_default))

        # Style modes (only one can be enabled at a time)
        cb_anim_glow = QCheckBox("Glow")
        cb_anim_shift = QCheckBox("Color shift")
        cb_anim_boom = QCheckBox("Boomerang")
        cb_anim_outline = QCheckBox("Neon outline")
        cb_anim_scanline = QCheckBox("Scanline")
        cb_anim_shimmer = QCheckBox("Shimmer")
        cb_anim_pop = QCheckBox("Pop")
        cb_anim_rand = QCheckBox("Random (each hover)")

        cb_anim_glow.setToolTip("Soft glowing hover effect.")
        cb_anim_shift.setToolTip("Smooth color cycling on hover.")
        cb_anim_boom.setToolTip("Animated left-right-left gradient sweep on hover.")
        cb_anim_outline.setToolTip("Pulsing neon outline while hovered.")
        cb_anim_scanline.setToolTip("A bright scanline band sweeps top-to-bottom on hover.")
        cb_anim_shimmer.setToolTip("Diagonal shine that sweeps across the button.")
        cb_anim_pop.setToolTip("Micro pop feeling via center highlight and lift shadow.")
        cb_anim_rand.setToolTip("Pick a different effect each time you hover (all effects).")

        mode_default = (s.value("animated_buttons_mode", "glow", type=str) or "glow").strip().lower()
        if mode_default not in ("glow", "shift", "boomerang", "outline", "scanline", "shimmer", "pop", "random"):
            mode_default = "glow"

        row_anim = QWidget(content)
        h_anim = QHBoxLayout(row_anim)
        h_anim.setContentsMargins(0, 0, 0, 0)
        h_anim.setSpacing(12)
        h_anim.addWidget(cb_animbtn)
        h_anim.addStretch(1)
        v.addWidget(row_anim)

        row_anim_modes = QWidget(content)
        v_anim2 = QVBoxLayout(row_anim_modes)
        v_anim2.setContentsMargins(18, 0, 0, 0)
        v_anim2.setSpacing(6)

        row_anim_modes_1 = QWidget(row_anim_modes)
        h_anim2_1 = QHBoxLayout(row_anim_modes_1)
        h_anim2_1.setContentsMargins(0, 0, 0, 0)
        h_anim2_1.setSpacing(12)
        h_anim2_1.addWidget(cb_anim_glow)
        h_anim2_1.addWidget(cb_anim_shift)
        h_anim2_1.addWidget(cb_anim_boom)
        h_anim2_1.addStretch(1)
        v_anim2.addWidget(row_anim_modes_1)

        row_anim_modes_2 = QWidget(row_anim_modes)
        h_anim2_2 = QHBoxLayout(row_anim_modes_2)
        h_anim2_2.setContentsMargins(0, 0, 0, 0)
        h_anim2_2.setSpacing(12)
        h_anim2_2.addWidget(cb_anim_outline)
        h_anim2_2.addWidget(cb_anim_scanline)
        h_anim2_2.addStretch(1)
        v_anim2.addWidget(row_anim_modes_2)

        row_anim_modes_3 = QWidget(row_anim_modes)
        h_anim2_3 = QHBoxLayout(row_anim_modes_3)
        h_anim2_3.setContentsMargins(0, 0, 0, 0)
        h_anim2_3.setSpacing(12)
        h_anim2_3.addWidget(cb_anim_shimmer)
        h_anim2_3.addWidget(cb_anim_pop)
        h_anim2_3.addWidget(cb_anim_rand)
        h_anim2_3.addStretch(1)
        v_anim2.addWidget(row_anim_modes_3)

        row_anim_modes.setVisible(bool(animbtn_default))
        v.addWidget(row_anim_modes)

        def _anim_apply_to_app():
            try:
                app = QtWidgets.QApplication.instance()
                AnimatedButtonsManager.install(app)
                AnimatedButtonsManager.apply_from_settings(app)
            except Exception:
                pass

        def _anim_set_mode(mode: str):
            mode = (mode or "").strip().lower()
            if mode not in ("glow", "shift", "boomerang", "outline", "scanline", "shimmer", "pop", "random"):
                mode = "glow"
            try:
                s.setValue("animated_buttons_mode", mode)
            except Exception:
                pass

            # Ensure only one is active (checkbox look, radio behavior)
            items = [
                (cb_anim_glow, mode == "glow"),
                (cb_anim_shift, mode == "shift"),
                (cb_anim_boom, mode == "boomerang"),
                (cb_anim_outline, mode == "outline"),
                (cb_anim_scanline, mode == "scanline"),
                (cb_anim_shimmer, mode == "shimmer"),
                (cb_anim_pop, mode == "pop"),
                (cb_anim_rand, mode == "random"),
            ]
            for w, on in items:
                try:
                    w.blockSignals(True)
                    w.setChecked(bool(on))
                except Exception:
                    pass
                try:
                    w.blockSignals(False)
                except Exception:
                    pass
                _sync_grey_state(w, bool(on))

            _anim_apply_to_app()

        def _anim_on_main_toggle(b: bool):
            try:
                s.setValue("animated_buttons_enabled", bool(b))
            except Exception:
                pass
            _sync_grey_state(cb_animbtn, bool(b))
            try:
                row_anim_modes.setVisible(bool(b))
            except Exception:
                pass

            # If turning ON, ensure a mode is selected.
            if b:
                cur = (s.value("animated_buttons_mode", "glow", type=str) or "glow").strip().lower()
                if cur not in ("glow", "shift", "boomerang", "outline", "scanline", "shimmer", "pop", "random"):
                    cur = "glow"
                _anim_set_mode(cur)
            else:
                _anim_apply_to_app()

        cb_animbtn.toggled.connect(_anim_on_main_toggle)

        def _anim_on_sub_toggle(mode: str, b: bool):
            if not cb_animbtn.isChecked():
                return
            if b:
                _anim_set_mode(mode)
            else:
                # Prevent ending up with zero mode selected while enabled.
                cur = (s.value("animated_buttons_mode", "glow", type=str) or "glow").strip().lower()
                if cur == (mode or "").strip().lower():
                    QTimer.singleShot(0, lambda: _anim_set_mode(cur))

        cb_anim_glow.toggled.connect(lambda b: _anim_on_sub_toggle("glow", b))
        cb_anim_shift.toggled.connect(lambda b: _anim_on_sub_toggle("shift", b))
        cb_anim_boom.toggled.connect(lambda b: _anim_on_sub_toggle("boomerang", b))
        cb_anim_outline.toggled.connect(lambda b: _anim_on_sub_toggle("outline", b))
        cb_anim_scanline.toggled.connect(lambda b: _anim_on_sub_toggle("scanline", b))
        cb_anim_shimmer.toggled.connect(lambda b: _anim_on_sub_toggle("shimmer", b))
        cb_anim_pop.toggled.connect(lambda b: _anim_on_sub_toggle("pop", b))
        cb_anim_rand.toggled.connect(lambda b: _anim_on_sub_toggle("random", b))

        # Apply initial mode states
        _anim_set_mode(mode_default)

        # -- Font size (global) ----------------------------------------------------------
        cb_font = QCheckBox("Change font size")
        cb_font.setToolTip("Adjust the global UI font size. Turn on to reveal the slider (in pixels).")
        font_enabled_default = s.value("ui_font_size_enabled", False, type=bool)
        cb_font.setChecked(font_enabled_default)
        _sync_grey_state(cb_font, bool(font_enabled_default))

        # Slider row (only visible when enabled)
        row_font = QWidget(content)
        h_font = QHBoxLayout(row_font)
        h_font.setContentsMargins(18, 0, 0, 0)
        h_font.setSpacing(10)

        lab_font = QLabel("Font size: (click 'apply' theme change the font size)")
        slider_font = QtWidgets.QSlider(Qt.Horizontal, row_font)
        slider_font.setRange(-5, 5)
        slider_font.setSingleStep(1)
        slider_font.setPageStep(1)
        slider_font.setTickInterval(1)
        slider_font.setTickPosition(QtWidgets.QSlider.TicksBelow)

        delta_default = s.value("ui_font_size_delta_px", 0)
        try:
            delta_default = int(delta_default)
        except Exception:
            delta_default = 0
        if delta_default < -5:
            delta_default = -5
        if delta_default > 5:
            delta_default = 5
        slider_font.setValue(delta_default)

        lab_font_val = QLabel(f"{int(slider_font.value()):+d} px")
        lab_font_val.setMinimumWidth(52)

        h_font.addWidget(lab_font)
        h_font.addWidget(slider_font, 1)
        h_font.addWidget(lab_font_val)
        h_font.addStretch(1)

        row_font.setVisible(bool(font_enabled_default))

        def _font_apply_now():
            try:
                app = QtWidgets.QApplication.instance()
                _font_size_apply(app, bool(cb_font.isChecked()), int(slider_font.value()))
            except Exception:
                pass

        def _on_font_toggle(b: bool):
            try:
                s.setValue("ui_font_size_enabled", bool(b))
            except Exception:
                pass
            _sync_grey_state(cb_font, bool(b))
            try:
                row_font.setVisible(bool(b))
            except Exception:
                pass
            _font_apply_now()

        def _on_font_slider(vv: int):
            try:
                vv = int(vv)
            except Exception:
                vv = 0
            try:
                s.setValue("ui_font_size_delta_px", int(vv))
            except Exception:
                pass
            try:
                lab_font_val.setText(f"{int(vv):+d} px")
            except Exception:
                pass
            if cb_font.isChecked():
                _font_apply_now()

        cb_font.toggled.connect(_on_font_toggle)
        slider_font.valueChanged.connect(_on_font_slider)

        v.addWidget(cb_font)
        v.addWidget(row_font)

        # Apply initial setting once the UI exists (safe if already applied elsewhere).
        try:
            _font_apply_now()
        except Exception:
            pass

        # -- Emoji labels toggle --------------------------------------------------------
        cb_emoji = QCheckBox("Emoji labels")
        cb_emoji.setToolTip("Replace feature labels with emoji like 🔍 ⏱️ 📝 📸 🖌️ 🤖 ⏳ ⚙️/⚒️. "
                            "Turn off if your system font renders emoji poorly.")
        emoji_default = s.value("emoji_labels_enabled", True, type=bool)
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

        cb_emoji_tabs = QCheckBox("Show labels with emojis on the tabs")
        cb_emoji_tabs.setToolTip("When Emoji labels are enabled, show both the emoji and the text on tab titles.")
        emoji_tabs_default = s.value("emoji_tabs_show_labels", True, type=bool)
        cb_emoji_tabs.setChecked(emoji_tabs_default)
        _sync_grey_state(cb_emoji_tabs, bool(emoji_tabs_default))
        cb_emoji_tabs.setVisible(bool(emoji_default))

        def _on_emoji_tabs_toggle(b: bool):
            s.setValue("emoji_tabs_show_labels", bool(b))
            _sync_grey_state(cb_emoji_tabs, bool(b))
            try:
                win = page.window()
                if cb_emoji.isChecked():
                    restore_emoji_labels_globally(win)
                    apply_emoji_labels_globally(win)
            except Exception:
                pass

        cb_emoji_tabs.toggled.connect(lambda b: _on_emoji_tabs_toggle(b))

        row_emoji = QWidget(content)
        h_emoji = QHBoxLayout(row_emoji)
        h_emoji.setContentsMargins(0,0,0,0)
        h_emoji.setSpacing(12)
        h_emoji.addWidget(cb_emoji)
        h_emoji.addWidget(cb_emoji_tabs)
        h_emoji.addStretch(1)
        v.addWidget(row_emoji)

        # -- Swap left/right (tabs ↔ player) --------------------------------------------
        cb_rswap = QCheckBox("Swap left/right layout")
        cb_rswap.setToolTip("Swap the tabs and media player positions (tabs on left, player on right) without changing text direction.")
        swap_default = s.value("rtl_layout_enabled", False, type=bool)
        cb_rswap.setChecked(swap_default)
        _sync_grey_state(cb_rswap, bool(swap_default))

        def _find_main_splitter(root: QWidget):
            from PySide6.QtWidgets import QSplitter
            win = root.window() if hasattr(root, 'window') else root
            cands = win.findChildren(QSplitter)
            best = None
            best_total = -1
            for sp in cands:
                try:
                    total = sum(sp.sizes()) if sp.count() >= 2 else -1
                    if total > best_total:
                        best_total = total
                        best = sp
                except Exception:
                    pass
            return best

        def _apply_swap_main(root: QWidget, enabled: bool):
            sp = _find_main_splitter(root)
            if not sp or sp.count() < 2:
                return
            orig = sp.property('orig_lr')
            if not orig:
                orig = [sp.widget(0), sp.widget(1)]
                sp.setProperty('orig_lr', orig)
            left, right = orig[0], orig[1]
            if enabled:
                sp.insertWidget(0, right)
                sp.insertWidget(1, left)
            else:
                sp.insertWidget(0, left)
                sp.insertWidget(1, right)
            sp.update()

        def _on_swap_toggle(b: bool):
            s.setValue("rtl_layout_enabled", bool(b))
            _sync_grey_state(cb_rswap, bool(b))
            try:
                _apply_swap_main(page, bool(b))
            except Exception:
                pass

        cb_rswap.toggled.connect(lambda b: _on_swap_toggle(b))
        v.addWidget(cb_rswap)

        try:
            _apply_swap_main(page, bool(swap_default))
        except Exception:
            pass

        # -- Reorder tabs ---------------------------------------------------------------
        cb_tabloc = QCheckBox("Reorder tabs")
        cb_tabloc.setToolTip("When enabled, you can drag tabs to rearrange them.")
        tabloc_default = s.value("tabs_reorder_enabled", True, type=bool)
        cb_tabloc.setChecked(tabloc_default)
        _sync_grey_state(cb_tabloc, bool(tabloc_default))

        btn_reset_tabs = QPushButton("Reset to default")
        btn_reset_tabs.setToolTip("Reset the tab order back to the default layout.")
        btn_reset_tabs.setVisible(bool(tabloc_default))

        def _apply_tab_reorder_enabled(root: QWidget, enabled: bool):
            try:
                win = root.window() if hasattr(root, 'window') else root
                if hasattr(win, "set_tabs_reorder_enabled"):
                    win.set_tabs_reorder_enabled(bool(enabled))
                    return
                tabs = win.findChild(QTabWidget, "main_tabs")
                if tabs:
                    tabs.tabBar().setMovable(bool(enabled))
            except Exception:
                pass

        def _on_tabloc_toggle(b: bool):
            s.setValue("tabs_reorder_enabled", bool(b))
            _sync_grey_state(cb_tabloc, bool(b))
            btn_reset_tabs.setVisible(bool(b))
            _apply_tab_reorder_enabled(page, bool(b))

        def _on_reset_tabs():
            try:
                win = page.window()
                if hasattr(win, "reset_tab_order_to_default"):
                    win.reset_tab_order_to_default()
            except Exception:
                pass

        cb_tabloc.toggled.connect(lambda b: _on_tabloc_toggle(b))
        btn_reset_tabs.clicked.connect(_on_reset_tabs)

        btn_reset_tabs.setSizePolicy(QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed))

        row_tabloc = QWidget(content)
        h_tabloc = QHBoxLayout(row_tabloc)
        h_tabloc.setContentsMargins(0,0,0,0)
        h_tabloc.setSpacing(12)
        h_tabloc.addWidget(cb_tabloc)
        h_tabloc.addStretch(1)
        h_tabloc.addWidget(btn_reset_tabs)
        v.addWidget(row_tabloc)

        try:
            _apply_tab_reorder_enabled(page, bool(tabloc_default))
        except Exception:
            pass

        # Apply persisted banner settings once now that the UI exists
        try:
            _banner_apply_visibility(root_for_banners, bool(banner_default))
            # Gradient overrides the colored/grey switch
            if bool(banner_default) and bool(banner_grad_default):
                _banner_apply_gradient(root_for_banners, True, int(speed_default))
            else:
                _banner_apply_gradient(root_for_banners, False, int(speed_default))
                _banner_apply_colored(root_for_banners, bool(banner_color_default))
        except Exception:
            pass

        # -- Temperature units row -------------------------------------------------------
        row = QWidget(content)
        h2 = QHBoxLayout(row)
        h2.setContentsMargins(0,4,0,0)
        h2.setSpacing(8)
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

    def _apply_expand_state(is_open: bool):
        is_open = bool(is_open)

        # build content only when opening for the first time
        if is_open:
            _ensure_built()

        try:
            content.setVisible(is_open)
        except Exception:
            pass

        try:
            btn.setArrowType(Qt.DownArrow if is_open else Qt.RightArrow)
        except Exception:
            pass

        try:
            s.setValue("settings_options_expanded", is_open)
        except Exception:
            pass

        _force_relayout()

    def _on_toggled(b: bool):
        _apply_expand_state(bool(b))

    try:
        btn.toggled.connect(_on_toggled)
    except Exception:
        pass

    # Initial state
    _apply_expand_state(bool(expanded_default))

    return g

def _buttons_row(page: QWidget) -> QWidget:
    row = QWidget(page)
    h = QHBoxLayout(row); h.setContentsMargins(0,0,0,0); h.setSpacing(8)

    btn_cache = QPushButton("Clear program cache…")
    def do_cache():
        # Non-blocking cleanup: run in a separate process so the UI never freezes.
        try:
            s = QSettings("FrameVision", "FrameVision")

            dlg = QtWidgets.QDialog(page)
            dlg.setWindowTitle("Clear program cache")
            dlg.setModal(True)

            v = QtWidgets.QVBoxLayout(dlg)
            v.setContentsMargins(12, 12, 12, 12)
            v.setSpacing(10)

            tip = QtWidgets.QLabel("Tip: Do this every now and then to keep a clean running app", dlg)
            tip.setWordWrap(True)
            v.addWidget(tip)

            cb_pyc = QtWidgets.QCheckBox("Python cache (__pycache__ / .pyc)", dlg)
            cb_logs = QtWidgets.QCheckBox("Logs (older than 24 hours)", dlg)
            cb_thumbs = QtWidgets.QCheckBox("Thumbnails (last results) — remove items older than 7 days", dlg)
            cb_qt = QtWidgets.QCheckBox("Qt cache (user profile)", dlg)
            cb_temp = QtWidgets.QCheckBox("Temp folders (output/_temp, work)", dlg)

            # HuggingFace cache is intentionally hidden by default (it can be huge).
            cb_hf = QtWidgets.QCheckBox("HuggingFace cache (can be large)", dlg)
            cb_hf.setChecked(False)
            cb_hf.setVisible(False)

            cb_pyc.setChecked(s.value("cleanup_cache_pyc", True, type=bool))
            cb_logs.setChecked(s.value("cleanup_cache_logs", True, type=bool))
            cb_thumbs.setChecked(s.value("cleanup_cache_thumbs", True, type=bool))
            cb_qt.setChecked(s.value("cleanup_cache_qt", True, type=bool))
            cb_temp.setChecked(s.value("cleanup_cache_temp", True, type=bool))

            v.addWidget(cb_pyc)
            v.addWidget(cb_logs)
            v.addWidget(cb_thumbs)
            v.addWidget(cb_qt)
            v.addWidget(cb_temp)

            btns = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
                parent=dlg
            )
            btns.accepted.connect(dlg.accept)
            btns.rejected.connect(dlg.reject)
            v.addWidget(btns)

            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return

            # Persist choices
            s.setValue("cleanup_cache_pyc", bool(cb_pyc.isChecked()))
            s.setValue("cleanup_cache_logs", bool(cb_logs.isChecked()))
            s.setValue("cleanup_cache_thumbs", bool(cb_thumbs.isChecked()))
            s.setValue("cleanup_cache_qt", bool(cb_qt.isChecked()))
            s.setValue("cleanup_cache_temp", bool(cb_temp.isChecked()))

            opts = {
                "pyc": bool(cb_pyc.isChecked()),
                "logs": bool(cb_logs.isChecked()),
                "thumbs": bool(cb_thumbs.isChecked()),
                "qt": bool(cb_qt.isChecked()),
                "hf": bool(cb_hf.isChecked()),
                "temp": bool(cb_temp.isChecked()),
            }
        except Exception:
            # If anything goes wrong building the UI, fall back to safe defaults.
            opts = {"pyc": True, "logs": True, "thumbs": True, "qt": True, "hf": False, "temp": True}

        project_root = _project_root()
        script = os.path.join(project_root, "helpers", "cleanup_cache.py")

        if os.path.exists(script):
            args = [script, "--project-root", project_root, "--json"]
        else:
            # Fallback: attempt module execution
            args = ["-m", "helpers.cleanup_cache", "--project-root", project_root, "--json"]

        if not opts.get("pyc", True):
            args.append("--no-pyc")
        if not opts.get("logs", True):
            args.append("--no-logs")
        if not opts.get("thumbs", True):
            args.append("--no-thumbs")
        if not opts.get("qt", True):
            args.append("--no-qt")
        if opts.get("hf", False):
            args.append("--clean-hf-cache")
        if opts.get("temp", False):
            args.append("--clean-temp")

        # Progress dialog (indeterminate), stays responsive while cleanup runs.
        pd = QtWidgets.QProgressDialog("Cleaning cache…", "Cancel", 0, 0, page)
        pd.setWindowTitle("Cleaning…")
        pd.setWindowModality(Qt.WindowModal)
        pd.setMinimumDuration(0)
        pd.setAutoClose(False)
        pd.setAutoReset(False)

        proc = QtCore.QProcess(page)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        out_chunks: list[str] = []

        def _read_out():
            try:
                b = proc.readAllStandardOutput()
                if b:
                    out_chunks.append(bytes(b).decode("utf-8", "replace"))
            except Exception:
                pass

        def _finish(_code: int, _status: QtCore.QProcess.ExitStatus):
            try:
                _read_out()
            except Exception:
                pass
            try:
                pd.close()
            except Exception:
                pass

            txt = "".join(out_chunks).strip()
            result = None
            try:
                # cleanup_cache prints JSON when --json is used; try to parse the last JSON object in output
                jpos = txt.rfind("{")
                if jpos >= 0:
                    result = json.loads(txt[jpos:])
            except Exception:
                result = None

            if isinstance(result, dict):
                try:
                    lines = []
                    total = 0
                    for k in ("pycache", "pyc_pyo", "logs", "thumbs", "qt", "hf", "temp"):
                        if k in result:
                            v = int(result.get(k) or 0)
                            total += v
                            lines.append(f"{k}: {v}")
                    msg = "Cleanup finished.\n\nRemoved:\n" + "\n".join(lines) + f"\n\nTotal: {total}"
                except Exception:
                    msg = "Cleanup finished."
                QtWidgets.QMessageBox.information(page, "Cleanup finished", msg)
            else:
                # If JSON parsing failed, show whatever output we got (trimmed)
                if len(txt) > 3000:
                    txt = txt[-3000:]
                QtWidgets.QMessageBox.information(page, "Cleanup finished", txt or "Cleanup finished.")

            try:
                page.setProperty("_fv_cleanup_proc", None)
                page.setProperty("_fv_cleanup_pd", None)
            except Exception:
                pass

        def _on_error(_err: QtCore.QProcess.ProcessError):
            try:
                _read_out()
            except Exception:
                pass
            try:
                pd.close()
            except Exception:
                pass
            txt = "".join(out_chunks).strip()
            if len(txt) > 3000:
                txt = txt[-3000:]
            QtWidgets.QMessageBox.warning(page, "Cleanup failed", txt or "Cleanup process failed to start or crashed.")

        def _cancel():
            try:
                proc.kill()
            except Exception:
                pass
            try:
                pd.close()
            except Exception:
                pass

        proc.readyReadStandardOutput.connect(_read_out)
        proc.finished.connect(_finish)
        proc.errorOccurred.connect(_on_error)
        pd.canceled.connect(_cancel)

        # Keep references alive
        try:
            page.setProperty("_fv_cleanup_proc", proc)
            page.setProperty("_fv_cleanup_pd", pd)
        except Exception:
            pass

        proc.start(sys.executable, args)
        pd.show()
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
    lab.setMinimumHeight(260); lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
  #  lab.setStyleSheet("QLabel { background:#1a1a1a; color:#aaaaaa; border-radius:2px; }")
    lab.setAutoFillBackground(True)
    from helpers.kv_index import attach_click_hint
    attach_click_hint(lab)
    v.addWidget(lab)

    def refresh():
        # Avoid background UI-thread work when Settings tab is not visible.
        try:
            if not g.isVisible():
                return
        except Exception:
            pass
        try:
            from helpers.intro_data import get_logo_sources
            from helpers.intro_screen import _load_pixmap
            urls = get_logo_sources(theme=None)
            pm = QtGui.QPixmap()
            if urls:
                import random
                pm = _load_pixmap(random.choice(urls))
            if not pm.isNull():
                pm = pm.scaled(lab.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            lab.setPixmap(pm)
        except Exception:
            pass

    lab.resizeEvent = lambda _e: refresh()
    QTimer.singleShot(0, refresh)
    tm = QTimer(g); tm.setInterval(4500); tm.timeout.connect(refresh); tm.start()

    try:
        from helpers.overlay_animations import apply_intro_overlay_from_settings, stop_overlay
    except Exception:
        apply_intro_overlay_from_settings = None; stop_overlay = None

    def _sync_overlay_preview():
        # Avoid polling / overlay work when Settings tab is not visible.
        try:
            if not g.isVisible():
                return
        except Exception:
            pass
        try:
            s = QSettings('FrameVision','FrameVision')
            en = s.value('intro_overlay_enabled', True, type=bool)
            prev_ok = s.value('intro_overlay_preview_enabled', False, type=bool)
            theme = s.value('theme','Auto')
            st = getattr(lab, "_fv_overlay_state", None)
            if st is None:
                st = {"applied": False, "last_en": None, "last_theme": None}
                setattr(lab, "_fv_overlay_state", st)
            if not en or not prev_ok:
                if stop_overlay and st.get("applied", False):
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

# --- Fancy banner helpers -------------------------------------------------------
def _banner__iter_banners(root: QWidget):
    """Yield all QLabel instances that act as fancy banners (per-tab headers)."""
    if root is None:
        return
    if isinstance(root, QWidget):
        win = root.window() or root
    else:
        win = root
    if not isinstance(win, QWidget):
        return
    for lab in win.findChildren(QLabel):
        try:
            name = lab.objectName() or ""
        except Exception:
            name = ""
        if name.endswith("Banner"):
            yield lab

def _banner_grey_style_from(style: str) -> str:
    """Return a stylesheet string with the banner gradient forced to greyscale."""
    try:
        style = style or ""
        key = "background:"
        idx = style.find(key)
        grey_block = (
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #e0e0e0,"
            "   stop:0.5 #b0b0b0,"
            "   stop:1 #707070"
            " );"
        )
        if idx == -1:
            return style + grey_block
        semi = style.find(";", idx)
        if semi == -1:
            semi = len(style)
        return style[:idx] + grey_block + style[semi+1:]
    except Exception:
        return style

def _banner_apply_visibility(root: QWidget, enabled: bool) -> None:
    """Show or hide all fancy banners in the main window."""
    try:
        for lab in _banner__iter_banners(root):
            try:
                lab.setVisible(bool(enabled))
            except Exception:
                pass
    except Exception:
        pass

def _banner_apply_colored(root: QWidget, colored: bool) -> None:
    """
    Switch all fancy banners between their original colorful gradients (colored=True)
    and a neutral light-grey to dark-grey gradient (colored=False).
    """
    try:
        for lab in _banner__iter_banners(root):
            try:
                orig = lab.property("fv_banner_orig_style")
                if orig is None:
                    orig = lab.styleSheet() or ""
                    lab.setProperty("fv_banner_orig_style", orig)
                if colored:
                    lab.setStyleSheet(orig)
                else:
                    lab.setStyleSheet(_banner_grey_style_from(orig))
            except Exception:
                pass
    except Exception:
        pass


# --- Fancy banner gradient animation -----------------------------------------------
_FV_BANNER_GRADIENT_ANIM = None

def _banner_style_with_bg(style: str, bg_block: str) -> str:
    """Replace the first 'background:' block in a stylesheet with bg_block."""
    try:
        style = style or ""
        key = "background:"
        idx = style.find(key)
        if idx == -1:
            return style + bg_block
        semi = style.find(";", idx)
        if semi == -1:
            semi = len(style)
        return style[:idx] + bg_block + style[semi+1:]
    except Exception:
        return style or ""

class _BannerGradientAnimator(QtCore.QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._root = None
        self._phase = 0.0
        self._speed = 20
        self._timer = QTimer(self)
        self._timer.setInterval(55)  # smooth, lightweight
        self._timer.timeout.connect(self._tick)

    def set_root(self, root: QWidget):
        try:
            self._root = root.window() if isinstance(root, QWidget) else root
        except Exception:
            self._root = root

    def set_speed(self, v: int):
        try:
            v = int(v)
        except Exception:
            v = 20
        if v < 1: v = 1
        if v > 50: v = 50
        self._speed = v

    def start(self):
        if not self._timer.isActive():
            self._timer.start()
        self._tick()

    def stop(self):
        try:
            if self._timer.isActive():
                self._timer.stop()
        except Exception:
            pass
        # Restore original styles so normal colored/grey logic can take over
        try:
            if self._root is None:
                return
            for lab in _banner__iter_banners(self._root):
                try:
                    orig = lab.property("fv_banner_orig_style")
                    if orig is None:
                        orig = lab.styleSheet() or ""
                        lab.setProperty("fv_banner_orig_style", orig)
                    lab.setStyleSheet(orig)
                except Exception:
                    pass
        except Exception:
            pass

    def _tick(self):
        root = self._root
        if root is None:
            return
        # speed maps to degrees-per-tick; keep it smooth without CPU spikes
        self._phase = (self._phase + (float(self._speed) * 0.18)) % 360.0

        def _rgba(qc):
            return f"rgba({qc.red()},{qc.green()},{qc.blue()},255)"

        try:
            c1 = QtGui.QColor.fromHsvF((self._phase % 360.0) / 360.0, 0.85, 0.95)
            c2 = QtGui.QColor.fromHsvF(((self._phase + 60.0) % 360.0) / 360.0, 0.85, 0.95)
            c3 = QtGui.QColor.fromHsvF(((self._phase + 120.0) % 360.0) / 360.0, 0.85, 0.95)
            bg = (
                " background: qlineargradient("
                "   x1:0, y1:0, x2:1, y2:0,"
                f"   stop:0 {_rgba(c1)},"
                f"   stop:0.5 {_rgba(c2)},"
                f"   stop:1 {_rgba(c3)}"
                " );"
            )
        except Exception:
            return

        for lab in _banner__iter_banners(root):
            try:
                orig = lab.property("fv_banner_orig_style")
                if orig is None:
                    orig = lab.styleSheet() or ""
                    lab.setProperty("fv_banner_orig_style", orig)
                lab.setStyleSheet(_banner_style_with_bg(orig, bg))
            except Exception:
                pass

def _banner_apply_gradient(root: QWidget, enabled: bool, speed: int = 20) -> None:
    """Enable/disable animated banner gradients for all *Banner labels."""
    global _FV_BANNER_GRADIENT_ANIM
    try:
        if _FV_BANNER_GRADIENT_ANIM is None:
            _FV_BANNER_GRADIENT_ANIM = _BannerGradientAnimator()
        _FV_BANNER_GRADIENT_ANIM.set_root(root)
        _FV_BANNER_GRADIENT_ANIM.set_speed(int(speed))
        if bool(enabled):
            _FV_BANNER_GRADIENT_ANIM.start()
        else:
            _FV_BANNER_GRADIENT_ANIM.stop()
    except Exception:
        pass

# --- Fancy banner startup sync --------------------------------------------------

def _banner_apply_from_qsettings(root: QWidget) -> bool:
    """Apply banner settings from QSettings to all *Banner labels.

    This is intentionally safe + idempotent so it can run during startup even
    when the Settings tab UI has never been opened.

    Returns True if at least one banner was found/applied.
    """
    try:
        s = QSettings("FrameVision", "FrameVision")
        enabled = bool(s.value("banner_enabled", True, type=bool))
        colored = bool(s.value("banner_colored", True, type=bool))
        grad_en = bool(s.value("banner_gradient_enabled", False, type=bool))
        speed = int(s.value("banner_gradient_speed", 20, type=int) or 20)
        if speed < 1:
            speed = 1
        if speed > 50:
            speed = 50
    except Exception:
        enabled, colored, grad_en, speed = True, True, False, 20

    # Capture the *current* banner stylesheet as "orig" before we mutate it.
    found = False
    try:
        for lab in _banner__iter_banners(root):
            found = True
            try:
                cur = lab.styleSheet() or ""
                orig = lab.property("fv_banner_orig_style")
                if orig is None or not str(orig).strip():
                    lab.setProperty("fv_banner_orig_style", cur)
            except Exception:
                pass
    except Exception:
        pass

    if not found:
        return False

    # Apply state.
    try:
        _banner_apply_visibility(root, bool(enabled))
    except Exception:
        pass

    # Gradient overrides colored/grey.
    try:
        if not bool(enabled):
            _banner_apply_gradient(root, False, int(speed))
            return True

        if bool(grad_en):
            _banner_apply_gradient(root, True, int(speed))
        else:
            _banner_apply_gradient(root, False, int(speed))
            _banner_apply_colored(root, bool(colored))
    except Exception:
        pass

    return True


def _banner_install_startup_sync() -> None:
    """Apply banner settings shortly after the UI exists.

    Fixes the case where banner preferences only "stick" after the Settings tab
    is opened (because the Options group is lazily built).
    """
    try:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        state = {"tries": 0, "done": False}

        def _pick_main_window():
            try:
                wins = list(app.topLevelWidgets())
            except Exception:
                wins = []
            for w in wins:
                if not isinstance(w, QWidget):
                    continue
                try:
                    tabs = w.findChild(QTabWidget, "main_tabs") or getattr(w, "tabs", None)
                except Exception:
                    tabs = None
                if isinstance(tabs, QTabWidget):
                    return w
            # fallback: any widget with banners
            for w in wins:
                if not isinstance(w, QWidget):
                    continue
                try:
                    for _ in _banner__iter_banners(w):
                        return w
                except Exception:
                    pass
            return None

        def _tick():
            if state["done"]:
                return
            state["tries"] += 1

            win = _pick_main_window()
            if win is not None:
                try:
                    if _banner_apply_from_qsettings(win):
                        state["done"] = True
                        return
                except Exception:
                    pass

            # keep trying briefly; banners may be created a few frames later
            if state["tries"] < 80:
                QTimer.singleShot(75, _tick)

        # kick once the event loop starts
        QTimer.singleShot(0, _tick)
    except Exception:
        pass



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
    "upscaling": "🧪",
    "engine": "🚀",
    "interpolator": "⏱️",
    "fps": "⏱️",
    "model": "🧩️",
    "cancel": "❌️",
    "planner": "📝",
    "character": "🎭",
    "storymode": "🐉",
    "generation": "🏡",
    "format": "🖼️",   
    "describe": "🖋️",
    "log": "📝",
    "steps": "👣",
    "folder": "🗂️",
#    "explorer": "🗂️"
    "speech": "🎙️",
    "generate": "✨",
    "batch": "📦",
    "clip": "🎬",
    "info": "💡",
    "txt2img": "📸",
    "loader": "📸",
    "txt to img": "📸",
    "Run": "🏃️",
    "pending": "💤️",
    "background": "🖌️",
    "preset": "💾️",
    "inpaint": "💫",
    "profile": "🧍",
    "queue": "⏳",
    "Queue": "⏳",
    "split": "📽️",
    "hunyuan": "📽️",
    "video": "📽️",
    "settings": "️⚙️",
    "tool": "🧱",
    "txttoimg": "📸",
    "thumbnail": "🖼️",
    "enhancement": "📝",
    "mixer": "🎼",
    "music": "🎶",
    "reverse": "⏪",
    "whisper": "💬",
    "metadata": "🏷️",
    "speedup": "🐢",
    "frames": "🎞️",
    "animated": "🔮️",
    "trim": "✂️",
    "cropping": "📐",
    "images": "📸️",
    "rename": "📗️",
    "copy": "🧬",
    "sound": "🎵️",
    "videoclip": "📺",
    "preview": "👁️",
    "wan22": "🎬",
    "video input": "🎬",
    "file": "📂",
    "rifefps": "⏱️",
    "interpolation": "⏱️",
    "ace": "🎵️",
    "seed": "🌱️",
    "rife fps": "⏱️",
    "cpu": "⚡️",
    "2511": "⚡️",
    "edit": "🖼️",
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
    "Update": "🌟",
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
        return bool(s.value("emoji_labels_enabled", True, type=bool))
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
        return bool(s.value("emoji_tabs_show_labels", True, type=bool))
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
    """Install FrameVision Settings UI into the Settings tab.

    This version makes the Settings tab self-contained and scrollable (single scroll area),
    and wipes any legacy placeholder/header UI on that tab to prevent duplicates.
    """
    try:
        page = _locate_settings_container(main_window)
        if not page:
            return

        def _make_settings_banner(parent: QWidget) -> QLabel:
            """Create the Settings tab banner label (fixed-height)."""
            banner = QLabel("Settings", parent)
            banner.setObjectName("settingsBanner")
            banner.setAlignment(Qt.AlignCenter)
            banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            banner.setFixedHeight(48)
            banner.setStyleSheet(
                "#settingsBanner {"
                " font-size: 15px;"
                " font-weight: 600;"
                " padding: 8px 17px;"
                " border-radius: 12px;"
                " margin: 0 0 6px 0;"
                " color: white;"
                " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                "   stop:0 rgba(255,75,75,255), stop:1 rgba(200,35,35,255)"
                " );"
                " letter-spacing: 0.5px;"
                "}"
            )
            return banner

        # Prevent double-install (can happen if called from multiple places).
        if getattr(page, "_fv_settings_installed", False):
            return
        setattr(page, "_fv_settings_installed", True)

        # Prepare a sticky banner + a dedicated scroll area/content widget.
        # The banner lives OUTSIDE the scroll area so it always stays visible.
        banner_host = None
        if isinstance(page, QScrollArea):
            # If the tab page itself is already a scroll area, convert it into a
            # simple container (no scrollbars) and place our own inner scroll area.
            scroll_outer = page
            scroll_outer.setWidgetResizable(True)
            try:
                scroll_outer.setFrameShape(QtWidgets.QFrame.NoFrame)
            except Exception:
                pass
            try:
                scroll_outer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                scroll_outer.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            except Exception:
                pass

            wrapper = QWidget(scroll_outer)
            wlay = QVBoxLayout(wrapper)
            wlay.setContentsMargins(0, 0, 0, 0)
            wlay.setSpacing(0)

            header = QWidget(wrapper)
            hlay = QVBoxLayout(header)
            hlay.setContentsMargins(12, 12, 12, 8)
            hlay.setSpacing(0)
            banner = _make_settings_banner(header)
            hlay.addWidget(banner)
            wlay.addWidget(header)
            banner_host = header

            scroll = QScrollArea(wrapper)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
            content = QWidget(scroll)
            scroll.setWidget(content)
            wlay.addWidget(scroll, 1)

            scroll_outer.setWidget(wrapper)
        else:
            # Wipe the entire tab page so we don't keep legacy placeholder UI above our banner.
            outer_lay = page.layout()
            if outer_lay is None:
                outer_lay = QVBoxLayout(page)
                page.setLayout(outer_lay)
            _wipe_layout(outer_lay)
            outer_lay.setContentsMargins(0, 0, 0, 0)
            outer_lay.setSpacing(0)

            header = QWidget(page)
            hlay = QVBoxLayout(header)
            hlay.setContentsMargins(12, 12, 12, 8)
            hlay.setSpacing(0)
            banner = _make_settings_banner(header)
            hlay.addWidget(banner)
            outer_lay.addWidget(header)
            banner_host = header

            scroll = QScrollArea(page)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

            content = QWidget(scroll)
            scroll.setWidget(content)
            outer_lay.addWidget(scroll, 1)

        # Respect the saved banner enabled state immediately (avoid a 1-frame flash).
        try:
            _s = QSettings("FrameVision", "FrameVision")
            en = bool(_s.value("banner_enabled", True, type=bool))
            try:
                banner.setVisible(en)
            except Exception:
                pass
            if banner_host is not None:
                try:
                    banner_host.setVisible(en)
                except Exception:
                    pass
        except Exception:
            pass

        # settings_more expects to be able to find this widget reliably.
        try:
            content.setObjectName("FvSettingsContent")
        except Exception:
            pass

        try:
            page.setObjectName("FvSettingsPage")
        except Exception:
            pass

        # Build content layout (this is what scrolls).
        lay = content.layout()
        if not isinstance(lay, QVBoxLayout):
            lay = QVBoxLayout(content)
            content.setLayout(lay)
        _wipe_layout(lay)
        # Top margin is handled by the sticky banner above.
        lay.setContentsMargins(12, 0, 12, 12)
        lay.setSpacing(12)

        content.setStyleSheet(
            "QGroupBox { margin-top: 10px; padding-top: 16px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }"
        )

        # (Banner is created above the scroll area so it stays sticky.)

        lay.addWidget(_theme_row(content))
        lay.addWidget(_options_group(content))
        lay.addWidget(_buttons_row(content))
        try:
            from PySide6.QtWidgets import QFrame
            hr = QFrame(content)
            hr.setFrameShape(QFrame.HLine)
            hr.setFrameShadow(QFrame.Sunken)
            lay.addWidget(hr)
        except Exception:
            pass
        lay.addWidget(_logo_group(content))
        lay.addStretch(1)

        # Auto-apply emoji labels if previously enabled
        try:
            _apply_emoji_on_start(main_window)
        except Exception:
            pass


        # Auto-apply font size from saved settings
        try:
            _font_size_apply_from_qsettings(QtWidgets.QApplication.instance())
        except Exception:
            pass


# Auto-install animated buttons manager (applies saved settings)
        try:
            app = QtWidgets.QApplication.instance()
            AnimatedButtonsManager.install(app)
            AnimatedButtonsManager.apply_from_settings(app)
        except Exception:
            pass


        # Minimal wiring: delegate Easter Eggs UI injection + tracker to settings_more
        try:
            from helpers import settings_more as _sm
            tabs = main_window.findChild(QTabWidget, "main_tabs") or getattr(main_window, "tabs", None)

            def _maybe_install_settings_more():
                try:
                    if getattr(page, "_fv_settings_more_installed", False):
                        return
                    if isinstance(tabs, QTabWidget):
                        cur = tabs.currentWidget()
                        if cur is not page and cur is not scroll:
                            return
                    _sm.install_settings_more(main_window)
                    setattr(page, "_fv_settings_more_installed", True)
                except Exception:
                    pass

            if isinstance(tabs, QTabWidget):
                try:
                    tabs.currentChanged.connect(lambda _i: _maybe_install_settings_more())
                except Exception:
                    pass

            QtCore.QTimer.singleShot(0, _maybe_install_settings_more)
        except Exception:
            pass

    except Exception:
        pass



# Ensure banner settings (including gradient mode) are applied during startup
# even before the Settings tab is opened.
try:
    _banner_install_startup_sync()
except Exception:
    pass
