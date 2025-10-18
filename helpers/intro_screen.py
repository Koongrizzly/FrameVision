# helpers/intro_screen.py — call theme auto-wire at startup
from __future__ import annotations
import os, random, hashlib
from urllib.request import urlopen, Request

from PySide6.QtCore import Qt, QSettings, QTimer, QEventLoop, QRect, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen, QWidget, QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
# --- Boot overlay animation support ---
try:
    from .overlay_animations import apply_intro_overlay_from_settings, stop_overlay  # type: ignore
except Exception:
    try:
        from helpers.overlay_animations import apply_intro_overlay_from_settings, stop_overlay  # type: ignore
    except Exception:
        apply_intro_overlay_from_settings = None
        stop_overlay = None

# startup cleanup
try:
    from .startup_cleanup import run_startup_cleanup, auto_install_cleanup_settings
except Exception:
    def run_startup_cleanup(): pass
    def auto_install_cleanup_settings(*a, **k): pass

# sysmon settings installer
try:
    from .sysmon import auto_install_sysmon_settings
except Exception:
    def auto_install_sysmon_settings(*a, **k): pass

# HUD colorizer
try:
    from .hud_colorizer import auto_install_hud_colorizer
except Exception:
    def auto_install_hud_colorizer(*a, **k): pass

# Settings scroll wrapper
try:
    from .settings_scroll import auto_wrap_settings_scroll
except Exception:
    def auto_wrap_settings_scroll(*a, **k): pass

# Scoped spacing (Settings only)
try:
    from .ui_tweaks import install_settings_spacing
except Exception:
    def install_settings_spacing(*a, **k): pass

# Theme auto-apply
try:
    from .theme_autowire import auto_wire_theme_combo
except Exception:
    def auto_wire_theme_combo(*a, **k): pass

ORG="FrameVision"; APP="FrameVision"
CACHE_DIR = os.path.join(os.path.expanduser("~"), "FrameVision", "cache", "intro_images")

def _fallback_pixmap() -> QPixmap:
    pm = QPixmap(640, 360); pm.fill(Qt.black); return pm

def _load_pixmap(src: str) -> QPixmap:
    try:
        if src.startswith(("http://","https://")):
            os.makedirs(CACHE_DIR, exist_ok=True)
            h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]
            fn = os.path.join(CACHE_DIR, f"{h}_{os.path.basename(src).split('?')[0]}")
            if not os.path.exists(fn):
                req = Request(src, headers={"User-Agent":"Mozilla/5.0"})
                with urlopen(req, timeout=5) as r, open(fn, "wb") as f: f.write(r.read())
            pm = QPixmap(fn)
            if not pm.isNull(): return pm
        pm = QPixmap(src)
        if not pm.isNull(): return pm
    except Exception:
        pass
    return _fallback_pixmap()

def _scaled_for_screen(pm: QPixmap, scale_percent: int, fullscreen: bool, stretch: bool) -> QPixmap:
    try:
        scr = QApplication.primaryScreen()
        geo = scr.availableGeometry() if scr else QRect(0,0,1280,720)
        if fullscreen and stretch:   return pm.scaled(geo.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        if fullscreen and not stretch: return pm.scaled(geo.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        target_w = max(600, int(geo.width() * max(10, min(100, scale_percent)) / 100.0))
        return pm.scaledToWidth(target_w, Qt.SmoothTransformation)
    except Exception:
        return pm

def _nice_dialog_style():
    s = QSettings(ORG, APP); theme = (s.value("theme","") or "").lower()
    if "night" in theme or "dark" in theme:
        return """
            QDialog{ background:#0e0e0e; }
            QLabel#joke{ background:#1e1e1e; color:#e6e6e6; padding:12px; border-radius:12px; font-size:14pt; }
            QPushButton{ padding:8px 14px; }
        """
    return """
        QDialog{ background:#f3f3f3; }
        QLabel#joke{ background:#ffffff; color:#111; padding:12px; border-radius:12px; font-size:14pt; }
        QPushButton{ padding:8px 14px; }
    """

def _show_joke_dialog_modeless(text: str, lifetime_ms:int=2800):
    try:
        dlg = QDialog(None, Qt.WindowStaysOnTopHint)
        dlg.setModal(False)
        dlg.setStyleSheet(_nice_dialog_style())
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        lay = QVBoxLayout(dlg); lay.setContentsMargins(18,18,18,18); lay.setSpacing(12)
        lbl = QLabel(text); lbl.setObjectName("joke"); lbl.setWordWrap(True); lay.addWidget(lbl)
        row = QHBoxLayout(); row.addStretch(1); btn = QPushButton("😂  Nice"); row.addWidget(btn); lay.addLayout(row)
        btn.clicked.connect(dlg.close)
        dlg.adjustSize()
        scr = QApplication.primaryScreen(); geo = scr.availableGeometry() if scr else QRect(0,0,1280,720)
        x = geo.center().x() - dlg.width()//2; y = geo.center().y() - dlg.height()//2
        dlg.move(max(geo.left(), x), max(geo.top(), y))
        # Auto-close disabled: wait for user to click the button to dismiss
        dlg.show()
        return dlg
    except Exception:
        return None

def _get_random_joke_text() -> str:
    try:
        from .kv_index import get_random_joke
        res = get_random_joke(None)
        if isinstance(res, tuple): return res[0]
        return res
    except Exception:
        return "I used to be a banker but I lost interest."

class EggSplash(QSplashScreen):
    def __init__(self, pixmap: QPixmap, on_time_ms: int, fullscreen: bool):
        super().__init__(pixmap, Qt.WindowStaysOnTopHint | Qt.SplashScreen | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self._clicks = 0
        self._reset_timer = QTimer(self); self._reset_timer.setSingleShot(True); self._reset_timer.setInterval(2500)
        self._reset_timer.timeout.connect(lambda: setattr(self, "_clicks", 0))

        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.setInterval(max(1200, on_time_ms))
        self._timer.timeout.connect(self.close)

        self._joke_keep = None

    def mousePressEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self._clicks += 1
            self._reset_timer.start()
            if self._clicks >= 4:
                self._clicks = 0
                text = _get_random_joke_text()
                self._joke_keep = _show_joke_dialog_modeless(text)
                return
        return

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close(); return
        return

def show_intro_if_enabled(main_window: QWidget | None = None):
    try:
        # Startup maintenance + UI installers
        run_startup_cleanup()
        QTimer.singleShot(0, auto_install_cleanup_settings)
        QTimer.singleShot(0, auto_install_sysmon_settings)
        QTimer.singleShot(0, auto_install_hud_colorizer)
        QTimer.singleShot(0, auto_wrap_settings_scroll)
        QTimer.singleShot(0, install_settings_spacing)
        QTimer.singleShot(0, auto_wire_theme_combo)  # <-- new

        s = QSettings(ORG, APP)
        if not bool(s.value("intro_enabled", True)): return None
        follow = bool(s.value("intro_follow_theme", False))
        fullscreen = bool(s.value("intro_fullscreen", True))
        scale_percent = int(s.value("intro_scale_percent", 100) or 100)
        stretch = bool(s.value("intro_stretch", True))
        duration_ms = int(s.value("intro_duration_ms", 4000) or 4000)
        if duration_ms < 2500: duration_ms = 2500

        try:
            from .intro_data import get_logo_sources
            sources = get_logo_sources() or []
            theme = (s.value("theme","") or "").lower()
            if follow and theme:
                themed = [u for u in sources if theme in os.path.basename(u).lower() or theme in str(u).lower()]
                if themed: sources = themed
        except Exception:
            sources = []

        pm = _fallback_pixmap()
        if sources:
            try: pm = _load_pixmap(random.choice(sources)) or pm
            except Exception: pass
        pm = _scaled_for_screen(pm, scale_percent, fullscreen, stretch)

        sp = EggSplash(pm, on_time_ms=duration_ms, fullscreen=fullscreen)
        scr = QApplication.primaryScreen(); geo = scr.availableGeometry() if scr else QRect(0,0,1280,720)
        sz: QSize = sp.pixmap().size(); x = geo.center().x() - sz.width()//2; y = geo.center().y() - sz.height()//2
        sp.move(max(geo.left(), x), max(geo.top(), y))
        sp.show(); QApplication.processEvents()
        # Start overlay on splash according to settings (safe try)
        if apply_intro_overlay_from_settings:
            try:
                theme_name = (s.value('theme','') or 'Auto')
                apply_intro_overlay_from_settings(sp, theme_name, force_topmost=False)
            except Exception:
                pass
        return sp
    except Exception:
        return None

def run_intro_if_enabled(main_window: QWidget | None = None):
    sp = show_intro_if_enabled(main_window)
    if sp is None:
        return None

    # Pre-show the main window shortly before the splash ends to avoid any flicker.
    def _fv_preschedule_main_show():
        try:
            if main_window is not None and not main_window.isVisible():
                main_window.show()
        except Exception:
            pass

    # Start a brief dim animation INSIDE the splash (no transparency), so it's gentle on the eyes.
    def _fv_start_dim(duration_ms: int = 500):
        try:
            if sp is None or not sp.isVisible():
                return
            if not hasattr(sp, "_fv_black_cover") or sp._fv_black_cover is None:
                cover = QWidget(sp)
                cover.setObjectName("_fv_black_cover")
                cover.setAttribute(Qt.WA_TransparentForMouseEvents, True)
                cover.setGeometry(0, 0, sp.width(), sp.height())
                cover.show()
                cover.raise_()
                sp._fv_black_cover = cover
            # Animate to black by increasing alpha over time using a timer
            steps = max(5, int(duration_ms / 16))  # ~60 fps target
            # Reset any prior dimmer
            if hasattr(sp, "_fv_dim_timer") and sp._fv_dim_timer:
                try: sp._fv_dim_timer.stop()
                except Exception: pass
            sp._fv_dim_step = 0
            sp._fv_dim_timer = QTimer(sp)
            def _tick():
                try:
                    sp._fv_dim_step += 1
                    t = min(1.0, sp._fv_dim_step / float(steps))
                    a = max(0, min(255, int(t * 255)))
                    sp._fv_black_cover.setStyleSheet(f"background: rgba(0,0,0,{a});")
                    if sp._fv_dim_step >= steps:
                        sp._fv_dim_timer.stop()
                except Exception:
                    try: sp._fv_dim_timer.stop()
                    except Exception: pass
            sp._fv_dim_timer.timeout.connect(_tick)
            sp._fv_dim_timer.start(max(10, int(duration_ms / max(1, steps))))
        except Exception:
            pass

    try:
        try:
            interval = int(sp._timer.interval())
        except Exception:
            interval = 2000  # reasonable default
        dim_duration = 500  # lengthen the transition a little
        pre_ms = 250        # show main ~250ms before the splash ends
        dim_at = max(0, interval - dim_duration)
        pre_at = max(0, interval - pre_ms)

        QTimer.singleShot(dim_at, lambda: _fv_start_dim(dim_duration))
        QTimer.singleShot(pre_at, _fv_preschedule_main_show)

        # Drive the splash lifetime (unchanged)
        loop = QEventLoop()
        sp._timer.timeout.connect(loop.quit)
        sp._timer.start()
        loop.exec()
    except Exception:
        pass
    return sp
