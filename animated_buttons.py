# helpers/animated_buttons.py
# Small UI-polish feature: animated hover backgrounds for selected buttons.
# Default matches (case-insensitive): "generate", "view results"
#
# Modes:
#  - glow: animated drop-shadow glow
#  - shift: background color hue-shift while hovering
#  - boomerang: animated horizontal gradient "left→right→left"
#  - random: picks one of (glow/shift/boomerang) on app start (saved until restart)

from __future__ import annotations

import random
from typing import Dict, Optional, List, Any

from PySide6.QtCore import QObject, QEvent, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QPushButton, QGraphicsDropShadowEffect
from PySide6.QtCore import QSettings


class AnimatedButtonsManager(QObject):
    def __init__(self, root=None, settings: Optional[QSettings] = None, keywords: Optional[List[str]] = None):
        super().__init__(root if root is not None else QApplication.instance())
        self._qs = settings if settings is not None else QSettings("FrameVision", "FrameVision")
        self.keywords = [k.lower().strip() for k in (keywords or ["generate", "view results"]) if (k or "").strip()]
        self._enabled = False
        self._mode = "glow"

        # btn -> state dict
        self._states: Dict[QPushButton, Dict[str, Any]] = {}
        self._tracked = set()

        # initial settings
        try:
            self._enabled = bool(self._qs.value("UI/animated_buttons_enabled", False, type=bool))
        except Exception:
            self._enabled = False
        try:
            self._mode = str(self._qs.value("UI/animated_buttons_mode", "glow", type=str) or "glow").strip().lower()
        except Exception:
            self._mode = "glow"

        # scan a few times (UI is still constructing in many apps)
        self._scan_count = 0
        self._scan_timer = QTimer(self)
        self._scan_timer.setInterval(700)
        self._scan_timer.timeout.connect(self._scan_buttons)
        self._scan_timer.start()
        QTimer.singleShot(0, self._scan_buttons)
        QTimer.singleShot(2500, self._scan_buttons)
        QTimer.singleShot(6000, self._scan_buttons)

    # ---------- Public API ----------
    def set_enabled(self, on: bool):
        self._enabled = bool(on)
        try:
            self._qs.setValue("UI/animated_buttons_enabled", bool(self._enabled))
        except Exception:
            pass
        if not self._enabled:
            # stop any running animations
            for btn in list(self._states.keys()):
                self._stop(btn)

    def set_mode(self, mode: str):
        mode = (mode or "glow").strip().lower()
        if mode not in ("glow", "shift", "boomerang", "random"):
            mode = "glow"
        self._mode = mode
        try:
            self._qs.setValue("UI/animated_buttons_mode", mode)
        except Exception:
            pass

        # if currently hovering, restart in new mode
        if self._enabled:
            for btn, st in list(self._states.items()):
                if st.get("hovering"):
                    self._start(btn)

    # ---------- Internals ----------
    def _effective_mode(self) -> str:
        mode = (self._mode or "glow").strip().lower()
        if mode != "random":
            return mode

        # Random pick persists until restart (stored key).
        pick = None
        try:
            pick = str(self._qs.value("UI/animated_buttons_random_pick", "", type=str) or "").strip().lower()
        except Exception:
            pick = ""

        if pick not in ("glow", "shift", "boomerang"):
            pick = random.choice(["glow", "shift", "boomerang"])
            try:
                self._qs.setValue("UI/animated_buttons_random_pick", pick)
            except Exception:
                pass
        return pick

    def _matches(self, btn: QPushButton) -> bool:
        try:
            t = (btn.text() or "").lower()
        except Exception:
            return False
        if not t:
            return False
        for kw in self.keywords:
            if kw and kw in t:
                return True
        return False

    def _scan_buttons(self):
        try:
            self._scan_count += 1
            # Stop timer after a few scans to avoid any long-term overhead.
            if self._scan_count >= 10:
                try:
                    self._scan_timer.stop()
                except Exception:
                    pass

            app = QApplication.instance()
            if app is None:
                return
            for w in app.allWidgets():
                if not isinstance(w, QPushButton):
                    continue
                if w in self._tracked:
                    continue
                if not self._matches(w):
                    continue
                self._tracked.add(w)
                self._states[w] = {
                    "orig_ss": w.styleSheet() or "",
                    "orig_eff": w.graphicsEffect(),
                    "timer": None,
                    "anim": None,
                    "phase": 0.0,
                    "dir": 1.0,
                    "hovering": False,
                }
                try:
                    w.installEventFilter(self)
                except Exception:
                    pass
        except Exception:
            pass

    def eventFilter(self, obj, ev):
        try:
            if isinstance(obj, QPushButton) and obj in self._states:
                et = ev.type()
                if et == QEvent.Enter:
                    self._states[obj]["hovering"] = True
                    if self._enabled:
                        self._start(obj)
                elif et in (QEvent.Leave, QEvent.HoverLeave):
                    self._states[obj]["hovering"] = False
                    self._stop(obj)
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # ---------- Effects ----------
    def _start(self, btn: QPushButton):
        self._stop(btn)
        mode = self._effective_mode()

        if mode == "glow":
            self._start_glow(btn)
        elif mode == "shift":
            self._start_shift(btn)
        elif mode == "boomerang":
            self._start_boomerang(btn)
        else:
            self._start_glow(btn)

    def _stop(self, btn: QPushButton):
        st = self._states.get(btn)
        if not st:
            return

        # stop timers/animations
        try:
            t = st.get("timer")
            if t is not None:
                t.stop()
        except Exception:
            pass
        try:
            a = st.get("anim")
            if a is not None:
                a.stop()
        except Exception:
            pass

        # restore style/effect
        try:
            btn.setStyleSheet(st.get("orig_ss", "") or "")
        except Exception:
            pass
        try:
            btn.setGraphicsEffect(st.get("orig_eff", None))
        except Exception:
            pass

        st["timer"] = None
        st["anim"] = None
        st["phase"] = 0.0
        st["dir"] = 1.0

    def _base_colors(self, btn: QPushButton):
        pal = btn.palette()
        try:
            c_btn = pal.color(QPalette.Button)
        except Exception:
            c_btn = QColor(60, 60, 60)
        try:
            c_hi = pal.color(QPalette.Highlight)
        except Exception:
            c_hi = QColor(90, 140, 255)
        # Ensure visible contrast
        if c_hi == c_btn:
            c_hi = c_hi.lighter(130)
        return c_btn, c_hi

    def _start_glow(self, btn: QPushButton):
        try:
            _, c_hi = self._base_colors(btn)
            eff = QGraphicsDropShadowEffect(btn)
            eff.setOffset(0, 0)
            eff.setColor(QColor(c_hi.red(), c_hi.green(), c_hi.blue(), 190))
            eff.setBlurRadius(10.0)
            btn.setGraphicsEffect(eff)

            anim = QPropertyAnimation(eff, b"blurRadius", btn)
            anim.setDuration(650)
            anim.setStartValue(8.0)
            anim.setEndValue(22.0)
            anim.setEasingCurve(QEasingCurve.InOutSine)
            anim.setLoopCount(-1)
            anim.start()

            st = self._states.get(btn)
            if st is not None:
                st["anim"] = anim
        except Exception:
            pass

    def _start_shift(self, btn: QPushButton):
        st = self._states.get(btn)
        if not st:
            return
        c_btn, c_hi = self._base_colors(btn)

        # seed hue from highlight for theme-awareness
        base = QColor(c_hi)
        h, s, v, a = base.getHsv()
        if h < 0:
            h = 200

        def tick():
            try:
                if not self._enabled or not st.get("hovering"):
                    self._stop(btn)
                    return
                st["phase"] = (st.get("phase", 0.0) + 0.018) % 1.0
                hh = int((h + st["phase"] * 360.0) % 360)
                col = QColor()
                col.setHsv(hh, max(40, s), max(90, v), 255)

                # append override so we don't destroy existing theming
                orig = st.get("orig_ss", "") or ""
                btn.setStyleSheet(orig + f"; background-color: {col.name()};")
            except Exception:
                pass

        t = QTimer(btn)
        t.setInterval(30)
        t.timeout.connect(tick)
        t.start()
        st["timer"] = t

    def _start_boomerang(self, btn: QPushButton):
        st = self._states.get(btn)
        if not st:
            return
        c_btn, c_hi = self._base_colors(btn)

        c1 = QColor(c_btn).lighter(105)
        c2 = QColor(c_hi).lighter(115)

        def tick():
            try:
                if not self._enabled or not st.get("hovering"):
                    self._stop(btn)
                    return

                ph = float(st.get("phase", 0.0))
                d = float(st.get("dir", 1.0))
                ph += 0.03 * d
                if ph >= 1.0:
                    ph = 1.0
                    d = -1.0
                elif ph <= 0.0:
                    ph = 0.0
                    d = 1.0
                st["phase"] = ph
                st["dir"] = d

                s = 0.15 + 0.70 * ph
                orig = st.get("orig_ss", "") or ""
                # Keep it simple: 3-stop gradient that "moves" by sliding the center stop
                qss = (
                    f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                    f"stop:0 {c1.name()}, stop:{s:.3f} {c2.name()}, stop:1 {c1.name()});"
                )
                btn.setStyleSheet(orig + "; " + qss)
            except Exception:
                pass

        t = QTimer(btn)
        t.setInterval(25)
        t.timeout.connect(tick)
        t.start()
        st["timer"] = t
