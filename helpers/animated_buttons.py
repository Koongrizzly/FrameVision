from __future__ import annotations
import math
import random
from typing import Optional, Dict, Any

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSettings, QTimer, QEvent


def _rgba(c: QtGui.QColor, a: Optional[int] = None) -> str:
    try:
        r, g, b, aa = c.getRgb()
        if a is None:
            a = aa
        a = max(0, min(255, int(a)))
        return f"rgba({int(r)},{int(g)},{int(b)},{a})"
    except Exception:
        return "rgba(80,80,80,255)"


def _blend(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    """Linear blend between colors a->b (t in [0..1])."""
    try:
        t = max(0.0, min(1.0, float(t)))
        ar, ag, ab, aa = a.getRgb()
        br, bg, bb, ba = b.getRgb()
        r = int(ar + (br - ar) * t)
        g = int(ag + (bg - ag) * t)
        b2 = int(ab + (bb - ab) * t)
        a2 = int(aa + (ba - aa) * t)
        return QtGui.QColor(r, g, b2, a2)
    except Exception:
        return b



def _clamp(v: float, lo: float, hi: float) -> float:
    try:
        v = float(v)
    except Exception:
        v = lo
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _bright_ui_for_btn(btn: QtWidgets.QPushButton, fallback_base: Optional[QtGui.QColor] = None) -> bool:
    """Best-effort 'is this a bright/day theme?' detector.
    We prefer the application's Window color over the button's base color because
    bright themes may still render dark buttons (e.g. accent buttons).
    """
    try:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            pal = app.palette()
            wl = int(pal.color(QtGui.QPalette.Window).lightness())
            if wl >= 150:
                return True
    except Exception:
        pass
    try:
        if fallback_base is not None:
            return int(fallback_base.lightness()) >= 150
    except Exception:
        pass
    return False



def _rel_lum(c: QtGui.QColor) -> float:
    # Relative luminance (WCAG-ish), sRGB -> linear
    try:
        r, g, b, _a = c.getRgb()
        def _lin(u: int) -> float:
            x = u / 255.0
            return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
        rl, gl, bl = _lin(r), _lin(g), _lin(b)
        return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl
    except Exception:
        return 0.0


def _contrast_ratio(a: QtGui.QColor, b: QtGui.QColor) -> float:
    try:
        la = _rel_lum(a)
        lb = _rel_lum(b)
        hi = max(la, lb)
        lo = min(la, lb)
        return (hi + 0.05) / (lo + 0.05)
    except Exception:
        return 1.0


def _pick_text_color(bg: QtGui.QColor, preferred: QtGui.QColor) -> QtGui.QColor:
    """Ensure readable text on top of bg.
    Keep preferred if it has decent contrast; otherwise choose black/white with best contrast.
    """
    try:
        if _contrast_ratio(bg, preferred) >= 3.5:
            return preferred
        black = QtGui.QColor(0, 0, 0)
        white = QtGui.QColor(255, 255, 255)
        return white if _contrast_ratio(bg, white) >= _contrast_ratio(bg, black) else black
    except Exception:
        return preferred


def _min_lightness(*colors: QtGui.QColor) -> int:
    vals = []
    for c in colors:
        try:
            vals.append(int(c.lightness()))
        except Exception:
            pass
    return min(vals) if vals else 128


def _max_lightness(*colors: QtGui.QColor) -> int:
    vals = []
    for c in colors:
        try:
            vals.append(int(c.lightness()))
        except Exception:
            pass
    return max(vals) if vals else 128


def _auto_fg_for_gradient(preferred: QtGui.QColor, *bg_colors: QtGui.QColor) -> QtGui.QColor:
    """Pick black/white in a way that stays readable across a gradient/animated fill.

    This is intentionally *more conservative* than _pick_text_color():
    if any significant part of the gradient is dark, we prefer white text (even on day themes).
    """
    try:
        min_l = _min_lightness(*bg_colors)
        max_l = _max_lightness(*bg_colors)
    except Exception:
        return preferred

    try:
        pref_l = int(preferred.lightness())
    except Exception:
        pref_l = 128

    # Day themes typically use dark text as preferred.
    if pref_l < 128:
        return QtGui.QColor(255, 255, 255) if min_l < 175 else preferred

    # Dark themes typically use light text as preferred.
    return QtGui.QColor(0, 0, 0) if max_l > 215 else preferred



def _boost_for_bright_ui(base: QtGui.QColor, accent: QtGui.QColor, accent2: QtGui.QColor, ui_bright: bool) -> tuple[QtGui.QColor, QtGui.QColor, QtGui.QColor]:
    """Subtle automatic boost for bright/day themes.
    Purpose: avoid dark/navy-looking hover fills on light UIs, without needing a slider.
    """
    if not ui_bright:
        return base, accent, accent2
    try:
        bl = int(base.lightness())
    except Exception:
        bl = 140

    # Accents: slightly brighter so the effect reads on light UIs.
    try:
        accent_b = QtGui.QColor(accent).lighter(150)
    except Exception:
        accent_b = accent
    try:
        accent2_b = QtGui.QColor(accent2).lighter(160)
    except Exception:
        accent2_b = accent2

    # Base: lift only a bit (more lift if the button itself is dark).
    try:
        t = 0.26 if bl < 160 else 0.14
        base_b = _blend(base, QtGui.QColor(255, 255, 255), t)
    except Exception:
        base_b = base

    return base_b, accent_b, accent2_b


class AnimatedButtonsManager(QtCore.QObject):
    """
    Hover animation manager for important buttons (e.g., "Generate", "View results").
    - Enabled/disabled + mode are stored in QSettings under:
        animated_buttons_enabled (bool)
        animated_buttons_mode (str): glow | shift | boomerang | outline | scanline | shimmer | pop | random
        animated_buttons_random_pick (legacy): unused (random now chooses per-hover)
    """
    _APP_PROP = "_fv_animated_buttons_manager"
    _HOVER_MARK = "/* fv_animbtn_hover */"
    _KEYWORDS = ("view results", "queue", "restart app", "restart app", "analyze", "clear program cache", "apply","show qsettings in cmd")
    _RANDOM_CHOICES = ("glow", "shift", "boomerang", "outline", "scanline", "shimmer", "pop")

    def __init__(self, app: QtWidgets.QApplication):
        super().__init__(app)
        self.app = app
        self.s = QSettings("FrameVision", "FrameVision")

        self.enabled: bool = False
        self.mode: str = "glow"          # requested mode (can be "random")
        # Effective mode is resolved per-hover (random chooses a new mode on each hover).

        self._active: Dict[QtWidgets.QPushButton, Dict[str, Any]] = {}

        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._tick)

        # We can catch Enter/Leave globally via app eventFilter.
        try:
            self.app.installEventFilter(self)
        except Exception:
            pass

        # Apply current saved settings.
        self._apply_from_settings_internal()

    # ---- public static API -----------------------------------------------------
    @staticmethod
    def install(app: Optional[QtWidgets.QApplication] = None) -> Optional["AnimatedButtonsManager"]:
        try:
            app = app or QtWidgets.QApplication.instance()
            if app is None:
                return None
            mgr = getattr(app, AnimatedButtonsManager._APP_PROP, None)
            if isinstance(mgr, AnimatedButtonsManager):
                return mgr
            mgr = AnimatedButtonsManager(app)
            setattr(app, AnimatedButtonsManager._APP_PROP, mgr)
            return mgr
        except Exception:
            return None

    @staticmethod
    def apply_from_settings(app: Optional[QtWidgets.QApplication] = None) -> None:
        mgr = AnimatedButtonsManager.install(app)
        if mgr is not None:
            try:
                mgr._apply_from_settings_internal()
            except Exception:
                pass

    # ---- internal helpers ------------------------------------------------------
    def _is_target(self, btn: QtWidgets.QPushButton) -> bool:
        try:
            t = (btn.text() or "").lower()
            if not t:
                return False
            return any(k in t for k in self._KEYWORDS)
        except Exception:
            return False

    def _effective_mode_for_hover(self) -> str:
        """Return effective mode for a hover instance.
        - If disabled -> glow (unused because we won't start anyway)
        - If mode is glow/shift/boomerang -> that mode
        - If mode is random -> choose a mode on EVERY hover
        """
        mode = (self.mode or "glow").strip().lower()
        if mode not in ("glow", "shift", "boomerang", "outline", "scanline", "shimmer", "pop", "random"):
            mode = "glow"
        if not self.enabled:
            return "glow"
        if mode == "random":
            return random.choice(self._RANDOM_CHOICES)
        return mode

    def _apply_from_settings_internal(self) -> None:
        try:
            enabled = bool(self.s.value("animated_buttons_enabled", False, type=bool))
        except Exception:
            enabled = False
        try:
            mode = (self.s.value("animated_buttons_mode", "glow", type=str) or "glow").strip().lower()
        except Exception:
            mode = "glow"

        prev_enabled = bool(getattr(self, "enabled", False))
        prev_mode = str(getattr(self, "mode", "glow") or "glow")

        self.enabled = bool(enabled)
        self.mode = mode

        # If settings changed while a button is hovered, reset active animations so the new mode applies cleanly.
        if (prev_enabled != self.enabled) or (prev_mode.strip().lower() != str(self.mode).strip().lower()):
            try:
                self._stop_all()
            except Exception:
                pass

        if not self.enabled:
            self._stop_all()
        # If enabled, nothing else needed until hover events occur.

    def _ensure_orig_style(self, btn: QtWidgets.QPushButton) -> str:
        """Return the button's current *base* stylesheet (without our hover marker).
        This is refreshed on every hover start so theme/font-size changes are preserved.
        """
        try:
            ss = btn.styleSheet() or ""
            mark = getattr(self, "_HOVER_MARK", "/* fv_animbtn_hover */")
            if mark and mark in ss:
                ss = ss.split(mark, 1)[0].rstrip()
            btn.setProperty("fv_animbtn_orig_style", ss)
            return str(ss or "")
        except Exception:
            return ""

    def _apply_hover_qss(self, btn: QtWidgets.QPushButton, st: Dict[str, Any], qss: str) -> None:
        """Apply hover-only visuals without nuking the button's existing stylesheet.
        We append a marked block so we can strip it back out later.
        """
        try:
            base = str(st.get("orig_style") or "")
            mark = getattr(self, "_HOVER_MARK", "/* fv_animbtn_hover */")
            # Defensive: if something already appended our mark, strip it first.
            if mark and mark in base:
                base = base.split(mark, 1)[0].rstrip()
            combined = base
            if qss:
                combined = (base + "\n" + mark + "\n" + qss) if base else (mark + "\n" + qss)
            if btn.styleSheet() != combined:
                btn.setStyleSheet(combined)
        except Exception:
            try:
                btn.setStyleSheet(qss or "")
            except Exception:
                pass

    def _start(self, btn: QtWidgets.QPushButton) -> None:
        if btn in self._active:
            return
        if not self._is_target(btn):
            return

        orig_style = self._ensure_orig_style(btn)
        st = {
            "phase": 0.0,
            "dir": 1.0,
            "orig_style": orig_style,
            "added_effect": False,
            "mode": self._effective_mode_for_hover(),
        }
        self._active[btn] = st

        # Some modes use a graphics effect; others are stylesheet-only.
        if st.get("mode") in ("glow", "pop", "outline"):
            try:
                if btn.graphicsEffect() is None:
                    eff = QtWidgets.QGraphicsDropShadowEffect(btn)
                    eff.setOffset(0, 0)
                    btn.setGraphicsEffect(eff)
                    st["added_effect"] = True
            except Exception:
                pass

        if not self._timer.isActive():
            self._timer.start()

    def _restore_btn(self, btn: QtWidgets.QPushButton, st: Dict[str, Any]) -> None:
        try:
            base = str(st.get("orig_style") or "")
            mark = getattr(self, "_HOVER_MARK", "/* fv_animbtn_hover */")
            if mark and mark in base:
                base = base.split(mark, 1)[0].rstrip()
            btn.setStyleSheet(base)
            btn.setProperty("fv_animbtn_orig_style", base)
        except Exception:
            pass
        try:
            if bool(st.get("added_effect", False)):
                btn.setGraphicsEffect(None)
        except Exception:
            pass

    def _stop(self, btn: QtWidgets.QPushButton) -> None:
        st = self._active.pop(btn, None)
        if st is None:
            return
        self._restore_btn(btn, st)
        if not self._active and self._timer.isActive():
            self._timer.stop()

    def _stop_all(self) -> None:
        for btn, st in list(self._active.items()):
            try:
                self._restore_btn(btn, st)
            except Exception:
                pass
        self._active.clear()
        try:
            if self._timer.isActive():
                self._timer.stop()
        except Exception:
            pass

    # ---- animation tick --------------------------------------------------------
    def _tick(self) -> None:
        if not self.enabled:
            self._stop_all()
            return

        # Update all hovered buttons.
        for btn, st in list(self._active.items()):
            try:
                # if button got deleted, remove
                try:
                    import shiboken6
                    if not shiboken6.isValid(btn):
                        self._active.pop(btn, None)
                        continue
                except Exception:
                    pass

                if not self._is_target(btn):
                    self._stop(btn)
                    continue

                phase = float(st.get("phase", 0.0))
                direction = float(st.get("dir", 1.0))

                if st.get("mode", "glow") == "shift":
                    phase = (phase + 0.045) % 1.0
                elif st.get("mode", "glow") == "boomerang":
                    phase += direction * 0.06
                    if phase >= 1.0:
                        phase = 1.0
                        direction = -1.0
                    elif phase <= 0.0:
                        phase = 0.0
                        direction = 1.0
                else:
                    # glow / outline / shimmer / pop
                    phase = (phase + 0.05) % 1.0

                st["phase"] = phase
                st["dir"] = direction

                pal = btn.palette()
                base = pal.color(QtGui.QPalette.Button)
                accent = pal.color(QtGui.QPalette.Highlight)
                link = pal.color(QtGui.QPalette.Link)
                accent2 = link if link.isValid() else accent.lighter(125)
                ui_bright = _bright_ui_for_btn(btn, base)
                base_b, accent_b, accent2_b = _boost_for_bright_ui(base, accent, accent2, ui_bright)

                # Preferred text color from theme, but we'll auto-contrast during hover if needed.
                btn_text = pal.color(QtGui.QPalette.ButtonText)


                if st.get("mode", "glow") == "shift":
                    # Rainbow hue cycling across the button.
                    # Palette highlight/link are often theme-blue, so we rotate hue ourselves.
                    try:
                        bl = int(base.lightness())
                    except Exception:
                        bl = 128

                    # Tune saturation/value so it looks good on both dark and light themes.
                    if bl >= 180:
                        sat, val, base_keep = 170, 210, 0.25  # lighter UI: keep it softer
                    elif bl <= 80:
                        sat, val, base_keep = 230, 235, 0.12  # dark UI: allow stronger colors
                    else:
                        sat, val, base_keep = 200, 225, 0.18

                    h1 = int((phase * 360.0) % 360)
                    h2 = int(((phase + 0.33) * 360.0) % 360)
                    raw1 = QtGui.QColor.fromHsv(h1, sat, val)
                    raw2 = QtGui.QColor.fromHsv(h2, sat, val)

                    # Blend a little of the theme base color back in so it still feels "FrameVision".
                    c1 = _blend(base, raw1, 1.0 - base_keep)
                    c2 = _blend(base, raw2, 1.0 - base_keep)

                    grad = (
                        f"qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                        f" stop:0 {_rgba(c1, 255)},"
                        f" stop:1 {_rgba(c2, 255)});"
                    )
                    fg = _pick_text_color(_blend(c1, c2, 0.5), btn_text)
                    self._apply_hover_qss(btn, st, f"QPushButton{{ background: {grad} color: {_rgba(fg,255)}; }}")

                elif st.get("mode", "glow") == "boomerang":
                    # Moving highlight band across the button
                    t = phase
                    w = 0.22
                    a = max(0.0, t - w)
                    b = t
                    c = min(1.0, t + w)
                    bg = _blend(base_b, accent_b, 0.25)
                    hi = _blend(base_b, accent_b, 0.85)
                    grad = (
                        "qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                        f" stop:0 {_rgba(bg,255)},"
                        f" stop:{a:.3f} {_rgba(bg,255)},"
                        f" stop:{b:.3f} {_rgba(hi,255)},"
                        f" stop:{c:.3f} {_rgba(bg,255)},"
                        f" stop:1 {_rgba(bg,255)});"
                    )
                    fg = _auto_fg_for_gradient(btn_text, bg, hi)
                    self._apply_hover_qss(btn, st, f"QPushButton{{ background: {grad} color: {_rgba(fg,255)}; }}")


                elif st.get("mode", "glow") == "outline":
                    # Neon outline pulse with color-changing (HSV) outline.
                    # This stays "border-first" (not a full fill effect) while still feeling alive.
                    pulse = 0.35 + 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * phase))
                    alpha = int(90 + 140 * pulse)

                    # Hue cycle across the outline (good on both light/dark themes).
                    try:
                        bl = int(base.lightness())
                    except Exception:
                        bl = 128
                    if bl >= 180:
                        sat, val = 170, 215
                    elif bl <= 80:
                        sat, val = 235, 240
                    else:
                        sat, val = 205, 230

                    h = int((phase * 360.0) % 360)
                    raw = QtGui.QColor.fromHsv(h, sat, val)
                    # Blend a bit with the theme accent so it still feels consistent.
                    border_c = _blend(accent_b, raw, 0.75)

                    # Slightly tint the background so the outline doesn't feel detached.
                    bg = _blend(base, border_c, 0.10 + 0.06 * math.sin(2 * math.pi * phase))

                    fg = _pick_text_color(bg, btn_text)
                    self._apply_hover_qss(
                        btn,
                        st,
                        "QPushButton{"
                        f" background: {_rgba(bg,255)};"
                        f" color: {_rgba(fg,255)};"
                        "}"
                    )
                    # Outline glow via shadow effect (no border => no size hint jump).
                    try:
                        eff = btn.graphicsEffect()
                        if not isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
                            eff = QtWidgets.QGraphicsDropShadowEffect(btn)
                            eff.setOffset(0, 0)
                            btn.setGraphicsEffect(eff)
                            st["added_effect"] = True
                        blur = 10 + int(14 * pulse)
                        eff.setBlurRadius(float(blur))
                        eff.setColor(QtGui.QColor(border_c.red(), border_c.green(), border_c.blue(), alpha))
                        eff.setOffset(0, 0)
                    except Exception:
                        pass

                



                elif st.get("mode", "glow") == "scanline":
                    # Bright scanline band sweeping top-to-bottom (very readable on both light/dark themes).
                    t = phase
                    w = 0.18
                    a = max(0.0, t - w)
                    b = t
                    c = min(1.0, t + w)
                    bg = _blend(base_b, accent_b, 0.18)
                    hi = _blend(base_b, accent2_b, 0.92)
                    grad = (
                        "qlineargradient(x1:0,y1:0,x2:0,y2:1,"
                        f" stop:0 {_rgba(bg,255)},"
                        f" stop:{a:.3f} {_rgba(bg,255)},"
                        f" stop:{b:.3f} {_rgba(hi,255)},"
                        f" stop:{c:.3f} {_rgba(bg,255)},"
                        f" stop:1 {_rgba(bg,255)});"
                    )
                    fg = _auto_fg_for_gradient(btn_text, bg, hi)
                    self._apply_hover_qss(btn, st, f"QPushButton{{ background: {grad} color: {_rgba(fg,255)}; }}")

                elif st.get("mode", "glow") == "shimmer":
                    # Diagonal shimmer band sliding across the button.
                    t = phase
                    w = 0.14
                    a = max(0.0, t - w)
                    b = t
                    c = min(1.0, t + w)
                    bg = _blend(base_b, accent_b, 0.12)
                    hi = _blend(base_b, accent2_b, 0.95)
                    grad = (
                        "qlineargradient(x1:0,y1:0,x2:1,y2:1,"
                        f" stop:0 {_rgba(bg,255)},"
                        f" stop:{a:.3f} {_rgba(bg,255)},"
                        f" stop:{b:.3f} {_rgba(hi,255)},"
                        f" stop:{c:.3f} {_rgba(bg,255)},"
                        f" stop:1 {_rgba(bg,255)});"
                    )
                    fg = _auto_fg_for_gradient(btn_text, bg, hi)
                    self._apply_hover_qss(btn, st, f"QPushButton{{ background: {grad} color: {_rgba(fg,255)}; }}")

                elif st.get("mode", "glow") == "pop":
                    # Micro "pop" feeling via center highlight + pulsing shadow (no geometry changes).
                    lift = 0.35 + 0.35 * (0.5 + 0.5 * math.sin(2 * math.pi * phase))
                    center = _blend(base_b, accent2_b, 0.18 + 0.22 * lift)
                    edge = _blend(base_b, accent_b, 0.10 + 0.06 * lift)
                    grad = (
                        "qradialgradient(cx:0.5, cy:0.5, radius:0.9,"
                        f" stop:0 {_rgba(center,255)},"
                        f" stop:1 {_rgba(edge,255)});"
                    )
                    fg = _pick_text_color(center, btn_text)
                    self._apply_hover_qss(btn, st, f"QPushButton{{ background: {grad} color: {_rgba(fg,255)}; }}")
                    try:
                        eff = btn.graphicsEffect()
                        if isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
                            blur = 8 + int(14 * lift)
                            alpha = 70 + int(130 * lift)
                            eff.setBlurRadius(float(blur))
                            eff.setColor(QtGui.QColor(accent_b.red(), accent_b.green(), accent_b.blue(), alpha))
                            eff.setOffset(0, -1)
                    except Exception:
                        pass

                else:  # glow
                    # Subtle tint + pulsing drop shadow
                    tint = 0.35 + 0.15 * math.sin(2 * math.pi * phase)
                    c1 = _blend(base_b, accent_b, tint)
                    grad = (
                        f"qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                        f" stop:0 {_rgba(c1,255)}, stop:1 {_rgba(base,255)});"
                    )
                    fg = _pick_text_color(_blend(c1, base_b, 0.55), btn_text)
                    self._apply_hover_qss(btn, st, f"QPushButton{{ background: {grad} color: {_rgba(fg,255)}; }}")
                    try:
                        eff = btn.graphicsEffect()
                        if isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
                            blur = 10 + int(10 * (0.5 + 0.5 * math.sin(2 * math.pi * phase)))
                            alpha = 60 + int(120 * (0.5 + 0.5 * math.sin(2 * math.pi * phase)))
                            eff.setBlurRadius(float(blur))
                            eff.setColor(QtGui.QColor(accent_b.red(), accent_b.green(), accent_b.blue(), alpha))
                            eff.setOffset(0, 0)
                    except Exception:
                        pass

            except Exception:
                # Defensive: if anything goes wrong with a button, restore it and remove.
                try:
                    self._stop(btn)
                except Exception:
                    pass

        if not self._active and self._timer.isActive():
            self._timer.stop()

    # ---- event filter ----------------------------------------------------------
    def eventFilter(self, obj, ev):
        try:
            et = ev.type()
            if isinstance(obj, QtWidgets.QPushButton):
                if self.enabled and self._is_target(obj):
                    if et == QEvent.Enter:
                        self._start(obj)
                    elif et == QEvent.Leave:
                        self._stop(obj)
                    elif et == QEvent.Destroy:
                        self._active.pop(obj, None)
            return False
        except Exception:
            return False
