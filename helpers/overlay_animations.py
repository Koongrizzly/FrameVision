# helpers/overlay_animations.py
from __future__ import annotations
import random, time
from typing import Tuple, Optional
from PySide6.QtCore import Qt, QTimer, QRect, QSize, QPoint, QEvent
from PySide6.QtGui import QColor, QPainter, QFont, QPen, QBrush
from PySide6.QtWidgets import QWidget

__all__ = [
    "MatrixRainOverlay",
    "BokehOverlay",
    "attach_random_intro_overlay",
    "start_overlay",
    "stop_overlay",
    "pick_overlay_for_theme",
    "apply_intro_overlay_from_settings",
]

# -------- Base overlay --------------------------------------------------------

class _BaseOverlay(QWidget):
    def __init__(self, target: QWidget, fps: int = 30, force_topmost: bool = False):
        parent = target if not force_topmost else None
        super().__init__(parent)
        self._target = target
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)
        self._timer = QTimer(self)
        self._force_topmost = bool(force_topmost)
        self._timer.setInterval(max(1, int(1000 / max(1, fps))))
        self._timer.timeout.connect(self._tick)
        self._running = False
        # Ensure we always cover our target
        self._sync_geometry()
        target.installEventFilter(self)
        if self._force_topmost:
            # Convert to a top-level tool window that sits above the parent window
            self.setParent(None)
            self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
            self._anchor = target.window()
            if self._anchor:
                self._anchor.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.Resize, QEvent.Show, QEvent.Hide, QEvent.Move):
            self._sync_geometry()
        return super().eventFilter(obj, ev)

    def _sync_geometry(self):
        if self._force_topmost:
            p = getattr(self, "_anchor", None)
            if not p: return
            try:
                top_left = p.mapToGlobal(QPoint(0, 0))
                self.setGeometry(QRect(top_left, QSize(p.width(), p.height())))
            except Exception:
                pass
            self.raise_()
        else:
            p = self._target
            if not p: return
            self.setGeometry(QRect(0, 0, p.width(), p.height()))
            self.raise_()

    def start(self):
        self._timer.start()
        self.show()
        self.raise_()

    def stop(self):
        self._timer.stop()
        self.hide()
        try: self.deleteLater()
        except Exception: pass

    def _tick(self):
        self.update()

# -------- Matrix Rain ---------------------------------------------------------

class MatrixRainOverlay(_BaseOverlay):
    CHARS = "アイウエオカキクケコサシスセソ0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def __init__(self, target: QWidget, style: str = "green", density: float = 0.9,
                 speed_px: float = 220.0, trail: float = 0.85, font_point_size: int = 14, fps: int = 30, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        self.style = style.lower().strip()
        self.density = max(0.1, min(1.0, density))
        self.speed = max(40.0, speed_px)
        self.trail = max(0.0, min(0.98, trail))
        self.font = QFont("Consolas")
        self.font.setPointSize(font_point_size)
        self._col_x = []
        self._ypos = []
        self._yspeed = []

    def _colors(self) -> Tuple[QColor, QColor]:
        if self.style == "blue":
            head = QColor(180, 220, 255)
            body = QColor(60, 150, 255)
        else:
            head = QColor(220, 255, 200)
            body = QColor(80, 255, 120)
        return head, body

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.setFont(self.font)
        metrics = p.fontMetrics()
        ch = max(12, metrics.height())
        cw = max(8, metrics.averageCharWidth())
        w, h = self.width(), self.height()
        if not self._col_x:
            cols = max(1, int(w / cw))
            self._col_x = [int(i * cw) for i in range(cols)]
            self._ypos = [-random.uniform(0, h) for _ in range(cols)]
            self._yspeed = [self.speed * random.uniform(0.7, 1.3) for _ in range(cols)]
        head, body = self._colors()
        # Trail fade
        p.fillRect(0, 0, w, h, QBrush(QColor(0, 0, 0, int(255 * (1.0 - self.trail)))))
        for i, x in enumerate(self._col_x):
            if random.random() > self.density:
                continue
            y = self._ypos[i]
            chain = random.randint(6, 14)
            yy = y
            for j in range(chain):
                c = random.choice(self.CHARS)
                if j == 0:
                    p.setPen(head)
                else:
                    a = max(40, 220 - j * 16)
                    col = QColor(body.red(), body.green(), body.blue(), max(60, a))
                    p.setPen(col)
                p.drawText(QPoint(x, int(yy)), c)
                yy -= ch
            # advance
            self._ypos[i] += self._yspeed[i] / 30.0
            if self._ypos[i] - ch > h + 20:
                self._ypos[i] = -random.uniform(0, h * 0.75)
        p.end()

# -------- Scanlines -----------------------------------------------------------


class BokehOverlay(_BaseOverlay):
    def __init__(self, target: QWidget, count: int = 28, fps: int = 30, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        self._dots = []
        import random as _r
        for _ in range(count):
            self._dots.append({
                "x": _r.random(),
                "y": _r.random(),
                "r": _r.uniform(14, 38),
                "dx": _r.uniform(-0.12, 0.12),
                "dy": _r.uniform(-0.06, 0.06),
                "a": _r.randint(70, 130),
            })

    def _tick(self):
        for d in self._dots:
            d["x"] += d["dx"] / 60.0
            d["y"] += d["dy"] / 60.0
            if d["x"] < -0.2: d["x"] = 1.2
            if d["x"] > 1.2: d["x"] = -0.2
            if d["y"] < -0.2: d["y"] = 1.2
            if d["y"] > 1.2: d["y"] = -0.2
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        w, h = self.width(), self.height()
        for d in self._dots:
            p.setBrush(QBrush(QColor(255, 255, 255, int(d["a"]))))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPoint(int(d["x"] * w), int(d["y"] * h)), int(d["r"]), int(d["r"]))
        p.end()

# -------- Film Grain ----------------------------------------------------------


# -------- Helpers -------------------------------------------------------------

def start_overlay(name: str, target_widget: QWidget, *, force_topmost: bool = True):
    n = (name or "").lower().strip()
    if n in ("matrix", "matrix_green", "matrix (green)"):
        o = MatrixRainOverlay(target_widget, style="green", force_topmost=force_topmost)
    elif n in ("matrix_blue", "matrix (blue)"):
        o = MatrixRainOverlay(target_widget, style="blue", force_topmost=force_topmost)
    elif n in ("bokeh",):
        o = BokehOverlay(target_widget, force_topmost=force_topmost)
    else:
        o = MatrixRainOverlay(target_widget, style="green", force_topmost=force_topmost)
    o.start()
    # keep a reference on widget to stop later
    if not hasattr(target_widget, "_fv_overlays"):
        target_widget._fv_overlays = []
    target_widget._fv_overlays.append(o)
    return o

def stop_overlay(target_widget: QWidget):
    try:
        for o in getattr(target_widget, "_fv_overlays", []):
            o.stop()
    except Exception:
        pass
    finally:
        try:
            target_widget._fv_overlays = []
        except Exception:
            pass

def attach_random_intro_overlay(target_widget: QWidget, choices=("matrix_green","matrix_blue","scanlines","bokeh","grain"), duration_ms: int = 0, *, force_topmost: bool = True):
    name = random.choice(list(choices) or ["matrix_green"])
    o = start_overlay(name, target_widget, force_topmost=force_topmost)
    if duration_ms and duration_ms > 0:
        t = QTimer(target_widget); t.setSingleShot(True); t.setInterval(int(duration_ms))
        def _stop():
            stop_overlay(target_widget)
            try: t.deleteLater()
            except Exception: pass
        t.timeout.connect(_stop); t.start()
    return o

def pick_overlay_for_theme(theme_name: str) -> str:
    """Map a theme to a default overlay name."""
    t = (theme_name or "").lower()
    if t.startswith("day") or "solar" in t:
        return "bokeh"
    if t.startswith("even") or "evening" in t:
        return "bokeh"
    if t.startswith("night") or "dark" in t or "slate" in t:
        return "matrix_green"
    # fun defaults
    if "cyber" in t or "neon" in t:
        return "matrix_blue"
    return "matrix_green"


# Convenience: start overlay based on QSettings + theme
def _norm_mode_name(txt: str) -> str:
    t = (txt or "").strip().lower()
    if t in ("random", "", "auto"): return "random"
    t = t.replace("(", "").replace(")", "").replace(" ", "_")
    if t.startswith("matrix_blue"): return "matrix_blue"
    if t.startswith("matrix_green") or t == "matrix": return "matrix_green"
    if t.startswith("scanline"): return "bokeh"
    if t.startswith("film_grain") or t.startswith("grain"): return "matrix_green"
    return t

def apply_intro_overlay_from_settings(target_widget: QWidget, theme_name: str | None = None, *, force_topmost: bool = True):
    """Reads QSettings for intro overlay and starts/stops accordingly.
    Keys:
        intro_overlay_enabled (bool)
        intro_overlay_mode (str) e.g., "Random", "Matrix (Blue)", "Film grain"
        intro_follow_theme (bool) — when True and mode is Random, choose by theme.
    """
    try:
        from PySide6.QtCore import QSettings
        s = QSettings("FrameVision","FrameVision")
        enabled = s.value("intro_overlay_enabled", False, type=bool)
        if not enabled:
            stop_overlay(target_widget)
            return None
        mode = s.value("intro_overlay_mode", "Random", type=str) or "Random"
        follow_theme = s.value("intro_follow_theme", False, type=bool)
        name = _norm_mode_name(mode)
        if name == "random":
            if follow_theme and (theme_name or ""):
                try:
                    pick = pick_overlay_for_theme(theme_name or "")
                    name = pick or "matrix_green"
                except Exception:
                    name = "matrix_green"
            else:
                name = random.choice(["matrix_green","matrix_blue","bokeh"])
        return start_overlay(name, target_widget, force_topmost=force_topmost)
    except Exception:
        return None
