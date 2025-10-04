# helpers/overlay_animations.py
from __future__ import annotations
import random, time
from typing import Tuple, Optional
from PySide6.QtCore import Qt, QTimer, QRect, QSize, QPoint, QEvent
from PySide6.QtGui import QColor, QPainter, QFont, QPen, QBrush, QLinearGradient
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
                "h": _r.uniform(0.0, 360.0),
                "dh": _r.uniform(-24.0, 24.0),
                "s": _r.randint(200, 255),
                "v": _r.randint(220, 255),
            })

    def _tick(self):
        for d in self._dots:
            d["x"] += d["dx"] / 60.0
            d["y"] += d["dy"] / 60.0
            d["h"] = (d["h"] + d["dh"] / 60.0) % 360.0
            if d["x"] < -0.2: d["x"] = 1.2
            if d["x"] > 1.2: d["x"] = -0.2
            if d["y"] < -0.2: d["y"] = 1.2
            if d["y"] > 1.2: d["y"] = -0.2
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        for d in self._dots:
            col = QColor.fromHsv(int(d["h"]) % 360, int(d["s"]), int(d["v"]), int(d["a"]))
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPoint(int(d["x"] * w), int(d["y"] * h)), int(d["r"]), int(d["r"]))
        p.end()

# -------- Film Grain ----------------------------------------------------------


# -------- Helpers -------------------------------------------------------------

def start_overlay(name: str, target_widget: QWidget, *, force_topmost: bool = True):
    n = _norm_mode_name(name)
    if n in ("matrix_green",):
        o = MatrixRainOverlay(target_widget, style="green", force_topmost=force_topmost)
    elif n in ("matrix_blue",):
        o = MatrixRainOverlay(target_widget, style="blue", force_topmost=force_topmost)
    elif n in ("bokeh",):
        o = BokehOverlay(target_widget, force_topmost=force_topmost)
    elif n == "rain":
        o = RainOverlay(target_widget, force_topmost=force_topmost)
    elif n == "fireflies":
        o = FirefliesParallaxOverlay(target_widget, force_topmost=force_topmost)
    elif n == "starfield":
        o = StarfieldHyperjumpOverlay(target_widget, force_topmost=force_topmost)
    elif n == "comets":
        o = CometTrailsOverlay(target_widget, force_topmost=force_topmost)
    elif n == "aurora":
        o = AuroraFlowOverlay(target_widget, force_topmost=force_topmost)
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
    if t.startswith("rain"): return "rain"
    if t in ("fireflies","fireflies_parallax","firefliesparallax","firefly"): return "fireflies"
    if t.startswith("starfield") or t.startswith("hyperjump"): return "starfield"
    if t.startswith("comet"): return "comets"
    if t.startswith("aurora"): return "aurora"
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
                name = random.choice(["matrix_green","matrix_blue","bokeh","rain","fireflies","starfield","comets","aurora"])
        return start_overlay(name, target_widget, force_topmost=force_topmost)
    except Exception:
        return None


# -------- Rain
class RainOverlay(_BaseOverlay):
    """Rain hits only (no falling streaks).
    - Heavy startup burst to instantly 'wet' the glass
    - Expanding rings that quickly fade
    Tune: rate (hits/min), burst_duration (s), gr (radius growth), ga (fade speed)
    """
    def __init__(self, target: QWidget, rate: int = 220, fps: int = 60, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        self._rate = max(0, int(rate))
        self._splats = []                # each: {x,y,r,alpha,gr,ga}
        self._max_splats = 600
        self._dt = 1.0 / float(max(30, fps))

        # startup burst so it's busy immediately
        self._age = 0.0
        self._burst_duration = 0.60      # seconds
        self._spawn_bias_top = 0.90      # bias toward top half for realism

        import random as _r
        # Prewarm a dense field
        for _ in range(160):
            self._spawn_hit(_prewarm=True, rnd=_r)

    def _spawn_hit(self, _prewarm=False, rnd=None):
        import random as _r
        r = rnd or _r
        w = self.width() if self.width() > 0 else 1280
        h = self.height() if self.height() > 0 else 720
        x = r.uniform(0, w)
        y = r.uniform(0, h * (0.15 + 0.75 * self._spawn_bias_top))
        self._splats.append({
            "x": x, "y": y,
            "r": r.uniform(2.0, 6.0),
            "alpha": 0.65 if not _prewarm else 0.4,
            "gr": r.uniform(360.0, 720.0),  # radius growth px/s
            "ga": r.uniform(1.8, 2.6),      # alpha fade /s
        })
        if len(self._splats) > self._max_splats:
            self._splats = self._splats[-self._max_splats:]

    def _tick(self):
        import random as _r
        dt = self._dt
        self._age += dt

        # Poisson rate + heavier during the initial burst
        lam = (self._rate / 60.0) * dt
        if self._age < self._burst_duration:
            lam *= 6.0

        n_new = 0
        if lam > 0.0:
            checks = 1 + int(lam * 6)
            p = lam / max(1, checks)
            for _ in range(checks):
                if _r.random() < p:
                    n_new += 1
        for _ in range(n_new):
            self._spawn_hit()

        # Update splats
        for s in self._splats:
            s["r"] += s["gr"] * dt
            s["alpha"] -= s["ga"] * dt

        # keep only visible rings
        self._splats[:] = [s for s in self._splats if s["alpha"] > 0.02]
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # Draw only the expanding impact rings (no trails)
        for s in self._splats:
            a = int(max(0, min(255, s["alpha"] * 255)))
            if a <= 3:
                continue
            ring = QColor(220, 235, 255, a)
            pen = QPen(ring)
            pen.setWidth(1)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPoint(int(s["x"]), int(s["y"])), int(s["r"]), int(s["r"]))

            # optional soft highlight dot at center for a glassy pop
            if a > 90:
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(230, 245, 255, int(a*0.35))))
                p.drawEllipse(QPoint(int(s["x"]), int(s["y"])), 2, 2)

        p.end()
# -------- Fireflies Parallax
class FirefliesParallaxOverlay(_BaseOverlay):
    def __init__(self, target: QWidget, count: int = 60, layers: int = 3, fps: int = 30, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        import random as _r
        self._flies = []
        layers = max(1, int(layers)); count = int(max(8, min(80, count)))
        for _ in range(count):
            z = _r.randint(0, layers-1)
            spd = [90.0, 50.0, 24.0][z] if z < 3 else 24.0
            twk = [0.9, 0.6, 0.35][z] if z < 3 else 0.35
            self._flies.append({
                "x": _r.uniform(0, 1),
                "y": _r.uniform(0, 1),
                "z": z,
                "vx": _r.uniform(-spd, spd),
                "vy": _r.uniform(-spd*0.4, spd*0.4),
                "twinkle": twk,
                "phase": _r.uniform(0, 6.28318),
                "size": [6,4,2][z] if z < 3 else 2,
            })

    def _tick(self):
        dt = 1.0 / 60.0
        w, h = float(self.width()), float(self.height())
        for f in self._flies:
            f["x"] += (f["vx"] * dt) / max(1.0, w)
            f["y"] += (f["vy"] * dt) / max(1.0, h)
            f["phase"] += 3.0 * dt
            if f["x"] < -0.05: f["x"] = 1.05
            if f["x"] > 1.05: f["x"] = -0.05
            if f["y"] < -0.05: f["y"] = 1.05
            if f["y"] > 1.05: f["y"] = -0.05
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        for f in self._flies:
            a = 160 if f["z"] == 0 else 120 if f["z"] == 1 else 90
            import math as _m
            tw = 0.5 + 0.5 * (1.0 + _m.sin(f["phase"])) * f["twinkle"]
            a = int(min(255, a * (0.7 + 0.3 * tw)))
            col = QColor(255, 255, 210, a)
            p.setBrush(QBrush(col)); p.setPen(Qt.NoPen)
            p.drawEllipse(QPoint(int(f["x"] * w), int(f["y"] * h)), f["size"], f["size"])
        p.end()

# -------- Starfield Hyperjump
class StarfieldHyperjumpOverlay(_BaseOverlay):
    def __init__(self, target: QWidget, count: int = 100, fps: int = 30, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        import random as _r
        self._stars = []
        self._cx = 0.5; self._cy = 0.5
        count = int(max(40, min(180, count)))
        for _ in range(count):
            self._stars.append(self._spawn(_r))

    def _spawn(self, r):
        return {"x": r.uniform(0.0, 1.0), "y": r.uniform(0.0, 1.0), "v": r.uniform(0.6, 1.8), "len": r.uniform(0.01, 0.06), "a": r.randint(120, 220)}

    def _tick(self):
        dt = 1.0 / 60.0
        cx, cy = self._cx, self._cy
        for s in self._stars:
            dx = (s["x"] - cx); dy = (s["y"] - cy)
            s["x"] += dx * s["v"] * dt * 1.8
            s["y"] += dy * s["v"] * dt * 1.8
            if s["x"] < -0.1 or s["x"] > 1.1 or s["y"] < -0.1 or s["y"] > 1.1:
                import random as _r
                ns = self._spawn(_r); s.update(ns)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        for s in self._stars:
            a = s["a"]
            pen = QPen(QColor(255, 255, 255, a))
            pen.setWidth(1)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            x = int(s["x"] * w); y = int(s["y"] * h)
            lx = int((s["x"] - (s["x"] - self._cx) * s["len"]) * w)
            ly = int((s["y"] - (s["y"] - self._cy) * s["len"]) * h)
            p.drawLine(lx, ly, x, y)
        p.end()


class SmokeWispOverlay(_BaseOverlay):
    """Realistic-looking smoke made from many soft puffs.
    - Particles spawn from the lower/left area and drift diagonally
    - Curl-ish motion via simple time-varying noise
    - Each puff grows and fades with a soft radial falloff
    Tune: emit_rate, base_speed, ttl range
    """
    def __init__(self, target: QWidget, emit_rate: float = 42.0, fps: int = 60, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        self._emit_rate = float(emit_rate)     # puffs per second
        self._puffs = []                       # list of dicts
        self._dt = 1.0 / float(max(30, fps))
        self._time = 0.0
        self._max_puffs = 420
        self._base_speed = 0.10                # normalized screen units per second
        self._source_rect = (0.05, 0.55, 0.25, 0.35)  # x,y,w,h in normalized coords

        # Prewarm so the first frame already looks smokey
        import random as _r
        for _ in range(180):
            self._spawn(_r, prewarm=True)

    def _spawn(self, rnd, prewarm=False):
        x0, y0, w, h = self._source_rect
        x = rnd.uniform(x0, x0 + w)
        y = rnd.uniform(y0, y0 + h)
        ttl = rnd.uniform(2.8, 4.6)
        r0 = rnd.uniform(8.0, 16.0) if not prewarm else rnd.uniform(20.0, 40.0)
        grow = rnd.uniform(28.0, 56.0)  # px/sec
        a0 = rnd.uniform(0.18, 0.30) if not prewarm else rnd.uniform(0.10, 0.22)
        aspr = rnd.uniform(0.8, 1.3)    # aspect ratio multiplier (x/y)
        ang = rnd.uniform(-10.0, 10.0)  # slight orientation

        # initial drift up-right
        vx = self._base_speed * rnd.uniform(0.6, 1.2)
        vy = -self._base_speed * rnd.uniform(0.4, 1.1)

        self._puffs.append({
            "x": x, "y": y, "r": r0, "grow": grow,
            "vx": vx, "vy": vy, "life": 0.0, "ttl": ttl,
            "a0": a0, "aspr": aspr, "ang": ang, "seed": rnd.uniform(0.0, 100.0)
        })
        if len(self._puffs) > self._max_puffs:
            self._puffs = self._puffs[-self._max_puffs:]

    def _tick(self):
        import random as _r, math as _m
        dt = self._dt
        self._time += dt

        # emission (Poisson-ish)
        lam = self._emit_rate * dt
        n_new = 0
        if lam > 0.0:
            checks = 1 + int(lam * 4)
            p = lam / max(1, checks)
            for _ in range(checks):
                if _r.random() < p:
                    n_new += 1
        for _ in range(n_new):
            self._spawn(_r)

        # update puffs with gentle curl-like noise
        for puf in self._puffs:
            puf["life"] += dt
            # normalized vel with noise
            t = self._time + puf["seed"] * 0.11
            # pseudo curl-ish motion field
            nx = _m.sin(5.1 * (puf["y"] + t * 0.13)) * 0.06 + _m.cos(3.3 * (puf["x"] - t * 0.17)) * 0.03
            ny = _m.cos(4.7 * (puf["x"] + t * 0.09)) * 0.05 - 0.06  # upward bias
            puf["x"] += (puf["vx"] + nx) * dt
            puf["y"] += (puf["vy"] + ny) * dt
            # grow slowly over life
            puf["r"] += puf["grow"] * dt * (0.4 + 0.6 * min(1.0, puf["life"] / (puf["ttl"] * 0.7)))

        # cull
        w, h = self.width(), self.height()
        self._puffs[:] = [
            p for p in self._puffs
            if p["life"] < p["ttl"] and -0.2 <= p["x"] <= 1.2 and -0.2 <= p["y"] <= 1.2
        ]

        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        w, h = float(self.width()), float(self.height())

        # draw far puffs first (approximate painter's depth by radius)
        for puff in sorted(self._puffs, key=lambda pp: pp["r"], reverse=False):
            # opacity over life: ramp in, hold, ramp out
            t = puff["life"] / puff["ttl"]
            if t <= 0.2:
                a = puff["a0"] * (t / 0.2)
            elif t >= 0.85:
                a = puff["a0"] * max(0.0, (1.0 - t) / 0.15)
            else:
                a = puff["a0"]

            if a <= 0.004:
                continue

            cx = puff["x"] * w
            cy = puff["y"] * h
            rx = puff["r"]
            ry = puff["r"] / max(0.1, puff["aspr"])  # slight ellipse

            # layered soft falloff (fake radial gradient)
            # 4 layers from dense core to wide halo
            core = int(a * 255)
            levels = [
                (1.00, core),
                (1.70, int(core * 0.45)),
                (2.40, int(core * 0.22)),
                (3.20, int(core * 0.10)),
            ]
            p.save()
            # small rotation adds organic look
            # (keep angles subtle to avoid cloud-like blobs)
            p.translate(cx, cy)
            p.rotate(puff["ang"])
            p.translate(-cx, -cy)
            for mul, aval in levels:
                if aval <= 2: 
                    continue
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor(220, 230, 240, aval)))
                p.drawEllipse(QPoint(int(cx), int(cy)), int(rx * mul), int(ry * mul))
            p.restore()

        p.end()
# -------- Comet Trails

# -------- Aurora Flow
class AuroraFlowOverlay(_BaseOverlay):
    """High-visibility aurora ribbons with a quick sweep flash in the first second."""
    def __init__(self, target: QWidget, bands: int = 3, fps: int = 60, speed: float = 0.6, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        import random as _r
        self._t = 0.0
        self._speed = float(speed)
        self._bands = []
        bands = max(1, min(6, int(bands)))
        for i in range(bands):
            self._bands.append({
                "base": 0.25 + 0.25 * i / max(1, bands-1),  # normalized vertical base
                "amp": 0.10 + 0.05 * (bands - i) / bands,   # vertical amplitude
                "thick": 0.035 + 0.015 * (bands - i) / bands,  # thickness relative to min(w,h)
                "ph1": _r.uniform(0.0, 6.28318),
                "ph2": _r.uniform(0.0, 6.28318),
                "f1": _r.uniform(0.6, 1.2),
                "f2": _r.uniform(1.3, 2.4),
                # color A->B per band (neon-ish)
                "colA": (60 + 40*i, 140 + 20*i, 255),
                "colB": (120 + 10*i, 80 + 20*i, 255),
            })
        # Strong opening sweep
        self._sweep_age = 0.0
        self._sweep_dur = 0.9  # seconds

    def _tick(self):
        dt = 1.0 / 60.0
        self._t += dt * self._speed
        self._sweep_age += dt
        self.update()

    def _y_for(self, band, u, t):
        import math as _m
        # Two sine layers create organic ribbon curves
        return (band["base"]
                + band["amp"] * _m.sin(2.0*_m.pi*(band["f1"]*u + band["ph1"] + 0.20*t))
                + 0.6*band["amp"] * _m.sin(2.0*_m.pi*(band["f2"]*(u+0.15) + band["ph2"] - 0.34*t)))

    def _lerp(self, a, b, k): return a + (b - a) * k

    def paintEvent(self, ev):
        import math as _m
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width()); h = float(self.height())
        s = min(w, h)

        # Use screen composition for glowy layering
        p.setCompositionMode(QPainter.CompositionMode_Screen)

        # Draw ribbons back-to-front
        for idx, band in enumerate(self._bands):
            # build a path as a thick polyline
            steps = 48
            pen = QPen()
            pen.setWidthF(max(2.0, band["thick"] * s))
            pen.setCapStyle(Qt.RoundCap)

            # gradient along the ribbon
            r,g,b = band["colA"]; r2,g2,b2 = band["colB"]
            grad = QLinearGradient(0, 0, self.width(), 0)
            grad.setColorAt(0.0, QColor(r, g, b, 160))
            grad.setColorAt(1.0, QColor(r2, g2, b2, 160))
            pen.setBrush(QBrush(grad))

            p.setPen(pen)
            lastx = None; lasty = None
            for i in range(steps+1):
                u = i/steps
                y = self._y_for(band, u, self._t)
                x = u
                # subtle vertical squeeze toward center for readability
                y = 0.5 + (y - 0.5) * 0.85
                px = int(x * w); py = int(y * h)
                if lastx is not None:
                    p.drawLine(lastx, lasty, px, py)
                lastx, lasty = px, py

        # Opening sweep flash: wide soft bar moving across once
        if self._sweep_age < self._sweep_dur:
            k = self._sweep_age / self._sweep_dur  # 0..1
            cx = int((-0.3 + 1.6*k) * w)  # traverse left->right
            sweep = QLinearGradient(cx-80, 0, cx+80, 0)
            a = int(70 * (1.0 - k*k))  # ease-out alpha
            sweep.setColorAt(0.0, QColor(255, 255, 255, 0))
            sweep.setColorAt(0.5, QColor(255, 255, 255, a))
            sweep.setColorAt(1.0, QColor(255, 255, 255, 0))
            p.fillRect(0, 0, int(w), int(h), QBrush(sweep))

        p.end()


class CometTrailsOverlay(_BaseOverlay):
    def __init__(self, target: QWidget, count: int = 4, fps: int = 30, force_topmost: bool = False):
        super().__init__(target, fps=fps, force_topmost=force_topmost)
        import random as _r
        self._comets = []
        count = int(max(1, min(6, count)))
        for _ in range(count):
            self._comets.append(self._spawn(_r))

    def _spawn(self, r):
        ang = r.uniform(-0.7, 0.7)
        spd = r.uniform(320.0, 520.0)
        x0 = -0.1 if r.random() < 0.5 else 1.1
        y0 = r.uniform(0.1, 0.9)
        return {"x": x0, "y": y0, "vx": spd * (1.0 if x0 < 0 else -1.0), "vy": spd * ang, "len": r.uniform(40.0, 120.0)}

    def _tick(self):
        dt = 1.0 / 60.0
        w, h = float(self.width()), float(self.height())
        for c in self._comets:
            c["x"] += (c["vx"] * dt) / max(1.0, w)
            c["y"] += (c["vy"] * dt) / max(1.0, h)
        for i, c in enumerate(list(self._comets)):
            if c["x"] < -0.2 or c["x"] > 1.2 or c["y"] < -0.2 or c["y"] > 1.2:
                import random as _r
                self._comets[i] = self._spawn(_r)
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        for c in self._comets:
            pen = QPen(QColor(255, 255, 240, 180))
            pen.setWidth(2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            x = int(c["x"] * w); y = int(c["y"] * h)
            tx = int(x - (c["vx"] * 0.06)); ty = int(y - (c["vy"] * 0.06))
            p.drawLine(tx, ty, x, y)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(255, 255, 255, 230)))
            p.drawEllipse(QPoint(x, y), 3, 3)
        p.end()