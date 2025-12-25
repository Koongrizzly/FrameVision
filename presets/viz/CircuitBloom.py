from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath, QLinearGradient, QRadialGradient, QPolygonF
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands:
        return 0.0, 0.0, 0.0
    n = len(bands)
    a = max(1, n // 6)
    b = max(a + 1, n // 2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b - a))
    hi = sum(bands[b:]) / max(1, (n - b))
    return lo, mid, hi

def _env_step(env, target, up=0.55, down=0.22):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target

def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _drive_from_audio(bands, rms, lo_w=0.65, mid_w=0.85, hi_w=0.55):
    lo, mid, hi = _split(bands)
    drive = lo_w * lo + mid_w * mid + hi_w * hi + 0.6 * rms
    drive = drive / (1.0 + 0.9 * drive)
    return _clamp01(drive)

def _hash01(x):
    x = (x * 1103515245 + 12345) & 0x7FFFFFFF
    return (x % 10000) / 10000.0

@register_visualizer
class CircuitBloom(BaseVisualizer):
    display_name = "Circuit Bloom"
    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._phase = 0.0
        self._prev_t = None
        self._seed = 1337

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        lo, mid, hi = _split(bands)
        drive = _drive_from_audio(bands, rms, 0.45, 0.95, 0.80)
        # quick "bloom" response on highs
        target = drive * (0.65 + 0.55 * hi)
        self._env = _env_step(self._env, target, 0.70, 0.22)

        if self._prev_t is None:
            self._prev_t = t
        dt = max(0.0, min(0.08, t - self._prev_t))
        self._prev_t = t
        if drive > 0.02:
            self._phase += dt * (0.10 + 2.4 * self._env)

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(5, 7, 10))

        # faint grid
        p.setPen(QPen(QColor(30, 35, 50, 35), 1))
        step = max(18, int(min(w, h) / 18))
        x = int(r.left())
        while x < int(r.right()):
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            x += step
        y = int(r.top())
        while y < int(r.bottom()):
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            y += step

        cx, cy = r.center().x(), r.center().y()
        # branching circuit petals
        branches = 16
        depth = 6
        max_len = min(w, h) * (0.38 + 0.20 * self._env)

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for b in range(branches):
            ang = (b / branches) * 2 * pi + self._phase * (0.35 + 0.65 * (b % 2))
            x, y = cx, cy
            for d in range(depth):
                # deterministic jitter per branch+depth, motion via phase
                k = (b * 97 + d * 131 + self._seed) & 0x7FFFFFFF
                turn = ( _hash01(k) - 0.5 ) * (0.9 + 1.4 * self._env)
                ang2 = ang + turn + 0.25 * sin(self._phase * 1.4 + b * 0.3 + d)
                seg = (max_len / depth) * (0.55 + 0.8 * self._env) * (0.85 + 0.15 * (d + 1))
                nx = x + cos(ang2) * seg
                ny = y + sin(ang2) * seg

                hue = int((120 + 200 * (b / branches) + 140 * self._env) % 360)
                alpha = int(10 + 90 * (1.0 - d / depth) * (0.25 + 0.75 * self._env))
                p.setPen(QPen(QColor.fromHsv(hue, 200, 255, alpha), 2))
                p.drawLine(QPointF(x, y), QPointF(nx, ny))

                # "node" squares
                sz = 2.5 + 5.0 * (0.25 + 0.75 * self._env) * (1.0 - d / depth)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(QColor.fromHsv((hue + 30) % 360, 160, 255, int(12 + 85 * self._env))))
                p.drawRect(QRectF(nx - sz, ny - sz, 2 * sz, 2 * sz))

                x, y = nx, ny

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        # center core
        core = QRadialGradient(QPointF(cx, cy), min(w, h) * 0.16)
        core.setColorAt(0.0, QColor(255, 255, 255, int(60 + 140 * self._env)))
        core.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(r, QBrush(core))
