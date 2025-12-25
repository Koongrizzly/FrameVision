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

@register_visualizer
class PrismTunnel(BaseVisualizer):
    display_name = "Prism Tunnel"
    def __init__(self):
        super().__init__()
        self._env = 0.0
        self._z = 0.0
        self._prev_t = None

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        drive = _drive_from_audio(bands, rms, 0.55, 0.80, 0.90)
        self._env = _env_step(self._env, drive, 0.62, 0.26)

        if self._prev_t is None:
            self._prev_t = t
        dt = max(0.0, min(0.08, t - self._prev_t))
        self._prev_t = t
        if drive > 0.02:
            self._z += dt * (0.25 + 3.6 * self._env)

        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(1, 2, 6))

        cx, cy = r.center().x(), r.center().y()
        base = min(w, h) * 0.9

        rings = 28
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(rings):
            # depth 0..1
            z = (i / rings) + (self._z % 1.0)
            z = z - 1.0 if z > 1.0 else z
            z = 1.0 - z
            # perspective scale
            s = (0.12 + 0.88 * (z ** 1.7))
            ww = base * s
            hh = base * s * (0.68 + 0.25 * sin(self._z * 0.7))

            # wobble rotates with mids
            ang = (self._z * 0.7 + i * 0.09) * (0.3 + 0.9 * self._env)
            ca, sa = cos(ang), sin(ang)

            def rot(px, py):
                dx, dy = px - cx, py - cy
                return QPointF(cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)

            rect = QRectF(cx - ww * 0.5, cy - hh * 0.5, ww, hh)
            pts = [
                rot(rect.left(), rect.top()),
                rot(rect.right(), rect.top()),
                rot(rect.right(), rect.bottom()),
                rot(rect.left(), rect.bottom())
            ]
            poly = QPolygonF(pts)

            hue = int((210 + 220 * (1.0 - z) + 160 * self._env) % 360)
            alpha = int(12 + 70 * (z ** 1.2) * (0.25 + 0.75 * self._env))
            col = QColor.fromHsv(hue, 210, int(90 + 150 * z), alpha)
            p.setPen(QPen(col, 2 + int(2 * z)))
            p.setBrush(Qt.NoBrush)
            p.drawPolygon(poly)

            # side shards on highs
            if self._env > 0.12 and i % 3 == 0:
                shard = QPainterPath()
                a = pts[0]
                b = pts[1]
                mid = QPointF((a.x() + b.x()) * 0.5, (a.y() + b.y()) * 0.5)
                spike = QPointF(mid.x() + (mid.x() - cx) * 0.15, mid.y() + (mid.y() - cy) * 0.15)
                shard.moveTo(a); shard.lineTo(b); shard.lineTo(spike); shard.closeSubpath()
                col2 = QColor.fromHsv((hue + 40) % 360, 230, 255, int(10 + 60 * z * self._env))
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(col2))
                p.drawPath(shard)

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        # center glow
        glow = QRadialGradient(QPointF(cx, cy), min(w, h) * 0.35)
        glow.setColorAt(0.0, QColor(255, 255, 255, int(18 + 70 * self._env)))
        glow.setColorAt(1.0, QColor(0, 0, 0, 0))
        p.fillRect(r, QBrush(glow))
