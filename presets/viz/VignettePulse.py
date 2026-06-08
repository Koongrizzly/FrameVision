from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
from PySide6.QtCore import QRectF, QPointF, Qt
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


def _env_step(env, target, up=0.6, down=0.23):
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target


@register_visualizer
class VignettePulse(BaseVisualizer):
    display_name = "Beat Vignette Pulse"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w = r.width()
        h = r.height()
        cx = w * 0.5
        cy = h * 0.5

        lo, mid, hi = _split(bands)
        lo += rms * 1.1
        mid += rms * 0.4
        hi += rms * 0.35

        self._env_lo = _env_step(self._env_lo, lo, up=0.7, down=0.25)
        self._env_mid = _env_step(self._env_mid, mid, up=0.5, down=0.22)
        self._env_hi = _env_step(self._env_hi, hi, up=0.65, down=0.24)

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        radius = (w + h) * 0.75 * (0.9 + 0.4 * min(1.5, self._env_lo * 1.8))
        hue = (int(t * 14.0) + int(self._env_mid * 120.0)) % 360

        # Soft beat "breathing" via brightness
        brightness = 180 + int(70 * max(0.0, sin(t * 3.0) * self._env_lo))
        alpha_outer = 30 + int(110 * self._env_lo)
        alpha_mid = int(alpha_outer * (0.6 + 0.3 * self._env_hi))

        c_outer = QColor.fromHsv(hue, 180, brightness, alpha_outer)
        c_mid = QColor.fromHsv(hue, 210, min(255, brightness + 30), alpha_mid)
        c_center = QColor(0, 0, 0, 0)

        grad = QRadialGradient(QPointF(cx, cy), radius)
        grad.setColorAt(0.0, c_center)
        grad.setColorAt(0.55 + 0.15 * self._env_mid, c_center)
        grad.setColorAt(0.8, c_mid)
        grad.setColorAt(1.0, c_outer)

        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(0.0, 0.0, w, h))

        p.restore()
