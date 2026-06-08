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
class CornerFlares(BaseVisualizer):
    display_name = "Corner Flares"

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
        lo += rms * 0.8
        mid += rms * 0.5
        hi += rms * 0.3

        self._env_lo = _env_step(self._env_lo, lo, up=0.55, down=0.18)
        self._env_mid = _env_step(self._env_mid, mid, up=0.5, down=0.2)
        self._env_hi = _env_step(self._env_hi, hi, up=0.6, down=0.22)

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        base_radius = 0.55 * (w * 0.5 + h * 0.5)
        pulse = 0.45 + 0.55 * min(1.5, self._env_lo * 1.7)
        radius = base_radius * pulse

        hue_base = int(t * 18.0) % 360

        corners = [
            QPointF(0.0, 0.0),
            QPointF(w, 0.0),
            QPointF(w, h),
            QPointF(0.0, h),
        ]

        for i, corner in enumerate(corners):
            hue = (hue_base + i * 45) % 360
            alpha_edge = 90 + int(120 * self._env_hi)
            c_inner = QColor.fromHsv(hue, 220, 255, alpha_edge)
            c_mid = QColor.fromHsv(hue, 200, 190, int(alpha_edge * 0.7))
            c_outer = QColor.fromHsv(hue, 180, 80, 0)

            grad = QRadialGradient(corner, radius)
            grad.setColorAt(0.0, c_inner)
            grad.setColorAt(0.4 + 0.15 * self._env_mid, c_mid)
            grad.setColorAt(1.0, c_outer)

            p.setBrush(QBrush(grad))
            p.setPen(Qt.NoPen)

            # Slight breathing of the wedge via rotation of the painter around the corner
            p.save()
            p.translate(corner)
            wobble = (sin(t * 0.7 + i * 0.8) * 10.0 * self._env_mid)
            p.rotate(wobble)
            p.translate(-corner)
            p.drawRect(QRectF(0.0, 0.0, w, h))
            p.restore()

        p.restore()
