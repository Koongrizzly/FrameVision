from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QPainterPath
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
class CornerBeams(BaseVisualizer):
    display_name = "Corner Beams"

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
        lo += rms * 0.9
        mid += rms * 0.45
        hi += rms * 0.35

        self._env_lo = _env_step(self._env_lo, lo, up=0.6, down=0.2)
        self._env_mid = _env_step(self._env_mid, mid, up=0.55, down=0.22)
        self._env_hi = _env_step(self._env_hi, hi, up=0.7, down=0.24)

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        beam_len = max(w, h) * (0.55 + 0.55 * min(1.2, self._env_lo * 1.4))
        spread = 22.0 + 18.0 * self._env_mid
        hue_base = int(t * 26.0) % 360

        corners = [
            QPointF(0.0, 0.0),
            QPointF(w, 0.0),
            QPointF(w, h),
            QPointF(0.0, h),
        ]

        for i, corner in enumerate(corners):
            hue = (hue_base + i * 40) % 360
            alpha_core = 120 + int(110 * self._env_hi)
            col_center = QColor.fromHsv(hue, 240, 255, alpha_core)
            col_fade = QColor.fromHsv(hue, 200, 120, 0)

            # Determine direction from corner towards center, then fan around it
            vx = cx - corner.x()
            vy = cy - corner.y()
            base_angle = atan2(vy, vx) if (vx or vy) else 0.0

            # Simple two-beam fan
            for j in range(-1, 2, 2):
                angle = base_angle + (spread * j * pi / 180.0)
                dx = cos(angle)
                dy = sin(angle)

                p1 = corner
                p2 = QPointF(corner.x() + dx * beam_len, corner.y() + dy * beam_len)
                p3 = QPointF(corner.x() + dx * beam_len * 0.5 - dy * beam_len * 0.15,
                             corner.y() + dy * beam_len * 0.5 + dx * beam_len * 0.15)

                path = QPainterPath()
                path.moveTo(p1)
                path.lineTo(p2)
                path.lineTo(p3)
                path.closeSubpath()

                # Gradient along the beam length
                grad = QLinearGradient(p1, p2)
                grad.setColorAt(0.0, col_center)
                grad.setColorAt(0.55 + 0.15 * self._env_mid, QColor(col_center.red(), col_center.green(), col_center.blue(), int(alpha_core * 0.5)))
                grad.setColorAt(1.0, col_fade)

                p.setBrush(QBrush(grad))
                p.setPen(Qt.NoPen)
                p.drawPath(path)

        p.restore()
