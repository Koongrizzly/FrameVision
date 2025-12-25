from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient
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
class NeonFrameRails(BaseVisualizer):
    display_name = "Neon Frame Rails"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w = r.width()
        h = r.height()

        lo, mid, hi = _split(bands)
        lo += rms * 0.8
        mid += rms * 0.4
        hi += rms * 0.25

        self._env_lo = _env_step(self._env_lo, lo, up=0.55, down=0.18)
        self._env_mid = _env_step(self._env_mid, mid, up=0.5, down=0.20)
        self._env_hi = _env_step(self._env_hi, hi, up=0.6, down=0.22)

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        thickness = 2.0 + 12.0 * min(1.5, self._env_lo * 1.5)
        glow_extra = 4.0 + 20.0 * self._env_mid

        hue_base = int(t * 25.0) % 360

        for i, edge in enumerate(("top", "bottom", "left", "right")):
            hue = (hue_base + i * 40) % 360
            alpha_core = 120 + int(90 * self._env_hi)
            col_main = QColor.fromHsv(hue, 255, 240, alpha_core)
            col_fade = QColor.fromHsv(hue, 220, 80, 0)

            if edge in ("top", "bottom"):
                y = 0.0 if edge == "top" else h - thickness
                grad = QLinearGradient(0.0, y, w, y)
                offset = (t * (0.18 + 0.08 * self._env_mid) + i * 0.21) % 1.0
                for j in range(4):
                    pos = (offset + j * 0.25) % 1.0
                    grad.setColorAt(pos, col_main)
                grad.setColorAt(0.0, col_fade)
                grad.setColorAt(1.0, col_fade)
                p.setBrush(QBrush(grad))
                p.setPen(Qt.NoPen)
                p.drawRect(QRectF(0.0, y, w, thickness + glow_extra * 0.4))
            else:
                x = 0.0 if edge == "left" else w - thickness
                grad = QLinearGradient(x, 0.0, x, h)
                offset = (t * (0.22 + 0.07 * self._env_mid) + i * 0.19) % 1.0
                for j in range(4):
                    pos = (offset + j * 0.25) % 1.0
                    grad.setColorAt(pos, col_main)
                grad.setColorAt(0.0, col_fade)
                grad.setColorAt(1.0, col_fade)
                p.setBrush(QBrush(grad))
                p.setPen(Qt.NoPen)
                p.drawRect(QRectF(x, 0.0, thickness + glow_extra * 0.4, h))

        p.restore()
