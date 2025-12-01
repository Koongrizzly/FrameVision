from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
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
class EdgeLightTrails(BaseVisualizer):
    display_name = "Edge Light Trails"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w = r.width()
        h = r.height()

        lo, mid, hi = _split(bands)
        lo += rms * 0.9
        mid += rms * 0.4
        hi += rms * 0.25

        self._env_lo = _env_step(self._env_lo, lo, up=0.55, down=0.18)
        self._env_mid = _env_step(self._env_mid, mid, up=0.5, down=0.20)
        self._env_hi = _env_step(self._env_hi, hi, up=0.6, down=0.22)

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        # Intensity/length
        base_len = max(w, h) * (0.35 + 0.45 * min(1.2, self._env_lo * 1.4))
        thickness = 1.5 + 6.0 * self._env_mid
        alpha_base = 60 + int(140 * self._env_hi)
        hue_base = int(t * 30.0) % 360

        p.setPen(Qt.NoPen)

        # We generate a small, deterministic swarm of streaks using time
        count = 12
        for i in range(count):
            phase = (t * (0.18 + 0.04 * self._env_mid) + i * 0.19) % 1.0
            fade = 1.0 - phase
            if fade <= 0.02:
                continue

            hue = (hue_base + i * 22) % 360
            alpha = int(alpha_base * fade)
            col = QColor.fromHsv(hue, 230, 255, alpha)

            edge_id = i % 4

            if edge_id == 0:
                # Top edge -> downwards
                x = phase * w
                y1 = 0.0
                y2 = min(h, base_len * fade)
                p.setBrush(col)
                p.drawRect(QRectF(x - thickness * 0.5, y1, thickness, y2))
            elif edge_id == 1:
                # Bottom edge -> upwards
                x = (1.0 - phase) * w
                y2 = h
                y1 = max(0.0, h - base_len * fade)
                p.setBrush(col)
                p.drawRect(QRectF(x - thickness * 0.5, y1, thickness, y2 - y1))
            elif edge_id == 2:
                # Left edge -> right
                y = phase * h
                x1 = 0.0
                x2 = min(w, base_len * fade)
                p.setBrush(col)
                p.drawRect(QRectF(x1, y - thickness * 0.5, x2, thickness))
            else:
                # Right edge -> left
                y = (1.0 - phase) * h
                x2 = w
                x1 = max(0.0, w - base_len * fade)
                p.setBrush(col)
                p.drawRect(QRectF(x1, y - thickness * 0.5, x2 - x1, thickness))

        p.restore()
