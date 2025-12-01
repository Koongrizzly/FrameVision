from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor
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
class EdgeOutlineFlicker(BaseVisualizer):
    display_name = "Edge Outline Flicker"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w = r.width()
        h = r.height()

        lo, mid, hi = _split(bands)
        lo += rms * 0.7
        mid += rms * 0.5
        hi += rms * 0.45

        self._env_lo = _env_step(self._env_lo, lo, up=0.55, down=0.20)
        self._env_mid = _env_step(self._env_mid, mid, up=0.5, down=0.22)
        self._env_hi = _env_step(self._env_hi, hi, up=0.65, down=0.24)

        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        thickness = 1.2 + 4.5 * self._env_mid
        segments = 24
        base_len = (w * 2.0 + h * 2.0)
        seg_len = base_len / segments
        hue_base = int(t * 24.0) % 360

        # We walk along the perimeter as a 1D line, then map back to 4 edges.
        offset = (t * (0.4 + 0.3 * self._env_mid)) % seg_len

        for i in range(segments):
            pos = i * seg_len + offset
            frac = pos / base_len
            flicker = 0.3 + 0.7 * max(0.0, sin(frac * pi * 2.0 + t * 6.0 * self._env_hi))
            alpha = int((80 + 140 * self._env_hi) * flicker)

            if alpha < 10:
                continue

            hue = (hue_base + i * 11) % 360
            col = QColor.fromHsv(hue, 240, 255, alpha)
            pen = QPen(col, thickness, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)

            # Map perimeter position to edge
            d = pos % base_len
            if d < w:
                x1 = d
                x2 = min(w, d + seg_len * 0.7)
                y1 = 0.0
                y2 = 0.0
            elif d < w + h:
                y1 = d - w
                y2 = min(h, y1 + seg_len * 0.7)
                x1 = w
                x2 = w
            elif d < 2 * w + h:
                x1 = (2 * w + h - d)
                x2 = max(0.0, x1 - seg_len * 0.7)
                y1 = h
                y2 = h
            else:
                y1 = (base_len - d)
                y2 = max(0.0, y1 - seg_len * 0.7)
                x1 = 0.0
                x2 = 0.0

            p.drawLine(x1, y1, x2, y2)

        p.restore()
