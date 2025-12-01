from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1, n//6); b=max(a+1, n//2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

def _env_step(env, target, up=0.55, down=0.22):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class EqualizerArc(BaseVisualizer):
    display_name = "Equalizer Arc"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(5, 6, 12))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.5*rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)

        cx, cy = w*0.5, h*0.65
        inner = min(w, h)*0.18
        outer = min(w, h)*0.48

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(10, 10, 20)))
        p.drawEllipse(QPointF(cx, cy), outer+14, outer+14)

        bins = 48
        span = pi * 1.3
        start = -span*0.5

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(bins):
            phase = i / max(1, bins-1)
            idx = int(phase * max(1, len(bands)-1)) if bands else 0
            amp = bands[idx] if bands else 0.0
            env = 0.35*self._env_lo + 0.6*self._env_mid + 0.4*amp
            ang = start + span * phase
            base_r = inner
            top_r = inner + (outer-inner)*(0.1 + 0.9*env)
            x1 = cx + cos(ang)*base_r
            y1 = cy + sin(ang)*base_r
            x2 = cx + cos(ang)*top_r
            y2 = cy + sin(ang)*top_r
            hue = (int(phase*220 + t*40) % 360)
            col = QColor.fromHsv(hue, 230, 255, int(80 + 140*self._env_hi))
            p.setPen(QPen(col, 3))
            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        p.setPen(QPen(QColor(130, 200, 255, 120), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(cx, cy), inner, inner)
