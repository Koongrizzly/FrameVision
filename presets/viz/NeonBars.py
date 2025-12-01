from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n = len(bands)
    a = max(1, n//6); b = max(a+1, n//2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b-a))
    hi = sum(bands[b:]) / max(1, (n-b))
    return lo, mid, hi

def _env_step(env, target, up=0.55, down=0.22):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class NeonBars(BaseVisualizer):
    display_name = "Neon Bars"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(4, 6, 12))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.7*rms, 0.7, 0.25)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.24)
        self._phase += 0.5 + 1.8*self._env_mid

        cols = 40
        bar_w = w / cols
        floor_y = h*0.68

        p.setPen(Qt.NoPen)
        p.setCompositionMode(QPainter.CompositionMode_Plus)

        for i in range(cols):
            x = (i + 0.1)*bar_w
            k = i/float(cols-1)
            local = bands[i % len(bands)] if bands else 0.0
            amp = 0.2*self._env_lo + 0.5*self._env_mid + 0.4*local
            bar_h = (h*0.45) * min(1.2, amp*2.4)

            hue = (int(self._phase*8) + int(k*180)) % 360
            top_col = QColor.fromHsv(hue, 220, 255, int(120 + 120*self._env_hi))
            glow_col = QColor.fromHsv(hue, 200, 255, 40)

            # Main bar
            p.setBrush(QBrush(top_col))
            p.drawRoundedRect(QRectF(x, floor_y-bar_h, bar_w*0.6, bar_h), 3, 3)

            # Glow halo
            p.setBrush(QBrush(glow_col))
            p.drawRect(QRectF(x, floor_y, bar_w*0.6, h*0.18))

            # Reflection
            ref_h = bar_h*0.35
            grad_a = int(80*self._env_hi)
            if grad_a > 0:
                ref_col = QColor.fromHsv(hue, 180, 200, grad_a)
                p.setBrush(QBrush(ref_col))
                p.drawRect(QRectF(x, floor_y, bar_w*0.6, ref_h))

        # Subtle scanline
        if self._env_hi > 0.45:
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            a = int(30 + 90*self._env_hi)
            p.setPen(QPen(QColor(255,255,255,a), 1))
            y = floor_y - (self._phase*4) % h
            p.drawLine(0, int(y), w, int(y))
