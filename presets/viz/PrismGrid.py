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

def _env_step(env, target, up=0.56, down=0.22):
    return (1-up)*env + up*target if target > env else (1-down)*env + down*target

@register_visualizer
class PrismGrid(BaseVisualizer):
    display_name = "Prism Grid"
    def __init__(self):
        super().__init__()
        self._env_lo = self._env_mid = self._env_hi = 0.0
        self._t = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(r, QColor(6, 8, 14))

        lo, mid, hi = _split(bands)
        self._env_lo = _env_step(self._env_lo, lo + 0.7*rms, 0.7, 0.24)
        self._env_mid = _env_step(self._env_mid, mid, 0.6, 0.22)
        self._env_hi = _env_step(self._env_hi, hi, 0.6, 0.26)
        self._t += 0.5 + 1.4*self._env_mid

        cols = 12
        rows = 7
        pad = 6
        cell_w = (w - pad*(cols+1)) / cols
        cell_h = (h - pad*(rows+1)) / rows

        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for gy in range(rows):
            for gx in range(cols):
                x = pad + gx*(cell_w+pad)
                y = pad + gy*(cell_h+pad)

                kx = gx/float(max(1,cols-1))
                ky = gy/float(max(1,rows-1))
                idx = (gx + gy*cols)
                local = bands[idx % len(bands)] if bands else 0.0

                wobble = sin(self._t*0.12 + kx*3.0 + ky*4.0)*0.6
                raise_amt = (0.3*self._env_lo + 0.5*self._env_mid + 0.4*local) + wobble*0.25
                raise_amt = max(0.0, raise_amt)

                top = y + cell_h*(1.0 - 0.75*min(1.3, raise_amt))
                hue = (int(self._t*3) + int(kx*140) + int(ky*110)) % 360

                # Base tile
                base_col = QColor.fromHsv(hue, 80, 80, 180)
                edge_col = QColor.fromHsv(hue, 120, 180, 220)
                p.setPen(QPen(edge_col, 1))
                p.setBrush(QBrush(base_col))
                p.drawRect(QRectF(x, y, cell_w, cell_h))

                # Raised prism
                a = int(80 + 160*self._env_hi*raise_amt)
                prism_col = QColor.fromHsv(hue, 220, 255, a)
                p.setPen(QPen(prism_col, 1))
                p.setBrush(QBrush(prism_col))
                p.drawRect(QRectF(x+cell_w*0.16, top, cell_w*0.68, y+cell_h - top))

        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
