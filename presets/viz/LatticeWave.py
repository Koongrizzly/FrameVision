from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# Bass emphasis with smoothing that keeps idle motion and adds punch on bass
_s_bass = 0.0
def bass_level(bands, rms):
    global _s_bass
    if bands:
        n = len(bands)
        lo = max(1, n//6)
        bass = sum(bands[:lo]) / lo
    else:
        bass = 0.0
    # light RMS mix, keep range gentle
    lvl = 0.2 + 0.8 * min(1.0, 0.9*bass + 0.3*rms)
    _s_bass = 0.80*_s_bass + 0.20*lvl
    return _s_bass

@register_visualizer
class LatticeWave(BaseVisualizer):
    display_name = "Lattice Wave"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.fillRect(r, QBrush(QColor(12, 12, 22)))

        cols, rows = 28, 16
        lvl = bass_level(bands, rms)

        AMP = 10.0   # base amplitude
        KICK = 12.0  # extra bass kick displacement
        for iy in range(rows):
            for ix in range(cols):
                x = r.left() + ix * (w/(cols-1))
                y = r.top()  + iy * (h/(rows-1))

                # Smooth waves + bass kick
                dx = AMP*(0.5 + 1.6*lvl) * sin(0.7*t + ix*0.28)
                dy = AMP*(0.5 + 1.4*lvl) * cos(0.65*t + iy*0.33)

                # A traveling ripple that strengthens with bass
                ripple = sin(0.9*t - (ix+iy)*0.18)
                dx += KICK*lvl*ripple
                dy += KICK*lvl*sin(0.9*t - (ix-iy)*0.18)

                nx, ny = x + dx, y + dy
                hue = int((ix*10 + iy*6 + t*28) % 360)
                p.setPen(QPen(QColor.fromHsv(hue, 210, 240, 200), 2))

                if ix < cols-1:
                    x2 = r.left() + (ix+1)*(w/(cols-1))
                    dx2 = AMP*(0.5 + 1.6*lvl) * sin(0.7*t + (ix+1)*0.28) + KICK*lvl*sin(0.9*t - (ix+1+iy)*0.18)
                    nx2 = x2 + dx2; ny2 = ny
                    p.drawLine(QPointF(nx, ny), QPointF(nx2, ny2))

                if iy < rows-1:
                    y2 = r.top() + (iy+1)*(h/(rows-1))
                    dy2 = AMP*(0.5 + 1.4*lvl) * cos(0.65*t + (iy+1)*0.33) + KICK*lvl*sin(0.9*t - (ix-(iy+1))*0.18)
                    ny2 = y2 + dy2; nx2 = nx
                    p.drawLine(QPointF(nx, ny), QPointF(nx2, ny2))
