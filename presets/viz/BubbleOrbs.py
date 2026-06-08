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

_rng = Random(123)
# x, y base in [0,1], size in [0.02,0.08], phase for bobbing
orbs = [(_rng.random(), _rng.random(), 0.02 + 0.06*_rng.random(), 2*pi*_rng.random()) for _ in range(120)]

@register_visualizer
class BubbleOrbs(BaseVisualizer):
    display_name = "Bubble Orbs"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        # Tuning knobs (feel free to tweak)
        DRIFT_BASE = 0.006   # vertical drift speed baseline
        DRIFT_BASS = 0.010   # added drift per bass level (lower than before)
        BOB_FREQ    = 0.8    # bobbing base frequency
        BOB_AMPL    = 0.04   # bobbing amplitude as fraction of height at max bass
        SIZE_BASE   = 0.55   # base orb size scalar
        SIZE_BASS   = 0.25   # size boost with bass (reduced so orbs don't "overgrow")

        p.fillRect(r, QBrush(QColor(8,10,18)))
        lvl = bass_level(bands, rms)

        for i,(ux, uy, sz, ph) in enumerate(orbs):
            # gentle horizontal sway, independent of bass (keeps life in quiet sections)
            x = r.left() + ((ux + 0.015*sin(t*0.6 + i*0.3)) % 1.0) * w

            # upward drift + subtle bass bobbing
            v_speed = DRIFT_BASE + DRIFT_BASS*lvl
            y_norm = (uy + v_speed*t) % 1.0
            y = r.bottom() - y_norm*h

            # add local bobbing that scales with bass but never cancels drift
            y += h * (BOB_AMPL*lvl) * sin(BOB_FREQ*t + ph)

            # size with mild bass coupling
            rad = sz * min(w,h) * (SIZE_BASE + SIZE_BASS*lvl)

            hue = int((i*7 + t*40) % 360)
            g = QRadialGradient(QPointF(x,y), rad)
            g.setColorAt(0.0, QColor.fromHsv(hue, 140, 255, 220))
            g.setColorAt(1.0, QColor.fromHsv(hue, 220, 30, 0))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,18), 1))
            p.drawEllipse(QPointF(x,y), rad, rad)
