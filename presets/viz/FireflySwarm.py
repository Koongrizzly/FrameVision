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

_rng = Random(321)
# base u,v in [0,1], personal phase, swirl radius
points = [(_rng.random(), _rng.random(), 2*pi*_rng.random(), 0.012 + 0.02*_rng.random()) for _ in range(260)]

@register_visualizer
class FireflySwarm(BaseVisualizer):
    display_name = "Firefly Swarm"
    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.fillRect(r, QBrush(QColor(6, 6, 14)))
        lvl = bass_level(bands, rms)

        BASE_SWAY = 0.018
        BASS_SWAY = 0.020  # extra speed from bass, but keep moderate
        SWIRL_FREQ = 2.2
        TRAIL_STEPS = 3     # short trail for visible motion

        for i,(ux, uy, ph, radn) in enumerate(points):
            # drift speed
            spd = BASE_SWAY + BASS_SWAY*lvl
            # main drift
            x_norm = (ux + spd*t + 0.010*sin(0.6*t + ph)) % 1.0
            y_norm = (uy + spd*0.9*t + 0.010*cos(0.5*t + ph)) % 1.0

            # local swirl that grows with bass
            swirl = radn * (0.6 + 0.9*lvl)
            x = r.left() + (x_norm + swirl*cos(SWIRL_FREQ*t + ph)) * w % w
            y = r.top()  + (y_norm + swirl*sin(SWIRL_FREQ*t + ph)) * h % h

            hue = int((i*4 + t*60) % 360)
            base_val = 180 + int(60*(0.5+0.5*sin(t*3 + ph)))
            val = min(255, int(base_val + 60*lvl))

            # draw tiny trail
            for k in range(TRAIL_STEPS, -1, -1):
                fade = (k+1)/(TRAIL_STEPS+1)
                xx = r.left() + (x_norm - k*0.012*spd + swirl*cos(SWIRL_FREQ*(t - k*0.05) + ph)) * w % w
                yy = r.top()  + (y_norm - k*0.012*spd + swirl*sin(SWIRL_FREQ*(t - k*0.05) + ph)) * h % h
                g = QRadialGradient(QPointF(xx,yy), 6+4*lvl)
                g.setColorAt(0.0, QColor.fromHsv(hue, 230, val, int(220*fade)))
                g.setColorAt(1.0, QColor.fromHsv(hue, 230, 0, 0))
                p.setBrush(QBrush(g))
                p.setPen(QPen(QColor(255,255,255,10), 1))
                p.drawEllipse(QPointF(xx,yy), 1.4+1.2*lvl, 1.4+1.2*lvl)
