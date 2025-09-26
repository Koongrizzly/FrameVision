from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QFont, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- Music envelopes (mid/high focus) ---
_prev = []
_env_fast = 0.0
_env_slow = 0.0
_rng = Random(24680)

def _midhi_energy(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s=0.0; c=0
    for i in range(cut, n):
        w = 0.4 + 0.6*((i-cut)/max(1, n-cut))
        s += bands[i]*w; c += 1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []
        return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6)
    f=0.0; c=0
    for i in range(cut, n):
        d = bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c += 1
    _prev = [0.85*_prev[i] + 0.15*bands[i] for i in range(n)]
    return f/max(1,c)

def env_hard(bands, rms):
    global _env_fast
    tgt = 0.5*_midhi_energy(bands) + 1.6*_flux(bands) + 0.2*rms
    tgt = tgt/(1+0.6*tgt)
    if tgt > _env_fast:
        _env_fast = 0.55*_env_fast + 0.45*tgt  # fast attack
    else:
        _env_fast = 0.80*_env_fast + 0.20*tgt  # faster release (punchy)
    return max(0.0, min(1.0, _env_fast))

def env_smooth(bands, rms):
    global _env_slow
    tgt = 0.7*_midhi_energy(bands) + 0.9*_flux(bands) + 0.2*rms
    tgt = tgt/(1+0.5*tgt)
    if tgt > _env_slow:
        _env_slow = 0.70*_env_slow + 0.30*tgt  # medium attack
    else:
        _env_slow = 0.93*_env_slow + 0.07*tgt  # slow release (decay)
    return max(0.0, min(1.0, _env_slow))

@register_visualizer
class ThunderLines(BaseVisualizer):
    display_name = "Thunder Lines"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(4,5,10)))
        d = env_hard(bands, rms)
        cx, cy = r.center().x(), r.center().y()
        spokes = 160
        for i in range(spokes):
            ang = 2*pi*i/spokes + 0.2*sin(t + i*0.03)
            lenr = (0.18 + 0.52*d)*(min(w,h))*(0.7 + 0.3*sin(t*2 + i))
            hue = int((t*50 + i*2) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 230, 255, 190), 2))
            p.drawLine(QPointF(cx,cy), QPointF(cx+lenr*cos(ang), cy+lenr*sin(ang)))
        # random strike bursts on very hard hits
        if d > 0.75:
            for k in range(10):
                ang = 2*pi*_rng.random()
                lenr = (0.25+0.5*_rng.random())*min(w,h)
                hue = int(_rng.random()*360)
                p.setPen(QPen(QColor.fromHsv(hue, 230, 255, 220), 3))
                p.drawLine(QPointF(cx,cy), QPointF(cx+lenr*cos(ang), cy+lenr*sin(ang)))
