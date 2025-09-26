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
class BurstPolygons(BaseVisualizer):
    display_name = "Burst Polygons"
    def paint(self, p: QPainter, r, bands, rms, t):
        from math import floor
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(8,8,16)))
        d = env_hard(bands, rms)
        s = env_smooth(bands, rms)
        # base swirl
        cx, cy = r.center().x(), r.center().y()
        layers = 5 + int(4*d)
        for L in range(layers):
            hue = int((t*30 + L*50) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 255, 170), 2))
            sides = 5 + (L%3)
            rad = (0.10+0.12*L)*min(w,h)*(0.8+0.7*s)
            prev = None
            for i in range(sides+1):
                th = 2*pi*i/sides + t*(0.4 + 1.2*d)
                x = cx + rad*cos(th)
                y = cy + rad*sin(th)
                if prev: p.drawLine(QPointF(prev[0],prev[1]), QPointF(x,y))
                prev = (x,y)
        # random burst when hard hits
        if d > 0.65:
            for i in range(16):
                ang = 2*pi*_rng.random()
                lenr = (30+120*_rng.random())*(0.6+0.8*d)
                hue = int(_rng.random()*360)
                p.setPen(QPen(QColor.fromHsv(hue, 220, 255, 200), 2))
                p.drawLine(QPointF(cx,cy), QPointF(cx+lenr*cos(ang), cy+lenr*sin(ang)))
