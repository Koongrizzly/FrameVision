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
class WarpRibbons(BaseVisualizer):
    display_name = "Warp Ribbons"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(5,6,12)))
        d = env_smooth(bands, rms)
        layers = 6
        for k in range(layers):
            hue = int((t*25 + k*40) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 200, 250, 200), 2))
            prev = None
            phase = k*0.7 + t*(0.4 + 1.2*d)
            step = max(4, w//260*4)
            amp = (0.06*h + 10*k)*(0.9 + 1.4*d)
            for i in range(0, w+step, step):
                x = r.left() + i
                v = 0.0
                if bands:
                    idx = int(i/max(1,w) * len(bands))
                    idx = max(0, min(len(bands)-1, idx))
                    v = bands[idx]
                y = r.center().y() + amp*sin(i*0.010 + phase) + (h*0.015)*sin(i*0.05 + t) - v*10
                if prev: p.drawLine(QPointF(prev[0],prev[1]), QPointF(x,y))
                prev = (x,y)
