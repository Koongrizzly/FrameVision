from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QConicalGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(9993)
_prev = []
_env = 0.0
_gate = 0.0

def _midhi_energy(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s = 0.0; c = 0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        s += bands[i]*w; c += 1
    return s/max(1,c)

def _spectral_flux(bands):
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
        if d>0: f += d*(0.35+0.65*((i-cut)/max(1,n-cut)))
        c += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def music_env(bands, rms):
    global _env, _gate
    e = _midhi_energy(bands)
    f = _spectral_flux(bands)
    target = 0.55*e + 1.25*f + 0.20*rms
    target = target/(1+0.7*target)
    if target > _env:
        _env = 0.68*_env + 0.32*target
    else:
        _env = 0.91*_env + 0.09*target
    thr_hi = 0.32
    thr_lo = 0.18
    g = 1.0 if f > thr_hi else (0.0 if f < thr_lo else _gate)
    _gate = 0.8*_gate + 0.2*g
    return max(0.0, min(1.0, _env)), max(0.0, min(1.0, _gate))

@register_visualizer
class HoloSigils(BaseVisualizer):
    display_name = "Holo-Sigils"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        cx, cy = r.center().x(), r.center().y()
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(6,6,14)))
        for ring in range(3):
            hue = int((t*25 + ring*60) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 255, 200), 2))
            rad = (0.12 + 0.12*ring)*min(w,h)*(0.8+0.6*env)
            sides = 6 + ring*2
            prev = None
            for i in range(sides+1):
                th = 2*pi*i/sides + t*(0.2+0.8*env) + ring*0.3
                x = cx + rad*cos(th); y = cy + rad*sin(th)
                if prev: p.drawLine(QPointF(prev[0],prev[1]), QPointF(x,y))
                prev = (x,y)
        if gate > 0.55:
            for k in range(14):
                th = 2*pi*k/14 + t*1.2
                L = (0.18 + 0.4*_rng.random())*min(w,h)
                col = QColor.fromHsv(int(_rng.random()*360), 220, 255, 200)
                p.setPen(QPen(col, 2))
                p.drawLine(QPointF(cx,cy), QPointF(cx+L*cos(th), cy+L*sin(th)))
