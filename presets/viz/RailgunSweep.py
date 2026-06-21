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
class RailgunSweep(BaseVisualizer):
    display_name = "Railgun Sweep"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        cx, cy = r.center().x(), r.center().y()
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(6,7,14)))
        speed = 0.6 + 2.2*env
        ang = t*speed + 0.4*sin(t*0.7)
        length = 0.52*min(w,h)*(0.7+0.6*env)
        hue = int((t*40) % 360)
        p.setPen(QPen(QColor.fromHsv(hue, 230, 255, 220), 3))
        p.drawLine(QPointF(cx,cy), QPointF(cx + length*cos(ang), cy + length*sin(ang)))
        for k in range(1,5):
            off = k*0.06
            col = QColor.fromHsv((hue+20*k)%360, 220, 240, 120)
            p.setPen(QPen(col, 2))
            p.drawLine(QPointF(cx,cy), QPointF(cx + length*cos(ang+off), cy + length*sin(ang+off)))
            p.drawLine(QPointF(cx,cy), QPointF(cx + length*cos(ang-off), cy + length*sin(ang-off)))
        if gate > 0.4:
            ring = ( (t*2.5) % 1.0 )
            rad = (0.1+0.9*ring)*min(w,h)*0.5
            alpha = int(200*(1.0-ring))
            p.setPen(QPen(QColor.fromHsv((hue+180)%360, 200, 255, alpha), 3))
            p.drawEllipse(QPointF(cx,cy), rad, rad)
