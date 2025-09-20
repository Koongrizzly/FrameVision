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
class SamuraiInkSlash(BaseVisualizer):
    display_name = "Samurai Ink Slash"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(5,6,12)))
        slashes = 6 + int(8*env)
        for s in range(slashes):
            ph = t*0.6 + s*0.5
            x1 = r.left() - 40 + (w+80)*( (s*0.17 + 0.2*sin(ph)) % 1.0 )
            y1 = r.top()  + h*(0.2 + 0.6*( (s*0.31 + 0.3*cos(ph)) % 1.0 ))
            ang = 0.6*sin(ph) + 0.9*cos(ph*1.3)
            lenr = 120 + 280*env
            hue = int((t*30 + s*33) % 360)
            core = QColor.fromHsv(hue, 230, 255, 200)
            trail = QColor.fromHsv((hue+40)%360, 220, 230, 120)
            p.setPen(QPen(core, 4))
            p.drawLine(QPointF(x1,y1), QPointF(x1+lenr*cos(ang), y1+lenr*sin(ang)))
            p.setPen(QPen(trail, 2))
            for k in range(1,5):
                a2 = ang + 0.05*k
                p.drawLine(QPointF(x1-10*k*cos(a2), y1-10*k*sin(a2)),
                           QPointF(x1+(lenr-10*k)*cos(a2), y1+(lenr-10*k)*sin(a2)))
        if gate > 0.5:
            for i in range(18):
                ang = 2*pi*_rng.random()
                d = (30+120*_rng.random())*(0.6+0.8*env)
                col = QColor.fromHsv(int(_rng.random()*360), 220, 255, 200)
                p.setPen(QPen(col, 2))
                x = r.center().x(); y = r.center().y()
                p.drawLine(QPointF(x,y), QPointF(x+d*cos(ang), y+d*sin(ang)))
