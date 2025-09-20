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
class PacSprintBoss(BaseVisualizer):
    display_name = "Pac Sprint (Boss Mode)"
    def paint(self, p: QPainter, r, bands, rms, t):
        from PySide6.QtGui import QPainterPath
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(5,6,12)))
        speed = 0.2 + 2.0*env + (1.5 if gate>0.6 else 0.0)
        x = r.left() + ( (t*speed*120) % (w+160) ) - 80
        y = r.center().y() + (h*0.28)*sin(t*0.9)
        rad = int(min(w,h)*0.07*(0.8+0.6*env))
        mouth = 20 + int(30*(0.6+0.4*sin(3.0*t + 2.0*env)))
        body = QPainterPath()
        body.arcMoveTo(x-rad, y-rad, 2*rad, 2*rad, mouth)
        body.arcTo(x-rad, y-rad, 2*rad, 2*rad, mouth, 360-2*mouth)
        body.closeSubpath()
        p.setBrush(QBrush(QColor(255, 230, 60)))
        p.setPen(QPen(QColor(255,255,160), 2))
        p.drawPath(body)
        for i in range(10):
            px = x - 18*i; py = y + 6*sin(t*0.8 + i*0.6)
            p.setBrush(QBrush(QColor.fromHsv(int((t*30+i*40)%360), 200, 255, 220)))
            p.setPen(QPen(QColor(255,255,255,20),1))
            p.drawEllipse(QPointF(px,py), 4,4)
