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
class FractalFireTrails(BaseVisualizer):
    display_name = "Fractal Fire Trails"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(4,5,10)))
        N = 220
        for i in range(N):
            th = 0.05*i + t*(0.6+1.2*env)
            x = r.center().x() + (0.35*min(w,h))*(0.6*sin(th*1.3)+0.4*cos(th*0.9))
            y = r.center().y() + (0.35*min(w,h))*(0.6*cos(th*1.1)+0.4*sin(th*0.8))
            hue = int((t*50 + i*2) % 360)
            val = 200
            col = QColor.fromHsv(hue, 230, val, 170)
            p.setPen(QPen(col, 2))
            p.drawPoint(int(x), int(y))
            x2 = x - 8*sin(th); y2 = y - 8*cos(th)
            p.drawLine(QPointF(x,y), QPointF(x2,y2))
        if gate>0.5:
            cx, cy = r.center().x(), r.center().y()
            g = QRadialGradient(QPointF(cx,cy), min(w,h)*0.4)
            g.setColorAt(0.0, QColor(255,200,120,80))
            g.setColorAt(1.0, QColor(0,0,0,0))
            p.setBrush(QBrush(g))
            p.setPen(QPen(QColor(255,255,255,10),1))
            p.drawEllipse(QPointF(cx,cy), min(w,h)*0.25, min(w,h)*0.25)
