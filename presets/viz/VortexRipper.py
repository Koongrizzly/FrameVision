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
class VortexRipper(BaseVisualizer):
    display_name = "Vortex Ripper"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        cx, cy = r.center().x(), r.center().y()
        env, gate = music_env(bands, rms)
        turns = 7
        steps = 750
        rot = t*(0.4 + 1.8*env)
        px, py = None, None
        for i in range(steps):
            frac = i/steps
            ang = 2*pi*turns*frac + rot
            rad = frac * min(w,h)*0.5 * (0.7 + 0.6*env)
            x = cx + rad*cos(ang)
            y = cy + rad*sin(ang)
            hue = int((t*30 + 360*frac) % 360)
            alpha = int(160*(1-frac))
            aoff = 0.0 if gate<0.4 else 2.5
            p.setPen(QPen(QColor.fromHsv(hue, 230, 255, alpha), 2))
            if px is not None:
                p.drawLine(QPointF(px-aoff,py), QPointF(x-aoff,y))
                p.drawLine(QPointF(px+aoff,py), QPointF(x+aoff,y))
            px, py = x, y
