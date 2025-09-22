from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QPainterPath, QFont
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(777)
_prev = []
_env = 0.0
_gate = 0.0
def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s=c=0.0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        s += w*bands[i]; c += 1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6); f=c=0.0
    for i in range(cut, n):
        d = bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def music_env(bands, rms):
    global _env,_gate
    e=_midhi(bands); f=_flux(bands)
    target = 0.55*e + 1.30*f + 0.20*rms
    target = target/(1+0.7*target)
    if target>_env: _env = 0.70*_env + 0.30*target
    else: _env = 0.92*_env + 0.08*target
    hi,lo = 0.30, 0.18
    g = 1.0 if f>hi else (0.0 if f<lo else _gate)
    _gate = 0.82*_gate + 0.18*g
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate))

_sdict = {}
def spring_to(key, target, t, k=28.0, c=6.5, lo=0.3, hi=3.0):
    s, v, pt = _sdict.get(key, (1.0, 0.0, None))
    if pt is None:
        _sdict[key] = (s, v, t)
        return 1.0
    dt = max(0.0, min(0.033, t-pt))
    a = -k*(s-target) - c*v
    v += a*dt
    s += v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key] = (s, v, t)
    return s

@register_visualizer
class VortexRipper(BaseVisualizer):
    display_name = "Vortex Ripper"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        cx, cy = r.center().x(), r.center().y()
        env, gate = music_env(bands, rms)
        tgt = 1.0 + 0.8*env + (1.2 if gate>0.55 else 0.0)
        amp = spring_to("vortex_amp", tgt, t, hi=2.4)
        turns = 7
        steps = 800
        rot = t*(0.6 + 2.4*amp)
        px, py = None, None
        for i in range(steps):
            frac = i/steps
            ang = 2*pi*turns*frac + rot
            rad = frac * min(w,h)*0.55 * (0.7 + 0.6*amp)
            x = cx + rad*cos(ang)
            y = cy + rad*sin(ang)
            hue = int((t*40 + 360*frac) % 360)
            alpha = int(170*(1-frac))
            p.setPen(QPen(QColor.fromHsv(hue, 230, 255, alpha), 2))
            if px is not None:
                p.drawLine(QPointF(px,py), QPointF(x,y))
            px, py = x, y
