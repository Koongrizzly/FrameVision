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
class VoxelCollapse(BaseVisualizer):
    display_name = "Voxel Collapse"
    def paint(self, p: QPainter, r, bands, rms, t):
        from math import floor
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(6,7,12)))
        size = max(12, int(min(w,h)*0.05))
        cols = int(w/size)+2
        rows = int(h/size)+2
        _rng.seed( int(t*2) )
        miss = set()
        if gate > 0.5:
            for _ in range( max(1, cols//6) ):
                miss.add(_rng.randint(0, cols-1))
        for iy in range(rows):
            for ix in range(cols):
                if ix in miss and iy%2==0:
                    continue
                x = r.left()+ix*size
                y = r.top()+iy*size + 6*sin(t*1.2 + ix*0.2 + iy*0.15)*env
                hue = int((t*25 + ix*8 + iy*6) % 360)
                col = QColor.fromHsv(hue, 220, 240, 200)
                p.setBrush(QBrush(col))
                p.setPen(QPen(QColor(0,0,0,40), 1))
                p.drawRect(QRectF(x,y,size-2,size-2))
