from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QConicalGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_prev_bands = []
_env = 0.0
_rng = Random(12345)

def _weighted_energy(bands):
    if not bands: return 0.0
    n = len(bands)
    cut = max(1, n//6)
    total = 0.0; cnt = 0
    for i in range(cut, n):
        w = 0.3 + 0.7*((i-cut)/max(1, n-cut))
        total += bands[i]*w
        cnt += 1
    return total/max(1, cnt)

def _spectral_flux(bands):
    global _prev_bands
    if not bands:
        _prev_bands = []
        return 0.0
    n = len(bands)
    if not _prev_bands or len(_prev_bands)!=n:
        _prev_bands = [0.0]*n
    cut = max(1, n//6)
    flux = 0.0; cnt=0
    for i in range(cut, n):
        w = 0.3 + 0.7*((i-cut)/max(1, n-cut))
        d = bands[i] - _prev_bands[i]
        if d>0: flux += d*w
        cnt += 1
    _prev_bands = [0.9*_prev_bands[i] + 0.1*bands[i] for i in range(n)]
    return flux/max(1, cnt)

def music_drive(bands, rms):
    global _env
    e = _weighted_energy(bands)
    f = _spectral_flux(bands)
    target = 0.6*e + 1.4*f + 0.2*rms
    target = target / (1.0 + 0.8*target)
    if target > _env:
        _env = 0.70*_env + 0.30*target
    else:
        _env = 0.92*_env + 0.08*target
    if _env < 0: _env = 0.0
    if _env > 1.0: _env = 1.0
    return _env

@register_visualizer
class SonicTornado(BaseVisualizer):
    display_name = "Sonic Tornado"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        cx, cy = r.center().x(), r.center().y()
        drive = music_drive(bands, rms)
        segs = 90
        rot = t*(0.6 + 2.0*drive)
        n = len(bands) if bands else 1
        for i in range(segs):
            frac = i/segs
            v = bands[int(frac*n)%n] if bands else 0.0
            lenr = (0.15 + 0.55*v)*(0.7 + 0.9*drive)*min(w,h)
            hue = int((t*30 + i*4) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 220, 240, 170), 2))
            ang = 2*pi*frac + rot + 0.25*sin(t + i*0.05)
            p.drawLine(QPointF(cx,cy), QPointF(cx + lenr*cos(ang), cy + lenr*sin(ang)))
