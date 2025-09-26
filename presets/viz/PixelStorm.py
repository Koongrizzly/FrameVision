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

_rng_pts = [_rng.random() for _ in range(900)]
@register_visualizer
class PixelStorm(BaseVisualizer):
    display_name = "Pixel Storm"
    def paint(self, p: QPainter, r, bands, rms, t):
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(5,6,12)))
        drive = music_drive(bands, rms)
        speed = 40 + 220*drive
        N = len(_rng_pts)//2
        for i in range(N):
            ux = _rng_pts[2*i]; uy = _rng_pts[2*i+1]
            x = r.left() + ((ux + 0.001*speed*sin(t*0.8 + i*0.02)) % 1.0) * w
            y = r.top()  + ((uy + 0.001*speed*cos(t*0.9 + i*0.03)) % 1.0) * h
            hue = int((t*60 + i*2) % 360)
            p.setPen(QPen(QColor.fromHsv(hue, 230, 255, 160), 1))
            x2 = r.left() + ((ux + 0.001*speed*sin(t*0.8 + i*0.02) - 0.008) % 1.0) * w
            y2 = r.top()  + ((uy + 0.001*speed*cos(t*0.9 + i*0.03) - 0.008) % 1.0) * h
            p.drawLine(QPointF(x,y), QPointF(x2,y2))
