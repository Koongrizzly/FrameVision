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

_cols = []
_init = False
_hue_shift = 0.0

@register_visualizer
class MatrixColorRain(BaseVisualizer):
    display_name = "Matrix Color Rain (Chromatic)"
    def paint(self, p: QPainter, r, bands, rms, t):
        global _init, _cols, _hue_shift
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if not _init or not _cols or len(_cols) != max(10, w//18):
            _init = True
            cols = max(10, w//18)
            _cols = [{"x": i*(w/cols)+5, "y": _rng.random()*h, "spd": 40+_rng.random()*120} for i in range(cols)]
        env, gate = music_env(bands, rms)
        p.fillRect(r, QBrush(QColor(0,0,0)))
        spd_boost = 1.0 + 2.2*env + (0.8 if gate>0.6 else 0.0)
        _hue_shift = (_hue_shift + 0.8 + 20*env) % 360
        font = QFont("Consolas", 12)
        p.setFont(font)
        for c in _cols:
            c["y"] += (c["spd"]*spd_boost) * 0.016
            if c["y"] > h + 40:
                c["y"] = -_rng.random()*200
                c["spd"] = 40 + _rng.random()*120
            y = c["y"]
            x = c["x"] + 8*sin(0.002*y + t*0.6)
            for k in range(12):
                yy = y - 18*k
                if yy < -20: break
                hue = int((_hue_shift + k*4) % 360)
                val = 160 if k==0 else 90
                if gate>0.5: val = min(255, val + 80)
                p.setPen(QPen(QColor.fromHsv(hue, 220, val, 220)))
                ch = chr(0x30A0 + int(_rng.random()*96))
                p.drawText(int(x), int(yy), ch)
