from math import sin, cos, pi, sqrt
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient, QFont, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# --- Music envelopes (mid/high focus) ---
_prev = []
_env_fast = 0.0
_env_slow = 0.0
_rng = Random(24680)

def _midhi_energy(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s=0.0; c=0
    for i in range(cut, n):
        w = 0.4 + 0.6*((i-cut)/max(1, n-cut))
        s += bands[i]*w; c += 1
    return s/max(1,c)

def _flux(bands):
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
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c += 1
    _prev = [0.85*_prev[i] + 0.15*bands[i] for i in range(n)]
    return f/max(1,c)

def env_hard(bands, rms):
    global _env_fast
    tgt = 0.5*_midhi_energy(bands) + 1.6*_flux(bands) + 0.2*rms
    tgt = tgt/(1+0.6*tgt)
    if tgt > _env_fast:
        _env_fast = 0.55*_env_fast + 0.45*tgt  # fast attack
    else:
        _env_fast = 0.80*_env_fast + 0.20*tgt  # faster release (punchy)
    return max(0.0, min(1.0, _env_fast))

def env_smooth(bands, rms):
    global _env_slow
    tgt = 0.7*_midhi_energy(bands) + 0.9*_flux(bands) + 0.2*rms
    tgt = tgt/(1+0.5*tgt)
    if tgt > _env_slow:
        _env_slow = 0.70*_env_slow + 0.30*tgt  # medium attack
    else:
        _env_slow = 0.93*_env_slow + 0.07*tgt  # slow release (decay)
    return max(0.0, min(1.0, _env_slow))

_cols = []
_init = False

@register_visualizer
class MatrixCodeRain(BaseVisualizer):
    display_name = "Matrix Code Rain"
    def paint(self, p: QPainter, r, bands, rms, t):
        global _init, _cols
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        if not _init or not _cols or len(_cols) != max(10, w//18):
            _init = True
            cols = max(10, w//18)
            _cols = [{"x": i*(w/cols)+5, "y": _rng.random()*h, "spd": 40+_rng.random()*120} for i in range(cols)]

        p.fillRect(r, QBrush(QColor(3,5,6)))
        drive = env_hard(bands, rms)
        spd_boost = 1.0 + 2.0*drive

        font = QFont("Consolas", 12)
        p.setFont(font)

        for c in _cols:
            c["y"] += (c["spd"]*spd_boost) * 0.016  # assume ~60fps
            if c["y"] > h + 40:
                c["y"] = -_rng.random()*200
                c["spd"] = 40 + _rng.random()*120

            y = c["y"]
            x = c["x"] + 8*sin(0.002*y + t*0.6)
            # draw 12 glyphs up the column
            for k in range(12):
                yy = y - 18*k
                if yy < -20: break
                val = 140 if k==0 else 70
                hue = 120  # green
                sat = 190
                # shimmer on hits
                val = min(255, int(val + 100*drive*(1.0-0.06*k)))
                p.setPen(QPen(QColor.fromHsv(hue, sat, val, 200)))
                ch = chr(0x30A0 + int(_rng.random()*96))  # Katakana block-ish
                p.drawText(int(x), int(yy), ch)
