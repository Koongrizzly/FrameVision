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

@register_visualizer
class PacChomp(BaseVisualizer):
    display_name = "Pac Chomp"
    def paint(self, p: QPainter, r, bands, rms, t):
        from PySide6.QtGui import QPainterPath
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r, QBrush(QColor(5,5,12)))

        drive = env_hard(bands, rms)
        # Path of the chomp: moves left->right, bounces top/bottom slowly
        speed = 0.15 + 0.65*drive
        x = r.left() + ( (t*speed*200) % (w+120) ) - 60
        y = r.center().y() + (h*0.25)*sin(t*0.8) 

        # Mouth animation harder on hits
        mouth = 0.5 + 0.5*sin(3.0*t + 4.0*drive)
        open_ang = 25 + int(40*mouth)
        radius = int(min(w,h)*0.08*(0.9+0.6*drive))

        body = QPainterPath()
        body.moveTo(x,y)
        # Pac-like sector (use arcs to shape mouth)
        body.arcMoveTo(x-radius, y-radius, 2*radius, 2*radius, open_ang)
        body.arcTo(x-radius, y-radius, 2*radius, 2*radius, open_ang, 360-2*open_ang)
        body.closeSubpath()

        p.setBrush(QBrush(QColor(255, 230, 50)))
        p.setPen(QPen(QColor(255,255,160), 2))
        p.drawPath(body)

        # eye
        p.setBrush(QBrush(QColor(30,30,30)))
        p.setPen(QPen(QColor(0,0,0,0), 0))
        p.drawEllipse(QPointF(x+radius*0.2, y-radius*0.35), radius*0.12, radius*0.12)

        # pellets: random-ish dotted line ahead
        for i in range(1,7):
            px = x + 40*i
            py = y + 6*sin(0.7*t + i)
            hue = int((t*20 + i*50) % 360)
            p.setBrush(QBrush(QColor.fromHsv(hue, 200, 255)))
            p.setPen(QPen(QColor(255,255,255,30), 1))
            p.drawEllipse(QPointF(px,py), 5,5)
