# HaloOrbitLogo.py — central logo with orbiting halo dots and spin
from math import sin, cos, pi
from random import random, randint
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

SIGN_TEXT = "FrameVision"

_prev = []
_last_t = None
_onset_avg = 0.0
_onset_peak = 1e-3

def _flux(bands):
    global _prev
    if not bands:
        _prev = []
        return 0.0
    if (not _prev) or (len(_prev) != len(bands)):
        _prev = [0.0]*len(bands)
    f = 0.0
    n = len(bands)
    for i,(x,px) in enumerate(zip(bands,_prev)):
        d = x - px
        if d>0:
            f += d * (0.35 + 0.65*(i/max(1,n-1)))
    _prev = [0.82*px + 0.18*x for x,px in zip(bands,_prev)]
    return f/max(1,n)

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _adaptive_onset(hi, fx, rms, dt):
    global _onset_avg, _onset_peak
    onset = 0.6*hi + 1.5*fx + 0.2*rms
    _onset_avg = 0.98*_onset_avg + 0.02*onset
    _onset_peak = max(_onset_peak*pow(0.5, dt/0.6), onset)  # decay ~0.6s half-life
    denom = max(1e-3, _onset_peak - 0.6*_onset_avg)
    norm = max(0.0, (onset - 0.8*_onset_avg) / denom)  # ~0..1 adaptive
    return norm

_spin = 0.0
_rings = []  # list of (ang, rad, life)

@register_visualizer
class HaloOrbitLogo(BaseVisualizer):
    display_name = "Halo Orbit Logo"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _spin, _rings
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx = _flux(bands)
        norm = _adaptive_onset(hi, fx, rms, dt)
        # spin speed from mids, kick on hits
        _spin += dt * (0.6 + 1.8*mid + 1.0*(norm>0.55))

        p.fillRect(r, QBrush(QColor(6,8,12)))
        base = min(w,h)
        size = int(base*(0.11 + 0.04*mid))
        p.setFont(QFont("Arial", size, QFont.Bold))

        cx,cy = w/2, h/2
        hue = int((t*18 + 240*mid) % 360)
        col = QColor.fromHsv(hue, 180, 255, 255)

        # draw logo with slight rotation
        p.save()
        p.translate(cx, cy)
        p.rotate(2.0*(0.4*mid + (1 if norm>0.35 else 0)))
        p.setPen(QPen(col, 6))
        p.drawText(QRectF(-w/2, -h/2, w, h), Qt.AlignCenter, SIGN_TEXT)
        p.restore()

        # halo dots
        dots = 24
        R = base*0.28
        for i in range(dots):
            a = (i/dots)*2*pi + _spin
            x = cx + R*cos(a)
            y = cy + R*sin(a)
            p.setPen(QPen(QColor.fromHsv((hue+int(360*i/dots))%360, 160, 255, 180), 3))
            p.drawPoint(QPointF(x,y))

        # ring ripple on hits
        if norm>0.6:
            _rings.append({'r': R, 'life': 0.7})
        alive=[]
        for ring in _rings:
            ring['r'] += 140*dt
            ring['life'] -= dt
            if ring['life']>0: alive.append(ring)
        _rings = alive
        for ring in _rings:
            a = int(200*ring['life'])
            p.setPen(QPen(QColor(255,255,255,a), 3))
            p.drawEllipse(QPointF(cx,cy), ring['r'], ring['r'])
