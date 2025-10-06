# NeonPulseLogo.py — size + glow pop
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

_pop = 0.0
_warm = 0.0

@register_visualizer
class NeonPulseLogo(BaseVisualizer):
    display_name = "Neon Pulse Logo"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _pop, _warm
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx = _flux(bands)
        norm = _adaptive_onset(hi, fx, rms, dt)
        # warm glow envelope follows mids/flux
        _warm = 0.85*_warm + 0.15*min(1.0, 0.9*mid + 0.8*fx)
        # impulse pop
        _pop = 1.0 if norm>0.55 else _pop*pow(0.5, dt/0.07)

        p.fillRect(r, QBrush(QColor(4,7,10)))
        base = min(w,h)
        size = int(base*(0.11 + 0.06*_warm) * (0.7 + 0.30*min(1.0, 0.35*norm + _pop)))
        p.setFont(QFont("Arial", size, QFont.Bold))

        hue = int((t*18 + 260*_warm) % 360)
        col = QColor.fromHsv(hue, 200, 255, 255)
        glow = int(60 + 180*_warm)

        # multi-stroke glow
        for thick in (20, 14, 9, 5, 2):
            tsc = max(1, int(thick * (1.0 + 0.25*_pop)))
            p.setPen(QPen(QColor(col.red(), col.green(), col.blue(), max(12, glow//tsc)), tsc))
            p.drawText(r, Qt.AlignCenter, SIGN_TEXT)
