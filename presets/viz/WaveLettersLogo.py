# WaveLettersLogo.py — per-letter sine wave baseline
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

_amp = 0.0

@register_visualizer
class WaveLettersLogo(BaseVisualizer):
    display_name = "Wave Letters Logo"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _amp
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx = _flux(bands)
        norm = _adaptive_onset(hi, fx, rms, dt)
        # amplitude rises on bass + hits
        target = 0.2 + 0.8*min(1.0, lo*1.5) + 0.6*(norm>0.55)
        _amp = 0.86*_amp + 0.14*target

        p.fillRect(r, QBrush(QColor(10,10,12)))
        base = min(w,h)
        size = int(base*0.12)
        font = QFont("Arial", size, QFont.Bold)
        fm = QFontMetrics(font)
        # compute total width
        spacing = int(size*0.06)
        total = sum(fm.horizontalAdvance(ch) for ch in SIGN_TEXT) + spacing*(len(SIGN_TEXT)-1)
        x = (w - total)//2
        y_mid = int(h*0.55)

        hue = int((t*22 + 200) % 360)
        p.setPen(QPen(QColor.fromHsv(hue, 200, 255, 255), 4))
        p.setFont(font)

        for idx,ch in enumerate(SIGN_TEXT):
            a = (idx*0.6 + t*2.2)
            y = y_mid + int(_amp*size*0.5 * sin(a))
            p.drawText(QPointF(x, y), ch)
            x += fm.horizontalAdvance(ch) + spacing
