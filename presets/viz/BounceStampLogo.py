# BounceStampLogo.py — vertical bounce with squash/stretch
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

_imp = 0.0
_phase = 0.0
_eq_env = [0.0]*12

@register_visualizer
class BounceStampLogo(BaseVisualizer):
    display_name = "Bounce Stamp Logo"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _imp, _phase
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx = _flux(bands)
        norm = _adaptive_onset(hi, fx, rms, dt)
        if norm>0.55: _imp = 1.0
        _imp *= pow(0.5, dt/0.05)  # bounce impulse fades ~100ms
        _phase += dt*(0.6 + 1.2*mid)

        p.fillRect(r, QBrush(QColor(8,9,12)))
        base = min(w,h)
        size = int(base*0.11)
        p.setFont(QFont("Arial", size, QFont.Black))

        # bounce + squash
        y_bounce = int(h*0.06 * (0.35*norm + _imp))
        sx = 1.0 + 0.06*(0.35*norm + _imp)
        sy = 1.0 - 0.08*(0.35*norm + _imp)

        p.save()
        p.translate(w/2, h/2 - y_bounce)
        p.scale(sx, sy)
        hue = int((t*20 + 200) % 360)
        col = QColor.fromHsv(hue, 160, 255, 255)
        # shadow
        p.setPen(QPen(QColor(0,0,0,150), 10))
        p.drawText(QRectF(-w/2, -h/2, w, h), Qt.AlignCenter, SIGN_TEXT)
        # fill
        p.setPen(QPen(col, 4))
        p.drawText(QRectF(-w/2, -h/2, w, h), Qt.AlignCenter, SIGN_TEXT)
        p.restore()

        
        # --- colorful 12‑band equalizer under the logo ---
        eq_n = 12
        global _eq_env
        if len(_eq_env) != eq_n:
            _eq_env = [0.0]*eq_n
        if bands:
            N = len(bands)
            for i in range(eq_n):
                a = int(i*N/eq_n)
                b = int((i+1)*N/eq_n)
                if b <= a: b = a+1
                val = sum(bands[a:b])/(b-a)
                target = 2.4*val + 1.2*fx
                _eq_env[i] = _env_step(_eq_env[i], target, up=0.55, down=0.22)
        else:
            for i in range(eq_n):
                _eq_env[i] *= 0.9

        base_y = int(h*0.62)
        left = int(w*0.15)
        right = int(w*0.85)
        eq_w = right - left
        bw = max(2, int(eq_w/eq_n * 0.7))
        gap = max(1, int(eq_w/eq_n - bw))
        x = left
        hue0 = int((t*18 + 120*mid) % 360)
        for i in range(eq_n):
            e = max(0.0, min(1.0, _eq_env[i]))
            bh = max(3, int(e * h * 0.24))
            p.setBrush(QBrush(QColor.fromHsv((hue0 + int(i*360/eq_n)) % 360, 200, 255, 220)))
            p.setPen(QPen(QColor(0,0,0,140), 1))
            p.drawRect(QRectF(x, base_y - bh, bw, bh))
            x += bw + gap
