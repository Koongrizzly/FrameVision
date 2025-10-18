# ArcEQBadge.py — circular arc equalizer around a badge logo
from math import sin, cos, pi
from random import random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QFont, QFontMetrics
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

SIGN_TEXT = "FrameVision"

_prev = []
_last_t = None

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

def _env_step(env, target, up=0.42, down=0.16):
    return (1-up)*env + up*target if target>env else (1-down)*env + down*target

def _sample_bands(bands, n_out):
    if not bands or n_out<=0: return [0.0]*n_out
    n = len(bands)
    vals = []
    for i in range(n_out):
        a = int(i*n/n_out); b = int((i+1)*n/n_out)
        if b<=a: b=a+1
        vals.append(sum(bands[a:b])/(b-a))
    return vals

_eq = [0.0]*24
_spin = 0.0

@register_visualizer
class ArcEQBadge(BaseVisualizer):
    display_name = "Arc EQ Badge"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _eq, _spin
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx=_flux(bands)
        _spin += dt*(0.4 + 1.6*mid)

        p.fillRect(r, QBrush(QColor(6,7,10)))

        # logo badge
        base = min(w,h)
        size = int(base*0.14)
        font = QFont("Arial", size, QFont.DemiBold)
        p.setFont(font)
        cx,cy = w/2, h/2
        hue = int((t*14 + 240*mid) % 360)
        p.setPen(QPen(QColor.fromHsv(hue, 180, 255, 255), 6))
        p.drawText(QRectF(0, cy-size*0.6, w, size*1.2), Qt.AlignCenter, SIGN_TEXT)

        # arc equalizer
        bins = 24
        vals = _sample_bands(bands, bins)
        if len(_eq)!=bins: _eq=[0.0]*bins
        for i,v in enumerate(vals):
            targ = 2.1*v + 1.2*fx
            _eq[i] = _env_step(_eq[i], targ, up=0.6, down=0.45)

        R = base*0.36
        for i,e in enumerate(_eq):
            lv = max(0.0, min(1.0, e))
            a0 = _spin + (i/bins)*2*pi
            a1 = a0 + (2*pi/bins)*0.75
            r0 = R
            r1 = R + base*(0.05 + 0.18*lv)
            huei = (hue + int(i*360/bins)) % 360
            col = QColor.fromHsv(huei, 200, 255, 220)
            p.setPen(QPen(col, 5))
            p.drawLine(QPointF(cx + r0*cos(a0), cy + r0*sin(a0)),
                       QPointF(cx + r1*cos(a0), cy + r1*sin(a0)))
            p.drawLine(QPointF(cx + r0*cos(a1), cy + r0*sin(a1)),
                       QPointF(cx + r1*cos(a1), cy + r1*sin(a1)))
