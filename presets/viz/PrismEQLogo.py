# PrismEQLogo.py — glossy prism bars + mirrored reflection, centered logo
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

_eq = [0.0]*16
_warm = 0.0

@register_visualizer
class PrismEQLogo(BaseVisualizer):
    display_name = "Prism EQ Logo"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _eq, _warm
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx=_flux(bands)
        _warm = _env_step(_warm, 0.9*mid + 0.8*fx, up=0.6, down=0.22)

        # background
        p.fillRect(r, QBrush(QColor(8,10,14)))
        # subtle vignette
        p.setPen(QPen(QColor(16,18,22,160), 8))
        p.drawRect(r)

        # logo
        base = min(w,h)
        size = int(base*0.12)
        font = QFont("Arial", size, QFont.Bold)
        p.setFont(font)
        fm = QFontMetrics(font)
        text_w = fm.horizontalAdvance(SIGN_TEXT)
        text_h = fm.height()
        x_text = (w - text_w)//2
        y_text = int(h*0.38)
        p.setPen(QPen(QColor.fromHsv(int((t*20+260*_warm)%360), 200, 255, 255), 6))
        p.drawText(QPointF(x_text, y_text), SIGN_TEXT)

        # equalizer under text
        bins = 16
        vals = _sample_bands(bands, bins)
        # update smoothing
        if len(_eq)!=bins: _eq=[0.0]*bins
        for i,v in enumerate(vals):
            targ = 1.8*v + 1.0*fx + 0.1*rms
            _eq[i] = _env_step(_eq[i], targ, up=0.6, down=0.24)

        left = int((w - min(w*0.7, text_w*1.1))//2)
        right = w - left
        base_y = y_text + int(text_h*0.6)
        bw = max(4, int((right-left)/(bins*1.25)))
        gap = max(2, int(bw*0.25))
        x = left
        hue0 = int((t*18 + 180*mid) % 360)
        for i in range(bins):
            e = max(0.0, min(1.0, _eq[i]))
            bh = max(4, int(e * h * 0.22))
            hue = (hue0 + int(i*360/bins)) % 360
            # main bar (prism look: light top, darker base)
            p.setBrush(QBrush(QColor.fromHsv(hue, 160, 255, 220)))
            p.setPen(QPen(QColor(30,40,60,180), 1))
            p.drawRect(QRectF(x, base_y - bh, bw, bh))
            # glossy top edge
            p.setPen(QPen(QColor(255,255,255,120), 2))
            p.drawLine(QPointF(x, base_y - bh), QPointF(x+bw, base_y - bh))
            # reflection
            rb = int(bh*0.45)
            p.setBrush(QBrush(QColor.fromHsv(hue, 120, 230, 70)))
            p.setPen(QPen(QColor(20,20,30,60), 1))
            p.drawRect(QRectF(x, base_y+4, bw, rb))
            x += bw + gap
