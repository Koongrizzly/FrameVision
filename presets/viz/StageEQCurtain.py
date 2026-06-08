# StageEQCurtain.py — stage vibe: logo as headline, thick floor EQ
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

def _env_step(env, target, up=0.32, down=0.16):
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

_eq = [0.0]*12
_warm = 0.0

@register_visualizer
class StageEQCurtain(BaseVisualizer):
    display_name = "Stage EQ Curtain"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _eq, _warm
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx=_flux(bands)
        _warm = _env_step(_warm, 0.8*mid + 0.9*fx, up=0.6, down=0.2)

        # background curtains
        p.fillRect(r, QBrush(QColor(10,10,14)))
        stripes = 20
        for i in range(stripes):
            x0 = int(i*w/stripes)
            x1 = int((i+1)*w/stripes)
            val = 40 + int(30*sin(i*0.6 + t*0.7))
            p.fillRect(QRectF(x0,0,x1-x0,h), QBrush(QColor(20,20,30,val)))

        # headline logo
        base = min(w,h)
        size = int(base*0.12)
        font = QFont("Arial", size, QFont.Black)
        p.setFont(font)
        hue = int((t*16 + 200*_warm) % 360)
        p.setPen(QPen(QColor.fromHsv(hue, 200, 255, 255), 6))
        p.drawText(QRectF(0, h*0.22 - size*0.4, w, size*1.2), Qt.AlignCenter, SIGN_TEXT)

        # 24-band floor EQ
        bins = 24
        vals = _sample_bands(bands, bins)
        if len(_eq)!=bins: _eq=[0.0]*bins
        for i,v in enumerate(vals):
            targ = 1.8*v + 1.0*fx
            _eq[i] = _env_step(_eq[i], targ, up=0.35, down=0.12)

        left = int(w*0.12); right=int(w*0.88)
        base_y = int(h*0.78)
        bw = max(8, int((right-left)/(bins*1.1)))
        gap = max(3, int(bw*0.1))
        x = left
        hue0 = int((t*18 + 160*mid) % 360)
        for i in range(bins):
            e = max(0.0, min(1.0, _eq[i]))
            bh = max(6, int(e*h*0.28))
            col = QColor.fromHsv((hue0 + int(i*360/bins)) % 360, 220, 255, 230)
            p.setBrush(QBrush(col))
            p.setPen(QPen(QColor(25,25,35,180), 2))
            p.drawRect(QRectF(x, base_y - bh, bw, bh))
            # top glow
            p.setPen(QPen(QColor(255,255,255,120), 2))
            p.drawLine(QPointF(x, base_y - bh), QPointF(x+bw, base_y - bh))
            x += bw + gap
