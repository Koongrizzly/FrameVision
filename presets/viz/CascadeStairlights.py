# CascadeStairlights.py — diagonal stair lights that accelerate on hits
# Shared helpers
from math import sin, cos, pi, sqrt
from random import random, randint, choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF, QFont
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_prev_spec = []
_last_t = None

def _flux(bands):
    global _prev_spec
    if not bands:
        _prev_spec = []
        return 0.0
    if not _prev_spec or len(_prev_spec)!=len(bands):
        _prev_spec = [0.0]*len(bands)
    f = 0.0
    n=len(bands)
    for i,(x,px) in enumerate(zip(bands,_prev_spec)):
        d = x - px
        if d>0:
            f += d * (0.35 + 0.65*(i/max(1,n-1)))
    _prev_spec = [0.82*px + 0.18*x for x,px in zip(bands,_prev_spec)]
    return f/max(1,n)

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env, target, up=0.38, down=0.16):
    return (1-up)*env + up*target if target>env else (1-down)*env + down*target

_offset = 0.0
_speed = 0.0
@register_visualizer
class CascadeStairlights(BaseVisualizer):
    display_name = "Cascade Stairlights"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _offset, _speed
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t - _last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx=_flux(bands)
        onset = 0.6*hi + 1.4*fx + 0.2*rms
        # base speed from mids, kick it on onset
        _speed = 0.85*_speed + 0.15*(60 + 280*mid)
        if onset>0.18:
            _speed += 120
        _offset = (_offset + _speed*dt) % (w*0.2 + h*0.2)

        p.fillRect(r, QBrush(QColor(7,8,12)))
        step = max(16, int(min(w,h)/24))
        hue0 = int((t*18 + 120*mid) % 360)
        for k in range(-h, w+step, step):
            # diagonal from left-bottom to right-top
            x0 = k + _offset*0.5
            y0 = h
            x1 = k + h
            y1 = 0
            val = int(120 + 120*abs(sin((k+_offset)*0.02)))
            hue = (hue0 + int(k*0.2)) % 360
            p.setPen(QPen(QColor.fromHsv(hue, 180, val, 180), 4))
            p.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        # add rows of bouncing dots on lows
        dots = 20
        for i in range(dots):
            x = i/(dots-1)*w
            y = h*0.8 - 40*sin(2*pi*(i*0.08 + t*(0.5+lo)))
            p.setPen(QPen(QColor(255,255,255,120), 3))
            p.drawPoint(QPointF(x,y))
