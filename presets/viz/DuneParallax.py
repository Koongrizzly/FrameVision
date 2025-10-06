# DuneParallax.py — layered dunes that swell with bass and drift with groove
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

t_offsets = [0.0, 0.0, 0.0, 0.0]
@register_visualizer
class DuneParallax(BaseVisualizer):
    display_name = "Dune Parallax"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, t_offsets
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t - _last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx=_flux(bands)
        amp = 14 + 120*min(1.0, lo*1.8)
        speed = [10+40*mid, 18+60*mid, 28+80*mid, 36+100*mid]
        for i in range(4):
            t_offsets[i] += speed[i]*dt

        p.fillRect(r, QBrush(QColor(8,10,14)))
        base_hue = int((t*10 + 40*mid) % 360)
        layers = 4
        for L in range(layers):
            y_base = h*(0.45 + 0.12*L)
            poly = QPolygonF()
            for x in range(0, w+8, 8):
                y = y_base + amp*(0.5+0.5*(L/layers))*sin( (x*0.01 + t_offsets[L]*0.02 + L) )
                poly.append(QPointF(x, y))
            poly.append(QPointF(w,h)); poly.append(QPointF(0,h))
            hue = (base_hue + 12*L) % 360
            p.setBrush(QBrush(QColor.fromHsv(hue, 140, 180 + 15*L, 220)))
            p.setPen(QPen(QColor(0,0,0,100), 1))
            p.drawPolygon(poly)
        # star glints on highs
        if (0.7*hi + 1.4*fx) > 0.18:
            p.setPen(QPen(QColor(255,255,255,180), 2))
            for i in range(20):
                x = randint(0,w); y = randint(0, int(h*0.4))
                p.drawPoint(QPointF(x,y))
