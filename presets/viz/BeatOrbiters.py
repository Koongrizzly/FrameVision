# BeatOrbiters.py — orbs orbit & radiate on each transient
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

_orbs = []  # {ang, r, speed, life}
@register_visualizer
class BeatOrbiters(BaseVisualizer):
    display_name = "Beat Orbiters"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _orbs
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t - _last_t)); _last_t = t

        lo,mid,hi = _split(bands); fx=_flux(bands)
        onset = 0.65*hi + 1.5*fx + 0.2*rms
        # spawn new orbs on onset (snappy)
        if onset > 0.18:
            for k in range(6):
                _orbs.append({
                    'ang': random()*2*pi,
                    'r': min(w,h)* (0.12 + 0.28*random()),
                    'speed': (0.5 + 2.5*mid + 0.6*lo) * (0.5 + random()*1.2),
                    'life': 0.8 + 0.6*random()
                })
        # update
        alive=[]
        for o in _orbs:
            o['ang'] += o['speed']*dt
            o['life'] -= 0.5*dt
            if o['life']>0: alive.append(o)
        _orbs = alive

        p.fillRect(r, QBrush(QColor(8,9,12)))
        cx,cy = w/2, h/2
        base_hue = int((t*25 + 220*mid) % 360)
        # draw faint nucleus
        p.setBrush(QBrush(QColor.fromHsv((base_hue+40)%360, 80, 140, 80)))
        p.setPen(QPen(QColor(0,0,0,0), 0))
        p.drawEllipse(QPointF(cx,cy), min(w,h)*0.06, min(w,h)*0.06)
        # orbs
        for o in _orbs:
            x = cx + o['r']*cos(o['ang'])
            y = cy + o['r']*sin(o['ang'])
            a = int(200 * min(1.0, o['life']))
            hue = (base_hue + int(180*(1.0-o['life']))) % 360
            p.setBrush(QBrush(QColor.fromHsv(hue, 180, 255, a)))
            p.setPen(QPen(QColor(0,0,0,0), 0))
            p.drawEllipse(QPointF(x,y), 6+4*mid, 6+4*mid)
            # radial line
            p.setPen(QPen(QColor.fromHsv(hue, 40, 255, int(a*0.7)), 2))
            p.drawLine(QPointF(cx,cy), QPointF(x,y))
