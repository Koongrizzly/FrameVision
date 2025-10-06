# TopDownDropViz.py
# Upside-down colorful visualizer.
# Bars fall from the TOP; sharp peaks spawn falling pixel droplets.
# Snappy to the beat via transient-sensitive envelopes.
from math import sin, pi
from random import Random, random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(7171)
_prev = []
_env = []
_last_t = None
_drops = []   # {x, y, vy, life}

def _flux(bands):
    global _prev
    if not bands:
        _prev=[]
        return [0.0]*0, 0.0
    if not _prev or len(_prev)!=len(bands):
        _prev = [0.0]*len(bands)
    fx_list = []
    fsum = 0.0
    for i,(x,px) in enumerate(zip(bands,_prev)):
        d = x - px
        v = d if d>0 else 0.0
        v *= (0.4 + 0.6*(i/len(bands)))
        fx_list.append(v)
        fsum += v
    _prev = [0.75*px + 0.25*x for x,px in zip(bands,_prev)]
    return fx_list, fsum/max(1,len(bands))

def _ensure_env(n):
    global _env
    if not _env or len(_env)!=n:
        _env = [0.0]*n

def _palette_hsv(i, n, t):
    # rainbow across bins, slowly orbiting
    hue = int(((i/n)*360 + t*12) % 360)
    sat = 200
    val = 240
    return QColor.fromHsv(hue, sat, val, 245)

@register_visualizer
class TopDownDropViz(BaseVisualizer):
    display_name = "Upside-Down Drop Visualizer"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _drops
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        fx_list, fx_avg = _flux(bands)
        n = max(8, len(bands))
        _ensure_env(n)

        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t - _last_t)); _last_t = t

        # background
        p.fillRect(r, QBrush(QColor(6,8,12)))

        # compute per-bin snappy heights (fast rise, quick fall)
        # Clamp bands length to n bins using simple sampling
        for i in range(n):
            src = bands[int(i*len(bands)/n)] if bands else 0.0
            target = 1.6*src + 0.6*fx_list[int(i*len(bands)/n)] if bands else 0.0
            env = _env[i]
            if target > env:
                env = 0.65*env + 0.35*target
            else:
                env = 0.78*env + 0.22*target
            _env[i] = env

        # bar geometry (from top)
        gap = max(1, int(w / n))
        bar_w = max(1, int(gap*0.7))
        left = (w - n*gap)/2

        # spawn drops on sharp local peaks
        for i in range(n):
            e = max(0.0, min(1.0, _env[i]))
            height = int(e * (h*0.85))
            # draw bar from top downward
            x = left + i*gap + (gap-bar_w)/2
            top_y = 0
            # color
            col = _palette_hsv(i, n, t)
            p.setBrush(QBrush(col))
            p.setPen(QPen(QColor(0,0,0,120), 1))
            p.drawRect(QRectF(x, top_y, bar_w, height))

            # spawn a drop when transient is high at this bin
            transient = fx_list[int(i*len(bands)/n)] if bands else 0.0
            if transient > 0.08 + 0.30*e and height>6 and random()<0.9:
                _drops.append({'x': x + bar_w*0.5, 'y': height, 'vy': 120 + 220*transient, 'life': 1.0})

        # update & draw drops
        alive = []
        for d in _drops:
            d['vy'] += 380*dt   # gravity
            d['y']  += d['vy']*dt
            d['life'] -= 0.8*dt
            if d['y'] < h and d['life']>0:
                alive.append(d)
        _drops = alive

        for d in _drops:
            a = int(255*min(1.0, max(0.0, d['life'])))
            p.setPen(QPen(QColor(255,255,255,a), 2))
            p.drawPoint(QPointF(d['x'], d['y']))
