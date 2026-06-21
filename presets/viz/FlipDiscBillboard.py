# Auto-generated music visualizer
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
    for i,(x,px) in enumerate(zip(bands,_prev_spec)):
        d = x - px
        if d>0: f += d * (0.35 + 0.65*(i/max(1,len(bands)-1)))
    _prev_spec = [0.82*px + 0.18*x for x,px in zip(bands,_prev_spec)]
    return f / max(1,len(bands))

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env, target, up=0.34, down=0.14):
    return (1-up)*env + up*target if target>env else (1-down)*env + down*target

_grid = []
_env_lo = _env_mid = _env_hi = 0.0
def _ensure_grid(w,h,cell=18):
    cols = max(8, int(w/cell))
    rows = max(6, int(h/cell))
    global _grid
    if not _grid or len(_grid)!=rows or len(_grid[0])!=cols:
        _grid = [[0.0]*cols for _ in range(rows)]
    return cols, rows, cell

@register_visualizer
class FlipDiscBillboard(BaseVisualizer):
    display_name = "Flip-Disc Billboard (Orchestra)"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _env_lo, _env_mid, _env_hi, _grid
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t-_last_t)); _last_t=t

        lo,mid,hi = _split(bands); fx=_flux(bands)
        _env_lo = _env_step(_env_lo, 1.2*lo + 0.2*rms)
        _env_mid = _env_step(_env_mid, 1.1*mid + 0.3*fx, up=0.45, down=0.12)
        _env_hi = _env_step(_env_hi, 0.7*hi + 1.4*fx, up=0.55, down=0.16)

        p.fillRect(r, QBrush(QColor(10,12,16)))
        cols, rows, cell = _ensure_grid(w,h,cell= max(12, int(min(w,h)/36)) )
        margin_x = (w - cols*cell)/2
        margin_y = (h - rows*cell)/2

        # Flip logic: lows create large-area flips; highs speckle flips
        for j in range(rows):
            for i in range(cols):
                v = _grid[j][i]
                # target brightness from mids pattern (sine sweep) + lows block sweep
                phase = (i*0.12 + j*0.07 + t*0.8*(_env_mid+0.01))
                block = 1.0 if (int((t*0.4 + j*0.12))%4==0 and _env_lo>0.05) else 0.0
                tgt = 0.5*(0.5+0.5*sin(phase*2*pi)) + 0.5*block
                # highs add sudden flips
                if random() < 0.08*min(1.0, _env_hi*3.0): tgt = 1.0 - v
                # settle quickly
                v = 0.65*v + 0.35*tgt
                _grid[j][i] = v

                # draw disc
                x = margin_x + i*cell + cell/2
                y = margin_y + j*cell + cell/2
                rad = cell*0.48
                col = QColor.fromHsv(int((120+ 120*v + 50*_env_mid)%360), 120, int(110+140*v), 255)
                p.setBrush(QBrush(col))
                p.setPen(QPen(QColor(20,30,40), 1))
                p.drawEllipse(QPointF(x,y), rad, rad)
