import math
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QRadialGradient
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_last_t=None
_env_lo=0.0
_env_mid=0.0
_env_hi=0.0
_prev_lo=0.0
_last_bump_t=-999.0

def _split(bands):
    if not bands:
        return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6)
    b=max(a+1,n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env,target,up=0.4,down=0.2):
    if target>env:
        return (1-up)*env+up*target
    else:
        return (1-down)*env+down*target

def _bass_bump(lo,t):
    global _prev_lo,_last_bump_t
    bump=(lo-_prev_lo)>0.22 and (t-_last_bump_t)>0.25
    _prev_lo=lo
    if bump:
        _last_bump_t=t
    return bump

@register_visualizer
class VectorGridDistort(BaseVisualizer):
    display_name="Vector Grid Distort"

    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_env_lo,_env_mid,_env_hi
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:
            return

        if _last_t is None:
            _last_t=t
        dt=max(0.0,min(0.05,t-_last_t))
        _last_t=t

        lo,mid,hi=_split(bands)
        _env_lo=_env_step(_env_lo, lo+0.4*rms, up=0.45,down=0.22)
        _env_mid=_env_step(_env_mid, mid+0.4*rms, up=0.4,down=0.2)
        _env_hi=_env_step(_env_hi, hi+0.8*rms, up=0.6,down=0.3)

        # bright synthwave-ish gradient sky
        sky_grad=QLinearGradient(r.left(), r.top(), r.left(), r.bottom())
        sky_grad.setColorAt(0.0, QColor.fromHsv(int((260+200*_env_hi)%360),255,255,255))
        sky_grad.setColorAt(0.4, QColor(40,10,60,255))
        sky_grad.setColorAt(1.0, QColor(0,0,0,255))
        p.fillRect(r,QBrush(sky_grad))

        p.setRenderHint(QPainter.Antialiasing,False)

        horizon_y = h*0.45
        bend_amt = 20.0*_env_lo
        wobble = 8.0*_env_mid*math.sin(t*3.0)

        line_col = QColor.fromHsv(int((160+120*_env_hi)%360),255,255,220)
        grid_pen = QPen(line_col,1)
        grid_pen.setCosmetic(True)
        p.setPen(grid_pen)

        rows=14
        for i in range(1,rows+1):
            k = i/float(rows+1)
            y = horizon_y + k*(h-horizon_y)
            y += bend_amt * (1.0-k) + wobble*(1.0-k)
            x0 = 0
            x1 = w*0.5
            x2 = w
            curve = 15.0*_env_lo*math.sin(k*4.0+t*2.0)
            p.drawLine(QPointF(x0,y+curve), QPointF(x1,y-curve*0.5))
            p.drawLine(QPointF(x1,y-curve*0.5), QPointF(x2,y+curve))

        cols=20
        for j in range(cols+1):
            k = (j/float(cols))-0.5
            xh = w*0.5 + k*w*0.7
            yh = horizon_y + wobble*0.1
            xb = w*0.5 + k*w*(1.2+0.5*_env_lo)
            yb = h
            p.drawLine(QPointF(xh,yh-bend_amt*0.5),
                       QPointF(xb,yb))

        if _bass_bump(lo,t):
            glow_pen = QPen(QColor(255,255,255,255),3)
            glow_pen.setCosmetic(True)
            p.setPen(glow_pen)
            rad=w*0.3+80*_env_lo
            p.drawArc(QRectF(w*0.5-rad, horizon_y-rad*0.4, rad*2, rad*0.8),
                      0, 16*360)

        # radial vignette to darken edges only
        cx,cy=w*0.5,h*0.5
        vign_r = max(w,h)*0.9
        vign = QRadialGradient(QPointF(cx,cy), vign_r)
        vign.setColorAt(0.0, QColor(0,0,0,0))
        vign.setColorAt(1.0, QColor(0,0,0,120))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(vign))
        p.drawRect(QRectF(r.left(), r.top(), w, h))
