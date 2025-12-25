import math, random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_rings = []
_last_t = None
_env_lo = 0.0
_env_mid = 0.0
_env_hi = 0.0
_prev_lo = 0.0
_last_hit_t = -999.0

def _split(bands):
    if not bands:
        return 0.0, 0.0, 0.0
    n=len(bands)
    a=max(1,n//6)
    b=max(a+1,n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env, target, up=0.4, down=0.2):
    if target>env:
        return (1-up)*env+up*target
    else:
        return (1-down)*env+down*target

def _kick(lo,t):
    global _prev_lo,_last_hit_t
    hit=(lo-_prev_lo)>0.2 and (t-_last_hit_t)>0.2
    _prev_lo=lo
    if hit:
        _last_hit_t=t
    return hit

def _ensure_rings():
    if len(_rings)<12:
        for _ in range(12-len(_rings)):
            _rings.append({"z": random.uniform(1.0,4.0)})
    _rings.sort(key=lambda r:r["z"], reverse=True)

@register_visualizer
class NeonTunnelWarp(BaseVisualizer):
    display_name = "Neon Tunnel Warp"

    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_env_lo,_env_mid,_env_hi

        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:
            return
        cx,cy=w*0.5,h*0.5

        if _last_t is None:
            _last_t=t
        dt=max(0.0,min(0.05,t-_last_t))
        _last_t=t

        lo,mid,hi=_split(bands)
        _env_lo=_env_step(_env_lo, lo+0.4*rms, up=0.5, down=0.25)
        _env_mid=_env_step(_env_mid, mid+0.2*rms, up=0.4, down=0.20)
        _env_hi=_env_step(_env_hi, hi+0.6*rms, up=0.6, down=0.30)

        # bright core gradient background
        core_col=QColor.fromHsv(int((200+150*_env_hi)%360),220,255,255)
        grad=QRadialGradient(QPointF(cx,cy), max(w,h)*0.7)
        grad.setColorAt(0.0, core_col)
        grad.setColorAt(1.0, QColor(0,0,20,255))
        p.fillRect(r,QBrush(grad))

        speed = 2.0 + 4.0*_env_mid

        _ensure_rings()
        new_rings=[]
        for ring in _rings:
            ring["z"]-=speed*dt
            if ring["z"]<0.3:
                ring["z"]=random.uniform(3.0,4.0)
            new_rings.append(ring)
        _rings[:]=new_rings
        _rings.sort(key=lambda rr: rr["z"], reverse=True)

        # glow tunnel rings
        for ring in _rings:
            z=ring["z"]
            scale=1.0/(z+0.2)
            radius=min(w,h)*0.45*scale*(1.0+0.3*_env_lo)
            thickness=2+4*scale+4*_env_lo
            hue=(t*40+_env_hi*200)%360
            alpha=int(200*scale+80*_env_hi)
            col=QColor.fromHsv(int(hue)%360,255,255,max(0,min(255,alpha)))
            pen=QPen(col, thickness)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(cx,cy), radius, radius*0.6)

        # star streaks
        p.setRenderHint(QPainter.Antialiasing,False)
        streaks=30
        streak_alpha=int(120+120*_env_hi)
        for i in range(streaks):
            ang= (i/float(streaks))*math.tau + t*0.3
            inner_r=min(w,h)*0.05
            outer_r=min(w,h)*0.5*(1.0+0.2*_env_lo)
            x0=cx+math.cos(ang)*inner_r
            y0=cy+math.sin(ang)*inner_r
            x1=cx+math.cos(ang)*outer_r
            y1=cy+math.sin(ang)*outer_r
            p.setPen(QPen(QColor(255,255,255,streak_alpha),1))
            p.drawLine(QPointF(x0,y0),QPointF(x1,y1))

        # soft radial vignette instead of dark full overlay
        vign_r = max(w,h)*0.8
        vign = QRadialGradient(QPointF(cx,cy), vign_r)
        vign.setColorAt(0.0, QColor(0,0,0,0))
        vign.setColorAt(1.0, QColor(0,0,0,120))
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(vign))
        p.drawRect(QRectF(r.left(), r.top(), w, h))
