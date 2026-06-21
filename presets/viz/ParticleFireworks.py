import math, random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_last_t=None
_env_lo=0.0
_env_mid=0.0
_env_hi=0.0
_prev_flux_spec=[]
_particles=[]
_prev_lo=0.0
_last_pop_t=-999.0

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

def _flux(bands):
    global _prev_flux_spec
    if not bands:
        _prev_flux_spec=[]
        return 0.0
    if (not _prev_flux_spec) or (len(_prev_flux_spec)!=len(bands)):
        _prev_flux_spec=[0.0]*len(bands)
    val=0.0
    L=max(1,len(bands)-1)
    for i,(x,px) in enumerate(zip(bands,_prev_flux_spec)):
        d=x-px
        if d>0:
            val+=d*(0.35+0.65*(i/float(L)))
    _prev_flux_spec=[0.8*px+0.2*x for x,px in zip(bands,_prev_flux_spec)]
    return val/max(1,len(bands))

def _env_step(env,target,up=0.5,down=0.25):
    if target>env:
        return (1-up)*env+up*target
    else:
        return (1-down)*env+down*target

def _spawn_burst(cx,cy,strength):
    N=int(30+strength*50)
    for _ in range(N):
        ang=random.random()*math.tau
        spd=(0.3+random.random()*0.7)* (200+300*strength)
        vx=math.cos(ang)*spd
        vy=math.sin(ang)*spd
        hue=random.randint(0,359)
        life=0.5+random.random()*0.5
        _particles.append({
            "x":cx,"y":cy,
            "vx":vx,"vy":vy,
            "life":life,
            "maxlife":life,
            "hue":hue
        })

@register_visualizer
class ParticleFireworks(BaseVisualizer):
    display_name="Particle Fireworks"

    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_env_lo,_env_mid,_env_hi,_prev_lo,_last_pop_t

        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:
            return
        cx,cy=w*0.5,h*0.5

        if _last_t is None:
            _last_t=t
        dt=max(0.0,min(0.05,t-_last_t))
        _last_t=t

        lo,mid,hi=_split(bands)
        fx=_flux(bands)

        _env_lo=_env_step(_env_lo, lo+0.6*rms, up=0.5,down=0.25)
        _env_mid=_env_step(_env_mid, mid+0.4*fx, up=0.45,down=0.22)
        _env_hi=_env_step(_env_hi, hi+1.2*fx, up=0.6,down=0.3)

        pop=((lo-_prev_lo)>0.22 or fx>0.12) and (t-_last_pop_t)>0.25
        _prev_lo=lo
        if pop:
            _last_pop_t=t
            _spawn_burst(cx,cy, strength=min(1.0, _env_hi+_env_lo))

        # bright haze background
        haze_col=QColor.fromHsv(int((200+160*_env_hi)%360),255,255,255)
        bg_grad=QRadialGradient(QPointF(cx,cy), max(w,h)*0.9)
        bg_grad.setColorAt(0.0,haze_col)
        bg_grad.setColorAt(1.0,QColor(0,0,0,255))
        p.fillRect(r,QBrush(bg_grad))

        # update particles
        alive=[]
        for part in _particles:
            part["x"]+=part["vx"]*dt
            part["y"]+=part["vy"]*dt
            part["vx"]*=0.92
            part["vy"]*=0.92
            part["vy"]+=80*dt
            part["life"]-=dt
            if part["life"]<=0:
                continue
            alive.append(part)
        _particles[:] = alive[-300:]

        # draw particles
        p.setRenderHint(QPainter.Antialiasing,True)
        for part in _particles:
            k=part["life"]/part["maxlife"]
            size=4+10*k
            alpha=int(255*k)
            col=QColor.fromHsv(part["hue"],255,255,alpha)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(part["x"],part["y"]), size, size)

        # center pulse
        core_r=min(w,h)*0.15*(1.0+0.5*_env_lo)
        core_col=QColor.fromHsv(int((40+200*_env_hi)%360),150,255,220)
        p.setBrush(QBrush(core_col))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx,cy),core_r,core_r)

        # edge vignette
        vign_r = max(w,h)*0.9
        vign = QRadialGradient(QPointF(cx,cy), vign_r)
        vign.setColorAt(0.0, QColor(0,0,0,0))
        vign.setColorAt(1.0, QColor(0,0,0,120))
        p.setBrush(QBrush(vign))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(r.left(), r.top(), w, h))
