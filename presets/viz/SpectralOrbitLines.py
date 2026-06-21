import math, random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_last_t=None
_env_lo=0.0
_env_mid=0.0
_env_hi=0.0

_trails = {
    "lo": [],
    "mid": [],
    "hi": [],
}

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

@register_visualizer
class SpectralOrbitLines(BaseVisualizer):
    display_name="Spectral Orbit Lines"

    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_env_lo,_env_mid,_env_hi,_trails
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0:
            return
        cx,cy=w*0.5,h*0.5

        if _last_t is None:
            _last_t=t
        dt=max(0.0,min(0.05,t-_last_t))
        _last_t=t

        lo,mid,hi=_split(bands)
        _env_lo=_env_step(_env_lo, lo+0.4*rms,up=0.45,down=0.22)
        _env_mid=_env_step(_env_mid, mid+0.4*rms,up=0.4,down=0.2)
        _env_hi=_env_step(_env_hi, hi+0.8*rms,up=0.6,down=0.3)

        # brighter background using radial gradient
        bg_grad=QRadialGradient(QPointF(cx,cy), max(w,h)*0.9)
        bg_grad.setColorAt(0.0, QColor.fromHsv(int((300+120*_env_hi)%360),255,255,255))
        bg_grad.setColorAt(1.0, QColor(10,10,20,255))
        p.fillRect(r,QBrush(bg_grad))

        p.setRenderHint(QPainter.Antialiasing,True)

        rad_lo=min(w,h)*0.35*(1.0+0.2*_env_lo)
        rad_mid=min(w,h)*0.27*(1.0+0.2*_env_mid)
        rad_hi=min(w,h)*0.18*(1.0+0.2*_env_hi)

        ang_lo=t*0.6
        ang_mid=t*1.2
        ang_hi=t*2.0

        x_lo = cx+math.cos(ang_lo)*rad_lo
        y_lo = cy+math.sin(ang_lo*1.2)*rad_lo*0.6
        x_mid= cx+math.cos(ang_mid*1.1)*rad_mid
        y_mid= cy+math.sin(ang_mid*0.9)*rad_mid*0.8
        x_hi = cx+math.cos(ang_hi*1.4)*rad_hi
        y_hi = cy+math.sin(ang_hi*1.7)*rad_hi*0.9

        _trails["lo"].append((x_lo,y_lo))
        _trails["mid"].append((x_mid,y_mid))
        _trails["hi"].append((x_hi,y_hi))

        max_len=80
        for k in _trails:
            if len(_trails[k])>max_len:
                _trails[k]=_trails[k][-max_len:]

        def draw_trail(points, base_hue, alpha_scale, width_scale):
            if len(points)<2:
                return
            for i in range(1,len(points)):
                x0,y0=points[i-1]
                x1,y1=points[i]
                k=i/float(len(points))
                hue=(base_hue+80*k)%360
                raw_alpha=alpha_scale*(k**1.5)
                alpha=max(0, min(255, int(raw_alpha)))
                col=QColor.fromHsv(int(hue),255,255,alpha)
                pen=QPen(col, 1+width_scale*k)
                pen.setCapStyle(Qt.RoundCap)
                pen.setCosmetic(True)
                p.setPen(pen)
                p.drawLine(QPointF(x0,y0),QPointF(x1,y1))

        draw_trail(_trails["lo"], base_hue=120, alpha_scale=min(255.0, 255*(0.5+_env_lo)), width_scale=4)
        draw_trail(_trails["mid"], base_hue=40,  alpha_scale=min(255.0, 255*(0.5+_env_mid)), width_scale=3)
        draw_trail(_trails["hi"], base_hue=300,  alpha_scale=min(255.0, 255*(0.5+_env_hi)), width_scale=2)

        center_r=min(w,h)*0.05*(1.0+0.5*(_env_lo+_env_mid+_env_hi)/3.0)
        center_col=QColor.fromHsv(int((60+200*_env_mid)%360),255,255,255)
        p.setBrush(QBrush(center_col))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx,cy),center_r,center_r)

        # very light edge vignette
        vign_r = max(w,h)*0.95
        vign = QRadialGradient(QPointF(cx,cy), vign_r)
        vign.setColorAt(0.0, QColor(0,0,0,0))
        vign.setColorAt(1.0, QColor(0,0,0,80))
        p.setBrush(QBrush(vign))
        p.setPen(Qt.NoPen)
        p.drawRect(QRectF(r.left(), r.top(), w, h))
