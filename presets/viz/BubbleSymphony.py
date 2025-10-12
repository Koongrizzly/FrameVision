
# --- Bubble Symphony (per-band pops + motion) ---
TUNE_KICK = 1.2
TUNE_BOOM = 1.0
SPR_K = 30.0
SPR_C = 6.0
SPR_MAX = 4.2

from math import sin, cos, pi
from random import Random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(7777)
_prev = []
_env=_gate=_punch=0.0
_pt=None
_beat_count=0
_last_t = None

def _low_avg(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    return sum(bands[:cut])/cut
def _mid_avg(bands):
    if not bands: return 0.0
    n=len(bands); a=max(1,n//6); b=max(a+1, n*3//6)
    return sum(bands[a:b]) / max(1,(b-a))
def _high_avg(bands):
    if not bands: return 0.0
    n=len(bands); b=max(1, n*3//6)
    return sum(bands[b:]) / max(1, n-b)

def _flux(bands):
    global _prev
    if not bands:
        _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n:
        _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut, n):
        d=bands[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt,_beat_count
    midhi = (_mid_avg(bands)+_high_avg(bands))*0.5
    f = _flux(bands); lo = _low_avg(bands)
    target = 0.58*midhi + 1.30*f + 0.18*rms + 0.22*lo*TUNE_BOOM
    target = target/(1+0.7*target)
    if target>_env: _env = 0.72*_env + 0.28*target
    else: _env = 0.92*_env + 0.08*target
    hi,lo_thr=0.30,0.18
    g = 1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    if g>0.6 and _gate<=0.6: _beat_count += 1
    _gate = 0.82*_gate + 0.18*g
    boom = min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    # fast-decay punch
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t-_pt)); _pt = t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return _env, _gate, boom, _punch, _beat_count

_sdict={}
def spring_to(key, target, t, k=SPR_K, c=SPR_C, lo=0.25, hi=SPR_MAX):
    s, v, pt = _sdict.get(key, (1.0, 0.0, None))
    if pt is None:
        _sdict[key]=(s,v,t); return 1.0
    dt=max(0.0, min(0.033, t-pt))
    a=-k*(s-target) - c*v
    v+=a*dt; s+=v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key]=(s,v,t); return s

# bubble state
_bubbles=[]   # [x,y,r,layer,phase]
_ripples=[]   # [x,y,rad,alpha,hue]

def _ensure_init(w,h):
    global _bubbles
    if _bubbles: return
    # 3 layers: 0=low,1=mid,2=high
    for layer in range(3):
        for i in range(24):  # 72 total
            x=_rng.random()*w; y=_rng.random()*h
            r=8+18*_rng.random()
            _bubbles.append([x,y,r,layer,_rng.random()*2*pi])

@register_visualizer
class BubbleSymphony(BaseVisualizer):
    display_name = "Bubble Symphony"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _ripples
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        _ensure_init(w,h)
        p.fillRect(r, QBrush(QColor(6,6,10)))

        env, gate, boom, punch, bc = beat_drive(bands, rms, t)
        lowE, midE, highE = _low_avg(bands), _mid_avg(bands), _high_avg(bands)
        # choose a layer to emphasize on onset
        if gate>0.6:
            layer = max([(0,lowE),(1,midE),(2,highE)], key=lambda kv: kv[1])[0]
            # pop 6-10 bubbles in that layer
            idxs=[i for i,b in enumerate(_bubbles) if b[3]==layer]
            _rng.shuffle(idxs)
            for i in idxs[:8]:
                x,y,r,layer,ph = _bubbles[i]
                _ripples.append([x,y, r, 1.0, int((t*90 + layer*60)%360)])
                # respawn bubble at bottom
                _bubbles[i]=[_rng.random()*w, h+10+_rng.random()*40, 8+18*_rng.random(), layer, ph]

        # spontaneous pops when a band's energy is high
        if lowE>0.22 and _rng.random()<0.3:
            i=_rng.choice([j for j,b in enumerate(_bubbles) if b[3]==0])
            x,y,r,layer,ph=_bubbles[i]; _ripples.append([x,y,r,1.0,200])
            _bubbles[i]=[_rng.random()*w, h+10, 8+18*_rng.random(), layer, ph]
        if midE>0.22 and _rng.random()<0.3:
            i=_rng.choice([j for j,b in enumerate(_bubbles) if b[3]==1])
            x,y,r,layer,ph=_bubbles[i]; _ripples.append([x,y,r,1.0,120])
            _bubbles[i]=[_rng.random()*w, h+10, 8+18*_rng.random(), layer, ph]
        if highE>0.22 and _rng.random()<0.3:
            i=_rng.choice([j for j,b in enumerate(_bubbles) if b[3]==2])
            x,y,r,layer,ph=_bubbles[i]; _ripples.append([x,y,r,1.0,40])
            _bubbles[i]=[_rng.random()*w, h+10, 8+18*_rng.random(), layer, ph]

        # update/draw bubbles
        dt=0.016
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        for i in range(len(_bubbles)):
            x,y,r,layer,ph = _bubbles[i]
            band_amp = [lowE, midE, highE][layer]
            # upward float + horizontal drift; faster when their band is hot
            y -= (8 + 90*band_amp + 30*env)*dt
            x += 30*sin(ph + t*0.6 + layer)*dt
            # size wobble
            r2 = r*(1.0 + 0.4*band_amp + 0.4*punch + 0.2*sin(t*2+ph))
            hue = int((t*60 + layer*90)%360)
            col = QColor.fromHsv(hue, 230, 255, 180)
            p.setPen(QPen(col,2))
            p.setBrush(QBrush(QColor(col.red(),col.green(),col.blue(),60)))
            p.drawEllipse(QPointF(x%w, y%h), r2, r2)
            _bubbles[i]=[x%w,y%h,r,layer,ph]

        # ripples
        nxt=[]
        for x,y,rad,a,hue in _ripples:
            col=QColor.fromHsv(hue,230,255, int(255*a))
            p.setPen(QPen(col,3)); p.setBrush(QBrush(QColor(0,0,0,0)))
            p.drawEllipse(QPointF(x,y), rad*(1.0+(1.5-a)*1.6), rad*(1.0+(1.5-a)*1.6))
            a *= 0.90
            if a>0.12: nxt.append([x,y,rad+12,a,hue])
        _ripples=nxt
