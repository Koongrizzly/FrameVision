
from math import sin
from random import choice, random
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# ---- Tunables ----
TUNE_BOOM = 1.0
SPR_K, SPR_C = 26.0, 8.0

# Silence / pause
SILENCE_RMS_THR, SILENCE_ENV_THR, SILENCE_BOOM_THR, SILENCE_GATE_THR = 0.015, 0.050, 0.120, 0.55
SILENCE_HOLD_SEC, WAKE_EAGER_SEC = 0.60, 0.10
PAUSE_STATIC_SEC, STATIC_FLUX_THR, STATIC_BANDS_EPS, STATIC_RMS_EPS = 0.35, 0.015, 0.002, 0.002

# Blink
BLINK_FLUX_THR   = 0.22
BLINK_BOOM_MAX   = 0.35
BLINK_HOLD       = 0.12
BEAT_BLINK_PER   = 8
IDLE_BLINK_GAP   = 2.0

# Gaze
GAZE_PX_FRAC = 0.12   # fraction of R for side glances
GAZE_BEAT_PER = 2     # saccade every N beats
GAZE_IDLE_SPEED = 0.5 # Hz drift

# Big hit sparkle
SPARK_PUNCH_THR = 0.80
SPARK_HOLD = 0.22

_prev_spec = []
_env = _gate = _punch = 0.0
_prev_frame_bands = None
_prev_rms = 0.0
_static_quiet_time = 0.0
_pt = None
_beat_count = 0

_sdict = {}
_last_active_t = None
_silence_mode = False
_frozen_env = _frozen_punch = _frozen_boom = 0.0

_mouth_state = 3
_mouth_last_t = None
_last_bc_mouth = -1

# blink timers
_blink_release_L = 0.0
_blink_release_R = 0.0
_blink_last_t = 0.0
_last_bc_blink = -1

# gaze target
_gaze_target = 0.0
_last_bc_gaze = -1

# sparkle
_spark_until = 0.0
_spark_x = 0.0
_spark_y = 0.0

def _midhi(b):
    if not b: return 0.0
    n=len(b); cut=max(1,n//6)
    s=c=0.0
    for i in range(cut,n):
        w=0.35+0.65*((i-cut)/max(1,n-cut))
        s+=w*b[i]; c+=1
    return s/max(1,c)

def _low(b):
    if not b: return 0.0
    n=len(b); cut=max(1,n//6)
    s=c=0.0
    for i in range(0,cut):
        w=1.0-0.4*(i/max(1,cut-1))
        s+=w*b[i]; c+=1
    return s/max(1,c)

def _flux(b):
    global _prev_spec
    if not b: _prev_spec=[]; return 0.0
    n=len(b)
    if not _prev_spec or len(_prev_spec)!=n: _prev_spec=[0.0]*n
    cut=max(1,n//6)
    f=c=0.0
    for i in range(cut,n):
        d=b[i]-_prev_spec[i]
        if d>0: f+=d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev_spec=[0.88*_prev_spec[i]+0.12*b[i] for i in range(n)]
    return f/max(1,c)

def beat_drive(bands, rms, t):
    global _env,_gate,_punch,_pt,_beat_count
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    tgt=(0.58*e+1.30*f+0.18*rms+0.22*lo*TUNE_BOOM); tgt=tgt/(1+0.7*tgt)
    _env=(0.72*_env+0.28*tgt) if tgt>_env else (0.92*_env+0.08*tgt)
    hi,lo_thr=0.30,0.18
    g=1.0 if f>hi else (0.0 if f<lo_thr else _gate)
    if g>0.6 and _gate<=0.6: _beat_count+=1
    _gate=0.82*_gate+0.18*g
    boom=min(1.0,max(0.0,lo*1.25+0.42*rms))
    if _pt is None: _pt=t
    dt=max(0.0,min(0.033,t-_pt)); _pt=t
    decay=pow(0.78,dt/0.016) if dt>0 else 0.78
    _punch=max(_punch*decay,1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)),max(0.0,min(1.0,_gate)),boom,max(0.0,min(1.0,_punch)),_beat_count,dt,f

def spring_to(key, target, t, k=SPR_K, c=SPR_C, lo=-1e9, hi=1e9):
    s,v,pt=_sdict.get(key,(target,0.0,t))
    dt=max(0.0,min(0.033,t-pt))
    a=-k*(s-target)-c*v
    v+=a*dt; s+=v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key]=(s,v,t); return s

def activity(bands,rms,t,env,gate,boom,flux_val,dt):
    global _prev_frame_bands,_prev_rms,_static_quiet_time,_last_active_t,_silence_mode
    avg=0.0
    if bands and _prev_frame_bands and len(_prev_frame_bands)==len(bands):
        s=0.0
        for a,b in zip(bands,_prev_frame_bands): s+=abs(a-b)
        avg=s/len(bands)
    if (avg<STATIC_BANDS_EPS) and (abs(rms-_prev_rms)<STATIC_RMS_EPS) and (flux_val<STATIC_FLUX_THR):
        _static_quiet_time+=dt
    else:
        _static_quiet_time=0.0
    paused=_static_quiet_time>PAUSE_STATIC_SEC
    _prev_frame_bands=list(bands) if bands else None; _prev_rms=rms
    active_now=(not paused) and ((rms>SILENCE_RMS_THR) or (env>SILENCE_ENV_THR) or (gate>SILENCE_GATE_THR) or (boom>SILENCE_BOOM_THR))
    if _last_active_t is None: _last_active_t=t
    if active_now: _last_active_t=t
    prev=_silence_mode
    _silence_mode=paused or ((t-_last_active_t)>SILENCE_HOLD_SEC)
    if _silence_mode and active_now and (t-_last_active_t)<=WAKE_EAGER_SEC:
        _silence_mode=False
    return paused, prev

def _switch_mouth(t):
    global _mouth_state,_mouth_last_t
    ns=choice([0,1,2,3])
    if ns==_mouth_state: ns=(_mouth_state+choice([1,2,3]))%4
    _mouth_state=ns; _mouth_last_t=t

def _maybe_blink(t, bc, flux_val, boom):
    global _blink_release_L,_blink_release_R,_blink_last_t,_last_bc_blink
    trig=False
    if flux_val>BLINK_FLUX_THR and boom<BLINK_BOOM_MAX: trig=True
    if bc>0 and (bc%BEAT_BLINK_PER==0) and (_last_bc_blink!=bc): trig=True; _last_bc_blink=bc
    if (t-_blink_last_t)>IDLE_BLINK_GAP: trig=True
    if trig:
        # 20% chance to only blink one eye (alien-y)
        if random()<0.2:
            if random()<0.5: _blink_release_L=t+BLINK_HOLD
            else: _blink_release_R=t+BLINK_HOLD
        else:
            _blink_release_L=t+BLINK_HOLD
            _blink_release_R=t+BLINK_HOLD
        _blink_last_t=t

@register_visualizer
class AlienFace(BaseVisualizer):
    display_name = "Alien Face"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _frozen_env,_frozen_punch,_frozen_boom,_silence_mode
        global _mouth_state,_mouth_last_t,_last_bc_mouth
        global _blink_release_L,_blink_release_R,_gaze_target,_last_bc_gaze,_spark_until,_spark_x,_spark_y

        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return

        env,gate,boom,punch,bc,dt,flux_val = beat_drive(bands,rms,t)
        paused, prev_sil = activity(bands,rms,t,env,gate,boom,flux_val,dt)
        if (not prev_sil) and _silence_mode:
            _frozen_env,_frozen_punch,_frozen_boom = env,punch,boom

        env_use = _frozen_env if _silence_mode else env
        punch_use = _frozen_punch if _silence_mode else punch
        boom_use = _frozen_boom if _silence_mode else boom

        # background
        p.fillRect(r, QBrush(QColor(4,8,14)))
        cx, cy = r.center().x(), r.center().y()
        amp = spring_to("amp", 1.0 + 0.6*env_use + 1.2*punch_use + 0.8*boom_use, t, k=24, c=8, lo=0.6, hi=3.8)
        R = min(w,h)*0.32*amp

        # head
        head_rect = QRectF(cx-R*0.95, cy-R*1.15, R*1.9, R*2.2)
        g = QRadialGradient(QPointF(cx,cy-R*0.2), R*1.4)
        g.setColorAt(0, QColor(130, 255, 150, 245))
        g.setColorAt(1, QColor(20, 60, 30, 240))
        p.setBrush(QBrush(g)); p.setPen(QPen(QColor(30,60,40,255), 6))
        p.drawEllipse(head_rect)

        # antenna
        stem_h = R*0.9
        p.setPen(QPen(QColor(60,120,80,230), max(3,int(R*0.07))))
        sway = sin(t*3.2 + env_use*2.0)*R*0.06
        p.drawLine(QPointF(cx, head_rect.top()+R*0.15),
                   QPointF(cx+sway, head_rect.top()-stem_h*0.1))
        bob_r = R*0.12 + R*0.05*punch_use
        p.setBrush(QBrush(QColor(200,255,200,245)))
        p.setPen(QPen(QColor(60,120,80,230), 2))
        p.drawEllipse(QPointF(cx+sway, head_rect.top()-stem_h*0.1), bob_r, bob_r)

        # saccade target (beat-driven), plus idle drift
        if not _silence_mode:
            if bc>0 and (bc%GAZE_BEAT_PER==0) and (_last_bc_gaze!=bc):
                _last_bc_gaze=bc
                _gaze_target = choice([-1,0,1])*GAZE_PX_FRAC
        idle = GAZE_PX_FRAC*0.5*sin(2*3.14159*GAZE_IDLE_SPEED*t)
        gaze = spring_to("gaze", _gaze_target + idle, t, k=40.0, c=10.0, lo=-0.5, hi=0.5)

        # energy squint shifts baseline blink openness
        if not _silence_mode:
            _maybe_blink(t, bc, flux_val, boom_use)
        targetL = max(0.05, 1.0 - 0.25*env_use)
        targetR = targetL
        if t < _blink_release_L: targetL = 0.05
        if t < _blink_release_R: targetR = 0.05
        blinkL = spring_to("blinkL", targetL, t, k=50.0, c=12.0, lo=0.02, hi=1.0)
        blinkR = spring_to("blinkR", targetR, t, k=50.0, c=12.0, lo=0.02, hi=1.0)

        eye_w = R*0.60
        eye_hL = R*0.30*blinkL
        eye_hR = R*0.30*blinkR
        eye_y = cy - R*0.25
        left_x = cx - R*0.60 + gaze*R
        right_x= cx + R*0.60 + gaze*R

        def almond(xc, yc, w, h):
            path = QPainterPath(QPointF(xc - w/2, yc))
            path.cubicTo(QPointF(xc - w*0.25, yc - h),
                         QPointF(xc + w*0.25, yc - h),
                         QPointF(xc + w/2, yc))
            path.cubicTo(QPointF(xc + w*0.25, yc + h),
                         QPointF(xc - w*0.25, yc + h),
                         QPointF(xc - w/2, yc))
            return path

        # spectral tilt for iris ring color
        mid = _midhi(bands); low = _low(bands); denom = max(1e-6, mid+low)
        tilt = (mid - low)/denom  # [-1..1] approx
        hue = int(200 + 80*tilt) % 360  # blue->magenta on more treble

        p.setBrush(QBrush(QColor(30,35,40,255)))
        p.setPen(QPen(QColor(10,15,20,240), 4))
        p.drawPath(almond(left_x, eye_y, eye_w, eye_hL))
        p.drawPath(almond(right_x, eye_y, eye_w, eye_hR))

        # iris ring
        ring_col = QColor.fromHsv(hue, 220, 255, 180)
        p.setPen(QPen(ring_col, max(2,int(R*0.05))))
        p.drawPath(almond(left_x, eye_y, eye_w*0.92, eye_hL*0.88))
        p.drawPath(almond(right_x, eye_y, eye_w*0.92, eye_hR*0.88))

        # pupils (hide when closed)
        if blinkL > 0.12:
            pup = R*(0.10 + 0.20*(env_use + 0.4*punch_use))*blinkL
            p.setBrush(QBrush(QColor(0,0,0,255))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawEllipse(QPointF(left_x, eye_y), pup, pup)
            p.setBrush(QBrush(QColor(255,255,255,200)))
            p.drawEllipse(QPointF(left_x-pup*0.35, eye_y-pup*0.35), pup*0.25, pup*0.25)
        if blinkR > 0.12:
            pup = R*(0.10 + 0.20*(env_use + 0.4*punch_use))*blinkR
            p.setBrush(QBrush(QColor(0,0,0,255))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawEllipse(QPointF(right_x, eye_y), pup, pup)
            p.setBrush(QBrush(QColor(255,255,255,200)))
            p.drawEllipse(QPointF(right_x-pup*0.35, eye_y-pup*0.35), pup*0.25, pup*0.25)

        # big hit sparkle
        if (not _silence_mode) and punch_use>SPARK_PUNCH_THR and t>_spark_until:
            _spark_until = t + SPARK_HOLD
            _spark_x = cx + R*0.75
            _spark_y = cy - R*0.85
        if t < _spark_until:
            p.setPen(QPen(QColor(255,255,255,230), max(2,int(R*0.04))))
            p.drawLine(QPointF(_spark_x-R*0.08, _spark_y), QPointF(_spark_x+R*0.08, _spark_y))
            p.drawLine(QPointF(_spark_x, _spark_y-R*0.08), QPointF(_spark_x, _spark_y+R*0.08))

        # mouth states
        base_y = cy + R*0.65; width=R*1.10
        ms=_mouth_state
        if not _silence_mode:
            if bc % 2 == 0 and bc>0 and _last_bc_mouth!=bc:
                _last_bc_mouth=bc; _switch_mouth(t)
            if _mouth_last_t is None: _mouth_last_t=t
            if t-_mouth_last_t>0.6: _switch_mouth(t)

        lip=QColor(40,15,60,255); fill=QColor(20,10,30,245)
        if ms==0:
            height=R*(0.20+0.52*(env_use+0.6*punch_use))
            p.setBrush(QBrush(lip)); p.setPen(QPen(QColor(50,30,80,255),4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.22, R*0.22)
            p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.85/2, base_y-height*0.70/2, width*0.85, height*0.70), R*0.14, R*0.14)
        elif ms==1:
            thick=max(4,int(R*0.14*(0.9+0.4*(env_use+punch_use))))
            p.setPen(QPen(lip,thick)); p.drawLine(QPointF(cx-width*0.55,base_y), QPointF(cx+width*0.55,base_y))
        elif ms==2:
            height=R*(0.10+0.32*(env_use+0.5*punch_use))
            p.setBrush(QBrush(lip)); p.setPen(QPen(QColor(50,30,80,255),4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.18, R*0.18)
            p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.80/2, base_y-height*0.65/2, width*0.80, height*0.65), R*0.12, R*0.12)
        else:
            curve=R*(0.22+0.20*(env_use+0.6*punch_use))
            path=QPainterPath(QPointF(cx-width*0.55,base_y))
            ctrlY=base_y+curve
            path.cubicTo(QPointF(cx-width*0.20,ctrlY), QPointF(cx+width*0.20,ctrlY), QPointF(cx+width*0.55,base_y))
            p.setPen(QPen(lip,max(5,int(R*0.14)))); p.setBrush(QBrush(QColor(0,0,0,0))); p.drawPath(path)
