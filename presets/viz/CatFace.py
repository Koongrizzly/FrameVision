
from math import sin
from random import choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

TUNE_BOOM = 1.0
SPR_K, SPR_C = 26.0, 8.0
SILENCE_RMS_THR, SILENCE_ENV_THR, SILENCE_BOOM_THR, SILENCE_GATE_THR = 0.015, 0.050, 0.120, 0.55
SILENCE_HOLD_SEC, WAKE_EAGER_SEC = 0.60, 0.10
PAUSE_STATIC_SEC, STATIC_FLUX_THR, STATIC_BANDS_EPS, STATIC_RMS_EPS = 0.35, 0.015, 0.002, 0.002

# Blinks
BLINK_FLUX_THR = 0.20
BLINK_HOLD     = 0.10
BEAT_BLINK_PER = 12
IDLE_BLINK_GAP = 2.2

# Calm pose
CALM_ENV_THR = 0.10
CALM_FLUX_THR = 0.05
CALM_HOLD_SEC = 0.6

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

_mouth_state = 2
_mouth_last_t = None
_last_bc_mouth = -1
_ear_phase = 0

_blink_release = 0.0
_blink_last_t = 0.0
_last_bc_blink = -1

_calm_time = 0.0

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
    global _prev_frame_bands,_prev_rms,_static_quiet_time,_last_active_t,_silence_mode,_calm_time
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
    # calm detector
    if (env < CALM_ENV_THR) and (flux_val < CALM_FLUX_THR) and (boom < CALM_ENV_THR):
        _calm_time += dt
    else:
        _calm_time = 0.0
    return paused, prev

def _switch_mouth(t):
    global _mouth_state,_mouth_last_t
    ns=choice([0,1,2,3])
    if ns==_mouth_state: ns=(_mouth_state+choice([1,2,3]))%4
    _mouth_state=ns; _mouth_last_t=t

def _maybe_blink(t, bc, flux_val, calm_pose):
    global _blink_release,_blink_last_t,_last_bc_blink
    trig=False
    if flux_val>BLINK_FLUX_THR: trig=True
    if bc>0 and (bc%BEAT_BLINK_PER==0) and (_last_bc_blink!=bc): trig=True; _last_bc_blink=bc
    if (t-_blink_last_t)>IDLE_BLINK_GAP: trig=True
    if calm_pose:  # slow blink (longer)
        _blink_release = t + BLINK_HOLD*2.0
        _blink_last_t = t
        return
    if trig:
        _blink_release=t+BLINK_HOLD
        _blink_last_t=t

@register_visualizer
class CatFace(BaseVisualizer):
    display_name = "Cat Face"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _frozen_env,_frozen_punch,_frozen_boom,_silence_mode
        global _mouth_state,_mouth_last_t,_last_bc_mouth,_ear_phase,_blink_release,_calm_time

        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return

        env,gate,boom,punch,bc,dt,flux_val = beat_drive(bands,rms,t)
        paused, prev_sil = activity(bands,rms,t,env,gate,boom,flux_val,dt)
        if (not prev_sil) and _silence_mode:
            _frozen_env,_frozen_punch,_frozen_boom = env,punch,boom

        env_use = _frozen_env if _silence_mode else env
        punch_use = _frozen_punch if _silence_mode else punch

        p.fillRect(r, QBrush(QColor(10,8,12)))
        cx, cy = r.center().x(), r.center().y()
        amp = spring_to("amp", 1.0 + 0.5*env_use + 1.0*punch_use, t, k=26, c=8, lo=0.7, hi=3.5)
        R = min(w,h)*0.30*amp

        # head
        face_col = QColor(245, 210, 180, 250)
        p.setBrush(QBrush(face_col)); p.setPen(QPen(QColor(90,70,60,255), 6))
        p.drawEllipse(QPointF(cx,cy), R, R)

        # ears twitch (stronger on punches)
        twitch = 0.0 if _silence_mode else (0.10 + 0.20*punch_use + 0.10*flux_val)
        if bc>0 and not _silence_mode: _ear_phase = 1 if bc%2==0 else -1
        p.setBrush(QBrush(QColor(230,190,160,255))); p.setPen(QPen(QColor(90,70,60,255), 5))
        off = R*0.12*_ear_phase*twitch
        p.drawPolygon([QPointF(cx-R*0.55, cy-R*0.55), QPointF(cx-R*0.15, cy-R*1.10+off), QPointF(cx-R*0.05, cy-R*0.55)])
        p.drawPolygon([QPointF(cx+R*0.55, cy-R*0.55), QPointF(cx+R*0.15, cy-R*1.10-off), QPointF(cx+R*0.05, cy-R*0.55)])

        # calm pose detection
        calm_pose = (_calm_time > CALM_HOLD_SEC) and (not _silence_mode)

        # eyes + blink
        if not _silence_mode: _maybe_blink(t, bc, flux_val, calm_pose)
        target = 0.08 if (t < _blink_release or calm_pose) else 1.0
        blink = spring_to("blink", target, t, k=50.0, c=12.0, lo=0.05, hi=1.0)

        eye_w = R*0.42; eye_h_open = R*0.26
        eye_h = eye_h_open*blink
        eye_y = cy - R*0.15
        # prowl gaze
        gaze = R*0.08*sin(t*0.8)
        left_x = cx - R*0.45 + gaze; right_x = cx + R*0.45 + gaze

        p.setBrush(QBrush(QColor(255,255,255,250))); p.setPen(QPen(QColor(60,50,50,220), 4))
        p.drawRoundedRect(QRectF(left_x-eye_w/2, eye_y-eye_h/2, eye_w, eye_h), eye_h*0.6, eye_h*0.6)
        p.drawRoundedRect(QRectF(right_x-eye_w/2, eye_y-eye_h/2, eye_w, eye_h), eye_h*0.6, eye_h*0.6)

        # slit pupils (hide on closed)
        if blink>0.12:
            slit = max(1.5, eye_w*(0.06 + 0.25*(1.0-env_use)))
            p.setBrush(QBrush(QColor(10,10,10,255))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(left_x-slit/2, eye_y-eye_h*0.45, slit, eye_h*0.90), eye_h*0.45, eye_h*0.45)
            p.drawRoundedRect(QRectF(right_x-slit/2, eye_y-eye_h*0.45, slit, eye_h*0.90), eye_h*0.45, eye_h*0.45)

        # nose
        nose_w = R*0.18
        p.setBrush(QBrush(QColor(210,120,120,255))); p.setPen(QPen(QColor(120,60,60,220), 3))
        p.drawEllipse(QPointF(cx, cy+R*0.05), nose_w*0.55, nose_w*0.40)

        # whiskers react to flux
        whisk = R*(0.55 + 0.25*flux_val)
        y_wh = cy + R*0.10
        p.setPen(QPen(QColor(120,100,100,230), 3))
        for offy in (-0.06, 0.0, 0.06):
            y = y_wh + R*offy
            p.drawLine(QPointF(cx - R*0.15, y), QPointF(cx - whisk, y - R*0.05))
            p.drawLine(QPointF(cx + R*0.15, y), QPointF(cx + whisk, y - R*0.05))

        # mouth with calm override
        base_y = cy + R*0.40; width  = R*0.95
        ms=_mouth_state
        if calm_pose:  # prefer half-open when calm
            ms = 2
        if not _silence_mode and not calm_pose:
            if bc % 2 == 0 and bc>0 and _last_bc_mouth!=bc:
                _last_bc_mouth=bc; _switch_mouth(t)
            if _mouth_last_t is None: _mouth_last_t=t
            if t-_mouth_last_t>0.6: _switch_mouth(t)

        lip=QColor(90,50,80,255); fill=QColor(40,20,40,245)
        if ms==0:
            height=R*(0.18+0.45*(env_use+0.4*punch_use))
            p.setBrush(QBrush(lip)); p.setPen(QPen(QColor(60,30,60,255), 4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.16, R*0.16)
            p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.80/2, base_y-height*0.65/2, width*0.80, height*0.65), R*0.12, R*0.12)
        elif ms==1:
            p.setPen(QPen(lip, max(4,int(R*0.12))))
            p.drawLine(QPointF(cx-width*0.42, base_y), QPointF(cx+width*0.42, base_y))
        elif ms==2:
            height=R*(0.08+0.26*(env_use+0.3*punch_use))
            p.setBrush(QBrush(lip)); p.setPen(QPen(QColor(60,30,60,255), 4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.14, R*0.14)
            p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.78/2, base_y-height*0.60/2, width*0.78, height*0.60), R*0.10, R*0.10)
        else:
            curve=R*(0.20+0.18*(env_use+0.5*punch_use))
            path=QPainterPath(QPointF(cx - width*0.45, base_y))
            ctrl=base_y+curve
            path.cubicTo(QPointF(cx - width*0.18, ctrl), QPointF(cx + width*0.18, ctrl), QPointF(cx + width*0.45, base_y))
            p.setPen(QPen(lip, max(5,int(R*0.12)))); p.setBrush(QBrush(QColor(0,0,0,0))); p.drawPath(path)
