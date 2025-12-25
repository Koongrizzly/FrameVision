
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
BEAT_BLINK_PER = 8
IDLE_BLINK_GAP = 2.0

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

_blink_release = 0.0
_blink_last_t = 0.0
_last_bc_blink = -1

# Gaze and faux head-tilt
GAZE_IDLE_SPEED = 0.35
GAZE_RECENTER_PUNCH = 0.75
TILT_FLUX_THR = 0.30
_tilt_sign = 1

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
    global _prev_frame_bands,_prev_rms,_static_quiet_time,_last_active_t,_silence_mode,_tilt_sign
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
    # tilt sign flips on strong snares
    if (flux_val>TILT_FLUX_THR) and (boom<0.2):
        _tilt_sign *= -1
    return paused, prev

def _switch_mouth(t):
    global _mouth_state,_mouth_last_t
    ns=choice([0,1,2,3])
    if ns==_mouth_state: ns=(_mouth_state+choice([1,2,3]))%4
    _mouth_state=ns; _mouth_last_t=t

def _maybe_blink(t, bc, flux_val):
    global _blink_release,_blink_last_t,_last_bc_blink
    trig=False
    if flux_val>BLINK_FLUX_THR: trig=True
    if bc>0 and (bc%BEAT_BLINK_PER==0) and (_last_bc_blink!=bc): trig=True; _last_bc_blink=bc
    if (t-_blink_last_t)>IDLE_BLINK_GAP: trig=True
    if trig:
        _blink_release=t+BLINK_HOLD
        _blink_last_t=t

@register_visualizer
class DogFace(BaseVisualizer):
    display_name = "Dog Face"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _frozen_env,_frozen_punch,_frozen_boom,_silence_mode
        global _mouth_state,_mouth_last_t,_last_bc_mouth,_blink_release,_tilt_sign

        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return

        env,gate,boom,punch,bc,dt,flux_val = beat_drive(bands,rms,t)
        paused, prev_sil = activity(bands,rms,t,env,gate,boom,flux_val,dt)
        if (not prev_sil) and _silence_mode:
            _frozen_env,_frozen_punch,_frozen_boom = env,punch,boom

        env_use = _frozen_env if _silence_mode else env
        punch_use = _frozen_punch if _silence_mode else punch
        boom_use = _frozen_boom if _silence_mode else boom

        p.fillRect(r, QBrush(QColor(10,10,8)))
        cx, cy = r.center().x(), r.center().y()
        amp = spring_to("amp", 1.0 + 0.55*env_use + 1.1*punch_use + 0.8*boom_use, t, k=24, c=8, lo=0.7, hi=3.6)
        R = min(w,h)*0.30*amp

        # faux head tilt via asymmetric vertical offsets
        tilt_px = spring_to("tilt", _tilt_sign * R*0.10*(0.2+0.8*flux_val), t, k=35.0, c=10.0, lo=-R*0.15, hi=R*0.15)

        # head
        face_col = QColor(235, 210, 150, 250)
        p.setBrush(QBrush(face_col)); p.setPen(QPen(QColor(90,70,50,255), 6))
        p.drawEllipse(QPointF(cx,cy), R, R)

        # floppy ears
        ear_w = R*0.55; ear_h = R*0.75
        swing = (0.25 + 0.55*boom_use) * sin(t*2.0)
        shift = R*(0.05 + 0.05*punch_use)
        p.setBrush(QBrush(QColor(205, 175, 120, 250))); p.setPen(QPen(QColor(90,70,50,255), 5))
        p.drawRoundedRect(QRectF(cx-R-ear_w*0.15-shift, cy-R*0.10-tilt_px, ear_w, ear_h*(1.0+swing*0.2)), R*0.20, R*0.20)
        p.drawRoundedRect(QRectF(cx+R-ear_w*0.85+shift, cy-R*0.10+tilt_px, ear_w, ear_h*(1.0-swing*0.2)), R*0.20, R*0.20)

        # eyes + blink + gaze
        if not _silence_mode: _maybe_blink(t, bc, flux_val)
        target = 0.08 if t < _blink_release else 1.0
        blink = spring_to("blink", target, t, k=50.0, c=12.0, lo=0.05, hi=1.0)

        eye_w = R*0.34; eye_h_open = R*0.22
        # slight eye-smile when mouth is smile (ms==3) — reduce top height
        ms_preview = _mouth_state
        eye_h_open *= (0.85 if ms_preview==3 else 1.0)
        eye_h = eye_h_open*blink
        eye_y = cy - R*0.10
        left_x = cx - R*0.42; right_x = cx + R*0.42

        p.setBrush(QBrush(QColor(255,255,255,250))); p.setPen(QPen(QColor(60,50,40,220), 4))
        p.drawRoundedRect(QRectF(left_x-eye_w/2, eye_y-eye_h/2-tilt_px, eye_w, eye_h), eye_h*0.8, eye_h*0.8)
        p.drawRoundedRect(QRectF(right_x-eye_w/2, eye_y-eye_h/2+tilt_px, eye_w, eye_h), eye_h*0.8, eye_h*0.8)

        # friendly gaze drift that recenters on strong punch
        gaze_idle = R*0.06*sin(t*0.7)
        center_bias = 0.0 if punch_use<GAZE_RECENTER_PUNCH else (-gaze_idle)  # snap towards center
        gaze = spring_to("gaze", gaze_idle + center_bias, t, k=45.0, c=12.0, lo=-R*0.12, hi=R*0.12)

        # pupils (hide on closed)
        if blink>0.12:
            pup = eye_h_open*(0.22 + 0.35*(env_use + 0.4*punch_use))*blink
            p.setBrush(QBrush(QColor(10,10,15,255))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawEllipse(QPointF(left_x+gaze, eye_y-tilt_px), pup, pup)
            p.drawEllipse(QPointF(right_x+gaze, eye_y+tilt_px), pup, pup)
            p.setBrush(QBrush(QColor(255,255,255,200)))
            p.drawEllipse(QPointF(left_x+gaze-pup*0.35, eye_y-tilt_px-pup*0.35), pup*0.22, pup*0.22)
            p.drawEllipse(QPointF(right_x+gaze-pup*0.35, eye_y+tilt_px-pup*0.35), pup*0.22, pup*0.22)

        # eyebrows pop on snare-ish hits
        if flux_val>0.30 and boom_use<0.2 and not _silence_mode:
            brow_h = R*0.10
            p.setPen(QPen(QColor(70,50,40,240), max(3,int(R*0.06))))
            p.drawLine(QPointF(left_x-eye_w*0.35, eye_y-eye_h*0.9-tilt_px-brow_h),
                       QPointF(left_x+eye_w*0.35, eye_y-eye_h*0.9-tilt_px-brow_h))
            p.drawLine(QPointF(right_x-eye_w*0.35, eye_y-eye_h*0.9+tilt_px-brow_h),
                       QPointF(right_x+eye_w*0.35, eye_y-eye_h*0.9+tilt_px-brow_h))

        # nose
        p.setBrush(QBrush(QColor(60,40,40,255))); p.setPen(QPen(QColor(40,25,25,220), 3))
        p.drawEllipse(QPointF(cx, cy+R*0.18), R*0.15, R*0.12)

        # mouth + tongue
        base_y = cy + R*0.48; width  = R*1.00
        ms=_mouth_state
        if not _silence_mode:
            if bc % 2 == 0 and bc>0 and _last_bc_mouth!=bc:
                _last_bc_mouth=bc; _switch_mouth(t)
            if _mouth_last_t is None: _mouth_last_t=t
            if t-_mouth_last_t>0.6: _switch_mouth(t)

        lip=QColor(80,40,55,255); fill=QColor(35,20,30,245)
        if ms==0:
            height=R*(0.22+0.52*(env_use+0.5*punch_use))
            p.setBrush(QBrush(lip)); p.setPen(QPen(QColor(60,30,50,255), 4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.18, R*0.18)
            p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.82/2, base_y-height*0.70/2, width*0.82, height*0.70), R*0.12, R*0.12)
            tongue_h=height*0.55
            p.setBrush(QBrush(QColor(220,110,120,250))); p.setPen(QPen(QColor(150,60,70,220), 3))
            p.drawRoundedRect(QRectF(cx - width*0.22, base_y-height*0.05, width*0.44, tongue_h), R*0.12, R*0.12)
        elif ms==1:
            p.setPen(QPen(lip, max(4,int(R*0.12))))
            p.drawLine(QPointF(cx-width*0.48, base_y), QPointF(cx+width*0.48, base_y))
        elif ms==2:
            height=R*(0.10+0.28*(env_use+0.3*punch_use))
            p.setBrush(QBrush(lip)); p.setPen(QPen(QColor(60,30,50,255), 4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.16, R*0.16)
            p.setBrush(QBrush(fill)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.80/2, base_y-height*0.62/2, width*0.80, height*0.62), R*0.10, R*0.10)
        else:
            curve=R*(0.22+0.18*(env_use+0.5*punch_use))
            path=QPainterPath(QPointF(cx - width*0.52, base_y))
            ctrl=base_y+curve
            path.cubicTo(QPointF(cx - width*0.18, ctrl), QPointF(cx + width*0.18, ctrl), QPointF(cx + width*0.52, base_y))
            p.setPen(QPen(lip, max(5,int(R*0.12)))); p.setBrush(QBrush(QColor(0,0,0,0))); p.drawPath(path)
