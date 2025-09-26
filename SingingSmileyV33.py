
# --- Singing / Talking Smiley (v3.3) ---
# Eyes: Left switches every 3 beats, Right every 4 beats (random L/C/R).
# Mouth: OPEN / CLOSE / SMILE every 2 beats.
# Idle behavior: if no onsets for IDLE_TO_CENTER_SEC, both eyes snap to CENTER and hold.
TUNE_KICK   = 1.2
TUNE_BOOM   = 1.0
SPR_K       = 30.0
SPR_C       = 6.0
SPR_MAX     = 4.2

EYE_PERIOD_LEFT   = 3
EYE_PERIOD_RIGHT  = 4
MOUTH_PERIOD      = 2

EYE_INTERVAL_LEFT_SEC  = 0.9
EYE_INTERVAL_RIGHT_SEC = 1.0
MOUTH_INTERVAL_SEC     = 0.55

IDLE_TO_CENTER_SEC     = 2.0

from math import sin, cos, pi
from random import Random, choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(70707)

# ---------- beat/env kit ----------
_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None
_beat_count = 0
_last_onset_t = 0.0

def _midhi(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s = c = 0.0
    for i in range(cut, n):
        w = 0.35 + 0.65*((i-cut)/max(1, n-cut))
        s += w*bands[i]; c += 1
    return s/max(1, c)

def _low(bands):
    if not bands: return 0.0
    n = len(bands); cut = max(1, n//6)
    s = c = 0.0
    for i in range(0, cut):
        w = 1.0 - 0.4*(i/max(1,cut-1))
        s += w*bands[i]; c += 1
    return s/max(1, c)

def _flux(bands):
    global _prev
    if not bands:
        _prev = []; return 0.0
    n = len(bands)
    if not _prev or len(_prev)!=n:
        _prev = [0.0]*n
    cut = max(1, n//6)
    f = c = 0.0
    for i in range(cut, n):
        d = bands[i] - _prev[i]
        if d > 0: f += d*(0.3 + 0.7*((i-cut)/max(1, n-cut)))
        c += 1
    _prev = [0.88*_prev[i] + 0.12*bands[i] for i in range(n)]
    return f/max(1, c)

def beat_drive(bands, rms, t):
    global _env, _gate, _punch, _pt, _beat_count, _last_onset_t
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)
    target = 0.58*e + 1.30*f + 0.18*rms + 0.22*lo*TUNE_BOOM
    target = target/(1+0.7*target)
    if target > _env: _env = 0.72*_env + 0.28*target
    else: _env = 0.92*_env + 0.08*target
    hi, lo_thr = 0.30, 0.18
    g = 1.0 if f > hi else (0.0 if f < lo_thr else _gate)
    if g>0.6 and _gate<=0.6:
        _beat_count += 1
        _last_onset_t = t
    _gate = 0.82*_gate + 0.18*g
    boom = min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch)), _beat_count, dt, _last_onset_t

_sdict = {}
def spring_to(key, target, t, k=SPR_K, c=SPR_C, lo=-1e9, hi=1e9):
    s, v, pt = _sdict.get(key, (0.0, 0.0, None))
    if pt is None:
        _sdict[key] = (target, 0.0, t)
        return target
    dt = max(0.0, min(0.033, t-pt))
    a = -k*(s-target) - c*v
    v += a*dt
    s += v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key] = (s, v, t)
    return s

# ---------- states for eyes / mouth with time fallback ----------
_last_bc_eye_L = -1
_last_bc_eye_R = -1
_last_bc_mouth = -1
_eye_state_L = 1   # 0=left,1=center,2=right
_eye_state_R = 2
_mouth_state = 0   # 0=open,1=close,2=smile
_eye_last_t_L = None
_eye_last_t_R = None
_mouth_last_t = None

def _switch_eye_L(t):
    global _eye_state_L, _eye_last_t_L
    new_state = choice([0,1,2])
    if new_state == _eye_state_L:
        new_state = (_eye_state_L + choice([1,2])) % 3
    _eye_state_L = new_state
    _eye_last_t_L = t

def _switch_eye_R(t):
    global _eye_state_R, _eye_last_t_R
    new_state = choice([0,1,2])
    if new_state == _eye_state_R:
        new_state = (_eye_state_R + choice([1,2])) % 3
    _eye_state_R = new_state
    _eye_last_t_R = t

def _switch_mouth(t):
    global _mouth_state, _mouth_last_t
    new_state = choice([0,1,2])  # OPEN/CLOSE/SMILE only
    if new_state == _mouth_state:
        new_state = (_mouth_state + choice([1,2])) % 3
    _mouth_state = new_state
    _mouth_last_t = t

@register_visualizer
class SingingSmileyV33(BaseVisualizer):
    display_name = "Singing Smiley v3.3"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_bc_eye_L,_last_bc_eye_R,_last_bc_mouth
        global _eye_state_L,_eye_state_R,_mouth_state
        global _eye_last_t_L,_eye_last_t_R,_mouth_last_t
        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return

        p.fillRect(r, QBrush(QColor(6,6,10)))
        env, gate, boom, punch, bc, dt, last_onset_t = beat_drive(bands, rms, t)
        if _eye_last_t_L is None: _eye_last_t_L = t
        if _eye_last_t_R is None: _eye_last_t_R = t
        if _mouth_last_t is None: _mouth_last_t = t

        idle = (t - last_onset_t) > IDLE_TO_CENTER_SEC

        # beat-driven switches with different periods (only if not idle)
        if not idle and bc % EYE_PERIOD_LEFT == 0 and bc>0 and _last_bc_eye_L != bc:
            _last_bc_eye_L = bc; _switch_eye_L(t)
        if not idle and bc % EYE_PERIOD_RIGHT == 0 and bc>0 and _last_bc_eye_R != bc:
            _last_bc_eye_R = bc; _switch_eye_R(t)
        if bc % MOUTH_PERIOD == 0 and bc>0 and _last_bc_mouth != bc:
            _last_bc_mouth = bc; _switch_mouth(t)

        # time fallbacks (only when not idle for eyes; mouth continues)
        if not idle and t - _eye_last_t_L > EYE_INTERVAL_LEFT_SEC:  _switch_eye_L(t)
        if not idle and t - _eye_last_t_R > EYE_INTERVAL_RIGHT_SEC: _switch_eye_R(t)
        if t - _mouth_last_t  > MOUTH_INTERVAL_SEC:                 _switch_mouth(t)

        # force-center eyes during idle and pause timers
        if idle:
            _eye_state_L = 1; _eye_state_R = 1
            _eye_last_t_L = t; _eye_last_t_R = t

        # Face scale to beat
        target = 1.0 + 0.7*env + TUNE_KICK*punch + 1.0*boom
        amp = spring_to("face_amp", target, t, k=SPR_K*0.8, c=SPR_C, lo=0.5, hi=4.5)
        cx, cy = r.center().x(), r.center().y()
        R = min(w,h)*0.30*amp

        # face glow
        g = QRadialGradient(QPointF(cx,cy), R*1.6)
        g.setColorAt(0.0, QColor.fromHsv(int((t*60+220*env)%360), 220, 255, 140))
        g.setColorAt(1.0, QColor(0,0,0,0))
        p.setBrush(QBrush(g)); p.setPen(QPen(QColor(255,255,255,22), 1))
        p.drawEllipse(QPointF(cx,cy), R*1.1, R*1.1)

        # face base
        p.setBrush(QBrush(QColor(255,228,120,235)))
        p.setPen(QPen(QColor(60,40,20,240), 6))
        p.drawEllipse(QPointF(cx,cy), R, R)

        # eyes (independent states, smooth horizontal slides; stiffer spring on idle for snap)
        eye_w = R*0.30; eye_h = R*0.18
        eye_y  = cy - R*0.20
        left_x = cx - R*0.38
        right_x= cx + R*0.38
        px_amp = eye_w*0.34

        def eye_px_from_state(state): 
            return (-1 if state==0 else (0 if state==1 else 1)) * px_amp

        k_eye = 90.0 if idle else 50.0
        cur_px_L = spring_to("eye_px_L", eye_px_from_state(_eye_state_L), t, k=k_eye, c=10.0, lo=-px_amp*1.5, hi=px_amp*1.5)
        cur_px_R = spring_to("eye_px_R", eye_px_from_state(_eye_state_R), t, k=k_eye, c=10.0, lo=-px_amp*1.5, hi=px_amp*1.5)

        # left eye
        p.setBrush(QBrush(QColor(250,250,250,250))); p.setPen(QPen(QColor(40,40,40,220), 4))
        p.drawRoundedRect(QRectF(left_x-eye_w/2, eye_y-eye_h/2, eye_w, eye_h), eye_h*0.5, eye_h*0.5)
        p.setBrush(QBrush(QColor(30,30,40,250))); p.setPen(QPen(QColor(0,0,0,0), 0))
        p.drawEllipse(QPointF(left_x+cur_px_L, eye_y), eye_h*0.24, eye_h*0.24)
        p.setBrush(QBrush(QColor(255,255,255,200)))
        p.drawEllipse(QPointF(left_x+cur_px_L-eye_h*0.07, eye_y-eye_h*0.07), eye_h*0.07, eye_h*0.07)

        # right eye
        p.setBrush(QBrush(QColor(250,250,250,250))); p.setPen(QPen(QColor(40,40,40,220), 4))
        p.drawRoundedRect(QRectF(right_x-eye_w/2, eye_y-eye_h/2, eye_w, eye_h), eye_h*0.5, eye_h*0.5)
        p.setBrush(QBrush(QColor(30,30,40,250))); p.setPen(QPen(QColor(0,0,0,0), 0))
        p.drawEllipse(QPointF(right_x+cur_px_R, eye_y), eye_h*0.24, eye_h*0.24)
        p.setBrush(QBrush(QColor(255,255,255,200)))
        p.drawEllipse(QPointF(right_x+cur_px_R-eye_h*0.07, eye_y-eye_h*0.07), eye_h*0.07, eye_h*0.07)

        # --- mouth states (OPEN / CLOSE / SMILE only) ---
        lip_col  = QColor(70,50,90,255)
        fill_col = QColor(25,15,35,245)
        base_y   = cy + R*0.18
        width    = R*0.95

        ms = _mouth_state  # 0=open,1=close,2=smile
        if ms == 0:  # OPEN
            openness = 0.42 + 0.42*(env + 0.6*punch)
            height = R*(0.18 + 0.52*openness)
            p.setBrush(QBrush(lip_col)); p.setPen(QPen(QColor(40,20,60,255), 4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.18, R*0.18)
            p.setBrush(QBrush(fill_col)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.86/2, base_y-height*0.70/2, width*0.86, height*0.70), R*0.12, R*0.12)

        elif ms == 1:  # CLOSE
            thickness = max(4, int(R*0.14*(0.9+0.4*(env+punch))))
            p.setPen(QPen(lip_col, thickness))
            p.drawLine(QPointF(cx-width*0.55, base_y), QPointF(cx+width*0.55, base_y))

        else:  # SMILE
            curve = R*(0.22 + 0.18*(env + 0.6*punch))
            path = QPainterPath(QPointF(cx - width*0.55, base_y))
            path.cubicTo(QPointF(cx - width*0.20, base_y - curve),
                         QPointF(cx + width*0.20, base_y - curve),
                         QPointF(cx + width*0.55, base_y))
            p.setPen(QPen(lip_col, max(5,int(R*0.14))))
            p.setBrush(QBrush(QColor(0,0,0,0)))
            p.drawPath(path)
