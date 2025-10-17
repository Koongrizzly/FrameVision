# --- Singing / Talking Smiley  ---
# Eyes: desynced. Left switches every 3 beats, right every 2 beats (random L/C/R).
# Mouth: four states (OPEN_WIDE / CLOSE / HALF_OPEN / SMILE) switching every 2 beats.
# No cheek lines. Smooth slides. No vertical pupil bob.
# v35: keeps pause/silence gating; fixes SMILE to curve upward (∪) correctly.

TUNE_KICK   = 1.2
TUNE_BOOM   = 1.0
SPR_K       = 30.0
SPR_C       = 6.0
SPR_MAX     = 4.2
EYE_PERIOD_LEFT   = 3
EYE_PERIOD_RIGHT  = 2
MOUTH_PERIOD      = 1
EYE_INTERVAL_LEFT_SEC  = 0.9
EYE_INTERVAL_RIGHT_SEC = 1.0
MOUTH_INTERVAL_SEC     = 0.55

# --- silence gate tuning ---
SILENCE_RMS_THR    = 0.015
SILENCE_ENV_THR    = 0.050
SILENCE_BOOM_THR   = 0.120
SILENCE_GATE_THR   = 0.55
SILENCE_HOLD_SEC   = 0.60
WAKE_EAGER_SEC     = 0.10

# --- pause/static detector ---
PAUSE_STATIC_SEC   = 0.35
STATIC_FLUX_THR    = 0.015
STATIC_BANDS_EPS   = 0.002
STATIC_RMS_EPS     = 0.002

from math import sin, cos, pi
from random import Random, choice
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

_rng = Random(50531)

# ---------- beat/env kit ----------
_prev = []
_env = 0.0
_gate = 0.0
_punch = 0.0
_pt = None
_beat_count = 0

# for pause/static detection
_prev_frame_bands = None
_prev_rms = 0.0
_static_quiet_time = 0.0

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
    global _env, _gate, _punch, _pt, _beat_count
    e = _midhi(bands); f = _flux(bands); lo = _low(bands)
    target = 0.58*e + 1.30*f + 0.18*rms + 0.22*lo*TUNE_BOOM
    target = target/(1+0.7*target)
    if target > _env: _env = 0.72*_env + 0.28*target
    else: _env = 0.92*_env + 0.08*target
    hi, lo_thr = 0.30, 0.18
    g = 1.0 if f > hi else (0.0 if f < lo_thr else _gate)
    if g>0.6 and _gate<=0.6: _beat_count += 1
    _gate = 0.82*_gate + 0.18*g
    boom = min(1.0, max(0.0, lo*1.25 + 0.42*rms))
    if _pt is None: _pt = t
    dt = max(0.0, min(0.033, t - _pt)); _pt = t
    decay = pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if g>0.6 else 0.0)
    return max(0.0,min(1.0,_env)), max(0.0,min(1.0,_gate)), boom, max(0.0,min(1.0,_punch)), _beat_count, dt, f

# spring for general scalar
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
# mouth states: 0=open_wide, 1=close, 2=half_open, 3=smile (true upward)
_mouth_state = 2
_eye_last_t_L = None
_eye_last_t_R = None
_mouth_last_t = None

# --- silence-mode bookkeeping ---
_last_active_t = None
_silence_mode  = False
_frozen_env    = 0.0
_frozen_punch  = 0.0
_frozen_boom   = 0.0

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
    new_state = choice([0,1,2,3])  # OPEN_WIDE / CLOSE / HALF_OPEN / SMILE
    if new_state == _mouth_state:
        new_state = (_mouth_state + choice([1,2,3])) % 4
    _mouth_state = new_state
    _mouth_last_t = t

@register_visualizer
class SingingSmileyV35(BaseVisualizer):
    display_name = "Singing Smiley"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_bc_eye_L,_last_bc_eye_R,_last_bc_mouth
        global _eye_state_L,_eye_state_R,_mouth_state
        global _eye_last_t_L,_eye_last_t_R,_mouth_last_t
        global _last_active_t,_silence_mode,_frozen_env,_frozen_punch,_frozen_boom
        global _prev_frame_bands,_prev_rms,_static_quiet_time

        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return

        p.fillRect(r, QBrush(QColor(6,6,10)))
        env, gate, boom, punch, bc, dt, flux_val = beat_drive(bands, rms, t)
        if _eye_last_t_L is None: _eye_last_t_L = t
        if _eye_last_t_R is None: _eye_last_t_R = t
        if _mouth_last_t is None: _mouth_last_t = t
        if _last_active_t is None: _last_active_t = t

        # --- PAUSE/static detection ---
        avg_delta = 0.0
        if bands and _prev_frame_bands and len(_prev_frame_bands)==len(bands):
            s=0.0
            for a,b in zip(bands,_prev_frame_bands):
                s += abs(a-b)
            avg_delta = s/len(bands)
        if (avg_delta < STATIC_BANDS_EPS) and (abs(rms - _prev_rms) < STATIC_RMS_EPS) and (flux_val < STATIC_FLUX_THR):
            _static_quiet_time += dt
        else:
            _static_quiet_time = 0.0
        paused_static = _static_quiet_time > PAUSE_STATIC_SEC
        _prev_frame_bands = list(bands) if bands else None
        _prev_rms = rms

        # --- determine active audio vs. silence/paused ---
        active_now = (not paused_static) and ((rms > SILENCE_RMS_THR) or (env > SILENCE_ENV_THR) or (gate > SILENCE_GATE_THR) or (boom > SILENCE_BOOM_THR))
        if active_now:
            _last_active_t = t

        prev_silence = _silence_mode
        _silence_mode = paused_static or ((t - _last_active_t) > SILENCE_HOLD_SEC)
        if _silence_mode and active_now and (t - _last_active_t) <= WAKE_EAGER_SEC:
            _silence_mode = False

        if (not prev_silence) and _silence_mode:
            _frozen_env   = env
            _frozen_punch = punch
            _frozen_boom  = boom

        env_use   = _frozen_env   if _silence_mode else env
        punch_use = _frozen_punch if _silence_mode else punch
        boom_use  = _frozen_boom  if _silence_mode else boom

        # beat-driven switches (disabled during silence)
        if not _silence_mode:
            if bc % EYE_PERIOD_LEFT == 0 and bc>0 and _last_bc_eye_L != bc:
                _last_bc_eye_L = bc; _switch_eye_L(t)
            if bc % EYE_PERIOD_RIGHT == 0 and bc>0 and _last_bc_eye_R != bc:
                _last_bc_eye_R = bc; _switch_eye_R(t)
            if bc % MOUTH_PERIOD == 0 and bc>0 and _last_bc_mouth != bc:
                _last_bc_mouth = bc; _switch_mouth(t)

            if t - _eye_last_t_L > EYE_INTERVAL_LEFT_SEC:  _switch_eye_L(t)
            if t - _eye_last_t_R > EYE_INTERVAL_RIGHT_SEC: _switch_eye_R(t)
            if t - _mouth_last_t  > MOUTH_INTERVAL_SEC:    _switch_mouth(t)

        # Face scale to beat (calm when silent)
        target = 1.0 + 0.7*env_use + TUNE_KICK*punch_use + 1.0*boom_use
        if _silence_mode: target = 1.0
        amp = spring_to("face_amp", target, t, k=SPR_K*0.8, c=SPR_C, lo=0.5, hi=4.5)

        cx, cy = r.center().x(), r.center().y()
        R = min(w,h)*0.30*amp

        # face glow
        g = QRadialGradient(QPointF(cx,cy), R*1.6)
        g.setColorAt(0.0, QColor.fromHsv(int((t*60+220*env_use)%360), 220, 255, 140))
        g.setColorAt(1.0, QColor(0,0,0,0))
        p.setBrush(QBrush(g)); p.setPen(QPen(QColor(255,255,255,22), 1))
        p.drawEllipse(QPointF(cx,cy), R*1.1, R*1.1)

        # face base
        p.setBrush(QBrush(QColor(255,228,120,235)))
        p.setPen(QPen(QColor(60,40,20,240), 6))
        p.drawEllipse(QPointF(cx,cy), R, R)

        # eyes
        eye_w = R*0.30; eye_h = R*0.18
        eye_y  = cy - R*0.20
        left_x = cx - R*0.38
        right_x= cx + R*0.38
        px_amp = eye_w*0.34

        def eye_px_from_state(state):
            return (-1 if state==0 else (0 if state==1 else 1)) * px_amp

        cur_px_L = spring_to("eye_px_L", eye_px_from_state(_eye_state_L), t, k=50.0, c=10.0, lo=-px_amp*1.5, hi=px_amp*1.5)
        cur_px_R = spring_to("eye_px_R", eye_px_from_state(_eye_state_R), t, k=50.0, c=10.0, lo=-px_amp*1.5, hi=px_amp*1.5)

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

        # --- mouth states ---
        lip_col  = QColor(70,50,90,255)
        fill_col = QColor(25,15,35,245)
        base_y   = cy + R*0.18
        width    = R*0.95

        ms = _mouth_state  # 0=open_wide,1=close,2=half_open,3=smile(upward)
        if ms == 0:  # OPEN_WIDE
            openness = 0.45 + 0.42*(env_use + 0.6*punch_use)
            height = R*(0.20 + 0.52*openness)
            p.setBrush(QBrush(lip_col)); p.setPen(QPen(QColor(40,20,60,255), 4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.18, R*0.18)
            p.setBrush(QBrush(fill_col)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.86/2, base_y-height*0.70/2, width*0.86, height*0.70), R*0.12, R*0.12)

        elif ms == 1:  # CLOSE
            thickness = max(4, int(R*0.14*(0.9+0.4*(env_use+punch_use))))
            p.setPen(QPen(lip_col, thickness))
            p.drawLine(QPointF(cx-width*0.55, base_y), QPointF(cx+width*0.55, base_y))

        elif ms == 2:  # HALF_OPEN
            openness = 0.25 + 0.25*(env_use + 0.5*punch_use)
            height = R*(0.10 + 0.30*openness)
            p.setBrush(QBrush(lip_col)); p.setPen(QPen(QColor(40,20,60,255), 4))
            p.drawRoundedRect(QRectF(cx-width/2, base_y-height/2, width, height), R*0.16, R*0.16)
            p.setBrush(QBrush(fill_col)); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawRoundedRect(QRectF(cx-width*0.84/2, base_y-height*0.65/2, width*0.84, height*0.65), R*0.10, R*0.10)

        elif ms == 3:  # SMILE (true upward ∪): control points BELOW base_y
            curve = R*(0.22 + 0.20*(env_use + 0.6*punch_use))
            path = QPainterPath(QPointF(cx - width*0.55, base_y))
            ctrlY = base_y + curve   # +Y is down, so this curves upward visually
            path.cubicTo(QPointF(cx - width*0.20, ctrlY),
                         QPointF(cx + width*0.20, ctrlY),
                         QPointF(cx + width*0.55, base_y))
            p.setPen(QPen(lip_col, max(5,int(R*0.14))))
            p.setBrush(QBrush(QColor(0,0,0,0)))
            p.drawPath(path)

        else:
            thickness = max(4, int(R*0.14))
            p.setPen(QPen(lip_col, thickness))
            p.drawLine(QPointF(cx-width*0.55, base_y), QPointF(cx+width*0.55, base_y))
