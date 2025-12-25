
from math import sin, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# ====== Tunables ======
# Beat-driven stepping; tweak to taste
EYE_STEP_BEATS_L = 1
EYE_STEP_BEATS_R = 1
MOUTH_STEP_BEATS = 1

# Gaze feel
GAZE_OFFSET_FRAC = 0.060   # fraction of face size for pupil shift
TILT_THR = 0.08            # smaller = more direction changes

# Silence/idle gate
PAUSE_STATIC_SEC  = 0.25
STATIC_FLUX_THR   = 0.010
STATIC_BANDS_EPS  = 0.0015
STATIC_RMS_EPS    = 0.0015
SILENCE_RMS_THR   = 0.010
SILENCE_HOLD_SEC  = 0.50
WAKE_EAGER_SEC    = 0.10

# BPM handling
FBK_BPM = 120.0            # fallback BPM
BPM_MIN = 60.0
BPM_MAX = 180.0

# Eyebrow pop
BROW_FLUX_THR = 0.28       # treble-ish hit threshold
BROW_BOOM_MAX = 0.25       # not too bassy
BROW_HOLD     = 0.16       # seconds to stay lit

# Pulse ring
RING_MIN_THICK_FR = 0.06   # of face size
RING_MAX_THICK_FR = 0.16
RING_ALPHA_MIN    = 80
RING_ALPHA_MAX    = 200

# ====== Beat drive & gating ======
_prev=[]; _env=0.0; _gate=0.0; _punch=0.0; _pt=None
_prev_frame_bands=None; _prev_rms=0.0; _static_quiet_time=0.0
_last_active_t=None; _silence_mode=False

# beat counters
_bc=0; _since_bc=0.0
_last_onset_t=None; _prev_onset_t=None; _bpm_ema=None

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
    global _prev
    if not b: _prev=[]; return 0.0
    n=len(b)
    if not _prev or len(_prev)!=n: _prev=[0.0]*n
    cut=max(1,n//6); f=c=0.0
    for i in range(cut,n):
        d=b[i]-_prev[i]
        if d>0: f += d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i]+0.12*b[i] for i in range(n)]
    return f/max(1,c)

def _drive(bands, rms):
    # compact envelope just for sizing/eye aperture
    e=_midhi(bands); f=_flux(bands); lo=_low(bands)
    tgt=(0.55*e + 1.30*f + 0.20*rms + 0.18*lo)
    tgt=tgt/(1+0.6*tgt)
    if not hasattr(_drive,'v'): _drive.v=0.0
    v=_drive.v
    v=0.62*v+0.38*tgt if tgt>v else 0.88*v+0.12*tgt
    _drive.v=v
    return v, f, lo, e

def beat_and_gate(bands, rms, t):
    """Returns (env, gate, punch, bc, dt, active, est_bpm, lo, mid, flux)"""
    global _env,_gate,_punch,_pt,_bc,_since_bc,_last_onset_t,_prev_onset_t,_bpm_ema
    global _prev_frame_bands,_prev_rms,_static_quiet_time,_last_active_t,_silence_mode

    env, flux, lo, mid = _drive(bands, rms)

    # onset gate
    hi, lo_thr = 0.30, 0.18
    g = 1.0 if flux>hi else (0.0 if flux<lo_thr else _gate)
    rising = (g>0.6 and _gate<=0.6)
    _gate = 0.82*_gate + 0.18*g

    # dt
    if _pt is None: _pt=t
    dt = max(0.0, min(0.033, t-_pt)); _pt=t

    # static measure (detect paused)
    avg=0.0
    if bands and _prev_frame_bands and len(_prev_frame_bands)==len(bands):
        s=0.0
        for a,b in zip(bands,_prev_frame_bands): s += abs(a-b)
        avg = s/len(bands)
    if (avg<STATIC_BANDS_EPS) and (abs(rms-_prev_rms)<STATIC_RMS_EPS) and (flux<STATIC_FLUX_THR):
        _static_quiet_time += dt
    else:
        _static_quiet_time = 0.0
    _prev_frame_bands = list(bands) if bands else None
    _prev_rms = rms

    # active check
    active_now = (rms>SILENCE_RMS_THR) or (env>0.05) or (g>0.2)
    if _last_active_t is None: _last_active_t = t
    if active_now: _last_active_t = t
    _silence_mode = (_static_quiet_time>PAUSE_STATIC_SEC) or (t-_last_active_t>SILENCE_HOLD_SEC)
    if _silence_mode and active_now and (t-_last_active_t)<=WAKE_EAGER_SEC:
        _silence_mode = False

    # beat counting (and bpm estimate) only when active
    _since_bc = max(0.0, _since_bc + dt)
    if not _silence_mode:
        if rising:
            _bc += 1
            # bpm estimate from IBI
            if _last_onset_t is not None:
                ibi = t - _last_onset_t
                if 0.25 <= ibi <= 2.0:
                    bpm = 60.0 / ibi
                    if _bpm_ema is None: _bpm_ema = bpm
                    else: _bpm_ema = 0.85*_bpm_ema + 0.15*bpm
            _prev_onset_t = _last_onset_t
            _last_onset_t = t
            _since_bc = 0.0
        # fallback stepping by estimated BPM
        bpm_use = min(BPM_MAX, max(BPM_MIN, _bpm_ema if _bpm_ema else FBK_BPM))
        sec_per = 60.0 / bpm_use
        if _since_bc >= sec_per:
            add = int(_since_bc // sec_per)
            if add>0:
                _bc += add
                _since_bc -= add*sec_per

    # punch for eye squeeze & ring
    decay=pow(0.78, dt/0.016) if dt>0 else 0.78
    _punch = max(_punch*decay, 1.0 if rising else 0.0)

    bpm_out = min(BPM_MAX, max(BPM_MIN, _bpm_ema if _bpm_ema else FBK_BPM))
    return env, _gate, _punch, _bc, dt, (not _silence_mode), bpm_out, lo, mid, flux

# ====== springs ======
_sdict={}
def spring_to(key, target, t, k=30.0, c=6.0, lo=-1e9, hi=1e9):
    s,v,pt=_sdict.get(key,(target,0.0,None))
    if pt is None: _sdict[key]=(target,0.0,t); return target
    dt=max(0.0, min(0.033, t-pt))
    a=-k*(s-target) - c*v
    v+=a*dt; s+=v*dt
    if s<lo: s=lo
    if s>hi: s=hi
    _sdict[key]=(s,v,t); return s

# ====== face state ======
_eye_pos_L=0
_eye_pos_R=0
_eye_blk_L=-1
_eye_blk_R=-1

_mouth_cycle=[0,1,2,1]   # OPEN -> LINE -> SMILE -> LINE -> repeat
_mouth_idx=0
_mouth_blk=-1
_mouth_timer=0.0          # time fallback

# eyebrow flash timer
_brow_until = 0.0

@register_visualizer
class PixelRobot(BaseVisualizer):
    display_name = "Pixel Robot"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _eye_pos_L,_eye_pos_R,_eye_blk_L,_eye_blk_R
        global _mouth_idx,_mouth_blk,_mouth_timer,_brow_until

        env, gate, punch, bc, dt, active, bpm, lo, mid, flux = beat_and_gate(bands, rms, t)

        w,h=int(r.width()),int(r.height())
        if w<=0 or h<=0: return
        p.fillRect(r,QBrush(QColor(6,6,10)))
        cx,cy=r.center().x(), r.center().y()

        # size reacts to env + spectrum
        s=spring_to("s", 1.0+0.6*env+0.6*lo+0.3*mid, t, k=24.0, c=7.0, hi=3.6)
        size=min(w,h)*0.60*s

        # ----- spectral tilt (for gaze & ring hue) -----
        tilt = (mid - lo) / max(1e-6, (mid+lo))
        side = -1 if tilt < -TILT_THR else (1 if tilt > TILT_THR else 0)

        # ----- eyebrows pop on treble/snare hits -----
        if active and (flux > BROW_FLUX_THR) and (lo < BROW_BOOM_MAX):
            _brow_until = t + BROW_HOLD
        brow_on = (t < _brow_until)

        # ----- Eye gaze -----
        blkL = bc // max(1,EYE_STEP_BEATS_L)
        if active and blkL != _eye_blk_L:
            _eye_blk_L = blkL
            if side == 0:
                _eye_pos_L = 0 if (blkL%2==0) else 1
            else:
                _eye_pos_L = -1 if side<0 else 1

        blkR = (bc+1) // max(1,EYE_STEP_BEATS_R)  # offset phase
        if active and blkR != _eye_blk_R:
            _eye_blk_R = blkR
            if side == 0:
                _eye_pos_R = 0 if (blkR%2==1) else -1
            else:
                _eye_pos_R = -1 if side<0 else 1

        if not active:
            _eye_pos_L = _eye_pos_R = 0

        # ----- Mouth stepping: beat blocks OR time fallback while active -----
        step_blk = bc // max(1, MOUTH_STEP_BEATS)
        stepped = False
        if active and step_blk != _mouth_blk:
            _mouth_blk = step_blk
            _mouth_idx = (_mouth_idx + 1) % len(_mouth_cycle)
            _mouth_timer = 0.0
            stepped = True

        if active and not stepped:
            _mouth_timer += dt
            sec_per_step = (60.0 / max(1e-3, bpm)) * MOUTH_STEP_BEATS
            if _mouth_timer >= sec_per_step * 1.05:  # small slack
                _mouth_timer = 0.0
                _mouth_idx = (_mouth_idx + 1) % len(_mouth_cycle)

        # ===== colorful pulse ring (draw BEHIND head) =====
        # Hue follows tilt (bass→warm, treble→cool); thickness & alpha pulse with env/punch
        hue = int((200 + 80*tilt + (t*40)%360) % 360)
        thick = (RING_MIN_THICK_FR + (RING_MAX_THICK_FR-RING_MIN_THICK_FR)*(0.35*env + 0.65*punch)) * size
        alpha = int(RING_ALPHA_MIN + (RING_ALPHA_MAX-RING_ALPHA_MIN)*(0.3*env + 0.7*punch))
        ring_col = QColor.fromHsv(hue, 220, 255, alpha)
        # ring radius
        base_r = size*0.58
        pulse = 0.02*sin(2*pi*(0.5*t)) + 0.06*punch
        rad = base_r*(1.0 + pulse)
        p.setBrush(QBrush(QColor(0,0,0,0)))
        p.setPen(QPen(ring_col, max(2,int(thick))))
        p.drawEllipse(QPointF(cx, cy), rad, rad)
        # faint outer halo
        ring_col2 = QColor.fromHsv((hue+40)%360, 180, 255, max(40, alpha//3))
        p.setPen(QPen(ring_col2, max(2,int(thick*0.5))))
        p.drawEllipse(QPointF(cx, cy), rad*1.12, rad*1.12)

        # ===== head =====
        p.setBrush(QBrush(QColor(20,26,36,230))); p.setPen(QPen(QColor(140,200,255,220),6))
        p.drawRoundedRect(QRectF(cx-size/2, cy-size*0.40, size, size*0.80), 16,16)

        # ===== eyes (lens + iris with pupil shift) =====
        ap_base = size*0.06*(1.0 + 0.35*env)
        ap_tgt = ap_base * (0.88 - 0.30*punch)
        irisL = spring_to("apL", ap_tgt, t, k=40.0, c=10.0, lo=size*0.02, hi=size*0.12)
        irisR = spring_to("apR", ap_tgt, t, k=40.0, c=10.0, lo=size*0.02, hi=size*0.12)

        off = size*GAZE_OFFSET_FRAC
        pxL = {-1:-off, 0:0, 1:off}[_eye_pos_L]
        pxR = {-1:-off, 0:0, 1:off}[_eye_pos_R]

        eye_y = cy-size*0.15
        for side_sign, iris, px in ((-1, irisL, pxL),(1, irisR, pxR)):
            x=cx+side_sign*size*0.28; y=eye_y
            p.setBrush(QBrush(QColor(12,16,24,230))); p.setPen(QPen(QColor(160,220,255,200),4))
            p.drawEllipse(QPointF(x,y), size*0.14, size*0.14)
            p.setBrush(QBrush(QColor(180,240,255,230))); p.setPen(QPen(QColor(0,0,0,0),0))
            p.drawEllipse(QPointF(x+px,y), iris, iris)

        # ===== eyebrow LEDs =====
        brow_w = size*0.18; brow_h = size*0.04
        brow_y = eye_y - size*0.15
        glow = 0.6 if brow_on else 0.15
        brow_col = QColor.fromHsv(int((200 + 100*tilt) % 360), 220, 255, int(255*glow))
        p.setBrush(QBrush(brow_col)); p.setPen(QPen(QColor(0,0,0,0),0))
        # left
        p.drawRoundedRect(QRectF(cx - size*0.28 - brow_w/2, brow_y, brow_w, brow_h), 4,4)
        # right
        p.drawRoundedRect(QRectF(cx + size*0.28 - brow_w/2, brow_y, brow_w, brow_h), 4,4)

        # ===== mouth LED matrix (same aesthetic) =====
        ms = _mouth_cycle[_mouth_idx] if active else 1  # line when inactive
        grid_w, grid_h = 10, 4
        cw = size*0.045; ch = size*0.045
        ox = cx - cw*grid_w/2; oy = cy + size*0.10

        on_col  = QColor(120,220,180, int(180 + 60*env))
        off_col = QColor(30,40,40,160)
        pen_off = QPen(QColor(0,0,0,0),0)

        for gy in range(grid_h):
            for gx in range(grid_w):
                on=False
                if ms==0:  # OPEN rectangle
                    on = 1<=gx<=8 and 1<=gy<=2
                elif ms==1:  # LINE (closed)
                    on = gy==2
                else:  # SMILE curve
                    on = (gy==3 and gx in (2,3,6,7)) or (gy==2 and gx in (1,4,5,8))
                p.setBrush(QBrush(on_col if on else off_col)); p.setPen(pen_off)
                p.drawRoundedRect(QRectF(ox+gx*cw, oy+gy*ch, cw-3, ch-3), 3,3)
