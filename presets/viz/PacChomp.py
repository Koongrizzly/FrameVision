
from math import sin
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QPointF, QRectF
from helpers.music import register_visualizer, BaseVisualizer

# ---------- Beat + Silence Gate ----------
_prev = []                 # for flux
_prev_frame_bands = None   # for static detection
_prev_rms = 0.0
_static_quiet_time = 0.0
_gate = 0.0
_pt = None
_since_bc = 0.0
_beat_count = 0
_silence_mode = False
_last_active_t = None

# Tunables
FBK_BPM = 120.0        # fallback tempo (only used when NOT silent)
PAUSE_STATIC_SEC = 0.35
STATIC_FLUX_THR = 0.015
STATIC_BANDS_EPS = 0.002
STATIC_RMS_EPS = 0.002
SILENCE_RMS_THR = 0.015
SILENCE_HOLD_SEC = 0.60
WAKE_EAGER_SEC   = 0.10

def _midhi(bands):
    if not bands: return 0.0
    n=len(bands); cut=max(1,n//6)
    s=c=0.0
    for i in range(cut,n):
        w=0.35+0.65*((i-cut)/max(1,n-cut))
        s+=w*bands[i]; c+=1
    return s/max(1,c)

def _flux(bands):
    global _prev
    if not bands: _prev=[]; return 0.0
    n=len(bands)
    if not _prev or len(_prev)!=n: _prev=[0.0]*n
    cut=max(1,n//6)
    f=c=0.0
    for i in range(cut,n):
        d=bands[i]-_prev[i]
        if d>0: f+=d*(0.3+0.7*((i-cut)/max(1,n-cut)))
        c+=1
    _prev=[0.88*_prev[i]+0.12*bands[i] for i in range(n)]
    return f/max(1,c)

def _drive(bands, rms):
    # compact envelope just for sizing/speed
    e=_midhi(bands); f=_flux(bands)
    tgt=(0.55*e + 1.35*f + 0.25*rms)
    tgt=tgt/(1+0.6*tgt)
    if not hasattr(_drive,'v'): _drive.v=0.0
    v=_drive.v
    v=0.62*v+0.38*tgt if tgt>v else 0.85*v+0.15*tgt
    _drive.v=v
    return v, f

def beat_and_gate(bands, rms, t):
    """Returns: (drive, beat_count, is_active) where is_active=False when paused/silent.
       Beat count increments on flux rising edges, or by fallback clock ONLY when active."""
    global _gate,_pt,_since_bc,_beat_count,_silence_mode,_prev_frame_bands,_prev_rms,_static_quiet_time,_last_active_t
    drive, flux = _drive(bands, rms)

    # Beat gate (rising edge)
    hi, lo = 0.30, 0.18
    g = 1.0 if flux>hi else (0.0 if flux<lo else _gate)
    rising = (g>0.6 and _gate<=0.6)
    _gate = 0.82*_gate + 0.18*g

    # dt
    if _pt is None: _pt=t
    dt = max(0.0, min(0.033, t-_pt)); _pt=t

    # Static/paused detection
    avg_delta = 0.0
    if bands and _prev_frame_bands and len(_prev_frame_bands)==len(bands):
        s=0.0
        for a,b in zip(bands,_prev_frame_bands): s += abs(a-b)
        avg_delta = s/len(bands)
    if (avg_delta < STATIC_BANDS_EPS) and (abs(rms-_prev_rms) < STATIC_RMS_EPS) and (flux < STATIC_FLUX_THR):
        _static_quiet_time += dt
    else:
        _static_quiet_time = 0.0
    paused_static = _static_quiet_time > PAUSE_STATIC_SEC
    _prev_frame_bands = list(bands) if bands else None
    _prev_rms = rms

    # Active now?
    active_now = (not paused_static) and (rms > SILENCE_RMS_THR or flux > lo or drive > 0.06)
    if _last_active_t is None: _last_active_t = t
    if active_now: _last_active_t = t

    _silence_mode = paused_static or ((t - _last_active_t) > SILENCE_HOLD_SEC)
    if _silence_mode and active_now and (t - _last_active_t) <= WAKE_EAGER_SEC:
        _silence_mode = False

    # Beat counting — only when NOT silent
    _since_bc = max(0.0, _since_bc + dt)
    sec_per = 60.0 / max(1e-3, FBK_BPM)
    if not _silence_mode:
        if rising:
            _beat_count += 1
            _since_bc = 0.0
        elif _since_bc >= sec_per:
            add = int(_since_bc // sec_per)
            if add>0:
                _beat_count += add
                _since_bc -= add*sec_per

    return drive, _beat_count, (not _silence_mode)

# ---------- Mouth pattern (discrete) ----------
OPEN_ANG  = 50   # degrees removed on each side (bigger = wider)
HALF_ANG  = 33
CLOSE_ANG = 6

MOUTH_STEP_BEATS = 1   # change to 2,3,4... to step every N beats
_mouth_phase = 0       # 0:open, 1:half, 2:closed, 3:half
_last_step_block = -1

@register_visualizer
class PacChomp(BaseVisualizer):
    display_name = "Pac Chomp"
    def paint(self, p: QPainter, r, bands, rms, t):
        global _mouth_phase, _last_step_block
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return

        # background
        p.fillRect(r, QBrush(QColor(5,5,12)))

        # music drive + beat counter + activity gate
        drive, bc, is_active = beat_and_gate(bands, rms, t)

        # Advance mouth only when active (not paused/silent)
        blk = bc // max(1, MOUTH_STEP_BEATS)
        if is_active and blk != _last_step_block:
            _last_step_block = blk
            _mouth_phase = (_mouth_phase + 1) % 4

        # pick angle
        open_ang = OPEN_ANG if _mouth_phase==0 else HALF_ANG if _mouth_phase in (1,3) else CLOSE_ANG

        # Pac position/size
        speed = 0.10 + 0.55*drive
        x = r.left() + ( (t*speed*200) % (w+120) ) - 60
        y = r.center().y() + (h*0.20)*sin(t*0.7)
        radius = int(min(w,h)*0.08*(0.9+0.5*drive))

        # Body sector (2*open_ang gap)
        body = QPainterPath()
        body.moveTo(x,y)
        body.arcMoveTo(x-radius, y-radius, 2*radius, 2*radius, open_ang)
        body.arcTo(x-radius, y-radius, 2*radius, 2*radius, open_ang, 360-2*open_ang)
        body.closeSubpath()
        p.setBrush(QBrush(QColor(255, 230, 50)))
        p.setPen(QPen(QColor(255,255,160), 2))
        p.drawPath(body)

        # eye
        p.setBrush(QBrush(QColor(30,30,30)))
        p.setPen(QPen(QColor(0,0,0,0), 0))
        p.drawEllipse(QPointF(x+radius*0.2, y-radius*0.35), radius*0.12, radius*0.12)

        # ---- Single color-changing ball ----
        ahead = radius*1.4 + 26
        px = x + ahead
        py = y + 6*sin(0.7*t + drive*3.0)
        hue = int((t*80 + 200*drive) % 360)
        sat = int(190 + 60*drive)
        col_main = QColor.fromHsv(hue, sat, 255, 255)
        col_edge = QColor.fromHsv((hue+40)%360, min(255, sat+30), 255, 200)
        p.setBrush(QBrush(col_main))
        p.setPen(QPen(col_edge, 2))
        size = 7 + 4*((_mouth_phase in (0,2)) and 1 or 0)
        p.drawEllipse(QPointF(px, py), size, size)
