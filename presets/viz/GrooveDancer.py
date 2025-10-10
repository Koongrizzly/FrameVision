# GrooveDancer.py — Beat-only dancing woman silhouette
from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QPointF, QRectF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# ======== Config ========
FREEZE_WHEN_IDLE   = True    # do not move if no beats
IDLE_TO_CENTER_SEC = 2.0     # time since last detected beat to treat as idle
IDLE_ENV_THR       = 0.04    # energy threshold to also consider idle (gate noise)
POP_HALF_LIFE      = 0.10    # seconds — how long a beat "pop" persists visually
ARM_HALF_LIFE      = 0.18    # seconds — arm flick persistence
STEP_HALF_LIFE     = 0.22    # seconds — step impulse persistence

# ======== State ========
_prev_spec = []
_last_t = None
_onset_avg = 0.0
_onset_peak = 1e-3
_gate = 0.0
_since_on = 10.0

_pop_hi = 0.0   # treble-driven (hands / head)
_pop_lo = 0.0   # bass-driven (hips / steps)
_arm_side = 1   # 1 = right arm up flick next, -1 = left
_step_side = 1  # 1 = right leg step next, -1 = left

def _flux(bands):
    """Positive spectral flux (onset-like)."""
    global _prev_spec
    if not bands:
        _prev_spec = []
        return 0.0
    if (not _prev_spec) or (len(_prev_spec) != len(bands)):
        _prev_spec = [0.0] * len(bands)
    f = 0.0
    n = len(bands)
    for i,(x,px) in enumerate(zip(bands,_prev_spec)):
        d = x - px
        if d > 0:
            f += d * (0.35 + 0.65 * (i / max(1, n-1)))
    _prev_spec = [0.82*px + 0.18*x for x,px in zip(bands,_prev_spec)]
    return f / max(1, n)

def _split(bands):
    if not bands: return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6); b=max(a+1, n//2)
    lo=sum(bands[:a])/a
    mid=sum(bands[a:b])/max(1,(b-a))
    hi=sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _onset_features(bands, rms, dt):
    """Adaptive onset with norm 0..~1; returns (lo,mid,hi,fx,env,norm,beat)."""
    global _onset_avg, _onset_peak, _gate, _since_on
    lo,mid,hi = _split(bands)
    fx = _flux(bands)
    env = 0.5*rms + 0.35*lo + 0.15*mid
    onset = 0.6*hi + 1.5*fx + 0.2*rms
    # adaptive normalization
    _onset_avg = 0.98*_onset_avg + 0.02*onset
    _onset_peak = max(_onset_peak*pow(0.5, dt/0.6), onset)  # 0.6s half-life
    denom = max(1e-3, _onset_peak - 0.6*_onset_avg)
    norm = max(0.0, (onset - 0.8*_onset_avg) / denom)
    beat = (norm > 0.55) and (_gate <= 0.5)
    _gate = 0.78*_gate + 0.22*(1.0 if norm>0.55 else 0.0)
    if beat:
        _since_on = 0.0
    return lo,mid,hi,fx,env,norm,beat

def _decay(x, dt, half_life):
    return x * pow(0.5, dt/max(1e-6, half_life))

def _rot(x, y, ang):
    ca, sa = cos(ang), sin(ang)
    return x*ca - y*sa, x*sa + y*ca

@register_visualizer
class GrooveDancer(BaseVisualizer):
    display_name = "Groove Dancer (Beat Silhouette)"
    def paint(self, p:QPainter, r, bands, rms, t):
        global _last_t, _since_on, _pop_hi, _pop_lo, _arm_side, _step_side
        w,h = int(r.width()), int(r.height())
        if w<=0 or h<=0: return
        if _last_t is None: _last_t = t
        dt = max(0.0, min(0.05, t - _last_t)); _last_t = t
        _since_on += dt

        lo,mid,hi,fx,env,norm,beat = _onset_features(bands, rms, dt)
        idle = (FREEZE_WHEN_IDLE and (_since_on > IDLE_TO_CENTER_SEC) and (env < IDLE_ENV_THR))

        # Trigger impulses ONLY on beats (no idle drift)
        if beat and not idle:
            if lo > 0.08:           # bass -> step / hips
                _pop_lo = 1.0
                _step_side *= -1
            if hi + fx > lo*0.6:    # treble/flux -> arms / head
                _pop_hi = 1.0
                _arm_side *= -1

        _pop_lo = _decay(_pop_lo, dt, STEP_HALF_LIFE)
        _pop_hi = _decay(_pop_hi, dt, ARM_HALF_LIFE)

        # --- draw background, subtle vignette ---
        p.fillRect(r, QBrush(QColor(8, 10, 14)))
        p.setPen(QPen(QColor(20,22,28,150), 8)); p.drawRect(r)

        # --- dancer silhouette anchor ---
        S = min(w,h)
        ground_y = int(h*0.82)
        hip_x = int(w*0.5)
        hip_y = int(ground_y - S*0.28)

        # global pose
        sway = (_step_side)*0.09*_pop_lo  # radians
        head_nod = 0.06*_pop_hi
        shoulder_lift = 0.10*_pop_hi*_arm_side

        # helper to draw limb from pivot with 2-segment angles
        def limb(pivot_x, pivot_y, ang1, len1, ang2, len2, thick):
            x1,y1 = _rot(0, -len1, ang1)   # up from pivot in local space
            x2,y2 = _rot(0, -len2, ang1+ang2)
            p.setPen(QPen(QColor(18,20,24), thick+2))
            p.drawLine(QPointF(pivot_x, pivot_y), QPointF(pivot_x + x1, pivot_y + y1))
            p.setPen(QPen(QColor(235,235,245), thick))
            p.drawLine(QPointF(pivot_x + x1, pivot_y + y1), QPointF(pivot_x + x1 + x2, pivot_y + y1 + y2))
            return pivot_x + x1 + x2, pivot_y + y1 + y2

        p.save()
        # whole body sway from hips
        p.translate(hip_x, hip_y)
        p.rotate(sway * 180/pi)
        p.translate(-hip_x, -hip_y)

        # torso (triangle-ish dress silhouette)
        p.setBrush(QBrush(QColor(235, 235, 245)))  # silhouette color
        p.setPen(Qt.NoPen)
        torso = QPainterPath()
        torso.moveTo(hip_x - S*0.06, hip_y)           # left hip
        torso.lineTo(hip_x + S*0.06, hip_y)           # right hip
        torso.lineTo(hip_x + S*0.12, hip_y - S*0.22 + shoulder_lift*S*0.12) # right shoulder
        torso.lineTo(hip_x - S*0.12, hip_y - S*0.22 - shoulder_lift*S*0.12) # left shoulder
        torso.closeSubpath()
        p.drawPath(torso)

        # head (slight nod)
        head_r = S*0.045*(1.0 + 0.4*_pop_hi)
        hx = hip_x
        hy = hip_y - S*0.28 - head_nod*S*0.06
        p.drawEllipse(QPointF(hx, hy), head_r, head_r)

        # arms — shoulder pivot points
        L_sh_x = hip_x - S*0.11; L_sh_y = hip_y - S*0.22 - shoulder_lift*S*0.12
        R_sh_x = hip_x + S*0.11; R_sh_y = hip_y - S*0.22 + shoulder_lift*S*0.12

        # arm angles (base pose + beat flick)
        base_up = -0.3  # radians relative to vertical
        flick = 0.6*_pop_hi
        # Left arm
        limb(L_sh_x, L_sh_y, base_up - 0.25*_arm_side, S*0.10, 0.35 + flick*(1 if _arm_side<0 else 0), S*0.12, int(S*0.010))
        # Right arm
        limb(R_sh_x, R_sh_y, base_up + 0.25*_arm_side, S*0.10, -0.35 - flick*(1 if _arm_side>0 else 0), S*0.12, int(S*0.010))

        # legs — hip pivot
        L_hip_x = hip_x - S*0.045; R_hip_x = hip_x + S*0.045; hip_y0 = hip_y
        step = 0.35*_pop_lo  # knee lift amount

        # Left leg (opposite of current stepping side to alternate)
        l1 = S*0.13; l2 = S*0.14
        ang_thigh_L = 0.20 + (step if _step_side<0 else 0.0)   # forward
        ax_L = L_hip_x; ay_L = hip_y0
        fx_L, fy_L = limb(ax_L, ay_L, ang_thigh_L, l1, -0.35 - step*0.6, l2, int(S*0.012))

        # Right leg
        ang_thigh_R = -0.20 + (step if _step_side>0 else 0.0)
        ax_R = R_hip_x; ay_R = hip_y0
        fx_R, fy_R = limb(ax_R, ay_R, ang_thigh_R, l1, 0.35 + step*0.6, l2, int(S*0.012))

        # ground shadow / floor
        p.setBrush(QBrush(QColor(40, 44, 56, 140)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(hip_x, ground_y), S*0.16*(1.0+0.2*_pop_lo), S*0.035*(1.0+0.2*_pop_lo))

        p.restore()
