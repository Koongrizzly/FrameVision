import math
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QRadialGradient, QLinearGradient
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

_last_t = None

_left_vals  = []
_right_vals = []
_left_peaks = []
_right_peaks= []

_env_lo=0.0
_env_mid=0.0
_env_hi=0.0
_prev_lo=0.0
_last_boom_t=-999.0

def _split_lo_mid_hi(bands):
    if not bands:
        return 0.0,0.0,0.0
    n=len(bands)
    a=max(1,n//6)
    b=max(a+1,n//2)
    lo = sum(bands[:a])/a
    mid= sum(bands[a:b])/max(1,(b-a))
    hi = sum(bands[b:])/max(1,(n-b))
    return lo,mid,hi

def _env_step(env,target,up=0.45,down=0.22):
    if target>env:
        return (1-up)*env+up*target
    else:
        return (1-down)*env+down*target

def _downsample(bands,count,start_frac,stop_frac):
    out=[]
    if not bands or count<=0:
        return [0.0]*count
    n=len(bands)
    lo_i=int(start_frac*n)
    hi_i=int(stop_frac*n)
    lo_i=max(0, min(n-1, lo_i))
    hi_i=max(lo_i+1, min(n, hi_i))
    seg=bands[lo_i:hi_i]
    if not seg:
        seg=[0.0]

    seg_n=len(seg)
    for i in range(count):
        s=int((i)*seg_n/count)
        e=int((i+1)*seg_n/count)
        if e<=s: e=min(s+1, seg_n)
        part=seg[s:e]
        out.append(sum(part)/len(part))
    return out

@register_visualizer
class DualFacingEQ(BaseVisualizer):
    """
    Two mirrored mini equalizers (left vs right), no center line.
    Tweaked: bar scale reduced ~10% so they don't instantly max out.
    """

    display_name = "Dual Facing EQ"

    def __init__(self):
        try:
            super().__init__()
        except Exception:
            pass

    def paint(self,p:QPainter,r,bands,rms,t):
        global _last_t,_left_vals,_right_vals,_left_peaks,_right_peaks
        global _env_lo,_env_mid,_env_hi,_prev_lo,_last_boom_t

        w=float(r.width())
        h=float(r.height())
        if w<=0 or h<=0:
            return

        if _last_t is None:
            _last_t = t
        dt=max(0.0,min(0.05,t-_last_t))
        _last_t = t

        bar_count=12
        left_now  = _downsample(bands, bar_count, 0.00, 0.50)
        right_now = _downsample(bands, bar_count, 0.40, 1.00)

        if len(_left_vals)!=bar_count:
            _left_vals =[0.0]*bar_count
        if len(_right_vals)!=bar_count:
            _right_vals=[0.0]*bar_count
        if len(_left_peaks)!=bar_count:
            _left_peaks=[0.0]*bar_count
        if len(_right_peaks)!=bar_count:
            _right_peaks=[0.0]*bar_count

        attack=0.5
        decay =0.2
        peak_fall=0.6

        for i,v in enumerate(left_now):
            cur=_left_vals[i]
            if v>cur:
                cur=(1.0-attack)*cur + attack*v
            else:
                cur=(1.0-decay)*cur + decay*v
            _left_vals[i]=cur

            if cur>_left_peaks[i]:
                _left_peaks[i]=cur
            else:
                _left_peaks[i]=max(0.0,_left_peaks[i]-peak_fall*dt)

        for i,v in enumerate(right_now):
            cur=_right_vals[i]
            if v>cur:
                cur=(1.0-attack)*cur + attack*v
            else:
                cur=(1.0-decay)*cur + decay*v
            _right_vals[i]=cur

            if cur>_right_peaks[i]:
                _right_peaks[i]=cur
            else:
                _right_peaks[i]=max(0.0,_right_peaks[i]-peak_fall*dt)

        lo,mid,hi = _split_lo_mid_hi(bands)
        _env_lo  = _env_step(_env_lo,  lo + 0.4*rms, up=0.5, down=0.25)
        _env_mid = _env_step(_env_mid, mid + 0.3*rms,up=0.4, down=0.2)
        _env_hi  = _env_step(_env_hi,  hi + 0.6*rms, up=0.6, down=0.3)

        boom = (lo-_prev_lo) > 0.22 and (t-_last_boom_t) > 0.25
        _prev_lo=lo
        if boom:
            _last_boom_t = t

        # background
        p.fillRect(r, QColor(5,5,10))

        # center glow from bass/mids/highs
        cx = w*0.5
        cy = h*0.5
        glow_radius = min(w,h)*0.3*(1.0+0.4*_env_lo)
        center_grad = QRadialGradient(QPointF(cx,cy), glow_radius)
        center_grad.setColorAt(0.0, QColor.fromHsv(int((200+150*_env_hi)%360),
                                                   200,255,
                                                   int(100+130*_env_lo)))
        center_grad.setColorAt(1.0, QColor(0,0,0,0))
        p.setBrush(QBrush(center_grad))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(cx,cy), glow_radius, glow_radius)

        total_height = h*0.6
        top_y        = (h-total_height)*0.5
        bar_area_h   = total_height / bar_count
        bar_gap      = bar_area_h*0.15
        bar_h        = bar_area_h - bar_gap

        max_bar_len = w*0.4
        left_origin_x  = 0.0
        right_origin_x = w

        p.setRenderHint(QPainter.Antialiasing, False)

        # scale factor for bar length
        # was 3.0, now 2.7 (~10% less) so bars don't hit center instantly
        scale_factor = 2.7

        # LEFT side bars -> >>> center
        for i,val in enumerate(_left_vals):
            lvl = max(0.0,min(1.0,val*scale_factor))
            cur_len = max_bar_len * lvl
            y = top_y + i*bar_area_h
            bar_rect = QRectF(left_origin_x, y, cur_len, bar_h)

            hue = int(((i/float(bar_count))*200.0 + t*10.0) % 360)
            col = QColor.fromHsv(hue, 255, 255)
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawRect(bar_rect)

            # peak cap matches scaling
            pk_lvl = max(0.0,min(1.0,_left_peaks[i]*scale_factor))
            pk_x = left_origin_x + max_bar_len*pk_lvl
            cap_w = 4.0
            cap_col = QColor(255,255,255)
            p.setBrush(QBrush(cap_col))
            p.drawRect(QRectF(pk_x-cap_w*0.5, y, cap_w, bar_h))

        # RIGHT side bars -> <<< center
        for i,val in enumerate(_right_vals):
            lvl = max(0.0,min(1.0,val*scale_factor))
            cur_len = max_bar_len * lvl
            y = top_y + i*bar_area_h
            bar_rect = QRectF(right_origin_x - cur_len, y, cur_len, bar_h)

            hue = int(((i/float(bar_count))*200.0 + t*10.0 + 120.0) % 360)
            col = QColor.fromHsv(hue, 255, 255)
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawRect(bar_rect)

            pk_lvl = max(0.0,min(1.0,_right_peaks[i]*scale_factor))
            pk_x = right_origin_x - max_bar_len*pk_lvl
            cap_w = 4.0
            cap_col = QColor(255,255,255)
            p.setBrush(QBrush(cap_col))
            p.drawRect(QRectF(pk_x-cap_w*0.5, y, cap_w, bar_h))

        # center guide line intentionally removed
