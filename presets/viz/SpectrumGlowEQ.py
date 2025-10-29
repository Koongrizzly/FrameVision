import math
from PySide6.QtGui import QPainter, QPen, QColor, QBrush
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer

# --- persistent state ---
_last_t = None
_smooth_vals = []      # smoothed bar heights
_peak_vals = []        # peak-hold/decay heads

def _downsample(bands, count):
    """Squash the full FFT 'bands' list into 'count' averaged buckets."""
    if count <= 0:
        return []
    if not bands:
        return [0.0]*count
    n = len(bands)
    out = []
    for i in range(count):
        start = int((i    ) * n / count)
        end   = int((i + 1) * n / count)
        if end <= start:
            end = min(start+1, n)
        seg = bands[start:end]
        out.append(sum(seg)/len(seg))
    return out

@register_visualizer
class SpectrumGlowEQ(BaseVisualizer):
    """
    Clean rainbow EQ:
      - dark background only
      - rainbow bars + white peak caps
      - smoothed rise/fall, peak-hold decay
    Tweaked: bar scale reduced ~10% so they don't slam the top so easily.
    """

    display_name = "Spectrum Glow EQ"

    def __init__(self):
        try:
            super().__init__()
        except Exception:
            pass

    def paint(self, p: QPainter, r, bands, rms, t):
        global _last_t, _smooth_vals, _peak_vals

        w = float(r.width())
        h = float(r.height())
        if w <= 0 or h <= 0:
            return

        # timing
        if _last_t is None:
            _last_t = t
        dt = max(0.0, min(0.05, t - _last_t))
        _last_t = t

        # --- audio to bars ---
        target_bar_count = 48
        vals_now = _downsample(bands, target_bar_count)

        if len(_smooth_vals) != target_bar_count:
            _smooth_vals = [0.0]*target_bar_count
        if len(_peak_vals) != target_bar_count:
            _peak_vals = [0.0]*target_bar_count

        attack = 0.5
        decay  = 0.2
        peak_fall = 0.6  # per-second fall speed for the cap

        for i, v in enumerate(vals_now):
            cur = _smooth_vals[i]
            if v > cur:
                cur = (1.0-attack)*cur + attack*v
            else:
                cur = (1.0-decay)*cur + decay*v
            _smooth_vals[i] = cur

            # update peak cap
            if cur > _peak_vals[i]:
                _peak_vals[i] = cur
            else:
                _peak_vals[i] = max(0.0, _peak_vals[i] - peak_fall*dt)

        # --- background: just dark ---
        p.fillRect(r, QColor(5,5,10))

        # --- draw EQ bars ---
        eq_top    = h*0.55
        eq_bottom = h*0.98
        eq_height = eq_bottom - eq_top
        left_pad  = w*0.03
        right_pad = w*0.03
        usable_w  = w - left_pad - right_pad

        bar_w     = usable_w / target_bar_count * 0.7
        gap_w     = usable_w / target_bar_count * 0.3

        p.setRenderHint(QPainter.Antialiasing, False)

        # scale factors:
        # previously: level = clamp(cur*2.5, 0..1)
        # now: 0.9 * that, so ~10% lower ceiling
        scale_factor = 2.5 * 0.9  # == 2.25 effectively
        for i, cur in enumerate(_smooth_vals):
            level = max(0.0, min(1.0, cur*scale_factor))
            bar_h = eq_height * level
            x0 = left_pad + i*(bar_w+gap_w)
            y0 = eq_bottom - bar_h

            hue_i = (i/float(target_bar_count))*300.0 + t*10.0
            col   = QColor.fromHsv(int(hue_i)%360, 255, 255)
            p.setBrush(QBrush(col))
            p.setPen(Qt.NoPen)
            p.drawRect(QRectF(x0, y0, bar_w, bar_h))

            # peak cap uses same reduced scaling so it matches visual height
            pk_level = max(0.0, min(1.0, _peak_vals[i]*scale_factor))
            pk_y = eq_bottom - eq_height*pk_level
            cap_h = 3.0
            cap_col = QColor(255,255,255)
            p.setBrush(QBrush(cap_col))
            p.drawRect(QRectF(x0, pk_y - cap_h*0.5, bar_w, cap_h))
