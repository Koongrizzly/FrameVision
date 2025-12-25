from math import sin, cos, pi
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPainterPath
from PySide6.QtCore import QRectF, QPointF, Qt
from helpers.music import register_visualizer, BaseVisualizer


def _split(bands):
    if not bands:
        return 0.0, 0.0, 0.0
    n = len(bands)
    a = max(1, n // 6)
    b = max(a + 1, n // 2)
    lo = sum(bands[:a]) / a
    mid = sum(bands[a:b]) / max(1, (b - a))
    hi = sum(bands[b:]) / max(1, (n - b))
    return lo, mid, hi


def _env_step(env, target, up=0.5, down=0.25):
    """Simple envelope follower: faster attack than release."""
    return (1 - up) * env + up * target if target > env else (1 - down) * env + down * target


@register_visualizer
class DJMascot(BaseVisualizer):
    display_name = "DJ Mascot"

    def __init__(self):
        super().__init__()
        self._env_lo = 0.0
        self._env_mid = 0.0
        self._env_hi = 0.0
        self._bounce_phase = 0.0
        self._arm_phase = 0.0

    def paint(self, p: QPainter, r, bands, rms, t):
        w, h = int(r.width()), int(r.height())
        if w <= 0 or h <= 0:
            return

        p.setRenderHint(QPainter.Antialiasing, True)

        # --- Background ---
        base_bg = QColor(3, 6, 14)
        p.fillRect(r, base_bg)

        # Slight vignette based on RMS
        vignette_strength = min(0.85, 0.35 + rms * 0.9)
        vignette_color = QColor(0, 0, 0, int(255 * vignette_strength))
        p.fillRect(r.adjusted(10, 10, -10, -10), vignette_color)

        # --- Audio envelopes ---
        lo, mid, hi = _split(bands or [])
        self._env_lo = _env_step(self._env_lo, lo + 0.8 * rms, 0.6, 0.22)
        self._env_mid = _env_step(self._env_mid, mid, 0.55, 0.2)
        self._env_hi = _env_step(self._env_hi, hi, 0.65, 0.3)

        # Phases for animation
        dt = 1 / 60.0
        self._bounce_phase += 4.0 * dt * (1.0 + 1.5 * self._env_lo)
        self._arm_phase += 5.0 * dt * (1.0 + 2.0 * self._env_mid)

        # --- Layout / mascot metrics ---
        cx, cy = w * 0.5, h * 0.58

        scale = min(w, h) * 0.22
        head_r = scale * 0.28
        body_w = scale * 0.8
        body_h = scale * 0.95
        leg_len = scale * 0.7
        arm_len = scale * 0.85

        # Bounce offset from low band
        bounce_amp = scale * 0.18 * min(1.3, 0.7 + self._env_lo * 1.8)
        bounce = sin(self._bounce_phase * 2 * pi) * bounce_amp * self._env_lo

        mascot_y = cy - bounce

        # --- Draw subtle stage under mascot ---
        stage_r = scale * 1.3
        stage_rect = QRectF(cx - stage_r, mascot_y + body_h * 0.5 + leg_len * 0.65,
                            stage_r * 2, stage_r * 0.6)
        grad_alpha = int(70 + 120 * self._env_lo)
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(18, 24, 40, grad_alpha)))
        p.drawEllipse(stage_rect)

        # --- Floor lights / beams reacting to low band (behind booth & mascot) ---
        p.setPen(Qt.NoPen)
        beam_base_y = stage_rect.bottom() - stage_rect.height() * 0.35
        beam_h = h * 0.35  # a bit shorter so they sit higher and don't cover the whole mascot
        beam_count = 6
        for i in range(beam_count):
            k = (i - (beam_count - 1) / 2.0) / max(1.0, (beam_count - 1) * 0.7)
            x = cx + k * stage_r * 1.3
            energy = max(0.0, self._env_lo * (0.5 + 0.8 * sin(t * 3.0 + k * 2.4)))
            alpha = int(210 * energy)
            if alpha <= 5:
                continue
            beam_col = QColor.fromHsv(int((t * 50) + k * 90) % 360, 180, 220, alpha)
            p.setBrush(beam_col)
            path = QPainterPath()
            path.moveTo(x - 6, beam_base_y)
            path.lineTo(x + 6, beam_base_y)
            path.lineTo(cx + k * stage_r * 0.25, beam_base_y - beam_h)
            path.closeSubpath()
            p.drawPath(path)

        # --- DJ booth in front ---
        booth_w = scale * 2.2
        booth_h = scale * 0.55
        booth_rect = QRectF(cx - booth_w * 0.5,
                            mascot_y - body_h * 0.25,
                            booth_w,
                            booth_h)
        booth_col = QColor(12, 16, 28)
        p.setBrush(QBrush(booth_col))
        p.setPen(QPen(QColor(40, 50, 80), 2))
        p.drawRoundedRect(booth_rect, scale * 0.15, scale * 0.15)

        # Booth glow strip (reacts to mid band)
        glow_h = booth_h * 0.18
        glow_rect = QRectF(booth_rect.left() + booth_w * 0.06,
                           booth_rect.bottom() - glow_h * 1.4,
                           booth_w * 0.88,
                           glow_h)
        glow_sat = 180
        glow_val = int(120 + 120 * min(1.0, self._env_mid * 2.2))
        glow_hue = (int(t * 50) % 360)
        glow_col = QColor.fromHsv(glow_hue, glow_sat, glow_val, 230)
        p.setBrush(QBrush(glow_col))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(glow_rect, glow_h * 0.4, glow_h * 0.4)

        # Small "VU" bars on booth front (quick-reacting hi band)
        bar_count = 7
        bar_gap = glow_rect.width() / (bar_count * 1.5)
        bar_w = bar_gap * 0.7
        max_bar_h = glow_h * 0.7
        for i in range(bar_count):
            k = i / max(1, bar_count - 1)
            bar_amp = (0.4 + 0.7 * self._env_hi) * (0.4 + 0.9 * sin(k * pi + t * 6.0))
            bar_h = max_bar_h * max(0.1, bar_amp)
            bx = glow_rect.left() + bar_gap * 0.75 + i * bar_gap * 1.2
            by = glow_rect.bottom() - glow_h * 0.15
            rect = QRectF(bx, by - bar_h, bar_w, bar_h)
            bar_col = QColor.fromHsv((glow_hue + int(k * 80)) % 360,
                                     220,
                                     int(140 + 100 * k),
                                     240)
            p.setBrush(bar_col)
            p.drawRoundedRect(rect, bar_w * 0.4, bar_w * 0.4)

        # --- Mascot: body ---
        body_rect = QRectF(cx - body_w * 0.5,
                           mascot_y - body_h * 0.3,
                           body_w,
                           body_h)
        body_col = QColor(18, 26, 38)
        p.setBrush(QBrush(body_col))
        p.setPen(QPen(QColor(60, 80, 120), 2))
        p.drawRoundedRect(body_rect, body_w * 0.3, body_w * 0.3)

        # Chest equalizer strip
        chest_rect = QRectF(body_rect.left() + body_w * 0.18,
                            body_rect.top() + body_h * 0.28,
                            body_w * 0.64,
                            body_h * 0.18)
        p.setPen(Qt.NoPen)
        chest_bg = QColor(8, 10, 16, 230)
        p.setBrush(chest_bg)
        p.drawRoundedRect(chest_rect, chest_rect.height() * 0.5, chest_rect.height() * 0.5)

        # Moving equalizer lines on chest
        chest_bars = 10
        gap = chest_rect.width() / (chest_bars * 1.4)
        bw = gap * 0.65
        max_ch = chest_rect.height() * 0.85
        for i in range(chest_bars):
            k = i / max(1, chest_bars - 1)
            ph = t * 4.0 + k * pi * 1.5
            amp = 0.25 + 0.7 * (0.4 * self._env_lo + 0.9 * self._env_mid + 0.6 * self._env_hi)
            ch = max_ch * max(0.1, amp * (0.5 + 0.5 * sin(ph)))
            cx_bar = chest_rect.left() + gap * 0.7 + i * gap * 1.2
            cy_bar = chest_rect.center().y()
            rect = QRectF(cx_bar, cy_bar - ch * 0.5, bw, ch)
            bar_hue = (glow_hue + 40 + int(k * 90)) % 360
            bar_col = QColor.fromHsv(bar_hue, 210, 250, 235)
            p.setBrush(bar_col)
            p.drawRoundedRect(rect, bw * 0.5, bw * 0.5)

        # --- Mascot: head ---
        head_center = QPointF(cx, body_rect.top() - head_r * 0.35)
        head_color = QColor(12, 18, 30)
        p.setBrush(QBrush(head_color))
        p.setPen(QPen(QColor(70, 90, 130), 2))
        p.drawEllipse(head_center, head_r, head_r)

        # Headphones arc
        hp_r_outer = head_r * 1.15
        hp_r_inner = head_r * 0.82
        arc_rect = QRectF(head_center.x() - hp_r_outer,
                          head_center.y() - hp_r_outer,
                          hp_r_outer * 2,
                          hp_r_outer * 2)
        p.setPen(QPen(QColor(80, 120, 200), 4))
        p.setBrush(Qt.NoBrush)
        p.drawArc(arc_rect, int(210 * 16), int(120 * 16))

        # Ear pads
        pad_w = head_r * 0.35
        pad_h = head_r * 0.65
        pad_offset_x = head_r * 0.95
        for side in (-1, 1):
            pad_rect = QRectF(head_center.x() + side * (pad_offset_x - pad_w * 0.5),
                              head_center.y() - pad_h * 0.5,
                              pad_w,
                              pad_h)
            p.setBrush(QBrush(QColor(20, 30, 50)))
            p.setPen(QPen(QColor(90, 130, 210), 2))
            p.drawRoundedRect(pad_rect, pad_w * 0.4, pad_w * 0.4)

        # --- Eyes (LED style, react to hi band) ---
        eye_offset_x = head_r * 0.4
        eye_offset_y = head_r * -0.05
        eye_w = head_r * 0.28
        eye_h = head_r * 0.18

        blink = max(0.05, 0.6 + 0.4 * sin(t * 14.0))
        hi_boost = min(1.0, 0.4 + self._env_hi * 2.5)
        eye_h_current = eye_h * (0.25 + 0.75 * hi_boost * blink)

        eye_hue = (glow_hue + 180) % 360
        eye_val = int(180 + 70 * hi_boost)
        eye_col = QColor.fromHsv(eye_hue, 40, eye_val, 255)

        p.setBrush(QBrush(eye_col))
        p.setPen(Qt.NoPen)

        for side in (-1, 1):
            ex = head_center.x() + side * eye_offset_x - eye_w * 0.5
            ey = head_center.y() + eye_offset_y - eye_h_current * 0.5
            eye_rect = QRectF(ex, ey, eye_w, eye_h_current)
            p.drawRoundedRect(eye_rect, eye_h_current * 0.5, eye_h_current * 0.5)

        # Tiny mouth strip
        mouth_w = head_r * 0.45
        mouth_h = head_r * 0.07
        mouth_rect = QRectF(head_center.x() - mouth_w * 0.5,
                            head_center.y() + head_r * 0.35,
                            mouth_w,
                            mouth_h)
        mouth_col = QColor(40, 60, 90)
        p.setBrush(mouth_col)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(mouth_rect, mouth_h * 0.5, mouth_h * 0.5)

        # --- Arms over booth (waving to beat) ---
        # Anchor points at shoulders
        shoulder_y = body_rect.top() + body_h * 0.26
        left_shoulder = QPointF(body_rect.left() + body_w * 0.22, shoulder_y)
        right_shoulder = QPointF(body_rect.right() - body_w * 0.22, shoulder_y)

        # Angles influenced by mid and hi bands
        base_left_angle = -130  # degrees
        base_right_angle = -50
        wave_mid = 25 * sin(self._arm_phase * 2 * pi)
        extra_hi = 18 * self._env_hi

        left_angle = (base_left_angle + wave_mid * (0.4 + 0.6 * self._env_mid)
                      + extra_hi * 0.4)
        right_angle = (base_right_angle - wave_mid * (0.4 + 0.6 * self._env_mid)
                       - extra_hi * 0.7)

        def arm_end(origin: QPointF, angle_deg: float, length: float) -> QPointF:
            ang = angle_deg * pi / 180.0
            return QPointF(origin.x() + cos(ang) * length,
                           origin.y() + sin(ang) * length)

        arm_thickness = scale * 0.15
        arm_col = QColor(26, 36, 54)
        p.setPen(QPen(QColor(70, 90, 130), 3))
        p.setBrush(QBrush(arm_col))

        for origin, ang in ((left_shoulder, left_angle), (right_shoulder, right_angle)):
            hand = arm_end(origin, ang, arm_len * (0.9 + 0.4 * self._env_mid))
            path = QPainterPath()
            path.moveTo(origin)
            ctrl = QPointF((origin.x() + hand.x()) * 0.5,
                           origin.y() - arm_len * 0.25)
            path.quadTo(ctrl, hand)
            p.drawPath(path)

            # Simple rounded "hand" circle
            hand_r = arm_thickness * (0.9 + 0.4 * self._env_hi)
            p.setBrush(QBrush(QColor(80, 120, 210)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(hand, hand_r, hand_r)

        # --- Simple legs (mostly static, slight wobble with low band) ---
        leg_offset_x = body_w * 0.22
        leg_top_y = body_rect.bottom() - body_h * 0.04
        leg_thick = scale * 0.16

        wobble = sin(self._bounce_phase * 2 * pi + pi * 0.3) * self._env_lo * 0.25

        p.setBrush(QBrush(QColor(20, 28, 44)))
        p.setPen(QPen(QColor(50, 70, 110), 2))

        for side in (-1, 1):
            lx = cx + side * leg_offset_x - leg_thick * 0.5 + wobble * side * scale * 0.1
            leg_rect = QRectF(lx,
                              leg_top_y,
                              leg_thick,
                              leg_len)
            p.drawRoundedRect(leg_rect, leg_thick * 0.45, leg_thick * 0.45)
