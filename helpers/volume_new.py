from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QFrame, QToolButton, QCheckBox, QGridLayout, QMenu, QWidgetAction, QApplication
)

import atexit

BANDS = [("60 Hz", 60), ("230 Hz", 230), ("910 Hz", 910), ("3.6 kHz", 3600), ("14 kHz", 14000)]

def _resolve_audio(pane):
    for name in ("audio", "audio_output", "audioOutput", "player_audio", "audio_out"):
        if hasattr(pane, name):
            return getattr(pane, name)
    for name in ("player", "media_player", "video_player"):
        p = getattr(pane, name, None)
        if p and hasattr(p, "audioOutput"):
            try:
                a = p.audioOutput()
                if a: return a
            except Exception:
                pass
    return None

def _resolve_player(pane):
    for name in ("player", "media_player", "video_player"):
        if hasattr(pane, name):
            return getattr(pane, name)
    return None

def _ffmpeg_eq_filter(gains):
    freqs = [60, 230, 910, 3600, 14000]
    parts = []
    for f, g in zip(freqs, gains):
        parts.append(f"equalizer=f={f}:t=q:w=1:g={int(g)}")
    return ",".join(parts)

# --- global emergency cleanup at interpreter exit ---
def _atexit_cleanup():
    try:
        from . import eq_ffmpeg
        eq_ffmpeg.stop()
    except Exception:
        pass
atexit.register(_atexit_cleanup)


class _MenuPopupWidget(QWidget):
    def __init__(self, pane, parent=None):
        super().__init__(parent)
        from . import eq_ffmpeg  # sidecar module
        self.eq_ffmpeg = eq_ffmpeg
        self.pane = pane
        self._eq_enabled = False      # actual sidecar running
        self._eq_armed = True         # desired state (EQ "on" even if no media yet)

        # start-gate state (prevents blips and first-second peaks)
        self._gate_active = False
        self._gate_t_play = QTimer(self); self._gate_t_play.setSingleShot(True)
        self._gate_t_play.timeout.connect(self._finalize_gate)
        self._gate_t_force = QTimer(self); self._gate_t_force.setSingleShot(True)
        self._gate_t_force.timeout.connect(self._finalize_gate)
        self._vol_guard = QTimer(self); self._vol_guard.setSingleShot(False)
        self._vol_guard.timeout.connect(self._enforce_desired_volume)

        # startup EQ default enforcement (to override late settings loaders)
        self._eq_default_enforce = True
        self._eq_default_timer = QTimer(self); self._eq_default_timer.setInterval(150); self._eq_default_timer.timeout.connect(self._enforce_eq_default)
        self._eq_default_deadline_ms = 2000  # enforce for ~2s max
        self._internal_toggle = False  # distinguishes programmatic vs user

        frame = QFrame(self); frame.setObjectName("fv_menu_volume_frame")
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.addWidget(frame)
        v = QVBoxLayout(frame); v.setContentsMargins(12,12,12,12); v.setSpacing(10)

        # Top: Volume + Mute
        top = QHBoxLayout(); top.setSpacing(8)
        vol_label = QLabel("Volume"); vol_label.setObjectName("fv_vol_label")
        top.addWidget(vol_label)
        self.vol = QSlider(Qt.Horizontal); self.vol.setRange(0,100); self.vol.setValue(100); top.addWidget(self.vol, 1)
        self.mute = QCheckBox("Mute"); top.addWidget(self.mute)
        v.addLayout(top)

        # EQ grid
        grid = QGridLayout(); grid.setHorizontalSpacing(16); grid.setVerticalSpacing(6)
        self.eq = []
        for i,(label,_freq) in enumerate(BANDS):
            sv = QSlider(Qt.Vertical); sv.setRange(-12,12); sv.setValue(0); self.eq.append(sv)
            grid.addWidget(sv,0,i,alignment=Qt.AlignHCenter|Qt.AlignBottom)
            lb = QLabel(label); lb.setObjectName('fv_freq_label'); lb.setAlignment(Qt.AlignHCenter); grid.addWidget(lb,1,i,alignment=Qt.AlignHCenter)
        v.addLayout(grid)

        # Bottom: Toggle EQ + Reset
        bottom = QHBoxLayout(); bottom.addStretch(1)
        self.toggle_btn = QPushButton("EQ")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setObjectName("fv_eq_toggle")
        self.reset = QPushButton("Reset EQ")
        bottom.addWidget(self.toggle_btn)
        bottom.addWidget(self.reset)
        bottom.addStretch(1)
        v.addLayout(bottom)

        # Debouncers
        self._eq_apply_timer = QTimer(self); self._eq_apply_timer.setSingleShot(True)
        self._eq_apply_timer.timeout.connect(self._apply_eq_now)

        self._reapply_timer = QTimer(self); self._reapply_timer.setSingleShot(True)
        self._reapply_timer.timeout.connect(lambda: self.reapply_to_audio(True, reason="debounced"))

        # Wire
        self.vol.valueChanged.connect(self._on_volume_changed)
        self.mute.toggled.connect(self._on_mute_toggled)
        self.reset.clicked.connect(self._reset_eq)
        self.toggle_btn.toggled.connect(self._toggle_eq)
        for idx, sv in enumerate(self.eq):
            sv.valueChanged.connect(lambda _g, i=idx: self._on_eq_changed())

        # init from audio
        self._sync_from_audio()

        # DEFAULT: EQ ON (armed) at app start regardless of media presence
        self._internal_toggle = True
        self.toggle_btn.setChecked(True)
        self._internal_toggle = False
        self._eq_armed = True

        self._update_visual_state()

        # style
        self.setStyleSheet(
            "#fv_menu_volume_frame{background: palette(window);border: 1px solid palette(mid);border-radius: 12px;}"
            "QLabel,QCheckBox{color: palette(window-text); background: transparent;}"
            "QLabel#fv_freq_label{font-weight:600; font-size:13px;}"
            "QLabel#fv_vol_label{font-weight:700; font-size:13px;}"
            "QPushButton{background: palette(button);color: palette(window-text);border: 1px solid palette(mid);border-radius: 8px; padding: 6px 12px;}"
            "QPushButton:hover{background: palette(light);}"
            "QPushButton:pressed{background: palette(dark);}"
            "QPushButton:checked{background: palette(midlight); border-color: palette(highlight);}"
            "QToolTip{color: palette(toolTipText); background-color: palette(toolTipBase); border: 1px solid palette(mid);}"
            "QPushButton#fv_eq_toggle { border-radius: 8px; font-weight: 600; }"
            "QPushButton#fv_eq_toggle:checked { background: #16a34a; color: white; }"
        )
        self.setMinimumWidth(420)

        # publish initial visual state
        self._publish_visual_state()

        # --- Reapply UI state automatically when media changes/loads ---
        self._install_media_change_hooks()

        # Ensure the current UI state is applied immediately with gate
        self.schedule_reapply(0)

        # Start EQ default enforcement window
        self._eq_default_timer.start()

    # ---------- Helpers to detect media readiness ----------
    def _media_ready(self):
        """Return True if player likely has a bound source capable of audio output."""
        p = _resolve_player(self.pane)
        if p is None:
            # If we at least have an audio output, we might still be okay
            return _resolve_audio(self.pane) is not None
        try:
            # Heuristic: if position exists or duration>0 or status indicates loaded/buffered
            if hasattr(p, "position") and callable(p.position) and p.position() > 0:
                return True
        except Exception:
            pass
        for attr in ("duration", "mediaStatus"):
            try:
                val = getattr(p, attr)() if callable(getattr(p, attr, None)) else getattr(p, attr, None)
                if val and int(val) > 0:
                    return True
            except Exception:
                pass
        # Fallback: if we have an audio output, allow applying; sidecar may still refuse — that's okay
        return _resolve_audio(self.pane) is not None

    # ---------- EQ default enforcer ----------
    def _enforce_eq_default(self):
        if not self._eq_default_enforce:
            self._eq_default_timer.stop()
            return
        self._eq_default_deadline_ms -= self._eq_default_timer.interval()
        if self._eq_default_deadline_ms <= 0:
            self._eq_default_timer.stop()
            self._eq_default_enforce = False
            return
        # Keep toggle visually ON
        if not self.toggle_btn.isChecked():
            self._internal_toggle = True
            self.toggle_btn.setChecked(True)
            self._internal_toggle = False
        # Arm EQ (apply when ready)
        self._eq_armed = True
        # Try to apply if media is ready
        if self._media_ready():
            self._apply_eq_pipeline(force_zero=True)

    # ---------- Start-gate logic to eliminate blips ----------
    def _begin_gate(self):
        if self._gate_active:
            return
        self._gate_active = True

        a = self._audio()
        if a:
            try: a.setMuted(True)
            except Exception: pass
            try: a.setVolume(0.0)
            except Exception: pass

        try: self.eq_ffmpeg.stop()
        except Exception: pass

        # If armed, prep pipeline at zero even if media not yet ready (sidecar may refuse; that's fine)
        if self._eq_armed:
            self._apply_eq_pipeline(force_zero=True)

        self._gate_t_force.start(1200)

    def _finalize_gate(self):
        if not self._gate_active:
            return
        self._gate_active = False
        try: self._gate_t_force.stop()
        except Exception: pass
        try: self._gate_t_play.stop()
        except Exception: pass

        a = self._audio()
        desired_muted = bool(self.mute.isChecked())
        desired_volume = max(0.0, min(1.0, float(self.vol.value())/100.0))

        if a:
            try: a.setVolume(0.0 if desired_muted else desired_volume)
            except Exception: pass
            try: a.setMuted(desired_muted)
            except Exception: pass

        if not desired_muted:
            try: self.eq_ffmpeg.unmute_qt(self.pane)
            except Exception: pass

        try:
            self._guard_deadline_ms = 900
            self._vol_guard.start(60)
        except Exception:
            pass

        self._publish_visual_state()
        self._update_visual_state()

    def _enforce_desired_volume(self):
        try:
            self._guard_deadline_ms -= 60
            if self._guard_deadline_ms <= 0:
                self._vol_guard.stop()
        except Exception:
            try: self._vol_guard.stop()
            except Exception: pass
            return

        a = self._audio()
        if not a:
            return
        desired_muted = bool(self.mute.isChecked())
        desired_volume = max(0.0, min(1.0, float(self.vol.value())/100.0))

        try:
            cur_vol = 0.0
            try: cur_vol = float(a.volume())
            except Exception: pass
            if desired_muted:
                if cur_vol != 0.0:
                    a.setVolume(0.0)
                if hasattr(a, "setMuted"):
                    a.setMuted(True)
            else:
                if abs(cur_vol - desired_volume) > 0.005:
                    a.setVolume(desired_volume)
                if hasattr(a, "setMuted"):
                    a.setMuted(False)
        except Exception:
            pass

    # ---------- Reapply orchestration ----------
    def schedule_reapply(self, delay_ms=120):
        try:
            self._reapply_timer.stop()
            self._reapply_timer.start(max(0, int(delay_ms)))
        except Exception:
            self.reapply_to_audio(True, reason="fallback")

    def reapply_to_audio(self, also_eq=True, reason="manual"):
        self._begin_gate()
        if also_eq and self._eq_armed:
            # Apply only when ready; otherwise stay armed
            if self._media_ready():
                self._apply_eq_pipeline(force_zero=self.mute.isChecked() or self._gate_active)
        self._publish_visual_state()
        self._update_visual_state()

    # ---------- Media hooks ----------
    def _install_media_change_hooks(self):
        if getattr(self.pane, "_fv_media_hooks_installed_v5_eq_armed", False):
            return
        setattr(self.pane, "_fv_media_hooks_installed_v5_eq_armed", True)

        p = _resolve_player(self.pane)
        if p is None:
            return

        def _on_status_change(*_):
            self.schedule_reapply(0)

        def _on_playback_change(*_):
            try:
                self._gate_t_play.start(150)
            except Exception:
                self._finalize_gate()

        def _on_position_changed(pos_ms):
            if pos_ms and int(pos_ms) > 40:
                self._finalize_gate()

        for sig_name, handler in (
            ("mediaStatusChanged", _on_status_change),
            ("sourceChanged", _on_status_change),
            ("currentMediaChanged", _on_status_change),
            ("mediaChanged", _on_status_change),
            ("playbackStateChanged", _on_playback_change),
            ("positionChanged", _on_position_changed),
        ):
            try:
                getattr(p, sig_name).connect(handler)
            except Exception:
                pass

        # Guard pane.open as well
        try:
            orig_open = getattr(self.pane, "open")
            if callable(orig_open) and not getattr(orig_open, "_fv_eq_wrapped_gate_v5_eq_armed", False):
                def wrapped_open(*args, **kwargs):
                    self._begin_gate()
                    try: self.eq_ffmpeg.stop()
                    except Exception: pass
                    res = orig_open(*args, **kwargs)
                    self.schedule_reapply(0)
                    return res
                wrapped_open._fv_eq_wrapped_gate_v5_eq_armed = True
                setattr(self.pane, "open", wrapped_open)
        except Exception:
            pass

    # ---------- EQ pipeline helpers ----------
    def _apply_eq_pipeline(self, force_zero=False):
        """Try to start/update the sidecar pipeline. Never flips the toggle OFF on failure.
        Keeps _eq_armed True and sets _eq_enabled based on success."""
        try:
            ok = self.eq_ffmpeg.apply_filter(self.pane, self._filter_chain(force_volume_zero=force_zero))
            self._eq_enabled = bool(ok)
            # Visual stays ON regardless; _eq_enabled just tracks running state
            self._apply_toggle_style(True)
        except Exception:
            self._eq_enabled = False
            # Keep toggle visually ON if armed
            if self._eq_armed:
                self._apply_toggle_style(True)

    # ----- toggle logic -----
    def _apply_toggle_style(self, on):
        self.toggle_btn.setChecked(bool(on))

    def _toggle_eq(self, on):
        # If user toggles during enforcement, stop enforcing
        if not self._internal_toggle and self._eq_default_enforce:
            self._eq_default_enforce = False
            try: self._eq_default_timer.stop()
            except Exception: pass

        if on:
            # Arm the EQ even if no media yet; do not revert toggle on failure
            self._eq_armed = True
            if self._media_ready():
                self._apply_eq_pipeline(force_zero=self.mute.isChecked() or self._gate_active)
            else:
                # Not ready: keep armed, will auto-apply on first track
                self._eq_enabled = False
                self._apply_toggle_style(True)
        else:
            # Disarm & stop sidecar
            self._eq_armed = False
            try: self.eq_ffmpeg.stop()
            except Exception: pass
            if not self.mute.isChecked():
                try: self.eq_ffmpeg.unmute_qt(self.pane)
                except Exception: pass
            self._eq_enabled = False
            self._apply_toggle_style(False)
        self._update_visual_state()

    # ----- UI change handlers -----
    def _on_volume_changed(self, _):
        a = self._audio()
        try:
            if a and hasattr(a,"setVolume"):
                a.setVolume(0.0 if self.mute.isChecked() else float(self.vol.value())/100.0)
        except Exception: pass
        self._schedule_eq_apply()
        self._publish_visual_state(); self._update_visual_state()

    def _on_mute_toggled(self, st):
        a = self._audio()
        try:
            if a and hasattr(a,"setMuted"):
                a.setMuted(bool(st))
        except Exception: pass
        try:
            if a and hasattr(a,"setVolume"):
                a.setVolume(0.0 if st else float(self.vol.value())/100.0)
        except Exception: pass
        self._schedule_eq_apply()
        if not st and not self._gate_active:
            try: self.eq_ffmpeg.unmute_qt(self.pane)
            except Exception: pass
        self._publish_visual_state(); self._update_visual_state()

    def _on_eq_changed(self):
        self._schedule_eq_apply()
        self._publish_visual_state(); self._update_visual_state()

    def _schedule_eq_apply(self):
        try:
            self._eq_apply_timer.stop()
            self._eq_apply_timer.start(80)
        except Exception:
            self._apply_eq_now()

    def _apply_eq_now(self):
        if self._eq_armed:
            self._apply_eq_pipeline(force_zero=self.mute.isChecked() or self._gate_active)

    def _reset_eq(self):
        for s in self.eq:
            s.blockSignals(True); s.setValue(0); s.blockSignals(False)
        self._schedule_eq_apply()
        self._publish_visual_state(); self._update_visual_state()

    # ----- misc helpers -----
    def _update_visual_state(self):
        try:
            vol = max(0.0, min(1.0, float(self.vol.value())/100.0))
            mute = bool(self.mute.isChecked())
            gains = [int(s.value()) for s in self.eq]
            eq_on = bool(self._eq_armed)  # reflect desired state, not sidecar running
            self.eq_ffmpeg.set_visual_from_ui(volume=vol, mute=mute, gains=gains, eq_on=eq_on)
        except Exception:
            pass

    def _publish_visual_state(self):
        try:
            p = self.pane
            p._fv_visual_gain = max(0.0, min(1.0, float(self.vol.value())/100.0)) if not self.mute.isChecked() else 0.0
            p._fv_visual_mute = bool(self.mute.isChecked())
            p._fv_eq_freqs = [f for (_label, f) in BANDS]
            p._fv_eq_gains_db = [int(s.value()) for s in self.eq]
        except Exception:
            pass

    def _audio(self):
        return _resolve_audio(self.pane)

    def _sync_from_audio(self):
        a = self._audio()
        try:
            if a and hasattr(a,"volume"):
                self.vol.setValue(int(round((a.volume() or 1.0)*100)))
        except Exception: pass
        try:
            if a and hasattr(a,"isMuted"):
                self.mute.setChecked(bool(a.isMuted()))
        except Exception: pass

    def _gains(self):
        return [int(s.value()) for s in self.eq]

    def _filter_chain(self, force_volume_zero=False):
        gains = self._gains()
        eq = _ffmpeg_eq_filter(gains)
        if force_volume_zero:
            vol_filter = "volume=0.0"
        else:
            vol = 0 if self.mute.isChecked() else max(0, min(100, int(self.vol.value())))
            if vol == 100: vol_filter = "volume=1.0"
            elif vol == 0: vol_filter = "volume=0.0"
            else: vol_filter = f"volume={vol/100.0:.3f}"
        return f"{eq},{vol_filter}" if eq else vol_filter


def _install_cleanup_strong(pane, btn):
    if getattr(pane, "_fv_eq_cleanup_installed_v5_eq_armed", False):
        return
    setattr(pane, "_fv_eq_cleanup_installed_v5_eq_armed", True)

    try:
        from . import eq_ffmpeg
    except Exception:
        return

    app = QApplication.instance()
    if app is not None:
        try:
            app.aboutToQuit.connect(lambda: (eq_ffmpeg.stop(), (eq_ffmpeg.unmute_qt(pane) if not getattr(pane, "volume_popup_widget", None) or not pane.volume_popup_widget.mute.isChecked() else None)))
        except Exception:
            pass

    win = btn.window()
    if win is not None:
        try:
            _orig_close = getattr(win, "closeEvent", None)
            def _wrapped(ev):
                try:
                    eq_ffmpeg.stop()
                    if not getattr(pane, "volume_popup_widget", None) or not pane.volume_popup_widget.mute.isChecked():
                        eq_ffmpeg.unmute_qt(pane)
                except Exception:
                    pass
                if callable(_orig_close):
                    return _orig_close(ev)
            win.closeEvent = _wrapped
        except Exception:
            pass

    try:
        stop_btn = getattr(pane, "btn_stop", None)
        if stop_btn is not None:
            stop_btn.clicked.connect(lambda: (eq_ffmpeg.stop(), (eq_ffmpeg.unmute_qt(pane) if not getattr(pane, "volume_popup_widget", None) or not pane.volume_popup_widget.mute.isChecked() else None)))
    except Exception:
        pass

    try:
        orig_open = getattr(pane, "open")
        if callable(orig_open) and not getattr(orig_open, "_fv_eq_wrapped_gate_v5_eq_armed", False):
            def wrapped_open(*args, **kwargs):
                w = getattr(pane, "volume_popup_widget", None)
                if w and hasattr(w, "_begin_gate"):
                    w._begin_gate()
                try: eq_ffmpeg.stop()
                except Exception: pass
                res = orig_open(*args, **kwargs)
                if w and hasattr(w, "schedule_reapply"):
                    w.schedule_reapply(0)
                return res
            wrapped_open._fv_eq_wrapped_gate_v5_eq_armed = True
            setattr(pane, "open", wrapped_open)
    except Exception:
        pass

    try:
        pane.destroyed.connect(lambda _=None: eq_ffmpeg.stop())
    except Exception:
        pass


def add_new_volume_popup(pane, bar_layout):
    for attr in ("btn_volume_new", "btn_volume"):
        old = getattr(pane, attr, None)
        try:
            if old: old.setParent(None); old.deleteLater()
        except Exception:
            pass

    btn = QToolButton(pane); btn.setObjectName("btn_volume_new")
    try: btn.setAutoRaise(True)
    except Exception: pass
    try: btn.setFocusPolicy(Qt.NoFocus)
    except Exception: pass
    try: btn.setCursor(Qt.PointingHandCursor)
    except Exception: pass
    btn.setText("🔊"); btn.setToolTip("Volume / EQ")
    try: btn.setFixedSize(48,48)
    except Exception: pass

    menu = QMenu(btn)
    w = _MenuPopupWidget(pane, parent=menu)
    wa = QWidgetAction(menu); wa.setDefaultWidget(w)
    menu.addAction(wa)
    btn.setMenu(menu)
    btn.setPopupMode(QToolButton.InstantPopup)
    btn.setStyleSheet("QToolButton#btn_volume_new{background:transparent;border:none;padding:0px;}"
                      "QToolButton#btn_volume_new:hover{background:rgba(0,0,0,0.08);border:none;}"
                      "QToolButton#btn_volume_new:pressed{background:rgba(0,0,0,0.16);border:none;}"
                      "QToolButton#btn_volume_new:checked{background:rgba(0,0,0,0.12);border:none;}"
                      "QToolButton::menu-indicator{image:none;width:0px;height:0px;}"
                      "QMenu{background:transparent;border:0px;}")

    bar_layout.addWidget(btn)

    _install_cleanup_strong(pane, btn)

    pane.btn_volume_new = btn
    pane.volume_popup_menu = menu
    pane.volume_popup_widget = w

    try:
        w.schedule_reapply(0)
    except Exception:
        pass

    return btn
