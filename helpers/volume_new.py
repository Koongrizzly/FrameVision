
from PySide6.QtCore import Qt, QObject
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
        from . import eq_ffmpeg  # local sidecar
        self.eq_ffmpeg = eq_ffmpeg
        self.pane = pane
        self._eq_enabled = False  # using sidecar
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

        # Wire
        self.vol.valueChanged.connect(self._on_volume_changed)
        self.mute.toggled.connect(self._on_mute_toggled)
        self.reset.clicked.connect(self._reset_eq)
        self.toggle_btn.toggled.connect(self._toggle_eq)
        for idx, sv in enumerate(self.eq):
            sv.valueChanged.connect(lambda g, i=idx: self._on_eq_changed())

        # init from audio
        self._sync_from_audio()
        self._apply_toggle_style(False)
        self._update_visual_state()

        # style
        self.setStyleSheet(
            "#fv_menu_volume_frame{background: palette(window);border: 1px solid palette(mid);border-radius: 12px;}QLabel,QCheckBox{color: palette(window-text); background: transparent;}QLabel#fv_freq_label{font-weight:600; font-size:13px;}QLabel#fv_vol_label{font-weight:700; font-size:13px;}QPushButton{background: palette(button);color: palette(window-text);border: 1px solid palette(mid);border-radius: 8px; padding: 6px 12px;}QPushButton:hover{background: palette(light);}QPushButton:pressed{background: palette(dark);}QPushButton:checked{background: palette(midlight); border-color: palette(highlight);}QToolTip{color: palette(toolTipText); background-color: palette(toolTipBase); border: 1px solid palette(mid);}"
            ""
            ""
            "QPushButton#fv_eq_toggle { border-radius: 8px; font-weight: 600; }"
            "QPushButton#fv_eq_toggle:checked { background: #16a34a; color: white; }"  # green when ON
        )
        self.setMinimumWidth(420)

        # publish initial visual state
        self._publish_visual_state()

    def _update_visual_state(self):
        """Push volume/mute/EQ to the visualizer shared state (eq_ffmpeg) if available."""
        try:
            vol = max(0.0, min(1.0, float(self.vol.value())/100.0))
            mute = bool(self.mute.isChecked())
            gains = [int(s.value()) for s in self.eq]
            # EQ considered ON only when toggle is checked
            eq_on = bool(self.toggle_btn.isChecked()) if hasattr(self, "toggle_btnQLabel{background: transparent;}") else False
            self.eq_ffmpeg.set_visual_from_ui(volume=vol, mute=mute, gains=gains, eq_on=eq_on)
        except Exception:
            pass




    def _publish_visual_state(self):
        """Expose volume/mute/EQ to the visualizer via attributes on the pane.
        - _fv_visual_gain: 0..1
        - _fv_visual_mute: bool
        - _fv_eq_freqs: list of Hz
        - _fv_eq_gains_db: list of dB (match BANDS order)
        """
        try:
            p = self.pane
            p._fv_visual_gain = max(0.0, min(1.0, float(self.vol.value())/100.0)) if not self.mute.isChecked() else 0.0
            p._fv_visual_mute = bool(self.mute.isChecked())
            p._fv_eq_freqs = [f for (_label, f) in BANDS]
            p._fv_eq_gains_db = [int(s.value()) for s in self.eq]
        except Exception:
            pass
    # ----- helpers -----
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

    def _filter_chain(self):
        gains = self._gains()
        eq = _ffmpeg_eq_filter(gains)
        vol = 0 if self.mute.isChecked() else max(0, min(100, int(self.vol.value())))
        if vol == 100: vol_filter = "volume=1.0"
        elif vol == 0: vol_filter = "volume=0.0"
        else: vol_filter = f"volume={vol/100.0:.3f}"
        return f"{eq},{vol_filter}" if eq else vol_filter

    # ----- toggle logic -----
    def _apply_toggle_style(self, on):
        self.toggle_btn.setChecked(bool(on))

    def _toggle_eq(self, on):
        if on:
            ok = self.eq_ffmpeg.apply_filter(self.pane, self._filter_chain())
            self._eq_enabled = bool(ok)
            self._apply_toggle_style(self._eq_enabled)
            if not ok:
                # failed -> revert toggle
                self.toggle_btn.blockSignals(True)
                self.toggle_btn.setChecked(False)
                self.toggle_btn.blockSignals(False)
        else:
            self.eq_ffmpeg.stop()
            try: self.eq_ffmpeg.unmute_qt(self.pane)
            except Exception: pass
            self._eq_enabled = False
            self._apply_toggle_style(False)
        self._update_visual_state()

    def _on_volume_changed(self, _):
        if self._eq_enabled:
            # reapply current filter chain
            self.eq_ffmpeg.apply_filter(self.pane, self._filter_chain())
        else:
            a = self._audio()
            try:
                if a and hasattr(a,"setVolume"):
                    a.setVolume(float(self.vol.value())/100.0)
            except Exception: pass

        try:
            self._publish_visual_state()
        except Exception:
            pass
        self._update_visual_state()

    def _on_mute_toggled(self, st):
        if self._eq_enabled:
            self.eq_ffmpeg.apply_filter(self.pane, self._filter_chain())
        else:
            a = self._audio()
            try:
                if a and hasattr(a,"setMuted"):
                    a.setMuted(bool(st))
            except Exception: pass

        try:
            self._publish_visual_state()
        except Exception:
            pass
        self._update_visual_state()

    def _on_eq_changed(self):
        if self._eq_enabled:
            self.eq_ffmpeg.apply_filter(self.pane, self._filter_chain())

        try:
            self._publish_visual_state()
        except Exception:
            pass
        self._update_visual_state()

    def _reset_eq(self):
        for s in self.eq:
            s.blockSignals(True); s.setValue(0); s.blockSignals(False)
        if self._eq_enabled:
            self.eq_ffmpeg.apply_filter(self.pane, self._filter_chain())


        try:
            self._publish_visual_state()
        except Exception:
            pass
        self._update_visual_state()

def _install_cleanup_strong(pane, btn):
    """Install robust cleanup hooks to stop the ffplay sidecar in all scenarios."""
    # Avoid double-install
    if getattr(pane, "_fv_eq_cleanup_installed", False):
        return
    setattr(pane, "_fv_eq_cleanup_installed", True)

    try:
        from . import eq_ffmpeg
    except Exception:
        return

    # 1) App about-to-quit
    app = QApplication.instance()
    if app is not None:
        try:
            app.aboutToQuit.connect(lambda: (eq_ffmpeg.stop(), getattr(eq_ffmpeg, "unmute_qt", lambda _p: None)(pane)))
        except Exception:
            pass

    # 2) Main window closeEvent wrapper
    win = btn.window()
    if win is not None:
        try:
            _orig_close = getattr(win, "closeEvent", None)
            def _wrapped(ev):
                try:
                    eq_ffmpeg.stop()
                    try: eq_ffmpeg.unmute_qt(pane)
                    except Exception: pass
                except Exception:
                    pass
                if callable(_orig_close):
                    return _orig_close(ev)
            win.closeEvent = _wrapped
        except Exception:
            pass

    # 3) Stop button in the UI (if present)
    try:
        stop_btn = getattr(pane, "btn_stop", None)
        if stop_btn is not None:
            stop_btn.clicked.connect(lambda: (eq_ffmpeg.stop(), getattr(eq_ffmpeg, "unmute_qt", lambda _p: None)(pane)))
    except Exception:
        pass

    # 4) Wrap pane.open(path) to stop old sidecar before loading new media
    try:
        orig_open = getattr(pane, "open")
        if callable(orig_open) and not getattr(orig_open, "_fv_eq_wrapped", False):
            def wrapped_open(path):
                try:
                    eq_ffmpeg.stop()
                    try: eq_ffmpeg.unmute_qt(pane)
                    except Exception: pass
                except Exception:
                    pass
                return orig_open(path)
            wrapped_open._fv_eq_wrapped = True
            setattr(pane, "open", wrapped_open)
    except Exception:
        pass

    # 5) As a last resort, when pane is destroyed
    try:
        pane.destroyed.connect(lambda _=None: eq_ffmpeg.stop())
    except Exception:
        pass


def add_new_volume_popup(pane, bar_layout):
    # Remove any old/new button remnants
    for attr in ("btn_volume_new", "btn_volume"):
        old = getattr(pane, attr, None)
        try:
            if old: old.setParent(None); old.deleteLater()
        except Exception:
            pass

    btn = QToolButton(pane); btn.setObjectName("btn_volume_new")
    try:
        btn.setAutoRaise(True)
    except Exception:
        pass
    try:
        btn.setFocusPolicy(Qt.NoFocus)
    except Exception:
        pass
    try:
        btn.setCursor(Qt.PointingHandCursor)
    except Exception:
        pass
    btn.setText("🔊"); btn.setToolTip("Volume / EQ")
    try: btn.setFixedSize(48,48)
    except Exception: pass

    # Menu + embedded widget
    menu = QMenu(btn)
    w = _MenuPopupWidget(pane, parent=menu)
    wa = QWidgetAction(menu); wa.setDefaultWidget(w)
    menu.addAction(wa)
    btn.setMenu(menu)
    btn.setPopupMode(QToolButton.InstantPopup)
    btn.setStyleSheet("QToolButton#btn_volume_new{background:transparent;border:none;padding:0px;}""QToolButton#btn_volume_new:hover{background:rgba(0,0,0,0.08);border:none;}""QToolButton#btn_volume_new:pressed{background:rgba(0,0,0,0.16);border:none;}""QToolButton#btn_volume_new:checked{background:rgba(0,0,0,0.12);border:none;}""QToolButton::menu-indicator{image:none;width:0px;height:0px;}""QMenu{background:transparent;border:0px;}")

    bar_layout.addWidget(btn)

    # Install robust cleanup
    _install_cleanup_strong(pane, btn)

    pane.btn_volume_new = btn
    pane.volume_popup_menu = menu
    pane.volume_popup_widget = w
    try:
        w._publish_visual_state()
    except Exception:
        pass
    return btn


