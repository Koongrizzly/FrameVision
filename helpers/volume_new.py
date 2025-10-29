from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QFrame, QToolButton, QCheckBox, QGridLayout, QMenu, QWidgetAction
)

"""
volume_new.py (safe drop-in with persistent mute/volume, fast re-mute on track change)

Public surface is the SAME as your old file:
    - add_new_volume_popup(pane, bar_layout)
    - class _MenuPopupWidget
    - pane.btn_volume_new
    - pane.volume_popup_menu
    - pane.volume_popup_widget

Behavior guarantees:
    - We DO NOT pause playback
    - We DO NOT resume playback
    - We DO NOT seek / jump / scrub
    - We DO NOT wrap pane.open()
    - We DO NOT run gating logic / ffmpeg sidecar / volume guards

We ONLY:
    - Give you a popup with Volume, Mute, EQ sliders (placeholder EQ)
    - Apply volume/mute on the audio output object
    - Remember desired mute/volume and re-apply them across tracks

What's new vs previous version:
    - We keep global desired mute/volume and enforce them more aggressively.
    - Poll loop is now ~75ms instead of 500ms so you don't get ~0.5s of "leak."
    - We also hook player signals (mediaChanged etc.) so when a new track starts,
      we instantly re-mute if the user wanted mute.

EQ sliders:
    - Still visual only for now. They don't affect audio yet.
    - You can read them from:
        pane.volume_popup_widget.get_eq_gains_db()
"""

# 5-band EQ like the old visible layout you showed.
# We can expand to 12 bands later, easy.
BANDS = [
    ("60 Hz",   60),
    ("230 Hz",  230),
    ("910 Hz",  910),
    ("3.6 kHz", 3600),
    ("14 kHz",  14000),
]

# Global desired state so it survives track changes / new popup instances
_DESIRED_VOL_PCT = 100   # 0-100
_DESIRED_MUTED   = False # True if user hit Mute


def _resolve_audio(pane):
    """
    Find the audio output object we are allowed to control.
    We ONLY ever call setVolume()/setMuted() on this.
    We NEVER touch transport timing.
    """
    for name in ("audio", "audio_output", "audioOutput", "player_audio", "audio_out"):
        if hasattr(pane, name):
            a = getattr(pane, name)
            if a is not None:
                return a

    # fallback via player.audioOutput()
    for nm in ("player", "media_player", "video_player"):
        p = getattr(pane, nm, None)
        if p is not None and hasattr(p, "audioOutput"):
            try:
                a = p.audioOutput()
                if a:
                    return a
            except Exception:
                pass

    return None


def _resolve_player(pane):
    """
    Best guess at the playback object (QMediaPlayer-like).
    We do NOT control it. We ONLY listen for its signals so we can
    instantly re-apply mute/volume when the source changes.
    """
    for nm in ("player", "media_player", "video_player"):
        p = getattr(pane, nm, None)
        if p is not None:
            return p
    return None


class _MenuPopupWidget(QWidget):
    """
    Popup widget that lives inside the menu.

    Compatibility:
        - self.vol  (QSlider, horizontal)
        - self.mute (QCheckBox)
        - self.eq   (list of band sliders)
        - schedule_reapply()  (now harmless no-op)

    Safe behavior only:
        - NO pause/resume/seek logic
        - NO 150ms push
        - NO ffmpeg sidecar ownership
    """

    def __init__(self, pane, parent=None):
        super().__init__(parent)
        self.pane = pane

        # prevent feedback loops while syncing
        self._internal_change = False

        # mirror globals locally, and update globals when user changes things
        global _DESIRED_VOL_PCT, _DESIRED_MUTED
        self._desired_vol_pct = _DESIRED_VOL_PCT
        self._desired_muted   = _DESIRED_MUTED

        # ---- UI BUILD ----
        frame = QFrame(self)
        frame.setObjectName("fv_menu_volume_frame")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(frame)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        # Top row: Volume + Mute
        top = QHBoxLayout()
        top.setSpacing(8)

        vol_label = QLabel("Volume")
        vol_label.setObjectName("fv_vol_label")

        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(self._desired_vol_pct)

        self.mute = QCheckBox("Mute")
        self.mute.setChecked(self._desired_muted)

        top.addWidget(vol_label)
        top.addWidget(self.vol, 1)
        top.addWidget(self.mute)
        outer.addLayout(top)

        # EQ grid (UI only for now)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(6)
        self.eq = []

        for i, (label_text, freq) in enumerate(BANDS):
            sv = QSlider(Qt.Vertical)
            sv.setRange(-12, 12)
            sv.setValue(0)
            sv.setToolTip(f"{freq} Hz")
            self.eq.append(sv)

            grid.addWidget(
                sv,
                0, i,
                alignment=Qt.AlignHCenter | Qt.AlignBottom,
            )

            lab = QLabel(label_text)
            lab.setObjectName("fv_freq_label")
            lab.setAlignment(Qt.AlignHCenter)
            grid.addWidget(lab, 1, i, alignment=Qt.AlignHCenter)

        outer.addLayout(grid)

        # Bottom row: Reset EQ button
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.reset = QPushButton("Reset EQ")
        self.reset.setObjectName("fv_eq_reset")
        bottom.addWidget(self.reset)
        bottom.addStretch(1)
        outer.addLayout(bottom)

        # Styling close to your old vibe
        self.setStyleSheet(
            "#fv_menu_volume_frame{"
            " background: palette(window);"
            " border: 1px solid palette(mid);"
            " border-radius: 12px;"
            "}"
            "QLabel,QCheckBox{"
            " color: palette(window-text);"
            " background: transparent;"
            "}"
            "QLabel#fv_freq_label{"
            " font-weight:600; font-size:13px;"
            "}"
            "QLabel#fv_vol_label{"
            " font-weight:700; font-size:13px;"
            "}"
            "QPushButton{"
            " background: palette(button);"
            " color: palette(window-text);"
            " border: 1px solid palette(mid);"
            " border-radius: 8px;"
            " padding: 6px 12px;"
            "}"
            "QPushButton:hover{background: palette(light);}"
            "QPushButton:pressed{background: palette(dark);}"
        )
        self.setMinimumWidth(420)

        # ---- SIGNALS ----
        self.vol.valueChanged.connect(self._on_volume_changed)
        self.mute.toggled.connect(self._on_mute_toggled)
        self.reset.clicked.connect(self._on_reset_clicked)
        for sv in self.eq:
            sv.valueChanged.connect(self._on_eq_slider_changed)

        # Poll loop:
        #   runs fast (~75ms), constantly forcing the audio output to match
        #   "what the user wants" (desired mute/volume).
        #   This makes mute stick immediately on new tracks.
        self._poll = QTimer(self)
        self._poll.setSingleShot(False)
        self._poll.timeout.connect(self._sync_loop)
        self._poll.start(75)  # was 500ms before → caused ~0.5s leak

        # Hook into player events so we re-apply desired mute/vol
        # instantly when track/media changes (no audible blip).
        self._hook_player_signals()

        # Do one immediate sync/apply right now
        self._sync_loop()

    # --------------------------------------------------
    # Compatibility stub from original file
    def schedule_reapply(self, delay_ms=0):
        # old code used this to trigger EQ pipeline; safe no-op now
        return

    # --------------------------------------------------
    # Internal helpers
    def _audio(self):
        return _resolve_audio(self.pane)

    def _apply_desired_state_to_audio(self):
        """
        Enforce our desired mute/volume on the *current* audio output.
        We call this a LOT, but it's lightweight.
        """
        a = self._audio()
        if not a:
            return

        effective_pct = 0 if self._desired_muted else self._desired_vol_pct
        if effective_pct < 0:
            effective_pct = 0
        if effective_pct > 100:
            effective_pct = 100

        effective_vol = float(effective_pct) / 100.0

        # set muted first if available
        try:
            if hasattr(a, "setMuted"):
                a.setMuted(bool(self._desired_muted))
        except Exception:
            pass

        # then force volume
        try:
            if hasattr(a, "setVolume"):
                a.setVolume(effective_vol)
        except Exception:
            pass

    def _sync_ui_from_desired(self):
        """
        Make sure the popup UI matches what we *want*, not what backend reset to.
        Backend is forced to follow us anyway.
        """
        if self.vol.value() != self._desired_vol_pct:
            self._internal_change = True
            try:
                self.vol.setValue(self._desired_vol_pct)
            finally:
                self._internal_change = False

        if self.mute.isChecked() != self._desired_muted:
            self._internal_change = True
            try:
                self.mute.setChecked(self._desired_muted)
            finally:
                self._internal_change = False

    def _sync_loop(self):
        """
        Runs every ~75ms:
        1. Apply desired mute/volume to the audio output.
        2. Reflect those desired values back into the UI.
        This fixes:
           - mute only lasting one track
           - "brief leak" of next track before mute hits
        """
        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    def _hook_player_signals(self):
        """
        Connect to player events (if they exist) so we immediately enforce
        mute/volume the moment a new file loads or playback state changes.

        We STILL do not pause/resume/seek. We only re-apply volume/mute.
        """
        p = _resolve_player(self.pane)
        if p is None:
            return

        def _reapply_from_player_event(*_):
            # When the source changes or playback restarts,
            # slam desired mute/volume immediately.
            self._apply_desired_state_to_audio()
            self._sync_ui_from_desired()

        for sig_name in (
            "mediaStatusChanged",
            "currentMediaChanged",
            "sourceChanged",
            "mediaChanged",
            "playbackStateChanged",
        ):
            sig = getattr(p, sig_name, None)
            try:
                sig.connect(_reapply_from_player_event)
            except Exception:
                pass

    # --------------------------------------------------
    # UI callbacks (user actions)
    def _on_volume_changed(self, _value):
        """
        User dragged the volume slider.
        Update what we WANT, then enforce it immediately.
        """
        if self._internal_change:
            return

        global _DESIRED_VOL_PCT
        vol_pct = int(self.vol.value())
        if vol_pct < 0: vol_pct = 0
        if vol_pct > 100: vol_pct = 100

        self._desired_vol_pct = vol_pct
        _DESIRED_VOL_PCT = vol_pct

        # Do not unmute automatically here.
        # If muted, keep muted. We just remember the new slider value.
        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    def _on_mute_toggled(self, state):
        """
        User toggled mute checkbox.
        Update what we WANT, and apply it now (and on future tracks).
        """
        if self._internal_change:
            return

        global _DESIRED_MUTED
        muted = bool(state)

        self._desired_muted = muted
        _DESIRED_MUTED = muted

        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    def _on_reset_clicked(self):
        """
        Reset all EQ sliders to 0 dB. Still UI-only.
        """
        self._internal_change = True
        try:
            for sv in self.eq:
                sv.setValue(0)
        finally:
            self._internal_change = False

    def _on_eq_slider_changed(self, _v):
        """
        EQ is still visual only. No audio processing here yet.
        """
        return

    # convenience getters (unchanged)
    def get_eq_gains_db(self):
        return [int(sv.value()) for sv in self.eq]

    def get_volume_scalar(self):
        if self._desired_muted:
            return 0.0
        return float(self._desired_vol_pct) / 100.0

    def is_muted(self):
        return bool(self._desired_muted)


def _install_cleanup_strong(pane, btn):
    """
    Old code had a ton of teardown logic. We keep a stub so caller code
    doesn't explode, but we intentionally do nothing here.
    """
    return


def add_new_volume_popup(pane, bar_layout):
    """
    This matches your original call style:
        add_new_volume_popup(self, self.bottom_bar_layout)

    What we do:
    - Create a 48x48 round-ish toolbutton with "🔊"
    - Attach a popup menu containing _MenuPopupWidget
    - Add that button to the bar_layout you passed in
    - Expose pane.btn_volume_new, pane.volume_popup_menu, pane.volume_popup_widget
    - DO NOT touch playback timing in any way
    """

    # Clean up any previous/stale button
    for attr in ("btn_volume_new", "btn_volume"):
        old = getattr(pane, attr, None)
        if old is not None:
            try:
                old.setParent(None)
                old.deleteLater()
            except Exception:
                pass

    btn = QToolButton(pane)
    btn.setObjectName("btn_volume_new")

    # match your control bar style: flat, hover highlight, 48x48
    try: btn.setAutoRaise(True)
    except Exception: pass
    try: btn.setFocusPolicy(Qt.NoFocus)
    except Exception: pass
    try: btn.setCursor(Qt.PointingHandCursor)
    except Exception: pass

    btn.setText("🔊")
    btn.setToolTip("Volume / EQ")
    try:
        btn.setFixedSize(48, 48)
    except Exception:
        pass

    menu = QMenu(btn)
    w = _MenuPopupWidget(pane, parent=menu)

    wa = QWidgetAction(menu)
    wa.setDefaultWidget(w)
    menu.addAction(wa)

    btn.setMenu(menu)
    btn.setPopupMode(QToolButton.InstantPopup)

    btn.setStyleSheet(
        "QToolButton#btn_volume_new{"
        " background:transparent;"
        " border:none;"
        " padding:0px;"
        " border-radius:24px;"
        "}"
        "QToolButton#btn_volume_new:hover{"
        " background:rgba(255,255,255,0.08);"
        "}"
        "QToolButton#btn_volume_new:pressed{"
        " background:rgba(255,255,255,0.16);"
        "}"
        "QToolButton#btn_volume_new:checked{"
        " background:rgba(255,255,255,0.12);"
        "}"
        "QToolButton::menu-indicator{"
        " image:none; width:0px; height:0px;"
        "}"
        "QMenu{"
        " background:transparent;"
        " border:0px;"
        "}"
    )

    # Put the button into your transport bar layout
    try:
        bar_layout.addWidget(btn)
    except Exception:
        # worst case: just show it anyway
        btn.setParent(pane)
        btn.show()

    # Old code used to install cleanup hooks etc.
    _install_cleanup_strong(pane, btn)

    # Expose on pane so other code can find it
    pane.btn_volume_new = btn
    pane.volume_popup_menu = menu
    pane.volume_popup_widget = w

    # Old code called w.schedule_reapply(0). Safe no-op here.
    try:
        w.schedule_reapply(0)
    except Exception:
        pass

    return btn
