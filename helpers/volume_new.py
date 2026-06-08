from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QFrame, QToolButton, QCheckBox, QMenu, QWidgetAction, QPushButton
)

"""
volume_new.py (mute + volume only)

Public surface remains compatible:
    - add_new_volume_popup(pane, bar_layout)
    - class _MenuPopupWidget
    - pane.btn_volume_new
    - pane.volume_popup_menu
    - pane.volume_popup_widget

Behavior guarantees:
    - We DO NOT pause/resume/seek or alter transport timing.
    - We ONLY control audio mute/volume and persist desired state across tracks.

Removed:
    - Equalizer UI and reset button
    - Presets and EQ-related methods/signals
"""

# Global desired state so it survives track changes / new popup instances
_DESIRED_VOL_PCT = 100   # 0-100
_DESIRED_MUTED   = False # True if user hit Mute


def _resolve_audio(pane):
    """
    Find the audio output object we are allowed to control.
    We ONLY ever call setVolume()/setMuted() on this.
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
    # --- aggressively remove any leftover small 5-band EQ UIs ---
    def _purge_unwanted_children(self):
        # Remove any vertical sliders, 'Reset EQ' buttons, and Hz labels added by legacy code.
        keep = {self, self.vol, self.mute}
        try:
            root_frame = self.findChild(QFrame, "fv_menu_volume_frame")
            if root_frame:
                keep.add(root_frame)
        except Exception:
            pass

        for w in list(self.findChildren(QWidget)):
            if w in keep:
                continue

            objn = getattr(w, "objectName", lambda: "")() or ""
            text = ""
            try:
                text = w.text() if hasattr(w, "text") else ""
            except Exception:
                text = ""

            try:
                if isinstance(w, QSlider) and w.orientation() == Qt.Vertical:
                    w.setParent(None); w.deleteLater(); continue
            except Exception:
                pass

            # Remove labels like "60 Hz", "3.6 kHz", etc.
            if isinstance(w, QLabel):
                t = (text or "").strip().lower()
                if "hz" in t or "kHz" in t or t.endswith("hz") or "eq" in objn.lower():
                    w.setParent(None); w.deleteLater(); continue

            if isinstance(w, QPushButton):
                if "eq" in (text or "").lower():
                    w.setParent(None); w.deleteLater(); continue

            if "eq" in objn.lower() or "equaliz" in objn.lower():
                try:
                    w.setParent(None); w.deleteLater(); continue
                except Exception:
                    pass

    
    """
    Popup widget that lives inside the menu.

    Compatibility:
        - self.vol  (QSlider, horizontal)
        - self.mute (QCheckBox)
        - schedule_reapply()  (harmless no-op)
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
        self.vol.setObjectName("fv_vol_slider")
        self.vol.setRange(0, 100)
        self.vol.setValue(self._desired_vol_pct)

        self.mute = QCheckBox("Mute")
        self.mute.setObjectName("fv_mute_checkbox")
        self.mute.setChecked(self._desired_muted)

        top.addWidget(vol_label)
        top.addWidget(self.vol, 1)
        top.addWidget(self.mute)
        outer.addLayout(top)

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
            "QLabel#fv_vol_label{"
            " font-weight:700; font-size:13px;"
            "}"
        )
        self.setMinimumWidth(260)

        # ---- SIGNALS ----
        self.vol.valueChanged.connect(self._on_volume_changed)
        self.mute.toggled.connect(self._on_mute_toggled)

        # Poll loop:
        #   runs fast (~75ms), constantly forcing the audio output to match
        #   "what the user wants" (desired mute/volume).
        self._poll = QTimer(self)
        self._poll.setSingleShot(False)
        self._poll.timeout.connect(self._sync_loop)
        self._poll.start(75)

        # Hook into player events so we re-apply desired mute/vol
        # instantly when track/media changes (no audible blip).
        self._hook_player_signals()

        # Do one immediate sync/apply right now
        self._purge_unwanted_children()
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

    def showEvent(self, ev):
        try:
            self._purge_unwanted_children()
        except Exception:
            pass
        super().showEvent(ev)

    def _sync_loop(self):
        """
        Runs every ~75ms:
        1. Apply desired mute/volume to the audio output.
        2. Reflect those desired values back into the UI.
        """
        self._purge_unwanted_children()
        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    def _hook_player_signals(self):
        """
        Connect to player events (if they exist) so we immediately enforce
        mute/volume the moment a new file loads or playback state changes.
        """
        p = _resolve_player(self.pane)
        if p is None:
            return

        def _reapply_from_player_event(*_):
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
        self._purge_unwanted_children()
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

        self._purge_unwanted_children()
        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    # convenience getters (unchanged)
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
    btn.setToolTip("Volume")
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
