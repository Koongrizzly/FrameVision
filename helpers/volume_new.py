import os
import json
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QFrame, QToolButton, QCheckBox, QGridLayout, QMenu, QWidgetAction,
    QComboBox, QInputDialog, QMessageBox
)

"""
volume_new.py  (v11)

What this gives you:
- Volume slider (0..100)  ✅ works
- Mute checkbox           ✅ works
- Sticky mute/volume across tracks (no popping, no skipping first beat)
- 12-band EQ sliders      ✅ UI
- Reset EQ button         ✅ UI, resets + saves
- Preset dropdown         ✅ new
- Save preset / Delete preset buttons ✅ new
- EQ state auto-saves and auto-loads across app restarts ✅ new

Paths (relative to current working directory when you launch the app):
    presets/setsave/eq_state.json
    presets/setsave/eqpresets/*.json

These folders/files are created automatically if missing.

Things we very intentionally DO NOT do (for safety/stability right now):
- We NEVER pause/resume/seek the player
- We NEVER do 150ms gates or first-beat skips
- We NEVER hijack playback or spawn a side ffmpeg process
- We DO NOT yet process audio through an EQ filter (that's the next phase where we build a proper master/mixer)

Public API is the same as before:
    add_new_volume_popup(pane, bar_layout)
    class _MenuPopupWidget
    pane.btn_volume_new
    pane.volume_popup_menu
    pane.volume_popup_widget
"""

# 12-band EQ (UI). Each is -12..+12 dB
BANDS = [
    ("60 Hz",      60),
    ("120 Hz",     120),
    ("230 Hz",     230),
    ("460 Hz",     460),
    ("910 Hz",     910),
    ("1.8 kHz",    1800),
    ("2.5 kHz",    2500),
    ("3.6 kHz",    3600),
    ("5 kHz",      5000),
    ("7 kHz",      7000),
    ("10 kHz",     10000),
    ("14 kHz",     14000),
]

# Where we persist EQ stuff
_BASE_EQ_DIR      = os.path.join(os.getcwd(), "presets", "setsave")
_EQ_STATE_FILE    = os.path.join(_BASE_EQ_DIR, "eq_state.json")
_EQ_PRESET_DIR    = os.path.join(_BASE_EQ_DIR, "eqpresets")

# Global desired state for sticky mute & volume
_DESIRED_VOL_PCT = 100    # slider % 0..100
_DESIRED_MUTED   = False  # True if user checked "Mute"


def _ensure_eq_dirs():
    """Make sure presets/setsave/ and presets/setsave/eqpresets/ actually exist."""
    try:
        os.makedirs(_BASE_EQ_DIR, exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(_EQ_PRESET_DIR, exist_ok=True)
    except Exception:
        pass


def _resolve_audio(pane):
    """
    Find the audio output object (QAudioOutput / similar).
    We ONLY touch setVolume()/setMuted() on this.
    We do NOT touch playback timing.
    """
    for name in ("audio", "audio_output", "audioOutput", "player_audio", "audio_out"):
        if hasattr(pane, name):
            a = getattr(pane, name)
            if a is not None:
                return a

    # fallback: pane.player.audioOutput()
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
    Try to find the playback object.
    We ONLY listen to its signals so we can re-apply mute/volume instantly on changes.
    We DO NOT pause/resume/seek it.
    """
    for nm in ("player", "media_player", "video_player"):
        p = getattr(pane, nm, None)
        if p is not None:
            return p
    return None


class _MenuPopupWidget(QWidget):
    """
    The popup content shown when you click the 🔊 button.

    Exposes:
      - self.vol (QSlider horizontal)
      - self.mute (QCheckBox)
      - self.eq (list of 12 vertical sliders)
      - self.preset_combo (QComboBox)
      - schedule_reapply() (compat no-op)

    Behavior:
      - Auto-load last EQ state from eq_state.json on startup
      - Auto-save EQ state whenever sliders move / reset is clicked
      - Preset dropdown lets you load presets from /eqpresets/
      - "Save Preset" writes a new file in /eqpresets/
      - "Delete Preset" removes the selected preset file
    """

    def __init__(self, pane, parent=None):
        super().__init__(parent)
        self.pane = pane

        _ensure_eq_dirs()

        # guard to avoid signal loops when we update UI ourselves
        self._internal_change = False

        # copy global desired volume/mute into this widget instance
        global _DESIRED_VOL_PCT, _DESIRED_MUTED
        self._desired_vol_pct = _DESIRED_VOL_PCT
        self._desired_muted   = _DESIRED_MUTED

        # remember last-applied audio state so we don't spam setVolume()/setMuted()
        self._last_applied_vol_pct = None
        self._last_applied_muted   = None

        # -------------------------------------------------
        # BUILD UI
        # -------------------------------------------------

        frame = QFrame(self)
        frame.setObjectName("fv_menu_volume_frame")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(frame)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(10)

        #
        # Top row: Volume slider | Mute | Reset EQ
        #
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        vol_label = QLabel("Volume")
        vol_label.setObjectName("fv_vol_label")

        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(self._desired_vol_pct)

        self.mute = QCheckBox("Mute")
        self.mute.setChecked(self._desired_muted)

        self.reset_eq = QPushButton("Reset EQ")
        self.reset_eq.setObjectName("fv_eq_reset")

        top_row.addWidget(vol_label)
        top_row.addWidget(self.vol, 1)
        top_row.addWidget(self.mute)
        top_row.addWidget(self.reset_eq)
        outer.addLayout(top_row)

        #
        # Preset row: [ Preset dropdown | Save Preset | Delete Preset ]
        #
        preset_row = QHBoxLayout()
        preset_row.setSpacing(8)

        preset_label = QLabel("Preset:")
        preset_label.setObjectName("fv_preset_label")

        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("fv_preset_combo")
        # style will follow palette, we can theme via stylesheet below

        self.btn_save_preset = QPushButton("Save Preset")
        self.btn_save_preset.setObjectName("fv_save_preset")

        self.btn_delete_preset = QPushButton("Delete Preset")
        self.btn_delete_preset.setObjectName("fv_delete_preset")

        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.btn_save_preset)
        preset_row.addWidget(self.btn_delete_preset)
        outer.addLayout(preset_row)

        #
        # 12-band EQ sliders
        #
        eq_grid = QGridLayout()
        eq_grid.setHorizontalSpacing(12)
        eq_grid.setVerticalSpacing(4)

        self.eq = []
        for i, (label_text, freq) in enumerate(BANDS):
            sv = QSlider(Qt.Vertical)
            sv.setRange(-12, 12)
            sv.setValue(0)
            sv.setToolTip(f"{freq} Hz")
            self.eq.append(sv)

            eq_grid.addWidget(
                sv,
                0, i,
                alignment=Qt.AlignHCenter | Qt.AlignBottom,
            )

            lab = QLabel(label_text)
            lab.setObjectName("fv_freq_label")
            lab.setAlignment(Qt.AlignHCenter)
            eq_grid.addWidget(
                lab,
                1, i,
                alignment=Qt.AlignHCenter,
            )

        outer.addLayout(eq_grid)

        #
        # Styling
        #
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
            " font-weight:600; font-size:12px;"
            "}"
            "QLabel#fv_vol_label{"
            " font-weight:700; font-size:13px;"
            "}"
            "QLabel#fv_preset_label{"
            " font-weight:600; font-size:12px;"
            "}"
            "QPushButton#fv_eq_reset,"
            "QPushButton#fv_save_preset,"
            "QPushButton#fv_delete_preset{"
            " background: palette(button);"
            " color: palette(window-text);"
            " border: 1px solid palette(mid);"
            " border-radius: 8px;"
            " padding: 4px 10px;"
            " font-size:12px;"
            "}"
            "QPushButton#fv_eq_reset:hover,"
            "QPushButton#fv_save_preset:hover,"
            "QPushButton#fv_delete_preset:hover{"
            " background: palette(light);"
            "}"
            "QPushButton#fv_eq_reset:pressed,"
            "QPushButton#fv_save_preset:pressed,"
            "QPushButton#fv_delete_preset:pressed{"
            " background: palette(dark);"
            "}"
            "QComboBox#fv_preset_combo{"
            " background: palette(base);"
            " color: palette(window-text);"
            " border: 1px solid palette(mid);"
            " border-radius: 6px;"
            " padding: 2px 8px;"
            " font-size:12px;"
            "}"
        )

        # -------------------------------------------------
        # SIGNALS
        # -------------------------------------------------

        self.vol.valueChanged.connect(self._on_volume_changed)
        self.mute.toggled.connect(self._on_mute_toggled)
        self.reset_eq.clicked.connect(self._on_reset_clicked)

        # EQ sliders
        for sv in self.eq:
            sv.valueChanged.connect(self._on_eq_slider_changed)

        # Preset interactions
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self.btn_save_preset.clicked.connect(self._on_save_preset_clicked)
        self.btn_delete_preset.clicked.connect(self._on_delete_preset_clicked)

        # Poll timer (every 250ms):
        # - enforce desired mute/volume if needed
        # - refresh UI to match desired values
        self._poll = QTimer(self)
        self._poll.setSingleShot(False
        )
        self._poll.timeout.connect(self._tick)
        self._poll.start(250)

        # Hook player signals so when a new track starts,
        # we instantly enforce mute/volume.
        self._hook_player_signals()

        # 1. Load last EQ state from disk (if present)
        #    This also updates sliders BEFORE we save anything new.
        self._load_eq_state()

        # 2. Load the preset list into dropdown
        self._refresh_presets()

        # 3. Initial sync
        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    # -------------------------------------------------
    # Backward compat with old code
    def schedule_reapply(self, delay_ms=0):
        # Old versions used this for EQ/ffmpeg.
        # We keep it so your app won't crash if something still calls it.
        return

    # -------------------------------------------------
    # Internal helpers for audio control
    def _audio(self):
        return _resolve_audio(self.pane)

    def _apply_desired_state_to_audio(self):
        """
        Enforce desired mute/volume on the actual audio output.
        We only call setVolume()/setMuted() if something changed,
        so CPU stays low and we don't spam.
        """
        a = self._audio()
        if not a:
            return

        pct = int(self._desired_vol_pct)
        if pct < 0:
            pct = 0
        if pct > 100:
            pct = 100

        muted_now = bool(self._desired_muted)
        vol_now   = float(0 if muted_now else pct) / 100.0

        if (self._last_applied_muted == muted_now and
            self._last_applied_vol_pct == pct):
            return

        try:
            if hasattr(a, "setMuted"):
                a.setMuted(muted_now)
        except Exception:
            pass

        try:
            if hasattr(a, "setVolume"):
                a.setVolume(vol_now)
        except Exception:
            pass

        self._last_applied_muted = muted_now
        self._last_applied_vol_pct = pct

    def _sync_ui_from_desired(self):
        """
        Make sure the widgets show what we WANT.
        We do NOT let backend resets change our UI.
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

    def _tick(self):
        """
        Runs every 250ms:
        1. Apply sticky mute/volume if needed.
        2. Sync UI to desired state.
        """
        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    # -------------------------------------------------
    # Track change hook
    def _hook_player_signals(self):
        """
        Listen for player state changes so we can:
        - instantly re-apply mute/volume on new track (prevents leaks)
        We STILL do NOT pause/resume/seek the player.
        """
        p = _resolve_player(self.pane)
        if p is None:
            return

        def _instant_enforce(*_):
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
                sig.connect(_instant_enforce)
            except Exception:
                pass

    # -------------------------------------------------
    # EQ state persistence
    def _slider_values_list(self):
        """
        Return EQ gains as list of 12 ints (-12..+12), in band order.
        """
        return [int(sv.value()) for sv in self.eq]

    def _apply_slider_values_list(self, values):
        """
        Given a list of 12 ints, set sliders to those values.
        (No audio processing yet, just UI + persistence.)
        """
        if len(values) != len(self.eq):
            return
        self._internal_change = True
        try:
            for sv, gain in zip(self.eq, values):
                try:
                    gain_i = int(gain)
                except Exception:
                    gain_i = 0
                if gain_i < -12:
                    gain_i = -12
                if gain_i > 12:
                    gain_i = 12
                sv.setValue(gain_i)
        finally:
            self._internal_change = False

    def _save_eq_state_file(self):
        """
        Save the CURRENT slider gains to eq_state.json.
        """
        _ensure_eq_dirs()
        data = {
            "bands": self._slider_values_list(),
            "labels": [lbl for (lbl, _freq) in BANDS],
        }
        try:
            with open(_EQ_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_eq_state(self):
        """
        On popup init: read eq_state.json (if exists) and apply to sliders.
        """
        if not os.path.exists(_EQ_STATE_FILE):
            return
        try:
            with open(_EQ_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            bands = data.get("bands", None)
            if isinstance(bands, list) and len(bands) == len(self.eq):
                self._apply_slider_values_list(bands)
        except Exception:
            pass

    # -------------------------------------------------
    # Preset handling
    def _refresh_presets(self):
        """
        Refresh dropdown with list of saved presets.
        We list files in eqpresets/, strip .json.
        """
        _ensure_eq_dirs()
        names = []
        try:
            for fname in os.listdir(_EQ_PRESET_DIR):
                if not fname.lower().endswith(".json"):
                    continue
                preset_name = fname[:-5]  # drop ".json"
                names.append(preset_name)
        except Exception:
            pass

        names.sort(key=str.lower)

        self._internal_change = True
        try:
            self.preset_combo.clear()
            # First item is just placeholder text "Choose preset..."
            self.preset_combo.addItem("Choose preset...")
            for n in names:
                self.preset_combo.addItem(n)
        finally:
            self._internal_change = False

    def _on_preset_selected(self, idx):
        """
        User picked something in the dropdown.
        Load that preset and apply its gains.
        """
        if self._internal_change:
            return
        if idx <= 0:
            # index 0 is "Choose preset..."
            return

        name = self.preset_combo.currentText()
        preset_path = os.path.join(_EQ_PRESET_DIR, f"{name}.json")
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                pdata = json.load(f)
            bands = pdata.get("bands", None)
            if isinstance(bands, list) and len(bands) == len(self.eq):
                self._apply_slider_values_list(bands)
                # after loading a preset, also save as 'current state'
                self._save_eq_state_file()
                # and call hook to eventually update DSP
                self._apply_eq_to_audio()
        except Exception:
            pass

    def _on_save_preset_clicked(self):
        """
        Ask user for a preset name, then save current slider values to that name.
        """
        _ensure_eq_dirs()
        name, ok = QInputDialog.getText(
            self,
            "Save EQ Preset",
            "Preset name:"
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # sanitize filename-ish
        safe_name = "".join(ch for ch in name if ch not in r'\/:*?"<>|').strip()
        if not safe_name:
            return

        preset_path = os.path.join(_EQ_PRESET_DIR, f"{safe_name}.json")
        data = {
            "bands": self._slider_values_list(),
            "labels": [lbl for (lbl, _freq) in BANDS],
        }
        try:
            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

        # refresh list so new preset shows up
        self._refresh_presets()

        # select it in the combo after save
        # (find its index)
        for i in range(self.preset_combo.count()):
            if self.preset_combo.itemText(i) == safe_name:
                self.preset_combo.setCurrentIndex(i)
                break

    def _on_delete_preset_clicked(self):
        """
        Delete currently selected preset file (except index 0).
        """
        idx = self.preset_combo.currentIndex()
        if idx <= 0:
            return

        name = self.preset_combo.currentText()
        preset_path = os.path.join(_EQ_PRESET_DIR, f"{name}.json")

        # Ask for confirmation just in case
        resp = QMessageBox.question(
            self,
            "Delete EQ Preset",
            f"Delete preset '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if resp != QMessageBox.Yes:
            return

        try:
            os.remove(preset_path)
        except Exception:
            pass

        # reload combo
        self._refresh_presets()
        # move back to "Choose preset..."
        self.preset_combo.setCurrentIndex(0)

    # -------------------------------------------------
    # User actions on the popup
    def _on_volume_changed(self, _value):
        """
        User moved the volume slider.
        Update desired volume and apply it now.
        Does NOT auto-unmute if muted; we just remember the new level.
        """
        if self._internal_change:
            return

        global _DESIRED_VOL_PCT
        pct = int(self.vol.value())
        if pct < 0:
            pct = 0
        if pct > 100:
            pct = 100

        self._desired_vol_pct = pct
        _DESIRED_VOL_PCT = pct

        self._apply_desired_state_to_audio()
        self._sync_ui_from_desired()

    def _on_mute_toggled(self, state):
        """
        User toggled mute.
        We store that and apply it right now
        (and also on every future track).
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
        Reset all EQ sliders to 0 dB.
        Save new state.
        Apply new state to audio path (placeholder).
        """
        self._internal_change = True
        try:
            for sv in self.eq:
                sv.setValue(0)
        finally:
            self._internal_change = False

        self._save_eq_state_file()
        self._apply_eq_to_audio()

    def _on_eq_slider_changed(self, _v):
        """
        Called whenever ANY band slider moves.
        We immediately:
            - save current EQ state to disk
            - call _apply_eq_to_audio() (placeholder for real DSP)
        """
        if self._internal_change:
            return

        self._save_eq_state_file()
        self._apply_eq_to_audio()

    # -------------------------------------------------
    # Hook for actual DSP (not implemented yet)
    def _apply_eq_to_audio(self):
        """
        This is where live EQ would actually be applied to the sound.

        Right now we are NOT touching playback audio,
        because we haven't built the shared "mixer"/"master bus" yet.

        But: the code that *will* do EQ will read gains from:
            self._slider_values_list()
        in exactly this method.

        For now it's a no-op so we don't break playback.
        """
        return

    # -------------------------------------------------
    # Convenience getters
    def get_eq_gains_db(self):
        """
        Returns a list of 12 integers, each -12..+12 dB.
        Order matches BANDS.
        """
        return self._slider_values_list()

    def get_volume_scalar(self):
        """
        0.0..1.0 effective volume after mute.
        """
        if self._desired_muted:
            return 0.0
        return float(self._desired_vol_pct) / 100.0

    def is_muted(self):
        return bool(self._desired_muted)


def _install_cleanup_strong(pane, btn):
    """
    Old code used to kill ffmpeg processes etc.
    We intentionally do nothing now.
    Stub stays so legacy calls won't crash.
    """
    return


def add_new_volume_popup(pane, bar_layout):
    """
    Call like:
        add_new_volume_popup(self, self.bottom_bar_layout)
    """

    # Remove any previous button if it was there
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

    btn.setText("🔊")
    btn.setToolTip("Volume / EQ")
    try:
        btn.setFixedSize(48, 48)
    except Exception:
        pass

    # Build popup
    menu = QMenu(btn)
    w = _MenuPopupWidget(pane, parent=menu)

    wa = QWidgetAction(menu)
    wa.setDefaultWidget(w)
    menu.addAction(wa)

    btn.setMenu(menu)
    btn.setPopupMode(QToolButton.InstantPopup)

    # Style to match your round-ish bottom bar buttons
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

    # Add to your control bar
    try:
        bar_layout.addWidget(btn)
    except Exception:
        btn.setParent(pane)
        btn.show()

    _install_cleanup_strong(pane, btn)

    pane.btn_volume_new = btn
    pane.volume_popup_menu = menu
    pane.volume_popup_widget = w

    try:
        w.schedule_reapply(0)  # harmless no-op for legacy code
    except Exception:
        pass

    return btn
