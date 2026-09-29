from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize, QTimer, QRect, QUrl
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QFontMetrics, QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFrame,
    QSplitter,
    QSizePolicy,
    QScrollArea,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QSlider,
    QInputDialog,
    QDialog,
    QDialogButtonBox,
    QApplication,
    QMenu,
)

try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
except Exception:
    QMediaPlayer = QAudioOutput = QVideoWidget = None

FPS = 24.0
MIN_SEGMENT_SECONDS = 0.3
# Backward-compatible internal alias; project JSON still uses the existing "segments" key.
MIN_CUT_SECONDS = MIN_SEGMENT_SECONDS
TIMELINE_SCHEMA_VERSION = 4
TIMELINE_MAX_REF_IMAGES = 9


class ClipTrimDialog(QDialog):
    """Non-destructive in/out editor for an existing rendered timeline clip."""

    def __init__(self, video_path: str, duration: float, trim_in: float = 0.0, trim_out: float | None = None, parent=None):
        super().__init__(parent)
        self.video_path = str(video_path)
        self.duration_seconds = max(0.01, float(duration or 0.01))
        self._previewing_range = False
        self._syncing = False
        self.setWindowTitle("Trim clip")
        self.resize(760, 610)

        root = QVBoxLayout(self)
        hint = QLabel("This trim is non-destructive. It changes only playback/assembly; the generated MP4 and generation duration stay unchanged.")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.video_widget = None
        self.player = None
        self.audio = None
        if QMediaPlayer is not None and QVideoWidget is not None:
            self.video_widget = QVideoWidget(self)
            self.video_widget.setMinimumHeight(330)
            self.video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            root.addWidget(self.video_widget, 1)
            self.player = QMediaPlayer(self)
            self.audio = QAudioOutput(self)
            self.player.setAudioOutput(self.audio)
            self.player.setVideoOutput(self.video_widget)
            self.player.setSource(QUrl.fromLocalFile(str(Path(self.video_path).resolve())))
            self.player.positionChanged.connect(self._on_position_changed)
            self.player.durationChanged.connect(self._on_duration_changed)
        else:
            unavailable = QLabel("Video preview is unavailable in this Qt build. Start/end trimming still works.")
            unavailable.setAlignment(Qt.AlignmentFlag.AlignCenter)
            unavailable.setMinimumHeight(180)
            root.addWidget(unavailable, 1)

        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, max(1, int(round(self.duration_seconds * 1000))))
        self.scrub.sliderMoved.connect(self._seek_ms)
        root.addWidget(self.scrub)

        time_row = QHBoxLayout()
        self.position_label = QLabel("0.00s")
        self.range_label = QLabel()
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        time_row.addWidget(self.position_label)
        time_row.addStretch(1)
        time_row.addWidget(self.range_label)
        root.addLayout(time_row)

        form = QFormLayout()
        self.start_spin = QDoubleSpinBox()
        self.end_spin = QDoubleSpinBox()
        for spin in (self.start_spin, self.end_spin):
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setRange(0.0, self.duration_seconds)
            spin.setSuffix(" s")
        initial_in = max(0.0, min(float(trim_in or 0.0), self.duration_seconds))
        initial_out = self.duration_seconds if trim_out is None else max(initial_in, min(float(trim_out), self.duration_seconds))
        self.start_spin.setValue(initial_in)
        self.end_spin.setValue(initial_out)
        self.start_spin.valueChanged.connect(self._range_changed)
        self.end_spin.valueChanged.connect(self._range_changed)
        form.addRow("Start", self.start_spin)
        form.addRow("End", self.end_spin)
        root.addLayout(form)

        controls = QHBoxLayout()
        self.play_btn = QPushButton("Play / Pause")
        self.preview_btn = QPushButton("Preview trim")
        self.set_start_btn = QPushButton("Set start at current")
        self.set_end_btn = QPushButton("Set end at current")
        self.reset_btn = QPushButton("Reset trim")
        controls.addWidget(self.play_btn)
        controls.addWidget(self.preview_btn)
        controls.addWidget(self.set_start_btn)
        controls.addWidget(self.set_end_btn)
        controls.addWidget(self.reset_btn)
        root.addLayout(controls)

        self.play_btn.clicked.connect(self._toggle_play)
        self.preview_btn.clicked.connect(self._preview_range)
        self.set_start_btn.clicked.connect(self._set_start_current)
        self.set_end_btn.clicked.connect(self._set_end_current)
        self.reset_btn.clicked.connect(self._reset_trim)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._range_changed()

    def _current_seconds(self):
        return (float(self.player.position()) / 1000.0) if self.player is not None else (float(self.scrub.value()) / 1000.0)

    def _seek_ms(self, value):
        if self.player is not None:
            self.player.setPosition(int(value))
        self.position_label.setText(f"{float(value)/1000.0:.2f}s")

    def _on_duration_changed(self, ms):
        if int(ms or 0) > 0:
            # Prefer the actual player duration if ffprobe rounded slightly differently.
            self.scrub.setMaximum(int(ms))

    def _on_position_changed(self, ms):
        if not self.scrub.isSliderDown():
            self.scrub.setValue(int(ms))
        sec = float(ms) / 1000.0
        self.position_label.setText(f"{sec:.2f}s")
        if self._previewing_range and sec >= float(self.end_spin.value()) - 0.015:
            self.player.pause()
            self.player.setPosition(int(round(float(self.start_spin.value()) * 1000)))
            self._previewing_range = False

    def _range_changed(self, *_args):
        if self._syncing:
            return
        self._syncing = True
        start = float(self.start_spin.value())
        end = float(self.end_spin.value())
        if end < start:
            sender = self.sender()
            if sender is self.start_spin:
                self.end_spin.setValue(start)
                end = start
            else:
                self.start_spin.setValue(end)
                start = end
        self._syncing = False
        used = max(0.0, end - start)
        self.range_label.setText(f"Selected: {start:.3f}s → {end:.3f}s   •   used {used:.3f}s")

    def _toggle_play(self):
        if self.player is None:
            return
        self._previewing_range = False
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _preview_range(self):
        if self.player is None:
            return
        if float(self.end_spin.value()) - float(self.start_spin.value()) < 0.04:
            return
        self._previewing_range = True
        self.player.setPosition(int(round(float(self.start_spin.value()) * 1000)))
        self.player.play()

    def _set_start_current(self):
        pos = min(self._current_seconds(), float(self.end_spin.value()))
        self.start_spin.setValue(pos)

    def _set_end_current(self):
        pos = max(self._current_seconds(), float(self.start_spin.value()))
        self.end_spin.setValue(pos)

    def _reset_trim(self):
        self.start_spin.setValue(0.0)
        self.end_spin.setValue(self.duration_seconds)
        if self.player is not None:
            self.player.setPosition(0)

    def _apply(self):
        if float(self.end_spin.value()) - float(self.start_spin.value()) < 0.04:
            QMessageBox.warning(self, "Trim clip", "The selected range is too short. Choose a longer in/out range.")
            return
        self.accept()

    def trim_values(self):
        return float(self.start_spin.value()), float(self.end_spin.value())

    def closeEvent(self, event):
        if self.player is not None:
            self.player.stop()
        super().closeEvent(event)


def _safe_project_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(text or "").strip())
    return text or "MiniMax Timeline"


def _clip_seconds(clip: dict) -> float:
    """Generation/source duration. Do not shorten this for assembly trims."""
    if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
        try:
            duration = float(clip.get("source_duration_seconds") or 0.0)
            if duration > 0:
                return duration
        except Exception:
            pass
    return max(1, int(clip.get("frames") or 124)) / FPS


def _clip_speed(clip: dict) -> float:
    """Post-process playback speed stored on a Timeline block."""
    try:
        return max(0.5, min(2.0, round(float(clip.get("speed_multiplier", 1.0) or 1.0), 1)))
    except Exception:
        return 1.0


def _clip_timeline_seconds(clip: dict) -> float:
    """Visible/effective timeline duration after trim and playback-speed edits.

    Generation duration remains owned by _clip_seconds(); speed is a non-destructive
    post-process edit used only by Timeline geometry, preview and final assembly.
    """
    source_duration = _clip_seconds(clip)
    try:
        trim_in = max(0.0, float(clip.get("trim_in") or 0.0))
    except Exception:
        trim_in = 0.0
    trim_out_raw = clip.get("trim_out")
    if trim_out_raw is None:
        used = source_duration
    else:
        try:
            trim_out = float(trim_out_raw)
        except Exception:
            trim_out = source_duration
        # trim_out is captured from the actual rendered media and can differ by a
        # few frames from the requested generation duration, so trust the saved
        # trim range itself rather than clamping it to the requested frame count.
        candidate = trim_out - trim_in
        used = candidate if candidate > 0.0 else source_duration
    return max(0.04, used / _clip_speed(clip))


TRANSITION_OPTIONS = [
    ("None", "none"),
    ("Fade", "fade"),
    ("Dissolve", "dissolve"),
    ("Wipe left", "wipeleft"),
    ("Wipe right", "wiperight"),
    ("Wipe up", "wipeup"),
    ("Wipe down", "wipedown"),
    ("Slide left", "slideleft"),
    ("Slide right", "slideright"),
    ("Slide up", "slideup"),
    ("Slide down", "slidedown"),
    ("Smooth left", "smoothleft"),
    ("Smooth right", "smoothright"),
    ("Smooth up", "smoothup"),
    ("Smooth down", "smoothdown"),
    ("Circle open", "circleopen"),
    ("Circle close", "circleclose"),
    ("Pixelize", "pixelize"),
    ("Radial", "radial"),
    ("Zoom", "zoomin"),
]
_TRANSITION_LABELS = {value: label for label, value in TRANSITION_OPTIONS}


def _transition_name(clip: dict) -> str:
    value = str(clip.get("transition_to_next") or "none").strip().lower()
    return value if value in _TRANSITION_LABELS else "none"


def _transition_requested_seconds(clip: dict) -> float:
    if _transition_name(clip) == "none":
        return 0.0
    try:
        return max(0.1, min(3.0, round(float(clip.get("transition_duration", 0.5) or 0.5), 1)))
    except Exception:
        return 0.5


def _transition_overlap_seconds(clip: dict, next_clip: dict | None) -> float:
    """Effective overlap represented on the Timeline for this outgoing edge."""
    if next_clip is None or _transition_name(clip) == "none":
        return 0.0
    requested = _transition_requested_seconds(clip)
    # xfade needs some non-transition material on both sides. Keep at least one
    # 24-fps frame available even when a user chooses a long transition.
    maximum = max(0.0, min(_clip_timeline_seconds(clip), _clip_timeline_seconds(next_clip)) - (1.0 / FPS))
    return max(0.0, min(requested, maximum))


def _timeline_total_seconds(clips: list[dict]) -> float:
    total = sum(_clip_timeline_seconds(c) for c in clips)
    for i in range(max(0, len(clips) - 1)):
        total -= _transition_overlap_seconds(clips[i], clips[i + 1])
    return max(0.0, total)


def _soundtrack_clip_duration(item: dict) -> float:
    try:
        source_duration = max(0.0, float(item.get("source_duration") or 0.0))
    except Exception:
        source_duration = 0.0
    try:
        trim_in = max(0.0, float(item.get("trim_in") or 0.0))
    except Exception:
        trim_in = 0.0
    raw_out = item.get("trim_out")
    try:
        trim_out = float(raw_out) if raw_out is not None else source_duration
    except Exception:
        trim_out = source_duration
    if source_duration > 0:
        trim_in = min(trim_in, source_duration)
        trim_out = min(max(trim_out, trim_in), source_duration)
    used = max(0.0, trim_out - trim_in)
    try:
        start = max(0.0, float(item.get("timeline_start") or 0.0))
    except Exception:
        start = 0.0
    raw_end = item.get("timeline_end")
    if raw_end is not None:
        try:
            requested = max(start, float(raw_end)) - start
            used = min(used, requested)
        except Exception:
            pass
    return max(0.0, used)


def _soundtrack_clip_end(item: dict) -> float:
    try:
        start = max(0.0, float(item.get("timeline_start") or 0.0))
    except Exception:
        start = 0.0
    return start + _soundtrack_clip_duration(item)


class TransitionDialog(QDialog):
    def __init__(self, clip: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transition to next clip")
        self.setMinimumWidth(410)
        root = QVBoxLayout(self)
        hint = QLabel("The transition overlaps the end of this clip with the start of the next clip. It is applied only during final assembly; source clips remain unchanged.")
        hint.setWordWrap(True)
        root.addWidget(hint)
        form = QFormLayout()
        self.kind = QComboBox()
        for label, value in TRANSITION_OPTIONS:
            self.kind.addItem(label, value)
        current = _transition_name(clip)
        idx = self.kind.findData(current)
        self.kind.setCurrentIndex(idx if idx >= 0 else 0)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 3.0)
        self.duration.setDecimals(1)
        self.duration.setSingleStep(0.1)
        self.duration.setSuffix(" s")
        self.duration.setValue(_transition_requested_seconds(clip) or 0.5)
        self.audio = QComboBox()
        self.audio.addItem("Crossfade", "crossfade")
        self.audio.addItem("Hard cut", "hard_cut")
        self.audio.addItem("Fade out + fade in", "fade_out_in")
        audio_mode = str(clip.get("transition_audio_mode") or "crossfade")
        aidx = self.audio.findData(audio_mode)
        self.audio.setCurrentIndex(aidx if aidx >= 0 else 0)
        form.addRow("Transition", self.kind)
        form.addRow("Duration", self.duration)
        form.addRow("Audio", self.audio)
        root.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.kind.currentIndexChanged.connect(self._sync_enabled)
        self._sync_enabled()

    def _sync_enabled(self):
        enabled = str(self.kind.currentData() or "none") != "none"
        self.duration.setEnabled(enabled)
        self.audio.setEnabled(enabled)

    def values(self):
        return (
            str(self.kind.currentData() or "none"),
            round(float(self.duration.value()), 1),
            str(self.audio.currentData() or "crossfade"),
        )


def _segment_seconds(clip: dict) -> list[float]:
    segments = clip.get("segments") or []
    if not segments:
        return []
    total = sum(max(0.0001, float(s.get("weight") or 1.0)) for s in segments) or 1.0
    duration = _clip_seconds(clip)
    return [duration * max(0.0001, float(s.get("weight") or 1.0)) / total for s in segments]


def _compiled_prompt(clip: dict) -> str:
    """Compile the repo-style timed-segment model into one H3 prompt.

    One segment stays a plain prompt. Multiple segments become timestamped instructions,
    which is the representation H3 already understands for visible changes inside
    a single generated clip.
    """
    segments = clip.get("segments") or []
    non_empty = [s for s in segments if str(s.get("prompt") or "").strip()]
    if not non_empty:
        return ""
    if len(segments) == 1:
        return str(segments[0].get("prompt") or "").strip()

    seconds = _segment_seconds(clip)
    cursor = 0.0
    lines: list[str] = []
    for seg, span in zip(segments, seconds):
        end = cursor + span
        prompt = str(seg.get("prompt") or "").strip()
        if prompt:
            lines.append(f"[{cursor:.1f}s - {end:.1f}s] {prompt}")
        cursor = end
    return "\n".join(lines).strip()


def _reference_entries(clip: dict) -> list[dict]:
    entries = []
    for item in (clip.get("reference_images") or []):
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        entry = {
            "path": path,
            "name": str(item.get("name") or Path(path).stem).strip(),
            "picture_description": str(item.get("picture_description") or "").strip(),
            "subject_description": str(item.get("subject_description") or "").strip(),
        }
        preset_id = str(item.get("preset_id") or "").strip()
        if preset_id:
            entry["preset_id"] = preset_id
        entries.append(entry)
    return entries[:TIMELINE_MAX_REF_IMAGES]


def _renumber_reference_description(text: str, token: str, index: int, default_body: str) -> str:
    """Normalize a saved preset line to the positional Picture/Subject number used by MiniMax."""
    value = str(text or "").strip()
    prefix = f"<{token} {index}>:"
    if not value:
        return f"{prefix} {default_body}"
    # Users commonly save examples as <Picture 1>: / <Subject 1>:.  Presets are
    # reusable, so the numeric token is always rewritten to the position chosen
    # for this particular clip.
    pattern = rf"^\s*<\s*{re.escape(token)}\s+\d+\s*>\s*:?\s*"
    body = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE).strip()
    return f"{prefix} {body or default_body}"


def _subject_tokens_present(text: str) -> bool:
    return bool(re.search(r"<\s*Subject\s+\d+\s*>", str(text or ""), flags=re.IGNORECASE))


def _strip_reference_prompt_prefix(text: str) -> str:
    """Remove a leading block of <Picture N>: / <Subject N>: definitions."""
    lines = str(text or "").splitlines()
    if not lines:
        return ""
    token_re = re.compile(r"^\s*<\s*(Picture|Subject)\s+\d+\s*>\s*:?.*$", re.IGNORECASE)
    i = 0
    saw_token = False
    while i < len(lines):
        stripped = lines[i].strip()
        if token_re.match(stripped):
            saw_token = True
            i += 1
            continue
        if not stripped:
            if not saw_token:
                i += 1
                continue
            j = i
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and token_re.match(lines[j].strip()):
                i = j
                continue
            i = j
            break
        break
    if not saw_token:
        return str(text or "").strip()
    return "\n".join(lines[i:]).strip()


def _reference_prompt_definition_lines(refs: list[dict]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for idx, ref in enumerate(refs, 1):
        friendly = str(ref.get("name") or f"Reference {idx}").strip() or f"Reference {idx}"
        picture_line = _renumber_reference_description(
            ref.get("picture_description"), "Picture", idx, f'reference image of {friendly}'
        )
        subject_line = _renumber_reference_description(
            ref.get("subject_description"), "Subject", idx,
            f'the subject shown in <Picture {idx}>. Preserve its exact identity and appearance.'
        )
        subject_line = re.sub(r"<\s*Picture\s+\d+\s*>", f"<Picture {idx}>", subject_line, flags=re.IGNORECASE)
        subject_line = re.sub(r"<\s*Subject\s+\d+\s*>", f"<Subject {idx}>", subject_line, count=1, flags=re.IGNORECASE)
        pairs.append((picture_line, subject_line))
    return pairs


def _reference_prompt_definitions_text(refs: list[dict]) -> str:
    if not refs:
        return ""
    pairs = _reference_prompt_definition_lines(refs)
    picture_lines = [picture for picture, _subject in pairs if picture]
    subject_lines = [subject for _picture, subject in pairs if subject]
    parts = []
    if picture_lines:
        parts.append("\n\n".join(picture_lines))
    if subject_lines:
        parts.append("\n\n".join(subject_lines))
    return "\n\n\n".join(parts).strip()


def _automatic_subject_usage_line(refs: list[dict]) -> str:
    if not refs:
        return ""
    tokens = [f"<Subject {idx}>" for idx in range(1, len(refs) + 1)]
    if len(tokens) == 1:
        return f"Feature {tokens[0]} as the main visible subject in the video."
    if len(tokens) == 2:
        joined = f"{tokens[0]} and {tokens[1]}"
    else:
        joined = ", ".join(tokens[:-1]) + f", and {tokens[-1]}"
    return f"Feature {joined} as visible subject references in the video."


def _compiled_reference_prompt(clip: dict) -> str:
    """Return the H3 prompt with stable Subject->Picture definitions for Ref2VA.

    MiniMax receives reference images as <Picture N> in input order.  The user-facing
    timeline uses <Subject N> for reusable visible content, matching MiniMax's official
    full-reference prompt guide.

    If the authored prompt body never mentions any <Subject N> token, automatically add
    one helper usage line so Ref2VA references still participate instead of being defined
    but never invoked.
    """
    body = _strip_reference_prompt_prefix(_compiled_prompt(clip))
    if not bool(clip.get("use_reference_images", False)):
        return body
    refs = _reference_entries(clip)
    if not refs:
        return body
    defs_text = _reference_prompt_definitions_text(refs)
    auto_use = "" if _subject_tokens_present(body) else _automatic_subject_usage_line(refs)
    parts = ([defs_text] if defs_text else []) + ([auto_use] if auto_use else []) + ([body] if body else [])
    return "\n\n".join(part for part in parts if part).strip()


class RefPresetDialog(QDialog):
    """Persistent reusable Ref2VA picture/subject presets for Timeline clips."""

    def __init__(self, storage_path: Path, parent=None):
        super().__init__(parent)
        self.storage_path = Path(storage_path)
        self.presets: list[dict] = []
        self._loading_editor = False
        self._check_sequence_counter = 0
        self.setWindowTitle("Reference presets")
        self.resize(900, 680)
        self._load_presets()

        root = QVBoxLayout(self)
        hint = QLabel(
            "Check the references to apply to the Timeline clip(s) selected before opening this window. "
            "The checked presets define the complete Ref2VA set for those clips. Checked presets are auto-numbered "
            "in the order you enable them: first checked becomes Subject/Picture 1, next becomes 2, and so on."
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        body = QSplitter(Qt.Orientation.Horizontal, self)
        body.setChildrenCollapsible(False)
        root.addWidget(body, 1)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        lv.addWidget(self.list_widget, 1)
        lbuttons = QHBoxLayout()
        self.add_btn = QPushButton("+ Add image")
        self.delete_btn = QPushButton("Delete preset")
        lbuttons.addWidget(self.add_btn)
        lbuttons.addWidget(self.delete_btn)
        lv.addLayout(lbuttons)
        body.addWidget(left)

        editor = QWidget()
        ev = QVBoxLayout(editor)
        ev.setContentsMargins(6, 0, 0, 0)
        self.thumb = QLabel("No preset selected")
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setFixedHeight(170)
        self.thumb.setFrameShape(QFrame.Shape.StyledPanel)
        ev.addWidget(self.thumb)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.subject_number = QSpinBox()
        self.subject_number.setRange(1, TIMELINE_MAX_REF_IMAGES)
        self.subject_number.setReadOnly(True)
        self.subject_number.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.subject_number.setToolTip("Subject/Picture numbers are assigned automatically in the order presets are checked.")
        form.addRow("Preset name", self.name_edit)
        form.addRow("Subject number", self.subject_number)
        ev.addLayout(form)

        ev.addWidget(QLabel("Picture description"))
        self.picture_edit = QPlainTextEdit()
        self.picture_edit.setPlaceholderText("<Picture 1>: image of a woman with blonde hair and a white dress")
        self.picture_edit.setMaximumHeight(115)
        ev.addWidget(self.picture_edit)

        ev.addWidget(QLabel("Subject description"))
        self.subject_edit = QPlainTextEdit()
        self.subject_edit.setPlaceholderText(
            "<Subject 1>: the woman shown in <Picture 1>. Preserve her exact facial identity, hair, body appearance, and outfit."
        )
        self.subject_edit.setMaximumHeight(145)
        ev.addWidget(self.subject_edit)
        self.save_btn = QPushButton("Save preset changes")
        ev.addWidget(self.save_btn)
        ev.addStretch(1)
        body.addWidget(editor)
        body.setSizes([350, 540])

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_btn = buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_btn.setText("Apply to selected clips")
        root.addWidget(buttons)

        self.add_btn.clicked.connect(self._add_preset)
        self.delete_btn.clicked.connect(self._delete_preset)
        self.save_btn.clicked.connect(self._save_current_editor)
        self.list_widget.currentItemChanged.connect(self._current_changed)
        self.list_widget.itemChanged.connect(self._item_changed)
        self.subject_number.valueChanged.connect(self._subject_number_changed)
        buttons.rejected.connect(self.reject)
        self.apply_btn.clicked.connect(self._validate_and_accept)

        self._rebuild_list()
        self._set_editor_enabled(bool(self.presets))

    def _load_presets(self):
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8")) if self.storage_path.is_file() else {}
        except Exception:
            data = {}
        raw = data.get("presets") if isinstance(data, dict) else []
        self.presets = []
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            self.presets.append({
                "id": str(item.get("id") or uuid.uuid4().hex),
                "name": str(item.get("name") or Path(path).stem).strip(),
                "path": path,
                "picture_description": str(item.get("picture_description") or "").strip(),
                "subject_description": str(item.get("subject_description") or "").strip(),
            })

    def _persist(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
            tmp.write_text(json.dumps({"version": 1, "presets": self.presets}, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.storage_path)
        except Exception as exc:
            QMessageBox.warning(self, "Reference presets", f"Could not save the preset list:\n{exc}")

    def _preset_by_id(self, preset_id: str):
        return next((p for p in self.presets if str(p.get("id")) == str(preset_id)), None)

    def _set_editor_enabled(self, enabled: bool):
        for w in (self.name_edit, self.subject_number, self.picture_edit, self.subject_edit, self.save_btn, self.delete_btn):
            w.setEnabled(bool(enabled))

    def _item_label(self, preset: dict, checked: bool, subject_no: int) -> str:
        name = str(preset.get("name") or Path(str(preset.get("path") or "Reference")).stem)
        return f"Subject {subject_no}  •  {name}" if checked else name

    def _rebuild_list(self, select_id: str | None = None):
        checked_state = {}
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            pid = str(item.data(Qt.ItemDataRole.UserRole) or "")
            checked_state[pid] = (
                item.checkState() == Qt.CheckState.Checked,
                int(item.data(Qt.ItemDataRole.UserRole + 1) or 1),
                int(item.data(Qt.ItemDataRole.UserRole + 2) or 0),
            )
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for pos, preset in enumerate(self.presets, 1):
            pid = str(preset.get("id"))
            checked, subject_no, check_order = checked_state.get(pid, (False, min(pos, TIMELINE_MAX_REF_IMAGES), 0))
            item = QListWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, pid)
            item.setData(Qt.ItemDataRole.UserRole + 1, subject_no)
            item.setData(Qt.ItemDataRole.UserRole + 2, int(check_order) if checked else 0)
            item.setText(self._item_label(preset, checked, subject_no))
            path = Path(str(preset.get("path") or ""))
            if not path.is_file():
                item.setToolTip(f"Image file is missing: {path}")
            self.list_widget.addItem(item)
            if select_id and pid == str(select_id):
                self.list_widget.setCurrentItem(item)
        if self.list_widget.currentItem() is None and self.list_widget.count():
            self.list_widget.setCurrentRow(0)
        self.list_widget.blockSignals(False)
        self._set_editor_enabled(bool(self.list_widget.currentItem()))
        if self.list_widget.currentItem():
            self._load_item_into_editor(self.list_widget.currentItem())
        self._renumber_checked_items_by_selection_order()

    def _current_changed(self, current, previous):
        if current is None:
            self._set_editor_enabled(False)
            self.thumb.clear(); self.thumb.setText("No preset selected")
            return
        self._set_editor_enabled(True)
        self._load_item_into_editor(current)

    def _load_item_into_editor(self, item):
        preset = self._preset_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
        if not preset:
            return
        self._loading_editor = True
        try:
            self.name_edit.setText(str(preset.get("name") or ""))
            self.picture_edit.setPlainText(str(preset.get("picture_description") or ""))
            self.subject_edit.setPlainText(str(preset.get("subject_description") or ""))
            self.subject_number.setValue(max(1, min(TIMELINE_MAX_REF_IMAGES, int(item.data(Qt.ItemDataRole.UserRole + 1) or 1))))
            pix = QPixmap(str(preset.get("path") or ""))
            if not pix.isNull():
                self.thumb.setPixmap(pix.scaled(320, 164, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                self.thumb.setToolTip(str(preset.get("path") or ""))
            else:
                self.thumb.clear(); self.thumb.setText("Image missing")
                self.thumb.setToolTip(str(preset.get("path") or ""))
        finally:
            self._loading_editor = False

    def _first_unused_subject(self) -> int:
        used = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                used.add(int(item.data(Qt.ItemDataRole.UserRole + 1) or 1))
        for n in range(1, TIMELINE_MAX_REF_IMAGES + 1):
            if n not in used:
                return n
        return TIMELINE_MAX_REF_IMAGES

    def _next_check_order(self) -> int:
        self._check_sequence_counter = int(self._check_sequence_counter or 0) + 1
        return self._check_sequence_counter

    def _checked_items_in_selection_order(self):
        items = [self.list_widget.item(i) for i in range(self.list_widget.count()) if self.list_widget.item(i).checkState() == Qt.CheckState.Checked]
        items.sort(key=lambda item: (int(item.data(Qt.ItemDataRole.UserRole + 2) or 0) or 10**9, self.list_widget.row(item)))
        return items

    def _renumber_checked_items_by_selection_order(self):
        self.list_widget.blockSignals(True)
        try:
            checked_items = self._checked_items_in_selection_order()
            for pos, item in enumerate(checked_items, 1):
                item.setData(Qt.ItemDataRole.UserRole + 1, pos)
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                preset = self._preset_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
                if preset:
                    item.setText(self._item_label(preset, item.checkState() == Qt.CheckState.Checked, int(item.data(Qt.ItemDataRole.UserRole + 1) or 1)))
        finally:
            self.list_widget.blockSignals(False)
        current = self.list_widget.currentItem()
        if current is not None:
            self._load_item_into_editor(current)

    def _item_changed(self, item):
        if item is None:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        self.list_widget.blockSignals(True)
        try:
            if checked:
                check_order = int(item.data(Qt.ItemDataRole.UserRole + 2) or 0)
                if check_order <= 0:
                    item.setData(Qt.ItemDataRole.UserRole + 2, self._next_check_order())
            else:
                item.setData(Qt.ItemDataRole.UserRole + 2, 0)
            preset = self._preset_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
            if preset:
                item.setText(self._item_label(preset, checked, int(item.data(Qt.ItemDataRole.UserRole + 1) or 1)))
        finally:
            self.list_widget.blockSignals(False)
        self._renumber_checked_items_by_selection_order()

    def _subject_number_changed(self, value: int):
        if self._loading_editor:
            return
        item = self.list_widget.currentItem()
        if item is None:
            return
        self.list_widget.blockSignals(True)
        try:
            item.setData(Qt.ItemDataRole.UserRole + 1, int(value))
            preset = self._preset_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
            if preset:
                item.setText(self._item_label(preset, item.checkState() == Qt.CheckState.Checked, int(value)))
        finally:
            self.list_widget.blockSignals(False)

    def _add_preset(self):
        name, _ = QFileDialog.getOpenFileName(self, "Add reference preset image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)")
        if not name:
            return
        path = Path(name)
        if not path.is_file():
            return
        preset = {
            "id": uuid.uuid4().hex,
            "name": path.stem,
            "path": str(path),
            "picture_description": f"<Picture 1>: reference image of {path.stem}",
            "subject_description": f"<Subject 1>: the subject shown in <Picture 1>. Preserve its exact identity and appearance.",
        }
        self.presets.append(preset)
        self._persist()
        self._rebuild_list(select_id=preset["id"])

    def _delete_preset(self):
        item = self.list_widget.currentItem()
        if item is None:
            return
        pid = str(item.data(Qt.ItemDataRole.UserRole) or "")
        preset = self._preset_by_id(pid)
        if not preset:
            return
        if QMessageBox.question(self, "Delete reference preset", f'Delete preset "{preset.get("name")}"?') != QMessageBox.StandardButton.Yes:
            return
        self.presets = [p for p in self.presets if str(p.get("id")) != pid]
        self._persist()
        self._rebuild_list()

    def _save_current_editor(self):
        item = self.list_widget.currentItem()
        if item is None:
            return
        preset = self._preset_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
        if not preset:
            return
        name = self.name_edit.text().strip() or Path(str(preset.get("path") or "Reference")).stem
        picture = self.picture_edit.toPlainText().strip()
        subject = self.subject_edit.toPlainText().strip()
        preset["name"] = name
        preset["picture_description"] = picture
        preset["subject_description"] = subject
        item.setText(self._item_label(preset, item.checkState() == Qt.CheckState.Checked, int(item.data(Qt.ItemDataRole.UserRole + 1) or 1)))
        self._persist()

    def _checked_items(self):
        return [self.list_widget.item(i) for i in range(self.list_widget.count()) if self.list_widget.item(i).checkState() == Qt.CheckState.Checked]

    def _validate_and_accept(self):
        self._save_current_editor()
        items = self._checked_items()
        if not items:
            QMessageBox.information(self, "Reference presets", "Select at least one reference preset to apply.")
            return
        if len(items) > TIMELINE_MAX_REF_IMAGES:
            QMessageBox.warning(self, "Reference presets", f"MiniMax Ref2VA supports at most {TIMELINE_MAX_REF_IMAGES} timeline reference images.")
            return
        numbers = [int(item.data(Qt.ItemDataRole.UserRole + 1) or 1) for item in items]
        wanted = list(range(1, len(items) + 1))
        if sorted(numbers) != wanted:
            QMessageBox.warning(
                self, "Reference presets",
                f"For {len(items)} selected presets, assign each Subject number exactly once: " + ", ".join(map(str, wanted)) + "."
            )
            return
        missing = []
        for item in items:
            preset = self._preset_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
            if preset and not Path(str(preset.get("path") or "")).is_file():
                missing.append(str(preset.get("name") or preset.get("path")))
        if missing:
            QMessageBox.warning(self, "Reference presets", "These preset image files are missing:\n\n" + "\n".join(missing))
            return
        self.accept()

    def selected_references(self) -> list[dict]:
        refs = []
        for item in self._checked_items_in_selection_order():
            preset = self._preset_by_id(str(item.data(Qt.ItemDataRole.UserRole) or ""))
            if not preset:
                continue
            refs.append({
                "path": str(preset.get("path") or ""),
                "name": str(preset.get("name") or "Reference"),
                "picture_description": str(preset.get("picture_description") or ""),
                "subject_description": str(preset.get("subject_description") or ""),
                "preset_id": str(preset.get("id") or ""),
            })
        return refs


class TimelineCanvas(QWidget):
    """Compact NLE-like overview of MiniMax H3 generation jobs."""

    clipSelected = Signal(str)  # legacy/plain selection compatibility
    clipSelectionRequested = Signal(str, int)
    clipsReordered = Signal(int, int)
    clipFramesChanged = Signal(str, int)
    clipContextMenuRequested = Signal(str, object)
    audioContextMenuRequested = Signal(str, object)

    RULER_H = 30
    CLIP_TOP = 40
    CLIP_H = 116
    SOUND_GAP = 32
    SOUND_ROW_H = 42
    SOUND_ROW_GAP = 6
    BOTTOM_PAD = 18
    THUMB_W = 46
    THUMB_H = 36

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clips: list[dict] = []
        self.audio_clips: list[dict] = []
        self.selected_id: str | None = None
        self.selected_ids: set[str] = set()
        self.pixels_per_second = 38.0
        self.allowed_frames: list[int] = list(range(124, 720, 17))
        self._rects: list[tuple[str, float, float]] = []
        self._audio_rects: list[tuple[str, QRect]] = []
        self._press_x = 0.0
        self._press_clip_index = -1
        self._dragging = False
        self._resizing = False
        self._thumb_cache: dict[str, QImage | None] = {}
        self._video_meta_cache: dict[str, dict] = {}
        self._thumb_cache_dir = Path(__file__).resolve().parent / "_timeline_thumb_cache"
        self.setMinimumHeight(self._required_height())
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _audio_row_assignments(self):
        rows_end: list[float] = []
        assignments: list[int] = []
        for item in self.audio_clips:
            try:
                start = max(0.0, float(item.get("timeline_start") or 0.0))
            except Exception:
                start = 0.0
            end = _soundtrack_clip_end(item)
            row = None
            for idx, row_end in enumerate(rows_end):
                if start >= row_end - 0.001:
                    row = idx
                    rows_end[idx] = end
                    break
            if row is None:
                row = len(rows_end)
                rows_end.append(end)
            assignments.append(row)
        return assignments, max(1, len(rows_end))

    def _sound_top(self) -> int:
        return self.CLIP_TOP + self.CLIP_H + self.SOUND_GAP

    def _required_height(self) -> int:
        _assignments, rows = self._audio_row_assignments()
        return int(self._sound_top() + rows * (self.SOUND_ROW_H + self.SOUND_ROW_GAP) + self.BOTTOM_PAD)

    def sizeHint(self):
        return QSize(max(900, self._content_width()), self._required_height())

    def _content_width(self) -> int:
        video_total = _timeline_total_seconds(self.clips)
        audio_total = max([_soundtrack_clip_end(x) for x in self.audio_clips] or [0.0])
        total = max(video_total, audio_total)
        return int(max(900, 36 + total * self.pixels_per_second + 36))

    def set_clips(self, clips: list[dict], selected_id: str | None = None, selected_ids=None):
        self.clips = clips
        self.selected_id = selected_id
        if selected_ids is None:
            self.selected_ids = {str(selected_id)} if selected_id else set()
        else:
            self.selected_ids = {str(x) for x in selected_ids if x}
        self.setMinimumWidth(self._content_width())
        self.setMinimumHeight(self._required_height())
        self.resize(self._content_width(), self._required_height())
        self.update()

    def set_audio_clips(self, audio_clips):
        self.audio_clips = list(audio_clips or [])
        self.setMinimumWidth(self._content_width())
        self.setMinimumHeight(self._required_height())
        self.resize(self._content_width(), self._required_height())
        self.update()

    def set_zoom(self, pixels_per_second: float):
        self.pixels_per_second = max(16.0, min(120.0, float(pixels_per_second)))
        self.setMinimumWidth(self._content_width())
        self.setMinimumHeight(self._required_height())
        self.resize(self._content_width(), self._required_height())
        self.update()

    def set_allowed_frames(self, values):
        vals = sorted({int(v) for v in (values or []) if int(v) > 0})
        if vals:
            self.allowed_frames = vals

    def _clip_at(self, x: float):
        for idx, (_, left, right) in enumerate(self._rects):
            if left <= x <= right:
                return idx, left, right
        return -1, 0.0, 0.0

    def _nearest_allowed_frames(self, wanted: int) -> int:
        if not self.allowed_frames:
            return max(1, int(wanted))
        return min(self.allowed_frames, key=lambda v: abs(v - wanted))

    def _ffmpeg_path(self) -> str:
        # The timeline helper can be imported from several different launch
        # locations, so do not rely on a single package-relative ffmpeg lookup.
        try:
            from runtime.ffmpeg_tools import tool_path as ffmpeg_tool_path
            for candidate in ("ffmpeg.exe", "ffmpeg"):
                try:
                    resolved = ffmpeg_tool_path(candidate)
                except Exception:
                    resolved = None
                if resolved:
                    path = Path(str(resolved))
                    if path.is_file():
                        return str(path)
        except Exception:
            pass

        found = shutil.which("ffmpeg.exe") or shutil.which("ffmpeg")
        if found:
            return found

        here = Path(__file__).resolve().parent
        roots = [here, *list(here.parents)[:5]]
        relative_candidates = (
            Path("presets/bin/ffmpeg.exe"),
            Path("presets/bin/ffmpeg"),
            Path("bin/ffmpeg.exe"),
            Path("bin/ffmpeg"),
            Path("ffmpeg.exe"),
            Path("ffmpeg"),
        )
        for root in roots:
            for rel in relative_candidates:
                candidate = root / rel
                if candidate.is_file():
                    return str(candidate)
        return ""

    def _ffprobe_path(self) -> str:
        try:
            from runtime.ffmpeg_tools import tool_path as ffmpeg_tool_path
            for candidate in ("ffprobe.exe", "ffprobe"):
                try:
                    resolved = ffmpeg_tool_path(candidate)
                except Exception:
                    resolved = None
                if resolved:
                    path = Path(str(resolved))
                    if path.is_file():
                        return str(path)
        except Exception:
            pass

        found = shutil.which("ffprobe.exe") or shutil.which("ffprobe")
        if found:
            return found

        here = Path(__file__).resolve().parent
        roots = [here, *list(here.parents)[:5]]
        for root in roots:
            for rel in (
                Path("presets/bin/ffprobe.exe"), Path("presets/bin/ffprobe"),
                Path("bin/ffprobe.exe"), Path("bin/ffprobe"),
                Path("ffprobe.exe"), Path("ffprobe"),
            ):
                candidate = root / rel
                if candidate.is_file():
                    return str(candidate)
        return ""

    @staticmethod
    def _resolution_height_from_text(value) -> int:
        text = str(value or "")
        # Covers values such as "1280 × 704", "1280x704" and preset labels
        # such as "High — 1792 × 768". Use the last WxH pair in the string.
        pairs = re.findall(r"(\d{2,5})\s*[x×X]\s*(\d{2,5})", text)
        if not pairs:
            return 0
        try:
            return int(pairs[-1][1])
        except Exception:
            return 0

    def _clip_rendered_height(self, clip: dict) -> int:
        """Best-effort height of the *active rendered clip*.

        Prefer the actual MP4/image dimensions, cached by file timestamp. This
        makes the HQ badge truthful for older projects too instead of relying on
        whether the clip happened to be produced through the HQ Restart button.
        Saved queue metadata is used only as a fallback.
        """
        path = self._thumbnail_source_path(clip)
        if path is not None:
            try:
                stat = path.stat()
                key = f"{path}|{int(stat.st_mtime_ns)}|{int(stat.st_size)}"
            except Exception:
                key = str(path)
            cached = self._video_meta_cache.get(key)
            if cached is not None:
                return int(cached.get("height") or 0)

            height = 0
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                try:
                    img = QImage(str(path))
                    if not img.isNull():
                        height = int(img.height())
                except Exception:
                    height = 0
            else:
                ffprobe = self._ffprobe_path()
                if ffprobe:
                    try:
                        cp = subprocess.run(
                            [ffprobe, "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=height", "-of", "json", str(path)],
                            capture_output=True, text=True, timeout=8,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        if cp.returncode == 0:
                            data = json.loads(cp.stdout or "{}")
                            stream = (data.get("streams") or [{}])[0]
                            height = int(stream.get("height") or 0)
                    except Exception:
                        height = 0
            self._video_meta_cache[key] = {"height": int(height)}
            if height:
                return int(height)

        generated = clip.get("last_generation_info") if isinstance(clip.get("last_generation_info"), dict) else {}
        for value in (
            generated.get("resolution"),
            (generated.get("settings") or {}).get("resolution") if isinstance(generated.get("settings"), dict) else None,
            (generated.get("settings") or {}).get("widescreen_quality") if isinstance(generated.get("settings"), dict) else None,
            (clip.get("settings") or {}).get("resolution") if isinstance(clip.get("settings"), dict) else None,
            (clip.get("settings") or {}).get("widescreen_quality") if isinstance(clip.get("settings"), dict) else None,
        ):
            height = self._resolution_height_from_text(value)
            if height:
                return height
        return 0

    @staticmethod
    def _actual_generation_settings(clip: dict) -> dict:
        generated = clip.get("last_generation_info") if isinstance(clip.get("last_generation_info"), dict) else {}
        saved = generated.get("settings") if isinstance(generated.get("settings"), dict) else None
        if isinstance(saved, dict):
            return saved
        settings = clip.get("settings")
        return settings if isinstance(settings, dict) else {}

    def _status_badges(self, idx: int, clip: dict) -> tuple[list[str], list[str]]:
        """Return compact left/right badge groups for the active clip."""
        left: list[str] = []
        right: list[str] = []
        settings = self._actual_generation_settings(clip)

        # HQ means actual 704p-or-higher output, independent of how it was made.
        if self._clip_rendered_height(clip) >= 704:
            left.append("HQ")

        # Provenance comes from the settings actually saved with the render when
        # available. Timeline continuation normally uses both latent history and
        # the exact final RGB frame, hence L and F may intentionally coexist.
        generated = clip.get("last_generation_info") if isinstance(clip.get("last_generation_info"), dict) else {}
        latent_used = bool(settings.get("latent_continuation") or generated.get("latent_continuation"))
        if latent_used:
            left.append("L")

        frame_used = bool(
            settings.get("last") or settings.get("first") or
            settings.get("continue_video") or settings.get("continue_last_result") or
            str(clip.get("generation_mode") or "") == "continue"
        )
        if frame_used:
            left.append("F")

        audio_used = bool(
            settings.get("continue_audio_memory") or settings.get("ref_audios") or
            settings.get("audio") or settings.get("audio_path") or settings.get("input_audio")
        )
        if audio_used:
            left.append("A")

        has_refs = bool(clip.get("use_reference_images", False)) and bool(_reference_entries(clip))
        if has_refs:
            right.append("Ref")
        if idx + 1 < len(self.clips):
            transition = _transition_name(clip)
            if transition != "none":
                right.append(f"{_TRANSITION_LABELS.get(transition, transition)} {_transition_overlap_seconds(clip, self.clips[idx + 1]):.1f}s")

        broken = False
        mode = str(clip.get("generation_mode") or "")
        if mode == "continue":
            if idx <= 0:
                broken = True
            else:
                prev = self.clips[idx - 1]
                prev_path = self._thumbnail_source_path(prev)
                if prev_path is None:
                    broken = True
        elif mode in {"source", "loaded"}:
            if self._thumbnail_source_path(clip) is None:
                broken = True
        if broken:
            right.insert(0, "⚠")
        return left, right

    def _thumbnail_source_path(self, clip: dict) -> Path | None:
        source = str(clip.get("output") or clip.get("start_source_video") or "").strip()
        if not source:
            return None
        path = Path(source).expanduser()
        if path.is_file():
            return path.resolve()
        # Older/saved timeline projects can contain relative result paths.
        for base in (Path.cwd(), Path(__file__).resolve().parent):
            candidate = (base / path).resolve()
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _load_thumbnail_image(path: Path) -> QImage | None:
        # Keep the timeline renderer entirely on QImage/raster data. QPixmap is
        # optimized for screen resources and proved unreliable here even though
        # Qt reported the extracted thumbnail as valid.
        try:
            data = path.read_bytes()
            if not data:
                return None
            image = QImage.fromData(data)
            if image.isNull() or image.width() < 2 or image.height() < 2:
                return None
            return image.convertToFormat(QImage.Format.Format_RGB32)
        except Exception:
            return None

    def _clip_thumbnail(self, clip: dict) -> QImage | None:
        path = self._thumbnail_source_path(clip)
        if path is None:
            return None
        try:
            stamp = int(path.stat().st_mtime_ns)
            file_size = int(path.stat().st_size)
        except Exception:
            stamp = 0
            file_size = 0

        # v4: keep decoded thumbnail pixels entirely in memory.  Previous
        # versions went through PNG/JPEG/BMP files and Qt image plugins; on the
        # user's Windows runtime those images reported as valid but painted as
        # empty rectangles.  FFmpeg now emits an exact THUMB_W x THUMB_H RGB24
        # frame to stdout and QImage owns a copied raster buffer.
        cache_key = f"v4raw|{path}|{stamp}|{file_size}|{self.THUMB_W}x{self.THUMB_H}"
        if cache_key in self._thumb_cache:
            return self._thumb_cache[cache_key]

        image: QImage | None = None
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            loaded = self._load_thumbnail_image(path)
            if loaded is not None:
                image = loaded.scaled(
                    self.THUMB_W, self.THUMB_H,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                ).copy(0, 0, self.THUMB_W, self.THUMB_H)
        else:
            ffmpeg = self._ffmpeg_path()
            if ffmpeg:
                try:
                    vf = (
                        f"scale={self.THUMB_W}:{self.THUMB_H}:"
                        "force_original_aspect_ratio=decrease,"
                        f"pad={self.THUMB_W}:{self.THUMB_H}:(ow-iw)/2:(oh-ih)/2:color=black"
                    )
                    proc = subprocess.run(
                        [
                            ffmpeg, "-hide_banner", "-loglevel", "error",
                            "-ss", "0.35", "-i", str(path), "-an",
                            "-frames:v", "1", "-vf", vf,
                            "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
                        ],
                        capture_output=True,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    expected = self.THUMB_W * self.THUMB_H * 3
                    if proc.returncode == 0 and len(proc.stdout) >= expected:
                        raw = proc.stdout[:expected]
                        # QImage initially references Python's buffer; .copy() is
                        # essential so the pixels remain valid after this method
                        # returns and the subprocess/output objects are released.
                        wrapped = QImage(
                            raw, self.THUMB_W, self.THUMB_H, self.THUMB_W * 3,
                            QImage.Format.Format_RGB888,
                        )
                        if not wrapped.isNull():
                            image = wrapped.copy()
                except Exception:
                    image = None

        self._thumb_cache[cache_key] = image
        return image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = self.palette()
        bg = palette.color(palette.ColorRole.Base)
        panel = palette.color(palette.ColorRole.AlternateBase)
        text = palette.color(palette.ColorRole.Text)
        muted = palette.color(palette.ColorRole.PlaceholderText)
        border = palette.color(palette.ColorRole.Mid)
        accent = palette.color(palette.ColorRole.Highlight)
        accent_text = palette.color(palette.ColorRole.HighlightedText)
        painter.fillRect(self.rect(), bg)

        audio_total = max([_soundtrack_clip_end(x) for x in self.audio_clips] or [0.0])
        total_seconds = max(1.0, _timeline_total_seconds(self.clips), audio_total)
        ruler_step = 1
        if self.pixels_per_second < 25:
            ruler_step = 10
        elif self.pixels_per_second < 45:
            ruler_step = 5
        elif self.pixels_per_second < 80:
            ruler_step = 2
        x0 = 26.0
        painter.setPen(QPen(border, 1))
        painter.drawLine(int(x0), self.RULER_H - 3, self.width() - 10, self.RULER_H - 3)
        font = painter.font()
        fm = QFontMetrics(font)
        for sec in range(0, int(math.ceil(total_seconds)) + ruler_step, ruler_step):
            x = x0 + sec * self.pixels_per_second
            painter.drawLine(int(x), self.RULER_H - 8, int(x), self.RULER_H - 2)
            label = f"{sec // 60}:{sec % 60:02d}" if sec >= 60 else f"{sec}s"
            painter.setPen(muted)
            painter.drawText(int(x + 3), 16, label)
            painter.setPen(border)

        self._rects = []
        cursor = x0
        cut_palette = [
            QColor("#375a7f"), QColor("#5b4b8a"), QColor("#476c5e"),
            QColor("#72553d"), QColor("#6a4f67"), QColor("#465c78"),
        ]
        state_colors = {
            "finished": QColor("#2f6b4f"),
            "running": QColor("#236b84"),
            "pending": QColor("#5c6470"),
            "failed": QColor("#783c3c"),
            "cancelled": QColor("#6b5353"),
            "stale": QColor("#806528"),
            "draft": panel,
        }

        for idx, clip in enumerate(self.clips):
            source_duration = _clip_seconds(clip)
            duration = _clip_timeline_seconds(clip)
            width = max(74.0, duration * self.pixels_per_second)
            left, right = cursor, cursor + width
            self._rects.append((str(clip.get("id")), left, right))
            clip_id = str(clip.get("id"))
            selected = clip_id in self.selected_ids or clip_id == self.selected_id
            primary_selected = clip_id == self.selected_id
            status = "stale" if clip.get("stale") else str(clip.get("status") or "draft")
            fill = state_colors.get(status, panel)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(accent if selected else border, 3 if primary_selected else (2 if selected else 1)))
            painter.drawRoundedRect(int(left), self.CLIP_TOP, int(width - 4), self.CLIP_H, 5, 5)

            # Clip header.
            painter.setPen(accent_text if selected else text)
            name = str(clip.get("name") or f"Clip {idx + 1}")
            name_width = int(max(20, width - (58 if bool(clip.get("locked", False)) else 18)))
            info_width = int(max(20, width - 18))
            elided = fm.elidedText(name, Qt.TextElideMode.ElideRight, name_width)
            painter.drawText(int(left + 8), self.CLIP_TOP + 18, elided)
            if bool(clip.get("locked", False)):
                lock_font = QFont(font)
                lock_font.setPointSize(max(7, font.pointSize() - 2))
                lock_font.setBold(True)
                painter.setFont(lock_font)
                lock_text = "LOCK"
                lock_w = painter.fontMetrics().horizontalAdvance(lock_text)
                painter.setPen(QColor("#f4d35e"))
                painter.drawText(int(right - lock_w - 10), self.CLIP_TOP + 18, lock_text)
                painter.setFont(font)
            if clip.get("generation_mode") == "source":
                mode = "▣ Loaded start clip"
            elif clip.get("generation_mode") == "loaded":
                mode = "▣ Loaded clip"
            else:
                mode = "↪ Continue" if clip.get("generation_mode") == "continue" else "◆ New"
            edit_mode = str(clip.get("edit_mode") or "")
            bridge = "  •  ⇥ Next frame" if edit_mode in {"bridge_both", "anchor_next"} else ""
            ref_count = len(_reference_entries(clip)) if bool(clip.get("use_reference_images", False)) else 0
            refs_info = f"  •  Ref2VA ×{ref_count}" if ref_count else ""
            if clip.get("trim_out") is not None:
                duration_text = f"{duration:.2f}s used"
            else:
                duration_text = f"{duration:.2f}s"
            info = f"{mode}  •  {int(clip.get('frames') or 0)}f  •  {duration_text}{bridge}{refs_info}"
            painter.setPen(muted if not selected else accent_text)
            painter.drawText(int(left + 8), self.CLIP_TOP + 38, fm.elidedText(info, Qt.TextElideMode.ElideRight, info_width))

            # Show a compact prompt preview. If an output clip already exists, also
            # draw a small thumbnail so timeline blocks are visually recognizable.
            prompt_y = self.CLIP_TOP + 53
            prompt_h = 42
            box_left = int(left + 5)
            box_w = max(1, int(width - 14))
            painter.setBrush(QBrush(QColor("#375a7f")))
            painter.setPen(QPen(bg, 1))
            painter.drawRect(box_left, prompt_y, box_w, prompt_h)
            if clip.get("generation_mode") in {"source", "loaded"}:
                source = str(clip.get("start_source_video") or clip.get("output") or "").strip()
                prompt = Path(source).name if source else "No loaded video"
            else:
                prompt = _compiled_prompt(clip).replace("\n", " ").strip() or "Empty prompt"

            text_left = box_left + 6
            text_w = max(12, box_w - 12)
            image = self._clip_thumbnail(clip)
            if image is not None and box_w >= 70:
                thumb_rect = QRect(box_left + 4, prompt_y + 3, self.THUMB_W, self.THUMB_H)
                painter.fillRect(thumb_rect, QColor("#223548"))
                # The v4 extractor already returns an exact THUMB_W x THUMB_H
                # raster. Paint it 1:1: no Qt-side resize and no source-rect
                # overloads are involved.
                painter.drawImage(thumb_rect.x(), thumb_rect.y(), image)
                painter.setPen(QPen(QColor("#8ea7bf"), 1))
                # drawRect() uses both the current pen and brush. The prompt
                # preview left a solid blue brush active, which was filling this
                # rectangle *after* drawImage() and hiding the thumbnail.
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(thumb_rect)
                text_left = thumb_rect.right() + 8
                text_w = max(12, box_left + box_w - text_left - 6)

                # Compact status/provenance badges below the thumbnail. They
                # describe the active rendered clip rather than merely the current
                # editor settings, so old clips remain truthful after later edits.
                badge_font = QFont(font)
                badge_font.setPointSize(max(7, font.pointSize() - 2))
                badge_font.setBold(True)
                painter.setFont(badge_font)
                badge_y = prompt_y + prompt_h + 13
                left_badges, right_badges = self._status_badges(idx, clip)
                painter.setPen(QColor("#f4f6f8"))
                if left_badges:
                    painter.drawText(thumb_rect.left(), badge_y, " ".join(left_badges))
                if right_badges:
                    right_text = " ".join(right_badges)
                    right_w = painter.fontMetrics().horizontalAdvance(right_text)
                    # Right-side badges describe the outgoing edge of this clip
                    # (notably "Transition to next"), so pin them to the clip's
                    # right edge.  This both gives long transition labels room and
                    # visually places the label at the cut where the effect occurs.
                    badge_right = int(right - 10)
                    painter.drawText(max(int(left + 8), badge_right - right_w), badge_y, right_text)
                painter.setFont(font)

            painter.setPen(QColor("#f4f6f8"))
            text_rect = QRect(text_left, prompt_y + 2, text_w, prompt_h - 4)
            painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine), fm.elidedText(prompt, Qt.TextElideMode.ElideRight, text_w))

            if clip.get("generation_mode") == "continue" and idx > 0:
                painter.setPen(QPen(accent, 2))
                painter.drawLine(int(left - 12), self.CLIP_TOP + 20, int(left - 3), self.CLIP_TOP + 20)
                painter.drawLine(int(left - 7), self.CLIP_TOP + 16, int(left - 3), self.CLIP_TOP + 20)
                painter.drawLine(int(left - 7), self.CLIP_TOP + 24, int(left - 3), self.CLIP_TOP + 20)

            overlap = _transition_overlap_seconds(clip, self.clips[idx + 1] if idx + 1 < len(self.clips) else None)
            cursor = right - overlap * self.pixels_per_second

        # Dedicated soundtrack lanes share the exact same time ruler/zoom as the
        # video timeline. Overlapping audio clips are automatically stacked into
        # additional rows so voice-over, music and effects remain individually clickable.
        sound_top = self._sound_top()
        painter.setPen(QPen(border, 1))
        painter.drawLine(int(x0), sound_top - 12, self.width() - 10, sound_top - 12)
        painter.setPen(muted)
        painter.drawText(int(x0), sound_top - 16, "Soundtrack")
        self._audio_rects = []
        assignments, _rows = self._audio_row_assignments()
        for ai, item in enumerate(self.audio_clips):
            try:
                start_sec = max(0.0, float(item.get("timeline_start") or 0.0))
            except Exception:
                start_sec = 0.0
            duration = _soundtrack_clip_duration(item)
            if duration <= 0.0005:
                continue
            row = assignments[ai] if ai < len(assignments) else 0
            left = x0 + start_sec * self.pixels_per_second
            width = max(42.0, duration * self.pixels_per_second)
            top = sound_top + row * (self.SOUND_ROW_H + self.SOUND_ROW_GAP)
            rect = QRect(int(left), int(top), int(width), self.SOUND_ROW_H)
            self._audio_rects.append((str(item.get("id") or ""), rect))
            painter.setBrush(QBrush(QColor("#5b4b8a")))
            painter.setPen(QPen(border, 1))
            painter.drawRoundedRect(rect, 5, 5)
            painter.setPen(QColor("#f4f6f8"))
            name = str(item.get("name") or Path(str(item.get("path") or "Audio")).stem or "Audio")
            volume = int(round(float(item.get("volume_percent", 100) or 100)))
            end_sec = start_sec + duration
            line1 = fm.elidedText(name, Qt.TextElideMode.ElideRight, max(20, rect.width() - 12))
            painter.drawText(rect.x() + 6, rect.y() + 17, line1)
            painter.setPen(QColor("#c9c5df"))
            try:
                fade_in = max(0.0, float(item.get("fade_in_seconds") or 0.0))
            except Exception:
                fade_in = 0.0
            try:
                fade_out = max(0.0, float(item.get("fade_out_seconds") or 0.0))
            except Exception:
                fade_out = 0.0
            fade_bits = []
            if fade_in > 0.001:
                fade_bits.append(f"FI {fade_in:.1f}s")
            if fade_out > 0.001:
                fade_bits.append(f"FO {fade_out:.1f}s")
            fade_text = ("  •  " + " / ".join(fade_bits)) if fade_bits else ""
            detail = f"{start_sec:.2f}s → {end_sec:.2f}s  •  {volume}%{fade_text}"
            painter.drawText(rect.x() + 6, rect.y() + 34, fm.elidedText(detail, Qt.TextElideMode.ElideRight, max(20, rect.width() - 12)))

        if not self.audio_clips:
            painter.setPen(muted)
            painter.drawText(int(x0), sound_top + 26, "No soundtrack audio — use + Add audio to add music, voice-over or effects.")

        if not self.clips:
            painter.setPen(muted)
            painter.drawText(28, self.CLIP_TOP + 50, "No clips yet — add a generation clip to start the timeline.")

    def contextMenuEvent(self, event):
        x = float(event.pos().x())
        pos = event.pos()
        for audio_id, rect in reversed(self._audio_rects):
            if audio_id and rect.contains(pos):
                self.audioContextMenuRequested.emit(audio_id, event.globalPos())
                event.accept()
                return
        idx, _left, _right = self._clip_at(x)
        if idx < 0 or pos.y() > self.CLIP_TOP + self.CLIP_H:
            return super().contextMenuEvent(event)
        clip_id = str(self.clips[idx].get("id") or "")
        if not clip_id:
            return
        self.clipContextMenuRequested.emit(clip_id, event.globalPos())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        x = event.position().x()
        if event.position().y() > self.CLIP_TOP + self.CLIP_H:
            return super().mousePressEvent(event)
        idx, left, right = self._clip_at(x)
        if idx < 0:
            return
        clip = self.clips[idx]
        clip_id = str(clip.get("id"))
        modifiers = event.modifiers()
        try:
            modifier_value = int(modifiers.value)
        except Exception:
            modifier_value = int(modifiers)
        self.clipSelectionRequested.emit(clip_id, modifier_value)

        # Ctrl/Shift clicks are selection gestures, not resize/reorder gestures.
        # Locked clips can be selected but cannot be structurally dragged/resized.
        modified_click = modifiers != Qt.KeyboardModifier.NoModifier
        locked = bool(clip.get("locked", False))
        self._press_x = x
        self._press_clip_index = -1 if modified_click or locked else idx
        self._dragging = False
        self._resizing = (
            not modified_click and not locked and (right - x) <= 10
            and len(self.clips) > 0
            and str(clip.get("generation_mode") or "") not in {"source", "loaded"}
        )
        self.update()

    def mouseMoveEvent(self, event):
        if self._press_clip_index < 0 or not (event.buttons() & Qt.MouseButton.LeftButton):
            return super().mouseMoveEvent(event)
        x = event.position().x()
        if self._resizing:
            clip = self.clips[self._press_clip_index]
            _, left, _ = self._rects[self._press_clip_index]
            seconds = max(1.0, (x - left) / self.pixels_per_second)
            wanted = int(round(seconds * FPS))
            frames = self._nearest_allowed_frames(wanted)
            if frames != int(clip.get("frames") or 0):
                clip["frames"] = frames
                self.clipFramesChanged.emit(str(clip.get("id")), frames)
                self.set_clips(self.clips, self.selected_id)
            return
        if abs(x - self._press_x) > 6:
            self._dragging = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.MouseButton.LeftButton and self._dragging and self._press_clip_index >= 0:
                x = event.position().x()
                target = len(self.clips) - 1
                for i, (_, left, right) in enumerate(self._rects):
                    if x < (left + right) / 2:
                        target = i
                        break
                if target != self._press_clip_index:
                    self.clipsReordered.emit(self._press_clip_index, target)
        finally:
            self._press_clip_index = -1
            self._dragging = False
            self._resizing = False
            self.unsetCursor()
            self.update()


class TimelineTab(QWidget):
    """MiniMax H3 long-form planning/generation timeline.

    Director's reusable structure is intentionally retained here: a project owns
    generation chunks, each chunk owns proportional prompt segments, and segment duration edits
    redistribute weight while keeping the generation length fixed. FrameVision's
    existing H3 pipeline remains authoritative for sampling and continuation.
    """

    def __init__(
        self,
        parent=None,
        *,
        settings_provider=None,
        queue_timeline_callback=None,
        assemble_timeline_callback=None,
        preview_result_callback=None,
        open_output_callback=None,
        frame_values=None,
    ):
        super().__init__(parent)
        self.setObjectName("minimaxTimelineTab")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.settings_provider = settings_provider
        self.queue_timeline_callback = queue_timeline_callback
        self.assemble_timeline_callback = assemble_timeline_callback
        self.preview_result_callback = preview_result_callback
        self.open_output_callback = open_output_callback
        self.frame_values = sorted({int(v) for v in (frame_values or range(124, 720, 17))})
        self.project = self._new_project_data()
        self.selected_clip_id: str | None = None
        self.selected_clip_ids: set[str] = set()
        self._selection_anchor_id: str | None = None
        self.selected_cut_index = 0
        self._loading_inspector = False
        self._project_path: Path | None = None

        # Session-wide Timeline edit history. Snapshots contain the complete project
        # model (including clip output paths/job ids), so Undo/Redo restores the
        # rendered clip version that belonged to that edit state instead of only
        # restoring the visible prompt/settings widgets. Video files themselves are
        # not duplicated; the Timeline already keeps previous renders on disk.
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []
        self._history_restoring = False
        self._history_coalesce_key: str | None = None
        self._history_coalesce_timer = QTimer(self)
        self._history_coalesce_timer.setSingleShot(True)
        self._history_coalesce_timer.setInterval(700)
        self._history_coalesce_timer.timeout.connect(self._end_history_coalesce)

        # Timeline recovery paths must exist before _build_ui/_refresh_all because
        # the summary tooltip already refers to the recovery location during the
        # first refresh at startup.
        module_dir = Path(__file__).resolve().parent
        app_root = module_dir.parent if module_dir.name.lower() == "helpers" else module_dir
        self._timeline_root = app_root / "output" / "timeline"
        self._timeline_root.mkdir(parents=True, exist_ok=True)
        self._autosave_temp_path = self._timeline_root / "minimax_timeline_autosave_temp.json"
        # Share the application's persistent QFileDialog history file so the
        # Timeline Load dialog opens where the user last loaded a project, even
        # after GrizzlyMax has been restarted.
        self._file_dialog_history_path = app_root / "presets" / "setsave" / "minimax_file_dialog_history.json"
        self._ref_presets_path = app_root / "presets" / "setsave" / "minimax_timeline_ref_presets.json"

        self._build_ui()
        self._ensure_initial_clip()
        self._refresh_all(select_first=True)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60_000)
        self._autosave_timer.timeout.connect(self._autosave_project)
        self._autosave_timer.start()

    # ------------------------------------------------------------------ model
    def _new_project_data(self):
        return {
            "schema_version": TIMELINE_SCHEMA_VERSION,
            "project_id": uuid.uuid4().hex,
            "name": "MiniMax Timeline",
            "project_folder": "",
            "clips": [],
            "soundtrack_clips": [],
            "soundtrack_mode": "mix",
            "assembled_output": "",
            "assembly_status": "",
            "auto_assemble": False,
            "auto_assemble_pending": False,
            "global_generation_settings": {},
            "hq_restart_override": {},
        }

    def _capture_settings(self):
        if callable(self.settings_provider):
            try:
                data = self.settings_provider()
                captured = copy.deepcopy(data) if isinstance(data, dict) else {}
                # Timeline always owns final assembly. Standalone FL2VA Glue is
                # intentionally ignored here so a Generation-tab checkbox can
                # never become a stale/fantasy Timeline setting.
                captured["glue_results"] = False
                return captured
            except Exception:
                return {}
        return {}

    def _timeline_global_settings(self, raw=None):
        """Return Generation-tab settings safe to use as a Timeline-wide baseline.

        Timeline owns prompt text, clip duration, Ref2VA inputs and continuation
        topology. Those fields must never be overwritten by the global settings
        capture button. Everything else (model, LoRAs, resolution, sampler,
        scheduler, seed, steps, VRAM/runtime options, etc.) remains global.
        """
        settings = copy.deepcopy(raw if isinstance(raw, dict) else self._capture_settings())
        for key in (
            "prompt", "frames", "experimental_long_duration",
            "ref_images", "ref_videos", "ref_audios",
            "first", "last", "continue_video", "continue_last_result",
        ):
            settings.pop(key, None)
        # Timeline decides T2VA / FL2VA / Ref2VA per block at queue time.
        settings["mode"] = 0
        settings["glue_results"] = False
        return settings

    def _blank_clip(self, *, continue_previous=False, source_settings=None):
        global_settings = self.project.get("global_generation_settings") if isinstance(getattr(self, "project", None), dict) else {}
        if isinstance(global_settings, dict) and global_settings:
            settings = copy.deepcopy(global_settings)
        elif isinstance(source_settings, dict):
            settings = copy.deepcopy(source_settings)
        else:
            settings = self._timeline_global_settings()
        # Duration belongs to the Timeline block itself. Global Generation settings
        # never get to change it. New blocks use the existing Timeline default.
        frames = 243
        frames = min(self.frame_values, key=lambda v: abs(v - frames)) if self.frame_values else frames
        prompt = ""
        # Timeline reference images are owned by each clip, not implicitly copied
        # from whatever happens to be loaded on the Generation tab.
        settings["ref_images"] = []
        settings["ref_videos"] = []
        settings["ref_audios"] = []
        settings["glue_results"] = False
        return {
            "id": uuid.uuid4().hex,
            "name": "",
            "frames": frames,
            "generation_mode": "continue" if continue_previous else "new",
            "start_source_video": "",
            "source_duration_seconds": 0.0,
            "edit_mode": "continue_previous" if continue_previous else "standalone",
            "match_next_first_frame": False,  # legacy compatibility; edit workflow owns this now
            "use_reference_images": False,
            "reference_images": [],
            "segments": [{"id": uuid.uuid4().hex, "prompt": prompt, "weight": 1.0}],
            "settings": settings,
            "status": "draft",
            "stale": False,
            "queue_job_id": None,
            "output": "",
            "hq_generated": False,
            "locked": False,
            "notes": "",
            "last_generation_info": {},
            "alternate_candidates": [],
            "speed_multiplier": 1.0,
        }

    def _clips(self):
        return self.project.setdefault("clips", [])

    def _ensure_initial_clip(self):
        if not self._clips():
            clip = self._blank_clip(continue_previous=False)
            clip["name"] = "Clip 1"
            self._clips().append(clip)
            self.selected_clip_id = clip["id"]
            self.selected_clip_ids = {str(clip["id"])}
            self._selection_anchor_id = str(clip["id"])

    def _selected_index(self):
        for i, c in enumerate(self._clips()):
            if str(c.get("id")) == str(self.selected_clip_id):
                return i
        return -1

    def _selected_clip(self):
        idx = self._selected_index()
        return self._clips()[idx] if idx >= 0 else None

    def _invalidate_assembly(self, message="Timeline changed — assemble again after generation."):
        self.project["assembled_output"] = ""
        self.project["assembly_status"] = str(message or "")
        self.project["auto_assemble_pending"] = False

    def _auto_assemble_changed(self, checked):
        self.project["auto_assemble"] = bool(checked)
        if not checked:
            self.project["auto_assemble_pending"] = False

    def _renumber_default_names(self):
        for i, clip in enumerate(self._clips(), 1):
            name = str(clip.get("name") or "").strip()
            if not name or re.fullmatch(r"Clip\s+\d+", name, flags=re.I):
                clip["name"] = f"Clip {i}"

    def _touch_clip(self, index: int, propagate=True):
        # The initial startup Timeline intentionally has no project folder yet.
        # On the first real edit, ask where this project belongs before generation
        # can continue writing clips to the generic main output directory.
        self._ensure_project_setup_for_first_edit()
        clips = self._clips()
        if not (0 <= index < len(clips)):
            return
        clip = clips[index]
        if bool(clip.get("locked", False)):
            return
        self._invalidate_assembly()
        clip.pop("_preserve_downstream_on_finish", None)
        # Stale is now informational only. Editing an already-rendered block marks
        # *that block only* stale so the orange state warns that its media no longer
        # matches the current editor values. Never propagate this downstream and
        # never use it to block preview, trim, generation, or assembly.
        output = str(clip.get("output") or "").strip()
        if output and Path(output).is_file():
            clip["stale"] = True
        else:
            clip.setdefault("stale", False)

    # --------------------------------------------------------------- undo/redo
    def _history_snapshot(self) -> dict:
        return {
            "project": copy.deepcopy(self.project),
            "selected_clip_id": self.selected_clip_id,
            "selected_cut_index": int(self.selected_cut_index or 0),
        }

    @staticmethod
    def _history_same(a: dict, b: dict) -> bool:
        return (
            isinstance(a, dict)
            and isinstance(b, dict)
            and a.get("project") == b.get("project")
            and a.get("selected_clip_id") == b.get("selected_clip_id")
            and int(a.get("selected_cut_index") or 0) == int(b.get("selected_cut_index") or 0)
        )

    def _end_history_coalesce(self):
        self._history_coalesce_key = None

    def _record_undo_state(self, label: str, *, coalesce_key: str | None = None):
        """Store the state immediately before a user edit.

        Repeated text/spin edits can be coalesced for a short interval, while
        structural actions always create their own undo step. Any new edit clears
        Redo, matching normal editor behaviour.
        """
        if self._history_restoring:
            return
        key = str(coalesce_key) if coalesce_key else None
        if key and key == self._history_coalesce_key:
            self._history_coalesce_timer.start()
            return

        snap = self._history_snapshot()
        if not self._undo_stack or not self._history_same(self._undo_stack[-1].get("snapshot", {}), snap):
            self._undo_stack.append({"label": str(label or "Edit"), "snapshot": snap})
        self._redo_stack.clear()
        self._history_coalesce_key = key
        if key:
            self._history_coalesce_timer.start()
        else:
            self._history_coalesce_timer.stop()
        self._update_history_buttons()

    def _reset_history(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._history_coalesce_key = None
        self._history_coalesce_timer.stop()
        self._update_history_buttons()

    def _update_history_buttons(self):
        if not hasattr(self, "undo_btn"):
            return
        self.undo_btn.setEnabled(bool(self._undo_stack))
        self.redo_btn.setEnabled(bool(self._redo_stack))
        if self._undo_stack:
            self.undo_btn.setToolTip(f"Undo: {self._undo_stack[-1].get('label') or 'last Timeline edit'}")
        else:
            self.undo_btn.setToolTip("Nothing to undo in this Timeline session.")
        if self._redo_stack:
            self.redo_btn.setToolTip(f"Redo: {self._redo_stack[-1].get('label') or 'last undone Timeline edit'}")
        else:
            self.redo_btn.setToolTip("Nothing to redo in this Timeline session.")

    def _restore_history_snapshot(self, snap: dict):
        self._history_restoring = True
        self._history_coalesce_key = None
        self._history_coalesce_timer.stop()
        try:
            self.project = copy.deepcopy(snap.get("project") or self._new_project_data())
            self.selected_clip_id = snap.get("selected_clip_id")
            self.selected_clip_ids = {str(self.selected_clip_id)} if self.selected_clip_id else set()
            self._selection_anchor_id = str(self.selected_clip_id) if self.selected_clip_id else None
            self.selected_cut_index = int(snap.get("selected_cut_index") or 0)
            self._refresh_all(select_first=True)
        finally:
            self._history_restoring = False
        self._update_history_buttons()

    def undo_timeline(self):
        self._end_history_coalesce()
        current = self._history_snapshot()
        # Skip accidental no-op entries (for example a cancelled file picker).
        while self._undo_stack and self._history_same(self._undo_stack[-1].get("snapshot", {}), current):
            self._undo_stack.pop()
        if not self._undo_stack:
            self._update_history_buttons()
            return False
        entry = self._undo_stack.pop()
        self._redo_stack.append({"label": entry.get("label") or "Edit", "snapshot": current})
        self._restore_history_snapshot(entry["snapshot"])
        return True

    def redo_timeline(self):
        self._end_history_coalesce()
        current = self._history_snapshot()
        while self._redo_stack and self._history_same(self._redo_stack[-1].get("snapshot", {}), current):
            self._redo_stack.pop()
        if not self._redo_stack:
            self._update_history_buttons()
            return False
        entry = self._redo_stack.pop()
        self._undo_stack.append({"label": entry.get("label") or "Edit", "snapshot": current})
        self._restore_history_snapshot(entry["snapshot"])
        return True

    def initialize_from_current_settings(self):
        """Populate the initial clip after the main window finished building.

        Timeline is inserted before Settings in the tab bar, so its constructor
        runs before every Generation setting widget exists.  Deferring this one
        capture avoids a half-empty first clip while keeping the desired tab order.
        """
        clips = self._clips()
        if len(clips) != 1:
            return
        clip = clips[0]
        if clip.get("settings"):
            return
        settings = self._capture_settings()
        if not settings:
            return
        settings = self._timeline_global_settings(settings)
        clip["settings"] = settings
        self._refresh_all()

    # --------------------------------------------------------------------- UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.new_btn = QPushButton("New")
        self.save_btn = QPushButton("Save")
        self.load_btn = QPushButton("Load")
        self.undo_btn = QPushButton("↶ Undo")
        self.redo_btn = QPushButton("↷ Redo")
        self.undo_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.add_clip_btn = QPushButton("+ Clip")
        self.dup_clip_btn = QPushButton("Duplicate")
        self.del_clip_btn = QPushButton("Delete")
        self.lock_clip_btn = QPushButton("Lock selected")
        self.lock_clip_btn.setToolTip("Lock selected timeline blocks so generation and editing actions cannot replace them.")
        self.selection_btn = QPushButton("Selection")
        self.selection_menu = QMenu(self.selection_btn)
        self.select_all_action = self.selection_menu.addAction("Select all")
        self.select_from_action = self.selection_menu.addAction("Select from here")
        self.select_to_action = self.selection_menu.addAction("Select to here")
        self.select_missing_action = self.selection_menu.addAction("Select all missing")
        self.select_stale_action = self.selection_menu.addAction("Select stale")
        self.selection_menu.addSeparator()
        self.clear_selection_action = self.selection_menu.addAction("Clear selection")
        self.selection_btn.setMenu(self.selection_menu)
        self.left_btn = QPushButton("◀")
        self.right_btn = QPushButton("▶")
        for w in (self.new_btn, self.save_btn, self.load_btn, self.undo_btn, self.redo_btn, self.add_clip_btn, self.dup_clip_btn, self.del_clip_btn, self.lock_clip_btn, self.selection_btn, self.left_btn, self.right_btn):
            toolbar.addWidget(w)
        toolbar.addSpacing(10)
        toolbar.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(16, 100)
        self.zoom_slider.setValue(38)
        self.zoom_slider.setFixedWidth(130)
        toolbar.addWidget(self.zoom_slider)
        toolbar.addStretch(1)
        self.summary_label = QLabel()
        toolbar.addWidget(self.summary_label)
        root.addLayout(toolbar)

        # Timeline actions live inside the Timeline tab so the main workflow is
        # obvious even when the global bottom Generate button is out of view.
        runbar = QHBoxLayout()
        runbar.setSpacing(8)
        self.generate_timeline_btn = QPushButton("▶ Generate Timeline")
        self.generate_timeline_btn.setMinimumHeight(36)
        self.generate_timeline_btn.setToolTip(
            "Choose which part of the timeline to generate."
        )
        self.generate_timeline_menu = QMenu(self.generate_timeline_btn)
        self.generate_all_action = self.generate_timeline_menu.addAction("Generate all clips")
        self.generate_missing_action = self.generate_timeline_menu.addAction("Generate all missing clips")
        self.generate_selected_only_action = self.generate_timeline_menu.addAction("Generate selected clips only")
        self.generate_from_selected_action = self.generate_timeline_menu.addAction("Start generate from selected block")
        self.generate_timeline_menu.addSeparator()
        self.generate_cancel_action = self.generate_timeline_menu.addAction("Cancel")
        self.generate_timeline_btn.setMenu(self.generate_timeline_menu)
        self.generate_all_action.triggered.connect(lambda: self.generate_timeline("all"))
        self.generate_missing_action.triggered.connect(lambda: self.generate_timeline("missing"))
        self.generate_selected_only_action.triggered.connect(lambda: self.generate_timeline("selected_only"))
        self.generate_from_selected_action.triggered.connect(lambda: self.generate_timeline("selected_onward"))
        self.hq_restart_btn = QPushButton("HQ restart")
        self.hq_restart_btn.setMinimumHeight(36)
        self.hq_restart_btn.setToolTip(
            "Use the HQ restart feature when your timeline is tested on a low resolution and results look good."
        )
        self.ref_presets_btn = QPushButton("Ref presets")
        self.ref_presets_btn.setMinimumHeight(36)
        self.ref_presets_btn.setToolTip(
            "Load reusable MiniMax Ref2VA presets and apply them to the currently selected Timeline clip(s)."
        )
        self.add_audio_btn = QPushButton("+ Add audio")
        self.add_audio_btn.setMinimumHeight(36)
        self.add_audio_btn.setToolTip("Add background music, voice-over or another audio file to the soundtrack lanes below the video timeline.")
        self.soundtrack_mode_combo = QComboBox()
        self.soundtrack_mode_combo.setMinimumHeight(36)
        self.soundtrack_mode_combo.addItem("Mix clip audio + soundtrack", "mix")
        self.soundtrack_mode_combo.addItem("Clip audio only", "clips_only")
        self.soundtrack_mode_combo.addItem("Soundtrack only", "soundtrack_only")
        self.soundtrack_mode_combo.setToolTip(
            "Choose the audio used by Assemble Video:\n"
            "• Mix clip audio + soundtrack: generated clip sound and added soundtrack are mixed together.\n"
            "• Clip audio only: soundtrack clips stay in the project but are ignored during assembly.\n"
            "• Soundtrack only: generated clip sound is muted; only added soundtrack audio is used."
        )
        self.assemble_timeline_btn = QPushButton("Assemble Video")
        self.assemble_timeline_btn.setMinimumHeight(36)
        self.assemble_timeline_btn.setToolTip(
            "Join the finished timeline clip outputs, in timeline order, into one final MP4."
        )
        self.use_generation_settings_btn = QPushButton("Use current Generation settings")
        self.use_generation_settings_btn.setMinimumHeight(36)
        self.use_generation_settings_btn.setToolTip(
            "Capture the current Generation-tab settings and apply them to every Timeline generation block. "
            "Timeline prompts, reference images and clip durations are preserved. New clips will inherit this global settings snapshot."
        )
        runbar.addWidget(self.generate_timeline_btn)
        runbar.addWidget(self.hq_restart_btn)
        runbar.addWidget(self.ref_presets_btn)
        runbar.addWidget(self.add_audio_btn)
        runbar.addWidget(self.soundtrack_mode_combo)
        runbar.addWidget(self.assemble_timeline_btn)
        runbar.addWidget(self.use_generation_settings_btn)
        self.generation_resolution_label = QLabel("Resolution: —")
        self.generation_resolution_label.setToolTip(
            "Resolution currently captured for new Timeline generations. "
            "Press 'Use current Generation settings' to refresh it from the Generation tab."
        )
        self.generation_resolution_label.setStyleSheet(
            "QLabel { color: #82e7ff; font-weight: 600; padding: 0 8px; }"
        )
        runbar.addWidget(self.generation_resolution_label)
        runbar.addStretch(1)
        root.addLayout(runbar)

        self.workspace_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(6)

        timeline_panel = QWidget(self.workspace_splitter)
        tl = QVBoxLayout(timeline_panel)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)

        self.chain_help = QLabel(
            "Each block is a new generation, You can drag blocks to reorder them or drag the right side of a block to make duration shorter/longer."
        )
        self.chain_help.setWordWrap(True)
        tl.addWidget(self.chain_help)

        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(False)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.timeline_scroll.setFrameShape(QFrame.Shape.StyledPanel)
        self.canvas = TimelineCanvas()
        self.canvas.set_allowed_frames(self.frame_values)
        self.timeline_scroll.setWidget(self.canvas)
        tl.addWidget(self.timeline_scroll, 1)

        # Compact edit/replacement panel. Only the option list scrolls; the
        # Regenerate button stays permanently visible at the bottom so this panel
        # cannot steal height from the actual timeline as more edit modes are added.
        edit_box = QGroupBox("Edit selected block")
        edit_outer = QVBoxLayout(edit_box)
        edit_outer.setContentsMargins(10, 8, 10, 10)
        edit_outer.setSpacing(6)

        edit_scroll_contents = QWidget()
        ev = QVBoxLayout(edit_scroll_contents)
        ev.setContentsMargins(2, 2, 8, 2)
        ev.setSpacing(6)
        self.edit_selected_label = QLabel("Select a block to choose how it should be regenerated.")
        self.edit_selected_label.setWordWrap(True)
        ev.addWidget(self.edit_selected_label)

        self.edit_mode_group = QButtonGroup(self)
        self.edit_mode_group.setExclusive(True)
        self.edit_bridge_both = QRadioButton("Continue previous block and end with start frame of the next block")
        self.edit_continue_previous = QRadioButton("Continue from previous block but do not end with the start frame of next block")
        self.edit_anchor_next = QRadioButton("Do not use previous block to start but end with the first frame of the next block")
        self.edit_standalone = QRadioButton("Do not use previous and next block")
        self.edit_bridge_both.setToolTip("Continue from the previous video + use first frame of the next video as the end frame")
        self.edit_continue_previous.setToolTip("Continue from the previous video and create a new free ending. Existing continuation clips after this block may need regeneration.")
        self.edit_anchor_next.setToolTip("Start independently from the previous timeline block and use the first frame of the next video as the end frame.")
        self.edit_standalone.setToolTip("Standalone clip")
        for button, mode in (
            (self.edit_bridge_both, "bridge_both"),
            (self.edit_continue_previous, "continue_previous"),
            (self.edit_anchor_next, "anchor_next"),
            (self.edit_standalone, "standalone"),
        ):
            self.edit_mode_group.addButton(button)
            button.setProperty("edit_mode", mode)
            ev.addWidget(button)
        # Keep a real scroll tail below the final radio button.  Qt can otherwise
        # report a content size that stops at the last control's top edge while
        # the sticky footer sits immediately below the viewport, making the final
        # option impossible to scroll fully above the Regenerate button.
        ev.addSpacing(28)
        edit_scroll_contents.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        # Four radio rows + the selected-clip label need more vertical content than
        # the intentionally compact viewport.  Use an explicit floor so the scroll
        # range always reaches past the final option on all font/DPI scales.
        edit_scroll_contents.setMinimumHeight(max(220, edit_scroll_contents.sizeHint().height() + 28))

        self.edit_scroll = QScrollArea()
        self.edit_scroll.setWidgetResizable(True)
        self.edit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.edit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.edit_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Keep this deliberately short: the group title + roughly two/three
        # option rows are visible at once.  Everything else belongs to the
        # inner vertical scrollbar, while the Regenerate button lives in a
        # separate footer and can never cover the scroll contents.
        self.edit_scroll.setFixedHeight(82)
        self.edit_scroll.setWidget(edit_scroll_contents)
        edit_outer.addWidget(self.edit_scroll, 0)

        self.generate_selected_btn = QPushButton("(Re)generate selected block")
        self.generate_selected_btn.setFixedHeight(36)
        self.generate_selected_btn.setToolTip("Regenerate only the selected block using the replacement continuity mode selected above.")
        edit_outer.addWidget(self.generate_selected_btn, 0)

        # Fixed overall height prevents this panel from expanding and forcing
        # the main timeline itself to scroll vertically.  The option list owns
        # its own scrollbar instead.
        edit_box.setFixedHeight(154)
        tl.addWidget(edit_box, 0)

        inspector_scroll = QScrollArea(self.workspace_splitter)
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setMinimumWidth(330)
        inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        inspector = QWidget()
        iv = QVBoxLayout(inspector)
        iv.setContentsMargins(4, 0, 4, 4)
        iv.setSpacing(8)
        inspector_scroll.setWidget(inspector)

        clip_box = QGroupBox("Selected generation clip")
        cf = QFormLayout(clip_box)
        self.clip_form = cf
        self.clip_name = QLineEdit()
        self.gen_mode = QComboBox()
        self.frames_combo = QComboBox()
        for frames in self.frame_values:
            self.frames_combo.addItem(f"{frames} frames — {frames / FPS:.2f} s", frames)
        self.state_label = QLabel("Draft")
        self.clip_output_label = QLabel("—")
        self.clip_output_label.setWordWrap(True)
        self.clip_output_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        clip_result_actions = QWidget()
        clip_result_l = QHBoxLayout(clip_result_actions)
        clip_result_l.setContentsMargins(0, 0, 0, 0)
        clip_result_l.setSpacing(6)
        self.preview_clip_btn = QPushButton("Preview result")
        self.open_clip_btn = QPushButton("Open folder")
        clip_result_l.addWidget(self.preview_clip_btn)
        clip_result_l.addWidget(self.open_clip_btn)
        clip_result_l.addStretch(1)
        cf.addRow("Name", self.clip_name)
        cf.addRow("Generation", self.gen_mode)
        self.start_source_label = QLabel("—")
        self.start_source_label.setWordWrap(True)
        self.start_source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        cf.addRow("Start video", self.start_source_label)
        cf.addRow("Duration", self.frames_combo)
        cf.addRow("State", self.state_label)
        cf.addRow("Output", self.clip_output_label)
        cf.addRow("", clip_result_actions)
        iv.addWidget(clip_box)

        self.prompt_box = QGroupBox("Prompt")
        prompt_box = self.prompt_box
        pv = QVBoxLayout(prompt_box)
        self.cut_prompt = QPlainTextEdit()
        self.cut_prompt.setPlaceholderText(
            "Prompt for this H3 generation. You can use the normal Prompt Builder output here, including multiple shots or timestamps."
        )
        self.cut_prompt.setMinimumHeight(240)
        pv.addWidget(self.cut_prompt)
        iv.addWidget(prompt_box)

        self.references_box = QGroupBox("Reference images")
        rv = QVBoxLayout(self.references_box)
        rv.setSpacing(6)
        self.use_refs_check = QCheckBox("Use reference images (Ref2VA)")
        self.use_refs_warning = QLabel("Using reference image(s) disables the selection to continue from a previous clip.")
        self.use_refs_warning.setWordWrap(True)
        self.use_refs_warning.setVisible(False)
        self.use_refs_guide = QLabel(
            "Tip: the Insert buttons add a ready-to-use subject snippet. If your prompt body never mentions any <Subject N>, Timeline now automatically adds one helper sentence so the loaded reference image(s) are actually invoked."
        )
        self.use_refs_guide.setWordWrap(True)
        self.use_refs_guide.setVisible(False)
        self.add_refs_btn = QPushButton("+ Add reference images")
        self.add_refs_btn.setToolTip(f"Add up to {TIMELINE_MAX_REF_IMAGES} reference images for this timeline clip.")
        self.refs_rows_widget = QWidget()
        self.refs_rows_layout = QVBoxLayout(self.refs_rows_widget)
        self.refs_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.refs_rows_layout.setSpacing(6)
        rv.addWidget(self.use_refs_check)
        rv.addWidget(self.use_refs_warning)
        rv.addWidget(self.use_refs_guide)
        rv.addWidget(self.add_refs_btn)
        rv.addWidget(self.refs_rows_widget)
        iv.addWidget(self.references_box)

        self.settings_box = QGroupBox("MiniMax settings")
        settings_box = self.settings_box
        sf = QFormLayout(settings_box)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 2147483647)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1, 100)
        self.scheduler_combo = QComboBox(); self.scheduler_combo.addItems(["simple", "beta"])
        self.audio_memory_check = QCheckBox("Carry audio memory")
        self.model_label = QLabel("—"); self.model_label.setWordWrap(True)
        self.refs_label = QLabel("—"); self.refs_label.setWordWrap(True)
        self.loras_label = QLabel("—"); self.loras_label.setWordWrap(True)
        sf.addRow("Seed", self.seed_spin)
        sf.addRow("Steps", self.steps_spin)
        sf.addRow("Scheduler", self.scheduler_combo)
        sf.addRow("", self.audio_memory_check)
        sf.addRow("Model", self.model_label)
        sf.addRow("References", self.refs_label)
        sf.addRow("LoRAs", self.loras_label)
        iv.addWidget(settings_box)
        iv.addStretch(1)

        self.workspace_splitter.addWidget(timeline_panel)
        self.workspace_splitter.addWidget(inspector_scroll)
        self.workspace_splitter.setStretchFactor(0, 4)
        self.workspace_splitter.setStretchFactor(1, 1)
        self.workspace_splitter.setSizes([950, 390])
        root.addWidget(self.workspace_splitter, 1)

        # Connections.
        self.new_btn.clicked.connect(self.new_project)
        self.save_btn.clicked.connect(self.save_project)
        self.load_btn.clicked.connect(self.load_project)
        self.undo_btn.clicked.connect(self.undo_timeline)
        self.redo_btn.clicked.connect(self.redo_timeline)
        self.hq_restart_btn.clicked.connect(self.hq_restart)
        self.generate_selected_btn.clicked.connect(self.generate_selected)
        self.assemble_timeline_btn.clicked.connect(self.assemble_timeline)
        self.use_generation_settings_btn.clicked.connect(self.use_current_generation_settings)
        self.preview_clip_btn.clicked.connect(self.preview_selected_result)
        self.open_clip_btn.clicked.connect(self.open_selected_output)
        self.add_clip_btn.clicked.connect(self.add_clip)
        self.dup_clip_btn.clicked.connect(self.duplicate_clip)
        self.del_clip_btn.clicked.connect(self.delete_clip)
        self.lock_clip_btn.clicked.connect(self.toggle_lock_selected)
        self.select_all_action.triggered.connect(self.select_all_clips)
        self.select_from_action.triggered.connect(self.select_from_here)
        self.select_to_action.triggered.connect(self.select_to_here)
        self.select_missing_action.triggered.connect(self.select_all_missing)
        self.select_stale_action.triggered.connect(self.select_all_stale)
        self.clear_selection_action.triggered.connect(self.clear_selection)
        self.left_btn.clicked.connect(lambda: self.move_clip(-1))
        self.right_btn.clicked.connect(lambda: self.move_clip(1))
        self.zoom_slider.valueChanged.connect(self.canvas.set_zoom)
        self.canvas.clipSelected.connect(self.select_clip)
        self.canvas.clipSelectionRequested.connect(self._selection_requested)
        self.canvas.clipsReordered.connect(self._reorder_clips)
        self.canvas.clipFramesChanged.connect(self._canvas_frames_changed)
        self.canvas.clipContextMenuRequested.connect(self._show_clip_context_menu)
        self.canvas.audioContextMenuRequested.connect(self._show_audio_context_menu)
        self.ref_presets_btn.clicked.connect(self._open_ref_presets)
        self.add_audio_btn.clicked.connect(self._add_soundtrack_audio)
        self.soundtrack_mode_combo.currentIndexChanged.connect(self._soundtrack_mode_changed)
        self.clip_name.textEdited.connect(self._clip_name_changed)
        self.gen_mode.currentIndexChanged.connect(self._mode_changed)
        self.frames_combo.currentIndexChanged.connect(self._frames_changed)
        self.seed_spin.valueChanged.connect(self._settings_changed)
        self.steps_spin.valueChanged.connect(self._settings_changed)
        self.scheduler_combo.currentTextChanged.connect(self._settings_changed)
        self.audio_memory_check.toggled.connect(self._settings_changed)
        self.edit_mode_group.buttonClicked.connect(self._edit_mode_changed)
        self.cut_prompt.textChanged.connect(self._cut_prompt_changed)
        self.use_refs_check.toggled.connect(self._reference_mode_changed)
        self.add_refs_btn.clicked.connect(self._add_reference_images)

    # -------------------------------------------------------------- refreshers
    def _refresh_all(self, *, select_first=False):
        self._update_history_buttons()
        self._refresh_generation_resolution_label()
        if hasattr(self, "soundtrack_mode_combo"):
            mode = str(self.project.get("soundtrack_mode") or "mix")
            if mode not in {"mix", "clips_only", "soundtrack_only"}:
                mode = "mix"
                self.project["soundtrack_mode"] = mode
            idx = self.soundtrack_mode_combo.findData(mode)
            self.soundtrack_mode_combo.blockSignals(True)
            try:
                self.soundtrack_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                self.soundtrack_mode_combo.blockSignals(False)
        valid_ids = {str(c.get("id")) for c in self._clips()}
        self.selected_clip_ids = {str(x) for x in self.selected_clip_ids if str(x) in valid_ids}
        if select_first and self._clips() and not self.selected_clip_id:
            self.selected_clip_id = str(self._clips()[0]["id"])
        if self.selected_clip_id and str(self.selected_clip_id) not in valid_ids:
            self.selected_clip_id = next(iter(self.selected_clip_ids), None)
        if self.selected_clip_id:
            self.selected_clip_ids.add(str(self.selected_clip_id))
        elif self.selected_clip_ids:
            self.selected_clip_id = next((str(c.get("id")) for c in self._clips() if str(c.get("id")) in self.selected_clip_ids), None)
        if select_first and self.selected_clip_id:
            self.selected_clip_ids.add(str(self.selected_clip_id))
            self._selection_anchor_id = str(self.selected_clip_id)
        total = _timeline_total_seconds(self._clips())
        selection_suffix = f"  •  {len(self.selected_clip_ids)} selected" if len(self.selected_clip_ids) > 1 else ""
        locked_count = sum(1 for c in self._clips() if bool(c.get("locked", False)))
        lock_suffix = f"  •  {locked_count} locked" if locked_count else ""
        self.summary_label.setText(f"{len(self._clips())} clips  •  {total:.2f}s  •  {round(total * FPS)} timeline frames{selection_suffix}{lock_suffix}")
        self.generate_timeline_btn.setEnabled(bool(self._clips()))
        self.hq_restart_btn.setEnabled(bool(self._clips()))
        ready, reason = self._assembly_ready()
        self.assemble_timeline_btn.setEnabled(
            bool(self._clips()) and not bool(getattr(self, "_timeline_assemble_click_busy", False))
        )
        self.assemble_timeline_btn.setToolTip(
            "Assemble every Timeline block that has an existing output file, in Timeline order. Stale status is ignored."
            if ready else reason
        )
        self.summary_label.setToolTip(
            str(self._project_path) if self._project_path
            else f"Autosaving recovery project to {self._autosave_temp_path}"
        )
        self.canvas.set_clips(self._clips(), self.selected_clip_id, self.selected_clip_ids)
        self.canvas.set_audio_clips(self.project.get("soundtrack_clips") or [])
        selected_clips = [c for c in self._clips() if str(c.get("id")) in self.selected_clip_ids]
        any_selected = bool(selected_clips)
        all_locked = any_selected and all(bool(c.get("locked", False)) for c in selected_clips)
        self.lock_clip_btn.setEnabled(any_selected)
        self.lock_clip_btn.setText("Unlock selected" if all_locked else "Lock selected")
        self.del_clip_btn.setEnabled(any(not bool(c.get("locked", False)) for c in selected_clips))
        primary = self._selected_clip()
        primary_locked = bool((primary or {}).get("locked", False))
        self.left_btn.setEnabled(bool(primary) and not primary_locked)
        self.right_btn.setEnabled(bool(primary) and not primary_locked)
        self._load_inspector()

    def _load_inspector(self):
        clip = self._selected_clip()
        self._loading_inspector = True
        try:
            enabled = clip is not None
            for w in (self.clip_name, self.gen_mode, self.frames_combo, self.seed_spin, self.steps_spin,
                      self.scheduler_combo, self.audio_memory_check, self.cut_prompt):
                w.setEnabled(enabled)
            if clip is None:
                self.state_label.setText("No clip selected")
                self.clip_output_label.setText("—")
                self.start_source_label.setText("—")
                self.start_source_label.setVisible(False)
                if hasattr(self.clip_form, "setRowVisible"):
                    self.clip_form.setRowVisible(self.start_source_label, False)
                self.prompt_box.setEnabled(False)
                self.references_box.setEnabled(False)
                self._clear_reference_rows()
                self.use_refs_warning.setVisible(False)
                self.use_refs_guide.setVisible(False)
                self.settings_box.setEnabled(False)
                self.preview_clip_btn.setEnabled(False); self.open_clip_btn.setEnabled(False)
                self._refresh_edit_workflow(None)
                self.cut_prompt.clear()
                return
            settings = clip.setdefault("settings", {})
            locked = bool(clip.get("locked", False))
            self.clip_name.setText(str(clip.get("name") or ""))
            selected_index = self._selected_index()
            self._populate_generation_modes(selected_index, str(clip.get("generation_mode") or "new"))
            mode_name = str(clip.get("generation_mode") or "")
            is_media_clip = mode_name in {"source", "loaded"}
            source_path = str(clip.get("start_source_video") or clip.get("output") or "")
            self.start_source_label.setText(source_path if is_media_clip and source_path else "—")
            self.start_source_label.setVisible(is_media_clip)
            if hasattr(self.clip_form, "setRowVisible"):
                self.clip_form.setRowVisible(self.start_source_label, is_media_clip)
            editable = enabled and not is_media_clip and not locked
            self.clip_name.setEnabled(enabled and not locked)
            self.gen_mode.setEnabled(enabled and not locked and not is_media_clip)
            self.prompt_box.setEnabled(editable)
            self.references_box.setEnabled(editable)
            self.settings_box.setEnabled(editable)
            self.use_refs_check.blockSignals(True)
            self.use_refs_check.setChecked(bool(clip.get("use_reference_images", False)))
            self.use_refs_check.blockSignals(False)
            refs_enabled = bool(clip.get("use_reference_images", False))
            self.use_refs_warning.setVisible(refs_enabled)
            self.use_refs_guide.setVisible(refs_enabled)
            self._refresh_reference_rows(clip)
            # A loaded start video is already the first timeline result; it needs no H3 prompt or generation settings.
            for w in (self.frames_combo, self.seed_spin, self.steps_spin, self.scheduler_combo,
                      self.audio_memory_check, self.cut_prompt):
                w.setEnabled(editable)
            fidx = self.frames_combo.findData(int(clip.get("frames") or 243))
            if fidx >= 0: self.frames_combo.setCurrentIndex(fidx)
            self.seed_spin.setValue(int(settings.get("seed", -1)))
            self.steps_spin.setValue(int(settings.get("steps", 15)))
            sched = str(settings.get("scheduler") or "beta")
            if self.scheduler_combo.findText(sched) < 0: self.scheduler_combo.addItem(sched)
            self.scheduler_combo.setCurrentText(sched)
            # Old projects may still contain glue_results=True. Timeline no longer
            # exposes or honors that flag; clips remain separate until assembly.
            settings["glue_results"] = False
            self.audio_memory_check.setChecked(bool(settings.get("continue_audio_memory", True)))
            self._refresh_edit_workflow(clip)
            state = str(clip.get("status") or "draft").title()
            if locked: state = f"Locked — {state}"
            self.state_label.setText(state)
            output = str(clip.get("output") or "")
            preview_path = self._clip_preview_path(clip)
            self.clip_output_label.setText(output if output else "—")
            # Stale is metadata only. Preview/open-folder availability depends on
            # whether preserved media still exists, not on generation validity.
            self.preview_clip_btn.setEnabled(bool(preview_path))
            self.open_clip_btn.setEnabled(bool(preview_path))
            self._refresh_settings_summary(clip)
            self._load_prompt_editor()
        finally:
            self._loading_inspector = False

    def _refresh_settings_summary(self, clip):
        s = clip.get("settings") or {}
        use_hybrid = bool(s.get("use_hybrid_model"))
        use_ref2va = bool(clip.get("use_reference_images", False)) or int(s.get("mode", 0)) == 2
        model = s.get("hybrid_model") if use_hybrid else (s.get("ref2va_model") if use_ref2va else s.get("fl2va_model"))
        self.model_label.setText(Path(str(model)).name if model else "Default / auto-resolved")
        timeline_refs = _reference_entries(clip) if bool(clip.get("use_reference_images", False)) else []
        refs = list(s.get("ref_videos") or []) + list(s.get("ref_audios") or [])
        if s.get("first"): refs.insert(0, s.get("first"))
        if s.get("last"): refs.append(s.get("last"))
        total_refs = len(timeline_refs) + len(refs)
        self.refs_label.setText(f"{len(timeline_refs)} timeline image reference(s)" if timeline_refs else (f"{total_refs} source/reference item(s)" if total_refs else "None"))
        loras = [x for x in (s.get("loras") or []) if isinstance(x, dict) and str(x.get("path") or "").strip()]
        if loras:
            self.loras_label.setText("; ".join(f"{Path(str(x.get('path'))).name} @ {float(x.get('strength', 1.0)):.2f}" for x in loras))
        else:
            self.loras_label.setText("None")

    def _load_prompt_editor(self):
        clip = self._selected_clip()
        self.cut_prompt.blockSignals(True)
        try:
            if not clip:
                self.cut_prompt.clear()
                self.cut_prompt.setEnabled(False)
                return
            if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
                self.cut_prompt.clear()
                self.cut_prompt.setPlaceholderText("Loaded video clips do not need a generation prompt.")
                self.cut_prompt.setEnabled(False)
                return
            self.cut_prompt.setPlaceholderText("Prompt for this H3 generation. You can use the normal Prompt Builder output here, including multiple shots or timestamps.")
            # Timeline v2 presents one normal H3 prompt per generation. Legacy
            # multi-segment projects are flattened into their timestamped compiled
            # prompt once, so no authored content is lost.
            segments = clip.get("segments") or []
            if len(segments) > 1:
                merged = _compiled_prompt(clip)
                clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": merged, "weight": 1.0}]
                segments = clip["segments"]
            if not segments:
                clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0}]
                segments = clip["segments"]
            self.cut_prompt.setEnabled(True)
            self.cut_prompt.setPlainText(str(segments[0].get("prompt") or ""))
        finally:
            self.cut_prompt.blockSignals(False)

    # ------------------------------------------------------------- project I/O
    def _write_project_json(self, path: Path):
        """Write the current timeline atomically so an interrupted autosave cannot corrupt it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.project["schema_version"] = TIMELINE_SCHEMA_VERSION
        payload = json.dumps(self.project, indent=2, ensure_ascii=False)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(path)

    def _autosave_project(self):
        """Silent 60-second autosave. Unsaved projects use a temporary JSON file."""
        try:
            target = self._project_path or self._autosave_temp_path
            self._write_project_json(target)
        except Exception:
            # Autosave must never interrupt generation or editing with a modal error.
            pass

    def _new_project_setup_dialog(self, *, first_edit=False):
        dialog = QDialog(self)
        dialog.setWindowTitle("New timeline project")
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)

        if first_edit:
            intro = QLabel(
                "You started a new project. Please enter a project name and "
                "(optional) an output folder."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

        form = QFormLayout()
        name_edit = QLineEdit()
        name_edit.setText("MiniMax Timeline")
        name_edit.selectAll()

        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(6)
        folder_edit = QLineEdit()
        folder_edit.setPlaceholderText(
            f"Leave empty for {self._timeline_root / '<project name>'}"
        )
        browse_btn = QPushButton("Browse…")

        def browse_folder():
            start_dir = folder_edit.text().strip() or str(self._timeline_root)
            chosen = QFileDialog.getExistingDirectory(
                dialog, "Choose timeline project folder", start_dir
            )
            if chosen:
                folder_edit.setText(chosen)

        browse_btn.clicked.connect(browse_folder)
        folder_layout.addWidget(folder_edit, 1)
        folder_layout.addWidget(browse_btn)
        form.addRow("Project name:", name_edit)
        form.addRow("Project folder:", folder_row)
        layout.addLayout(form)

        note = QLabel(
            f"Project folders start from {self._timeline_root}. "
            "If Project folder is left empty, a folder named after the project is created there automatically."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        while True:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            project_name = str(name_edit.text() or "").strip()
            if not project_name:
                QMessageBox.warning(dialog, "Project name required", "Enter a name for the new timeline project.")
                continue
            folder_text = str(folder_edit.text() or "").strip()
            if folder_text:
                project_folder = Path(folder_text).expanduser()
                if not project_folder.is_absolute():
                    project_folder = self._timeline_root / project_folder
            else:
                project_folder = self._timeline_root / _safe_project_name(project_name)
            try:
                project_folder.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                QMessageBox.critical(dialog, "Project folder", f"Could not create the project folder:\n{project_folder}\n\n{exc}")
                continue
            return project_name, project_folder

    def _ensure_project_setup_for_first_edit(self):
        """Ask for project identity the first time the startup Timeline is edited.

        Pressing New already performs this setup. This guard exists for the common
        case where the user starts editing the default Timeline immediately after
        opening the app.
        """
        current_folder = str(self.project.get("project_folder") or "").strip()
        if current_folder:
            return True

        setup = self._new_project_setup_dialog(first_edit=True)
        if setup is None:
            return False

        project_name, project_folder = setup
        self.project["name"] = project_name
        self.project["project_folder"] = str(project_folder)

        # Keep the recovery file in output/timeline/ as requested. Only its content
        # changes to reflect the newly named project.
        try:
            self._write_project_json(self._autosave_temp_path)
        except Exception:
            pass

        # Refresh the visible project name without turning the programmatic update
        # into another user edit.
        if hasattr(self, "project_name"):
            self.project_name.blockSignals(True)
            try:
                self.project_name.setText(project_name)
            finally:
                self.project_name.blockSignals(False)
        self.summary_label.setToolTip(
            f"Project folder: {project_folder}\nRecovery: {self._autosave_temp_path}"
        )
        return True

    def new_project(self):
        # If the current Timeline contains work, make the close decision explicit.
        # Save Before Close never discards the recovery copy unless the real save
        # completed successfully; Cancel leaves the current project untouched.
        has_content = bool(self._clips()) and (
            len(self._clips()) > 1
            or any(str(s.get("prompt") or "").strip() for c in self._clips() for s in c.get("segments") or [])
            or any(str(c.get("output") or "").strip() for c in self._clips())
        )
        if has_content:
            box = QMessageBox(self)
            box.setWindowTitle("New timeline")
            box.setIcon(QMessageBox.Icon.Question)
            box.setText("Close the current timeline and start a new project?")
            box.setInformativeText("Save the current project first or discard it.")
            save_btn = box.addButton("Save before close", QMessageBox.ButtonRole.AcceptRole)
            discard_btn = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
            cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_btn or clicked is None:
                return
            if clicked is save_btn:
                if not self.save_project():
                    return
            elif clicked is not discard_btn:
                return

        setup = self._new_project_setup_dialog()
        if setup is None:
            return
        project_name, project_folder = setup

        # The previous recovery file belongs to the project being closed. Once the
        # user has explicitly saved or discarded it, remove it before starting a
        # fresh recovery stream for the new project.
        try:
            if self._autosave_temp_path.exists():
                self._autosave_temp_path.unlink()
        except Exception:
            pass

        self.project = self._new_project_data()
        self.project["name"] = project_name
        self.project["project_folder"] = str(project_folder)
        self._project_path = None
        self.selected_clip_id = None
        self.selected_clip_ids = set()
        self._selection_anchor_id = None
        self.selected_cut_index = 0
        self._ensure_initial_clip()
        self._reset_history()
        # Write the first recovery snapshot immediately; subsequent autosaves keep
        # updating the same file under output/timeline/.
        try:
            self._write_project_json(self._autosave_temp_path)
        except Exception:
            pass
        self._refresh_all(select_first=True)

    def save_project(self):
        # The temporary autosave is only a recovery file. It must never silently
        # become the user's "saved project" and it must never be deleted until a
        # real user-chosen destination has been written successfully.
        current_path = Path(self._project_path) if self._project_path else None
        needs_destination = (
            current_path is None
            or current_path == self._autosave_temp_path
            or current_path.name == self._autosave_temp_path.name
        )

        if needs_destination:
            suggested_name = _safe_project_name(self.project.get("name")).replace(" ", "_") + ".json"
            project_folder = str(self.project.get("project_folder") or "").strip()
            suggested = str((Path(project_folder) if project_folder else self._timeline_root) / suggested_name)
            name, _ = QFileDialog.getSaveFileName(
                self,
                "Save MiniMax timeline",
                suggested,
                "MiniMax Timeline (*.json);;JSON (*.json)",
            )
            if not name:
                # User cancelled: keep the recovery autosave untouched.
                return False
            chosen_path = Path(name)
        else:
            chosen_path = current_path

        try:
            self._write_project_json(chosen_path)
            if not chosen_path.is_file():
                raise OSError(f"Timeline save did not create the requested file:\n{chosen_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save timeline failed", str(exc))
            return False

        # Only now commit the real project path.
        self._project_path = chosen_path
        self.project["project_folder"] = str(chosen_path.parent)

        # Delete the temporary recovery copy only after a distinct real save file
        # definitely exists. If cleanup fails, leave it in place; an extra recovery
        # file is safer than losing the project.
        try:
            if (
                self._autosave_temp_path.exists()
                and self._autosave_temp_path.resolve() != chosen_path.resolve()
            ):
                self._autosave_temp_path.unlink()
        except Exception:
            pass

        self.summary_label.setToolTip(str(self._project_path))
        QMessageBox.information(
            self,
            "Timeline saved",
            f"Timeline project saved to:\n{self._project_path}",
        )
        return True

    def _timeline_load_start_dir(self) -> str:
        """Return the last successfully used Timeline Load folder."""
        try:
            path = self._file_dialog_history_path
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    folders = data.get("folders")
                    if isinstance(folders, dict):
                        candidate = folders.get("minimax_timeline_load")
                        if candidate and Path(candidate).is_dir():
                            return str(Path(candidate))
        except Exception:
            pass
        # Existing project folder is a useful fallback for older installs that
        # do not have dialog history yet. Otherwise use the Timeline output root.
        try:
            if self._project_path and self._project_path.parent.is_dir():
                return str(self._project_path.parent)
        except Exception:
            pass
        return str(self._timeline_root)

    def _remember_timeline_load_folder(self, selected: str | Path) -> None:
        """Persist the directory of a successfully selected Timeline JSON."""
        if not selected:
            return
        try:
            q = Path(str(selected)).expanduser()
            folder = q if q.is_dir() else q.parent
            if not folder.is_dir():
                return
            path = self._file_dialog_history_path
            data = {}
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            if not isinstance(data, dict):
                data = {}
            folders = data.get("folders")
            if not isinstance(folders, dict):
                folders = {}
            folders["minimax_timeline_load"] = str(folder)
            data["folders"] = folders
            # Keep compatibility with the rest of GrizzlyMax's file dialogs.
            data["last_folder"] = str(folder)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            # File-dialog history should never stop a project from loading.
            pass

    def load_project(self):
        name, _ = QFileDialog.getOpenFileName(
            self,
            "Load MiniMax timeline",
            self._timeline_load_start_dir(),
            "MiniMax Timeline (*.json);;JSON (*.json)",
        )
        if not name:
            return
        try:
            data = json.loads(Path(name).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
                raise ValueError("File does not contain a MiniMax timeline project.")
            self.project = data
            loaded_schema_version = int(self.project.get("schema_version") or 1)
            self.project["schema_version"] = TIMELINE_SCHEMA_VERSION
            self.project.setdefault("project_id", uuid.uuid4().hex)
            self.project.setdefault("name", Path(name).stem)
            self.project.setdefault("project_folder", str(Path(name).parent))
            self.project.setdefault("assembled_output", "")
            self.project.setdefault("assembly_status", "")
            self.project["auto_assemble"] = False
            self.project["auto_assemble_pending"] = False
            self.project.setdefault("global_generation_settings", {})
            self.project.setdefault("hq_restart_override", {})
            self.project.setdefault("soundtrack_clips", [])
            self.project.setdefault("soundtrack_mode", "mix")
            if str(self.project.get("soundtrack_mode") or "mix") not in {"mix", "clips_only", "soundtrack_only"}:
                self.project["soundtrack_mode"] = "mix"
            for audio in self.project.get("soundtrack_clips") or []:
                if not isinstance(audio, dict):
                    continue
                audio.setdefault("id", uuid.uuid4().hex)
                audio.setdefault("path", "")
                audio.setdefault("name", Path(str(audio.get("path") or "Audio")).stem)
                audio.setdefault("source_duration", 0.0)
                audio.setdefault("trim_in", 0.0)
                audio.setdefault("trim_out", audio.get("source_duration"))
                audio.setdefault("timeline_start", 0.0)
                audio.setdefault("timeline_end", None)
                audio.setdefault("volume_percent", 100)
                audio.setdefault("fade_in_seconds", 0.0)
                audio.setdefault("fade_out_seconds", 0.0)
            for clip in self.project.get("clips") or []:
                legacy_match = bool(clip.get("match_next_first_frame", False))
                if not clip.get("edit_mode"):
                    if legacy_match and str(clip.get("generation_mode") or "new") == "continue":
                        clip["edit_mode"] = "bridge_both"
                    elif legacy_match:
                        clip["edit_mode"] = "anchor_next"
                    elif str(clip.get("generation_mode") or "new") == "continue":
                        clip["edit_mode"] = "continue_previous"
                    else:
                        clip["edit_mode"] = "standalone"
                clip["match_next_first_frame"] = False
            for clip in self._clips():
                clip.setdefault("id", uuid.uuid4().hex)
                clip.setdefault("name", "")
                clip.setdefault("generation_mode", "new")
                clip.setdefault("start_source_video", "")
                clip.setdefault("source_duration_seconds", 0.0)
                clip.setdefault("frames", 243)
                clip.setdefault("use_reference_images", False)
                clip.setdefault("reference_images", [])
                clip.setdefault("settings", {})
                # Legacy Timeline projects may contain a captured standalone Glue
                # flag. Generated Timeline outputs were already forced to remain
                # separate, so clearing this metadata is safe and must not mark the
                # finished clip stale or require regeneration.
                clip["settings"]["glue_results"] = False
                clip.setdefault("status", "draft")
                # Stale is informational only. Preserve it across save/load so an
                # already-rendered clip whose editor state changed can remain visibly
                # orange, but never use this flag to block media operations.
                clip.setdefault("stale", False)
                clip.setdefault("queue_job_id", None)
                clip.setdefault("output", "")
                clip.setdefault("hq_generated", False)
                clip.setdefault("locked", False)
                clip.setdefault("notes", "")
                clip.setdefault("last_generation_info", {})
                clip.setdefault("alternate_candidates", [])
                clip["speed_multiplier"] = _clip_speed(clip)
                if not clip.get("segments"):
                    clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0}]
                for seg in clip["segments"]:
                    seg.setdefault("id", uuid.uuid4().hex); seg.setdefault("prompt", ""); seg.setdefault("weight", 1.0)
                if len(clip["segments"]) > 1:
                    merged_prompt = _compiled_prompt(clip)
                    clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": merged_prompt, "weight": 1.0}]

            # Schema v1 could accidentally persist every rendered continuation clip
            # as stale after deleting an unrelated final clip. There was no stale
            # reason/fingerprint in that schema, so those saved flags cannot be
            # distinguished from a real edit after reload. Recover the specific
            # legacy "mass stale continuation chain" signature once:
            #   - Clip 1 is still valid
            #   - every rendered continuation clip from Clip 2 onward is stale
            #   - every one of those clips still has its rendered output on disk
            # This is the exact state produced by the old delete-last-clip bug.
            if loaded_schema_version < 2:
                clips = self._clips()
                rendered_continuations = [
                    c for c in clips[1:]
                    if str(c.get("generation_mode") or "") == "continue"
                    and str(c.get("status") or "") == "finished"
                    and str(c.get("output") or "").strip()
                ]
                legacy_mass_stale = bool(rendered_continuations)
                if clips and bool(clips[0].get("stale")):
                    legacy_mass_stale = False
                if legacy_mass_stale:
                    for c in rendered_continuations:
                        out_path = Path(str(c.get("output") or "")).expanduser()
                        if not bool(c.get("stale")) or not out_path.is_file():
                            legacy_mass_stale = False
                            break
                if legacy_mass_stale:
                    for c in rendered_continuations:
                        c["stale"] = False
                    self.project["assembly_status"] = ""
                    print(
                        f"[TIMELINE] Recovered {len(rendered_continuations)} legacy stale "
                        "continuation clip(s) from the old delete-last-clip bug.",
                        flush=True,
                    )

            self._project_path = Path(name)
            self._remember_timeline_load_folder(self._project_path)
            self.selected_clip_id = self._clips()[0]["id"] if self._clips() else None
            self.selected_clip_ids = {str(self.selected_clip_id)} if self.selected_clip_id else set()
            self._selection_anchor_id = str(self.selected_clip_id) if self.selected_clip_id else None
            self.selected_cut_index = 0
            self._reset_history()
            self._refresh_all(select_first=True)
        except Exception as exc:
            QMessageBox.critical(self, "Load timeline failed", str(exc))

    # ------------------------------------------------------------- clip actions
    def add_clip(self):
        if not self._ensure_project_setup_for_first_edit():
            return
        self._record_undo_state("Add clip")
        clips = self._clips()
        idx = self._selected_index()
        insert_at = len(clips) if idx < 0 else idx + 1
        base_settings = clips[idx].get("settings") if idx >= 0 else None
        clip = self._blank_clip(continue_previous=bool(clips), source_settings=base_settings)
        clip["name"] = f"Clip {insert_at + 1}"
        # A newly continued clip starts with an empty prompt instead of silently copying stale action.
        if clip["generation_mode"] == "continue":
            clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0}]
        clips.insert(insert_at, clip)
        self._invalidate_assembly()
        self.selected_clip_id = clip["id"]
        self.selected_clip_ids = {str(clip["id"])}
        self._selection_anchor_id = str(clip["id"])
        self.selected_cut_index = 0
        self._renumber_default_names()
        self._refresh_all()

    def duplicate_clip(self):
        if not self._ensure_project_setup_for_first_edit():
            return
        idx = self._selected_index()
        if idx < 0: return
        if str(self._clips()[idx].get("generation_mode") or "") in {"source", "loaded"}:
            QMessageBox.information(self, "Timeline", "Loaded video clips cannot be duplicated as generation blocks. Use Load clip… to add another ready-made video.")
            return
        self._record_undo_state("Duplicate clip")
        clone = copy.deepcopy(self._clips()[idx])
        clone["id"] = uuid.uuid4().hex
        clone["name"] = str(clone.get("name") or f"Clip {idx + 1}") + " copy"
        clone["queue_job_id"] = None; clone["output"] = ""; clone["status"] = "draft"; clone["stale"] = False
        clone.pop("trim_in", None); clone.pop("trim_out", None)
        clone["locked"] = False
        clone["last_generation_info"] = {}
        clone["alternate_candidates"] = []
        for seg in clone.get("segments") or []: seg["id"] = uuid.uuid4().hex
        self._clips().insert(idx + 1, clone)
        self._invalidate_assembly()
        self.selected_clip_id = clone["id"]
        self.selected_clip_ids = {str(clone["id"])}
        self._selection_anchor_id = str(clone["id"])
        self._refresh_all()

    def delete_clip(self):
        indices = self._selected_indices()
        if not indices:
            return
        unlocked = [i for i in indices if not bool(self._clips()[i].get("locked", False))]
        if not unlocked:
            QMessageBox.information(self, "Delete clip", "The selected block(s) are locked. Unlock them before deleting.")
            return
        named = [str(self._clips()[i].get("name") or f"Clip {i + 1}") for i in unlocked]
        has_content = any(
            any(str(seg.get("prompt") or "").strip() for seg in self._clips()[i].get("segments") or [])
            for i in unlocked
        )
        locked_skipped = len(indices) - len(unlocked)
        if has_content or locked_skipped:
            if len(unlocked) > 1:
                msg = f"Delete {len(unlocked)} selected unlocked blocks?"
            else:
                msg = f"Delete {named[0]}?"
            if locked_skipped:
                msg += f"\n\n{locked_skipped} locked selected block(s) will be kept."
            if QMessageBox.question(self, "Delete clip", msg) != QMessageBox.StandardButton.Yes:
                return
        self._record_undo_state("Delete clips" if len(unlocked) > 1 else "Delete clip")
        first = min(unlocked)
        removed_ids = {str(self._clips()[i].get("id")) for i in unlocked}
        for i in sorted(unlocked, reverse=True):
            self._clips().pop(i)
        self.selected_clip_ids -= removed_ids
        self._invalidate_assembly()
        self._renumber_default_names()
        if self._clips():
            fallback = self._clips()[min(first, len(self._clips()) - 1)]
            self.selected_clip_id = str(fallback.get("id"))
            self.selected_clip_ids = {self.selected_clip_id}
            self._selection_anchor_id = self.selected_clip_id
        else:
            self.selected_clip_id = None
            self.selected_clip_ids = set()
            self._selection_anchor_id = None
        self.selected_cut_index = 0
        stale_from = first if first < len(self._clips()) else None
        self._repair_continuation_chain(mark_stale=True, stale_from=stale_from)
        self._refresh_all()

    def move_clip(self, delta):
        idx = self._selected_index()
        target = idx + int(delta)
        if idx < 0 or bool((self._selected_clip() or {}).get("locked", False)) or not (0 <= target < len(self._clips())): return
        self._reorder_clips(idx, target)

    def _reorder_clips(self, old, new):
        clips = self._clips()
        if not (0 <= old < len(clips) and 0 <= new < len(clips)): return
        if bool(clips[old].get("locked", False)):
            return
        lo, hi = sorted((old, new))
        if any(bool(clips[i].get("locked", False)) for i in range(lo, hi + 1) if i != old):
            return
        # A user-loaded start video is the root of the chain and must remain first.
        if clips and str(clips[0].get("generation_mode") or "") == "source" and (old == 0 or new == 0):
            return
        self._record_undo_state("Reorder clips")
        item = clips.pop(old); clips.insert(new, item)
        self._invalidate_assembly()
        # Reordering only changes continuation ancestry starting at the earliest
        # position involved in the move. Earlier rendered clips stay valid.
        self._repair_continuation_chain(mark_stale=True, stale_from=min(old, new))
        self._refresh_all()

    def _repair_continuation_chain(self, mark_stale=False, stale_from=None):
        """Repair impossible continuation topology without invalidating unrelated clips.

        ``stale_from`` is the first index whose incoming continuation boundary may
        have changed. Previously this routine marked every rendered continuation
        clip stale, which meant deleting Clip 10 incorrectly invalidated Clips 2-9.
        """
        for i, clip in enumerate(self._clips()):
            if i == 0 and clip.get("generation_mode") == "continue":
                clip["generation_mode"] = "new"
            # Do not clear informational stale flags here. Reordering/deleting a
            # different block must not silently erase the user's warning state.
            clip.setdefault("stale", False)

    def _ordered_selected_ids(self):
        return [str(c.get("id")) for c in self._clips() if str(c.get("id")) in self.selected_clip_ids]

    def _selected_indices(self):
        return [i for i, c in enumerate(self._clips()) if str(c.get("id")) in self.selected_clip_ids]

    def _apply_selection(self, ids, primary=None, *, anchor=None):
        valid = {str(c.get("id")) for c in self._clips()}
        self.selected_clip_ids = {str(x) for x in ids if str(x) in valid}
        if primary is not None and str(primary) in self.selected_clip_ids:
            self.selected_clip_id = str(primary)
        elif self.selected_clip_ids:
            self.selected_clip_id = next((str(c.get("id")) for c in self._clips() if str(c.get("id")) in self.selected_clip_ids), None)
        else:
            self.selected_clip_id = None
        if anchor is not None:
            self._selection_anchor_id = str(anchor) if str(anchor) in valid else None
        elif not self.selected_clip_ids:
            self._selection_anchor_id = None
        self.selected_cut_index = 0
        self._refresh_all()

    def select_clip(self, clip_id):
        # Legacy/plain selection path.
        clip_id = str(clip_id)
        self._apply_selection({clip_id}, clip_id, anchor=clip_id)

    def _selection_requested(self, clip_id, modifiers_value=0):
        clip_id = str(clip_id)
        ctrl = bool(int(modifiers_value) & int(Qt.KeyboardModifier.ControlModifier.value))
        shift = bool(int(modifiers_value) & int(Qt.KeyboardModifier.ShiftModifier.value))
        clips = self._clips()
        ids = [str(c.get("id")) for c in clips]
        if clip_id not in ids:
            return

        if shift:
            anchor = self._selection_anchor_id or self.selected_clip_id or clip_id
            try:
                a = ids.index(str(anchor)); b = ids.index(clip_id)
            except ValueError:
                a = b = ids.index(clip_id)
            lo, hi = sorted((a, b))
            range_ids = set(ids[lo:hi + 1])
            if ctrl:
                range_ids |= set(self.selected_clip_ids)
            self._apply_selection(range_ids, clip_id, anchor=anchor)
            return

        if ctrl:
            new_ids = set(self.selected_clip_ids)
            if clip_id in new_ids:
                new_ids.remove(clip_id)
                primary = self.selected_clip_id if self.selected_clip_id in new_ids else None
            else:
                new_ids.add(clip_id)
                primary = clip_id
            self._apply_selection(new_ids, primary, anchor=clip_id if clip_id in new_ids else self._selection_anchor_id)
            return

        self._apply_selection({clip_id}, clip_id, anchor=clip_id)

    def select_all_clips(self):
        ids = [str(c.get("id")) for c in self._clips()]
        self._apply_selection(ids, self.selected_clip_id or (ids[0] if ids else None), anchor=self.selected_clip_id or (ids[0] if ids else None))

    def select_from_here(self):
        idx = self._selected_index()
        if idx < 0:
            return
        ids = [str(c.get("id")) for c in self._clips()[idx:]]
        self._apply_selection(ids, self.selected_clip_id, anchor=self.selected_clip_id)

    def select_to_here(self):
        idx = self._selected_index()
        if idx < 0:
            return
        ids = [str(c.get("id")) for c in self._clips()[:idx + 1]]
        self._apply_selection(ids, self.selected_clip_id, anchor=self.selected_clip_id)

    def select_all_missing(self):
        ids = [str(c.get("id")) for c in self._clips() if not self._clip_has_usable_output(c)]
        self._apply_selection(ids, ids[0] if ids else None, anchor=ids[0] if ids else None)

    def select_all_stale(self):
        ids = [str(c.get("id")) for c in self._clips() if bool(c.get("stale", False))]
        self._apply_selection(ids, ids[0] if ids else None, anchor=ids[0] if ids else None)

    def clear_selection(self):
        self._apply_selection(set(), None, anchor=None)

    def toggle_lock_selected(self):
        selected = [c for c in self._clips() if str(c.get("id")) in self.selected_clip_ids]
        if not selected:
            return False
        unlock = all(bool(c.get("locked", False)) for c in selected)
        self._record_undo_state("Unlock clips" if unlock else "Lock clips")
        for clip in selected:
            clip["locked"] = not unlock
        self._refresh_all()
        return True

    def _selected_clip_locked(self):
        clip = self._selected_clip()
        return bool((clip or {}).get("locked", False))

    # ---------------------------------------------------------- inspector slots
    def _project_name_changed(self, text):
        self.project["name"] = text

    def _clip_name_changed(self, text):
        if self._loading_inspector: return
        clip = self._selected_clip()
        if not clip or bool(clip.get("locked", False)): return
        self._record_undo_state("Rename clip", coalesce_key=f"clip-name:{clip.get('id')}")
        clip["name"] = text
        self.canvas.update()

    def _populate_generation_modes(self, index: int, current_mode: str = "new"):
        self.gen_mode.blockSignals(True)
        try:
            clip = self._selected_clip()
            using_refs = bool((clip or {}).get("use_reference_images", False))
            self.gen_mode.clear()
            if current_mode == "loaded":
                self.gen_mode.addItem("Loaded video clip", "loaded")
            else:
                self.gen_mode.addItem("New generation / new chain", "new")
                if index == 0:
                    self.gen_mode.addItem("Load a start clip", "source")
                elif not using_refs:
                    self.gen_mode.addItem("Continue previous clip", "continue")
            target = self.gen_mode.findData(current_mode)
            if target < 0:
                target = 0
            self.gen_mode.setCurrentIndex(target)
            self.gen_mode.setToolTip(
                "Continue previous clip is disabled while this block uses Ref2VA reference images."
                if using_refs and index > 0 else ""
            )
        finally:
            self.gen_mode.blockSignals(False)

    # --------------------------------------------------------- Ref2VA presets
    def _open_ref_presets(self):
        target_indices = [
            i for i, clip in enumerate(self._clips())
            if str(clip.get("id")) in self.selected_clip_ids
            and not bool(clip.get("locked", False))
            and str(clip.get("generation_mode") or "") not in {"source", "loaded"}
        ]
        if not target_indices:
            QMessageBox.information(
                self, "Reference presets",
                "Select at least one unlocked generation clip before opening Ref presets."
            )
            return

        dialog = RefPresetDialog(self._ref_presets_path, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        refs = dialog.selected_references()
        if not refs:
            return

        continuation_indices = []
        for i in target_indices:
            clip = self._clips()[i]
            mode = str(clip.get("generation_mode") or "new")
            edit_mode = str(clip.get("edit_mode") or "")
            if mode == "continue" or edit_mode in {"continue_previous", "bridge_both", "anchor_next"}:
                continuation_indices.append(i)

        if continuation_indices:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Ref2VA replaces clip-to-clip frame references")
            count = len(continuation_indices)
            box.setText(
                f"{count} selected clip{'s were' if count != 1 else ' was'} set up to use a previous/next clip frame.\n\n"
                "MiniMax Ref2VA reference-image mode does not use the first or last frame of other clips. "
                "Applying these presets will change those clips to standalone Ref2VA generation while keeping their prompts, duration, and other settings."
            )
            continue_btn = box.addButton("Continue anyway", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is not continue_btn:
                return

        self._record_undo_state("Apply reference presets")
        self._invalidate_assembly("Reference presets changed — assemble again after regenerating affected clips.")
        defs_text = _reference_prompt_definitions_text(refs)
        for i in target_indices:
            clip = self._clips()[i]
            clip["reference_images"] = copy.deepcopy(refs)
            clip["use_reference_images"] = True
            existing_prompt = _compiled_prompt(clip)
            prompt_body = _strip_reference_prompt_prefix(existing_prompt)
            merged_prompt = defs_text if defs_text else ""
            if prompt_body:
                merged_prompt = ((merged_prompt + "\n\n\n" + prompt_body) if merged_prompt else prompt_body).strip()
            segments = clip.setdefault("segments", [])
            if not segments:
                segments.append({"id": uuid.uuid4().hex, "prompt": merged_prompt, "weight": 1.0})
            elif len(segments) > 1:
                clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": merged_prompt, "weight": 1.0}]
            else:
                segments[0]["prompt"] = merged_prompt
                segments[0]["weight"] = 1.0
            # Ref2VA cannot simultaneously consume continuation/bridge first/last
            # frames, so applying a preset set makes the selected clip standalone.
            if str(clip.get("generation_mode") or "new") == "continue" or str(clip.get("edit_mode") or "") in {
                "continue_previous", "bridge_both", "anchor_next"
            }:
                clip["generation_mode"] = "new"
                clip["edit_mode"] = "standalone"
                clip["match_next_first_frame"] = False
            self._touch_clip(i, propagate=False)
        self._refresh_all()

    # --------------------------------------------------------- Ref2VA references
    def _clear_reference_rows(self):
        layout = getattr(self, "refs_rows_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_reference_rows(self, clip=None):
        self._clear_reference_rows()
        clip = clip or self._selected_clip()
        if not clip:
            return
        refs = _reference_entries(clip)
        for idx, ref in enumerate(refs, 1):
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)

            thumb = QLabel()
            thumb.setFixedSize(58, 58)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setFrameShape(QFrame.Shape.StyledPanel)
            pix = QPixmap(ref["path"])
            if not pix.isNull():
                thumb.setPixmap(pix.scaled(54, 54, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            else:
                thumb.setText("Image")
            desc_tip = "\n".join(x for x in (str(ref.get("picture_description") or "").strip(), str(ref.get("subject_description") or "").strip()) if x)
            thumb.setToolTip(ref["path"] + (("\n\n" + desc_tip) if desc_tip else ""))

            token = QLabel(f"<Subject {idx}>\n<Picture {idx}>")
            token.setToolTip(
                f"Use <Subject {idx}> in the prompt for reusable visible content. "
                f"MiniMax receives this file as <Picture {idx}>."
            )
            token.setMinimumWidth(105)

            name_edit = QLineEdit(ref.get("name") or Path(ref["path"]).stem)
            name_edit.setPlaceholderText(f"Reference {idx} name")
            name_edit.setToolTip(
                "Friendly reference name. Preset-specific <Picture N> and <Subject N> descriptions are preserved and renumbered automatically."
            )
            name_edit.editingFinished.connect(lambda i=idx-1, w=name_edit: self._reference_name_changed(i, w.text()))

            insert_btn = QPushButton(f"Insert <Subject {idx}>")
            insert_btn.setToolTip(f"Insert a ready-to-use Ref2VA prompt snippet for <Subject {idx}> at the prompt cursor.")
            insert_btn.clicked.connect(lambda _=False, n=idx: self._insert_reference_token(n))
            remove_btn = QPushButton("Remove")
            remove_btn.clicked.connect(lambda _=False, i=idx-1: self._remove_reference_image(i))

            hl.addWidget(thumb)
            hl.addWidget(token)
            hl.addWidget(name_edit, 1)
            hl.addWidget(insert_btn)
            hl.addWidget(remove_btn)
            self.refs_rows_layout.addWidget(row)

        if not refs:
            empty = QLabel("No reference images loaded.")
            self.refs_rows_layout.addWidget(empty)
        self.add_refs_btn.setEnabled(bool(clip) and len(refs) < TIMELINE_MAX_REF_IMAGES and str(clip.get("generation_mode") or "") not in {"source", "loaded"})

    def _reference_mode_changed(self, enabled):
        if self._loading_inspector:
            return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or bool(clip.get("locked", False)) or str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            return
        enabled = bool(enabled)
        self._record_undo_state("Toggle reference images")
        clip["use_reference_images"] = enabled
        self.use_refs_warning.setVisible(enabled)
        self.use_refs_guide.setVisible(enabled)
        if enabled and str(clip.get("generation_mode") or "new") == "continue":
            clip["generation_mode"] = "new"
            clip["edit_mode"] = "standalone"
        self._touch_clip(idx)
        self._populate_generation_modes(idx, str(clip.get("generation_mode") or "new"))
        self._refresh_edit_workflow(clip)
        self._refresh_settings_summary(clip)
        self.canvas.update()

    def _add_reference_images(self):
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or bool(clip.get("locked", False)) or str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            return
        refs = _reference_entries(clip)
        remaining = TIMELINE_MAX_REF_IMAGES - len(refs)
        if remaining <= 0:
            QMessageBox.information(self, "Reference images", f"This timeline clip already has the maximum of {TIMELINE_MAX_REF_IMAGES} reference images.")
            return
        names, _ = QFileDialog.getOpenFileNames(
            self, "Add reference images", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)"
        )
        if not names:
            return
        self._record_undo_state("Add reference images")
        added = 0
        existing = {str(Path(r["path"]).resolve()) for r in refs if Path(r["path"]).exists()}
        for name in names:
            if added >= remaining:
                break
            path = Path(name)
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in existing:
                continue
            refs.append({"path": str(path), "name": path.stem})
            existing.add(key)
            added += 1
        clip["reference_images"] = refs
        if refs:
            clip["use_reference_images"] = True
            if str(clip.get("generation_mode") or "new") == "continue":
                clip["generation_mode"] = "new"
                clip["edit_mode"] = "standalone"
        self._touch_clip(idx)
        self._refresh_all()

    def _remove_reference_image(self, ref_index: int):
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or bool(clip.get("locked", False)):
            return
        refs = _reference_entries(clip)
        if not (0 <= ref_index < len(refs)):
            return
        old_count = len(refs)
        self._record_undo_state("Remove reference image")
        refs.pop(ref_index)
        clip["reference_images"] = refs
        # Ref2VA Picture numbering is positional. Renumber Subject tokens in the
        # authored prompt so deleting a middle image cannot silently point later
        # subjects at the wrong picture.
        prompt = _compiled_prompt(clip)
        if prompt and ref_index < old_count - 1:
            placeholders = {}
            for old_n in range(ref_index + 2, old_count + 1):
                placeholder = f"__MMH3_SUBJECT_RENUMBER_{old_n}__"
                prompt = prompt.replace(f"<Subject {old_n}>", placeholder)
                placeholders[placeholder] = f"<Subject {old_n - 1}>"
            for placeholder, token in placeholders.items():
                prompt = prompt.replace(placeholder, token)
            segments = clip.setdefault("segments", [{"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0}])
            if not segments:
                segments.append({"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0})
            clip["segments"] = [{"id": segments[0].get("id") or uuid.uuid4().hex, "prompt": prompt, "weight": 1.0}]
        if not refs:
            clip["use_reference_images"] = False
        self._touch_clip(idx)
        self._refresh_all()

    def _reference_name_changed(self, ref_index: int, text: str):
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or bool(clip.get("locked", False)):
            return
        refs = _reference_entries(clip)
        if not (0 <= ref_index < len(refs)):
            return
        value = str(text or "").strip() or Path(refs[ref_index]["path"]).stem
        if value == str(refs[ref_index].get("name") or ""):
            return
        self._record_undo_state("Rename reference image")
        refs[ref_index]["name"] = value
        clip["reference_images"] = refs
        self._touch_clip(idx, propagate=False)
        self._refresh_settings_summary(clip)
        self.canvas.update()

    def _insert_reference_token(self, number: int):
        if not self.cut_prompt.isEnabled():
            return
        clip = self._selected_clip() or {}
        refs = _reference_entries(clip)
        friendly = ""
        if 1 <= int(number) <= len(refs):
            friendly = str(refs[int(number) - 1].get("name") or "").strip()
        snippet = f"Feature <Subject {int(number)}> prominently in the scene."
        if friendly:
            snippet = f"Feature <Subject {int(number)}> prominently in the scene as {friendly}."
        cursor = self.cut_prompt.textCursor()
        current_text = self.cut_prompt.toPlainText()
        if cursor.position() > 0 and current_text and not current_text.endswith((" ", "\n")):
            cursor.insertText(" ")
        cursor.insertText(snippet + " ")
        self.cut_prompt.setTextCursor(cursor)
        self.cut_prompt.setFocus()

    def _probe_start_video_duration(self, path: str) -> float:
        try:
            from runtime.ffmpeg_tools import tool_path as ffmpeg_tool_path
            exe = str(ffmpeg_tool_path("ffprobe.exe"))
            cp = subprocess.run(
                [exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if cp.returncode == 0:
                value = float((cp.stdout or "0").strip() or 0)
                return value if value > 0 else 0.0
        except Exception:
            pass
        return 0.0

    def _choose_start_video(self, clip: dict) -> bool:
        name, _ = QFileDialog.getOpenFileName(
            self, "Load start video", "",
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All files (*.*)"
        )
        if not name:
            return False
        path = Path(name)
        if not path.is_file():
            return False
        clip["generation_mode"] = "source"
        clip["start_source_video"] = str(path)
        clip["source_duration_seconds"] = self._probe_start_video_duration(str(path))
        clip["output"] = str(path)
        clip.pop("trim_in", None); clip.pop("trim_out", None)
        clip["queue_job_id"] = None
        clip["status"] = "finished"
        clip["stale"] = False
        clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0}]
        self._invalidate_assembly("Start video changed — assemble again after generation.")
        return True

    def _mode_changed(self):
        if self._loading_inspector:
            return
        idx = self._selected_index(); clip = self._selected_clip()
        if not clip or bool(clip.get("locked", False)):
            return
        mode = str(self.gen_mode.currentData() or "new")
        if str(clip.get("generation_mode") or "") == "loaded":
            return
        if idx == 0 and mode == "source":
            previous_mode = str(clip.get("generation_mode") or "new")
            self._record_undo_state("Load start clip")
            if not self._choose_start_video(clip):
                self._loading_inspector = True
                try:
                    self._populate_generation_modes(idx, previous_mode)
                finally:
                    self._loading_inspector = False
                return
            self._refresh_all()
            return
        # Switching back from a loaded start clip turns block 1 into a normal draft generation.
        if idx == 0 and str(clip.get("generation_mode") or "") == "source" and mode == "new":
            self._record_undo_state("Change generation mode")
            clip["generation_mode"] = "new"
            clip["start_source_video"] = ""
            clip["source_duration_seconds"] = 0.0
            clip["output"] = ""
            clip["queue_job_id"] = None
            clip["status"] = "draft"
            clip["stale"] = False
            self._touch_clip(idx)
            self._refresh_all()
            return
        if mode != str(clip.get("generation_mode") or "new"):
            self._record_undo_state("Change generation mode")
        clip["generation_mode"] = mode
        self._touch_clip(idx)
        self._refresh_all()

    def _frames_changed(self):
        if self._loading_inspector: return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or bool(clip.get("locked", False)): return
        frames = int(self.frames_combo.currentData() or clip.get("frames") or 243)
        if frames != int(clip.get("frames") or 0):
            self._record_undo_state("Change clip duration")
            clip["frames"] = frames
            clip.setdefault("settings", {})["frames"] = frames
            clip["settings"]["experimental_long_duration"] = frames > 719
            self._touch_clip(idx)
            self._refresh_all()

    def _canvas_frames_changed(self, clip_id, frames):
        for i, clip in enumerate(self._clips()):
            if str(clip.get("id")) == str(clip_id):
                if bool(clip.get("locked", False)):
                    break
                if int(frames) == int(clip.get("frames") or 0):
                    break
                self._record_undo_state("Resize clip")
                clip["frames"] = int(frames)
                clip.setdefault("settings", {})["frames"] = int(frames)
                clip["settings"]["experimental_long_duration"] = int(frames) > 719
                self._touch_clip(i)
                break
        self._refresh_all()

    def _settings_changed(self, *args):
        if self._loading_inspector: return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or bool(clip.get("locked", False)): return
        self._record_undo_state("Change clip settings", coalesce_key=f"settings:{clip.get('id')}")
        s = clip.setdefault("settings", {})
        s["seed"] = self.seed_spin.value(); s["steps"] = self.steps_spin.value(); s["scheduler"] = self.scheduler_combo.currentText()
        s["glue_results"] = False
        s["continue_audio_memory"] = self.audio_memory_check.isChecked()
        self._touch_clip(idx)
        self._refresh_all()

    def _refresh_edit_workflow(self, clip):
        buttons = (self.edit_bridge_both, self.edit_continue_previous, self.edit_anchor_next, self.edit_standalone)
        idx = self._selected_index()
        if clip is None or idx < 0:
            self.edit_selected_label.setText("Select a block to choose how it should be regenerated.")
            for b in buttons:
                b.setEnabled(False)
            self.generate_selected_btn.setEnabled(False)
            return

        if bool(clip.get("locked", False)):
            self.edit_selected_label.setText("Locked block — unlock it before changing regeneration mode or recreating it.")
            for b in buttons:
                b.setEnabled(False)
            self.generate_selected_btn.setEnabled(False)
            return

        if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            self.edit_selected_label.setText("Loaded start clip — this block is the source for the continuation chain and is not regenerated by H3.")
            for b in buttons:
                b.setEnabled(False)
            self.generate_selected_btn.setEnabled(False)
            return

        has_prev = idx > 0
        has_next = idx < len(self._clips()) - 1
        using_refs = bool(clip.get("use_reference_images", False))
        self.edit_bridge_both.setEnabled(has_prev and has_next and not using_refs)
        self.edit_continue_previous.setEnabled(has_prev and not using_refs)
        self.edit_anchor_next.setEnabled(has_next)
        self.edit_standalone.setEnabled(True)
        self.generate_selected_btn.setEnabled(True)

        mode = str(clip.get("edit_mode") or ("continue_previous" if clip.get("generation_mode") == "continue" else "standalone"))
        valid = {
            "bridge_both": self.edit_bridge_both,
            "continue_previous": self.edit_continue_previous,
            "anchor_next": self.edit_anchor_next,
            "standalone": self.edit_standalone,
        }
        button = valid.get(mode, self.edit_standalone)
        if not button.isEnabled():
            if using_refs:
                mode = "standalone"
            else:
                mode = "anchor_next" if has_next and not has_prev else ("continue_previous" if has_prev and not has_next else "standalone")
            clip["edit_mode"] = mode
            button = valid[mode]
        button.setChecked(True)

        prev_state = "available" if has_prev else "none"
        next_state = "available" if has_next else "none"
        if has_prev:
            prev = self._clips()[idx - 1]
            prev_state = "ready" if self._clip_preview_path(prev) else "not rendered"
        if has_next:
            nxt = self._clips()[idx + 1]
            next_state = "ready" if self._clip_preview_path(nxt) else "not rendered"
        ref_note = "  •  Ref2VA: previous continuation disabled" if using_refs else ""
        self.edit_selected_label.setText(
            f"{clip.get('name') or f'Clip {idx + 1}'}  •  previous: {prev_state}  •  next: {next_state}{ref_note}"
        )

    def _edit_mode_changed(self, button):
        if self._loading_inspector or button is None:
            return
        clip = self._selected_clip()
        if not clip or bool(clip.get("locked", False)):
            return
        mode = str(button.property("edit_mode") or "standalone")
        if mode == str(clip.get("edit_mode") or ""):
            return
        self._record_undo_state("Change regeneration mode")
        clip["edit_mode"] = mode
        # This is an edit strategy, not a content change. Choosing it must not
        # stale the clip or mutate the normal first-time creation workflow.
        self.canvas.update()

    @staticmethod
    def _generation_resolution_text(settings: dict) -> str:
        """Resolve the actual MiniMax output dimensions from a captured settings dict."""
        if not isinstance(settings, dict) or not settings:
            return "—"
        aspect = str(settings.get("aspect") or "16:9").strip()
        if aspect == "21:9":
            value = str(settings.get("widescreen_quality") or "").strip()
            match = re.search(r"(\d+)\s*[×xX]\s*(\d+)", value)
            if match:
                return f"{int(match.group(1))} × {int(match.group(2))}"
            return "—"

        value = str(settings.get("resolution") or "").strip()
        match = re.search(r"(\d+)\s*[×xX]\s*(\d+)", value)
        if not match:
            return "—"
        w, h = int(match.group(1)), int(match.group(2))
        # The Generation tab deliberately labels this preset 1280 × 720 for
        # familiarity, while MiniMax actually generates on its valid 32px grid.
        if (w, h) == (1280, 720):
            w, h = 1280, 704
        if aspect == "9:16":
            w, h = h, w
        elif aspect == "1:1":
            edge = h
            if (w, h) == (1280, 704):
                edge = 704
            w, h = edge, edge
        return f"{w} × {h}"

    def _refresh_generation_resolution_label(self):
        label = getattr(self, "generation_resolution_label", None)
        if label is None:
            return
        settings = self.project.get("global_generation_settings", {}) if isinstance(self.project, dict) else {}
        label.setText(f"Resolution: {self._generation_resolution_text(settings)}")

    def use_current_generation_settings(self):
        """Apply one Generation-tab snapshot globally without touching Timeline content/duration."""
        if not self._ensure_project_setup_for_first_edit():
            return False
        settings = self._timeline_global_settings()
        if not settings:
            QMessageBox.warning(self, "Timeline settings", "No Generation-tab settings were available to capture.")
            return False

        self._record_undo_state("Use current Generation settings")
        self.project["global_generation_settings"] = copy.deepcopy(settings)
        self._refresh_generation_resolution_label()
        changed_finished = False
        for idx, clip in enumerate(self._clips()):
            if bool(clip.get("locked", False)) or str(clip.get("generation_mode") or "") in {"source", "loaded"}:
                continue
            # Preserve all Timeline-owned values. Queue-time logic will add the
            # block's frames, prompt, refs and continuation source separately.
            clip["settings"] = copy.deepcopy(settings)
            clip["settings"]["glue_results"] = False
            if clip.get("queue_job_id") or clip.get("status") in {"pending", "running", "finished"}:
                changed_finished = True
            clip["stale"] = False

        self._invalidate_assembly(
            "Generation settings changed — regenerate affected clips before assembling."
            if changed_finished else "Generation settings updated."
        )
        self._refresh_all()
        return True

    # Compatibility for older signal/project code; this is now Timeline-global.
    def capture_current_settings(self):
        return self.use_current_generation_settings()

    def _cut_prompt_changed(self):
        if self._loading_inspector:
            return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or bool(clip.get("locked", False)):
            return
        self._record_undo_state("Edit prompt", coalesce_key=f"prompt:{clip.get('id')}")
        segments = clip.setdefault("segments", [])
        if not segments:
            segments.append({"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0})
        if len(segments) > 1:
            merged = _compiled_prompt(clip)
            clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": merged, "weight": 1.0}]
            segments = clip["segments"]
        segments[0]["prompt"] = self.cut_prompt.toPlainText()
        segments[0]["weight"] = 1.0
        self._touch_clip(idx)
        self.canvas.update()

    def _accept_preserved_downstream_chain(self, anchor_index: int) -> int:
        """Accept the already-rendered chain after an anchored replacement.

        bridge_both / anchor_next deliberately regenerate one clip *toward* the
        existing next clip. Once that replacement has actually finished, the
        existing next result and its unchanged continuation chain are valid
        again and must not remain stale just because an earlier delete/recreate
        operation marked them so.
        """
        clips = self._clips()
        if not (0 <= int(anchor_index) < len(clips) - 1):
            return 0
        anchor_clip = clips[int(anchor_index)]
        if str(anchor_clip.get("edit_mode") or "") not in {"bridge_both", "anchor_next"}:
            return 0
        if str(anchor_clip.get("status") or "") != "finished":
            return 0
        anchor_output = Path(str(anchor_clip.get("output") or ""))
        if not anchor_output.is_file():
            return 0

        cleared = 0
        for j in range(int(anchor_index) + 1, len(clips)):
            downstream = clips[j]
            if j > int(anchor_index) + 1 and str(downstream.get("generation_mode") or "") != "continue":
                break
            output = Path(str(downstream.get("output") or ""))
            if str(downstream.get("status") or "") != "finished" or not output.is_file():
                break
            if bool(downstream.get("stale")):
                downstream["stale"] = False
                cleared += 1
        return cleared

    def _recover_old_anchor_preserve_state(self) -> int:
        """Recover projects made before preserve-on-finish was recorded.

        Older builds could regenerate Clip N toward the already-existing
        Clip N+1 correctly, but leave Clip N+1.. stale forever.  A strong
        signature is that the anchored replacement's output is newer than the
        preserved next clip's output.  Recover that already-rendered chain once
        so the user does not have to regenerate it again just to assemble.
        """
        clips = self._clips()
        total = 0
        for i in range(len(clips) - 1):
            clip = clips[i]
            nxt = clips[i + 1]
            if str(clip.get("edit_mode") or "") not in {"bridge_both", "anchor_next"}:
                continue
            if str(clip.get("status") or "") != "finished" or str(nxt.get("status") or "") != "finished":
                continue
            if not bool(nxt.get("stale")):
                continue
            cur_out = Path(str(clip.get("output") or ""))
            next_out = Path(str(nxt.get("output") or ""))
            if not cur_out.is_file() or not next_out.is_file():
                continue
            try:
                # The preserved target existed first; the replacement was
                # rendered afterwards specifically to meet it.
                if cur_out.stat().st_mtime + 0.001 < next_out.stat().st_mtime:
                    continue
            except OSError:
                continue
            cleared = self._accept_preserved_downstream_chain(i)
            if cleared:
                total += cleared
                clip["_anchor_preserve_recovered"] = True
        if total:
            print(f"[TIMELINE] Recovered {total} preserved finished clip(s) from obsolete stale state.", flush=True)
        return total

    # --------------------------------------------------------------- results
    def _assembly_ready(self):
        """Assembly is intentionally independent from Timeline dependency state.

        "Stale" only means a generated clip may no longer match the current
        continuation plan. It must never prevent the user from concatenating
        files that already exist. Assembly therefore checks only whether every
        visible Timeline block has a real output file.
        """
        clips = self._clips()
        if not clips:
            return False, "Timeline has no clips."
        for i, clip in enumerate(clips):
            name = str(clip.get("name") or f"Clip {i + 1}")
            output = str(clip.get("output") or "")
            if not output or not Path(output).is_file():
                return False, f"{name} has no usable output file."
        return True, ""

    def assemble_timeline(self):
        # Assembly is a read-only operation on the Timeline project. Repeated
        # clicks while FFmpeg is being started/running are ignored so the action
        # cannot trigger duplicate requests or interact with edit-state logic.
        if getattr(self, "_timeline_assemble_click_busy", False):
            return False

        ready, reason = self._assembly_ready()
        if not ready:
            QMessageBox.information(self, "Timeline assembly", reason)
            return False

        stale_clips = [c for c in self._clips() if bool(c.get("stale", False))]
        if stale_clips:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Stale clips in timeline")
            count = len(stale_clips)
            box.setText(f"{count} clip{'s are' if count != 1 else ' is'} marked stale.")
            box.setInformativeText(
                "Stale only means the existing render may not match the current clip settings. "
                "You can still assemble it exactly as it exists on disk."
            )
            assemble_anyway = box.addButton("Assemble anyway", QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(assemble_anyway)
            box.exec()
            if box.clickedButton() is not assemble_anyway:
                return False

        if not callable(self.assemble_timeline_callback):
            QMessageBox.warning(self, "Timeline assembly", "Timeline assembly is not connected to the standalone GUI.")
            return False

        self._timeline_assemble_click_busy = True
        self.project["assembly_status"] = "Assembling…"
        self.project["assembled_output"] = ""
        self.assemble_timeline_btn.setEnabled(False)
        self.assemble_timeline_btn.setText("Assembling…")
        self._refresh_all()
        try:
            result = self.assemble_timeline_callback(copy.deepcopy(self.project))
        except Exception as exc:
            self._timeline_assemble_click_busy = False
            self.project["assembly_status"] = f"Assembly failed: {exc}"
            self.assemble_timeline_btn.setText("Assemble Video")
            self._refresh_all()
            QMessageBox.critical(self, "Timeline assembly failed", str(exc))
            return False

        if not result:
            self._timeline_assemble_click_busy = False
            if self.project.get("assembly_status") == "Assembling…":
                self.project["assembly_status"] = "Assembly did not start."
            self.assemble_timeline_btn.setText("Assemble Video")
            self._refresh_all()
        return bool(result)

    def mark_assembly_queued(self, output, job_id=None):
        self.project["assembled_output"] = str(output or "")
        self.project["assembly_status"] = "Queued for assembly"
        self.project["assembly_queue_job_id"] = str(job_id or "")
        self._timeline_assemble_click_busy = True
        if hasattr(self, "assemble_timeline_btn"):
            self.assemble_timeline_btn.setText("Assembly queued…")
        self._refresh_all()

    def mark_assembly_started(self, output):
        self.project["assembled_output"] = str(output or "")
        self.project["assembly_status"] = "Assembling…"
        self._timeline_assemble_click_busy = True
        if hasattr(self, "assemble_timeline_btn"):
            self.assemble_timeline_btn.setText("Assembling…")
        self._refresh_all()

    def mark_assembly_finished(self, output):
        self.project["assembled_output"] = str(output or "")
        self.project["assembly_status"] = "Finished"
        self.project["assembly_queue_job_id"] = ""
        self._timeline_assemble_click_busy = False
        if hasattr(self, "assemble_timeline_btn"):
            self.assemble_timeline_btn.setText("Assemble Video")
        self._refresh_all()

    def mark_assembly_failed(self, message):
        self.project["assembled_output"] = ""
        self.project["assembly_status"] = "Assembly failed: " + str(message or "Unknown error")
        self.project["assembly_queue_job_id"] = ""
        self._timeline_assemble_click_busy = False
        if hasattr(self, "assemble_timeline_btn"):
            self.assemble_timeline_btn.setText("Assemble Video")
        self._refresh_all()

    def _clip_by_id(self, clip_id):
        target = str(clip_id or "")
        for i, clip in enumerate(self._clips()):
            if str(clip.get("id") or "") == target:
                return i, clip
        return -1, None


    def _clip_preview_path(self, clip: dict) -> str:
        """Return an existing media path for preview/open-folder operations.

        Preview is deliberately independent of the Timeline stale flag.  A stale
        render is still valid media and remains inspectable until the file itself
        is removed.  Older projects/jobs can also have the authoritative path in
        last_generation_info rather than the mutable block output field, so check
        both before deciding that no preview exists.
        """
        if not isinstance(clip, dict):
            return ""

        candidates = []
        def add(value):
            value = str(value or "").strip().strip('\"')
            if value and value not in candidates:
                candidates.append(value)

        add(clip.get("output"))
        generated = clip.get("last_generation_info")
        if isinstance(generated, dict):
            add(generated.get("output"))
        if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            add(clip.get("start_source_video"))

        bases = []
        if self._project_path is not None:
            try:
                bases.append(Path(self._project_path).resolve().parent)
            except Exception:
                bases.append(Path(self._project_path).parent)
        bases.append(self._timeline_root)

        for raw in candidates:
            path = Path(raw).expanduser()
            try:
                if path.is_file():
                    return str(path)
            except OSError:
                pass
            if not path.is_absolute():
                for base in bases:
                    candidate = base / path
                    try:
                        if candidate.is_file():
                            return str(candidate)
                    except OSError:
                        pass

            # Timeline renders are versioned (..._001.mp4, ..._002.mp4, ...).
            # If an older saved version path disappeared, recover the newest
            # surviving sibling instead of treating the block as unpreviewable.
            try:
                parent = path.parent
                if parent.is_dir() and path.suffix:
                    stem = path.stem
                    base_stem = re.sub(r"_\d{3,}$", "", stem)
                    matches = sorted(
                        parent.glob(base_stem + "_*" + path.suffix),
                        key=lambda x: x.stat().st_mtime if x.is_file() else -1,
                        reverse=True,
                    )
                    for match in matches:
                        if match.is_file():
                            return str(match)
            except (OSError, ValueError):
                pass
        return ""

    def _probe_audio_duration(self, path: str) -> float:
        try:
            ffprobe = self.canvas._ffprobe_path()
            if not ffprobe:
                return 0.0
            cp = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if cp.returncode == 0:
                return max(0.0, float((cp.stdout or "0").strip() or 0.0))
        except Exception:
            pass
        return 0.0

    def _soundtrack_mode_changed(self):
        if self._loading_inspector:
            return
        if not hasattr(self, "soundtrack_mode_combo"):
            return
        mode = str(self.soundtrack_mode_combo.currentData() or "mix")
        if mode not in {"mix", "clips_only", "soundtrack_only"}:
            mode = "mix"
        old_mode = str(self.project.get("soundtrack_mode") or "mix")
        if old_mode == mode:
            return
        self._record_undo_state("Change soundtrack mix mode")
        self.project["soundtrack_mode"] = mode
        self._invalidate_assembly("Soundtrack mix mode changed — assemble again when ready.")
        self._refresh_all()

    def _soundtrack_by_id(self, audio_id):
        for idx, item in enumerate(self.project.get("soundtrack_clips") or []):
            if str(item.get("id") or "") == str(audio_id):
                return idx, item
        return -1, None

    def _add_soundtrack_audio(self):
        if not self._ensure_project_setup_for_first_edit():
            return
        names, _ = QFileDialog.getOpenFileNames(
            self, "Add soundtrack audio", "",
            "Audio / media (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma *.mp4 *.mov *.mkv *.webm);;All files (*.*)"
        )
        if not names:
            return
        additions = []
        for name in names:
            path = Path(name)
            if not path.is_file():
                continue
            duration = self._probe_audio_duration(str(path))
            if duration <= 0.001:
                QMessageBox.warning(self, "Soundtrack", f"Could not read an audio duration from:\n{path}")
            additions.append({
                "id": uuid.uuid4().hex,
                "path": str(path),
                "name": path.stem,
                "source_duration": duration,
                "trim_in": 0.0,
                "trim_out": duration,
                "timeline_start": 0.0,
                "timeline_end": None,
                "volume_percent": 100,
                "fade_in_seconds": 0.0,
                "fade_out_seconds": 0.0,
            })
        if not additions:
            return
        self._record_undo_state("Add soundtrack audio")
        self.project.setdefault("soundtrack_clips", []).extend(additions)
        self._invalidate_assembly("Soundtrack changed — assemble again when ready.")
        self._refresh_all()

    def _show_audio_context_menu(self, audio_id, global_pos):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None:
            return
        menu = QMenu(self)
        trim = menu.addAction("Trim audio…")
        volume = menu.addAction(f"Volume…  {int(round(float(item.get('volume_percent', 100) or 100)))}%")
        fade_in_action = menu.addAction(f"Fade in…  {max(0.0, float(item.get('fade_in_seconds') or 0.0)):.1f}s")
        fade_out_action = menu.addAction(f"Fade out…  {max(0.0, float(item.get('fade_out_seconds') or 0.0)):.1f}s")
        menu.addSeparator()
        split_action = menu.addAction("Split audio at…")
        start_action = menu.addAction(f"Set exact timeline start…  {float(item.get('timeline_start') or 0.0):.2f}s")
        end_action = menu.addAction(f"Set exact timeline end…  {_soundtrack_clip_end(item):.2f}s")
        menu.addSeparator()
        remove = menu.addAction("Remove audio from soundtrack")
        chosen = menu.exec(global_pos)
        if chosen is trim:
            self._trim_soundtrack_audio(audio_id)
        elif chosen is volume:
            self._set_soundtrack_volume(audio_id)
        elif chosen is fade_in_action:
            self._set_soundtrack_fade(audio_id, "in")
        elif chosen is fade_out_action:
            self._set_soundtrack_fade(audio_id, "out")
        elif chosen is split_action:
            self._split_soundtrack_audio(audio_id)
        elif chosen is start_action:
            self._set_soundtrack_start(audio_id)
        elif chosen is end_action:
            self._set_soundtrack_end(audio_id)
        elif chosen is remove:
            self._remove_soundtrack_audio(audio_id)

    def _trim_soundtrack_audio(self, audio_id):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None:
            return
        duration = max(0.001, float(item.get("source_duration") or self._probe_audio_duration(item.get("path")) or 0.001))
        dlg = QDialog(self)
        dlg.setWindowTitle("Trim soundtrack audio")
        form = QFormLayout(dlg)
        start = QDoubleSpinBox(); end = QDoubleSpinBox()
        for spin in (start, end):
            spin.setRange(0.0, duration); spin.setDecimals(3); spin.setSingleStep(0.1); spin.setSuffix(" s")
        start.setValue(max(0.0, min(duration, float(item.get("trim_in") or 0.0))))
        raw_out = item.get("trim_out")
        end.setValue(max(start.value(), min(duration, float(raw_out) if raw_out is not None else duration)))
        form.addRow("Source start", start); form.addRow("Source end", end)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); form.addRow(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_in = float(start.value()); new_out = float(end.value())
        if new_out <= new_in + 0.001:
            QMessageBox.warning(self, "Soundtrack", "The audio end must be after its start.")
            return
        self._record_undo_state("Trim soundtrack audio")
        item["trim_in"] = round(new_in, 3); item["trim_out"] = round(new_out, 3)
        natural_end = float(item.get("timeline_start") or 0.0) + (new_out - new_in)
        if item.get("timeline_end") is not None:
            item["timeline_end"] = min(float(item.get("timeline_end")), natural_end)
        self._invalidate_assembly("Soundtrack trim changed — assemble again when ready.")
        self._refresh_all()

    def _set_soundtrack_volume(self, audio_id):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None: return
        current = int(round(float(item.get("volume_percent", 100) or 100)))
        value, ok = QInputDialog.getInt(self, "Soundtrack volume", "Volume", current, 0, 150, 1)
        if not ok or value == current: return
        self._record_undo_state("Change soundtrack volume")
        item["volume_percent"] = int(value)
        self._invalidate_assembly("Soundtrack volume changed — assemble again when ready.")
        self._refresh_all()

    def _set_soundtrack_fade(self, audio_id, direction: str):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None:
            return
        key = "fade_in_seconds" if str(direction).lower() == "in" else "fade_out_seconds"
        label = "Fade in" if key == "fade_in_seconds" else "Fade out"
        duration = max(0.001, _soundtrack_clip_duration(item))
        try:
            current = max(0.0, min(duration, float(item.get(key) or 0.0)))
        except Exception:
            current = 0.0
        value, ok = QInputDialog.getDouble(
            self, f"Soundtrack {label.lower()}",
            f"{label} duration in seconds (0 = off)",
            current, 0.0, duration, 3,
        )
        if not ok:
            return
        value = max(0.0, min(duration, float(value)))
        if abs(value - current) < 0.0005:
            return
        self._record_undo_state(f"Change soundtrack {label.lower()}")
        item[key] = round(value, 3)
        self._invalidate_assembly(f"Soundtrack {label.lower()} changed — assemble again when ready.")
        self._refresh_all()

    def _split_soundtrack_audio(self, audio_id):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None:
            return
        start = max(0.0, float(item.get("timeline_start") or 0.0))
        duration = max(0.0, _soundtrack_clip_duration(item))
        end = start + duration
        if duration <= 0.002:
            QMessageBox.information(self, "Split soundtrack audio", "This soundtrack block is too short to split.")
            return
        default = start + duration / 2.0
        value, ok = QInputDialog.getDouble(
            self, "Split soundtrack audio",
            "Split at video timeline time (seconds)",
            default, start + 0.001, end - 0.001, 3,
        )
        if not ok:
            return
        split_time = max(start + 0.001, min(end - 0.001, float(value)))
        left_duration = split_time - start
        right_duration = end - split_time
        try:
            source_in = max(0.0, float(item.get("trim_in") or 0.0))
        except Exception:
            source_in = 0.0
        source_split = source_in + left_duration
        source_end = source_split + right_duration

        self._record_undo_state("Split soundtrack audio")
        original_fade_in = max(0.0, float(item.get("fade_in_seconds") or 0.0))
        original_fade_out = max(0.0, float(item.get("fade_out_seconds") or 0.0))

        # Left half keeps the original entrance fade; the newly-created cut itself
        # is hard by default so each side can be adjusted independently afterward.
        item["trim_out"] = round(source_split, 3)
        item["timeline_end"] = round(split_time, 3)
        item["fade_in_seconds"] = round(min(original_fade_in, left_duration), 3)
        item["fade_out_seconds"] = 0.0

        right = copy.deepcopy(item)
        right["id"] = uuid.uuid4().hex
        right["name"] = str(item.get("name") or "Audio") + " (split)"
        right["trim_in"] = round(source_split, 3)
        right["trim_out"] = round(source_end, 3)
        right["timeline_start"] = round(split_time, 3)
        right["timeline_end"] = round(end, 3)
        right["fade_in_seconds"] = 0.0
        right["fade_out_seconds"] = round(min(original_fade_out, right_duration), 3)
        self.project.setdefault("soundtrack_clips", []).insert(idx + 1, right)
        self._invalidate_assembly("Soundtrack split changed — assemble again when ready.")
        self._refresh_all()

    def _set_soundtrack_start(self, audio_id):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None: return
        total = max(0.0, _timeline_total_seconds(self._clips()))
        current = max(0.0, float(item.get("timeline_start") or 0.0))
        value, ok = QInputDialog.getDouble(self, "Soundtrack start", "Start on video timeline (seconds)", current, 0.0, max(36000.0, total), 3)
        if not ok: return
        self._record_undo_state("Move soundtrack audio")
        old_duration = _soundtrack_clip_duration(item)
        item["timeline_start"] = round(float(value), 3)
        item["timeline_end"] = round(float(value) + old_duration, 3)
        self._invalidate_assembly("Soundtrack timing changed — assemble again when ready.")
        self._refresh_all()

    def _set_soundtrack_end(self, audio_id):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None: return
        start = max(0.0, float(item.get("timeline_start") or 0.0))
        natural_end = start + max(0.0, float(item.get("trim_out") or item.get("source_duration") or 0.0) - float(item.get("trim_in") or 0.0))
        current = _soundtrack_clip_end(item)
        value, ok = QInputDialog.getDouble(self, "Soundtrack end", "End on video timeline (seconds)", current, start + 0.001, max(start + 0.001, natural_end), 3)
        if not ok: return
        self._record_undo_state("Set soundtrack end")
        item["timeline_end"] = round(float(value), 3)
        self._invalidate_assembly("Soundtrack timing changed — assemble again when ready.")
        self._refresh_all()

    def _remove_soundtrack_audio(self, audio_id):
        idx, item = self._soundtrack_by_id(audio_id)
        if item is None: return
        self._record_undo_state("Remove soundtrack audio")
        del self.project.setdefault("soundtrack_clips", [])[idx]
        self._invalidate_assembly("Soundtrack changed — assemble again when ready.")
        self._refresh_all()

    def _load_external_clip_after(self, clip_id):
        """Insert an existing ready-made video immediately after the clicked block."""
        idx, _clicked = self._clip_by_id(clip_id)
        if idx < 0:
            return False
        if not self._ensure_project_setup_for_first_edit():
            return False
        name, _ = QFileDialog.getOpenFileName(
            self, "Load clip onto timeline", "",
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All files (*.*)"
        )
        if not name:
            return False
        path = Path(name)
        if not path.is_file():
            return False
        details = self._probe_video_details(str(path))
        duration = float(details.get("duration") or self._probe_start_video_duration(str(path)) or 0.0)
        if duration <= 0.001:
            QMessageBox.warning(self, "Load clip", "Could not read the duration of this video file.")
            return False
        self._record_undo_state("Load clip")
        clip = self._blank_clip(continue_previous=False)
        clip["name"] = path.stem or f"Loaded clip {idx + 2}"
        clip["generation_mode"] = "loaded"
        clip["start_source_video"] = str(path)
        clip["source_duration_seconds"] = duration
        clip["frames"] = max(1, int(round(duration * FPS)))
        clip["edit_mode"] = "standalone"
        clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0}]
        clip["settings"] = {}
        clip["use_reference_images"] = False
        clip["reference_images"] = []
        clip["queue_job_id"] = None
        clip["output"] = str(path)
        clip["status"] = "finished"
        clip["stale"] = False
        clip["hq_generated"] = False
        clip["last_generation_info"] = {}
        if int(details.get("width") or 0) > 0 and int(details.get("height") or 0) > 0:
            clip["loaded_resolution"] = f"{int(details['width'])} × {int(details['height'])}"
        if details.get("fps"):
            clip["loaded_fps"] = str(details.get("fps"))
        self._clips().insert(idx + 1, clip)
        self._renumber_default_names()
        self._invalidate_assembly("Loaded clip added — assemble again when ready.")
        self.selected_clip_id = str(clip["id"])
        self.selected_clip_ids = {str(clip["id"])}
        self._selection_anchor_id = str(clip["id"])
        self._refresh_all()
        return True

    def _show_clip_context_menu(self, clip_id, global_pos):
        idx, clip = self._clip_by_id(clip_id)
        if clip is None:
            return

        # Right click makes that block the primary target, but does not destroy an
        # existing multi-selection when the clicked block is already part of it.
        clip_id = str(clip_id)
        if clip_id not in self.selected_clip_ids:
            self._apply_selection({clip_id}, clip_id, anchor=clip_id)
        elif self.selected_clip_id != clip_id:
            self.selected_clip_id = clip_id
            self._refresh_all()

        menu = QMenu(self)
        regenerate = menu.addAction("(Re)generate this clip")
        regenerate_alternate = menu.addAction("Regenerate as alternate")
        low_res_test = menu.addAction("Create low res test")
        ref_presets = menu.addAction("Ref presets…")

        candidates_menu = menu.addMenu("Alternate candidates")
        self._populate_candidate_menu(candidates_menu, clip_id, clip)

        load_clip_action = menu.addAction("Load clip…")
        preview = menu.addAction("Preview clip")
        open_folder = menu.addAction("Open clip folder")
        trim = menu.addAction("Trim clip…")
        volume_action = menu.addAction(f"Volume…  {int(round(float(clip.get('volume_percent', 100) or 100)))}%")
        fade_in_action = menu.addAction(f"Audio fade in…  {max(0.0, float(clip.get('audio_fade_in_seconds') or 0.0)):.1f}s")
        fade_out_action = menu.addAction(f"Audio fade out…  {max(0.0, float(clip.get('audio_fade_out_seconds') or 0.0)):.1f}s")
        video_fade_in_action = None
        video_fade_out_action = None
        if idx == 0:
            video_fade_in_action = menu.addAction(
                f"Video fade in from black…  {max(0.0, float(clip.get('video_fade_in_seconds') or 0.0)):.1f}s"
            )
        if idx == len(self._clips()) - 1:
            video_fade_out_action = menu.addAction(
                f"Video fade out to black…  {max(0.0, float(clip.get('video_fade_out_seconds') or 0.0)):.1f}s"
            )
        speed_action = menu.addAction(f"Speed…  {_clip_speed(clip):.1f}x")
        transition_name = _transition_name(clip)
        transition_suffix = "" if transition_name == "none" else f"  {_TRANSITION_LABELS.get(transition_name, transition_name)} {_transition_requested_seconds(clip):.1f}s"
        transition_action = menu.addAction(f"Transition to next clip…{transition_suffix}")
        remove = menu.addAction("Remove clip from timeline")
        menu.addSeparator()
        lock_action = menu.addAction("Unlock clip" if bool(clip.get("locked", False)) else "Lock clip")
        notes = menu.addAction("Add / read notes")
        info = menu.addAction("Info")

        locked = bool(clip.get("locked", False))
        media_clip = str(clip.get("generation_mode") or "") in {"source", "loaded"}
        preview_path = self._clip_preview_path(clip)
        output_exists = bool(preview_path)
        regenerate.setEnabled(not locked and not media_clip)
        regenerate_alternate.setEnabled(not locked and not media_clip and output_exists)
        low_res_test.setEnabled(not locked and not media_clip)
        ref_presets.setEnabled(any(
            not bool(c.get("locked", False)) and str(c.get("generation_mode") or "") not in {"source", "loaded"}
            for c in self._clips() if str(c.get("id")) in self.selected_clip_ids
        ))
        # Stale is metadata only: preserved media remains previewable/openable.
        preview.setEnabled(output_exists)
        open_folder.setEnabled(output_exists)
        trim.setEnabled(output_exists and not locked)
        transition_action.setEnabled(not locked and idx + 1 < len(self._clips()))
        transition_action.setToolTip("Transitions belong to the outgoing cut after this block." if idx + 1 < len(self._clips()) else "The final timeline clip has no next clip to transition into.")
        remove.setEnabled(not locked)

        chosen = menu.exec(global_pos)
        if chosen is regenerate:
            self.selected_clip_id = clip_id
            self.generate_selected()
        elif chosen is regenerate_alternate:
            self._generate_alternate_by_id(clip_id)
        elif chosen is low_res_test:
            self._generate_low_res_test_by_id(clip_id)
        elif chosen is ref_presets:
            self._open_ref_presets()
        elif chosen is load_clip_action:
            self._load_external_clip_after(clip_id)
        elif chosen is preview:
            self._preview_clip_by_id(clip_id)
        elif chosen is open_folder:
            self._open_clip_folder_by_id(clip_id)
        elif chosen is trim:
            self._trim_clip_by_id(clip_id)
        elif chosen is volume_action:
            self._set_clip_volume_by_id(clip_id)
        elif chosen is fade_in_action:
            self._set_clip_audio_fade_by_id(clip_id, "in")
        elif chosen is fade_out_action:
            self._set_clip_audio_fade_by_id(clip_id, "out")
        elif video_fade_in_action is not None and chosen is video_fade_in_action:
            self._set_clip_video_edge_fade_by_id(clip_id, "in")
        elif video_fade_out_action is not None and chosen is video_fade_out_action:
            self._set_clip_video_edge_fade_by_id(clip_id, "out")
        elif chosen is speed_action:
            self._set_clip_speed_by_id(clip_id)
        elif chosen is transition_action:
            self._set_clip_transition_by_id(clip_id)
        elif chosen is remove:
            self._remove_clip_by_id(clip_id)
        elif chosen is lock_action:
            self._toggle_clip_lock_by_id(clip_id)
        elif chosen is notes:
            self._edit_clip_notes(clip_id)
        elif chosen is info:
            self._show_clip_info(clip_id)

    @staticmethod
    def _candidate_snapshot_from_active(clip: dict) -> dict:
        """Capture the current active render so it can be swapped back later."""
        data = {
            "candidate_id": uuid.uuid4().hex,
            "output": str(clip.get("output") or ""),
            "queue_job_id": clip.get("queue_job_id"),
            "status": str(clip.get("status") or "finished"),
            "last_generation_info": copy.deepcopy(clip.get("last_generation_info") or {}),
            "hq_generated": bool(clip.get("hq_generated", False)),
            "stale": bool(clip.get("stale", False)),
        }
        if "trim_in" in clip:
            data["trim_in"] = clip.get("trim_in")
        if "trim_out" in clip:
            data["trim_out"] = clip.get("trim_out")
        return data

    def _archive_active_candidate_before_replacement(self, clip: dict, replacement_output: str = "") -> bool:
        """Preserve the current active render before a normal regeneration replaces it.

        This makes ordinary retries participate in the same A/B history as explicit
        ``Regenerate as alternate`` jobs.  Only a real media file is archived and
        the same file is never inserted twice.
        """
        if not isinstance(clip, dict) or str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            return False

        active = self._clip_preview_path(clip)
        if not active:
            return False
        active_path = Path(str(active))
        if not active_path.is_file():
            return False

        try:
            active_key = str(active_path.resolve()).lower()
        except Exception:
            active_key = str(active_path).lower()

        if replacement_output:
            try:
                replacement_key = str(Path(str(replacement_output)).resolve()).lower()
            except Exception:
                replacement_key = str(replacement_output).lower()
            if replacement_key == active_key:
                return False

        candidates = clip.setdefault("alternate_candidates", [])
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_output = str(candidate.get("output") or "")
            if not candidate_output:
                continue
            try:
                candidate_key = str(Path(candidate_output).resolve()).lower()
            except Exception:
                candidate_key = candidate_output.lower()
            if candidate_key == active_key:
                return False

        snapshot = self._candidate_snapshot_from_active(clip)
        snapshot["output"] = str(active_path)
        # An existing playable render is a finished candidate even when the block
        # itself is currently marked stale/informational.
        snapshot["status"] = "finished"
        candidates.append(snapshot)
        return True

    @staticmethod
    def _candidate_label(candidate: dict, number: int) -> str:
        state = str(candidate.get("status") or "").strip().lower()
        path = Path(str(candidate.get("output") or ""))
        info = candidate.get("last_generation_info") if isinstance(candidate.get("last_generation_info"), dict) else {}
        resolution = str(info.get("resolution") or "").strip()
        seed = info.get("actual_seed", info.get("seed"))
        bits = [f"Candidate {number}"]
        if resolution:
            bits.append(resolution)
        if seed not in (None, ""):
            bits.append(f"seed {seed}")
        if state and state != "finished":
            bits.append(state)
        elif path.name:
            bits.append(path.name)
        return " — ".join(bits)

    def _populate_candidate_menu(self, menu, clip_id, clip):
        active = self._clip_preview_path(clip)
        active_name = Path(active).name if active else "no active render"
        active_action = menu.addAction(f"Active — {active_name}")
        active_action.setEnabled(False)
        menu.addSeparator()

        candidates = clip.get("alternate_candidates") if isinstance(clip.get("alternate_candidates"), list) else []
        if not candidates:
            empty = menu.addAction("No alternate candidates yet")
            empty.setEnabled(False)
            return
        for n, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                continue
            action = menu.addAction(self._candidate_label(candidate, n))
            output = str(candidate.get("output") or "")
            ready = str(candidate.get("status") or "") == "finished" and bool(output) and Path(output).is_file()
            action.setEnabled(ready)
            action.triggered.connect(lambda _checked=False, cid=str(clip_id), cand_id=str(candidate.get("candidate_id") or ""): self._activate_candidate(cid, cand_id))

    def _generate_alternate_by_id(self, clip_id):
        idx, clip = self._clip_by_id(clip_id)
        if clip is None or idx < 0:
            return False
        if bool(clip.get("locked", False)):
            QMessageBox.information(self, "Alternate candidate", "This block is locked. Unlock it before creating an alternate.")
            return False
        if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            QMessageBox.information(self, "Alternate candidate", "A loaded video clip cannot be regenerated as an alternate.")
            return False
        if not self._clip_preview_path(clip):
            QMessageBox.information(self, "Alternate candidate", "Create the first render normally before adding alternate candidates.")
            return False
        self.selected_clip_id = str(clip_id)
        return self._generate_selected_impl(alternate_candidate=True)

    def _activate_candidate(self, clip_id, candidate_id):
        idx, clip = self._clip_by_id(clip_id)
        if clip is None:
            return False
        candidates = clip.get("alternate_candidates") if isinstance(clip.get("alternate_candidates"), list) else []
        target_index = next((i for i, c in enumerate(candidates) if isinstance(c, dict) and str(c.get("candidate_id") or "") == str(candidate_id)), -1)
        if target_index < 0:
            return False
        target = candidates[target_index]
        output = str(target.get("output") or "")
        if str(target.get("status") or "") != "finished" or not output or not Path(output).is_file():
            QMessageBox.warning(self, "Alternate candidate", "That candidate is not a finished render on disk yet.")
            return False

        self._record_undo_state("Switch alternate candidate")
        current = self._candidate_snapshot_from_active(clip)
        # The chosen candidate becomes active; the previous active render takes
        # its place in the alternate list so switching never destroys a render.
        candidates[target_index] = current
        clip["alternate_candidates"] = candidates
        clip["output"] = output
        clip["queue_job_id"] = target.get("queue_job_id")
        clip["status"] = str(target.get("status") or "finished")
        clip["last_generation_info"] = copy.deepcopy(target.get("last_generation_info") or {})
        clip["hq_generated"] = bool(target.get("hq_generated", False))
        clip["stale"] = bool(target.get("stale", False))
        if "trim_in" in target:
            clip["trim_in"] = target.get("trim_in")
        else:
            clip.pop("trim_in", None)
        if "trim_out" in target:
            clip["trim_out"] = target.get("trim_out")
        else:
            clip.pop("trim_out", None)
        self._invalidate_assembly("Active clip candidate changed — assemble again.")
        self._refresh_all()
        return True

    def _generate_low_res_test_by_id(self, clip_id):
        """Regenerate exactly one block at MiniMax's supported 384p test size.

        This deliberately reuses the exact same single-block regeneration path
        as (Re)generate this clip.  Resolution is the only one-shot difference,
        so continuation clips at the end of a timeline behave identically.
        """
        idx, clip = self._clip_by_id(clip_id)
        if clip is None or idx < 0:
            return False
        if bool(clip.get("locked", False)):
            QMessageBox.information(self, "Low res test", "This block is locked. Unlock it before creating a low-res test.")
            return False
        if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            QMessageBox.information(self, "Low res test", "A loaded video clip cannot be regenerated as a low-res test.")
            return False

        settings = clip.get("settings") if isinstance(clip.get("settings"), dict) else {}
        aspect = str(settings.get("aspect") or "16:9").strip()
        if aspect not in {"16:9", "9:16", "1:1", "21:9"}:
            aspect = "16:9"
        if aspect == "21:9":
            override = {"aspect": "21:9", "widescreen_quality": "Low — 896 × 384"}
        else:
            override = {"aspect": aspect, "resolution": "736 × 384"}

        # Right-click already makes this block primary, but set it explicitly so
        # this helper is also correct if called from elsewhere later.
        self.selected_clip_id = str(clip_id)
        return self._generate_selected_impl(
            one_shot_resolution_override=override,
            low_res_test=True,
        )

    def _preview_clip_by_id(self, clip_id):
        _idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        output = self._clip_preview_path(clip)
        if not output:
            return False
        if callable(self.preview_result_callback):
            return bool(self.preview_result_callback(
                output,
                clip.get("queue_job_id"),
                clip.get("trim_in"),
                clip.get("trim_out"),
                clip.get("volume_percent", 100),
                _clip_speed(clip),
            ))
        return False



    def _open_clip_folder_by_id(self, clip_id):
        _idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        output = self._clip_preview_path(clip)
        if not output:
            return False
        if callable(self.open_output_callback):
            return bool(self.open_output_callback(output))
        return False

    def _toggle_clip_lock_by_id(self, clip_id):
        _idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        currently_locked = bool(clip.get("locked", False))
        self._record_undo_state("Unlock clip" if currently_locked else "Lock clip")
        clip["locked"] = not currently_locked
        self._refresh_all()
        return True

    def _set_clip_volume_by_id(self, clip_id):
        _idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        try:
            current = int(round(float(clip.get("volume_percent", 100) or 100)))
        except Exception:
            current = 100
        current = max(0, min(150, current))
        value, accepted = QInputDialog.getInt(
            self,
            "Clip volume",
            "Volume for this clip (0% = mute, 100% = original, 150% = boost):",
            current, 0, 150, 1,
        )
        if not accepted or int(value) == current:
            return bool(accepted)
        self._record_undo_state("Change clip volume")
        value = int(value)
        if value == 100:
            clip.pop("volume_percent", None)
        else:
            clip["volume_percent"] = value
        self._invalidate_assembly("Clip volume changed — assemble again.")
        self._refresh_all()
        return True

    def _set_clip_audio_fade_by_id(self, clip_id, direction):
        _idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        key = "audio_fade_in_seconds" if str(direction).lower() == "in" else "audio_fade_out_seconds"
        label = "Audio fade in" if key == "audio_fade_in_seconds" else "Audio fade out"
        try:
            current = max(0.0, float(clip.get(key) or 0.0))
        except Exception:
            current = 0.0
        try:
            duration = max(0.001, float(_clip_timeline_seconds(clip)))
        except Exception:
            duration = max(0.001, float(clip.get("frames") or 24) / FPS)
        value, accepted = QInputDialog.getDouble(
            self,
            label,
            "Fade duration in assembled video (seconds; 0 disables the fade):",
            min(current, duration),
            0.0,
            duration,
            3,
        )
        if not accepted:
            return False
        value = max(0.0, min(duration, float(value)))
        if abs(value - current) < 0.0005:
            return True
        self._record_undo_state(f"Change {label.lower()}")
        if value <= 0.0005:
            clip.pop(key, None)
        else:
            clip[key] = round(value, 3)
        self._invalidate_assembly(f"{label} changed — assemble again.")
        self._refresh_all()
        return True

    def _set_clip_video_edge_fade_by_id(self, clip_id, direction):
        idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        direction = str(direction).lower()
        clips = self._clips()
        if direction == "in":
            if idx != 0:
                return False
            key = "video_fade_in_seconds"
            label = "Video fade in from black"
        else:
            if idx != len(clips) - 1:
                return False
            key = "video_fade_out_seconds"
            label = "Video fade out to black"
        try:
            current = max(0.0, float(clip.get(key) or 0.0))
        except Exception:
            current = 0.0
        try:
            duration = max(0.001, float(_clip_timeline_seconds(clip)))
        except Exception:
            duration = max(0.001, float(clip.get("frames") or 24) / FPS)
        value, accepted = QInputDialog.getDouble(
            self,
            label,
            "Fade duration in assembled video (seconds; 0 disables the fade):",
            min(current, duration),
            0.0,
            duration,
            3,
        )
        if not accepted:
            return False
        value = max(0.0, min(duration, float(value)))
        if abs(value - current) < 0.0005:
            return True
        self._record_undo_state(f"Change {label.lower()}")
        if value <= 0.0005:
            clip.pop(key, None)
        else:
            clip[key] = round(value, 3)
        self._invalidate_assembly(f"{label} changed — assemble again.")
        self._refresh_all()
        return True

    def _set_clip_speed_by_id(self, clip_id):
        _idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        current = _clip_speed(clip)
        dlg = QInputDialog(self)
        dlg.setWindowTitle("Clip speed")
        dlg.setLabelText("Playback speed for this clip (0.5x = half speed, 2.0x = double speed):")
        dlg.setInputMode(QInputDialog.InputMode.DoubleInput)
        dlg.setDoubleRange(0.5, 2.0)
        dlg.setDoubleDecimals(1)
        dlg.setDoubleStep(0.1)
        dlg.setDoubleValue(current)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        value = max(0.5, min(2.0, round(float(dlg.doubleValue()), 1)))
        if abs(value - current) < 0.0001:
            return True
        self._record_undo_state("Change clip speed")
        if abs(value - 1.0) < 0.0001:
            clip.pop("speed_multiplier", None)
        else:
            clip["speed_multiplier"] = value
        self._invalidate_assembly("Clip speed changed — assemble again.")
        self._refresh_all()
        return True

    def _set_clip_transition_by_id(self, clip_id):
        idx, clip = self._clip_by_id(clip_id)
        if clip is None or idx < 0:
            return False
        if idx + 1 >= len(self._clips()):
            QMessageBox.information(self, "Timeline transition", "The final clip has no next clip to transition into.")
            return False
        if bool(clip.get("locked", False)):
            QMessageBox.information(self, "Timeline transition", "This block is locked. Unlock it before changing its outgoing transition.")
            return False
        dlg = TransitionDialog(clip, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        kind, duration, audio_mode = dlg.values()
        old = (_transition_name(clip), _transition_requested_seconds(clip), str(clip.get("transition_audio_mode") or "crossfade"))
        new = (kind, duration if kind != "none" else 0.0, audio_mode)
        if old == new:
            return True
        self._record_undo_state("Change clip transition")
        if kind == "none":
            clip.pop("transition_to_next", None)
            clip.pop("transition_duration", None)
            clip.pop("transition_audio_mode", None)
        else:
            clip["transition_to_next"] = kind
            clip["transition_duration"] = duration
            clip["transition_audio_mode"] = audio_mode
        self._invalidate_assembly("Timeline transition changed — assemble again.")
        self._refresh_all()
        return True

    def _trim_clip_by_id(self, clip_id):
        _idx, clip = self._clip_by_id(clip_id)
        if not clip:
            return False
        output = self._clip_preview_path(clip)
        if not output:
            return False
        details = self._probe_video_details(output)
        duration = float(details.get("duration") or 0.0)
        if duration <= 0:
            duration = _clip_seconds(clip)
        trim_in = max(0.0, float(clip.get("trim_in") or 0.0))
        saved_out = clip.get("trim_out")
        try:
            trim_out = float(saved_out) if saved_out is not None else duration
        except Exception:
            trim_out = duration
        trim_out = max(trim_in, min(trim_out, duration))
        dlg = ClipTrimDialog(output, duration, trim_in, trim_out, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        new_in, new_out = dlg.trim_values()
        # Treat a full-source range as no trim so old/default projects stay clean.
        full_range = new_in <= 0.001 and abs(new_out - duration) <= 0.02
        old_in = float(clip.get("trim_in") or 0.0)
        old_out_raw = clip.get("trim_out")
        old_out = duration if old_out_raw is None else float(old_out_raw)
        if abs(old_in - new_in) <= 0.001 and abs(old_out - new_out) <= 0.001:
            return True
        self._record_undo_state("Trim clip")
        if full_range:
            clip.pop("trim_in", None)
            clip.pop("trim_out", None)
        else:
            clip["trim_in"] = round(new_in, 3)
            clip["trim_out"] = round(new_out, 3)
        self._invalidate_assembly("Clip trim changed — assemble again.")
        self._refresh_all()
        return True

    def _remove_clip_by_id(self, clip_id):
        idx, clip = self._clip_by_id(clip_id)
        if clip is None or bool(clip.get("locked", False)):
            return False
        name = str(clip.get("name") or f"Clip {idx + 1}")
        answer = QMessageBox.question(
            self,
            "Remove clip from timeline",
            f"Remove {name} from the timeline?\n\nThe rendered video file will not be deleted from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        self._record_undo_state("Remove clip from timeline")
        removed_id = str(clip.get("id") or "")
        self._clips().pop(idx)
        self._renumber_default_names()
        self._repair_continuation_chain(mark_stale=True, stale_from=(idx if idx < len(self._clips()) else None))
        self._invalidate_assembly()
        self.selected_clip_ids.discard(removed_id)
        if self._clips():
            fallback_index = min(idx, len(self._clips()) - 1)
            fallback = self._clips()[fallback_index]
            self.selected_clip_id = str(fallback.get("id"))
            self.selected_clip_ids.add(self.selected_clip_id)
            self._selection_anchor_id = self.selected_clip_id
        else:
            self.selected_clip_id = None
            self.selected_clip_ids.clear()
            self._selection_anchor_id = None
            self._ensure_initial_clip()
        self.selected_cut_index = 0
        self._refresh_all()
        return True

    def _edit_clip_notes(self, clip_id):
        _idx, clip = self._clip_by_id(clip_id)
        if clip is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Notes — {clip.get('name') or 'Timeline clip'}")
        dlg.resize(620, 420)
        layout = QVBoxLayout(dlg)
        hint = QLabel("Director notes are stored with this timeline block and are never sent to MiniMax for generation.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        editor = QPlainTextEdit(dlg)
        editor.setPlaceholderText("Add continuity reminders, fixes to make later, performance notes, story intent, etc.")
        editor.setPlainText(str(clip.get("notes") or ""))
        layout.addWidget(editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, parent=dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        value = editor.toPlainText().rstrip()
        if value == str(clip.get("notes") or ""):
            return
        self._record_undo_state("Edit clip notes")
        clip["notes"] = value
        self._refresh_all()

    @staticmethod
    def _format_bytes(value):
        try:
            size = float(value)
        except Exception:
            return "Unknown"
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if size < 1024.0 or unit == units[-1]:
                return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
            size /= 1024.0
        return "Unknown"

    def _probe_video_details(self, output):
        path = Path(str(output or ""))
        if not path.is_file():
            return {}
        details = {"file_size": path.stat().st_size}
        try:
            from runtime.ffmpeg_tools import tool_path as ffmpeg_tool_path
            exe = str(ffmpeg_tool_path("ffprobe.exe"))
            cp = subprocess.run(
                [exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if cp.returncode == 0:
                data = json.loads(cp.stdout or "{}")
                stream = (data.get("streams") or [{}])[0]
                fmt = data.get("format") or {}
                details["width"] = int(stream.get("width") or 0)
                details["height"] = int(stream.get("height") or 0)
                details["duration"] = float(fmt.get("duration") or 0.0)
                details["fps"] = str(stream.get("r_frame_rate") or "")
        except Exception:
            pass
        return details

    @staticmethod
    def _generation_info_from_job(job):
        if not isinstance(job, dict):
            return {}
        info = {}
        for key in (
            "id", "job_number", "mode_name", "model_label", "output", "seed", "actual_seed",
            "resolution", "frames", "steps", "prompt", "created_at", "started_at", "finished_at",
        ):
            value = job.get(key)
            if value is not None:
                info[key] = copy.deepcopy(value)
        settings = job.get("settings")
        if isinstance(settings, dict):
            try:
                info["settings"] = json.loads(json.dumps(settings, default=str))
            except Exception:
                info["settings"] = {str(k): str(v) for k, v in settings.items()}
        return info

    @staticmethod
    def _append_settings_lines(lines, settings):
        if not isinstance(settings, dict):
            return
        preferred = (
            "model", "model_path", "checkpoint", "checkpoint_path", "hybrid_checkpoint", "aspect", "resolution",
            "widescreen_quality", "seed", "steps", "cfg", "shift", "audio_shift", "sampler", "scheduler",
            "lora", "loras", "lora_paths", "lora_strength", "ref_images", "continue_audio_memory", "latent_continuation",
        )
        shown = set()
        for key in preferred:
            if key not in settings:
                continue
            value = settings.get(key)
            if value in (None, "", [], {}):
                continue
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}")
            shown.add(key)
        extras = []
        for key in sorted(settings):
            if key in shown:
                continue
            value = settings.get(key)
            if value in (None, "", [], {}):
                continue
            extras.append(f"{key.replace('_', ' ').title()}: {value}")
        if extras:
            lines.append("")
            lines.append("Other saved settings:")
            lines.extend(extras)

    def _show_clip_info(self, clip_id):
        idx, clip = self._clip_by_id(clip_id)
        if clip is None:
            return
        output = str(clip.get("output") or "").strip()
        disk = self._probe_video_details(output)
        trim_in = float(clip.get("trim_in") or 0.0)
        trim_out = clip.get("trim_out")
        if trim_out is not None:
            try:
                trim_text = f"{trim_in:.3f}s → {float(trim_out):.3f}s ({max(0.0, float(trim_out)-trim_in):.3f}s used)"
            except Exception:
                trim_text = "Invalid trim metadata"
        else:
            trim_text = "None"
        generated = clip.get("last_generation_info") if isinstance(clip.get("last_generation_info"), dict) else {}
        settings = generated.get("settings") if isinstance(generated.get("settings"), dict) else clip.get("settings", {})
        prompt = str(generated.get("prompt") or _compiled_reference_prompt(clip) or _compiled_prompt(clip) or "")

        lines = [
            f"Clip: {clip.get('name') or f'Clip {idx + 1}'}",
            f"Timeline position: {idx + 1}",
            f"Status: {clip.get('status') or 'draft'}",
            f"Locked: {'Yes' if clip.get('locked') else 'No'}",
            f"HQ: {'Yes' if clip.get('hq_generated') else 'No'}",
            f"References: {len(_reference_entries(clip))}",
            f"Assembly trim: {trim_text}",
            f"Speed: {_clip_speed(clip):.1f}x",
            f"Transition to next: {(_TRANSITION_LABELS.get(_transition_name(clip), 'None') + ('  ' + format(_transition_requested_seconds(clip), '.1f') + 's' if _transition_name(clip) != 'none' else ''))}",
        ]
        if generated:
            if generated.get("job_number") is not None:
                lines.append(f"Queue job: {generated.get('job_number')}")
            if generated.get("model_label"):
                lines.append(f"Model / checkpoint: {generated.get('model_label')}")
            seed = generated.get("actual_seed") if generated.get("actual_seed") is not None else generated.get("seed")
            if seed is not None:
                lines.append(f"Seed used: {seed}")
            if generated.get("steps") is not None:
                lines.append(f"Steps: {generated.get('steps')}")
            if generated.get("resolution"):
                lines.append(f"Queued resolution: {generated.get('resolution')}")
            if generated.get("frames") is not None:
                lines.append(f"Frames: {generated.get('frames')}")
        else:
            lines.append(f"Seed: {(clip.get('settings') or {}).get('seed', 'Unknown')}")
            lines.append(f"Frames: {clip.get('frames') or 'Unknown'}")

        if output:
            lines.append("")
            lines.append(f"Output: {output}")
            if Path(output).is_file():
                lines.append(f"Disk size: {self._format_bytes(disk.get('file_size'))}")
                if disk.get("duration"):
                    lines.append(f"Actual duration: {disk['duration']:.3f} s")
                if disk.get("width") and disk.get("height"):
                    lines.append(f"Actual resolution: {disk['width']} × {disk['height']}")
                if disk.get("fps"):
                    lines.append(f"Video frame rate: {disk['fps']}")
            else:
                lines.append("Output file: Missing from disk")

        lines.append("")
        lines.append("Prompt used:" if generated else "Current saved prompt:")
        lines.append(prompt or "(empty)")
        lines.append("")
        lines.append("Generation settings:" if generated else "Current saved generation settings:")
        self._append_settings_lines(lines, settings)

        notes = str(clip.get("notes") or "").strip()
        if notes:
            lines.extend(["", "Director notes:", notes])

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Clip info — {clip.get('name') or f'Clip {idx + 1}'}")
        dlg.resize(760, 680)
        layout = QVBoxLayout(dlg)
        view = QPlainTextEdit(dlg)
        view.setReadOnly(True)
        view.setPlainText("\n".join(str(x) for x in lines))
        layout.addWidget(view, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=dlg)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        buttons.clicked.connect(lambda _button: dlg.accept())
        layout.addWidget(buttons)
        dlg.exec()

    def preview_selected_result(self):
        clip = self._selected_clip()
        if not clip:
            return
        output = self._clip_preview_path(clip)
        if callable(self.preview_result_callback) and output:
            self.preview_result_callback(
                output,
                clip.get("queue_job_id"),
                clip.get("trim_in"),
                clip.get("trim_out"),
                clip.get("volume_percent", 100),
                _clip_speed(clip),
            )

    def open_selected_output(self):
        clip = self._selected_clip()
        output = self._clip_preview_path(clip or {})
        if callable(self.open_output_callback) and output:
            self.open_output_callback(output)

    def preview_final_result(self):
        output = str(self.project.get("assembled_output") or "")
        if callable(self.preview_result_callback) and output:
            self.preview_result_callback(output, None)

    def open_final_output(self):
        output = str(self.project.get("assembled_output") or "")
        if callable(self.open_output_callback) and output:
            self.open_output_callback(output)

    # -------------------------------------------------------------- generation
    def validate_timeline(self):
        clips = self._clips()
        if not clips:
            return False, "Timeline has no clips."
        for i, clip in enumerate(clips):
            mode = str(clip.get("generation_mode") or "new")
            if i == 0 and mode == "continue":
                return False, "Clip 1 cannot continue a previous timeline clip."
            if mode in {"source", "loaded"}:
                if mode == "source" and i != 0:
                    return False, "Only Clip 1 can be a loaded start clip."
                source = str(clip.get("start_source_video") or clip.get("output") or "")
                if not source or not Path(source).is_file():
                    return False, f"{clip.get('name') or f'Clip {i + 1}'} is a loaded video clip, but its video file is missing."
                continue
            prompt = _compiled_prompt(clip)
            if not prompt:
                return False, f"{clip.get('name') or f'Clip {i + 1}'} has no prompt."
            if bool(clip.get("use_reference_images", False)):
                refs = _reference_entries(clip)
                if not refs:
                    return False, f"{clip.get('name') or f'Clip {i + 1}'} has reference mode enabled but no reference images."
                if len(refs) > TIMELINE_MAX_REF_IMAGES:
                    return False, f"{clip.get('name') or f'Clip {i + 1}'} has more than {TIMELINE_MAX_REF_IMAGES} reference images."
                missing = [r["path"] for r in refs if not Path(r["path"]).is_file()]
                if missing:
                    return False, f"{clip.get('name') or f'Clip {i + 1}'} has a missing reference image: {missing[0]}"
                if mode == "continue":
                    return False, f"{clip.get('name') or f'Clip {i + 1}'} uses reference images and cannot Continue Previous Clip."
            if int(clip.get("frames") or 0) not in self.frame_values:
                return False, f"{clip.get('name') or f'Clip {i + 1}'} has an invalid H3 frame count."
        return True, ""

    def generation_specs(self):
        specs = []
        clips = self._clips()
        hq_override = self.project.get("hq_restart_override") or {}
        if not isinstance(hq_override, dict):
            hq_override = {}
        for i, clip in enumerate(clips):
            spec = copy.deepcopy(clip)
            spec["timeline_index"] = i
            spec["compiled_prompt"] = _compiled_reference_prompt(clip)
            spec["timeline_reference_images"] = _reference_entries(clip)
            # HQ restart is a project-level generation mode. Once selected, keep
            # using the same override for later single-clip regenerations and
            # subsequent timeline generation, including after save/reload.
            if hq_override and str(clip.get("generation_mode") or "") not in {"source", "loaded"}:
                spec["timeline_hq_override"] = copy.deepcopy(hq_override)
            # Edit/replacement topology is used only by Regenerate selected block.
            spec["match_next_first_frame"] = False
            if i > 0:
                prev = clips[i - 1]
                spec["timeline_previous_output"] = str(prev.get("output") or "")
                spec["timeline_previous_job_id"] = str(prev.get("queue_job_id") or "")
                spec["timeline_previous_status"] = str(prev.get("status") or "")
                spec["timeline_previous_stale"] = False
                spec["timeline_previous_is_source"] = str(prev.get("generation_mode") or "") in {"source", "loaded"}
                # Preserve the predecessor's visible boundary metadata so queue-side
                # bridge helpers can resolve against the effective timeline clip,
                # not blindly against source frame 0 / the untrimmed source end.
                spec["timeline_previous_trim_in"] = prev.get("trim_in")
                spec["timeline_previous_trim_out"] = prev.get("trim_out")
            if i + 1 < len(clips):
                nxt = clips[i + 1]
                spec["timeline_next_output"] = str(nxt.get("output") or "")
                spec["timeline_next_status"] = str(nxt.get("status") or "")
                spec["timeline_next_stale"] = False
                # Anchor-next replacement mode must land on the visible start of
                # the next clip. When that next clip is trimmed, frame 0 of the
                # source file is no longer the correct destination.
                spec["timeline_next_trim_in"] = nxt.get("trim_in")
                spec["timeline_next_trim_out"] = nxt.get("trim_out")
            specs.append(spec)
        return specs

    def generate_selected(self):
        """Regenerate the selected block using its normal saved/current resolution."""
        return self._generate_selected_impl()

    def _generate_selected_impl(self, one_shot_resolution_override=None, low_res_test=False, alternate_candidate=False):
        """Shared single-block regeneration path.

        ``one_shot_resolution_override`` changes only the queued job; it does not
        mutate the block's saved settings or the project's HQ restart override.
        This keeps normal regeneration and low-res testing on exactly the same
        continuation/predecessor logic.
        """
        if not self._ensure_project_setup_for_first_edit():
            return False

        # Guard this synchronous preparation path. Whatever happens after the
        # button enters "Preparing regeneration…", the finally block below must
        # restore it. Validation warnings and queue-preparation failures used to
        # return early before cleanup, leaving Timeline permanently locked until
        # the app was restarted.
        if getattr(self, "_regenerate_selected_busy", False):
            return False
        self._regenerate_selected_busy = True
        old_button_text = self.generate_selected_btn.text() if hasattr(self, "generate_selected_btn") else "(Re)generate selected block"
        if hasattr(self, "generate_selected_btn"):
            self.generate_selected_btn.setEnabled(False)
            self.generate_selected_btn.setText("Preparing regeneration…")
        QApplication.processEvents()

        try:
            clip = self._selected_clip()
            idx = self._selected_index()
            if clip is None or idx < 0:
                return False
            if bool(clip.get("locked", False)):
                QMessageBox.information(self, "Timeline edit", "This block is locked. Unlock it before regenerating it.")
                return False

            if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
                QMessageBox.information(
                    self,
                    "Timeline edit",
                    "This is a loaded video clip, not an H3 generation. Choose another block to regenerate it.",
                )
                return False

            prompt = _compiled_prompt(clip)
            if not prompt:
                QMessageBox.warning(self, "Timeline not ready", "The selected clip has no prompt.")
                return False
            if int(clip.get("frames") or 0) not in self.frame_values:
                QMessageBox.warning(self, "Timeline not ready", "The selected clip has an invalid H3 frame count.")
                return False

            specs = self.generation_specs()
            spec = specs[idx]
            spec["timeline_single_regeneration"] = True
            edit_mode = str(
                clip.get("edit_mode")
                or ("continue_previous" if clip.get("generation_mode") == "continue" else "standalone")
            )
            use_previous = edit_mode in {"bridge_both", "continue_previous"}
            use_next = edit_mode in {"bridge_both", "anchor_next"}

            if bool(clip.get("use_reference_images", False)) and use_previous:
                QMessageBox.warning(
                    self,
                    "Timeline edit",
                    "Reference images use Ref2VA and cannot continue from the previous block. Choose a non-previous edit mode.",
                )
                return False

            spec["edit_mode"] = edit_mode
            spec["generation_mode"] = "continue" if use_previous else "new"
            spec["match_next_first_frame"] = bool(use_next)

            # One-shot render overrides (currently used by Create low res test)
            # must be applied *after* the normal single-clip continuation spec is
            # built.  That way the only difference from (Re)generate is resolution.
            if isinstance(one_shot_resolution_override, dict) and one_shot_resolution_override:
                spec["timeline_hq_override"] = copy.deepcopy(one_shot_resolution_override)
            if low_res_test:
                spec["timeline_low_res_test"] = True
            if alternate_candidate:
                spec["timeline_alternate_candidate"] = True
                spec["timeline_alternate_candidate_id"] = uuid.uuid4().hex

            if use_previous:
                if idx == 0:
                    QMessageBox.warning(
                        self,
                        "Timeline edit",
                        "This replacement mode needs a previous block, but the selected block is first.",
                    )
                    return False
                prev = self._clips()[idx - 1]
                prev_output = Path(str(prev.get("output") or ""))
                if (
                    str(prev.get("status") or "") != "finished"
                    or not prev_output.is_file()
                ):
                    QMessageBox.warning(
                        self,
                        "Timeline edit",
                        "The previous block must have a valid finished result before this replacement can continue from it.",
                    )
                    return False

            if use_next:
                if idx + 1 >= len(self._clips()):
                    QMessageBox.warning(
                        self,
                        "Timeline edit",
                        "This replacement mode needs a next block, but the selected block is last.",
                    )
                    return False
                nxt = self._clips()[idx + 1]
                next_output = Path(str(nxt.get("output") or ""))

                # A next block used only as an END ANCHOR does not need to be
                # current in the continuation chain. Deleting/recreating earlier
                # clips may legitimately mark it stale, but its saved first frame
                # is still a perfectly valid visual destination. Require only an
                # existing finished result file here.
                if str(nxt.get("status") or "") != "finished" or not next_output.is_file():
                    QMessageBox.warning(
                        self,
                        "Timeline edit",
                        "The next block must have a finished result file before its first frame can anchor this replacement.",
                    )
                    return False

            if not callable(self.queue_timeline_callback):
                QMessageBox.warning(self, "Timeline", "The timeline is not connected to the MiniMax queue.")
                return False

            self._record_undo_state(
                "Regenerate as alternate" if alternate_candidate else
                ("Create low res test" if low_res_test else "Regenerate selected clip")
            )
            try:
                result = self.queue_timeline_callback([spec])
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Timeline regeneration failed",
                    f"Could not prepare the selected block for regeneration:\n\n{exc}",
                )
                return False

            if result:
                if not alternate_candidate:
                    if low_res_test:
                        # This queued render intentionally overrides HQ only for this
                        # job.  Do not change the project's persistent HQ mode.
                        clip["hq_generated"] = False
                    else:
                        active_hq = self.project.get("hq_restart_override") or {}
                        if isinstance(active_hq, dict) and active_hq:
                            clip["hq_generated"] = True
                    self._invalidate_assembly(
                        "Low-res test queued — assemble again when ready."
                        if low_res_test else
                        "Selected clip regenerated — assemble again when ready."
                    )

                # A free-ending continuation changes the boundary consumed by the
                # next continuation clip, so downstream continuation results really
                # do become stale.
                if not alternate_candidate and edit_mode == "continue_previous":
                    # Existing downstream renders remain usable; do not invalidate them.
                    for j in range(idx + 1, len(self._clips())):
                        if self._clips()[j].get("generation_mode") != "continue":
                            break
                        self._clips()[j]["stale"] = False

                # bridge_both / anchor_next are specifically chosen to PRESERVE the
                # existing next clip by targeting its exact first frame. If that
                # next clip was marked stale only because an earlier clip was
                # deleted/recreated, keeping that stale flag defeats the whole point
                # of the anchored replacement and makes Assemble Video demand an
                # unnecessary re-render.
                #
                # The next clip's output is not changed here. Once it is accepted as
                # the destination anchor, its outgoing boundary is also unchanged, so
                # an existing contiguous finished continuation chain after it remains
                # valid as well. Clear only clips that still have real finished files.
                elif edit_mode in {"bridge_both", "anchor_next"}:
                    # Do not decide this at queue time. The replacement still
                    # has to render successfully. sync_queue_jobs() will clear
                    # the preserved chain when this clip reaches "finished".
                    clip["_preserve_downstream_on_finish"] = True

                self._refresh_all()
            else:
                QMessageBox.warning(
                    self,
                    "Timeline regeneration did not queue",
                    "The selected block was not added to the queue. Check the MiniMax log for the validation message; no duplicate regeneration request was started.",
                )
            return bool(result)

        finally:
            self._regenerate_selected_busy = False
            if hasattr(self, "generate_selected_btn"):
                self.generate_selected_btn.setText(old_button_text)
                self._refresh_edit_workflow(self._selected_clip())

    def _clip_has_usable_output(self, clip: dict) -> bool:
        """Return True only when this block already has a video that can be reused."""
        if str(clip.get("generation_mode") or "") in {"source", "loaded"}:
            source = str(clip.get("start_source_video") or clip.get("output") or "")
            return bool(source and Path(source).is_file())
        output = str(clip.get("output") or "")
        return bool(output and Path(output).is_file())

    def _validate_generation_indices(self, indices: list[int]):
        """Validate only blocks that are part of this Generate Timeline request.

        This deliberately does not validate unrelated earlier blocks for
        'start from selected'. A selected continuation block is allowed to run
        without its previous block; _generation_specs_for_indices() detaches the
        first queued block when no usable predecessor exists.
        """
        clips = self._clips()
        if not clips:
            return False, "Timeline has no clips."
        if not indices:
            return False, "There are no clips to generate for this selection."

        for i in indices:
            if i < 0 or i >= len(clips):
                continue
            clip = clips[i]
            mode = str(clip.get("generation_mode") or "new")
            name = str(clip.get("name") or f"Clip {i + 1}")
            if mode in {"source", "loaded"}:
                source = str(clip.get("start_source_video") or clip.get("output") or "")
                if not source or not Path(source).is_file():
                    return False, f"{name} is a loaded video clip, but its video file is missing."
                continue
            prompt = _compiled_prompt(clip)
            if not prompt:
                return False, f"{name} has no prompt."
            if bool(clip.get("use_reference_images", False)):
                refs = _reference_entries(clip)
                if not refs:
                    return False, f"{name} has reference mode enabled but no reference images."
                if len(refs) > TIMELINE_MAX_REF_IMAGES:
                    return False, f"{name} has more than {TIMELINE_MAX_REF_IMAGES} reference images."
                missing = [r["path"] for r in refs if not Path(r["path"]).is_file()]
                if missing:
                    return False, f"{name} has a missing reference image: {missing[0]}"
                if mode == "continue":
                    return False, f"{name} uses reference images and cannot Continue Previous Clip."
            if int(clip.get("frames") or 0) not in self.frame_values:
                return False, f"{name} has an invalid H3 frame count."
        return True, ""

    def _generation_specs_for_indices(self, indices: list[int]) -> list[dict]:
        """Build a queue batch for an arbitrary subset of Timeline blocks.

        Original timeline_index values are preserved so queue/result mapping
        still targets the correct blocks. If the first selected continuation
        block has no usable predecessor outside the batch, generate it as a
        standalone start for this run instead of refusing to queue the batch.
        """
        all_specs = self.generation_specs()
        wanted = {int(i) for i in indices}
        specs = [copy.deepcopy(all_specs[i]) for i in indices if 0 <= i < len(all_specs)]
        clips = self._clips()

        for spec in specs:
            i = int(spec.get("timeline_index", -1))
            if i <= 0 or str(spec.get("generation_mode") or "") != "continue":
                continue
            prev_index = i - 1
            # If the predecessor is also part of this queue batch, normal queue
            # chaining can connect the jobs in sequence. If it is not in this
            # batch, only keep Continue mode when a real previous video exists.
            if prev_index in wanted:
                continue
            previous = clips[prev_index] if 0 <= prev_index < len(clips) else None
            if previous is not None and self._clip_has_usable_output(previous):
                continue
            spec["generation_mode"] = "new"
            spec["edit_mode"] = "standalone"
            spec["timeline_previous_output"] = ""
            spec["timeline_previous_job_id"] = ""
            spec["timeline_previous_status"] = ""
            spec["timeline_previous_stale"] = False
            spec["timeline_previous_is_source"] = False
            spec["timeline_detached_start"] = True
        return specs

    def generate_timeline(self, mode="all"):
        if not self._ensure_project_setup_for_first_edit():
            return False
        if not callable(self.queue_timeline_callback):
            QMessageBox.warning(self, "Timeline", "The timeline is not connected to the MiniMax queue.")
            return False

        clips = self._clips()
        if mode == "missing":
            indices = [i for i, clip in enumerate(clips) if not bool(clip.get("locked", False)) and not self._clip_has_usable_output(clip)]
            if not indices:
                if any(bool(c.get("locked", False)) and not self._clip_has_usable_output(c) for c in clips):
                    QMessageBox.information(self, "Generate Timeline", "There are no unlocked missing clips to generate. Locked missing clips were left untouched.")
                else:
                    QMessageBox.information(self, "Generate Timeline", "All timeline clips already have usable output files.")
                return False
            undo_label = "Generate missing timeline clips"
        elif mode == "selected_only":
            selected_indices = self._selected_indices()
            if not selected_indices:
                QMessageBox.information(self, "Generate Timeline", "Select one or more timeline blocks first.")
                return False
            indices = [i for i in selected_indices if not bool(clips[i].get("locked", False))]
            if not indices:
                QMessageBox.information(self, "Generate Timeline", "All selected clips are locked. Unlock at least one selected clip to generate it.")
                return False
            undo_label = "Generate selected timeline clips"
        elif mode == "selected_onward":
            selected = self._selected_index()
            if selected < 0:
                QMessageBox.information(self, "Generate Timeline", "Select a timeline block first.")
                return False
            indices = [i for i in range(selected, len(clips)) if not bool(clips[i].get("locked", False))]
            undo_label = "Generate timeline from selected block"
        else:
            indices = [i for i, clip in enumerate(clips) if not bool(clip.get("locked", False))]
            undo_label = "Generate all timeline clips"

        if not indices:
            QMessageBox.information(self, "Generate Timeline", "There are no unlocked clips to generate for this selection.")
            return False
        ok, error = self._validate_generation_indices(indices)
        if not ok:
            QMessageBox.warning(self, "Timeline not ready", error)
            return False

        specs = self._generation_specs_for_indices(indices)
        if not specs:
            QMessageBox.information(self, "Generate Timeline", "There are no clips to generate for this selection.")
            return False

        self._record_undo_state(undo_label)
        result = self.queue_timeline_callback(specs)
        if result:
            active_hq = self.project.get("hq_restart_override") or {}
            if isinstance(active_hq, dict) and active_hq:
                for i in indices:
                    if 0 <= i < len(clips) and str(clips[i].get("generation_mode") or "") not in {"source", "loaded"}:
                        clips[i]["hq_generated"] = True
            self.project["assembled_output"] = ""
            self.project["assembly_status"] = ""
            self.project["auto_assemble"] = False
            self.project["auto_assemble_pending"] = False
            self._refresh_all()
        return bool(result)

    def hq_restart(self):
        """Requeue the complete timeline with one HQ resolution override.

        Every other generation setting remains owned by each timeline clip. The
        orientation is taken from the live MiniMax GUI for normal HQ presets so a
        9:16 or 1:1 project stays in that orientation. Explicit 21:9 choices force
        21:9 because those entries already describe their complete output shape.
        """
        if not self._ensure_project_setup_for_first_edit():
            return False
        unlocked_indices = [i for i, clip in enumerate(self._clips()) if not bool(clip.get("locked", False))]
        if not unlocked_indices:
            QMessageBox.information(self, "HQ restart", "Every timeline block is locked. Unlock the blocks you want to recreate in HQ.")
            return False
        ok, error = self._validate_generation_indices(unlocked_indices)
        if not ok:
            QMessageBox.warning(self, "Timeline not ready", error)
            return False
        if not callable(self.queue_timeline_callback):
            QMessageBox.warning(self, "Timeline", "The timeline is not connected to the MiniMax queue.")
            return False

        choices = [
            "1280x704",
            "1344x768",
            "1920x1080",
            "1344x576 (21:9)",
            "1792x768 (21:9)",
        ]
        choice, accepted = QInputDialog.getItem(
            self,
            "HQ restart",
            "Select HQ resolution:",
            choices,
            0,
            False,
        )
        if not accepted:
            return False

        current = self._capture_settings()
        current_aspect = str(current.get("aspect") or "16:9")
        if current_aspect not in {"9:16", "1:1"}:
            current_aspect = "16:9"

        if choice == "1280x704":
            override = {"aspect": current_aspect, "resolution": "1280 × 720"}
        elif choice == "1344x768":
            override = {"aspect": current_aspect, "resolution": "1344 × 768"}
        elif choice == "1920x1080":
            # The GUI's 1080p choice intentionally maps to MiniMax's valid
            # 1920x1088 tensor size, exactly like the normal Generation tab.
            override = {"aspect": current_aspect, "resolution": "1920 × 1088"}
        elif choice == "1344x576 (21:9)":
            override = {"aspect": "21:9", "widescreen_quality": "Medium — 1344 × 576"}
        else:
            override = {"aspect": "21:9", "widescreen_quality": "High — 1792 × 768"}

        specs = self._generation_specs_for_indices(unlocked_indices)
        for spec in specs:
            spec["timeline_hq_override"] = copy.deepcopy(override)

        self._record_undo_state("HQ restart")
        result = self.queue_timeline_callback(specs)
        if result:
            # Persist the chosen HQ mode at project level. This is saved in the
            # timeline JSON and is automatically re-applied by generation_specs()
            # for later single-clip regeneration after an app restart.
            self.project["hq_restart_override"] = copy.deepcopy(override)
            for i in unlocked_indices:
                clip = self._clips()[i]
                if str(clip.get("generation_mode") or "") not in {"source", "loaded"}:
                    clip["hq_generated"] = True
            self.project["assembled_output"] = ""
            self.project["assembly_status"] = ""
            self.project["auto_assemble"] = False
            self.project["auto_assemble_pending"] = False
            self._refresh_all()
        return bool(result)

    def mark_queued(self, clip_id, job_id, output="", job_info=None):
        is_alternate = bool(isinstance(job_info, dict) and job_info.get("timeline_alternate_candidate"))
        alternate_id = str((job_info or {}).get("timeline_alternate_candidate_id") or uuid.uuid4().hex) if is_alternate else ""
        for clip in self._clips():
            if str(clip.get("id")) == str(clip_id):
                info = self._generation_info_from_job(job_info)
                if is_alternate:
                    candidates = clip.setdefault("alternate_candidates", [])
                    candidates.append({
                        "candidate_id": alternate_id,
                        "queue_job_id": str(job_id),
                        "status": "pending",
                        "output": str(output or ""),
                        "last_generation_info": info,
                        "hq_generated": bool((job_info or {}).get("timeline_hq_override")),
                        "stale": False,
                    })
                else:
                    # Normal regeneration used to overwrite the active attempt and
                    # leave Undo as the only route back. Preserve the currently
                    # playable render as an alternate *before* replacing its queue
                    # metadata/output. This applies to ordinary retries as well as
                    # batch/HQ/low-res recreations.
                    self._archive_active_candidate_before_replacement(clip, str(output or ""))
                    clip["queue_job_id"] = str(job_id)
                    clip["status"] = "pending"
                    clip["stale"] = False
                    clip["output"] = str(output or "")
                    # A trim belongs to the previous rendered version. It was copied
                    # into the archived candidate above; the new render starts with a
                    # clean full-range assembly state.
                    clip.pop("trim_in", None); clip.pop("trim_out", None)
                    if info:
                        clip["last_generation_info"] = info
                break
        self._refresh_all()

    def sync_queue_jobs(self, jobs):
        by_id = {str(j.get("id")): j for j in (jobs or []) if j.get("id")}
        changed = False

        # Keep queued Timeline assembly state synchronized with the normal queue,
        # including after queue reload/recovery. Assembly is not a clip-generation
        # dependency; this only drives the button/status/result path.
        assembly_job_id = str(self.project.get("assembly_queue_job_id") or "")
        if assembly_job_id and assembly_job_id in by_id:
            assembly_job = by_id[assembly_job_id]
            assembly_state = str(assembly_job.get("state") or "pending")
            assembly_output = str(assembly_job.get("output") or "")
            if assembly_output and self.project.get("assembled_output") != assembly_output:
                self.project["assembled_output"] = assembly_output; changed = True
            if assembly_state in {"pending", "interrupted"}:
                wanted = "Queued for assembly"
                self._timeline_assemble_click_busy = True
                if self.project.get("assembly_status") != wanted:
                    self.project["assembly_status"] = wanted; changed = True
            elif assembly_state == "running":
                wanted = "Assembling…"
                self._timeline_assemble_click_busy = True
                if self.project.get("assembly_status") != wanted:
                    self.project["assembly_status"] = wanted; changed = True
            elif assembly_state == "finished":
                self._timeline_assemble_click_busy = False
                if self.project.get("assembly_status") != "Finished":
                    self.project["assembly_status"] = "Finished"; changed = True
            elif assembly_state in {"failed", "cancelled", "canceled"}:
                self._timeline_assemble_click_busy = False
                wanted = "Assembly failed: " + str(assembly_job.get("error") or assembly_job.get("cancel_reason") or assembly_state)
                if self.project.get("assembly_status") != wanted:
                    self.project["assembly_status"] = wanted; changed = True

        clips = self._clips()
        for idx, clip in enumerate(clips):
            job_id = str(clip.get("queue_job_id") or "")
            if job_id and job_id in by_id:
                job = by_id[job_id]
                state = str(job.get("state") or "pending")
                previous_state = str(clip.get("status") or "")
                if previous_state != state:
                    clip["status"] = state; changed = True
                output = str(job.get("output") or "")
                if output and clip.get("output") != output:
                    clip["output"] = output; changed = True
                job_info = self._generation_info_from_job(job)
                if job_info and clip.get("last_generation_info") != job_info:
                    clip["last_generation_info"] = job_info; changed = True

                if state == "finished" and bool(clip.get("_preserve_downstream_on_finish")):
                    clip.pop("_preserve_downstream_on_finish", None)
                    cleared = self._accept_preserved_downstream_chain(idx)
                    if cleared:
                        print(
                            f"[TIMELINE] Anchored replacement finished; preserved {cleared} existing downstream clip(s).",
                            flush=True,
                        )
                        changed = True
                elif state in {"failed", "cancelled", "canceled"}:
                    clip.pop("_preserve_downstream_on_finish", None)

            # Alternate candidates have their own queue jobs and must never
            # overwrite the active render while they are generating.
            candidates = clip.get("alternate_candidates") if isinstance(clip.get("alternate_candidates"), list) else []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                alt_job_id = str(candidate.get("queue_job_id") or "")
                if not alt_job_id or alt_job_id not in by_id:
                    continue
                alt_job = by_id[alt_job_id]
                alt_state = str(alt_job.get("state") or "pending")
                if str(candidate.get("status") or "") != alt_state:
                    candidate["status"] = alt_state; changed = True
                alt_output = str(alt_job.get("output") or "")
                if alt_output and str(candidate.get("output") or "") != alt_output:
                    candidate["output"] = alt_output; changed = True
                alt_info = self._generation_info_from_job(alt_job)
                if alt_info and candidate.get("last_generation_info") != alt_info:
                    candidate["last_generation_info"] = alt_info; changed = True

        if changed:
            self._refresh_all()
