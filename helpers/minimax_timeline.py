from __future__ import annotations

import copy
import json
import math
import re
import subprocess
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFontMetrics, QPixmap
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
)

FPS = 24.0
MIN_SEGMENT_SECONDS = 0.3
# Backward-compatible internal alias; project JSON still uses the existing "segments" key.
MIN_CUT_SECONDS = MIN_SEGMENT_SECONDS
TIMELINE_SCHEMA_VERSION = 1


def _safe_project_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(text or "").strip())
    return text or "MiniMax Timeline"


def _clip_seconds(clip: dict) -> float:
    if str(clip.get("generation_mode") or "") == "source":
        try:
            duration = float(clip.get("source_duration_seconds") or 0.0)
            if duration > 0:
                return duration
        except Exception:
            pass
    return max(1, int(clip.get("frames") or 124)) / FPS


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
        entries.append({"path": path, "name": str(item.get("name") or Path(path).stem).strip()})
    return entries[:5]


def _compiled_reference_prompt(clip: dict) -> str:
    """Return the H3 prompt with stable Subject->Picture definitions for Ref2VA.

    MiniMax receives reference images as <Picture N> in input order.  The user-facing
    timeline uses <Subject N> for reusable visible content, matching MiniMax's official
    full-reference prompt guide.
    """
    body = _compiled_prompt(clip)
    if not bool(clip.get("use_reference_images", False)):
        return body
    refs = _reference_entries(clip)
    if not refs:
        return body
    defs = []
    for idx, ref in enumerate(refs, 1):
        friendly = str(ref.get("name") or f"Reference {idx}").strip() or f"Reference {idx}"
        defs.append(f'<Subject {idx}> is the reusable visible content named "{friendly}" from <Picture {idx}>.')
    return "\n".join(defs + ([body] if body else [])).strip()


class TimelineCanvas(QWidget):
    """Compact NLE-like overview of MiniMax H3 generation jobs."""

    clipSelected = Signal(str)
    clipsReordered = Signal(int, int)
    clipFramesChanged = Signal(str, int)

    RULER_H = 30
    CLIP_TOP = 40
    CLIP_H = 116
    BOTTOM_PAD = 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clips: list[dict] = []
        self.selected_id: str | None = None
        self.pixels_per_second = 38.0
        self.allowed_frames: list[int] = list(range(124, 720, 17))
        self._rects: list[tuple[str, float, float]] = []
        self._press_x = 0.0
        self._press_clip_index = -1
        self._dragging = False
        self._resizing = False
        self.setMinimumHeight(self.CLIP_TOP + self.CLIP_H + self.BOTTOM_PAD)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QSize(max(900, self._content_width()), self.CLIP_TOP + self.CLIP_H + self.BOTTOM_PAD)

    def _content_width(self) -> int:
        total = sum(_clip_seconds(c) for c in self.clips)
        return int(max(900, 36 + total * self.pixels_per_second + 36))

    def set_clips(self, clips: list[dict], selected_id: str | None = None):
        self.clips = clips
        if selected_id is not None:
            self.selected_id = selected_id
        self.setMinimumWidth(self._content_width())
        self.resize(self._content_width(), self.minimumHeight())
        self.update()

    def set_zoom(self, pixels_per_second: float):
        self.pixels_per_second = max(16.0, min(120.0, float(pixels_per_second)))
        self.setMinimumWidth(self._content_width())
        self.resize(self._content_width(), self.minimumHeight())
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

        total_seconds = max(1.0, sum(_clip_seconds(c) for c in self.clips))
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
        fm = QFontMetrics(painter.font())
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
            duration = _clip_seconds(clip)
            width = max(74.0, duration * self.pixels_per_second)
            left, right = cursor, cursor + width
            self._rects.append((str(clip.get("id")), left, right))
            selected = str(clip.get("id")) == self.selected_id
            status = "stale" if clip.get("stale") else str(clip.get("status") or "draft")
            fill = state_colors.get(status, panel)
            painter.setBrush(QBrush(fill))
            painter.setPen(QPen(accent if selected else border, 2 if selected else 1))
            painter.drawRoundedRect(int(left), self.CLIP_TOP, int(width - 4), self.CLIP_H, 5, 5)

            # Clip header.
            painter.setPen(accent_text if selected else text)
            name = str(clip.get("name") or f"Clip {idx + 1}")
            name_width = int(max(20, width - 18))
            elided = fm.elidedText(name, Qt.TextElideMode.ElideRight, name_width)
            painter.drawText(int(left + 8), self.CLIP_TOP + 18, elided)
            if clip.get("generation_mode") == "source":
                mode = "▣ Loaded start clip"
            else:
                mode = "↪ Continue" if clip.get("generation_mode") == "continue" else "◆ New"
            edit_mode = str(clip.get("edit_mode") or "")
            bridge = "  •  ⇥ Next frame" if edit_mode in {"bridge_both", "anchor_next"} else ""
            ref_count = len(_reference_entries(clip)) if bool(clip.get("use_reference_images", False)) else 0
            refs_info = f"  •  Ref2VA ×{ref_count}" if ref_count else ""
            info = f"{mode}  •  {int(clip.get('frames') or 0)}f  •  {duration:.2f}s{bridge}{refs_info}"
            painter.setPen(muted if not selected else accent_text)
            painter.drawText(int(left + 8), self.CLIP_TOP + 38, fm.elidedText(info, Qt.TextElideMode.ElideRight, name_width))

            # Show a compact prompt preview. A timeline block is one complete H3 job;
            # shot/timestamp structure stays inside the normal H3 prompt itself.
            prompt_y = self.CLIP_TOP + 53
            prompt_h = 42
            painter.setBrush(QBrush(QColor("#375a7f")))
            painter.setPen(QPen(bg, 1))
            painter.drawRect(int(left + 5), prompt_y, max(1, int(width - 14)), prompt_h)
            if clip.get("generation_mode") == "source":
                source = str(clip.get("start_source_video") or clip.get("output") or "").strip()
                prompt = Path(source).name if source else "No start video loaded"
            else:
                prompt = _compiled_prompt(clip).replace("\n", " ").strip() or "Empty prompt"
            painter.setPen(QColor("#f4f6f8"))
            painter.drawText(
                int(left + 10), prompt_y + 26,
                fm.elidedText(prompt, Qt.TextElideMode.ElideRight, int(max(12, width - 24)))
            )

            if clip.get("generation_mode") == "continue" and idx > 0:
                painter.setPen(QPen(accent, 2))
                painter.drawLine(int(left - 12), self.CLIP_TOP + 20, int(left - 3), self.CLIP_TOP + 20)
                painter.drawLine(int(left - 7), self.CLIP_TOP + 16, int(left - 3), self.CLIP_TOP + 20)
                painter.drawLine(int(left - 7), self.CLIP_TOP + 24, int(left - 3), self.CLIP_TOP + 20)

            cursor = right

        if not self.clips:
            painter.setPen(muted)
            painter.drawText(28, self.CLIP_TOP + 50, "No clips yet — add a generation clip to start the timeline.")

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        x = event.position().x()
        idx, left, right = self._clip_at(x)
        if idx < 0:
            return
        self._press_x = x
        self._press_clip_index = idx
        self._dragging = False
        self._resizing = (right - x) <= 10 and len(self.clips) > 0 and str(self.clips[idx].get("generation_mode") or "") != "source"
        clip_id = str(self.clips[idx].get("id"))
        self.selected_id = clip_id
        self.clipSelected.emit(clip_id)
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
        self.selected_cut_index = 0
        self._loading_inspector = False
        self._project_path: Path | None = None
        self._build_ui()
        self._ensure_initial_clip()
        self._refresh_all(select_first=True)

        # Autosave timeline projects once per minute. Before the user chooses a
        # project file, keep a quiet recovery copy next to this module; after a
        # normal Save/Load establishes a project path, autosave writes there.
        self._autosave_temp_path = Path(__file__).resolve().parent / "minimax_timeline_autosave_temp.json"
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
            "clips": [],
            "assembled_output": "",
            "assembly_status": "",
            "auto_assemble": True,
            "auto_assemble_pending": False,
        }

    def _capture_settings(self):
        if callable(self.settings_provider):
            try:
                data = self.settings_provider()
                return copy.deepcopy(data) if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    def _blank_clip(self, *, continue_previous=False, source_settings=None):
        settings = copy.deepcopy(source_settings if isinstance(source_settings, dict) else self._capture_settings())
        frames = int(settings.get("frames") or 243)
        frames = min(self.frame_values, key=lambda v: abs(v - frames)) if self.frame_values else frames
        prompt = str(settings.get("prompt") or "")
        # Timeline reference images are owned by each clip, not implicitly copied
        # from whatever happens to be loaded on the Generation tab.
        settings["ref_images"] = []
        settings["ref_videos"] = []
        settings["ref_audios"] = []
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
        }

    def _clips(self):
        return self.project.setdefault("clips", [])

    def _ensure_initial_clip(self):
        if not self._clips():
            clip = self._blank_clip(continue_previous=False)
            clip["name"] = "Clip 1"
            self._clips().append(clip)
            self.selected_clip_id = clip["id"]

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
        clips = self._clips()
        if not (0 <= index < len(clips)):
            return
        clip = clips[index]
        self._invalidate_assembly()
        if clip.get("queue_job_id") or clip.get("status") in {"pending", "running", "finished"}:
            clip["stale"] = True
        # Edit topology determines whether the already-rendered next block is
        # intentionally preserved. Only "continue previous -> free ending" creates
        # a new outgoing boundary that invalidates an existing continuation chain.
        edit_mode = str(clip.get("edit_mode") or ("continue_previous" if clip.get("generation_mode") == "continue" else "standalone"))
        preserve_next = edit_mode in {"bridge_both", "anchor_next", "standalone"}
        if propagate and not preserve_next:
            for j in range(index + 1, len(clips)):
                if clips[j].get("generation_mode") != "continue":
                    break
                if clips[j].get("queue_job_id") or clips[j].get("status") in {"pending", "running", "finished"}:
                    clips[j]["stale"] = True

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
        settings["ref_images"] = []
        settings["ref_videos"] = []
        settings["ref_audios"] = []
        clip["settings"] = settings
        frames = int(settings.get("frames") or clip.get("frames") or 243)
        if self.frame_values:
            frames = min(self.frame_values, key=lambda v: abs(v - frames))
        clip["frames"] = frames
        if len(clip.get("segments") or []) == 1 and not str(clip["segments"][0].get("prompt") or "").strip():
            clip["segments"][0]["prompt"] = str(settings.get("prompt") or "")
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
        self.add_clip_btn = QPushButton("+ Clip")
        self.dup_clip_btn = QPushButton("Duplicate")
        self.del_clip_btn = QPushButton("Delete")
        self.left_btn = QPushButton("◀")
        self.right_btn = QPushButton("▶")
        for w in (self.new_btn, self.save_btn, self.load_btn, self.add_clip_btn, self.dup_clip_btn, self.del_clip_btn, self.left_btn, self.right_btn):
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
            "Queue every clip on this timeline as a separate MiniMax H3 generation in timeline order."
        )
        self.assemble_timeline_btn = QPushButton("Assemble Video")
        self.assemble_timeline_btn.setMinimumHeight(36)
        self.assemble_timeline_btn.setToolTip(
            "Join the finished timeline clip outputs, in timeline order, into one final MP4."
        )
        self.auto_assemble_check = QCheckBox("Auto assemble when finished")
        self.auto_assemble_check.setChecked(bool(self.project.get("auto_assemble", True)))
        self.auto_assemble_check.setToolTip(
            "After Generate Timeline, automatically assemble the final MP4 when every timeline clip has finished."
        )
        self.preview_final_btn = QPushButton("Preview final")
        self.open_final_btn = QPushButton("Open final folder")
        self.preview_final_btn.setEnabled(False)
        self.open_final_btn.setEnabled(False)
        runbar.addWidget(self.generate_timeline_btn)
        runbar.addWidget(self.assemble_timeline_btn)
        runbar.addWidget(self.auto_assemble_check)
        runbar.addWidget(self.preview_final_btn)
        runbar.addWidget(self.open_final_btn)
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
        ev.addStretch(1)

        self.edit_scroll = QScrollArea()
        self.edit_scroll.setWidgetResizable(True)
        self.edit_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.edit_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.edit_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # Roughly three to four text/control rows are visible at once; remaining
        # edit choices are reached with the vertical scrollbar.
        self.edit_scroll.setFixedHeight(118)
        self.edit_scroll.setWidget(edit_scroll_contents)
        edit_outer.addWidget(self.edit_scroll, 1)

        self.generate_selected_btn = QPushButton("Regenerate selected block")
        self.generate_selected_btn.setMinimumHeight(36)
        self.generate_selected_btn.setToolTip("Regenerate only the selected block using the replacement continuity mode selected above.")
        edit_outer.addWidget(self.generate_selected_btn, 0)

        edit_box.setMaximumHeight(190)
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
        self.add_refs_btn = QPushButton("+ Add reference images")
        self.add_refs_btn.setToolTip("Add up to 5 reference images for this timeline clip.")
        self.refs_rows_widget = QWidget()
        self.refs_rows_layout = QVBoxLayout(self.refs_rows_widget)
        self.refs_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.refs_rows_layout.setSpacing(6)
        rv.addWidget(self.use_refs_check)
        rv.addWidget(self.use_refs_warning)
        rv.addWidget(self.add_refs_btn)
        rv.addWidget(self.refs_rows_widget)
        iv.addWidget(self.references_box)

        self.settings_box = QGroupBox("MiniMax settings")
        settings_box = self.settings_box
        sf = QFormLayout(settings_box)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(-1, 2147483647)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1, 100)
        self.scheduler_combo = QComboBox(); self.scheduler_combo.addItems(["simple", "beta"])
        self.glue_check = QCheckBox("Glue result to source")
        self.audio_memory_check = QCheckBox("Carry audio memory")
        self.latent_check = QCheckBox("Latent continuation")
        self.capture_btn = QPushButton("Capture current Generation-tab settings")
        self.model_label = QLabel("—"); self.model_label.setWordWrap(True)
        self.refs_label = QLabel("—"); self.refs_label.setWordWrap(True)
        self.loras_label = QLabel("—"); self.loras_label.setWordWrap(True)
        sf.addRow("", self.capture_btn)
        sf.addRow("Seed", self.seed_spin)
        sf.addRow("Steps", self.steps_spin)
        sf.addRow("Scheduler", self.scheduler_combo)
        sf.addRow("", self.glue_check)
        sf.addRow("", self.audio_memory_check)
        sf.addRow("", self.latent_check)
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
        self.generate_timeline_btn.clicked.connect(self.generate_timeline)
        self.generate_selected_btn.clicked.connect(self.generate_selected)
        self.assemble_timeline_btn.clicked.connect(self.assemble_timeline)
        self.auto_assemble_check.toggled.connect(self._auto_assemble_changed)
        self.preview_clip_btn.clicked.connect(self.preview_selected_result)
        self.open_clip_btn.clicked.connect(self.open_selected_output)
        self.preview_final_btn.clicked.connect(self.preview_final_result)
        self.open_final_btn.clicked.connect(self.open_final_output)
        self.add_clip_btn.clicked.connect(self.add_clip)
        self.dup_clip_btn.clicked.connect(self.duplicate_clip)
        self.del_clip_btn.clicked.connect(self.delete_clip)
        self.left_btn.clicked.connect(lambda: self.move_clip(-1))
        self.right_btn.clicked.connect(lambda: self.move_clip(1))
        self.zoom_slider.valueChanged.connect(self.canvas.set_zoom)
        self.canvas.clipSelected.connect(self.select_clip)
        self.canvas.clipsReordered.connect(self._reorder_clips)
        self.canvas.clipFramesChanged.connect(self._canvas_frames_changed)
        self.clip_name.textEdited.connect(self._clip_name_changed)
        self.gen_mode.currentIndexChanged.connect(self._mode_changed)
        self.frames_combo.currentIndexChanged.connect(self._frames_changed)
        self.seed_spin.valueChanged.connect(self._settings_changed)
        self.steps_spin.valueChanged.connect(self._settings_changed)
        self.scheduler_combo.currentTextChanged.connect(self._settings_changed)
        self.glue_check.toggled.connect(self._settings_changed)
        self.audio_memory_check.toggled.connect(self._settings_changed)
        self.latent_check.toggled.connect(self._settings_changed)
        self.capture_btn.clicked.connect(self.capture_current_settings)
        self.edit_mode_group.buttonClicked.connect(self._edit_mode_changed)
        self.cut_prompt.textChanged.connect(self._cut_prompt_changed)
        self.use_refs_check.toggled.connect(self._reference_mode_changed)
        self.add_refs_btn.clicked.connect(self._add_reference_images)

    # -------------------------------------------------------------- refreshers
    def _refresh_all(self, *, select_first=False):
        if select_first and self._clips() and not self.selected_clip_id:
            self.selected_clip_id = self._clips()[0]["id"]
        if self.selected_clip_id and not any(c.get("id") == self.selected_clip_id for c in self._clips()):
            self.selected_clip_id = self._clips()[0]["id"] if self._clips() else None
        total = sum(_clip_seconds(c) for c in self._clips())
        self.summary_label.setText(f"{len(self._clips())} clips  •  {total:.2f}s  •  {round(total * FPS)} timeline frames")
        self.generate_timeline_btn.setEnabled(bool(self._clips()))
        ready, reason = self._assembly_ready()
        self.assemble_timeline_btn.setEnabled(bool(self._clips()))
        self.assemble_timeline_btn.setToolTip(
            "Join the finished timeline clip outputs, in timeline order, into one final MP4."
            if ready else reason
        )
        self.auto_assemble_check.blockSignals(True)
        self.auto_assemble_check.setChecked(bool(self.project.get("auto_assemble", True)))
        self.auto_assemble_check.blockSignals(False)
        final_path = str(self.project.get("assembled_output") or "")
        final_exists = bool(final_path and Path(final_path).is_file())
        self.preview_final_btn.setEnabled(final_exists)
        self.open_final_btn.setEnabled(final_exists)
        final_status = str(self.project.get("assembly_status") or "").strip()
        self.preview_final_btn.setToolTip(final_path if final_exists else (final_status or "No assembled timeline result yet."))
        self.open_final_btn.setToolTip(final_path if final_exists else (final_status or "No assembled timeline result yet."))
        self.summary_label.setToolTip(str(self._project_path) if self._project_path else "Autosaving to temporary timeline JSON until you choose Save.")
        self.canvas.set_clips(self._clips(), self.selected_clip_id)
        self._load_inspector()

    def _load_inspector(self):
        clip = self._selected_clip()
        self._loading_inspector = True
        try:
            enabled = clip is not None
            for w in (self.clip_name, self.gen_mode, self.frames_combo, self.seed_spin, self.steps_spin,
                      self.scheduler_combo, self.glue_check, self.audio_memory_check, self.latent_check,
                      self.capture_btn, self.cut_prompt):
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
                self.settings_box.setEnabled(False)
                self.preview_clip_btn.setEnabled(False); self.open_clip_btn.setEnabled(False)
                self._refresh_edit_workflow(None)
                self.cut_prompt.clear()
                return
            settings = clip.setdefault("settings", {})
            self.clip_name.setText(str(clip.get("name") or ""))
            selected_index = self._selected_index()
            self._populate_generation_modes(selected_index, str(clip.get("generation_mode") or "new"))
            is_source = selected_index == 0 and str(clip.get("generation_mode") or "") == "source"
            source_path = str(clip.get("start_source_video") or clip.get("output") or "")
            self.start_source_label.setText(source_path if is_source and source_path else "—")
            self.start_source_label.setVisible(is_source)
            if hasattr(self.clip_form, "setRowVisible"):
                self.clip_form.setRowVisible(self.start_source_label, is_source)
            self.prompt_box.setEnabled(enabled and not is_source)
            self.references_box.setEnabled(enabled and not is_source)
            self.settings_box.setEnabled(enabled and not is_source)
            self.use_refs_check.blockSignals(True)
            self.use_refs_check.setChecked(bool(clip.get("use_reference_images", False)))
            self.use_refs_check.blockSignals(False)
            self.use_refs_warning.setVisible(bool(clip.get("use_reference_images", False)))
            self._refresh_reference_rows(clip)
            # A loaded start video is already the first timeline result; it needs no H3 prompt or generation settings.
            for w in (self.frames_combo, self.seed_spin, self.steps_spin, self.scheduler_combo, self.glue_check,
                      self.audio_memory_check, self.latent_check, self.capture_btn, self.cut_prompt):
                w.setEnabled(enabled and not is_source)
            fidx = self.frames_combo.findData(int(clip.get("frames") or 243))
            if fidx >= 0: self.frames_combo.setCurrentIndex(fidx)
            self.seed_spin.setValue(int(settings.get("seed", -1)))
            self.steps_spin.setValue(int(settings.get("steps", 15)))
            sched = str(settings.get("scheduler") or "beta")
            if self.scheduler_combo.findText(sched) < 0: self.scheduler_combo.addItem(sched)
            self.scheduler_combo.setCurrentText(sched)
            self.glue_check.setChecked(bool(settings.get("glue_results", False)))
            self.audio_memory_check.setChecked(bool(settings.get("continue_audio_memory", True)))
            self.latent_check.setChecked(bool(settings.get("latent_continuation", False)))
            self._refresh_edit_workflow(clip)
            state = str(clip.get("status") or "draft").title()
            if clip.get("stale"): state = "Stale — edited after queue/render"
            self.state_label.setText(state)
            output = str(clip.get("output") or "")
            output_exists = bool(output and Path(output).is_file())
            self.clip_output_label.setText(output if output else "—")
            self.preview_clip_btn.setEnabled(output_exists and str(clip.get("status") or "") == "finished")
            self.open_clip_btn.setEnabled(output_exists)
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
            if str(clip.get("generation_mode") or "") == "source":
                self.cut_prompt.clear()
                self.cut_prompt.setPlaceholderText("Loaded start clips do not need a prompt.")
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

    def new_project(self):
        if self._clips() and any(str(s.get("prompt") or "").strip() for c in self._clips() for s in c.get("segments") or []):
            if QMessageBox.question(self, "New timeline", "Clear the current timeline and start a new project?") != QMessageBox.StandardButton.Yes:
                return
        self.project = self._new_project_data()
        self._project_path = None
        self.selected_clip_id = None
        self.selected_cut_index = 0
        self._ensure_initial_clip()
        self._refresh_all(select_first=True)

    def save_project(self):
        if not self._project_path:
            suggested = _safe_project_name(self.project.get("name")).replace(" ", "_") + ".json"
            name, _ = QFileDialog.getSaveFileName(self, "Save MiniMax timeline", suggested, "MiniMax Timeline (*.json);;JSON (*.json)")
            if not name:
                return
            self._project_path = Path(name)
        try:
            self._write_project_json(self._project_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save timeline failed", str(exc))
            return
        # Once the project has a real file, the temporary recovery copy is no
        # longer authoritative. Future autosaves go to the saved JSON.
        try:
            if self._autosave_temp_path.exists():
                self._autosave_temp_path.unlink()
        except Exception:
            pass

    def load_project(self):
        name, _ = QFileDialog.getOpenFileName(self, "Load MiniMax timeline", "", "MiniMax Timeline (*.json);;JSON (*.json)")
        if not name:
            return
        try:
            data = json.loads(Path(name).read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("clips"), list):
                raise ValueError("File does not contain a MiniMax timeline project.")
            self.project = data
            self.project.setdefault("schema_version", TIMELINE_SCHEMA_VERSION)
            self.project.setdefault("project_id", uuid.uuid4().hex)
            self.project.setdefault("name", Path(name).stem)
            self.project.setdefault("assembled_output", "")
            self.project.setdefault("assembly_status", "")
            self.project.setdefault("auto_assemble", True)
            self.project.setdefault("auto_assemble_pending", False)
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
                clip.setdefault("status", "draft")
                clip.setdefault("stale", False)
                clip.setdefault("queue_job_id", None)
                clip.setdefault("output", "")
                if not clip.get("segments"):
                    clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": "", "weight": 1.0}]
                for seg in clip["segments"]:
                    seg.setdefault("id", uuid.uuid4().hex); seg.setdefault("prompt", ""); seg.setdefault("weight", 1.0)
                if len(clip["segments"]) > 1:
                    merged_prompt = _compiled_prompt(clip)
                    clip["segments"] = [{"id": uuid.uuid4().hex, "prompt": merged_prompt, "weight": 1.0}]
            self._project_path = Path(name)
            self.selected_clip_id = self._clips()[0]["id"] if self._clips() else None
            self.selected_cut_index = 0
            self._refresh_all(select_first=True)
        except Exception as exc:
            QMessageBox.critical(self, "Load timeline failed", str(exc))

    # ------------------------------------------------------------- clip actions
    def add_clip(self):
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
        self.selected_cut_index = 0
        self._renumber_default_names()
        self._refresh_all()

    def duplicate_clip(self):
        idx = self._selected_index()
        if idx < 0: return
        if str(self._clips()[idx].get("generation_mode") or "") == "source":
            QMessageBox.information(self, "Timeline", "The loaded start clip is a source video and cannot be duplicated as a generation block.")
            return
        clone = copy.deepcopy(self._clips()[idx])
        clone["id"] = uuid.uuid4().hex
        clone["name"] = str(clone.get("name") or f"Clip {idx + 1}") + " copy"
        clone["queue_job_id"] = None; clone["output"] = ""; clone["status"] = "draft"; clone["stale"] = False
        for seg in clone.get("segments") or []: seg["id"] = uuid.uuid4().hex
        self._clips().insert(idx + 1, clone)
        self._invalidate_assembly()
        self.selected_clip_id = clone["id"]
        self._refresh_all()

    def delete_clip(self):
        idx = self._selected_index()
        if idx < 0: return
        clip = self._clips()[idx]
        has_content = any(str(s.get("prompt") or "").strip() for s in clip.get("segments") or [])
        if has_content and QMessageBox.question(self, "Delete clip", f"{clip.get('name')} contains prompt text. Delete it anyway?") != QMessageBox.StandardButton.Yes:
            return
        self._clips().pop(idx)
        self._invalidate_assembly()
        self._renumber_default_names()
        if self._clips():
            self.selected_clip_id = self._clips()[min(idx, len(self._clips()) - 1)]["id"]
        else:
            self.selected_clip_id = None
        self.selected_cut_index = 0
        self._repair_continuation_chain(mark_stale=True)
        self._refresh_all()

    def move_clip(self, delta):
        idx = self._selected_index()
        target = idx + int(delta)
        if idx < 0 or not (0 <= target < len(self._clips())): return
        self._reorder_clips(idx, target)

    def _reorder_clips(self, old, new):
        clips = self._clips()
        if not (0 <= old < len(clips) and 0 <= new < len(clips)): return
        # A user-loaded start video is the root of the chain and must remain first.
        if clips and str(clips[0].get("generation_mode") or "") == "source" and (old == 0 or new == 0):
            return
        item = clips.pop(old); clips.insert(new, item)
        self._invalidate_assembly()
        self._repair_continuation_chain(mark_stale=True)
        self._refresh_all()

    def _repair_continuation_chain(self, mark_stale=False):
        for i, clip in enumerate(self._clips()):
            if i == 0 and clip.get("generation_mode") == "continue":
                clip["generation_mode"] = "new"
                if mark_stale: clip["stale"] = True
            elif mark_stale and clip.get("generation_mode") == "continue" and clip.get("queue_job_id"):
                clip["stale"] = True

    def select_clip(self, clip_id):
        self.selected_clip_id = str(clip_id)
        self.selected_cut_index = 0
        self.canvas.selected_id = self.selected_clip_id
        self.canvas.update()
        self._load_inspector()

    # ---------------------------------------------------------- inspector slots
    def _project_name_changed(self, text):
        self.project["name"] = text

    def _clip_name_changed(self, text):
        if self._loading_inspector: return
        clip = self._selected_clip()
        if not clip: return
        clip["name"] = text
        self.canvas.update()

    def _populate_generation_modes(self, index: int, current_mode: str = "new"):
        self.gen_mode.blockSignals(True)
        try:
            clip = self._selected_clip()
            using_refs = bool((clip or {}).get("use_reference_images", False))
            self.gen_mode.clear()
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
            thumb.setToolTip(ref["path"])

            token = QLabel(f"<Subject {idx}>\n<Picture {idx}>")
            token.setToolTip(
                f"Use <Subject {idx}> in the prompt for reusable visible content. "
                f"MiniMax receives this file as <Picture {idx}>."
            )
            token.setMinimumWidth(105)

            name_edit = QLineEdit(ref.get("name") or Path(ref["path"]).stem)
            name_edit.setPlaceholderText(f"Reference {idx} name")
            name_edit.setToolTip("Friendly name used in the automatic <Subject N> definition sent to MiniMax.")
            name_edit.editingFinished.connect(lambda i=idx-1, w=name_edit: self._reference_name_changed(i, w.text()))

            insert_btn = QPushButton(f"Insert <Subject {idx}>")
            insert_btn.setToolTip(f"Insert <Subject {idx}> at the prompt cursor.")
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
        self.add_refs_btn.setEnabled(bool(clip) and len(refs) < 5 and str(clip.get("generation_mode") or "") != "source")

    def _reference_mode_changed(self, enabled):
        if self._loading_inspector:
            return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip or str(clip.get("generation_mode") or "") == "source":
            return
        enabled = bool(enabled)
        clip["use_reference_images"] = enabled
        self.use_refs_warning.setVisible(enabled)
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
        if not clip or str(clip.get("generation_mode") or "") == "source":
            return
        refs = _reference_entries(clip)
        remaining = 5 - len(refs)
        if remaining <= 0:
            QMessageBox.information(self, "Reference images", "This timeline clip already has the maximum of 5 reference images.")
            return
        names, _ = QFileDialog.getOpenFileNames(
            self, "Add reference images", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)"
        )
        if not names:
            return
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
        if not clip:
            return
        refs = _reference_entries(clip)
        if not (0 <= ref_index < len(refs)):
            return
        old_count = len(refs)
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
        if not clip:
            return
        refs = _reference_entries(clip)
        if not (0 <= ref_index < len(refs)):
            return
        value = str(text or "").strip() or Path(refs[ref_index]["path"]).stem
        refs[ref_index]["name"] = value
        clip["reference_images"] = refs
        self._touch_clip(idx, propagate=False)
        self._refresh_settings_summary(clip)
        self.canvas.update()

    def _insert_reference_token(self, number: int):
        if not self.cut_prompt.isEnabled():
            return
        cursor = self.cut_prompt.textCursor()
        cursor.insertText(f"<Subject {int(number)}>" )
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
        if not clip:
            return
        mode = str(self.gen_mode.currentData() or "new")
        if idx == 0 and mode == "source":
            previous_mode = str(clip.get("generation_mode") or "new")
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
        clip["generation_mode"] = mode
        self._touch_clip(idx)
        self._refresh_all()

    def _frames_changed(self):
        if self._loading_inspector: return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip: return
        frames = int(self.frames_combo.currentData() or clip.get("frames") or 243)
        if frames != int(clip.get("frames") or 0):
            clip["frames"] = frames
            clip.setdefault("settings", {})["frames"] = frames
            clip["settings"]["experimental_long_duration"] = frames > 719
            self._touch_clip(idx)
            self._refresh_all()

    def _canvas_frames_changed(self, clip_id, frames):
        for i, clip in enumerate(self._clips()):
            if str(clip.get("id")) == str(clip_id):
                clip["frames"] = int(frames)
                clip.setdefault("settings", {})["frames"] = int(frames)
                clip["settings"]["experimental_long_duration"] = int(frames) > 719
                self._touch_clip(i)
                break
        self._refresh_all()

    def _settings_changed(self, *args):
        if self._loading_inspector: return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip: return
        s = clip.setdefault("settings", {})
        s["seed"] = self.seed_spin.value(); s["steps"] = self.steps_spin.value(); s["scheduler"] = self.scheduler_combo.currentText()
        s["glue_results"] = self.glue_check.isChecked(); s["continue_audio_memory"] = self.audio_memory_check.isChecked(); s["latent_continuation"] = self.latent_check.isChecked()
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

        if str(clip.get("generation_mode") or "") == "source":
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
            prev_state = "ready" if str(prev.get("status") or "") == "finished" and bool(prev.get("output")) and not bool(prev.get("stale")) else "not rendered / stale"
        if has_next:
            nxt = self._clips()[idx + 1]
            next_state = "ready" if str(nxt.get("status") or "") == "finished" and bool(nxt.get("output")) and not bool(nxt.get("stale")) else "not rendered / stale"
        ref_note = "  •  Ref2VA: previous continuation disabled" if using_refs else ""
        self.edit_selected_label.setText(
            f"{clip.get('name') or f'Clip {idx + 1}'}  •  previous: {prev_state}  •  next: {next_state}{ref_note}"
        )

    def _edit_mode_changed(self, button):
        if self._loading_inspector or button is None:
            return
        clip = self._selected_clip()
        if not clip:
            return
        mode = str(button.property("edit_mode") or "standalone")
        clip["edit_mode"] = mode
        # This is an edit strategy, not a content change. Choosing it must not
        # stale the clip or mutate the normal first-time creation workflow.
        self.canvas.update()

    def capture_current_settings(self):
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip:
            return

        # Capture technical Generation-tab settings only. The timeline prompt is
        # authored independently and must never be replaced by whatever prompt
        # happens to be present on the Generation tab.
        current_prompt = _compiled_prompt(clip)
        current_segments = copy.deepcopy(clip.get("segments") or [])
        settings = self._capture_settings()
        settings.pop("prompt", None)
        # Reference images are edited per timeline clip in the dedicated Ref2VA
        # section; Capture Settings must not overwrite them from the Generation tab.
        settings["ref_images"] = []
        settings["ref_videos"] = []
        settings["ref_audios"] = []
        clip["settings"] = settings

        frames = int(settings.get("frames") or clip.get("frames") or 243)
        if self.frame_values:
            frames = min(self.frame_values, key=lambda v: abs(v - frames))
        clip["frames"] = frames

        # Preserve the timeline prompt verbatim. Keep the existing segment id so
        # saved project state remains stable even though the UI exposes one prompt.
        if current_segments:
            clip["segments"] = current_segments
        else:
            clip["segments"] = [{
                "id": uuid.uuid4().hex,
                "prompt": current_prompt,
                "weight": 1.0,
            }]

        self._touch_clip(idx)
        self._refresh_all()

    # -------------------------------------------------------------- prompt edit
    def _cut_prompt_changed(self):
        if self._loading_inspector:
            return
        clip = self._selected_clip(); idx = self._selected_index()
        if not clip:
            return
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

    # --------------------------------------------------------------- results
    def _assembly_ready(self):
        clips = self._clips()
        if not clips:
            return False, "Timeline has no clips."
        for i, clip in enumerate(clips):
            name = str(clip.get("name") or f"Clip {i + 1}")
            if clip.get("stale"):
                return False, f"{name} is stale and must be regenerated first."
            if str(clip.get("status") or "") != "finished":
                return False, f"{name} has not finished generating."
            output = str(clip.get("output") or "")
            if not output or not Path(output).is_file():
                return False, f"{name} has no usable output file."
            if i > 0 and clip.get("generation_mode") == "continue" and bool((clip.get("settings") or {}).get("glue_results", False)):
                return False, (
                    f"{name} has 'Glue result to source' enabled. Timeline assembly expects each block "
                    "to contain only its own generated section; disable Glue and regenerate that continuation clip."
                )
        return True, ""

    def assemble_timeline(self):
        ready, reason = self._assembly_ready()
        if not ready:
            QMessageBox.information(self, "Timeline assembly", reason)
            return False
        if not callable(self.assemble_timeline_callback):
            QMessageBox.warning(self, "Timeline assembly", "Timeline assembly is not connected to the standalone GUI.")
            return False
        self.project["assembly_status"] = "Assembling…"
        self.project["assembled_output"] = ""
        self._refresh_all()
        result = self.assemble_timeline_callback(copy.deepcopy(self.project))
        if not result and self.project.get("assembly_status") == "Assembling…":
            self.project["assembly_status"] = "Assembly did not start."
            self._refresh_all()
        return bool(result)

    def mark_assembly_started(self, output):
        self.project["assembled_output"] = str(output or "")
        self.project["assembly_status"] = "Assembling…"
        self._refresh_all()

    def mark_assembly_finished(self, output):
        self.project["assembled_output"] = str(output or "")
        self.project["assembly_status"] = "Finished"
        self._refresh_all()

    def mark_assembly_failed(self, message):
        self.project["assembled_output"] = ""
        self.project["assembly_status"] = "Assembly failed: " + str(message or "Unknown error")
        self._refresh_all()

    def preview_selected_result(self):
        clip = self._selected_clip()
        if not clip:
            return
        output = str(clip.get("output") or "")
        if callable(self.preview_result_callback) and output:
            self.preview_result_callback(output, clip.get("queue_job_id"))

    def open_selected_output(self):
        clip = self._selected_clip()
        output = str((clip or {}).get("output") or "")
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
            if mode == "source":
                if i != 0:
                    return False, "Only Clip 1 can be a loaded start clip."
                source = str(clip.get("start_source_video") or clip.get("output") or "")
                if not source or not Path(source).is_file():
                    return False, "Clip 1 is set to Load a start clip, but the video file is missing."
                continue
            prompt = _compiled_prompt(clip)
            if not prompt:
                return False, f"{clip.get('name') or f'Clip {i + 1}'} has no prompt."
            if bool(clip.get("use_reference_images", False)):
                refs = _reference_entries(clip)
                if not refs:
                    return False, f"{clip.get('name') or f'Clip {i + 1}'} has reference mode enabled but no reference images."
                if len(refs) > 5:
                    return False, f"{clip.get('name') or f'Clip {i + 1}'} has more than 5 reference images."
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
        for i, clip in enumerate(clips):
            spec = copy.deepcopy(clip)
            spec["timeline_index"] = i
            spec["compiled_prompt"] = _compiled_reference_prompt(clip)
            spec["timeline_reference_images"] = _reference_entries(clip)
            # Edit/replacement topology is used only by Regenerate selected block.
            spec["match_next_first_frame"] = False
            if i > 0:
                prev = clips[i - 1]
                spec["timeline_previous_output"] = str(prev.get("output") or "")
                spec["timeline_previous_job_id"] = str(prev.get("queue_job_id") or "")
                spec["timeline_previous_status"] = str(prev.get("status") or "")
                spec["timeline_previous_stale"] = bool(prev.get("stale"))
                spec["timeline_previous_is_source"] = str(prev.get("generation_mode") or "") == "source"
            if i + 1 < len(clips):
                nxt = clips[i + 1]
                spec["timeline_next_output"] = str(nxt.get("output") or "")
                spec["timeline_next_status"] = str(nxt.get("status") or "")
                spec["timeline_next_stale"] = bool(nxt.get("stale"))
            specs.append(spec)
        return specs

    def generate_selected(self):
        clip = self._selected_clip()
        idx = self._selected_index()
        if clip is None or idx < 0:
            return False
        if str(clip.get("generation_mode") or "") == "source":
            QMessageBox.information(self, "Timeline edit", "The loaded start clip is a source video, not an H3 generation. Choose another block to regenerate it.")
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
        edit_mode = str(clip.get("edit_mode") or ("continue_previous" if clip.get("generation_mode") == "continue" else "standalone"))
        use_previous = edit_mode in {"bridge_both", "continue_previous"}
        use_next = edit_mode in {"bridge_both", "anchor_next"}
        if bool(clip.get("use_reference_images", False)) and use_previous:
            QMessageBox.warning(self, "Timeline edit", "Reference images use Ref2VA and cannot continue from the previous block. Choose a non-previous edit mode.")
            return False
        spec["edit_mode"] = edit_mode
        spec["generation_mode"] = "continue" if use_previous else "new"
        spec["match_next_first_frame"] = bool(use_next)

        if use_previous:
            if idx == 0:
                QMessageBox.warning(self, "Timeline edit", "This replacement mode needs a previous block, but the selected block is first.")
                return False
            prev = self._clips()[idx - 1]
            prev_output = Path(str(prev.get("output") or ""))
            if str(prev.get("status") or "") != "finished" or bool(prev.get("stale")) or not prev_output.is_file():
                QMessageBox.warning(self, "Timeline edit", "The previous block must have a valid finished result before this replacement can continue from it.")
                return False
        if use_next:
            if idx + 1 >= len(self._clips()):
                QMessageBox.warning(self, "Timeline edit", "This replacement mode needs a next block, but the selected block is last.")
                return False
            nxt = self._clips()[idx + 1]
            next_output = Path(str(nxt.get("output") or ""))
            if str(nxt.get("status") or "") != "finished" or bool(nxt.get("stale")) or not next_output.is_file():
                QMessageBox.warning(self, "Timeline edit", "The next block must have a valid finished result before its first frame can anchor this replacement.")
                return False
        if not callable(self.queue_timeline_callback):
            QMessageBox.warning(self, "Timeline", "The timeline is not connected to the MiniMax queue.")
            return False
        result = self.queue_timeline_callback([spec])
        if result:
            self._invalidate_assembly("Selected clip regenerated — assemble again when ready.")
            # A free-ending continuation changes the boundary consumed by the next
            # continuation clip. The other three edit modes intentionally preserve
            # the next block via anchoring or a hard cut.
            if edit_mode == "continue_previous":
                for j in range(idx + 1, len(self._clips())):
                    if self._clips()[j].get("generation_mode") != "continue":
                        break
                    if self._clips()[j].get("queue_job_id") or self._clips()[j].get("status") in {"pending", "running", "finished"}:
                        self._clips()[j]["stale"] = True
            self._refresh_all()
        return bool(result)

    def generate_timeline(self):
        ok, error = self.validate_timeline()
        if not ok:
            QMessageBox.warning(self, "Timeline not ready", error)
            return False
        if not callable(self.queue_timeline_callback):
            QMessageBox.warning(self, "Timeline", "The timeline is not connected to the MiniMax queue.")
            return False
        result = self.queue_timeline_callback(self.generation_specs())
        if result:
            self.project["assembled_output"] = ""
            self.project["assembly_status"] = "Waiting for timeline clips to finish…" if self.auto_assemble_check.isChecked() else ""
            self.project["auto_assemble"] = self.auto_assemble_check.isChecked()
            self.project["auto_assemble_pending"] = self.auto_assemble_check.isChecked()
            self._refresh_all()
        return bool(result)

    def mark_queued(self, clip_id, job_id, output=""):
        for clip in self._clips():
            if str(clip.get("id")) == str(clip_id):
                clip["queue_job_id"] = str(job_id)
                clip["status"] = "pending"
                clip["stale"] = False
                clip["output"] = str(output or "")
                break
        self._refresh_all()

    def sync_queue_jobs(self, jobs):
        by_id = {str(j.get("id")): j for j in (jobs or []) if j.get("id")}
        changed = False
        for clip in self._clips():
            job_id = clip.get("queue_job_id")
            if not job_id or str(job_id) not in by_id:
                continue
            job = by_id[str(job_id)]
            state = str(job.get("state") or "pending")
            if clip.get("status") != state:
                clip["status"] = state; changed = True
            output = str(job.get("output") or "")
            if output and clip.get("output") != output:
                clip["output"] = output; changed = True
        if changed:
            self._refresh_all()
        if self.project.get("auto_assemble_pending"):
            ready, _reason = self._assembly_ready()
            if ready:
                self.project["auto_assemble_pending"] = False
                self.assemble_timeline()
