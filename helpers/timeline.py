"""
Timeline UI components for the Music Clip Creator.

This module is intentionally separate from auto_music_sync.py so that
the timeline / music-structure view can grow over time without bloating
the main tool file.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from pathlib import Path
import json
import re
from datetime import datetime

from PySide6.QtCore import Qt, Signal, QEvent
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFrame,
    QSizePolicy,
    QMenu,
    QScrollArea,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)


# Simple color map for section types. Colors are chosen to fit the dark UI theme.
SECTION_COLORS = {
    "intro": "#44486e",
    "verse": "#305d39",
    "chorus": "#255a92",
    "drop": "#a63a3a",
    "break": "#8c7b2f",
}


# Simple icon map for section types (colored dot emojis) so the table
# and tooltips visually match the colored bar.
SECTION_ICONS = {
    "intro": "🟣",
    "verse": "🟢",
    "chorus": "🔵",
    "drop": "🔴",
    "break": "🟡",
}

# Optional pyqtgraph-based energy overlay (if pyqtgraph is available).
try:
    import pyqtgraph as pg  # type: ignore
except Exception:  # pragma: no cover - UI only
    pg = None



class _SectionBlock(QFrame):
    """Clickable block in the mini-timeline bar.

    Right-click opens a small menu with Add / Remove / Select actions.
    The block forwards user intent back to the owning TimelinePanel via
    private callbacks so the main tool can react (e.g. by linking clips).
    """

    def __init__(self, panel: "TimelinePanel", section: Any, color: str, tooltip: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._section = section
        self.setFrameShape(QFrame.NoFrame)
        self.setMinimumHeight(32)
        self.setStyleSheet(f"background-color: {color}; border-radius: 2px;")
        self.setToolTip(tooltip)

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        menu = QMenu(self)
        act_add = menu.addAction("Add")
        act_remove = menu.addAction("Remove")
        act_select = menu.addAction("Select")
        chosen = menu.exec_(event.globalPos())
        if chosen is None:
            return
        if chosen == act_add:
            self._panel._on_block_add(self._section)
        elif chosen == act_remove:
            self._panel._on_block_remove(self._section)
        elif chosen == act_select:
            self._panel._on_block_select(self._section)
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        # Left-click also selects the section for convenience.
        try:
            if event.button() == Qt.LeftButton:
                self._panel._on_block_select(self._section)
        except Exception:
            pass
        super().mousePressEvent(event)



class Segment:
    """Simple data holder for a custom user-defined segment on the song timeline.

    A segment has a start / end time in seconds and an optional human-readable label.
    """

    def __init__(self, start: float, end: float, label: str = "") -> None:
        self.start = float(start)
        self.end = float(end)
        # Optional name for use in UIs / JSON / signals.
        self.label = str(label) if label is not None else ""

    def as_tuple(self) -> tuple[float, float]:
        """Return (start, end) for backwards-compatible callers."""
        return float(self.start), float(self.end)

    def as_dict(self) -> dict[str, Any]:
        """Return a small payload suitable for JSON save / signals."""
        return {
            "start": float(self.start),
            "end": float(self.end),
            "label": self.label,
        }


class _SegmentBar(QWidget):
    """Interactive bar that lets the user create and tweak custom segments.

    - Ctrl+click to create a new segment around the mouse position.
    - Drag left / right edge of a segment to adjust its start / end time.
    - Drag inside a segment to move it.
    - Right-click a segment to delete it (or press Delete when focused).
    """

    def __init__(self, panel: "TimelinePanel", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._panel = panel
        self._duration: float = 0.0
        self._pixels_per_second: float = 40.0
        self._edge_margin_px: int = 6

        self._drag_index: Optional[int] = None
        self._drag_edge: str = ""
        self._drag_offset_time: float = 0.0
        self._drag_initial_start: float = 0.0
        self._drag_initial_end: float = 0.0

        self.setMinimumHeight(22)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip(
            "Custom segments: Ctrl+click to add, drag edges to resize, right-click to rename/delete, Delete to remove."
        )

    # ----------------------------- geometry -----------------------------

    def set_timeline_params(self, duration: float, pixels_per_second: float) -> None:
        """Synchronise width / scaling with the main timeline bar."""
        self._duration = max(0.0, float(duration or 0.0))
        self._pixels_per_second = max(1.0, float(pixels_per_second or 1.0))
        width = int(self._duration * self._pixels_per_second)
        self.setMinimumWidth(max(1, width))
        self.update()

    def _time_to_x(self, t: float) -> int:
        t = max(0.0, min(self._duration, float(t or 0.0)))
        return int(t * self._pixels_per_second)

    def _x_to_time(self, x: float) -> float:
        if self._pixels_per_second <= 0:
            return 0.0
        t = float(x) / self._pixels_per_second
        return max(0.0, min(self._duration, t))

    # ------------------------------- paint ------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        from PySide6.QtGui import QPainter, QColor, QPen, QBrush
        from PySide6.QtCore import QRectF

        painter = QPainter(self)
        rect = self.rect()

        # Subtle background so the bar stands out from the section bar.
        painter.fillRect(rect, QColor(32, 32, 32, 190))

        segments = getattr(self._panel, "_segments", [])
        if not segments or self._duration <= 0.0:
            painter.setPen(QColor(160, 160, 160, 210))
            hint = "No custom segments yet – Ctrl+click to add one."
            painter.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, hint)
            painter.end()
            return

        selected_index = getattr(self._panel, "_selected_segment_index", None)

        for idx, seg in enumerate(segments):
            start, end = float(seg.start), float(seg.end)
            if end <= start:
                continue
            x1 = self._time_to_x(start)
            x2 = self._time_to_x(end)
            if x2 <= x1:
                continue

            w = max(4, x2 - x1)
            inner = QRectF(x1 + 1, rect.top() + 3, w - 2, rect.height() - 6)

            base = QColor(210, 210, 120, 180)
            border = QColor(245, 245, 200) if idx == selected_index else QColor(220, 220, 170)

            painter.fillRect(inner, QBrush(base))
            pen = QPen(border, 1.4)
            painter.setPen(pen)
            painter.drawRect(inner)

            # Tiny handles on the edges to suggest they are draggable.
            handle_w = 3
            left_handle = QRectF(x1, rect.top() + 2, handle_w, rect.height() - 4)
            right_handle = QRectF(x2 - handle_w, rect.top() + 2, handle_w, rect.height() - 4)
            painter.fillRect(left_handle, border)
            painter.fillRect(right_handle, border)

            # Optional label text inside the segment, if available.
            label = getattr(seg, "label", "")
            if label:
                painter.setPen(QColor(20, 20, 20, 230))
                text_rect = inner.adjusted(4, 0, -4, 0)
                painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, str(label))

        painter.end()

    # --------------------------- interactions ---------------------------

    def _hit_test(self, x: float) -> tuple[Optional[int], str]:
        """Return (segment_index, region) for a given x position."""
        segments = getattr(self._panel, "_segments", [])
        if not segments:
            return None, ""

        for idx, seg in enumerate(segments):
            start, end = float(seg.start), float(seg.end)
            if end <= start:
                continue
            x1 = self._time_to_x(start)
            x2 = self._time_to_x(end)
            if x2 <= x1:
                continue

            if abs(x - x1) <= self._edge_margin_px:
                return idx, "left"
            if abs(x - x2) <= self._edge_margin_px:
                return idx, "right"
            if x1 < x < x2:
                return idx, "inside"

        return None, ""

    def _create_segment_at(self, x: float) -> None:
        if self._duration <= 0.0:
            return

        center_time = self._x_to_time(x)
        default_len = min(4.0, self._duration) or self._duration
        start = max(0.0, center_time - default_len / 2.0)
        end = min(self._duration, start + default_len)
        if end - start < 0.1:
            end = min(self._duration, start + 0.1)

        seg = Segment(start, end)
        self._panel._segments.append(seg)
        self._panel._segments.sort(key=lambda s: float(s.start))
        idx = self._panel._segments.index(seg)
        self._panel._on_segment_selected(idx)
        self._panel._on_segments_changed()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        x = float(getattr(event, "position", lambda: event.pos())().x())
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                self._create_segment_at(x)
                event.accept()
                return

            idx, region = self._hit_test(x)
            if idx is not None:
                self._panel._on_segment_selected(idx)
                seg = self._panel._segments[idx]
                self._drag_index = idx
                self._drag_edge = region or "inside"
                self._drag_initial_start = float(seg.start)
                self._drag_initial_end = float(seg.end)
                self._drag_offset_time = self._x_to_time(x) - self._drag_initial_start
                event.accept()
                return

        elif event.button() == Qt.RightButton:
            idx, _ = self._hit_test(x)
            if idx is not None:
                self._panel._on_segment_selected(idx)
                seg = self._panel._segments[idx]
                menu = QMenu(self)
                act_rename = menu.addAction("Rename segment…")
                act_delete = menu.addAction("Delete segment (Del)")
                chosen = menu.exec_(event.globalPos())
                if chosen == act_rename:
                    current_label = getattr(seg, "label", "")
                    new_label, ok = QInputDialog.getText(
                        self,
                        "Rename segment",
                        "Segment label:",
                        text=current_label,
                    )
                    if ok:
                        seg.label = str(new_label).strip()
                        self._panel._on_segments_changed()
                elif chosen == act_delete:
                    self._panel._delete_selected_segment()
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_index is None or self._duration <= 0.0:
            super().mouseMoveEvent(event)
            return

        segments = getattr(self._panel, "_segments", [])
        if not (0 <= self._drag_index < len(segments)):
            super().mouseMoveEvent(event)
            return

        seg = segments[self._drag_index]
        x = float(getattr(event, "position", lambda: event.pos())().x())
        t = self._x_to_time(x)

        if self._drag_edge == "left":
            new_start = min(t, float(seg.end) - 0.1)
            seg.start = max(0.0, new_start)
        elif self._drag_edge == "right":
            new_end = max(t, float(seg.start) + 0.1)
            seg.end = min(self._duration, new_end)
        else:
            length = self._drag_initial_end - self._drag_initial_start
            new_start = t - self._drag_offset_time
            new_start = max(0.0, min(self._duration - length, new_start))
            seg.start = new_start
            seg.end = new_start + length

        self._panel._on_segments_changed()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._drag_index is not None:
            # Keep segments ordered after adjustments.
            self._panel._segments.sort(key=lambda s: float(s.start))
            self._panel._on_segments_changed()
            self._drag_index = None
            self._drag_edge = ""
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_Delete:
            if self._panel._delete_selected_segment():
                event.accept()
                return
        super().keyPressEvent(event)


class TimelinePanel(QWidget):
    """
    Lightweight visual overview for the analyzed music structure.

    - Top: a compact colored timeline bar showing section types over time.
    - Bottom: a table with one row per section (kind, start, end, duration).

    This widget does not perform any analysis; it only visualises the
    MusicAnalysisResult passed in from auto_music_sync.
    """

    # Emitted when the user chooses an action from the mini-timeline bar.
    # The payload is the underlying section object from the analysis.
    segmentAddRequested = Signal(object)
    segmentRemoveRequested = Signal(object)
    segmentSelected = Signal(object)

    # Emitted whenever the custom user-defined segments change in any way.
    # Payload: list of dicts [{ "start": float, "end": float, "label": str }, ...].
    customSegmentsChanged = Signal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._analysis: Optional[Any] = None
        # Keep a simple list of sections so we can map clicks back to table rows.
        self._sections: list[Any] = []
        # Map table rows to mini-timeline blocks for tooltip updates.
        self._section_blocks: dict[int, _SectionBlock] = {}
        # Optional label (file name) for media attached to each section.
        self._section_media_labels: dict[int, str] = {}
        # Custom user-defined segments over the song timeline.
        self._segments: list[Segment] = []
        self._selected_segment_index: Optional[int] = None


        # Zoom state for the mini-timeline bar.
        self._zoom_factor: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 8.0
        # Base pixels-per-second for the bar at zoom = 1.0
        self._pixels_per_second: float = 40.0
        # Panning state for middle-mouse drag on the mini-timeline bar.
        self._is_panning: bool = False
        self._pan_start_x: int = 0
        self._pan_start_scroll: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Music structure timeline", self)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        # Stack for energy overlay + colored mini-timeline bar
        self._timeline_box = QVBoxLayout()
        self._timeline_box.setContentsMargins(0, 0, 0, 0)
        self._timeline_box.setSpacing(2)
        layout.addLayout(self._timeline_box)

        # Optional pyqtgraph energy overlay (thin waveform / energy envelope)
        if pg is not None:
            self._energy_plot = pg.PlotWidget(self)
            self._energy_plot.setMaximumHeight(56)
            self._energy_plot.setMinimumHeight(36)
            self._energy_plot.setMenuEnabled(False)
            self._energy_plot.setMouseEnabled(x=False, y=False)
            self._energy_plot.hideAxis("left")
            self._energy_plot.hideAxis("bottom")
            self._energy_plot.setFrameStyle(QFrame.NoFrame)
            self._energy_plot.setStyleSheet("background: transparent;")
            self._timeline_box.addWidget(self._energy_plot)
        else:
            self._energy_plot = None  # type: ignore

        # Colored mini-timeline bar (sections) inside a scroll area to allow zooming.
        self._bar_scroll = QScrollArea(self)
        self._bar_scroll.setWidgetResizable(False)
        self._bar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._bar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._bar_scroll.setFrameShape(QFrame.NoFrame)

        self._bar_container = QWidget(self._bar_scroll)
        self._bar_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._bar_container.setMinimumHeight(56)

        # Stack the analysed-section bar and the user-editable segment bar
        # inside a single scroll area so they share the same horizontal scroll.
        self._bar_container_layout = QVBoxLayout(self._bar_container)
        self._bar_container_layout.setContentsMargins(0, 0, 0, 0)
        self._bar_container_layout.setSpacing(2)

        self._section_bar = QWidget(self._bar_container)
        self._bar_layout = QHBoxLayout(self._section_bar)
        self._bar_layout.setContentsMargins(0, 0, 0, 0)
        self._bar_layout.setSpacing(2)
        self._bar_container_layout.addWidget(self._section_bar)

        # Custom user segments live in a slim bar below the main structure view.
        self._segment_bar = _SegmentBar(self, self._bar_container)
        self._bar_container_layout.addWidget(self._segment_bar)

        self._bar_scroll.setWidget(self._bar_container)
        self._timeline_box.addWidget(self._bar_scroll)

        # Enable mouse tracking so middle-mouse drag panning feels responsive.
        self._bar_scroll.viewport().setMouseTracking(True)

        # Listen for wheel events over the bar so we can zoom with the scroll wheel.
        self._bar_scroll.viewport().installEventFilter(self)

        # Table with per-section info
        self._table = QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Section", "Start (s)", "End (s)", "Duration (s)", "Media"])
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

        # When the selection changes in the info table, keep the mini-timeline
        # in sync and make sure the matching block is visible when zoomed in.
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)

        # Right-click context menu on the table rows to mirror the timeline bar
        # actions (Add / Remove / Select).
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._on_table_context_menu)

        # Compact textual summary of the structure
        self._summary = QLabel(self)
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(self._summary)

        hint = QLabel(
            "This view updates after you run 'Analyze Music'. Colors and rows show "
            "where intros, verses, breaks and drops occur.",
            self,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #999999; font-size: 10px;")
        layout.addWidget(hint)

        # Save / load buttons for custom segment timelines.
        buttons_row = QHBoxLayout()
        buttons_row.addStretch(1)

        self._btn_save_segments = QPushButton("Save timeline as JSON", self)
        self._btn_save_segments.setToolTip(
            "Save your custom segments to presets/setsave/timelines as a JSON file."
        )
        buttons_row.addWidget(self._btn_save_segments)

        self._btn_load_segments = QPushButton("Load timeline JSON", self)
        self._btn_load_segments.setToolTip(
            "Reload a previously saved custom segment timeline from presets/setsave/timelines."
        )
        buttons_row.addWidget(self._btn_load_segments)

        layout.addLayout(buttons_row)

        self._btn_save_segments.clicked.connect(self._on_save_segments_clicked)
        self._btn_load_segments.clicked.connect(self._on_load_segments_clicked)

    # ------------------------------------------------------------------ API

    def set_analysis(self, analysis: Any) -> None:
        """Populate the timeline from a MusicAnalysisResult-like object.

        Expected attributes on *analysis*:
            - duration: float
            - sections: iterable of objects with .kind, .start, .end
        """
        self._analysis = analysis
        sections = list(getattr(analysis, "sections", None) or [])
        self._sections = sections
        self._section_blocks.clear()
        self._section_media_labels.clear()

        # Reset custom segments whenever a new analysis is applied.
        self._segments.clear()
        self._selected_segment_index = None
        duration = float(getattr(analysis, "duration", 0.0) or 0.0)
        energy_times = getattr(analysis, "energy_times", None) or []
        energy_values = getattr(analysis, "energy_values", None) or []
        beats = getattr(analysis, "beats", None) or []
        waveform_times = getattr(analysis, "waveform_times", None) or []
        waveform_values = getattr(analysis, "waveform_values", None) or []

        self._populate_table(sections)

        # Clear any previous clip labels in the Media column when a new track
        # is analysed so labels do not "stick" across different songs.
        if self._table.columnCount() >= 5:
            for row in range(self._table.rowCount()):
                item = self._table.item(row, 4)
                if item is not None:
                    item.setText("")
        self._section_media_labels.clear()

        self._populate_bar(sections, duration)
        self._populate_energy_overlay(
            duration,
            energy_times,
            energy_values,
            beats,
            waveform_times,
            waveform_values,
        )
        self._update_summary(sections, duration)

        # Keep the custom segment bar aligned with the current zoom + duration.
        if hasattr(self, "_segment_bar"):
            pixels_per_second = max(1.0, self._pixels_per_second * self._zoom_factor)
            self._segment_bar.set_timeline_params(duration, pixels_per_second)

        # Custom segments may be reset when a new analysis is applied, so notify listeners.
        self._on_segments_changed()

    # ----------------------------------------------------------------- internals

    def _populate_table(self, sections: Iterable[Any]) -> None:
        rows = list(sections)
        self._table.setRowCount(len(rows))

        for row, sec in enumerate(rows):
            kind = getattr(sec, "kind", "section") or "section"
            start = float(getattr(sec, "start", 0.0) or 0.0)
            end = float(getattr(sec, "end", start) or start)
            dur = max(0.0, end - start)

            icon = SECTION_ICONS.get(kind, "●")
            pretty_kind = f"{icon} {kind.capitalize()}"
            self._table.setItem(row, 0, QTableWidgetItem(pretty_kind))
            self._table.setItem(row, 1, QTableWidgetItem(f"{start:.1f}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{end:.1f}"))
            self._table.setItem(row, 3, QTableWidgetItem(f"{dur:.1f}"))

            # Lightly tint the row based on section type so patterns are easy to see.
            color = SECTION_COLORS.get(kind)
            if color:
                for col in range(self._table.columnCount()):
                    item = self._table.item(row, col)
                    if item is not None:
                        item.setBackground(self._make_brush(color))

        self._table.resizeColumnsToContents()

    def _section_index_for(self, section: Any) -> Optional[int]:
        """Return the row index for a section object, if known."""
        if not self._sections:
            return None
        try:
            return self._sections.index(section)
        except ValueError:
            target = (
                float(getattr(section, "start", 0.0) or 0.0),
                float(getattr(section, "end", 0.0) or 0.0),
                str(getattr(section, "kind", "section") or "section"),
            )
            for i, sec in enumerate(self._sections):
                cur = (
                    float(getattr(sec, "start", 0.0) or 0.0),
                    float(getattr(sec, "end", 0.0) or 0.0),
                    str(getattr(sec, "kind", "section") or "section"),
                )
                if cur == target:
                    return i
        return None

    def _select_section_in_table(self, section: Any) -> None:
        """Highlight the matching section in the table, if possible."""
        idx = self._section_index_for(section)
        if idx is None or idx < 0:
            return
        self._table.selectRow(idx)
        item = self._table.item(idx, 0)
        if item is not None:
            self._table.scrollToItem(item)

    def set_section_media_label(self, section: Any, label: str) -> None:
        """Update the 'Media' column and tooltip for a given section."""
        idx = self._section_index_for(section)
        if idx is None or idx < 0:
            return
        if idx >= self._table.rowCount() or self._table.columnCount() < 5:
            return

        text = label or ""

        # Update the table cell in the Media column.
        item = self._table.item(idx, 4)
        if item is None:
            item = QTableWidgetItem(text)
            self._table.setItem(idx, 4, item)
        else:
            item.setText(text)

        # Remember label so we can reapply if the bar is rebuilt.
        if text:
            self._section_media_labels[idx] = text
        else:
            self._section_media_labels.pop(idx, None)

        # Update tooltip on the mini-timeline block, if present.
        block = self._section_blocks.get(idx)
        if block is not None:
            base = block.toolTip().split("\n", 1)[0]
            if text:
                block.setToolTip(f"{base}\nMedia: {text}")
            else:
                block.setToolTip(base)

    def _on_block_add(self, section: Any) -> None:
        self._select_section_in_table(section)
        self.segmentAddRequested.emit(section)

    def _on_block_remove(self, section: Any) -> None:
        self._select_section_in_table(section)
        self.segmentRemoveRequested.emit(section)

    def _on_block_select(self, section: Any) -> None:
        self._select_section_in_table(section)
        self.segmentSelected.emit(section)

    def _on_table_context_menu(self, pos) -> None:
        """Context menu on the info table.

        Right-clicking a row lets the user:
          - Add / Remove / Select the matching section (for the auto system)
          - Rename or clear the clip label in the Media column
        """
        row = self._table.rowAt(pos.y())
        if row < 0 or row >= len(self._sections):
            return

        self._table.selectRow(row)
        section = self._sections[row]

        # Current media label (if any) for this section.
        current_label = self._section_media_labels.get(row, "")
        media_item = self._table.item(row, 4)
        if media_item is not None and media_item.text().strip():
            current_label = media_item.text().strip()

        menu = QMenu(self._table)
        act_add = menu.addAction("Add")
        act_remove = menu.addAction("Remove")
        act_select = menu.addAction("Select")
        menu.addSeparator()
        act_rename_media = menu.addAction("Rename clip label…")
        act_clear_media = menu.addAction("Clear clip label")
        chosen = menu.exec_(self._table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_add:
            self._on_block_add(section)
        elif chosen == act_remove:
            self._on_block_remove(section)
        elif chosen == act_select:
            self._on_block_select(section)
        elif chosen == act_rename_media:
            new_label, ok = QInputDialog.getText(
                self,
                "Rename clip label",
                "Clip label for this section:",
                text=current_label,
            )
            if ok:
                self.set_section_media_label(section, str(new_label).strip())
        elif chosen == act_clear_media:
            self.set_section_media_label(section, "")

    def _on_table_selection_changed(self) -> None:
        """When the user selects a row, ensure the matching block is visible."""
        if not self._sections or not self._section_blocks:
            return
        sel_model = self._table.selectionModel()
        if sel_model is None:
            return
        selected_rows = sel_model.selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if row < 0 or row >= len(self._sections):
            return
        block = self._section_blocks.get(row)
        if block is None:
            return
        self._ensure_block_visible(block)

    def _ensure_block_visible(self, block: _SectionBlock) -> None:
        """Scroll the mini-timeline so that *block* is visible when zoomed in."""
        if self._bar_scroll is None:
            return
        hbar = self._bar_scroll.horizontalScrollBar()
        if hbar is None:
            return

        x = block.x()
        w = block.width()
        if w <= 0:
            return

        viewport = self._bar_scroll.viewport()
        if viewport is None:
            return
        viewport_width = viewport.width()
        if viewport_width <= 0:
            return

        current = hbar.value()
        visible_start = current
        visible_end = current + viewport_width

        # Already fully visible, nothing to do.
        if x >= visible_start and (x + w) <= visible_end:
            return

        # Center the block in the viewport where possible.
        new_center = x + w // 2
        new_value = max(0, new_center - viewport_width // 2)
        hbar.setValue(new_value)

    def _populate_bar(self, sections: Iterable[Any], duration: float) -> None:
        # Clear existing widgets
        while self._bar_layout.count():
            item = self._bar_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        sections = list(sections)
        if not sections or duration <= 0:
            self._bar_container.setMinimumWidth(0)
            return

        self._section_blocks.clear()

        # Convert durations to fixed pixel widths so zooming via the scroll
        # wheel actually changes the visual size of each block instead of
        # just rescaling layout stretch factors.
        pixels_per_second = max(1.0, self._pixels_per_second * self._zoom_factor)

        total_width = 0
        spacing = self._bar_layout.spacing()

        for sec in sections:
            kind = getattr(sec, "kind", "section") or "section"
            start = float(getattr(sec, "start", 0.0) or 0.0)
            end = float(getattr(sec, "end", start) or start)
            dur = max(0.0, end - start)
            if dur <= 0:
                continue

            # Minimum width so very short sections are still clickable.
            width = max(4, int(dur * pixels_per_second))
            total_width += width + spacing

            color = SECTION_COLORS.get(kind, "#555555")
            icon = SECTION_ICONS.get(kind, "●")
            tooltip = f"{icon} {kind.capitalize()}  {start:.1f}s–{end:.1f}s"
            block = _SectionBlock(self, sec, color, tooltip, self._bar_container)
            block.setFixedWidth(width)

            # Remember this block so we can tweak its tooltip when media is attached.
            try:
                idx = self._sections.index(sec)
            except ValueError:
                idx = -1
            if idx >= 0:
                self._section_blocks[idx] = block
                label = self._section_media_labels.get(idx)
                if label:
                    block.setToolTip(f"{tooltip}\nMedia: {label}")

            self._bar_layout.addWidget(block)

        # Set a fixed content width so the scroll area can show a horizontal
        # scrollbar whenever the zoomed timeline is wider than the viewport.
        total_width = max(1, total_width)
        self._bar_container.setMinimumWidth(total_width)
        self._bar_container.setMaximumWidth(total_width)

    # Small helper so we do not import QBrush/QColor at top-level if not needed.
    def _make_brush(self, color_name: str):
        from PySide6.QtGui import QBrush, QColor

        return QBrush(QColor(color_name))



    def _populate_energy_overlay(
        self,
        duration: float,
        energy_times: Iterable[float],
        energy_values: Iterable[float],
        beats: Iterable[Any],
        waveform_times: Iterable[float],
        waveform_values: Iterable[float],
    ) -> None:
        """Draw waveform + energy / beat overlay above the section bar.

        - Waveform: thin line taken from analysis.waveform_times / waveform_values
        - Energy: normalised 0..1 envelope from analysis.energy_times / energy_values
        - Beats: major beat markers near the top edge

        All data is optional; the plot falls back gracefully when any part
        is missing or pyqtgraph is unavailable.
        """
        if self._energy_plot is None or pg is None or duration <= 0:
            return

        self._energy_plot.clear()

        # ---------------------- Waveform (optional) ----------------------
        wt = [float(t) for t in waveform_times]
        wv = [float(v) for v in waveform_values]
        n_wave = min(len(wt), len(wv))

        if n_wave >= 2:
            # Clamp to song duration and drop obviously invalid times.
            wt = [max(0.0, min(duration, wt[i])) for i in range(n_wave)]
            wv = [wv[i] for i in range(n_wave)]

            # Lightweight downsample so very long tracks stay responsive.
            max_points = 4000
            if n_wave > max_points:
                step = max(1, n_wave // max_points)
                wt = wt[::step]
                wv = wv[::step]

            # Normalise amplitudes into 0..1 for display.
            min_v = min(wv)
            max_v = max(wv)
            span = (max_v - min_v) or 1.0
            w_ys = [(v - min_v) / span for v in wv]

            try:
                wave_pen = pg.mkPen(width=1)
            except Exception:
                wave_pen = None
            self._energy_plot.plot(wt, w_ys, pen=wave_pen)

        # ---------------- Energy envelope (optional) ---------------------
        times = [float(t) for t in energy_times]
        values = [float(v) for v in energy_values]
        n = min(len(times), len(values))

        if n >= 2:
            times = [max(0.0, min(duration, times[i])) for i in range(n)]
            values = [max(0.0, values[i]) for i in range(n)]
            max_v = max(values) or 1.0
            ys = [v / max_v for v in values]

            try:
                pen = pg.mkPen(width=2)
            except Exception:
                pen = None
            self._energy_plot.plot(times, ys, pen=pen)

        # Set visible ranges even if only one of the above was drawn.
        self._energy_plot.setXRange(0, duration, padding=0.0)
        self._energy_plot.setYRange(0, 1.05, padding=0.0)

        # ------------------------- Beat markers --------------------------
        major_times = [
            float(getattr(b, "time", 0.0) or 0.0)
            for b in beats
            if getattr(b, "kind", "") == "major"
        ]
        major_times = [t for t in major_times if 0.0 <= t <= duration]
        if major_times:
            y = [1.02] * len(major_times)
            self._energy_plot.plot(major_times, y, pen=None, symbol="o", symbolSize=4)

    def _update_summary(self, sections: Iterable[Any], duration: float) -> None:
        """Update the compact textual summary below the table."""
        if not hasattr(self, "_summary"):
            return

        rows = list(sections)
        if not rows or duration <= 0:
            self._summary.setText(
                "No music analysis yet. Run 'Analyze Music' to see structure and stats."
            )
            return

        durations = []
        counts: dict[str, int] = {}
        longest: dict[str, float] = {}
        for sec in rows:
            kind = getattr(sec, "kind", "section") or "section"
            start = float(getattr(sec, "start", 0.0) or 0.0)
            end = float(getattr(sec, "end", start) or start)
            d = max(0.0, end - start)
            durations.append(d)
            counts[kind] = counts.get(kind, 0) + 1
            longest[kind] = max(longest.get(kind, 0.0), d)

        total = len(rows)
        avg_len = sum(durations) / len(durations) if durations else 0.0

        lines = [
            f"Total sections: {total}",
            f"Song duration: {duration:.1f} s",
        ]
        for key in ("intro", "verse", "chorus", "drop", "break"):
            if key in counts:
                lines.append(f"{key.capitalize()} count: {counts[key]}")
        if "break" in longest:
            lines.append(f"Longest break: {longest['break']:.1f} s")
        if "chorus" in longest:
            lines.append(f"Longest chorus: {longest['chorus']:.1f} s")
        if "drop" in longest:
            lines.append(f"Longest drop: {longest['drop']:.1f} s")
        lines.append(f"Average section length: {avg_len:.1f} s")

        self._summary.setText("   • " + "\n   • ".join(lines))


    # ----------------------------------------------------------------- segments

    def _on_segments_changed(self) -> None:
        """Internal helper: repaint the custom segment bar when data changes and emit a snapshot."""
        # Repaint the custom segment bar.
        if hasattr(self, "_segment_bar"):
            self._segment_bar.update()

        # Emit a snapshot of the segments so external tools can react (e.g. clip lists).
        try:
            payload: list[dict[str, Any]] = []
            for seg in sorted(self._segments, key=lambda s: float(s.start)):
                if float(seg.end) <= float(seg.start):
                    continue
                label = getattr(seg, "label", "")
                payload.append(
                    {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "label": label,
                    }
                )
            self.customSegmentsChanged.emit(payload)
        except Exception:
            # Best-effort only; never let UI break if something goes wrong here.
            pass

    def _on_segment_selected(self, index: int) -> None:
        """Remember which user segment is selected for Delete / editing."""
        if not self._segments:
            self._selected_segment_index = None
            self._on_segments_changed()
            return

        if index < 0 or index >= len(self._segments):
            self._selected_segment_index = None
        else:
            self._selected_segment_index = index
        self._on_segments_changed()

    def _delete_selected_segment(self) -> bool:
        """Delete the currently selected custom segment, if any."""
        idx = self._selected_segment_index
        if idx is None or not (0 <= idx < len(self._segments)):
            return False

        del self._segments[idx]
        if self._segments:
            self._selected_segment_index = min(idx, len(self._segments) - 1)
        else:
            self._selected_segment_index = None

        self._on_segments_changed()
        return True

    def get_custom_segments(self) -> list[tuple[float, float]]:
        """Return current custom segments as (start, end) tuples, sorted by start."""
        if not self._segments:
            return []
        return [seg.as_tuple() for seg in sorted(self._segments, key=lambda s: float(s.start))]

    def set_custom_segments(self, segments: Iterable[tuple[float, float]]) -> None:
        """Replace current custom segments with *segments* (start, end) tuples."""
        self._segments = []
        for start, end in segments:
            try:
                start_f = float(start)
                end_f = float(end)
            except Exception:
                continue
            if end_f <= start_f:
                continue
            self._segments.append(Segment(start_f, end_f))
        self._segments.sort(key=lambda s: float(s.start))
        self._selected_segment_index = 0 if self._segments else None

        duration = 0.0
        if self._analysis is not None:
            duration = float(getattr(self._analysis, "duration", 0.0) or 0.0)
        if hasattr(self, "_segment_bar"):
            pixels_per_second = max(1.0, self._pixels_per_second * self._zoom_factor)
            self._segment_bar.set_timeline_params(duration, pixels_per_second)

        # Custom segments may be reset when a new analysis is applied, so notify listeners.
        self._on_segments_changed()

        self._on_segments_changed()

    # ----------------------------------------------------------------- save / load helpers

    def _segments_folder(self) -> Path:
        """Return the folder where custom timeline JSON files are stored.

        Default: <root>/presets/setsave/timelines where <root> is one level
        above the folder that contains this module.
        """
        base = Path(__file__).resolve().parent
        root = base.parent
        folder = root / "presets" / "setsave" / "timelines"
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; caller will surface any write errors.
            pass
        return folder

    def _infer_track_stub(self) -> str:
        """Infer a short, filesystem-safe name for the current track."""
        if self._analysis is None:
            return "timeline"

        # Try a few common attribute names on the analysis object.
        candidates = []
        for attr in ("track_name", "title", "file_name", "file", "source_name", "source_path", "path"):
            value = getattr(self._analysis, attr, None)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        text = candidates[0] if candidates else "timeline"

        # If it looks like a path, keep only the file name.
        try:
            if any(ch in text for ch in ("/", "\\") ):
                text = Path(text).stem
        except Exception:
            pass

        # Allow only safe characters in the final stub.
        stub = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
        return stub or "timeline"

    def _on_save_segments_clicked(self) -> None:
        """Save the full timeline (sections, colors, clip labels, custom segments) to a JSON preset.

        This is intended to capture the advanced editor state for a given track so users can
        reload their manual tweaks later. The payload includes:

            - duration: song length in seconds
            - bpm: optional, if the analysis object exposes a .bpm or .tempo attribute
            - sections: list of {kind, start, end, duration, color, media_label, media_path}
            - custom_segments: list of {start, end, label}

        Notes:
            * media_path is currently best-effort; if the main tool does not provide a separate
              path, the clip label from the "Media" column is stored in both media_label and
              media_path so future versions can distinguish them when the API grows.
        """
        if self._analysis is None:
            QMessageBox.warning(
                self,
                "Save timeline",
                "No music analysis is loaded yet, so the song duration is unknown.",
            )
            return

        duration = float(getattr(self._analysis, "duration", 0.0) or 0.0)
        if duration <= 0.0:
            QMessageBox.warning(self, "Save timeline", "Song duration is zero or unknown; cannot save.")
            return

        # Optional BPM support for the future: we record it if available so the
        # main tool can wire BPM detection into the analysis object later.
        bpm = None
        for attr in ("bpm", "tempo", "detected_bpm", "detected_tempo"):
            value = getattr(self._analysis, attr, None)
            if isinstance(value, (int, float)):
                bpm = float(value)
                break

        # ------------------------------- sections -------------------------------
        sections_payload = []
        for row, sec in enumerate(self._sections):
            kind = str(getattr(sec, "kind", "section") or "section")
            start = float(getattr(sec, "start", 0.0) or 0.0)
            end = float(getattr(sec, "end", start) or start)
            sec_dur = max(0.0, end - start)

            # Clamp to known duration just in case.
            start = max(0.0, min(duration, start))
            end = max(0.0, min(duration, end))
            if end < start:
                start, end = end, start
            sec_dur = max(0.0, end - start)

            # Color used for this section kind in the UI.
            color = SECTION_COLORS.get(kind.lower())

            # Clip label from the Media column.
            media_label = ""
            if self._table.columnCount() >= 5 and row < self._table.rowCount():
                item = self._table.item(row, 4)
                if item is not None and item.text().strip():
                    media_label = item.text().strip()

            # For now we store the same value for media_label and media_path. In the future,
            # the main tool can introduce a dedicated API to provide the underlying file path.
            media_path = media_label

            sections_payload.append(
                {
                    "kind": kind,
                    "start": start,
                    "end": end,
                    "duration": sec_dur,
                    "color": color,
                    "media_label": media_label,
                    "media_path": media_path,
                }
            )

        # --------------------------- custom segments ----------------------------
        custom_segments_payload = []
        for seg in self._segments:
            start = max(0.0, min(duration, float(seg.start)))
            end = max(0.0, min(duration, float(seg.end)))
            if end <= start:
                continue
            label = getattr(seg, "label", "")
            custom_segments_payload.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                }
            )

        # Build final payload; we always include sections, and include custom_segments
        # even if empty so the schema is predictable.
        folder = self._segments_folder()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stub = self._infer_track_stub()
        filename = f"{timestamp}_{stub}.json"
        path = folder / filename

        payload: dict[str, object] = {
            "version": 2,
            "duration": duration,
            "track": stub,
            "sections": sections_payload,
            "custom_segments": custom_segments_payload,
        }
        if bpm is not None:
            payload["bpm"] = bpm

        try:
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save timeline", f"Could not write JSON file:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Save timeline",
            f"Saved timeline preset to:\n{path}",
        )
    def _on_load_segments_clicked(self) -> None:
        """Reload a full timeline preset (sections + clip labels + custom segments) from JSON.

        Loading a preset will OVERWRITE the current clip labels and custom segments for this
        track. The underlying analysis structure (section kinds + timings) is not modified yet;
        instead, we align the saved sections to the current ones by row index.
        """
        folder = self._segments_folder()
        if not folder.exists():
            QMessageBox.information(
                self,
                "Load timeline",
                "No saved timelines found yet. Save a timeline preset first.",
            )
            return

        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load timeline JSON",
            str(folder),
            "Timeline JSON (*.json);;All files (*)",
        )
        if not path_str:
            return

        # Confirm overwrite as requested.
        reply = QMessageBox.question(
            self,
            "Load timeline",
            "Loading a timeline preset will overwrite clip labels and custom segments for this track.\n\n"
            "Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        try:
            with open(path_str, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Load timeline", f"Could not read JSON file:\n{exc}")
            return

        # --- Determine duration ---
        duration = 0.0
        if self._analysis is not None:
            duration = float(getattr(self._analysis, "duration", 0.0) or 0.0)
        if duration <= 0.0:
            duration = float(data.get("duration") or 0.0)

        # --- Restore section clip labels (by index) ---
        sections_data = data.get("sections")
        if isinstance(sections_data, list) and self._sections:
            max_idx = min(len(self._sections), len(sections_data), self._table.rowCount())
            for idx in range(max_idx):
                item = sections_data[idx]
                if not isinstance(item, dict):
                    continue
                # Prefer media_label, fall back to media_path if needed.
                label = item.get("media_label") or item.get("media_path") or ""
                if label is None:
                    label = ""
                try:
                    section = self._sections[idx]
                except IndexError:
                    continue
                self.set_section_media_label(section, str(label))

        # --- Restore custom segments ---
        # Prefer the new "custom_segments" key, but fall back to legacy "segments" if present.
        segments_data = data.get("custom_segments")
        if not isinstance(segments_data, list):
            segments_data = data.get("segments")

        new_segments: list[Segment] = []
        if isinstance(segments_data, list):
            for item in segments_data:
                if not isinstance(item, dict):
                    continue
                try:
                    start = float(item.get("start", 0.0))
                    end = float(item.get("end", 0.0))
                except Exception:
                    continue
                if end <= start:
                    continue
                if duration > 0.0:
                    if start >= duration:
                        continue
                    start = max(0.0, min(duration, start))
                    end = max(0.0, min(duration, end))
                    if end <= start:
                        continue
                label = item.get("label", "") if isinstance(item, dict) else ""
                new_segments.append(Segment(start, end, label))

        # Apply new segments + refresh bar.
        if new_segments:
            new_segments.sort(key=lambda s: float(s.start))
            self._segments = new_segments
            self._selected_segment_index = 0
        else:
            # If there are no segments in the file, just clear existing ones.
            self._segments = []
            self._selected_segment_index = None

        if duration > 0.0 and hasattr(self, "_segment_bar"):
            pixels_per_second = max(1.0, self._pixels_per_second * self._zoom_factor)
            self._segment_bar.set_timeline_params(duration, pixels_per_second)

        self._on_segments_changed()

        QMessageBox.information(self, "Load timeline", "Timeline preset loaded successfully.")
    # ------------------------------------------------------------------ events

    def eventFilter(self, obj, event):  # type: ignore[override]
        """Intercept wheel + middle-button drag events over the mini-timeline.

        - Mouse wheel zooms in/out, anchored to the mouse cursor (DAW style).
        - Middle mouse button drag pans the timeline left/right smoothly.
        """
        if getattr(self, "_bar_scroll", None) is not None and obj is self._bar_scroll.viewport():
            etype = event.type()

            # --------------------------- Panning ---------------------------
            if etype == QEvent.MouseButtonPress:
                if event.button() == Qt.MiddleButton:
                    hbar = self._bar_scroll.horizontalScrollBar()
                    if hbar is not None:
                        self._is_panning = True
                        self._pan_start_x = event.pos().x()
                        self._pan_start_scroll = hbar.value()
                        self._bar_scroll.setCursor(Qt.ClosedHandCursor)
                        event.accept()
                        return True

            elif etype == QEvent.MouseMove:
                if self._is_panning:
                    hbar = self._bar_scroll.horizontalScrollBar()
                    if hbar is not None:
                        dx = event.pos().x() - self._pan_start_x
                        new_value = self._pan_start_scroll - dx
                        new_value = max(hbar.minimum(), min(hbar.maximum(), new_value))
                        hbar.setValue(new_value)
                    event.accept()
                    return True

            elif etype == QEvent.MouseButtonRelease:
                if self._is_panning and event.button() == Qt.MiddleButton:
                    self._is_panning = False
                    self._bar_scroll.unsetCursor()
                    event.accept()
                    return True

            # ---------------------------- Zoom -----------------------------
            elif etype == QEvent.Wheel:
                delta = event.angleDelta().y()
                if not delta:
                    return True

                step = 1 if delta > 0 else -1
                factor = 1.1 ** step
                new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom_factor * factor))
                if abs(new_zoom - self._zoom_factor) < 1e-4:
                    return True

                viewport = self._bar_scroll.viewport()
                hbar = self._bar_scroll.horizontalScrollBar()

                if viewport is not None and hbar is not None:
                    # Position of cursor within the viewport.
                    try:
                        mouse_x = int(event.position().x())
                    except AttributeError:
                        mouse_x = event.pos().x()

                    # Absolute content coordinate currently under the cursor.
                    content_x_at_cursor = hbar.value() + mouse_x

                    # Apply zoom and rebuild the bar.
                    self._zoom_factor = new_zoom
                    if self._analysis is not None:
                        duration = float(getattr(self._analysis, "duration", 0.0) or 0.0)
                        pixels_per_second = max(1.0, self._pixels_per_second * self._zoom_factor)
                        self._populate_bar(self._sections, duration)
                        if hasattr(self, "_segment_bar"):
                            self._segment_bar.set_timeline_params(duration, pixels_per_second)

                    # Re-anchor so the same content point stays under the mouse cursor.
                    new_value = content_x_at_cursor - mouse_x
                    new_value = max(hbar.minimum(), min(hbar.maximum(), new_value))
                    hbar.setValue(new_value)
                else:
                    # Fallback: zoom without anchoring.
                    self._zoom_factor = new_zoom
                    if self._analysis is not None:
                        duration = float(getattr(self._analysis, "duration", 0.0) or 0.0)
                        pixels_per_second = max(1.0, self._pixels_per_second * self._zoom_factor)
                        self._populate_bar(self._sections, duration)
                        if hasattr(self, "_segment_bar"):
                            self._segment_bar.set_timeline_params(duration, pixels_per_second)

                return True

        return super().eventFilter(obj, event)